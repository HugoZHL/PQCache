# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import hashlib
import logging
import os
import struct
import unittest.mock as um
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import (
    Any,
    Callable,
    ContextManager,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    cast,
)

import torch
import torch.nn.functional as F
import transformers
from torch import Tensor
from torch.nn.functional import pad
from typing_extensions import TypeAlias

logger = logging.getLogger(__name__)

KVCache: TypeAlias = Tuple[Tuple[Tensor, Tensor], ...]
Model: TypeAlias = transformers.PreTrainedModel
ModelContext: TypeAlias = Callable[[Model], ContextManager[Model]]


@contextmanager
def null_model_context(model: Model) -> Iterator[Model]:
    yield model


def patch_for_model(
    target: str, fn: Callable[..., Any], *args: Any, **kwargs: Any
) -> ModelContext:
    """Patch a function (globally) during ModelContext execution."""

    @contextmanager
    def model_context(model: Model) -> Iterator[Model]:
        with um.patch(target, partial(fn, *args, **kwargs)):
            yield model

    return model_context


DEFAULT_CACHE_DIR = f"/net/group/research/{os.environ.get('USER')}/cache"


class Adapter:
    """A simplified interface for HuggingFace models paired with a tokeniser."""

    DEFAULT_BATCH_SIZE = 32

    def __init__(
        self,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizerBase,
        batch_size: int,
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self._batch_size = batch_size

        # Set padding token
        self.tokenizer.pad_token = self.tokenizer.eos_token

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        batch_size: int = DEFAULT_BATCH_SIZE,
        dtype: Optional[torch.dtype] = None,
    ) -> "Adapter":
        return cls.from_model(
            transformers.AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path, torch_dtype=dtype
            ),
            batch_size=batch_size,
        )

    @classmethod
    def from_model(
        cls, model: transformers.PreTrainedModel, batch_size: int = DEFAULT_BATCH_SIZE
    ) -> "Adapter":
        return cls(
            model=model,
            tokenizer=transformers.AutoTokenizer.from_pretrained(
                model.config._name_or_path
            ),
            batch_size=batch_size,
        )

    # lm_eval API

    @property
    def eot_token_id(self) -> int:
        return cast(int, self.tokenizer.eos_token_id)

    @property
    def max_length(self) -> int:
        try:
            return cast(int, self.model.config.max_sequence_length)
        except AttributeError:
            return cast(int, self.model.config.max_position_embeddings)

    @property
    def max_gen_toks(self) -> int:
        return 128  # arbitary?

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def device(self) -> torch.device:
        return cast(torch.device, self.model.device)

    def tok_encode(self, string: str) -> List[int]:
        return cast(List[int], self.tokenizer.encode(string, add_special_tokens=False))

    def tok_decode(self, tokens: Iterable[int]) -> str:
        return cast(str, self.tokenizer.decode(tokens))

    def _model_call(self, inps: Tensor) -> Tensor:
        with torch.no_grad():
            logits: Tensor = self.model(inps).logits
            return logits

    def _model_generate(
        self, context: Tensor, max_length: int, eos_token_id: int
    ) -> Tensor:
        generation: Tensor = self.model.generate(
            context, max_length=max_length, eos_token_id=eos_token_id, do_sample=False
        )
        return generation

    @contextmanager
    def tokenizer_override(self, padding: str, truncation: str) -> Iterator[None]:
        try:
            defaults = self.tokenizer.padding_side, self.tokenizer.truncation_side
            self.tokenizer.padding_side = padding
            self.tokenizer.truncation_side = truncation
            yield
        finally:
            self.tokenizer.padding_side, self.tokenizer.truncation_side = defaults

    def _get_cache_str(self, s: str, limit: int) -> str:
        """Return a hash identifier for the KV cache
        based on the input string and the model config.

        Args:
            s (str): Context string that will be cached
            limit (int): Token length limit, before truncation occurs

        Returns:
            str: Filename hash used for caching
        """
        model_config = self.model.config.to_json_string(use_diff=False)
        hash_fn = hashlib.md5()
        for item in [model_config, s]:
            hash_fn.update(item.encode())
        hash_fn.update(struct.pack("<L", limit))

        return f"{hash_fn.hexdigest()}"

    @staticmethod
    def _kv_to_tuple(kv_cache: Tensor) -> KVCache:
        return tuple((layer[0], layer[1]) for layer in kv_cache)

    @staticmethod
    def _kv_to_tensor(kv_cache: KVCache) -> Tensor:
        """
        Args:
            past_key_values (KVCache): per-layer cache of (K, V) matrices, each
            of shape (batch_size, num_heads, max_sequence_length, embed_size_per_head)

        Returns:
            Tensor:
            (layer, 2, batch_size, num_heads, max_sequence_length, embed_size_per_head)
        """
        return torch.stack([torch.stack(kv) for kv in kv_cache])

    def _save_kv_cache(
        self,
        past_key_values: Tensor,
        sequence_lens: Tensor,
        filepaths: List[Path],
    ) -> None:
        """
        Given KV cache for a batch of sequences, save the cache for each sequence i
        separately into a Tensor of shape
        (num_layers, 2, 1, num_heads, sequence_lens[i], embed_size_per_head)

        Args:
            past_key_values (Tensor): as returned by _kv_to_tensor
            sequence_lens (Tensor): per-sequence sequence_length
            filepaths (List[Path]): per-sequence filepath
        """
        batch_size = past_key_values.shape[2]
        assert (
            len(filepaths) == batch_size
        ), "Number of filepaths needs to match the batch size"
        assert (
            len(sequence_lens) == batch_size
        ), "Number of sequence lengths needs to match the batch size"

        for i, (sequence_len, filepath) in enumerate(zip(sequence_lens, filepaths)):
            filepath.parent.mkdir(parents=True, exist_ok=True)
            # Expect right-padded batch
            torch.save(
                past_key_values[:, :, i : i + 1, :, :sequence_len, :].clone(),
                filepath,
            )

    def prefill_with_cache(
        self, text: List[str], max_context_length: int, dir_path: Optional[str]
    ) -> Tuple[KVCache, Tensor]:
        """Given a batch of input strings, run the model and save
        the KV cache for each sequence individually.
        Note: uses right padding when batching.

        Args:
            text (List[str]): Batch of text inputs
            max_context_length (int): Maximum length of context (trims from the left
            to fit inside this limit)
            dir_path (Optional[str]): Directory for storing the KV cache

        Returns:
            (KVCache, Tensor): (kv cache, sequence lengths)
        """
        with self.tokenizer_override(padding="right", truncation="left"):
            enc = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_context_length,
            )

        input_ids, attention_mask = enc["input_ids"], enc["attention_mask"]

        with torch.no_grad():
            out = self.model(
                input_ids=input_ids.to(self.model.device),
                attention_mask=attention_mask.to(self.model.device),
            )

        # Get the actual length of each sequence in the batch
        sequence_lens = attention_mask.sum(dim=-1).tolist()

        if dir_path:
            # Get the filepath hash for each sequence
            filepaths = [
                Path(dir_path) / (self._get_cache_str(s, max_context_length) + ".pt")
                for s in text
            ]
            self._save_kv_cache(
                self._kv_to_tensor(out.past_key_values), sequence_lens, filepaths
            )

        return out.past_key_values, sequence_lens

    def greedy_sample(
        self,
        ctxs: List[str],
        prompts: List[str],
        num_generated_tokens: int,
        max_prompt_and_generated_tokens: int = 256,
        use_cache: bool = False,
        cache_dir: str = DEFAULT_CACHE_DIR,
        generation_context: Optional[ModelContext] = None,
        combine_context_and_prompt: bool = True,
    ) -> Tensor:
        """
        Sample greedily from the model assuming the prompts are ctx + prompt,
        where KV cache for the ctx can be loaded from memory.

        Args:
            ctxs (List[str]): List of context strings prepended to prompts
            prompts (List[str]): List of prompt strings
            num_generated_tokens (int): Number of generated tokens
            max_prompt_and_generated_tokens (int, optional): Number of tokens
            to allocate to prompt and generation. Defaults to 256.
            use_cache (bool, optional): Load ctx KV cache from disk if it exits.
            If not, first generate and save the cache. Defaults to True.
            cache_dir (str, optional): Directory for saving/loading KV cache.
            Defaults to "cache/".
            generation_context (ModelContext, optional): A contextmanager function
            that accepts and yields a PreTrainedModel, used during generation only,
            for each batch being processed (for example to enable certain behaviours
            or reset state between batches). If not specified, looks for a generation
            context in self.model.generation_context.
            combine_context_and_prompt (bool, optional): Combine the context &
            prompt into a single prefill string. Not compatible with `use_cache`.

        Returns:
            Tensor: Generated tokens. Shape - (batch_size, num_generated_tokens)
        """
        batch_size = len(ctxs)
        assert len(ctxs) == len(
            prompts
        ), "Number of context and prompt strings should be the same"
        max_context_length = self.max_length - max_prompt_and_generated_tokens

        if combine_context_and_prompt:
            assert not use_cache, "cannot combine_context_and_prompt and use_cache"
            with self.tokenizer_override(padding="left", truncation="left"):
                enc = self.tokenizer(
                    [c + p for c, p in zip(ctxs, prompts)],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_context_length,
                )
            input_ids = enc["input_ids"]
            attention_mask = enc["attention_mask"]
            past_key_values = None
            position_ids = None
        else:
            if use_cache:
                filepaths = [
                    Path(cache_dir)
                    / (self._get_cache_str(ctx, max_context_length) + ".pt")
                    for ctx in ctxs
                ]
                cache_exists = [filepath.exists() for filepath in filepaths]

                # Generate KV cache where missing
                if not all(cache_exists):
                    ctxs_to_cache = [
                        ctx for ctx, exist in zip(ctxs, cache_exists) if not exist
                    ]
                    self.prefill_with_cache(
                        ctxs_to_cache,
                        max_context_length=max_context_length,
                        dir_path=cache_dir,
                    )

                # All KV caches should be available now
                cache_exists = [filepath.exists() for filepath in filepaths]
                assert all(cache_exists), "Issue generating KV cache"
                cache = [torch.load(filepath) for filepath in filepaths]
                seq_lens = torch.tensor([pkv.shape[-2] for pkv in cache])
                # Right-pad the KV caches (this ensures that the actual token
                # position doesn't change between prefill and generation, for
                # techniques that want to aggregate "KV stats")
                pkv = torch.cat(
                    [
                        pad(pkv, (0, 0, 0, max(seq_lens) - seq_len))
                        for pkv, seq_len in zip(cache, seq_lens)
                    ],
                    dim=2,
                ).to(self.model.device)
                past_key_values = self._kv_to_tuple(pkv)
            else:
                past_key_values, seq_lens = self.prefill_with_cache(
                    ctxs, max_context_length=max_context_length, dir_path=None
                )

            attention_mask_left = torch.tensor(
                [
                    [i < len for i in range(past_key_values[0][0].shape[-2])]
                    for len in seq_lens
                ]
            ).long()

            # Tokenize prompts with left padding (easier generation)
            with self.tokenizer_override(padding="left", truncation="left"):
                prompts_enc = self.tokenizer(
                    prompts, return_tensors="pt", padding=True, truncation=False
                )

            input_ids = prompts_enc["input_ids"]
            attention_mask_right = prompts_enc["attention_mask"]
            if (
                max_prompt_and_generated_tokens
                < input_ids.shape[1] + num_generated_tokens
            ):
                logger.warning(
                    "length of prompt (%d) + generation (%d)"
                    " is more than reserved (%d)",
                    input_ids.shape[1],
                    num_generated_tokens,
                    max_prompt_and_generated_tokens,
                )

            # Apply correct position ids given padding tokens
            q_len = input_ids.shape[-1]
            q_no_pad_lens = attention_mask_right.sum(dim=-1)
            position_ids = torch.stack(
                [
                    torch.arange(
                        seq_len - (q_len - q_no_pad_len), seq_len + q_no_pad_len
                    )
                    for seq_len, q_no_pad_len in zip(seq_lens, q_no_pad_lens)
                ]
            ).to(self.model.device)
            position_ids.masked_fill_(position_ids < 0, 0)

            # Concatenate cache and prompts attention masks
            attention_mask = torch.cat(
                [attention_mask_left, attention_mask_right], dim=1
            )

        # Generate tokens one by one
        generated_tokens = []
        context = (
            generation_context
            or getattr(self.model, "generation_context", None)
            or null_model_context
        )
        with torch.no_grad(), context(self.model) as model:
            for _ in range(num_generated_tokens):
                out = model(
                    input_ids=input_ids.to(model.device),
                    attention_mask=attention_mask.to(model.device),
                    past_key_values=past_key_values,
                    position_ids=position_ids,
                )
                logits = out.logits[:, -1, :]
                next_token = logits.argmax(dim=-1, keepdim=True)
                generated_tokens.append(next_token)

                # Prepare inputs for next iteration
                input_ids = next_token
                attention_mask = torch.cat(
                    [attention_mask, torch.ones(batch_size, 1)], dim=1
                )
                past_key_values = out.past_key_values
                if position_ids is not None:
                    position_ids = position_ids[:, -1:] + 1

        return torch.cat(generated_tokens, dim=1)

    def forced_sample(
        self,
        prefill: List[str],
        reference: List[str],
        max_reference_tokens: int = 256,
        generation_context: Optional[ModelContext] = None,
    ) -> Tensor:
        """
        Sample from the model using teacher-forcing, token-by-token.

        Prefills are provided, which are initially fed to the model to generate an
        initial batch of negative log likelihoods. Rather than feeding the most likely
        tokens back into the model, the reference texts are instead tokenized and fed in
        for the next iteration. This function returns the concatenation of the negative
        log likelihoods corresponding to the target tokens.

        Args:
            prefill (List[str]): List of prefill strings processed before generation
            reference (List[str]): List of reference strings for teacher-forcing
            max_reference_tokens (int): Maximium number of generated reference tokens.
            generation_context (ModelContext, optional): A contextmanager function
            that accepts and yields a PreTrainedModel, used during generation only,
            for each batch being processed (for example to enable certain behaviours
            or reset state between batches). If not specified, looks for a generation
            context in self.model.generation_context.

        Returns:
            Tensor: negative log likelihoods. Shape - (batch_size, num_generated_logits)
        """
        with self.tokenizer_override(padding="left", truncation="left"):
            prefill_enc = self.tokenizer(
                prefill,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length - max_reference_tokens,
            )
            prefill_length = prefill_enc["input_ids"].shape[-1]
        with self.tokenizer_override(padding="right", truncation="right"):
            reference_enc = self.tokenizer(
                reference, return_tensors="pt", padding=True, add_special_tokens=False
            )
            reference_length = reference_enc["input_ids"].shape[-1]
            if reference_length > max_reference_tokens:
                raise ValueError(
                    "Reference string exceeds max_reference_tokens"
                    f"={max_reference_tokens}, reference: {reference}"
                )

        full_input_ids = torch.concat(
            [prefill_enc["input_ids"], reference_enc["input_ids"]], dim=1
        )
        full_attention_mask = torch.concat(
            [prefill_enc["attention_mask"], reference_enc["attention_mask"]], dim=1
        )
        full_position_ids = torch.nn.functional.pad(
            full_attention_mask.cumsum(dim=1)[:, :-1], (1, 0)
        )
        full_length = prefill_length + reference_length

        # Generate tokens one by one
        neg_log_likelihoods = []
        context = (
            generation_context
            or getattr(self.model, "generation_context", None)
            or null_model_context
        )

        past_key_values = None
        with torch.no_grad(), context(self.model) as model:
            idxs = [0] + list(range(prefill_length, full_length + 1))
            for i, j, k in zip(idxs, idxs[1:], idxs[2:]):
                input_ids = full_input_ids[:, i:j].to(model.device)
                position_ids = full_position_ids[:, i:j].to(model.device)
                attention_mask = full_attention_mask[:, :j].to(model.device)
                target_ids = full_input_ids[:, j:k]

                out = model(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                )
                past_key_values = out.past_key_values

                logits = out.logits[:, -1, :]
                nll = F.cross_entropy(
                    logits, target_ids.squeeze(-1).to(logits.device), reduction="none"
                ).cpu()
                nll.masked_fill_(~full_attention_mask[:, j:k].squeeze(-1).bool(), 0)
                neg_log_likelihoods.append(nll)

        return torch.stack(neg_log_likelihoods, dim=1)
