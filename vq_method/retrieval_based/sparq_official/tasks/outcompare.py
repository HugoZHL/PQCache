# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

"""A basic evaluation harness for language model adaptation.

The idea is to compare outputs against a reference model.
"""

import itertools as it
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import datasets
import torch
import torchaudio
import tqdm
import transformers
from torch import Tensor

from .. import utility


@dataclass
class Datum:
    prompt: Tensor
    completion: Tensor
    entropy: Tensor


@dataclass
class Dataset:
    model: str
    dataset: str
    data: List[Datum]

    def to_json(self) -> str:
        d = self.__dict__.copy()
        d["data"] = [{k: v.tolist() for k, v in x.__dict__.items()} for x in d["data"]]
        d["_version"] = 0
        return json.dumps(d)

    @classmethod
    def from_json(cls, text: str) -> "Dataset":
        d = json.loads(text)
        d.pop("_version")
        d["data"] = [
            Datum(**{k: torch.tensor(v) for k, v in x.items()}) for x in d["data"]
        ]
        return cls(**d)

    def save(self, path: Union[str, Path]) -> None:
        Path(path).write_text(self.to_json())

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Dataset":
        return cls.from_json(Path(path).read_text())


def _get_prompts(
    dataset_name: str,
    tokenizer: transformers.PreTrainedTokenizerBase,
    length: int,
    seed: int = 283492,
) -> Iterable[Tensor]:
    """Extract a list of prompts from a dataset.

    dataset_name -- path:name:split e.g. "wikitext:wikitext-103-raw-v1:validation"

    length -- number of tokens per prompt (discards examples with fewer tokens)
    """
    path, name, split = dataset_name.split(":")
    dataset = datasets.load_dataset(path, name=name, split=split)
    for line in dataset.shuffle(seed=seed)["text"]:
        tokens = tokenizer.encode(line)
        if len(tokens) > length:
            yield torch.tensor(tokens[:length])


def _batched_causal_cross_entropy(logits: Tensor, tokens: Tensor) -> Tensor:
    """Compute the X-Ent for each element of a batch of shape (..., sequence_length)."""
    next_tokens = tokens[..., 1:]
    return torch.nn.functional.cross_entropy(
        logits[..., :-1, :].flatten(end_dim=-2), next_tokens.flatten(), reduction="none"
    ).reshape(next_tokens.shape)


def _complete(
    model: transformers.PreTrainedModel, input: Tensor, length: int
) -> Tensor:
    completion: Tensor = model.generate(
        input,
        attention_mask=torch.ones_like(input),
        max_new_tokens=length,
        pad_token_id=model.config.eos_token_id,
    )
    return completion


def _generate_datums(
    model: transformers.PreTrainedModel,
    prompts: Iterable[Tensor],
    completion_length: int,
    batch_size: int,
) -> Iterable[Datum]:
    with torch.no_grad():
        for batch in utility.batches(prompts, batch_size):
            input = torch.stack(batch)
            completions = _complete(model, input, completion_length)
            entropies = _batched_causal_cross_entropy(
                model(completions).logits, completions
            )
            for prompt, completion, entropy in zip(
                batch, completions[:, input.shape[1] :], entropies[:, input.shape[1] :]
            ):
                yield Datum(prompt, completion, entropy)


def generate_dataset(
    model_name: str,
    prompt_length: int,
    completion_length: int,
    batch_size: int,
    dataset: str = "wikitext:wikitext-103-raw-v1:validation",
    limit: Optional[int] = None,
    progress: bool = True,
) -> Dataset:
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    prompts = list(it.islice(_get_prompts(dataset, tokenizer, prompt_length), limit))
    data = _generate_datums(model, prompts, completion_length, batch_size)
    if progress:
        data = tqdm.tqdm(data, total=len(prompts), desc=model_name)
    return Dataset(model=model_name, dataset=dataset, data=list(data))


METRICS = ["exact_match_length", "edit_distance_L16", "entropy_rmse"]


def _evaluate_rmse(
    model: transformers.PreTrainedModel, data: Iterable[Datum], batch_size: int
) -> Iterable[float]:
    for batch in utility.batches(data, batch_size):
        tokens = torch.stack([torch.cat([d.prompt, d.completion]) for d in batch])
        entropy = _batched_causal_cross_entropy(model(tokens).logits, tokens)
        target_entropy = torch.stack([d.entropy for d in batch])
        mse = torch.nn.functional.mse_loss(
            target_entropy,
            entropy[..., -target_entropy.shape[-1] :],
            reduction="none",
        )
        yield from torch.sqrt(torch.mean(mse, dim=1)).tolist()


def _evaluate_exact_match(
    model: transformers.PreTrainedModel,
    data: Iterable[Datum],
    batch_size: int,
    edit_distance_length: int,
) -> Iterable[Tuple[int, int]]:
    for batch in utility.batches(data, batch_size):
        length = len(batch[0].completion)
        completion = _complete(model, torch.stack([d.prompt for d in batch]), length)[
            :, -length:
        ]
        target_completion = torch.stack([d.completion for d in batch])
        for expected, actual in zip(target_completion, completion):
            yield (
                int(torch.cummin(expected[: len(actual)] == actual, 0).values.sum()),
                torchaudio.functional.edit_distance(
                    expected[:edit_distance_length], actual[:edit_distance_length]
                ),
            )


def _mean_and_stderr(samples: Tensor, name: str) -> Dict[str, float]:
    fsamples = samples.float()
    return {
        name: float(fsamples.mean()),
        f"{name}_stderr": float(fsamples.std() / fsamples.nelement() ** 0.5),
    }


def evaluate(
    model: transformers.PreTrainedModel,
    dataset: Dataset,
    batch_size: int,
    limit: Optional[int] = None,
    metrics: List[str] = METRICS,
) -> Dict[str, float]:
    """Evaluate a model adaptation against a reference (the original model)."""

    if any(m not in METRICS for m in metrics):
        raise ValueError(
            f"Unknown metrics {[m for m in metrics if m not in METRICS]}"
            f" (expected {METRICS})"
        )
    if model.config._name_or_path != dataset.model:
        print(
            f"Warning: evaluating the model {model.config._name_or_path!r} against a"
            f" different reference {dataset.model!r}",
            file=sys.stderr,
        )

    results = {}
    with torch.no_grad():
        if "entropy_rmse" in metrics:
            rmse = torch.tensor(
                list(_evaluate_rmse(model, dataset.data[:limit], batch_size))
            )
            results.update(_mean_and_stderr(rmse, "entropy_rmse"))
        if "exact_match_length" in metrics or "edit_distance_L16" in metrics:
            stats = torch.tensor(
                list(
                    _evaluate_exact_match(
                        model, dataset.data[:limit], batch_size, edit_distance_length=16
                    )
                )
            )
            if "exact_match_length" in metrics:
                results.update(_mean_and_stderr(stats[:, 0], "exact_match_length"))
            if "edit_distance_L16" in metrics:
                results.update(_mean_and_stderr(stats[:, 1], "edit_distance_L16"))
    return results
