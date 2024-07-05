# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

"""Methods for evaluating models on synthetic text-repetition task.
"""
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import datasets
from tqdm import tqdm

from ..eval_adapter import DEFAULT_CACHE_DIR, Adapter, ModelContext
from ..utility import AnyDict, batches


def split_on_whitespace(text: str, length: int) -> Iterable[Tuple[int, str]]:
    for m in re.finditer(rf"(.{{{length},}}?)(?=\s|$)", text, flags=re.DOTALL):
        yield m.start(), m.group()


def split_into_prompt_and_reference(
    text: str, prompt_length: int, reference_length: int
) -> Iterable[Tuple[int, str, str]]:
    for m in re.finditer(
        rf"(.{{{prompt_length},}}?)(\s.{{{reference_length-1},}}?)(?=\s|$)",
        text,
        flags=re.DOTALL,
    ):
        yield m.start(), m.group(1), m.group(2)


def generate_examples(
    text: str,
    context_length: int = 6000,
    prompt_length: int = 128,
    reference_length: int = 256,
) -> Iterable[AnyDict]:
    for context_start, context in split_on_whitespace(text, context_length):
        for prompt_start, prompt, reference in split_into_prompt_and_reference(
            context, prompt_length, reference_length
        ):
            yield dict(
                id=context_start + prompt_start,
                context_id=context_start,
                context=context,
                prompt=prompt,
                reference=reference,
            )


class Shakespeare:
    @classmethod
    def data(cls, shuffle_seed: int = 544085, **settings: Any) -> datasets.Dataset:
        ds = datasets.load_dataset("tiny_shakespeare")
        text = "".join(v["text"][0] for v in ds.values())
        examples = list(generate_examples(text, **settings))
        return datasets.Dataset.from_list(examples).shuffle(shuffle_seed)


LEADING_SPACE_PATTERN = re.compile(r"^\s+")


def evaluate_match_length(generation: str, reference: str) -> Dict[str, int]:
    generation = LEADING_SPACE_PATTERN.sub("", generation)
    reference = LEADING_SPACE_PATTERN.sub("", reference)
    diff_idxs = [i for i, (c1, c2) in enumerate(zip(generation, reference)) if c1 != c2]
    return dict(
        match_length_char=diff_idxs[0] if diff_idxs else len(reference),
        reference_length_char=len(reference),
    )


PROMPT_PREFIX = "\n"


def evaluate(
    adapter: Adapter,
    examples: List[AnyDict],
    batch_size: int,
    prompt_prefix: str = PROMPT_PREFIX,
    max_generated_tokens: int = 128,
    open_book: bool = True,
    generation_context: Optional[ModelContext] = None,
    use_cache: bool = False,
    cache_dir: str = DEFAULT_CACHE_DIR,
    combine_context_and_prompt: bool = True,
    progress: Union[bool, str] = True,
) -> Iterable[AnyDict]:
    for batch in tqdm(
        list(batches(examples, batch_size)),
        desc=progress
        if isinstance(progress, str)
        else f"Evaluating {adapter.model.name_or_path}",
        disable=progress is False,
    ):
        contexts = [b["context"] if open_book else "" for b in batch]
        prompts = [prompt_prefix + b["prompt"] for b in batch]
        prefill_lengths = [
            len(adapter.tok_encode(context + prompt))
            for context, prompt in zip(contexts, prompts)
        ]
        reference_lengths = [len(adapter.tok_encode(b["reference"])) for b in batch]
        outputs = adapter.greedy_sample(
            ctxs=contexts,
            prompts=prompts,
            num_generated_tokens=min(max(reference_lengths), max_generated_tokens),
            generation_context=generation_context,
            use_cache=use_cache,
            cache_dir=cache_dir,
            combine_context_and_prompt=combine_context_and_prompt,
        )
        for b, reference_length, prefill_length, output_ids in zip(
            batch, reference_lengths, prefill_lengths, list(outputs)
        ):
            output = adapter.tok_decode(output_ids[:reference_length])
            yield dict(
                # ids = positions (in chars) within the text
                id=b["id"],
                context_id=b["context_id"],
                output=output,
                # Needs to be in tokens for memory transfer calculation
                prefill_length=prefill_length,
                # Evaluate match length in characters
                **evaluate_match_length(output, b["reference"]),
            )
