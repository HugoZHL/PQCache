# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

"""Methods for evaluating models for generative text summarisation.

For example:

    data = summarisation.CnnDailymail.data()
    results = list(summarisation.evaluate(
        adapter,
        [data[i] for i in range(10)],
        batch_size=5,
    ))
    print(results[0])  # => {"id": ..., "output": ..., "rougeL": ...}

    rougeL = sum(r["rougeL"] for r in results) / len(results)
    print(rougeL)
"""

from functools import partial
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import datasets
import rouge_score.rouge_scorer
from tqdm import tqdm

from .. import utility
from ..eval_adapter import DEFAULT_CACHE_DIR, Adapter, ModelContext
from ..utility import AnyDict


class CnnDailymail:
    """Filtered version of cnn_dailymail:3.0.0 from HuggingFace.

    See: "Get To The Point: Summarization with Pointer-Generator Networks"
    https://www.aclweb.org/anthology/P17-1099
    """

    @staticmethod
    def preprocess(
        d: Dict[str, Any], chars_range: Tuple[int, int] = (4000, 8000)
    ) -> Dict[str, Any]:
        """Generate a summarisation example from cnn_dailymail.

        Yields: {"id": str,
                 "context": str,
                 "reference": str}
        """
        if chars_range[0] <= len(d["article"]) <= chars_range[1]:
            return dict(
                id=d["id"],
                context="Article: " + d["article"],
                reference=d["highlights"],
            )

    @classmethod
    def data(
        cls, shuffle_seed: int = 2353669, **preprocess_args: Any
    ) -> datasets.Dataset:
        return utility.map_and_filter(
            datasets.load_dataset("cnn_dailymail", name="3.0.0")["validation"],
            partial(cls.preprocess, **preprocess_args),
        ).shuffle(shuffle_seed)


DEFAULT_PROMPT = "\nSummary:"


def evaluate(
    adapter: Adapter,
    examples: List[AnyDict],
    batch_size: int,
    prompt: str = DEFAULT_PROMPT,
    max_generated_tokens: int = 128,
    generation_context: Optional[ModelContext] = None,
    use_cache: bool = False,
    cache_dir: str = DEFAULT_CACHE_DIR,
    combine_context_and_prompt: bool = True,
    progress: Union[bool, str] = True,
) -> Iterable[AnyDict]:
    """Evaluate a generic summarisation task, comparing model output against
    a reference using ROUGE score.

    See: "ROUGE: A Package for Automatic Evaluation of Summaries"
    https://aclanthology.org/W04-1013/

    Args:
        adapter (L.Adapter): Adapter wrapper for the LM model.
        examples (List[AnyDict]): Summarisation dataset containing:
          {context: str, id: str, reference: str}
        batch_size (int): Batch examples for greedy sample steps.
        prompt (str, optional): Defaults to "\nSummary:".
        max_generated_tokens (int, optional): Maximum generated summary length.
        Defaults to 128.
        generation_context (ModelContext, optional): Override model
        during generation.
        use_cache (bool, optional): Use cached context when sampling.
        Defaults to True.
        cache_dir (str, optional): Context cache path. Defaults to "cache".
        progress (bool|str, optional): enable tqdm (True) and set description
        (str), or disable (False).

    Yields:
        {
            id: int,
            output: str,
            rougeL: float,
            prefill_length: int,
            reference_length:int,
        }: For each input.
    """
    scorer = rouge_score.rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    for batch in tqdm(
        list(utility.batches(examples, batch_size)),
        desc=progress
        if isinstance(progress, str)
        else f"Evaluating {adapter.model.name_or_path}",
        disable=progress is False,
    ):
        reference_lengths = [len(adapter.tok_encode(b["reference"])) for b in batch]
        outputs = adapter.greedy_sample(
            [b["context"] for b in batch],
            [prompt for _ in batch],
            num_generated_tokens=min(max_generated_tokens, max(reference_lengths)),
            generation_context=generation_context,
            use_cache=use_cache,
            cache_dir=cache_dir,
            combine_context_and_prompt=combine_context_and_prompt,
        )
        for b, reference_length, output_ids in zip(
            batch, reference_lengths, list(outputs)
        ):
            output = adapter.tok_decode(output_ids[:reference_length])
            yield dict(
                id=b["id"],
                output=output,
                rougeL=scorer.score(b["reference"], output)["rougeL"].fmeasure,
                prefill_length=len(adapter.tok_encode(b["context"] + prompt)),
                reference_length=reference_length,
            )
