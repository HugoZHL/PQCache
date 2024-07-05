# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

"""Methods for evaluation on question answering tasks.

For example:

    data = qa.TriviaQA.data()
    results = list(qa.evaluate(
        adapter,
        [qa.add_zero_shot_prompt(data[i]) for i in range(10)],
        batch_size=5,
        open_book=False,
    ))
    print(results[0])  # => {"id": ..., "output": ..., "match": True|False}

    em_accuracy = sum(r["match"] for r in results) / len(results)
    print(em_accuracy)
"""

import collections
from functools import partial
from typing import Any, Iterable, List, Optional, Tuple, Union

import datasets
import numpy as np
import regex as re
from tqdm import tqdm

from ..eval_adapter import DEFAULT_CACHE_DIR, Adapter, ModelContext
from ..utility import AnyDict, batches, map_full_batch


class TriviaQA:
    @staticmethod
    def make_example(row: AnyDict) -> AnyDict:
        return dict(
            id=row["question_id"],
            question=row["question"],
            answers=row["answer"]["aliases"],
        )

    @classmethod
    def preprocess(
        cls,
        rows: List[AnyDict],
        examples: int = 5,
        chars_range: Tuple[int, int] = (4000, 8000),
        seed: int = 2751584,
        context: str = "search",
    ) -> Iterable[AnyDict]:
        """Generate a few-shot generative evaluation dataset from TriviaQA.

        Yields: {"id": str,
                 "question": str,
                 "answers": [str],
                 "context": str,
                 "examples": [{"id": str, "question": str, "answers": [str]}]}
        """
        if context not in ["search", "wiki"]:
            raise ValueError(
                f"{context} is not a valid value for context"
                ", set to either 'search' or 'wiki'."
            )
        rng = np.random.RandomState(seed)
        for row in rows:
            # Take a random context doc that has an in-range length
            contexts = (
                row["entity_pages"]["wiki_context"]
                if context == "wiki"
                else row["search_results"]["search_context"]
            )
            filtered_contexts = [
                c for c in contexts if chars_range[0] <= len(c) <= chars_range[1]
            ]
            if filtered_contexts:
                # Ensure we never use this question as a few-shot example
                # (note that each question_id can appear twice in TriviaQA, hence +2)
                few_shot_examples = [
                    cls.make_example(rows[i])
                    for i in rng.choice(len(rows), examples + 2)
                    if rows[i]["question_id"] != row["question_id"]
                ][:examples]
                doc_idx = rng.randint(len(filtered_contexts))
                yield dict(
                    **cls.make_example(row),
                    context=filtered_contexts[doc_idx],
                    examples=few_shot_examples,
                )

    @classmethod
    def data(
        cls, shuffle_seed: int = 487523, **preprocess_args: Any
    ) -> datasets.Dataset:
        return map_full_batch(
            datasets.load_dataset("trivia_qa", name="rc")["validation"],
            partial(cls.preprocess, **preprocess_args),
        ).shuffle(shuffle_seed)


class SQuAD:
    @staticmethod
    def format_context(row: AnyDict) -> str:
        return f"Title: {row['title'].replace('_', ' ')}. Background: {row['context']}"

    @staticmethod
    def make_example(row: AnyDict) -> AnyDict:
        return dict(
            id=row["id"],
            question=row["question"],
            answers=row["answers"]["text"],
        )

    @classmethod
    def preprocess(
        cls,
        rows: List[AnyDict],
        confusion_contexts: int = 7,
        examples: int = 5,
        seed: int = 82735235,
    ) -> Iterable[AnyDict]:
        """Generate a few-shot generative evaluation dataset from SQuAD.

        Yields: {"id": str,
                 "context": str,
                 "question": str,
                 "answers": [str],
                 "examples": [{"id": str, "question": str, "answers": [str]}]}
        """
        # Note: some care is required to keep this deterministic, given `seed` and
        # the order of incoming `rows`
        rng = np.random.RandomState(seed)
        all_examples = [cls.make_example(row) for row in rows]
        context_to_examples = collections.defaultdict(list)
        for example, row in zip(all_examples, rows):
            context_to_examples[row["context"]].append(example)
        all_contexts = sorted({cls.format_context(row) for row in rows})

        for row in rows:
            # Find `confusion_contexts` other contexts, shuffle & concatenate
            contexts = [cls.format_context(row)] + [
                all_contexts[i]
                for i in rng.choice(len(all_contexts), confusion_contexts)
            ]
            rng.shuffle(contexts)
            full_context = (
                "\n".join(contexts)
                + f"\nFrom what you've just read about {row['title'].replace('_', ' ')}"
                + ", please answer the following questions."
            )
            # Find `examples` other example questions, starting with the same
            # context, but then broadening the search if there aren't enough.
            same_context_examples = [
                c for c in context_to_examples[row["context"]] if c["id"] != row["id"]
            ]
            rng.shuffle(same_context_examples)  # type:ignore[arg-type]
            other_examples = [
                all_examples[i]
                for i in rng.choice(len(all_examples), examples + 1)
                if all_examples[i]["id"] != row["id"]
            ]
            yield dict(
                **cls.make_example(row),
                context=full_context,
                examples=(same_context_examples + other_examples)[:examples],
            )

    @classmethod
    def data(
        cls,
        shuffle_seed: int = 945237,
        part: str = "validation",
        **preprocess_args: Any,
    ) -> datasets.Dataset:
        return map_full_batch(
            datasets.load_dataset("squad")[part],
            partial(cls.preprocess, **preprocess_args),
        ).shuffle(shuffle_seed)


def get_default_prompt_template(config_name_or_path: str, shots: int) -> str:
    """Get a default prompt template for this model class & k-shot evaluation."""
    if shots == 0 and ("pythia" in config_name_or_path):
        # This prompt performed better for Pythia, preventing it from returning
        # unmatched outputs such as "the answer is <correct answer>"
        return "Question: {question}\nSingle-word answer:"
    return "Question: {question}\nAnswer:"


def add_few_shot_prompt(
    datum: AnyDict, k: int, prompt_template: str, answer_template: str = " {answer}"
) -> AnyDict:
    """Add a key "prompt" to the returned dictionary, following `prompt_template`.

    Args:
        datum (AnyDict): Single example from the dataset, containing:
            {"id": str,
             "question": str,
             "answers": List[str],
             "examples": List[{"question": str, "answers": [str]}]}
        k (int): Number of k-shot examples to provide.
        prompt_template (str): String template e.g. "Q: {question}, A:".
        answer_template (str): String template e.g. " {answer}", used for k-shot
        examples only.

    Returns:
        {**datum, "prompt": str}: As `datum` but with a formatted "prompt".
    """
    examples = datum.get("examples", [])
    if len(examples) < k:
        raise ValueError(
            f"Cannot form a (k={k})-shot prompt for question {datum['id']}"
            f", which only provides {len(examples)} examples"
        )
    prompt = "\n".join(
        [
            prompt_template.format(**eg)
            + answer_template.format(answer=min(eg["answers"], key=len))
            for eg in examples[:k]
        ]
        + [prompt_template.format(**datum)]
    )
    return dict(**datum, prompt=prompt)


# Strip leading & trailing space/punctuation, and leading "the " etc.
EVALUATE_NORMALISATION_PATTERN = re.compile(
    r"""^[\s\n"'_.,]*(the |a |an )?|[\s\n"'_.,]*$""",
)


def evaluate_prediction(out: str, answers: List[str]) -> bool:
    norm_out = re.sub(EVALUATE_NORMALISATION_PATTERN, "", out.lower())
    norm_answers = [
        re.sub(EVALUATE_NORMALISATION_PATTERN, "", answer.lower()) for answer in answers
    ]
    return any(norm_out.startswith(answer) for answer in norm_answers if answer)


def evaluate(
    adapter: Adapter,
    examples: List[AnyDict],
    batch_size: int,
    output_token_limit: int = 30,
    output_spare_tokens: int = 5,
    open_book: bool = True,
    generation_context: Optional[ModelContext] = None,
    use_cache: bool = False,
    cache_dir: str = DEFAULT_CACHE_DIR,
    combine_context_and_prompt: bool = True,
    progress: Union[bool, str] = True,
) -> Iterable[AnyDict]:
    """Evaluate a generic QA task consisting of a list of examples, each a
    dictionary with keys "context", "prompt", and "answers".

    Args:
        adapter (L.Adapter): Adapter wrapper for the LM model
        examples (List[AnyDict]): QA dataset containing:
          {context: str, prompt: str, id: str, answers: List[str]}
        batch_size (int): Batch examples for greedy sample steps
        output_token_limit (int, optional): Defaults to 30.
        output_spare_tokens (int, optional): Defaults to 5.
        open_book (bool, optional): Prepend context to the prompt.
        Defaults to True.
        generation_context (ModelContext, optional): Override model
        during generation.
        use_cache (bool, optional): Use cached context when sampling.
        Defaults to True.
        cache_dir (str, optional): Context cache path. Defaults to "cache".
        progress (bool|str, optional): enable tqdm (True) and set description
        (str), or disable (False).

    Yields:
        {"id": int, "output": str, "match": bool}: One for each input
    """
    for batch in tqdm(
        list(batches(examples, batch_size)),
        desc=progress
        if isinstance(progress, str)
        else f"Evaluating {adapter.model.name_or_path}",
        disable=progress is False,
    ):
        max_answer_tokens = max(
            len(adapter.tok_encode(a)) for x in batch for a in x["answers"]
        )
        out_ids_batch = adapter.greedy_sample(
            [x["context"] + "\n" if open_book else "" for x in batch],
            [x["prompt"] for x in batch],
            num_generated_tokens=min(
                output_token_limit, output_spare_tokens + max_answer_tokens
            ),
            generation_context=generation_context,
            use_cache=use_cache and open_book,
            cache_dir=cache_dir,
            combine_context_and_prompt=combine_context_and_prompt,
        )
        for x, ids in zip(batch, out_ids_batch):
            output = adapter.tok_decode(ids)
            yield dict(
                id=x["id"],
                output=output,
                match=evaluate_prediction(output, x["answers"]),
                prefill_length=len(
                    adapter.tok_encode(
                        (x["context"] if open_book else "\n") + x["prompt"]
                    )
                ),
            )
