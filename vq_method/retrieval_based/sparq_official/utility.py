# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

"""Generic utilities"""

import copy
import datetime
import json
import multiprocessing
import os
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    TypeVar,
    Union,
    cast,
)

import datasets
import torch
import tqdm
import transformers
from torch import nn

T = TypeVar("T")


def batches(iterable: Iterable[T], batch_size: int) -> Iterable[List[T]]:
    """Chunks `iterable` into batches of consecutive values."""
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


AnyDict = Dict[str, Any]


def _sweep_runner(
    task: Callable[..., AnyDict], task_args: AnyDict, n_threads: int
) -> AnyDict:
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")
    torch.set_num_threads(n_threads)
    t0 = time.time()
    try:
        result = task(**task_args)
        result["_duration"] = time.time() - t0
        return result
    except Exception as error:
        return dict(
            _args={k: str(v) for k, v in task_args.items()},
            _error=repr(error),
            _error_tb=traceback.format_exception(
                type(error), error, error.__traceback__
            ),
            _duration=time.time() - t0,
        )


@contextmanager
def jsonlines_writer(path: Union[str, Path]) -> Iterator[Callable[[AnyDict], None]]:
    """Create a function that writes results to a jsonlines file.

    Fails if `path` already exists (so we don't accidentally overwrite results).
    """
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"File {path} already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yield lambda line: print(json.dumps(line), file=f, flush=True)


def run_multiprocess_sweep(
    task: Callable[..., AnyDict],
    settings: List[AnyDict],
    dest: Path,
    n_workers: int,
    max_threads_per_worker: int = 32,
) -> None:
    """Run a sweep in worker processes, saving the results to a .jsonl file."""
    if dest.exists():
        time = datetime.datetime.now().isoformat(timespec="seconds")
        dest = dest.parent / f"{dest.stem}-{time}{dest.suffix}"

    print(f"Sweeping {len(settings)} settings -> {dest}", file=sys.stderr)

    n_error = 0
    n_threads = min(max_threads_per_worker, cast(int, os.cpu_count()) // n_workers)
    with multiprocessing.Pool(n_workers, maxtasksperchild=1) as pool, jsonlines_writer(
        dest
    ) as write:
        results = tqdm.tqdm(
            [
                pool.apply_async(
                    _sweep_runner,
                    kwds=dict(task=task, task_args=s, n_threads=n_threads),
                )
                for s in settings
            ],
            ncols=120,
            miniters=1,
        )
        for n, result in enumerate(results):
            out = result.get()
            write(out)
            n_error += "_error" in out
            if n_error:
                results.set_description(f"{n_error}/{n+1} failed")
    print(
        f"Finished sweep ({n_error}/{n+1} failed) -> {dest}",
        file=sys.stderr,
    )


def map_full_batch(
    data: datasets.Dataset, fn: Callable[[List[AnyDict]], Iterable[AnyDict]]
) -> datasets.Dataset:
    """Transform a dataset with `fn`, mapping a list of input rows to output rows.

    Also, only return output columns from fn(), don't pass-through any original columns.
    """

    def mapper(batch: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        batch_size = len(next(iter(batch.values())))
        # Unstack the batch to call fn()
        rows_in = [{k: batch[k][i] for k in batch} for i in range(batch_size)]
        rows_out = list(fn(rows_in))
        # Restack the batch for the output
        return {k: [r[k] for r in rows_out] for k in rows_out[0]} if rows_out else {}

    return data.map(
        mapper, batched=True, batch_size=None, remove_columns=list(data.features)
    )


def map_and_filter(
    data: datasets.Dataset, fn: Callable[[AnyDict], Optional[AnyDict]]
) -> datasets.Dataset:
    """Like `Dataset.map`, but if the function returns None, filter out that row.

    Also, only return output columns from fn(), don't pass-through any original columns.
    """
    return map_full_batch(data, lambda rows: filter(None, map(fn, rows)))


def convert_module(
    model: nn.Module, replace: Callable[[nn.Module], Optional[nn.Module]]
) -> nn.Module:
    """Generic recursive module conversion."""

    def _convert(original: nn.Module) -> nn.Module:
        replacement = replace(original)
        if replacement is not None:
            replacement.to(next(original.parameters()).dtype)
            replacement.to(next(original.parameters()).device)
            replacement.load_state_dict(original.state_dict(), strict=False)
            return replacement

        # Recursive (lazy) copy
        result = original
        for name, child in original.named_children():
            replacement = _convert(child)
            if replacement is not child:
                if result is original:
                    result = copy.copy(original)
                    # Copy _modules, otherwise add_module() modifies `original`
                    result._modules = original._modules.copy()
                result.add_module(name, replacement)
        return result

    return _convert(model)


TRANSFORMERS_VERSION = "4.34.1"


def check_transformers_version(type: type) -> None:
    assert transformers.__version__ == TRANSFORMERS_VERSION, (
        f"{type.__name__} is version-locked to"
        f" transformers=={TRANSFORMERS_VERSION} for your safety"
    )
