# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

"""High-level experiment interface, supporting multiple tasks, models & logging."""

import dataclasses
import datetime
import logging
import multiprocessing
import multiprocessing.pool
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, cast

import torch
import tqdm
import wandb
from transformers import PreTrainedModel
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXForCausalLM
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.mistral.modeling_mistral import MistralForCausalLM

from . import eval_adapter, utility
from .methods import ann_attention, eviction_attention, sparse_attention
from .models import pipelined_models
from .tasks import bpc, qa, repetition, summarisation

# A dictionary of code changes that may affect the numbers, which is always
# logged alongside experiment results.
CODE_CHANGES: Dict[str, Any] = {
    "ann-local-token-for-free": True,
    "repetition-ignore-leading-space": True,
    "forced-sample-no-specials": True,
}

WANDB_PROJECT = "sparse-attention"
WANDB_URL = "https://wandb.sourcevertex.net"
logger = logging.getLogger(__name__)


# Configuration

TASKS = (
    "triviaqa",
    "squad",
    "squad_train",
    "cnn_dailymail",
    "wikitext_bpc",
    "repetition",
)
MODELS = (GPTNeoXForCausalLM, LlamaForCausalLM, MistralForCausalLM)


@dataclass
class Task:
    name: str  # TASKS
    shots: int
    samples: int
    confusion_contexts: int


@dataclass
class Sparsity:
    name: str  # method name from SparsityMethods
    # additional keys are passed to the sparsity method

    def __init__(self, name: str, **kwargs: Any):
        self.name = name
        self.__dict__.update(kwargs)

    def __str__(self) -> str:
        return str(self.__dict__)


@dataclass
class Execution:
    device: str
    dtype: str
    batch_size: int
    pipeline_stages: int
    wandb: Union[bool, str]  # False | True | "offline"

    @classmethod
    def auto(cls, batch_size: Optional[int] = None) -> "Execution":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipeline_stages = torch.cuda.device_count() if device == "cuda" else 1
        return cls(
            device=device,
            dtype=dict(cpu="float32", cuda="float16")[device],
            batch_size=batch_size or dict(cpu=10, cuda=5)[device],
            pipeline_stages=pipeline_stages,
            wandb=True,
        )


@dataclass
class Experiment:
    name: str
    task: Task
    model: str  # e.g. EleutherAI/pythia-1b
    sparsity: Sparsity
    execution: Execution
    # This shouldn't be specified directly, and is set via `CODE_CHANGES`
    code_changes: Dict[str, Any] = dataclasses.field(
        default_factory=lambda: CODE_CHANGES.copy()
    )

    def to_dict(self) -> Dict[str, Any]:
        def convert(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if hasattr(obj, "__dict__"):
                return convert(obj.__dict__)
            return obj

        return cast(Dict[str, Any], convert(self))


Outcome = Dict[str, Any]


# Method
class SparsityMethods:
    @classmethod
    def apply(cls, sparsity: Sparsity, model: PreTrainedModel) -> PreTrainedModel:
        method = getattr(cls, sparsity.name)
        return method(
            model, **{k: v for k, v in sparsity.__dict__.items() if k != "name"}
        )

    @staticmethod
    def dense(model: PreTrainedModel) -> PreTrainedModel:
        return model

    @staticmethod
    def sparse_v(model: PreTrainedModel, **settings: Any) -> PreTrainedModel:
        if isinstance(model, MistralForCausalLM):
            settings["kv_group_size"] = (
                model.config.num_attention_heads // model.config.num_key_value_heads
            )
        model.generation_context = eval_adapter.patch_for_model(
            "torch.nn.functional.softmax",
            sparse_attention.sparse_softmax_fixed_k,
            **settings,
        )
        return model

    @staticmethod
    def local(model: PreTrainedModel, **settings: Any) -> PreTrainedModel:
        model.generation_context = eval_adapter.patch_for_model(
            "torch.nn.functional.softmax",
            sparse_attention.local_softmax,
            **settings,
        )
        return model

    @staticmethod
    def eviction(model: PreTrainedModel, **settings: Any) -> PreTrainedModel:
        assert isinstance(model, MODELS)
        return eviction_attention.convert(
            model, eviction_attention.Settings(**settings)
        )

    @staticmethod
    def ann(model: PreTrainedModel, **settings: Any) -> PreTrainedModel:
        assert isinstance(model, MODELS)
        return ann_attention.convert(model, ann_attention.Settings(**settings))


# Running
def _evaluate(
    task: Task, adapter: eval_adapter.Adapter, batch_size: int, progress: bool
) -> Dict[str, Any]:
    if task.name in ["triviaqa", "squad", "squad_train"]:
        if task.name == "triviaqa":
            assert task.confusion_contexts == 0
            data = qa.TriviaQA.data(context="wiki")
        if task.name == "squad":
            data = qa.SQuAD.data(confusion_contexts=task.confusion_contexts)
        if task.name == "squad_train":
            data = qa.SQuAD.data(
                part="train", confusion_contexts=task.confusion_contexts
            )
        examples = [
            qa.add_few_shot_prompt(
                data[i],
                k=task.shots,
                prompt_template=qa.get_default_prompt_template(
                    adapter.model.config._name_or_path, task.shots
                ),
            )
            for i in range(task.samples)
        ]
        evaluate_fn: Any = qa.evaluate
    elif task.name == "cnn_dailymail":
        assert task.shots == 0 and task.confusion_contexts == 0
        data = summarisation.CnnDailymail.data()
        examples = [data[i] for i in range(task.samples)]
        evaluate_fn = summarisation.evaluate
    elif task.name == "wikitext_bpc":
        assert task.shots == 0 and task.confusion_contexts == 0
        data = bpc.WikiText.data()
        examples = [data[i] for i in range(task.samples)]
        evaluate_fn = bpc.evaluate
    elif task.name == "repetition":
        assert task.shots == 0 and task.confusion_contexts == 0
        data = repetition.Shakespeare.data()
        examples = [data[i] for i in range(task.samples)]
        evaluate_fn = repetition.evaluate
    else:
        raise ValueError(f"Task {task.name} not found")

    results = list(
        evaluate_fn(
            adapter=adapter,
            examples=examples,
            batch_size=batch_size,
            progress=progress,
        )
    )
    return dict(
        results=results,
        count=len(results),
        **{
            k: sum(x[k] for x in results) / len(results)
            for k in [
                "prefill_length",
                "reference_length",
                "match",
                "rougeL",
                "bpc",
                "match_length_char",
            ]
            if k in results[0]
        },
    )


def run_one(xp: Experiment, progress: bool = True) -> Outcome:
    """Run a single experiment, optionally logging to wandb."""
    if xp.execution.wandb:
        mode = "offline" if xp.execution.wandb == "offline" else "online"
        if mode == "online" and wandb.api.api_url != WANDB_URL:
            mode = "offline"
            logger.warning(
                f"Wandb not logged in to {WANDB_URL}; running in offline mode"
            )
        wandb.init(
            config=xp.to_dict(),
            mode=mode,
            entity="research",
            project=WANDB_PROJECT,
            reinit=True,
        )
    adapter = eval_adapter.Adapter.from_pretrained(
        xp.model, dtype=getattr(torch, xp.execution.dtype)
    )
    if xp.execution.pipeline_stages > 1:
        adapter.model = pipelined_models.pipeline_model(
            adapter.model, xp.execution.pipeline_stages
        )
    else:
        adapter.model.to(torch.device(xp.execution.device))

    adapter.model = SparsityMethods.apply(xp.sparsity, adapter.model)

    out = {}
    out["parameters"] = sum(p.nelement() for p in adapter.model.parameters())
    out["model_config"] = adapter.model.config.to_diff_dict()
    t0 = time.time()
    try:
        out.update(
            _evaluate(xp.task, adapter, xp.execution.batch_size, progress=progress)
        )
    except Exception as error:
        out["error"] = repr(error)
        out["backtrace"] = traceback.format_exc()
        logger.error(f"Error: {error}")
        logger.error(traceback.format_exc())
    out["duration"] = time.time() - t0
    if xp.execution.wandb:
        wandb.summary.update(out)
        assert wandb.run is not None
        out["wandb"] = dict(
            id=wandb.run.id, name=wandb.run.name, url=wandb.run.get_url()
        )
        wandb.finish(exit_code=1 if "error" in out else 0)
    return dict(**xp.to_dict(), **out)


def _run_many_task(xp: Experiment) -> Dict[str, Any]:
    """Run an experiment in a non-daemonic subprocess (for sake of wandb)."""
    queue: "multiprocessing.Queue[Dict[str, Any]]" = multiprocessing.Queue()
    num_threads = torch.get_num_threads()

    def _run() -> None:
        torch.set_num_threads(num_threads)
        queue.put(run_one(xp, progress=False))

    p = multiprocessing.get_context("fork").Process(target=_run, daemon=False)
    p.start()
    try:
        return queue.get()
    finally:
        p.join()


def run_many(
    xps: List[Experiment], n_workers: int = 1, out: Optional[Path] = None
) -> None:
    """Run multiple experiments, optionally as a multiprocess sweep."""
    if out is None:
        out = (
            Path("out") / datetime.datetime.now().isoformat(sep="/", timespec="seconds")
        ).with_suffix(".jsonl")

    with utility.jsonlines_writer(out) as writer:
        if n_workers >= 2:
            with multiprocessing.pool.ThreadPool(n_workers) as pool:
                for result in tqdm.tqdm(
                    pool.map(_run_many_task, xps), total=len(xps), desc="experiments"
                ):
                    writer(result)
        else:
            for xp in tqdm.tqdm(xps, desc="experiments"):
                writer(run_one(xp, progress=False))
