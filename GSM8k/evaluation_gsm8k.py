import argparse
import json
import os
import logging
import re
import sys
import time
import torch
import numpy as np
import datasets
import transformers
from pathlib import Path
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm.auto import tqdm
from datasets import load_dataset
from typing import Any, Callable, Dict, Sequence, cast, List
from dataclasses import dataclass
from dataclasses_json import DataClassJsonMixin
from transformers import LlamaTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.models.llama.configuration_llama import LlamaConfig
from vq_method.llama_vq_attention import VQLlamaForCausalLM
from vq_method.mistral_patch import VQMistralForCausalLM
from h2o_method.h2o_attention import H2OLlamaForCausalLM, H2OLlamaAttention


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

MODEL_GENERATION_SPLIT = "\nQuestion: "
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvaluationSample:
    """Wrapper around format evaluation sample."""

    question: str
    generation: str
    answer: str
    list_from_pred: List[str]
    list_from_answer: List[str]
    pred: float
    label: float
    is_pred_true: bool


@dataclass(frozen=True)
class EvaluationMetrics(DataClassJsonMixin):
    """Wrapper around aggregated evaluation metrics."""

    accuracy: float


@dataclass(frozen=True)
class EvaluationResults(DataClassJsonMixin):
    """Wrapper around evaluation results"""

    samples: List[EvaluationSample]
    metrics: EvaluationMetrics


def evaluate_pred_answer(pred_str, ans_str):
    pattern = "\d*\.?\d+"
    pred_str, ans_str = pred_str.replace(",", ""), ans_str.replace(",", "")
    pred_list = re.findall(pattern, pred_str)
    gold_list = re.findall(pattern, ans_str)
    if len(pred_list) >= 1:
        pred = float(pred_list[-1])
        gold = float(gold_list[-1])
        is_pred_true = pred == gold
    else:
        is_pred_true = False
        pred = None
        gold = float(gold_list[-1])
    return (
        is_pred_true,
        pred,
        pred_list,
        gold,
        gold_list,
    )


def test_answer(pred_str, ans_str):
    pattern = "\d*\.?\d+"
    pred = re.findall(pattern, pred_str)
    if len(pred) >= 1:
        print("#####\n Pred string:", pred_str, "\n pred_list", pred)
        pred = float(pred[-1].replace(",", ""))
        gold = re.findall(pattern, ans_str)
        print("\n Gold_answer", ans_str, "\n gold_list", gold)
        gold = float(gold[-1].replace(",", ""))
        print("\n result", gold, pred, gold == pred)
        return pred == gold
    else:
        return False


def parse_pred_ans(filename):
    with open(filename) as fd:
        lines = fd.readlines()
    am, a = None, None
    num_q, acc = 0, 0
    current_mode = "none"
    questions = []
    ans_pred = []
    ans_gold = []
    am_others = []
    for l in lines:
        if l.startswith("Q: "):
            if am is not None and a is not None:
                questions.append(q)
                ans_pred.append(am)
                ans_gold.append(a)
                if test_answer(am, a):
                    acc += 1
            current_mode = "q"
            q = l
            num_q += 1
        elif l.startswith("A_model:"):
            current_mode = "am"
            am = l
        elif l.startswith("A:"):
            current_mode = "a"
            a = l
        # TODO
        elif current_mode == "am" and l.startswith("Question: "):
            current_mode = "am_other"
            am_other = l
        else:
            if current_mode == "q":
                q += l
            elif current_mode == "am":
                am += l
            elif current_mode == "a":
                a += l
            elif current_mode == "am_other":
                am_other += l
            else:
                raise ValueError(current_mode)

    questions.append(q)
    ans_pred.append(am)
    ans_gold.append(a)
    am_others.append(am_other)
    if test_answer(am, a):
        acc += 1
    print("######\n num_q %d correct %d ratio %.4f" % (num_q, acc, float(acc / num_q)))
    return questions, ans_pred, ans_gold


def get_config_str_list(args):
    if args.compressor in ["pq_search","infllm"]:
        config_str = [
            f"budget_{args.compress_ratio}",
            f"rec_{args.recent_ratio}",
            f"sink_{args.sink_size}",
            f"mode_{args.compressor}",
            f"gqa_{args.gqa}",
            f"subvec_{args.n_subvec_per_head}",
            f"subbit_{args.n_subbits}",
        ]
    if args.compressor in ["sparq","sparq_f"]:
        config_str = [
            f"budget_{args.compress_ratio}",
            f"rec_{args.recent_ratio}",
            f"sink_{args.sink_size}",
            f"mode_{args.compressor}",
            f"gqa_{args.gqa}",
            f"topr_{args.topr}",
            f"mean_v_trick_{args.sparq_mean_v_trick}"
        ]
    elif args.compressor in ["off","no_drop_lb"]:
        config_str = [
            f"budget_{args.compress_ratio}",
            f"topk_{args.important_ratio}",
            f"rec_{args.recent_ratio}",
            f"sink_{args.sink_size}",
            f"drop_{args.drop_ratio}",
            f"mode_{args.compressor}",
            f"skip_layer_{args.preserve_layer}",
            f"score_{args.score_func}",
            f"gqa_{args.gqa}",
        ]
    return config_str


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate GSM8K Dataset")
    compressor_choices = ["off", "no_drop_lb", "pq_search", "infllm", "sparq_f"]
    parser.add_argument("--model", type=str, default="mistral-7b-Instruct-32k", help="Model name or path.")
    parser.add_argument("--prompt_file", type=str, default="gsm8k_prompt_formal.txt", help="")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")
    parser.add_argument("--example_subset", type=str, default=None, help="")
    parser.add_argument("--max_length", type=int, default=None, help="")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="")
    parser.add_argument("--model_max_length", type=int, default=31500, help="")
    parser.add_argument("--sink-size", type=int, default=32)
    parser.add_argument("--do_sample", action="store_true", default=False, help="")
    parser.add_argument("--temperature", type=float, default=0.0, help="")
    parser.add_argument("--top_k", type=int, default=50, help="")
    parser.add_argument("--top_p", type=float, default=1.0, help="")
    parser.add_argument("--preserve_layer", type=int, default=0)
    parser.add_argument("--gqa", type=str, default="True")
    parser.add_argument("--exp_name", type=str, default="dafault_exp")
    parser.add_argument("--sparq_mean_v_trick", type=str, default="False")
    parser.add_argument("--generation_split", type=str, default=MODEL_GENERATION_SPLIT, help="")
    parser.add_argument("--score_func", type=str, default="sum")
    parser.add_argument("--compress_ratio", type=float, default=1)
    parser.add_argument("--important_ratio", type=float, default=0.5)
    parser.add_argument("--recent_ratio", type=float, default=0.5)
    parser.add_argument("--drop_ratio", type=float, default=0)
    parser.add_argument('--pp-size', type=int, choices=[1,2,4,8])
    parser.add_argument("--threshold", type=float, default=1)
    parser.add_argument("--topr", type=int, default=1)
    parser.add_argument("--compressor", type=str, default="off", choices=compressor_choices)
    parser.add_argument('--enable_h2o_cache', action='store_true')
    parser.add_argument('--enable_vq_cache', action='store_true')
    parser.add_argument("--n_subvec_per_head", type=int, default=2)
    parser.add_argument("--n_subbits", type=int, default=6)
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    args = parser.parse_args()

    # Setup output dir
    dataset = "gsm8k_test"
    model_name = args.model
    exp_name = args.exp_name
    eval_dataset = load_dataset('json', data_files='./GSM8k/'+dataset+'.jsonl', split='train')
    
    if not os.path.exists(f"pred/{model_name}/{dataset}"):
        os.makedirs(f"pred/{model_name}/{dataset}")
    if not os.path.exists(f"pred/{model_name}/{dataset}/{exp_name}"):
        os.makedirs(f"pred/{model_name}/{dataset}/{exp_name}")
    if not os.path.exists(f"pred/{model_name}/{dataset}/{exp_name}/generate"):
        os.makedirs(f"pred/{model_name}/{dataset}/{exp_name}/generate")
    if not os.path.exists(f"pred/{model_name}/{dataset}/{exp_name}/evaluate"):
        os.makedirs(f"pred/{model_name}/{dataset}/{exp_name}/evaluate")

    config_str_list = get_config_str_list(args)
    output_dir = Path(f"pred/{model_name}/{dataset}/{exp_name}")
    generation_file = Path(f"pred/{model_name}/{dataset}/{exp_name}/generate/{'_'.join(config_str_list)}.jsonl")
    evaluation_result_file = Path(f"pred/{model_name}/{dataset}/{exp_name}/evaluate/{'_'.join(config_str_list)}.json")
    
    logging.basicConfig(
        filename=os.path.join(output_dir.resolve(), "log.txt"),
        filemode="a",
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    # Load Model and Tokenizer
    if "mistral" in model_name:
        config = AutoConfig.from_pretrained("./Mistral-7B-Instruct")
        config.recent_ratio = args.recent_ratio
        config.compress_ratio = args.compress_ratio
        config.important_ratio = args.important_ratio
        config.sink_size = args.sink_size
        config.pp_size = args.pp_size
        config.drop_ratio = args.drop_ratio
        config.preserve_layer = args.preserve_layer
        config.score_func = args.score_func
        config.compressor = args.compressor
        config.threshold = args.threshold
        config.n_subvec_per_head = args.n_subvec_per_head
        config.n_subbits = args.n_subbits
        config.topr = args.topr
        config.gqa = True
        config.mean_v_trick = False
        print(config.compressor)
        tokenizer = AutoTokenizer.from_pretrained("./Mistral-7B-Instruct", padding_side="left", model_max_length=args.model_max_length, use_fast=False, config=config)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = VQMistralForCausalLM.from_pretrained("./Mistral-7B-Instruct", config=config)
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        model = model.half().eval().to(device)
    elif "llama" in model_name:
        config = AutoConfig.from_pretrained("./llama2-7b-chat-hf")
        config.compress_ratio = args.compress_ratio
        config.important_ratio = args.important_ratio
        config.sink_size = args.sink_size
        config.pp_size = args.pp_size
        config.preserve_layer = args.preserve_layer
        config.score_func = args.score_func
        config.compressor = args.compressor
        config.drop_ratio = args.drop_ratio
        print(config.compressor)
        config.threshold = args.threshold
        config.n_subvec_per_head = args.n_subvec_per_head
        config.n_subbits = args.n_subbits
        config.topr = args.topr
        config.gqa = False
        config.mean_v_trick = False
        config.recent_ratio = args.recent_ratio
        if args.enable_vq_cache:
            config.important_ratio = args.important_ratio
        elif args.enable_h2o_cache:
            config.hh_ratio = args.important_ratio
        
        tokenizer = AutoTokenizer.from_pretrained("./llama2-7b-chat-hf", padding_side="left", model_max_length=args.model_max_length, use_fast=False)
        tokenizer.pad_token = tokenizer.eos_token

        if args.enable_vq_cache:
            model = VQLlamaForCausalLM.from_pretrained("./llama2-7b-chat-hf", config=config)
        elif args.enable_h2o_cache:
            model = H2OLlamaForCausalLM.from_pretrained("./llama2-7b-chat-hf", config=config)

        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            
        model = model.half().eval().to(device)

    logging.info("Preprocessing the dataset.")
    with open(f"./GSM8k/{args.prompt_file}", "r") as handle:
        prompt_cot = handle.read()

    dataloader = torch.utils.data.DataLoader(
        cast(torch.utils.data.Dataset, eval_dataset),
        batch_size=args.batch_size,
    )

    acc_list = []
    all_samples = []
    all_question, all_generation, all_answer = [], [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluate GSM8K"):
            questions = batch["question"]
            answers = batch["answer"]
          
            prompts = [
                prompt_cot + "\nQuestion: " + question + "\n"
                for question in questions
            ]

            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding="longest",
                truncation=True,
            )
            inputs = inputs.to("cuda")

            generate_kwargs = dict(
                return_dict_in_generate=True,
                max_length=args.max_length,
                max_new_tokens=args.max_new_tokens,
                output_scores=True,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
            
            if args.do_sample:
                generate_kwargs["do_sample"] = True
                generate_kwargs["temperature"] = args.temperature
                generate_kwargs["top_k"] = args.top_k
                generate_kwargs["top_p"] = args.top_p
            else:
                generate_kwargs["do_sample"] = False
                generate_kwargs["temperature"] = 0
            
            outputs = model.generate(                
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask, 
                **generate_kwargs
            )

            generations = tokenizer.batch_decode(
                outputs.sequences[:, inputs.input_ids.shape[1] :],
                skip_special_tokens=True,
            )

            all_question += questions
            all_generation += generations
            all_answer += answers

            for question, generation, answer in zip(questions, generations, answers):
                is_pred_true, pred, pred_list, gold, gold_list = evaluate_pred_answer(
                    generation.split(args.generation_split)[0], answer
                )
                sample = EvaluationSample(
                    question=question,
                    generation=generation,
                    answer=answer,
                    list_from_pred=pred_list,
                    list_from_answer=gold_list,
                    pred=pred,
                    label=gold,
                    is_pred_true=is_pred_true,
                )
                all_samples.append(sample)

            acc_list.append(sample.is_pred_true)
            print('acc: {:.5f}'.format(np.mean(acc_list)))

        accuracy = sum([sample.is_pred_true for sample in all_samples]) / len(all_samples)
        evaluation_metric = EvaluationMetrics(accuracy=accuracy)
        evaluation_result = EvaluationResults(samples=all_samples, metrics=evaluation_metric)

    logging.info(f"Accuracy: {accuracy}")

    with evaluation_result_file.open("w") as handle:
        json.dump(evaluation_result.to_dict(), handle)

    with generation_file.open("w") as handle:
        for question, generation, answer in zip(all_question, all_generation, all_answer):
            handle.write("Q: %s\nA_model:\n%s\nA:\n%s\n\n" % (question, generation, answer))