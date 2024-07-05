import os
from datasets import load_dataset
import torch
import json
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from tqdm import tqdm
import numpy as np
import random
import argparse
from vq_method.llama_patch import VQLlamaForCausalLM
from vq_method.mistral_patch import VQMistralForCausalLM
from h2o_method.h2o_attention import H2OLlamaForCausalLM, H2OLlamaAttention
from vq_method.retrieval_based.pq_search import initialize_objects
import torch.distributed as dist
import torch.multiprocessing as mp
import time

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    compressor_choices = ["off", "original", "no_drop_lb", "pq_search","sparq_f"]
    parser.add_argument('--model', type=str, default=None, choices=[
        "llama-7b", "llama2-7b-chat-4k", "llama2-7b-32K", "mistral-7b-Instruct-32k", "longchat-v1.5-7b-32k",
        "xgen-7b-8k", "internlm-7b-8k", "chatglm2-6b", "chatglm2-6b-32k", "chatglm3-6b-32k", "vicuna-v1.5-7b-16k"])
    parser.add_argument('--e', action='store_true',
                        help="Evaluate on LongBench-E")
    parser.add_argument("--compress_ratio", type=float, default=1)
    parser.add_argument("--important_ratio", type=float, default=0)
    parser.add_argument("--recent_ratio", type=float, default=1)
    parser.add_argument('--enable_vq_cache', action='store_true')
    parser.add_argument('--enable_h2o_cache', action='store_true')
    parser.add_argument("--sink-size", type=int, default=32)
    parser.add_argument("--exp_name", type=str, default="dafault_exp")
    parser.add_argument("--compressor", type=str, default="off", choices=compressor_choices)
    parser.add_argument("--n_subvec_per_head", type=int, default=0)
    parser.add_argument("--n_subbits", type=int, default=0)
    parser.add_argument("--topr", type=int, default=32)
    parser.add_argument("--gqa", type=str, default="True")
    parser.add_argument("--sparq_mean_v_trick", type=str, default="False")
    parser.add_argument("--max_iter", type=int, default=0)
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument('--dp', action='store_true',help='whether data parallel')
    parser.add_argument('--pp-size', type=int, choices=[1,2,4,8])
    return parser.parse_args(args)

# This is the customized building prompt for chat models

def build_chat(tokenizer, prompt, model_name):
    if "chatglm3" in model_name:
        prompt = tokenizer.build_chat_input(prompt)
    elif "chatglm" in model_name:
        prompt = tokenizer.build_prompt(prompt)
    elif "longchat" in model_name or "vicuna" in model_name:
        from fastchat.model import get_conversation_template
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif "llama" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    elif "mistral" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    elif "xgen" in model_name:
        header = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
        )
        prompt = header + f" ### Human: {prompt}\n###"
    elif "internlm" in model_name:
        prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
    return prompt


def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response


def get_pred(args, model, tokenizer, rank, world_size, data, max_length, max_gen, prompt_format, dataset, model_name, model2path, out_path):
    data_idx = None
    # data_idx = 1
    
    for i, json_obj in tqdm(enumerate(data)):
        a = time.perf_counter()
        if data_idx is not None and i != (data_idx - 1):
            continue
        
        prompt = prompt_format.format(**json_obj)

        tokenized_prompt = tokenizer(
            prompt, truncation=False, return_tensors="pt").input_ids[0]
        if "chatglm3" in model_name:
            tokenized_prompt = tokenizer(
                prompt, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]
        original_token_cnt = len(tokenized_prompt) # 可能不准，因为可能要应用template。
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(
                tokenized_prompt[-half:], skip_special_tokens=True)
            original_token_cnt = max_length

        # chat models are better off without build prompts on these tasks
        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
            prompt = build_chat(tokenizer, prompt, model_name)
        if "chatglm3" in model_name:
            if dataset in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
                input = tokenizer(prompt, truncation=False,
                                  return_tensors="pt").to(device)
            else:
                input = prompt.to(device)
        else:
            input = tokenizer(prompt, truncation=False,
                              return_tensors="pt").to(device)
            
        context_length = input.input_ids.shape[-1]
        print("MAX_GEN", max_gen, "context_length", context_length)
        # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
        if dataset == "samsum":
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length+1,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.encode(
                    "\n", add_special_tokens=False)[-1]],
            )[0]
        else:
            output = model.generate(
                input_ids=input.input_ids,
                attention_mask=input.attention_mask,
                pad_token_id=tokenizer.eos_token_id,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
            )[0]

        if args.enable_h2o_cache:
            for name, m in model.named_modules():
                if isinstance(m, H2OLlamaAttention):
                    m._clean_cache()

        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        pred = post_process(pred, model_name)
        
        if data_idx is not None and i == (data_idx - 1):
            print("output:",pred)
            exit()
        
        if eval(os.environ.get("JUST_ONE_SAMPLE","False")):
            print(json_obj["answers"])
            exit()

        with open(out_path, "a", encoding="utf-8") as f:
            json.dump({"pred": pred, "answers": json_obj["answers"], 
                        "all_classes": json_obj["all_classes"], "length": json_obj["length"], 
                        "request_time": {"batch_time": 0, "batch_size": 1}, 
                        "input_tokens":int(original_token_cnt)}, f, ensure_ascii=False)
            f.write('\n')
        print(f"elapsed:{time.perf_counter()-a}")
    # dist.destroy_process_group()


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(args, path, model_name, device, pp_size = 1):
    if "chatglm" in model_name or "internlm" in model_name or "xgen" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            path, trust_remote_code=True, torch_dtype=torch.bfloat16)
        model = model.half().eval().to(device)
    elif "mistral" in model_name:
        config = AutoConfig.from_pretrained(path)
        config.recent_ratio = args.recent_ratio
        config.compress_ratio = args.compress_ratio
        config.important_ratio = args.important_ratio
        config.pp_size = pp_size
        config.sink_size = args.sink_size
        config.compressor = args.compressor
        config.n_subvec_per_head = args.n_subvec_per_head
        config.n_subbits = args.n_subbits
        config.topr = args.topr
        config.gqa = (args.gqa == "True")
        config.mean_v_trick = (args.sparq_mean_v_trick == "True")
        config.max_iter = args.max_iter
        config.device = torch.device("cuda:0")

        if config.compressor == "pq_search":
            config.max_seq_len = 36000
            config.cache_block_size = 128
            config.global_cache_size = 4096
            config.cache_topk = 32
            initialize_objects(config, model="mistral")

        tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True, config=config)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = VQMistralForCausalLM.from_pretrained(path, config=config)
        model = model.half().eval()
    elif "llama" in model_name:
        # replace_llama_attn_with_flash_attn()
        # tokenizer = LlamaTokenizer.from_pretrained(path)
        config = AutoConfig.from_pretrained(path)
        config.compress_ratio = args.compress_ratio
        config.important_ratio = args.important_ratio
        config.pp_size = pp_size
        config.sink_size = args.sink_size
        config.drop_ratio = args.drop_ratio
        config.preserve_layer = args.preserve_layer
        config.score_func = args.score_func
        config.compressor = args.compressor
        config.threshold = args.threshold
        config.n_subvec_per_head = args.n_subvec_per_head
        config.n_subbits = args.n_subbits
        config.topr = args.topr
        config.gqa = False
        config.max_iter = args.max_iter
        config.device = torch.device("cuda:0")
        if args.enable_vq_cache:
            config.compress_ratio = args.compress_ratio
            config.important_ratio = args.important_ratio
        elif args.enable_h2o_cache:
            config.hh_ratio = args.important_ratio
        config.recent_ratio = args.recent_ratio
        if config.compressor == "pq_search":
            config.max_seq_len = 32768
            config.cache_block_size = 128
            config.global_cache_size = 4096
            config.cache_topk = 32
            initialize_objects(config, model="mistral")
        tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
        if args.enable_vq_cache:
            model = VQLlamaForCausalLM.from_pretrained(path, config=config)
        elif args.enable_h2o_cache:
            model = H2OLlamaForCausalLM.from_pretrained(path, config=config)
        model = model.half().eval().to(device)
    elif "longchat" in model_name or "vicuna" in model_name:
        from fastchat.model import load_model
        replace_llama_attn_with_flash_attn()
        model, _ = load_model(
            path,
            device='cpu',
            num_gpus=0,
            load_8bit=False,
            cpu_offloading=False,
            debug=False,
        )
        model = model.bfloat16()
        tokenizer = AutoTokenizer.from_pretrained(
            path, trust_remote_code=True, use_fast=False)
        model = model.half().eval().to(device)
    
    return model, tokenizer

def get_config_str_list(args):
    if args.compressor in ["original"]:
        config_str = [
            "original"
        ]
    if args.compressor in ["pq_search","infllm"]:
        config_str = [
            f"budget_{args.compress_ratio}",
            f"rec_{args.recent_ratio}",
            f"sink_{args.sink_size}",
            f"mode_{args.compressor}",
            f"gqa_{args.gqa}",
            f"subvec_{args.n_subvec_per_head}",
            f"subbit_{args.n_subbits}",
            f"max_iter_{args.max_iter}"
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
    elif args.compressor in ["only_h2o_hw", "only_h2o_lw", "no_drop_vb"]:
        config_str = [
            f"threshold_{args.threshold}",
            f"score_{args.score_func}",
            f"mode_{args.compressor}",
            f"sink_{args.sink_size}",
        ]
    elif args.compressor in ["off","vq_h2o","no_drop_lb"]:
        config_str = [
            f"budget_{args.compress_ratio}",
            f"topk_{args.important_ratio}",
            f"rec_{args.recent_ratio}",
            f"sink_{args.sink_size}",
            f"mode_{args.compressor}",
            f"skip_layer_{args.preserve_layer}",
            f"score_{args.score_func}",
            f"gqa_{args.gqa}",
        ]
    return config_str


if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()
    assert args.enable_vq_cache + args.enable_h2o_cache == 1
    world_size = torch.cuda.device_count()
    model2path = json.load(open("config/model2path.json", "r"))
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    model_name = args.model
    # define your model
    max_length = model2maxlen[model_name]
    if args.e:
        datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news",
                    "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    else:
        datasets = ["gov_report", "multi_news","hotpotqa","2wikimqa","musique","multifieldqa_en","narrativeqa","qasper"]


    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    # predict on each dataset
    if not os.path.exists("pred"):
        os.makedirs("pred")
    if not os.path.exists("pred_e"):
        os.makedirs("pred_e")
        
    device = torch.device("cuda:0")
    model, tokenizer = load_model_and_tokenizer(args, model2path[model_name], model_name, device, args.pp_size)
    for dataset in datasets:
        if args.e:
            data = load_dataset('./data', f"{dataset}_e", split='test')
            if not os.path.exists(f"pred_e/{model_name}"):
                os.makedirs(f"pred_e/{model_name}")
            out_path = f"pred_e/{model_name}/{dataset}.jsonl"
        else:
            # data = load_dataset('./data', dataset, split='test')
            data = load_dataset('json', data_files='./data/' +
                                dataset+'.jsonl', split='train')
            exp_name = args.exp_name
            if not os.path.exists(f"pred/{model_name}"):
                os.makedirs(f"pred/{model_name}")
            if not os.path.exists(f"pred/{model_name}/{dataset}"):
                os.makedirs(f"pred/{model_name}/{dataset}")
            if not os.path.exists(f"pred/{model_name}/{dataset}/{exp_name}"):
                os.makedirs(f"pred/{model_name}/{dataset}/{exp_name}")
            config_str_list = get_config_str_list(args)
            if args.enable_h2o_cache:
                out_path = f"pred/{model_name}/{dataset}/{exp_name}/h2o_hh_{args.important_ratio}_recent_{args.recent_ratio}.jsonl"
            elif args.enable_vq_cache:
                out_path = f"pred/{model_name}/{dataset}/{exp_name}/{'_'.join(config_str_list)}.jsonl"

        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        data_all = [data_sample for data_sample in data]
        
        get_pred(args, model, tokenizer, 0, world_size, data_all, max_length, max_gen,
                    prompt_format, dataset, model_name, model2path, out_path)
