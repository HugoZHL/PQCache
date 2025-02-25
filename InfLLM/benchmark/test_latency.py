# Based on inf-llm codebase, make input text consistent with other baselines
from datasets import load_from_disk, load_dataset
import torch
import json
from tqdm import tqdm
import argparse
from omegaconf import OmegaConf
from inf_llm.utils import patch_hf, GreedySearch, patch_model_center
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import random
import os
import pandas as pd
import time

# NOTE: same as inf-llm
def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--output_dir_path", required=True)
    parser.add_argument("--datasets", type=str, default=None)
    parser.add_argument("--model_center", action="store_true", default=False)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--world_size", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    args, extra_args = parser.parse_known_args()
    conf = OmegaConf.load(args.config_path)
    cli_conf = OmegaConf.from_cli(extra_args)
    conf = OmegaConf.merge(conf, cli_conf)
    print(OmegaConf.to_yaml(conf))
    conf.output_dir_path = args.output_dir_path
    conf.model.model_center = args.model_center
    conf.rank = args.rank
    conf.world_size = args.world_size
    conf.verbose = args.verbose
    if not hasattr(conf.model, "tokenizer_path"):
        conf.model.tokenizer_path = conf.model.path
    if not hasattr(conf, "truncation"):
        conf.truncation = None

    datasets_str = args.datasets.strip().strip(",")
    datasets_list = datasets_str.split(",")
    conf.datasets = []
    for d in datasets_list:
        conf.datasets.append(d.strip())
    return conf

# NOTE: Change the dtype to float16 to do inference
def get_model_and_tokenizer(config):
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(config.path, torch_dtype=torch.float16, trust_remote_code=True, device_map="cuda")
    model = patch_hf(model, config.type, **config)
    return model, tokenizer

# NOTE: different from infllm
# 我们保证input id和其他baseline一致 and post process的逻辑与其他baseline一致即可
def get_pred(model, tokenizer, data, max_length,
    max_gen, prompt_format, dataset, model_name, 
    gen_chunk_size = None, truncation: str = None, 
    rank: int = None, world_size: int = None,
    verbose: bool = False
):  
    nah_file_path = "./pqcache/mistral_refactor/nah_input.jsonl"
    df = pd.read_json(nah_file_path, lines=True) # 如果空文件有可能出问题
    searcher = GreedySearch(model, tokenizer)

    # NOTE: check correctness
    for index, row in df.iterrows():
        input_string = row["prompt"]
        break

    prompt = f"[INST]{input_string}[/INST]"
    input_ = tokenizer(prompt, truncation=False, return_tensors="pt").to("cuda:0")
    output = searcher.generate(
                input_ids = input_.input_ids,
                max_length=128,
                chunk_size=gen_chunk_size,
                extra_end_token_ids=[]
            )
    print(f"Check correctness: {output[0]} \n")
    searcher.clear()

    
    prompt = "".join([input_string for _ in range(0,30)])

    prompt = f"[INST]{prompt}[/INST]"
    input_ = tokenizer(prompt, truncation=False, return_tensors="pt").to("cuda:0")

    gen_max_token = 20
    for idx in range(5):
        for seqlen in [2000, 4000, 8000, 16000]:
            if idx == 4 and seqlen == 8000:
                torch.cuda.cudart().cudaProfilerStart()
            begin = time.perf_counter()
            output = searcher.generate(
                        input_ids = input_.input_ids[:,:seqlen],
                        max_length=1,
                        chunk_size=gen_chunk_size,
                        extra_end_token_ids=[]
                    )
            print(f"{output[0][-1]} \r")
            end = time.perf_counter()
            ttft = end - begin
            searcher.clear()

            # -------------------------------

            begin = time.perf_counter()
            output = searcher.generate(
                        input_ids = input_.input_ids[:,:seqlen],
                        max_length=2,
                        chunk_size=gen_chunk_size,
                        extra_end_token_ids=[]
                    )
            print(f"{output[0][-1]} \r")
            end = time.perf_counter()
            tt2t = end - begin
            searcher.clear()

            # -------------------------------

            begin = time.perf_counter()
            output = searcher.generate(
                        input_ids = input_.input_ids[:,:seqlen],
                        max_length=gen_max_token,
                        chunk_size=gen_chunk_size,
                        extra_end_token_ids=[]
                    )
            print(f"{output[0][-1]} \r")
            end = time.perf_counter()

            decoding_elapsed = end - begin - ttft
            print(f"Given input len is:{seqlen}, gen seq_len:{gen_max_token},"
                    f"ttft is {ttft},"
                    f"tt2t is {tt2t},"
                    f"decoding elasped:{decoding_elapsed},"
                    f"{decoding_elapsed / (gen_max_token - 1)} per decoding token.")
            searcher.clear()
            if idx == 4 and seqlen == 8000:
                torch.cuda.cudart().cudaProfilerStop()

# NOTE: change the way of loading longbench dataset
if __name__ == '__main__':
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # define your model
    model, tokenizer = get_model_and_tokenizer(args.model)

    preds = get_pred(
            model, tokenizer, None, 
            args.max_len, 128, 
            None, None, 
            args.conv_type, 
            args.chunk_size, args.truncation,
            args.rank, args.world_size,
            args.verbose
        )
