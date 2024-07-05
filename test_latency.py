import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import numpy as np
from vq_method.retrieval_based.pq_search import initialize_objects, del_objects
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.generation.utils import GenerationConfig
from vq_method.mistral_patch import VQMistralForCausalLM
from vq_method.llama_patch import VQLlamaForCausalLM
import torch.multiprocessing as mp
import pandas as pd
import tqdm
import torch
from loguru import logger

# NOTE: We use Llama2-7b to benchmark the latency.

def main():
    os.environ["SUBVEC"] = "2"
    os.environ["SUBBITS"] = "6"
    os.environ["MODE"] = "off"

    # model_path = "./pqcache/Llama-2-7b-chat-hf"
    model_path = "./pqcache/llama-32k"
    # model_path = "./pqcache/Mistral-32k"
    print("Testing latency, make sure you are using llama2 model")
    
    config = AutoConfig.from_pretrained(model_path)
    config.compress_ratio = 0.2
    config.recent_ratio = 0.5
    config.important_ratio = 0.5
    config.sink_size = 32
    config.compressor = "pq_search" # We only support 
    config.n_subvec_per_head = 2
    config.n_subbits = 6
    config.topr = 1
    config.gqa = True
    config.mean_v_trick = False
    config.score_func = "sum"
    config.max_iter = 0 # 3 or 4合适，再高就不行了。
    config.pp_size = 1
    config.device = torch.device("cuda:0")
    
    config.max_seq_len = 32768
    config.cache_block_size = 128
    config.global_cache_size = 4096
    config.lfu_cache_topk = 16

    df = pd.read_json("./nah_input.jsonl", lines=True) # 读取文件

    if config.compressor == "pq_search":
        initialize_objects(config, model=model_path)
    
    # tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast = True, config = config)
    # model = VQMistralForCausalLM.from_pretrained(model_path, config=config, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast = True, config = config)
    model = VQLlamaForCausalLM.from_pretrained(model_path, config=config, trust_remote_code=True)
    model = model.half().eval()

    print("We are loading model", model_path)

    for index, row in df.iterrows():
        input_string = row["prompt"]
        break

    repeat_prompt = ",".join([input_string for _ in range(200)])
    prompt = f"[INST]{repeat_prompt}[/INST]"
    input_ = tokenizer(prompt, truncation=False, return_tensors="pt").to("cuda:0")

    gen_max_token = 20
    for idx in range(5):
        for seqlen in tqdm.tqdm([2000, 4000, 8000, 16000]):
        # for seqlen in [2000, 4000]:
            begin = time.perf_counter()
            output = model.generate(
                        input_ids=input_.input_ids[:, :seqlen],
                        attention_mask=None,
                        pad_token_id=tokenizer.eos_token_id,
                        max_new_tokens=1, 
                        num_beams=1,
                        do_sample=False,
                        temperature=1.0,
                    )[0]
            print(f"{output.flatten()[-1]} \r")
            end = time.perf_counter()
            ttft = end - begin
            
            time.sleep(2)
            
            begin = time.perf_counter()
            output = model.generate(
                        input_ids=input_.input_ids[:, :seqlen],
                        attention_mask=None,
                        pad_token_id=tokenizer.eos_token_id,
                        max_new_tokens=2, 
                        num_beams=1,
                        do_sample=False,
                        temperature=1.0,
                    )[0]
            print(f"{output.flatten()[-1]} \r")
            end = time.perf_counter()
            tt2t = end - begin
            
            time.sleep(2)

            begin = time.perf_counter()
            output = model.generate(
                        input_ids=input_.input_ids[:, :seqlen],
                        attention_mask=None,
                        pad_token_id=tokenizer.eos_token_id,
                        max_new_tokens=gen_max_token, 
                        num_beams=1,
                        do_sample=False,
                        temperature=1.0,
                    )[0]
            print(f"{output.flatten()[-1]}")
            end = time.perf_counter()
            decoding_elapsed = end - begin - ttft
            print(f"Given input len is:{seqlen}, gen seq_len:{gen_max_token},"
                    f"ttft is {ttft},"
                    f"tt2t is {tt2t},"
                    f"decoding elasped:{decoding_elapsed},"
                    f"{decoding_elapsed / (gen_max_token - 1)} per decoding token.")
    del model
    if config.compressor == "pq_search":
        del_objects()
    logger.info(f"del objects done.")   
    exit()

if __name__ == "__main__":
    main()

