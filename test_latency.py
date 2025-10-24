import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import time
import numpy as np
from vq_method.retrieval_based.pq_search import initialize_objects, del_objects, wait
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.generation.utils import GenerationConfig
from vq_method.mistral_patch import VQMistralForCausalLM
from vq_method.llama31_patch import VQLlama31ForCausalLM
import torch.multiprocessing as mp
import pandas as pd
import tqdm
import torch
from loguru import logger
from vq_method.retrieval_based.global_timer import global_timer

SYNC_TEST_TIME = eval(os.environ.get("SYNC_TEST_TIME","0"))

def main():
    # model_path = "./pqcache/mistral-7b-Instruct-32k"
    model_path = "[MODEL_PATH]"
    print("Using mistral to profile")
    # print("Testing latency, make sure you are using llama3.1 model")
    try:
        config = AutoConfig.from_pretrained(model_path)
    except:
        raise Exception(f"Cannot find the model parameters directory: {model_path}")

    config.compress_ratio = 0.2
    config.recent_ratio = 0.5
    config.important_ratio = 0.5
    config.sink_size = 32
    config.compressor = "pq_search"
    config.n_subvec_per_head = 2
    config.n_subbits = 6
    os.environ["SUBVEC"] = f"{config.n_subvec_per_head}"
    os.environ["SUBBITS"] = f"{config.n_subbits}"
    os.environ["MODE"] = "off"
    config.topr = 1
    config.gqa = True
    config.mean_v_trick = False
    config.score_func = "sum"
    config.max_iter = 0
    config.pp_size = 1
    config.device = torch.device("cuda:0")
    
    config.max_seq_len = 32768
    config.cache_block_size = 128
    config.global_cache_size = 4096
    config.cache_topk = config.global_cache_size // config.cache_block_size

    with open("./test_input.txt", "r") as f:
        input_string = f.read()

    if config.compressor == "pq_search":
        initialize_objects(config, model=model_path)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast = True, config = config)
    model = VQMistralForCausalLM.from_pretrained(model_path, config=config, trust_remote_code=True)
    # tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast = True, config = config)
    # model = VQLlama31ForCausalLM.from_pretrained(model_path, config=config, trust_remote_code=True)
    model = model.half().eval()
    print("We are loading model", model_path)

    repeat_prompt = ",".join([input_string for _ in range(50)])
    prompt = f"[INST]{repeat_prompt}[/INST]"
    input_ = tokenizer(prompt, truncation=False, return_tensors="pt").to("cuda:0")

    if SYNC_TEST_TIME:
        beginning_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

    gen_max_token = 30
    for idx in range(4):
        if idx == 3:
            torch.cuda.cudart().cudaProfilerStart()
        for seqlen in tqdm.tqdm([4000, 8000, 12000, 16000, 20000, 24000]):
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
            wait()
            print(f"{output.flatten()[-1]} \r")
            end = time.perf_counter()
            ttft = end - begin
            
            time.sleep(3)
            
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

            # global_timer.transfer_time = 0
            global_timer.pq_compute_time = 0
            
            time.sleep(3)

            begin = time.perf_counter()
            if SYNC_TEST_TIME:
                beginning_event.record()

            output = model.generate(
                        input_ids=input_.input_ids[:, :seqlen],
                        attention_mask=None,
                        pad_token_id=tokenizer.eos_token_id,
                        max_new_tokens=gen_max_token, 
                        num_beams=1,
                        do_sample=False,
                        temperature=1.0,
                    )[0]
            if SYNC_TEST_TIME:
                end_event.record()
            print(f"{output.flatten()[-1]}, {output.shape}")
            end = time.perf_counter()
            decoding_elapsed = end - begin - tt2t
            print(f"Given input len is:{seqlen}, gen seq_len:{gen_max_token},"
                    f"ttft is {ttft},"
                    f"tt2t is {tt2t},"
                    f"decoding elasped:{decoding_elapsed},"
                    f"{decoding_elapsed / (gen_max_token - 2)} per decoding token.")
            if SYNC_TEST_TIME:
                pq, non_pq = global_timer.get_decode_time_parts(beginning_event, end_event)
                print("pq time elapsed is ", pq / 1000)
                print("non pq time elapsed is ", non_pq / 1000)
        if idx == 4:
            torch.cuda.cudart().cudaProfilerStop()
    del model
    if config.compressor == "pq_search":
        del_objects()
    logger.info(f"del objects done.")   
    exit()

if __name__ == "__main__":
    main()

