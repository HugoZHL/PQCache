import os
import time
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch.multiprocessing as mp
from vq_method.mistral_patch import VQMistralForCausalLM
from transformers.generation.utils import GenerationConfig
from tqdm import tqdm
import pandas as pd
import faulthandler
faulthandler.enable()
import numpy as np
from vq_method.retrieval_based.pq_search import initialize_objects


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    os.environ["SUBVEC"] = "2"
    os.environ["SUBBITS"] = "6"
    os.environ["TOKENIZERS_PARALLELISM"]="false"

    initialize_objects()
    mp.set_start_method("spawn", force=True)
    model_path = "mistralai/Mistral-7B-Instruct-v0.2"

    def save_data(df, output_file, columns=[]):
        if type(columns) is list and len(columns) > 0:
            save_df = df[columns]
        else:
            save_df = df
        if output_file.endswith(".xlsx"):
            save_df.to_excel(output_file, index=False)
        elif output_file.endswith(".csv"):
            save_df.to_csv(output_file)
        elif output_file.endswith(".jsonl"):
            save_df.to_json(output_file, orient='records', lines=True, force_ascii=False)
        else:
            save_df.to_json(output_file + ".jsonl", orient='records', lines=True, force_ascii=False)

    nah_input = []
    nah_file_path = "./nah_input.jsonl"
    df = pd.read_json(nah_file_path, lines=True) 

    # torch.cuda.memory._record_memory_history()
    config = AutoConfig.from_pretrained(model_path)
    config.compress_ratio = 0.1
    config.recent_ratio = 0.5
    config.important_ratio = 0.5
    config.sink_size = 32
    config.compressor = "pq_search"
    config.n_subvec_per_head = 2
    config.n_subbits = 6
    config.topr = 1
    config.gqa = True
    config.mean_v_trick = False
    # useless config
    config.score_func = "sum"
    config.threshold = 10000
    config.preserve_layer = 0
    config.pp_size = 1
    config.max_iter = 10
    tokenizer = AutoTokenizer.from_pretrained(model_path, 
                                                use_fast=True, 
                                                config = config)
    model = VQMistralForCausalLM.from_pretrained(model_path,
                                                config=config,
                                                trust_remote_code=True)
    model = model.half().eval()
    for index, row in df.iterrows():
        input_string = row["prompt"]
        prompt = f"[INST]{input_string}[/INST]"
        input_ = tokenizer(prompt, truncation=False, return_tensors="pt").to("cuda:0")
        context_length = input_.input_ids.shape[-1]
        output = model.generate(
                    input_ids=input_.input_ids,
                    attention_mask=input_.attention_mask,
                    pad_token_id=tokenizer.eos_token_id,
                    max_new_tokens=128, 
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                )[0]
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        df.loc[index, "response"] = pred

    print("Generation done, dumping result....")
    os.makedirs(f"./bc_eval/eval_result_mistral/{config.compressor}/needlehaystack")
    save_data(df, f"./bc_eval/eval_result_mistral/{config.compressor}/needlehaystack/response.jsonl")

if __name__ == "__main__":
    main()
