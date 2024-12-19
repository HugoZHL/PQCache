import torch
import torch.nn as nn
from kmeans_gpu import KMeans  # try kmeans on GPU
from typing import Optional, List, Tuple
import numpy as np
import math
import time
from .sparq_official.methods.ann_attention import AnnAttention, Settings
from flash_attn import flash_attn_func
import os
from .retrieval_based_compressor import *

# KV on CPU: used to profile latency
class SparQCompressor(RetrievalBasedCompressor):
    def __init__(self, compress, recent_ratio, sink, gqa, **kwargs) -> None:
        self.r = kwargs["r"]
        self.compress = compress 
        self.sink = sink
        self.local_ratio = recent_ratio
        self.mean_v = None
        self.decoding_time = 0
        self.decoding_count = 0
        self.idx = kwargs["idx"]
        self.model_config = kwargs["model_config"]
        self.mean_v_trick = self.model_config.mean_v_trick
        self.cpu_kvcache = None 
        
        self.official_ann_attn = None 
        self.GQA = gqa

        global D2HStream
        global H2DStream
        D2HStream = torch.cuda.Stream()
        H2DStream = torch.cuda.Stream()

        self.calc_event = torch.cuda.Event(blocking=True, interprocess=False)
        self.offload_event = torch.cuda.Event(blocking=True, interprocess=False)

        if self.idx <= 1:
            print(f"Initializing sparq compressor, using official codebase, GQA is {self.GQA}")
        super().__init__(**kwargs)
    
    def prefill_attn(
        self,
        query, 
        past_key_value: torch.Tensor,
        use_gpu = True
    ) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        key_states, value_states = past_key_value
        self.calc_event.record() # Mark that key_states and value_states is ready

        self.key_states = key_states
        self.value_states = value_states
        self.s_contiguous_key = key_states.transpose(2,3)
    
        self.cpu_kvcache = (torch.empty_like(key_states, device=torch.device("cpu"), pin_memory=True), 
                            torch.empty_like(value_states, device=torch.device("cpu"), pin_memory=True),
                            torch.empty_like(self.s_contiguous_key, device=torch.device("cpu"), pin_memory=True))

        bsz, kv_head, seq_len, dim = value_states.shape
        self.mean_v = value_states.unsqueeze(2).mean(dim=-2, keepdim=True).to(torch.float32)

        self.local_size = int(seq_len * self.compress * self.local_ratio)
        self.budget_size = int(seq_len * self.compress)
        official_setting = Settings(self.budget_size, self.local_size, self.sink, self.mean_v_trick, "sparse_q", rank=self.r)
        self.official_ann_attn = AnnAttention(official_setting, kv_head, dim, self.mean_v)

        attn_output = flash_attn_func(
            query.transpose(1,2),
            key_states.transpose(1,2),
            value_states.transpose(1,2),
            causal = True
        ).transpose(1,2)

        with torch.cuda.stream(D2HStream):
            self.calc_event.wait()
            self.cpu_kvcache[0].copy_(self.key_states, non_blocking = True)
            self.cpu_kvcache[1].copy_(self.value_states, non_blocking = True)
            self.cpu_kvcache[2].copy_(self.s_contiguous_key, non_blocking = True)
            self.offload_event.record(D2HStream)
        
        return attn_output, None

    def decoding_attn(self, num_key_value_groups: int,
                    Q, # bsz, n_heads, q_len, dim
                    repeat_K, repeat_V
                ):
        K, V = repeat_K[:,::num_key_value_groups,:,:], repeat_V[:,::num_key_value_groups,:,:]
        
        self.offload_event.wait()

        self.key_states = None # Prepare for gc
        self.value_states = None
        self.s_contiguous_key = None

        result = self.official_ann_attn(Q, K, V, 
                                        Q.new_zeros(*Q.shape[:-1], K.shape[2] + self.cpu_kvcache[0].shape[-2]), 
                                        self.cpu_kvcache)
        key_states = torch.concat([self.cpu_kvcache[0], K.cpu()], dim = -2)
        value_states = torch.concat([self.cpu_kvcache[1], V.cpu()], dim = -2)
        s_contiguous_key = torch.concat([self.cpu_kvcache[2], K.transpose(2,3).cpu()], dim = -1)

        self.cpu_kvcache = (key_states, value_states, s_contiguous_key)
        return result


# KV on GPU: Used to test accuracy
class SparQCompressorGPU(RetrievalBasedCompressor):
    def __init__(self, compress, recent_ratio, sink, gqa, **kwargs) -> None:
        self.r = kwargs["r"]
        self.compress = compress 
        self.sink = sink
        self.local_ratio = recent_ratio
        self.mean_v = None
        self.decoding_time = 0
        self.decoding_count = 0
        self.idx = kwargs["idx"]
        self.model_config = kwargs["model_config"]
        self.mean_v_trick = self.model_config.mean_v_trick
        self.gpu_kvcache = None 
        
        self.official_ann_attn = None 
        self.GQA = gqa

        global D2HStream
        global H2DStream
        D2HStream = torch.cuda.Stream()
        H2DStream = torch.cuda.Stream()

        self.calc_event = torch.cuda.Event(blocking=True, interprocess=False)
        self.offload_event = torch.cuda.Event(blocking=True, interprocess=False)

        if self.idx <= 1:
            print(f"Initializing sparq compressor, using official codebase, GQA is {self.GQA}")
        super().__init__(**kwargs)
    
    def prefill_attn(
        self,
        query, 
        past_key_value: torch.Tensor,
        use_gpu = True
    ) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        key_states, value_states = past_key_value

        self.gpu_kvcache = (key_states, value_states, key_states.transpose(2,3))

        bsz, kv_head, seq_len, dim = value_states.shape
        self.mean_v = value_states.unsqueeze(2).mean(dim=-2, keepdim=True).to(torch.float32)

        self.local_size = int(seq_len * self.compress * self.local_ratio)
        self.budget_size = int(seq_len * self.compress)
        official_setting = Settings(self.budget_size, self.local_size, self.sink, self.mean_v_trick, "sparse_q", rank=self.r)
        if np.random.randint(0,10000) % 500 == 1:
            print(f"Using sparq official code, mean v trick up? {official_setting.reallocate_to_mean_value}")
        self.official_ann_attn = AnnAttention(official_setting, kv_head, dim, self.mean_v)

        attn_output = flash_attn_func(
            query.transpose(1,2),
            key_states.transpose(1,2),
            value_states.transpose(1,2),
            causal = True
        ).transpose(1,2)
            
        return attn_output, None

    def decoding_attn(self, num_key_value_groups: int,
                    Q, # bsz, n_heads, q_len, dim
                    repeat_K, repeat_V
                ):
        K, V = repeat_K[:,::num_key_value_groups,:,:], repeat_V[:,::num_key_value_groups,:,:]
        
        result = self.official_ann_attn.forward_gpu_kv(Q, K, V, 
                                                        Q.new_zeros(*Q.shape[:-1], K.shape[2] + self.gpu_kvcache[0].shape[-2]), 
                                                        self.gpu_kvcache)
        key_states = torch.concat([self.gpu_kvcache[0], K], dim = -2)
        value_states = torch.concat([self.gpu_kvcache[1], V], dim = -2)
        s_contiguous_key = torch.concat([self.gpu_kvcache[2], K.transpose(2,3)], dim = -1)

        self.gpu_kvcache = (key_states, value_states, s_contiguous_key)
        return result

