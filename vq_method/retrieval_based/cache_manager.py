from multiprocessing.managers import SharedMemoryManager
from einops import reduce
import torch
import torch.multiprocessing as mp
from kmeans_gpu import KMeans as KMeans_gpu  # try kmeans on GPU
from typing import Optional, List, Tuple
import numpy as np
from flash_attn import flash_attn_func
from .retrieval_based_compressor import *
import sys
import os.path as osp
from loguru import logger
import math

sys.path.append(osp.join(osp.abspath(osp.dirname(__file__)), "lfu/build"))
# print(sys.path)
import lfucache

cache_class = lfucache.LFUCache

def init_gpu_cache_manager(**kwargs):
    global D2HStream, H2DStream
    D2HStream = torch.cuda.Stream()
    H2DStream = torch.cuda.Stream()

    mp.set_start_method("spawn", force=True)
    cache_manager = GPUCacheManager(
        **kwargs
    )
    # We only support single GPU inference right now.
    logger.info("Cache manager init done. !!!!!!Warning you're running system in refactor mode!")
    return cache_manager

def pin_shm(bufs: List[torch.Tensor]):
    # Copy from https://gist.github.com/colesbury/aae4361feb596353fcd219339d6c44ab
    cudart = torch.cuda.cudart()
    for buf in bufs:
        err = cudart.cudaHostRegister(buf.data_ptr(), buf.numel() * buf.element_size(), 0)
        assert buf.is_shared()
        assert buf.is_pinned()
        assert err == 0, err

def unpin_shm(bufs: List[torch.Tensor]):
    # Copy from https://gist.github.com/colesbury/aae4361feb596353fcd219339d6c44ab
    lib = torch.cuda.cudart()
    for buf in bufs:
        err = lib.cudaHostUnregister(buf.data_ptr())
        assert err == 0, err

class GPUCacheManager:
    def __init__(self, 
                layer_cnt, 
                n_kv_head, 
                total_max_len, 
                dim, 
                device, 
                dtype,
                compress_ratio, 
                local_ratio, 
                sink_size,
                global_cache_size,
                cache_block_size,
                cache_topk = -1
                ) -> None:
        self.bsz, self.n_kv_head, self.dim = 1, n_kv_head, dim
        self.local_ratio, self.compress_ratio, self.global_cache_size = local_ratio, compress_ratio, global_cache_size
        self.max_idx = total_max_len
        self.device = device
        self.sink_size = sink_size
        self.layer_cnt = layer_cnt
        self.cache_block_size = cache_block_size
        if cache_topk < 0:
            self.cache_topk = global_cache_size / cache_block_size
        else:
            self.cache_topk = cache_topk

        # This memory layout can facilitate truly async data transfer
        self.cpu_key_buffer = [torch.empty(
                                        [1, total_max_len, n_kv_head, dim], 
                                        dtype = dtype,
                                    ).share_memory_() for _ in range(layer_cnt)]
        pin_shm(self.cpu_key_buffer)
        
        # No need to share between procs.
        self.cpu_value_buffer = [torch.empty(
                                        [1, total_max_len, n_kv_head, dim], 
                                        dtype = dtype, 
                                        pin_memory=True
                                    ) for _ in range(layer_cnt)]
        
        self.indice_buffer = [torch.empty(
                                        [10000, self.n_kv_head], 
                                        dtype=torch.int64,
                                        pin_memory=True
                                    ) for _ in range(layer_cnt)]


        # We will at most fetch 6400 kv vecs per head.
        self.fetch_k_pin_buffer = [torch.empty(
                                            [6400 * n_kv_head, dim],
                                            dtype=dtype,
                                            pin_memory=True
                                        ) for _  in range(layer_cnt)]
        self.fetch_v_pin_buffer = [torch.empty(
                                            [6400 * n_kv_head, dim],
                                            dtype=dtype,
                                            pin_memory=True
                                        ) for _  in range(layer_cnt)]
        
        # TODO: Mem layout改变了
        self.global_key_cache = torch.empty([self.layer_cnt, 1, self.global_cache_size, self.n_kv_head, self.dim], device=device, dtype=dtype)
        self.global_value_cache = torch.empty([self.layer_cnt, 1, self.global_cache_size, self.n_kv_head, self.dim], device=device, dtype=dtype)        
                                    
        self.max_block_cnt_perhead = (self.max_idx // cache_block_size)
        self.cache_block_cnt = global_cache_size // cache_block_size
        
        # A tool tensor to support us computing cache block hit count.
        self.block_idx_pad = torch.arange(self.n_kv_head, device=device, dtype=torch.int64) \
                                    .unsqueeze(1) \
                                    * self.max_block_cnt_perhead
        
        # self.block_pos_record = np.empty([layer_cnt, 1, self.max_block_cnt_perhead], dtype=torch.int32)
        self.block_pos_record = torch.empty([layer_cnt, 1, self.max_block_cnt_perhead], dtype=torch.int32, pin_memory=True)
        self.block_pos_record_gpu = torch.empty([layer_cnt, 1, self.max_block_cnt_perhead], dtype=torch.int32, device=device)
        
        self.global_keys = [] # Preserve python object refs.
        self.global_values = []
        
        self.kv_ready_events = [torch.cuda.Event() for _ in range(layer_cnt)]
        self.offload_events = [torch.cuda.Event(interprocess=True) for _ in range(layer_cnt)]
        self.cache_update_events = [torch.cuda.Event() for _ in range(layer_cnt)]
        
        self.hi_scatter_idx_table = None
        
        print(f"Initializing cache manager with cache size: {global_cache_size}, cache_block size: {cache_block_size}, cache_topk {self.cache_topk}")
        
    def __del__(self):
        print("del invoked")
        unpin_shm(self.cpu_key_buffer)

    def refresh_config(self):
        # TODO: modify here.
        self.prefill_len = 0

    def init(self, key:torch.Tensor, value:torch.Tensor, layer_idx: int, topk_size: int):
        assert key.device != torch.device("cpu") and value.device != torch.device("cpu")

        if layer_idx == 0: # We only refresh key/value cache in the first layer
            # Leave the very first area of gpu key/value buffer to sink token and local token.
            self.prefill_len = key.shape[-2]
            self.local_size = int((self.prefill_len - self.sink_size) * self.compress_ratio * self.local_ratio)
            self.topk_size = int((self.prefill_len - self.sink_size) * self.compress_ratio * (1 - self.local_ratio))

            self.global_token_cnt = self.prefill_len - self.local_size - self.sink_size

            self.topk_index = self.sink_size + self.local_size 
            self.total_budget = self.topk_size + self.sink_size + self.local_size + 1 # the last "1" for current new generated token
            
            # Compute buffer. It carry tokens which are going to participate attention computation.
            self.key_buffer = self.global_key_cache.new_empty([self.layer_cnt, 1, self.n_kv_head, self.total_budget, self.dim])
            self.value_buffer = self.global_key_cache.new_empty([self.layer_cnt, 1, self.n_kv_head, self.total_budget, self.dim])

            self.local_to_evict_idx = 0
            self.offloaded_cnt = self.global_token_cnt

            self.block_pos_record[:] = -1
            self.block_pos_record_gpu.fill_(-1)
            self.caches = [cache_class(self.cache_block_cnt) for _ in range(self.layer_cnt)]

            self.global_keys = []
            self.global_values = []
            
            self.hit_scatter_idx_table = torch.hstack([
                                            (torch.arange(self.topk_size, dtype=torch.int64) + (h * self.total_budget + self.topk_index)) 
                                            for h in range(self.n_kv_head)
                                        ])
            self.miss_scatter_idx_table = torch.hstack([
                                            (torch.arange(-2,  - self.topk_size - 2, -1, dtype=torch.int64) + (h+1) * self.total_budget) 
                                            for h in range(self.n_kv_head)
                                        ]).to(self.device, non_blocking=True)

        self.global_keys.append(key[..., self.sink_size: self.prefill_len - self.local_size,:].transpose(1,2))
        self.global_values.append(value[..., self.sink_size: self.prefill_len - self.local_size,:].transpose(1,2))
        self.kv_ready_events[layer_idx].record()

        self.key_buffer[layer_idx,:,:,:self.local_size,:].copy_(key[...,-self.local_size:,:], non_blocking=True)
        self.value_buffer[layer_idx,:,:,:self.local_size,:].copy_(value[...,-self.local_size:,:], non_blocking=True)
        self.key_buffer[layer_idx,:,:,self.local_size : self.sink_size + self.local_size,:].copy_(key[...,:self.sink_size,:], non_blocking=True)
        self.value_buffer[layer_idx,:,:,self.local_size : self.sink_size + self.local_size,:].copy_(value[...,:self.sink_size,:], non_blocking=True)
            
        with torch.cuda.stream(D2HStream):
            self.kv_ready_events[layer_idx].wait(D2HStream)
            self.cpu_key_buffer[layer_idx][:,:self.global_token_cnt,:,:].copy_(self.global_keys[layer_idx], non_blocking=True)
            self.cpu_value_buffer[layer_idx][:,:self.global_token_cnt,:,:].copy_(self.global_values[layer_idx], non_blocking=True)
            self.offload_events[layer_idx].record(D2HStream)

    def add_new_token(self, new_key, new_value, layer_idx):
        assert new_key.shape == (self.bsz, self.n_kv_head, 1, self.dim), new_key.shape

        self.to_evict_key = self.key_buffer[layer_idx,:,:,self.local_to_evict_idx,:]
        self.to_evict_value = self.value_buffer[layer_idx,:,:,self.local_to_evict_idx,:]
        self.key_buffer[layer_idx,:,:,self.local_to_evict_idx,:] = new_key.squeeze(2)
        self.value_buffer[layer_idx,:,:,self.local_to_evict_idx,:] = new_value.squeeze(2)

        self.cpu_key_buffer[layer_idx][:,self.offloaded_cnt,:,:].copy_(self.to_evict_key, non_blocking=True) 
        self.cpu_value_buffer[layer_idx][:,self.offloaded_cnt,:,:].copy_(self.to_evict_value, non_blocking=True)
        
        if layer_idx == 0:
            self.offloaded_cnt += 1
            self.local_to_evict_idx = (self.local_to_evict_idx + 1) % self.local_size
        
        return self.to_evict_key

    def gpu_trick_diff(self, cur_indices, cache_indices):
        target = torch.zeros([self.n_kv_head, self.max_idx], dtype=torch.int8, device = cur_indices.device)
        on_gpu_idx = torch.zeros([self.n_kv_head, self.max_idx],  dtype=torch.int8, device = cur_indices.device)
        off_gpu_idx = ~on_gpu_idx
        target.scatter_(-1, cur_indices, 1)
        on_gpu_idx.scatter_(-1, cache_indices, 1)
        off_gpu_idx = ~on_gpu_idx
        to_fetch_idx  = torch.nonzero(target & off_gpu_idx, as_tuple=True)
        hit_idx = torch.nonzero(target & on_gpu_idx, as_tuple=True)
        return to_fetch_idx, hit_idx
    
    def get_qualified_blocks(self, block_indices):
        block_indices_pad = block_indices + self.block_idx_pad
        
        max_possible_block_idx = self.n_kv_head * self.max_block_cnt_perhead
        block_qualified_cnts = torch.bincount(block_indices_pad.flatten(), minlength=max_possible_block_idx) 
    
        reduced_qualified_cnts = block_qualified_cnts.reshape([self.n_kv_head, self.max_block_cnt_perhead]).sum(dim = 0)
        return  torch.topk(
                            reduced_qualified_cnts, 
                            self.cache_topk,
                            dim = -1
                        ) # [topk]
    
    def gpu_diff(self, cur_indices, layer_idx):
        # cur_indices shape [self.n_kv_head, self.topk_size]
        # pos_record shape [layer_cnt, 1, n_kv_head, total_max_len]
        token_pos_record = self.block_pos_record_gpu[layer_idx, 0][:,None].expand([-1, self.cache_block_size]) \
                                                                            * self.cache_block_size \
                                                                            + torch.arange(self.cache_block_size, device=self.device)
                                                                    
        pos_arr_record = torch.gather(token_pos_record.flatten()[None,:].expand([self.n_kv_head, -1]), -1, cur_indices) # mixed with "real idx" and -1
    
        block_indices = cur_indices // self.cache_block_size
        qualified_block_result = self.get_qualified_blocks(block_indices)
        
        on_gpu = torch.nonzero(pos_arr_record >= 0, as_tuple=True)
        off_gpu = torch.nonzero(pos_arr_record < 0, as_tuple=True)
        
        per_head_miss_cnt = torch.bincount(off_gpu[0], minlength=self.n_kv_head) # ([n_kv_head])
        per_head_hit_cnt = self.topk_size - per_head_miss_cnt
        
        assert per_head_miss_cnt.numel() == self.n_kv_head
        
        on_gpu_pos = pos_arr_record[on_gpu] # [hit_cnt]
        off_gpu_token_idx = cur_indices[off_gpu] # [miss_cnt]
        
        to_fetch_tuple_idx = (off_gpu[0], off_gpu_token_idx)
        hit_tuple_idx = (on_gpu[0], on_gpu_pos)
        
        return to_fetch_tuple_idx, per_head_miss_cnt, hit_tuple_idx, per_head_hit_cnt, qualified_block_result
    
    # Only for debug.
    def fetch_and_concat_kv_wo_cache(self, indices: torch.Tensor, layer_idx):
        assert indices.shape == (self.n_kv_head, self.topk_size), f"{indices.shape}, {self.n_kv_head}, {self.topk_size}"
        assert indices.device != torch.device("cpu")

        indices = indices.cpu()
        self.offload_events[layer_idx].wait()

        indices.transpose_(0,1)
        selected_key = self.cpu_key_buffer[layer_idx].gather(1, indices[...,None].expand([1, -1, -1, 128]))
        selected_value = self.cpu_value_buffer[layer_idx].gather(1, indices[...,None].expand([1, -1, -1, 128]))

        self.key_buffer[layer_idx,:,:,self.topk_index:-1,:] = selected_key.to(self.key_buffer.device).transpose(1,2)
        self.value_buffer[layer_idx,:,:,self.topk_index:-1,:] = selected_value.to(self.value_buffer.device).transpose(1,2)

        return self.key_buffer[layer_idx], self.value_buffer[layer_idx]
        
    def fetch_and_concat_kv_w_cache(self, indices:torch.Tensor, layer_idx):
        assert indices.shape == (self.n_kv_head, self.topk_size), f"{indices.shape}, {self.n_kv_head}, {self.topk_size}"
        assert indices.device != torch.device("cpu")
        
        if np.random.randint(0, 10000) % 5000 == 1:
            logger.info("Using pq_search w cache.")
        
        self.cache_update_events[layer_idx].wait()
        
        # Select the "on gpu" token set and "not on gpu" token set
        to_fetch_idx, miss_cnt, hit_idx, hit_cnt, qualified_block_result = self.gpu_diff(indices, layer_idx)
        assert len(to_fetch_idx) == 2 and len(hit_idx) == 2
        
        # What we need on cpu
        to_fetch_idx = (to_fetch_idx[0].cpu(), to_fetch_idx[1].cpu())
        miss_cnt = miss_cnt.cpu() 

        block2token_times, qualified_block_idx = qualified_block_result.values.cpu(), qualified_block_result.indices.cpu()
        
        last_valid_block_idx = self.offloaded_cnt // self.cache_block_size
        
        with torch.cuda.stream(H2DStream):
            old_cache_buf_pos = torch.gather(self.block_pos_record[layer_idx, 0], -1, qualified_block_idx)
            
            q_b_idx_ = qualified_block_idx
            # update lfu cache
            cache_obj = self.caches[layer_idx]
            
            # 去除：hit token计数为0的block以及off-band的block
            q_b_idx_ = q_b_idx_[block2token_times > 0]
            selected_block_indices = q_b_idx_[q_b_idx_ <= last_valid_block_idx]
            
            cache_obj.BatchedInsertArray(
                np.ascontiguousarray(selected_block_indices.to(torch.int32).numpy()), 
                self.block_pos_record[layer_idx,0].numpy() 
            )
            
            new_cache_buf_pos = self.block_pos_record[layer_idx,0][selected_block_indices]
            # update cache buffer 
            for i in range(selected_block_indices.shape[0]):
                new_gpu_pos = new_cache_buf_pos[i]
                old_gpu_pos = old_cache_buf_pos[i]
                if old_gpu_pos == -1 and new_gpu_pos >= 0:
                    new_gpu_pos_offset = new_gpu_pos * self.cache_block_size
                    cpu_pos_offset = selected_block_indices[i] * self.cache_block_size
                    
                    self.global_key_cache[layer_idx, :, new_gpu_pos_offset:new_gpu_pos_offset+self.cache_block_size,:,:] \
                            .copy_(self.cpu_key_buffer[layer_idx][0, cpu_pos_offset:cpu_pos_offset+self.cache_block_size, :, :], non_blocking=True)
                                                
                    self.global_value_cache[layer_idx, :, new_gpu_pos_offset:new_gpu_pos_offset+self.cache_block_size,:,:] \
                            .copy_(self.cpu_value_buffer[layer_idx][0, cpu_pos_offset:cpu_pos_offset+self.cache_block_size, :, :], non_blocking=True)
                elif old_gpu_pos >= 0 and new_gpu_pos >= 0 and (old_gpu_pos != new_gpu_pos):
                    new_gpu_pos_offset = new_gpu_pos * self.cache_block_size
                    cpu_pos_offset = selected_block_indices[i] * self.cache_block_size
                    
                    self.global_key_cache[layer_idx, :, new_gpu_pos_offset:new_gpu_pos_offset+self.cache_block_size,:,:] \
                            .copy_(self.cpu_key_buffer[layer_idx][0, cpu_pos_offset:cpu_pos_offset+self.cache_block_size, :, :], non_blocking=True)
                                                
                    self.global_value_cache[layer_idx, :, new_gpu_pos_offset:new_gpu_pos_offset+self.cache_block_size,:,:] \
                            .copy_(self.cpu_value_buffer[layer_idx][0, cpu_pos_offset:cpu_pos_offset+self.cache_block_size, :, :], non_blocking=True)

            
            self.block_pos_record_gpu[layer_idx, 0, :].copy_(self.block_pos_record[layer_idx, 0, :], non_blocking=True)
            
            self.cache_update_events[layer_idx].record(H2DStream)
        
        # Gather "on gpu" tokens using advanced indexing.
        selected_global_key = self.global_key_cache[(layer_idx, 0, hit_idx[1], hit_idx[0])] # [hit_cnt, dim]
        selected_global_value = self.global_value_cache[(layer_idx, 0, hit_idx[1], hit_idx[0])] # [hit_cnt, dim]
        
        hit_cnt = hit_cnt.cpu()
        hit_scatter_index = torch.hstack([
                                    (torch.arange(hit_cnt[h], dtype=torch.int64) + (h * self.total_budget + self.topk_index)) 
                                    for h in range(self.n_kv_head)
                                ]).to(self.device)
        
        # Scatter "on gpu" tokens into compute buffer.
        self.key_buffer[layer_idx].view([self.n_kv_head * self.total_budget, self.dim]) \
                        .scatter_(-2, hit_scatter_index.unsqueeze(-1).expand_as(selected_global_key), selected_global_key)
        self.value_buffer[layer_idx].view([self.n_kv_head * self.total_budget, self.dim]) \
                        .scatter_(-2, hit_scatter_index.unsqueeze(-1).expand_as(selected_global_value), selected_global_value)
        
        self.global_keys[layer_idx] = None # Let python gc free the cuda mem space.
        self.global_values[layer_idx] = None 

        fetched_token_cnt = to_fetch_idx[1].numel()
        self.fetch_k_pin_buffer[layer_idx][:fetched_token_cnt,:] = self.cpu_key_buffer[layer_idx][(0, to_fetch_idx[1], to_fetch_idx[0])]
        self.fetch_v_pin_buffer[layer_idx][:fetched_token_cnt,:] = self.cpu_value_buffer[layer_idx][(0, to_fetch_idx[1], to_fetch_idx[0])]

        fetch_global_key = self.fetch_k_pin_buffer[layer_idx][:fetched_token_cnt,:].to(self.device, non_blocking=True)
        fetch_global_value = self.fetch_v_pin_buffer[layer_idx][:fetched_token_cnt,:].to(self.device, non_blocking=True)

        # Prepare index for "not on gpu" tokens to scatter into compute buffer.
        miss_scatter_index = torch.hstack([
                                    (torch.arange(-2,  - int(miss_cnt[h]) - 2, -1, dtype=torch.int64) + (h+1) * self.total_budget) 
                                    for h in range(self.n_kv_head)
                                ]).to(self.device, non_blocking=True)

        # Scatter "not on gpu" tokens into compute buffer.
        self.key_buffer[layer_idx].view([self.n_kv_head * self.total_budget, self.dim]) \
                        .scatter_(-2, miss_scatter_index.unsqueeze(-1).expand_as(fetch_global_key), fetch_global_key)
        self.value_buffer[layer_idx].view([self.n_kv_head * self.total_budget, self.dim]) \
                        .scatter_(-2, miss_scatter_index.unsqueeze(-1).expand_as(fetch_global_value), fetch_global_value)
        
        return self.key_buffer[layer_idx], self.value_buffer[layer_idx]