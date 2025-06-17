import torch
from kmeans_gpu import KMeans as KMeans_gpu
from typing import Optional, List, Tuple
import numpy as np
from flash_attn import flash_attn_func
from .retrieval_based_compressor import *
import sys
import os.path as osp
from loguru import logger
import math

sys.path.append(osp.join(osp.abspath(osp.dirname(__file__)), "lfu/build"))
import lfucache
from .global_timer import global_timer

SYNC_TEST_TIME = eval(os.environ.get("SYNC_TEST_TIME","0"))

cache_class = lfucache.LFUCache

def init_gpu_cache_manager(**kwargs):
    cache_manager = GPUCacheManager(
        **kwargs,
    )
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

def create_event(device, interprocess=False):
    e = torch.cuda.Event(interprocess=interprocess)
    # A workaround to set the device of torch.cuda.Event()
    with torch.cuda.device(device):
        if interprocess:
            e.ipc_handle()
        else:
            e.record()
    return e

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
                cache_topk = -1,
                ) -> None:
        self.bsz, self.n_kv_head, self.dim = 1, n_kv_head, dim
        self.local_ratio, self.compress_ratio, self.global_cache_size = local_ratio, compress_ratio, global_cache_size
        self.max_idx = total_max_len
        self.device = device
        self.sink_size = sink_size
        self.layer_cnt = layer_cnt
        self.cache_block_size = cache_block_size
        self.D2HStream = torch.cuda.Stream(device=device)
        self.H2DStream = torch.cuda.Stream(device=device)
        self.defaultStream = torch.cuda.default_stream(device=device)
        if SYNC_TEST_TIME:
            self.transfer_default_starts = [torch.cuda.Event(enable_timing=True) for _ in range(layer_cnt)]
            self.transfer_default_ends = [torch.cuda.Event(enable_timing=True) for _ in range(layer_cnt)]
            self.transfer_other_starts = [torch.cuda.Event(enable_timing=True) for _ in range(layer_cnt)]
            self.transfer_other_ends = [torch.cuda.Event(enable_timing=True) for _ in range(layer_cnt)]
        if cache_topk < 0:
            self.cache_topk = global_cache_size / cache_block_size
        else:
            self.cache_topk = cache_topk

        # This memory layout can facilitate truly async data transfer
        base = torch.randn([1, total_max_len, n_kv_head, dim], 
                            dtype = dtype,
                            )
        self.cpu_key_buffers = [base.clone().share_memory_() for _ in range(layer_cnt)]
        pin_shm(self.cpu_key_buffers)
        
        # No need to share between procs.
        self.cpu_value_buffer = [torch.empty(
                                        [1, total_max_len, n_kv_head, dim], 
                                        dtype = dtype, 
                                        pin_memory=True
                                    ) for _ in range(layer_cnt)]
        
        self.indice_buffer = [torch.empty(
                                        [12800, self.n_kv_head], 
                                        dtype=torch.int64,
                                        pin_memory=True
                                    ) for _ in range(layer_cnt)]

        self.fetch_k_pin_buffer = [torch.empty(
                                            [12800 * n_kv_head, dim],
                                            dtype=dtype,
                                            pin_memory=True
                                        ) for _  in range(layer_cnt)]
        self.fetch_v_pin_buffer = [torch.empty(
                                            [12800 * n_kv_head, dim],
                                            dtype=dtype,
                                            pin_memory=True
                                        ) for _  in range(layer_cnt)]
        
        self.global_key_cache = torch.empty([self.layer_cnt, 1, self.global_cache_size, self.n_kv_head, self.dim], device=device, dtype=dtype)
        self.global_value_cache = torch.empty([self.layer_cnt, 1, self.global_cache_size, self.n_kv_head, self.dim], device=device, dtype=dtype)        
                                    
        self.max_block_cnt_perhead = (self.max_idx // cache_block_size)
        self.cache_block_cnt = global_cache_size // cache_block_size
        
        # A tool tensor to support us computing cache block hit count.
        self.block_idx_pad = torch.arange(self.n_kv_head, device=device, dtype=torch.int64) \
                                    .unsqueeze(1) \
                                    * self.max_block_cnt_perhead
        
        self.block_pos_record = torch.empty([layer_cnt, 1, self.max_block_cnt_perhead], dtype=torch.int32, pin_memory=True)
        self.block_pos_record_gpu = torch.empty([layer_cnt, 1, self.max_block_cnt_perhead], dtype=torch.int32, device=device)
        
        self.token_pos_record_gpu = torch.empty([layer_cnt, 1, self.max_block_cnt_perhead, self.cache_block_size], dtype=torch.int32, device=device)

        self.global_keys = [] # Preserve python object refs.
        self.global_values = []
        
        self.kv_ready_events = [create_event(device, False) for _ in range(layer_cnt)]
        self.offload_events = [create_event(device, True) for _ in range(layer_cnt)]
        self.kv_fetch_done_events = [create_event(device, False) for _ in range(layer_cnt)]
        self.cache_update_events = [create_event(device, False) for _ in range(layer_cnt)]

        self.reserved_tokens = []
        
        self.hi_scatter_idx_table = None
        
        print(f"Initializing cache manager with cache size: {global_cache_size}, cache_block size: {cache_block_size}, cache_topk {self.cache_topk}")
        
    def __del__(self):
        print("del invoked")
        unpin_shm(self.cpu_key_buffers)

    def refresh_config(self):
        # TODO: modify here.
        self.prefill_len = 0

    def init(self, key:torch.Tensor, value:torch.Tensor, layer_idx: int, topk_size: int):
        layer_idx = layer_idx % self.layer_cnt
        assert key.device != torch.device("cpu") and value.device != torch.device("cpu")

        if layer_idx == 0: # We only refresh key/value cache in the first layer
            self.reserved_tokens = []
            # Leave the very first area of gpu key/value buffer to sink token and local token.
            self.prefill_len = key.shape[-2]
            self.local_size = int((self.prefill_len - self.sink_size) * self.compress_ratio * self.local_ratio)
            self.topk_size = int((self.prefill_len - self.sink_size) * self.compress_ratio * (1 - self.local_ratio))

            self.global_token_cnt = self.prefill_len - self.local_size - self.sink_size

            self.topk_index = self.sink_size + self.local_size 
            self.total_budget = self.topk_size + self.sink_size + self.local_size + 1 # the last "1" for current new generated token
            
            # Compute buffer. It carrys tokens which are going to participate attention computation.
            self.key_buffer = self.global_key_cache.new_empty([self.layer_cnt, 1, self.n_kv_head, self.topk_index, self.dim])
            self.value_buffer = self.global_key_cache.new_empty([self.layer_cnt, 1, self.n_kv_head, self.topk_index, self.dim])
            self.k = self.global_key_cache.new_empty([1, self.n_kv_head, self.total_budget, self.dim])
            self.v = self.k.clone()

            self.local_to_evict_idx = 0
            self.offloaded_cnt = self.global_token_cnt

            self.block_pos_record[:] = -1
            self.block_pos_record_gpu.fill_(-1)
            self.caches = [cache_class(self.cache_block_cnt) for _ in range(self.layer_cnt)]

            self.global_keys = []
            self.global_values = []
            
            self.hit_scatter_idx_table = torch.hstack([
                                            (torch.arange(self.topk_size, dtype=torch.int64, device=key.device) + (h * self.total_budget + self.topk_index)) 
                                            for h in range(self.n_kv_head)
                                        ])
            self.miss_scatter_idx_table = torch.hstack([
                                            (torch.arange(-2,  - self.topk_size - 2, -1, dtype=torch.int64, device=key.device) + (h+1) * self.total_budget) 
                                            for h in range(self.n_kv_head)
                                        ])

        self.kv_ready_events[layer_idx].record(self.defaultStream)

        self.key_buffer[layer_idx,:,:,:self.local_size,:].copy_(key[...,-self.local_size:,:], non_blocking=True)
        self.value_buffer[layer_idx,:,:,:self.local_size,:].copy_(value[...,-self.local_size:,:], non_blocking=True)
        self.key_buffer[layer_idx,:,:,self.local_size : self.sink_size + self.local_size,:].copy_(key[...,:self.sink_size,:], non_blocking=True)
        self.value_buffer[layer_idx,:,:,self.local_size : self.sink_size + self.local_size,:].copy_(value[...,:self.sink_size,:], non_blocking=True)
        
        with torch.cuda.stream(self.D2HStream):
            self.kv_ready_events[layer_idx].wait(self.D2HStream)
            
            self.cpu_key_buffers[layer_idx][:,:self.global_token_cnt,:,:].copy_(key[..., self.sink_size: self.prefill_len - self.local_size,:].transpose(1,2), non_blocking=True)
            self.cpu_value_buffer[layer_idx][:,:self.global_token_cnt,:,:].copy_(value[..., self.sink_size: self.prefill_len - self.local_size,:].transpose(1,2), non_blocking=True)
            self.offload_events[layer_idx].record(self.D2HStream)

    def add_new_token(self, new_key, new_value, layer_idx):
        layer_idx = layer_idx % self.layer_cnt
        assert new_key.shape == (self.bsz, self.n_kv_head, 1, self.dim), new_key.shape

        self.to_evict_key = self.key_buffer[layer_idx,:,:,self.local_to_evict_idx,:]
        self.to_evict_value = self.value_buffer[layer_idx,:,:,self.local_to_evict_idx,:]
        self.key_buffer[layer_idx,:,:,self.local_to_evict_idx,:] = new_key.squeeze(2)
        self.value_buffer[layer_idx,:,:,self.local_to_evict_idx,:] = new_value.squeeze(2)

        self.cpu_key_buffers[layer_idx][:,self.offloaded_cnt,:,:].copy_(self.to_evict_key, non_blocking=True) 
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
    
    def get_qualified_blocks(self, block_indices:torch.Tensor):
        reduced_qualified_cnts = torch.bincount(block_indices.flatten(), minlength=self.max_block_cnt_perhead)
        return  torch.topk(
                            reduced_qualified_cnts, 
                            self.cache_topk,
                            dim = -1,
                            sorted=False
                        )
    
    def gpu_diff(self, cur_indices, layer_idx):
        layer_idx = layer_idx % self.layer_cnt
        token_pos_record = self.token_pos_record_gpu[layer_idx, 0]
                                                                    
        pos_arr_record = torch.gather(token_pos_record.flatten()[None,:].expand([self.n_kv_head, -1]), -1, cur_indices)
    
        block_indices = cur_indices // self.cache_block_size
        qualified_block_result = self.get_qualified_blocks(block_indices)
        
        on_gpu = torch.nonzero(pos_arr_record >= 0, as_tuple=True)
        off_gpu = torch.nonzero(pos_arr_record < 0, as_tuple=True)
        
        per_head_miss_cnt = torch.bincount(off_gpu[0], minlength=self.n_kv_head) # ([n_kv_head])
        per_head_hit_cnt = self.topk_size - per_head_miss_cnt
        
        on_gpu_pos = pos_arr_record[on_gpu] # [hit_cnt]
        off_gpu_token_idx = cur_indices[off_gpu] # [miss_cnt]
        
        to_fetch_tuple_idx = (off_gpu[0], off_gpu_token_idx)
        hit_tuple_idx = (on_gpu[0], on_gpu_pos)
        
        return to_fetch_tuple_idx, per_head_miss_cnt, hit_tuple_idx, per_head_hit_cnt, qualified_block_result
    
    def fetch_all_key_value(self, layer_idx, seq_len):
        key = self.cpu_key_buffers[layer_idx][:, :seq_len].cuda()
        value = self.cpu_value_buffer[layer_idx][:, :seq_len].cuda()
        return key, value

    # Only for debug.
    def fetch_and_concat_kv_wo_cache(self, indices: torch.Tensor, layer_idx):
        layer_idx = layer_idx % self.layer_cnt
        assert indices.shape == (self.n_kv_head, self.topk_size), f"{indices.shape}, {self.n_kv_head}, {self.topk_size}"
        assert indices.device != torch.device("cpu")

        indices = indices.cpu()
        self.offload_events[layer_idx].wait()

        indices.transpose_(0,1)
        selected_key = self.cpu_key_buffers[layer_idx].gather(1, indices[...,None].expand([1, -1, -1, 128]))
        selected_value = self.cpu_value_buffer[layer_idx].gather(1, indices[...,None].expand([1, -1, -1, 128]))

        # Optimize memory usage
        self.k[...,:self.topk_index,:].copy_(self.key_buffer[layer_idx], non_blocking=True)
        self.v[...,:self.topk_index,:].copy_(self.value_buffer[layer_idx], non_blocking=True)
        self.k[...,self.topk_index:-1,:] = selected_key.to(self.key_buffer.device).transpose(1,2)
        self.v[...,self.topk_index:-1,:] = selected_value.to(self.value_buffer.device).transpose(1,2)

        return self.k, self.v

    def fetch_and_concat_kv_w_cache(self, indices:torch.Tensor, layer_idx):
        layer_idx = layer_idx % self.layer_cnt
        assert indices.shape == (self.n_kv_head, self.topk_size), f"{indices.shape}, {self.n_kv_head}, {self.topk_size}"
        assert indices.device != torch.device("cpu")
        
        if np.random.randint(0, 10000) % 5000 == 1:
            logger.info("Using pq_search w cache.")
        
        # Optimize memory usage
        self.k[...,:self.topk_index,:].copy_(self.key_buffer[layer_idx], non_blocking=True)
        self.v[...,:self.topk_index,:].copy_(self.value_buffer[layer_idx], non_blocking=True)
        
        self.cache_update_events[layer_idx].wait()
        
        if self.global_cache_size > 0:
            # Select the "on gpu" token set and "not on gpu" token set
            # This api is done fully on gpu.
            to_fetch_idx, miss_cnt, hit_idx, hit_cnt, qualified_block_result = self.gpu_diff(indices, layer_idx) # 1ms
            assert len(to_fetch_idx) == 2 and len(hit_idx) == 2
            
            # What we need on cpu
            to_fetch_idx = (to_fetch_idx[0].cpu(), to_fetch_idx[1].cpu())
            miss_cnt = miss_cnt.cpu() 
            hit_cnt = hit_cnt.cpu()
            
            block2token_times, qualified_block_idx = qualified_block_result.values.cpu().tolist(), qualified_block_result.indices.cpu()
            
            last_valid_block_idx = self.offloaded_cnt // self.cache_block_size
            
            # Gather "on gpu" tokens using advanced indexing.
            selected_global_key = self.global_key_cache[(layer_idx, 0, hit_idx[1], hit_idx[0])] # [hit_cnt, dim]
            selected_global_value = self.global_value_cache[(layer_idx, 0, hit_idx[1], hit_idx[0])] # [hit_cnt, dim]
            
            hit_scatter_index = torch.concat([self.hit_scatter_idx_table[self.topk_size*h:self.topk_size*h + hit_cnt[h]] for h in range(self.n_kv_head)], dim=0)
            # Scatter "on gpu" tokens into compute buffer.
            self.k.view([self.n_kv_head * self.total_budget, self.dim]) \
                            .scatter_(-2, hit_scatter_index.unsqueeze(-1).expand_as(selected_global_key), selected_global_key)
            self.v.view([self.n_kv_head * self.total_budget, self.dim]) \
                            .scatter_(-2, hit_scatter_index.unsqueeze(-1).expand_as(selected_global_value), selected_global_value)
            
            fetched_token_cnt = to_fetch_idx[1].numel()
            self.fetch_k_pin_buffer[layer_idx][:fetched_token_cnt,:] = self.cpu_key_buffers[layer_idx][(0, to_fetch_idx[1], to_fetch_idx[0])]
            self.fetch_v_pin_buffer[layer_idx][:fetched_token_cnt,:] = self.cpu_value_buffer[layer_idx][(0, to_fetch_idx[1], to_fetch_idx[0])]

            if SYNC_TEST_TIME and global_timer.can_record():
                self.transfer_default_starts[layer_idx].record()

            fetch_global_key = self.fetch_k_pin_buffer[layer_idx][:fetched_token_cnt,:].to(self.device, non_blocking=True)
            fetch_global_value = self.fetch_v_pin_buffer[layer_idx][:fetched_token_cnt,:].to(self.device, non_blocking=True)

            if SYNC_TEST_TIME and global_timer.can_record():
                self.transfer_default_ends[layer_idx].record()
                self.transfer_default_ends[layer_idx].synchronize()
                global_timer.append_transfer_time_tuples(self.transfer_default_starts[layer_idx], self.transfer_default_ends[layer_idx])

            self.kv_fetch_done_events[layer_idx].record(self.defaultStream)

            miss_scatter_index = torch.concat([self.miss_scatter_idx_table[self.topk_size*h:self.topk_size*h + miss_cnt[h]] for h in range(self.n_kv_head)], dim=0)
            # Scatter "not on gpu" tokens into compute buffer.

            self.k.view([self.n_kv_head * self.total_budget, self.dim]) \
                            .scatter_(-2, miss_scatter_index.unsqueeze(-1).expand_as(fetch_global_key), fetch_global_key)
            self.v.view([self.n_kv_head * self.total_budget, self.dim]) \
                            .scatter_(-2, miss_scatter_index.unsqueeze(-1).expand_as(fetch_global_value), fetch_global_value)

            old_cache_buf_pos = torch.gather(self.block_pos_record[layer_idx, 0], -1, qualified_block_idx).tolist()
            
            q_b_idx_ = qualified_block_idx.tolist()
            # update lfu cache
            cache_obj = self.caches[layer_idx]
            
            selected_block_indices = []
            for i in range(len(q_b_idx_)):
                if block2token_times[i] > 0 and q_b_idx_[i] <= last_valid_block_idx:
                    selected_block_indices.append(q_b_idx_[i])
            
            cache_obj.BatchedInsertArray(
                np.array(selected_block_indices, dtype=np.int32), 
                self.block_pos_record[layer_idx,0].numpy()
            )

            new_cache_buf_pos = self.block_pos_record[layer_idx,0][selected_block_indices].tolist()
            
            with torch.cuda.stream(self.H2DStream):
                self.kv_fetch_done_events[layer_idx].wait(self.H2DStream)
                
                if SYNC_TEST_TIME and global_timer.can_record():
                    self.transfer_other_starts[layer_idx].record()
                
                for i in range(len(selected_block_indices)):
                    new_gpu_pos = new_cache_buf_pos[i]
                    old_gpu_pos = old_cache_buf_pos[i]
                    if old_gpu_pos == -1 and new_gpu_pos >= 0:
                        new_gpu_pos_offset = new_gpu_pos * self.cache_block_size
                        cpu_pos_offset = selected_block_indices[i] * self.cache_block_size
                        
                        self.global_key_cache[layer_idx, :, new_gpu_pos_offset:new_gpu_pos_offset+self.cache_block_size,:,:] \
                                .copy_(self.cpu_key_buffers[layer_idx][0, cpu_pos_offset:cpu_pos_offset+self.cache_block_size, :, :], non_blocking=True)
                                                    
                        self.global_value_cache[layer_idx, :, new_gpu_pos_offset:new_gpu_pos_offset+self.cache_block_size,:,:] \
                                .copy_(self.cpu_value_buffer[layer_idx][0, cpu_pos_offset:cpu_pos_offset+self.cache_block_size, :, :], non_blocking=True)
                    elif old_gpu_pos >= 0 and new_gpu_pos >= 0 and (old_gpu_pos != new_gpu_pos):
                        new_gpu_pos_offset = new_gpu_pos * self.cache_block_size
                        cpu_pos_offset = selected_block_indices[i] * self.cache_block_size
                        
                        self.global_key_cache[layer_idx, :, new_gpu_pos_offset:new_gpu_pos_offset+self.cache_block_size,:,:] \
                                .copy_(self.cpu_key_buffers[layer_idx][0, cpu_pos_offset:cpu_pos_offset+self.cache_block_size, :, :], non_blocking=True)
                                                    
                        self.global_value_cache[layer_idx, :, new_gpu_pos_offset:new_gpu_pos_offset+self.cache_block_size,:,:] \
                                .copy_(self.cpu_value_buffer[layer_idx][0, cpu_pos_offset:cpu_pos_offset+self.cache_block_size, :, :], non_blocking=True)

                self.block_pos_record_gpu[layer_idx, 0, :].copy_(self.block_pos_record[layer_idx, 0, :], non_blocking=True)
                self.token_pos_record_gpu[layer_idx, 0] = self.block_pos_record_gpu[layer_idx, 0][:,None].expand([-1, self.cache_block_size]) \
                                                                            * self.cache_block_size \
                                                                            + torch.arange(self.cache_block_size, device=self.device)
                if SYNC_TEST_TIME and global_timer.can_record():
                    self.transfer_other_ends[layer_idx].record()
                    global_timer.append_transfer_time_tuples(self.transfer_other_starts[layer_idx], self.transfer_other_ends[layer_idx])

                self.cache_update_events[layer_idx].record(self.H2DStream)
            
        else:
            indices = indices.cpu()
            indices = indices.unsqueeze(0).unsqueeze(3).expand([1, self.n_kv_head, self.topk_size, self.dim])
            to_fetch_key = torch.gather(self.cpu_key_buffers[layer_idx].transpose(1,2), dim = 2, index=indices)
            to_fetch_value = torch.gather(self.cpu_value_buffer[layer_idx].transpose(1,2), dim = 2, index=indices)
            self.k[...,-self.topk_size-1:-1].copy_(to_fetch_key, non_blocking=True)
            self.v[...,-self.topk_size-1:-1].copy_(to_fetch_value, non_blocking=True)    
        
        return self.k, self.v