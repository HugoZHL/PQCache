from multiprocessing.managers import SharedMemoryManager
import os
import torch
import torch.multiprocessing as mp
from kmeans_gpu import KMeans as KMeans_gpu  # try kmeans on GPU
from typing import Optional, List, Tuple
import numpy as np
import math
import time
from .sparq_official.methods.ann_attention import MistralAttentionWithANN, Settings
from flash_attn import flash_attn_func

from .retrieval_based_compressor import *
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans as KMeans_sklr
from .multi_core_compressor_v2 import MultiCoreCompressor_v2
import sys
import os.path as osp
from .cache_manager import init_gpu_cache_manager
from loguru import logger
from .global_timer import global_timer

CHECK_RECALL = eval(os.environ.get("CHECK_RECALL", "0"))
SYNC_TEST_TIME = eval(os.environ.get("SYNC_TEST_TIME","0"))

pq_compute_time = 0

# All those configs are based on mistral model architecture.
# TODO: Only init those two object for master process.
def initialize_objects(config, model):
    global global_compressor
    global cache_managers
    global total_layer_num, pp_size, layer_per_rank
    
    global H2DStream
    H2DStream = torch.cuda.Stream()
    
    MAX_CPU_IN_USE=int(eval(os.environ.get("MAX_CPU_IN_USE", "32")))
    MAX_WORKER_CNT=64

    cache_managers = []
    cpu_key_bufs = []
    offload_events = []

    total_layer_num = config.num_hidden_layers
    pp_size = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    layer_per_rank = total_layer_num // pp_size

    for rank in range(pp_size):
        cache_manager = init_gpu_cache_manager(
                                        layer_cnt = config.num_hidden_layers // pp_size,
                                        n_kv_head = config.num_key_value_heads,
                                        total_max_len = config.max_seq_len,
                                        dim = config.hidden_size // config.num_attention_heads,
                                        device = torch.device(f"cuda:{rank}"),
                                        dtype = torch.float16,
                                        compress_ratio = config.compress_ratio,
                                        local_ratio = config.recent_ratio,
                                        sink_size = config.sink_size,
                                        global_cache_size = config.global_cache_size,
                                        cache_block_size = config.cache_block_size,
                                        cache_topk = config.cache_topk,
                                    )
        cache_managers.append(cache_manager)
        cpu_key_bufs += cache_manager.cpu_key_buffers
        offload_events += cache_manager.offload_events

    # Assume that we utilize 64 cpu cores.
    process_cnt = min(config.num_key_value_heads  * eval(os.environ.get("SUBVEC",2)), MAX_WORKER_CNT)
    global_compressor = MultiCoreCompressor_v2(cpu_key_bufs,
                                                offload_events,
                                                process_cnt = process_cnt,
                                                core_per_process = MAX_CPU_IN_USE // process_cnt, 
                                                max_km_groups=config.num_key_value_heads * eval(os.environ.get("SUBVEC",2)),
                                                max_seq_len=config.max_seq_len,
                                                dim=(config.hidden_size // config.num_attention_heads) // eval(os.environ.get("SUBVEC",2)),
                                                max_cent_cnt= 2 ** eval(os.environ.get("SUBBITS","6")),
                                                max_task_cnt=32,
                                                metric=os.environ.get("METRIC","euc"),
                                                layer_cnt = config.num_hidden_layers,
                                                model_name=model)

    logger.info("Multi-core compressor init done.")

def wait():
    global global_compressor
    global_compressor.wait_for_km_result()

def del_objects():
    global global_compressor
    global cache_managers
    del global_compressor
    for m in cache_managers:
        del m
    
class PqBasedSearchCompressor(RetrievalBasedCompressor):
    all_pq_compressors = []
    
    def __init__(self, compress_ratio, recent_ratio, n_subvec_per_head, n_subbits, gqa, sink_size = 32, **kwargs):
        self.compress_ratio = compress_ratio 
        self.recent_ratio = recent_ratio
        self.sink_size = sink_size
        self.topk_ratio = 1 - self.recent_ratio
        if n_subvec_per_head not in [1,2,4,8,16]:
            raise Exception("PQ subvec must in 1 2 4 8 16")
        self.n_subvec_per_head = n_subvec_per_head
        self.n_subbits = n_subbits
        self.recent_size = 0
        self.prefill_length = 0
        self.topk_size = 0
        self.layer_idx = kwargs["layer_idx"]
        self.rank = self.layer_idx // layer_per_rank
        self.code_book = None
        self.centroids = None
        self.future = None
        self.km_done = False
        self.ip2l2_phi = None
        self.GQA = gqa

        self.all_layer_cnt = kwargs["num_layer_cnt"]

        self.selected_idx_arr = []
        self.seq_cnt = 0

        n_kv_heads = kwargs["kv_head"]
        dim = kwargs["dim"]
        device = kwargs["cur_device"]
        self.max_iter = kwargs["max_iter"]
        
        if SYNC_TEST_TIME:
            self.prefetch_event = torch.cuda.Event(enable_timing=True)
            self.prefetch_event_start = torch.cuda.Event(enable_timing=True)
            self.prefetch_event_end = torch.cuda.Event(enable_timing=True)

            self.pq_start_event = torch.cuda.Event(enable_timing=True)
            self.pq_end_event = torch.cuda.Event(enable_timing=True)
            global_timer.append_compute_event(self.pq_start_event, self.pq_end_event)
            if self.layer_idx == 0:
                self.layer_0_start = torch.cuda.Event(enable_timing=True)
                self.layer_0_end = torch.cuda.Event(enable_timing=True)
        
        self.prefetch_event = torch.cuda.Event()
        
        self.gpu_key_for_recall_check = None
    
        if self.layer_idx <= 1:
            print(f"GQA is {self.GQA}")
        super().__init__(**kwargs)
        
        # Used for prefetch
        PqBasedSearchCompressor.all_pq_compressors.append(self)
    
    def build_index_cpu_multi_core_sklr(self, xb, cent_cnt) -> torch.Tensor:
        bsz, kv_heads, n_subvec_per_head, n_xb, subvec_d = xb.shape
        if n_xb > cent_cnt:
            self.valid_n_xb = n_xb
            xb = xb.reshape([bsz * kv_heads * n_subvec_per_head, n_xb, subvec_d])
            self.centroids, self.code_book, self.shm_set_idx, ip2l2_phi = global_compressor.compress(xb, 
                                                                                          cent_cnt=cent_cnt, 
                                                                                          max_iter=self.max_iter, 
                                                                                          layer_idx=self.layer_idx)
            # code_book is a big buffer that reserve places for generated token in the future.
            self.code_book = self.code_book.reshape([bsz, -1, kv_heads, n_subvec_per_head])
            self.centroids = self.centroids.reshape([bsz, kv_heads, n_subvec_per_head, cent_cnt, -1])
            return ip2l2_phi
        return None


    def _ip2l2_preprocess(self, xb: torch.Tensor, phi):
        assert xb.device != torch.device("cpu")
        assert phi.shape == (xb.shape[0], 1, 1)
        norms = (xb ** 2).sum(dim=2, keepdim=True) # n_groups, n_xb, 1
        extracol = torch.sqrt(phi - norms)
        return torch.concat((xb, extracol), dim=2)

    def prefetch_codebook(self):
        with torch.cuda.stream(H2DStream):
            if SYNC_TEST_TIME and global_timer.can_record():
                self.prefetch_event_start.record()
            self.gpu_centroids = self.centroids.to(self.device, non_blocking=True).to(torch.float16)
            self.gpu_code_book = self.code_book.to(self.device, non_blocking=True).permute([0,2,3,1]).to(torch.int64)
            self.prefetch_event.record()

            if SYNC_TEST_TIME and global_timer.can_record():
                self.prefetch_event_end.record()
                global_timer.append_transfer_time_tuples(self.prefetch_event_start, self.prefetch_event_end)

    def predict_index_cpu(self, vec: np.ndarray):
        assert vec.shape[-2] == 1
        bsz, n_kv_heads, n_subvec_per_head, q_len, subvec_d = vec.shape
        if global_compressor.metric == "ip":
            vec = self._ip2l2_preprocess(vec.reshape([bsz * n_kv_heads * n_subvec_per_head, q_len, subvec_d]), self.ip2l2_phi)
            subvec_d = vec.shape[-1]
        assert subvec_d == self.centroids.shape[-1]
        cent_cnt = self.centroids.shape[-2]
        vec = vec.reshape((-1, subvec_d))[:,None,:] # n_subspace, 1, subvec_d
        centroids = self.centroids.reshape((-1, cent_cnt, subvec_d)) # n_subspace, cent_cnt, subvec_d
        distances = torch.tensor(np.sum((centroids - vec) ** 2, axis=-1)) # n_subspace, cent_cnt
        return distances.min(dim=-1).indices.reshape([bsz, n_kv_heads, n_subvec_per_head, 1]).numpy()
    
    def predict_index_gpu(self, vec: torch.Tensor):
        assert vec.shape[-2] == 1
        bsz, n_kv_heads, n_subvec_per_head, q_len, subvec_d = vec.shape
        if global_compressor.metric == "ip":
            vec = self._ip2l2_preprocess(vec.reshape([bsz * n_kv_heads * n_subvec_per_head, q_len, subvec_d]), self.ip2l2_phi)
            subvec_d = vec.shape[-1]
        assert subvec_d == self.centroids.shape[-1]
        cent_cnt = self.centroids.shape[-2]
        vec = vec.reshape((-1, subvec_d))[:,None,:] # n_subspace, 1, subvec_d
        centroids = self.gpu_centroids.reshape((-1, cent_cnt, subvec_d)) # n_subspace, cent_cnt, subvec_d
        distances = torch.tensor(torch.sum((centroids - vec) ** 2, axis=-1)) # n_subspace, cent_cnt
        return distances.min(dim=-1).indices.reshape([bsz, 1, n_kv_heads, n_subvec_per_head])

    def prefill_attn(
        self,
        query,
        past_key_value: torch.Tensor,
        use_gpu = True
    ) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        self.gpu_centroids = None
        self.centroids = None
        self.code_book = None
        self.gpu_code_book = None
        self.km_done = False
        self.ip2l2_phi = None
        self.past_token_cnt = 0
        self.seq_cnt += 1
        if self.layer_idx == 0:
            global_compressor.refresh_pool()

        key_states, value_states = past_key_value
        bsz, kv_heads, kv_seq_len, dim = key_states.shape

        assert bsz == 1, "Do not support bsz > 1 in adaptive compression mode yet."
        self.recent_size = int((kv_seq_len-self.sink_size) * self.compress_ratio * self.recent_ratio)
        self.prefill_length = kv_seq_len
        self.topk_size = int((kv_seq_len-self.sink_size) * self.compress_ratio * (1 - self.recent_ratio))

        # There is no need to compress sink token
        xb = key_states[:,:, self.sink_size:, :]
        n_xb = kv_seq_len - self.sink_size

        subvec_d = dim // self.n_subvec_per_head
        centroid_cnt = 2 ** self.n_subbits
        xb = xb.reshape(bsz, kv_heads, n_xb, self.n_subvec_per_head, subvec_d).transpose(2,3)
        
        cache_managers[self.rank].init(key_states, value_states, self.layer_idx,self.topk_size)
        # Do compression, in async manner. self.ip2l2 will be set to None if clustering metric is euc.
        self.ip2l2_phi = self.build_index_cpu_multi_core_sklr(xb, centroid_cnt)

        attn_output = flash_attn_func(
            query.transpose(1,2),
            key_states.transpose(1,2),
            value_states.transpose(1,2),
            causal = True
        ).transpose(1,2)

        self.kv_cache_cnt = np.zeros([bsz*kv_heads], dtype=np.int64)
        self.past_token_cnt = key_states.shape[-2]
        
        # self.gpu_key_for_recall_check = key_states
        
        return attn_output, self.kv_cache_cnt

    def decoding_attn_GQA_euc(
        self,
        num_key_value_groups: int,
        query, # bsz, n_heads, q_len, dim
        repeat_k, repeat_v   
    ):
        if self.code_book is None: # skip this situation
            attn_output = torch.matmul(torch.softmax(query @ repeat_k.transpose(2,3) / math.sqrt(query.shape[-1]), dim=-1), repeat_v)
            return attn_output

        if SYNC_TEST_TIME and global_timer.can_record():
            self.pq_start_event.record()

        bsz, n_heads, n_kv_seqlen, dim = repeat_k.shape
        _, kv_head, n_subvec_per_head, cent_cnt, subvec_d = self.centroids.shape
        assert query.shape[2] == 1, "Do not support multi query pq_search yet."

        recent_index = self.past_token_cnt - self.recent_size
        n_topk_candidate = recent_index - self.sink_size
        
        k, v = unrepeat(repeat_k, num_key_value_groups, 1), unrepeat(repeat_v, num_key_value_groups, 1)

        if not self.km_done:
            global_compressor.wait_for_km_result(self.shm_set_idx)
            self.km_done = True
        
        if self.layer_idx == 0:
            if SYNC_TEST_TIME and global_timer.can_record():
                self.layer_0_start.record()
            
            self.gpu_centroids = torch.Tensor(self.centroids).to(self.device, non_blocking=True).to(torch.float16)
            self.gpu_code_book = torch.Tensor(self.code_book).to(self.device, non_blocking=True).permute([0,2,3,1]).to(torch.int64)

            if SYNC_TEST_TIME and global_timer.can_record():
                self.layer_0_end.record()
                global_timer.append_transfer_time_tuples(self.layer_0_start, self.layer_0_end)
        else:
            self.prefetch_event.wait()
        
        if self.layer_idx < (self.all_layer_cnt - 1):
            PqBasedSearchCompressor.all_pq_compressors[self.layer_idx+1].prefetch_codebook()

        query_trans = query.reshape([bsz, n_heads, 1, n_subvec_per_head, subvec_d]) \
                                    .transpose(2,3) # query: [bsz, n_heads, n_subvec_per_head, q_len, subvec_d]
        
        repeat_centroids = repeat(self.gpu_centroids, size=num_key_value_groups, dim_idx=1).transpose(3,4)
        repeat_code_book = repeat(self.gpu_code_book, size=num_key_value_groups, dim_idx=1)

        # Sink token don't have their pq indices, and tokens within local window can be ignored.
        repeat_code_book = repeat_code_book[...,:n_topk_candidate] 

        qk_table = torch.matmul(query_trans, repeat_centroids) # [bsz, n_heads, n_subvec_per_head, q_len, cent_cnt]      
        dummy_weight = torch.gather(qk_table[:,:,:,0,:], -1, repeat_code_book[:,:,:,:]).sum(dim=-2)
        dummy_softmax_scale = math.sqrt(dim)
        dummy_score = torch.softmax(dummy_weight / dummy_softmax_scale, dim=-1)
        
        dummy_score = torch.sum(dummy_score.reshape([bsz, kv_head, num_key_value_groups, 1, n_topk_candidate]), dim = 2) # reduce
        topk_indices = dummy_score.topk(self.topk_size, dim=-1, largest = True, sorted=False).indices # [bsz, kv_head, q_len, topk]

        if CHECK_RECALL:
            k_, v_ = cache_managers[self.rank].fetch_all_key_value(self.layer_idx, self.past_token_cnt)
            recall, recall_mean, recall_var = calc_recall(query, k_.transpose(1,2), topk_indices, num_key_value_groups, self.topk_size)
            if self.layer_idx == 0:
                logger.info(f"{recall},{recall_mean},{recall_var}")

        final_k_gpu, final_v_gpu = cache_managers[self.rank].fetch_and_concat_kv_w_cache(topk_indices.squeeze(2).squeeze(0), self.layer_idx)

        assert final_k_gpu.shape[-2] == self.sink_size + self.recent_size + self.topk_size + 1, f"{final_k_gpu.shape[-2]},{self.sink_size + self.recent_size + self.topk_size + 1}"
        final_k_gpu[:,:,-1:,:].copy_(k, non_blocking=True)
        final_v_gpu[:,:,-1:,:].copy_(v, non_blocking=True)

        attn_output = flash_attn_func(
            query.transpose(1,2),
            repeat(final_k_gpu, num_key_value_groups, 1).transpose(1,2),
            repeat(final_v_gpu, num_key_value_groups, 1).transpose(1,2),
            causal=True
        ).transpose(1,2)

        to_evict_key = cache_managers[self.rank].add_new_token(k, v, self.layer_idx)
        # If one token gonna pass local window in next decoding step while do not have its pq indices, 
        # we need to predict its pq indices.
        if n_topk_candidate == self.valid_n_xb:
            if self.layer_idx <= 0:
                print("Predicting generated token")
            to_pass_index = self.sink_size + self.valid_n_xb
            assert (to_pass_index + self.recent_size) == self.past_token_cnt, f"{to_pass_index}, {self.recent_size}, {n_kv_seqlen}, {self.past_token_cnt}"
            to_predict_k = to_evict_key
            indices = self.predict_index_gpu(to_predict_k.reshape([bsz, kv_head, 1, n_subvec_per_head, subvec_d]).transpose(2,3))
            self.code_book[:, n_topk_candidate:n_topk_candidate+1,:,:].copy_(indices, non_blocking=True) # NOTE: Let's neglect its overhead for now.
            self.valid_n_xb += 1
        
        if SYNC_TEST_TIME and global_timer.can_record():
            self.pq_end_event.record()

        self.past_token_cnt += 1
        return attn_output

    def decoding_attn_GQA_ip(
        self,
        num_key_value_groups: int,
        query, # bsz, n_heads, q_len, dim
        repeat_k, repeat_v   
    ):
        if self.code_book is None: # skip this situation
            attn_output = torch.matmul(torch.softmax(query @ repeat_k.transpose(2,3) / math.sqrt(query.shape[-1]), dim=-1), repeat_v)
            return attn_output
        if np.random.randint(0,10000) % 5000 == 0:
            logger.info("Using ip2l2 metric to decoding!")
    
        bsz, n_heads, n_kv_seqlen, dim = repeat_k.shape
        _, kv_head, n_subvec_per_head, cent_cnt, subvec_d = self.centroids.shape
        assert query.shape[2] == 1, "Do not support multi query pq_search yet."

        recent_index = self.past_token_cnt - self.recent_size
        n_topk_candidate = recent_index - self.sink_size
        
        k, v = unrepeat(repeat_k, num_key_value_groups, 1), unrepeat(repeat_v, num_key_value_groups, 1)

        if not self.km_done:
            global_compressor.wait_for_km_result(self.shm_set_idx)
            self.km_done = True
        
        if self.layer_idx == 0:
            self.gpu_centroids = torch.Tensor(self.centroids).to(self.device, non_blocking=True).to(torch.float16)
            self.gpu_code_book = torch.Tensor(self.code_book).to(self.device, non_blocking=True).permute([0,2,3,1]).to(torch.int64)
        else:
            self.prefetch_event.wait()
        
        # NOTE: prefetch.
        if self.layer_idx < (self.all_layer_cnt - 1):
            PqBasedSearchCompressor.all_pq_compressors[self.layer_idx+1].prefetch_codebook()

        query_trans = query.reshape([bsz, n_heads, 1, n_subvec_per_head, dim // n_subvec_per_head]) \
                                    .transpose(2,3) # query: [bsz, n_heads, n_subvec_per_head, q_len, subvec_d]
        aug_query_trans = self.augment_xq(query_trans.reshape([-1, dim // n_subvec_per_head])).reshape([bsz, n_heads,n_subvec_per_head,  1, subvec_d])
        
        repeat_centroids = repeat(self.gpu_centroids, size=num_key_value_groups, dim_idx=1)
        repeat_code_book = repeat(self.gpu_code_book, size=num_key_value_groups, dim_idx=1)

        # Sink token don't have their pq indices, and tokens within local window can be igonored.
        repeat_code_book = repeat_code_book[...,:n_topk_candidate] 

        # NOTE: Main method
        qk_table = torch.sum((aug_query_trans - repeat_centroids) ** 2, dim = -1, keepdim=True) # [bsz, n_heads, n_subvec_per_head, cent_cnt, 1]
        dummy_distance = query.new_zeros([bsz, n_heads, 1, n_topk_candidate])
        
        # TODO: optimize here
        for i in range(0, n_subvec_per_head):
            distance_piece = torch.gather(qk_table[:,:,i,:,0], -1, repeat_code_book[:,:,i,:])
            dummy_distance[:,:,0,:] += distance_piece
        
        dummy_score = torch.sum(dummy_distance.reshape([bsz, kv_head, num_key_value_groups, 1, n_topk_candidate]), dim = 2) # reduce
        topk_indices = dummy_score.topk(self.topk_size, dim=-1, largest = False).indices # [bsz, kv_head, q_len, n_xb]
        # END NOTE

        ratio, x, y = calc_recall(query, self.gpu_key_for_recall_check[...,self.sink_size:recent_index,:], topk_indices, num_key_value_groups, topk_indices.shape[-1])
        if np.random.randint(0,10000) % 5000 == 0:
            print(f"Recall in cur attn:{ratio}, mean:{x}, var:{y}, PQ_search_GQA, Ignore local and sink? yes!")

        final_k_gpu, final_v_gpu = cache_managers[self.rank].fetch_and_concat_kv_w_cache(topk_indices.squeeze(2).squeeze(0), self.layer_idx)

        assert final_k_gpu.shape[-2] == self.sink_size + self.recent_size + self.topk_size + 1, f"{final_k_gpu.shape[-2]},{self.sink_size + self.recent_size + self.topk_size + 1}"
        final_k_gpu[:,:,-1:,:].copy_(k, non_blocking=True)
        final_v_gpu[:,:,-1:,:].copy_(v, non_blocking=True)

        attn_output = flash_attn_func(
            query.transpose(1,2),
            repeat(final_k_gpu, num_key_value_groups, 1).transpose(1,2),
            repeat(final_v_gpu, num_key_value_groups, 1).transpose(1,2),
            causal=True
        ).transpose(1,2)

        to_evict_key = cache_managers[self.rank].add_new_token(k, v, self.layer_idx)
        if self.gpu_key_for_recall_check is not None:
            self.gpu_key_for_recall_check = torch.concat([self.gpu_key_for_recall_check, k], dim = -2)
        # If one token gonna pass local window in next decoding step while do not have its pq indices, 
        # we need to predict its pq indices.
        if n_topk_candidate == self.code_book.shape[-1]:
            if self.layer_idx <= 0:
                print("Predicting generated token")
            to_pass_index = self.sink_size + self.code_book.shape[-1]
            assert (to_pass_index + self.recent_size) == self.past_token_cnt, f"{to_pass_index}, {self.recent_size}, {n_kv_seqlen}"
            to_predict_k = to_evict_key
            indices = self.predict_index_gpu(to_predict_k.reshape([bsz, kv_head, 1, n_subvec_per_head, subvec_d]).transpose(2,3))
            self.code_book[0, n_topk_candidate:n_topk_candidate+1,:,:].copy_(indices, non_blocking=True) # NOTE: Let's neglect its overhead for now.
            
        self.past_token_cnt += 1

        return attn_output


    def augment_xq(self, xq): 
        extracol = torch.zeros(len(xq), dtype=xq.dtype, device = xq.device)
        return torch.hstack((xq, extracol.reshape(-1, 1)))

    def decoding_attn(
        self,
        num_key_value_groups: int,
        query, # bsz, n_heads, q_len, dim
        repeat_k, repeat_v
    ):
        if self.GQA:
            if global_compressor.metric == "euc":
                target_func = self.decoding_attn_GQA_euc
            elif global_compressor.metric == "ip":
                target_func = self.decoding_attn_GQA_ip
        else:
            raise Exception("wo GQA not supported currently")

        return target_func(num_key_value_groups, query, repeat_k, repeat_v)