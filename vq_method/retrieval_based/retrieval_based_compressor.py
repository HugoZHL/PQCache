import torch
import time
import os
import numpy as np

def repeat(a:torch.Tensor, size, dim_idx):
    shape = a.shape
    return a.unsqueeze(dim_idx+1) \
            .expand(*shape[:dim_idx], shape[dim_idx], size, *shape[dim_idx+1:]) \
            .reshape(*shape[:dim_idx], shape[dim_idx] * size, *shape[dim_idx+1:])

def unrepeat(a:torch.Tensor, size, dim_idx):
    shape = a.shape
    return a.reshape(*shape[:dim_idx], shape[dim_idx] // size, size, *shape[dim_idx+1:]) \
            .select(dim_idx+1, 0) # NOTE: By default it will squeeze the target dimension.

recall_history = []

def calc_recall(query, key, dummy_topk_indices, num_kv_group, topk_size):
    bsz, kv_head, kv_seq_len, dim = key.shape
    _, n_head, q_len, _ = query.shape
    if key.shape[1] * num_kv_group == query.shape[1]:
        real_weight = query @ repeat(key, num_kv_group, 1).transpose(2,3)
    elif key.shape[1] == query.shape[1]:
        real_weight = query @ key.transpose(2,3)
    else:
        raise Exception(f"?{key.shape},{query.shape},{num_kv_group}")
    
    real_topk_indices = real_weight.topk(k = topk_size, dim = -1, largest=True).indices
    assert real_topk_indices.shape == (bsz, n_head, q_len, topk_size)

    if dummy_topk_indices.shape[1] != real_topk_indices.shape[1]:
        dummy_topk_indices = repeat(dummy_topk_indices, num_kv_group, 1)
    
    dummy_topk_indices = dummy_topk_indices.flatten(0,1)
    real_topk_indices = real_topk_indices.flatten(0,1)
    ground_truth_idx_cnt = topk_size * n_head * bsz
    hit_idx_cnt = 0

    for i in range(dummy_topk_indices.shape[0]):
        dummy = dummy_topk_indices[i,0,:]
        real = real_topk_indices[i,0,:]
        assert dummy.numel() == torch.unique(dummy).numel()
        comparison = torch.isin(dummy, real, assume_unique=True)
        hit_idx_cnt += torch.sum(comparison.int())

    result = hit_idx_cnt / ground_truth_idx_cnt
    
    recall_history.append(result.item())
    np_history = np.array(recall_history)
    mean, var = np.mean(np_history), np.var(np_history)
    return result, mean, var


class RetrievalBasedCompressor:
    def __init__(self, **kwargs) -> None:
        self.profile_metric = {
            "prefill_time" : 0,
            "prefill_attn_time" : 0,
            "prefill_cnt": 0,
            "prefill_per_layer_time": 0,

            "prepare_idx_elapsed" : 0,
            "prepare_idx_cnt" : 0,

            "decoding_time" : 0,
            "decoding_cnt" : 0,
            "decoding_attn_time" : 0,

            "offload_pq_ref_elapsed" : 0,
            "offload_kv_elapsed" : 0,
            "offload_cnt": 0,
            "offload_pq_ref_bytes" : 0,
            "offload_kv_bytes" : 0,

            "fetch_ref_elapsed" : 0,
            "fetch_kv_elapsed" : 0,
            "cpu_gather_elapsed" : 0,
            "calc_dummy_weight_elapsed" : 0,
            "fetch_ref_data_bytes" : 0,
            "fetch_kv_data_bytes" : 0,
        }

        self.device = kwargs["cur_device"]
    
    def profile_ckpt(self):
        torch.cuda.synchronize(self.device)
        return time.perf_counter()

    def reset(self):
        for k,_ in self.profile_metric.items():
            self.profile_metric[k] = 0

    def showtime(self):
        result_str = "\n".join([f"{key} : {value}" for key, value in self.profile_metric.items()])
        print("-----profile result show:\n", result_str)
        with open(f"./profile_result/mistral_profile_{os.getpid()}","a") as f:
            f.write(result_str)
