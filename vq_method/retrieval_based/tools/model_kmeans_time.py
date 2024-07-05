# 测试单组聚类性能
import time
import psutil as p
import os
from sympy import false
import torch
import numpy as np
import multiprocessing
import json
import tqdm

np.random.seed(1234)

def kmeans_profile(max_iter, cpu_allocated, seq_len, dim, cent_cnt):
    try:
        import sklearnex
        sklearnex.patch_sklearn(verbose=False) # for Intel Xeon CPU to acclerate
    except:
        pass
    from sklearn.cluster import KMeans
    all_time = 0
    for _ in range(0,10):
        k = torch.randn([1, 1, 30000, dim])
        bsz, kv_head, n_xb, dim = k.shape
        cent_cnt = cent_cnt
        n_subset = seq_len

        subset_idx = np.random.choice(np.arange(n_xb), size=n_subset,replace=False)
        subset_cent_idx = np.random.choice(np.arange(n_subset), size=cent_cnt,replace=False) # TODO: 改为16
        
        target = k[0,0,subset_idx,:]
        target = target.reshape([-1, dim]).cpu().numpy()
        # print(f"set the max number of cpu used to {cpu_allocated}", target.shape, "cent_cnt", cent_cnt)
        
        compressor = KMeans(
                        n_clusters = cent_cnt,
                        n_init=1,
                        init=target[subset_cent_idx],
                        # init="k-means++",
                        tol = 0.0001,
                        # copy_x=True,
                        verbose=False,
                        max_iter=max_iter,
                        random_state=0,
                        algorithm="lloyd"
                    )
        
        # ok = compressor.fit(target)
        a = time.perf_counter()
        inertias = 0
        iters = 0
        ok = compressor.fit(target)
        inertias += ok.inertia_
        iters += ok.n_iter_
        b = time.perf_counter()
        all_time += b - a

    elapsed = ( all_time / 10)
    return elapsed

def profile(dim, cent_cnt, cpu_allocated = 1):
    result_dict = dict()
    seq_lens = []
    base_latency = []
    per_iter_latency = []

    cpu_num = multiprocessing.cpu_count()
    cur_pid = os.getpid()
    os.sched_setaffinity(cur_pid, list(range(cpu_num))[-cpu_allocated:])

    for seq_len in range(1000, 20000, 1000):
        seq_lens.append(seq_len)
        result_dict.setdefault(seq_len, dict())
        for max_iter in [3, 27]:
            result_dict[seq_len][max_iter] = kmeans_profile(
                                                max_iter, 
                                                cpu_allocated = cpu_allocated,
                                                seq_len = seq_len,
                                                dim = dim,
                                                cent_cnt = cent_cnt
                                            )
            
    for seq_len, time in result_dict.items():
        base_latency.append(time[3]) 
        per_iter_latency.append((time[27] - time[3]) / 24)

    iter_3_coef = np.polyfit(seq_lens, base_latency, 1)
    per_iter_coef = np.polyfit(seq_lens, per_iter_latency, 1)
    return iter_3_coef, per_iter_coef, result_dict
    
if __name__ == "__main__":
    dim, cent_cnt, cpu = 64, 64, 1
    _,_, result_dict = profile(dim, cent_cnt, cpu)
    print(result_dict)
    with open(f"./kmeans_{dim}_{cent_cnt}_{cpu}.json", "w") as f:
        json.dump(result_dict, f)