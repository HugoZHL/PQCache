from bz2 import compress
import concurrent.futures as future
import torch
import numpy as np
import time
import multiprocessing
import os
import math
from typing import List

import torch.multiprocessing as mp
from torch.multiprocessing import Condition, Event, Value
from torch.multiprocessing import Queue
from .tools.model_kmeans_time import profile
import json
from loguru import logger


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

class SharedMemSet:
    def __init__(self, cpu_key_buf, offload_event, max_km_groups, max_cent_cnt, max_seq_len, dim, **kwargs) -> None:
        self.xb_shared_tensor = cpu_key_buf
        self.cb_shared_tensor = torch.empty(    
                                        [max_km_groups, max_cent_cnt, dim],
                                        dtype = torch.float16, 
                                        ).share_memory_()
        pin_shm([self.cb_shared_tensor])
        self.id_shared_tensor = torch.empty(    
                                        [max_seq_len, max_km_groups],
                                        dtype = torch.int64,     
                                        ).share_memory_()
        pin_shm([self.id_shared_tensor])
        self.offload_cpu_event = Event() # New event
        
        self.km_task_event = Event()
        self.km_task_event.set()
        self.done_flag = Value("i", 0)
        self.offload_event = offload_event
        self.in_use = False # Not shared by worker process

    def free(self):
        self.km_task_event.clear()
        self.offload_cpu_event.clear()
        self.in_use = False
        

# No synchorization operation is needed.
class MultiProcSharedTensorPool:
    def __init__(self, cpu_key_bufs, offload_events,**kwargs) -> None:
        self.shared_mem_block_sets = [
            SharedMemSet(cpu_key_bufs[i], offload_events[i], **kwargs) 
            for i in range(kwargs["max_size"])
        ]

    def get(self):
        while True:
            for i, shm_set in enumerate(self.shared_mem_block_sets):
                if not shm_set.in_use:
                    shm_set.in_use = True       
                    return i

def compute_kmeans_worker(
        idx, n_proc, n_core, 
        shmpool: MultiProcSharedTensorPool,
        task_queue: Queue,
        layer_cnt: int
):
    try:
        import sklearnex
        sklearnex.patch_sklearn(verbose = False) # for Intel Xeon CPU to acclerate
    except:
        pass
    from sklearn.cluster import KMeans
    
    cpu_num = multiprocessing.cpu_count()
    cur_pid = os.getpid()
    if n_core >= 1:
        cpu_use = int(n_core)
        os.sched_setaffinity(cur_pid, list(range(cpu_num))[cpu_use * idx:cpu_use * (idx + 1)])
    else:
        cpu_use = 1
        core_idx = math.floor(n_core*idx)
        os.sched_setaffinity(cur_pid, list(range(cpu_num))[core_idx:core_idx + 1])

    init_cent_idx = None
    last_n_xb, last_cent_cnt = 0, 0

    while True:
        shm_set_idx, shape, cent_cnt, max_iter = task_queue.get()
        assert shm_set_idx == 0, shm_set_idx
        if shape is None and cent_cnt == 0:
            print(f"Worker {idx} exit gracefully.")
            exit()
        n_groups, n_xb, dim = shape
        
        if last_n_xb != shape[1] or last_cent_cnt != cent_cnt:
            init_cent_idx = np.random.choice(np.arange(n_xb), size = cent_cnt, replace = False)
            last_n_xb = n_xb
            last_cent_cnt = cent_cnt
        
        for shm_set_idx in range(layer_cnt):
            shm_set = shmpool.shared_mem_block_sets[shm_set_idx]

            shm_set.offload_cpu_event.wait() # 等待主进程record event
            shm_set.offload_event.synchronize() # Wait for the completion of KV offload

            cpu_key_buf = shm_set.xb_shared_tensor # [1, total_max_len, n_kv_head, dim]
            _, total_max_len, n_kv_head, hidden_size = cpu_key_buf.shape
            
            xb_array = shm_set.xb_shared_tensor \
                .reshape([1, total_max_len, n_kv_head, hidden_size // dim, dim]) \
                .reshape([1, total_max_len, n_groups, dim])[0, :n_xb, idx].numpy()

            cb_shm_tensor = shm_set.cb_shared_tensor
            indices_shm_tensor = shm_set.id_shared_tensor
            
            result = KMeans(
                    n_clusters = cent_cnt,
                    n_init=1,
                    init=xb_array[init_cent_idx], 
                    tol = 0.0001,
                    copy_x=True,
                    verbose=False,
                    max_iter=max_iter,
                    random_state=0,
                    algorithm="lloyd"
            ).fit(xb_array)
            
            cb_shm_tensor[idx].copy_(torch.from_numpy(result.cluster_centers_))
            indices_shm_tensor[:n_xb, idx].copy_(torch.from_numpy(result.labels_))

            shm_set.done_flag.acquire()
            shm_set.done_flag.value += 1
            if shm_set.done_flag.value == n_proc:
                # TODO: 需要唤醒吗？
                shm_set.km_task_event.set()
                shm_set.offload_cpu_event.clear()
                shm_set.done_flag.value = 0
            shm_set.done_flag.release()
        # torch.cuda.synchronize()

# dim=64, metric = "euc"下的配置
prefill_coef = [5.56006974e-11, 1.60809797e-06, 7.46724923e-03]

class MultiCoreCompressor_v2:
    def __init__(
            self, 
            # 临时设计，为了快速验证能否提速
            cpu_key_bufs,
            offload_events,
            # /
            process_cnt, 
            core_per_process, 
            max_km_groups,
            max_seq_len,
            dim,
            max_cent_cnt,
            max_task_cnt,
            metric = "euc",
            layer_cnt = 32
        ) -> None:
        self.metric = metric
        if self.metric == "ip":
            dim += 1
        
        self.max_km_groups = max_km_groups
        self.max_seq_len = max_seq_len

        self.proc_cnt = process_cnt
        self.n_core_per_proc = core_per_process

        self.shm_pool = MultiProcSharedTensorPool(
                                    cpu_key_bufs,
                                    offload_events,
                                    max_km_groups = max_km_groups, 
                                    max_cent_cnt = max_cent_cnt,
                                    max_seq_len = max_seq_len,
                                    dim = dim, 
                                    max_size = 32, 
                                    process_cnt = process_cnt,
                                   )

        self.queues = [Queue(max_task_cnt) for _ in range(process_cnt)]

        self.processes = []
        for i in range(self.proc_cnt):
            p = multiprocessing.Process(
                    target = compute_kmeans_worker, 
                    args = (i, self.proc_cnt, self.n_core_per_proc, self.shm_pool, self.queues[i], layer_cnt)
                )
            p.start()
            self.processes.append(p)
        
        # TODO: 空闲worker抢掉队者的任务来做，目的是减少queue的数量
        self.kmeans_coef = None

        # Try to load existing clustering coef. 
        # If no coef conf have been archived, we will profile clustering time to generate a new item.
        if os.path.exists("./cluster_config.json"):
            with open("./cluster_config.json", "r") as f:
                cfg = json.load(f)
        else:
            cfg = dict()
            
        try:
            self.kmeans_coef = cfg[f"{dim}_{max_cent_cnt}_{core_per_process}"]
            print(f"KMeans coef is {self.kmeans_coef}. If you are running this system in a new hardware environment, please generate a new configuration file")
        except:
            print("Cannot find existing clustering config. We will do some profiling now.")
            self.profile_executor = future.ProcessPoolExecutor(max_workers = 1)
            if core_per_process >= 1:
                self.profile_f = self.profile_executor.submit(profile, dim, max_cent_cnt, int(core_per_process))
            else: 
                self.profile_f = self.profile_executor.submit(profile, dim, max_cent_cnt, 1)
        
            self.iter_3_coef, self.per_iter_coef, _ = self.profile_f.result()
            if self.n_core_per_proc < 1:
                self.iter_3_coef, self.per_iter_coef = self.iter_3_coef * (1/self.n_core_per_proc), self.per_iter_coef * (1/self.n_core_per_proc)

            cfg[f"{dim}_{max_cent_cnt}_{core_per_process}"] = {
                                                                "3_iter":self.iter_3_coef.tolist(), 
                                                                "per_iter":self.per_iter_coef.tolist()
                                                            }
            
            with open("./cluster_config.json", "w") as f:
                json.dump(cfg, f)
            
            self.kmeans_coef = {"3_iter":self.iter_3_coef, "per_iter":self.per_iter_coef}
            self.profile_executor.shutdown(wait = True)
            print(f"KMeans coef is {self.kmeans_coef}. New cluster coef has been archived.")

        print(f"Compressor init done, config: \n \
                process_cnt = {process_cnt}, \n \
                core_per_process = {core_per_process}, \n \
                max_km_groups = {max_km_groups}, \n \
                max_seq_len = {max_seq_len}, \n \
                dim = {dim}, \n \
                max_cent_cnt = {max_cent_cnt}, \n \
                max_task_cnt = {max_task_cnt} \n \
                metric is {metric}")

    
    def __del__(self):
        for queue in self.queues:
            queue.put((0, None, 0, 0))

    # NOTE: 目的两个：1. 让shm_set的xb_shm共用cache_manager的cpu buffer；2. 让shm_set持有cpu buffer的offload event，这样便于sync.
    def borrow_cpu_buffer(self, bufs: List[torch.Tensor], offload_events:List[torch.cuda.Event]):
        for i, shm_set in enumerate(self.shm_pool.shared_mem_block_sets):
            shm_set.xb_shared_tensor = bufs[i]
            shm_set.offload_event = offload_events[i]

    # GPU来做,CPU来做消耗不小.
    def _ip2l2_preprocess(self, xb: torch.Tensor):
        assert xb.device != torch.device("cpu")
        norms = (xb ** 2).sum(dim=2, keepdim=True) # n_groups, n_xb, 1
        phi = norms.max(dim = 1, keepdim=True).values
        extracol = torch.sqrt(phi - norms)
        return torch.concat((xb, extracol), dim=2), phi

    def compress(self, x: torch.Tensor, cent_cnt: int, max_iter: int, layer_idx: int):
        assert len(x.shape) == 3
        assert x.dtype == torch.float16

        n_groups, n_xb, dim = x.shape

        free_shm_set_idx = layer_idx
        if self.metric == "ip":
            dst_codebook = np.ndarray([n_groups, cent_cnt, dim + 1], dtype=np.float16)
        else:
            dst_codebook = self.shm_pool.shared_mem_block_sets[free_shm_set_idx].cb_shared_tensor[:n_groups, :cent_cnt, :]
        dst_indices = self.shm_pool.shared_mem_block_sets[free_shm_set_idx].id_shared_tensor

        if max_iter is None or max_iter == 0:
            gpu_compute_time = (prefill_coef[0]*n_xb**2 + prefill_coef[1]*n_xb + prefill_coef[2])
            kmeans_base = self.kmeans_coef["3_iter"][0] * n_xb + self.kmeans_coef["3_iter"][1]
            kmeans_per_round = self.kmeans_coef["per_iter"][0] * n_xb + self.kmeans_coef["per_iter"][1]
            max_iter = int((gpu_compute_time / 2 - kmeans_base) / kmeans_per_round  +  3) 
            max_iter = max(max_iter, 3)
            max_iter = min(max_iter, 10)
            # if np.random.randint(0, 1000) % 30 == 1:
            #     logger.info(f"友情提示，正在使用multi core压缩器，max_iter:{max_iter}, {kmeans_base}, {kmeans_per_round}, {gpu_compute_time}") 
        else:
            # if np.random.randint(0, 1000) % 30 == 1:
            #     logger.info(f"友情提示，正在使用multi core压缩器，max_iter:{max_iter}") 
            pass

        # 压缩任务是周期性的。我们只需要在每个周期开始的时候issue一次任务即可
        if layer_idx == 0:
            # NOTE: 这里会直接返回，只是压缩任务会排队
            self._compress_task(cent_cnt, x, free_shm_set_idx, max_iter)
        
        self.shm_pool.shared_mem_block_sets[free_shm_set_idx].offload_cpu_event.set()

        return dst_codebook, dst_indices, free_shm_set_idx

    def _compress_task(
                self,
                cent_cnt,
                input_tensor,
                shm_set_idx,
                max_iter,
            ):
        shm_set: SharedMemSet

        ip2l2_phi = None
        if self.metric == "ip":
            input_tensor, ip2l2_phi = self._ip2l2_preprocess(input_tensor)
        n_groups, n_xb, input_dim = input_tensor.shape

        # TODO: 这个queue是非常非常耗时的. 尝试每个sequence只issue一次。
        for queue in self.queues:
            queue.put((shm_set_idx, (n_groups, n_xb, input_dim), cent_cnt, max_iter))
    
    def wait_for_km_result(self, shm_set_idx):
        done = self.shm_pool.shared_mem_block_sets[shm_set_idx].km_task_event.is_set()
        if not done:
            while not self.shm_pool.shared_mem_block_sets[shm_set_idx].km_task_event.wait(timeout=2):
                logger.info("Waiting for km task to complete.")
        self.shm_pool.shared_mem_block_sets[shm_set_idx].km_task_event.wait()

    def refresh_pool(self):
        for shm_set in self.shm_pool.shared_mem_block_sets:
            shm_set.in_use = False
            if not shm_set.km_task_event.is_set():
                while not shm_set.km_task_event.wait(timeout=2):
                    logger.info("Waiting for km task to complete.")
            # shm_set.km_task_event.wait()
            shm_set.km_task_event.clear()

def test():
    n_subvec_per_head = 2
    cent_cnt = 64
    compressor = MultiCoreCompressor_v2(16, 4)
    a = time.perf_counter()
    futures = []
    for i in range(16):
        key = torch.load(f"./kv_tensors/key_{i}_sample1.pt").to("cuda:0")
        bsz, n_kv_heads, n_xb, dim = key.shape
        key = key.reshape([bsz, n_kv_heads, n_xb, n_subvec_per_head, dim // n_subvec_per_head])
        key = key.transpose(2,3).flatten(1,2).flatten(0,1)
        futures.append(compressor.compress(key, cent_cnt, "euc")[2])
    
    for f in futures:
        f.result()
    
    print(f"{time.perf_counter() - a}")
    a = time.perf_counter()
    futures = []
    for i in range(16):
        key = torch.load(f"./kv_tensors/key_{i}_sample1.pt").to("cuda:0")
        bsz, n_kv_heads, n_xb, dim = key.shape
        key = key.reshape([bsz, n_kv_heads, n_xb, n_subvec_per_head, dim // n_subvec_per_head])
        key = key.transpose(2,3).flatten(1,2).flatten(0,1)
        futures.append(compressor.compress(key, cent_cnt, "euc")[2])
    for f in futures:
        f.result()
    print(f"{time.perf_counter() - a}")

    # for i in range(16):
    #     key = torch.load(f"./kv_tensors/key_{i}_sample0.pt").to("cuda:0")
    #     bsz, n_kv_heads, n_xb, dim = key.shape
    #     key = key.reshape([bsz, n_kv_heads, n_xb, n_subvec_per_head, dim // n_subvec_per_head])
    #     key = key.transpose(2,3).flatten(1,2).flatten(0,1)
    #     compressor.compress(key, cent_cnt, "euc")
    
    time.sleep(10)
    # del compressor