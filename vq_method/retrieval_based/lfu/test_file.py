# please first install pybind11: pip install pybind11[global]
# then compile the codes: mkdir build; cd build; cmake ..; make

import sys
import os.path as osp
import numpy as np
sys.path.append(osp.join(osp.abspath(osp.dirname(__file__)), "build"))
print(sys.path)
import lfucache
import torch.multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager

def test():
    cache = lfucache.LFUCache(5) # cache size
    arr = np.ndarray([10],  dtype=np.int32)
    arr.fill(-1)
    ids = np.arange(5, dtype=np.int32)
    arr = np.ascontiguousarray(arr)
    ids = np.ascontiguousarray(ids)
    print(ids, arr)
    cache.BatchedInsertArray(ids, arr)
    # x.BatchedInsertArray(ids, arr)
    print(ids, arr)
    ids = np.random.randint(0, 10, size=(5,), dtype=np.int32)
    ids = np.ascontiguousarray(ids)
    # x.BatchedInsertArray(ids, arr)
    cache.BatchedInsertArray(ids, arr)
    print(ids, arr)

    # test async
    ids = np.random.randint(0, 10, size=(7,), dtype=np.int32)
    ids = np.ascontiguousarray(ids)
    print(ids, arr)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    test()