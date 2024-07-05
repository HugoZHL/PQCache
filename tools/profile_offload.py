import torch
import time



def get_tensor(shape, dtype):
    return torch.randn(shape, dtype=dtype)

def to_device(tensor, device):
    return tensor.to(device)

def from_device(tensor):
    return tensor.cpu()


results = {}

for seqlen in range(1000, 21000, 1000):
    cur_shape = (1, 8, seqlen, 128)
    tensors = [get_tensor(cur_shape, torch.float16) for _ in range(25)]
    tensors = [to_device(t, 'cuda:7') for t in tensors]
    for i in range(5):
        _ = from_device(tensors[i])
    
    start = time.perf_counter()
    for i in range(5, 25):
        _ = from_device(tensors[i])
    during = time.perf_counter()-start
    during = during / 20 * 2
    print("Seqlen", seqlen, "total time elapsed:", during)
    results[seqlen] = during

import json
with open(f"./result_offload_small.json","w") as f:
    json.dump(results, f, ensure_ascii = False, indent=4)

    

