import torch
from typing import Optional
from .context_manager import ContextManager
import math
import os

def inf_llm_forward(
    n_local, n_init, topk, 
    block_size, max_cached_block,
    exc_block_size, fattn,
    repr_topk: int = 1,
    cache_strategy="lru",
    score_decay=None,
    chunk_topk_calc=None,
    async_global_stream=True,
    pin_memory=False,
    faiss=False,
    perhead=False,
    *args, **kwargs
):

    def forward(self, query : torch.Tensor,
                    key_value : torch.Tensor,
                    position_bias : Optional[torch.Tensor],
                    use_cache: bool,
                    past_key_value,
                    project_q, project_k, project_v, attention_out, 
                    dim_head, num_heads, num_heads_kv
    ):

        batch_size = query.size(0)
        len_q = query.size(1)
        len_k = key_value.size(1)

        assert use_cache

        h_q = project_q(query)             # (batch, len_q, num_heads * dim_head)
        h_k = project_k(key_value)         # (batch, len_k, num_heads * dim_head)
        h_v = project_v(key_value)         # (batch, len_k, num_heads * dim_head)


        h_q = h_q.view(batch_size, len_q, num_heads, dim_head).permute(0, 2, 1, 3).contiguous()   # (batch, num_heads, len_q, dim_head)
        h_k = h_k.view(batch_size, len_k, num_heads_kv, dim_head).permute(0, 2, 1, 3).contiguous()   # (batch, num_heads_kv, len_k, dim_head)
        h_v = h_v.view(batch_size, len_k, num_heads_kv, dim_head).permute(0, 2, 1, 3).contiguous()   # (batch, num_heads_kv, len_k, dim_head)


        if past_key_value is None:
            compress_ratio = eval(os.environ["COMPRESS_RATIO"])
            local_ratio = eval(os.environ["LOCAL_RATIO"])
            no_sink_len = len_k - n_init
            new_n_local = int(no_sink_len * compress_ratio * local_ratio)
            topk = math.ceil((no_sink_len * compress_ratio - new_n_local) / block_size)
            
            new_max_cached_block = max_cached_block
            if max_cached_block < topk:
                cache_extra_size = (2**30)
                mem_unit_size = block_size * num_heads * dim_head * 2 * 2
                cache_extra_block = math.ceil(cache_extra_size/mem_unit_size)
                new_max_cached_block = topk + cache_extra_block
            
            new_exc_block_size = exc_block_size
            if exc_block_size > new_n_local:
                # new_exc_block_size = 1
                # while new_exc_block_size * 2 <= new_n_local:
                #     new_exc_block_size *= 2
                new_exc_block_size = new_n_local

            past_key_value = ContextManager(
                position_bias, n_init,
                new_n_local, block_size,
                new_max_cached_block, topk,
                new_exc_block_size,
                score_decay, fattn, repr_topk,
                cache_strategy,
                chunk_topk_calc,
                async_global_stream,
                pin_memory,
                faiss,
                perhead
            )

        local_q, local_k, local_v = h_q, h_k, h_v
        global_q, global_k, global_v = h_q, h_k, h_v

        o = past_key_value.append(
            local_q, local_k, local_v,
            global_q, global_k, global_v,
        )


        o = o.view(batch_size, num_heads, len_q, dim_head).permute(0, 2, 1, 3)
        o = o.reshape(batch_size, len_q, dim_head * num_heads)
        o = attention_out(o)

        if use_cache:
            return o, past_key_value
        else:
            return o


    return forward
