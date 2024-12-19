# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

"""Approximate nearest neighbour methods that approximate `Q @ K.T`."""

from dataclasses import dataclass
from functools import partial
from typing import Any, List, Optional, Tuple, Union, cast

import torch
from torch import Tensor, nn
from transformers.models.gpt_neox.configuration_gpt_neox import GPTNeoXConfig
from transformers.models.gpt_neox.modeling_gpt_neox import (
    GPTNeoXAttention,
    GPTNeoXForCausalLM,
)
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaForCausalLM
from transformers.models.mistral.configuration_mistral import MistralConfig
from transformers.models.mistral.modeling_mistral import (
    MistralAttention,
    MistralForCausalLM,
)

from .. import utility
from ..models import llama_attention, mistral_attention
from . import sparse_attention


def gather(t: Tensor, dim: int, i: Tensor) -> Tensor:
    """A broadcasting version of torch.gather."""
    dim += (dim < 0) * t.ndim
    return t.gather(dim, i.expand(*t.shape[:dim], i.shape[dim], *t.shape[dim + 1 :]))


class LowRank(nn.Module):
    """Use a random orthonormal projection to down-project Q & K."""

    @dataclass
    class Settings:
        rank: int
        name: str = "low_rank"

    def __init__(self, settings: Settings, n_kv_heads: int, head_size: int):
        super().__init__()
        self.settings = settings
        self.weight = nn.Parameter(torch.empty(n_kv_heads, 1, head_size, settings.rank))
        for i in range(n_kv_heads):  # can't batch this!
            nn.init.orthogonal_(self.weight[i, 0])  # type:ignore[no-untyped-call]

    def forward(self, query: Tensor, key: Tensor) -> Tensor:
        """Compute approximate score for each (query, key).

        query -- (batch, n_kv_heads, n_heads_per_kv, query, head_size)

        key -- (batch, n_kv_heads, 1, key, head_size)

        returns -- (batch, n_kv_heads, n_heads_per_kv, query, key)
        """
        head_size = query.shape[-1]
        query_proj = query.to(self.weight.dtype) @ self.weight
        key_proj = key.to(self.weight.dtype) @ self.weight
        return cast(Tensor, query_proj @ key_proj.transpose(-1, -2) * head_size**-0.5)


class SparseQ(nn.Module):
    """Gather the top (absolute) components of Q from Q & K."""

    @dataclass
    class Settings:
        rank: int
        name: str = "sparse_q"

    def __init__(self, settings: Settings):
        super().__init__()
        self.settings = settings

    def forward(self, query: Tensor, cpu_key: Tensor, gpu_cur_key: Tensor) -> Tensor:
        """Compute approximate score for each (query, key).

        query -- (batch, n_kv_heads, n_heads_per_kv, 1, head_size)

        cpu_key -- (batch, n_kv_heads, head_size, seq_len)

        gpu_cur_key -- (batch, n_kv_heads, 1, head_size, 1)

        returns -- (batch, n_kv_heads, n_heads_per_kv, 1, key)
        """
        assert cpu_key.device == torch.device("cpu") and gpu_cur_key != torch.device("cpu")
        assert query.shape[-2] == 1, "no support for multiple queries"
        head_size = query.shape[-1]

        # Sum the magnitudes within KV groups before top-k
        # topk indices shape -- (batch, n_kv_heads, 1, 1, rank)
        topk_indices = query.abs() \
                            .sum(dim=2, keepdim=True) \
                            .topk(dim=-1, k=self.settings.rank) \
                            .indices
                    
        query_proj = gather(query, -1, topk_indices)
        key_proj_buffer = query_proj.new_empty([*gpu_cur_key.shape[:-2], self.settings.rank, cpu_key.shape[-1] + 1])
                                    
        key_proj_buffer[...,:-1].copy_(gather(cpu_key.unsqueeze(2), -2, topk_indices.transpose(-1, -2).to("cpu",non_blocking=True)), non_blocking=True)
        key_proj_buffer[...,-1:] = gather(gpu_cur_key, -2, topk_indices.transpose(-1, -2))

        # Scale could be:
        #  - sqrt(head_size) -- if we think our approximation is exact
        #  - sqrt(rank)      -- if our approximation is no better than random
        #  - sqrt(q_coverage * head_size) -- used below
        #       q_coverage estimates the variance of Q K^T from the approximated
        #       product, and the L1 coverage of Q by the topk components
        scale = (
            query_proj.abs()
            .sum(-1)
            .div_(query.abs().sum(-1))
            .mul_(head_size)
            .pow_(0.5)
            .unsqueeze(-1)
        )
        return (query_proj @ key_proj_buffer).div_(scale)

    def forward_gpu(self, query: Tensor, gpu_key: Tensor, gpu_cur_key: Tensor) -> Tensor:
        """Compute approximate score for each (query, key).

        query -- (batch, n_kv_heads, n_heads_per_kv, 1, head_size)

        gpu_key -- (batch, n_kv_heads, head_size, seq_len)

        gpu_cur_key -- (batch, n_kv_heads, 1, head_size, 1)

        returns -- (batch, n_kv_heads, n_heads_per_kv, 1, key)
        """
        assert gpu_key.device != torch.device("cpu") and gpu_cur_key != torch.device("cpu")
        assert query.shape[-2] == 1, "no support for multiple queries"
        head_size = query.shape[-1]

        # Sum the magnitudes within KV groups before top-k
        # topk indices shape -- (batch, n_kv_heads, 1, 1, rank)
        topk_indices = query.abs() \
                            .sum(dim=2, keepdim=True) \
                            .topk(dim=-1, k=self.settings.rank) \
                            .indices
                    
        query_proj = gather(query, -1, topk_indices)
        key_proj_buffer = query_proj.new_empty([*gpu_cur_key.shape[:-2], self.settings.rank, gpu_key.shape[-1] + 1])
                                    
        key_proj_buffer[...,:-1].copy_(gather(gpu_key.unsqueeze(2), -2, topk_indices.transpose(-1, -2)), non_blocking=True)
        key_proj_buffer[...,-1:] = gather(gpu_cur_key, -2, topk_indices.transpose(-1, -2))

        # Scale could be:
        #  - sqrt(head_size) -- if we think our approximation is exact
        #  - sqrt(rank)      -- if our approximation is no better than random
        #  - sqrt(q_coverage * head_size) -- used below
        #       q_coverage estimates the variance of Q K^T from the approximated
        #       product, and the L1 coverage of Q by the topk components
        scale = (
            query_proj.abs()
            .sum(-1)
            .div_(query.abs().sum(-1))
            .mul_(head_size)
            .pow_(0.5)
            .unsqueeze(-1)
        )
        return (query_proj @ key_proj_buffer).div_(scale)


ScoreSettings = Union[LowRank.Settings, SparseQ.Settings]


@dataclass
class Settings:
    k: int
    local_k: int
    reallocate_to_mean_value: bool
    score: ScoreSettings
    sink: int

    def __init__(
        self,
        k: int,
        local_k: int,
        sink: int,
        reallocate_to_mean_value: bool,
        score: Union[ScoreSettings, str],
        **args: Any,
    ):
        if isinstance(score, str):
            ctor: Any = dict(low_rank=LowRank.Settings, sparse_q=SparseQ.Settings)[
                score
            ]
            score_settings: ScoreSettings = ctor(**args)
        else:
            assert (
                not args
            ), "ann_attention.Setting only accepts **args when `score` is a string"
            score_settings = score
        self.k = k
        self.local_k = local_k
        self.sink = sink
        self.reallocate_to_mean_value = reallocate_to_mean_value
        self.score = score_settings


class AnnAttention(nn.Module):
    """Generic ANN with local windowing and masking."""

    def __init__(self, settings: Settings, n_kv_heads: int, head_size: int, mean_v):
        super().__init__()
        self.settings = settings
        self.score: nn.Module
        self.mean_v = mean_v
        if isinstance(settings.score, LowRank.Settings):
            self.score = LowRank(settings.score, n_kv_heads, head_size)
        elif isinstance(settings.score, SparseQ.Settings):
            self.score = SparseQ(settings.score)
        else:
            raise ValueError(f"Unexpected settings.score = {settings.score}")
        # Set to an empty list to turn on ANN index logging
        self.debug_indices: Optional[List[Tensor]] = None

    def _attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        logmask: Tensor,
        kv_weight: Tensor,
        mean_value: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Dense attention, with left-over weight reallocation.

        query -- (batch, n_kv_heads, n_heads_per_kv, n_query, head_size)

        key -- (batch, n_kv_heads, 1, n_kv, head_size)

        value -- (batch, n_kv_heads, 1, n_heads, n_kv, head_size)

        logmask -- (batch, n_kv_heads, n_heads_per_kv, n_query, n_kv)

        kv_weight -- (batch, n_kv_heads, n_heads_per_kv, n_query) | ()
                  -- 1.0 for regular attention (no reallocation)

        mean_value -- (batch, n_kv_heads, n_heads_per_kv, n_query, head_size)
        """
        scores = (
            (query @ key.transpose(-1, -2)).div_(query.shape[-1] ** 0.5).add_(logmask)
        )
        weights = torch.softmax(scores, -1, dtype=torch.float32).to(value.dtype)
        # Value-mixing with reallocation
        weights *= kv_weight[..., None]
        output = weights @ value
        output += (1 - kv_weight[..., None]) * mean_value.expand([-1, -1, query.shape[2], -1, -1])
        return output, weights

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, logmask: Tensor, cpu_kv
    ) -> Tuple[Tensor, Tensor]:
        """Preprocess (key, value, mask) for ANN attention.

        query -- (batch, n_heads, 1, head_size)

        key -- (batch, n_kv_heads, 1, head_size)

        value -- (batch, n_kv_heads, 1, head_size)

        logmask -- (batch, n_heads, 1, seq)

        returns -- (output, weights)
                   output -- (batch, n_heads, 1, head_size)
                   weights -- (batch, n_heads, 1, seq)
        """
        batch, n_kv_heads, _, head_size = key.shape
        key_cpu, value_cpu, s_contiguous_key_cpu = cpu_kv
        seq = key_cpu.shape[-2] + 1
        n_heads_per_kv = query.shape[1] // n_kv_heads

        # Group by KV head
        query, key, value, logmask = map(
            partial(torch.unflatten, dim=1, sizes=(n_kv_heads, -1)),
            [query, key, value, logmask],
        )

        assert query.shape == (batch, n_kv_heads, n_heads_per_kv, 1, head_size), query.shape
        assert key.shape == (batch, n_kv_heads, 1, 1, head_size), key.shape
        assert value.shape == (batch, n_kv_heads, 1, 1, head_size), value.shape
        assert logmask.shape == (batch, n_kv_heads, n_heads_per_kv, 1, seq), logmask.shape

        # Calculate an approximate score for each (query, key) pair
        # shape -- (batch, n_kv_heads, n_heads_per_kv, 1, seq)
        score = (self.score(query, s_contiguous_key_cpu, key.transpose(3,4)) + logmask).float()

        # Set the score of local keys (+1 current) to max
        causal_index = sparse_attention.causal_index(logmask[...,:-1])

        # MODIFIED: add "sink token" trick.
        is_local = (0 <= causal_index) & (causal_index < self.settings.local_k)
        is_sink = (causal_index >= (seq - self.settings.sink - 1))
        assert torch.sum(is_sink[0,0,0,0]) == self.settings.sink
        topk_score = score[...,:-1].masked_fill(torch.logical_or(is_local, is_sink), torch.finfo(score.dtype).max).sum(
            dim=2, keepdim=True
        )
        # Find max-score keys (note: +1 because the current token's k comes "for free")
        indices = topk_score.topk(
            min(self.settings.k + self.settings.sink, score.shape[-1]), -1
        ).indices  # (batch, n_kv_heads, 1, 1, k + sink + 1)
        if self.debug_indices is not None:
            self.debug_indices.append(indices)

        # Optional "mean_value" kv, but we discard this trick here.
        # NOTE: different from original implementation, here we assumes logmask are full of zeros.
        # value_mask = (
        #     logmask[:, :, :1].squeeze(-2).unsqueeze(-1).exp()
        # )  # (batch, n_kv_heads, 1, seq, 1)

        # MODIFIED: datatype conversion, half -> float -> half
        self.mean_v = ((self.mean_v * (seq - 1) + value.to(torch.float32)) / seq) # (batch, n_kv_heads, 1, 1, dim)
        cur_mean_v = self.mean_v.half()

        # Do not use all value tensor to calculate mean value
        # mean_value = ((value * value_mask).to(torch.float32).sum(-2) / value_mask.to(torch.float32).sum(-2)).unsqueeze(
        #     -2
        # ).half()  # (batch, n_kv_heads, 1, 1, 1)
        kv_weight = torch.tensor(1.0, device=query.device)
        if self.settings.reallocate_to_mean_value:
            norm_score = torch.softmax(score, -1) # (batch, n_kv_heads, n_heads_per_kv, 1, seq)
            kv_weight = (
                gather(norm_score, -1, indices)  # no need to expand here
                .sum(-1)
                .add(norm_score[...,-1])
                .to(value.dtype)
            )  # (batch, n_kv_heads, n_heads_per_kv, 1)

        # Slice key, value, logmask for attention
        kv_indices = indices.squeeze(-2).unsqueeze(-1).cpu()  # (batch, n_kv_heads, 1, k, 1)
        
        key_gpu = query.new_empty([batch, n_kv_heads, 1, kv_indices.shape[-2] + 1, head_size])
        value_gpu = query.new_empty([batch, n_kv_heads, 1, kv_indices.shape[-2] + 1, head_size])
        
        key_gpu[...,:-1,:].copy_(gather(key_cpu.unsqueeze(2), -2, kv_indices), non_blocking=True)
        value_gpu[...,:-1,:].copy_(gather(value_cpu.unsqueeze(2), -2, kv_indices), non_blocking=True)
        
        key_gpu[...,-1:,:], value_gpu[...,-1:,:] = key, value
        cur_index = indices.new_zeros([batch, n_kv_heads, 1, 1, 1]) + seq - 1
        output, weights = self._attention(
            query,
            key_gpu,
            value_gpu,
            gather(logmask, -1, torch.concat([indices, cur_index], dim = -1)),
            kv_weight=kv_weight,
            mean_value=cur_mean_v,
        )
        # Note: expand indices as scatter does not broadcast (!)
        return output.flatten(1, 2)

    # Put KV cache on GPU, only for accuracy test
    def forward_gpu_kv(
        self, query: Tensor, key: Tensor, value: Tensor, logmask: Tensor, gpu_kv
    ) -> Tuple[Tensor, Tensor]:
        """Preprocess (key, value, mask) for ANN attention.

        query -- (batch, n_heads, 1, head_size)

        key -- (batch, n_kv_heads, 1, head_size)

        value -- (batch, n_kv_heads, 1, head_size)

        logmask -- (batch, n_heads, 1, seq)

        returns -- (output, weights)
                   output -- (batch, n_heads, 1, head_size)
                   weights -- (batch, n_heads, 1, seq)
        """
        batch, n_kv_heads, _, head_size = key.shape
        key_gpu, value_gpu, s_contiguous_key_gpu = gpu_kv
        seq = key_gpu.shape[-2] + 1
        n_heads_per_kv = query.shape[1] // n_kv_heads

        # Group by KV head
        query, key, value, logmask = map(
            partial(torch.unflatten, dim=1, sizes=(n_kv_heads, -1)),
            [query, key, value, logmask],
        )

        assert query.shape == (batch, n_kv_heads, n_heads_per_kv, 1, head_size), query.shape
        assert key.shape == (batch, n_kv_heads, 1, 1, head_size), key.shape
        assert value.shape == (batch, n_kv_heads, 1, 1, head_size), value.shape
        assert logmask.shape == (batch, n_kv_heads, n_heads_per_kv, 1, seq), logmask.shape

        # Calculate an approximate score for each (query, key) pair
        # shape -- (batch, n_kv_heads, n_heads_per_kv, 1, seq)
        score = (self.score.forward_gpu(query, s_contiguous_key_gpu, key.transpose(3,4)) + logmask).float()

        # Set the score of local keys (+1 current) to max
        causal_index = sparse_attention.causal_index(logmask[...,:-1])

        # MODIFIED: add "sink token" trick.
        is_local = (0 <= causal_index) & (causal_index < self.settings.local_k)
        is_sink = (causal_index >= (seq - self.settings.sink - 1))
        assert torch.sum(is_sink[0,0,0,0]) == self.settings.sink
        topk_score = score[...,:-1].masked_fill(torch.logical_or(is_local, is_sink), torch.finfo(score.dtype).max).sum(
            dim=2, keepdim=True
        )
        # Find max-score keys (note: +1 because the current token's k comes "for free")
        indices = topk_score.topk(
            min(self.settings.k + self.settings.sink, score.shape[-1]), -1
        ).indices  # (batch, n_kv_heads, 1, 1, k + sink + 1)
        if self.debug_indices is not None:
            self.debug_indices.append(indices)

        # Optional "mean_value" kv, but we discard this trick here.
        # NOTE: different from original implementation, here we assumes logmask are full of zeros.
        # value_mask = (
        #     logmask[:, :, :1].squeeze(-2).unsqueeze(-1).exp()
        # )  # (batch, n_kv_heads, 1, seq, 1)

        # MODIFIED: datatype conversion, half -> float -> half
        self.mean_v = ((self.mean_v * (seq - 1) + value.to(torch.float32)) / seq) # (batch, n_kv_heads, 1, 1, dim)
        cur_mean_v = self.mean_v.half()

        # Do not use all value tensor to calculate mean value
        # mean_value = ((value * value_mask).to(torch.float32).sum(-2) / value_mask.to(torch.float32).sum(-2)).unsqueeze(
        #     -2
        # ).half()  # (batch, n_kv_heads, 1, 1, 1)
        kv_weight = torch.tensor(1.0, device=query.device)
        if self.settings.reallocate_to_mean_value:
            norm_score = torch.softmax(score, -1) # (batch, n_kv_heads, n_heads_per_kv, 1, seq)
            kv_weight = (
                gather(norm_score, -1, indices)  # no need to expand here
                .sum(-1)
                .add(norm_score[...,-1])
                .to(value.dtype)
            )  # (batch, n_kv_heads, n_heads_per_kv, 1)

        # Slice key, value, logmask for attention
        kv_indices = indices.squeeze(-2).unsqueeze(-1)  # (batch, n_kv_heads, 1, k, 1)
        
        final_key = query.new_empty([batch, n_kv_heads, 1, kv_indices.shape[-2] + 1, head_size])
        final_value = query.new_empty([batch, n_kv_heads, 1, kv_indices.shape[-2] + 1, head_size])
        
        final_key[...,:-1,:].copy_(gather(key_gpu.unsqueeze(2), -2, kv_indices), non_blocking=True)
        final_value[...,:-1,:].copy_(gather(value_gpu.unsqueeze(2), -2, kv_indices), non_blocking=True)
        
        final_key[...,-1:,:], final_value[...,-1:,:] = key, value
        cur_index = indices.new_zeros([batch, n_kv_heads, 1, 1, 1]) + seq - 1
        output, weights = self._attention(
            query,
            final_key,
            final_value,
            gather(logmask, -1, torch.concat([indices, cur_index], dim = -1)),
            kv_weight=kv_weight,
            mean_value=cur_mean_v,
        )
        # Note: expand indices as scatter does not broadcast (!)
        return output.flatten(1, 2)


Model = Union[GPTNeoXForCausalLM, LlamaForCausalLM, MistralForCausalLM]


class GPTNeoXAttentionWithANN(GPTNeoXAttention):  # type:ignore[misc]
    def __init__(self, config: GPTNeoXConfig, settings: Settings):
        utility.check_transformers_version(type(self))
        super().__init__(config)
        self.ann = AnnAttention(settings, self.num_attention_heads, self.head_size)

    def _attn(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        assert attention_mask is not None
        assert head_mask is None

        # Only enable ANN during autoregressive generation
        if query.shape[-2] == 1:
            return self.ann(  # type:ignore[no-any-return]
                query,
                key,
                value,
                attention_mask.broadcast_to(key.unsqueeze(-3).shape[:-1]),
            )

        return super()._attn(  # type:ignore[no-any-return]
            query, key, value, attention_mask, head_mask
        )


class LlamaAttentionWithANN(llama_attention.LlamaAttention):
    def __init__(self, config: LlamaConfig, settings: Settings):
        utility.check_transformers_version(type(self))
        super().__init__(config)
        self.settings = settings
        self.ann = AnnAttention(settings, self.num_heads, self.head_dim)

    def _attn(
        self, query: Tensor, key: Tensor, value: Tensor, logmask: Tensor
    ) -> Tuple[Tensor, Tensor]:
        if query.shape[-2] == 1:
            return self.ann(  # type:ignore[no-any-return]
                query, key, value, logmask.broadcast_to(key.unsqueeze(-3).shape[:-1])
            )
        return super()._attn(query, key, value, logmask)


class MistralAttentionWithANN(mistral_attention.MistralAttention):
    def __init__(self, config: MistralConfig, settings: Settings, mean_v: torch.Tensor):
        utility.check_transformers_version(type(self))
        super().__init__(config)
        self.settings = settings
        self.ann = AnnAttention(
            settings,
            self.num_key_value_heads,
            self.head_dim,
            mean_v
        )

    def _attn(
        self, query: Tensor, key: Tensor, value: Tensor, logmask: Tensor, cpu_kv
    ) -> Tuple[Tensor, Tensor]:
        if query.shape[-2] == 1:
            return self.ann(  # type:ignore[no-any-return]
                query,
                key,
                value,
                # reshape to (batch, n_heads, 1, seq_len)
                # Do not change the last dimension size.
                logmask.broadcast_to(*query.shape[:-1], -1),
                cpu_kv
            )
        return super()._attn(query, key, value, logmask)


def convert(model: Model, settings: Settings) -> Model:
    """Convert a model to use KV cache compression using ANN."""

    def _replace(m: nn.Module) -> Optional[nn.Module]:
        if isinstance(m, GPTNeoXAttention):
            return GPTNeoXAttentionWithANN(model.config, settings)
        if isinstance(m, LlamaAttention):
            return LlamaAttentionWithANN(model.config, settings)
        if isinstance(m, MistralAttention):
            return MistralAttentionWithANN(model.config, settings)

    return utility.convert_module(model, _replace)
