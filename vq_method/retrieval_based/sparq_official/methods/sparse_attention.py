# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

"""Sparse attention mechanisms.

Note the following definitions:

 - 'mask' is `True` or `1` for unmasked/retained tokens, `False` or `0` for
   masked-out tokens
 - `score` (often `x`) is near `FP16_MIN` for masked-out tokens
"""
import unittest.mock as um
from contextlib import contextmanager
from functools import partial
from typing import Any, Iterator, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers.models.gpt_neox.modeling_gpt_neox import (
    GPTNeoXAttention,
    GPTNeoXModel,
)

_softmax = F.softmax


def topk_mask(x: Tensor, k: int, dim: int = -1) -> Tensor:
    """Mask selecting top-k positions along a dimension of `x` (default: last).

    Ties are broken arbitarily, enforcing the constraint
    `(out.sum(-1) <= k).all()`
    """
    fmin = torch.finfo(x.dtype).min
    topk_idxs = x.nan_to_num(fmin).topk(k, dim).indices
    return torch.zeros_like(x, dtype=torch.bool).scatter(dim, topk_idxs, 1)


def score_to_mask(score: Tensor, threshold: float = 0.5) -> Tensor:
    """Convert a score tensor to a mask.

    threshold (float): how strict to be (values below `threshold*FP16_MIN` are
    considered masked-out); between 0 and 1.

    Returns: boolean 'mask' tensor, the same shape as `score`
    """
    return threshold * torch.finfo(torch.float16).min < score


def causal_index(score: Tensor) -> Tensor:
    """Get the 'causal index' of a tokens in history, from masked attention scores.

    The causal index is the number of unmasked tokens between key token and
    the query, counting backwards

    Args:
        score (Tensor): shape (*, q_len, k_len), should be set to -finfo.min
        for masked-out values

    Returns:
        Tensor: shape (*, q_len, k_len), containing the caual index of each
        token according to the mask, or -1 if the token is masked-out
    """
    mask = score_to_mask(score)
    cumsum = mask.flip(-1).cumsum(-1, dtype=torch.int32).flip(-1)
    return cumsum.sub_(1).masked_fill_(mask.logical_not_(), -1)


def sparse_softmax_fixed_k(
    x: Tensor,
    k: int,
    dtype: Optional[torch.dtype] = None,
    dim: int = -1,
    add_avg: bool = False,
    apply_after_softmax: bool = True,
    out_weights: Optional[Tensor] = None,
    generation_only: bool = True,
    kv_group_size: int = 1,
) -> Tensor:
    """Applies softmax accross last dimension, keeping top k
    elements of the output.

    Args:
        x (Tensor): shape (batch_size, num_heads, q_len, k_len)
        k (int): Number of attention scores to keep
        dim (int, optional): Assumed dim = -1
        add_avg (bool, optional): If True, assign the non-top-k
        softmax weight equally to non-top-k tokens.
        apply_after_softmax (bool, optional): apply the top-k mask after softmax
        out_weights (Optional[Tensor], optional): shape (broadcastable to x)
        If passed, multiplies softmax scores with out_weights
        before choosing top-k.
        generation_only (bool, optional): only apply the sparse softmax when
        x.shape[-2] == 1 (causal generation)
        kv_group_size (int, optional): number of query heads per kv head (GQA)

    Returns:
        Tensor: shape (batch_size, num_heads, q_len, k_len)
    """
    assert dim == -1
    assert not (
        add_avg and not apply_after_softmax
    ), "add_avg requires apply_after_softmax"
    assert (
        kv_group_size == 1 or apply_after_softmax
    ), "GQA supported only for when applying top-k after softmax"
    if dtype is not None:
        x = x.to(dtype)
    if out_weights is None:
        out_weights = torch.tensor(1.0, dtype=x.dtype, device=x.device)

    if k >= x.shape[-1] or (generation_only and x.shape[-2] != 1):
        return _softmax(x, dim=-1)

    if not apply_after_softmax:
        mask = topk_mask(x + torch.log(out_weights), k)
        return _softmax(x.masked_fill(~mask, torch.finfo(x.dtype).min), dim=-1)

    y = _softmax(x, dim=-1)
    y_grouped = torch.unflatten(y * out_weights, dim=1, sizes=(-1, kv_group_size))
    mask = (
        topk_mask(y_grouped.sum(dim=2, keepdim=True), k)
        .expand_as(y_grouped)
        .flatten(1, 2)
    )
    y *= mask
    if add_avg:
        # '1' for tokens that were removed by the topk operation, which can
        # receive the average reallocation
        removed_by_topk = (~mask) & score_to_mask(x)
        y += (
            removed_by_topk
            * (1 - y.sum(dim=-1, keepdim=True))
            / removed_by_topk.sum(-1, keepdim=True)
        )
    return y


def sparse_softmax_fixed_p(
    x: Tensor, p: float, k_min: int = 1, dim: int = -1
) -> Tensor:
    """Applies softmax accross last dimension, keeping top k
    elements of the output.
    k is calculated as p * num of tokens attended to by the query

    Args:
        x (Tensor): shape (batch_size, num_heads, q_len, k_len)
        p (float): Proportion of attention scores to keep
        k_min (int): Minimum number of attention scores kept
        dim (int, optional): Assumed dim = -1

    Returns:
        Tensor: shape (batch_size, num_heads, q_len, k_len)
    """
    assert dim == -1
    assert len(x.shape) >= 2
    q_len, k_len = x.shape[-2:]
    y = _softmax(x, dim=-1)
    if k_min >= x.shape[-1]:
        return y

    # For each query, take `kn` (shape: (q_len,)) keys
    kn = torch.maximum(
        torch.tensor(k_min),
        (p * torch.arange(start=k_len - q_len + 1, end=k_len + 1)).long(),
    )
    # Take a topk based on the max kn, then mask before scattering to obey kn
    knmax = int(kn[-1].item())
    topk = torch.topk(y, knmax, dim=-1)
    return torch.zeros_like(y).scatter(
        -1, topk.indices, topk.values * (torch.arange(knmax) < kn[:, None])
    )


def local_softmax(
    x: Tensor,
    k: int,
    initial_k: int = 0,
    apply_after_softmax: bool = False,
    dim: int = -1,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """Applies softmax across last dimension, keeping k values.

    Always keeps a maximum of `k` values.
     - `k - initial_k` most recent
     - `initial_k` oldest

    Args:
        x (Tensor): shape (batch_size, num_heads, q_len, k_len)
        k (int): Keep `k` output scores
        initial_k (int, optional): Keep `initial_k` from the beginning
        apply_after_softmax (bool, optional):
        If True, set corresponding softmax output elements to 0.
        If False, mask corresponding inputs to softmax to `finfo.min`.
        Defaults to False.
        dim (int, optional): Assumed dim = -1.

    Returns:
        Tensor: shape (batch_size, num_heads, q_len, k_len)
    """
    assert dim == -1
    assert len(x.shape) >= 2
    if dtype is not None:
        x = x.to(dtype)
    index = causal_index(x)
    max_index = index.max(dim=-1, keepdim=True).values
    local_mask = (index < k - initial_k).logical_or_(max_index - initial_k < index)
    if apply_after_softmax:
        return _softmax(x, dim=-1).mul_(local_mask)
    return _softmax(
        x.masked_fill(local_mask.logical_not_(), torch.finfo(x.dtype).min), dim=-1
    )


@contextmanager
def number_attention_layers(model: GPTNeoXModel) -> Iterator[GPTNeoXModel]:
    try:
        for i, layer in enumerate(model.layers):
            layer.attention.layer_idx = i
        yield model
    finally:
        for layer in model.layers:
            del layer.attention.layer_idx


original_attn = GPTNeoXAttention._attn


def sparse_attn(
    self: GPTNeoXAttention,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attention_mask: Optional[Tensor] = None,
    head_mask: Optional[Tensor] = None,
    use_v_mag: bool = False,
    k_per_layer: Optional[List[int]] = None,
    **sa_args: Any
) -> Tuple[Tensor, Tensor]:
    if use_v_mag:
        # value.shape = (batch_size, num_heads, k_len, head_dim)
        # v_norm.shape = (batch_size, num_heads, 1, k_len)
        v_norm = value.norm(dim=-1).unsqueeze(dim=-2)
    else:
        v_norm = None
    if k_per_layer is not None:
        assert hasattr(self, "layer_idx"), (
            "Attention object does not have layer_idx attribute,"
            " wrap within number_attention_layers contextmanager beforehand."
        )
        sa_args["k"] = k_per_layer[self.layer_idx]
    with um.patch(
        "torch.nn.functional.softmax",
        partial(sparse_softmax_fixed_k, **sa_args, out_weights=v_norm),
    ):
        out: Tuple[Tensor, Tensor] = original_attn(
            self, query, key, value, attention_mask, head_mask
        )
    return out
