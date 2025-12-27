"""
Retrieval-based compressor base module for PQCache.

This module provides the base class and utility functions for retrieval-based
KV-cache compression, including recall calculation for evaluating compression quality.
"""

import os
import time
from typing import Tuple

import numpy as np
import torch

from ..utils import repeat, unrepeat

recall_history: list = []


def calc_recall(
    query: torch.Tensor,
    key: torch.Tensor,
    dummy_topk_indices: torch.Tensor,
    num_kv_group: int,
    topk_size: int
) -> Tuple[float, float, float]:
    """
    Calculate recall rate between predicted top-k indices and ground truth.

    This function measures how well the PQ-based approximate search identifies
    the truly important tokens compared to exact attention computation.

    Args:
        query: Query tensor of shape [batch, n_heads, q_len, dim].
        key: Key tensor of shape [batch, kv_heads, kv_seq_len, dim].
        dummy_topk_indices: Predicted top-k indices from PQ search.
        num_kv_group: Number of query heads per KV head (for GQA).
        topk_size: Number of top tokens selected.

    Returns:
        Tuple of (current_recall, historical_mean, historical_variance).

    Raises:
        ValueError: If query and key shapes are incompatible.
    """
    bsz, kv_head, kv_seq_len, dim = key.shape
    _, n_head, q_len, _ = query.shape

    if key.shape[1] * num_kv_group == query.shape[1]:
        real_weight = query @ repeat(key, num_kv_group, 1).transpose(2, 3)
    elif key.shape[1] == query.shape[1]:
        real_weight = query @ key.transpose(2, 3)
    else:
        raise ValueError(
            f"Incompatible shapes: key {key.shape}, query {query.shape}, "
            f"num_kv_group {num_kv_group}"
        )

    real_topk_indices = real_weight.topk(k=topk_size, dim=-1, largest=True).indices
    assert real_topk_indices.shape == (bsz, n_head, q_len, topk_size)

    if dummy_topk_indices.shape[1] != real_topk_indices.shape[1]:
        dummy_topk_indices = repeat(dummy_topk_indices, num_kv_group, 1)

    dummy_topk_indices = dummy_topk_indices.flatten(0, 1)
    real_topk_indices = real_topk_indices.flatten(0, 1)
    ground_truth_idx_cnt = topk_size * n_head * bsz
    hit_idx_cnt = 0

    for i in range(dummy_topk_indices.shape[0]):
        dummy = dummy_topk_indices[i, 0, :]
        real = real_topk_indices[i, 0, :]
        assert dummy.numel() == torch.unique(dummy).numel()
        comparison = torch.isin(dummy, real, assume_unique=True)
        hit_idx_cnt += torch.sum(comparison.int())

    result = hit_idx_cnt / ground_truth_idx_cnt

    recall_history.append(result.item())
    np_history = np.array(recall_history)
    mean, var = np.mean(np_history), np.var(np_history)
    return result, mean, var


class RetrievalBasedCompressor:
    """
    Base class for retrieval-based KV-cache compression.

    This class provides the foundation for compression strategies that use
    retrieval mechanisms to identify and retain important tokens in the
    KV-cache during LLM inference.

    Subclasses implement specific retrieval strategies (e.g., PQ-based search,
    H2O scoring) while this base class handles common profiling and device
    management.

    Attributes:
        device: The CUDA device for tensor operations.
        profile_metric: Dictionary tracking various timing and data metrics
            for performance profiling.
    """

    def __init__(self, **kwargs) -> None:
        """
        Initialize the compressor with device and profiling infrastructure.

        Args:
            **kwargs: Keyword arguments. Must include 'cur_device' specifying
                the CUDA device to use.
        """
        self.profile_metric = {
            "prefill_time": 0,
            "prefill_attn_time": 0,
            "prefill_cnt": 0,
            "prefill_per_layer_time": 0,
            "prepare_idx_elapsed": 0,
            "prepare_idx_cnt": 0,
            "decoding_time": 0,
            "decoding_cnt": 0,
            "decoding_attn_time": 0,
            "offload_pq_ref_elapsed": 0,
            "offload_kv_elapsed": 0,
            "offload_cnt": 0,
            "offload_pq_ref_bytes": 0,
            "offload_kv_bytes": 0,
            "fetch_ref_elapsed": 0,
            "fetch_kv_elapsed": 0,
            "cpu_gather_elapsed": 0,
            "calc_dummy_weight_elapsed": 0,
            "fetch_ref_data_bytes": 0,
            "fetch_kv_data_bytes": 0,
        }

        self.device = kwargs["cur_device"]

    def profile_ckpt(self) -> float:
        """
        Create a profiling checkpoint by synchronizing CUDA and recording time.

        Returns:
            Current time in seconds from perf_counter.
        """
        torch.cuda.synchronize(self.device)
        return time.perf_counter()

    def reset(self) -> None:
        """Reset all profiling metrics to zero."""
        for k in self.profile_metric:
            self.profile_metric[k] = 0

    def showtime(self) -> None:
        """
        Display and save profiling results.

        Prints metrics to stdout and appends to a profile result file
        in the ./profile_result/ directory.
        """
        result_str = "\n".join(
            [f"{key} : {value}" for key, value in self.profile_metric.items()]
        )
        print("-----profile result show:\n", result_str)
        with open(f"./profile_result/mistral_profile_{os.getpid()}", "a") as f:
            f.write(result_str)
