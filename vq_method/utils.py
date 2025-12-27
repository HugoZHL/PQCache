"""
Shared utility functions for PQCache.

This module contains common tensor operations and helper functions used across
the PQCache codebase to eliminate code duplication and provide a single source
of truth for these utilities.
"""

from typing import TypeVar
import torch

T = TypeVar('T', torch.Tensor, 'torch.Tensor')


def repeat(tensor: torch.Tensor, size: int, dim_idx: int) -> torch.Tensor:
    """
    Repeat a tensor along a specified dimension (for GQA - Grouped Query Attention).

    This function is used to expand key-value heads to match query heads in
    grouped query attention mechanisms. It inserts a new dimension after dim_idx,
    expands along that dimension, and then reshapes to merge the expanded dimension.

    Args:
        tensor: Input tensor to repeat.
        size: Number of times to repeat along the dimension.
        dim_idx: The dimension index after which to insert and expand.

    Returns:
        Tensor with shape[dim_idx] multiplied by size.

    Example:
        >>> x = torch.randn(2, 4, 8)  # [batch, kv_heads, dim]
        >>> y = repeat(x, size=4, dim_idx=1)  # Repeat heads 4x for GQA
        >>> y.shape
        torch.Size([2, 16, 8])  # [batch, kv_heads * 4, dim]
    """
    shape = tensor.shape
    return (
        tensor.unsqueeze(dim_idx + 1)
        .expand(*shape[:dim_idx], shape[dim_idx], size, *shape[dim_idx + 1:])
        .reshape(*shape[:dim_idx], shape[dim_idx] * size, *shape[dim_idx + 1:])
    )


def unrepeat(tensor: torch.Tensor, size: int, dim_idx: int) -> torch.Tensor:
    """
    Reverse the repeat operation - reduce repeated heads back to original count.

    This function is the inverse of repeat(). It reshapes the tensor to separate
    the repeated dimension, then selects the first element along that dimension.

    Args:
        tensor: Input tensor with repeated dimension.
        size: The repeat factor that was used (number of groups).
        dim_idx: The dimension index that was repeated.

    Returns:
        Tensor with shape[dim_idx] divided by size.

    Example:
        >>> x = torch.randn(2, 16, 8)  # [batch, repeated_heads, dim]
        >>> y = unrepeat(x, size=4, dim_idx=1)  # Unrepeated from 4 groups
        >>> y.shape
        torch.Size([2, 4, 8])  # [batch, kv_heads, dim]

    Note:
        The select operation automatically squeezes the target dimension.
    """
    shape = tensor.shape
    return (
        tensor.reshape(*shape[:dim_idx], shape[dim_idx] // size, size, *shape[dim_idx + 1:])
        .select(dim_idx + 1, 0)
    )


class PQCacheError(Exception):
    """Base exception for PQCache errors."""
    pass


class ConfigurationError(PQCacheError):
    """Raised when configuration parameters are invalid."""
    pass


class CompressionError(PQCacheError):
    """Raised when compression operations fail."""
    pass


class CacheError(PQCacheError):
    """Raised when cache operations fail."""
    pass


class UnsupportedOperationError(PQCacheError):
    """Raised when an unsupported operation is attempted."""
    pass


def validate_batch_size(batch_size: int, max_supported: int = 1) -> None:
    """
    Validate that batch size is within supported range.

    Args:
        batch_size: The batch size to validate.
        max_supported: Maximum supported batch size.

    Raises:
        ConfigurationError: If batch size exceeds maximum supported.
    """
    if batch_size > max_supported:
        raise ConfigurationError(
            f"Batch size {batch_size} exceeds maximum supported batch size {max_supported}. "
            f"Multi-batch inference is not yet supported in adaptive compression mode."
        )


def validate_subvec_count(n_subvec: int) -> None:
    """
    Validate that the number of sub-vectors is a valid power of 2.

    Args:
        n_subvec: Number of sub-vectors per head for PQ.

    Raises:
        ConfigurationError: If n_subvec is not in [1, 2, 4, 8, 16].
    """
    valid_values = [1, 2, 4, 8, 16]
    if n_subvec not in valid_values:
        raise ConfigurationError(
            f"PQ sub-vector count must be one of {valid_values}, got {n_subvec}"
        )


def validate_gqa_support(use_gqa: bool, operation: str) -> None:
    """
    Validate that GQA mode is enabled for operations that require it.

    Args:
        use_gqa: Whether GQA mode is enabled.
        operation: Name of the operation being attempted.

    Raises:
        UnsupportedOperationError: If GQA is disabled for an operation requiring it.
    """
    if not use_gqa:
        raise UnsupportedOperationError(
            f"Operation '{operation}' currently requires GQA (Grouped Query Attention) mode. "
            f"Non-GQA mode is not yet supported."
        )


class SampledLogger:
    """
    Logger that samples output at a configurable rate to reduce log spam.

    This replaces the pattern of using random number checks for occasional
    logging, providing a more predictable and configurable approach.

    Attributes:
        sample_interval: Log every Nth call.
        _counters: Dictionary tracking call counts per message key.
    """

    def __init__(self, sample_interval: int = 1000):
        """
        Initialize the sampled logger.

        Args:
            sample_interval: Log every Nth call (default: 1000).
        """
        self.sample_interval = sample_interval
        self._counters: dict = {}

    def should_log(self, key: str = "default") -> bool:
        """
        Check if this call should produce log output.

        Args:
            key: Unique key to track separate call sites.

        Returns:
            True if this call should log, False otherwise.
        """
        if key not in self._counters:
            self._counters[key] = 0

        self._counters[key] += 1

        if self._counters[key] >= self.sample_interval:
            self._counters[key] = 0
            return True
        return False

    def reset(self, key: str = None) -> None:
        """
        Reset counter(s).

        Args:
            key: Specific key to reset, or None to reset all.
        """
        if key is None:
            self._counters.clear()
        elif key in self._counters:
            del self._counters[key]


# Global sampled logger instance
sampled_logger = SampledLogger(sample_interval=1000)
