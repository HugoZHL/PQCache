"""
Pytest configuration and shared fixtures for PQCache tests.
"""

import pytest
import torch
import numpy as np


@pytest.fixture
def device():
    """Get appropriate device for testing."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


@pytest.fixture
def dtype():
    """Default dtype for test tensors."""
    return torch.float16


@pytest.fixture
def sample_kv_tensors(device, dtype):
    """Create sample key-value tensors for testing."""
    batch_size = 1
    n_heads = 8
    seq_len = 1024
    head_dim = 128

    keys = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device, dtype=dtype)
    values = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device, dtype=dtype)

    return keys, values


@pytest.fixture
def sample_query(device, dtype):
    """Create a sample query tensor for testing."""
    batch_size = 1
    n_heads = 32  # More query heads than KV heads for GQA
    q_len = 1
    head_dim = 128

    return torch.randn(batch_size, n_heads, q_len, head_dim, device=device, dtype=dtype)


@pytest.fixture
def compression_config():
    """Default compression configuration for tests."""
    return {
        "compress_ratio": 0.2,
        "recent_ratio": 0.5,
        "sink_size": 32,
        "n_subvec_per_head": 2,
        "n_subbits": 6,
    }
