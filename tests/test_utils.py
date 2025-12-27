"""
Unit tests for vq_method.utils module.
"""

import pytest
import torch
import numpy as np

from vq_method.utils import (
    repeat,
    unrepeat,
    validate_batch_size,
    validate_subvec_count,
    validate_gqa_support,
    ConfigurationError,
    UnsupportedOperationError,
)


class TestRepeatUnrepeat:
    """Tests for repeat and unrepeat tensor operations."""

    def test_repeat_basic(self):
        """Test basic repeat operation."""
        x = torch.randn(2, 4, 8)
        result = repeat(x, size=3, dim_idx=1)

        assert result.shape == (2, 12, 8)

    def test_repeat_first_dim(self):
        """Test repeat on first dimension."""
        x = torch.randn(4, 8, 16)
        result = repeat(x, size=2, dim_idx=0)

        assert result.shape == (8, 8, 16)

    def test_repeat_last_dim(self):
        """Test repeat on last dimension."""
        x = torch.randn(2, 4, 8)
        result = repeat(x, size=4, dim_idx=2)

        assert result.shape == (2, 4, 32)

    def test_unrepeat_basic(self):
        """Test basic unrepeat operation."""
        x = torch.randn(2, 12, 8)
        result = unrepeat(x, size=3, dim_idx=1)

        assert result.shape == (2, 4, 8)

    def test_repeat_unrepeat_roundtrip(self):
        """Test that unrepeat reverses repeat."""
        original = torch.randn(2, 4, 8)
        repeated = repeat(original, size=3, dim_idx=1)
        unrepeated = unrepeat(repeated, size=3, dim_idx=1)

        # First slice should match original
        assert torch.allclose(unrepeated, original)

    def test_repeat_preserves_dtype(self):
        """Test that repeat preserves tensor dtype."""
        x = torch.randn(2, 4, 8, dtype=torch.float16)
        result = repeat(x, size=2, dim_idx=1)

        assert result.dtype == torch.float16

    def test_repeat_preserves_device(self):
        """Test that repeat preserves tensor device."""
        if torch.cuda.is_available():
            x = torch.randn(2, 4, 8, device="cuda:0")
            result = repeat(x, size=2, dim_idx=1)

            assert result.device == x.device

    def test_gqa_use_case(self):
        """Test repeat/unrepeat for GQA (Grouped Query Attention) use case."""
        # Simulate KV heads (8) being expanded to match query heads (32)
        kv_heads = 8
        num_kv_groups = 4  # 32 / 8

        kv = torch.randn(1, kv_heads, 100, 128)
        expanded_kv = repeat(kv, size=num_kv_groups, dim_idx=1)

        assert expanded_kv.shape == (1, 32, 100, 128)

        # Verify unrepeat gets back to original shape
        reduced_kv = unrepeat(expanded_kv, size=num_kv_groups, dim_idx=1)
        assert reduced_kv.shape == (1, kv_heads, 100, 128)


class TestValidation:
    """Tests for validation functions."""

    def test_validate_batch_size_valid(self):
        """Test valid batch size passes."""
        validate_batch_size(1)  # Should not raise

    def test_validate_batch_size_invalid(self):
        """Test invalid batch size raises error."""
        with pytest.raises(ConfigurationError) as exc_info:
            validate_batch_size(2)

        assert "Batch size 2 exceeds maximum" in str(exc_info.value)

    def test_validate_batch_size_custom_max(self):
        """Test validation with custom max."""
        validate_batch_size(4, max_supported=4)  # Should not raise

        with pytest.raises(ConfigurationError):
            validate_batch_size(5, max_supported=4)

    def test_validate_subvec_count_valid(self):
        """Test valid subvec counts pass."""
        for valid in [1, 2, 4, 8, 16]:
            validate_subvec_count(valid)  # Should not raise

    def test_validate_subvec_count_invalid(self):
        """Test invalid subvec counts raise error."""
        for invalid in [0, 3, 5, 7, 32]:
            with pytest.raises(ConfigurationError) as exc_info:
                validate_subvec_count(invalid)

            assert "must be one of" in str(exc_info.value)

    def test_validate_gqa_support_enabled(self):
        """Test GQA validation passes when enabled."""
        validate_gqa_support(use_gqa=True, operation="test_op")  # Should not raise

    def test_validate_gqa_support_disabled(self):
        """Test GQA validation fails when disabled."""
        with pytest.raises(UnsupportedOperationError) as exc_info:
            validate_gqa_support(use_gqa=False, operation="test_op")

        assert "test_op" in str(exc_info.value)
        assert "GQA" in str(exc_info.value)


class TestExceptions:
    """Tests for custom exception hierarchy."""

    def test_configuration_error_message(self):
        """Test ConfigurationError can be raised with message."""
        with pytest.raises(ConfigurationError) as exc_info:
            raise ConfigurationError("Test config error")

        assert "Test config error" in str(exc_info.value)

    def test_unsupported_operation_error(self):
        """Test UnsupportedOperationError can be raised."""
        with pytest.raises(UnsupportedOperationError):
            raise UnsupportedOperationError("Not supported")
