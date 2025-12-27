# Contributing to PQCache

Thank you for your interest in contributing to PQCache! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Reporting Issues](#reporting-issues)

## Code of Conduct

Please be respectful and constructive in all interactions. We welcome contributions from everyone regardless of experience level.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/PQCache.git
   cd PQCache
   ```
3. Add the upstream repository:
   ```bash
   git remote add upstream https://github.com/HugoZHL/PQCache.git
   ```

## Development Setup

### Prerequisites

- Python 3.10+
- CUDA 12.1+ (for GPU support)
- GPU with Compute Capability >= 80 (A100, RTX 3090/4090, etc.)

### Installation

1. Create a conda environment:
   ```bash
   conda create -n pqcache-dev python=3.10
   conda activate pqcache-dev
   ```

2. Install PyTorch with CUDA:
   ```bash
   pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121
   ```

3. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

4. Build the C++ LFU cache extension:
   ```bash
   cd vq_method/retrieval_based/lfu
   mkdir -p build && cd build
   cmake .. && make
   cd ../../../..
   ```

## Code Style

We follow these coding conventions:

### Python Style

- Use [Black](https://github.com/psf/black) for code formatting (line length: 100)
- Use [isort](https://github.com/PyCQA/isort) for import sorting
- Use [Ruff](https://github.com/astral-sh/ruff) for linting
- Add type hints to all function signatures
- Write docstrings for all public functions and classes (Google style)

### Running Formatters

```bash
# Format code
black .
isort .

# Check linting
ruff check .

# Type checking
mypy vq_method/
```

### Docstring Example

```python
def compress_kv_cache(
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    compress_ratio: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compress key-value cache using PQ-based selection.

    Args:
        key_states: Key tensor of shape [batch, heads, seq_len, dim].
        value_states: Value tensor of shape [batch, heads, seq_len, dim].
        compress_ratio: Target compression ratio (0-1).

    Returns:
        Tuple of compressed (keys, values) tensors.

    Raises:
        ValueError: If compress_ratio is not in (0, 1].
    """
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=vq_method --cov-report=html

# Run specific test file
pytest tests/test_utils.py

# Run in parallel (requires pytest-xdist)
pytest -n auto
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files as `test_<module>.py`
- Use descriptive test names: `test_repeat_preserves_dtype`
- Group related tests in classes
- Use fixtures from `conftest.py` for common setup

### Test Categories

1. **Unit tests**: Test individual functions/methods
2. **Integration tests**: Test component interactions
3. **GPU tests**: Mark with `@pytest.mark.gpu` if requiring CUDA

## Submitting Changes

### Pull Request Process

1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit:
   ```bash
   git add .
   git commit -m "Add feature: description of changes"
   ```

3. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

4. Open a Pull Request on GitHub

### PR Guidelines

- **Title**: Use a clear, descriptive title
- **Description**: Explain what and why, not just how
- **Tests**: Add tests for new functionality
- **Documentation**: Update docs if needed
- **Breaking changes**: Clearly note any breaking changes

### Commit Messages

Follow conventional commits format:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation only
- `test:` Adding tests
- `refactor:` Code refactoring
- `perf:` Performance improvement
- `chore:` Maintenance tasks

Example:
```
feat: add support for Llama-3.2 model

- Update llama_patch.py for new architecture
- Add configuration for 3.2 variants
- Update README with new model support
```

## Reporting Issues

### Bug Reports

Include:
1. **Description**: What happened vs. what you expected
2. **Reproduction steps**: Minimal code to reproduce
3. **Environment**: Python version, CUDA version, GPU model
4. **Error messages**: Full traceback if applicable

### Feature Requests

Include:
1. **Use case**: Why is this feature needed?
2. **Proposed solution**: How might it work?
3. **Alternatives**: Other approaches considered

## Questions?

- Open a [GitHub Issue](https://github.com/HugoZHL/PQCache/issues) for questions
- Check existing issues before creating new ones

Thank you for contributing to PQCache!
