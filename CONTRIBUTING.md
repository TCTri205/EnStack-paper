# Contributing to EnStack

Thank you for considering contributing to EnStack! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)

## Code of Conduct

This project adheres to professional standards of conduct. By participating, you are expected to:

- Be respectful and inclusive
- Focus on technical merit and facts
- Accept constructive criticism gracefully
- Prioritize the project's best interests

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/EnStack-paper.git
   cd EnStack-paper
   ```
3. **Set up the development environment** (see below)

## Development Setup

### Prerequisites

- Python 3.8 or higher
- Git
- Virtual environment tool (venv, conda, etc.)

### Setup Instructions

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies (if not in requirements.txt)
pip install pytest black ruff mypy

# Verify installation
python -c "import torch; import transformers; print('Setup successful!')"
```

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:

- A clear, descriptive title
- Steps to reproduce the problem
- Expected behavior
- Actual behavior
- Python version and OS
- Relevant logs or error messages

### Suggesting Enhancements

Enhancement suggestions are welcome! Please provide:

- A clear use case
- Expected behavior
- Potential implementation approach (optional)
- Any relevant examples or references

### Contributing Code

1. **Check existing issues** to see if your idea is already being discussed
2. **Create a new issue** if needed to discuss your proposal
3. **Fork and create a branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Make your changes** following the coding standards
5. **Add tests** for new functionality
6. **Update documentation** as needed
7. **Submit a pull request**

## Coding Standards

Please follow the guidelines in [AGENTS.md](AGENTS.md). Key points:

### Style

- **Line length**: Maximum 88 characters (Black default)
- **Naming conventions**:
  - Classes: `PascalCase`
  - Functions/Methods: `snake_case`
  - Constants: `UPPER_SNAKE_CASE`
  - Private members: `_leading_underscore`

### Type Hints

**Required** for all function signatures:

```python
def process_data(input_path: str, max_size: int = 1000) -> List[Dict[str, Any]]:
    """Process data from file."""
    ...
```

### Docstrings

Use **Google-style** docstrings:

```python
def train_model(data: pd.DataFrame, epochs: int) -> Dict[str, float]:
    """
    Trains the model on provided data.

    Args:
        data (pd.DataFrame): Training data.
        epochs (int): Number of training epochs.

    Returns:
        Dict[str, float]: Training metrics.

    Raises:
        ValueError: If data is empty.
    """
    ...
```

### Logging

**Never use `print()`** for output. Use the `logging` module:

```python
import logging
logger = logging.getLogger("EnStack")

logger.info("Training started")
logger.warning("Low memory available")
logger.error("Failed to load checkpoint")
```

### Error Handling

Use specific exceptions:

```python
# Good
try:
    data = load_file(path)
except FileNotFoundError:
    logger.error(f"File not found: {path}")
    raise
except PermissionError:
    logger.error(f"Permission denied: {path}")
    raise

# Bad
try:
    data = load_file(path)
except:
    print("Error!")
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_dataset.py

# Run with coverage
pytest --cov=src tests/

# Run with verbose output
pytest -v
```

### Writing Tests

- Place tests in `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Use fixtures for common setup
- Test both success and failure cases

Example:

```python
import pytest
from src.models import EnStackModel

def test_model_initialization():
    """Test that model initializes correctly."""
    model = EnStackModel("microsoft/codebert-base", num_labels=5)
    assert model.num_labels == 5
    assert model.model is not None

def test_model_invalid_labels():
    """Test that invalid num_labels raises error."""
    with pytest.raises(ValueError):
        EnStackModel("microsoft/codebert-base", num_labels=-1)
```

## Pull Request Process

1. **Update documentation** if you're changing functionality
2. **Add tests** for new features or bug fixes
3. **Run the test suite** and ensure all tests pass:
   ```bash
   pytest
   ```
4. **Run code quality checks**:
   ```bash
   black src/ tests/
   ruff check src/ tests/
   mypy src/
   ```
5. **Update CHANGELOG.md** with your changes
6. **Create a pull request** with:
   - Clear title describing the change
   - Reference to related issues
   - Description of changes and rationale
   - Screenshots (if UI changes)
   - Test results

### PR Review Checklist

Before requesting review, ensure:

- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] New code has tests
- [ ] Documentation is updated
- [ ] Type hints are added
- [ ] No linting errors
- [ ] Commit messages are clear

### Commit Message Guidelines

Write clear, descriptive commit messages:

```
# Good
Add feature extraction method to EnStackModel
Fix tokenization bug for GraphCodeBERT
Update README with installation instructions

# Bad
Update
Fix bug
Changes
```

## Code Review Process

- Maintainers will review your PR
- Address feedback and update your PR
- Once approved, a maintainer will merge your changes
- PRs are typically reviewed within 3-5 business days

## Questions?

If you have questions:

1. Check the [documentation](README.md)
2. Look at [existing issues](https://github.com/YOUR_USERNAME/EnStack-paper/issues)
3. Open a new issue with the "question" label

## Recognition

Contributors will be recognized in:
- README.md contributors section
- CHANGELOG.md for significant contributions
- Release notes

Thank you for contributing to EnStack!
