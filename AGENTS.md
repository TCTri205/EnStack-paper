# EnStack Agent Guidelines

This document provides comprehensive guidelines for AI agents and developers working on the **EnStack** project. EnStack is a stacking ensemble framework for vulnerability detection using Large Language Models (LLMs) like CodeBERT, GraphCodeBERT, and UniXcoder.

## 1. Project Overview & Environment

- **Language:** Python 3.8+
- **Frameworks:** PyTorch, Transformers (Hugging Face), Scikit-learn
- **Key Dependencies:** `torch`, `transformers`, `pandas`, `tree-sitter`
- **Runtime:** Local development with execution often targeted for Google Colab.

### Environment Setup
Agents should assume a standard Python environment.

```bash
# Install dependencies
pip install -r requirements.txt

# Install dev dependencies (if not in requirements.txt)
pip install pytest ruff mypy black
```

## 2. Build, Lint, and Test Commands

Agents must verify all changes using the following commands before submitting code.

### Running Tests
Since standard `pytest` is preferred for automated verification:

```bash
# Run all tests
pytest

# Run a specific test file
pytest tests/test_dataset.py

# Run a specific test case
pytest tests/test_models.py::test_enstack_model_forward
```

*Note: If `tests/` directory is missing, agents should propose creating one based on `src/` logic.*

### Linting & Formatting
Strict adherence to code quality is required.

```bash
# Check for linting errors (using Ruff or Flake8)
ruff check src/ tests/

# Format code (using Black)
black src/ tests/

# Type checking
mypy src/
```

## 3. Code Style Guidelines

All code modifications must adhere to the following conventions.

### 3.1 Naming Conventions
- **Classes:** `PascalCase` (e.g., `EnStackModel`, `VulnerabilityDataset`)
- **Functions/Methods:** `snake_case` (e.g., `train_model`, `extract_features`)
- **Variables:** `snake_case` (e.g., `input_ids`, `validation_loss`)
- **Constants:** `UPPER_SNAKE_CASE` (e.g., `MAX_SEQUENCE_LENGTH`, `BATCH_SIZE`)
- **Private Members:** Prefix with `_` (e.g., `_load_weights`)

### 3.2 Type Hinting
**Mandatory** for all function signatures. Use the `typing` module for complex types.

```python
from typing import List, Dict, Optional
import numpy as np

def extract_features(self, loader: DataLoader) -> List[np.ndarray]:
    """Extracts features from the data loader."""
    ...
```

### 3.3 Docstrings
Use **Google Style** docstrings for all modules, classes, and functions.

```python
def evaluate(self, loader: DataLoader) -> Dict[str, float]:
    """
    Evaluates the model on a given dataset.

    Args:
        loader (DataLoader): DataLoader containing evaluation data.

    Returns:
        Dict[str, float]: Dictionary containing metrics (accuracy, F1, etc.).
    """
    ...
```

### 3.4 Imports
Group imports in the following order:
1.  Standard library imports (`os`, `sys`, `typing`)
2.  Third-party library imports (`torch`, `transformers`, `sklearn`)
3.  Local application imports (`src.models`, `src.utils`)

### 3.5 Error Handling
Use specific exception handling. **Never** use bare `except:`.

```python
try:
    checkpoint = torch.load(path)
except FileNotFoundError:
    logger.error(f"Checkpoint not found at {path}")
    raise
except KeyError as e:
    logger.error(f"Invalid checkpoint format: {e}")
    raise ValueError(f"Checkpoint missing key: {e}")
```

### 3.6 Logging
**Do not use `print()`**. Use the `logging` module.

```python
import logging
logger = logging.getLogger(__name__)

logger.info("Starting training epoch...")
logger.warning("Validation loss increasing.")
```

## 4. Project Structure & Architecture

Agents should respect the existing modular structure in `src/`.

- **`src/dataset.py`**:
    - Class `VulnerabilityDataset(Dataset)`: Handles data loading and preprocessing.
    - Must handle tokenization for CodeBERT/GraphCodeBERT/UniXcoder.
- **`src/models.py`**:
    - Class `EnStackModel(nn.Module)`: Wrapper for transformer models.
    - Implements `get_embedding()` for the stacking phase.
- **`src/trainer.py`**:
    - Class `EnStackTrainer`: Manages training, evaluation, and saving.
- **`src/utils.py`**:
    - Configuration loading, logging setup, and metric calculations.
- **`configs/config.yaml`**:
    - Centralized configuration for hyperparameters and paths.

## 5. Workflow & Deployment

### Development Loop
1.  **Local Dev:** Edit code in `src/`.
2.  **Verification:** Run tests and linters locally.
3.  **Colab Integration:** The code is designed to be pulled into Google Colab.
    - Agents should ensure `src/` modules are importable when the repo is cloned in Colab.
    - Avoid hardcoded paths that only work on local machines (use `config.yaml`).

### Data Handling
- Large datasets (Draper VDISC) and model checkpoints are stored in **Google Drive**, not the repo.
- Code should expect data paths to be provided via config or arguments, defaulting to standard Drive mount paths (e.g., `/content/drive/MyDrive/...`).

## 6. Security & Best Practices

- **Secrets:** NEVER commit API keys, tokens, or credentials. Use environment variables.
- **File Permissions:** Check permissions before writing to sensitive files.
- **Paths:** Always use absolute paths in tool calls when operating as an agent.
- **Model Safety:** Validate model inputs and handle potential OOM (Out of Memory) errors gracefully (e.g., by suggesting batch size reduction).

## 7. Configuration Management

Configuration should be decoupled from code. Use `configs/config.yaml` for:
- Hyperparameters (batch size, learning rate, epochs)
- Model selection (`base_models`, `meta_classifier`)
- Data paths (`root_dir`, `train_file`)

Example `config.yaml` structure to adhere to:
```yaml
data:
  root_dir: "/content/drive/MyDrive/EnStack_Data"
training:
  batch_size: 16
  max_length: 512
model:
  base_models: ["codebert", "graphcodebert", "unixcoder"]
```

---
*This file is auto-generated to guide autonomous agents. Please update if project conventions change.*
