# EnStack: Stacking Ensemble for Vulnerability Detection

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.20+-orange.svg)](https://huggingface.co/transformers/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

EnStack is a stacking ensemble framework for vulnerability detection using Large Language Models (LLMs) like CodeBERT, GraphCodeBERT, and UniXcoder.

## Overview

This project implements a two-stage ensemble approach:
1. **Base Models**: Train multiple transformer-based models (CodeBERT, GraphCodeBERT, UniXcoder) on vulnerability detection tasks
2. **Meta-Classifier**: Combine predictions from base models using a stacking ensemble with SVM/Logistic Regression/Random Forest

## Project Structure

```
EnStack-paper/
├── src/                      # Core source code
│   ├── __init__.py
│   ├── dataset.py           # Dataset and data loading utilities
│   ├── models.py            # Model architectures (EnStackModel)
│   ├── trainer.py           # Training and evaluation logic
│   ├── stacking.py          # Stacking ensemble implementation
│   └── utils.py             # Utility functions (config, logging, etc.)
├── configs/                  # Configuration files
│   └── config.yaml          # Main configuration file
├── notebooks/                # Jupyter notebooks for Colab
│   └── main_pipeline.ipynb  # End-to-end training pipeline
├── tests/                    # Unit tests
│   ├── test_dataset.py
│   ├── test_models.py
│   └── test_trainer.py
├── scripts/                  # Helper scripts
├── docs/                     # Documentation
├── requirements.txt          # Python dependencies
├── .gitignore
├── AGENTS.md                 # Agent guidelines
└── README.md
```

## Installation

### Local Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/EnStack-paper.git
cd EnStack-paper

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Google Colab Setup

The project is designed to run on Google Colab. See the [main_pipeline.ipynb](notebooks/main_pipeline.ipynb) notebook for a complete walkthrough.

1. Open the notebook in Google Colab
2. Mount your Google Drive
3. Clone the repository
4. Install dependencies
5. Run the pipeline cells

## Quick Start

### 1. Prepare Your Data

Ensure your data is in the format expected by the dataset module:
- Pickle, CSV, or Parquet files
- Columns: `func` (source code), `target` (vulnerability label)

### 2. Configure the Project

Edit `configs/config.yaml` to set:
- Data paths (local or Google Drive)
- Model selection (codebert, graphcodebert, unixcoder)
- Training hyperparameters (batch size, learning rate, epochs)
- Meta-classifier type (svm, lr, rf, xgboost)

### 3. Train Models

#### Using Python Script

```python
from src.utils import load_config, setup_logging, set_seed, get_device
from src.models import create_model
from src.dataset import create_dataloaders
from src.trainer import EnStackTrainer

# Load configuration
config = load_config("configs/config.yaml")
set_seed(config["training"]["seed"])
device = get_device()

# Create model and data
model, tokenizer = create_model("codebert", config)
train_loader, val_loader, test_loader = create_dataloaders(config, tokenizer)

# Train
trainer = EnStackTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
)
trainer.train(num_epochs=10)
```

#### Using Colab Notebook

Run all cells in `notebooks/main_pipeline.ipynb` for a complete training pipeline.

## Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_dataset.py

# Run with coverage
pytest --cov=src tests/
```

## Code Quality

```bash
# Format code with Black
black src/ tests/

# Lint with Ruff
ruff check src/ tests/

# Type checking with mypy
mypy src/
```

## Configuration

The `configs/config.yaml` file contains all configurable parameters:

```yaml
data:
  root_dir: "/content/drive/MyDrive/EnStack_Data"
  train_file: "train_processed.pkl"
  val_file: "val_processed.pkl"
  test_file: "test_processed.pkl"

model:
  base_models: 
    - "codebert"
    - "graphcodebert"
    - "unixcoder"
  meta_classifier: "svm"
  num_labels: 5

training:
  batch_size: 16
  epochs: 10
  learning_rate: 2.0e-5
  max_length: 512
  seed: 42
  output_dir: "/content/drive/MyDrive/EnStack_Data/checkpoints"
```

## Key Features

- **Modular Architecture**: Clean separation of data, models, training, and stacking logic
- **Multiple Base Models**: Support for CodeBERT, GraphCodeBERT, and UniXcoder
- **Flexible Meta-Classifier**: Choose from SVM, Logistic Regression, Random Forest, or XGBoost
- **Google Colab Integration**: Seamless execution on cloud GPUs
- **Comprehensive Testing**: Unit tests for all core components
- **Type Safety**: Full type hints for better code quality
- **Logging**: Structured logging instead of print statements
- **Configuration Management**: Centralized YAML-based configuration

## Architecture

### Base Models
Each base model is trained independently on the vulnerability detection task:
- **CodeBERT**: Pre-trained on code and natural language
- **GraphCodeBERT**: Incorporates data flow information
- **UniXcoder**: Universal cross-modal pre-training

### Stacking Ensemble
1. Extract feature embeddings (CLS token) from each base model
2. Concatenate features to form meta-features
3. Train a meta-classifier on these meta-features
4. Final predictions come from the meta-classifier

## Performance Metrics

The framework reports the following metrics:
- **Accuracy**: Overall classification accuracy
- **F1 Score**: Weighted F1 score across all classes
- **Precision**: Weighted precision
- **Recall**: Weighted recall

## Development Guidelines

Please refer to [AGENTS.md](AGENTS.md) for detailed guidelines on:
- Code style and naming conventions
- Type hinting requirements
- Testing procedures
- Logging best practices
- Configuration management

## Contributing

1. Follow the code style guidelines in AGENTS.md
2. Write tests for new features
3. Run linters and formatters before committing
4. Update documentation as needed

## License

[Specify your license here]

## Citation

If you use this code in your research, please cite:

```bibtex
@article{enstack2024,
  title={EnStack: Stacking Ensemble for Vulnerability Detection},
  author={Your Name},
  journal={Your Journal/Conference},
  year={2024}
}
```

## 📦 Handover & Quick Start

- **For New Users**: See [QUICKSTART_USER.md](QUICKSTART_USER.md) - Get started in 5 minutes on Colab
- **For Project Handover**: See [HANDOVER.md](HANDOVER.md) - Complete handover documentation

## Contact

For questions or issues, please open an issue on GitHub: https://github.com/TCTri205/EnStack-paper/issues

## Acknowledgments

This project uses the following pre-trained models:
- [CodeBERT](https://github.com/microsoft/CodeBERT) by Microsoft
- [GraphCodeBERT](https://github.com/microsoft/CodeBERT) by Microsoft
- [UniXcoder](https://github.com/microsoft/CodeBERT) by Microsoft

---

**Note**: Remember to update the Google Drive paths in `configs/config.yaml` to match your own Google Drive structure before running on Colab.
