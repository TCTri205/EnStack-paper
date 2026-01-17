# Changelog

All notable changes to the EnStack project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-01-16

### Added
- Initial release of EnStack framework
- Core modules:
  - `src/utils.py`: Configuration management, logging, and utility functions
  - `src/dataset.py`: VulnerabilityDataset class for data loading and preprocessing
  - `src/models.py`: EnStackModel wrapper for transformer-based models
  - `src/trainer.py`: EnStackTrainer for training and evaluation
  - `src/stacking.py`: Stacking ensemble implementation with meta-classifiers
- Support for three base models:
  - CodeBERT
  - GraphCodeBERT
  - UniXcoder
- Support for four meta-classifiers:
  - Support Vector Machine (SVM)
  - Logistic Regression (LR)
  - Random Forest (RF)
  - XGBoost
- Google Colab integration:
  - `notebooks/main_pipeline.ipynb`: Complete training pipeline for Colab
  - `scripts/setup_colab.sh`: Automated setup script
- Testing suite:
  - `tests/test_dataset.py`: Dataset module tests
  - `tests/test_models.py`: Model module tests
  - `tests/test_trainer.py`: Trainer module tests
- Documentation:
  - `README.md`: Comprehensive project documentation
  - `AGENTS.md`: Guidelines for AI agents and developers
  - `TODO_PLAN.md`: Development roadmap
- Configuration:
  - `configs/config.yaml`: Centralized YAML configuration
  - `pyproject.toml`: Tool configuration for pytest, black, ruff, mypy
- Scripts:
  - `scripts/train.py`: Command-line training interface
  - `scripts/quickstart.py`: Quick start example
- Development tools:
  - `.gitignore`: Git ignore rules
  - `requirements.txt`: Python dependencies

### Features
- Modular and extensible architecture
- Type hints throughout the codebase
- Comprehensive logging instead of print statements
- Flexible configuration management
- Support for multiple data formats (pickle, CSV, parquet)
- Automatic checkpoint saving with best model selection
- Feature extraction for ensemble learning
- Comprehensive evaluation metrics (accuracy, F1, precision, recall)
- Google Colab compatibility for cloud GPU training

### Documentation
- Complete README with installation and usage instructions
- Inline documentation with Google-style docstrings
- Agent guidelines for code quality and best practices
- FAQ and troubleshooting guides in `docs/`

### Testing
- Unit tests for all core modules
- pytest configuration with coverage support
- Continuous testing workflow setup

## [Unreleased]

### Planned
- Support for additional transformer models
- Advanced hyperparameter tuning
- Model interpretability features
- Performance benchmarking suite
- Docker containerization
- CI/CD pipeline setup
- Integration with MLflow for experiment tracking

---

For more details on each release, see the [release notes](https://github.com/YOUR_USERNAME/EnStack-paper/releases).
