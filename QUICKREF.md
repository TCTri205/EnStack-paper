# Quick Reference Guide

Quick commands and reference for common tasks in EnStack.

## Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/EnStack-paper.git
cd EnStack-paper

# Install dependencies
pip install -r requirements.txt
```

## Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=src tests/

# Specific module
pytest tests/test_dataset.py

# Verbose
pytest -v

# Stop on first failure
pytest -x
```

## Code Quality

```bash
# Format all code
black src/ tests/ scripts/

# Check formatting (without changing files)
black --check src/ tests/ scripts/

# Lint
ruff check src/ tests/ scripts/

# Type check
mypy src/

# Fix auto-fixable issues
ruff check --fix src/ tests/ scripts/
```

## Training

### Quick Start (Single Model)
```bash
python scripts/quickstart.py
```

### Full Pipeline (All Models + Stacking)
```bash
python scripts/train.py --config configs/config.yaml
```

### Custom Training
```bash
# Train specific models
python scripts/train.py --models codebert graphcodebert

# Custom epochs and batch size
python scripts/train.py --epochs 5 --batch-size 32

# Custom output directory
python scripts/train.py --output-dir ./my_checkpoints

# Skip base model training (use existing)
python scripts/train.py --skip-training

# Train only base models (skip stacking)
python scripts/train.py --skip-stacking
```

## Python API Usage

### Load Configuration
```python
from src.utils import load_config, setup_logging, set_seed

config = load_config("configs/config.yaml")
logger = setup_logging()
set_seed(42)
```

### Create Model
```python
from src.models import create_model

model, tokenizer = create_model("codebert", config, pretrained=True)
```

### Create DataLoaders
```python
from src.dataset import create_dataloaders

train_loader, val_loader, test_loader = create_dataloaders(
    config, tokenizer
)
```

### Train Model
```python
from src.trainer import EnStackTrainer
from src.utils import get_device

device = get_device()

trainer = EnStackTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
)

history = trainer.train(num_epochs=10)
```

### Extract Features
```python
features = trainer.extract_features(train_loader)
```

### Train Meta-Classifier
```python
from src.stacking import (
    prepare_meta_features,
    train_meta_classifier,
    evaluate_meta_classifier
)

# Prepare meta features from multiple models
meta_features, _ = prepare_meta_features(
    [features_model1, features_model2, features_model3],
    labels
)

# Train meta-classifier
meta_classifier = train_meta_classifier(
    meta_features,
    labels,
    classifier_type="svm"
)

# Evaluate
metrics = evaluate_meta_classifier(
    meta_classifier,
    test_meta_features,
    test_labels
)
```

## Configuration

### Edit config.yaml
```yaml
data:
  root_dir: "/path/to/your/data"
  train_file: "train.pkl"
  
model:
  base_models: ["codebert", "graphcodebert"]
  meta_classifier: "svm"
  
training:
  batch_size: 16
  epochs: 10
  learning_rate: 2.0e-5
```

## Google Colab

### Setup
```python
# In Colab notebook
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/YOUR_USERNAME/EnStack-paper.git
%cd EnStack-paper
!pip install -r requirements.txt
```

### Run Pipeline
Open and run `notebooks/main_pipeline.ipynb`

## Common Issues

### CUDA Out of Memory
```python
# Reduce batch size in config.yaml
training:
  batch_size: 8  # or smaller
```

### Data Not Found
```python
# Update data paths in config.yaml
data:
  root_dir: "/correct/path/to/data"
```

### Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

## File Locations

- **Source Code**: `src/`
- **Tests**: `tests/`
- **Configuration**: `configs/config.yaml`
- **Notebooks**: `notebooks/main_pipeline.ipynb`
- **Scripts**: `scripts/`
- **Checkpoints**: Defined in `config.yaml` (default: `output_dir`)

## Environment Variables

```bash
# Optional: Set custom config path
export ENSTACK_CONFIG=/path/to/config.yaml

# Optional: Set custom output directory
export ENSTACK_OUTPUT_DIR=/path/to/output
```

## Useful Python Snippets

### Load Saved Model
```python
from src.models import EnStackModel

model = EnStackModel.load_pretrained(
    "/path/to/checkpoint",
    num_labels=5
)
```

### Save Meta-Classifier
```python
from src.stacking import save_meta_classifier

save_meta_classifier(meta_classifier, "meta_clf.pkl")
```

### Load Meta-Classifier
```python
from src.stacking import load_meta_classifier

meta_classifier = load_meta_classifier("meta_clf.pkl")
```

### Get Predictions
```python
# From base model
predictions = torch.argmax(logits, dim=-1)

# From meta-classifier
predictions = meta_classifier.predict(meta_features)
```

## Performance Tips

1. **Use GPU**: Ensure CUDA is available
   ```python
   import torch
   print(torch.cuda.is_available())
   ```

2. **Optimize Batch Size**: Use largest batch size that fits in memory

3. **Use Mixed Precision**: For faster training (requires AMP)
   ```python
   from torch.cuda.amp import autocast, GradScaler
   ```

4. **Data Loading**: Increase `num_workers` in DataLoader
   ```python
   DataLoader(dataset, num_workers=4)
   ```

## Debugging

### Enable Debug Logging
```python
from src.utils import setup_logging
import logging

logger = setup_logging(level=logging.DEBUG)
```

### Check Model Output Shapes
```python
batch = next(iter(train_loader))
outputs = model(**batch)
print(f"Logits shape: {outputs['logits'].shape}")
print(f"Loss: {outputs['loss']}")
```

### Verify Data Loading
```python
print(f"Train samples: {len(train_loader.dataset)}")
print(f"Batches per epoch: {len(train_loader)}")

# Check first batch
batch = next(iter(train_loader))
print(f"Input IDs shape: {batch['input_ids'].shape}")
print(f"Labels: {batch['labels']}")
```

## Getting Help

1. Check [README.md](README.md)
2. Review [CONTRIBUTING.md](CONTRIBUTING.md)
3. Look at example in [scripts/quickstart.py](scripts/quickstart.py)
4. Open an issue on GitHub

---

For more detailed information, see the full [README.md](README.md) and [documentation](docs/).
