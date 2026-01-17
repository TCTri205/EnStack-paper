"""
Unit tests for the dataset module.
"""

import pickle
import tempfile
from pathlib import Path

import pandas as pd
import pytest
import torch
from transformers import AutoTokenizer

from src.dataset import VulnerabilityDataset, create_dataloaders


@pytest.fixture
def sample_data():
    """Creates sample data for testing."""
    data = {
        "func": [
            "int add(int a, int b) { return a + b; }",
            "void print() { printf('Hello'); }",
            "int mul(int x, int y) { return x * y; }",
        ],
        "target": [0, 1, 0],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_data_file(sample_data):
    """Creates a temporary pickle file with sample data."""
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".pkl", delete=False) as f:
        pickle.dump(sample_data, f)
        temp_path = f.name
    yield temp_path
    # Cleanup
    Path(temp_path).unlink()


@pytest.fixture
def tokenizer():
    """Creates a tokenizer for testing."""
    return AutoTokenizer.from_pretrained("microsoft/codebert-base")


def test_vulnerability_dataset_init(sample_data_file, tokenizer):
    """Test VulnerabilityDataset initialization."""
    dataset = VulnerabilityDataset(sample_data_file, tokenizer, max_length=128)
    assert len(dataset) == 3
    assert dataset.max_length == 128


def test_vulnerability_dataset_getitem(sample_data_file, tokenizer):
    """Test VulnerabilityDataset __getitem__ method."""
    dataset = VulnerabilityDataset(sample_data_file, tokenizer, max_length=128)

    item = dataset[0]

    # Check keys
    assert "input_ids" in item
    assert "attention_mask" in item
    assert "labels" in item

    # Check shapes
    # No padding in __getitem__, so length should be <= max_length
    assert item["input_ids"].shape[0] <= 128
    assert item["attention_mask"].shape[0] <= 128
    assert item["labels"].shape == ()

    # Check types
    assert isinstance(item["input_ids"], torch.Tensor)
    assert isinstance(item["attention_mask"], torch.Tensor)
    assert isinstance(item["labels"], torch.Tensor)


def test_vulnerability_dataset_file_not_found(tokenizer):
    """Test VulnerabilityDataset with non-existent file."""
    with pytest.raises(FileNotFoundError):
        VulnerabilityDataset("non_existent_file.pkl", tokenizer, max_length=128)


def test_vulnerability_dataset_csv(tokenizer):
    """Test VulnerabilityDataset with CSV file."""
    data = pd.DataFrame(
        {
            "func": ["void foo() {}", "int bar() { return 0; }"],
            "target": [0, 1],
        }
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        data.to_csv(f.name, index=False)
        temp_path = f.name

    try:
        dataset = VulnerabilityDataset(temp_path, tokenizer, max_length=64)
        assert len(dataset) == 2
    finally:
        Path(temp_path).unlink()


def test_create_dataloaders(sample_data_file, tokenizer):
    """Test create_dataloaders function."""
    config = {
        "data": {
            "root_dir": str(Path(sample_data_file).parent),
            "train_file": Path(sample_data_file).name,
        },
        "training": {
            "batch_size": 2,
            "max_length": 128,
        },
    }

    train_loader, val_loader, test_loader = create_dataloaders(
        config, tokenizer, train_path=sample_data_file
    )

    # Check train loader
    assert train_loader is not None
    assert len(train_loader.dataset) == 3

    # Check batch
    batch = next(iter(train_loader))
    assert batch["input_ids"].shape[0] <= 2  # batch_size
    # Dynamic padding: length will be max length in batch, which is likely < 128 for short samples
    assert batch["input_ids"].shape[1] <= 128


def test_dataset_labels(sample_data_file, tokenizer):
    """Test that labels are correctly loaded."""
    dataset = VulnerabilityDataset(sample_data_file, tokenizer, max_length=64)

    labels = [dataset[i]["labels"].item() for i in range(len(dataset))]
    assert labels == [0, 1, 0]
