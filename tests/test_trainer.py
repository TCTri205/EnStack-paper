"""
Unit tests for the trainer module.
"""

import tempfile
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models import EnStackModel
from src.trainer import EnStackTrainer


@pytest.fixture
def device():
    """Returns CPU device for testing."""
    return torch.device("cpu")


@pytest.fixture
def dummy_model():
    """Creates a dummy model for testing."""
    return EnStackModel(
        model_name="microsoft/codebert-base",
        num_labels=5,
        pretrained=False,
    )


@pytest.fixture
def dummy_dataloader():
    """Creates a dummy dataloader for testing."""
    batch_size = 2
    seq_length = 64
    num_samples = 10

    # Create dummy data
    input_ids = torch.randint(0, 1000, (num_samples, seq_length))
    attention_mask = torch.ones(num_samples, seq_length)
    labels = torch.randint(0, 5, (num_samples,))

    # Create dataset
    dataset = TensorDataset(input_ids, attention_mask, labels)

    # Create custom collate function
    def collate_fn(batch):
        input_ids = torch.stack([item[0] for item in batch])
        attention_mask = torch.stack([item[1] for item in batch])
        labels = torch.stack([item[2] for item in batch])
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)


def test_trainer_init(dummy_model, dummy_dataloader, device):
    """Test EnStackTrainer initialization."""
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = EnStackTrainer(
            model=dummy_model,
            train_loader=dummy_dataloader,
            learning_rate=2e-5,
            device=device,
            output_dir=tmpdir,
        )

        assert trainer.model is not None
        assert trainer.train_loader is not None
        assert trainer.optimizer is not None
        assert trainer.device == device


def test_trainer_train_epoch(dummy_model, dummy_dataloader, device):
    """Test training for one epoch."""
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = EnStackTrainer(
            model=dummy_model,
            train_loader=dummy_dataloader,
            learning_rate=2e-5,
            device=device,
            output_dir=tmpdir,
        )

        # Train one epoch
        metrics = trainer.train_epoch(epoch=1)

        # Check metrics
        assert "loss" in metrics
        assert "accuracy" in metrics
        assert "f1" in metrics
        assert isinstance(metrics["loss"], float)
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["f1"] <= 1


def test_trainer_evaluate(dummy_model, dummy_dataloader, device):
    """Test evaluation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = EnStackTrainer(
            model=dummy_model,
            val_loader=dummy_dataloader,
            learning_rate=2e-5,
            device=device,
            output_dir=tmpdir,
        )

        # Evaluate
        metrics = trainer.evaluate(dummy_dataloader, split_name="Test")

        # Check metrics
        assert "loss" in metrics
        assert "accuracy" in metrics
        assert "f1" in metrics
        assert "precision" in metrics
        assert "recall" in metrics


def test_trainer_extract_features(dummy_model, dummy_dataloader, device):
    """Test feature extraction."""
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = EnStackTrainer(
            model=dummy_model,
            learning_rate=2e-5,
            device=device,
            output_dir=tmpdir,
        )

        # Extract features
        features = trainer.extract_features(dummy_dataloader)

        # Check shape (10 samples, 768 hidden dimensions for CodeBERT)
        assert features.shape == (10, 768)


def test_trainer_save_checkpoint(dummy_model, dummy_dataloader, device):
    """Test checkpoint saving."""
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = EnStackTrainer(
            model=dummy_model,
            train_loader=dummy_dataloader,
            learning_rate=2e-5,
            device=device,
            output_dir=tmpdir,
        )

        # Save checkpoint
        trainer.save_checkpoint("test_checkpoint")

        # Check if files exist
        checkpoint_dir = Path(tmpdir) / "test_checkpoint"
        assert checkpoint_dir.exists()
        assert (checkpoint_dir / "training_state.pth").exists()


def test_trainer_train_no_loader(dummy_model, device):
    """Test training without data loader raises error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = EnStackTrainer(
            model=dummy_model,
            learning_rate=2e-5,
            device=device,
            output_dir=tmpdir,
        )

        # Should raise error when train is called without train_loader
        with pytest.raises(ValueError):
            trainer.train(num_epochs=1)
