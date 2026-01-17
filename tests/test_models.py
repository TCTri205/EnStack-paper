"""
Unit tests for the models module.
"""

import pytest
import torch

from src.models import EnStackModel, create_model


@pytest.fixture
def device():
    """Returns CPU device for testing."""
    return torch.device("cpu")


@pytest.fixture
def sample_config():
    """Returns a sample configuration."""
    return {
        "model": {
            "model_map": {
                "codebert": "microsoft/codebert-base",
                "graphcodebert": "microsoft/graphcodebert-base",
            },
            "num_labels": 5,
        },
        "training": {
            "seed": 42,
        },
    }


def test_enstack_model_init():
    """Test EnStackModel initialization."""
    model = EnStackModel(
        model_name="microsoft/codebert-base",
        num_labels=5,
        pretrained=False,
    )

    assert model.num_labels == 5
    assert model.model is not None


def test_enstack_model_forward(device):
    """Test EnStackModel forward pass."""
    model = EnStackModel(
        model_name="microsoft/codebert-base",
        num_labels=5,
        pretrained=False,
    )
    model.to(device)
    model.eval()

    # Create dummy input
    batch_size = 2
    seq_length = 128
    input_ids = torch.randint(0, 1000, (batch_size, seq_length)).to(device)
    attention_mask = torch.ones(batch_size, seq_length).to(device)
    labels = torch.randint(0, 5, (batch_size,)).to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids, attention_mask, labels)

    # Check outputs
    assert "logits" in outputs
    assert "loss" in outputs
    assert outputs["logits"].shape == (batch_size, 5)
    assert outputs["loss"].numel() == 1


def test_enstack_model_get_embedding(device):
    """Test EnStackModel get_embedding method."""
    model = EnStackModel(
        model_name="microsoft/codebert-base",
        num_labels=5,
        pretrained=False,
    )
    model.to(device)
    model.eval()

    # Create dummy input
    batch_size = 2
    seq_length = 128
    input_ids = torch.randint(0, 1000, (batch_size, seq_length)).to(device)
    attention_mask = torch.ones(batch_size, seq_length).to(device)

    # Get embedding
    with torch.no_grad():
        embeddings = model.get_embedding(input_ids, attention_mask)

    # Check shape (CodeBERT has 768 hidden dimensions)
    assert embeddings.shape == (batch_size, 768)


def test_enstack_model_get_logits(device):
    """Test EnStackModel get_logits method."""
    model = EnStackModel(
        model_name="microsoft/codebert-base",
        num_labels=5,
        pretrained=False,
    )
    model.to(device)
    model.eval()

    # Create dummy input
    batch_size = 2
    seq_length = 64
    input_ids = torch.randint(0, 1000, (batch_size, seq_length)).to(device)
    attention_mask = torch.ones(batch_size, seq_length).to(device)

    # Get logits
    with torch.no_grad():
        logits = model.get_logits(input_ids, attention_mask)

    assert logits.shape == (batch_size, 5)


def test_create_model(sample_config):
    """Test create_model factory function."""
    model, tokenizer = create_model("codebert", sample_config, pretrained=False)

    assert isinstance(model, EnStackModel)
    assert tokenizer is not None
    assert model.num_labels == 5


def test_create_model_invalid_name(sample_config):
    """Test create_model with invalid model name."""
    with pytest.raises(ValueError):
        create_model("invalid_model", sample_config, pretrained=False)


def test_model_forward_without_labels(device):
    """Test forward pass without labels."""
    model = EnStackModel(
        model_name="microsoft/codebert-base",
        num_labels=5,
        pretrained=False,
    )
    model.to(device)
    model.eval()

    # Create dummy input
    batch_size = 2
    seq_length = 64
    input_ids = torch.randint(0, 1000, (batch_size, seq_length)).to(device)
    attention_mask = torch.ones(batch_size, seq_length).to(device)

    # Forward pass without labels
    with torch.no_grad():
        outputs = model(input_ids, attention_mask, labels=None)

    # Check outputs
    assert "logits" in outputs
    assert "loss" not in outputs
    assert outputs["logits"].shape == (batch_size, 5)
