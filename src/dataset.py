"""
Dataset module for EnStack vulnerability detection.

This module provides dataset classes for loading and preprocessing vulnerability data
for training and evaluation with transformer-based models.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger("EnStack")


class VulnerabilityDataset(Dataset):
    """
    PyTorch Dataset for vulnerability detection tasks.

    This dataset handles tokenization and preprocessing of source code
    for vulnerability detection using transformer models.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        text_column: str = "func",
        label_column: str = "target",
    ) -> None:
        """
        Initialize the VulnerabilityDataset.

        Args:
            data_path (str): Path to the data file (pickle, csv, or parquet).
            tokenizer (PreTrainedTokenizer): HuggingFace tokenizer instance.
            max_length (int): Maximum sequence length for tokenization.
            text_column (str): Name of the column containing source code.
            label_column (str): Name of the column containing labels.

        Raises:
            FileNotFoundError: If the data file does not exist.
            ValueError: If required columns are missing from the data.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_column = text_column
        self.label_column = label_column

        # Load data
        data_file = Path(data_path)
        if not data_file.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        self.data = self._load_data(data_file)

        # Validate columns
        if text_column not in self.data.columns:
            raise ValueError(f"Text column '{text_column}' not found in data")
        if label_column not in self.data.columns:
            raise ValueError(f"Label column '{label_column}' not found in data")

        logger.info(
            f"Loaded {len(self.data)} samples from {data_path} "
            f"(max_length={max_length})"
        )

    def _load_data(self, data_file: Path) -> pd.DataFrame:
        """
        Load data from various file formats.

        Args:
            data_file (Path): Path to the data file.

        Returns:
            pd.DataFrame: Loaded data.

        Raises:
            ValueError: If file format is not supported.
        """
        suffix = data_file.suffix.lower()

        if suffix == ".pkl" or suffix == ".pickle":
            with open(data_file, "rb") as f:
                data = pickle.load(f)
            # Convert to DataFrame if it's a dict or list
            if isinstance(data, dict):
                data = pd.DataFrame(data)
            elif isinstance(data, list):
                data = pd.DataFrame(data)
        elif suffix == ".csv":
            data = pd.read_csv(data_file)
        elif suffix == ".parquet":
            data = pd.read_parquet(data_file)
        else:
            raise ValueError(
                f"Unsupported file format: {suffix}. "
                "Supported formats: .pkl, .csv, .parquet"
            )

        # Ensure it's a DataFrame
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        return data

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns a single sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing:
                - input_ids: Token IDs (shape: [max_length])
                - attention_mask: Attention mask (shape: [max_length])
                - labels: Target label (scalar)
        """
        row = self.data.iloc[idx]
        text = str(row[self.text_column])
        label = int(row[self.label_column])

        # Tokenize the text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def create_dataloaders(
    config: Dict,
    tokenizer: PreTrainedTokenizer,
    train_path: Optional[str] = None,
    val_path: Optional[str] = None,
    test_path: Optional[str] = None,
) -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
    """
    Creates DataLoader instances for training, validation, and testing.

    Args:
        config (Dict): Configuration dictionary containing training parameters.
        tokenizer (PreTrainedTokenizer): Tokenizer to use for all datasets.
        train_path (Optional[str]): Path to training data. If None, uses config.
        val_path (Optional[str]): Path to validation data. If None, uses config.
        test_path (Optional[str]): Path to test data. If None, uses config.

    Returns:
        Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
            Train, validation, and test DataLoaders (None if path not provided).
    """
    batch_size = config["training"]["batch_size"]
    max_length = config["training"]["max_length"]
    root_dir = config["data"]["root_dir"]

    # Use config paths if not explicitly provided
    if train_path is None and "train_file" in config["data"]:
        train_path = str(Path(root_dir) / config["data"]["train_file"])
    if val_path is None and "val_file" in config["data"]:
        val_path = str(Path(root_dir) / config["data"]["val_file"])
    if test_path is None and "test_file" in config["data"]:
        test_path = str(Path(root_dir) / config["data"]["test_file"])

    # Create datasets
    train_loader = None
    val_loader = None
    test_loader = None

    if train_path and Path(train_path).exists():
        train_dataset = VulnerabilityDataset(
            train_path, tokenizer, max_length=max_length
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )
        logger.info(f"Created train DataLoader with {len(train_dataset)} samples")

    if val_path and Path(val_path).exists():
        val_dataset = VulnerabilityDataset(val_path, tokenizer, max_length=max_length)
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )
        logger.info(f"Created validation DataLoader with {len(val_dataset)} samples")

    if test_path and Path(test_path).exists():
        test_dataset = VulnerabilityDataset(test_path, tokenizer, max_length=max_length)
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )
        logger.info(f"Created test DataLoader with {len(test_dataset)} samples")

    return train_loader, val_loader, test_loader
