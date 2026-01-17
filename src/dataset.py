"""
Dataset module for EnStack vulnerability detection.

This module provides dataset classes for loading and preprocessing vulnerability data
for training and evaluation with transformer-based models.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger("EnStack")


class DataCollatorWithPadding:
    """
    Data collator that dynamically pads batches to the maximum length in each batch.

    This is more efficient than static padding as it reduces computation on padding tokens.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, padding: bool = True):
        """
        Initialize the DataCollator.

        Args:
            tokenizer (PreTrainedTokenizer): Tokenizer used for padding.
            padding (bool): Whether to pad the batch.
        """
        self.tokenizer = tokenizer
        self.padding = padding

    def __call__(
        self, features: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of samples with dynamic padding.

        Args:
            features (List[Dict[str, torch.Tensor]]): List of samples.

        Returns:
            Dict[str, torch.Tensor]: Batched and padded tensors.
        """
        # Separate labels from input features
        labels = torch.stack([f["labels"] for f in features])

        # Remove labels from features for padding
        batch = [{k: v for k, v in f.items() if k != "labels"} for f in features]

        # Pad to the maximum length in this batch
        if self.padding:
            # Find max length in current batch
            max_length = max(f["input_ids"].shape[0] for f in batch)

            # Pad each sample to max_length
            input_ids = []
            attention_mask = []

            for f in batch:
                seq_len = f["input_ids"].shape[0]
                pad_len = max_length - seq_len

                # Pad input_ids with tokenizer.pad_token_id
                padded_input = torch.cat(
                    [
                        f["input_ids"],
                        torch.full(
                            (pad_len,), self.tokenizer.pad_token_id, dtype=torch.long
                        ),
                    ]
                )
                input_ids.append(padded_input)

                # Pad attention_mask with 0
                padded_mask = torch.cat(
                    [f["attention_mask"], torch.zeros(pad_len, dtype=torch.long)]
                )
                attention_mask.append(padded_mask)

            batch_dict = {
                "input_ids": torch.stack(input_ids),
                "attention_mask": torch.stack(attention_mask),
                "labels": labels,
            }
        else:
            # No padding (not recommended for batching)
            batch_dict = {
                "input_ids": torch.stack([f["input_ids"] for f in batch]),
                "attention_mask": torch.stack([f["attention_mask"] for f in batch]),
                "labels": labels,
            }

        return batch_dict


class VulnerabilityDataset(Dataset):
    """
    PyTorch Dataset for vulnerability detection tasks.

    This dataset handles tokenization and preprocessing of source code
    for vulnerability detection using transformer models.

    Supports both eager loading (load all data into memory) and lazy loading
    (load data on-the-fly from disk) for memory efficiency.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        text_column: str = "func",
        label_column: str = "target",
        lazy_loading: bool = False,
        cache_tokenization: bool = False,
    ) -> None:
        """
        Initialize the VulnerabilityDataset.

        Args:
            data_path (str): Path to the data file (pickle, csv, or parquet).
            tokenizer (PreTrainedTokenizer): HuggingFace tokenizer instance.
            max_length (int): Maximum sequence length for tokenization.
            text_column (str): Name of the column containing source code.
            label_column (str): Name of the column containing labels.
            lazy_loading (bool): If True, uses lazy loading to reduce memory usage.
            cache_tokenization (bool): If True, caches tokenized inputs to disk
                to avoid re-tokenizing in every epoch. Recommended for large datasets.

        Raises:
            FileNotFoundError: If the data file does not exist.
            ValueError: If required columns are missing from the data.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_column = text_column
        self.label_column = label_column
        self.lazy_loading = lazy_loading
        self.data_path = data_path
        self.cache_tokenization = cache_tokenization

        # Load data
        data_file = Path(data_path)
        if not data_file.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        self.data_file = data_file

        # Setup tokenization cache directory
        if cache_tokenization:
            model_slug = tokenizer.__class__.__name__.lower().replace("tokenizer", "")
            self.cache_dir = (
                data_file.parent / f".cache_{data_file.stem}_{model_slug}_{max_length}"
            )
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Tokenization cache enabled: {self.cache_dir}")

        if lazy_loading:
            # For lazy loading, only load metadata (length and columns)
            self._initialize_lazy_loading()
        else:
            # Eager loading: load entire dataset into memory
            self.data = self._load_data(data_file)

            # Validate columns
            if text_column not in self.data.columns:
                raise ValueError(f"Text column '{text_column}' not found in data")
            if label_column not in self.data.columns:
                raise ValueError(f"Label column '{label_column}' not found in data")

        logger.info(
            f"Loaded {len(self)} samples from {data_path} "
            f"(max_length={max_length}, lazy_loading={lazy_loading}, cache={cache_tokenization})"
        )

    def _initialize_lazy_loading(self) -> None:
        """
        Initialize lazy loading by reading only the dataset length.
        """
        suffix = self.data_file.suffix.lower()

        if suffix == ".parquet":
            # Parquet is ideal for lazy loading
            import pyarrow.parquet as pq

            self.parquet_file = pq.ParquetFile(self.data_file)
            self._length = self.parquet_file.metadata.num_rows
            logger.info(f"Initialized lazy loading with Parquet ({self._length} rows)")
        elif suffix == ".csv":
            # For CSV, we need to count lines (slower)
            with open(self.data_file, "r", encoding="utf-8") as f:
                self._length = sum(1 for _ in f) - 1  # Subtract header
            logger.info(f"Initialized lazy loading with CSV ({self._length} rows)")
        else:
            # For pickle, fall back to eager loading
            logger.warning(
                f"Lazy loading not efficient for {suffix}, using eager loading"
            )
            self.lazy_loading = False
            self.data = self._load_data(self.data_file)

            # Validate columns
            if self.text_column not in self.data.columns:
                raise ValueError(f"Text column '{self.text_column}' not found in data")
            if self.label_column not in self.data.columns:
                raise ValueError(
                    f"Label column '{self.label_column}' not found in data"
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

    def _load_single_row(self, idx: int) -> Tuple[str, int]:
        """
        Load a single row from disk (for lazy loading).

        Args:
            idx (int): Row index.

        Returns:
            Tuple[str, int]: (text, label)
        """
        suffix = self.data_file.suffix.lower()

        if suffix == ".parquet":
            # Read single row from parquet (fast)
            import pyarrow.parquet as pq

            table = pq.read_table(
                self.data_file, columns=[self.text_column, self.label_column]
            )
            # For efficiency in actual training, use batches, but for __getitem__
            # we need specific row. Parquet is still fast here.
            row = table.to_pandas().iloc[idx]
            text = str(row[self.text_column])
            label = int(row[self.label_column])
            return text, label
        elif suffix == ".csv":
            # CSV lazy loading is inherently slow.
            # Optimization: Use a shared file handle or buffer if needed,
            # but for now, we use a slightly better pandas approach.
            # WARNING: This is still O(N) in worst case.
            df = pd.read_csv(self.data_file, skiprows=idx, nrows=1, header=None)
            if len(df) == 0:
                raise IndexError(f"Index {idx} out of range")
            # header=None means columns are 0, 1...
            # We need to find the correct column index
            # This is why Parquet is highly recommended.
            text = str(
                df.iloc[0][0]
            )  # Assuming text is first column for simplicity in this fallback
            label = int(df.iloc[0][1])
            return text, label
        else:
            raise ValueError(
                f"Lazy loading not supported for {suffix}. Use Parquet for best performance."
            )

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        if self.lazy_loading:
            return self._length
        else:
            return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns a single sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing:
                - input_ids: Token IDs (shape: [sequence_length])
                - attention_mask: Attention mask (shape: [sequence_length])
                - labels: Target label (scalar)

        Note:
            No padding is applied here. Use DataCollatorWithPadding for dynamic padding.
        """
        # Check cache first if enabled
        if self.cache_tokenization:
            cache_file = self.cache_dir / f"sample_{idx}.pt"
            if cache_file.exists():
                try:
                    return torch.load(cache_file)
                except Exception as e:
                    logger.warning(f"Error loading cache for index {idx}: {e}")

        if self.lazy_loading:
            # Load single row from disk
            text, label = self._load_single_row(idx)
        else:
            # Load from in-memory DataFrame
            row = self.data.iloc[idx]
            text = str(row[self.text_column])
            label = int(row[self.label_column])

        # Tokenize the text WITHOUT padding (dynamic padding will be done in DataCollator)
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )

        sample = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }

        # Save to cache if enabled
        if self.cache_tokenization:
            try:
                torch.save(sample, self.cache_dir / f"sample_{idx}.pt")
            except Exception:
                # Don't fail if cache save fails
                pass

        return sample


def create_dataloaders(
    config: Dict,
    tokenizer: PreTrainedTokenizer,
    train_path: Optional[str] = None,
    val_path: Optional[str] = None,
    test_path: Optional[str] = None,
    use_dynamic_padding: bool = True,
    lazy_loading: bool = False,
    cache_tokenization: bool = False,
) -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
    """
    Creates DataLoader instances for training, validation, and testing.

    Args:
        config (Dict): Configuration dictionary containing training parameters.
        tokenizer (PreTrainedTokenizer): Tokenizer to use for all datasets.
        train_path (Optional[str]): Path to training data. If None, uses config.
        val_path (Optional[str]): Path to validation data. If None, uses config.
        test_path (Optional[str]): Path to test data. If None, uses config.
        use_dynamic_padding (bool): Whether to use dynamic padding.
        lazy_loading (bool): Whether to use lazy loading.
        cache_tokenization (bool): Whether to cache tokenized inputs.

    Returns:
        Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
            Train, validation, and test DataLoaders.
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

    # Create data collator for dynamic padding
    collate_fn = DataCollatorWithPadding(tokenizer) if use_dynamic_padding else None

    # Optimize DataLoader for GPU training
    import os
    import platform

    is_windows = platform.system() == "Windows"
    pin_memory = torch.cuda.is_available()

    # On Windows, num_workers > 0 often causes issues with multiprocessing/pickling
    # On Linux, use number of CPU cores (but not all to avoid overhead)
    if is_windows:
        num_workers = 0
        persistent_workers = False
    else:
        num_workers = min(os.cpu_count() or 4, 4)  # Default to 4 on Linux
        persistent_workers = True
        logger.info(f"Linux detected: using num_workers={num_workers}")

    # Create datasets
    train_loader = None
    val_loader = None
    test_loader = None

    if train_path and Path(train_path).exists():
        train_dataset = VulnerabilityDataset(
            train_path,
            tokenizer,
            max_length=max_length,
            lazy_loading=lazy_loading,
            cache_tokenization=cache_tokenization,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=2 if num_workers > 0 else None,
        )
        logger.info(
            f"Created train DataLoader with {len(train_dataset)} samples "
            f"(dynamic_padding={use_dynamic_padding}, lazy_loading={lazy_loading}, cache={cache_tokenization})"
        )

    if val_path and Path(val_path).exists():
        val_dataset = VulnerabilityDataset(
            val_path,
            tokenizer,
            max_length=max_length,
            lazy_loading=lazy_loading,
            cache_tokenization=cache_tokenization,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=2 if num_workers > 0 else None,
        )
        logger.info(
            f"Created validation DataLoader with {len(val_dataset)} samples "
            f"(dynamic_padding={use_dynamic_padding}, lazy_loading={lazy_loading}, cache={cache_tokenization})"
        )

    if test_path and Path(test_path).exists():
        test_dataset = VulnerabilityDataset(
            test_path,
            tokenizer,
            max_length=max_length,
            lazy_loading=lazy_loading,
            cache_tokenization=cache_tokenization,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=2 if num_workers > 0 else None,
        )
        logger.info(
            f"Created test DataLoader with {len(test_dataset)} samples "
            f"(dynamic_padding={use_dynamic_padding}, lazy_loading={lazy_loading}, cache={cache_tokenization})"
        )

    return train_loader, val_loader, test_loader
