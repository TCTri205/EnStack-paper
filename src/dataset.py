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
from tqdm import tqdm
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
        smart_batching: bool = True,
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
            smart_batching (bool): If True, sorts dataset by sequence length to minimize padding.
                Only applies when lazy_loading=False. Default: True.

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
        self.smart_batching = smart_batching
        self.samples: List[Dict[str, torch.Tensor]] = []
        self._length: int = 0

        # Load data
        data_file = Path(data_path)
        if not data_file.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        self.data_file = data_file

        # Setup tokenization cache directory (Only for lazy loading / disk cache)
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

            # OPTIMIZATION: Pre-tokenize all data into RAM if not lazy loading
            # This avoids repeated tokenization and disk I/O during training
            self._pre_tokenize_data()

            # OPTIMIZATION: Smart Batching (Sort by length)
            # Sorting samples by length reduces padding significantly during batched inference
            # We sort descending to handle longest sequences first (often helps with OOM early detection)
            # Note: DataLoader with shuffle=True will undo this for training (which is desired),
            # but DataLoader with shuffle=False (Val/Test) will benefit immensely.
            if self.smart_batching:
                self.samples.sort(key=lambda x: x["input_ids"].shape[0], reverse=True)
                logger.info(
                    "Dataset sorted by sequence length (Smart Batching enabled)"
                )

        logger.info(
            f"Loaded {len(self)} samples from {data_path} "
            f"(max_length={max_length}, lazy_loading={lazy_loading}, cache={cache_tokenization})"
        )

    def _pre_tokenize_data(self) -> None:
        """
        Pre-tokenizes the entire dataset and stores it in memory.
        This drastically speeds up training by removing tokenizer overhead and disk I/O.
        """
        logger.info("Pre-tokenizing dataset into memory...")

        # Iterate over dataframe and tokenize
        for idx in tqdm(range(len(self.data)), desc="Tokenizing"):
            row = self.data.iloc[idx]
            text = str(row[self.text_column])
            label = int(row[self.label_column])

            # Tokenize
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
            self.samples.append(sample)

        logger.info(f"Successfully pre-tokenized {len(self.samples)} samples in RAM")

    def _initialize_lazy_loading(self) -> None:
        """
        Initialize lazy loading by reading only the dataset length.

        OPTIMIZATION: For CSV, builds an offset map for O(1) random access.
        """
        suffix = self.data_file.suffix.lower()

        if suffix == ".parquet":
            # Parquet is ideal for lazy loading
            import pyarrow.parquet as pq

            self.parquet_file = pq.ParquetFile(self.data_file)
            self._length = self.parquet_file.metadata.num_rows
            logger.info(f"Initialized lazy loading with Parquet ({self._length} rows)")
        elif suffix == ".csv":
            # OPTIMIZATION: Build offset map for O(1) random access
            logger.info("Building CSV offset map for fast random access...")
            self.csv_offsets = []
            self.csv_header = None

            with open(self.data_file, "rb") as f:
                # Read header
                header_line = f.readline()
                self.csv_header = header_line.decode("utf-8").strip()

                # Build offset map: store file position of each row
                while True:
                    offset = f.tell()
                    line = f.readline()
                    if not line:
                        break
                    self.csv_offsets.append(offset)

            self._length = len(self.csv_offsets)
            logger.info(
                f"Initialized lazy loading with CSV ({self._length} rows, "
                f"offset map built for O(1) access)"
            )
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

        OPTIMIZATION: Uses offset map for O(1) CSV access instead of O(N).

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
            # OPTIMIZATION: Use offset map for O(1) random access
            import csv

            if not hasattr(self, "csv_offsets") or idx >= len(self.csv_offsets):
                raise IndexError(f"Index {idx} out of range")

            with open(self.data_file, "r", encoding="utf-8") as f:
                # Seek directly to the row position (O(1) operation)
                f.seek(self.csv_offsets[idx])
                line = f.readline().strip()

                # Parse the CSV line
                reader = csv.reader([line])
                row = next(reader)

                # Parse header to find column indices
                if not hasattr(self, "_csv_column_indices"):
                    if self.csv_header is None:
                        raise ValueError("CSV header is missing")
                    header_reader = csv.reader([self.csv_header])
                    from typing import cast

                    header = cast(List[str], next(header_reader))
                    self._csv_column_indices = {col: i for i, col in enumerate(header)}

                # Extract text and label using column names
                text_idx = self._csv_column_indices.get(self.text_column, 0)
                label_idx = self._csv_column_indices.get(self.label_column, 1)

                text = str(row[text_idx])
                label = int(row[label_idx])

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
        # OPTIMIZATION: Return from RAM cache if available
        if not self.lazy_loading and idx < len(self.samples):
            return self.samples[idx]

        # Fallback for lazy loading (or if cache failed)
        # Check disk cache first if enabled
        if self.cache_tokenization:
            cache_file = self.cache_dir / f"sample_{idx}.pt"
            if cache_file.exists():
                try:
                    from typing import cast

                    return cast(Dict[str, torch.Tensor], torch.load(cache_file))
                except Exception as e:
                    logger.warning(f"Error loading cache for index {idx}: {e}")

        if self.lazy_loading:
            # Load single row from disk
            text, label = self._load_single_row(idx)
        else:
            # Load from in-memory DataFrame (should be covered by self.samples, but safety net)
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

        # Save to disk cache if enabled
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


def create_dataloaders_from_hf_dataset(
    config: Dict,
    tokenizer: PreTrainedTokenizer,
    dataset_name_or_path: Optional[str] = None,
    train_split: str = "train",
    val_split: str = "validation",
    test_split: str = "test",
    text_column: str = "func",
    label_column: str = "target",
    use_dynamic_padding: bool = True,
) -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
    """
    Creates DataLoader instances using HuggingFace datasets library (memory-mapped).

    OPTIMIZATION: Uses Apache Arrow memory mapping for zero-copy data access.
    This is faster and more memory-efficient than custom CSV/Parquet loading.

    Args:
        config (Dict): Configuration dictionary containing training parameters.
        tokenizer (PreTrainedTokenizer): Tokenizer to use for all datasets.
        dataset_name_or_path (Optional[str]): HF dataset name or local path.
        train_split (str): Name of training split.
        val_split (str): Name of validation split.
        test_split (str): Name of test split.
        text_column (str): Column name containing source code.
        label_column (str): Column name containing labels.
        use_dynamic_padding (bool): Whether to use dynamic padding.

    Returns:
        Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
            Train, validation, and test DataLoaders.
    """
    try:
        from datasets import load_dataset, load_from_disk
    except ImportError as e:
        raise ImportError(
            "HuggingFace datasets library not installed. "
            "Install with: pip install datasets"
        ) from e

    batch_size = config["training"]["batch_size"]
    max_length = config["training"]["max_length"]

    # Load dataset
    if dataset_name_or_path is None:
        raise ValueError("dataset_name_or_path must be provided")

    # Check if it's a local path or HF Hub dataset
    from pathlib import Path

    if Path(dataset_name_or_path).exists():
        logger.info(f"Loading dataset from disk: {dataset_name_or_path}")
        dataset = load_from_disk(dataset_name_or_path)
    else:
        logger.info(f"Loading dataset from HuggingFace Hub: {dataset_name_or_path}")
        dataset = load_dataset(dataset_name_or_path)

    # Tokenization function
    def tokenize_function(examples):
        encodings = tokenizer(
            examples[text_column],
            max_length=max_length,
            padding=False,  # Dynamic padding will be done by collator
            truncation=True,
        )
        encodings["labels"] = examples[label_column]
        # OPTIMIZATION: Add length for Smart Batching
        encodings["length"] = [len(x) for x in encodings["input_ids"]]
        return encodings

    # Apply tokenization with multiprocessing
    logger.info("Tokenizing dataset with multiprocessing...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=4,  # Use 4 CPU cores for tokenization
        remove_columns=dataset[train_split].column_names,
        desc="Tokenizing",
    )

    # OPTIMIZATION: Sort by length for Smart Batching (Desc)
    # This optimizes inference speed by grouping similar length sequences
    logger.info("Sorting dataset by length for Smart Batching...")
    for split in tokenized_dataset:
        try:
            tokenized_dataset[split] = tokenized_dataset[split].sort(
                "length", reverse=True
            )
        except Exception as e:
            logger.warning(f"Could not sort split {split}: {e}")

    # Set format to PyTorch tensors
    tokenized_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )

    # Create data collator
    collate_fn = DataCollatorWithPadding(tokenizer) if use_dynamic_padding else None

    # Optimize DataLoader settings
    import os
    import platform

    is_windows = platform.system() == "Windows"
    pin_memory = torch.cuda.is_available()

    if is_windows:
        num_workers = 0
        persistent_workers = False
    else:
        num_workers = min(os.cpu_count() or 4, 4)
        persistent_workers = True

    # Create DataLoaders
    train_loader = None
    val_loader = None
    test_loader = None

    if train_split in tokenized_dataset:
        train_loader = DataLoader(
            tokenized_dataset[train_split],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=2 if num_workers > 0 else None,
        )
        logger.info(
            f"Created train DataLoader with {len(tokenized_dataset[train_split])} samples"
        )

    if val_split in tokenized_dataset:
        val_loader = DataLoader(
            tokenized_dataset[val_split],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=2 if num_workers > 0 else None,
        )
        logger.info(
            f"Created validation DataLoader with {len(tokenized_dataset[val_split])} samples"
        )

    if test_split in tokenized_dataset:
        test_loader = DataLoader(
            tokenized_dataset[test_split],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=2 if num_workers > 0 else None,
        )
        logger.info(
            f"Created test DataLoader with {len(tokenized_dataset[test_split])} samples"
        )

    return train_loader, val_loader, test_loader
