"""
Utility functions for EnStack project.

This module provides configuration management, logging setup, and other helper functions.
"""

import logging
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Loads configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        Dict[str, Any]: Configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the YAML file is malformed.
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        if not isinstance(config, dict):
            raise ValueError(f"Config file {config_path} is not a valid dictionary")
        logging.info(f"Configuration loaded from {config_path}")
        return config
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML configuration: {e}")
        raise


def setup_logging(
    log_file: Optional[str] = None, level: int = logging.INFO
) -> logging.Logger:
    """
    Sets up logging configuration for console and optional file output.

    Args:
        log_file (Optional[str]): Path to log file. If None, logs only to console.
        level (int): Logging level (default: logging.INFO).

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger("EnStack")
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicate logs
    if logger.handlers:
        logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(console_format)
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}")

    return logger


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """
    Sets random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): Random seed value (default: 42).
        deterministic (bool): If True, ensures fully deterministic behavior.
            WARNING: Setting to True can reduce performance by 20-30%.
            Only use for debugging or strict reproducibility requirements.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if deterministic:
            # Fully deterministic (slower)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            logging.info(f"Random seed set to {seed} (DETERMINISTIC mode - slower)")
        else:
            # Faster but may have minor non-determinism
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
            logging.info(f"Random seed set to {seed} (optimized mode)")
    else:
        logging.info(f"Random seed set to {seed}")


def ensure_dir(directory: str) -> None:
    """
    Ensures that a directory exists, creating it if necessary.

    Args:
        directory (str): Path to the directory.
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def get_device() -> torch.device:
    """
    Returns the appropriate device (CUDA if available, else CPU).

    Returns:
        torch.device: Device to use for computations.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logging.info("CUDA not available, using CPU")
    return device


def count_parameters(model: torch.nn.Module) -> int:
    """
    Counts the total number of trainable parameters in a model.

    Args:
        model (torch.nn.Module): PyTorch model.

    Returns:
        int: Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
