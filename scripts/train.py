"""
Main training script for EnStack.

This script provides a command-line interface to train the complete EnStack pipeline,
including base models and meta-classifier.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from typing import Dict, List

import numpy as np

from src.dataset import create_dataloaders
from src.models import create_model
from src.stacking import (
    evaluate_meta_classifier,
    prepare_meta_features,
    save_meta_classifier,
    train_meta_classifier,
)
from src.trainer import EnStackTrainer
from src.utils import get_device, load_config, set_seed, setup_logging


def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train EnStack vulnerability detection model"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Base models to train (overrides config)",
    )

    parser.add_argument(
        "--epochs", type=int, default=None, help="Number of epochs (overrides config)"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (overrides config)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (overrides config)",
    )

    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip base model training (use existing checkpoints)",
    )

    parser.add_argument(
        "--skip-stacking",
        action="store_true",
        help="Skip meta-classifier training",
    )

    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to log file",
    )

    return parser.parse_args()


def load_labels_from_file(data_path: str) -> np.ndarray:
    """
    Loads labels from a data file.

    Args:
        data_path (str): Path to the data file.

    Returns:
        np.ndarray: Labels array.
    """
    import pickle

    import pandas as pd

    data_file = Path(data_path)

    if data_file.suffix == ".pkl":
        with open(data_file, "rb") as f:
            data = pickle.load(f)
    elif data_file.suffix == ".csv":
        data = pd.read_csv(data_file)
    else:
        raise ValueError(f"Unsupported file format: {data_file.suffix}")

    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    return data["target"].values


def train_base_models(
    config: Dict, model_names: List[str], num_epochs: int, device
) -> Dict:
    """
    Trains all base models.

    Args:
        config (Dict): Configuration dictionary.
        model_names (List[str]): List of model names to train.
        num_epochs (int): Number of training epochs.
        device: Device to use for training.

    Returns:
        Dict: Dictionary of trained trainers.
    """
    logger = logging.getLogger("EnStack")
    trainers = {}

    for model_name in model_names:
        logger.info("=" * 60)
        logger.info(f"Training {model_name.upper()}")
        logger.info("=" * 60)

        # Create model and tokenizer
        model, tokenizer = create_model(model_name, config, pretrained=True)

        # Create dataloaders
        train_loader, val_loader, test_loader = create_dataloaders(config, tokenizer)

        # Create trainer
        output_dir = f"{config['training']['output_dir']}/{model_name}"
        trainer = EnStackTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            learning_rate=config["training"]["learning_rate"],
            device=device,
            output_dir=output_dir,
        )

        # Train
        history = trainer.train(num_epochs=num_epochs, save_best=True)

        # Evaluate on test set
        if test_loader is not None:
            test_metrics = trainer.evaluate(test_loader, split_name="Test")
            logger.info(f"{model_name} Test Results: {test_metrics}")

        trainers[model_name] = trainer
        logger.info(f"{model_name} training completed\n")

    return trainers


def extract_all_features(config: Dict, trainers: Dict) -> Dict:
    """
    Extracts features from all base models.

    Args:
        config (Dict): Configuration dictionary.
        trainers (Dict): Dictionary of trained trainers.

    Returns:
        Dict: Dictionary containing feature arrays for each split.
    """
    logger = logging.getLogger("EnStack")
    logger.info("=" * 60)
    logger.info("EXTRACTING FEATURES FOR STACKING")
    logger.info("=" * 60)

    train_features_list = []
    val_features_list = []
    test_features_list = []

    for model_name, trainer in trainers.items():
        logger.info(f"Extracting features from {model_name}...")

        # Get tokenizer
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            config["model"]["model_map"][model_name]
        )

        # Create dataloaders
        train_loader, val_loader, test_loader = create_dataloaders(config, tokenizer)

        # Extract features
        if train_loader:
            train_features = trainer.extract_features(train_loader)
            train_features_list.append(train_features)

        if val_loader:
            val_features = trainer.extract_features(val_loader)
            val_features_list.append(val_features)

        if test_loader:
            test_features = trainer.extract_features(test_loader)
            test_features_list.append(test_features)

    return {
        "train": train_features_list,
        "val": val_features_list,
        "test": test_features_list,
    }


def main():
    """Main training pipeline."""
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override config with command-line arguments
    if args.models:
        config["model"]["base_models"] = args.models
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.output_dir:
        config["training"]["output_dir"] = args.output_dir

    # Setup logging
    logger = setup_logging(log_file=args.log_file, level=logging.INFO)
    logger.info("EnStack Training Pipeline Started")
    logger.info(f"Configuration: {args.config}")

    # Set random seed
    set_seed(config["training"]["seed"])

    # Get device
    device = get_device()

    # Get model names and epochs
    model_names = config["model"]["base_models"]
    num_epochs = config["training"]["epochs"]

    # Step 1: Train base models
    if not args.skip_training:
        trainers = train_base_models(config, model_names, num_epochs, device)
    else:
        logger.info("Skipping base model training (using existing checkpoints)")
        # Load existing models (implementation depends on your checkpoint structure)
        raise NotImplementedError(
            "Loading from checkpoints not yet implemented. "
            "Remove --skip-training flag."
        )

    # Step 2: Extract features
    if not args.skip_stacking:
        features_dict = extract_all_features(config, trainers)

        # Load labels
        root_dir = Path(config["data"]["root_dir"])
        train_labels = load_labels_from_file(root_dir / config["data"]["train_file"])
        val_labels = load_labels_from_file(root_dir / config["data"]["val_file"])
        test_labels = load_labels_from_file(root_dir / config["data"]["test_file"])

        # Prepare meta-features
        train_meta_features, _ = prepare_meta_features(
            features_dict["train"], train_labels
        )
        val_meta_features, _ = prepare_meta_features(features_dict["val"], val_labels)
        test_meta_features, _ = prepare_meta_features(
            features_dict["test"], test_labels
        )

        # Step 3: Train meta-classifier
        logger.info("=" * 60)
        logger.info("TRAINING META-CLASSIFIER")
        logger.info("=" * 60)

        meta_classifier_type = config["model"].get("meta_classifier", "svm")

        # Get specific parameters for the chosen classifier
        meta_params = (
            config["model"]
            .get("meta_classifier_params", {})
            .get(meta_classifier_type, {})
        )
        logger.info(f"Using meta-classifier params: {meta_params}")

        meta_classifier = train_meta_classifier(
            train_meta_features,
            train_labels,
            classifier_type=meta_classifier_type,
            random_state=config["training"]["seed"],
            **meta_params,
        )

        # Save meta-classifier
        meta_save_path = f"{config['training']['output_dir']}/meta_classifier.pkl"
        save_meta_classifier(meta_classifier, meta_save_path)

        # Step 4: Evaluate
        logger.info("=" * 60)
        logger.info("ENSEMBLE EVALUATION")
        logger.info("=" * 60)

        logger.info("Validation Set Results:")
        val_metrics = evaluate_meta_classifier(
            meta_classifier, val_meta_features, val_labels
        )

        logger.info("\nTest Set Results:")
        test_metrics = evaluate_meta_classifier(
            meta_classifier, test_meta_features, test_labels
        )

        # Print summary
        print("\n" + "=" * 60)
        print("FINAL RESULTS SUMMARY")
        print("=" * 60)
        print("\nValidation Metrics:")
        for key, value in val_metrics.items():
            print(f"  {key.capitalize()}: {value:.4f}")
        print("\nTest Metrics:")
        for key, value in test_metrics.items():
            print(f"  {key.capitalize()}: {value:.4f}")
        print("\n" + "=" * 60)

    logger.info("Pipeline completed successfully!")


if __name__ == "__main__":
    main()
