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

from typing import Dict, List, Tuple

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
from src.visualization import (
    plot_confusion_matrix,
    plot_meta_feature_importance,
    plot_training_history,
)


def log_experiment_results(
    config: Dict, metrics: Dict, output_dir: str, name: str = "ensemble"
) -> None:
    """
    Logs experiment configuration and results to a CSV file.
    """
    import csv
    from datetime import datetime

    log_file = Path(output_dir) / "experiment_results.csv"
    file_exists = log_file.exists()

    fieldnames = [
        "timestamp",
        "model_name",
        "base_models",
        "meta_classifier",
        "accuracy",
        "f1",
        "precision",
        "recall",
        "auc",
        "epochs",
        "batch_size",
    ]

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": name,
        "base_models": ",".join(config["model"]["base_models"]),
        "meta_classifier": config["model"].get("meta_classifier", "svm"),
        "accuracy": metrics.get("accuracy", 0),
        "f1": metrics.get("f1", 0),
        "precision": metrics.get("precision", 0),
        "recall": metrics.get("recall", 0),
        "auc": metrics.get("auc", 0),
        "epochs": config["training"]["epochs"],
        "batch_size": config["training"]["batch_size"],
    }

    with open(log_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    logger = logging.getLogger("EnStack")
    logger.info(f"Experiment results logged to {log_file}")


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

    parser.add_argument(
        "--save-steps",
        type=int,
        default=500,
        help="Save checkpoint every X steps (default: 500)",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from last checkpoint if available",
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
    config: Dict,
    model_names: List[str],
    num_epochs: int,
    device,
    resume: bool = False,
    save_steps: int = 500,
) -> Tuple[Dict, Dict]:
    """
    Trains all base models.

    Args:
        config (Dict): Configuration dictionary.
        model_names (List[str]): List of model names to train.
        num_epochs (int): Number of training epochs.
        device: Device to use for training.
        resume (bool): Whether to resume from checkpoint.
        save_steps (int): Steps between checkpoints.

    Returns:
        Tuple[Dict, Dict]: (trainers dict, dataloaders dict)
            - trainers: Dictionary of trained trainers
            - dataloaders: Dictionary mapping model_name -> {train/val/test: DataLoader}
    """
    logger = logging.getLogger("EnStack")
    trainers = {}
    dataloaders = {}

    for model_name in model_names:
        logger.info("=" * 60)
        logger.info(f"Training {model_name.upper()}")
        logger.info("=" * 60)

        output_dir = f"{config['training']['output_dir']}/{model_name}"

        # Determine if we should resume
        resume_path = None
        if resume:
            last_checkpoint = Path(output_dir) / "last_checkpoint"
            if last_checkpoint.exists():
                logger.info(f"Found checkpoint at {last_checkpoint}, will resume.")
                resume_path = str(last_checkpoint)
            else:
                logger.info("No checkpoint found, starting fresh.")

        # Create model and tokenizer
        model, tokenizer = create_model(model_name, config, pretrained=True)

        # Create dataloaders (ONLY ONCE per model)
        train_loader, val_loader, test_loader = create_dataloaders(
            config,
            tokenizer,
            use_dynamic_padding=config["training"].get("use_dynamic_padding", True),
            lazy_loading=config["training"].get("lazy_loading", False),
            cache_tokenization=config["training"].get("cache_tokenization", True),
        )

        # Store dataloaders for later use
        dataloaders[model_name] = {
            "train": train_loader,
            "val": val_loader,
            "test": test_loader,
        }

        # Create trainer
        trainer = EnStackTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            learning_rate=config["training"]["learning_rate"],
            device=device,
            output_dir=output_dir,
            use_amp=config["training"].get("use_amp", True),
            gradient_accumulation_steps=config["training"].get(
                "gradient_accumulation_steps", 1
            ),
            early_stopping_patience=config["training"].get(
                "early_stopping_patience", 3
            ),
            early_stopping_metric=config["training"].get("early_stopping_metric", "f1"),
        )

        # Train
        if num_epochs > 0:
            history = trainer.train(
                num_epochs=num_epochs,
                save_best=True,
                resume_from=resume_path,
                save_steps=save_steps,
                scheduler_type=config["training"].get("scheduler", "cosine"),
                use_swa=config["training"].get("use_swa", False),
                swa_start=config["training"].get("swa_start", 5),
            )

            # Plot training history
            plot_training_history(
                history, save_path=f"{output_dir}/training_history.png"
            )
        elif resume_path:
            logger.info(f"Loading weights from {resume_path} (epochs=0)...")
            trainer.load_checkpoint(resume_path)
        else:
            logger.warning(f"epochs=0 and no resume_path for {model_name}. Skipping.")

        # Evaluate on test set
        if test_loader is not None:
            test_metrics = trainer.evaluate(test_loader, split_name="Test")
            logger.info(f"{model_name} Test Results: {test_metrics}")

        trainers[model_name] = trainer
        logger.info(f"{model_name} training completed\n")

    return trainers, dataloaders


def extract_all_features(
    config: Dict,
    trainers: Dict,
    dataloaders: Dict,
    mode: str = "logits",
    pooling: str = "mean",
    use_cache: bool = True,
) -> Dict:
    """
    Extracts features from all base models.

    Args:
        config (Dict): Configuration dictionary.
        trainers (Dict): Dictionary of trained trainers.
        dataloaders (Dict): Dictionary of dataloaders (train/val/test) for each model.
        mode (str): Type of features to extract ('logits' or 'embedding').
        pooling (str): Pooling strategy for embeddings.
        use_cache (bool): Whether to use feature caching.

    Returns:
        Dict: Dictionary containing feature arrays for each split.
    """
    logger = logging.getLogger("EnStack")
    logger.info("=" * 60)
    logger.info(f"EXTRACTING {mode.upper()} FOR STACKING")
    logger.info("=" * 60)

    train_features_list = []
    val_features_list = []
    test_features_list = []

    cache_dir = (
        Path(config["training"]["output_dir"]) / "feature_cache" if use_cache else None
    )

    for model_name, trainer in trainers.items():
        logger.info(f"Extracting {mode} from {model_name}...")

        # Get pre-created dataloaders for this model
        model_loaders = dataloaders.get(model_name, {})
        train_loader = model_loaders.get("train")
        val_loader = model_loaders.get("val")
        test_loader = model_loaders.get("test")

        # Extract features with caching
        if train_loader:
            cache_path = (
                str(cache_dir / f"{model_name}_train_{mode}.npy") if cache_dir else None
            )
            train_features = trainer.extract_features(
                train_loader, mode=mode, pooling=pooling, cache_path=cache_path
            )
            train_features_list.append(train_features)

        if val_loader:
            cache_path = (
                str(cache_dir / f"{model_name}_val_{mode}.npy") if cache_dir else None
            )
            val_features = trainer.extract_features(
                val_loader, mode=mode, pooling=pooling, cache_path=cache_path
            )
            val_features_list.append(val_features)

        if test_loader:
            cache_path = (
                str(cache_dir / f"{model_name}_test_{mode}.npy") if cache_dir else None
            )
            test_features = trainer.extract_features(
                test_loader, mode=mode, pooling=pooling, cache_path=cache_path
            )
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
    output_dir = Path(config["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(
        log_file=args.log_file or str(output_dir / "train.log"), level=logging.INFO
    )
    logger.info("EnStack Training Pipeline Started")
    logger.info(f"Configuration: {args.config}")

    # Check for pyarrow (needed for parquet lazy loading)
    if config["training"].get("lazy_loading", False):
        try:
            import pyarrow
        except ImportError:
            logger.warning(
                "pyarrow not installed. Lazy loading with Parquet will fail. Install with: pip install pyarrow"
            )

    # Set random seed
    set_seed(config["training"]["seed"])

    # Get device
    device = get_device()

    # Get model names and epochs
    model_names = config["model"]["base_models"]
    num_epochs = config["training"]["epochs"]

    # Step 1: Train base models
    if not args.skip_training:
        trainers, dataloaders = train_base_models(
            config,
            model_names,
            num_epochs,
            device,
            resume=args.resume,
            save_steps=args.save_steps,
        )
    else:
        logger.info("Skipping base model training (using existing checkpoints)")
        # Load existing models (implementation depends on your checkpoint structure)
        raise NotImplementedError(
            "Loading from checkpoints not yet implemented. Remove --skip-training flag."
        )

    # Step 2: Extract features
    if not args.skip_stacking:
        # Use configuration for extraction mode and pooling
        mode = config["training"].get("stacking_mode", "logits")
        pooling = config["training"].get("pooling_mode", "mean")

        # Use cached features if available
        features_dict = extract_all_features(
            config, trainers, dataloaders, mode=mode, pooling=pooling, use_cache=True
        )

        # Load labels
        root_dir = Path(config["data"]["root_dir"])
        train_labels = load_labels_from_file(root_dir / config["data"]["train_file"])
        val_labels = load_labels_from_file(root_dir / config["data"]["val_file"])
        test_labels = load_labels_from_file(root_dir / config["data"]["test_file"])

        # Prepare meta-features with PCA and Scaling
        use_pca = config["training"].get("use_pca", True)
        pca_components = config["training"].get("pca_components", None)
        use_scaling = config["training"].get("use_scaling", True)

        train_meta_features, _, pca_model, scaler = prepare_meta_features(
            features_dict["train"],
            train_labels,
            use_pca=use_pca,
            pca_components=pca_components,
            use_scaling=use_scaling,
        )
        val_meta_features, _, _, _ = prepare_meta_features(
            features_dict["val"],
            val_labels,
            pca_model=pca_model,
            scaler=scaler,
            use_pca=use_pca,
            use_scaling=use_scaling,
        )
        test_meta_features, _, _, _ = prepare_meta_features(
            features_dict["test"],
            test_labels,
            pca_model=pca_model,
            scaler=scaler,
            use_pca=use_pca,
            use_scaling=use_scaling,
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

        # Step 5: Advanced Visualization & Logging
        logger.info("=" * 60)
        logger.info("FINAL VISUALIZATION & LOGGING")
        logger.info("=" * 60)

        # Plot confusion matrix
        plot_confusion_matrix(
            test_labels,
            meta_classifier.predict(test_meta_features),
            save_path=f"{config['training']['output_dir']}/confusion_matrix.png",
        )

        # Plot feature importance
        feature_names = []
        for model_name in model_names:
            # Add features for each class if it's probability mode
            num_classes = config["model"].get("num_labels", 5)
            for c in range(num_classes):
                feature_names.append(f"{model_name}_prob_{c}")

        plot_meta_feature_importance(
            meta_classifier,
            feature_names,
            save_path=f"{config['training']['output_dir']}/feature_importance.png",
        )

        # Log results to CSV
        log_experiment_results(config, test_metrics, config["training"]["output_dir"])

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
