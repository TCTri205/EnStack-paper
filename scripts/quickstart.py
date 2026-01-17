"""
Quick start example for EnStack.

This script demonstrates a minimal example of using EnStack for vulnerability detection.
"""

import logging

from src.dataset import create_dataloaders
from src.models import create_model
from src.trainer import EnStackTrainer
from src.utils import get_device, load_config, set_seed, setup_logging


def main():
    """
    Quick start example.

    This example shows how to:
    1. Load configuration
    2. Create a model and tokenizer
    3. Create data loaders
    4. Train the model
    5. Evaluate the model
    """
    # Setup logging
    logger = setup_logging(level=logging.INFO)
    logger.info("EnStack Quick Start Example")

    # Load configuration
    config_path = "configs/config.yaml"
    config = load_config(config_path)

    # Set random seed for reproducibility
    set_seed(config["training"]["seed"])

    # Get device (CUDA if available, else CPU)
    device = get_device()

    # Select a base model (e.g., CodeBERT)
    model_name = "codebert"
    logger.info(f"Training {model_name}")

    # Create model and tokenizer
    model, tokenizer = create_model(model_name, config, pretrained=True)
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Create data loaders
    # NOTE: Make sure your data files exist at the paths specified in config.yaml
    train_loader, val_loader, test_loader = create_dataloaders(config, tokenizer)

    if train_loader is None:
        logger.error(
            "Training data not found. Please ensure data files exist at the paths "
            "specified in configs/config.yaml"
        )
        return

    logger.info(f"Training samples: {len(train_loader.dataset)}")
    if val_loader:
        logger.info(f"Validation samples: {len(val_loader.dataset)}")
    if test_loader:
        logger.info(f"Test samples: {len(test_loader.dataset)}")

    # Create trainer
    trainer = EnStackTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        learning_rate=config["training"]["learning_rate"],
        device=device,
        output_dir=f"{config['training']['output_dir']}/{model_name}",
    )

    # Train the model
    num_epochs = config["training"]["epochs"]
    logger.info(f"Starting training for {num_epochs} epochs...")

    history = trainer.train(num_epochs=num_epochs, save_best=True)

    # Print training history
    logger.info("\nTraining History:")
    for epoch in range(len(history["train_loss"])):
        logger.info(
            f"Epoch {epoch + 1}: "
            f"Train Loss={history['train_loss'][epoch]:.4f}, "
            f"Train F1={history['train_f1'][epoch]:.4f}"
        )
        if history["val_loss"]:
            logger.info(
                f"         Val Loss={history['val_loss'][epoch]:.4f}, "
                f"Val F1={history['val_f1'][epoch]:.4f}"
            )

    # Evaluate on test set
    if test_loader is not None:
        logger.info("\nEvaluating on test set...")
        test_metrics = trainer.evaluate(test_loader, split_name="Test")

        logger.info("\nTest Results:")
        for metric, value in test_metrics.items():
            logger.info(f"  {metric.capitalize()}: {value:.4f}")

    logger.info("\nTraining completed! Model saved to:")
    logger.info(f"  {trainer.output_dir}")

    # Extract features (for stacking)
    logger.info("\nExtracting features for stacking...")
    if train_loader:
        train_features = trainer.extract_features(train_loader)
        logger.info(f"Extracted train features: {train_features.shape}")

    logger.info("\nQuick start example completed successfully!")


if __name__ == "__main__":
    main()
