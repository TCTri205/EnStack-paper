"""
Training module for EnStack vulnerability detection.

This module provides the EnStackTrainer class for training, evaluation,
and feature extraction from transformer-based models.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, cast

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from .models import EnStackModel

logger = logging.getLogger("EnStack")


class EnStackTrainer:
    """
    Trainer class for EnStack vulnerability detection models.

    Handles training loops, evaluation, checkpointing, and feature extraction.
    """

    def __init__(
        self,
        model: EnStackModel,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        learning_rate: float = 2e-5,
        device: Optional[torch.device] = None,
        output_dir: str = "./checkpoints",
    ) -> None:
        """
        Initialize the EnStackTrainer.

        Args:
            model (EnStackModel): Model to train.
            train_loader (Optional[DataLoader]): Training data loader.
            val_loader (Optional[DataLoader]): Validation data loader.
            test_loader (Optional[DataLoader]): Test data loader.
            learning_rate (float): Learning rate for optimizer.
            device (Optional[torch.device]): Device to use for training.
            output_dir (str): Directory to save checkpoints.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.learning_rate = learning_rate
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model.to(self.device)
        logger.info(f"Model moved to {self.device}")

        # Initialize optimizer
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)

        # Scheduler will be set during training
        self.scheduler: Optional[LambdaLR] = None

        # Track best validation metrics
        self.best_val_f1 = 0.0
        self.best_val_acc = 0.0

    def load_checkpoint(self, checkpoint_path: str) -> tuple[int, int]:
        """
        Loads the model and training state from a checkpoint.

        Args:
            checkpoint_path (str): Path to the checkpoint directory.

        Returns:
            tuple[int, int]: A tuple containing (epoch, step).
                             - epoch: The epoch to resume from (or the last completed one).
                             - step: The step within that epoch to resume from.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

        # Load model weights
        from transformers import RobertaForSequenceClassification

        # We need to reload the inner model using HF's from_pretrained
        # This ensures we get the weights correctly
        self.model.model = RobertaForSequenceClassification.from_pretrained(
            checkpoint_path, num_labels=self.model.num_labels
        )
        self.model.to(self.device)
        logger.info(f"Loaded model weights from {checkpoint_path}")

        # Load training state
        state_path = checkpoint_path / "training_state.pth"
        if state_path.exists():
            state = torch.load(state_path, map_location=self.device)
            self.optimizer.load_state_dict(state["optimizer_state_dict"])
            self.best_val_f1 = state.get("best_val_f1", 0.0)
            self.best_val_acc = state.get("best_val_acc", 0.0)

            if self.scheduler is not None and "scheduler_state_dict" in state:
                self.scheduler.load_state_dict(state["scheduler_state_dict"])

            epoch = state.get("epoch", 0)
            step = state.get("step", 0)
            logger.info(f"Loaded training state (Epoch {epoch}, Step {step})")
            return epoch, step

        return 0, 0

    def _get_linear_schedule_with_warmup(
        self, num_training_steps: int, num_warmup_steps: int = 0
    ) -> LambdaLR:
        """
        Creates a linear learning rate scheduler with warmup.

        Args:
            num_training_steps (int): Total number of training steps.
            num_warmup_steps (int): Number of warmup steps.

        Returns:
            LambdaLR: Learning rate scheduler.
        """

        def lr_lambda(current_step: int) -> float:
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0,
                float(num_training_steps - current_step)
                / float(max(1, num_training_steps - num_warmup_steps)),
            )

        return LambdaLR(self.optimizer, lr_lambda)

    def train_epoch(
        self, epoch: int, resume_step: int = 0, save_steps: int = 500
    ) -> Dict[str, float]:
        """
        Trains the model for one epoch.

        Args:
            epoch (int): Current epoch number.
            resume_step (int): Step to resume from within the epoch.
            save_steps (int): Save checkpoint every X steps.

        Returns:
            Dict[str, float]: Dictionary containing training metrics.
        """
        if self.train_loader is None:
            raise ValueError("Training loader is not provided")

        self.model.train()
        total_loss = 0.0
        all_preds: List[int] = []
        all_labels: List[int] = []

        # Adjust total batches for progress bar if resuming
        total_batches = len(self.train_loader)

        progress_bar = tqdm(
            enumerate(self.train_loader),
            total=total_batches,
            desc=f"Epoch {epoch} [Train]",
            leave=False,
            initial=resume_step,
        )

        for step, batch in progress_bar:
            # Skip steps if resuming
            if step < resume_step:
                continue

            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(input_ids, attention_mask, labels)

            loss = outputs["loss"]
            logits = outputs["logits"]

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            # Track metrics
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Update progress bar
            progress_bar.set_postfix({"loss": loss.item()})

            # Step-based Checkpointing
            if (step + 1) % save_steps == 0:
                self.save_checkpoint("last_checkpoint", epoch=epoch, step=(step + 1))
                # We can also save a backup like 'checkpoint-epoch-X-step-Y' if desired,
                # but 'last_checkpoint' is enough for crash recovery.

        # Calculate metrics (accounting for skipped steps if any, though average is per-batch)
        # Note: If we resumed, len(all_labels) < len(dataset). This gives metrics for the *trained part* of the epoch.
        if len(all_labels) == 0:
            return {"loss": 0.0, "accuracy": 0.0, "f1": 0.0}

        avg_loss = (
            total_loss / len(all_labels) if len(all_labels) > 0 else 0.0
        )  # Approximation using batches
        # Better: total_loss / (total_batches - resume_step)
        steps_trained = total_batches - resume_step
        avg_loss = total_loss / steps_trained if steps_trained > 0 else 0.0

        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

        metrics = {
            "loss": avg_loss,
            "accuracy": accuracy,
            "f1": f1,
        }

        logger.info(
            f"Epoch {epoch} [Train] - Loss: {avg_loss:.4f}, "
            f"Acc: {accuracy:.4f}, F1: {f1:.4f}"
        )

        return metrics

    def evaluate(self, loader: DataLoader, split_name: str = "Val") -> Dict[str, float]:
        """
        Evaluates the model on a given dataset.

        Args:
            loader (DataLoader): DataLoader containing evaluation data.
            split_name (str): Name of the split (e.g., 'Val', 'Test').

        Returns:
            Dict[str, float]: Dictionary containing evaluation metrics.
        """
        self.model.eval()
        total_loss = 0.0
        all_preds: List[int] = []
        all_labels: List[int] = []

        progress_bar = tqdm(loader, desc=f"{split_name} Evaluation", leave=False)

        with torch.no_grad():
            for batch in progress_bar:
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Forward pass
                outputs = self.model(input_ids, attention_mask, labels)

                loss = outputs["loss"]
                logits = outputs["logits"]

                # Track metrics
                total_loss += loss.item()
                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        avg_loss = total_loss / len(loader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
        precision = precision_score(
            all_labels, all_preds, average="weighted", zero_division=0
        )
        recall = recall_score(
            all_labels, all_preds, average="weighted", zero_division=0
        )

        metrics = {
            "loss": avg_loss,
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall,
        }

        logger.info(
            f"{split_name} - Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}, "
            f"F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}"
        )

        return metrics

    def train(
        self,
        num_epochs: int,
        save_best: bool = True,
        resume_from: Optional[str] = None,
        save_steps: int = 500,
    ) -> Dict[str, List[float]]:
        """
        Trains the model for multiple epochs.

        Args:
            num_epochs (int): Number of epochs to train.
            save_best (bool): Whether to save the best model based on validation F1.
            resume_from (Optional[str]): Path to checkpoint directory to resume from.
            save_steps (int): Save checkpoint every X steps.

        Returns:
            Dict[str, List[float]]: Dictionary containing training history.
        """
        if self.train_loader is None:
            raise ValueError("Training loader is not provided")

        # Setup scheduler
        num_training_steps = len(self.train_loader) * num_epochs
        num_warmup_steps = num_training_steps // 10
        self.scheduler = self._get_linear_schedule_with_warmup(
            num_training_steps, num_warmup_steps
        )

        start_epoch = 1
        start_step = 0

        # Resume if requested
        if resume_from:
            logger.info(f"Resuming training from {resume_from}...")
            loaded_epoch, loaded_step = self.load_checkpoint(resume_from)

            # Determine where to start
            if loaded_step == 0:
                # Previous checkpoint was at end of epoch (old format or finished epoch)
                if loaded_epoch > 0:
                    start_epoch = loaded_epoch + 1
                    start_step = 0
                    logger.info(f"Resuming from start of epoch {start_epoch}")
            else:
                # Previous checkpoint was mid-epoch
                start_epoch = loaded_epoch
                start_step = loaded_step
                logger.info(f"Resuming from epoch {start_epoch}, step {start_step}")

            # Adjust scheduler to skip steps
            # Total steps already taken = (completed_epochs * steps_per_epoch) + steps_in_current_epoch
            # completed_epochs = start_epoch - 1
            if self.scheduler:
                steps_per_epoch = len(self.train_loader)
                steps_to_skip = ((start_epoch - 1) * steps_per_epoch) + start_step

                if steps_to_skip > 0:
                    logger.info(
                        f"Fast-forwarding scheduler by {steps_to_skip} steps..."
                    )
                    # Note: We can just set _step_count if accessing private member is safe,
                    # but calling step() is safer for compatibility.
                    # Since this might be large, tqdm it if very large?
                    # Usually fast enough unless millions.
                    for _ in range(steps_to_skip):
                        self.scheduler.step()

        history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_acc": [],
            "train_f1": [],
            "val_loss": [],
            "val_acc": [],
            "val_f1": [],
        }

        logger.info(f"Starting training for {num_epochs} epochs (from {start_epoch})")

        for epoch in range(start_epoch, num_epochs + 1):
            # If this is the first epoch of the loop, use start_step. Otherwise 0.
            current_resume_step = start_step if epoch == start_epoch else 0

            # Train
            train_metrics = self.train_epoch(
                epoch, resume_step=current_resume_step, save_steps=save_steps
            )
            history["train_loss"].append(train_metrics["loss"])
            history["train_acc"].append(train_metrics["accuracy"])
            history["train_f1"].append(train_metrics["f1"])

            # Validate
            if self.val_loader is not None:
                val_metrics = self.evaluate(self.val_loader, split_name="Val")
                history["val_loss"].append(val_metrics["loss"])
                history["val_acc"].append(val_metrics["accuracy"])
                history["val_f1"].append(val_metrics["f1"])

                # Save best model
                if save_best and val_metrics["f1"] > self.best_val_f1:
                    self.best_val_f1 = val_metrics["f1"]
                    self.best_val_acc = val_metrics["accuracy"]
                    self.save_checkpoint(
                        f"best_model_epoch_{epoch}", epoch=epoch, step=0
                    )  # Best model is always at end of epoch
                    logger.info(f"New best model saved (F1: {self.best_val_f1:.4f})")

            # ALWAYS save last checkpoint for resuming (reset step to 0 as epoch finished)
            self.save_checkpoint("last_checkpoint", epoch=epoch, step=0)

            # Reset start_step for next epochs
            start_step = 0

        logger.info("Training completed")
        return history

    def save_checkpoint(
        self, checkpoint_name: str = "checkpoint", epoch: int = 0, step: int = 0
    ) -> None:
        """
        Saves the model checkpoint.

        Args:
            checkpoint_name (str): Name of the checkpoint.
            epoch (int): Current epoch number.
            step (int): Current step within the epoch.
        """
        save_path = self.output_dir / checkpoint_name
        save_path.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(str(save_path))

        # Save optimizer state
        state_dict = {
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_f1": self.best_val_f1,
            "best_val_acc": self.best_val_acc,
            "epoch": epoch,
            "step": step,
        }

        if self.scheduler is not None:
            state_dict["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(state_dict, save_path / "training_state.pth")

        logger.info(f"Checkpoint saved to {save_path}")

    def extract_features(self, loader: DataLoader, mode: str = "logits") -> np.ndarray:
        """
        Extracts features (embeddings or logits) from the model for stacking.

        Args:
            loader (DataLoader): DataLoader containing data.
            mode (str): Type of features to extract ('logits' or 'embedding').
                'logits' returns probability distributions (standard for stacking).
                'embedding' returns hidden states (high-dimensional).

        Returns:
            np.ndarray: Extracted features (shape: [num_samples, feature_dim]).
        """
        self.model.eval()
        all_features = []

        progress_bar = tqdm(loader, desc=f"Extracting {mode}", leave=False)

        with torch.no_grad():
            for batch in progress_bar:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                if mode == "embedding":
                    # Get hidden states (768 dim)
                    features = self.model.get_embedding(input_ids, attention_mask)
                else:
                    # Get logits (num_labels dim)
                    features = self.model.get_logits(input_ids, attention_mask)
                    # Convert to probabilities
                    features = torch.softmax(features, dim=-1)

                all_features.append(features.cpu().numpy())

        features = np.concatenate(all_features, axis=0)
        logger.info(f"Extracted {mode} with shape: {features.shape}")

        return cast(np.ndarray, features)
