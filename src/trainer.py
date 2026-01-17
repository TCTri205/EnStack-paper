"""
Training module for EnStack vulnerability detection.

This module provides the EnStackTrainer class for training, evaluation,
and feature extraction from transformer-based models.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, cast

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
        use_amp: bool = True,
        gradient_accumulation_steps: int = 1,
        early_stopping_patience: int = 0,
        early_stopping_metric: str = "f1",
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
            use_amp (bool): Whether to use Automatic Mixed Precision (FP16).
            gradient_accumulation_steps (int): Number of steps to accumulate gradients.
            early_stopping_patience (int): Number of epochs to wait before stopping.
                If 0, early stopping is disabled. Default: 0 (disabled).
            early_stopping_metric (str): Metric to monitor for early stopping ('f1' or 'loss').
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.learning_rate = learning_rate
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_amp = use_amp
        self.gradient_accumulation_steps = gradient_accumulation_steps

        # Early stopping parameters
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_metric = early_stopping_metric
        self.early_stopping_counter = 0
        self.best_metric_value = 0.0 if early_stopping_metric == "f1" else float("inf")

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model.to(self.device)
        logger.info(f"Model moved to {self.device}")

        # Initialize optimizer
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)

        # Initialize AMP GradScaler if using mixed precision
        self.scaler = (
            torch.cuda.amp.GradScaler()
            if use_amp and self.device.type == "cuda"
            else None
        )
        if self.scaler:
            logger.info("Automatic Mixed Precision (AMP) enabled")

        # Scheduler will be set during training
        self.scheduler: Optional[LambdaLR] = None

        self.best_val_f1 = 0.0
        self.best_val_acc = 0.0

        # Initialize TensorBoard writer
        self.writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter

            log_dir = self.output_dir / "logs"
            self.writer = SummaryWriter(log_dir=str(log_dir))
            logger.info(f"TensorBoard logging enabled: {log_dir}")
        except ImportError:
            logger.warning(
                "TensorBoard not installed, logging disabled. Install with 'pip install tensorboard'"
            )

    def load_checkpoint(self, checkpoint_path: str) -> Tuple[int, int]:
        """
        Loads the model and training state from a checkpoint.

        Args:
            checkpoint_path (str): Path to the checkpoint directory.

        Returns:
            Tuple[int, int]: A tuple containing (epoch, step).
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
            saved_total_batches = state.get("total_batches", 0)

            # Enhanced logging for debugging
            logger.info("=" * 60)
            logger.info("LOADED CHECKPOINT STATE:")
            logger.info(f"  Epoch: {epoch}")
            logger.info(f"  Step: {step}")
            logger.info(f"  Total Batches (saved): {saved_total_batches}")
            logger.info(f"  Best Val F1: {self.best_val_f1:.4f}")
            logger.info(f"  Best Val Acc: {self.best_val_acc:.4f}")

            # Determine completion status
            if step == 0:
                logger.info(f"  Status: ✅ Epoch {epoch} COMPLETED")
            elif saved_total_batches > 0 and step >= saved_total_batches:
                logger.info(
                    f"  Status: ✅ Epoch {epoch} COMPLETED (step >= total_batches)"
                )
            else:
                if saved_total_batches > 0:
                    progress = (step / saved_total_batches) * 100
                    logger.info(
                        f"  Status: ⏸️  Epoch {epoch} INCOMPLETE ({progress:.1f}% done)"
                    )
                else:
                    logger.info(f"  Status: ⏸️  Epoch {epoch} INCOMPLETE (step={step})")
            logger.info("=" * 60)

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

        # Create progress bar that shows actual remaining work
        if resume_step > 0:
            remaining_batches = total_batches - resume_step
            logger.info(
                f"⏭️  Resuming: will skip {resume_step} batches, "
                f"train {remaining_batches} batches"
            )

        progress_bar = tqdm(
            enumerate(self.train_loader),
            total=total_batches,
            desc=f"Epoch {epoch} [Train]",
            leave=False,
            initial=resume_step,  # Start counter at resume_step
        )

        trained_count = 0  # Track actual batches trained (excluding skipped)

        for step, batch in progress_bar:
            # Skip steps if resuming (fast - just continue)
            if step < resume_step:
                continue

            trained_count += 1

            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            # Forward pass with AMP
            if self.scaler:
                # Mixed precision training
                with torch.cuda.amp.autocast():
                    outputs = self.model(input_ids, attention_mask, labels)
                    loss = outputs["loss"]
                    logits = outputs["logits"]

                # Scale loss for gradient accumulation
                loss = loss / self.gradient_accumulation_steps

                # Backward pass with scaled loss
                self.scaler.scale(loss).backward()

                # Update weights only after accumulating gradients OR at the end of epoch
                if (step + 1) % self.gradient_accumulation_steps == 0 or (
                    step + 1
                ) == len(self.train_loader):
                    # Unscale gradients and clip
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0
                    )

                    # Step optimizer and scaler
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                    # Zero gradients AFTER optimizer step
                    self.optimizer.zero_grad(set_to_none=True)

                    if self.scheduler is not None:
                        self.scheduler.step()
            else:
                # Standard training (FP32)
                outputs = self.model(input_ids, attention_mask, labels)
                loss = outputs["loss"]
                logits = outputs["logits"]

                # Scale loss for gradient accumulation
                loss = loss / self.gradient_accumulation_steps

                # Backward pass
                loss.backward()

                # Update weights only after accumulating gradients OR at the end of epoch
                if (step + 1) % self.gradient_accumulation_steps == 0 or (
                    step + 1
                ) == len(self.train_loader):
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0
                    )
                    self.optimizer.step()

                    # Zero gradients AFTER optimizer step
                    self.optimizer.zero_grad(set_to_none=True)

                    if self.scheduler is not None:
                        self.scheduler.step()

            # Track metrics (use unscaled loss for logging)
            total_loss += loss.item() * self.gradient_accumulation_steps
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Update progress bar
            progress_bar.set_postfix(
                {"loss": loss.item() * self.gradient_accumulation_steps}
            )

            # Log to TensorBoard
            if self.writer:
                global_step = (epoch - 1) * len(self.train_loader) + step
                self.writer.add_scalar(
                    "Train/Loss",
                    loss.item() * self.gradient_accumulation_steps,
                    global_step,
                )
                self.writer.add_scalar(
                    "Train/LR", self.optimizer.param_groups[0]["lr"], global_step
                )

            # Step-based Checkpointing (save to separate file to avoid overwriting end-of-epoch)
            # Only save if save_steps > 0 (can be disabled for faster training)
            if save_steps > 0 and (step + 1) % save_steps == 0:
                logger.info(
                    f"💾 Saving mid-epoch checkpoint (Epoch {epoch}, Step {step + 1})"
                )
                # Use different checkpoint name to avoid overwriting
                self.save_checkpoint(
                    f"checkpoint_epoch{epoch}_step{step + 1}",
                    epoch=epoch,
                    step=(step + 1),
                )
                # Also update a "recovery" checkpoint that can be cleaned up later
                self.save_checkpoint(
                    "recovery_checkpoint", epoch=epoch, step=(step + 1)
                )

                # Clear VRAM cache after checkpoint to prevent OOM
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Calculate metrics
        if len(all_labels) == 0:
            return {"loss": 0.0, "accuracy": 0.0, "f1": 0.0}

        # Use actual trained count for average loss
        avg_loss = total_loss / trained_count if trained_count > 0 else 0.0

        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

        metrics = {
            "loss": avg_loss,
            "accuracy": accuracy,
            "f1": f1,
        }

        logger.info(
            f"Epoch {epoch} [Train] - "
            f"Trained {trained_count}/{total_batches} batches, "
            f"Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}, F1: {f1:.4f}"
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

        # Use inference_mode instead of no_grad for better performance
        with torch.inference_mode():
            for batch in progress_bar:
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(
                    self.device, non_blocking=True
                )
                labels = batch["labels"].to(self.device, non_blocking=True)

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

        # Clear VRAM after evaluation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return metrics

    def _get_cosine_schedule_with_warmup(
        self,
        num_training_steps: int,
        num_warmup_steps: int = 0,
        num_cycles: float = 0.5,
    ) -> LambdaLR:
        """
        Creates a cosine learning rate scheduler with warmup.

        Args:
            num_training_steps (int): Total number of training steps.
            num_warmup_steps (int): Number of warmup steps.
            num_cycles (float): Number of waves in the cosine schedule (default: 0.5).

        Returns:
            LambdaLR: Learning rate scheduler.
        """
        import math

        def lr_lambda(current_step: int) -> float:
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(
                max(1, num_training_steps - num_warmup_steps)
            )
            return max(
                0.0,
                0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)),
            )

        return LambdaLR(self.optimizer, lr_lambda)

    def train(
        self,
        num_epochs: int,
        save_best: bool = True,
        resume_from: Optional[str] = None,
        save_steps: int = 500,
        scheduler_type: str = "linear",
        use_swa: bool = False,
        swa_start: int = 5,
    ) -> Dict[str, List[float]]:
        """
        Trains the model for multiple epochs.

        Args:
            num_epochs (int): Number of epochs to train.
            save_best (bool): Whether to save the best model based on validation F1.
            resume_from (Optional[str]): Path to checkpoint directory to resume from.
            save_steps (int): Save checkpoint every X steps.
            scheduler_type (str): Type of scheduler ('linear' or 'cosine').
            use_swa (bool): Whether to use Stochastic Weight Averaging.
            swa_start (int): Epoch to start SWA.

        Returns:
            Dict[str, List[float]]: Dictionary containing training history.
        """
        if self.train_loader is None:
            raise ValueError("Training loader is not provided")

        # Setup scheduler
        num_training_steps = len(self.train_loader) * num_epochs
        num_warmup_steps = num_training_steps // 10

        if scheduler_type == "cosine":
            self.scheduler = self._get_cosine_schedule_with_warmup(
                num_training_steps, num_warmup_steps
            )
            logger.info(f"Using Cosine Annealing scheduler (warmup={num_warmup_steps})")
        else:
            self.scheduler = self._get_linear_schedule_with_warmup(
                num_training_steps, num_warmup_steps
            )
            logger.info(f"Using Linear scheduler (warmup={num_warmup_steps})")

        # Setup SWA
        swa_model = None
        swa_scheduler = None
        if use_swa:
            from torch.optim.swa_utils import AveragedModel, SWALR

            swa_model = AveragedModel(self.model)
            swa_scheduler = SWALR(self.optimizer, swa_lr=self.learning_rate)
            logger.info(f"SWA enabled, starting at epoch {swa_start}")

        start_epoch = 1
        start_step = 0

        # Resume if requested
        if resume_from:
            logger.info("=" * 60)
            logger.info("RESUMING TRAINING FROM CHECKPOINT")
            logger.info(f"Checkpoint path: {resume_from}")
            logger.info("=" * 60)

            loaded_epoch, loaded_step = self.load_checkpoint(resume_from)

            # Determine where to start
            steps_per_epoch = len(self.train_loader)
            logger.info(f"\nCurrent dataset: {steps_per_epoch} batches/epoch")

            # Check if the checkpoint was saved at the end of an epoch
            # Step=0 typically means end of epoch (when we explicitly save with step=0)
            # OR step >= total batches means we completed the epoch
            if loaded_step == 0:
                # Checkpoint saved at end of epoch, start next epoch
                if loaded_epoch > 0:
                    start_epoch = loaded_epoch + 1
                    start_step = 0
                    logger.info(
                        f"\n✅ Epoch {loaded_epoch} was COMPLETED\n"
                        f"➡️  Will resume from START of epoch {start_epoch}"
                    )
            else:
                # Checkpoint saved mid-epoch
                # Check if we actually completed the epoch by comparing with current dataset size
                if loaded_step >= steps_per_epoch:
                    # The checkpoint step is >= current dataset batches
                    # This means the epoch was likely completed with a different/larger dataset
                    start_epoch = loaded_epoch + 1
                    start_step = 0
                    logger.info(
                        f"\n✅ Checkpoint step ({loaded_step}) >= current batches ({steps_per_epoch})\n"
                        f"   Treating epoch {loaded_epoch} as COMPLETED\n"
                        f"➡️  Will resume from START of epoch {start_epoch}"
                    )
                else:
                    # Mid-epoch, continue from where we left off
                    start_epoch = loaded_epoch
                    start_step = loaded_step
                    remaining = steps_per_epoch - loaded_step
                    progress = (loaded_step / steps_per_epoch) * 100
                    logger.info(
                        f"\n⏸️  Epoch {loaded_epoch} is INCOMPLETE\n"
                        f"   Progress: {loaded_step}/{steps_per_epoch} batches ({progress:.1f}%)\n"
                        f"   Remaining: {remaining} batches\n"
                        f"➡️  Will resume WITHIN epoch {start_epoch} from step {start_step}"
                    )

            # Adjust scheduler to skip steps
            if self.scheduler:
                steps_per_epoch = len(self.train_loader)
                steps_to_skip = ((start_epoch - 1) * steps_per_epoch) + start_step

                if steps_to_skip > 0:
                    logger.info(
                        f"Fast-forwarding scheduler by {steps_to_skip} steps..."
                    )
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

        logger.info("=" * 60)
        logger.info(
            f"STARTING TRAINING: {num_epochs} epochs (from epoch {start_epoch})"
        )
        logger.info("=" * 60)

        for epoch in range(start_epoch, num_epochs + 1):
            # If this is the first epoch of the loop, use start_step. Otherwise 0.
            current_resume_step = start_step if epoch == start_epoch else 0

            logger.info("\n" + "=" * 60)
            logger.info(f"EPOCH {epoch}/{num_epochs}")
            if current_resume_step > 0:
                logger.info(f"  Resuming from step {current_resume_step}")
            logger.info("=" * 60)

            # Train
            train_metrics = self.train_epoch(
                epoch, resume_step=current_resume_step, save_steps=save_steps
            )
            history["train_loss"].append(train_metrics["loss"])
            history["train_acc"].append(train_metrics["accuracy"])
            history["train_f1"].append(train_metrics["f1"])

            if self.writer:
                self.writer.add_scalar("Epoch/Train_Loss", train_metrics["loss"], epoch)
                self.writer.add_scalar(
                    "Epoch/Train_Acc", train_metrics["accuracy"], epoch
                )
                self.writer.add_scalar("Epoch/Train_F1", train_metrics["f1"], epoch)

            # Validate
            should_stop = False
            if self.val_loader is not None:
                val_metrics = self.evaluate(self.val_loader, split_name="Val")
                history["val_loss"].append(val_metrics["loss"])
                history["val_acc"].append(val_metrics["accuracy"])
                history["val_f1"].append(val_metrics["f1"])

                if self.writer:
                    self.writer.add_scalar("Epoch/Val_Loss", val_metrics["loss"], epoch)
                    self.writer.add_scalar(
                        "Epoch/Val_Acc", val_metrics["accuracy"], epoch
                    )
                    self.writer.add_scalar("Epoch/Val_F1", val_metrics["f1"], epoch)

                # Update SWA model
                if use_swa and epoch >= swa_start:
                    swa_model.update_parameters(self.model)
                    swa_scheduler.step()
                    logger.info(f"Epoch {epoch}: Updated SWA parameters")

                # Save best model
                if save_best and val_metrics["f1"] > self.best_val_f1:
                    self.best_val_f1 = val_metrics["f1"]
                    self.best_val_acc = val_metrics["accuracy"]
                    self.save_checkpoint(
                        f"best_model_epoch_{epoch}", epoch=epoch, step=0
                    )
                    logger.info(f"New best model saved (F1: {self.best_val_f1:.4f})")

                # Early stopping check
                if self.early_stopping_patience > 0:
                    current_metric = val_metrics[self.early_stopping_metric]

                    if self.early_stopping_metric == "f1":
                        # For F1, higher is better
                        if current_metric > self.best_metric_value:
                            self.best_metric_value = current_metric
                            self.early_stopping_counter = 0
                        else:
                            self.early_stopping_counter += 1
                    else:  # loss
                        # For loss, lower is better
                        if current_metric < self.best_metric_value:
                            self.best_metric_value = current_metric
                            self.early_stopping_counter = 0
                        else:
                            self.early_stopping_counter += 1

                    if self.early_stopping_counter >= self.early_stopping_patience:
                        logger.info(
                            f"Early stopping triggered after {epoch} epochs "
                            f"(no improvement for {self.early_stopping_patience} epochs)"
                        )
                        should_stop = True

            # ALWAYS save last checkpoint for resuming (reset step to 0 as epoch finished)
            logger.info(f"📥 Saving end-of-epoch checkpoint (Epoch {epoch} completed)")
            self.save_checkpoint("last_checkpoint", epoch=epoch, step=0)

            # Clean up mid-epoch recovery checkpoint as epoch completed successfully
            recovery_path = self.output_dir / "recovery_checkpoint"
            if recovery_path.exists():
                import shutil

                try:
                    shutil.rmtree(recovery_path)
                    logger.debug(
                        f"Cleaned up recovery checkpoint (epoch {epoch} completed)"
                    )
                except Exception as e:
                    logger.warning(f"Failed to clean up recovery checkpoint: {e}")

            # Reset start_step for next epochs
            start_step = 0

            # Break if early stopping triggered
            if should_stop:
                break

        # Finalize SWA
        if use_swa:
            logger.info("Finalizing SWA: Updating BN and copying weights to model")
            from torch.optim.swa_utils import update_bn

            update_bn(self.train_loader, swa_model, device=self.device)
            # Copy SWA weights back to main model
            self.model.load_state_dict(swa_model.module.state_dict())
            self.save_checkpoint("swa_model", epoch=num_epochs, step=0)

        if self.writer:
            self.writer.close()

        logger.info("Training completed")
        return history

    def save_checkpoint(
        self, checkpoint_name: str = "checkpoint", epoch: int = 0, step: int = 0
    ) -> None:
        """
        Saves the model checkpoint with atomic write to prevent corruption.

        Args:
            checkpoint_name (str): Name of the checkpoint.
            epoch (int): Current epoch number.
            step (int): Current step within the epoch.
        """
        import tempfile
        import shutil

        save_path = self.output_dir / checkpoint_name
        save_path.mkdir(parents=True, exist_ok=True)

        # Use temporary directory for atomic save
        temp_dir = None
        try:
            # Create temp directory in same parent to ensure same filesystem
            temp_dir = Path(
                tempfile.mkdtemp(dir=self.output_dir, prefix=f".tmp_{checkpoint_name}_")
            )

            # Save model to temp directory first
            logger.debug(f"Saving model to temporary location: {temp_dir}")
            self.model.save_pretrained(str(temp_dir))

            # Save optimizer and training state
            state_dict = {
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_val_f1": self.best_val_f1,
                "best_val_acc": self.best_val_acc,
                "epoch": epoch,
                "step": step,
                "total_batches": len(self.train_loader) if self.train_loader else 0,
            }

            if self.scheduler is not None:
                state_dict["scheduler_state_dict"] = self.scheduler.state_dict()

            torch.save(state_dict, temp_dir / "training_state.pth")

            # Atomic move: only if everything succeeded
            # First, backup existing checkpoint if it exists
            backup_path = None
            if save_path.exists():
                backup_path = self.output_dir / f".backup_{checkpoint_name}"
                if backup_path.exists():
                    shutil.rmtree(backup_path)
                logger.debug(f"Creating backup: {backup_path}")
                shutil.move(str(save_path), str(backup_path))

            # Move temp to final location
            logger.debug(f"Moving checkpoint to final location: {save_path}")
            shutil.move(str(temp_dir), str(save_path))
            temp_dir = None  # Successfully moved, don't clean up

            # Remove backup on success
            if backup_path and backup_path.exists():
                shutil.rmtree(backup_path)

            logger.info(
                f"✅ Checkpoint saved: {checkpoint_name} (epoch={epoch}, step={step})"
            )

        except Exception as e:
            logger.error(f"❌ Failed to save checkpoint {checkpoint_name}: {e}")
            logger.error(f"   Epoch: {epoch}, Step: {step}")
            # Don't raise - allow training to continue even if checkpoint save fails
            # But log prominently so user knows
            logger.warning("⚠️  Training will continue but checkpoint may be lost!")
        finally:
            # Clean up temp directory if it still exists (save failed)
            if temp_dir and temp_dir.exists():
                try:
                    shutil.rmtree(temp_dir)
                except:
                    pass

    def extract_features(
        self,
        loader: DataLoader,
        mode: str = "logits",
        pooling: str = "mean",
        cache_path: Optional[str] = None,
        force_recompute: bool = False,
    ) -> np.ndarray:
        """
        Extracts features (embeddings or logits) from the model for stacking.

        Args:
            loader (DataLoader): DataLoader containing data.
            mode (str): Type of features to extract ('logits' or 'embedding').
                'logits' returns probability distributions (standard for stacking).
                'embedding' returns hidden states (high-dimensional).
            pooling (str): Pooling strategy for embeddings ('cls' or 'mean').
                Only used when mode='embedding'. 'mean' is recommended for code.
            cache_path (Optional[str]): Path to cache file. If provided and exists,
                features will be loaded from cache instead of recomputed.
            force_recompute (bool): If True, ignore cache and recompute features.

        Returns:
            np.ndarray: Extracted features (shape: [num_samples, feature_dim]).
        """
        # Check cache first
        if cache_path and not force_recompute:
            cache_file = Path(cache_path)
            if cache_file.exists():
                logger.info(f"Loading cached features from {cache_path}")
                features = np.load(cache_path)
                logger.info(f"Loaded cached features with shape: {features.shape}")
                return features

        # Extract features
        self.model.eval()
        all_features = []

        progress_bar = tqdm(loader, desc=f"Extracting {mode}", leave=False)

        # Use inference_mode for maximum performance during inference
        with torch.inference_mode():
            for batch in progress_bar:
                input_ids = batch["input_ids"].to(self.device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(
                    self.device, non_blocking=True
                )

                if mode == "embedding":
                    # Get hidden states with specified pooling strategy
                    features = self.model.get_embedding(
                        input_ids, attention_mask, pooling=pooling
                    )
                else:
                    # Get logits (num_labels dim)
                    features = self.model.get_logits(input_ids, attention_mask)
                    # Convert to probabilities
                    features = torch.softmax(features, dim=-1)

                all_features.append(features.cpu().numpy())

        features = np.concatenate(all_features, axis=0)
        logger.info(
            f"Extracted {mode} (pooling={pooling if mode == 'embedding' else 'N/A'}) "
            f"with shape: {features.shape}"
        )

        # Save to cache if path provided
        if cache_path:
            cache_file = Path(cache_path)
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, features)
            logger.info(f"Cached features saved to {cache_path}")

        # Clear VRAM after feature extraction
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return cast(np.ndarray, features)
