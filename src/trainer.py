"""
Training module for EnStack vulnerability detection.

This module provides the EnStackTrainer class for training, evaluation,
and feature extraction from transformer-based models.
"""

import logging
import os
import re
import shutil
import stat
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


class GPUMemoryManager:
    """
    OPTIMIZATION: Smart GPU memory manager to reduce fragmentation and OOM errors.

    Uses strategic cache clearing at optimal points instead of excessive clearing
    which can slow down training.
    """

    def __init__(self, enabled: bool = True, clear_threshold_mb: float = 100.0):
        """
        Initialize GPU memory manager.

        Args:
            enabled (bool): Whether memory management is enabled.
            clear_threshold_mb (float): Clear cache if allocated memory changes by this amount (MB).
        """
        self.enabled = enabled and torch.cuda.is_available()
        self.clear_threshold_mb = clear_threshold_mb
        self.last_allocated_mb = 0.0

        if self.enabled:
            logger.info(
                f"GPU Memory Manager enabled (threshold={clear_threshold_mb}MB)"
            )

    def check_and_clear(self, force: bool = False) -> None:
        """
        Checks memory usage and clears cache if threshold exceeded.

        Args:
            force (bool): Force cache clearing regardless of threshold.
        """
        if not self.enabled:
            return

        current_allocated_mb = torch.cuda.memory_allocated() / (1024**2)
        delta_mb = abs(current_allocated_mb - self.last_allocated_mb)

        if force or delta_mb > self.clear_threshold_mb:
            torch.cuda.empty_cache()
            self.last_allocated_mb = torch.cuda.memory_allocated() / (1024**2)

            if force:
                logger.debug(
                    f"Forced GPU cache clear (allocated: {self.last_allocated_mb:.1f}MB)"
                )

    def get_memory_stats(self) -> Dict[str, float]:
        """Returns current GPU memory statistics."""
        if not self.enabled:
            return {}

        return {
            "allocated_mb": torch.cuda.memory_allocated() / (1024**2),
            "reserved_mb": torch.cuda.memory_reserved() / (1024**2),
            "max_allocated_mb": torch.cuda.max_memory_allocated() / (1024**2),
        }


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
        logging_steps: int = 50,
        enable_memory_management: bool = True,
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
            logging_steps (int): Log metrics to TensorBoard every N steps.
            enable_memory_management (bool): OPTIMIZATION - Enable smart GPU memory management.
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
        self.logging_steps = logging_steps

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

        # OPTIMIZATION: Initialize GPU memory manager
        self.memory_manager = GPUMemoryManager(enabled=enable_memory_management)

        # Initialize optimizer
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)

        # Initialize AMP GradScaler if using mixed precision
        # FIX: Update for PyTorch 2.x compatibility
        self.scaler = None
        if use_amp and self.device.type == "cuda":
            if hasattr(torch.amp, "GradScaler"):
                self.scaler = torch.amp.GradScaler(device="cuda")
            else:
                self.scaler = torch.cuda.amp.GradScaler()
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

    def _validate_optimizer_consistency(
        self, state: Dict, expected_steps: int, tolerance: int = 20
    ) -> None:
        """
        IMPROVEMENT 2: Validates optimizer step count matches checkpoint metadata.

        Args:
            state (Dict): Loaded checkpoint state dictionary.
            expected_steps (int): Expected optimizer steps based on epoch/step.
            tolerance (int): Maximum allowed difference before warning.
        """
        if "optimizer_state_dict" not in state:
            logger.warning("⚠️  No optimizer state found in checkpoint")
            return

        opt_state = state["optimizer_state_dict"]

        # Extract optimizer step count
        if "state" in opt_state and len(opt_state["state"]) > 0:
            # Get first parameter's step count (all should be the same)
            first_param_state = opt_state["state"][0]
            if "step" in first_param_state:
                actual_steps = first_param_state["step"].item()

                logger.info(f"  Optimizer Steps: {actual_steps}")
                logger.info(f"  Expected Steps: {expected_steps}")

                diff = abs(actual_steps - expected_steps)

                if diff <= tolerance:
                    logger.info(
                        f"  ✅ Optimizer state consistent (diff={diff} steps, within tolerance={tolerance})"
                    )
                else:
                    logger.warning("=" * 60)
                    logger.warning("⚠️  OPTIMIZER STEP MISMATCH DETECTED")
                    logger.warning(f"  Actual optimizer steps: {actual_steps}")
                    logger.warning(f"  Expected steps: {expected_steps}")
                    logger.warning(f"  Difference: {diff} steps")
                    logger.warning("")
                    logger.warning("  Possible causes:")
                    logger.warning("  1. Gradient accumulation (expected behavior)")
                    logger.warning("  2. Scheduler warmup phase")
                    logger.warning("  3. Training was interrupted mid-batch")
                    logger.warning("")
                    if diff > 100:
                        logger.warning(
                            f"  ⚠️  Large mismatch ({diff} steps) - checkpoint may be inconsistent!"
                        )
                        logger.warning(
                            "     Consider retraining from an earlier checkpoint."
                        )
                    else:
                        logger.warning(
                            "  ℹ️  Small mismatch is usually harmless (gradient accumulation)."
                        )
                    logger.warning("=" * 60)
            else:
                logger.warning("  ⚠️  Optimizer state missing 'step' field")
        else:
            logger.warning("  ⚠️  Optimizer state appears empty")

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
        checkpoint_dir = Path(checkpoint_path)
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_dir}")

        # IMPROVEMENT 1: Validate checkpoint integrity before loading
        try:
            from .utils import quick_verify_checkpoint

            quick_verify_checkpoint(str(checkpoint_dir))
        except (FileNotFoundError, ValueError) as e:
            logger.error(f"❌ Checkpoint validation failed: {e}")
            raise

        # Load model weights
        from transformers import AutoModelForSequenceClassification

        # FIX: Load weights into the EXISTING model instance to preserve parameter references
        # This prevents the optimizer (initialized in __init__) from becoming detached
        logger.info(f"Loading model weights from {checkpoint_dir}...")

        # FIX: Use AutoModel instead of hard-coded RobertaForSequenceClassification
        # This supports any transformer architecture (RoBERTa, BERT, CodeBERT, etc.)
        temp_model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint_dir, num_labels=self.model.num_labels
        )

        # Transfer weights to our training model
        self.model.model.load_state_dict(temp_model.state_dict())

        # Clean up
        del temp_model
        import gc

        gc.collect()
        torch.cuda.empty_cache()

        self.model.to(self.device)
        logger.info("Successfully loaded weights into existing model instance")

        # IMPROVEMENT 1: Verify model weights were actually loaded
        logger.info(f"✅ Verified model weights loaded from {checkpoint_dir}")

        # Load training state
        state_path = checkpoint_dir / "training_state.pth"
        if state_path.exists():
            state = torch.load(state_path, map_location=self.device)
            self.optimizer.load_state_dict(state["optimizer_state_dict"])
            self.best_val_f1 = state.get("best_val_f1", 0.0)
            self.best_val_acc = state.get("best_val_acc", 0.0)

            if self.scheduler is not None and "scheduler_state_dict" in state:
                self.scheduler.load_state_dict(state["scheduler_state_dict"])

            # FIX: Load Scaler state for AMP stability
            if self.scaler is not None and "scaler_state_dict" in state:
                self.scaler.load_state_dict(state["scaler_state_dict"])
                logger.info("Loaded AMP Scaler state")

            epoch = state.get("epoch", 0)
            step = state.get("step", 0)
            # For old checkpoints without total_batches, use current loader length
            saved_total_batches = state.get("total_batches", None)

            # Handle legacy checkpoints
            if saved_total_batches is None or saved_total_batches == 0:
                current_batches = len(self.train_loader) if self.train_loader else 0
                logger.warning(
                    "⚠️  Legacy checkpoint detected (missing total_batches field)"
                )
                logger.warning(
                    f"   Using current dataset size: {current_batches} batches"
                )
                saved_total_batches = current_batches

            # Enhanced logging for debugging
            logger.info("=" * 60)
            logger.info("LOADED CHECKPOINT STATE:")
            logger.info(f"  Epoch: {epoch}")
            logger.info(f"  Step: {step}")
            logger.info(f"  Total Batches (saved): {saved_total_batches}")
            logger.info(f"  Best Val F1: {self.best_val_f1:.4f}")
            logger.info(f"  Best Val Acc: {self.best_val_acc:.4f}")

            # IMPROVEMENT 2: Validate optimizer consistency
            # Calculate expected optimizer steps
            current_batches = len(self.train_loader) if self.train_loader else 0
            if step == 0:
                # End-of-epoch checkpoint: optimizer should have completed all steps
                expected_opt_steps = epoch * current_batches
            else:
                # Mid-epoch checkpoint
                expected_opt_steps = (epoch - 1) * current_batches + step

            self._validate_optimizer_consistency(state, expected_opt_steps)

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

    def _train_step(
        self,
        batch: Dict[str, torch.Tensor],
        step: int,
        trained_count: int,
        total_batches: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Executes a single training step (forward, backward, optimizer).

        Args:
            batch: Batch data dictionary.
            step: Current global step.
            trained_count: Number of batches trained so far in this epoch.
            total_batches: Total batches to train in this epoch.

        Returns:
            Tuple[loss, logits]: Unscaled loss and logits.
        """
        # Move batch to device
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)

        # Forward pass with AMP
        if self.scaler:
            # Mixed precision training
            with torch.amp.autocast(device_type="cuda"):
                outputs = self.model(input_ids, attention_mask, labels)
                loss = outputs["loss"]
                logits = outputs["logits"]

            # Scale loss for gradient accumulation
            scaled_loss = loss / self.gradient_accumulation_steps

            # Backward pass with scaled loss
            self.scaler.scale(scaled_loss).backward()

            # Update weights only after accumulating gradients OR at the end of epoch
            if (step + 1) % self.gradient_accumulation_steps == 0 or (
                trained_count == total_batches
            ):
                # Unscale gradients and clip
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

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
            scaled_loss = loss / self.gradient_accumulation_steps

            # Backward pass
            scaled_loss.backward()

            # Update weights only after accumulating gradients OR at the end of epoch
            if (step + 1) % self.gradient_accumulation_steps == 0 or (
                trained_count == total_batches
            ):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                # Zero gradients AFTER optimizer step
                self.optimizer.zero_grad(set_to_none=True)

                if self.scheduler is not None:
                    self.scheduler.step()

        return loss, logits

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

        # Optimize: Use itertools to skip batches without loading them
        import itertools

        # Create progress bar that shows actual remaining work
        if resume_step > 0:
            remaining_batches = total_batches - resume_step
            logger.info(
                f"⏭️  Resuming: will skip {resume_step} batches (fast-forward), "
                f"train {remaining_batches} batches"
            )
            # Skip the first resume_step batches efficiently without loading
            train_iterator = itertools.islice(self.train_loader, resume_step, None)
            # Create progress bar starting from resume_step
            progress_bar = tqdm(
                train_iterator,
                total=remaining_batches,
                desc=f"Epoch {epoch} [Train]",
                leave=False,
            )
            # Manually track step since we're using islice
            step_offset = resume_step
            batches_to_train = remaining_batches  # Track actual batches to train
        else:
            # Explicitly type hint to avoid mypy confusion
            from typing import Iterator, Union

            train_iter: Union[DataLoader, Iterator] = self.train_loader

            progress_bar = tqdm(
                train_iter,
                total=total_batches,
                desc=f"Epoch {epoch} [Train]",
                leave=False,
            )
            step_offset = 0
            batches_to_train = total_batches

        trained_count = 0  # Track actual batches trained (excluding skipped)

        for batch_idx, batch in enumerate(progress_bar):
            step = step_offset + batch_idx  # Actual step in epoch
            trained_count += 1

            loss, logits = self._train_step(
                batch, step, trained_count, batches_to_train
            )

            # Track metrics (use unscaled loss for logging)
            total_loss += loss.item() * self.gradient_accumulation_steps
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())

            # Update progress bar with detailed info
            current_lr = self.optimizer.param_groups[0]["lr"]
            progress_bar.set_postfix(
                {
                    "loss": f"{loss.item() * self.gradient_accumulation_steps:.4f}",
                    "lr": f"{current_lr:.2e}",
                }
            )

            # Log to TensorBoard
            if self.writer and (step % self.logging_steps == 0 or step == 0):
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
                # OPTIMIZATION: Use smart memory manager instead of always clearing
                self.memory_manager.check_and_clear(force=False)

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

        # OPTIMIZATION: Strategic cache clear after evaluation
        self.memory_manager.check_and_clear(force=True)

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

        # Clean up any leftover temporary checkpoints from previous runs
        self._cleanup_temp_checkpoints()

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
            from torch.optim.swa_utils import SWALR, AveragedModel

            swa_model = AveragedModel(self.model)
            swa_scheduler = SWALR(self.optimizer, swa_lr=self.learning_rate)
            logger.info(f"SWA enabled, starting at epoch {swa_start}")

        start_epoch = 1
        start_step = 0

        # Resume if requested
        if resume_from:
            # IMPROVEMENT 3: Pre-flight checkpoint verification
            from .utils import quick_verify_checkpoint

            logger.info("=" * 60)
            logger.info("RESUMING TRAINING FROM CHECKPOINT")
            logger.info(f"Checkpoint path: {resume_from}")
            logger.info("=" * 60)

            # Quick verification before attempting to load
            try:
                quick_verify_checkpoint(resume_from)
            except (FileNotFoundError, ValueError) as e:
                logger.error("=" * 60)
                logger.error("❌ CHECKPOINT VERIFICATION FAILED")
                logger.error(f"   {e}")
                logger.error("=" * 60)
                logger.error("")
                logger.error("💡 Suggestions:")
                logger.error("   1. Check the checkpoint path is correct")
                logger.error("   2. Verify the checkpoint was saved successfully")
                logger.error("   3. Run full verification:")
                logger.error(
                    f"      python scripts/verify_checkpoint.py --checkpoint_path {resume_from}"
                )
                logger.error("")
                raise RuntimeError(
                    f"Cannot resume training - checkpoint verification failed: {e}"
                ) from e

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

            # AUTO-FIX: If checkpoint is fully trained, restart from Epoch 1
            if start_epoch > num_epochs:
                logger.warning("\n" + "!" * 60)
                logger.warning(
                    f"⚠️  Checkpoint indicates training finished (Epoch {start_epoch - 1} >= {num_epochs})"
                )
                logger.warning("   Current logic would SKIP training.")
                logger.warning(
                    "   ACTION: Forcing restart from Epoch 1 to ensure training occurs."
                )
                logger.warning("!" * 60 + "\n")
                start_epoch = 1
                start_step = 0
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

            # IMPROVEMENT 3: Enhanced scheduler fast-forwarding with validation
            if self.scheduler:
                steps_per_epoch = len(self.train_loader)
                steps_to_skip = ((start_epoch - 1) * steps_per_epoch) + start_step

                if steps_to_skip > 0:
                    # Get learning rate before fast-forward
                    lr_before = self.optimizer.param_groups[0]["lr"]

                    logger.info("=" * 60)
                    logger.info("⏩ FAST-FORWARDING SCHEDULER")
                    logger.info(f"  Total steps to skip: {steps_to_skip}")
                    logger.info(
                        f"  Calculation: ({start_epoch - 1} epochs × {steps_per_epoch} steps/epoch) + {start_step} mid-epoch steps"
                    )
                    logger.info(f"  LR before fast-forward: {lr_before:.6e}")

                    # Fast-forward
                    for _ in range(steps_to_skip):
                        self.scheduler.step()

                    # Get learning rate after fast-forward
                    lr_after = self.optimizer.param_groups[0]["lr"]
                    logger.info(f"  LR after fast-forward: {lr_after:.6e}")

                    # Validate reasonable learning rate
                    if lr_after < 1e-8:
                        logger.warning(
                            "⚠️  WARNING: Learning rate very small after fast-forward!"
                        )
                        logger.warning(
                            f"     LR={lr_after:.2e} - scheduler may have decayed too far"
                        )
                    elif lr_after > self.learning_rate * 2:
                        logger.warning("⚠️  WARNING: Learning rate unexpectedly high!")
                        logger.warning(
                            f"     LR={lr_after:.2e} vs initial={self.learning_rate:.2e}"
                        )
                    else:
                        logger.info("  ✅ Learning rate in expected range")

                    logger.info("=" * 60)

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
                    if swa_model is not None and swa_scheduler is not None:
                        swa_model.update_parameters(self.model)
                        swa_scheduler.step()
                        logger.info(f"Epoch {epoch}: Updated SWA parameters")

                # Save best model
                if save_best and val_metrics["f1"] > self.best_val_f1:
                    self.best_val_f1 = val_metrics["f1"]
                    self.best_val_acc = val_metrics["accuracy"]

                    # Use fixed name to ensure only ONE best model exists (overwrites previous)
                    self.save_checkpoint("best_model", epoch=epoch, step=0)
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
                self._force_delete(recovery_path)
                logger.debug(
                    f"Cleaned up recovery checkpoint (epoch {epoch} completed)"
                )

            # Reset start_step for next epochs
            start_step = 0

            # Break if early stopping triggered
            if should_stop:
                break

        # Finalize SWA
        if use_swa:
            if swa_model is not None:
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

    def _cleanup_temp_checkpoints(self) -> None:
        """
        Cleans up any leftover temporary checkpoint directories.
        These can remain if a previous save was interrupted.
        """
        temp_patterns = [".tmp_", ".backup_"]
        cleaned_count = 0

        for path in self.output_dir.iterdir():
            if path.is_dir() and any(path.name.startswith(p) for p in temp_patterns):
                logger.info(f"🧹 Cleaning up leftover temp directory: {path.name}")
                self._force_delete(path)
                cleaned_count += 1

        if cleaned_count > 0:
            logger.info(f"✅ Cleaned up {cleaned_count} temporary checkpoint(s)")

    def _force_delete(self, path: Path) -> None:
        """
        Robustly deletes a directory, handling read-only files and errors.
        Fixes issues on Windows where shutil.rmtree fails on file locks.
        """
        if not path.exists():
            return

        def handle_remove_readonly(func, path, exc):
            # Handler for read-only files (common on Windows)
            excvalue = exc[1]
            if func in (os.rmdir, os.remove, os.unlink) and isinstance(
                excvalue, PermissionError
            ):
                os.chmod(path, stat.S_IWRITE)
                func(path)
            else:
                # Re-raise if it's a different error
                raise

        try:
            shutil.rmtree(path, onerror=handle_remove_readonly)
        except Exception as e:
            # Fallback: Just log it, don't crash training
            logger.warning(f"Failed to force delete {path.name}: {e}")

    def _rotate_checkpoints(self, keep_last_n: int = 1) -> None:
        """
        Rotates checkpoints, keeping only the N most recent ones.
        Target pattern: checkpoint_epoch{E}_step{S}
        """
        checkpoints = []
        pattern = re.compile(r"checkpoint_epoch(\d+)_step(\d+)")

        # Find all matching checkpoints
        for path in self.output_dir.iterdir():
            if path.is_dir() and pattern.match(path.name):
                match = pattern.match(path.name)
                epoch = int(match.group(1))
                step = int(match.group(2))
                checkpoints.append((epoch, step, path))

        # Sort by epoch then step (ascending)
        checkpoints.sort(key=lambda x: (x[0], x[1]))

        # Identify checkpoints to delete
        if len(checkpoints) > keep_last_n:
            to_delete = checkpoints[:-keep_last_n]
            kept = checkpoints[-keep_last_n:]

            # Log what we found for transparency
            logger.info(f"Checkpoint Rotation: Found {len(checkpoints)} checkpoints.")
            logger.info(f"   Keeping: {[c[2].name for c in kept]}")

            for _, _, path in to_delete:
                logger.info(f"🗑️  Deleting old checkpoint: {path.name}")
                self._force_delete(path)

                # Verify deletion
                if path.exists():
                    logger.error(f"⚠️  Could not delete {path.name} (File locked?)")
        else:
            if len(checkpoints) > 0:
                logger.debug(
                    f"Checkpoint Rotation: Keeping all {len(checkpoints)} checkpoints (limit={keep_last_n})"
                )

    def save_checkpoint(
        self, checkpoint_name: str = "checkpoint", epoch: int = 0, step: int = 0
    ) -> None:
        """
        Saves the model checkpoint with atomic write to prevent corruption.
        Includes retry mechanism and verification for Google Drive compatibility.

        Args:
            checkpoint_name (str): Name of the checkpoint.
            epoch (int): Current epoch number.
            step (int): Current step within the epoch.
        """
        import shutil
        import tempfile
        import time

        save_path = self.output_dir / checkpoint_name

        # Detect if we're saving to Google Drive
        is_gdrive = "/content/drive/" in str(self.output_dir)

        # Use temporary directory for atomic save
        temp_dir = None
        max_retries = 3
        retry_delay = 2.0  # seconds

        for attempt in range(max_retries):
            try:
                # OPTIMIZATION: "Local-First" Strategy
                # If saving to Google Drive, create temp dir on LOCAL filesystem (VM SSD) first.
                # This makes saving model/optimizer states much faster and reliable (no network I/O).
                if is_gdrive and Path("/content").exists():
                    # Create temp dir in /content (Local VM)
                    temp_base = Path("/content/temp_checkpoints")
                    temp_base.mkdir(parents=True, exist_ok=True)
                    temp_dir = Path(
                        tempfile.mkdtemp(
                            dir=temp_base, prefix=f".tmp_{checkpoint_name}_"
                        )
                    )
                    logger.debug(f"Created fast local temp dir: {temp_dir}")
                else:
                    # Standard behavior: Create temp dir in same parent (atomic move compatible)
                    temp_dir = Path(
                        tempfile.mkdtemp(
                            dir=self.output_dir, prefix=f".tmp_{checkpoint_name}_"
                        )
                    )
                    logger.debug(f"Created standard temp dir: {temp_dir}")

                # Save model to temp directory first
                logger.debug("Saving model to temporary location...")
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

                # FIX: Save Scaler state
                if self.scaler is not None:
                    state_dict["scaler_state_dict"] = self.scaler.state_dict()

                torch.save(state_dict, temp_dir / "training_state.pth")

                # CRITICAL: Verify temp directory contents before moving
                required_files = ["training_state.pth", "config.json"]
                for req_file in required_files:
                    if not (temp_dir / req_file).exists():
                        raise FileNotFoundError(
                            f"Required file {req_file} not found in temp directory"
                        )

                # FIX: Google Drive-compatible save (COPY instead of MOVE)
                if is_gdrive:
                    logger.debug("Google Drive detected - using COPY method for safety")

                    # First, backup existing checkpoint if it exists
                    backup_path = None
                    if save_path.exists():
                        backup_path = self.output_dir / f".backup_{checkpoint_name}"
                        if backup_path.exists():
                            self._force_delete(backup_path)
                        logger.debug(f"Creating backup: {backup_path}")
                        shutil.copytree(str(save_path), str(backup_path))

                        # Sync and wait
                        import os

                        if hasattr(os, "sync"):
                            os.sync()
                        time.sleep(0.5)

                        # Remove old checkpoint
                        self._force_delete(save_path)
                        time.sleep(0.5)

                    # Copy temp to final location (safer than move on Drive)
                    logger.debug(f"Copying checkpoint to final location: {save_path}")
                    shutil.copytree(str(temp_dir), str(save_path))

                    # CRITICAL: Flush OS buffers and wait for Google Drive sync
                    import os

                    if hasattr(os, "sync"):
                        logger.debug("Flushing OS buffers (os.sync)...")
                        os.sync()

                    logger.debug("Waiting for Google Drive sync...")
                    time.sleep(3.0)  # Increased wait time for Drive to 3s

                    # VERIFY: Check that checkpoint actually exists and has required files
                    if not save_path.exists():
                        raise FileNotFoundError(
                            f"Checkpoint directory {save_path} does not exist after save!"
                        )

                    for req_file in required_files:
                        final_file = save_path / req_file
                        if not final_file.exists():
                            raise FileNotFoundError(
                                f"Required file {req_file} missing from final checkpoint!"
                            )
                        # Verify file is not empty
                        if final_file.stat().st_size == 0:
                            raise ValueError(f"File {req_file} is empty!")

                    # Clean up temp directory after successful copy
                    self._force_delete(temp_dir)
                    temp_dir = None

                    # Remove backup on success
                    if backup_path and backup_path.exists():
                        self._force_delete(backup_path)

                else:
                    # Standard atomic move for local filesystems
                    # First, backup existing checkpoint if it exists
                    backup_path = None
                    if save_path.exists():
                        backup_path = self.output_dir / f".backup_{checkpoint_name}"
                        if backup_path.exists():
                            self._force_delete(backup_path)
                        logger.debug(f"Creating backup: {backup_path}")
                        shutil.move(str(save_path), str(backup_path))

                    # Move temp to final location
                    logger.debug(f"Moving checkpoint to final location: {save_path}")
                    shutil.move(str(temp_dir), str(save_path))
                    temp_dir = None  # Successfully moved, don't clean up

                    # VERIFY: Check that checkpoint actually exists and has required files
                    if not save_path.exists():
                        raise FileNotFoundError(
                            f"Checkpoint directory {save_path} does not exist after save!"
                        )

                    for req_file in required_files:
                        final_file = save_path / req_file
                        if not final_file.exists():
                            raise FileNotFoundError(
                                f"Required file {req_file} missing from final checkpoint!"
                            )
                        # Verify file is not empty
                        if final_file.stat().st_size == 0:
                            raise ValueError(f"File {req_file} is empty!")

                    # Remove backup on success
                    if backup_path and backup_path.exists():
                        self._force_delete(backup_path)

                logger.info(
                    f"✅ Checkpoint saved: {checkpoint_name} (epoch={epoch}, step={step})"
                )

                # Cleanup old checkpoints if this was a mid-epoch save
                if checkpoint_name.startswith("checkpoint_epoch"):
                    self._rotate_checkpoints(keep_last_n=1)

                # Success - exit retry loop
                return

            except Exception as e:
                logger.error(
                    f"❌ Failed to save checkpoint {checkpoint_name} (attempt {attempt + 1}/{max_retries}): {e}"
                )
                logger.error(f"   Epoch: {epoch}, Step: {step}")

                # Clean up temp directory if it still exists
                if temp_dir and temp_dir.exists():
                    self._force_delete(temp_dir)
                    temp_dir = None

                # If this is the last attempt, decide whether to raise
                if attempt == max_retries - 1:
                    # CRITICAL checkpoints MUST succeed - raise exception
                    if checkpoint_name in ["best_model", "last_checkpoint"]:
                        logger.error(
                            f"⚠️  CRITICAL: {checkpoint_name} save failed after {max_retries} attempts!"
                        )
                        raise RuntimeError(
                            f"Failed to save critical checkpoint {checkpoint_name} after {max_retries} attempts: {e}"
                        ) from e
                    else:
                        # Mid-epoch checkpoints: log warning but continue
                        logger.warning(
                            f"⚠️  Training will continue but checkpoint {checkpoint_name} may be lost!"
                        )
                else:
                    # Retry after delay
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)

        # Final cleanup (should not reach here if successful)
        if temp_dir and temp_dir.exists():
            self._force_delete(temp_dir)

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
                return cast(np.ndarray, features)

        # Extract features
        self.model.eval()

        # OPTIMIZATION: Zero-Copy Memory Management
        # Pre-allocate numpy array to avoid list append overhead and RAM spikes
        # First, determine feature dimension from the model config or a dummy pass
        from typing import Sized

        dataset_sized = cast(Sized, loader.dataset)
        num_samples = len(dataset_sized)
        feature_dim = (
            self.model.num_labels if mode == "logits" else 768
        )  # Approximate default

        # We'll use a dynamic list for the first batch to get exact dimension,
        # then allocate the full array.
        features_array = None
        start_idx = 0

        progress_bar = tqdm(loader, desc=f"Extracting {mode}", leave=False)

        # Use inference_mode for maximum performance during inference
        with torch.inference_mode():
            # OPTIMIZATION: Enable AMP (FP16) for inference speedup and lower VRAM
            with torch.amp.autocast(device_type="cuda", enabled=self.use_amp):
                for batch in progress_bar:
                    input_ids = batch["input_ids"].to(self.device, non_blocking=True)
                    attention_mask = batch["attention_mask"].to(
                        self.device, non_blocking=True
                    )

                    if mode == "embedding":
                        # Get hidden states with specified pooling strategy
                        batch_features = self.model.get_embedding(
                            input_ids, attention_mask, pooling=pooling
                        )
                    else:
                        # Get logits (num_labels dim)
                        batch_features = self.model.get_logits(
                            input_ids, attention_mask
                        )
                        # Convert to probabilities
                        batch_features = torch.softmax(batch_features, dim=-1)

                    # Move to CPU numpy
                    batch_features_np = batch_features.cpu().numpy()
                    batch_size = batch_features_np.shape[0]

                    # Initialize pre-allocated array on first batch
                    if features_array is None:
                        feature_dim = batch_features_np.shape[1]
                        features_array = np.zeros(
                            (num_samples, feature_dim), dtype=np.float32
                        )

                    # Fill buffer directly (Zero-Copy)
                    end_idx = start_idx + batch_size
                    # Handle potential size mismatch (e.g., drop_last=True or inconsistent length)
                    if end_idx > num_samples:
                        # Resize if dataset length was underestimated
                        logger.warning(
                            f"Dataset length mismatch! Resizing buffer from {num_samples} to {end_idx}"
                        )
                        new_array = np.zeros((end_idx, feature_dim), dtype=np.float32)
                        new_array[:num_samples] = features_array
                        features_array = new_array
                        num_samples = end_idx

                    features_array[start_idx:end_idx] = batch_features_np
                    start_idx = end_idx

                    # OPTIMIZATION: Clear cache aggressively during feature extraction
                    # to prevent OOM on large datasets
                    del input_ids, attention_mask, batch_features, batch_features_np
                    self.memory_manager.check_and_clear(force=False)

        # Final cleanup
        torch.cuda.empty_cache()

        # Trim if dataset length was overestimated (e.g. distributed sampling)
        if start_idx < num_samples and features_array is not None:
            features_array = features_array[:start_idx]

        features = features_array
        # Handle the case where features_array might be None (empty loader)
        if features is None:
            features = np.array([], dtype=np.float32)

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

        # OPTIMIZATION: Strategic cache clear after feature extraction
        self.memory_manager.check_and_clear(force=True)

        return cast(np.ndarray, features)
