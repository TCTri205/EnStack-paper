"""
IMPROVEMENT 3: Pre-flight checkpoint verification script.

Comprehensive validation tool that checks checkpoint integrity and consistency
BEFORE attempting to resume training. This prevents training failures and
detects corrupted checkpoints early.

Usage:
    python scripts/verify_checkpoint.py --checkpoint_path <path> [--strict]

Features:
    - File integrity checks (existence, size, format)
    - Metadata consistency validation
    - Optimizer state verification
    - Model weights verification
    - Training state sanity checks
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

import torch

# Fix Windows console encoding for emoji support
if sys.platform == "win32":
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


class CheckpointVerifier:
    """Comprehensive checkpoint verification tool."""

    def __init__(self, checkpoint_path: str, strict: bool = False):
        """
        Initialize checkpoint verifier.

        Args:
            checkpoint_path (str): Path to checkpoint directory.
            strict (bool): If True, treat warnings as errors.
        """
        self.checkpoint_dir = Path(checkpoint_path)
        self.strict = strict
        self.errors = []
        self.warnings = []
        self.passed = 0
        self.failed = 0

    def error(self, msg: str) -> None:
        """Log an error."""
        self.errors.append(msg)
        self.failed += 1
        print(f"❌ ERROR: {msg}")

    def warning(self, msg: str) -> None:
        """Log a warning."""
        self.warnings.append(msg)
        if self.strict:
            self.errors.append(f"STRICT MODE: {msg}")
            self.failed += 1
        print(f"⚠️  WARNING: {msg}")

    def success(self, msg: str) -> None:
        """Log a success."""
        self.passed += 1
        print(f"✅ {msg}")

    def check_directory_exists(self) -> bool:
        """Check if checkpoint directory exists."""
        print("\n🔍 CHECK 1: Directory Existence")
        print("=" * 70)

        if not self.checkpoint_dir.exists():
            self.error(f"Checkpoint directory does not exist: {self.checkpoint_dir}")
            return False

        if not self.checkpoint_dir.is_dir():
            self.error(f"Path is not a directory: {self.checkpoint_dir}")
            return False

        self.success(f"Directory exists: {self.checkpoint_dir}")
        return True

    def check_required_files(self) -> bool:
        """Check if all required files exist and are valid."""
        print("\n🔍 CHECK 2: Required Files")
        print("=" * 70)

        required_files = {
            "training_state.pth": "Training state (optimizer, metrics, epoch/step)",
            "config.json": "Model configuration",
        }

        model_weights = {
            "model.safetensors": "Model weights (SafeTensors format - preferred)",
            "pytorch_model.bin": "Model weights (PyTorch format - legacy)",
        }

        all_valid = True

        # Check required files
        for filename, description in required_files.items():
            filepath = self.checkpoint_dir / filename
            if not filepath.exists():
                self.error(f"Missing required file: {filename} ({description})")
                all_valid = False
            elif filepath.stat().st_size == 0:
                self.error(f"File is empty: {filename}")
                all_valid = False
            else:
                size_mb = filepath.stat().st_size / (1024 * 1024)
                self.success(f"{filename}: {size_mb:.2f} MB - {description}")

        # Check model weights (at least one must exist)
        has_weights = False
        for filename, description in model_weights.items():
            filepath = self.checkpoint_dir / filename
            if filepath.exists() and filepath.stat().st_size > 0:
                size_mb = filepath.stat().st_size / (1024 * 1024)
                self.success(f"{filename}: {size_mb:.2f} MB - {description}")
                has_weights = True

        if not has_weights:
            self.error("No valid model weights found (need .safetensors or .bin)")
            all_valid = False

        return all_valid

    def check_training_state(self) -> Optional[Dict]:
        """Load and validate training state."""
        print("\n🔍 CHECK 3: Training State Integrity")
        print("=" * 70)

        state_path = self.checkpoint_dir / "training_state.pth"

        try:
            state = torch.load(state_path, map_location="cpu")
            self.success("Training state loaded successfully")
        except Exception as e:
            self.error(f"Failed to load training state: {e}")
            return None

        # Check required fields
        required_fields = {
            "epoch": "Current epoch number",
            "step": "Current step within epoch",
            "optimizer_state_dict": "Optimizer state",
            "best_val_f1": "Best validation F1 score",
            "best_val_acc": "Best validation accuracy",
        }

        for field, description in required_fields.items():
            if field not in state:
                self.error(f"Missing field '{field}' in training state ({description})")
            else:
                if field in ["epoch", "step"]:
                    self.success(f"{field}={state[field]} ({description})")
                elif field in ["best_val_f1", "best_val_acc"]:
                    self.success(f"{field}={state[field]:.4f} ({description})")
                else:
                    self.success(f"'{field}' present ({description})")

        # Check optional but recommended fields
        if "total_batches" not in state:
            self.warning(
                "Field 'total_batches' missing (legacy checkpoint). "
                "May cause issues if dataset size changed."
            )
        else:
            self.success(f"total_batches={state['total_batches']} (good!)")

        if "scaler_state_dict" not in state:
            self.warning(
                "Field 'scaler_state_dict' missing. "
                "AMP training may not resume correctly."
            )

        return state

    def check_metadata_consistency(self, state: Dict) -> bool:
        """Validate metadata consistency."""
        print("\n🔍 CHECK 4: Metadata Consistency")
        print("=" * 70)

        epoch = state.get("epoch", 0)
        step = state.get("step", 0)
        total_batches = state.get("total_batches", None)

        # Basic sanity checks
        if epoch < 0:
            self.error(f"Invalid epoch value: {epoch} (must be >= 0)")
            return False

        if step < 0:
            self.error(f"Invalid step value: {step} (must be >= 0)")
            return False

        self.success(f"Epoch and step values are valid (epoch={epoch}, step={step})")

        # Check step vs total_batches consistency
        if total_batches is not None:
            if step > total_batches:
                self.warning(
                    f"Step ({step}) > total_batches ({total_batches}). "
                    f"Dataset may have changed since checkpoint."
                )
            else:
                if step == 0:
                    self.success(
                        f"End-of-epoch checkpoint (step=0, epoch {epoch} completed)"
                    )
                else:
                    progress = (step / total_batches) * 100
                    self.success(
                        f"Mid-epoch checkpoint (step={step}/{total_batches}, {progress:.1f}% done)"
                    )

        # Check best metrics
        best_f1 = state.get("best_val_f1", 0.0)
        best_acc = state.get("best_val_acc", 0.0)

        if best_f1 > 1.0 or best_f1 < 0.0:
            self.warning(f"Best F1 out of range: {best_f1} (expected 0-1)")
        else:
            self.success(f"Best validation F1: {best_f1:.4f}")

        if best_acc > 1.0 or best_acc < 0.0:
            self.warning(f"Best accuracy out of range: {best_acc} (expected 0-1)")
        else:
            self.success(f"Best validation accuracy: {best_acc:.4f}")

        return True

    def check_optimizer_state(self, state: Dict) -> bool:
        """Validate optimizer state."""
        print("\n🔍 CHECK 5: Optimizer State")
        print("=" * 70)

        if "optimizer_state_dict" not in state:
            self.error("Optimizer state missing from checkpoint")
            return False

        opt_state = state["optimizer_state_dict"]

        # Check optimizer has parameter groups
        if "param_groups" not in opt_state:
            self.error("Optimizer missing 'param_groups'")
            return False

        num_param_groups = len(opt_state["param_groups"])
        self.success(f"Found {num_param_groups} parameter group(s)")

        # Check optimizer state
        if "state" not in opt_state or len(opt_state["state"]) == 0:
            self.warning(
                "Optimizer state is empty. "
                "This is normal for newly initialized optimizers."
            )
            return True

        # Get step count from first parameter
        first_param_state = opt_state["state"][0]
        if "step" in first_param_state:
            opt_steps = first_param_state["step"].item()
            self.success(f"Optimizer has performed {opt_steps} steps")

            # Verify consistency with checkpoint metadata
            epoch = state.get("epoch", 0)
            step = state.get("step", 0)
            total_batches = state.get("total_batches", 0)

            if total_batches > 0:
                if step == 0:
                    # End-of-epoch: expect optimizer_steps ≈ epoch * total_batches
                    expected_steps = epoch * total_batches
                else:
                    # Mid-epoch: expect optimizer_steps ≈ (epoch-1) * total_batches + step
                    expected_steps = (epoch - 1) * total_batches + step

                diff = abs(opt_steps - expected_steps)
                tolerance = max(20, total_batches // 10)  # 10% tolerance

                if diff <= tolerance:
                    self.success(
                        f"Optimizer steps ({opt_steps}) consistent with metadata "
                        f"(expected ~{expected_steps}, diff={diff})"
                    )
                else:
                    self.warning(
                        f"Optimizer steps ({opt_steps}) differ from expected ({expected_steps}) "
                        f"by {diff} steps. Possible causes: gradient accumulation, warmup."
                    )
        else:
            self.warning("Optimizer state missing 'step' field")

        return True

    def verify(self) -> bool:
        """Run all verification checks."""
        print("\n" + "=" * 70)
        print("CHECKPOINT VERIFICATION REPORT")
        print(f"Checkpoint: {self.checkpoint_dir}")
        print(f"Strict Mode: {self.strict}")
        print("=" * 70)

        # Run all checks
        if not self.check_directory_exists():
            return False

        if not self.check_required_files():
            return False

        state = self.check_training_state()
        if state is None:
            return False

        self.check_metadata_consistency(state)
        self.check_optimizer_state(state)

        # Print summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"✅ Passed: {self.passed}")
        print(f"⚠️  Warnings: {len(self.warnings)}")
        print(f"❌ Errors: {len(self.errors)}")

        if self.errors:
            print("\n❌ VERIFICATION FAILED")
            print("\nErrors found:")
            for i, error in enumerate(self.errors, 1):
                print(f"  {i}. {error}")
            return False
        elif self.warnings and self.strict:
            print("\n❌ VERIFICATION FAILED (strict mode)")
            print("\nWarnings (treated as errors in strict mode):")
            for i, warning in enumerate(self.warnings, 1):
                print(f"  {i}. {warning}")
            return False
        elif self.warnings:
            print("\n⚠️  VERIFICATION PASSED WITH WARNINGS")
            print("\nWarnings:")
            for i, warning in enumerate(self.warnings, 1):
                print(f"  {i}. {warning}")
            print(
                "\n💡 These warnings may be harmless, but review them before resuming."
            )
            return True
        else:
            print("\n✅ VERIFICATION PASSED")
            print("Checkpoint is valid and safe to resume training from.")
            return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Verify checkpoint integrity before resuming training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic verification
  python scripts/verify_checkpoint.py --checkpoint_path /path/to/checkpoint

  # Strict mode (treat warnings as errors)
  python scripts/verify_checkpoint.py --checkpoint_path /path/to/checkpoint --strict

Exit codes:
  0 - Verification passed
  1 - Verification failed
        """,
    )

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to checkpoint directory",
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help="Strict mode: treat warnings as errors",
    )

    args = parser.parse_args()

    # Run verification
    verifier = CheckpointVerifier(args.checkpoint_path, strict=args.strict)
    success = verifier.verify()

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
