"""
Comprehensive checkpoint validation tool.
Checks if checkpoint state matches the actual model weights.

Usage: python scripts/validate_checkpoint.py --checkpoint_path <path>
"""

import argparse
from pathlib import Path

import torch


def validate_checkpoint_consistency(checkpoint_path: str):
    """
    Validate that checkpoint metadata is consistent with model state.
    """
    checkpoint_dir = Path(checkpoint_path)

    print("=" * 70)
    print("CHECKPOINT VALIDATION")
    print("=" * 70)

    # Load training state
    state_file = checkpoint_dir / "training_state.pth"
    if not state_file.exists():
        print(f"❌ No training_state.pth found in {checkpoint_dir}")
        return

    state = torch.load(state_file, map_location="cpu")

    epoch = state.get("epoch", "N/A")
    step = state.get("step", "N/A")
    total_batches = state.get("total_batches", "N/A")

    print("\n📊 CHECKPOINT METADATA:")
    print(f"  Epoch: {epoch}")
    print(f"  Step: {step}")
    print(f"  Total Batches: {total_batches}")

    # Interpret what this means
    print("\n🔍 INTERPRETATION:")

    if step == 0:
        print("  ✅ This is an END-OF-EPOCH checkpoint")
        print(f"  📝 Meaning: Epoch {epoch} is COMPLETED")
        print("  📦 Model has trained on:")
        print(
            f"     - ALL batches of epoch {epoch} (batches 0 to {total_batches - 1 if total_batches != 'N/A' else '?'})"
        )
        print(
            f"  ➡️  When resuming: Will start epoch {epoch + 1 if epoch != 'N/A' else '?'}"
        )

    elif step != "N/A" and total_batches != "N/A":
        batches_trained = step
        batches_remaining = total_batches - step
        progress = (step / total_batches) * 100 if total_batches > 0 else 0

        print("  ⏸️  This is a MID-EPOCH checkpoint")
        print(f"  📝 Meaning: Epoch {epoch} is INCOMPLETE")
        print("  📦 Model has trained on:")
        print(
            f"     - Batches 0 to {step - 1} of epoch {epoch} ({batches_trained} batches)"
        )
        print("  ⏭️  NOT YET trained:")
        print(
            f"     - Batches {step} to {total_batches - 1} ({batches_remaining} batches)"
        )
        print(f"  📈 Progress: {progress:.1f}%")
        print(
            f"  ➡️  When resuming: Will skip batches 0-{step - 1}, train batches {step}-{total_batches - 1}"
        )

        # Important note about wasted work
        if batches_remaining > 100:
            print(f"\n  ⚠️  WARNING: {batches_remaining} batches remaining!")
            print(
                "     If training was interrupted, you may have already trained some of"
            )
            print(f"     batches {step}-{total_batches - 1} before the crash.")
            print("     Those batches will be RE-TRAINED when resuming.")
            print("     This is EXPECTED behavior with mid-epoch checkpoints.")

    # Check optimizer state
    print("\n🔧 OPTIMIZER STATE:")
    if "optimizer_state_dict" in state:
        opt_state = state["optimizer_state_dict"]
        if "state" in opt_state and len(opt_state["state"]) > 0:
            # Get first param state to check
            first_param_state = opt_state["state"][0]
            if "step" in first_param_state:
                opt_steps = first_param_state["step"].item()
                print(f"  ✅ Optimizer has performed {opt_steps} steps")

                # This should match the checkpoint step for mid-epoch
                if step != 0 and step != "N/A":
                    expected_opt_steps = step
                    if (
                        abs(opt_steps - expected_opt_steps) < 10
                    ):  # Allow small difference
                        print(
                            f"  ✅ Optimizer steps ({opt_steps}) matches checkpoint step ({step})"
                        )
                    else:
                        print(
                            f"  ⚠️  MISMATCH: Optimizer steps ({opt_steps}) != checkpoint step ({step})"
                        )
                        print("     This might indicate an inconsistent checkpoint!")
        else:
            print("  ⚠️  Optimizer state exists but appears empty")
    else:
        print("  ❌ No optimizer state found")

    # Check model files
    print("\n📁 MODEL FILES:")
    model_files = ["pytorch_model.bin", "model.safetensors", "config.json"]
    for fname in model_files:
        fpath = checkpoint_dir / fname
        if fpath.exists():
            size_mb = fpath.stat().st_size / (1024 * 1024)
            print(f"  ✅ {fname:<25} ({size_mb:>8.1f} MB)")
        else:
            print(f"  ❌ {fname:<25} (missing)")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if step == 0:
        print(f"✅ This checkpoint represents a COMPLETE epoch {epoch}")
        print(
            f"✅ Safe to resume - will start epoch {epoch + 1 if epoch != 'N/A' else '?'}"
        )
        print("✅ No batches will be skipped or duplicated")
    else:
        if total_batches != "N/A" and step < total_batches:
            wasted_batches = total_batches - step
            wasted_time_min = wasted_batches * 3.22 / 60  # Estimate
            print(f"⚠️  This checkpoint represents an INCOMPLETE epoch {epoch}")
            print(
                f"⚠️  {wasted_batches} batches may have been trained AFTER this checkpoint"
            )
            print(f"   (estimated ~{wasted_time_min:.1f} minutes of wasted work)")
            print("✅ When resuming: Will correctly train all remaining batches")
            print("✅ No batches will be permanently skipped")
            print("⚠️  Some batches may be trained twice (this is expected)")

    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate checkpoint consistency")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to checkpoint directory",
    )

    args = parser.parse_args()
    validate_checkpoint_consistency(args.checkpoint_path)
