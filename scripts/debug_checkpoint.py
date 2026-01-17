"""
Debug checkpoint state and analyze resume behavior.
Usage: python scripts/debug_checkpoint.py --checkpoint_path <path>
"""

import argparse
from pathlib import Path

import torch


def analyze_checkpoint(checkpoint_path: str):
    """
    Analyze checkpoint state in detail.
    """
    checkpoint_dir = Path(checkpoint_path)
    state_file = checkpoint_dir / "training_state.pth"

    print("=" * 70)
    print("CHECKPOINT ANALYSIS")
    print("=" * 70)

    if not state_file.exists():
        print(f"❌ Training state file NOT found: {state_file}")
        print("\nChecking directory contents:")
        if checkpoint_dir.exists():
            for f in checkpoint_dir.iterdir():
                print(f"  - {f.name}")
        else:
            print(f"❌ Checkpoint directory does not exist: {checkpoint_dir}")
        return

    # Load checkpoint
    print(f"\n📂 Loading checkpoint from: {state_file}")
    state = torch.load(state_file, map_location="cpu")

    print("\n📊 CHECKPOINT STATE:")
    print(f"  {'Key':<25} {'Value':<20} {'Type':<15}")
    print("  " + "-" * 60)

    for key in sorted(state.keys()):
        if key in ["epoch", "step", "best_val_f1", "best_val_acc", "total_batches"]:
            value = state[key]
            print(f"  {key:<25} {str(value):<20} {type(value).__name__:<15}")

    # Get key values
    epoch = state.get("epoch", "N/A")
    step = state.get("step", "N/A")
    total_batches = state.get("total_batches", "N/A")
    best_f1 = state.get("best_val_f1", "N/A")

    print("\n🔍 INTERPRETATION:")
    print(f"  Current Epoch: {epoch}")
    print(f"  Current Step: {step}")
    print(f"  Total Batches per Epoch: {total_batches}")
    print(f"  Best Validation F1: {best_f1}")

    # Analyze completion status
    print("\n✅ COMPLETION STATUS:")

    if epoch == "N/A":
        print("  ⚠️  No epoch information (old checkpoint format)")
    elif step == "N/A":
        print("  ⚠️  No step information (old checkpoint format)")
    elif step == 0:
        print(f"  ✅ Epoch {epoch} is COMPLETED")
        print(f"  ➡️  Resume will start from epoch {epoch + 1}")
    elif total_batches != "N/A" and step >= total_batches:
        print(f"  ✅ Epoch {epoch} is COMPLETED (step {step} >= {total_batches})")
        print(f"  ➡️  Resume will start from epoch {epoch + 1}")
    else:
        if total_batches != "N/A":
            remaining = total_batches - step
            progress = (step / total_batches) * 100
            print(f"  ⏸️  Epoch {epoch} is INCOMPLETE")
            print(f"     Progress: {step}/{total_batches} batches ({progress:.1f}%)")
            print(f"     Remaining: {remaining} batches")
            print(f"  ➡️  Resume will continue epoch {epoch} from step {step}")
        else:
            print(f"  ⏸️  Epoch {epoch} appears INCOMPLETE (step={step})")
            print("     Cannot determine exact progress (no total_batches info)")
            print(f"  ➡️  Resume will continue epoch {epoch} from step {step}")

    # Check for optimizer state
    print("\n📦 OPTIMIZER STATE:")
    if "optimizer_state_dict" in state:
        opt_state = state["optimizer_state_dict"]
        if "state" in opt_state:
            num_params = len(opt_state["state"])
            print(f"  ✅ Optimizer state exists ({num_params} parameter groups)")
        else:
            print("  ⚠️  Optimizer state exists but appears empty")
    else:
        print("  ❌ No optimizer state found")

    # Check for scheduler state
    if "scheduler_state_dict" in state:
        print("  ✅ Scheduler state exists")
    else:
        print("  ⚠️  No scheduler state found")

    print("\n" + "=" * 70)

    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    if step == 0 or (total_batches != "N/A" and step >= total_batches):
        print(f"  • This checkpoint marks the END of epoch {epoch}")
        print(f"  • Resuming will correctly start from epoch {epoch + 1}")
    elif step != "N/A" and step != 0:
        print("  • This checkpoint was saved MID-EPOCH")
        print("  • If you believe the epoch completed, the checkpoint may be outdated")
        print("  • Possible causes:")
        print("    1. Training crashed/stopped during epoch")
        print("    2. Checkpoint at end of epoch failed to save")
        print(
            "    3. End-of-epoch checkpoint was overwritten by a later mid-epoch save"
        )
        print("  • Solutions:")
        print(
            "    - Use scripts/fix_checkpoint_epoch.py to manually mark epoch as complete"
        )
        print("    - OR delete checkpoint and restart training")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze checkpoint state")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to checkpoint directory (containing training_state.pth)",
    )

    args = parser.parse_args()
    analyze_checkpoint(args.checkpoint_path)
