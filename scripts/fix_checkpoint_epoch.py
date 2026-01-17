"""
Script to manually set checkpoint epoch to mark it as completed.
Usage: python scripts/fix_checkpoint_epoch.py --checkpoint_path <path> --epoch <epoch_number>
"""

import argparse
from pathlib import Path

import torch


def fix_checkpoint_epoch(checkpoint_path: str, target_epoch: int):
    """
    Manually set checkpoint to mark a specific epoch as completed.

    Args:
        checkpoint_path: Path to the checkpoint directory
        target_epoch: Epoch number to mark as completed
    """
    checkpoint_dir = Path(checkpoint_path)
    state_file = checkpoint_dir / "training_state.pth"

    if not state_file.exists():
        print(f"ERROR: Training state file not found: {state_file}")
        return

    # Load checkpoint
    print(f"Loading checkpoint from {state_file}...")
    state = torch.load(state_file, map_location="cpu")

    # Display current state
    print("\nCurrent checkpoint state:")
    print(f"  Epoch: {state.get('epoch', 'N/A')}")
    print(f"  Step: {state.get('step', 'N/A')}")
    print(f"  Total Batches: {state.get('total_batches', 'N/A')}")
    print(f"  Best Val F1: {state.get('best_val_f1', 'N/A')}")

    # Update state
    state["epoch"] = target_epoch
    state["step"] = 0  # Mark as completed

    # Save backup
    backup_file = state_file.parent / "training_state.pth.backup"
    print(f"\nCreating backup: {backup_file}")
    torch.save(torch.load(state_file, map_location="cpu"), backup_file)

    # Save updated state
    print("Saving updated state...")
    torch.save(state, state_file)

    # Verify
    updated_state = torch.load(state_file, map_location="cpu")
    print("\nUpdated checkpoint state:")
    print(f"  Epoch: {updated_state.get('epoch')}")
    print(f"  Step: {updated_state.get('step')}")
    print("\n✅ Checkpoint updated successfully!")
    print(f"   When resuming, training will start from epoch {target_epoch + 1}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix checkpoint epoch")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to checkpoint directory",
    )
    parser.add_argument(
        "--epoch", type=int, required=True, help="Epoch number to mark as completed"
    )

    args = parser.parse_args()
    fix_checkpoint_epoch(args.checkpoint_path, args.epoch)
