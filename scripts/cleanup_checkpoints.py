"""
Utility to clean up old mid-epoch checkpoints to save disk space.
Usage: python scripts/cleanup_checkpoints.py --checkpoint_dir <path> [--keep-last N]
"""

import argparse
import shutil
from pathlib import Path
import re


def cleanup_mid_epoch_checkpoints(checkpoint_dir: str, keep_last: int = 0):
    """
    Remove old mid-epoch checkpoints, keeping only the most recent N.

    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_last: Number of most recent mid-epoch checkpoints to keep (0 = delete all)
    """
    checkpoint_path = Path(checkpoint_dir)

    if not checkpoint_path.exists():
        print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
        return

    print(f"🔍 Scanning for mid-epoch checkpoints in: {checkpoint_dir}")

    # Find all mid-epoch checkpoints (format: checkpoint_epochX_stepY)
    pattern = re.compile(r"checkpoint_epoch(\d+)_step(\d+)")
    mid_epoch_checkpoints = []

    for item in checkpoint_path.iterdir():
        if item.is_dir():
            match = pattern.match(item.name)
            if match:
                epoch = int(match.group(1))
                step = int(match.group(2))
                mid_epoch_checkpoints.append(
                    {"path": item, "name": item.name, "epoch": epoch, "step": step}
                )

    if not mid_epoch_checkpoints:
        print("✅ No mid-epoch checkpoints found.")
        return

    # Sort by epoch and step (most recent last)
    mid_epoch_checkpoints.sort(key=lambda x: (x["epoch"], x["step"]))

    print(f"\n📊 Found {len(mid_epoch_checkpoints)} mid-epoch checkpoints:")
    total_size = 0
    for ckpt in mid_epoch_checkpoints:
        size = sum(f.stat().st_size for f in ckpt["path"].rglob("*") if f.is_file())
        total_size += size
        size_mb = size / (1024 * 1024)
        print(f"  - {ckpt['name']:<40} ({size_mb:>8.1f} MB)")

    print(f"\n💾 Total disk usage: {total_size / (1024 * 1024):.1f} MB")

    # Determine which to delete
    if keep_last > 0:
        to_delete = mid_epoch_checkpoints[:-keep_last]
        to_keep = mid_epoch_checkpoints[-keep_last:]
    else:
        to_delete = mid_epoch_checkpoints
        to_keep = []

    if not to_delete:
        print(f"\n✅ No checkpoints to delete (keeping last {keep_last})")
        return

    print(f"\n🗑️  Will DELETE {len(to_delete)} checkpoint(s):")
    delete_size = 0
    for ckpt in to_delete:
        size = sum(f.stat().st_size for f in ckpt["path"].rglob("*") if f.is_file())
        delete_size += size
        size_mb = size / (1024 * 1024)
        print(f"  - {ckpt['name']:<40} ({size_mb:>8.1f} MB)")

    if to_keep:
        print(f"\n✅ Will KEEP {len(to_keep)} checkpoint(s):")
        for ckpt in to_keep:
            size = sum(f.stat().st_size for f in ckpt["path"].rglob("*") if f.is_file())
            size_mb = size / (1024 * 1024)
            print(f"  - {ckpt['name']:<40} ({size_mb:>8.1f} MB)")

    print(f"\n💰 Will free up: {delete_size / (1024 * 1024):.1f} MB")

    # Confirm deletion
    response = input("\n⚠️  Proceed with deletion? (yes/no): ").strip().lower()

    if response != "yes":
        print("❌ Cancelled. No checkpoints were deleted.")
        return

    # Delete checkpoints
    print("\n🗑️  Deleting checkpoints...")
    deleted_count = 0
    for ckpt in to_delete:
        try:
            shutil.rmtree(ckpt["path"])
            print(f"  ✅ Deleted: {ckpt['name']}")
            deleted_count += 1
        except Exception as e:
            print(f"  ❌ Failed to delete {ckpt['name']}: {e}")

    print(f"\n✅ Successfully deleted {deleted_count}/{len(to_delete)} checkpoints")
    print(f"💾 Freed up ~{delete_size / (1024 * 1024):.1f} MB of disk space")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clean up old mid-epoch checkpoints to save disk space"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Directory containing checkpoints (e.g., /path/to/checkpoints/codebert)",
    )
    parser.add_argument(
        "--keep-last",
        type=int,
        default=0,
        help="Number of most recent mid-epoch checkpoints to keep (default: 0 = delete all)",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Skip confirmation prompt (use with caution!)",
    )

    args = parser.parse_args()

    if args.auto:
        # Auto mode: patch input() to auto-confirm
        import builtins

        builtins.input = lambda _: "yes"

    cleanup_mid_epoch_checkpoints(args.checkpoint_dir, args.keep_last)
