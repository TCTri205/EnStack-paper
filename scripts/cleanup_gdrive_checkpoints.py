#!/usr/bin/env python3
"""
Script to clean up orphaned temporary checkpoint directories in Google Drive.
Use this after fixing the save_checkpoint() bug to remove leftover .tmp folders.
"""

import os
import shutil
import sys
from pathlib import Path


def force_delete(path):
    """Force delete a directory, handling permission issues."""
    try:
        shutil.rmtree(path)
        print(f"✅ Deleted: {path.name}")
        return True
    except Exception as e:
        print(f"❌ Failed to delete {path.name}: {e}")
        return False


def cleanup_temp_checkpoints(checkpoint_dir, dry_run=False):
    """
    Clean up temporary checkpoint directories.

    Args:
        checkpoint_dir: Path to the checkpoint directory
        dry_run: If True, only show what would be deleted
    """
    checkpoint_path = Path(checkpoint_dir)

    if not checkpoint_path.exists():
        print(f"❌ Directory not found: {checkpoint_dir}")
        return

    print(f"📂 Scanning: {checkpoint_dir}")
    print(
        f"{'🔍 DRY RUN MODE - No files will be deleted' if dry_run else '⚠️  DELETION MODE - Files will be permanently removed'}\n"
    )

    # Find all .tmp and .backup directories
    tmp_dirs = []
    for item in checkpoint_path.iterdir():
        if item.is_dir() and (
            item.name.startswith(".tmp_") or item.name.startswith(".backup_")
        ):
            tmp_dirs.append(item)

    if not tmp_dirs:
        print("✨ No temporary directories found - checkpoint folder is clean!")
        return

    print(f"Found {len(tmp_dirs)} temporary directories:\n")
    for tmp_dir in tmp_dirs:
        # Get size
        try:
            size_mb = sum(
                f.stat().st_size for f in tmp_dir.rglob("*") if f.is_file()
            ) / (1024 * 1024)
            print(f"  📦 {tmp_dir.name} ({size_mb:.1f} MB)")
        except:
            print(f"  📦 {tmp_dir.name}")

    print()

    if dry_run:
        print("ℹ️  To actually delete these, run without --dry-run flag")
        return

    # Confirm deletion
    response = input("⚠️  Delete all temporary directories? [y/N]: ")
    if response.lower() != "y":
        print("❌ Aborted")
        return

    print("\n🗑️  Deleting temporary directories...\n")

    deleted = 0
    failed = 0
    for tmp_dir in tmp_dirs:
        if force_delete(tmp_dir):
            deleted += 1
        else:
            failed += 1

    print(f"\n✅ Cleanup complete: {deleted} deleted, {failed} failed")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Clean up temporary checkpoint directories in Google Drive"
    )
    parser.add_argument(
        "--path",
        type=str,
        default="/content/drive/MyDrive/EnStack_Data/checkpoints/codebert",
        help="Path to checkpoint directory (default: Google Drive path)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting",
    )

    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Scan all model subdirectories in the parent folder",
    )

    args = parser.parse_args()

    if args.all_models:
        parent_dir = Path(args.path).parent
        print(f"🔍 Scanning all model directories in: {parent_dir}")
        for model_dir in parent_dir.iterdir():
            if model_dir.is_dir() and not model_dir.name.startswith("."):
                cleanup_temp_checkpoints(str(model_dir), dry_run=args.dry_run)
    else:
        cleanup_temp_checkpoints(args.path, dry_run=args.dry_run)
