import logging
import os
import re
import shutil
import stat
import sys
from pathlib import Path
from typing import List, Tuple

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("Cleaner")


def force_delete(path: Path) -> None:
    """Robustly deletes a directory, handling read-only files."""
    if not path.exists():
        return

    def handle_error(func, path, exc_info):
        # Check if access denied
        if not os.access(path, os.W_OK):
            os.chmod(path, stat.S_IWUSR)
            func(path)
        else:
            raise

    try:
        shutil.rmtree(path, onerror=handle_error)
        logger.info(f"✅ Deleted: {path.name}")
    except Exception as e:
        logger.error(f"❌ Failed to delete {path.name}: {e}")


def cleanup_checkpoints(checkpoint_dir: str, dry_run: bool = False):
    """
    Cleans up temporary and redundant checkpoints.

    Args:
        checkpoint_dir: Path to the checkpoints directory.
        dry_run: If True, only list what would be deleted.
    """
    root = Path(checkpoint_dir)
    if not root.exists():
        logger.error(f"Directory not found: {root}")
        return

    logger.info(f"Scanning directory: {root}")
    if dry_run:
        logger.info("--- DRY RUN MODE (No files will be deleted) ---")

    # 1. Scan for .tmp folders (Trash)
    tmp_folders = list(root.glob(".tmp_*"))

    # 2. Scan for recovery checkpoint (Redundant if others exist)
    recovery = root / "recovery_checkpoint"

    # 3. Scan for numbered checkpoints for rotation
    pattern = re.compile(r"checkpoint_epoch(\d+)_step(\d+)")
    checkpoints: List[Tuple[int, int, Path]] = []

    for path in root.iterdir():
        if path.is_dir() and pattern.match(path.name):
            match = pattern.match(path.name)
            checkpoints.append((int(match.group(1)), int(match.group(2)), path))

    # Sort: Keep only the single latest checkpoint (highest epoch, then highest step)
    checkpoints.sort(key=lambda x: (x[0], x[1]))

    to_delete = []

    # Add tmp folders
    to_delete.extend(tmp_folders)

    # Add recovery if it exists
    if recovery.exists():
        to_delete.append(recovery)

    # Add redundant checkpoints (keep last 1)
    if len(checkpoints) > 1:
        redundant = checkpoints[:-1]  # All except last one
        for _, _, path in redundant:
            to_delete.append(path)

    # Report status
    logger.info(f"Found {len(to_delete)} items to clean up.")

    if not to_delete:
        logger.info("Directory is clean! ✨")
        return

    # Execute
    for path in to_delete:
        if dry_run:
            logger.info(f"Would delete: {path.name}")
        else:
            force_delete(path)

    if dry_run:
        logger.info(f"\nTo actually delete, run this script without --dry-run")


if __name__ == "__main__":
    # Default path from your logs, can be overridden via args
    default_path = "/content/drive/MyDrive/EnStack_Data/checkpoints/codebert"

    import argparse

    parser = argparse.ArgumentParser(description="Clean up EnStack checkpoints")
    parser.add_argument(
        "--path", type=str, default=default_path, help="Path to checkpoints folder"
    )
    parser.add_argument("--dry-run", action="store_true", help="Simulate deletion only")
    parser.add_argument(
        "--confirm", action="store_true", help="Skip confirmation prompt"
    )

    args = parser.parse_args()

    print(f"Target Path: {args.path}")
    if not args.dry_run and not args.confirm:
        response = input("Are you sure you want to delete files? [y/N]: ")
        if response.lower() != "y":
            print("Aborted.")
            sys.exit(0)

    cleanup_checkpoints(args.path, dry_run=args.dry_run)
