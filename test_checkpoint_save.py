#!/usr/bin/env python3
"""
Test script to verify the OPTIMIZED save_checkpoint() mechanism.
Tests "Local-First" strategy for Google Drive and os.sync() flushing.
"""

import sys
import tempfile
import time
import shutil
import os
from pathlib import Path


# Mock minimal checkpoint structure
def test_checkpoint_save(test_dir):
    """
    Simulates the optimized checkpoint saving process.
    """
    test_dir = Path(test_dir)
    test_dir.mkdir(parents=True, exist_ok=True)

    is_gdrive = "/content/drive/" in str(test_dir)
    print(f"📂 Test directory: {test_dir}")
    print(f"🔍 Google Drive detected: {is_gdrive}")
    print()

    checkpoint_name = "test_checkpoint"
    save_path = test_dir / checkpoint_name

    # 1. OPTIMIZATION TEST: Create Temp Directory
    print(f"🧪 Test 1: Creating temporary checkpoint...")

    temp_dir = None
    if is_gdrive and Path("/content").exists():
        print("   🚀 Strategy: Local-First (Fast VM SSD)")
        temp_base = Path("/content/temp_test_checkpoints")
        temp_base.mkdir(parents=True, exist_ok=True)
        temp_dir = Path(
            tempfile.mkdtemp(dir=temp_base, prefix=f".tmp_{checkpoint_name}_")
        )
    else:
        print("   🐢 Strategy: Direct (Same Filesystem)")
        temp_dir = Path(
            tempfile.mkdtemp(dir=test_dir, prefix=f".tmp_{checkpoint_name}_")
        )

    print(f"   Created temp dir: {temp_dir}")

    # Write big dummy file to simulate model weights (10MB)
    print("   Writing dummy model weights (10MB)...")
    with open(temp_dir / "model.safetensors", "wb") as f:
        f.write(os.urandom(10 * 1024 * 1024))

    (temp_dir / "config.json").write_text('{"test": true}')
    print(f"   ✅ Files written to temp directory")

    # 2. SAVE TEST
    if is_gdrive:
        print(f"\n🧪 Test 2: Copying to Drive with Sync...")

        if save_path.exists():
            print(f"   Removing existing checkpoint...")
            shutil.rmtree(save_path)
            time.sleep(0.5)

        print(f"   Copying {temp_dir} -> {save_path}")
        shutil.copytree(str(temp_dir), str(save_path))

        print(f"   🔄 Flushing OS buffers (os.sync)...")
        if hasattr(os, "sync"):
            os.sync()

        print(f"   ⏳ Waiting for Drive sync (3s)...")
        time.sleep(3.0)

        # Verify
        if save_path.exists():
            print(f"   ✅ Directory exists: {checkpoint_name}")
        else:
            print(f"   ❌ FAILED: Directory does not exist!")
            return False

        test_file = save_path / "model.safetensors"
        if test_file.exists():
            size_mb = test_file.stat().st_size / (1024 * 1024)
            print(f"   ✅ Files verified (size: {size_mb:.2f} MB)")
            if size_mb < 9.9:
                print(f"   ⚠️ WARNING: File size mismatch!")
        else:
            print(f"   ❌ FAILED: Files missing!")
            return False

        # Clean up temp
        shutil.rmtree(temp_dir)
        print(f"   ✅ Local temp directory cleaned up")

        # Clean up test dir on drive
        shutil.rmtree(temp_base, ignore_errors=True)

    else:
        print(f"\n🧪 Test 2: Atomic Move (Local)...")
        shutil.move(str(temp_dir), str(save_path))

        if save_path.exists():
            print(f"   ✅ Checkpoint saved successfully")
        else:
            print(f"   ❌ FAILED: Checkpoint not found!")
            return False

    print(f"\n✅ All tests passed!")
    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test optimized checkpoint saving")
    parser.add_argument(
        "--path",
        type=str,
        default="/content/drive/MyDrive/EnStack_Data/checkpoints/test_opt",
        help="Test directory path",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("EnStack Optimized Checkpoint Test")
    print("=" * 60)

    try:
        success = test_checkpoint_save(args.path)
        if success:
            print("\n✅ OPTIMIZATION VERIFIED")
            sys.exit(0)
        else:
            print("\n❌ TEST FAILED")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
