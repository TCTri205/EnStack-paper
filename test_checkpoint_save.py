"""
Quick test script to verify checkpoint saving mechanism.
This tests the atomic save and verification logic.
"""

import torch
import tempfile
from pathlib import Path
from transformers import RobertaForSequenceClassification, AutoTokenizer
from src.trainer import EnStackTrainer


def test_checkpoint_save():
    """Test checkpoint save with verification."""
    print("=" * 60)
    print("Testing Checkpoint Save & Verification")
    print("=" * 60)

    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        print(f"\n✓ Created temp directory: {temp_path}")

        # Create minimal model for testing
        model = RobertaForSequenceClassification.from_pretrained(
            "microsoft/codebert-base", num_labels=2
        )
        print("✓ Loaded test model")

        # Create trainer
        trainer = EnStackTrainer(
            model=model,
            output_dir=temp_path,
            learning_rate=2e-5,
        )
        print("✓ Created trainer")

        # Test 1: Save checkpoint
        print("\n" + "-" * 60)
        print("Test 1: Saving checkpoint...")
        print("-" * 60)
        try:
            trainer.save_checkpoint("test_checkpoint", epoch=1, step=100)
            checkpoint_path = temp_path / "test_checkpoint"

            # Verify checkpoint exists
            assert checkpoint_path.exists(), "Checkpoint directory does not exist!"
            print(f"✓ Checkpoint directory exists: {checkpoint_path}")

            # Verify required files
            required_files = ["training_state.pth", "config.json", "pytorch_model.bin"]
            for req_file in required_files:
                file_path = checkpoint_path / req_file
                # pytorch_model.bin might be model.safetensors instead
                if req_file == "pytorch_model.bin":
                    safetensors_path = checkpoint_path / "model.safetensors"
                    if safetensors_path.exists():
                        print(f"✓ Found {safetensors_path.name}")
                        continue

                assert file_path.exists(), f"Required file {req_file} not found!"
                assert file_path.stat().st_size > 0, f"File {req_file} is empty!"
                print(f"✓ Found {req_file} ({file_path.stat().st_size} bytes)")

            print("\n✅ Test 1 PASSED: Checkpoint saved successfully")

        except Exception as e:
            print(f"\n❌ Test 1 FAILED: {e}")
            raise

        # Test 2: Verify cleanup of temp directories
        print("\n" + "-" * 60)
        print("Test 2: Verifying temp directory cleanup...")
        print("-" * 60)
        temp_dirs = [p for p in temp_path.iterdir() if p.name.startswith(".tmp_")]
        if len(temp_dirs) == 0:
            print("✓ No leftover .tmp_ directories")
            print("\n✅ Test 2 PASSED: Cleanup successful")
        else:
            print(f"❌ Found {len(temp_dirs)} leftover temp directories:")
            for tmp_dir in temp_dirs:
                print(f"   - {tmp_dir.name}")
            raise AssertionError("Temp directories not cleaned up!")

        # Test 3: Load checkpoint
        print("\n" + "-" * 60)
        print("Test 3: Loading checkpoint...")
        print("-" * 60)
        try:
            loaded_epoch, loaded_step = trainer.load_checkpoint(str(checkpoint_path))
            assert loaded_epoch == 1, f"Expected epoch 1, got {loaded_epoch}"
            assert loaded_step == 100, f"Expected step 100, got {loaded_step}"
            print(f"✓ Loaded checkpoint: epoch={loaded_epoch}, step={loaded_step}")
            print("\n✅ Test 3 PASSED: Checkpoint loaded successfully")

        except Exception as e:
            print(f"\n❌ Test 3 FAILED: {e}")
            raise

    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    test_checkpoint_save()
