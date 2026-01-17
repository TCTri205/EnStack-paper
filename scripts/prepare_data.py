"""
Data preparation script for EnStack.

This script provides multiple options to prepare vulnerability detection data:
1. Use a publicly available vulnerability dataset from Hugging Face
2. Generate synthetic data for testing
3. Manual upload guide for Draper VDISC
"""

import logging
import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("DataPrep")


def generate_synthetic_data(output_dir: str, num_samples: Dict[str, int]):
    """
    Generate synthetic vulnerability data for testing the pipeline.

    Args:
        output_dir: Directory to save the data
        num_samples: Dict with keys 'train', 'val', 'test' and sample counts
    """
    logger.info("Generating synthetic vulnerability data...")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Sample vulnerable code patterns for each CWE type
    patterns = {
        0: "void func() { char buf[10]; gets(buf); }",  # CWE-119 Buffer
        1: "void func() { char *p = malloc(10); strcpy(p, input); }",  # CWE-120
        2: "int func(int x) { return x * 1000000; }",  # CWE-469 Integer overflow
        3: "void func() { int *p = NULL; *p = 5; }",  # CWE-476 NULL pointer
        4: "void func() { system(user_input); }",  # CWE-other
    }

    for split, count in num_samples.items():
        logger.info(f"Generating {count} samples for {split} split...")

        # Generate random samples
        funcs = []
        targets = []

        for i in range(count):
            # Random CWE type (0-4)
            cwe_type = np.random.randint(0, 5)

            # Create a variation of the pattern
            base_code = patterns[cwe_type]
            code_variant = f"// Sample {i}\n{base_code}\n// Line {i}"

            funcs.append(code_variant)
            targets.append(cwe_type)

        df = pd.DataFrame({"func": funcs, "target": targets})

        # Save as pickle
        filename = f"{split}_processed.pkl"
        save_path = output_path / filename
        df.to_pickle(save_path)
        logger.info(f"Saved {len(df)} samples to {save_path}")

        # Also save CSV sample
        df.head(20).to_csv(output_path / f"{split}_sample.csv", index=False)

    logger.info("✅ Synthetic data generation complete!")
    logger.info(f"📁 Data saved to: {output_path}")


def use_public_vulnerability_dataset(output_dir: str, sample_size: int = None):
    """
    Download and process a publicly available vulnerability dataset.

    Using the 'code_x_glue_cc_defect_detection' dataset which is similar
    and publicly available on Hugging Face.
    """
    try:
        from datasets import load_dataset

        logger.info(
            "Downloading public vulnerability dataset (code_x_glue_cc_defect_detection)..."
        )
        dataset = load_dataset("code_x_glue_cc_defect_detection")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Map splits
        split_mapping = {"train": "train", "validation": "val", "test": "test"}

        for hf_split, our_split in split_mapping.items():
            if hf_split not in dataset:
                continue

            logger.info(f"Processing {hf_split} split...")
            ds = dataset[hf_split]

            # Sample if requested
            if sample_size and len(ds) > sample_size:
                ds = ds.select(range(sample_size))

            df = ds.to_pandas()

            # The dataset has 'func' and 'target' columns (0=non-vulnerable, 1=vulnerable)
            # We need to map to 5 classes for our experiment
            # Simple approach: distribute vulnerable samples across 5 CWE types
            def map_target(target):
                if target == 0:
                    return 4  # Safe -> Other
                else:
                    # Randomly assign vulnerable code to CWE types 0-3
                    return np.random.randint(0, 4)

            df["target"] = df["target"].apply(map_target)

            # Save
            filename = f"{our_split}_processed.pkl"
            save_path = output_path / filename
            df[["func", "target"]].to_pickle(save_path)
            logger.info(f"Saved {len(df)} samples to {save_path}")

        logger.info("✅ Public dataset download complete!")
        return True

    except Exception as e:
        logger.error(f"Failed to download public dataset: {e}")
        return False


def print_manual_upload_guide(output_dir: str):
    """Print instructions for manually uploading Draper VDISC data."""
    logger.info("\n" + "=" * 60)
    logger.info("MANUAL DATA UPLOAD GUIDE - Draper VDISC Dataset")
    logger.info("=" * 60)
    logger.info("\nTo use the real Draper VDISC dataset:")
    logger.info("\n1. Download the dataset from the original source:")
    logger.info("   https://osf.io/d45bw/ (Russell et al., 2018)")
    logger.info("\n2. Process the data to have these columns:")
    logger.info("   - 'func': Source code (string)")
    logger.info("   - 'target': Label 0-4 (int)")
    logger.info("     0: CWE-119, 1: CWE-120, 2: CWE-469, 3: CWE-476, 4: Other")
    logger.info("\n3. Save as pickle files:")
    logger.info(f"   - {output_dir}/train_processed.pkl")
    logger.info(f"   - {output_dir}/val_processed.pkl")
    logger.info(f"   - {output_dir}/test_processed.pkl")
    logger.info("\n4. Upload these files to your Google Drive:")
    logger.info(f"   /content/drive/MyDrive/EnStack_Data/")
    logger.info("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare data for EnStack training")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data",
        help="Directory to save processed data",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="auto",
        choices=["auto", "synthetic", "public", "manual"],
        help="Data preparation mode",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Number of samples per split (for testing)",
    )
    parser.add_argument(
        "--train_size",
        type=int,
        default=10000,
        help="Training samples for synthetic mode",
    )
    parser.add_argument(
        "--val_size",
        type=int,
        default=2000,
        help="Validation samples for synthetic mode",
    )
    parser.add_argument(
        "--test_size", type=int, default=2000, help="Test samples for synthetic mode"
    )

    args = parser.parse_args()

    # Auto mode: try public dataset, fallback to synthetic
    if args.mode == "auto":
        logger.info("Auto mode: Attempting to use public dataset...")
        success = use_public_vulnerability_dataset(args.output_dir, args.sample)

        if not success:
            logger.info("Public dataset failed, using synthetic data...")
            num_samples = {
                "train": args.sample or args.train_size,
                "val": args.sample or args.val_size,
                "test": args.sample or args.test_size,
            }
            generate_synthetic_data(args.output_dir, num_samples)

    elif args.mode == "synthetic":
        num_samples = {
            "train": args.sample or args.train_size,
            "val": args.sample or args.val_size,
            "test": args.sample or args.test_size,
        }
        generate_synthetic_data(args.output_dir, num_samples)

    elif args.mode == "public":
        success = use_public_vulnerability_dataset(args.output_dir, args.sample)
        if not success:
            logger.error(
                "Failed to download public dataset. Try synthetic mode instead."
            )

    elif args.mode == "manual":
        print_manual_upload_guide(args.output_dir)

    logger.info("\n🎉 Data preparation script completed!")
