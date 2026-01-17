"""
Data preparation script for EnStack.

This script provides multiple options to prepare vulnerability detection data:
1. Process Draper VDISC HDF5 files (RECOMMENDED for paper results)
2. Use a publicly available vulnerability dataset from Hugging Face
3. Generate synthetic data for testing
"""

import logging
import os
import pickle
import pandas as pd
import numpy as np
import h5py
from pathlib import Path
from typing import Dict, List, Tuple

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("DataPrep")

# Mapping from column name in HDF5 to Class ID
CWE_MAPPING = {
    "CWE-119": 0,
    "CWE-120": 1,
    "CWE-469": 2,
    "CWE-476": 3,
    "CWE-other": 4,
}

# Target distribution from Table I of EnStack paper
PAPER_COUNTS = {
    "train": {0: 5942, 1: 5777, 2: 249, 3: 2755, 4: 5582},
    "val": {0: 1142, 1: 1099, 2: 53, 3: 535, 4: 1071},
    "test": {0: 1142, 1: 1099, 2: 53, 3: 535, 4: 1071},
}


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


def process_draper_files(data_dir: str, output_dir: str, match_paper: bool = False):
    """
    Process raw Draper VDISC HDF5 files to create preprocessed pkl files.

    Args:
        data_dir: Directory containing .hdf5 files
        output_dir: Directory to save processed .pkl files
        match_paper: If True, downsample to match EnStack paper distribution
    """
    logger.info("Processing Draper VDISC HDF5 files...")
    input_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Expected filenames
    files = {
        "train": "VDISC_train.hdf5",
        "val": "VDISC_validate.hdf5",
        "test": "VDISC_test.hdf5",
    }

    for split, filename in files.items():
        filepath = input_path / filename
        if not filepath.exists():
            logger.warning(
                f"File {filename} not found in {input_path}. Skipping {split}."
            )
            continue

        logger.info(f"Processing {split} split: {filepath}")

        try:
            with h5py.File(filepath, "r") as f:
                keys = list(f.keys())
                logger.info(f"Keys in HDF5: {keys}")

                if "functionSource" not in keys:
                    logger.error(f"Invalid HDF5 format. Missing 'functionSource'.")
                    continue

                # Read CWE columns
                cwe_cols = [c for c in CWE_MAPPING.keys() if c in keys]
                logger.info(f"Found CWE columns: {cwe_cols}")

                # Create labels dataframe
                df_labels = pd.DataFrame()
                for col in cwe_cols:
                    df_labels[col] = f[col][:]

                logger.info(f"Total samples in {split}: {len(df_labels)}")

                # Assign labels with priority
                df_labels["target"] = -1

                if "CWE-other" in cwe_cols:
                    df_labels.loc[df_labels["CWE-other"] == True, "target"] = 4
                if "CWE-476" in cwe_cols:
                    df_labels.loc[df_labels["CWE-476"] == True, "target"] = 3
                if "CWE-469" in cwe_cols:
                    df_labels.loc[df_labels["CWE-469"] == True, "target"] = 2
                if "CWE-120" in cwe_cols:
                    df_labels.loc[df_labels["CWE-120"] == True, "target"] = 1
                if "CWE-119" in cwe_cols:
                    df_labels.loc[df_labels["CWE-119"] == True, "target"] = 0

                # Filter valid samples
                valid_indices = df_labels[df_labels["target"] != -1].index
                logger.info(f"Found {len(valid_indices)} vulnerable samples.")

                # Downsampling
                final_indices = []

                if match_paper and split in PAPER_COUNTS:
                    target_counts = PAPER_COUNTS[split]
                    logger.info(f"Downsampling to match paper distribution...")

                    for label, count in target_counts.items():
                        label_indices = df_labels[
                            df_labels["target"] == label
                        ].index.tolist()
                        if len(label_indices) >= count:
                            selected = np.random.choice(
                                label_indices, count, replace=False
                            )
                        else:
                            logger.warning(
                                f"Label {label} has fewer samples than needed. Using all."
                            )
                            selected = label_indices
                        final_indices.extend(selected)
                else:
                    final_indices = valid_indices.tolist()

                final_indices = sorted(final_indices)
                logger.info(f"Selected {len(final_indices)} samples for final dataset.")

                # Read source code
                logger.info("Reading source code... (this may take a moment)")
                raw_sources = f["functionSource"]
                sources = []

                for i in final_indices:
                    sources.append(raw_sources[i].decode("utf-8"))

                # Create final dataframe
                df_final = pd.DataFrame(
                    {
                        "func": sources,
                        "target": df_labels.loc[final_indices, "target"].values,
                    }
                )

                # Save
                output_file = output_path / f"{split}_processed.pkl"
                df_final.to_pickle(output_file)
                logger.info(f"Saved processed data to {output_file}")
                logger.info(
                    f"Class distribution:\n{df_final['target'].value_counts().sort_index()}"
                )

        except Exception as e:
            logger.error(f"Error processing {filepath}: {e}")
            import traceback

            traceback.print_exc()

    logger.info("✅ Draper VDISC processing complete!")


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
        choices=["auto", "synthetic", "public", "draper"],
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
    parser.add_argument(
        "--draper_dir",
        type=str,
        default=None,
        help="Directory containing Draper VDISC HDF5 files (for draper mode)",
    )
    parser.add_argument(
        "--match_paper",
        action="store_true",
        help="Downsample Draper data to match paper counts exactly",
    )

    args = parser.parse_args()

    # Auto mode: check for draper files first, then public, fallback to synthetic
    if args.mode == "auto":
        draper_dir = args.draper_dir or f"{args.output_dir}/raw_data"
        if Path(draper_dir).exists() and any(Path(draper_dir).glob("*.hdf5")):
            logger.info(
                f"Auto mode: Found HDF5 files in {draper_dir}. Processing Draper data..."
            )
            process_draper_files(draper_dir, args.output_dir, args.match_paper)
        else:
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

    elif args.mode == "draper":
        draper_dir = args.draper_dir or f"{args.output_dir}/raw_data"
        if not Path(draper_dir).exists():
            logger.error(
                f"Draper directory not found: {draper_dir}. Please provide --draper_dir."
            )
            print_manual_upload_guide(args.output_dir)
        else:
            process_draper_files(draper_dir, args.output_dir, args.match_paper)

    elif args.mode == "synthetic":
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

    logger.info("\n🎉 Data preparation script completed!")
