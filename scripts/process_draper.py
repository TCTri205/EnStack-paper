"""
Script to process the original Draper VDISC Dataset (.hdf5 files) for EnStack.

This script converts the raw HDF5 files from the Draper VDISC dataset into the
preprocessed pickle format required by EnStack. It handles:
1. Extraction of source code and labels.
2. Mapping specific CWEs to class IDs (0-4).
3. Downsampling to match the distribution described in the EnStack paper (Table I).

Usage:
    python scripts/process_draper.py --data_dir /path/to/hdf5/files --output_dir /path/to/save/pkl
"""

import argparse
import logging
import pandas as pd
import numpy as np
import h5py
from pathlib import Path
from typing import Dict, List, Optional

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("DraperPrep")

# Mapping from column name in HDF5 to Class ID
# Based on EnStack Paper:
# 0: CWE-119 (Memory)
# 1: CWE-120 (Buffer Overflow)
# 2: CWE-469 (Integer Overflow)
# 3: CWE-476 (Null Pointer)
# 4: CWE-other (Miscellaneous)

CWE_MAPPING = {
    "CWE-119": 0,
    "CWE-120": 1,
    "CWE-469": 2,
    "CWE-476": 3,
    "CWE-other": 4,  # Note: Verify specific column name in dataset, often 'CWE-other' or similar
}

# Target distribution from Table I of the paper (approximate ratios)
# We will use these as limits if --match_paper is set
PAPER_COUNTS = {
    "train": {0: 5942, 1: 5777, 2: 249, 3: 2755, 4: 5582},
    "val": {0: 1142, 1: 1099, 2: 53, 3: 535, 4: 1071},
    "test": {0: 1142, 1: 1099, 2: 53, 3: 535, 4: 1071},
}


def get_label(row, columns):
    """
    Determine the class label for a row.
    Prioritizes specific CWEs over 'CWE-other'.
    """
    # Check specific CWEs first
    for cwe, label_id in CWE_MAPPING.items():
        if cwe == "CWE-other":
            continue
        if row[cwe]:  # If boolean is true
            return label_id

    # If no specific CWE, check CWE-other
    if "CWE-other" in columns and row["CWE-other"]:
        return CWE_MAPPING["CWE-other"]

    return -1  # Non-vulnerable (or not in our interest set)


def process_file(
    filepath: Path, split_name: str, output_dir: Path, match_paper: bool = False
):
    """
    Process a single HDF5 file.
    """
    logger.info(f"Processing {split_name} file: {filepath}")

    if not filepath.exists():
        logger.error(f"File not found: {filepath}")
        return

    try:
        with h5py.File(filepath, "r") as f:
            # Check available keys
            keys = list(f.keys())
            logger.info(f"Keys in HDF5: {keys}")

            # Usually the dataset is under 'functionSource' and CWE columns
            if "functionSource" not in keys:
                logger.error(f"Invalid HDF5 format. Missing 'functionSource'.")
                return

            # Read columns we need
            # Note: Draper VDISC is huge, reading all to memory might crash on small RAM.
            # We will read in batches or carefully.
            # For simplicity in this script, we assume we can read the indices of vulnerabilities.

            # Let's read the CWE columns first to filter indices
            cwe_cols = [c for c in CWE_MAPPING.keys() if c in keys]
            logger.info(f"Found CWE columns: {cwe_cols}")

            # Create a dataframe for labels first to save memory
            df_labels = pd.DataFrame()
            for col in cwe_cols:
                df_labels[col] = f[col][:]

            logger.info(f"Total samples in {split_name}: {len(df_labels)}")

            # Assign labels
            # We want to filter for rows that have at least one of our targets
            # and map them to 0-4

            # Logic:
            # 1. Create a 'target' column initialized to -1
            # 2. Update based on priority

            df_labels["target"] = -1

            # Priority: 469 (Rare) > 476 > 119 > 120 > Other (Just an heuristic to handle multi-label)
            # Or strictly follow mapping. Paper implies single classification.
            # We'll stick to a simple order or the order in dictionary.

            # Vectorized assignment is faster
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

            # Filter only valid targets (0-4)
            # The paper says "Null entries were then removed" (meaning non-vulnerable lines or unlabelled)
            valid_indices = df_labels[df_labels["target"] != -1].index

            logger.info(f"Found {len(valid_indices)} vulnerable samples.")

            # Downsampling
            final_indices = []

            if match_paper and split_name in PAPER_COUNTS:
                target_counts = PAPER_COUNTS[split_name]
                logger.info(
                    f"Downsampling to match paper distributions for {split_name}..."
                )

                for label, count in target_counts.items():
                    # Get indices for this label
                    label_indices = df_labels[
                        df_labels["target"] == label
                    ].index.tolist()

                    if len(label_indices) >= count:
                        selected = np.random.choice(label_indices, count, replace=False)
                    else:
                        logger.warning(
                            f"Label {label} has {len(label_indices)} samples, wanted {count}. Using all."
                        )
                        selected = label_indices

                    final_indices.extend(selected)
            else:
                # If not matching paper exact counts, we might still want to limit
                # or just take all vulnerable ones.
                # For now, let's take all valid vulnerable samples.
                final_indices = valid_indices.tolist()

            final_indices = sorted(final_indices)
            logger.info(f"Selected {len(final_indices)} samples for final dataset.")

            # Now retrieve the source code for these indices
            # Handling huge string arrays in HDF5
            sources = []

            # Reading by index list from h5py dataset is not directly supported efficiently
            # like numpy array[list]. We might need to iterate or read chunks.
            # Optimized approach: read all if memory allows, or read in sorted order.

            logger.info("Reading source code... (this may take a moment)")
            raw_sources = f["functionSource"]

            # If dataset is < 100k, reading all to memory is fine.
            # VDISC is 1.2M. Strings can take GBs.
            # Let's read only what we need.

            for i in final_indices:
                sources.append(raw_sources[i].decode("utf-8"))  # Decode bytes to string

            # Create final DataFrame
            df_final = pd.DataFrame(
                {
                    "func": sources,
                    "target": df_labels.loc[final_indices, "target"].values,
                }
            )

            # Save
            output_file = output_dir / f"{split_name}_processed.pkl"
            df_final.to_pickle(output_file)
            logger.info(f"Saved processed data to {output_file}")

            # Verification statistics
            logger.info(
                f"Class distribution:\n{df_final['target'].value_counts().sort_index()}"
            )

    except Exception as e:
        logger.error(f"Error processing {filepath}: {e}")
        import traceback

        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Process Draper VDISC HDF5 files")
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Directory containing .hdf5 files"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save .pkl files"
    )
    parser.add_argument(
        "--match_paper",
        action="store_true",
        help="Downsample to match paper counts exactly",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Expected filenames
    files = {
        "train": "VDISC_train.hdf5",
        "val": "VDISC_validate.hdf5",
        "test": "VDISC_test.hdf5",
    }

    for split, filename in files.items():
        filepath = data_dir / filename
        if filepath.exists():
            process_file(filepath, split, output_dir, args.match_paper)
        else:
            logger.warning(
                f"File {filename} not found in {data_dir}. Skipping {split} split."
            )


if __name__ == "__main__":
    main()
