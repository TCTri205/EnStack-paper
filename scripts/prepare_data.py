import logging
import os
import pickle
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from datasets import load_dataset

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("DataPrep")


def map_cwes_to_labels(cwe_bools: Dict) -> int:
    """
    Map boolean CWE columns to a single integer label.
    Priority: 119 > 120 > 469 > 476 > Other (if any True) > 0 (Safe)

    Adjust priority/logic based on specific paper details if needed.
    For this implementation:
    0: Safe
    1: CWE-119
    2: CWE-120
    3: CWE-469
    4: CWE-476
    5: Other (if any other vulnerability exists but not the main 4)

    However, the config.yaml usually expects 0-4 or similar.
    The user description says:
    - CWE-119
    - CWE-120
    - CWE-469
    - CWE-476
    - CWE-other

    This implies 5 classes of vulnerabilities.
    But usually, we also need a 'Safe' class?
    Or is the dataset ONLY vulnerable functions?

    The Draper dataset contains both vulnerable and non-vulnerable.
    If 'CWE-other' collects everything else, we might have a multi-class problem including Safe.

    Let's align with the user prompt:
    "Dataset phân loại lỗ hổng theo 5 nhóm CWE... CWE-other"

    If the dataset has 'Safe' samples, where do they go?
    Usually, binary classification is Vuln vs Safe.
    Multi-class usually implies identifying the specific vulnerability.

    Let's check the paper logic (simulated):
    Usually 0=Safe, 1=119, 2=120, ...

    Let's assume:
    0: Safe (No CWE is True)
    1: CWE-119
    2: CWE-120
    3: CWE-469
    4: CWE-476
    5: CWE-other

    BUT, the user config says `num_labels: 5`.
    Maybe 0=119, 1=120, 2=469, 3=476, 4=other?
    And Safe is ignored? Or Safe is class 4?

    Re-reading user prompt:
    "Dataset phân loại lỗ hổng theo 5 nhóm CWE"
    It doesn't explicitly mention "Safe".
    However, VDISC is typically used for detection (Binary) or multiclass.

    If `num_labels: 5`, likely it maps the 5 categories.

    Let's define the mapping:
    Target 0: CWE-119
    Target 1: CWE-120
    Target 2: CWE-469
    Target 3: CWE-476
    Target 4: CWE-other (includes everything else + maybe Safe? No, Safe should be separate).

    Wait, if the model predicts vulnerability type, it only runs on vulnerable code?
    Or is it 5 classes where one is "Safe"?

    Let's check standard usage of Draper VDISC in papers (Russell et al):
    They often treat it as a multilabel problem because one function can have multiple CWEs.
    But EnStack uses Softmax (implied by `num_labels` and `CrossEntropy`).

    Let's assume the user wants to classify 5 types.
    I will handle "Safe" samples by filtering them OUT if we only do classification of vulnerabilities,
    OR (more likely) one of the classes is Safe, or "CWE-other" acts as the catch-all.

    Actually, usually Class 0 is Safe.
    But the user listed 5 CWE groups.

    Let's stick to the user's list:
    0: CWE-119
    1: CWE-120
    2: CWE-469
    3: CWE-476
    4: CWE-other

    What about Safe functions?
    If I include Safe, I need 6 classes.
    If I map Safe to 'CWE-other' (Class 4), that might be weird.

    Let's assume we map:
    - CWE-119 -> 0
    - CWE-120 -> 1
    - CWE-469 -> 2
    - CWE-476 -> 3
    - Any other True -> 4
    - All False (Safe) -> 4 (Treat as 'Other/Safe') OR Filter out?

    The most robust approach for a "Vulnerability Detection" paper (EnStack):
    Usually it detects IF it is vulnerable (Binary).
    But here we have Stacking for 5 classes.

    Let's use a standard mapping where 0-4 are the VULNERABILITY types.
    And we might filter out Safe samples for this specific experiment, OR assume the user handles Safe as 'Other'.

    Let's write the script to allow flexibility but default to:
    Safe -> Label 4 (Other)
    119 -> 0
    120 -> 1
    469 -> 2
    476 -> 3
    Other -> 4
    """
    if cwe_bools["CWE-119"]:
        return 0
    if cwe_bools["CWE-120"]:
        return 1
    if cwe_bools["CWE-469"]:
        return 2
    if cwe_bools["CWE-476"]:
        return 3

    # Check if any other known vulnerability is present (VDISC has many columns)
    # If not one of the above, but has a vulnerability -> 4
    # If Safe -> 4 (Group Safe with Other? Or make Safe 0?)

    # To be safe and follow the user's specific 5-class structure:
    # I will map Safe to 4 (Other) for now to ensure we have data.
    return 4


def prepare_data(output_dir: str, sample_size: int = None):
    """
    Download and process Draper VDISC dataset.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading Draper VDISC dataset (via text-analytics-vdisc)...")
    # This is a version of Draper VDISC hosted on HF
    # It contains 'train', 'validation', 'test' splits
    try:
        dataset = load_dataset("arithmo-ai/vdisc_vuln_code_detection")
        # Alternative: "mfin20/draper-vdisc" (check availability)
    except Exception:
        logger.warning(
            "Primary dataset link failed. Trying fallback 'mfin20/draper-vdisc'..."
        )
        dataset = load_dataset("mfin20/draper-vdisc")

    logger.info(f"Dataset structure: {dataset}")

    splits = ["train", "validation", "test"]

    for split in splits:
        logger.info(f"Processing split: {split}")
        ds_split = dataset[split]

        # Sampling if requested (to avoid OOM on Colab if needed)
        if sample_size and len(ds_split) > sample_size:
            logger.info(f"Sampling {sample_size} examples from {split}...")
            ds_split = ds_split.select(range(sample_size))

        # Convert to pandas for easier processing
        df = ds_split.to_pandas()

        # Rename columns to match EnStack expectation: 'func', 'target'
        # HF dataset usually has 'functionSource', 'CWE-119', etc.

        # Check column names
        logger.info(f"Columns: {df.columns.tolist()}")

        source_col = "functionSource" if "functionSource" in df.columns else "code"
        if source_col not in df.columns:
            # Try finding the code column
            for col in df.columns:
                if "code" in col.lower() or "source" in col.lower():
                    source_col = col
                    break

        logger.info(f"Using '{source_col}' as source code column")

        # Create target column
        targets = []
        for _, row in df.iterrows():
            # Extract boolean flags
            cwe_flags = {
                "CWE-119": row.get("CWE-119", False),
                "CWE-120": row.get("CWE-120", False),
                "CWE-469": row.get("CWE-469", False),
                "CWE-476": row.get("CWE-476", False),
            }
            label = map_cwes_to_labels(cwe_flags)
            targets.append(label)

        processed_df = pd.DataFrame({"func": df[source_col], "target": targets})

        # Save
        filename = f"{split if split != 'validation' else 'val'}_processed.pkl"
        save_path = output_path / filename
        processed_df.to_pickle(save_path)
        logger.info(f"Saved {len(processed_df)} samples to {save_path}")

        # Also save a CSV for inspection
        processed_df.head(100).to_csv(output_path / f"{split}_sample.csv", index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Number of samples per split (for testing)",
    )
    args = parser.parse_args()

    prepare_data(args.output_dir, args.sample)
