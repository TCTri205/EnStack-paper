"""
Setup script for Draper VDISC dataset.
This script helps downloading the dataset and preparing it for EnStack.
"""

import logging
import os
from pathlib import Path
import subprocess
import sys

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("DraperSetup")


def print_setup_guide():
    """Print step-by-step setup instructions."""
    logger.info("\n" + "=" * 70)
    logger.info("🚀 DRAPER VDISC DATASET SETUP GUIDE")
    logger.info("=" * 70)

    logger.info("\n📋 Step 1: Download the dataset files")
    logger.info("--------------------------------------------------")
    logger.info("Download the following 3 files from the official repository:")
    logger.info("https://osf.io/d45bw/")
    logger.info("\nRequired files:")
    logger.info("  1. VDISC_train.hdf5    (~1.63 GB)")
    logger.info("  2. VDISC_validate.hdf5 (~207 MB)")
    logger.info("  3. VDISC_test.hdf5     (~208 MB)")

    logger.info("\n💡 Quick download command (if you have the direct link):")
    logger.info("  mkdir -p data/raw_data")
    logger.info("  cd data/raw_data")
    logger.info("  wget <link-to-train>")
    logger.info("  wget <link-to-validate>")
    logger.info("  wget <link-to-test>")

    logger.info("\n📂 Step 2: Upload to your environment")
    logger.info("--------------------------------------------------")
    logger.info("Place the downloaded files in one of these locations:")
    logger.info("  - /content/drive/MyDrive/EnStack_Data/raw_data/ (for Colab)")
    logger.info("  - ./data/raw_data/ (for local machine)")

    logger.info("\n⚙️  Step 3: Run data processing")
    logger.info("--------------------------------------------------")
    logger.info("Execute the following command to process the HDF5 files:")
    logger.info("\n  python scripts/prepare_data.py \\")
    logger.info("      --mode draper \\")
    logger.info("      --draper_dir /content/drive/MyDrive/EnStack_Data/raw_data \\")
    logger.info("      --output_dir /content/drive/MyDrive/EnStack_Data \\")
    logger.info("      --match_paper")

    logger.info("\n🚀 Step 4: Start training")
    logger.info("--------------------------------------------------")
    logger.info("After processing completes, run your training:")
    logger.info("  python scripts/train.py")

    logger.info("\n" + "=" * 70)
    logger.info("ℹ️  Note: The --match_paper flag will downsample the dataset")
    logger.info("to match the exact distribution used in the EnStack paper (Table I).")
    logger.info("=" * 70 + "\n")


def download_sample():
    """
    Attempt to download sample data (if available).
    Note: Full Draper dataset requires manual download due to size.
    """
    logger.info("Attempting to download sample data...")

    # Since full download requires manual access, we'll just guide the user
    logger.warning("⚠️  The Draper VDISC dataset is too large for automatic download.")
    logger.info("Please follow the manual download instructions above.")

    return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Setup Draper VDISC dataset for EnStack"
    )
    parser.add_argument("--download", action="store_true", help="Try to download data")
    parser.add_argument("--guide", action="store_true", help="Print setup guide")

    args = parser.parse_args()

    if args.guide or len(sys.argv) == 1:
        print_setup_guide()

    if args.download:
        download_sample()
