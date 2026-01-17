#!/bin/bash

# Quick setup script for Draper VDISC Dataset on Colab

echo "=============================================================="
echo "  🚀 EnStack - Draper VDISC Dataset Setup"
echo "=============================================================="
echo ""

# Check if running in Colab
if [ ! -d "/content/drive" ]; then
    echo "⚠️  Warning: Not running in Google Colab?"
    echo "This script is optimized for Colab environment."
    echo ""
fi

# Create directories
echo "📁 Creating necessary directories..."
mkdir -p /content/drive/MyDrive/EnStack_Data/raw_data
mkdir -p /content/drive/MyDrive/EnStack_Data/checkpoints

# Check for HDF5 files
DRAPER_DIR="/content/drive/MyDrive/EnStack_Data/raw_data"
echo ""
echo "🔍 Checking for data in: $DRAPER_DIR"
echo ""

FILE_COUNT=$(find "$DRAPER_DIR" -name "*.hdf5" 2>/dev/null | wc -l)

if [ "$FILE_COUNT" -eq 0 ]; then
    echo "⚠️  No HDF5 files found locally. Downloading from OSF..."
    echo ""
    
    cd "$DRAPER_DIR" || exit

    echo "⬇️  Downloading VDISC_train.hdf5 (~862 MB)..."
    wget -O VDISC_train.hdf5 "https://osf.io/download/6fexn/" --show-progress

    echo "⬇️  Downloading VDISC_validate.hdf5 (~108 MB)..."
    wget -O VDISC_validate.hdf5 "https://osf.io/download/43mzd/" --show-progress

    echo "⬇️  Downloading VDISC_test.hdf5 (~107 MB)..."
    wget -O VDISC_test.hdf5 "https://osf.io/download/f9t6z/" --show-progress
    
    cd - > /dev/null || exit
    echo ""
    echo "✅ Download complete!"
else
    echo "✅ Found $FILE_COUNT HDF5 file(s):"
    ls -lh "$DRAPER_DIR"/*.hdf5
    echo ""
fi

# Check if we need to clean up old "Dummy" data
# Logic: If processed files exist, BUT raw HDF5 files are missing, 
# it implies the previous run was using Dummy/Synthetic data.
PROCESSED_EXIST=$(ls /content/drive/MyDrive/EnStack_Data/*_processed.pkl 2>/dev/null | wc -l)
RAW_EXIST=$(find "$DRAPER_DIR" -name "*.hdf5" 2>/dev/null | wc -l)

if [ "$PROCESSED_EXIST" -ge 3 ] && [ "$RAW_EXIST" -eq 0 ]; then
    echo "⚠️  DETECTED POTENTIAL DUMMY DATA!"
    echo "   Found processed files but NO raw Draper files."
    echo "   This suggests previous runs used synthetic data."
    echo ""
    echo "🧹 AUTOMATIC CLEANUP ACTIVATED:"
    echo "   - Deleting old processed files (*.pkl)..."
    rm /content/drive/MyDrive/EnStack_Data/*_processed.pkl
    
    echo "   - Deleting old checkpoints (trained on dummy data)..."
    rm -rf /content/drive/MyDrive/EnStack_Data/checkpoints
    
    echo "✅ Cleanup complete. Proceeding with fresh Draper setup..."
    echo ""
    PROCESSED_COUNT=0
else
    # Check if processed files already exist to skip processing
    PROCESSED_COUNT=$(ls /content/drive/MyDrive/EnStack_Data/*_processed.pkl 2>/dev/null | wc -l)
fi

if [ "$PROCESSED_COUNT" -eq 3 ] && [ "$RAW_EXIST" -gt 0 ]; then
    echo "✅ Processed data files found! Skipping processing step."
    echo "   (Remove files in /content/drive/MyDrive/EnStack_Data/ if you want to re-process)"
else
    # Run processing
    echo "=============================================================="
    echo "  ⚙️  Processing Draper VDISC Data"
    echo "=============================================================="
    echo ""

    python scripts/prepare_data.py \
        --mode draper \
        --draper_dir "$DRAPER_DIR" \
        --output_dir "/content/drive/MyDrive/EnStack_Data" \
        --match_paper
fi

# Check results
echo ""
echo "=============================================================="
if [ $? -eq 0 ]; then
    echo "  ✅ Data processing completed successfully!"
    echo ""
    echo "📊 VERIFICATION: Class distribution should match Paper Table I:"
    echo "--------------------------------------------------------------"
    echo "  Label 0 (CWE-119): Memory-related"
    echo "  Label 1 (CWE-120): Buffer Overflow"
    echo "  Label 2 (CWE-469): Integer Overflow"
    echo "  Label 3 (CWE-476): NULL Pointer"
    echo "  Label 4 (Other)  : Miscellaneous"
    echo "--------------------------------------------------------------"
    echo ""
    echo "Next steps:"
    echo "  1. Check files on Drive:"
    ls -lh /content/drive/MyDrive/EnStack_Data/*_processed.pkl
    echo ""
    echo "  2. Run the Training cell in the notebook."
else
    echo "  ❌ Data processing failed."
    echo "  Please check the error messages above."
fi
echo "=============================================================="
