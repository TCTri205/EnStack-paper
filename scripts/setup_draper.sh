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
