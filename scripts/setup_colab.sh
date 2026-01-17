#!/bin/bash
# Setup script for Google Colab environment

set -e

echo "====================================="
echo "EnStack Setup Script for Google Colab"
echo "====================================="

# Check if running in Colab
if [ ! -d "/content" ]; then
    echo "Warning: This script is designed for Google Colab"
    echo "Running outside Colab may cause issues"
fi

# Install dependencies
echo ""
echo "[1/4] Installing dependencies..."
pip install -q torch>=1.10.0
pip install -q transformers>=4.20.0
pip install -q scikit-learn>=1.0
pip install -q pandas>=1.3
pip install -q tree-sitter>=0.20
pip install -q tqdm pyyaml
pip install -q pytest black ruff mypy
pip install -q matplotlib seaborn tensorboard xgboost psutil

echo "✓ Dependencies installed"

# Check CUDA availability
echo ""
echo "[2/4] Checking GPU availability..."
python3 -c "import torch; print('✓ CUDA available:', torch.cuda.is_available()); print('  Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# Create necessary directories
echo ""
echo "[3/4] Creating directories..."
mkdir -p /content/drive/MyDrive/EnStack_Data/checkpoints
mkdir -p /content/drive/MyDrive/EnStack_Data/logs
echo "✓ Directories created"

# Verify installation
echo ""
echo "[4/4] Verifying installation..."
python3 -c "
import torch
import transformers
import sklearn
import pandas
import yaml
print('✓ All packages imported successfully')
print(f'  PyTorch version: {torch.__version__}')
print(f'  Transformers version: {transformers.__version__}')
print(f'  Scikit-learn version: {sklearn.__version__}')
print(f'  Pandas version: {pandas.__version__}')
"

echo ""
echo "====================================="
echo "Setup completed successfully!"
echo "====================================="
echo ""
echo "Next steps:"
echo "1. Upload your data to /content/drive/MyDrive/EnStack_Data/"
echo "2. Update configs/config.yaml with your data paths"
echo "3. Run the main_pipeline.ipynb notebook"
echo ""
