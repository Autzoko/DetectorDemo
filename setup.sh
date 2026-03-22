#!/bin/bash
# ============================================================
# Setup Inference Environment
#
# Installs dependencies from the local copy included in this
# project folder. Works on any machine with conda.
#
# Usage:
#   bash setup.sh
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "============================================"
echo "  Inference Environment Setup"
echo "============================================"

# ---- Step 1: Create conda env ----
ENV_NAME="detdemo"

if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Conda env '$ENV_NAME' already exists."
    echo "To recreate: conda env remove -n $ENV_NAME && bash setup.sh"
else
    echo "Creating conda env: $ENV_NAME (Python 3.8)"
    conda create -n "$ENV_NAME" python=3.8 -y
fi

echo ""
echo "Activating conda env..."
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

# ---- Step 2: Install PyTorch ----
echo ""
echo "Installing PyTorch 1.11 + CUDA 11.3..."
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 \
    -f https://download.pytorch.org/whl/torch_stable.html

# ---- Step 3: Install detection framework ----
echo ""
if [ ! -d "$SCRIPT_DIR/nnDetection" ]; then
    echo "Cloning detection framework..."
    git clone https://github.com/MIC-DKFZ/nnDetection.git "$SCRIPT_DIR/nnDetection"
fi
echo "Installing detection framework from local source..."
cd "$SCRIPT_DIR/nnDetection"
pip install -e .

# ---- Step 4: Install additional dependencies ----
echo ""
echo "Installing additional dependencies..."
pip install nibabel loguru omegaconf hydra-core gdown

# ---- Step 5: Download model files ----
MODEL_DIR="$SCRIPT_DIR/models/Task100_BreastABUS/RetinaUNetV001_D3V001_3d/fold0__0"
if [ ! -f "$MODEL_DIR/model_best.ckpt" ]; then
    echo ""
    echo "Downloading model files..."
    mkdir -p "$MODEL_DIR"
    GDRIVE_ID="1tyrooafGsEBBanttL-Sxjs5YePWxrcTm"
    gdown "$GDRIVE_ID" -O "$SCRIPT_DIR/models/model_files.zip"
    unzip -o "$SCRIPT_DIR/models/model_files.zip" -d "$MODEL_DIR"
    rm -f "$SCRIPT_DIR/models/model_files.zip"
    echo "Model files extracted to: $MODEL_DIR"
else
    echo ""
    echo "Model files already exist at: $MODEL_DIR"
fi

echo ""
echo "============================================"
echo "  Setup Complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Update config.json:"
echo "     - Set 'det_data' to your raw data directory"
echo ""
echo "  2. Run:"
echo "     conda activate $ENV_NAME"
echo "     cd $SCRIPT_DIR"
echo "     python predict.py --test_data /path/to/nii_files   # inference"
echo "     python postprocess.py                                # post-process"
