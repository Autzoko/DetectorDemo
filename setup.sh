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

# ---- Step 3: Install detection framework from local copy ----
echo ""
echo "Installing detection framework from local source..."
cd "$SCRIPT_DIR/nnDetection"
pip install -e .

# ---- Step 4: Install additional dependencies ----
echo ""
echo "Installing additional dependencies..."
pip install nibabel loguru omegaconf hydra-core

echo ""
echo "============================================"
echo "  Setup Complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Update config.json:"
echo "     - Set 'det_data' to your raw data directory"
echo "     - Set 'predict.task' and 'predict.model' to match your model"
echo ""
echo "  2. Place model files (from HPC) at the path matching config.json:"
echo "     models/<task>/<model>/fold0__0/"
echo "       config.yaml"
echo "       plan_inference.pkl (or plan.pkl + run: python generate_plan_inference.py)"
echo "       model_best.ckpt"
echo ""
echo "  3. Run:"
echo "     conda activate $ENV_NAME"
echo "     cd $SCRIPT_DIR"
echo "     python predict.py --test_data /path/to/nii_files   # inference"
echo "     python postprocess.py                                # post-process"
