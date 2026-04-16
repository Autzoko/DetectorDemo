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
    git clone https://github.com/Autzoko/Detection.git "$SCRIPT_DIR/nnDetection"
fi
echo "Installing detection framework from local source..."
echo ""
echo "NOTE: For GPU support, run this setup on a GPU node (e.g., via srun --gres=gpu:1)."
echo "      If installed on a login node, CUDA extensions will not compile."
echo ""

# Auto-detect CUDA_HOME if not set
if [ -z "$CUDA_HOME" ]; then
    _NVCC_PATH=$(which nvcc 2>/dev/null)
    if [ -n "$_NVCC_PATH" ]; then
        export CUDA_HOME=$(dirname $(dirname "$_NVCC_PATH"))
        echo "  Auto-detected CUDA_HOME=$CUDA_HOME"
    fi
fi

# Auto-detect GPU architecture for CUDA compilation
if [ -z "$TORCH_CUDA_ARCH_LIST" ] && python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    _GPU_ARCH=$(python3 -c "
import torch
cap = torch.cuda.get_device_capability(0)
print(f'{cap[0]}.{cap[1]}')
" 2>/dev/null)
    if [ -n "$_GPU_ARCH" ]; then
        export TORCH_CUDA_ARCH_LIST="$_GPU_ARCH"
        echo "  Auto-detected TORCH_CUDA_ARCH_LIST=$_GPU_ARCH"
    fi
fi

cd "$SCRIPT_DIR/nnDetection"
rm -rf build/ nndet/_C*.so  # Clean old builds
pip install -e .

# Patch: ensure NMS falls back to CPU if CUDA extension not available
NMS_FILE="$SCRIPT_DIR/nnDetection/nndet/core/boxes/nms.py"
if grep -q "if boxes.is_cuda:" "$NMS_FILE" 2>/dev/null; then
    sed -i.bak 's/if boxes.is_cuda:/if boxes.is_cuda and nms_gpu is not None:/' "$NMS_FILE"
    rm -f "${NMS_FILE}.bak"
    echo "Applied NMS CPU fallback patch."
fi

# ---- Step 4: Install additional dependencies ----
echo ""
echo "Installing additional dependencies..."
pip install nibabel loguru omegaconf hydra-core gdown

# ---- Step 5: Download model files ----
MODEL_DIR="$SCRIPT_DIR/models/Task101_BreastBIRADS/RetinaUNetV001_D3V001_3d/fold0__0"
if [ ! -f "$MODEL_DIR/model_best.ckpt" ]; then
    echo ""
    echo "Downloading model files..."
    mkdir -p "$MODEL_DIR"
    GDRIVE_URL="https://drive.google.com/uc?id=1tyrooafGsEBBanttL-Sxjs5YePWxrcTm"
    TMP_ZIP="$SCRIPT_DIR/models/_model_files.zip"
    gdown "$GDRIVE_URL" -O "$TMP_ZIP"
    # Unzip to temp dir, then move files to MODEL_DIR (handles nested dirs)
    TMP_UNZIP="$SCRIPT_DIR/models/_tmp_unzip"
    mkdir -p "$TMP_UNZIP"
    unzip -o "$TMP_ZIP" -d "$TMP_UNZIP"
    # Find and move model files (config.yaml, *.pkl, *.ckpt)
    find "$TMP_UNZIP" -type f \( -name "*.yaml" -o -name "*.pkl" -o -name "*.ckpt" \) \
        -exec mv {} "$MODEL_DIR/" \;
    rm -rf "$TMP_ZIP" "$TMP_UNZIP"
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
echo "  conda activate $ENV_NAME"
echo "  cd $SCRIPT_DIR"
echo ""
echo "  # From raw NIfTI files:"
echo "  python prepare_data.py --input /path/to/raw/nii_files"
echo "  bash run_pipeline.sh"
echo ""
echo "  # Or specify data path directly:"
echo "  bash run_pipeline.sh --data /path/to/data"
