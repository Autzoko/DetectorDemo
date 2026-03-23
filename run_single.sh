#!/bin/bash
# ============================================================
# Single-Case Inference
#
# Usage:
#   bash run_single.sh --input /path/to/scan.nii.gz
#   bash run_single.sh --input /path/to/scan.nii.gz --no_tta
#   bash run_single.sh --case_id case_00000 --no_tta
#   bash run_single.sh --case_id case_00000 --raw_data_dir /path/to/raw
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ---- Fix CUDA/library issues (common on HPC) ----
if echo "$LD_LIBRARY_PATH" | grep -q "stubs"; then
    export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | sed 's|[^:]*stubs[^:]*:||g; s|:[^:]*stubs[^:]*||g; s|^[^:]*stubs[^:]*$||g')
fi
_TORCH_LIB=$(python3 -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))" 2>/dev/null)
if [ -n "$_TORCH_LIB" ] && [ -d "$_TORCH_LIB" ]; then
    export LD_LIBRARY_PATH="$_TORCH_LIB:$LD_LIBRARY_PATH"
fi
if [ -z "$CUDA_HOME" ]; then
    _NVCC_PATH=$(which nvcc 2>/dev/null)
    if [ -n "$_NVCC_PATH" ]; then
        export CUDA_HOME=$(dirname $(dirname "$_NVCC_PATH"))
    fi
fi

# ---- Run ----
if [ $# -lt 1 ]; then
    echo "Usage:"
    echo "  bash run_single.sh --input /path/to/scan.nii.gz [--no_tta]"
    echo "  bash run_single.sh --case_id case_00000 [--no_tta] [--raw_data_dir /path]"
    exit 1
fi

python3 "$SCRIPT_DIR/run_single.py" "$@"
