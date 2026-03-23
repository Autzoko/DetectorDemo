#!/bin/bash
# ============================================================
# Step 1: Run Inference
#
# Reads task/model/fold/paths from config.json automatically.
#
# Prerequisites:
#   - Conda env activated (conda activate detdemo)
#   - Model files in place (config.yaml, plan_inference.pkl, *.ckpt)
#   - Raw test data (nii.gz files)
#
# Usage:
#   bash run_predict.sh                                    # full pipeline
#   bash run_predict.sh --no_preprocess                    # skip preprocessing
#   bash run_predict.sh --test_data /path/to/nii_files     # custom test data
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
CONFIG="$SCRIPT_DIR/config.json"

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

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: config.json not found at $CONFIG"
    exit 1
fi

# ---- Read config.json for defaults ----
TASK="${TASK:-$(python3 -c "import json; print(json.load(open('$CONFIG'))['predict']['task'])")}"
MODEL="${MODEL:-$(python3 -c "import json; print(json.load(open('$CONFIG'))['predict']['model'])")}"
FOLD="${FOLD:-$(python3 -c "import json; print(json.load(open('$CONFIG'))['predict']['fold'])")}"

export det_data="${det_data:-$(python3 -c "import json; print(json.load(open('$CONFIG'))['env']['det_data'])")}"
export det_models="${det_models:-$(python3 -c "import json; c=json.load(open('$CONFIG'))['env']['det_models']; import os; print(os.path.abspath(c) if not os.path.isabs(c) else c)")}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export det_num_threads="${det_num_threads:-6}"


# ---- Validate model directory ----
TRAINING_DIR="$det_models/$TASK/$MODEL"
if [ ! -d "$TRAINING_DIR" ]; then
    echo "ERROR: Model directory not found: $TRAINING_DIR"
    echo ""
    echo "Expected:"
    echo "  $TRAINING_DIR/fold${FOLD}__0/"
    echo "    config.yaml  plan_inference.pkl  model_best.ckpt"
    echo ""
    echo "Download from HPC or update config.json."
    exit 1
fi

# Find fold directory
if [ "$FOLD" = "-1" ]; then
    FOLD_DIR="$TRAINING_DIR/consolidated"
else
    FOLD_DIR=$(ls -d "$TRAINING_DIR/fold${FOLD}"__* 2>/dev/null | sort -t_ -k2 -n | tail -1)
fi

if [ -z "$FOLD_DIR" ] || [ ! -d "$FOLD_DIR" ]; then
    echo "ERROR: Fold directory not found for fold=$FOLD"
    ls -d "$TRAINING_DIR"/*/ 2>/dev/null || echo "  (none)"
    exit 1
fi

# Check required files
MISSING=0
for f in config.yaml plan_inference.pkl; do
    [ ! -f "$FOLD_DIR/$f" ] && echo "MISSING: $FOLD_DIR/$f" && MISSING=1
done

CKPT_COUNT=$(ls "$FOLD_DIR"/*.ckpt 2>/dev/null | wc -l | tr -d ' ')
[ "$CKPT_COUNT" = "0" ] && echo "MISSING: No .ckpt files in $FOLD_DIR/" && MISSING=1

if [ "$MISSING" = "1" ]; then
    echo ""
    echo "If plan_inference.pkl is missing:"
    echo "  python $SCRIPT_DIR/generate_plan_inference.py --plan $FOLD_DIR/plan.pkl"
    exit 1
fi

# ---- Run prediction ----
python3 "$SCRIPT_DIR/predict.py" --config "$CONFIG" "$@"
