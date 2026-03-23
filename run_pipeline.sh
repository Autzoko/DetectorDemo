#!/bin/bash
# ============================================================
# Inference Pipeline
#
# Usage:
#   bash run_pipeline.sh --data /path/to/data              # full pipeline
#   bash run_pipeline.sh --skip_predict                     # post-process only
#   bash run_pipeline.sh --data /path --no_preprocess       # skip preprocessing
#   bash run_pipeline.sh --test_data /path/to/nii_files     # custom test NIfTI
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ---- Fix CUDA/library issues (common on HPC) ----
# Remove CUDA stubs from LD_LIBRARY_PATH so runtime uses real CUDA drivers
if echo "$LD_LIBRARY_PATH" | grep -q "stubs"; then
    export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | sed 's|[^:]*stubs[^:]*:||g; s|:[^:]*stubs[^:]*||g; s|^[^:]*stubs[^:]*$||g')
fi
# Ensure PyTorch shared libraries are findable (needed for CUDA NMS extension)
_TORCH_LIB=$(python3 -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))" 2>/dev/null)
if [ -n "$_TORCH_LIB" ] && [ -d "$_TORCH_LIB" ]; then
    export LD_LIBRARY_PATH="$_TORCH_LIB:$LD_LIBRARY_PATH"
fi
# Auto-detect CUDA_HOME if not set
if [ -z "$CUDA_HOME" ]; then
    _NVCC_PATH=$(which nvcc 2>/dev/null)
    if [ -n "$_NVCC_PATH" ]; then
        export CUDA_HOME=$(dirname $(dirname "$_NVCC_PATH"))
    fi
fi

# ---- Parse arguments ----
SKIP_PREDICT=false
PREDICT_ARGS=""
PP_ARGS=""
PRED_DIR=""
DATA_PATH=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --data)
            DATA_PATH="$2"
            shift 2 ;;
        --skip_predict)
            SKIP_PREDICT=true
            shift ;;
        --no_preprocess)
            PREDICT_ARGS="$PREDICT_ARGS --no_preprocess"
            shift ;;
        --no_tta)
            PREDICT_ARGS="$PREDICT_ARGS --no_tta"
            shift ;;
        --case_id)
            PREDICT_ARGS="$PREDICT_ARGS --case_id $2"
            shift 2 ;;
        --test_data)
            PREDICT_ARGS="$PREDICT_ARGS --test_data $2"
            shift 2 ;;
        --pred_dir)
            PRED_DIR="$2"
            shift 2 ;;
        --iou_thresh|--output_dir|--min_score|--density_radius|--density_power|--cluster_iou|--top_k)
            PP_ARGS="$PP_ARGS $1 $2"
            shift 2 ;;
        *)
            PREDICT_ARGS="$PREDICT_ARGS $1"
            shift ;;
    esac
done

# ---- Auto-configure data path ----
if [ -n "$DATA_PATH" ]; then
    DATA_PATH="$(cd "$DATA_PATH" 2>/dev/null && pwd || echo "$DATA_PATH")"
    echo "Configuring data path: $DATA_PATH"
    python3 -c "
import json, sys
cfg_path = '$SCRIPT_DIR/config.json'
with open(cfg_path) as f:
    cfg = json.load(f)
cfg['env']['det_data'] = '$DATA_PATH'
with open(cfg_path, 'w') as f:
    json.dump(cfg, f, indent=4)
print('  config.json updated: det_data =', '$DATA_PATH')
"
    echo ""
fi

# ---- Step 1: Prediction ----
if [ "$SKIP_PREDICT" = false ]; then
    python3 "$SCRIPT_DIR/predict.py" --config "$SCRIPT_DIR/config.json" $PREDICT_ARGS

    if [ -z "$PRED_DIR" ]; then
        PRED_DIR="$SCRIPT_DIR/test_predictions"
    fi
else
    if [ -z "$PRED_DIR" ]; then
        PRED_DIR="$SCRIPT_DIR/test_predictions"
    fi
fi

# ---- Step 2: Post-processing ----
python3 "$SCRIPT_DIR/postprocess.py" \
    --config "$SCRIPT_DIR/config.json" \
    --pred_dir "$PRED_DIR" \
    $PP_ARGS
