#!/bin/bash
# ============================================================
# Complete Inference Pipeline
#
# Raw 3D data -> Preprocessing -> Model Inference -> Post-Processing
#
# All settings read from config.json.
#
# Usage:
#   bash run_all.sh                                   # full pipeline
#   bash run_all.sh --skip_predict                    # post-process only
#   bash run_all.sh --skip_predict --pred_dir /path   # post-process custom dir
#   bash run_all.sh --no_preprocess                   # skip preprocessing
#   bash run_all.sh --test_data /path/to/nii_files    # custom test data
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

# ---- Parse arguments ----
SKIP_PREDICT=false
PRED_DIR=""
PREDICT_ARGS=""
PP_ARGS=""
DATA_PATH=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --data)
            DATA_PATH="$2"
            shift 2 ;;
        --skip_predict)
            SKIP_PREDICT=true
            shift ;;
        --pred_dir)
            PRED_DIR="$2"
            shift 2 ;;
        --no_preprocess)
            PREDICT_ARGS="$PREDICT_ARGS $1"
            shift ;;
        --no_tta)
            PREDICT_ARGS="$PREDICT_ARGS --no_tta"
            shift ;;
        --case_id)
            PREDICT_ARGS="$PREDICT_ARGS --case_id $2"
            shift 2 ;;
        --test_data)
            PREDICT_ARGS="$PREDICT_ARGS $1 $2"
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
import json
cfg_path = '$CONFIG'
with open(cfg_path) as f:
    cfg = json.load(f)
cfg['env']['det_data'] = '$DATA_PATH'
with open(cfg_path, 'w') as f:
    json.dump(cfg, f, indent=4)
print('  config.json updated: det_data =', '$DATA_PATH')
"
    echo ""
fi

echo "============================================"
echo "  Inference Pipeline"
echo "============================================"
echo ""

# ---- Step 1: Prediction ----
if [ "$SKIP_PREDICT" = false ]; then
    echo ">>> Step 1: Model Prediction"
    echo "-------------------------------------------"
    python3 "$SCRIPT_DIR/predict.py" --config "$CONFIG" $PREDICT_ARGS
    echo ""

    if [ -z "$PRED_DIR" ]; then
        PRED_DIR=$(python3 -c "import json; print(json.load(open('$CONFIG'))['paths']['predictions_dir'])")
        # Resolve relative path
        [[ "$PRED_DIR" != /* ]] && PRED_DIR="$SCRIPT_DIR/$PRED_DIR"
    fi
else
    echo ">>> Step 1: SKIPPED (--skip_predict)"
    if [ -z "$PRED_DIR" ]; then
        PRED_DIR=$(python3 -c "import json; print(json.load(open('$CONFIG'))['paths']['predictions_dir'])")
        [[ "$PRED_DIR" != /* ]] && PRED_DIR="$SCRIPT_DIR/$PRED_DIR"
    fi
    echo "  Using predictions from: $PRED_DIR"
    echo ""
fi

# ---- Step 2: Post-processing ----
echo ">>> Step 2: Post-Processing"
echo "-------------------------------------------"
python3 "$SCRIPT_DIR/postprocess.py" \
    --config "$CONFIG" \
    --pred_dir "$PRED_DIR" \
    $PP_ARGS

echo ""
echo "============================================"
echo "  Pipeline Complete!"
echo "  Results: $SCRIPT_DIR/results/"
echo "============================================"
