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
CONFIG="$SCRIPT_DIR/config.json"

# ---- Parse arguments ----
SKIP_PREDICT=false
PRED_DIR=""
PREDICT_ARGS=""
PP_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip_predict)
            SKIP_PREDICT=true
            shift ;;
        --pred_dir)
            PRED_DIR="$2"
            shift 2 ;;
        --no_preprocess)
            PREDICT_ARGS="$PREDICT_ARGS $1"
            shift ;;
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
