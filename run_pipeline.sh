#!/bin/bash
# ============================================================
# nnDetection Inference Pipeline (via predict.py)
#
# Raw 3D ABUS data -> Preprocessing -> Model Prediction -> Post-Processing
#
# Directory structure:
#   nndet_patchcls_inference/
#   ├── nnDetection/               <- full nnDetection source (pip install -e)
#   ├── models/<TASK>/<MODEL>/fold0__0/
#   │   ├── config.yaml            <- download from HPC
#   │   ├── plan_inference.pkl     <- download from HPC (or generate)
#   │   └── model_best.ckpt       <- download from HPC
#   ├── test_predictions/          <- prediction output (pkl files)
#   ├── results/                   <- post-processing output (CSV files)
#   ├── predict.py                 <- Step 1: nnDetection inference
#   ├── postprocess.py             <- Step 2: post-processing
#   └── config.json                <- all configuration
#
# Usage:
#   bash run_pipeline.sh                                  # full pipeline
#   bash run_pipeline.sh --skip_predict                   # only post-process
#   bash run_pipeline.sh --no_preprocess                  # skip preprocessing
#   bash run_pipeline.sh --test_data /path/to/nii_files   # custom test data
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ---- Parse arguments ----
SKIP_PREDICT=false
PREDICT_ARGS=""
PP_ARGS=""
PRED_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip_predict)
            SKIP_PREDICT=true
            shift ;;
        --no_preprocess)
            PREDICT_ARGS="$PREDICT_ARGS --no_preprocess"
            shift ;;
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

echo "============================================================"
echo "  nnDetection Inference Pipeline"
echo "============================================================"
echo ""

# ---- Step 1: Prediction ----
if [ "$SKIP_PREDICT" = false ]; then
    echo ">>> Step 1: nnDetection Prediction"
    echo "------------------------------------------------------------"
    python3 "$SCRIPT_DIR/predict.py" --config "$SCRIPT_DIR/config.json" $PREDICT_ARGS
    echo ""

    # Use default predictions dir if not specified
    if [ -z "$PRED_DIR" ]; then
        PRED_DIR="$SCRIPT_DIR/test_predictions"
    fi
else
    echo ">>> Step 1: SKIPPED (--skip_predict)"
    if [ -z "$PRED_DIR" ]; then
        PRED_DIR="$SCRIPT_DIR/test_predictions"
    fi
    echo "  Using predictions from: $PRED_DIR"
    echo ""
fi

# ---- Step 2: Post-processing ----
echo ">>> Step 2: Post-Processing"
echo "------------------------------------------------------------"
python3 "$SCRIPT_DIR/postprocess.py" \
    --config "$SCRIPT_DIR/config.json" \
    --pred_dir "$PRED_DIR" \
    $PP_ARGS

echo ""
echo "============================================================"
echo "  Pipeline Complete!"
echo "  Results: $SCRIPT_DIR/results/"
echo "============================================================"
