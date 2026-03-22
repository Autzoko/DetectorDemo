# 3D ABUS Lesion Detection Inference

3D breast ultrasound (ABUS) lesion detection inference pipeline.

## Environment Setup

```bash
bash setup.sh
conda activate detdemo
```

## Quick Start

### 1. Prepare Data

Convert raw NIfTI files to the expected format:

```bash
python prepare_data.py --input /path/to/raw/nii_files
```

### 2. Run Inference

```bash
python predict.py
```

Or specify custom test data:

```bash
python predict.py --test_data /path/to/nii_files
```

Skip preprocessing (if already done):

```bash
python predict.py --no_preprocess
```

### 3. Post-Processing

```bash
python postprocess.py
```

Specify custom prediction directory:

```bash
python postprocess.py --pred_dir /path/to/predictions
```

### Full Pipeline (One Command)

```bash
bash run_pipeline.sh
bash run_pipeline.sh --test_data /path/to/nii_files
bash run_pipeline.sh --skip_predict    # post-process only
```

## Configuration

All settings are in `config.json`:

- `predict.task` / `predict.model` / `predict.fold`: model selection
- `env.det_data`: raw data root directory (**must be changed to your path**)
- `env.det_models`: model weights directory (default: `./models`)
- `density_wbc.*`: post-processing parameters

## Model Files

Place model files at:

```
models/<task>/<model>/fold0__0/
  config.yaml
  plan_inference.pkl
  model_best.ckpt
```

## Output

- `test_predictions/`: raw model predictions (pkl files)
- `results/predictions.csv`: filtered detections per case
- `results/summary.csv`: per-case summary
