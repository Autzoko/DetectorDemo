# 3D ABUS Lesion Detection Inference

3D breast ultrasound (ABUS) lesion detection inference pipeline.

## Setup

```bash
# Clone the repo
git clone https://github.com/Autzoko/DetectorDemo.git
cd DetectorDemo

# Setup environment (creates conda env, installs dependencies, downloads model)
bash setup.sh

# Activate environment
conda activate detdemo
```

## Usage

### Step 1: Configure data path

Edit `config.json`, set `env.det_data` to your raw data root:

```bash
# Example: your data is at /scratch/user/data/nnDet
# Edit config.json -> "det_data": "/scratch/user/data/nnDet"
```

The data directory should contain:

```
<det_data>/
└── Task100_BreastABUS/
    └── raw_splitted/
        └── imagesTs/
            ├── case_00000_0000.nii.gz
            ├── case_00001_0000.nii.gz
            └── ...
```

### Step 2: Prepare data (if using raw NIfTI files)

```bash
python prepare_data.py --input /path/to/raw/nii_files
```

### Step 3: Run inference

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

### Step 4: Post-processing

```bash
python postprocess.py
```

### Full pipeline (one command)

```bash
bash run_pipeline.sh
bash run_pipeline.sh --test_data /path/to/nii_files
bash run_pipeline.sh --skip_predict    # post-process only
```

## Output

- `test_predictions/`: raw model predictions (pkl files)
- `results/predictions.csv`: filtered detections per case
- `results/summary.csv`: per-case summary
