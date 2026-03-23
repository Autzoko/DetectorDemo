# 3D ABUS Lesion Detection Inference

3D breast ultrasound (ABUS) lesion detection inference pipeline.

## Setup

```bash
git clone https://github.com/Autzoko/DetectorDemo.git
cd DetectorDemo
bash setup.sh
conda activate detdemo
```

## Usage

### From raw NIfTI files (recommended)

```bash
# Step 1: Convert data (auto-configures paths)
python prepare_data.py --input /path/to/raw/nii_files

# Step 2: Run inference + post-processing
bash run_pipeline.sh
```

### If data is already in standard format

```bash
bash run_pipeline.sh --data /path/to/your/data
```

Data directory structure:

```
/path/to/your/data/
└── Task100_BreastABUS/
    └── raw_splitted/
        └── imagesTs/
            ├── case_00000_0000.nii.gz
            └── ...
```

### Other options

```bash
# Skip preprocessing (data already preprocessed)
bash run_pipeline.sh --no_preprocess

# Only run post-processing (predictions already exist)
bash run_pipeline.sh --skip_predict

# Run steps separately
python predict.py
python postprocess.py
```

## Output

- `results/predictions.csv`: filtered detections per case
- `results/summary.csv`: per-case summary
