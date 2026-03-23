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

Put your data at any path, then one command runs everything:

```bash
bash run_pipeline.sh --data /path/to/your/data
```

This will automatically update `config.json` and run the full pipeline (preprocessing + inference + post-processing).

### Data directory structure

```
/path/to/your/data/
└── Task100_BreastABUS/
    └── raw_splitted/
        └── imagesTs/
            ├── case_00000_0000.nii.gz
            ├── case_00001_0000.nii.gz
            └── ...
```

### If using raw NIfTI files (arbitrary filenames)

```bash
# Step 1: Convert to expected format
python prepare_data.py --input /path/to/raw/nii_files

# Step 2: Run pipeline
bash run_pipeline.sh --data /path/to/your/data
```

### Other options

```bash
# Skip preprocessing (data already preprocessed)
bash run_pipeline.sh --data /path/to/your/data --no_preprocess

# Only run post-processing (predictions already exist)
bash run_pipeline.sh --skip_predict

# Run steps separately
python predict.py --test_data /path/to/nii_files
python postprocess.py
```

## Output

- `results/predictions.csv`: filtered detections per case
- `results/summary.csv`: per-case summary
