# 3D ABUS Lesion Detection Inference

3D breast ultrasound (ABUS) lesion detection inference pipeline.

## Setup

**Important**: Run setup on a GPU node for CUDA NMS compilation.

```bash
# On HPC, request a GPU node first:
srun --gres=gpu:1 --mem=32G --time=01:00:00 --pty /bin/bash

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
bash run_pipeline.sh --no_tta          # disable TTA (8x faster)
```

### Single-case inference

```bash
# From a raw NIfTI file
bash run_single.sh --input /path/to/scan.nii.gz --no_tta

# From an already preprocessed case
bash run_single.sh --case_id case_00000 --no_tta
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

# Run a specific case from batch data
bash run_pipeline.sh --case_id case_00005 --no_tta

# Run steps separately
python predict.py --no_tta
python postprocess.py
```

## Output

- `results/predictions.csv`: filtered detections per case
- `results/summary.csv`: per-case summary

## Scripts

| Script | Purpose |
|--------|---------|
| `setup.sh` | Environment setup (conda, PyTorch, nnDetection, model download) |
| `prepare_data.py` | Convert raw NIfTI files to standard format |
| `run_pipeline.sh` | Full pipeline: predict + post-process (batch) |
| `run_single.sh` | Single-case inference (raw NIfTI or preprocessed) |
| `predict.py` | Model inference only |
| `postprocess.py` | Post-processing only |
| `viewer.py` | 3D visualization viewer (PyQt5) |

## Note on nnDetection

This project depends on [nnDetection](https://github.com/MIC-DKFZ/nnDetection) for model architecture and inference. The nnDetection source code is **not** included in this repository — it is automatically cloned and installed during setup (`bash setup.sh`). For training, architecture details, or further customization, please refer to the [nnDetection repository](https://github.com/MIC-DKFZ/nnDetection).

## Acknowledgements

This project builds upon [nnDetection](https://github.com/MIC-DKFZ/nnDetection), a self-configuring framework for medical object detection developed by the Division of Medical Image Computing at the German Cancer Research Center (DKFZ). We gratefully acknowledge their work in providing the foundational detection architecture used in this inference pipeline.

## Citation

If you use this project, please cite the underlying nnDetection framework:

```bibtex
@inproceedings{Baumgartner2021nnDetection,
    title     = {nnDetection: A Self-configuring Method for Medical Object Detection},
    author    = {Baumgartner, Michael and J{\"a}ger, Paul F. and Isensee, Fabian and Maier-Hein, Klaus H.},
    booktitle = {Medical Image Computing and Computer Assisted Intervention -- MICCAI 2021},
    year      = {2021},
    publisher = {Springer International Publishing},
    pages     = {530--539},
    doi       = {10.1007/978-3-030-87240-3_51},
}
```

## License

This project is licensed under the Apache License 2.0 — see the [LICENSE](LICENSE) file for details.

nnDetection is subject to its own licenses; see the [nnDetection repository](https://github.com/MIC-DKFZ/nnDetection) for details.
