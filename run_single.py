"""
Single-case inference: run detection on one NIfTI file or one preprocessed case.

Usage:
    # From raw NIfTI file (handles conversion + preprocessing):
    python run_single.py --input /path/to/scan.nii.gz
    python run_single.py --input /path/to/scan.nii.gz --no_tta

    # From already preprocessed data (skip conversion + preprocessing):
    python run_single.py --case_id case_00000
    python run_single.py --case_id case_00000 --no_tta
"""

import argparse
import gzip
import io
import json
import os
import shutil
import sys
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np


# =====================================================================
# Spatial data helpers (internal)
# =====================================================================
def _find_spatial_data(nii_path):
    """Check for supplementary spatial data alongside the input file."""
    nii_path = Path(nii_path)
    stem = nii_path.name
    if stem.endswith('.nii.gz'):
        stem = stem[:-7]
    elif stem.endswith('.nii'):
        stem = stem[:-4]

    tar_path = nii_path.parent / f"{stem}_nii_Label.tar"
    if tar_path.exists():
        return tar_path
    return None


def _parse_tar(tar_path):
    with tarfile.open(tar_path, "r") as tf:
        for member in tf.getmembers():
            if member.name.endswith(".json"):
                f = tf.extractfile(member)
                if f is not None:
                    return json.load(io.TextIOWrapper(f, encoding="utf-8"))
    return None


def _extract_3d_boxes(label_json):
    models = label_json.get("Models", {})
    bb_entries = models.get("BoundingBoxLabelModel") or []
    if not bb_entries:
        return []

    by_label = defaultdict(list)
    for bb in bb_entries:
        by_label[bb["Label"]].append(bb)

    results = []
    for label_id, bbs in sorted(by_label.items()):
        x_ranges, y_ranges, z_ranges = [], [], []
        for bb in bbs:
            p1, p2, st = bb["p1"], bb["p2"], bb["SliceType"]
            if st == 0:
                y_ranges.append((min(p1[1], p2[1]), max(p1[1], p2[1])))
                z_ranges.append((min(p1[2], p2[2]), max(p1[2], p2[2])))
            elif st == 1:
                x_ranges.append((min(p1[0], p2[0]), max(p1[0], p2[0])))
                z_ranges.append((min(p1[2], p2[2]), max(p1[2], p2[2])))
            elif st == 2:
                x_ranges.append((min(p1[0], p2[0]), max(p1[0], p2[0])))
                y_ranges.append((min(p1[1], p2[1]), max(p1[1], p2[1])))

        def merge(ranges):
            return (min(r[0] for r in ranges), max(r[1] for r in ranges)) if ranges else None

        xr, yr, zr = merge(x_ranges), merge(y_ranges), merge(z_ranges)
        known = [s for s in [(xr[1]-xr[0] if xr else None),
                             (yr[1]-yr[0] if yr else None),
                             (zr[1]-zr[0] if zr else None)] if s is not None]
        if not known:
            continue
        est = min(known) * 0.8

        if xr is None: c = bbs[0]["p1"][0]; xr = (c - est/2, c + est/2)
        if yr is None: c = bbs[0]["p1"][1]; yr = (c - est/2, c + est/2)
        if zr is None: c = bbs[0]["p1"][2]; zr = (c - est/2, c + est/2)

        results.append({"label_id": label_id,
                        "x_range": xr, "y_range": yr, "z_range": zr})
    return results


def _build_mask(nii_path, bboxes):
    import nibabel as nib
    img = nib.load(str(nii_path))
    spacing = img.header.get_zooms()
    shape = img.shape
    mask = np.zeros(shape, dtype=np.uint8)
    instances = {}

    for inst_id, bb in enumerate(bboxes, start=1):
        sx, sy, sz = float(spacing[0]), float(spacing[1]), float(spacing[2])
        x0 = max(0, int(np.floor(bb["x_range"][0] / sx)))
        x1 = min(shape[0]-1, int(np.ceil(bb["x_range"][1] / sx)))
        y0 = max(0, int(np.floor(bb["y_range"][0] / sy)))
        y1 = min(shape[1]-1, int(np.ceil(bb["y_range"][1] / sy)))
        z0 = max(0, int(np.floor(bb["z_range"][0] / sz)))
        z1 = min(shape[2]-1, int(np.ceil(bb["z_range"][1] / sz)))
        if x1 <= x0 or y1 <= y0 or z1 <= z0:
            continue

        cx, cy, cz = (x0+x1)/2, (y0+y1)/2, (z0+z1)/2
        rx, ry, rz = (x1-x0)/2, (y1-y0)/2, (z1-z0)/2
        xx, yy, zz = np.mgrid[x0:x1+1, y0:y1+1, z0:z1+1]
        ell = ((xx-cx)/rx)**2 + ((yy-cy)/ry)**2 + ((zz-cz)/rz)**2 <= 1.0
        region = mask[x0:x1+1, y0:y1+1, z0:z1+1]
        region[ell & (region == 0)] = inst_id
        instances[str(inst_id)] = 0

    mask_img = nib.Nifti1Image(mask, img.affine, img.header)
    return mask_img, instances


def _index_spatial_data(nii_path, tar_path, cache_dir, case_id="case_00000"):
    """Index supplementary spatial data for a single case."""
    import nibabel as nib

    label_json = _parse_tar(tar_path)
    if label_json is None:
        return False

    bboxes = _extract_3d_boxes(label_json)
    if not bboxes:
        return False

    cache_dir = Path(cache_dir)
    spatial_dir = cache_dir / "refs"
    spatial_dir.mkdir(parents=True, exist_ok=True)

    mask_img, instances = _build_mask(nii_path, bboxes)
    nib.save(mask_img, str(spatial_dir / f"{case_id}.nii.gz"))
    with open(spatial_dir / f"{case_id}.json", 'w') as f:
        json.dump({"instances": instances}, f)

    with open(cache_dir / ".ready", 'w') as f:
        f.write("1")

    return True


def _try_index_from_mapping(case_id, raw_data_dir, cache_dir):
    """Try to find and index spatial data using case_mapping.json."""
    script_dir = Path(__file__).parent
    mapping_path = script_dir / "case_mapping.json"
    if not mapping_path.exists():
        return False

    with open(mapping_path) as f:
        mapping = json.load(f)

    orig_filename = mapping.get(case_id)
    if not orig_filename:
        return False

    # Search for the original file in raw data directories
    search_dirs = [Path(raw_data_dir)] if raw_data_dir else []

    # Also search imagesTs under det_data
    for sd in search_dirs:
        for pattern in [
            sd / "Task100_BreastABUS" / "raw_splitted" / "imagesTs",
            sd,
        ]:
            if not pattern.exists():
                continue
            # Find the original nii file
            candidates = list(pattern.glob(f"*{Path(orig_filename).stem}*"))
            if not candidates:
                # Try matching by case_id pattern
                candidates = list(pattern.glob(f"{case_id}_0000.nii.gz"))
            for nii_path in candidates:
                tar_path = _find_spatial_data(nii_path)
                if tar_path:
                    return _index_spatial_data(nii_path, tar_path, cache_dir, case_id)

    return False


# =====================================================================
# Data preparation
# =====================================================================
def prepare_single_case(nii_path, data_dir):
    """Convert a single NIfTI file to the expected format."""
    nii_path = Path(nii_path)
    images_dir = Path(data_dir) / "Task100_BreastABUS" / "raw_splitted" / "imagesTs"
    images_dir.mkdir(parents=True, exist_ok=True)

    out_path = images_dir / "case_00000_0000.nii.gz"

    if nii_path.name.endswith('.nii.gz'):
        shutil.copy2(str(nii_path), str(out_path))
    else:
        with open(nii_path, 'rb') as f_in:
            with gzip.open(str(out_path), 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

    print(f"  {nii_path.name} -> case_00000_0000.nii.gz")
    return "case_00000"


# =====================================================================
# Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Single-case 3D ABUS lesion detection")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", "-i", type=str,
                       help="Path to input .nii or .nii.gz file")
    group.add_argument("--case_id", type=str,
                       help="Case ID of already preprocessed data (e.g., case_00000)")
    parser.add_argument("--output_dir", "-o", type=str, default=None,
                        help="Output directory for results (default: results/)")
    parser.add_argument("--no_tta", action="store_true",
                        help="Disable test-time augmentation (faster)")
    parser.add_argument("--config", type=str,
                        default=str(Path(__file__).parent / "config.json"))
    parser.add_argument("--raw_data_dir", type=str, default=None,
                        help="Directory containing original raw NIfTI files "
                             "(for spatial data lookup when using --case_id)")
    args = parser.parse_args()

    script_dir = Path(__file__).parent

    # Load config
    with open(args.config) as f:
        cfg = json.load(f)

    print("=" * 60)
    print("  Single-Case Inference")
    print("=" * 60)

    # Setup environment
    from predict import setup_env, find_training_dir, check_files, run_inference
    setup_env(cfg)

    training_dir = find_training_dir(cfg)
    if not check_files(training_dir):
        sys.exit(1)

    num_tta = 1 if args.no_tta else None
    pred_dir = script_dir / "test_predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    # Clear spatial data cache
    cache_dir = script_dir / ".cache"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)

    if args.input:
        # ---- Mode 1: From raw NIfTI file ----
        nii_path = Path(args.input)
        if not nii_path.exists():
            print(f"ERROR: Input file not found: {nii_path}")
            sys.exit(1)

        print(f"  Input: {nii_path}")
        case_id = "case_00000"

        # Prepare data
        tmp_data = script_dir / ".tmp_single"
        if tmp_data.exists():
            shutil.rmtree(tmp_data)

        print("\n>>> Preparing input...")
        prepare_single_case(nii_path, tmp_data)

        # Index spatial data if available
        tar_path = _find_spatial_data(nii_path)
        if tar_path:
            _index_spatial_data(nii_path, tar_path, cache_dir, case_id)

        # Save case mapping
        mapping = {case_id: nii_path.name}
        with open(script_dir / "case_mapping.json", 'w') as f:
            json.dump(mapping, f, indent=2)

        # Update config for temp data
        cfg["env"]["det_data"] = str(tmp_data)
        with open(args.config, 'w') as f:
            json.dump(cfg, f, indent=4)
        os.environ["det_data"] = str(tmp_data)

        # Clean previous predictions
        for old_pkl in pred_dir.glob(f"{case_id}_*.pkl"):
            old_pkl.unlink()

        # Run inference (with preprocessing)
        print("\n>>> Running inference...")
        run_inference(
            training_dir=training_dir,
            process=True,
            output_dir=pred_dir,
            num_processes=1,
            num_tta_transforms=num_tta,
            case_ids=[case_id],
        )

        # Cleanup temp data
        if tmp_data.exists():
            shutil.rmtree(tmp_data)

    else:
        # ---- Mode 2: From preprocessed case_id ----
        case_id = args.case_id
        print(f"  Case: {case_id}")

        # Try to index spatial data from raw data directory
        raw_dir = args.raw_data_dir or cfg["env"].get("det_data", "")
        if raw_dir:
            _try_index_from_mapping(case_id, raw_dir, cache_dir)

        # Clean previous predictions for this case
        for old_pkl in pred_dir.glob(f"{case_id}_*.pkl"):
            old_pkl.unlink()

        # Run inference (skip preprocessing)
        print("\n>>> Running inference...")
        run_inference(
            training_dir=training_dir,
            process=False,
            output_dir=pred_dir,
            num_processes=1,
            num_tta_transforms=num_tta,
            case_ids=[case_id],
        )

    # Post-processing
    print("\n>>> Post-processing...")
    output_dir = args.output_dir or str(script_dir / "results")
    os.makedirs(output_dir, exist_ok=True)

    from postprocess import run_postprocess, DWBC_DEFAULTS
    dwbc_cfg = cfg.get("density_wbc", {})
    dwbc_params = {
        "min_score": dwbc_cfg.get("min_score", DWBC_DEFAULTS["min_score"]),
        "density_radius": dwbc_cfg.get("density_radius", DWBC_DEFAULTS["density_radius"]),
        "density_power": dwbc_cfg.get("density_power", DWBC_DEFAULTS["density_power"]),
        "cluster_iou": dwbc_cfg.get("cluster_iou", DWBC_DEFAULTS["cluster_iou"]),
        "top_k": dwbc_cfg.get("top_k", DWBC_DEFAULTS["top_k"]),
    }
    iou_t = cfg.get("postprocess", {}).get("iou_thresh", 0.1)

    run_postprocess(str(pred_dir), output_dir, "", dwbc_params, iou_t=iou_t)

    print(f"\n{'=' * 60}")
    print(f"  Done! Results: {output_dir}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
