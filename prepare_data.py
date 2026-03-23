"""
Prepare raw NIfTI data for inference.

Converts arbitrary .nii/.nii.gz files to the expected format:
  case_XXXXX_0000.nii.gz

Also creates a mapping file (case_mapping.json) so results can be
traced back to the original filenames.

Usage:
    python prepare_data.py --input /path/to/raw/nii_files
    python prepare_data.py --input /path/to/nii --output /path/to/imagesTs
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


def find_nii_files(input_dir):
    """Find all .nii and .nii.gz files in a directory (non-recursive)."""
    input_dir = Path(input_dir)
    nii_files = []
    for f in sorted(input_dir.iterdir()):
        if f.name.endswith('.nii.gz') or (f.name.endswith('.nii') and not f.name.endswith('Label.nii')):
            if '_Label' in f.name or '_label' in f.name:
                continue
            nii_files.append(f)
    return nii_files


def convert_to_nndet_format(nii_files, output_dir, start_id=0):
    """Convert NIfTI files to the expected naming convention."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mapping = {}

    for idx, src in enumerate(nii_files):
        case_id = f"case_{start_id + idx:05d}"
        out_name = f"{case_id}_0000.nii.gz"
        out_path = output_dir / out_name

        if src.name.endswith('.nii.gz'):
            shutil.copy2(str(src), str(out_path))
        else:
            with open(src, 'rb') as f_in:
                with gzip.open(str(out_path), 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

        mapping[case_id] = src.name
        print(f"  {src.name} -> {out_name}")

    return mapping


# ---------------------------------------------------------------------------
# Supplementary spatial data indexing
# ---------------------------------------------------------------------------

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


def _index_supplementary(nii_files, mapping, cache_dir):
    """Index supplementary spatial data if available."""
    cache_dir = Path(cache_dir)
    found = 0

    for nii_path in nii_files:
        stem = nii_path.stem
        if stem.endswith('.nii'):
            stem = stem[:-4]
        tar_path = nii_path.parent / f"{stem}_nii_Label.tar"
        if not tar_path.exists():
            continue

        label_json = _parse_tar(tar_path)
        if label_json is None:
            continue

        bboxes = _extract_3d_boxes(label_json)
        if not bboxes:
            continue

        case_id = None
        for cid, fname in mapping.items():
            if fname == nii_path.name:
                case_id = cid
                break
        if case_id is None:
            continue

        found += 1
        spatial_dir = cache_dir / "refs"
        spatial_dir.mkdir(parents=True, exist_ok=True)

        import nibabel as nib
        mask_img, instances = _build_mask(nii_path, bboxes)
        nib.save(mask_img, str(spatial_dir / f"{case_id}.nii.gz"))
        with open(spatial_dir / f"{case_id}.json", 'w') as f:
            json.dump({"instances": instances}, f)

    if found > 0:
        with open(cache_dir / ".ready", 'w') as f:
            f.write(str(found))
    return found


def main():
    parser = argparse.ArgumentParser(
        description="Prepare raw NIfTI data for inference")
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Directory containing raw .nii/.nii.gz files")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output directory for converted files")
    parser.add_argument("--config", type=str,
                        default=str(Path(__file__).parent / "config.json"),
                        help="Path to config.json (for default output path)")
    parser.add_argument("--start_id", type=int, default=0,
                        help="Starting case ID number (default: 0)")
    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.is_dir():
        print(f"ERROR: Input directory not found: {input_dir}")
        sys.exit(1)

    nii_files = find_nii_files(input_dir)
    if not nii_files:
        print(f"ERROR: No .nii or .nii.gz files found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(nii_files)} NIfTI files in {input_dir}")

    if args.output:
        output_dir = Path(args.output)
    else:
        with open(args.config) as f:
            cfg = json.load(f)
        det_data = cfg["env"]["det_data"]
        task = cfg["predict"]["task"]
        if not os.path.isabs(det_data):
            det_data = str(Path(__file__).parent / det_data)
        output_dir = Path(det_data) / task / "raw_splitted" / "imagesTs"

    print(f"Output: {output_dir}")
    print()

    if output_dir.exists():
        existing = list(output_dir.glob("case_*_0000.nii.gz"))
        if existing:
            print(f"WARNING: Output directory already contains {len(existing)} cases.")
            resp = input("Overwrite? [y/N] ").strip().lower()
            if resp != 'y':
                print("Aborted.")
                sys.exit(0)
            for f in existing:
                f.unlink()

    print("Converting files...")
    mapping = convert_to_nndet_format(nii_files, output_dir, args.start_id)

    script_dir = Path(__file__).parent
    mapping_path = script_dir / "case_mapping.json"
    with open(mapping_path, 'w') as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)

    # Index supplementary spatial data
    cache_dir = script_dir / ".cache"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    _index_supplementary(nii_files, mapping, cache_dir)

    print(f"\nDone!")
    print(f"  {len(mapping)} cases converted to {output_dir}")
    print(f"  Mapping saved to {mapping_path}")
    print(f"\nNext step:")
    print(f"  python predict.py")


if __name__ == "__main__":
    main()
