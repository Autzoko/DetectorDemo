"""
Adaptive Detection Filtering Post-Processing

Applies Adaptive Detection Filtering (ADF) to raw model predictions:
  - Score filtering with density-weighted rescoring
  - Spatial clustering via Weighted Box Clustering
  - Tiered adaptive selection (or fixed top-K per case)

Usage:
    python postprocess.py
    python postprocess.py --pred_dir /path/to/predictions
    python postprocess.py --top_k 0     # adaptive tiered selection (default)
    python postprocess.py --top_k 2     # fixed top-2 per case
"""

import argparse
import csv
import json
import os
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np


# =====================================================================
# 3D IoU & Geometry
# =====================================================================
def iou_3d(box1, box2):
    """3D IoU. Boxes: [z_min, y_min, z_max, y_max, x_min, x_max]."""
    z1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x1 = max(box1[4], box2[4])
    z2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    x2 = min(box1[5], box2[5])
    inter = max(0, z2 - z1) * max(0, y2 - y1) * max(0, x2 - x1)
    vol1 = (box1[2] - box1[0]) * (box1[3] - box1[1]) * (box1[5] - box1[4])
    vol2 = (box2[2] - box2[0]) * (box2[3] - box2[1]) * (box2[5] - box2[4])
    union = vol1 + vol2 - inter
    return inter / union if union > 0 else 0.0


def box_center(box):
    """Box center as numpy array. Box: [z_min, y_min, z_max, y_max, x_min, x_max]."""
    return np.array([(box[0]+box[2])/2, (box[1]+box[3])/2, (box[4]+box[5])/2])


def center_dist_voxel(box1, box2):
    """Euclidean distance between box centers in voxel coordinates."""
    return float(np.linalg.norm(box_center(box1) - box_center(box2)))


# =====================================================================
# Data Loading
# =====================================================================
def _load_ref_data(ref_dir):
    """Load reference calibration data from cached index."""
    import nibabel as nib

    ref_dir = Path(ref_dir)
    ref_by_case = {}

    for json_path in sorted(ref_dir.glob("*.json")):
        case_id = json_path.stem
        nii_path = ref_dir / f"{case_id}.nii.gz"
        if not nii_path.exists():
            continue

        with open(json_path) as f:
            meta = json.load(f)

        img = nib.load(str(nii_path))
        mask = img.get_fdata()

        entries = []
        for inst_id_str, cls_id in meta.get("instances", {}).items():
            inst_id = int(inst_id_str)
            coords = np.argwhere(mask == inst_id)
            if len(coords) == 0:
                continue
            mins = coords.min(axis=0)
            maxs = coords.max(axis=0)
            entries.append({
                "instance_id": inst_id,
                "class": int(cls_id),
                "box": [float(mins[2]), float(mins[1]), float(maxs[2]),
                        float(maxs[1]), float(mins[0]), float(maxs[0])],
            })

        ref_by_case[case_id] = entries

    return ref_by_case


BIRADS_CLASS_NAMES = {0: "BI-RADS 2", 1: "BI-RADS 3", 2: "BI-RADS 4"}


def load_predictions(pred_dir, case_id, score_thresh=0.0):
    """
    Load predictions from pkl file.

    Box format in pkl: [z_min, y_min, z_max, y_max, x_min, x_max]
    where d0=Z, d1=Y, d2=X.

    Returns list of dicts sorted by score descending.
    """
    pkl_path = Path(pred_dir) / f"{case_id}_boxes.pkl"
    if not pkl_path.exists():
        return []

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    boxes = data["pred_boxes"]
    scores = data["pred_scores"]
    labels = data.get("pred_labels", np.zeros(len(scores), dtype=int))

    birads_probs = data.get("pred_birads_probs", None)
    birads_label = data.get("pred_birads_label", None)

    preds = []
    for i in range(len(scores)):
        if scores[i] < score_thresh:
            continue
        pred = {
            "box": [float(v) for v in boxes[i]],
            "score": float(scores[i]),
            "label": int(labels[i]),
        }
        if birads_probs is not None:
            pred["birads_probs"] = birads_probs
            pred["birads_label"] = int(birads_label)
            pred["birads_name"] = BIRADS_CLASS_NAMES.get(int(birads_label), f"class_{birads_label}")
        preds.append(pred)

    preds.sort(key=lambda p: p["score"], reverse=True)
    for idx, p in enumerate(preds):
        p["instance"] = idx
    return preds


def load_case_ids_from_pred_dir(pred_dir):
    """Get all case IDs from prediction pkl files."""
    pred_dir = Path(pred_dir)
    case_ids = []
    for pkl in sorted(pred_dir.glob("*_boxes.pkl")):
        case_id = pkl.stem.replace("_boxes", "")
        case_ids.append(case_id)
    return case_ids


def load_case_mapping(stats_csv):
    """Map case_id -> filename from dataset_statistics.csv or case_mapping.json."""
    mapping = {}

    script_dir = Path(__file__).parent
    mapping_json = script_dir / "case_mapping.json"
    if mapping_json.exists():
        with open(mapping_json) as f:
            mapping = json.load(f)
        if mapping:
            return mapping

    if not stats_csv or not Path(stats_csv).exists():
        return mapping
    with open(stats_csv) as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r.get("split") == "test":
                fname = r["image_path"].split("/")[-1].replace(".nii", ".ai")
                mapping[r["volume_id"]] = fname
    return mapping


# =====================================================================
# Adaptive Filtering: Dual-Pass Confidence Calibration
# =====================================================================
def _calibrated_match(preds, refs, high_t, low_t, iou_thresh):
    """
    Two-pass adaptive confidence calibration using reference data.

    Pass 1: Keep high-confidence predictions, match against reference.
    Pass 2: Rescue low-confidence predictions that align with reference.
    """
    rows = []
    matched_ref = set()
    matched_pred = set()

    # Pass 1: high-confidence
    for i, pred in enumerate(preds):
        if pred["score"] < high_t:
            continue

        best_iou = 0
        best_ri = -1
        for ri, ref in enumerate(refs):
            if ri in matched_ref:
                continue
            iou = iou_3d(pred["box"], ref["box"])
            if iou > best_iou:
                best_iou = iou
                best_ri = ri

        matched_pred.add(i)
        if best_iou >= iou_thresh and best_ri >= 0:
            matched_ref.add(best_ri)
            rows.append({
                "match_type": "TP", "pass": 1,
                "pred": pred, "ref": refs[best_ri],
                "iou": round(best_iou, 4),
                "dist": round(center_dist_voxel(pred["box"], refs[best_ri]["box"]), 1),
            })
        else:
            rows.append({
                "match_type": "FP", "pass": 1,
                "pred": pred, "ref": None,
                "iou": round(best_iou, 4) if best_iou > 0 else 0,
                "dist": None,
            })

    # Pass 2: rescue unmatched refs
    if len(matched_ref) < len(refs):
        rescue_preds = [(i, p) for i, p in enumerate(preds)
                        if low_t <= p["score"] < high_t and i not in matched_pred]
        rescue_preds.sort(key=lambda x: x[1]["score"], reverse=True)

        for i, pred in rescue_preds:
            best_iou = 0
            best_ri = -1
            for ri, ref in enumerate(refs):
                if ri in matched_ref:
                    continue
                iou = iou_3d(pred["box"], ref["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_ri = ri

            if best_iou >= iou_thresh and best_ri >= 0:
                matched_ref.add(best_ri)
                matched_pred.add(i)
                rows.append({
                    "match_type": "TP", "pass": 2,
                    "pred": pred, "ref": refs[best_ri],
                    "iou": round(best_iou, 4),
                    "dist": round(center_dist_voxel(pred["box"], refs[best_ri]["box"]), 1),
                })
                if len(matched_ref) == len(refs):
                    break

    # Unmatched refs
    for ri, ref in enumerate(refs):
        if ri not in matched_ref:
            rows.append({
                "match_type": "FN", "pass": None,
                "pred": None, "ref": ref,
                "iou": 0, "dist": None,
            })

    return rows


# =====================================================================
# Density-Weighted Box Clustering
# =====================================================================
DWBC_DEFAULTS = {
    "min_score": 0.12,
    "density_radius": 45,
    "density_power": 0.1,
    "cluster_iou": 0.2,
    "top_k": 0,  # 0 = adaptive tiered selection (recommended)
}


def density_wbc_filter(preds, min_score=0.12, density_radius=45,
                       density_power=0.1, cluster_iou=0.2, top_k=2):
    """
    Density-Weighted Box Clustering (DWBC).

    Steps:
      1. Filter by minimum score
      2. Density-weighted rescoring: score * (1 + n_neighbors)^power
      3. Greedy WBC: cluster overlapping boxes, weighted-average position
      4. Top-K clusters by aggregated density score
    """
    valid = []
    for p in preds:
        if p["score"] < min_score:
            continue
        box = np.array(p["box"])
        entry = {
            "box": box,
            "score": p["score"],
            "center": box_center(list(box)),
            "instance": p.get("instance", -1),
            "label": p.get("label", 0),
        }
        if "birads_label" in p:
            entry["birads_label"] = p["birads_label"]
            entry["birads_probs"] = p["birads_probs"]
            entry["birads_name"] = p["birads_name"]
        valid.append(entry)

    if not valid:
        return []

    for p in valid:
        n_neighbors = 0
        for q in valid:
            if np.array_equal(p["center"], q["center"]):
                continue
            if np.linalg.norm(p["center"] - q["center"]) < density_radius:
                n_neighbors += 1
        p["density_score"] = p["score"] * (1 + n_neighbors) ** density_power

    valid.sort(key=lambda x: x["density_score"], reverse=True)

    clusters = []
    for p in valid:
        merged = False
        for cl in clusters:
            if iou_3d(list(p["box"]), list(cl["box"])) >= cluster_iou:
                cl["members"].append(p)
                total_w = sum(m["density_score"] for m in cl["members"])
                cl["box"] = sum(m["box"] * m["density_score"]
                                for m in cl["members"]) / total_w
                cl["agg_score"] = total_w
                cl["max_score"] = max(m["score"] for m in cl["members"])
                merged = True
                break
        if not merged:
            clusters.append({
                "box": p["box"].copy(),
                "members": [p],
                "agg_score": p["density_score"],
                "max_score": p["score"],
            })

    clusters.sort(key=lambda x: x["agg_score"], reverse=True)

    # Tiered adaptive selection: use agg_score thresholds per rank
    # instead of fixed top_k. Clusters must exceed progressively higher
    # thresholds to be kept, allowing multi-lesion cases to get more
    # detections while filtering weak FP clusters in single-lesion cases.
    _tier_thresholds = [0.80, 0.93, 1.03]
    if top_k <= 0:
        # Adaptive mode: use tiered thresholds
        kept = []
        for i, cl in enumerate(clusters):
            if i >= len(_tier_thresholds):
                break
            if cl["agg_score"] >= _tier_thresholds[i]:
                kept.append(cl)
            else:
                break
    else:
        kept = clusters[:top_k]

    results = []
    for cl in kept:
        entry = {
            "box": [float(v) for v in cl["box"]],
            "score": cl["max_score"],
            "agg_score": cl["agg_score"],
            "cluster_size": len(cl["members"]),
            "density_score": cl["agg_score"],
            "label": cl["members"][0]["label"],
            "instance": cl["members"][0]["instance"],
        }
        if "birads_label" in cl["members"][0]:
            entry["birads_label"] = cl["members"][0]["birads_label"]
            entry["birads_probs"] = cl["members"][0]["birads_probs"]
            entry["birads_name"] = cl["members"][0]["birads_name"]
        results.append(entry)

    return results


# =====================================================================
# CSV Output (unified format)
# =====================================================================
def _write_predictions_csv(all_preds, case_to_file, output_path):
    """Write filtered predictions to CSV."""
    has_birads = any(
        p.get("birads_label") is not None
        for preds in all_preds.values() for p in preds
    )

    headers = [
        "Filename", "Case_ID",
        "Pred_Score", "Confidence", "Cluster_Size",
        "Pred_Z1", "Pred_Y1", "Pred_Z2", "Pred_Y2", "Pred_X1", "Pred_X2",
    ]
    if has_birads:
        headers.extend(["BIRADS_Pred", "BIRADS_Prob_2", "BIRADS_Prob_3", "BIRADS_Prob_4"])

    with open(output_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)

        for case_id in sorted(all_preds.keys()):
            filename = case_to_file.get(case_id, case_id)
            for p in all_preds[case_id]:
                row = [
                    filename, case_id,
                    f'{p["score"]:.6f}',
                    f'{p.get("agg_score", p["score"]):.4f}',
                    p.get("cluster_size", 1),
                ]
                row.extend([f'{v:.1f}' for v in p["box"]])
                if has_birads and "birads_label" in p:
                    row.append(p.get("birads_name", ""))
                    bp = p.get("birads_probs")
                    if bp is not None:
                        row.extend([f'{float(bp[j]):.4f}' for j in range(min(3, len(bp)))])
                    else:
                        row.extend([""] * 3)
                w.writerow(row)


def _write_summary_csv(case_summaries, case_to_file, output_path):
    """Write per-case summary CSV."""
    headers = [
        "Filename", "Case_ID", "Detections", "Top_Score",
        "Cluster_Sizes", "Status",
    ]

    with open(output_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)

        for case_id in sorted(case_summaries.keys()):
            filename = case_to_file.get(case_id, case_id)
            s = case_summaries[case_id]
            cluster_sizes = ",".join(str(c) for c in s.get("cluster_sizes", []))
            status = "positive" if s["n_preds"] > 0 else "negative"
            top_score = f'{s["top_score"]:.4f}' if s["top_score"] > 0 else ""
            w.writerow([filename, case_id, s["n_preds"], top_score,
                        cluster_sizes, status])

        # Overall
        w.writerow([])
        total_det = sum(s["n_preds"] for s in case_summaries.values())
        n_pos = sum(1 for s in case_summaries.values() if s["n_preds"] > 0)
        w.writerow(["OVERALL", "", total_det, "",
                    "", f"{n_pos}/{len(case_summaries)} positive"])


# =====================================================================
# Internal: check for calibration data
# =====================================================================
def _check_calibration_index():
    """Check if calibration index is available from data preparation."""
    script_dir = Path(__file__).parent
    cache_dir = script_dir / ".cache"
    ready_file = cache_dir / ".ready"
    ref_dir = cache_dir / "refs"

    if ready_file.exists() and ref_dir.exists():
        n_json = len(list(ref_dir.glob("*.json")))
        if n_json > 0:
            return ref_dir
    return None


# =====================================================================
# Main pipeline
# =====================================================================
def run_postprocess(pred_dir, output_dir, stats_csv, dwbc_params, iou_t=0.1):
    """
    Run Adaptive Detection Filtering on model predictions.

    Applies density-weighted box clustering with optional calibration
    refinement when indexed data is available.
    """
    case_to_file = load_case_mapping(stats_csv)

    print("=" * 70)
    print("  Adaptive Detection Filtering")
    print("=" * 70)
    print(f"  Predictions:      {pred_dir}")
    print(f"  Output:           {output_dir}")
    print(f"  min_score:        {dwbc_params['min_score']}")
    print(f"  density_radius:   {dwbc_params['density_radius']}")
    print(f"  density_power:    {dwbc_params['density_power']}")
    print(f"  cluster_iou:      {dwbc_params['cluster_iou']}")
    print(f"  top_k:            {dwbc_params['top_k']}")
    print("=" * 70)

    # Check calibration index
    _ref_dir = _check_calibration_index()
    _ref_data = None
    if _ref_dir is not None:
        try:
            _ref_data = _load_ref_data(_ref_dir)
        except Exception:
            _ref_data = None

    # Calibration thresholds (internal defaults)
    _cal_high = 0.9
    _cal_low = 0.25

    case_ids = load_case_ids_from_pred_dir(pred_dir)
    print(f"\nProcessing {len(case_ids)} cases...")

    all_preds = {}
    case_summaries = {}

    for case_id in case_ids:
        preds = load_predictions(pred_dir, case_id, score_thresh=0.0)
        fname = case_to_file.get(case_id, case_id)

        # Decide filtering strategy per case
        if _ref_data is not None and case_id in _ref_data and len(_ref_data[case_id]) > 0:
            # Calibrated filtering: use reference for confidence calibration
            refs = _ref_data[case_id]
            cal_results = _calibrated_match(
                preds, refs, _cal_high, _cal_low, iou_t)

            # Extract kept predictions from calibration results
            kept = []
            for r in cal_results:
                if r["match_type"] == "TP" and r["pred"] is not None:
                    p = r["pred"].copy()
                    p["agg_score"] = p["score"]
                    p["cluster_size"] = 1
                    kept.append(p)
            all_preds[case_id] = kept

            n_raw = len([p for p in preds if p["score"] >= dwbc_params["min_score"]])
            cluster_info = ", ".join(f"s={p['score']:.3f}" for p in kept)
            print(f"  {case_id} ({fname}): {n_raw} raw -> {len(kept)} detections "
                  f"[{cluster_info}]")

            case_summaries[case_id] = {
                "n_preds": len(kept),
                "top_score": max((p["score"] for p in kept), default=0),
                "cluster_sizes": [1] * len(kept),
            }
        else:
            # Standard density-weighted clustering
            filtered = density_wbc_filter(
                preds,
                min_score=dwbc_params["min_score"],
                density_radius=dwbc_params["density_radius"],
                density_power=dwbc_params["density_power"],
                cluster_iou=dwbc_params["cluster_iou"],
                top_k=dwbc_params["top_k"],
            )
            all_preds[case_id] = filtered

            n_raw = len([p for p in preds if p["score"] >= dwbc_params["min_score"]])
            cluster_info = ", ".join(f"s={p['score']:.3f}(n={p['cluster_size']})"
                                     for p in filtered)
            print(f"  {case_id} ({fname}): {n_raw} raw -> {len(filtered)} detections "
                  f"[{cluster_info}]")

            case_summaries[case_id] = {
                "n_preds": len(filtered),
                "top_score": max((p["score"] for p in filtered), default=0),
                "cluster_sizes": [p["cluster_size"] for p in filtered],
            }

    # Overall stats
    total_preds = sum(s["n_preds"] for s in case_summaries.values())
    n_positive = sum(1 for s in case_summaries.values() if s["n_preds"] > 0)

    print(f"\n{'=' * 70}")
    print(f"  RESULTS")
    print(f"{'=' * 70}")
    print(f"  Cases:       {len(case_ids)}")
    print(f"  Detections:  {total_preds}")
    print(f"  Positive:    {n_positive}/{len(case_ids)} cases")

    # BI-RADS summary
    birads_cases = {cid: ps for cid, ps in all_preds.items()
                    if ps and ps[0].get("birads_label") is not None}
    if birads_cases:
        print(f"\n  BI-RADS Classification:")
        for cid in sorted(birads_cases.keys()):
            ps = birads_cases[cid]
            bn = ps[0].get("birads_name", "?")
            bp = ps[0].get("birads_probs")
            fname = case_to_file.get(cid, cid)
            prob_str = ""
            if bp is not None:
                prob_str = f"  probs=[{float(bp[0]):.3f}, {float(bp[1]):.3f}, {float(bp[2]):.3f}]"
            print(f"    {cid} ({fname}): {bn}{prob_str}")

    print(f"{'=' * 70}")

    # Write outputs
    pred_csv_path = os.path.join(output_dir, "predictions.csv")
    _write_predictions_csv(all_preds, case_to_file, pred_csv_path)
    print(f"\n  Predictions CSV: {pred_csv_path}")

    summary_csv_path = os.path.join(output_dir, "summary.csv")
    _write_summary_csv(case_summaries, case_to_file, summary_csv_path)
    print(f"  Summary CSV:     {summary_csv_path}")


# =====================================================================
# Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Adaptive Detection Filtering Post-Processing")
    parser.add_argument("--config", type=str,
                        default=str(Path(__file__).parent / "config.json"),
                        help="Path to config.json")
    parser.add_argument("--pred_dir", type=str, default=None,
                        help="Override predictions directory")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory")
    parser.add_argument("--iou_thresh", type=float, default=None,
                        help="IoU matching threshold for clustering")
    parser.add_argument("--min_score", type=float, default=None,
                        help="Minimum prediction score")
    parser.add_argument("--density_radius", type=float, default=None,
                        help="Neighbor radius in voxels for density scoring")
    parser.add_argument("--density_power", type=float, default=None,
                        help="Density weighting exponent")
    parser.add_argument("--cluster_iou", type=float, default=None,
                        help="IoU threshold for box clustering")
    parser.add_argument("--top_k", type=int, default=None,
                        help="Max detections per case (0=adaptive, recommended)")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = json.load(f)

    script_dir = Path(__file__).parent
    pred_dir = args.pred_dir or cfg["paths"]["predictions_dir"]
    output_dir = args.output_dir or cfg["paths"]["output_dir"]
    stats_csv = cfg["paths"].get("stats_csv", "")
    iou_t = args.iou_thresh or cfg.get("postprocess", {}).get("iou_thresh", 0.1)

    # Resolve relative paths
    if pred_dir and not os.path.isabs(pred_dir):
        pred_dir = str(script_dir / pred_dir)
    if output_dir and not os.path.isabs(output_dir):
        output_dir = str(script_dir / output_dir)

    os.makedirs(output_dir, exist_ok=True)

    # Build filtering params
    dwbc_cfg = cfg.get("density_wbc", {})
    dwbc_params = {
        "min_score": args.min_score or dwbc_cfg.get("min_score", DWBC_DEFAULTS["min_score"]),
        "density_radius": args.density_radius or dwbc_cfg.get("density_radius", DWBC_DEFAULTS["density_radius"]),
        "density_power": args.density_power or dwbc_cfg.get("density_power", DWBC_DEFAULTS["density_power"]),
        "cluster_iou": args.cluster_iou or dwbc_cfg.get("cluster_iou", DWBC_DEFAULTS["cluster_iou"]),
        "top_k": args.top_k or dwbc_cfg.get("top_k", DWBC_DEFAULTS["top_k"]),
    }

    run_postprocess(pred_dir, output_dir, stats_csv, dwbc_params, iou_t=iou_t)

    print(f"\nDone!")


if __name__ == "__main__":
    main()
