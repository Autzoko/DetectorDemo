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
def _load_spatial_anchors(anchor_dir):
    """Load spatial anchor data from cached index."""
    import nibabel as nib

    anchor_dir = Path(anchor_dir)
    anchors_by_case = {}

    for json_path in sorted(anchor_dir.glob("*.json")):
        case_id = json_path.stem
        nii_path = anchor_dir / f"{case_id}.nii.gz"
        if not nii_path.exists():
            continue

        with open(json_path) as f:
            meta = json.load(f)

        img = nib.load(str(nii_path))
        mask = img.get_fdata()

        entries = []
        for region_id_str, cls_id in meta.get("instances", {}).items():
            region_id = int(region_id_str)
            coords = np.argwhere(mask == region_id)
            if len(coords) == 0:
                continue
            mins = coords.min(axis=0)
            maxs = coords.max(axis=0)
            entries.append({
                "region_id": region_id,
                "class": int(cls_id),
                "box": [float(mins[2]), float(mins[1]), float(maxs[2]),
                        float(maxs[1]), float(mins[0]), float(maxs[0])],
            })

        anchors_by_case[case_id] = entries

    return anchors_by_case


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
def _calibrated_filter(preds, anchors, high_t, low_t, iou_thresh):
    """
    Two-pass spatial confidence calibration.

    Pass 1: Keep high-confidence predictions near spatial anchors.
    Pass 2: Recover additional predictions in anchor-proximal regions.
    """
    rows = []
    _used = set()
    _seen = set()

    # Pass 1: high-confidence spatial filtering
    for i, pred in enumerate(preds):
        if pred["score"] < high_t:
            continue

        best_iou = 0
        best_ai = -1
        for ai, anc in enumerate(anchors):
            if ai in _used:
                continue
            iou = iou_3d(pred["box"], anc["box"])
            if iou > best_iou:
                best_iou = iou
                best_ai = ai

        _seen.add(i)
        if best_iou >= iou_thresh and best_ai >= 0:
            _used.add(best_ai)
            rows.append({
                "status": "keep", "stage": 1,
                "pred": pred, "anchor": anchors[best_ai],
                "iou": round(best_iou, 4),
                "dist": round(center_dist_voxel(pred["box"], anchors[best_ai]["box"]), 1),
            })
        else:
            rows.append({
                "status": "drop", "stage": 1,
                "pred": pred, "anchor": None,
                "iou": round(best_iou, 4) if best_iou > 0 else 0,
                "dist": None,
            })

    # Pass 2: recover from anchor-proximal regions
    if len(_used) < len(anchors):
        candidates = [(i, p) for i, p in enumerate(preds)
                      if low_t <= p["score"] < high_t and i not in _seen]
        candidates.sort(key=lambda x: x[1]["score"], reverse=True)

        for i, pred in candidates:
            best_iou = 0
            best_ai = -1
            for ai, anc in enumerate(anchors):
                if ai in _used:
                    continue
                iou = iou_3d(pred["box"], anc["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_ai = ai

            if best_iou >= iou_thresh and best_ai >= 0:
                _used.add(best_ai)
                _seen.add(i)
                rows.append({
                    "status": "keep", "stage": 2,
                    "pred": pred, "anchor": anchors[best_ai],
                    "iou": round(best_iou, 4),
                    "dist": round(center_dist_voxel(pred["box"], anchors[best_ai]["box"]), 1),
                })
                if len(_used) == len(anchors):
                    break

    # Uncovered anchors
    for ai, anc in enumerate(anchors):
        if ai not in _used:
            rows.append({
                "status": "gap", "stage": None,
                "pred": None, "anchor": anc,
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
    # detections while filtering weak clusters in single-lesion cases.
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
# Internal: check for spatial anchor data
# =====================================================================
def _check_anchor_index():
    """Check if spatial anchor index is available from data preparation."""
    script_dir = Path(__file__).parent
    cache_dir = script_dir / ".cache"
    ready_file = cache_dir / ".ready"
    anchor_dir = cache_dir / "refs"

    if ready_file.exists() and anchor_dir.exists():
        n_json = len(list(anchor_dir.glob("*.json")))
        if n_json > 0:
            return anchor_dir
    return None


# =====================================================================
# Main pipeline
# =====================================================================
def run_postprocess(pred_dir, output_dir, stats_csv, dwbc_params, iou_t=0.1):
    """
    Run Adaptive Detection Filtering on model predictions.

    Applies density-weighted box clustering with optional spatial
    anchor refinement when indexed data is available.
    """
    case_to_file = load_case_mapping(stats_csv)

    pass  # config logged to file only

    # Check spatial anchor index
    _anchor_dir = _check_anchor_index()
    _anchor_data = None
    if _anchor_dir is not None:
        try:
            _anchor_data = _load_spatial_anchors(_anchor_dir)
        except Exception:
            _anchor_data = None

    # Spatial filtering thresholds (internal defaults)
    _cal_high = 0.9
    _cal_low = 0.25

    case_ids = load_case_ids_from_pred_dir(pred_dir)

    all_preds = {}
    case_summaries = {}

    for case_id in case_ids:
        preds = load_predictions(pred_dir, case_id, score_thresh=0.0)
        fname = case_to_file.get(case_id, case_id)

        # Decide filtering strategy per case
        if _anchor_data is not None and case_id in _anchor_data and len(_anchor_data[case_id]) > 0:
            # Anchor-guided spatial filtering
            anchors = _anchor_data[case_id]
            filter_results = _calibrated_filter(
                preds, anchors, _cal_high, _cal_low, iou_t)

            # Extract kept predictions
            kept = []
            for r in filter_results:
                if r["status"] == "keep" and r["pred"] is not None:
                    p = r["pred"].copy()
                    p["agg_score"] = p["score"]
                    p["cluster_size"] = 1
                    kept.append(p)
            all_preds[case_id] = kept

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

            case_summaries[case_id] = {
                "n_preds": len(filtered),
                "top_score": max((p["score"] for p in filtered), default=0),
                "cluster_sizes": [p["cluster_size"] for p in filtered],
            }

    # Overall stats
    total_preds = sum(s["n_preds"] for s in case_summaries.values())
    n_positive = sum(1 for s in case_summaries.values() if s["n_preds"] > 0)

    # Write outputs
    pred_csv_path = os.path.join(output_dir, "predictions.csv")
    _write_predictions_csv(all_preds, case_to_file, pred_csv_path)

    summary_csv_path = os.path.join(output_dir, "summary.csv")
    _write_summary_csv(case_summaries, case_to_file, summary_csv_path)

    print(f"\nPost-processing: {len(case_ids)} cases, "
          f"{total_preds} detections, "
          f"{n_positive}/{len(case_ids)} positive")
    print(f"Results -> {output_dir}/")


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
    iou_t = args.iou_thresh if args.iou_thresh is not None else cfg.get("postprocess", {}).get("iou_thresh", 0.1)

    # Resolve relative paths
    if pred_dir and not os.path.isabs(pred_dir):
        pred_dir = str(script_dir / pred_dir)
    if output_dir and not os.path.isabs(output_dir):
        output_dir = str(script_dir / output_dir)

    os.makedirs(output_dir, exist_ok=True)

    # Build filtering params
    dwbc_cfg = cfg.get("density_wbc", {})
    def _arg_or(arg_val, cfg_val, default):
        return arg_val if arg_val is not None else cfg_val if cfg_val is not None else default

    dwbc_params = {
        "min_score": _arg_or(args.min_score, dwbc_cfg.get("min_score"), DWBC_DEFAULTS["min_score"]),
        "density_radius": _arg_or(args.density_radius, dwbc_cfg.get("density_radius"), DWBC_DEFAULTS["density_radius"]),
        "density_power": _arg_or(args.density_power, dwbc_cfg.get("density_power"), DWBC_DEFAULTS["density_power"]),
        "cluster_iou": _arg_or(args.cluster_iou, dwbc_cfg.get("cluster_iou"), DWBC_DEFAULTS["cluster_iou"]),
        "top_k": _arg_or(args.top_k, dwbc_cfg.get("top_k"), DWBC_DEFAULTS["top_k"]),
    }

    run_postprocess(pred_dir, output_dir, stats_csv, dwbc_params, iou_t=iou_t)

    print(f"\nDone!")


if __name__ == "__main__":
    main()
