"""
Lightweight test: run inference on a single case using CPU.
Tests the full pipeline: load model -> preprocess -> predict -> save pkl.

Reads all paths from config.json (no hardcoded paths).

Usage:
    python test_single_case.py
    python test_single_case.py --case_id case_00000
    python test_single_case.py --skip_preprocess  # if already preprocessed
"""

import argparse
import importlib
import json
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR / "nnDetection"))


def find_training_dir(cfg):
    """Find model directory from config."""
    pred_cfg = cfg["predict"]
    det_models = os.environ.get("det_models", "")
    base = Path(det_models) / pred_cfg["task"] / pred_cfg["model"]
    fold = pred_cfg["fold"]
    candidates = sorted(base.glob(f"fold{fold}__*"))
    if candidates:
        return candidates[-1]
    return base / f"fold{fold}__0"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(SCRIPT_DIR / "config.json"),
                        help="Path to config.json")
    parser.add_argument("--case_id", default="case_00000",
                        help="Case ID to predict (default: case_00000)")
    parser.add_argument("--skip_preprocess", action="store_true",
                        help="Skip preprocessing (use already preprocessed data)")
    parser.add_argument("--output_dir", default=str(SCRIPT_DIR / "test_output"),
                        help="Output directory")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = json.load(f)

    # Set environment from config
    for key, val in cfg.get("env", {}).items():
        if key.startswith("_"):
            continue
        # Resolve relative det_models path
        if key == "det_models" and not os.path.isabs(str(val)):
            val = str(SCRIPT_DIR / val)
        os.environ[key] = str(val)

    import torch
    from omegaconf import OmegaConf
    from loguru import logger

    from nndet.io.load import load_pickle
    from nndet.inference.loading import load_all_models
    from nndet.inference.helper import predict_dir

    training_dir = find_training_dir(cfg)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(sys.stdout, format="<level>{level} {message}</level>",
               level="INFO", colorize=True)

    print("=" * 60)
    print("  Lightweight Inference Test (CPU)")
    print("=" * 60)
    print(f"  Case: {args.case_id}")
    print(f"  Model: {cfg['predict']['model']}")
    print(f"  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  Training dir: {training_dir}")
    print(f"  Output: {output_dir}")
    print("=" * 60)

    if not training_dir.exists():
        print(f"\nERROR: Training directory not found: {training_dir}")
        print("Make sure model files are downloaded from HPC.")
        sys.exit(1)

    # Load plan & config
    plan = load_pickle(training_dir / "plan_inference.pkl")
    model_cfg = OmegaConf.load(str(training_dir / "config.yaml"))
    model_cfg.merge_with_dotlist([
        "host.parent_data=${oc.env:det_data}",
        "host.parent_results=${oc.env:det_models}",
    ])

    # Additional imports for model registry
    for imp in model_cfg.get("additional_imports", []):
        print(f"  Importing: {imp}")
        importlib.import_module(imp)

    resolved = OmegaConf.to_container(model_cfg, resolve=True)
    preprocessed_output_dir = Path(resolved["host"]["preprocessed_output_dir"])

    # Step 1: Preprocess
    if not args.skip_preprocess:
        print("\n>>> Step 1: Preprocessing...")
        from nndet.planning import PLANNER_REGISTRY
        splitted_dir = resolved["host"]["splitted_4d_output_dir"]

        planner_cls = PLANNER_REGISTRY.get(plan["planner_id"])
        t0 = time.time()
        planner_cls.run_preprocessing_test(
            preprocessed_output_dir=preprocessed_output_dir,
            splitted_4d_output_dir=splitted_dir,
            plan=plan,
            num_processes=1,
        )
        print(f"  Preprocessing done in {time.time()-t0:.1f}s")
    else:
        print("\n>>> Step 1: Skipping preprocessing")

    # Step 2: Predict
    source_dir = preprocessed_output_dir / plan["data_identifier"] / "imagesTs"
    print(f"\n>>> Step 2: Running inference on {args.case_id}...")
    print(f"  Source: {source_dir}")

    t0 = time.time()
    predict_dir(
        source_dir=source_dir,
        target_dir=output_dir,
        cfg=resolved,
        plan=plan,
        source_models=training_dir,
        num_models=1,
        num_tta_transforms=1,
        model_fn=load_all_models,
        restore=True,
        case_ids=[args.case_id],
    )
    elapsed = time.time() - t0

    # Check output
    pkl_files = list(output_dir.glob("*_boxes.pkl"))
    print(f"\n{'=' * 60}")
    print(f"  Test Complete! ({elapsed:.1f}s)")
    print(f"  Output files: {len(pkl_files)}")
    for pf in pkl_files:
        print(f"    {pf.name}")

    if pkl_files:
        import pickle
        with open(pkl_files[0], "rb") as f:
            data = pickle.load(f)
        print(f"\n  Prediction contents:")
        for k, v in data.items():
            if hasattr(v, 'shape'):
                print(f"    {k}: shape={v.shape}")
            else:
                print(f"    {k}: {type(v).__name__}")
        if 'pred_scores' in data:
            print(f"    Score range: [{data['pred_scores'].min():.4f}, {data['pred_scores'].max():.4f}]")
            n_high = (data["pred_scores"] >= 0.5).sum()
            print(f"    Predictions with score >= 0.5: {n_high}")
        # V003: BI-RADS predictions
        if 'pred_birads_probs' in data:
            bp = data['pred_birads_probs']
            bl = data.get('pred_birads_label', bp.argmax())
            names = {0: "BI-RADS 2", 1: "BI-RADS 3", 2: "BI-RADS 4"}
            print(f"    BI-RADS prediction: {names.get(int(bl), bl)}")
            print(f"    BI-RADS probs: {bp}")
    print("=" * 60)


if __name__ == "__main__":
    main()
