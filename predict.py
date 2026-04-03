"""
3D ABUS Lesion Detection Inference: Raw 3D data → Prediction pkl files.

This script wraps the inference pipeline into a single entry point.
It handles: environment setup → preprocessing → model loading → prediction.

Requirements:
  - Detection framework installed (pip install -e ./framework/)
  - Model files in training_dir: config.yaml, plan_inference.pkl, *.ckpt
  - Raw test data: imagesTs/case_XXXXX_0000.nii.gz

Usage:
    python predict.py --config config.json
    python predict.py --config config.json --no_preprocess
    python predict.py --training_dir /path/to/fold0__0 --test_data /path/to/imagesTs
"""

import argparse
import importlib
import json
import os
import pickle
import sys
from pathlib import Path


def setup_env(cfg):
    """Set environment variables from config.
    Resolves relative paths (e.g., './models') relative to the script directory.
    """
    script_dir = Path(__file__).parent
    env = cfg.get("env", {})
    for key, val in env.items():
        if key.startswith("_"):
            continue
        val = str(val)
        # Resolve relative paths for directory-type env vars
        if key in ("det_data", "det_models") and not os.path.isabs(val):
            val = str(script_dir / val)
        os.environ[key] = val


def find_training_dir(cfg):
    """Locate the training directory from config."""
    pred_cfg = cfg["predict"]
    task = pred_cfg["task"]
    model = pred_cfg["model"]
    fold = pred_cfg["fold"]

    det_models = os.environ.get("det_models", "")
    base = Path(det_models) / task / model

    if fold == -1:
        fold_dir = base / "consolidated"
    else:
        # Find latest fold directory
        candidates = sorted(base.glob(f"fold{fold}__*"))
        if candidates:
            fold_dir = candidates[-1]
        else:
            fold_dir = base / f"fold{fold}__0"

    return fold_dir


def check_files(training_dir):
    """Verify all required files exist."""
    required = ["config.yaml", "plan_inference.pkl"]
    missing = []
    for f in required:
        if not (training_dir / f).exists():
            missing.append(f)

    ckpts = list(training_dir.glob("*.ckpt"))
    if not ckpts:
        missing.append("*.ckpt")

    if missing:
        print(f"\nERROR: Missing files in {training_dir}:")
        for f in missing:
            print(f"  - {f}")
        if "plan_inference.pkl" in missing and (training_dir / "plan.pkl").exists():
            print(f"\nplan.pkl exists. Generate plan_inference.pkl:")
            print(f"  python generate_plan_inference.py --plan {training_dir / 'plan.pkl'}")
        return False

    return True


def run_inference(training_dir, process=True, num_models=None,
                  num_tta_transforms=None, num_processes=3,
                  output_dir=None, test_data_dir=None, case_ids=None):
    """
    Run inference pipeline.

    This replicates scripts/predict.py run() but with explicit paths.

    Args:
        training_dir: path to training fold directory (config.yaml + *.ckpt)
        process: whether to preprocess raw data
        num_models: number of models to ensemble (None = all)
        num_tta_transforms: TTA transforms (None = all, 1 = no TTA)
        num_processes: preprocessing workers
        output_dir: prediction output directory
        test_data_dir: path to raw test data (nii.gz files). If provided,
            symlinks data to the expected location so config.yaml paths work.
    """
    from omegaconf import OmegaConf
    from loguru import logger

    from nndet.planning import PLANNER_REGISTRY
    from nndet.io.load import load_pickle
    from nndet.inference.loading import load_all_models
    from nndet.inference.helper import predict_dir

    # Load plan and config
    plan = load_pickle(training_dir / "plan_inference.pkl")
    cfg = OmegaConf.load(str(training_dir / "config.yaml"))

    # Merge environment paths
    overwrites = [
        f"host.parent_data=${{oc.env:det_data}}",
        f"host.parent_results=${{oc.env:det_models}}",
    ]
    cfg.merge_with_dotlist(overwrites)

    # Additional imports (model registry)
    for imp in cfg.get("additional_imports", []):
        print(f"  Importing: {imp}")
        importlib.import_module(imp)

    # Resolve all paths
    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    preprocessed_output_dir = Path(resolved_cfg["host"]["preprocessed_output_dir"])

    # If user specified test data directory, symlink it to expected location
    if test_data_dir:
        test_data_dir = Path(test_data_dir)
        expected_test_dir = Path(resolved_cfg["host"]["splitted_4d_output_dir"]) / "imagesTs"
        if test_data_dir.resolve() != expected_test_dir.resolve():
            expected_test_dir.parent.mkdir(parents=True, exist_ok=True)
            if expected_test_dir.is_symlink():
                expected_test_dir.unlink()
            if not expected_test_dir.exists():
                expected_test_dir.symlink_to(test_data_dir.resolve())
                print(f"  Linked test data: {test_data_dir} -> {expected_test_dir}")
            else:
                print(f"  Test data dir already exists: {expected_test_dir}")

    # Determine output directory
    if output_dir:
        prediction_dir = Path(output_dir)
    else:
        prediction_dir = training_dir / "test_predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)

    # Setup logger
    logger.remove()
    logger.add(sys.stdout, format="<level>{level} {message}</level>",
               level="INFO", colorize=True)
    logger.add(prediction_dir / "inference.log", level="INFO")

    # Step 1: Preprocess test data
    if process:
        print("\n>>> Preprocessing test data...")
        planner_cls = PLANNER_REGISTRY.get(plan["planner_id"])
        planner_cls.run_preprocessing_test(
            preprocessed_output_dir=preprocessed_output_dir,
            splitted_4d_output_dir=resolved_cfg["host"]["splitted_4d_output_dir"],
            plan=plan,
            num_processes=num_processes,
        )
        print("  Preprocessing complete.")

    # Step 2: Run prediction
    source_dir = preprocessed_output_dir / plan["data_identifier"] / "imagesTs"

    predict_dir(
        source_dir=source_dir,
        target_dir=prediction_dir,
        cfg=OmegaConf.to_container(cfg, resolve=True),
        plan=plan,
        source_models=training_dir,
        num_models=num_models,
        num_tta_transforms=num_tta_transforms,
        model_fn=load_all_models,
        restore=True,
        case_ids=case_ids,
        **resolved_cfg.get("inference_kwargs", {}),
    )

    return prediction_dir


def main():
    parser = argparse.ArgumentParser(
        description="3D ABUS Lesion Detection Inference")
    parser.add_argument("--config", type=str,
                        default=str(Path(__file__).parent / "config.json"))
    parser.add_argument("--training_dir", type=str, default=None,
                        help="Override training directory path")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override prediction output directory")
    parser.add_argument("--test_data", type=str, default=None,
                        help="Path to raw test data dir (nii.gz files). "
                             "Overrides the default data location from config.yaml.")
    parser.add_argument("--no_preprocess", action="store_true",
                        help="Skip data preprocessing")
    parser.add_argument("--num_processes", type=int, default=3,
                        help="Number of preprocessing workers")
    parser.add_argument("--no_tta", action="store_true",
                        help="Disable test-time augmentation (8x faster)")
    parser.add_argument("--case_id", type=str, default=None,
                        help="Only predict a specific case (e.g., case_00000)")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = json.load(f)

    # Setup environment
    setup_env(cfg)

    # Find training directory
    if args.training_dir:
        training_dir = Path(args.training_dir)
    else:
        training_dir = find_training_dir(cfg)

    # Validate
    if not check_files(training_dir):
        sys.exit(1)

    # Determine output
    output_dir = args.output_dir or cfg["paths"].get("predictions_dir")
    if output_dir:
        output_dir = Path(output_dir)
        if not output_dir.is_absolute():
            output_dir = Path(__file__).parent / output_dir

    # Run inference
    num_tta = 1 if args.no_tta else None
    case_ids = [args.case_id] if args.case_id else None
    pred_dir = run_inference(
        training_dir=training_dir,
        process=not args.no_preprocess,
        output_dir=output_dir,
        num_processes=args.num_processes,
        num_tta_transforms=num_tta,
        test_data_dir=args.test_data,
        case_ids=case_ids,
    )

    n_files = len(list(pred_dir.glob("*_boxes.pkl")))
    print(f"\nDone! {n_files} cases predicted -> {pred_dir}")


if __name__ == "__main__":
    main()
