"""
main.py - improvements
=======
YOLOrtho: A Unified Framework for Teeth Enumeration and Dental Disease Detection.

Paper:   https://arxiv.org/abs/2308.05967
Dataset: https://www.kaggle.com/competitions/dentex-challenge-2023

Pipeline Modes:
  preprocess    → Convert COCO JSON annotations to YOLO format
  pseudo_label  → Generate pseudo-labels for healthy teeth (Part 3) and unlabelled images
  train         → Train YOLOrtho (2-phase: detection + attributes)
  evaluate      → Evaluate model on validation set (mAP metrics)
  predict       → Run inference on image(s) with visualization
  full          → Run entire pipeline end-to-end (no download — dataset must be on disk)

Usage Examples:
  # ── Step-by-step ──
  python main.py --mode preprocess --dentex-root D:/path/to/DENTEX
  python main.py --mode pseudo_label
  python main.py --mode train
  python main.py --mode evaluate --weights weights/yolortho_best.pt
  python main.py --mode predict --input data/processed/images/test/sample.jpg

  # ── Full pipeline ──
  python main.py --mode full --dentex-root D:/path/to/DENTEX

  # ── Resume training ──
  python main.py --mode train --resume

  # ── Predict on a directory ──
  python main.py --mode predict --input path/to/xrays/ --weights weights/yolortho_best.pt
"""

import argparse
import logging
import sys
from pathlib import Path

# ─── Project root setup ───────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# The log FileHandler below is created at import time and writes to
# outputs/yolortho.log, so that directory must exist first. On a fresh checkout
# (e.g. a clean Colab clone) outputs/ does not yet exist → without this the
# program crashes before any --mode can run.
(PROJECT_ROOT / "outputs").mkdir(parents=True, exist_ok=True)

# ─── Logging configuration ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / "outputs" / "yolortho.log", mode="a"),
    ],
)
logger = logging.getLogger("yolortho.main")


# ─── Argument Parser ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="YOLOrtho",
        description="YOLOrtho: Unified Teeth Enumeration + Dental Disease Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["preprocess", "pseudo_label", "train", "train_attr",
                 "evaluate", "predict", "full"],
        default="full",
        help="Pipeline stage to execute (default: full — runs all stages).",
    )
    parser.add_argument(
        "--dentex-root",
        type=str,
        default=None,
        dest="dentex_root",
        help=(
            "Path to the DENTEX dataset root folder. Required for --mode preprocess "
            "and --mode full. Must contain training_data/, validation_data/, test_data/. "
            "Example: --dentex-root D:/WILP/sem-4/Dataset/DENTEX/DENTEXsample"
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/train_config.yaml",
        help="Path to training configuration YAML (default: config/train_config.yaml).",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to model weights (.pt) for evaluate/predict modes.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input image path or directory for predict mode.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/predictions",
        help="Output directory for prediction results (default: outputs/predictions).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Compute device: 'cuda', 'cuda:0', 'cpu' (default: cuda).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from last checkpoint (for --mode train).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Detection confidence threshold for predict/evaluate (default: 0.25).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        dest="val_ratio",
        help="Kept for backward compatibility. Validation set is now the official validation_triple.json.",
    )

    return parser.parse_args()


# ─── Pipeline Stage Functions ──────────────────────────────────────────────────

def stage_preprocess(args: argparse.Namespace) -> None:
    """Stage 1: Convert COCO JSON annotations to YOLO extended format.

    Reads the DENTEX dataset from --dentex-root and writes processed
    YOLO-format labels to data/processed/.

    Data sources:
      training_data/quadrant/              → data_type=0 (quadrant only) → train/
      training_data/quadrant_enumeration/  → data_type=1 (FDI numbers)  → train/
      training_data/quadrant-enumeration-disease/  → data_type=2 (+ disease) → train/
      validation_data/validation_triple.json       → data_type=2          → val/
      test_data/disease/input/             → images only               → test/
      training_data/unlabelled/xrays/      → copied to data/unlabelled/ for pseudo labeling
    """
    _section_header("STAGE 1 — Preprocess Data to YOLO Format")

    if args.dentex_root is None:
        logger.error(
            "--dentex-root is required for preprocess mode. "
            "Example: --dentex-root D:/WILP/sem-4/Dataset/DENTEX/DENTEXsample"
        )
        sys.exit(1)

    from src.data.preprocess import preprocess_dentex
    preprocess_dentex(
        dentex_root=Path(args.dentex_root),
        out_dir=PROJECT_ROOT / "data" / "processed",
    )


def stage_pseudo_label(args: argparse.Namespace) -> None:
    """Stage 2: Generate pseudo-labels for healthy teeth.

    Processes two sets of images:
      1. Part 3 training images — detections not overlapping disease labels
         are added as healthy pseudo labels.
      2. Unlabelled images (data/unlabelled/) — all detected teeth become
         healthy pseudo labels.

    Requires Phase 1 training to be complete first (--mode train).
    """
    _section_header("STAGE 2 — Generate Pseudo Labels for Healthy Teeth")
    logger.info("This requires Phase 1 model weights (run --mode train first if not done).")

    from src.data.pseudo_label import generate_pseudo_labels
    generate_pseudo_labels(
        data_dir=PROJECT_ROOT / "data" / "processed",
        pseudo_dir=PROJECT_ROOT / "data" / "pseudo",
        unlabelled_dir=PROJECT_ROOT / "data" / "unlabelled",
        weights_path=_find_phase1_weights(),
        device=args.device,
    )


def stage_train_attr(args: argparse.Namespace) -> None:
    """Stage 4b: Re-train attribute heads only (Phase 2b).

    Use this when attr_best.pt is missing or was trained with the no_grad bug
    (all disease predictions show ~0.001 probability / everything healthy).

    Requires:
      - Phase 2 best.pt backbone weights (or supply --weights)
      - data/processed/images/train/  populated with training images
      - data/processed/labels_ext/train/  populated (disease labels)
    """
    _section_header("STAGE 4b — Train Attribute Heads Only (Phase 2b)")
    logger.info("Config: %s", args.config)
    logger.info("Device: %s", args.device)

    from src.training.trainer import YOLOrthoTrainer
    trainer = YOLOrthoTrainer(
        config_path=args.config,
        resume=False,
        device=args.device,
    )
    # If user supplied --weights, use that as the backbone; otherwise auto-detect
    base_weights = args.weights if args.weights else None
    trainer.train_attr_only(base_weights=base_weights)


def stage_train(args: argparse.Namespace) -> None:
    """Stage 4: Train YOLOrtho with hierarchical loss.

    Two-phase training:
      Phase 1 — YOLOv8x + CoordConv on Parts 1+2 (detection only)
      Phase 2 — Fine-tune with attribute heads on all data + pseudo labels

    Total loss = bbox(7.5) + cls(0.5) + DFL(1.5) + attr×4(8.0 each)
    """
    _section_header("STAGE 4 — Train YOLOrtho")
    logger.info("Config: %s", args.config)
    logger.info("Device: %s", args.device)
    logger.info("Resume: %s", args.resume)

    from src.training.trainer import YOLOrthoTrainer
    trainer = YOLOrthoTrainer(
        config_path=args.config,
        resume=args.resume,
        device=args.device,
    )
    trainer.train()


def stage_evaluate(args: argparse.Namespace) -> None:
    """Stage 5: Evaluate the trained model on the validation set.

    Reports standard COCO detection metrics:
      AP-Quadrant, AP-Enumeration, AP-Diagnosis (as per Dentex challenge)
      mAP@0.5, mAP@0.5:0.95
    """
    _section_header("STAGE 5 — Evaluate YOLOrtho")

    weights = args.weights or str(PROJECT_ROOT / "weights" / "yolortho_best.pt")
    if not Path(weights).exists():
        logger.error(
            "Weights not found at '%s'. "
            "Run --mode train or provide --weights path.",
            weights,
        )
        sys.exit(1)

    logger.info("Using weights: %s", weights)

    from src.inference.predictor import YOLOrthoPredictor
    predictor = YOLOrthoPredictor(
        weights_path=weights,
        device=args.device,
        conf_threshold=args.conf,
    )
    metrics = predictor.evaluate(
        data_yaml=str(PROJECT_ROOT / "config" / "dataset.yaml"),
        split="val",
    )

    logger.info("Evaluation Results:")
    for k, v in metrics.items():
        logger.info("  %s: %.4f", k, v)


def stage_predict(args: argparse.Namespace) -> None:
    """Stage 6: Run inference on image(s) and save annotated results.

    Outputs per image:
      <stem>_vis.jpg   — Annotated panoramic X-ray with bounding boxes + FDI labels
      <stem>_result.json — Structured JSON with tooth detections + disease attributes
    """
    _section_header("STAGE 6 — Predict (Inference)")

    if args.input is None:
        logger.error("--input argument required for predict mode.")
        sys.exit(1)

    weights = args.weights
    if weights is None:
        weights = str(PROJECT_ROOT / "weights" / "yolortho_best.pt")
        logger.warning("No --weights specified. Using default: '%s'", weights)

    if not Path(weights).exists():
        logger.error(
            "Weights not found at '%s'. Provide --weights or run --mode train.",
            weights,
        )
        sys.exit(1)

    logger.info("Input:   %s", args.input)
    logger.info("Weights: %s", weights)
    logger.info("Output:  %s", args.output)

    from src.inference.predictor import YOLOrthoPredictor
    predictor = YOLOrthoPredictor(
        weights_path=weights,
        device=args.device,
        conf_threshold=args.conf,
    )
    results = predictor.predict(
        input_path=args.input,
        output_dir=args.output,
        save_json=True,
        save_vis=True,
    )

    # Print summary for each processed image
    for img_name, teeth in results.items():
        n_total = len(teeth)
        n_diseased = sum(1 for t in teeth if t.diseases)
        logger.info(
            "  %s → %d teeth detected (%d healthy, %d diseased)",
            img_name,
            n_total,
            n_total - n_diseased,
            n_diseased,
        )


# ─── Full Pipeline ────────────────────────────────────────────────────────────

def run_full_pipeline(args: argparse.Namespace) -> None:
    """Execute the complete YOLOrtho pipeline end-to-end.

    Requires --dentex-root pointing to the DENTEX dataset folder.

    Order:
      1. Preprocess dataset to YOLO format
      2. Train Phase 1 (detection only, needed for pseudo labeling)
      3. Generate pseudo labels (Part 3 healthy + unlabelled images)
      4. Train Phase 2 (full model with disease attributes)
      5. Evaluate on validation set
    """
    logger.info("=" * 70)
    logger.info("  YOLOrtho FULL PIPELINE")
    logger.info("  Paper: https://arxiv.org/abs/2308.05967")
    logger.info("  Dataset: DENTEX (on disk at --dentex-root)")
    logger.info("=" * 70)

    if args.dentex_root is None:
        logger.error(
            "--dentex-root is required for full pipeline. "
            "Example: --dentex-root D:/WILP/sem-4/Dataset/DENTEX/DENTEXsample"
        )
        sys.exit(1)

    stage_preprocess(args)

    # Phase 1 only: train detector on Parts 1+2 (needed to produce pseudo labels).
    # Phase 2 is intentionally skipped here so pseudo labels can be generated first.
    from src.training.trainer import YOLOrthoTrainer
    trainer = YOLOrthoTrainer(
        config_path=args.config,
        resume=args.resume,
        device=args.device,
    )
    trainer.train_phase1_only()

    stage_pseudo_label(args)

    # Full training: Phase 1 weights exist → skipped automatically.
    # Phase 2 + attribute heads trained on all data including pseudo labels.
    stage_train(args)
    stage_evaluate(args)

    logger.info("=" * 70)
    logger.info("  Full pipeline complete.")
    logger.info("  Best weights: weights/yolortho_best.pt")
    logger.info("  Predictions:  outputs/predictions/")
    logger.info("=" * 70)


# ─── Utilities ────────────────────────────────────────────────────────────────

def _section_header(title: str) -> None:
    logger.info("")
    logger.info("=" * 60)
    logger.info("  %s", title)
    logger.info("=" * 60)


def _find_phase1_weights() -> Path:
    """Try to locate Phase 1 model weights."""
    candidates = [
        PROJECT_ROOT / "outputs" / "runs" / "phase1" / "weights" / "best.pt",
        PROJECT_ROOT / "weights" / "phase1_best.pt",
    ]
    for p in candidates:
        if p.exists():
            return p
    # Also search recursively
    found = sorted(
        (PROJECT_ROOT / "outputs" / "runs").rglob("best.pt")
        if (PROJECT_ROOT / "outputs" / "runs").exists() else [],
        key=lambda x: x.stat().st_mtime,
    )
    return found[-1] if found else None


def _ensure_output_dirs() -> None:
    """Create required output directories."""
    for d in [
        "outputs/runs",
        "outputs/predictions",
        "weights",
    ]:
        (PROJECT_ROOT / d).mkdir(parents=True, exist_ok=True)

    # Ensure log output dir exists
    (PROJECT_ROOT / "outputs").mkdir(parents=True, exist_ok=True)


# ─── Entry Point ──────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    _ensure_output_dirs()

    logger.info("YOLOrtho — Mode: %s | Device: %s", args.mode, args.device)

    dispatch = {
        "preprocess":   stage_preprocess,
        "pseudo_label": stage_pseudo_label,
        "train":        stage_train,
        "train_attr":   stage_train_attr,
        "evaluate":     stage_evaluate,
        "predict":      stage_predict,
        "full":         run_full_pipeline,
    }

    stage_fn = dispatch.get(args.mode)
    if stage_fn is None:
        logger.error("Unknown mode: '%s'", args.mode)
        sys.exit(1)

    try:
        stage_fn(args)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logger.exception("Pipeline failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
