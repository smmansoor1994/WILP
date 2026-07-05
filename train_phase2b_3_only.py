"""
train_phase2b_3_only.py — Train Only Phase 2b and/or Phase 3

Simplified training script for Phase 2b (attribute heads) and Phase 3 (hybrid components).
Does NOT require preprocessing, labels, labels_ext, epoch.pts.
Only needs: weights/ and outputs/ folders (pre-trained Phase 2 weights).

Usage:
    python train_phase2b_3_only.py --phase 2b --device cpu
    python train_phase2b_3_only.py --phase 3 --device cpu
    python train_phase2b_3_only.py --phase 2b3 --device cpu  # Both phases

    With custom weights:
    python train_phase2b_3_only.py --phase 2b --weights path/to/weights.pt --device cpu

Output:
    Phase 2b results → outputs/runs/attr/weights/best.pt
    Phase 3 results  → weights/hybrid_best.pt (or outputs/runs/hybrid/weights/best.pt)
"""

import argparse
import logging
import sys
from pathlib import Path

# ─── Project root setup ───────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# ─── Logging configuration ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("archon.train_phase2b_3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ARCHON Phase 2b & 3 Training",
        description="Train only Phase 2b (attribute heads) and/or Phase 3 (hybrid components)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--phase",
        type=str,
        choices=["2b", "3", "2b3"],
        default="2b3",
        help="Training phase: 2b (attributes), 3 (hybrid), or 2b3 (both). Default: 2b3",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/train_config.yaml",
        help="Path to training config (default: config/train_config.yaml)",
    )

    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help=(
            "Path to base Phase 2 weights (.pt). "
            "If not provided, auto-detects from weights/ or outputs/runs/phase2/weights/"
        ),
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device: 'cpu' or 'cuda' (default: cpu)",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint (if exists)",
    )

    return parser.parse_args()


def find_phase2_weights() -> Path:
    """Auto-detect Phase 2 best.pt from standard locations."""
    candidates = [
        PROJECT_ROOT / "weights" / "archon_best.pt",
        PROJECT_ROOT / "outputs" / "runs" / "phase2" / "weights" / "best.pt",
        PROJECT_ROOT / "outputs" / "runs" / "detect" / "weights" / "best.pt",
    ]

    for path in candidates:
        if path.exists():
            logger.info("✓ Found Phase 2 weights: %s", path)
            return path

    logger.error(
        "Phase 2 weights not found. Checked:\n  %s\n"
        "Provide --weights or ensure Phase 2 training completed.",
        "\n  ".join(str(p) for p in candidates),
    )
    sys.exit(1)


def train_phase2b_only(config_path: str, weights_path: str, device: str, resume: bool):
    """Train Phase 2b (attribute heads only)."""
    logger.info("=" * 80)
    logger.info("PHASE 2b — Train Attribute Heads Only")
    logger.info("=" * 80)
    logger.info("Base weights: %s", weights_path)
    logger.info("Config: %s", config_path)
    logger.info("Device: %s", device)

    from src.training.trainer import ARCHONTrainer

    trainer = ARCHONTrainer(
        config_path=config_path,
        resume=resume,
        device=device,
    )
    trainer.train_attr_only(base_weights=weights_path)

    logger.info("=" * 80)
    logger.info("✓ Phase 2b training completed")
    logger.info("  Weights saved to: outputs/runs/attr/weights/best.pt")
    logger.info("=" * 80)

    return PROJECT_ROOT / "outputs" / "runs" / "attr" / "weights" / "best.pt"


def train_phase3_only(config_path: str, weights_path: str, device: str, resume: bool):
    """Train Phase 3 (hybrid components)."""
    logger.info("=" * 80)
    logger.info("PHASE 3 — Train Hybrid Components (Swin + Cross-Attention + Severity)")
    logger.info("=" * 80)
    logger.info("Base weights: %s", weights_path)
    logger.info("Config: %s", config_path)
    logger.info("Device: %s", device)

    from src.training.trainer import ARCHONHybridTrainer

    trainer = ARCHONHybridTrainer(
        config_path=config_path,
        resume=resume,
        device=device,
    )
    trainer.train_hybrid(base_weights=weights_path)

    logger.info("=" * 80)
    logger.info("✓ Phase 3 training completed")
    logger.info("  Weights saved to: outputs/runs/hybrid/weights/best.pt")
    logger.info("=" * 80)

    return PROJECT_ROOT / "outputs" / "runs" / "hybrid" / "weights" / "best.pt"


def copy_weights_to_output(src: Path, dest: Path):
    """Copy trained weights to output weights/ folder."""
    if not src.exists():
        logger.warning("Source weights not found: %s", src)
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy2(src, dest)
    logger.info("✓ Copied weights: %s → %s", src, dest)


def main():
    args = parse_args()

    # Validate config exists
    config_path = PROJECT_ROOT / args.config
    if not config_path.exists():
        logger.error("Config not found: %s", config_path)
        sys.exit(1)

    # Auto-detect or validate weights path
    base_weights = args.weights or str(find_phase2_weights())
    if not Path(base_weights).exists():
        logger.error("Weights file not found: %s", base_weights)
        sys.exit(1)

    logger.info("Starting ARCHON Phase 2b/3 Training")
    logger.info("─" * 80)
    logger.info("Phase: %s", args.phase)
    logger.info("Base weights: %s", base_weights)
    logger.info("Config: %s", args.config)
    logger.info("Device: %s", args.device)
    logger.info("Resume: %s", args.resume)
    logger.info("─" * 80)

    phase2b_weights = None
    phase3_weights = None

    # ─── Phase 2b: Train attribute heads ───────────────────────────────────────
    if args.phase in ["2b", "2b3"]:
        phase2b_weights = train_phase2b_only(
            config_path=str(config_path),
            weights_path=base_weights,
            device=args.device,
            resume=args.resume,
        )
        # Copy to weights/ folder
        copy_weights_to_output(
            phase2b_weights,
            PROJECT_ROOT / "weights" / "attr_best.pt"
        )

    # ─── Phase 3: Train hybrid components ──────────────────────────────────────
    if args.phase in ["3", "2b3"]:
        # For Phase 3, use Phase 2b weights if just trained, else Phase 2
        phase3_base_weights = (
            str(phase2b_weights) if phase2b_weights else base_weights
        )
        phase3_weights = train_phase3_only(
            config_path=str(config_path),
            weights_path=phase3_base_weights,
            device=args.device,
            resume=args.resume,
        )
        # Copy to weights/ folder
        copy_weights_to_output(
            phase3_weights,
            PROJECT_ROOT / "weights" / "hybrid_best.pt"
        )

    # ─── Summary ───────────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 80)
    logger.info("✓ TRAINING COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)

    if phase2b_weights:
        logger.info("Phase 2b (Attributes):")
        logger.info("  Output: %s", phase2b_weights)
        logger.info("  Weights: %s", PROJECT_ROOT / "weights" / "attr_best.pt")

    if phase3_weights:
        logger.info("Phase 3 (Hybrid):")
        logger.info("  Output: %s", phase3_weights)
        logger.info("  Weights: %s", PROJECT_ROOT / "weights" / "hybrid_best.pt")

    logger.info("")
    logger.info("Next steps:")
    logger.info("  Validate: python validate_with_groundtruth.py")
    logger.info("  Infer:    python verify_local_image.py --image <path> --weights weights/hybrid_best.pt")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
