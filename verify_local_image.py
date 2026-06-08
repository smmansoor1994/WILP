"""
verify_local_image.py
======================
Run inference on a local panoramic dental X-ray image using the trained
YOLOrtho model (phase2/best.pt or yolortho_best.pt) and display results.

Usage:
    python verify_local_image.py --image path/to/xray.jpg
    python verify_local_image.py --image path/to/xray.jpg --weights path/to/best.pt
    python verify_local_image.py --image path/to/xray.jpg --show
    python verify_local_image.py --image path/to/xray.jpg --save-dir C:/Users/You/Downloads

Save outputs:
    By default, results are saved to outputs/predictions/.
    Use --save-dir to choose any local folder (e.g. Downloads).
    Use --no-save to skip saving entirely.

Default weight search order:
  1. weights/yolortho_best.pt        (final trained weights)
  2. outputs/runs/phase2/weights/best.pt
  3. outputs/runs/phase1/weights/best.pt
"""

import argparse
import logging
import sys
from pathlib import Path

# ── Make sure WILP src is on the path ────────────────────────────────────────
WILP_DIR = Path(__file__).resolve().parent
if str(WILP_DIR) not in sys.path:
    sys.path.insert(0, str(WILP_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("verify_local_image")


# ── Weight search ─────────────────────────────────────────────────────────────
CANDIDATE_WEIGHTS = [
    WILP_DIR / "weights" / "yolortho_best.pt",
    WILP_DIR / "outputs" / "runs" / "phase2" / "weights" / "best.pt",
    WILP_DIR / "outputs" / "runs" / "phase1" / "weights" / "best.pt",
]


def find_weights(override: str = None) -> Path:
    if override:
        p = Path(override)
        if not p.exists():
            logger.error("Specified weights not found: %s", p)
            sys.exit(1)
        return p
    for candidate in CANDIDATE_WEIGHTS:
        if candidate.exists():
            return candidate
    logger.error(
        "No model weights found. Searched:\n%s\n"
        "Download the trained weights from Colab or run training first.\n"
        "Pass --weights <path> to specify a custom location.",
        "\n".join(f"  {c}" for c in CANDIDATE_WEIGHTS),
    )
    sys.exit(1)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Verify YOLOrtho detection on a local panoramic X-ray image."
    )
    parser.add_argument(
        "--image", required=True,
        help="Path to a panoramic dental X-ray image (.jpg / .png)."
    )
    parser.add_argument(
        "--weights", default=None,
        help="Path to model weights (.pt). Auto-detected if omitted."
    )
    parser.add_argument(
        "--attr-weights", default=None,
        help="Path to attribute head weights (attr_best.pt). "
             "Auto-searched in weights directory and its parent if omitted."
    )
    parser.add_argument(
        "--conf", type=float, default=0.10,
        help="Detection confidence threshold (default: 0.10). "
             "YOLOrtho has 32 FDI classes so per-class confidence is naturally "
             "lower than binary detectors — use 0.05-0.15 to see all teeth. "
             "Raise toward 0.25 to reduce false positives."
    )
    parser.add_argument(
        "--iou", type=float, default=0.45,
        help="NMS IoU threshold (default: 0.45)."
    )
    parser.add_argument(
        "--attr-threshold", type=float, default=0.3,
        help="Disease attribute probability threshold (default: 0.3). "
             "Lower values increase disease sensitivity."
    )
    parser.add_argument(
        "--attr-mode", default="per_tooth",
        choices=["per_tooth", "global_avg"],
        help="Attribute inference mode (default: per_tooth). "
             "Use 'global_avg' for models trained before June 2026 per-tooth fix. "
             "Use 'per_tooth' for models retrained with the current trainer.py."
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Torch device: 'cpu' or 'cuda' (default: cpu)."
    )
    parser.add_argument(
        "--save-dir", default=None,
        help="Directory to save annotated image and JSON results. "
             "Overrides --output-dir. Example: C:/Users/You/Downloads"
    )
    parser.add_argument(
        "--output-dir", default=str(WILP_DIR / "outputs" / "predictions"),
        help="Fallback save directory if --save-dir is not set "
             "(default: outputs/predictions/)."
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Display the annotated image in a window after inference."
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Skip saving output files entirely."
    )
    args = parser.parse_args()

    # ── Validate image path ───────────────────────────────────────────────────
    img_path = Path(args.image)
    if not img_path.exists():
        logger.error("Image not found: %s", img_path)
        sys.exit(1)

    # ── Resolve save directory ───────────────────────────────────────────────
    effective_save_dir = args.save_dir if args.save_dir else args.output_dir

    # ── Resolve weights ───────────────────────────────────────────────────────
    weights_path = find_weights(args.weights)
    logger.info("Using weights : %s", weights_path)
    logger.info("Image         : %s", img_path)
    logger.info("Device        : %s", args.device)
    if not args.no_save:
        logger.info("Save dir      : %s", effective_save_dir)

    # ── Run inference ─────────────────────────────────────────────────────────
    from src.inference.predictor import YOLOrthoPredictor

    predictor = YOLOrthoPredictor(
        weights_path=str(weights_path),
        device=args.device,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        attr_threshold=args.attr_threshold,
        img_size=(640, 1280),
        attr_weights_path=args.attr_weights,
        attr_inference_mode=args.attr_mode,
    )

    save = not args.no_save
    logger.info("Attr threshold : %.2f  mode=%s", args.attr_threshold, args.attr_mode)

    results = predictor.predict(
        input_path=str(img_path),
        output_dir=effective_save_dir if save else None,
        save_json=save,
        save_vis=save,
    )

    # ── Print per-tooth summary ───────────────────────────────────────────────
    for img_name, teeth in results.items():
        print(f"\n{'='*60}")
        print(f"Image   : {img_name}")
        print(f"Teeth detected: {len(teeth)}")
        print(f"Attr threshold : {args.attr_threshold}")
        print(f"{'='*60}")

        if not teeth:
            print("  No teeth detected. Try lowering --conf threshold.")
            continue

        healthy = [t for t in teeth if not t.diseases]
        diseased = [t for t in teeth if t.diseases]

        print(f"  Healthy  : {len(healthy)}")
        print(f"  Diseased : {len(diseased)}")
        print()

        for tooth in sorted(teeth, key=lambda t: t.fdi):
            disease_str = ", ".join(tooth.diseases) if tooth.diseases else "Healthy"
            # Show raw attribute probabilities for each tooth to aid threshold tuning
            attr_raw = (
                f"  [imp={tooth.is_impacted and 1 or 0}  "
                f"car={tooth.has_caries and 1 or 0}  "
                f"deep={tooth.has_deepcaries and 1 or 0}  "
                f"les={tooth.has_lesion and 1 or 0}]"
            )
            print(
                f"  FDI {tooth.fdi:>2}  ({tooth.fdi_name:<35})  "
                f"conf={tooth.conf:.2f}  {disease_str:<25}{attr_raw}"
            )

    # ── Show image if requested ───────────────────────────────────────────────
    if args.show and results:
        import cv2
        import matplotlib.pyplot as plt

        img_name = list(results.keys())[0]
        out_dir = Path(effective_save_dir)
        stem = Path(img_name).stem
        vis_path = out_dir / f"{stem}_vis.jpg"

        if vis_path.exists():
            img = cv2.imread(str(vis_path))
        else:
            # Fall back: draw on the original
            from src.utils.visualize import draw_teeth_detections
            img = cv2.imread(str(img_path))
            teeth = results.get(img_name, [])
            if img is not None and teeth:
                img = draw_teeth_detections(img, teeth)

        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(16, 8))
            plt.imshow(img_rgb)
            plt.title(f"YOLOrtho — {img_name}")
            plt.axis("off")
            plt.tight_layout()
            plt.show()

    if save and results:
        logger.info("Results saved to: %s", effective_save_dir)


if __name__ == "__main__":
    main()
