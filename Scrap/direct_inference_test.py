"""
direct_inference_test.py — Direct inference without full prediction pipeline

Tests model thresholds by directly using the model, bypassing some initialization overhead.
Useful for quick diagnostics when full prediction pipeline has issues.

Usage:
    python direct_inference_test.py --images 3 --device cpu
"""

import argparse
import logging
from pathlib import Path
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("direct_test")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Direct inference test with minimal overhead",
        epilog=__doc__,
    )
    parser.add_argument(
        "--images",
        type=int,
        default=3,
        help="Number of images to test (default: 3)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="weights/hybrid_best.pt",
        help="Path to model weights",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device: cpu or cuda",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed/images/train",
        help="Training images directory",
    )
    return parser.parse_args()


def main():
    import sys
    import random
    from ultralytics import YOLO
    import cv2
    import numpy as np
    
    args = parse_args()
    
    # Check files
    weights_path = Path(args.weights)
    data_dir = Path(args.data_dir)
    
    if not weights_path.exists():
        logger.error(f"❌ Weights not found: {weights_path}")
        return
    
    if not data_dir.exists():
        logger.error(f"❌ Data directory not found: {data_dir}")
        logger.info("Available in current directory:")
        for d in Path(".").iterdir():
            if d.is_dir():
                logger.info(f"  📁 {d.name}/")
        return
    
    # Find images
    image_files = sorted(list(data_dir.glob("*.png"))) + sorted(list(data_dir.glob("*.jpg")))
    if not image_files:
        logger.error(f"❌ No images found in {data_dir}")
        return
    
    sample_images = random.sample(image_files, min(args.images, len(image_files)))
    logger.info(f"Found {len(image_files)} total images, testing {len(sample_images)}\n")
    
    # Load model ONCE
    logger.info(f"Loading model: {weights_path}")
    try:
        model = YOLO(str(weights_path))
        logger.info("✓ Model loaded\n")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        logger.info("\nTroubleshooting:")
        logger.info("1. Check if weights file is corrupted")
        logger.info("2. Try running on CUDA instead: --device cuda")
        logger.info("3. Check if weights is a valid YOLO format")
        return
    
    # Test configurations
    configs = [
        {"conf": 0.10, "iou": 0.45, "name": "Very Permissive"},
        {"conf": 0.15, "iou": 0.45, "name": "Recommended"},
        {"conf": 0.20, "iou": 0.45, "name": "Balanced"},
        {"conf": 0.25, "iou": 0.45, "name": "Strict"},
    ]
    
    logger.info("=" * 80)
    logger.info("DIRECT INFERENCE TEST")
    logger.info("=" * 80)
    logger.info(f"Model: {weights_path}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Testing on {len(sample_images)} images\n")
    
    # Test each config
    for config in configs:
        logger.info("-" * 80)
        logger.info(f"Config: {config['name']}")
        logger.info(f"  Confidence: {config['conf']}")
        logger.info("-" * 80)
        
        total_detections = 0
        error_count = 0
        
        for img_path in sample_images:
            try:
                # Read image
                img = cv2.imread(str(img_path))
                if img is None:
                    logger.warning(f"  Could not read {img_path.name}")
                    error_count += 1
                    continue
                
                # Run inference
                results = model.predict(
                    source=str(img_path),
                    conf=config["conf"],
                    iou=config["iou"],
                    verbose=False,
                )
                
                if results and len(results) > 0:
                    boxes = results[0].boxes
                    num_detections = len(boxes)
                    total_detections += num_detections
                    logger.info(f"  {img_path.name:30} → {num_detections:2} detections")
                else:
                    logger.info(f"  {img_path.name:30} → 0 detections")
                    
            except Exception as e:
                error_count += 1
                logger.warning(f"  ❌ {img_path.name}: {str(e)[:40]}")
        
        avg_detections = total_detections / len(sample_images) if sample_images else 0
        logger.info(f"\n  Average detections: {avg_detections:.1f}/image")
        if error_count > 0:
            logger.warning(f"  Errors: {error_count}/{len(sample_images)}")
        logger.info("")
    
    logger.info("=" * 80)
    logger.info("✓ TEST COMPLETE")
    logger.info("=" * 80)
    logger.info("\nIf you see detections above, the model is working!")
    logger.info("Choose the config with good balance of detections and try:")
    logger.info("  python verify_local_image.py --image <image> --conf 0.20 --device cpu")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
