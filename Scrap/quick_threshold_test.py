"""
quick_threshold_test.py — Fast threshold testing (minimal requirements)

Tests a few key threshold combinations without full validation loop.
Works even if you have limited labeled data.

Usage:
    python quick_threshold_test.py --images 5 --device cpu
    python quick_threshold_test.py --images 20 --device cpu
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("quick_test")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Quick threshold testing with minimal data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--images",
        type=int,
        default=5,
        help="Number of images to test (default: 5)",
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
    from src.inference.predictor import ARCHONHybridPredictor
    from pathlib import Path
    import random
    
    args = parse_args()
    
    # Check weights exist
    weights_path = Path(args.weights)
    if not weights_path.exists():
        logger.error(f"❌ Weights not found: {weights_path}")
        logger.info("Expected to find model weights at 'weights/hybrid_best.pt'")
        return
    
    # Check data exists
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"❌ Data directory not found: {data_dir}")
        logger.info("Expected training images at 'data/processed/images/train/'")
        logger.info("\nTo get data, use one of these options:")
        logger.info("1. python main.py --mode preprocess --dentex-root <DENTEX_PATH>")
        logger.info("2. Copy from existing trained model")
        logger.info("3. Use available test images in your project")
        return
    
    # Find available images
    image_files = sorted(list(data_dir.glob("*.png"))) + sorted(list(data_dir.glob("*.jpg")))
    if not image_files:
        logger.error(f"❌ No images found in {data_dir}")
        return
    
    # Sample images
    sample_images = random.sample(image_files, min(args.images, len(image_files)))
    logger.info(f"Found {len(image_files)} total images, testing {len(sample_images)}\n")
    
    # Test key thresholds
    test_configs = [
        {"conf": 0.10, "attr": 0.15, "name": "Very Permissive"},
        {"conf": 0.15, "attr": 0.20, "name": "Recommended (Start Here)"},
        {"conf": 0.20, "attr": 0.25, "name": "Balanced"},
        {"conf": 0.25, "attr": 0.30, "name": "Strict"},
    ]
    
    logger.info("=" * 80)
    logger.info("QUICK THRESHOLD TEST")
    logger.info("=" * 80)
    logger.info(f"Model: {args.weights}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Testing on {len(sample_images)} images\n")
    
    # Load model once
    logger.info("Loading model...")
    try:
        predictor = ARCHONHybridPredictor(
            weights_path=str(weights_path),
            device=args.device,
            conf_threshold=0.15,
            attr_threshold=0.20,
        )
        logger.info("✓ Model loaded\n")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return
    
    # Test each configuration
    results = []
    for config in test_configs:
        logger.info("-" * 80)
        logger.info(f"Testing: {config['name']}")
        logger.info(f"  Confidence threshold: {config['conf']}")
        logger.info(f"  Disease threshold: {config['attr']}")
        logger.info("-" * 80)
        
        # Update thresholds
        predictor.conf_threshold = config["conf"]
        predictor.attr_threshold = config["attr"]
        
        total_teeth = 0
        total_diseases = 0
        error_count = 0
        
        # Test on sample images
        for img_path in sample_images:
            try:
                result = predictor.predict(str(img_path), save_vis=False, save_json=False)
                img_name = img_path.name
                
                teeth = result.get(img_name, [])
                diseases = sum(len(t.diseases) if hasattr(t, 'diseases') else 0 for t in teeth)
                
                total_teeth += len(teeth)
                total_diseases += diseases
                
            except Exception as e:
                error_count += 1
                logger.warning(f"  ⚠️  Error on {img_path.name}: {str(e)[:50]}")
        
        avg_teeth = total_teeth / len(sample_images) if sample_images else 0
        avg_diseases = total_diseases / len(sample_images) if sample_images else 0
        
        result_data = {
            "name": config["name"],
            "conf": config["conf"],
            "attr": config["attr"],
            "avg_teeth": avg_teeth,
            "avg_diseases": avg_diseases,
            "errors": error_count,
        }
        results.append(result_data)
        
        logger.info(f"  Avg teeth per image: {avg_teeth:.1f}")
        logger.info(f"  Avg diseases per image: {avg_diseases:.1f}")
        if error_count > 0:
            logger.warning(f"  Errors: {error_count}/{len(sample_images)}")
        logger.info("")
    
    # Recommendations
    logger.info("=" * 80)
    logger.info("RECOMMENDATIONS")
    logger.info("=" * 80)
    
    # Find balanced option
    for r in results:
        logger.info(f"\n{r['name']}:")
        logger.info(f"  conf={r['conf']}, attr={r['attr']}")
        logger.info(f"  Avg {r['avg_teeth']:.1f} teeth/img, {r['avg_diseases']:.1f} diseases/img")
        
        if r['avg_diseases'] > r['avg_teeth']:
            logger.warning(f"  ⚠️  Too many diseases per tooth (likely false positives)")
        elif r['avg_diseases'] < r['avg_teeth'] * 0.3:
            logger.info(f"  ✓ Good balance (healthy teeth + detected diseases)")
    
    logger.info("\n" + "=" * 80)
    logger.info("NEXT STEP")
    logger.info("=" * 80)
    logger.info("Based on above results, choose a config and run:")
    logger.info("  python verify_local_image.py --image test.jpg \\")
    logger.info("    --conf <chosen_conf> --attr-threshold <chosen_attr> --device cpu")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
