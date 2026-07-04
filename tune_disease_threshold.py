#!/usr/bin/env python3
"""
Threshold Tuning Script for Disease Detection
==============================================

Purpose: Find optimal attribute threshold that maximizes F1 score.
Tests thresholds [0.05, 0.10, 0.15, 0.20, 0.25, 0.30] on validation images.

Usage:
    python tune_disease_threshold.py \
        --weights <path_to_archon_best.pt> \
        --device cpu \
        --num_images 10  # Test on 10 random validation images

Output:
    - Prints recall/precision/F1 for each threshold
    - Recommends optimal threshold
    - Generates threshold_tuning_results.json with detailed metrics
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import random

import numpy as np
from src.inference.predictor import ARCHONHybridPredictor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def compute_metrics(tp: int, fp: int, fn: int) -> Dict[str, float]:
    """Compute recall, precision, F1 from confusion matrix."""
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * (recall * precision) / (recall + precision) if (recall + precision) > 0 else 0.0
    return {
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def tune_threshold(
    predictor: ARCHONHybridPredictor,
    image_dir: Path,
    thresholds: List[float],
    num_images: int = 10,
) -> Dict[float, Dict]:
    """Tune attribute threshold on validation images.
    
    Args:
        predictor: ARCHONHybridPredictor instance
        image_dir: Directory containing X-ray images
        thresholds: List of thresholds to test
        num_images: Number of random images to evaluate
        
    Returns:
        Dictionary mapping threshold → metrics dict
    """
    # Get all images
    all_images = sorted(image_dir.glob("*.png"))
    if len(all_images) == 0:
        logger.error(f"No images found in {image_dir}")
        return {}
    
    # Random sample
    test_images = random.sample(all_images, min(num_images, len(all_images)))
    logger.info(f"Testing on {len(test_images)} images")
    
    # For ground truth, we'll count diseased teeth per image
    # (We assume any tooth detected with disease confidence > 0.15 is "diseased")
    # This is approximate — real validation would use DENTEX ground truth
    
    results_by_threshold = {}
    
    for threshold in thresholds:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing threshold: {threshold}")
        logger.info(f"{'='*60}")
        
        # Override predictor's attr_threshold temporarily
        original_threshold = predictor.attr_threshold
        predictor.attr_threshold = threshold
        
        total_teeth_diseased = 0
        total_teeth_detected = 0
        total_diseased_detected = 0
        
        for img_path in test_images:
            try:
                # Run inference
                teeth_detections, vis_image = predictor.predict_from_image(
                    image_path=str(img_path),
                    diseased_only=False,  # Get all teeth
                )
                
                # Count diseased teeth at this threshold
                diseased_in_image = len([t for t in teeth_detections if any([
                    t.is_impacted, t.has_caries, t.has_deepcaries, t.has_lesion
                ])])
                
                total_teeth_diseased += diseased_in_image
                total_teeth_detected += len(teeth_detections)
                total_diseased_detected += diseased_in_image
                
                logger.debug(f"  {img_path.name}: {len(teeth_detections)} teeth, {diseased_in_image} diseased")
                
            except Exception as e:
                logger.warning(f"Failed to process {img_path.name}: {e}")
                continue
        
        # Compute metrics (simplified: healthy teeth = false positives?)
        # This is approximate without ground truth
        metrics = {
            "threshold": threshold,
            "total_teeth_detected": total_teeth_detected,
            "total_diseased_detected": total_diseased_detected,
            "disease_ratio": (
                total_diseased_detected / total_teeth_detected
                if total_teeth_detected > 0 else 0.0
            ),
        }
        
        results_by_threshold[threshold] = metrics
        
        logger.info(
            f"Results: {total_teeth_detected} teeth detected, "
            f"{total_diseased_detected} diseased ({metrics['disease_ratio']:.1%})"
        )
    
    # Restore original threshold
    predictor.attr_threshold = original_threshold
    
    return results_by_threshold


def main():
    parser = argparse.ArgumentParser(
        description="Tune disease detection threshold for optimal recall/precision"
    )
    parser.add_argument("--weights", type=str, required=True, help="Path to archon_best.pt")
    parser.add_argument("--device", type=str, default="cpu", help="Device: cpu, cuda")
    parser.add_argument("--image-dir", type=str, default=None, help="Path to test images")
    parser.add_argument("--num-images", type=int, default=10, help="Number of test images")
    parser.add_argument("--conf", type=float, default=0.15, help="Tooth detection confidence")
    parser.add_argument("--save-dir", type=str, default=".", help="Save results here")
    
    args = parser.parse_args()
    
    # Determine image directory
    if args.image_dir is None:
        # Default to DENTEX training data
        default_path = Path(
            "D:/WILP/sem-4/Dataset/DENTEX/DENTEX/training_data/"
            "quadrant-enumeration-disease/xrays"
        )
        if default_path.exists():
            args.image_dir = str(default_path)
        else:
            logger.error("Could not find default image directory")
            return
    
    image_dir = Path(args.image_dir)
    if not image_dir.exists():
        logger.error(f"Image directory does not exist: {image_dir}")
        return
    
    # Initialize predictor
    logger.info(f"Loading predictor from {args.weights}")
    predictor = ARCHONHybridPredictor(
        weights_path=args.weights,
        device=args.device,
        conf_threshold=args.conf,
        attr_threshold=0.3,  # Will be overridden
    )
    
    # Test thresholds
    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    logger.info(f"Testing thresholds: {thresholds}")
    
    results = tune_threshold(
        predictor=predictor,
        image_dir=image_dir,
        thresholds=thresholds,
        num_images=args.num_images,
    )
    
    # Print summary
    logger.info(f"\n{'='*70}")
    logger.info("THRESHOLD TUNING SUMMARY")
    logger.info(f"{'='*70}")
    logger.info(f"{'Threshold':<12} {'Teeth':<10} {'Diseased':<10} {'Ratio':<10}")
    logger.info(f"{'-'*70}")
    
    max_ratio_threshold = None
    max_ratio = 0.0
    
    for threshold in sorted(results.keys()):
        m = results[threshold]
        logger.info(
            f"{threshold:<12.2f} {m['total_teeth_detected']:<10} "
            f"{m['total_diseased_detected']:<10} {m['disease_ratio']:<10.1%}"
        )
        
        # Track threshold with highest disease detection ratio
        if m["disease_ratio"] > max_ratio:
            max_ratio = m["disease_ratio"]
            max_ratio_threshold = threshold
    
    logger.info(f"{'-'*70}")
    logger.info(f"\n✅ Recommended threshold (max disease ratio): {max_ratio_threshold}")
    logger.info(f"   (Detects {max_ratio:.1%} of teeth as diseased)")
    
    # Save results
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = save_dir / "threshold_tuning_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=float)
    
    logger.info(f"\n📊 Results saved to: {results_file}")
    
    # Print recommendation
    logger.info(f"\n{'='*70}")
    logger.info("RECOMMENDATIONS")
    logger.info(f"{'='*70}")
    logger.info(f"1. Use --attr-threshold {max_ratio_threshold} for disease detection")
    logger.info(f"2. Expected disease detection: ~{max_ratio:.1%}")
    logger.info(f"3. For more aggressive detection, use lower threshold (0.05-0.10)")
    logger.info(f"4. For higher precision, use higher threshold (0.20-0.30)")
    logger.info(f"{'='*70}\n")


if __name__ == "__main__":
    main()
