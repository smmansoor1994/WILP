"""
tune_thresholds.py — Find optimal confidence & disease thresholds

Tests combinations of:
- Tooth detection confidence: 0.10, 0.15, 0.20, 0.25, 0.30
- Disease attribute threshold: 0.15, 0.20, 0.25, 0.30

Runs validation on training images and recommends best combination.

Usage:
    python tune_thresholds.py --num-images 50 --weights weights/hybrid_best.pt --device cpu
    python tune_thresholds.py --num-images 100 --report results.json
"""

import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("threshold_tuning")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tune confidence and disease thresholds for ARCHON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=50,
        help="Number of training images to test (default: 50)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="weights/hybrid_best.pt",
        help="Path to model weights (default: weights/hybrid_best.pt)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device: cpu or cuda (default: cpu)",
    )
    parser.add_argument(
        "--report",
        type=str,
        default="threshold_tuning_report.json",
        help="Output report file (default: threshold_tuning_report.json)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed/images/train",
        help="Path to training images directory",
    )
    parser.add_argument(
        "--labels-dir",
        type=str,
        default="data/processed/labels_ext/train",
        help="Path to ground truth labels directory",
    )

    return parser.parse_args()


def compute_metrics(
    tp: int, fp: int, fn: int
) -> Dict[str, float]:
    """Compute precision, recall, F1 score."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def run_validation_with_thresholds(
    predictor,
    conf_threshold: float,
    attr_threshold: float,
    num_images: int,
    data_dir: str,
    labels_dir: str,
) -> Tuple[Dict, Dict]:
    """
    Run validation with specific thresholds.
    
    Note: predictor is initialized once and reused.
    Thresholds are updated via predictor attributes.
    
    Returns:
        (teeth_metrics, disease_metrics)
    """
    try:
        import random
        import cv2
        from pathlib import Path
        
        # Update predictor thresholds (don't reload model)
        predictor.conf_threshold = conf_threshold
        predictor.attr_threshold = attr_threshold
        
        # Load ground truth labels
        labels_dir = Path(labels_dir)
        image_dir = Path(data_dir)
        
        # Get random sample of images
        all_label_files = sorted(list(labels_dir.glob("*.txt")))
        sample_files = random.sample(all_label_files, min(num_images, len(all_label_files)))
        
        teeth_tp, teeth_fp, teeth_fn = 0, 0, 0
        disease_tp, disease_fp, disease_fn = 0, 0, 0
        
        processed = 0
        for label_file in sample_files:
            img_name = label_file.stem + ".png"
            img_path = image_dir / img_name
            
            if not img_path.exists():
                continue
            
            # Read ground truth
            gt_teeth = {}
            with open(label_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        fdi = int(parts[0])
                        diseases = [int(d) for d in parts[5:]]
                        gt_teeth[fdi] = diseases
            
            # Run prediction
            try:
                results = predictor.predict(str(img_path), save_vis=False, save_json=False)
                pred_teeth = results.get(img_name, [])
            except Exception as e:
                logger.warning(f"Prediction failed for {img_name}: {e}")
                continue
            
            # Compare: compute TP, FP, FN for teeth and diseases
            pred_fdi_set = {t.fdi for t in pred_teeth}
            gt_fdi_set = set(gt_teeth.keys())
            
            # Teeth metrics
            teeth_tp += len(pred_fdi_set & gt_fdi_set)
            teeth_fp += len(pred_fdi_set - gt_fdi_set)
            teeth_fn += len(gt_fdi_set - pred_fdi_set)
            
            # Disease metrics
            for tooth in pred_teeth:
                fdi = tooth.fdi
                pred_diseases = set(tooth.disease_indices) if hasattr(tooth, 'disease_indices') else set()
                gt_diseases = set(gt_teeth.get(fdi, []))
                
                disease_tp += len(pred_diseases & gt_diseases)
                disease_fp += len(pred_diseases - gt_diseases)
            
            for fdi, diseases in gt_teeth.items():
                if fdi not in pred_fdi_set:
                    disease_fn += len(diseases)
                else:
                    pred_diseases = set()
                    for t in pred_teeth:
                        if t.fdi == fdi:
                            pred_diseases = set(t.disease_indices) if hasattr(t, 'disease_indices') else set()
                            break
                    disease_fn += len(set(diseases) - pred_diseases)
            
            processed += 1
            if processed % 10 == 0:
                logger.info(f"  Processed: {processed}/{len(sample_files)}")
        
        teeth_metrics = compute_metrics(teeth_tp, teeth_fp, teeth_fn)
        disease_metrics = compute_metrics(disease_tp, disease_fp, disease_fn)
        
        return teeth_metrics, disease_metrics
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return {}, {}


def main():
    from src.inference.predictor import ARCHONHybridPredictor
    from pathlib import Path
    
    args = parse_args()
    
    logger.info("=" * 80)
    logger.info("ARCHON Threshold Tuning")
    logger.info("=" * 80)
    
    # Verify weights exist
    weights_path = Path(args.weights)
    if not weights_path.exists():
        logger.error(f"Weights not found: {weights_path}")
        import sys
        sys.exit(1)
    
    # Load predictor ONCE
    logger.info(f"Loading model once: {args.weights}")
    try:
        predictor = ARCHONHybridPredictor(
            weights_path=str(weights_path),
            device=args.device,
            conf_threshold=0.15,  # Default, will be updated per test
            iou_threshold=0.45,
            attr_threshold=0.20,  # Default, will be updated per test
        )
        logger.info("✓ Model loaded successfully\n")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        import sys
        sys.exit(1)
    
    # Threshold combinations to test
    conf_thresholds = [0.10, 0.15, 0.20, 0.25, 0.30]
    attr_thresholds = [0.15, 0.20, 0.25, 0.30]
    
    logger.info(f"Testing combinations:")
    logger.info(f"  Confidence thresholds: {conf_thresholds}")
    logger.info(f"  Disease thresholds: {attr_thresholds}")
    logger.info(f"  Images per test: {args.num_images}")
    logger.info(f"  Total combinations: {len(conf_thresholds) * len(attr_thresholds)}")
    logger.info("")
    
    results = {
        "config": {
            "num_images": args.num_images,
            "weights": args.weights,
            "device": args.device,
            "confidence_thresholds": conf_thresholds,
            "disease_thresholds": attr_thresholds,
        },
        "results": [],
    }
    
    best_perfect = {"score": 0, "params": None}
    best_precision = {"score": 0, "params": None}
    best_f1 = {"score": 0, "params": None}
    
    for conf in conf_thresholds:
        for attr_thresh in attr_thresholds:
            logger.info(f"Testing: conf={conf:.2f}, attr_threshold={attr_thresh:.2f}")
            
            # Run validation with reused predictor
            teeth_metrics, disease_metrics = run_validation_with_thresholds(
                predictor=predictor,
                conf_threshold=conf,
                attr_threshold=attr_thresh,
                num_images=args.num_images,
                data_dir=args.data_dir,
                labels_dir=args.labels_dir,
            )
            
            if not teeth_metrics:
                logger.warning(f"  Skipped (validation error)")
                continue
            
            # Compute combined score (both precision and F1)
            combined_score = (
                teeth_metrics["precision"] * 0.3 +  # 30% weight: tooth precision
                teeth_metrics["recall"] * 0.2 +      # 20% weight: tooth recall
                disease_metrics["precision"] * 0.25 +  # 25% weight: disease precision
                disease_metrics["recall"] * 0.25     # 25% weight: disease recall
            )
            
            result = {
                "conf_threshold": conf,
                "attr_threshold": attr_thresh,
                "teeth": teeth_metrics,
                "disease": disease_metrics,
                "combined_score": combined_score,
            }
            results["results"].append(result)
            
            # Log results
            logger.info(
                f"  Teeth: Prec={teeth_metrics['precision']:.1%}, Rec={teeth_metrics['recall']:.1%}, F1={teeth_metrics['f1']:.3f}"
            )
            logger.info(
                f"  Disease: Prec={disease_metrics['precision']:.1%}, Rec={disease_metrics['recall']:.1%}, F1={disease_metrics['f1']:.3f}"
            )
            logger.info(f"  Combined Score: {combined_score:.4f}\n")
            
            # Track best results
            if teeth_metrics["precision"] > best_precision["score"]:
                best_precision = {
                    "score": teeth_metrics["precision"],
                    "params": (conf, attr_thresh),
                    "teeth": teeth_metrics,
                    "disease": disease_metrics,
                }
            
            if disease_metrics["f1"] > best_f1["score"]:
                best_f1 = {
                    "score": disease_metrics["f1"],
                    "params": (conf, attr_thresh),
                    "teeth": teeth_metrics,
                    "disease": disease_metrics,
                }
    
    # Sort results by combined score
    results["results"].sort(key=lambda x: x["combined_score"], reverse=True)
    results["best_combinations"] = {
        "by_tooth_precision": {
            "params": {"conf": best_precision["params"][0], "attr_threshold": best_precision["params"][1]},
            "metrics": {
                "teeth": best_precision["teeth"],
                "disease": best_precision["disease"],
            },
        } if best_precision["params"] else None,
        "by_disease_f1": {
            "params": {"conf": best_f1["params"][0], "attr_threshold": best_f1["params"][1]},
            "metrics": {
                "teeth": best_f1["teeth"],
                "disease": best_f1["disease"],
            },
        } if best_f1["params"] else None,
    }
    
    # Top 5 overall
    results["top_5_combinations"] = [
        {
            "conf": r["conf_threshold"],
            "attr_threshold": r["attr_threshold"],
            "combined_score": r["combined_score"],
            "teeth_precision": r["teeth"]["precision"],
            "disease_precision": r["disease"]["precision"],
        }
        for r in results["results"][:5]
    ]
    
    # Save report
    output_path = Path(args.report)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info("=" * 80)
    logger.info("TUNING RESULTS")
    logger.info("=" * 80)
    logger.info(f"\nReport saved to: {output_path}\n")
    
    logger.info("TOP 5 BEST COMBINATIONS:")
    logger.info("-" * 80)
    for i, combo in enumerate(results["top_5_combinations"], 1):
        logger.info(
            f"{i}. conf={combo['conf']:.2f}, attr_threshold={combo['attr_threshold']:.2f} "
            f"(Score: {combo['combined_score']:.4f})"
        )
        logger.info(
            f"   Teeth Precision: {combo['teeth_precision']:.1%}, "
            f"Disease Precision: {combo['disease_precision']:.1%}"
        )
    
    logger.info("\n" + "=" * 80)
    logger.info("RECOMMENDATION")
    logger.info("=" * 80)
    if results["top_5_combinations"]:
        best = results["top_5_combinations"][0]
        logger.info(
            f"Use: --conf {best['conf']:.2f} --attr-threshold {best['attr_threshold']:.2f}"
        )
        logger.info(f"Combined Score: {best['combined_score']:.4f}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
