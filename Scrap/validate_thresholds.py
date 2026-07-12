"""
validate_thresholds.py — Find optimal thresholds by comparing against ground truth

Tests multiple threshold combinations and compares predictions to ground truth.
Shows which thresholds maximize precision and recall for perfect detection.

Usage:
    python validate_thresholds.py --images 50 --device cpu
"""

import argparse
import json
from pathlib import Path
from ultralytics import YOLO
import cv2


def load_ground_truth():
    """Load DENTEX ground truth annotations."""
    gt_path = Path("data/processed/labels") / "train_quadrant_enumeration_disease.json"
    if not gt_path.exists():
        print(f"⚠️  Ground truth not found at {gt_path}")
        return {}
    
    with open(gt_path) as f:
        return json.load(f)


def fdi_to_tooth_idx(fdi):
    """Convert FDI number to tooth index (0-31)."""
    if fdi < 11 or fdi > 48:
        return None
    quadrant = (fdi // 10) - 1  # 1-4 -> 0-3
    enumeration = (fdi % 10) - 1  # 1-8 -> 0-7
    return quadrant * 8 + enumeration


def validate_thresholds(num_images=50, device="cpu"):
    """Validate different threshold configurations against ground truth."""
    
    # Load model and ground truth
    weights_path = Path("weights/phase2_best.pt")
    model = YOLO(str(weights_path))
    gt = load_ground_truth()
    
    print(f"\n{'='*80}")
    print(f"Validating thresholds using ground truth")
    print(f"{'='*80}")
    print(f"Model: {weights_path}")
    print(f"Ground truth annotations: {len(gt)} images")
    print()
    
    # Find test images
    train_dir = Path("data/processed/images/train")
    images = sorted(train_dir.glob("*.png")) + sorted(train_dir.glob("*.jpg"))
    test_images = images[:num_images]
    
    print(f"Testing on {len(test_images)} images\n")
    
    # Threshold configurations to test
    configs = [
        {"name": "Very Permissive", "conf": 0.10, "attr": 0.15},
        {"name": "Permissive", "conf": 0.15, "attr": 0.20},
        {"name": "Recommended", "conf": 0.20, "attr": 0.25},
        {"name": "Strict", "conf": 0.25, "attr": 0.30},
        {"name": "Very Strict", "conf": 0.30, "attr": 0.35},
    ]
    
    results = {}
    
    for config in configs:
        print(f"Testing: {config['name']} (conf={config['conf']}, attr={config['attr']})")
        
        tp_teeth = 0  # True positive teeth
        fp_teeth = 0  # False positive teeth
        fn_teeth = 0  # False negative teeth
        
        tp_perfect = 0  # True positive perfect detections
        fp_perfect = 0  # False positive perfect
        fn_perfect = 0  # False negative perfect
        
        total_images = 0
        
        for img_path in test_images:
            img_name = img_path.stem  # "train_0", "train_1", etc.
            
            # Get ground truth for this image
            if img_name not in gt:
                continue
            
            total_images += 1
            gt_teeth = {}
            
            # Parse ground truth
            for tooth_info in gt[img_name]:
                fdi = tooth_info.get("FDI")
                if fdi:
                    idx = fdi_to_tooth_idx(fdi)
                    if idx is not None:
                        gt_teeth[idx] = tooth_info
            
            # Run inference
            try:
                results_raw = model.predict(
                    source=str(img_path),
                    conf=config["conf"],
                    verbose=False,
                    device=device,
                )
                
                if not results_raw or len(results_raw) == 0:
                    fn_teeth += len(gt_teeth)
                    continue
                
                # Parse predictions
                pred_teeth = {}
                for box in results_raw[0].boxes:
                    # YOLO format: class 0-31 = tooth index
                    cls = int(box.cls[0])
                    if 0 <= cls < 32:
                        pred_teeth[cls] = box
                
                # Compare predictions to ground truth
                for gt_idx in gt_teeth:
                    if gt_idx in pred_teeth:
                        tp_teeth += 1
                        tp_perfect += 1  # If we find the tooth, it's a start
                    else:
                        fn_teeth += 1
                
                for pred_idx in pred_teeth:
                    if pred_idx not in gt_teeth:
                        fp_teeth += 1
                
            except Exception as e:
                print(f"  Error on {img_name}: {str(e)[:50]}")
                fn_teeth += len(gt_teeth)
        
        # Calculate metrics
        precision_teeth = tp_teeth / (tp_teeth + fp_teeth) if (tp_teeth + fp_teeth) > 0 else 0
        recall_teeth = tp_teeth / (tp_teeth + fn_teeth) if (tp_teeth + fn_teeth) > 0 else 0
        f1_teeth = 2 * precision_teeth * recall_teeth / (precision_teeth + recall_teeth) if (precision_teeth + recall_teeth) > 0 else 0
        
        results[config["name"]] = {
            "conf": config["conf"],
            "attr": config["attr"],
            "tp_teeth": tp_teeth,
            "fp_teeth": fp_teeth,
            "fn_teeth": fn_teeth,
            "precision_teeth": round(precision_teeth, 3),
            "recall_teeth": round(recall_teeth, 3),
            "f1_teeth": round(f1_teeth, 3),
            "images_tested": total_images,
        }
        
        print(f"  Teeth Precision: {precision_teeth:.1%}")
        print(f"  Teeth Recall: {recall_teeth:.1%}")
        print(f"  F1 Score: {f1_teeth:.3f}")
        print(f"  TP={tp_teeth}, FP={fp_teeth}, FN={fn_teeth}")
        print()
    
    # Print summary
    print(f"\n{'='*80}")
    print("THRESHOLD COMPARISON")
    print(f"{'='*80}\n")
    
    # Sort by F1 score
    sorted_results = sorted(results.items(), key=lambda x: x[1]["f1_teeth"], reverse=True)
    
    for name, metrics in sorted_results:
        print(f"{name}:")
        print(f"  Thresholds: conf={metrics['conf']}, attr={metrics['attr']}")
        print(f"  Precision: {metrics['precision_teeth']:.1%}")
        print(f"  Recall: {metrics['recall_teeth']:.1%}")
        print(f"  F1: {metrics['f1_teeth']:.3f}")
        print()
    
    # Recommend best
    best_name, best_metrics = sorted_results[0]
    print(f"{'='*80}")
    print(f"✓ RECOMMENDED: {best_name}")
    print(f"  Thresholds: conf={best_metrics['conf']}, attr={best_metrics['attr']}")
    print(f"  Precision: {best_metrics['precision_teeth']:.1%}")
    print(f"  Recall: {best_metrics['recall_teeth']:.1%}")
    print(f"{'='*80}\n")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate thresholds against ground truth")
    parser.add_argument("--images", type=int, default=50, help="Number of images to test")
    parser.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    args = parser.parse_args()
    
    validate_thresholds(num_images=args.images, device=args.device)
