"""
working_inference_test.py — Threshold testing using working Phase 2 model

Uses phase2_best.pt (which loads successfully) to test threshold combinations.
Phase 2 has disease attributes, so this is valid for threshold optimization.

Usage:
    python working_inference_test.py --images 10 --device cpu
"""

import argparse
from pathlib import Path
import json
from ultralytics import YOLO
import cv2


def run_inference_tests(num_images=10, device="cpu"):
    """Test multiple threshold configurations."""
    
    # Load working model
    weights_path = Path("weights/phase2_best.pt")
    print(f"\n{'='*80}")
    print(f"Loading model: {weights_path}")
    model = YOLO(str(weights_path))
    print(f"✓ Model loaded successfully")
    print(f"  Task: {model.task}")
    print(f"  Device: {model.device}")
    print(f"{'='*80}\n")
    
    # Find test images
    train_dir = Path("data/processed/images/train")
    if not train_dir.exists():
        print(f"❌ Images not found at {train_dir}")
        return
    
    images = sorted(train_dir.glob("*.png")) + sorted(train_dir.glob("*.jpg"))
    test_images = images[:num_images]
    
    if not test_images:
        print(f"❌ No images found")
        return
    
    print(f"Testing on {len(test_images)} images\n")
    
    # Test configurations
    configs = [
        {"name": "Very Permissive", "conf": 0.10, "iou": 0.45},
        {"name": "Recommended", "conf": 0.15, "iou": 0.45},
        {"name": "Balanced", "conf": 0.20, "iou": 0.45},
        {"name": "Strict", "conf": 0.25, "iou": 0.45},
    ]
    
    results = {}
    
    for config in configs:
        print(f"Config: {config['name']} (conf={config['conf']}, iou={config['iou']})")
        
        total_detections = 0
        errors = 0
        
        for img_path in test_images:
            try:
                result = model.predict(
                    source=str(img_path),
                    conf=config["conf"],
                    iou=config["iou"],
                    verbose=False,
                    device=device,
                )
                
                if result and len(result) > 0:
                    detections = len(result[0].boxes)
                    total_detections += detections
                else:
                    detections = 0
                    
            except Exception as e:
                errors += 1
                print(f"  ❌ Error on {img_path.name}: {str(e)[:60]}")
        
        avg_detections = total_detections / len(test_images) if test_images else 0
        
        results[config["name"]] = {
            "conf": config["conf"],
            "iou": config["iou"],
            "total_detections": total_detections,
            "avg_per_image": round(avg_detections, 2),
            "errors": errors,
            "success_rate": round(100 * (1 - errors/len(test_images)), 1),
        }
        
        print(f"  Total detections: {total_detections}")
        print(f"  Avg/image: {avg_detections:.2f}")
        print(f"  Success rate: {100*(1-errors/len(test_images)):.1f}%")
        print()
    
    # Print summary
    print(f"{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    for name, result in results.items():
        print(f"\n{name}:")
        for key, value in result.items():
            print(f"  {key}: {value}")
    
    print(f"\n{'='*80}")
    print("✓ Inference tests completed successfully!")
    print(f"  Model: {weights_path}")
    print(f"  Images tested: {len(test_images)}")
    print(f"{'='*80}\n")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test thresholds on working model")
    parser.add_argument("--images", type=int, default=10, help="Number of images to test")
    parser.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    args = parser.parse_args()
    
    run_inference_tests(num_images=args.images, device=args.device)
