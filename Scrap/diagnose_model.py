"""
diagnose_model.py — Diagnose model and environment issues

Checks:
1. Model file integrity
2. YOLO compatibility  
3. PyTorch installation
4. Device availability
5. Can load and run basic inference

Usage:
    python diagnose_model.py --weights weights/hybrid_best.pt --device cpu
"""

import argparse
import sys
from pathlib import Path


def diagnose():
    parser = argparse.ArgumentParser(description="Diagnose ARCHON model setup")
    parser.add_argument("--weights", default="weights/hybrid_best.pt")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("ARCHON MODEL DIAGNOSTIC")
    print("=" * 80)
    
    # 1. Check weights file
    print("\n[1/6] Checking weights file...")
    weights_path = Path(args.weights)
    if weights_path.exists():
        size_mb = weights_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ Found: {weights_path}")
        print(f"  ✓ Size: {size_mb:.1f} MB")
    else:
        print(f"  ❌ Not found: {weights_path}")
        print(f"  Checked: {weights_path.absolute()}")
        return False
    
    # 2. Check PyTorch
    print("\n[2/6] Checking PyTorch...")
    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__}")
        print(f"  ✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  ✓ CUDA device: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"  ❌ PyTorch error: {e}")
        return False
    
    # 3. Check Ultralytics
    print("\n[3/6] Checking Ultralytics YOLO...")
    try:
        from ultralytics import YOLO
        print(f"  ✓ Ultralytics installed")
    except Exception as e:
        print(f"  ❌ Ultralytics error: {e}")
        return False
    
    # 4. Check OpenCV
    print("\n[4/6] Checking OpenCV...")
    try:
        import cv2
        print(f"  ✓ OpenCV {cv2.__version__}")
    except Exception as e:
        print(f"  ❌ OpenCV error: {e}")
        return False
    
    # 5. Try loading model
    print("\n[5/6] Loading model...")
    try:
        from ultralytics import YOLO
        print(f"  Loading from: {weights_path}")
        model = YOLO(str(weights_path))
        print(f"  ✓ Model loaded successfully")
        print(f"  ✓ Model device: {model.device}")
        print(f"  ✓ Model task: {model.task}")
    except Exception as e:
        print(f"  ❌ Model load failed: {e}")
        print(f"  Error type: {type(e).__name__}")
        import traceback
        print("\n  Full traceback:")
        traceback.print_exc()
        return False
    
    # 6. Try inference
    print("\n[6/6] Testing inference...")
    try:
        # Find a test image
        test_dirs = [
            Path("data/processed/images/train"),
            Path("data/processed/images/val"),
            Path("."),
        ]
        
        test_image = None
        for test_dir in test_dirs:
            if test_dir.exists():
                images = list(test_dir.glob("*.png")) + list(test_dir.glob("*.jpg"))
                if images:
                    test_image = images[0]
                    break
        
        if not test_image:
            print(f"  ⚠️  No test images found")
        else:
            print(f"  Using test image: {test_image.name}")
            results = model.predict(
                source=str(test_image),
                conf=0.15,
                verbose=False,
            )
            
            if results:
                detections = len(results[0].boxes)
                print(f"  ✓ Inference successful")
                print(f"  ✓ Found {detections} detections")
            else:
                print(f"  ✓ Inference ran (no detections)")
    
    except Exception as e:
        print(f"  ❌ Inference failed: {e}")
        import traceback
        print("\n  Full traceback:")
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 80)
    print("✓ ALL CHECKS PASSED")
    print("=" * 80)
    print("\nYour setup is working! You can now run:")
    print("  python direct_inference_test.py --images 5 --device cpu")
    print("  python verify_local_image.py --image test.jpg --device cpu")
    print("=" * 80 + "\n")
    return True


if __name__ == "__main__":
    success = diagnose()
    sys.exit(0 if success else 1)
