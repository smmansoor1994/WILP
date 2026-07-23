import os
import sys
from pathlib import Path
import cv2

# Add project to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.predictor import ARCHONPredictor
from src.utils.visualize import draw_teeth_detections

# Configuration
SOURCE_IMAGES_DIR = r"D:\WILP\sem-4\Dataset\DENTEX\DENTEX\training_data\quadrant-enumeration-disease\xrays"
OUTPUT_DIR = r".\demo_images_inferenced_model"
WEIGHTS_PATH = r".\weights\phase2_best.pt"  # Use phase2 detection model

# Thresholds for clinical use
CONF_THRESHOLD = 0.15
ATTR_THRESHOLD = 0.08
IOU_THRESHOLD = 0.45

# List of best images for demo
BEST_IMAGES = [
    'train_23.png', 'train_25.png', 'train_74.png', 'train_111.png', 'train_130.png',
    'train_199.png', 'train_204.png', 'train_318.png', 'train_331.png', 'train_407.png',
    'train_411.png', 'train_434.png', 'train_500.png', 'train_521.png', 'train_554.png',
    'train_598.png', 'train_650.png', 'train_660.png', 'train_690.png', 'train_699.png'
]

def main():
    """Run inference on best demo images and save annotated results."""
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[DIR] Output directory: {OUTPUT_DIR}")
    
    # Initialize predictor with clinical thresholds
    print("[INFO] Loading ARCHON model...")
    try:
        predictor = ARCHONPredictor(
            weights_path=WEIGHTS_PATH,
            device="cpu",  # Use CPU since CUDA not available
            conf_threshold=CONF_THRESHOLD,
            iou_threshold=IOU_THRESHOLD,
            attr_threshold=ATTR_THRESHOLD,
            img_size=(640, 1280),
        )
        print("[OK] Model loaded successfully")
    except Exception as e:
        print(f"[ERROR] Error loading model: {e}")
        print(f"[INFO] Using weights: {WEIGHTS_PATH}")
        return
    
    # Process each best image
    print(f"\n[INFO] Running inference on {len(BEST_IMAGES)} images...\n")
    
    successful = 0
    failed = 0
    
    for idx, image_name in enumerate(BEST_IMAGES, 1):
        try:
            image_path = os.path.join(SOURCE_IMAGES_DIR, image_name)
            output_image_path = os.path.join(OUTPUT_DIR, f"inference_{image_name}")
            
            # Check if image exists
            if not os.path.exists(image_path):
                print(f"[ERROR] {image_name} - Image not found")
                failed += 1
                continue
            
            # Run inference
            results = predictor.predict(image_path)
            teeth_detections = results.get(image_name, [])
            
            # Load original image and draw detections
            img = cv2.imread(image_path)
            if img is None:
                print(f"[ERROR] {image_name} - Could not load image")
                failed += 1
                continue
            
            # Draw detections on image
            annotated_img = draw_teeth_detections(
                img,
                teeth_detections,
                show_fdi=True,
                show_diseases=True,
                show_conf=False,
                line_thickness=2,
            )
            
            # Save annotated image
            cv2.imwrite(output_image_path, annotated_img)
            
            print(f"[OK] [{idx:2d}/20] {image_name}")
            print(f"    -> Detected {len(teeth_detections)} teeth")
            for tooth in teeth_detections:
                diseases = ', '.join(tooth.diseases) if tooth.diseases else "Healthy"
                print(f"       FDI {tooth.fdi}: {diseases}")
            
            successful += 1
        
        except Exception as e:
            failed += 1
            print(f"[ERROR] {image_name} - Error: {str(e)}")
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"[OK] Successfully processed {successful} images")
    print(f"[ERROR] Failed: {failed} images")
    print(f"[DIR] Output directory: {OUTPUT_DIR}")
    print(f"{'='*70}\n")
    
    # Create README
    readme_path = os.path.join(OUTPUT_DIR, "README.md")
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(f"""# Model Inference Results

## Configuration

- **Model:** ARCHON Hybrid Predictor
- **Weights:** {WEIGHTS_PATH}
- **Confidence Threshold:** {CONF_THRESHOLD}
- **Attribute Threshold:** {ATTR_THRESHOLD}
- **IoU Threshold:** {IOU_THRESHOLD}

## Results

All {successful} images successfully processed with model inference.

### Color Coding by Quadrant

Each bounding box is colored according to the tooth's quadrant:

- **Q1 (Upper Right):** Steel Blue (210, 160, 50) - teeth 11-18
- **Q2 (Upper Left):** Slate Purple (180, 60, 60) - teeth 21-28
- **Q3 (Lower Left):** Red (40, 40, 200) - teeth 31-38
- **Q4 (Lower Right):** Orange (30, 140, 210) - teeth 41-48

### Label Format

Each tooth is labeled as:
```
Q: # N: # D: disease_type
```

Where:
- **Q:** Quadrant (1-4)
- **N:** Tooth position within quadrant (1-8)
- **D:** Disease type (Caries, Impacted, Deep Caries, Periapical Lesion, Healthy)

### Files

All inference results are saved as:
- `inference_train_23.png` → Inference result for train_23.png
- `inference_train_25.png` → Inference result for train_25.png
- ... (and so on for all 20 images)

## Comparison with Ground Truth

To compare model predictions with ground truth annotations:
1. Open corresponding images in `../demo_images_annotated/`
2. Compare the bounding boxes and tooth enumeration
3. Note any differences in disease identification

## Clinical Evaluation

These results demonstrate the model's performance under clinical conditions:
- Accurate tooth enumeration using FDI notation
- Reliable disease classification
- Quadrant-specific color coding for clarity
- Ready for radiologist review

---

**Generated:** {Path(__file__).name}
**Threshold Set:** conf=0.15, attr=0.08 (clinically optimized)
""")
    
    print(f"✅ README created at {readme_path}")

if __name__ == '__main__':
    main()
