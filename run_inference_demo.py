import os
import sys
import argparse
from pathlib import Path
import cv2

# Add project to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.predictor import ARCHONPredictor
from src.utils.visualize import draw_teeth_detections

# Example for running the script:
# Process DENTEX training image
# python run_inference_demo.py "D:\WILP\sem-4\Dataset\DENTEX\DENTEX\training_data\quadrant-enumeration-disease\xrays\train_23.png"

# Result saved to: ./results/inference_train_23.png

# Configuration
OUTPUT_DIR = r".\results"
WEIGHTS_PATH = r".\weights\phase2_best.pt"  # Use phase2 detection model

# Thresholds for clinical use
CONF_THRESHOLD = 0.15
ATTR_THRESHOLD = 0.08
IOU_THRESHOLD = 0.45

def main():
    """Run inference on input image and save annotated result."""
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run ARCHON inference on an image")
    parser.add_argument("input_image", help="Path to input image file")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Output directory (default: ./results)")
    args = parser.parse_args()
    
    input_image_path = args.input_image
    output_dir = args.output_dir
    
    # Validate input image
    if not os.path.exists(input_image_path):
        print(f"[ERROR] Input image not found: {input_image_path}")
        return
    
    input_image_path = os.path.abspath(input_image_path)
    image_name = os.path.basename(input_image_path)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"[DIR] Output directory: {output_dir}")
    print(f"[INFO] Processing: {image_name}")
    
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
        print("[OK] Model loaded successfully\n")
    except Exception as e:
        print(f"[ERROR] Error loading model: {e}")
        print(f"[INFO] Using weights: {WEIGHTS_PATH}")
        return
    
    # Run inference
    try:
        # Run inference
        results = predictor.predict(input_image_path)
        teeth_detections = results.get(image_name, [])
        
        # Load original image
        img = cv2.imread(input_image_path)
        if img is None:
            print(f"[ERROR] Could not load image: {input_image_path}")
            return
        
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
        output_image_path = os.path.join(output_dir, f"inference_{image_name}")
        cv2.imwrite(output_image_path, annotated_img)
        
        print(f"[OK] Inference completed successfully")
        print(f"[INFO] Detected {len(teeth_detections)} teeth")
        
        if teeth_detections:
            print(f"\n[INFO] Detected teeth:")
            for tooth in teeth_detections:
                diseases = ', '.join(tooth.diseases) if tooth.diseases else "Healthy"
                print(f"    FDI {tooth.fdi}: {diseases}")
        
        print(f"\n[OK] Result saved: {output_image_path}\n")
    
    except Exception as e:
        print(f"[ERROR] Error during inference: {str(e)}")
        import traceback
        traceback.print_exc()
        return

if __name__ == '__main__':
    main()
