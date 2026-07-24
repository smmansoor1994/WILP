#!/usr/bin/env python3
"""
Annotate ground truth bounding boxes from COCO JSON on input image and save to results/gt.
"""

import os
import sys
import json
import argparse
from pathlib import Path
import cv2

# Annotate a single image
# python annotate_ground_truth.py "path/to/image.png"

# Or specify custom output directory
# python annotate_ground_truth.py "path/to/image.png" --output-dir "./custom_output"
# python annotate_ground_truth.py "D:\WILP\sem-4\Dataset\DENTEX\DENTEX\training_data\quadrant-enumeration-disease\xrays\train_23.png"

# Add project to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration
GT_JSON_PATH = r"D:\WILP\sem-4\Dataset\DENTEX\DENTEX\training_data\quadrant-enumeration-disease\train_quadrant_enumeration_disease.json"
OUTPUT_DIR = str(PROJECT_ROOT / "results" / "gt")

# Quadrant colors (BGR format for OpenCV)
QUADRANT_COLORS = {
    1: (210, 160, 50),    # Q1 Upper-Right - Steel Blue
    2: (180, 60, 60),     # Q2 Upper-Left - Slate Purple
    3: (40, 40, 200),     # Q3 Lower-Left - Red
    4: (30, 140, 210),    # Q4 Lower-Right - Orange
}

# Disease names mapping
DISEASE_NAMES = {
    0: "Impacted",
    1: "Caries",
    2: "Periapical Lesion",
    3: "Deep Caries",
}

# Quadrant and tooth position names
QUADRANT_NAMES = {0: "1", 1: "2", 2: "3", 3: "4"}
TOOTH_POSITIONS = {i: str(i+1) for i in range(8)}

def load_ground_truth(gt_json_path):
    """Load ground truth data from COCO JSON."""
    with open(gt_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def get_image_annotations(gt_data, image_name):
    """Extract annotations for a specific image."""
    # Find image by name
    image_id = None
    for img in gt_data['images']:
        if img['file_name'] == image_name:
            image_id = img['id']
            break
    
    if image_id is None:
        return None
    
    # Get all annotations for this image
    annotations = [ann for ann in gt_data['annotations'] if ann['image_id'] == image_id]
    return annotations

def get_fdi_code(quadrant, position):
    """Calculate FDI code from quadrant and position."""
    # quadrant is 0-3, convert to 1-4
    # position is 0-7, convert to 1-8
    fdi = (quadrant + 1) * 10 + (position + 1)
    return fdi

def annotate_image(image_path, gt_data, output_path):
    """
    Annotate image with ground truth bounding boxes and save.
    
    Args:
        image_path: Path to input image
        gt_data: Ground truth data from COCO JSON
        output_path: Path to save annotated image
    """
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        return False, "Could not load image"
    
    image_name = os.path.basename(image_path)
    
    # Get annotations for this image
    annotations = get_image_annotations(gt_data, image_name)
    if annotations is None:
        return False, "Image not found in ground truth data"
    
    if not annotations:
        # No annotations for this image, just save original
        cv2.imwrite(str(output_path), img)
        return True, "No annotations found (saved original)"
    
    # Draw each annotation
    tooth_count = 0
    for ann in annotations:
        bbox = ann['bbox']  # [x, y, width, height]
        
        # Extract category IDs - DENTEX uses category_id_1, category_id_2, category_id_3
        # category_id_1: Quadrant (0-3)
        # category_id_2: Tooth position (0-7)
        # category_id_3: Disease type (0-3)
        quadrant = ann.get('category_id_1', 0)  # 0-3
        position = ann.get('category_id_2', 0)  # 0-7
        disease = ann.get('category_id_3', 0)   # 0-3
        
        # Extract bbox coordinates
        x, y, w, h = bbox
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w), int(y + h)
        
        # Get color based on quadrant
        color = QUADRANT_COLORS.get(quadrant + 1, (200, 200, 200))
        
        # Calculate FDI code
        fdi = get_fdi_code(quadrant, position)
        
        # Get disease name
        disease_name = DISEASE_NAMES.get(disease, "Unknown")
        
        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Create label
        label = f"Q:{quadrant+1} N:{position+1} D:{disease_name}"
        
        # Get text size
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        
        # Draw text background
        text_x = x1
        text_y = y1 - 5
        if text_y < 20:
            text_y = y2 + 15
        
        cv2.rectangle(img, (text_x, text_y - text_size[1] - 4), 
                     (text_x + text_size[0] + 4, text_y + 4), color, -1)
        
        # Draw text
        cv2.putText(img, label, (text_x + 2, text_y - 2), 
                   font, font_scale, (255, 255, 255), thickness)
        
        tooth_count += 1
    
    # Save annotated image
    cv2.imwrite(str(output_path), img)
    return True, f"Annotated with {tooth_count} teeth"

def main():
    """Main function to annotate ground truth."""
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Annotate ground truth from COCO JSON")
    parser.add_argument("input_image", help="Path to input image file")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Output directory (default: ./results/gt)")
    args = parser.parse_args()
    
    input_image_path = args.input_image
    output_dir = args.output_dir
    
    # Validate input image
    if not os.path.exists(input_image_path):
        print(f"[ERROR] Input image not found: {input_image_path}")
        return
    
    input_image_path = os.path.abspath(input_image_path)
    image_name = os.path.basename(input_image_path)
    
    # Validate ground truth JSON
    if not os.path.exists(GT_JSON_PATH):
        print(f"[ERROR] Ground truth JSON not found: {GT_JSON_PATH}")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"[DIR] Output directory: {output_dir}")
    print(f"[INFO] Processing: {image_name}")
    
    # Load ground truth data
    print("[INFO] Loading ground truth data...")
    try:
        gt_data = load_ground_truth(GT_JSON_PATH)
        print(f"[OK] Loaded ground truth with {len(gt_data['images'])} images")
    except Exception as e:
        print(f"[ERROR] Error loading ground truth: {e}")
        return
    
    # Annotate image
    print("[INFO] Annotating ground truth...")
    try:
        output_image_path = os.path.join(output_dir, f"gt_{image_name}")
        success, message = annotate_image(input_image_path, gt_data, output_image_path)
        
        if success:
            print(f"[OK] {message}")
            print(f"[OK] Result saved: {output_image_path}\n")
        else:
            print(f"[ERROR] {message}\n")
    
    except Exception as e:
        print(f"[ERROR] Error during annotation: {str(e)}")
        import traceback
        traceback.print_exc()
        return

if __name__ == '__main__':
    main()
