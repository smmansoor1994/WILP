#!/usr/bin/env python3
"""
Create side-by-side comparison images combining ground truth and model inference results.
"""

import os
import cv2
import numpy as np
from pathlib import Path

# Configuration
GT_DIR = r".\demo_images_annotated"
INFERENCE_DIR = r".\demo_images_inferenced_model"
OUTPUT_DIR = r".\demo_images_combined"

# Best 20 images
BEST_IMAGES = [
    "train_23.png", "train_25.png", "train_74.png", "train_111.png", 
    "train_130.png", "train_199.png", "train_204.png", "train_318.png",
    "train_331.png", "train_407.png", "train_411.png", "train_434.png",
    "train_500.png", "train_521.png", "train_554.png", "train_598.png",
    "train_650.png", "train_660.png", "train_690.png", "train_699.png",
]

def create_combined_image(gt_path, inference_path, output_path):
    """
    Create a side-by-side comparison image.
    
    Args:
        gt_path: Path to ground truth annotated image
        inference_path: Path to inference annotated image
        output_path: Path to save combined image
    """
    # Load images
    gt_img = cv2.imread(str(gt_path))
    inf_img = cv2.imread(str(inference_path))
    
    if gt_img is None or inf_img is None:
        return False
    
    # Get dimensions
    h_gt, w_gt = gt_img.shape[:2]
    h_inf, w_inf = inf_img.shape[:2]
    
    # Resize to same height if needed
    if h_gt != h_inf:
        max_h = max(h_gt, h_inf)
        gt_img = cv2.resize(gt_img, (int(w_gt * max_h / h_gt), max_h))
        inf_img = cv2.resize(inf_img, (int(w_inf * max_h / h_inf), max_h))
    
    # Create white separator (10 pixels)
    h = gt_img.shape[0]
    w = gt_img.shape[1]
    separator = np.ones((h, 10, 3), dtype=np.uint8) * 255
    
    # Combine horizontally
    combined = np.hstack([gt_img, separator, inf_img])
    
    # Add title bar with labels
    title_height = 40
    title_img = np.ones((title_height, combined.shape[1], 3), dtype=np.uint8) * 240
    
    # Add text labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    color = (0, 0, 0)
    thickness = 2
    
    # Ground truth label
    cv2.putText(title_img, "Ground Truth Annotations", (20, 28), font, font_scale, color, thickness)
    
    # Inference label
    inf_label_x = gt_img.shape[1] + 10 + 20
    cv2.putText(title_img, "Model Inference Results", (inf_label_x, 28), font, font_scale, color, thickness)
    
    # Combine title with image
    final_img = np.vstack([title_img, combined])
    
    # Save
    cv2.imwrite(str(output_path), final_img)
    return True

def main():
    """Main processing function."""
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[DIR] Output directory: {OUTPUT_DIR}\n")
    
    successful = 0
    failed = 0
    
    print(f"[INFO] Creating combined images...\n")
    
    for idx, image_name in enumerate(BEST_IMAGES, 1):
        try:
            # GT image path
            gt_image_path = os.path.join(GT_DIR, f"annotated_{image_name}")
            
            # Inference image path
            inf_image_path = os.path.join(INFERENCE_DIR, f"inference_{image_name}")
            
            # Output path
            output_image_path = os.path.join(OUTPUT_DIR, f"combined_{image_name}")
            
            # Check if both images exist
            if not os.path.exists(gt_image_path):
                print(f"[ERROR] [{idx:2d}/20] {image_name} - GT image not found")
                failed += 1
                continue
            
            if not os.path.exists(inf_image_path):
                print(f"[ERROR] [{idx:2d}/20] {image_name} - Inference image not found")
                failed += 1
                continue
            
            # Create combined image
            if create_combined_image(gt_image_path, inf_image_path, output_image_path):
                print(f"[OK] [{idx:2d}/20] {image_name} - Combined successfully")
                successful += 1
            else:
                print(f"[ERROR] [{idx:2d}/20] {image_name} - Failed to combine")
                failed += 1
        
        except Exception as e:
            failed += 1
            print(f"[ERROR] [{idx:2d}/20] {image_name} - Error: {str(e)}")
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"[OK] Successfully combined {successful} images")
    print(f"[ERROR] Failed: {failed} images")
    print(f"[DIR] Output directory: {OUTPUT_DIR}")
    print(f"{'='*70}\n")
    
    # Create README
    readme_path = os.path.join(OUTPUT_DIR, "README.md")
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write("""# Combined Demo Images - Ground Truth vs Model Inference

## Overview

This folder contains side-by-side comparison images showing:
- **Left side:** Ground truth annotations with FDI codes and disease labels
- **Right side:** Model inference results with detected teeth and diseases

## Purpose

These combined images allow for direct visual comparison between:
1. **What the model should detect** (ground truth from radiographs)
2. **What the model actually detected** (ARCHON predictions)

## Evaluation Methodology

For each image pair, assess:
- **Tooth Detection Accuracy:** Are all diseased teeth detected?
- **Quadrant Alignment:** Do teeth align correctly in their quadrants?
- **Disease Classification:** Are disease types correctly identified?
- **False Positives:** Are any healthy teeth incorrectly marked as diseased?

## Clinical Thresholds Used

- **Confidence Threshold:** 0.15 (minimizes false positives)
- **Attribute Threshold:** 0.08 (balanced disease detection)
- **IoU Threshold:** 0.45 (standard post-processing)

## Color Coding Reference

Bounding boxes are colored by quadrant (same in both GT and inference):
- **Q1 (Upper Right):** Steel Blue (210, 160, 50) - teeth 11-18
- **Q2 (Upper Left):** Slate Purple (180, 60, 60) - teeth 21-28
- **Q3 (Lower Left):** Red (40, 40, 200) - teeth 31-38
- **Q4 (Lower Right):** Orange (30, 140, 210) - teeth 41-48

## Label Format

Each tooth label follows the FDI World Dental Federation numbering system:
- **FDI XY** where:
  - X = Quadrant (1-4)
  - Y = Tooth position within quadrant (1-8)
  
Disease types shown:
- **Caries:** Tooth decay
- **Impacted:** Unerupted or trapped tooth
- **Deep Caries:** Advanced cavity
- **Periapical Lesion:** Bone infection at tooth root

## Files

All 20 combined images are in the format:
```
combined_train_23.png
combined_train_25.png
... (20 total)
```

## Next Steps

1. Review each comparison image to assess model performance
2. Identify patterns in detection accuracy and errors
3. Note areas for improvement (if any)
4. Use for clinical validation and demonstration

## Summary Statistics

- **Total Images:** 20
- **Model:** ARCHON Phase 2 Detection
- **Dataset:** DENTEX Training Data (quadrant-enumeration-disease)
- **Quadrant Distribution:** All four quadrants represented
- **Disease Types Detected:** Impacted, Caries, Deep Caries, Periapical Lesion
""")
    
    print(f"[OK] README created at {readme_path}")

if __name__ == "__main__":
    main()
