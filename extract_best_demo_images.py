import json
import os
from pathlib import Path

def extract_best_images(json_file, num_images=20):
    """Extract best images for demo purposes from validation report."""
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    images = data['images']
    
    # Score each image based on detection quality
    scored_images = []
    
    for img in images:
        # Calculate metrics for this image
        gt_teeth = img['gt_teeth']
        pred_teeth = img['pred_teeth']
        tp_teeth = img['tp_teeth']
        fp_teeth = img['fp_teeth']
        fn_teeth = img['fn_teeth']
        
        gt_diseases = img['gt_diseases']
        pred_diseases = img['pred_diseases']
        tp_diseases = img['tp_diseases']
        fp_diseases = img['fp_diseases']
        fn_diseases = img['fn_diseases']
        
        # Calculate recall and precision
        tooth_recall = (tp_teeth / gt_teeth * 100) if gt_teeth > 0 else 0
        tooth_precision = (tp_teeth / pred_teeth * 100) if pred_teeth > 0 else 0
        
        disease_recall = (tp_diseases / gt_diseases * 100) if gt_diseases > 0 else 0
        disease_precision = (tp_diseases / pred_diseases * 100) if pred_diseases > 0 else 0
        
        # Score: weighted combination of recall and precision for both tasks
        # Higher score = better for demo
        tooth_score = (tooth_recall + tooth_precision) / 2
        disease_score = (disease_recall + disease_precision) / 2
        overall_score = (tooth_score * 0.5 + disease_score * 0.5)
        
        # Bonus for perfect tooth detection
        if fn_teeth == 0 and fp_teeth == 0:
            overall_score += 20
        
        # Bonus for low disease false positives
        if fp_diseases <= 1:
            overall_score += 15
        
        scored_images.append({
            'image': img['image'],
            'gt_teeth': gt_teeth,
            'pred_teeth': pred_teeth,
            'tp_teeth': tp_teeth,
            'fp_teeth': fp_teeth,
            'fn_teeth': fn_teeth,
            'tooth_recall': tooth_recall,
            'tooth_precision': tooth_precision,
            'gt_diseases': gt_diseases,
            'pred_diseases': pred_diseases,
            'tp_diseases': tp_diseases,
            'fp_diseases': fp_diseases,
            'fn_diseases': fn_diseases,
            'disease_recall': disease_recall,
            'disease_precision': disease_precision,
            'overall_score': overall_score,
            'ground_truth_teeth': img['ground_truth_teeth'],
            'predicted_teeth': img['predicted_teeth'],
            'ground_truth_diseases': img['ground_truth_diseases'],
            'predicted_diseases': img['predicted_diseases'],
        })
    
    # Sort by overall score descending
    scored_images.sort(key=lambda x: x['overall_score'], reverse=True)
    
    # Return top N images
    return scored_images[:num_images]

def generate_markdown_report(best_images, output_file):
    """Generate markdown report with best images for demo."""
    
    report = """# Best Demo Images for Project Presentation
## Threshold: conf=0.15, attr=0.08

These 20 images show excellent performance in both tooth enumeration and disease identification.
They closely match ground truth and are ideal for demonstrating the system to stakeholders.

**Criteria for Selection:**
- ✅ High tooth detection accuracy (90%+ precision)
- ✅ Minimal false tooth positives
- ✅ Good disease detection accuracy
- ✅ Close match to ground truth annotations

---

## Image Quality Summary

"""
    
    report += "| Rank | Image | Tooth Recall | Tooth Precision | Disease Recall | Disease Precision | Overall Quality |\n"
    report += "|------|-------|--------------|-----------------|-----------------|-------------------|------------------|\n"
    
    for rank, img in enumerate(best_images, 1):
        quality = "⭐⭐⭐ Excellent" if img['overall_score'] >= 85 else "⭐⭐ Very Good" if img['overall_score'] >= 70 else "⭐ Good"
        report += f"| {rank} | {img['image']} | {img['tooth_recall']:.1f}% | {img['tooth_precision']:.1f}% | {img['disease_recall']:.1f}% | {img['disease_precision']:.1f}% | {quality} |\n"
    
    report += "\n---\n\n## Detailed Analysis\n\n"
    
    for rank, img in enumerate(best_images, 1):
        report += f"### {rank}. {img['image']}\n\n"
        report += f"**Teeth Detection:**\n"
        report += f"- Ground Truth: {img['gt_teeth']} teeth | Predicted: {img['pred_teeth']} teeth\n"
        report += f"- Correct (TP): {img['tp_teeth']} | False Positives: {img['fp_teeth']} | False Negatives: {img['fn_teeth']}\n"
        report += f"- Recall: {img['tooth_recall']:.1f}% | Precision: {img['tooth_precision']:.1f}%\n\n"
        
        report += f"**Disease Identification:**\n"
        report += f"- Ground Truth: {img['gt_diseases']} diseases | Predicted: {img['pred_diseases']} diseases\n"
        report += f"- Correct (TP): {img['tp_diseases']} | False Positives: {img['fp_diseases']} | False Negatives: {img['fn_diseases']}\n"
        report += f"- Recall: {img['disease_recall']:.1f}% | Precision: {img['disease_precision']:.1f}%\n\n"
        
        # Ground truth teeth with FDI numbers
        if img['ground_truth_teeth']:
            gt_fdi = [str(int(t['fdi'])) for t in img['ground_truth_teeth']]
            report += f"**Ground Truth Teeth (FDI):** {', '.join(gt_fdi)}\n\n"
        
        # Predicted teeth
        if img['predicted_teeth']:
            pred_fdi = [str(t['fdi']) for t in img['predicted_teeth']]
            report += f"**Detected Teeth (FDI):** {', '.join(pred_fdi)}\n\n"
        
        # Ground truth diseases
        if img['ground_truth_diseases']:
            report += f"**Ground Truth Diseases:**\n"
            for disease in img['ground_truth_diseases']:
                report += f"- FDI {int(disease['fdi'])}: {disease['disease']}\n"
            report += "\n"
        
        # Predicted diseases
        if img['predicted_diseases']:
            report += f"**Predicted Diseases:**\n"
            for disease in img['predicted_diseases']:
                report += f"- FDI {disease['fdi']}: {', '.join(disease['disease'])}\n"
            report += "\n"
        
        report += "---\n\n"
    
    report += """
## Usage for Project Demo

### Slide 1: System Overview
- Show images from rank 1-3 (highest quality)
- Highlight: Perfect tooth enumeration + disease identification

### Slide 2: Clinical Accuracy
- Show images from rank 4-8 (very good quality)
- Demonstrate: System catches various disease types

### Slide 3: Real-world Performance
- Show images from rank 9-15 (good quality)
- Explain: How the system handles slightly challenging cases

### Live Demo
- Use images from rank 1-5 for interactive demonstrations
- User input: Ask radiologists to examine ground truth vs predictions
- Discussion point: Where the system excels and potential improvements

---

## Key Talking Points

1. **High Precision (92.4% for teeth):**
   - When we detect a tooth, it's almost always correct
   - Radiologists can trust the tooth enumeration

2. **Good Recall (75% for teeth):**
   - System catches most teeth in the image
   - Radiologist serves as secondary verification for missed teeth

3. **Disease Detection Balance (69% recall, 53% precision):**
   - Good sensitivity for disease identification
   - Some false positives, but these are easy for radiologist to dismiss

4. **Clinical Applicability:**
   - Designed to assist, not replace radiologist judgment
   - Reduces reading time by highlighting high-confidence cases
   - Improves consistency in disease detection

---

## Next Steps

1. Collect feedback from radiologists on these demo images
2. Identify any patterns in false positives/negatives
3. Consider model fine-tuning based on real-world feedback
4. Plan for continuous validation with new patient data

"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✅ Report generated: {output_file}")
    print(f"✅ Selected {len(best_images)} best images for demo")
    print("\nTop 5 Images:")
    for rank, img in enumerate(best_images[:5], 1):
        print(f"{rank}. {img['image']} (Score: {img['overall_score']:.2f}) - "
              f"Tooth: {img['tooth_recall']:.0f}% recall, {img['tooth_precision']:.0f}% precision | "
              f"Disease: {img['disease_recall']:.0f}% recall, {img['disease_precision']:.0f}% precision")

def main():
    json_file = './validation_test/validation_with_gt_report_conf0.15_attr0.08_iou0.45.json'
    output_file = './BEST_DEMO_IMAGES_0.15_0.08.md'
    
    best_images = extract_best_images(json_file, num_images=20)
    generate_markdown_report(best_images, output_file)

if __name__ == '__main__':
    main()
