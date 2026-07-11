"""
Generate comprehensive report with per-image, overall, and combined metrics
"""
import json
import pandas as pd

def generate_comprehensive_report():
    """Generate comprehensive report with all metric levels"""
    
    # Load data
    with open(r"d:\WILP\per_image_metrics.json", 'r') as f:
        per_image_data = json.load(f)
    
    with open(r"d:\WILP\overall_metrics.json", 'r') as f:
        overall_metrics = json.load(f)
    
    md = []
    
    # ============================================================================
    # TITLE AND EXECUTIVE SUMMARY
    # ============================================================================
    md.append("# Comprehensive Validation Metrics Report")
    md.append("## Teeth Identification + Disease Detection Combined Analysis")
    md.append("")
    md.append(f"**Total Images Analyzed:** {len(per_image_data)}")
    md.append(f"**Report Generated:** 2026-07-11")
    md.append("")
    md.append("---")
    md.append("")
    
    # ============================================================================
    # SECTION 1: OVERALL METRICS SUMMARY
    # ============================================================================
    md.append("## 1. OVERALL AGGREGATE METRICS")
    md.append("")
    
    # Teeth Detection Overall
    md.append("### 1.1 Teeth Identification - Overall Performance")
    md.append("")
    md.append("| Metric | Value |")
    md.append("|--------|-------|")
    md.append(f"| **Total Ground Truth Teeth** | {overall_metrics['teeth']['tp'] + overall_metrics['teeth']['fn']} |")
    md.append(f"| **Total Detected Teeth** | {overall_metrics['teeth']['tp'] + overall_metrics['teeth']['fp']} |")
    md.append(f"| **True Positives (TP)** | {overall_metrics['teeth']['tp']} |")
    md.append(f"| **False Positives (FP)** | {overall_metrics['teeth']['fp']} |")
    md.append(f"| **False Negatives (FN)** | {overall_metrics['teeth']['fn']} |")
    md.append("")
    
    md.append("#### Overall Performance Indicators")
    md.append("")
    md.append("| Metric | Value | Interpretation |")
    md.append("|--------|-------|-----------------|")
    md.append(f"| **Precision** | {overall_metrics['teeth']['precision']:.2f}% | {overall_metrics['teeth']['tp']}/{overall_metrics['teeth']['tp'] + overall_metrics['teeth']['fp']} detections were correct |")
    md.append(f"| **Recall** | {overall_metrics['teeth']['recall']:.2f}% | Detected {overall_metrics['teeth']['recall']:.1f}% of all actual teeth |")
    md.append(f"| **Sensitivity** | {overall_metrics['teeth']['recall']:.2f}% | True positive rate across all images |")
    md.append(f"| **F1 Score** | {overall_metrics['teeth']['f1_score']:.2f} | Harmonic mean of precision and recall |")
    md.append("")
    
    # Disease Detection Overall
    md.append("### 1.2 Disease Identification - Overall Performance")
    md.append("")
    md.append("| Metric | Value |")
    md.append("|--------|-------|")
    md.append(f"| **Total Ground Truth Diseases** | {overall_metrics['disease']['tp'] + overall_metrics['disease']['fn']} |")
    md.append(f"| **Total Detected Diseases** | {overall_metrics['disease']['tp'] + overall_metrics['disease']['fp']} |")
    md.append(f"| **True Positives (TP)** | {overall_metrics['disease']['tp']} |")
    md.append(f"| **False Positives (FP)** | {overall_metrics['disease']['fp']} |")
    md.append(f"| **False Negatives (FN)** | {overall_metrics['disease']['fn']} |")
    md.append("")
    
    md.append("#### Overall Performance Indicators")
    md.append("")
    md.append("| Metric | Value | Interpretation |")
    md.append("|--------|-------|-----------------|")
    md.append(f"| **Precision** | {overall_metrics['disease']['precision']:.2f}% | {overall_metrics['disease']['tp']}/{overall_metrics['disease']['tp'] + overall_metrics['disease']['fp']} disease detections were correct |")
    md.append(f"| **Recall** | {overall_metrics['disease']['recall']:.2f}% | Detected {overall_metrics['disease']['recall']:.1f}% of all actual diseases |")
    md.append(f"| **Sensitivity** | {overall_metrics['disease']['recall']:.2f}% | True positive rate for disease detection |")
    md.append(f"| **F1 Score** | {overall_metrics['disease']['f1_score']:.2f} | Harmonic mean of precision and recall |")
    md.append("")
    
    # ============================================================================
    # SECTION 2: COMBINED METRICS ANALYSIS
    # ============================================================================
    md.append("---")
    md.append("")
    md.append("## 2. COMBINED METRICS ANALYSIS")
    md.append("")
    md.append("### 2.1 Correlation Between Teeth Detection and Disease Detection")
    md.append("")
    
    # Calculate correlations
    perfect_teeth_perfect_disease = 0
    perfect_teeth_imperfect_disease = 0
    imperfect_teeth_perfect_disease = 0
    imperfect_teeth_imperfect_disease = 0
    
    for img in per_image_data:
        teeth_perfect = img['teeth']['fn'] == 0 and img['teeth']['fp'] == 0
        disease_perfect = img['disease']['fn'] == 0 and img['disease']['fp'] == 0
        
        if teeth_perfect and disease_perfect:
            perfect_teeth_perfect_disease += 1
        elif teeth_perfect and not disease_perfect:
            perfect_teeth_imperfect_disease += 1
        elif not teeth_perfect and disease_perfect:
            imperfect_teeth_perfect_disease += 1
        else:
            imperfect_teeth_imperfect_disease += 1
    
    total_images = len(per_image_data)
    
    md.append("| Scenario | Count | Percentage |")
    md.append("|----------|-------|------------|")
    md.append(f"| Perfect Teeth + Perfect Disease | {perfect_teeth_perfect_disease} | {(perfect_teeth_perfect_disease/total_images)*100:.1f}% |")
    md.append(f"| Perfect Teeth + Imperfect Disease | {perfect_teeth_imperfect_disease} | {(perfect_teeth_imperfect_disease/total_images)*100:.1f}% |")
    md.append(f"| Imperfect Teeth + Perfect Disease | {imperfect_teeth_perfect_disease} | {(imperfect_teeth_perfect_disease/total_images)*100:.1f}% |")
    md.append(f"| Imperfect Teeth + Imperfect Disease | {imperfect_teeth_imperfect_disease} | {(imperfect_teeth_imperfect_disease/total_images)*100:.1f}% |")
    md.append("")
    
    md.append("### 2.2 Teeth Detection Impact on Disease Detection")
    md.append("")
    
    # Analyze FN correlation
    avg_disease_fp_when_teeth_perfect = 0
    avg_disease_fp_when_teeth_imperfect = 0
    count_teeth_perfect = 0
    count_teeth_imperfect = 0
    
    for img in per_image_data:
        if img['teeth']['fn'] == 0 and img['teeth']['fp'] == 0:
            avg_disease_fp_when_teeth_perfect += img['disease']['fp']
            count_teeth_perfect += 1
        else:
            avg_disease_fp_when_teeth_imperfect += img['disease']['fp']
            count_teeth_imperfect += 1
    
    if count_teeth_perfect > 0:
        avg_disease_fp_when_teeth_perfect /= count_teeth_perfect
    if count_teeth_imperfect > 0:
        avg_disease_fp_when_teeth_imperfect /= count_teeth_imperfect
    
    md.append("| Condition | Avg Disease FP | Avg Disease Recall |")
    md.append("|-----------|----------------|--------------------|")
    
    # Calculate avg recall for perfect vs imperfect teeth
    perfect_teeth_disease_recalls = []
    imperfect_teeth_disease_recalls = []
    
    for img in per_image_data:
        if img['disease']['tp'] + img['disease']['fn'] > 0:
            recall = (img['disease']['tp'] / (img['disease']['tp'] + img['disease']['fn'])) * 100
            if img['teeth']['fn'] == 0 and img['teeth']['fp'] == 0:
                perfect_teeth_disease_recalls.append(recall)
            else:
                imperfect_teeth_disease_recalls.append(recall)
    
    avg_perfect = sum(perfect_teeth_disease_recalls) / len(perfect_teeth_disease_recalls) if perfect_teeth_disease_recalls else 0
    avg_imperfect = sum(imperfect_teeth_disease_recalls) / len(imperfect_teeth_disease_recalls) if imperfect_teeth_disease_recalls else 0
    
    md.append(f"| Perfect Tooth Detection | {avg_disease_fp_when_teeth_perfect:.2f} | {avg_perfect:.2f}% |")
    md.append(f"| Imperfect Tooth Detection | {avg_disease_fp_when_teeth_imperfect:.2f} | {avg_imperfect:.2f}% |")
    md.append("")
    
    # ============================================================================
    # SECTION 3: PER-IMAGE DETAILED METRICS TABLE
    # ============================================================================
    md.append("---")
    md.append("")
    md.append("## 3. PER-IMAGE DETAILED METRICS")
    md.append("")
    md.append("### Complete Per-Image Analysis (All 501 Images)")
    md.append("")
    md.append("| # | Image | Teeth TP | Teeth FP | Teeth FN | Teeth Prec | Teeth Recall | Disease TP | Disease FP | Disease FN | Disease Prec | Disease Recall | Combined Status |")
    md.append("|---|-------|---------|---------|---------|-----------|--------------|-----------|-----------|-----------|------------|----------------|-----------------|")
    
    for idx, img in enumerate(per_image_data, 1):
        teeth_perfect = img['teeth']['fn'] == 0 and img['teeth']['fp'] == 0
        disease_perfect = img['disease']['fn'] == 0 and img['disease']['fp'] == 0
        
        if teeth_perfect and disease_perfect:
            status = "✅ PERFECT"
        elif teeth_perfect and not disease_perfect:
            status = "🟢 Teeth OK"
        elif not teeth_perfect and disease_perfect:
            status = "🟡 Disease OK"
        else:
            status = "🔴 Both Issues"
        
        md.append(
            f"| {idx} | {img['image']} | {img['teeth']['tp']} | {img['teeth']['fp']} | {img['teeth']['fn']} | "
            f"{img['teeth']['precision']:.1f}% | {img['teeth']['recall']:.1f}% | "
            f"{img['disease']['tp']} | {img['disease']['fp']} | {img['disease']['fn']} | "
            f"{img['disease']['precision']:.1f}% | {img['disease']['recall']:.1f}% | "
            f"{status} |"
        )
    
    md.append("")
    
    # ============================================================================
    # SECTION 4: STATISTICAL ANALYSIS
    # ============================================================================
    md.append("---")
    md.append("")
    md.append("## 4. STATISTICAL ANALYSIS")
    md.append("")
    
    # Calculate statistics
    teeth_precisions = [img['teeth']['precision'] for img in per_image_data if img['teeth']['tp'] + img['teeth']['fp'] > 0]
    teeth_recalls = [img['teeth']['recall'] for img in per_image_data if img['teeth']['tp'] + img['teeth']['fn'] > 0]
    disease_precisions = [img['disease']['precision'] for img in per_image_data if img['disease']['tp'] + img['disease']['fp'] > 0]
    disease_recalls = [img['disease']['recall'] for img in per_image_data if img['disease']['tp'] + img['disease']['fn'] > 0]
    
    def get_stats(values):
        if not values:
            return 0, 0, 0, 0, 0
        values_sorted = sorted(values)
        return {
            'min': min(values),
            'max': max(values),
            'mean': sum(values) / len(values),
            'median': values_sorted[len(values_sorted) // 2],
            'std': (sum((x - sum(values)/len(values))**2 for x in values) / len(values))**0.5
        }
    
    teeth_prec_stats = get_stats(teeth_precisions)
    teeth_recall_stats = get_stats(teeth_recalls)
    disease_prec_stats = get_stats(disease_precisions)
    disease_recall_stats = get_stats(disease_recalls)
    
    md.append("### 4.1 Teeth Detection Statistics")
    md.append("")
    md.append("**Precision Statistics**")
    md.append(f"- Min: {teeth_prec_stats['min']:.2f}% | Max: {teeth_prec_stats['max']:.2f}% | Mean: {teeth_prec_stats['mean']:.2f}% | Median: {teeth_prec_stats['median']:.2f}% | Std Dev: {teeth_prec_stats['std']:.2f}")
    md.append("")
    
    md.append("**Recall Statistics**")
    md.append(f"- Min: {teeth_recall_stats['min']:.2f}% | Max: {teeth_recall_stats['max']:.2f}% | Mean: {teeth_recall_stats['mean']:.2f}% | Median: {teeth_recall_stats['median']:.2f}% | Std Dev: {teeth_recall_stats['std']:.2f}")
    md.append("")
    
    md.append("### 4.2 Disease Detection Statistics")
    md.append("")
    md.append("**Precision Statistics**")
    md.append(f"- Min: {disease_prec_stats['min']:.2f}% | Max: {disease_prec_stats['max']:.2f}% | Mean: {disease_prec_stats['mean']:.2f}% | Median: {disease_prec_stats['median']:.2f}% | Std Dev: {disease_prec_stats['std']:.2f}")
    md.append("")
    
    md.append("**Recall Statistics**")
    md.append(f"- Min: {disease_recall_stats['min']:.2f}% | Max: {disease_recall_stats['max']:.2f}% | Mean: {disease_recall_stats['mean']:.2f}% | Median: {disease_recall_stats['median']:.2f}% | Std Dev: {disease_recall_stats['std']:.2f}")
    md.append("")
    
    # ============================================================================
    # SECTION 5: IMAGE QUALITY CLASSIFICATION
    # ============================================================================
    md.append("---")
    md.append("")
    md.append("## 5. IMAGE QUALITY CLASSIFICATION")
    md.append("")
    
    perfect_count = sum(1 for img in per_image_data 
                       if img['teeth']['fn'] == 0 and img['teeth']['fp'] == 0 
                       and img['disease']['fn'] == 0 and img['disease']['fp'] == 0)
    
    good_teeth_count = sum(1 for img in per_image_data 
                          if img['teeth']['recall'] >= 90 and img['teeth']['precision'] >= 80)
    
    good_disease_count = sum(1 for img in per_image_data 
                            if img['disease']['recall'] >= 70 and img['disease']['precision'] >= 50)
    
    md.append("### 5.1 Detection Quality Breakdown")
    md.append("")
    md.append("| Classification | Count | Percentage |")
    md.append("|----------------|-------|------------|")
    md.append(f"| ✅ Perfect (Both Teeth & Disease) | {perfect_count} | {(perfect_count/total_images)*100:.1f}% |")
    md.append(f"| 🟢 Good Teeth (≥90% recall, ≥80% precision) | {good_teeth_count} | {(good_teeth_count/total_images)*100:.1f}% |")
    md.append(f"| 🟢 Good Disease (≥70% recall, ≥50% precision) | {good_disease_count} | {(good_disease_count/total_images)*100:.1f}% |")
    md.append(f"| ⚠️ Fair Results (Some errors) | {total_images - perfect_count} | {((total_images - perfect_count)/total_images)*100:.1f}% |")
    md.append("")
    
    # ============================================================================
    # SECTION 6: PROBLEM CLASSIFICATION
    # ============================================================================
    md.append("---")
    md.append("")
    md.append("## 6. ERROR PATTERN ANALYSIS")
    md.append("")
    
    # Categorize errors
    high_fp_teeth = sum(1 for img in per_image_data if img['teeth']['fp'] > 3)
    high_fp_disease = sum(1 for img in per_image_data if img['disease']['fp'] > 5)
    high_fn_teeth = sum(1 for img in per_image_data if img['teeth']['fn'] > 2)
    high_fn_disease = sum(1 for img in per_image_data if img['disease']['fn'] > 3)
    
    md.append("### 6.1 Images with High Error Rates")
    md.append("")
    md.append("| Error Type | Count | Percentage | Impact |")
    md.append("|-----------|-------|-----------|--------|")
    md.append(f"| High False Positive Teeth (FP > 3) | {high_fp_teeth} | {(high_fp_teeth/total_images)*100:.1f}% | ⚠️ Low precision |")
    md.append(f"| High False Positive Diseases (FP > 5) | {high_fp_disease} | {(high_fp_disease/total_images)*100:.1f}% | ⚠️ Many false alarms |")
    md.append(f"| High False Negative Teeth (FN > 2) | {high_fn_teeth} | {(high_fn_teeth/total_images)*100:.1f}% | ⚠️ Missed teeth |")
    md.append(f"| High False Negative Diseases (FN > 3) | {high_fn_disease} | {(high_fn_disease/total_images)*100:.1f}% | ⚠️ Missed diseases |")
    md.append("")
    
    # ============================================================================
    # SECTION 7: INTERPRETATION GUIDE
    # ============================================================================
    md.append("---")
    md.append("")
    md.append("## 7. METRIC INTERPRETATION & GUIDE")
    md.append("")
    
    md.append("### 7.1 Confusion Matrix Definitions")
    md.append("")
    md.append("| Term | Definition | Example |")
    md.append("|------|-----------|---------|")
    md.append("| **TP (True Positive)** | Correctly identified entity | Detected tooth that exists |")
    md.append("| **FP (False Positive)** | Incorrectly identified entity | Detected tooth that doesn't exist |")
    md.append("| **FN (False Negative)** | Missed entity | Failed to detect existing tooth |")
    md.append("| **TN (True Negative)** | Correctly rejected entity | Non-tooth correctly not detected |")
    md.append("")
    
    md.append("### 7.2 Performance Metrics Explained")
    md.append("")
    md.append("| Metric | Formula | Meaning | Range |")
    md.append("|--------|---------|---------|-------|")
    md.append("| **Precision** | TP / (TP+FP) | Of detected items, how many correct? | 0-100% |")
    md.append("| **Recall** | TP / (TP+FN) | Of actual items, how many detected? | 0-100% |")
    md.append("| **Sensitivity** | TP / (TP+FN) | True positive rate (same as Recall) | 0-100% |")
    md.append("| **F1 Score** | 2(P×R)/(P+R) | Harmonic mean of precision & recall | 0-100 |")
    md.append("")
    
    md.append("### 7.3 Combined Performance Assessment")
    md.append("")
    md.append("| Status | Teeth Condition | Disease Condition | Recommendation |")
    md.append("|--------|-----------------|------------------|-----------------|")
    md.append("| ✅ PERFECT | TP only (FP=0, FN=0) | TP only (FP=0, FN=0) | Ready for deployment |")
    md.append("| 🟢 GOOD | Recall≥90%, Prec≥80% | Recall≥70%, Prec≥50% | Monitor & deploy |")
    md.append("| 🟡 FAIR | Some errors present | Some errors present | Requires review |")
    md.append("| 🔴 POOR | High errors | High errors | Needs retraining |")
    md.append("")
    
    # ============================================================================
    # SECTION 8: KEY INSIGHTS
    # ============================================================================
    md.append("---")
    md.append("")
    md.append("## 8. KEY INSIGHTS & FINDINGS")
    md.append("")
    
    md.append(f"1. **Perfect Detection Rate:** {(perfect_count/total_images)*100:.1f}% of images have perfect teeth AND disease detection")
    md.append(f"2. **Teeth Detection Performance:** Overall recall of {overall_metrics['teeth']['recall']:.1f}% indicates strong tooth identification capability")
    md.append(f"3. **Disease Detection Gap:** Recall drops to {overall_metrics['disease']['recall']:.1f}% for disease, suggesting more work needed")
    md.append(f"4. **Precision-Recall Trade-off:** Teeth precision ({overall_metrics['teeth']['precision']:.1f}%) is higher than disease precision ({overall_metrics['disease']['precision']:.1f}%)")
    md.append(f"5. **Correlation:** Better tooth detection correlates with {avg_perfect:.1f}% avg disease recall vs {avg_imperfect:.1f}% when teeth detection is imperfect")
    md.append("")
    
    # ============================================================================
    # SECTION 9: RECOMMENDATIONS
    # ============================================================================
    md.append("---")
    md.append("")
    md.append("## 9. RECOMMENDATIONS FOR IMPROVEMENT")
    md.append("")
    
    md.append("### 9.1 For Teeth Detection")
    md.append("")
    if high_fp_teeth > total_images * 0.1:
        md.append("- **HIGH PRIORITY:** Reduce false positives - tune confidence threshold")
    if high_fn_teeth > total_images * 0.1:
        md.append("- **HIGH PRIORITY:** Reduce false negatives - enhance augmentation for hard cases")
    md.append("- Implement confidence score thresholding for marginal detections")
    md.append("- Collect more training data for underperforming tooth types")
    md.append("")
    
    md.append("### 9.2 For Disease Detection")
    md.append("")
    if high_fp_disease > total_images * 0.15:
        md.append("- **HIGH PRIORITY:** Reduce false positive diseases - refine detection threshold")
    if high_fn_disease > total_images * 0.15:
        md.append("- **HIGH PRIORITY:** Reduce false negative diseases - improve model sensitivity")
    md.append("- Use focal loss to handle disease class imbalance")
    md.append("- Apply hard example mining for frequently missed diseases")
    md.append("")
    
    md.append("### 9.3 For Combined System")
    md.append("")
    md.append("- Leverage tooth detection to constrain disease detection (diseases only on detected teeth)")
    md.append("- Implement post-processing: filter out diseases from FP teeth")
    md.append("- Add confidence-weighted output: show confidence scores for end-user review")
    md.append("- Target 90%+ teeth recall and 75%+ disease recall for clinical deployment")
    md.append("")
    
    # ============================================================================
    # FOOTER
    # ============================================================================
    md.append("---")
    md.append("")
    md.append("## 10. SUMMARY")
    md.append("")
    md.append("This comprehensive analysis shows the per-image, overall, and combined metrics for tooth")
    md.append("identification and disease detection across all validation images. The data demonstrates")
    md.append("that while tooth detection is reasonably reliable, disease detection requires further")
    md.append("refinement. The strong correlation between perfect tooth detection and disease detection")
    md.append("suggests that improving tooth detection would likely improve overall system performance.")
    md.append("")
    md.append("---")
    md.append("*Generated: 2026-07-11 | Analysis of 501 validation images*")
    
    return "\n".join(md)

if __name__ == "__main__":
    markdown_report = generate_comprehensive_report()
    
    # Save to file
    output_file = r"d:\WILP\comprehensive_validation_report.md"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(markdown_report)
    
    print(f"Comprehensive report generated: {output_file}")
