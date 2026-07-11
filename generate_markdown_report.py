"""
Generate comprehensive markdown report from validation metrics
"""
import json
import math

def calculate_metrics(tp, tn, fp, fn):
    """Calculate precision, recall, sensitivity, specificity"""
    metrics = {}
    
    # Precision (TP / (TP + FP))
    metrics['precision'] = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0
    
    # Recall / Sensitivity (TP / (TP + FN))
    metrics['recall'] = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0
    
    # Specificity (TN / (TN + FP))
    metrics['specificity'] = (tn / (tn + fp)) * 100 if (tn + fp) > 0 else 0
    
    # Sensitivity = Recall
    metrics['sensitivity'] = metrics['recall']
    
    # F1 Score
    if metrics['precision'] + metrics['recall'] > 0:
        metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
    else:
        metrics['f1_score'] = 0
    
    return metrics

def generate_markdown_report():
    """Generate comprehensive markdown report"""
    
    # Load data
    with open(r"d:\WILP\validation_metrics.json", 'r') as f:
        data = json.load(f)
    
    per_tooth_teeth = data['per_tooth_teeth']
    overall_teeth = data['overall_teeth']
    overall_disease = data['overall_disease']
    all_teeth = sorted(data['all_teeth'])
    
    # Sort teeth by FDI number for proper ordering
    fdi_order = {
        '11': 0, '12': 1, '13': 2, '14': 3, '15': 4, '16': 5, '17': 6, '18': 7,
        '21': 8, '22': 9, '23': 10, '24': 11, '25': 12, '26': 13, '27': 14, '28': 15,
        '31': 16, '32': 17, '33': 18, '34': 19, '35': 20, '36': 21, '37': 22, '38': 23,
        '41': 24, '42': 25, '43': 26, '44': 27, '45': 28, '46': 29, '47': 30, '48': 31,
    }
    
    def get_fdi_number(fdi_str):
        return fdi_str.replace('FDI', '')
    
    sorted_teeth = sorted(all_teeth, key=lambda x: fdi_order.get(get_fdi_number(x), 999))
    
    # Calculate overall metrics
    overall_teeth_metrics = calculate_metrics(
        overall_teeth['TP'], 0, overall_teeth['FP'], overall_teeth['FN']
    )
    overall_disease_metrics = calculate_metrics(
        overall_disease['TP'], 0, overall_disease['FP'], overall_disease['FN']
    )
    
    # Start building markdown
    md = []
    md.append("# Dental X-ray Analysis - Comprehensive Validation Report")
    md.append("")
    md.append("## Executive Summary")
    md.append("")
    md.append("This report presents detailed per-tooth metrics for teeth identification and disease detection,")
    md.append("along with comprehensive performance metrics including Precision, Recall, Specificity, and Sensitivity.")
    md.append("")
    md.append(f"**Validation Dataset:** 470 images")
    md.append(f"**Analysis Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md.append("")
    
    # ============================================================================
    # SECTION 1: OVERALL METRICS
    # ============================================================================
    md.append("---")
    md.append("")
    md.append("## 1. OVERALL PERFORMANCE METRICS")
    md.append("")
    
    # Overall Teeth Detection
    md.append("### 1.1 Teeth Identification (Overall)")
    md.append("")
    md.append("| Metric | Value |")
    md.append("|--------|-------|")
    md.append(f"| **True Positives (TP)** | {overall_teeth['TP']} |")
    md.append(f"| **False Positives (FP)** | {overall_teeth['FP']} |")
    md.append(f"| **False Negatives (FN)** | {overall_teeth['FN']} |")
    md.append(f"| **Total Detected** | {overall_teeth['TP'] + overall_teeth['FP']} |")
    md.append(f"| **Total Ground Truth** | {overall_teeth['TP'] + overall_teeth['FN']} |")
    md.append("")
    
    md.append("#### Performance Indicators")
    md.append("")
    md.append("| Metric | Value |")
    md.append("|--------|-------|")
    md.append(f"| **Precision** | {overall_teeth_metrics['precision']:.2f}% |")
    md.append(f"| **Recall** | {overall_teeth_metrics['recall']:.2f}% |")
    md.append(f"| **Sensitivity** | {overall_teeth_metrics['sensitivity']:.2f}% |")
    md.append(f"| **Specificity** | {overall_teeth_metrics['specificity']:.2f}% |")
    md.append(f"| **F1 Score** | {overall_teeth_metrics['f1_score']:.2f} |")
    md.append("")
    
    # Overall Disease Detection
    md.append("### 1.2 Disease Identification (Overall)")
    md.append("")
    md.append("| Metric | Value |")
    md.append("|--------|-------|")
    md.append(f"| **True Positives (TP)** | {overall_disease['TP']} |")
    md.append(f"| **False Positives (FP)** | {overall_disease['FP']} |")
    md.append(f"| **False Negatives (FN)** | {overall_disease['FN']} |")
    md.append(f"| **Total Detected** | {overall_disease['TP'] + overall_disease['FP']} |")
    md.append(f"| **Total Ground Truth** | {overall_disease['TP'] + overall_disease['FN']} |")
    md.append("")
    
    md.append("#### Performance Indicators")
    md.append("")
    md.append("| Metric | Value |")
    md.append("|--------|-------|")
    md.append(f"| **Precision** | {overall_disease_metrics['precision']:.2f}% |")
    md.append(f"| **Recall** | {overall_disease_metrics['recall']:.2f}% |")
    md.append(f"| **Sensitivity** | {overall_disease_metrics['sensitivity']:.2f}% |")
    md.append(f"| **Specificity** | {overall_disease_metrics['specificity']:.2f}% |")
    md.append(f"| **F1 Score** | {overall_disease_metrics['f1_score']:.2f} |")
    md.append("")
    
    # ============================================================================
    # SECTION 2: PER-TOOTH TEETH IDENTIFICATION METRICS
    # ============================================================================
    md.append("---")
    md.append("")
    md.append("## 2. PER-TOOTH IDENTIFICATION METRICS")
    md.append("")
    md.append("### Confusion Matrix and Performance for Each Tooth")
    md.append("")
    md.append("| FDI Tooth | Missed | TP | FP | FN | Precision | Recall | Sensitivity | F1 Score |")
    md.append("|-----------|--------|----|----|----|-----------|--------|-------------|----------|")
    
    for tooth in sorted_teeth:
        if tooth in per_tooth_teeth:
            metrics = per_tooth_teeth[tooth]
            tp = metrics['TP']
            fp = metrics['FP']
            fn = metrics['FN']
            tn = metrics['TN']
            
            tooth_metrics = calculate_metrics(tp, tn, fp, fn)
            
            missed = "❌" if fn > 0 else "✓"
            
            md.append(
                f"| **{tooth}** | {missed} | {tp} | {fp} | {fn} | "
                f"{tooth_metrics['precision']:.1f}% | {tooth_metrics['recall']:.1f}% | "
                f"{tooth_metrics['sensitivity']:.1f}% | {tooth_metrics['f1_score']:.2f} |"
            )
    
    md.append("")
    
    # ============================================================================
    # SECTION 3: DETAILED PER-TOOTH ANALYSIS
    # ============================================================================
    md.append("---")
    md.append("")
    md.append("## 3. DETAILED PER-TOOTH ANALYSIS")
    md.append("")
    
    for tooth in sorted_teeth:
        if tooth in per_tooth_teeth:
            metrics = per_tooth_teeth[tooth]
            tp = metrics['TP']
            fp = metrics['FP']
            fn = metrics['FN']
            tn = metrics['TN']
            
            tooth_metrics = calculate_metrics(tp, tn, fp, fn)
            
            md.append(f"### Tooth {tooth}")
            md.append("")
            md.append("#### Confusion Matrix")
            md.append("")
            md.append("| Metric | Count |")
            md.append("|--------|-------|")
            md.append(f"| True Positives (TP) - Correctly identified | {tp} |")
            md.append(f"| False Positives (FP) - Incorrectly identified | {fp} |")
            md.append(f"| False Negatives (FN) - Missed teeth | {fn} |")
            md.append(f"| True Negatives (TN) | {tn} |")
            md.append(f"| **Total Detected** | {tp + fp} |")
            md.append(f"| **Total Ground Truth** | {tp + fn} |")
            md.append("")
            
            md.append("#### Performance Metrics")
            md.append("")
            md.append("| Metric | Formula | Value |")
            md.append("|--------|---------|-------|")
            md.append(f"| **Precision** | TP / (TP + FP) | {tooth_metrics['precision']:.2f}% |")
            md.append(f"| **Recall** | TP / (TP + FN) | {tooth_metrics['recall']:.2f}% |")
            md.append(f"| **Sensitivity** | TP / (TP + FN) | {tooth_metrics['sensitivity']:.2f}% |")
            md.append(f"| **Specificity** | TN / (TN + FP) | {tooth_metrics['specificity']:.2f}% |")
            md.append(f"| **F1 Score** | 2 × (P × R) / (P + R) | {tooth_metrics['f1_score']:.2f} |")
            md.append("")
            
            # Quality assessment
            quality = "🟢 EXCELLENT"
            if tooth_metrics['precision'] < 70 or tooth_metrics['recall'] < 70:
                quality = "🟡 NEEDS IMPROVEMENT"
            if tooth_metrics['precision'] < 50 or tooth_metrics['recall'] < 50:
                quality = "🔴 POOR"
            
            md.append(f"**Quality Assessment:** {quality}")
            md.append("")
    
    # ============================================================================
    # SECTION 4: INTERPRETATION GUIDE
    # ============================================================================
    md.append("---")
    md.append("")
    md.append("## 4. METRIC INTERPRETATION GUIDE")
    md.append("")
    md.append("### Confusion Matrix Terms")
    md.append("")
    md.append("- **TP (True Positive):** Tooth correctly identified as present")
    md.append("- **FP (False Positive):** Non-existent tooth incorrectly identified")
    md.append("- **FN (False Negative):** Existing tooth was missed/not identified")
    md.append("- **TN (True Negative):** Non-existent tooth correctly not identified")
    md.append("")
    
    md.append("### Performance Metrics Explained")
    md.append("")
    md.append("| Metric | Formula | Meaning | Ideal Value |")
    md.append("|--------|---------|---------|-------------|")
    md.append("| **Precision** | TP/(TP+FP) | Of all detected teeth, how many were correct? | 100% |")
    md.append("| **Recall** | TP/(TP+FN) | Of all actual teeth, how many were detected? | 100% |")
    md.append("| **Sensitivity** | TP/(TP+FN) | Same as Recall - True Positive Rate | 100% |")
    md.append("| **Specificity** | TN/(TN+FP) | How many non-teeth were correctly rejected? | 100% |")
    md.append("| **F1 Score** | 2(P×R)/(P+R) | Harmonic mean of Precision & Recall | 1.00 |")
    md.append("")
    
    md.append("### Performance Thresholds")
    md.append("")
    md.append("| Range | Assessment | Recommendation |")
    md.append("|-------|------------|-----------------|")
    md.append("| 90-100% | Excellent | Use for clinical deployment |")
    md.append("| 80-89% | Good | Suitable with monitoring |")
    md.append("| 70-79% | Fair | Needs improvement |")
    md.append("| Below 70% | Poor | Requires significant tuning |")
    md.append("")
    
    # ============================================================================
    # SECTION 5: KEY FINDINGS
    # ============================================================================
    md.append("---")
    md.append("")
    md.append("## 5. KEY FINDINGS & INSIGHTS")
    md.append("")
    
    # Find best and worst performing teeth
    best_teeth = []
    worst_teeth = []
    
    for tooth in sorted_teeth:
        if tooth in per_tooth_teeth:
            metrics = per_tooth_teeth[tooth]
            tp = metrics['TP']
            fp = metrics['FP']
            fn = metrics['FN']
            
            if tp + fn > 0:  # Only if we have data
                recall = (tp / (tp + fn)) * 100
                best_teeth.append((tooth, recall))
                worst_teeth.append((tooth, recall))
    
    best_teeth.sort(key=lambda x: x[1], reverse=True)
    worst_teeth.sort(key=lambda x: x[1])
    
    md.append("### Top 5 Performing Teeth (by Recall)")
    md.append("")
    for i, (tooth, recall) in enumerate(best_teeth[:5], 1):
        md.append(f"{i}. **{tooth}**: {recall:.1f}% recall")
    md.append("")
    
    md.append("### Top 5 Underperforming Teeth (by Recall)")
    md.append("")
    for i, (tooth, recall) in enumerate(worst_teeth[:5], 1):
        md.append(f"{i}. **{tooth}**: {recall:.1f}% recall ⚠️")
    md.append("")
    
    # Calculate statistics
    all_recalls = [(per_tooth_teeth[t]['TP'] / (per_tooth_teeth[t]['TP'] + per_tooth_teeth[t]['FN']) * 100)
                   for t in sorted_teeth 
                   if t in per_tooth_teeth and per_tooth_teeth[t]['TP'] + per_tooth_teeth[t]['FN'] > 0]
    
    if all_recalls:
        avg_recall = sum(all_recalls) / len(all_recalls)
        md.append(f"### Average Recall Across All Teeth: **{avg_recall:.1f}%**")
        md.append("")
    
    # ============================================================================
    # SECTION 6: RECOMMENDATIONS
    # ============================================================================
    md.append("---")
    md.append("")
    md.append("## 6. RECOMMENDATIONS")
    md.append("")
    
    md.append("### For System Improvement")
    md.append("")
    md.append("1. **High False Positive Rate (FP):** Review detection threshold tuning")
    md.append("2. **False Negative Teeth:** Focus on augmentation for frequently missed teeth")
    md.append("3. **Precision-Recall Trade-off:** Adjust model confidence thresholds based on clinical needs")
    md.append("")
    
    md.append("### For Clinical Deployment")
    md.append("")
    md.append("1. Use this model as **screening assistant**, not primary diagnostic tool")
    md.append("2. Implement **human-in-the-loop** validation for teeth with recall < 80%")
    md.append("3. Maintain **confidence score thresholds** for high-stakes decisions")
    md.append("")
    
    md.append("### Priority Improvements")
    md.append("")
    underperforming = worst_teeth[:5]
    md.append("Focus training efforts on these underperforming teeth:")
    for tooth, recall in underperforming[:3]:
        md.append(f"- **{tooth}**: Increase training samples, data augmentation (Current recall: {recall:.1f}%)")
    md.append("")
    
    # ============================================================================
    # FOOTER
    # ============================================================================
    md.append("---")
    md.append("")
    md.append("## 7. TECHNICAL NOTES")
    md.append("")
    md.append("- Analysis based on 470 validation images")
    md.append("- All 32 FDI tooth numbers analyzed")
    md.append("- Metrics calculated independently per tooth")
    md.append("- Performance thresholds based on clinical ML standards")
    md.append("")
    md.append("---")
    md.append("*Report Generated Automatically - For detailed analysis, review validation logs*")
    
    return "\n".join(md)

if __name__ == "__main__":
    import pandas as pd
    
    markdown_report = generate_markdown_report()
    
    # Save to file
    output_file = r"d:\WILP\validation_metrics_report.md"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(markdown_report)
    
    print(f"Report generated successfully: {output_file}")
