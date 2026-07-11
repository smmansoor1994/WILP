"""
Extract per-image metrics from validation log and generate comprehensive report
"""
import re
import json
from collections import defaultdict

def parse_per_image_metrics(log_file):
    """Parse the validation log and extract per-image metrics"""
    
    per_image_data = []
    
    # Read log file with UTF-8 encoding
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Split by image processing blocks
    image_blocks = re.split(r'(?=\d{2}:\d{2}:\d{2} \[INFO\] Processing:)', content, flags=re.MULTILINE)
    
    for block in image_blocks:
        if not block.strip():
            continue
        
        # Extract image name
        image_match = re.search(r'Processing:\s+(\S+\.png)', block)
        if not image_match:
            continue
        
        image_name = image_match.group(1)
        
        # Extract teeth metrics
        teeth_tp = re.search(r'Teeth Detection:.*?TP=(\d+)', block)
        teeth_fp = re.search(r'Teeth Detection:.*?FP=(\d+)', block)
        teeth_fn = re.search(r'Teeth Detection:.*?FN=(\d+)', block)
        teeth_recall = re.search(r'Teeth Detection:.*?Recall=\s*([\d.]+)%', block)
        teeth_precision = re.search(r'Teeth Detection:.*?Precision=\s*([\d.]+)%', block)
        
        # Extract disease metrics
        disease_tp = re.search(r'Disease Detection:.*?TP=(\d+)', block)
        disease_fp = re.search(r'Disease Detection:.*?FP=(\d+)', block)
        disease_fn = re.search(r'Disease Detection:.*?FN=(\d+)', block)
        disease_recall = re.search(r'Disease Detection:.*?Recall=\s*([\d.]+)%', block)
        disease_precision = re.search(r'Disease Detection:.*?Precision=\s*([\d.]+)%', block)
        
        # Extract ground truth and detected counts
        gt_teeth_count = re.search(r'Ground Truth:\s+(\d+)\s+teeth', block)
        gt_disease_count = re.search(r'Ground Truth:.*?\|\s+(\d+)\s+diseases', block)
        pred_teeth_count = re.search(r'Predictions:\s+(\d+)\s+teeth', block)
        pred_disease_count = re.search(r'Predictions:.*?\|\s+(\d+)\s+diseases', block)
        
        if not (teeth_tp and disease_tp):
            continue
        
        # Build image data
        image_data = {
            'image': image_name,
            'teeth': {
                'tp': int(teeth_tp.group(1)),
                'fp': int(teeth_fp.group(1)),
                'fn': int(teeth_fn.group(1)),
                'recall': float(teeth_recall.group(1)) if teeth_recall else 0,
                'precision': float(teeth_precision.group(1)) if teeth_precision else 0,
                'gt_count': int(gt_teeth_count.group(1)) if gt_teeth_count else 0,
                'pred_count': int(pred_teeth_count.group(1)) if pred_teeth_count else 0,
            },
            'disease': {
                'tp': int(disease_tp.group(1)),
                'fp': int(disease_fp.group(1)),
                'fn': int(disease_fn.group(1)),
                'recall': float(disease_recall.group(1)) if disease_recall else 0,
                'precision': float(disease_precision.group(1)) if disease_precision else 0,
                'gt_count': int(gt_disease_count.group(1)) if gt_disease_count else 0,
                'pred_count': int(pred_disease_count.group(1)) if pred_disease_count else 0,
            }
        }
        
        # Calculate F1 and specificity
        for task in ['teeth', 'disease']:
            data = image_data[task]
            if data['tp'] + data['fp'] > 0:
                data['specificity'] = 0  # Not enough info for TN
            if data['precision'] + data['recall'] > 0:
                data['f1_score'] = 2 * (data['precision'] * data['recall']) / (data['precision'] + data['recall'])
            else:
                data['f1_score'] = 0
        
        per_image_data.append(image_data)
    
    return per_image_data

def calculate_overall_metrics(per_image_data):
    """Calculate overall metrics across all images"""
    
    overall = {
        'teeth': {'tp': 0, 'fp': 0, 'fn': 0},
        'disease': {'tp': 0, 'fp': 0, 'fn': 0}
    }
    
    for img_data in per_image_data:
        for task in ['teeth', 'disease']:
            for metric in ['tp', 'fp', 'fn']:
                overall[task][metric] += img_data[task][metric]
    
    # Calculate metrics
    for task in ['teeth', 'disease']:
        tp = overall[task]['tp']
        fp = overall[task]['fp']
        fn = overall[task]['fn']
        
        overall[task]['precision'] = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0
        overall[task]['recall'] = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0
        overall[task]['f1_score'] = 2 * (overall[task]['precision'] * overall[task]['recall']) / (overall[task]['precision'] + overall[task]['recall']) if (overall[task]['precision'] + overall[task]['recall']) > 0 else 0
    
    return overall

def main():
    log_file = r"d:\WILP\validation_test\validation log.txt"
    
    per_image_data = parse_per_image_metrics(log_file)
    overall_metrics = calculate_overall_metrics(per_image_data)
    
    # Save data
    with open(r"d:\WILP\per_image_metrics.json", 'w') as f:
        json.dump(per_image_data, f, indent=2)
    
    with open(r"d:\WILP\overall_metrics.json", 'w') as f:
        json.dump(overall_metrics, f, indent=2)
    
    print(f"Parsed {len(per_image_data)} images")
    print("\nOverall Metrics:")
    print(f"Teeth - TP: {overall_metrics['teeth']['tp']}, FP: {overall_metrics['teeth']['fp']}, FN: {overall_metrics['teeth']['fn']}")
    print(f"Teeth - Precision: {overall_metrics['teeth']['precision']:.2f}%, Recall: {overall_metrics['teeth']['recall']:.2f}%, F1: {overall_metrics['teeth']['f1_score']:.2f}")
    print(f"Disease - TP: {overall_metrics['disease']['tp']}, FP: {overall_metrics['disease']['fp']}, FN: {overall_metrics['disease']['fn']}")
    print(f"Disease - Precision: {overall_metrics['disease']['precision']:.2f}%, Recall: {overall_metrics['disease']['recall']:.2f}%, F1: {overall_metrics['disease']['f1_score']:.2f}")

if __name__ == "__main__":
    main()
