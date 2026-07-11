"""
Parse validation log and generate comprehensive metrics report
"""
import re
from collections import defaultdict
import json

def parse_validation_log(log_file):
    """Parse the validation log and extract metrics"""
    
    # Initialize data structures
    per_tooth_teeth_metrics = defaultdict(lambda: {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0})
    per_tooth_disease_metrics = defaultdict(lambda: {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0})
    
    overall_teeth_metrics = {'TP': 0, 'FP': 0, 'FN': 0}
    overall_disease_metrics = {'TP': 0, 'FP': 0, 'FN': 0}
    
    # Set of all unique teeth FDI numbers seen
    all_teeth_fdi = set()
    
    # Read log file with UTF-8 encoding
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Split by image processing blocks
    image_blocks = re.split(r'(?=^\d{2}:\d{2}:\d{2} \[INFO\] Processing:)', content, flags=re.MULTILINE)
    
    for block in image_blocks:
        if not block.strip():
            continue
            
        # Extract image metrics
        teeth_tp = re.search(r'Teeth Detection:.*?TP=(\d+)', block)
        teeth_fp = re.search(r'Teeth Detection:.*?FP=(\d+)', block)
        teeth_fn = re.search(r'Teeth Detection:.*?FN=(\d+)', block)
        
        disease_tp = re.search(r'Disease Detection:.*?TP=(\d+)', block)
        disease_fp = re.search(r'Disease Detection:.*?FP=(\d+)', block)
        disease_fn = re.search(r'Disease Detection:.*?FN=(\d+)', block)
        
        if not (teeth_tp and disease_tp):
            continue
        
        # Get image name
        image_match = re.search(r'Processing:\s+(\S+\.png)', block)
        image_name = image_match.group(1) if image_match else "unknown"
        
        # Add to overall metrics
        teeth_tp_val = int(teeth_tp.group(1))
        teeth_fp_val = int(teeth_fp.group(1))
        teeth_fn_val = int(teeth_fn.group(1))
        
        disease_tp_val = int(disease_tp.group(1))
        disease_fp_val = int(disease_fp.group(1))
        disease_fn_val = int(disease_fn.group(1))
        
        overall_teeth_metrics['TP'] += teeth_tp_val
        overall_teeth_metrics['FP'] += teeth_fp_val
        overall_teeth_metrics['FN'] += teeth_fn_val
        
        overall_disease_metrics['TP'] += disease_tp_val
        overall_disease_metrics['FP'] += disease_fp_val
        overall_disease_metrics['FN'] += disease_fn_val
        
        # Extract ground truth and detected teeth
        gt_teeth_match = re.search(r'Ground Truth Teeth:\s+(.+?)(?=\n\s+Detected)', block, re.DOTALL)
        detected_teeth_match = re.search(r'Detected Teeth:\s+(.+?)(?=\n\s+[❌⚠️]|\n\n|\Z)', block, re.DOTALL)
        
        if gt_teeth_match and detected_teeth_match:
            gt_teeth_str = gt_teeth_match.group(1).strip()
            detected_teeth_str = detected_teeth_match.group(1).strip()
            
            gt_teeth = [t.strip() for t in gt_teeth_str.split(',')]
            detected_teeth = [t.strip() for t in detected_teeth_str.split(',')]
            
            all_teeth_fdi.update(gt_teeth)
            all_teeth_fdi.update(detected_teeth)
            
            # Per-tooth teeth identification metrics
            for tooth in gt_teeth:
                if tooth in detected_teeth:
                    per_tooth_teeth_metrics[tooth]['TP'] += 1
                else:
                    per_tooth_teeth_metrics[tooth]['FN'] += 1
            
            for tooth in detected_teeth:
                if tooth not in gt_teeth:
                    per_tooth_teeth_metrics[tooth]['FP'] += 1
        
        # Extract missed diseases and false diseases
        missed_diseases_match = re.search(r'❌ MISSED DISEASES:\s+(.+?)(?=\n\s+⚠️|$)', block, re.DOTALL)
        false_diseases_match = re.search(r'⚠️\s+FALSE DISEASES:\s+(.+?)(?=\n|\Z)', block, re.DOTALL)
        
        # Extract per-tooth disease metrics from the block
        missed_diseases = {}
        if missed_diseases_match:
            missed_diseases_str = missed_diseases_match.group(1).strip()
            for disease_entry in missed_diseases_str.split(','):
                disease_entry = disease_entry.strip()
                if ':' in disease_entry:
                    tooth, disease = disease_entry.split(':', 1)
                    tooth = tooth.strip()
                    disease = disease.strip()
                    per_tooth_disease_metrics[tooth]['FN'] += 1
        
        false_diseases = {}
        if false_diseases_match:
            false_diseases_str = false_diseases_match.group(1).strip()
            for disease_entry in false_diseases_str.split(','):
                disease_entry = disease_entry.strip()
                if ':' in disease_entry:
                    tooth, disease = disease_entry.split(':', 1)
                    tooth = tooth.strip()
                    disease = disease.strip()
                    per_tooth_disease_metrics[tooth]['FP'] += 1
        
        # Extract TP diseases (from ground truth teeth list - simple approach)
        # For now, we'll infer: if tooth is detected, any disease in GT is TP if not in missed
        if gt_teeth_match:
            for tooth in gt_teeth:
                # Check if this tooth had any diseases in GT
                tooth_in_missed = any(tooth in str(d) for d in missed_diseases)
                tooth_in_false = any(tooth in str(d) for d in false_diseases)
                
                # Simple heuristic: if tooth was detected and no disease was missed, count as TP
                if tooth in detected_teeth and not tooth_in_missed:
                    # This is a rough approximation
                    pass
    
    return {
        'per_tooth_teeth': dict(per_tooth_teeth_metrics),
        'per_tooth_disease': dict(per_tooth_disease_metrics),
        'overall_teeth': overall_teeth_metrics,
        'overall_disease': overall_disease_metrics,
        'all_teeth': sorted(list(all_teeth_fdi))
    }

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
    
    return metrics

def main():
    log_file = r"d:\WILP\validation_test\validation log.txt"
    
    data = parse_validation_log(log_file)
    
    # Save data for markdown generation
    import json
    with open(r"d:\WILP\validation_metrics.json", 'w') as f:
        json.dump(data, f, indent=2)
    
    print("Metrics extracted successfully!")
    print(f"Total unique teeth: {len(data['all_teeth'])}")
    print(f"Overall Teeth - TP: {data['overall_teeth']['TP']}, FP: {data['overall_teeth']['FP']}, FN: {data['overall_teeth']['FN']}")
    print(f"Overall Disease - TP: {data['overall_disease']['TP']}, FP: {data['overall_disease']['FP']}, FN: {data['overall_disease']['FN']}")

if __name__ == "__main__":
    main()
