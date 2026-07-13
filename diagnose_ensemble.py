"""
Comprehensive ensemble diagnostics
"""
import json
from pathlib import Path
import numpy as np

print("="*80)
print("ENSEMBLE DIAGNOSTICS")
print("="*80)

# 1. Check if Stage-2 models exist and are trained
print("\n[1] CHECKING STAGE-2 MODELS")
print("-"*80)
model_dir = Path("outputs/disease_validation_option2/stage2_models")
if model_dir.exists():
    models = list(model_dir.glob("*.pt"))
    print(f"✓ Found {len(models)} trained models:")
    for m in sorted(models):
        size_mb = m.stat().st_size / (1024*1024)
        print(f"  • {m.name} ({size_mb:.1f} MB)")
else:
    print("✗ Stage-2 models directory not found!")

# 2. Check ground truth labels for test set
print("\n[2] CHECKING TEST SET GROUND TRUTH LABELS")
print("-"*80)
labels_dir = Path("data/processed/labels_ext/test")
if labels_dir.exists():
    label_files = list(labels_dir.glob("*.txt"))
    print(f"✓ Found {len(label_files)} label files")
    
    # Sample some labels
    positive_diseases = {'has_caries': 0, 'has_deepcaries': 0, 'has_lesion': 0, 'has_impacted': 0}
    total_labels = 0
    
    for label_file in label_files[:10]:  # Check first 10
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 9:
                    total_labels += 1
                    has_caries = int(parts[5])
                    has_deepcaries = int(parts[6])
                    has_lesion = int(parts[7])
                    has_impacted = int(parts[8])
                    
                    if has_caries: positive_diseases['has_caries'] += 1
                    if has_deepcaries: positive_diseases['has_deepcaries'] += 1
                    if has_lesion: positive_diseases['has_lesion'] += 1
                    if has_impacted: positive_diseases['has_impacted'] += 1
    
    print(f"\n  Sample from first 10 images:")
    print(f"  Total labels: {total_labels}")
    print(f"  Disease breakdown:")
    for disease, count in positive_diseases.items():
        pct = 100 * count / total_labels if total_labels > 0 else 0
        print(f"    • {disease}: {count} ({pct:.1f}%)")
else:
    print("✗ Labels directory not found!")

# 3. Check test images exist
print("\n[3] CHECKING TEST IMAGES")
print("-"*80)
test_images_dir = Path("data/processed/images/test")
if test_images_dir.exists():
    images = list(test_images_dir.glob("*.png"))
    print(f"✓ Found {len(images)} test images")
else:
    print("✗ Test images directory not found!")

# 4. Check if ensemble output was created
print("\n[4] CHECKING ENSEMBLE OUTPUT")
print("-"*80)
ensemble_out = Path("outputs/ensemble_predictions")
if ensemble_out.exists():
    files = list(ensemble_out.glob("*"))
    print(f"✓ Found ensemble output directory with {len(files)} files:")
    for f in sorted(files):
        if f.is_file():
            size_kb = f.stat().st_size / 1024
            print(f"  • {f.name} ({size_kb:.1f} KB)")
        else:
            print(f"  • {f.name}/ (directory)")
    
    # Try to load and inspect predictions
    pred_file = ensemble_out / "test_predictions.json"
    if pred_file.exists():
        try:
            with open(pred_file, 'r') as f:
                predictions = json.load(f)
            
            print(f"\n  Predictions JSON inspection:")
            print(f"    • Images: {len(predictions)}")
            
            # Count teeth and diseases
            total_teeth = 0
            total_positive = 0
            disease_counts = {'has_caries': 0, 'has_deepcaries': 0, 'has_lesion': 0, 'has_impacted': 0}
            scores_list = {'stage1': [], 'stage2': [], 'fused': []}
            
            for img, results in predictions.items():
                for tooth in results['teeth']:
                    total_teeth += 1
                    
                    # Check if any disease
                    diseases = tooth.get('diseases', {})
                    if any(diseases.values()):
                        total_positive += 1
                    
                    # Count by disease
                    for disease in disease_counts:
                        if diseases.get(disease, False):
                            disease_counts[disease] += 1
                    
                    # Collect scores
                    fused = tooth.get('fused_scores', {})
                    s1 = tooth.get('stage1_scores', {})
                    s2 = tooth.get('stage2_scores', {})
                    
                    for disease in disease_counts:
                        if disease in fused:
                            scores_list['fused'].append(fused[disease])
                        if disease in s1:
                            scores_list['stage1'].append(s1[disease])
                        if disease in s2:
                            scores_list['stage2'].append(s2[disease])
            
            print(f"    • Total teeth: {total_teeth}")
            print(f"    • Predicted as diseased: {total_positive} ({100*total_positive/total_teeth:.1f}%)")
            print(f"\n    Disease predictions:")
            for disease, count in disease_counts.items():
                pct = 100 * count / total_teeth if total_teeth > 0 else 0
                print(f"      • {disease}: {count} ({pct:.1f}%)")
            
            # Score statistics
            print(f"\n    Score statistics (across all diseases):")
            for score_type in ['stage1', 'stage2', 'fused']:
                if scores_list[score_type]:
                    scores = np.array(scores_list[score_type])
                    print(f"      {score_type}:")
                    print(f"        Min: {scores.min():.4f}, Max: {scores.max():.4f}")
                    print(f"        Mean: {scores.mean():.4f}, Std: {scores.std():.4f}")
        except Exception as e:
            print(f"  ✗ Error reading predictions: {e}")
else:
    print("✗ Ensemble predictions directory not found!")

print("\n" + "="*80)
print("DIAGNOSTIC COMPLETE")
print("="*80)
