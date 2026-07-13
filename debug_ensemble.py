"""Debug ensemble predictions"""
import json

with open('outputs/ensemble_predictions/test_predictions.json', 'r') as f:
    data = json.load(f)

print("="*80)
print("ENSEMBLE PREDICTIONS DEBUG")
print("="*80)

# Check first 3 images
for i, (img, results) in enumerate(list(data.items())[:3]):
    print(f"\n[{i+1}] Image: {img}")
    print(f"    Teeth detected: {len(results['teeth'])}")
    
    if results['teeth']:
        tooth = results['teeth'][0]
        print(f"\n    First tooth FDI: {tooth.get('fdi')}")
        print(f"    Diseases: {tooth.get('diseases')}")
        print(f"    Confidence: {tooth.get('confidence'):.4f}")
        print(f"\n    Fused scores: {tooth.get('fused_scores')}")
        print(f"    Stage-1 scores: {tooth.get('stage1_scores')}")
        print(f"    Stage-2 scores: {tooth.get('stage2_scores')}")

# Check for any positive predictions
print("\n" + "="*80)
print("CHECKING FOR POSITIVE PREDICTIONS")
print("="*80)

total_teeth = 0
predicted_disease = 0

for img, results in data.items():
    for tooth in results['teeth']:
        total_teeth += 1
        diseases = tooth.get('diseases', [])
        if any(diseases):
            predicted_disease += 1
            print(f"Found positive: {img} FDI {tooth['fdi']} - {diseases}")

print(f"\nTotal teeth: {total_teeth}")
print(f"Predicted as diseased: {predicted_disease}")
print(f"Percentage diseased: {100*predicted_disease/total_teeth:.1f}%")
