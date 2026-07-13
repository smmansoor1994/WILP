from pathlib import Path
from src.inference.disease_validation import DiseaseValidationPipeline

pipeline = DiseaseValidationPipeline(
    model_weights='weights/archon_best.pt',  # Reuse existing
    device='cpu'
)

# Step 1: Quick analysis (5 min)
print("Step 1: Extracting crops...")
s1 = pipeline.stage_1_enumerate_and_crop(
    images_dir=Path('data/processed/images/train'),
    labels_dir=Path('data/processed/labels/train'),
    output_crops_dir=Path('outputs/stage1_crops'),
    sample_images=100  # Quick test
)
print(f"  • Crops extracted: {s1['total_crops']}")
print(f"  • With ground truth: {s1['crops_with_gt']}\n")

# Step 2: Train disease classifiers on crops (30-45 min)
print("Step 2: Training disease classifiers on crops...")
s2 = pipeline.stage_2_train_classifiers(
    crops_dir=Path('outputs/stage1_crops'),
    output_models_dir=Path('outputs/stage2_models'),
    epochs=20,  # Quick
    val_split=0.2
)
print(f"  • Models trained\n")

# Step 3: Validate & compare (5 min)
print("Step 3: Validating and comparing...")
s3 = pipeline.stage_3_validate_and_compare(
    model_dir=Path('outputs/stage2_models')
)
print(f"  • Agreement with stage-1: {s3['summary']['agreement_with_stage1']:.2%}")
print(f"  • Improvement in recall: {s3['summary']['improvement_recall']:+.2%}")
print(f"  • Hard examples found: {s3['hard_examples']['false_positives_in_stage1']}\n")

# Step 4: Get recommendations
print("Step 4: Generating recommendations...")
s4 = pipeline.stage_4_generate_improvement_report(
    output_path=Path('outputs/disease_validation_report.json')
)
print(f"  • Report saved")