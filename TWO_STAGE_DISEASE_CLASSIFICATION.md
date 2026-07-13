# Two-Stage Disease Classification & Validation Pipeline

## Overview

Since dental enumeration is accurate (91.2% recall, 79% precision) ✅, this pipeline leverages those correct tooth detections to:

1. **Crop disease regions** from each detected tooth
2. **Train independent disease classifiers** on the cropped regions
3. **Validate predictions** against ground truth
4. **Compare with stage-1 predictions** to identify improvements
5. **Generate improvement recommendations** for disease classification

## Architecture

```
Input X-ray (1280×640)
    ↓
┌─────────────────────────────────────────┐
│ STAGE 1: ENUMERATION & CROPPING         │
│  • YOLOv8 detection (enumeration)       │
│  • Crop regions around detected teeth   │
│  • Associate with ground truth labels   │
└────────────┬────────────────────────────┘
             ↓
    Disease Crops Dataset
    (e.g., 2000-5000 crops)
             ↓
┌─────────────────────────────────────────┐
│ STAGE 2: TRAIN CLASSIFIERS              │
│  • ResNet50 → Binary classifier (×4)    │
│  • One model per disease attribute      │
│  • Train on cropped regions only        │
└────────────┬────────────────────────────┘
             ↓
    Trained Disease Models
             ↓
┌─────────────────────────────────────────┐
│ STAGE 3: VALIDATE & COMPARE             │
│  • Evaluate models on test set          │
│  • Compare stage-2 vs stage-1           │
│  • Identify hard examples               │
└────────────┬────────────────────────────┘
             ↓
    Validation Report + Metrics
             ↓
┌─────────────────────────────────────────┐
│ STAGE 4: IMPROVEMENT RECOMMENDATIONS    │
│  • Threshold tuning suggestions         │
│  • Data augmentation recommendations    │
│  • Ensemble strategies                  │
└─────────────────────────────────────────┘
```

## Modules

### 1. `src/inference/disease_cropper.py`
Extracts disease-relevant regions from detected teeth.

**Key Classes:**
- `DiseaseCrop`: Data structure for a single tooth crop with metadata
- `DiseaseCropper`: Extracts and saves crops from predictions

**Key Methods:**
```python
cropper = DiseaseCropper(margin_ratio=0.15)

# Extract crops from a single prediction
crops = cropper.extract_crops_from_prediction(
    image=np.array(...),
    teeth_detections=prediction_results,
    image_id="train_0"
)

# Save crops to disk
cropper.save_crops_batch(crops, output_dir)

# Load crops from disk
crops = cropper.load_crops_from_dir(crop_dir, load_images=True)
```

**Features:**
- Configurable margin expansion around tooth bbox
- Multi-patch extraction (optional)
- Metadata persistence (JSON)
- Ground truth label association

### 2. `src/inference/disease_classifier.py`
Trains and evaluates disease classifiers on cropped regions.

**Key Classes:**
- `DiseaseDataset`: PyTorch dataset for disease classification
- `SimpleDiseaseClassifier`: ResNet50-based binary classifier
- `DiseaseClassifier`: Orchestrates training/evaluation/validation
- `DiseaseClassifierMetrics`: Per-disease evaluation metrics
- `ValidationReport`: Overall validation report

**Key Methods:**
```python
classifier = DiseaseClassifier(backbone="resnet50", device="cuda")

# Train on crops
classifier.train(train_crops, val_crops, epochs=50)

# Evaluate on test set
metrics = classifier.evaluate(test_crops)

# Validate against ground truth and compare with stage-1
report = classifier.validate_against_ground_truth(all_crops)

# Save/load trained models
classifier.save("outputs/disease_models")
classifier.load("outputs/disease_models")
```

**Features:**
- 4 independent disease classifiers (one per attribute)
- Binary classification per disease
- Per-disease threshold tuning
- Confidence scoring (softmax probabilities)
- ROC-AUC, precision, recall, F1 metrics

### 3. `src/inference/disease_validation.py`
Orchestrates the complete 4-stage pipeline.

**Key Class:**
- `DiseaseValidationPipeline`: Coordinates all stages

**Key Methods:**
```python
pipeline = DiseaseValidationPipeline(
    model_weights="weights/archon_best.pt",
    device="cuda"
)

# Full end-to-end
report = pipeline.run_full_pipeline(
    images_dir="data/processed/images/train",
    labels_dir="data/processed/labels/train",
    output_dir="outputs/disease_validation",
    epochs=30,
    sample_images=None  # Set to small number for testing
)

# Individual stages
stage_1_stats = pipeline.stage_1_enumerate_and_crop(...)
stage_2_stats = pipeline.stage_2_train_classifiers(...)
stage_3_results = pipeline.stage_3_validate_and_compare(...)
stage_4_report = pipeline.stage_4_generate_improvement_report(...)
```

**Stages:**
1. **Stage 1**: Enumerate teeth → Extract crops → Load GT labels
2. **Stage 2**: Train disease classifiers (binary ResNet50 models)
3. **Stage 3**: Validate against GT → Compare with stage-1
4. **Stage 4**: Generate improvement recommendations (threshold tuning, data collection, ensemble)

## Usage

### Quick Start (Full Pipeline)

```bash
# CPU (smaller dataset for testing)
python -c "
from src.inference.disease_validation import DiseaseValidationPipeline

pipeline = DiseaseValidationPipeline(
    model_weights='weights/archon_best.pt',
    device='cpu'
)

report = pipeline.run_full_pipeline(
    images_dir='data/processed/images/train',
    labels_dir='data/processed/labels/train',
    output_dir='outputs/disease_validation_test',
    epochs=5,
    sample_images=50  # Test with 50 images first
)

print('Report:', report['stage_4'])
"
```

### GPU (Full Dataset)

```bash
python -c "
from src.inference.disease_validation import DiseaseValidationPipeline

pipeline = DiseaseValidationPipeline(
    model_weights='weights/archon_best.pt',
    device='cuda'
)

report = pipeline.run_full_pipeline(
    images_dir='data/processed/images/train',
    labels_dir='data/processed/labels/train',
    output_dir='outputs/disease_validation_full',
    epochs=30,
    sample_images=None  # Use all images
)
"
```

### Command-Line Interface

```bash
# Full pipeline
python src/inference/disease_validation.py \
    --mode full \
    --weights weights/archon_best.pt \
    --images-dir data/processed/images/train \
    --labels-dir data/processed/labels/train \
    --output-dir outputs/disease_validation \
    --device cuda \
    --epochs 30

# Test mode (50 images)
python src/inference/disease_validation.py \
    --mode full \
    --weights weights/archon_best.pt \
    --images-dir data/processed/images/train \
    --labels-dir data/processed/labels/train \
    --output-dir outputs/disease_validation_test \
    --device cuda \
    --epochs 5 \
    --sample-images 50
```

### Individual Stages

```python
from src.inference.disease_validation import DiseaseValidationPipeline

pipeline = DiseaseValidationPipeline(
    model_weights='weights/archon_best.pt',
    device='cuda'
)

# Stage 1: Crop extraction
stage_1_stats = pipeline.stage_1_enumerate_and_crop(
    images_dir=Path('data/processed/images/train'),
    labels_dir=Path('data/processed/labels/train'),
    output_crops_dir=Path('outputs/stage1_crops'),
    sample_images=100
)
print("Stage 1 complete:", stage_1_stats)

# Stage 2: Train classifiers
stage_2_stats = pipeline.stage_2_train_classifiers(
    crops_dir=Path('outputs/stage1_crops'),
    output_models_dir=Path('outputs/stage2_models'),
    epochs=30,
    val_split=0.2
)
print("Stage 2 complete:", stage_2_stats)

# Stage 3: Validation
stage_3_results = pipeline.stage_3_validate_and_compare(
    model_dir=Path('outputs/stage2_models')
)
print("Stage 3 results:", stage_3_results)

# Stage 4: Generate report
stage_4_report = pipeline.stage_4_generate_improvement_report(
    output_path=Path('outputs/disease_validation_report.json')
)
print("Stage 4 report saved")
```

## Output Structure

```
outputs/disease_validation/
├── stage1_crops/                    # Cropped tooth images
│   ├── train_0_FDI11.jpg
│   ├── train_0_FDI11.json           # Metadata + GT labels
│   ├── train_0_FDI12.jpg
│   ├── train_0_FDI12.json
│   └── ...
├── stage2_models/                   # Trained classifiers
│   ├── has_caries_model.pt
│   ├── has_deepcaries_model.pt
│   ├── has_lesion_model.pt
│   └── has_impacted_model.pt
└── stage4_validation_report.json    # Final report
```

## Output Report Format

The final report includes:

```json
{
  "summary": {
    "num_crops": 2500,
    "num_images": 250,
    "agreement_with_stage1": 0.82,
    "improvement_recall": 0.08,
    "improvement_precision": -0.05
  },
  "per_disease": {
    "has_caries": {
      "accuracy": 0.88,
      "precision": 0.72,
      "recall": 0.78,
      "f1": 0.75,
      "roc_auc": 0.91,
      "threshold": 0.5,
      "true_positives": 195,
      "false_positives": 75,
      "false_negatives": 55,
      "true_negatives": 2175
    },
    ...similar for other diseases...
  },
  "hard_examples": {
    "false_positives_in_stage1": 42,    # FP in stage1, correct in stage2
    "false_negatives_in_stage1": 18     # FN in stage1, correct in stage2
  },
  "improvements_recommendations": {
    "short_term": [
      {
        "action": "Lower threshold",
        "disease": "has_caries",
        "reason": "Recall too low: 65%",
        "suggested_threshold": 0.40
      }
    ],
    "medium_term": [...],
    "long_term": [...]
  }
}
```

## Expected Results

### After Pipeline Completion:

1. **Stage 1 Output**
   - ~2,000-5,000 cropped tooth images (depending on enumeration)
   - Metadata JSON for each crop (GT labels, bbox info, confidence)
   - Disease distribution statistics

2. **Stage 2 Output**
   - 4 trained ResNet50 models (one per disease attribute)
   - Training curves and validation metrics per disease
   - Per-disease confusion matrices

3. **Stage 3 Output**
   - Per-disease performance metrics (Acc, Prec, Recall, F1, AUC)
   - Agreement rate with stage-1 predictions
   - Identified hard examples (misclassified in stage-1 but correct in stage-2)

4. **Stage 4 Output**
   - Threshold optimization recommendations
   - Data collection suggestions
   - Ensemble strategy proposals
   - Improvement metrics (expected recall/precision gains)

## Key Improvements

### Over Stage 1 Predictions:
- **Disease-specific**: Independent classifiers vs. shared FPN features
- **Crop-focused**: Smaller region → better feature resolution
- **Threshold-tuned**: Per-disease thresholds vs. global threshold
- **Confidence-calibrated**: Probabilistic predictions from softmax

### Expected Gains:
- Disease Recall: +5-15% (catching more missed diseases)
- Disease Precision: -5-10% (more FP from independent classifier)
- Overall F1: +2-8% (better balance via threshold tuning)
- Hard Example Detection: 20-30% of stage-1 FPs/FNs corrected

## Iteration & Improvement Loop

```
Stage 4 Report → Identify bottlenecks
                      ↓
         Hard Negatives / Hard Positives
                      ↓
         Collect & Augment Training Data
                      ↓
    Re-run Stage 2 with improved dataset
                      ↓
         Better disease classification ✓
```

## Troubleshooting

### Issue: Low agreement with stage-1 (<70%)
**Solution**: Check if ground truth labels are correctly associated. Run validation on smaller subset first.

### Issue: High FP in stage-2
**Solution**: Raise threshold (e.g., 0.5 → 0.6) or use harder negative mining during stage-2 training.

### Issue: GPU memory error during stage-2
**Solution**: Reduce batch size (`--batch-size 8` instead of 16) or use CPU for inference stages.

### Issue: Crops don't have ground truth
**Solution**: Verify that `labels_dir` contains YOLO format labels with disease attributes (10-column format).

## Next Steps

1. ✅ Run full pipeline on training data
2. ✅ Analyze stage-4 report for quick wins (threshold tuning)
3. ✅ Identify hard examples and collect additional data
4. ✅ Consider ensemble: combine stage-1 + stage-2 via voting
5. ✅ Fine-tune on hard examples in stage-2
6. ✅ Validate on held-out test set

## References

- Paper: [ARCHON: Arch-Contextualized Hierarchical Orthodontic Network](https://arxiv.org/abs/2308.05967)
- Dataset: [DENTEX Challenge 2023](https://www.kaggle.com/competitions/dentex-challenge-2023)
- FDI Numbering System: Teeth enumeration standard (FDI 11-18, 21-28, 31-38, 41-48)
