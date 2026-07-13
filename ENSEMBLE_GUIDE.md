"""
ENSEMBLE MODEL GUIDE: Two-Stage Disease Classification

Overview:
  The ensemble combines ARCHON enumeration (Stage 1) with independent disease
  classifiers (Stage 2) to produce robust disease predictions.

Files Created:
  1. src/inference/disease_ensemble.py       - Core ensemble class
  2. run_ensemble_predictions.py             - Batch prediction script
  3. run_ensemble_threshold_tuner.py         - Threshold optimization
  4. ENSEMBLE_GUIDE.md                       - This guide
"""

# ============================================================================
# QUICK START
# ============================================================================

## Step 1: Run Full Dataset Training (if not done)

```bash
# Update run_option2_quick_finetune.py with:
#   epochs: 50
#   sample_images: None

python run_option2_quick_finetune.py
# Expected time: 2-3 hours on GPU
```

This creates:
- 3,000-5,000 tooth crops (Stage 1)
- 4 trained disease classifiers (Stage 2)
- Validation report with metrics


## Step 2: Run Ensemble Predictions

```bash
python run_ensemble_predictions.py
```

**Configuration** (in script):
```python
CONFIG = {
    "archon_weights": Path("weights/archon_best.pt"),
    "stage2_models_dir": Path("outputs/disease_validation_option2/stage2_models"),
    "test_images_dir": Path("data/processed/images/test"),
    "labels_dir": Path("data/processed/labels_ext/test"),
    
    "stage1_weight": 0.3,    # ARCHON importance
    "stage2_weight": 0.7,    # Disease classifiers importance
    
    "thresholds": {          # Prediction thresholds
        "has_caries": 0.45,
        "has_deepcaries": 0.45,
        "has_lesion": 0.40,
        "has_impacted": 0.50
    },
    
    "output_dir": Path("outputs/ensemble_predictions"),
    "device": "cuda",
}
```

**Output:**
- `test_predictions.json`       - Raw predictions on test set
- `ensemble_evaluation_report.json` - Performance metrics
- `ensemble_config.json`        - Configuration snapshot

**Performance metrics generated:**
```
Per-Disease:
  • Accuracy, Precision, Recall, F1 Score
  • Confusion matrix (TP, FP, FN, TN)

Overall:
  • Average F1 score
  • Stage-1 agreement rate
  • Timestamp and configuration
```


## Step 3: Optimize Thresholds (Optional but Recommended)

```bash
python run_ensemble_threshold_tuner.py
```

**Configuration** (in script):
```python
CONFIG = {
    "threshold_range": np.arange(0.3, 0.8, 0.05),  # 0.3 to 0.75 in 0.05 steps
    "metric_to_optimize": "f1",  # Options: 'f1', 'accuracy', 'precision', 'recall'
    "sample_images": None,       # Use all images (or limit to N)
}
```

**Output:**
```
threshold_tuning_report_YYYYMMDD_HHMMSS.json

Contains:
  • Optimal threshold for each disease
  • Performance at optimal threshold
  • All tested thresholds and their metrics
```

**Example output:**
```
has_caries: Optimal threshold = 0.40 (F1 = 0.85)
has_deepcaries: Optimal threshold = 0.45 (F1 = 0.92)
has_lesion: Optimal threshold = 0.35 (F1 = 0.78)
has_impacted: Optimal threshold = 0.50 (F1 = 0.88)
```

Copy the optimal thresholds back into `disease_ensemble.py`:
```python
self.thresholds = {
    'has_caries': 0.40,
    'has_deepcaries': 0.45,
    'has_lesion': 0.35,
    'has_impacted': 0.50
}
```


# ============================================================================
# ADVANCED USAGE
# ============================================================================

## Using Ensemble Programmatically

```python
from pathlib import Path
from src.inference.disease_ensemble import create_ensemble

# Create ensemble
ensemble = create_ensemble(
    archon_weights="weights/archon_best.pt",
    stage2_models_dir="outputs/disease_validation_option2/stage2_models",
    device="cuda"
)

# Set thresholds
ensemble.set_thresholds({
    'has_caries': 0.40,
    'has_deepcaries': 0.45,
    'has_lesion': 0.35,
    'has_impacted': 0.50
})

# Predict on single image
image_path = Path("data/processed/images/test/test_0.png")
predictions = ensemble.predict_single(image_path)

# predictions is a dict: FDI number → EnsemblePrediction
for fdi, pred in predictions.items():
    print(f"Tooth FDI {fdi}:")
    print(f"  Confidence: {pred.confidence:.2%}")
    print(f"  Diseases: {pred.final_predictions}")
    print(f"  Scores: {pred.fused_predictions}")

# Predict on batch
image_paths = list(Path("data/processed/images/test").glob("*.png"))
all_predictions = ensemble.predict_batch(image_paths)

# Save configuration
ensemble.save_config(Path("my_ensemble_config.json"))

# Load configuration
ensemble.load_config(Path("my_ensemble_config.json"))
```

**EnsemblePrediction structure:**
```python
@dataclass
class EnsemblePrediction:
    image_path: Path                       # Source image
    fdi_number: int                        # FDI tooth number
    stage1_predictions: Dict[str, float]   # Raw ARCHON scores (0-1)
    stage2_predictions: Dict[str, float]   # Raw ResNet50 scores (0-1)
    fused_predictions: Dict[str, float]    # Weighted average (0-1)
    fused_probabilities: Dict[str, float]  # After normalization (0-1)
    final_predictions: Dict[str, bool]     # After thresholding
    confidence: float                      # Minimum disease confidence
```


## Understanding the Fusion Process

```
Stage 1 Score (ARCHON):     [0.1, 0.9, 0.3, 0.2]
Stage 2 Score (ResNet50):   [0.05, 0.85, 0.1, 0.15]

Weight Stage 1:             0.3
Weight Stage 2:             0.7

Fused = 0.3 * S1 + 0.7 * S2
      = 0.3 * [0.1, 0.9, 0.3, 0.2] + 0.7 * [0.05, 0.85, 0.1, 0.15]
      = [0.03, 0.27, 0.09, 0.06] + [0.035, 0.595, 0.07, 0.105]
      = [0.065, 0.865, 0.16, 0.165]

Apply Thresholds:
  has_caries (0.40):       0.065 < 0.40 → False
  has_deepcaries (0.45):   0.865 > 0.45 → True
  has_lesion (0.35):       0.16 < 0.35 → False
  has_impacted (0.50):     0.165 < 0.50 → False

Final Prediction: {
    'has_caries': False,
    'has_deepcaries': True,
    'has_lesion': False,
    'has_impacted': False
}
```


## Tuning Weights (Advanced)

To change the relative importance of Stage 1 vs Stage 2:

```python
# More weight on ARCHON enumeration
ensemble = DiseaseEnsemble(
    archon_weights="weights/archon_best.pt",
    stage2_models_dir="outputs/disease_validation_option2/stage2_models",
    stage1_weight=0.5,   # Increased from 0.3
    stage2_weight=0.5    # Decreased from 0.7
)

# Opposite: More weight on disease classifiers
ensemble = DiseaseEnsemble(
    ...,
    stage1_weight=0.2,
    stage2_weight=0.8
)
```

**When to adjust:**
- Stage 1 weight ↑: If ARCHON has high recall, trust its disease detection
- Stage 2 weight ↑: If classifiers are highly accurate, rely more on them


# ============================================================================
# PERFORMANCE INTERPRETATION
# ============================================================================

## Key Metrics

**Accuracy**: (TP + TN) / Total
  - Overall correctness
  - Good for balanced datasets

**Precision**: TP / (TP + FP)
  - Of predicted positives, how many are correct?
  - Important when false positives are costly

**Recall**: TP / (TP + FN)
  - Of actual positives, how many are detected?
  - Important when false negatives are costly

**F1 Score**: 2 * (Precision * Recall) / (Precision + Recall)
  - Harmonic mean of precision and recall
  - Good when both matter equally


## Example Report

```json
{
  "has_deepcaries": {
    "accuracy": 0.989,
    "precision": 1.0,
    "recall": 0.7,
    "f1": 0.823,
    "tp": 7,
    "fp": 0,
    "fn": 3,
    "tn": 277
  }
}
```

**Interpretation:**
- ✅ No false positives (precision = 1.0) - very clean detections
- ⚠️ Missing 30% of cases (recall = 0.7) - some cases go undetected
- 💡 Could lower threshold to increase recall at cost of precision


## Common Scenarios

**Scenario 1: Low Recall, High Precision**
- Missing positive cases
- Solution: Lower threshold (be more aggressive)

**Scenario 2: High Recall, Low Precision**
- Too many false alarms
- Solution: Raise threshold (be more conservative)

**Scenario 3: High Recall, High Precision**
- Excellent performance - model is well-calibrated
- May need to validate on external data

**Scenario 4: Both Low**
- Model struggling - may need more training data or different architecture


# ============================================================================
# TROUBLESHOOTING
# ============================================================================

## Issue: "Models not found" Error

**Cause**: Stage-2 models not trained yet

**Solution**:
```bash
python run_option2_quick_finetune.py  # Run full pipeline first
```

## Issue: Low Recall on All Diseases

**Cause**: Thresholds set too high

**Solution**:
```bash
python run_ensemble_threshold_tuner.py  # Find optimal thresholds
```

## Issue: Too Many False Positives

**Cause**: Thresholds set too low

**Solution**:
1. Increase thresholds manually:
   ```python
   ensemble.set_thresholds({
       'has_caries': 0.55,      # Was 0.45
       'has_deepcaries': 0.55,  # Was 0.45
       ...
   })
   ```

2. Or run tuner with recall emphasis:
   - In tuner config, change weights differently
   - Or use higher threshold_range

## Issue: Stage-1 vs Stage-2 Disagreement

**Cause**: Models have different biases

**Options**:
1. Adjust stage weights:
   ```python
   DiseaseEnsemble(..., stage1_weight=0.5, stage2_weight=0.5)
   ```

2. Investigate hard examples:
   ```python
   # Find predictions where stage1 ≠ stage2
   for pred in predictions:
       if pred.stage1_predictions != pred.stage2_predictions:
           print(f"Disagreement: {pred}")
   ```


# ============================================================================
# WORKFLOW CHECKLIST
# ============================================================================

After each major step, verify:

- [ ] Full dataset training (50 epochs, all images)
  - Check: outputs/disease_validation_option2/stage2_models/ has 4 .pt files
  - Check: stage4_validation_report.json exists
  
- [ ] Ensemble predictions
  - Check: outputs/ensemble_predictions/test_predictions.json exists
  - Check: ensemble_evaluation_report.json has per-disease metrics
  - Check: Average F1 > previous run
  
- [ ] Threshold tuning
  - Check: threshold_tuning_report_*.json exists
  - Check: Optimal thresholds make sense (typically 0.3-0.6)
  - Check: Updated disease_ensemble.py with new thresholds
  
- [ ] Final validation
  - Check: Rerun ensemble_predictions.py with new thresholds
  - Check: Metrics improved (especially recall)


# ============================================================================
# OUTPUT DIRECTORY STRUCTURE
# ============================================================================

```
outputs/disease_validation_option2/
├── stage1_crops/              # 3,000-5,000 tooth images
├── stage2_models/             # 4 trained ResNet50 models
│   ├── has_caries_model.pt
│   ├── has_deepcaries_model.pt
│   ├── has_lesion_model.pt
│   └── has_impacted_model.pt
└── stage4_validation_report.json

outputs/ensemble_predictions/
├── test_predictions.json           # Raw predictions
├── ensemble_evaluation_report.json  # Performance metrics
├── ensemble_config.json            # Configuration snapshot
└── logs/                           # Execution logs
    └── ensemble_*.log

outputs/ensemble_threshold_tuning/
└── threshold_tuning_report_*.json  # Tuning results
```


# ============================================================================
# NEXT STEPS
# ============================================================================

**Short-term (This week):**
1. ✅ Create ensemble scripts (DONE)
2. ⬜ Run full dataset training (2-3 hours)
3. ⬜ Run ensemble predictions
4. ⬜ Review evaluation report
5. ⬜ Run threshold tuning
6. ⬜ Apply optimal thresholds
7. ⬜ Revalidate with new thresholds

**Medium-term (This month):**
8. Collect more ground truth labels (hard examples)
9. Retrain Stage-2 models with expanded labels
10. Compare ensemble vs. Stage-1 only performance
11. Test on external validation set
12. Document final performance metrics

**Long-term:**
13. Deploy ensemble to production
14. Monitor performance on new images
15. Periodic retraining with new data


# ============================================================================
# SUPPORT
# ============================================================================

For questions about:
- Ensemble architecture: See disease_ensemble.py comments
- Threshold tuning: See run_ensemble_threshold_tuner.py
- Integration: See run_ensemble_predictions.py
- Theory: See TWO_STAGE_DISEASE_CLASSIFICATION.md
"""

# Quick reference: Running everything

print("""
================================================================================
                       ENSEMBLE WORKFLOW SUMMARY
================================================================================

1. TRAIN (if not done):
   python run_option2_quick_finetune.py
   
2. PREDICT:
   python run_ensemble_predictions.py
   
3. TUNE THRESHOLDS:
   python run_ensemble_threshold_tuner.py
   
4. CHECK RESULTS:
   - outputs/ensemble_predictions/ensemble_evaluation_report.json
   - outputs/ensemble_threshold_tuning/threshold_tuning_report_*.json
   
5. UPDATE THRESHOLDS IN disease_ensemble.py
   
6. REVALIDATE:
   python run_ensemble_predictions.py (with updated thresholds)

Expected timeline: 2-3 hours (training) + 30 min (tuning) + 15 min (validation)
Expected improvement: +5-15% F1 score vs. single-stage

================================================================================
""")
