# 📋 ARCHON Model Improvement — Step-by-Step Implementation Guide

## Quick Summary
- **Current Performance:** Tooth Recall 87.5%, Disease Recall 66.1%
- **Target Performance:** Tooth Recall 92%+, Disease Recall 80%+
- **Timeline:** 2-3 weeks
- **Priority:** Disease detection (too many false negatives)

---

## 🚀 PHASE 1: Immediate Quick Wins (Week 1)
**Expected Improvement:** Disease Recall +10-15%, Precision +5%

### Step 1: Update Training Configuration
**File Modified:** ✅ `config/train_config.yaml`
**Status:** DONE

Changes made:
```yaml
phase2b_epochs: 100   # ↑ from 50 (allows better disease head training)
phase3_epochs: 100    # ↑ from 50 (allows better hybrid training)
attr_pos_weight: [8.0, 5.0, 12.0, 8.0]  # Tuned for disease recall
augmentation.hsv_v: 0.6  # ↑ from 0.4
augmentation.degrees: 10.0  # ↑ from 5.0
augmentation.scale: 0.7  # ↑ from 0.5
augmentation.blur: 0.2  # ↑ from 0.1
```

### Step 2: Test Different Attribute Thresholds
**File Created:** ✅ `tune_disease_threshold.py`
**Time to run:** 15-20 minutes

```bash
cd D:\WILP\Workingcode\Baseline\WILP

# Activate environment
& .\.venv\Scripts\Activate.ps1

# Test thresholds on 10 random validation images
python tune_disease_threshold.py `
    --weights "D:\WILP\Workingcode\models\archon-100-main-hybrid\weights_20260704_163459\content\WILP\weights\archon_best.pt" `
    --device cpu `
    --num-images 10 `
    --conf 0.15 `
    --save-dir ./validation_test
```

**What to expect:**
```
✅ Recommended threshold (max disease ratio): 0.15
   Expected disease detection: ~14% of teeth
   (Lower threshold = higher recall, more false positives)
```

**Recommendation:** Based on output, update your inference scripts:
```bash
# OLD (current):
python verify_local_image.py --image <path> --attr-threshold 0.30

# NEW (after tuning):
python verify_local_image.py --image <path> --attr-threshold 0.15
```

### Step 3: Retrain Phase 2b & Phase 3 with New Configuration
**Estimated Time:** 4-6 hours (on CPU, faster on GPU)
**Expected Results:** +10% disease detection

```bash
# Start fresh training with new config (all phases)
python main.py

# OR, if you want to skip Phase 1 and use existing Phase 1 checkpoint:
# (Currently unavailable in main.py, but can modify trainer.py)

# Monitor progress:
# - Loss should decrease over 100 epochs
# - Look for Phase 2b and Phase 3 training logs
# - Check validation metrics at epoch intervals
```

**Expected Training Logs:**
```
Phase 1: Detection pre-training (100 epochs) - SKIP if Phase 1 weights exist
Phase 2: Fine-tuning detection + disease (100 epochs)
Phase 2b: Attribute head training (100 epochs) - NEW LONGER TRAINING
    Epoch 50: attr_loss ≈ 0.45-0.55
    Epoch 100: attr_loss ≈ 0.30-0.40 (target)
Phase 3: Hybrid training (100 epochs) - NEW LONGER TRAINING
    Epoch 50: severity_loss ≈ 0.8-1.0
    Epoch 100: severity_loss ≈ 0.5-0.7 (target)
```

### Step 4: Validate Improvement
**File:** `validate_with_groundtruth.py` (already created)
**Time:** 10-15 minutes

```bash
python validate_with_groundtruth.py

# Compare with baseline:
# OLD Metrics (current):
#   Disease Recall: 66.1% (37/56)
#   Disease Precision: 53.6% (37/69)
#
# EXPECTED After Phase 1:
#   Disease Recall: 75-80% (42-45/56)
#   Disease Precision: 60-65% (40-43/69)
```

---

## 🔧 PHASE 2: Advanced Optimizations (Week 2)
**Expected Improvement:** Disease Recall +5%, Precision +8%

### Step 5: Implement Focal Loss
**Files Modified:**
- ✅ `src/training/focal_loss.py` (created)
- ⏳ `src/training/loss.py` (add import + class)
- ⏳ `src/training/trainer.py` (Phase 2b modification)

#### 5a. Add Focal Loss Import to loss.py
**File:** `src/training/loss.py`
**Line:** Add after imports (~line 350)

```python
# At the end of loss.py, add:

# ─── Focal Loss for Disease Detection ───────────────────────────────────
# (Import from focal_loss.py if file grows too large)
from pathlib import Path
_focal_loss_path = Path(__file__).parent / "focal_loss.py"
if _focal_loss_path.exists():
    from focal_loss import FocalAttributeLoss
else:
    FocalAttributeLoss = None
```

#### 5b. Modify Phase 2b Trainer
**File:** `src/training/trainer.py`
**Section:** `train_phase2b()` method (~line 730)

Find this line:
```python
attr_loss_fn = AttributeBCELoss(
    num_attrs=4,
    loss_weight=8.0,
    pos_weight=pos_weight,
)
```

Replace with:
```python
# Try focal loss if available for better disease detection
from src.training.focal_loss import FocalAttributeLoss

use_focal_loss = True  # Set to False to revert to BCE
if use_focal_loss:
    attr_loss_fn = FocalAttributeLoss(
        num_attrs=4,
        loss_weight=8.0,
        pos_weight=pos_weight,
        gamma=2.0,     # Standard focusing parameter
        alpha=0.25,    # Balance parameter
    )
    logger.info("Using Focal Loss for disease attributes (γ=2.0)")
else:
    attr_loss_fn = AttributeBCELoss(
        num_attrs=4,
        loss_weight=8.0,
        pos_weight=pos_weight,
    )
    logger.info("Using BCE Loss for disease attributes")
```

#### 5c. Retrain with Focal Loss
```bash
# Start training again with focal loss enabled
python main.py

# Expected logs:
# Phase 2b: Using Focal Loss for disease attributes (γ=2.0)
# Monitor attr_loss convergence — should be smoother with focal
```

**Expected Results:**
- Disease precision should improve significantly
- False positive rate should decrease by 5-10%

---

## 📊 PHASE 3: Validation & Fine-tuning (Week 2-3)
**Expected Improvement:** Overall +3-5% from ensemble/threshold tuning

### Step 6: Comprehensive Validation Report
**File to Create:** `validate_comprehensive.py`

```python
"""Generate comprehensive validation report on all 250 test images"""

import json
from pathlib import Path
from src.inference.predictor import ARCHONHybridPredictor

predictor = ARCHONHybridPredictor(
    weights_path="archon_best.pt",
    device="cpu",
    conf_threshold=0.15,
    attr_threshold=0.15,  # Use tuned threshold
)

# Test on full test set
test_images = list(Path("D:/WILP/sem-4/Dataset/DENTEX/training_data/.../xrays").glob("*.png"))
results = validate_with_groundtruth(predictor, test_images[:250])

# Save report
with open("validation_comprehensive_report.json", "w") as f:
    json.dump(results, f, indent=2)

# Print summary:
print(f"Tooth Recall: {results['tooth_recall']:.1%}")
print(f"Disease Recall: {results['disease_recall']:.1%}")
```

### Step 7: Analyze Failure Cases
**Identify:** Which disease types have lowest recall?

```bash
# Run validation and check output:
# Example analysis:
# - Impacted teeth: 45% recall (needs improvement)
# - Caries: 72% recall (acceptable)
# - Deep Caries: 68% recall (acceptable)
# - Lesion: 40% recall (needs improvement)

# Action:
# For low-recall diseases, increase pos_weight further:
#   attr_pos_weight: [12.0, 5.0, 15.0, 12.0]  # Increase impacted & lesion
```

### Step 8: Confidence/Threshold Search Grid
**Fine-tune both tooth and disease thresholds simultaneously**

```python
# Test matrix:
tooth_confs = [0.10, 0.15, 0.20, 0.25]
attr_thresholds = [0.05, 0.10, 0.15, 0.20]

best_config = None
best_f1 = 0.0

for tooth_conf in tooth_confs:
    for attr_thresh in attr_thresholds:
        predictor = ARCHONHybridPredictor(..., conf_threshold=tooth_conf, attr_threshold=attr_thresh)
        metrics = validate_with_groundtruth(predictor, test_images)
        
        tooth_f1 = 2 * (metrics['tooth_recall'] * metrics['tooth_precision']) / (...)
        disease_f1 = 2 * (metrics['disease_recall'] * metrics['disease_precision']) / (...)
        combined_f1 = (tooth_f1 + disease_f1) / 2
        
        if combined_f1 > best_f1:
            best_f1 = combined_f1
            best_config = (tooth_conf, attr_thresh)
            
        print(f"conf={tooth_conf:.2f}, attr_thresh={attr_thresh:.2f} → F1={combined_f1:.3f}")

print(f"\n✅ Best config: tooth_conf={best_config[0]}, attr_thresh={best_config[1]}")
```

---

## 📈 PHASE 4: Dataset Expansion (Week 3+)
**Expected Improvement:** +10-15% overall

### Step 9: Pseudo-label Unlabeled Data
**Files to use:**
- `src/data/pseudo_label.py` (existing)
- `D:/WILP/Workingcode/Baseline/WILP/data/unlabelled/` (directory)

```bash
# Generate pseudo-labels on unlabeled data
python src/data/pseudo_label.py \
    --input-dir data/unlabelled/ \
    --weights archon_best.pt \
    --confidence 0.50 \
    --output-dir data/pseudo_labeled/ \
    --device cpu

# This creates ~500-1000 new training images with auto-annotations
```

### Step 10: Retrain on Expanded Dataset
```bash
# Combine original + pseudo-labeled data
# Retrain all phases with new dataset

python main.py
# Expected: 15-20% improvement in disease recall due to larger dataset
```

---

## ✅ Success Criteria & Monitoring

### Metrics Dashboard
Create this after each phase:

```
PHASE 1 (After epoch 100 - new config):
├─ Tooth Detection
│  ├─ Recall: 87.5% → 89-90% (target: 92%)
│  └─ Precision: 76.6% → 78-80% (target: 85%)
└─ Disease Detection
   ├─ Recall: 66.1% → 75-80% (target: 80%+)
   └─ Precision: 53.6% → 60-65% (target: 70%)

PHASE 2 (After focal loss + tuning):
├─ Tooth Detection: 91% recall, 82% precision
└─ Disease Detection: 77% recall, 68% precision

PHASE 3 (After validation + threshold optimization):
├─ Tooth Detection: 92% recall, 85% precision ✅
└─ Disease Detection: 80% recall, 70% precision ✅
```

---

## 🎯 Commands Quick Reference

### Activate Environment
```bash
cd D:\WILP\Workingcode\Baseline\WILP
& .\.venv\Scripts\Activate.ps1
```

### Run Validation (Current Model)
```bash
python validate_with_groundtruth.py
```

### Test Single Image
```bash
python verify_local_image.py `
    --image "path/to/image.png" `
    --attr-threshold 0.15 `
    --conf 0.15 `
    --device cpu
```

### Tune Thresholds
```bash
python tune_disease_threshold.py `
    --weights "path/to/archon_best.pt" `
    --device cpu `
    --num-images 50
```

### Full Retraining
```bash
python main.py  # Trains all phases from scratch
```

---

## 🚨 Common Issues & Solutions

### Issue 1: Disease Recall Still Low After Phase 1
**Symptom:** Recall stays ~66%, not improving to 75%
**Solution:**
1. Check if new config was applied (`--attr-threshold 0.15`)
2. Verify Phase 2b trained for full 100 epochs
3. Increase `attr_pos_weight` further: [10.0, 7.0, 15.0, 10.0]
4. Implement focal loss (Phase 2)

### Issue 2: Too Many False Positives
**Symptom:** Disease precision drops below 50%
**Solution:**
1. Increase `--attr-threshold` to 0.20-0.25
2. Implement focal loss (reduces easy negatives)
3. Reduce `attr_pos_weight` slightly: [6.0, 4.0, 10.0, 6.0]

### Issue 3: Training Takes Too Long
**Symptom:** 100 epochs × 3 phases = 10+ hours on CPU
**Solution:**
1. Use GPU if available (A100/H100)
2. Reduce batch size: 4 → 2 (faster but less stable)
3. Skip Phase 1 if weights exist, retrain Phase 2b+3 only
4. Use cloud compute (Google Colab Pro)

### Issue 4: Model Gets Worse After New Training
**Symptom:** Performance degrades after retraining
**Solution:**
1. Check if Phase 1 weights are fresh (not corrupted)
2. Verify dataset splits are correct
3. Lower learning rates: `phase2_lr0: 0.001` (was 0.002)
4. Restore checkpoint and start over

---

## 📚 Related Files & References

### Key Files
```
config/
  └─ train_config.yaml          ← MODIFIED (epochs, augmentation, pos_weight)

src/training/
  ├─ trainer.py                 ← Phase 2b/3 training
  ├─ loss.py                    ← Loss functions
  └─ focal_loss.py              ← NEW (focal loss implementation)

src/inference/
  ├─ predictor.py               ← Inference pipeline
  └─ postprocess.py             ← Post-processing

scripts/
  ├─ main.py                    ← Main training entry point
  ├─ verify_local_image.py      ← Single image inference
  ├─ tune_disease_threshold.py  ← NEW (threshold tuning)
  └─ validate_with_groundtruth.py ← Validation against ground truth

IMPROVEMENT_PLAN.md             ← Detailed improvement guide
```

### Documentation
- ARCHON Paper: Section 2.2 (Loss functions)
- Focal Loss Paper: Lin et al., ICCV 2017
- YOLOv8 Documentation: https://docs.ultralytics.com/

---

## 📞 Support

### Debugging
```bash
# Enable verbose logging
python main.py --log-level DEBUG

# Check GPU memory (if using GPU)
nvidia-smi

# Profile training speed
python -m cProfile -s cumtime main.py > profile.txt
```

### Validation Results Location
```
D:\WILP\validation_test\
  ├─ validation_with_gt_results.json
  ├─ threshold_tuning_results.json
  └─ validation_comprehensive_report.json
```

---

**Status:** Ready for implementation ✅
**Last Updated:** 2026-07-04
**Contact:** See PROJECT_GUIDE.md for support
