# 🚀 ARCHON Model Performance Improvement Plan

## Current Performance (10 test images)

### Tooth Detection
- **Recall: 87.5%** (49/56 teeth detected) ✅ Good
- **Precision: 76.6%** (49/64 predictions correct) ⚠️ 15 false positives

### Disease Detection
- **Recall: 66.1%** (37/56 diseases detected) ❌ Critical gap
- **Precision: 53.6%** (37/69 predictions correct) ❌ Too many false positives

---

## 🎯 Improvement Roadmap (Prioritized)

### Priority 1: DISEASE DETECTION RECALL (66.1% → 85%+)
**Problem:** Missing 19/56 diseases (mainly Impacted, Deep Caries, Lesion)

#### 1.1 Increase Phase 2b Training Epochs (QUICK WIN)
**File:** `config/train_config.yaml`
```yaml
# Current: 50 epochs
phase2b_epochs: 100  # Increase 2x — allows attribute heads more iterations to learn

# Rationale:
# - Phase 2b trains disease heads on frozen backbone
# - 50 epochs may be insufficient for fine-grained disease discrimination
# - More epochs → deeper feature representations for disease features
# - No risk: backbone frozen, so detection won't degrade
```

**Impact:** +5-10% recall improvement expected

---

#### 1.2 Increase Phase 3 (Hybrid) Training Epochs
**File:** `config/train_config.yaml`
```yaml
# Current: 50 epochs
phase3_epochs: 100  # Increase 2x — more training for severity head

# Rationale:
# - Phase 3 trains severity head + cross-attention + global encoder
# - 50 epochs may not be enough for complex feature fusion
# - Severity → binary conversion depends on confidence calibration
# - More epochs with CosineAnnealingLR helps smooth convergence
```

**Impact:** +5-10% recall improvement expected

---

#### 1.3 Lower Disease Detection Threshold (IMMEDIATE)
**File:** `verify_local_image.py`
```python
# Current: --attr-threshold 0.3
python verify_local_image.py \
  --image <path> \
  --attr-threshold 0.15 \  # Reduce from 0.3
  --weights <path> \
  --device cpu
```

**Rationale:**
- Threshold 0.3 is too high for imbalanced disease classes
- Severity head output → binary conversion may produce ~0.1-0.25 range for true diseases
- Lowering threshold increases recall at cost of precision
- Test on validation images to find optimal threshold

**Action:**
1. Test with thresholds: 0.05, 0.10, 0.15, 0.20
2. Plot recall vs precision curve
3. Choose threshold that maximizes F1 score (harmonic mean of recall/precision)

**Expected outcome:** +10-20% recall improvement

---

#### 1.4 Increase attr_pos_weight for Disease Classes (MEDIUM EFFORT)
**File:** `config/train_config.yaml`
```yaml
# Current: null (auto-computed)
# Auto values: [5.0, 3.0, 8.0, 5.0] for [impacted, caries, deepcaries, lesion]

# Proposed: Increase weights to penalize missing diseases more
attr_pos_weight: [8.0, 5.0, 12.0, 8.0]  # Increase by 1.5-1.7x

# Rationale:
# - pos_weight controls false negative penalty vs false positive penalty
# - Higher weight = penalize missing diseases MORE
# - Current: balanced for both; new: optimize for RECALL (find diseases)
# - Trade-off: may increase false positives slightly (acceptable from clinical view)

# Recommended tuning:
# Start with [7.0, 5.0, 10.0, 7.0] (mid-conservative)
# If still low recall, increase to [10.0, 7.0, 15.0, 10.0] (aggressive)
```

**Impact:** +3-8% recall improvement

---

### Priority 2: DISEASE DETECTION PRECISION (53.6% → 70%+)
**Problem:** Too many false positives (32/69 predictions wrong)

#### 2.1 Implement Focal Loss for Disease Attributes (ADVANCED)
**File:** `src/training/loss.py`
```python
# Add after AttributeBCELoss class:

class FocalAttributeLoss(nn.Module):
    """Focal loss for disease attributes — reduces impact of easy negatives.
    
    From paper: "Focal Loss for Dense Object Detection"
    Loss = -α * (1 - p_t)^γ * log(p_t)
    
    γ > 0: down-weights easy examples
    α: balance between classes
    
    Clinical motivation: Most teeth are healthy (easy negatives).
    Focal loss focuses training on hard positives (diseased teeth).
    """
    def __init__(self, num_attrs=4, loss_weight=8.0, pos_weight=None, 
                 gamma=2.0, alpha=0.25):
        super().__init__()
        self.num_attrs = num_attrs
        self.loss_weight = loss_weight
        self.gamma = gamma  # Focus parameter (2.0 recommended)
        self.alpha = alpha  # Balance parameter (0.25 recommended)
        _pw = pos_weight if pos_weight is not None else [5.0, 3.0, 8.0, 5.0]
        self.register_buffer("_pos_weight", torch.tensor(_pw, dtype=torch.float32))
    
    def forward(self, pred_attrs, target_attrs, data_types):
        mask = (data_types == 2).float()
        if mask.sum() == 0:
            return pred_attrs.sum() * 0.0
        
        target_f = target_attrs.float()
        pw = self._pos_weight.to(pred_attrs.device)
        
        # Sigmoid probability
        p = torch.sigmoid(pred_attrs)
        p_t = torch.where(target_f == 1, p, 1 - p)
        
        # Focal weight: (1 - p_t)^γ
        focal_weight = (1 - p_t) ** self.gamma
        
        # BCE loss
        bce_loss = (
            (1 - target_f) * pred_attrs
            - (1 + (pw - 1) * target_f) * F.logsigmoid(pred_attrs)
        )
        
        # Focal loss = focal_weight * BCE
        focal_loss = focal_weight * bce_loss * mask.unsqueeze(1)
        
        n_valid = mask.sum() * self.num_attrs
        loss = focal_loss.sum() / (n_valid + 1e-6)
        
        return self.loss_weight * loss
```

**Implementation:**
1. Create above class in `src/training/loss.py`
2. Update Phase 2b trainer to use FocalAttributeLoss instead of AttributeBCELoss
3. Train Phase 2b for 100 epochs with focal loss
4. Evaluate on validation set

**Impact:** -5-10% false positives (higher precision), minimal recall impact

---

#### 2.2 Add Hard Negative Mining (ADVANCED)
**File:** `src/training/trainer.py` (Phase 2b section)
```python
# Pseudocode for Phase 2b loss computation:

# After getting attr predictions:
if use_hard_negative_mining:
    # Identify predictions that are:
    # - Confident but wrong (pred > 0.5 but label = 0)
    false_positives = (pred_attrs > 0.5) & (target_attrs == 0)
    
    # Increase loss weight for these hard negatives
    hard_weight = torch.ones_like(pred_attrs)
    hard_weight[false_positives] = 2.0  # 2x weight on hard FPs
    
    # Apply to BCE loss
    loss = loss * hard_weight
```

**Benefit:** Reduces false disease predictions by explicitly penalizing confident misclassifications

---

### Priority 3: TOOTH DETECTION PRECISION (76.6% → 85%+)
**Problem:** 15 false positive detections

#### 3.1 Increase Confidence Threshold
**File:** `verify_local_image.py`
```python
# Current: --conf 0.15
python verify_local_image.py \
  --image <path> \
  --conf 0.20 \  # Increase from 0.15
  --weights <path>
```

**Trade-off:** Recall will drop slightly (miss ~1-2 teeth) but precision improves significantly

**Optimal search:**
- Test: 0.10, 0.15, 0.20, 0.25, 0.30
- Pick threshold where F1 score maximizes

---

#### 3.2 Increase NMS IoU Threshold
**File:** `src/inference/predictor.py` (line ~270)
```python
# Current: iou_threshold=0.45
results = model(
    img_rgb,
    conf=self.conf_threshold,
    iou=0.50,  # Increase from 0.45
    device=self.device,
    verbose=False,
)
```

**Rationale:** Higher IoU means stricter overlap requirements for NMS merging
- Reduces duplicate/overlapping tooth detections
- Keeps only the highest-confidence bounding boxes
- Clinical benefit: Cleaner output visualization

---

### Priority 4: DATASET & TRAINING INFRASTRUCTURE

#### 4.1 Expand Training Dataset (HIGH IMPACT)
**Current:** 705 training images with 56 disease annotations (10% disease prevalence)
**Target:** 1500+ training images with 150+ disease annotations

**Action items:**
```bash
# Check current DENTEX split
ls -la D:\WILP\sem-4\Dataset\DENTEX\DENTEX\training_data\

# Strategy:
# 1. Use unlabelled data with pseudo-labels from Phase 1 detector
# 2. Manually annotate 100-200 additional images for disease
# 3. Use data augmentation more aggressively (see below)
```

**Expected impact:** +5-15% overall performance

---

#### 4.2 Improve Data Augmentation for Disease Features
**File:** `config/train_config.yaml`
```yaml
augmentation:
  hsv_v: 0.6        # Increase from 0.4 — simulate different X-ray exposures
  degrees: 10.0     # Increase from 5.0 — slight rotation variation
  scale: 0.7        # Increase from 0.5 — more zoom variation
  blur: 0.2         # Increase from 0.1 — simulate motion blur
  noise: 0.05       # ADD NEW — Gaussian noise for X-ray artifacts
```

**Rationale:** Disease features (cavities, lesions) appear under various X-ray conditions; more augmentation helps generalize

---

#### 4.3 Implement Mixed Batch Sampling
**File:** `src/data/dataset.py`
```python
# During training, ensure balanced batches:
# - Batch of 4 should have:
#   - 2 images with diseases
#   - 2 images without (or with fewer) diseases
# This prevents the model from overfitting to healthy-only images

# Implementation: WeightedRandomSampler
from torch.utils.data import WeightedRandomSampler

# Compute weights: samples with diseases get higher weight
disease_counts = [count_diseases_in_image(img) for img in dataset]
weights = [1 + (count * 2) for count in disease_counts]  # 3x weight for disease images
sampler = WeightedRandomSampler(weights, len(dataset), replacement=True)
```

---

### Priority 5: INFERENCE-TIME OPTIMIZATIONS

#### 5.1 Implement Post-Processing Smoothing
**File:** `src/inference/postprocess.py`
```python
def smooth_disease_predictions(teeth_detections, smoothing_window=3):
    """Smooth disease predictions based on neighboring teeth.
    
    Rationale: If surrounding teeth have same disease, local confidence increases.
    Example: If FDI 15, 16, 17 all have caries, FDI16's caries prob increases.
    """
    # Group teeth by quadrant
    for quad_id in range(4):
        quad_teeth = [t for t in teeth_detections if t.fdi // 10 == quad_id + 1]
        quad_teeth.sort(key=lambda t: t.fdi % 10)  # Sort by tooth number
        
        # Smooth each disease across teeth
        for disease_idx in range(4):  # 4 disease types
            probs = [t.disease_probs[disease_idx] for t in quad_teeth]
            smoothed = moving_average(probs, window=smoothing_window)
            for tooth, prob in zip(quad_teeth, smoothed):
                tooth.disease_probs[disease_idx] = prob
```

---

## 📋 Recommended Implementation Order

### **Week 1: Quick Wins (Expected +15% recall, +5% precision)**
1. ✅ Lower `--attr-threshold` from 0.3 to 0.15 (test on val set)
2. ✅ Increase Phase 2b epochs: 50 → 100
3. ✅ Increase Phase 3 epochs: 50 → 100
4. ✅ Increase `attr_pos_weight`: [5, 3, 8, 5] → [8, 5, 12, 8]
5. ✅ Increase confidence threshold 0.15 → 0.20 if precision low

### **Week 2: Moderate Improvements (Expected +10% recall, +8% precision)**
6. ✅ Implement focal loss for disease attributes
7. ✅ Increase data augmentation params (blur, noise, scale)
8. ✅ Test multiple confidence thresholds (0.05-0.30)
9. ✅ Rerun validation on full test set (250 images)

### **Week 3: Infrastructure (Expected +10-15% overall)**
10. ✅ Expand dataset with pseudo-labeled images
11. ✅ Implement weighted batch sampling
12. ✅ Add hard negative mining in loss computation

---

## 🧪 Validation & Monitoring

### Create a Validation Dashboard
**File:** `validate_improved_model.py`
```python
"""Compare old vs new model metrics"""

results = {
    'baseline': {
        'tooth_recall': 0.875,
        'tooth_precision': 0.766,
        'disease_recall': 0.661,
        'disease_precision': 0.536,
    },
    'after_threshold_tune': None,
    'after_phase_epochs_increase': None,
    'after_focal_loss': None,
    'after_dataset_expansion': None,
}

# Save after each experiment
```

### Test Scenarios
```python
# Test on different image types:
test_sets = {
    'small_lesions': 'images with small cavities',
    'deep_caries': 'images with deep decay',
    'impacted_teeth': 'images with impacted teeth',
    'healthy_teeth': 'images with no disease',
}

# Per-test-set metrics → identify which diseases model struggles with
```

---

## 💡 Advanced Techniques (If Time Permits)

### Ensemble Methods
- Train 3 models with different random seeds
- Average predictions at inference time
- Typical improvement: +3-5% recall, +5% precision

### Confidence Calibration
- After training, use validation set to calibrate probability outputs
- Makes thresholds more meaningful across different disease types
- Tool: Temperature scaling or Platt scaling

### Knowledge Distillation
- Train a larger YOLOv8l model as teacher
- Distill into YOLOv8x student
- Improved generalization and robustness

---

## 📊 Success Criteria

| Metric | Current | Target |
|--------|---------|--------|
| Tooth Recall | 87.5% | 92%+ |
| Tooth Precision | 76.6% | 85%+ |
| Disease Recall | 66.1% | 80%+ |
| Disease Precision | 53.6% | 70%+ |

**Estimated timeline:** 2-3 weeks with implementation of priorities 1-3
**Expected final F1 scores:** ~0.87 (teeth), ~0.74 (disease)

---

## 🔗 Related Files to Modify

```
config/
  ├─ train_config.yaml          (epochs, augmentation, pos_weight)
  └─ dataset.yaml               (class/disease balancing)

src/training/
  ├─ trainer.py                 (Phase 2b/3 epochs, batch sampling)
  └─ loss.py                    (focal loss, hard negative mining)

src/inference/
  ├─ predictor.py               (NMS IoU threshold)
  ├─ postprocess.py             (smoothing, post-processing)
  └─ (new) validate_improved_model.py (metrics tracking)

scripts/
  └─ verify_local_image.py      (threshold tuning)
```

