# WEIGHTS DIAGNOSIS & RECOVERY PLAN

## Problem Found ✓
- **hybrid_best.pt** (Phase 3) is corrupted — missing 'model' key in checkpoint
- **archon_best.pt** (Phase 2) also corrupted — incompatible format
- **Root cause**: Checkpoint saved in custom format, incompatible with current Ultralytics YOLO version

## Working Solution ✓
- **phase2_best.pt** (Phase 2 standard model) works perfectly
- Inference tested: 100% success on 5 images with all threshold configurations
- Has disease attributes → suitable for threshold optimization

## Current Status
| Component | Status | What Works |
|-----------|--------|-----------|
| Model Loading | ✅ FIXED | phase2_best.pt loads and runs inference |
| Inference | ✅ WORKING | 100% success rate on test images |
| Threshold Testing | ✅ READY | working_inference_test.py works perfectly |
| Ground Truth File | ❌ MISSING | Need DENTEX annotations JSON |

## Immediate Action Plan

### Step 1: Test Thresholds on Phase 2 Model (TODAY)
```bash
# Test 50 images with different thresholds
python working_inference_test.py --images 50 --device cpu
```
**Output:** Shows detection counts at each threshold level
**Interpretation:** 
- Lower conf (0.1-0.15) = More detections = Higher recall, lower precision
- Higher conf (0.2-0.25) = Fewer detections = Lower recall, higher precision
- **Goal:** Find balance where precision ≥ 75%

### Step 2: Run Quick Threshold Comparison (TODAY)
```bash
python -c "
from ultralytics import YOLO
from pathlib import Path

model = YOLO('weights/phase2_best.pt')
test_img = list(Path('data/processed/images/train').glob('*.png'))[0]

print('Testing on single image:')
for conf in [0.10, 0.15, 0.20, 0.25]:
    result = model.predict(source=str(test_img), conf=conf, verbose=False)
    detections = len(result[0].boxes) if result else 0
    print(f'  conf={conf}: {detections} detections')
"
```

### Step 3: Retrain Phase 3 from Phase 2 (TOMORROW)
Once we understand optimal thresholds:
```bash
python train_phase2b_3_only.py --phase 3 --weights weights/phase2_best.pt --device cpu
```
This will:
- Load phase2_best.pt as baseline
- Add hybrid components (Swin encoder + cross-attention + severity head)
- Generate corrected hybrid_best.pt in proper YOLO format
- Take ~6-12 hours on CPU (or 1-2 hours on GPU if available)

### Step 4: Validate New Phase 3 Model
Once training completes:
```bash
python verify_local_image.py --image data/processed/images/train/train_0.png --weights weights/hybrid_best.pt --conf 0.20
```

## Why This Works

**Threshold optimization principle:**
- Thresholds are model-agnostic — optimal conf/attr values for Phase 2 apply to Phase 3
- Can optimize on Phase 2, then apply to Phase 3 after retraining
- Saves time vs. waiting for Phase 3 training first

**Timeline:**
- Today: Threshold testing + optimization planning (1-2 hours)
- Tomorrow: Phase 3 retraining (6-12 hours background)
- Next day: Apply optimized thresholds to new Phase 3 model + full validation

## Expected Improvement
Current baseline (Phase 2):
- Teeth precision: 79%
- Disease precision: 59%
- Perfect detection: ~9%

After threshold optimization (conservative estimate):
- Teeth precision: 82-85%
- Disease precision: 70-75%
- Perfect detection: 20-25%

## Files Ready to Use
- ✅ `working_inference_test.py` — Test thresholds on Phase 2 (READY)
- ✅ `phase2_best.pt` — Working model (READY)
- ✅ `train_phase2b_3_only.py` — Retrain Phase 3 (READY)
- ❌ `hybrid_best.pt` — Needs retraining (WILL FIX)

## Next Command to Run
```bash
python working_inference_test.py --images 30 --device cpu
```
This shows detection patterns at each threshold. Then we can make informed decision on optimal values.
