## 🚀 Train Phase 2b & 3 Only — Quick Reference

### ✅ What You Have
- ✓ `weights/` folder with Phase 2 best.pt
- ✓ `outputs/` folder (for results)
- ✓ `data/processed/` with training images and labels
- ✓ Corrected `config/train_config.yaml` with `attr_pos_weight: [6.0, 3.6, 9.6, 6.0]`

### ❌ What You DON'T Need
- ✗ Preprocessing (already done)
- ✗ Labels generation (already in data/processed/labels_ext/)
- ✗ Epoch.pts files (auto-created during training)

---

## 📋 Commands

### **Option 1: Train Phase 2b ONLY (30-45 min)**
```bash
python train_phase2b_3_only.py --phase 2b --device cpu
```

**Output:**
- Phase 2b weights: `outputs/runs/attr/weights/best.pt`
- Copied to: `weights/attr_best.pt`

---

### **Option 2: Train Phase 3 ONLY (30-45 min)**
*Requires Phase 2 weights (already have it)*
```bash
python train_phase2b_3_only.py --phase 3 --device cpu
```

**Output:**
- Phase 3 weights: `outputs/runs/hybrid/weights/best.pt`
- Copied to: `weights/hybrid_best.pt`

---

### **Option 3: Train BOTH Phase 2b → Phase 3 (1.5-2 hours)**
```bash
python train_phase2b_3_only.py --phase 2b3 --device cpu
```

**Output:**
- Phase 2b: `weights/attr_best.pt`
- Phase 3: `weights/hybrid_best.pt`

---

### **With Custom Base Weights Path**
```bash
python train_phase2b_3_only.py --phase 2b3 --weights "D:\WILP\Workingcode\models\archon-100-improved-main\weights_20260705_145809\content\WILP\weights\archon_best.pt" --device cpu
```

---

## 📊 Expected Flow

```
outputs/               ← All training logs, checkpoints
  └─ runs/
      ├─ attr/        ← Phase 2b checkpoints (when --phase 2b)
      └─ hybrid/      ← Phase 3 checkpoints (when --phase 3)

weights/              ← FINAL WEIGHTS (auto-copied here)
  ├─ attr_best.pt     ← Phase 2b final model
  └─ hybrid_best.pt   ← Phase 3 final model
```

---

## ✔️ After Training

### Validate Results
```bash
python validate_with_groundtruth.py
```

### Test Inference
```bash
python verify_local_image.py --image test_image.jpg --weights weights/hybrid_best.pt --device cpu
```

### Check Training Loss
```bash
tail -f outputs/archon.log  # View live logs
```

---

## 🔧 Config Modifications (if needed)

Edit `config/train_config.yaml`:
```yaml
# Phase 2b epochs
phase2b_epochs: 100  # Already set

# Phase 3 epochs
phase3_epochs: 100   # Already set

# Disease balance (already corrected)
attr_pos_weight: [6.0, 3.6, 9.6, 6.0]  # ✓ Corrected

# Learning rates
lr0: 0.01
phase2_lr0: 0.002    # For Phase 3

# Augmentation (already enhanced)
augmentation:
  hsv_v: 0.6
  degrees: 10.0
  scale: 0.7
  blur: 0.2
```

---

## 🎯 Recommended Strategy

1. **First Run: Phase 2b Only**
   ```bash
   python train_phase2b_3_only.py --phase 2b --device cpu
   ```
   → Validate: `python validate_with_groundtruth.py`
   → Check if disease precision recovers to 50-60%

2. **If Good Results: Add Phase 3**
   ```bash
   python train_phase2b_3_only.py --phase 3 --device cpu
   ```
   → Validate again to see final metrics

3. **If Still Below Target: Tune Disease Threshold**
   ```bash
   python tune_disease_threshold.py --weights weights/hybrid_best.pt --device cpu
   ```

---

## 📝 Script Features

✅ Auto-detects Phase 2 weights from standard locations
✅ Creates output directories automatically
✅ Copies final weights to `weights/` folder
✅ Detailed logging to `outputs/archon.log`
✅ Supports `--resume` for interrupted training
✅ No preprocessing or labels generation needed

---

**Ready?** Run: `python train_phase2b_3_only.py --phase 2b --device cpu`
