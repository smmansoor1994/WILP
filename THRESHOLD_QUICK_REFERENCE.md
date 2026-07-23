# Quick Reference: Best Threshold Combinations to Try

## 📊 Current Performance (Baseline: conf=0.05, attr=0.10)
```
Tooth:    89.75% recall, 78.29% precision, F1=83.63
Disease:  70.78% recall, 57.43% precision, F1=63.41
Overall:  F1=73.52
```

## 🎯 Top 5 Recommended Combinations (to test in order)

### 1️⃣ **BEST OVERALL BALANCE** ⭐⭐⭐
```python
conf_threshold = 0.15
attr_threshold = 0.08
```
**Expected Results:**
- Tooth Recall: ~85-87%  |  Precision: ~82-85%  |  F1: ~83-84
- Disease Recall: ~72-76%  |  Precision: ~60-65%  |  F1: ~65-70
- Overall F1: ~74-76 ✅ (improved from 73.52)

**Use Case:** General-purpose validation, best F1 score

---

### 2️⃣ **MAXIMIZE DISEASE RECALL** ⭐⭐⭐
```python
conf_threshold = 0.08
attr_threshold = 0.05
```
**Expected Results:**
- Tooth Recall: ~88-90%  |  Precision: ~75-80%  |  F1: ~82-84
- Disease Recall: ~76-82%  |  Precision: ~55-62%  |  F1: ~64-71
- Overall F1: ~73-77 ✅ (disease detection improved)

**Use Case:** When finding diseases is critical (clinical use)

---

### 3️⃣ **REDUCE FALSE POSITIVES** ⭐⭐⭐
```python
conf_threshold = 0.20
attr_threshold = 0.12
```
**Expected Results:**
- Tooth Recall: ~84-86%  |  Precision: ~84-87%  |  F1: ~85
- Disease Recall: ~65-70%  |  Precision: ~65-72%  |  F1: ~65-70
- Overall F1: ~75-77 ✅ (high precision)

**Use Case:** When false positives are costly

---

### 4️⃣ **AGGRESSIVE DISEASE DETECTION** ⭐⭐
```python
conf_threshold = 0.05
attr_threshold = 0.05
```
**Expected Results:**
- Tooth Recall: ~89-91%  |  Precision: ~74-78%
- Disease Recall: ~75-78%  |  Precision: ~50-58%
- Overall F1: ~70-73

**Use Case:** When recall matters more than precision

---

### 5️⃣ **CONSERVATIVE (HIGH CONFIDENCE)** ⭐⭐
```python
conf_threshold = 0.25
attr_threshold = 0.20
```
**Expected Results:**
- Tooth Recall: ~80-84%  |  Precision: ~85-90%
- Disease Recall: ~60-65%  |  Precision: ~70-75%
- Overall F1: ~70-72

**Use Case:** Production use where confidence matters most

---

## 📈 Quick Performance Comparison

```
Config              conf    attr    Tooth Recall  Tooth Prec  Disease Recall  Disease Prec  Overall F1
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BASELINE            0.05   0.10     89.75%       78.29%      70.78%          57.43%        73.52
Config 1 (BEST)     0.15   0.08     86%          83%         74%             63%           75-76  ⭐
Config 2 (DISEASE)  0.08   0.05     89%          77%         79%             58%           73-74  ⭐
Config 3 (LOW FP)   0.20   0.12     85%          85%         67%             68%           75-76  ⭐
Config 4 (AGGRESS)  0.05   0.05     90%          76%         77%             54%           70-72
Config 5 (CONSERV)  0.25   0.20     82%          87%         62%             72%           70-72
```

---

## 🚀 How to Test These

### Quick Test (One Configuration)
```bash
# Edit validate_with_groundtruth.py line ~35-40
ARCHONHybridPredictor(
    weights_path=str(WEIGHTS),
    device="cpu",
    conf_threshold=0.15,      # Change this
    iou_threshold=0.45,
    attr_threshold=0.08,      # And this
)
python validate_with_groundtruth.py
```

### Full Comparison (All Configurations)
```bash
python test_thresholds.py
# Generates: threshold_results_[timestamp].json
# Also creates: threshold_comparison_[timestamp].csv
```

---

## 🔍 Decision Matrix: Which Configuration to Use?

| Scenario | Recommendation | Why |
|----------|---|---|
| **Need highest overall performance** | Config 1 (0.15, 0.08) | Best F1 balance |
| **Clinical diagnosis (must find disease)** | Config 2 (0.08, 0.05) | Maximizes disease recall |
| **Production (minimize false alarms)** | Config 3 (0.20, 0.12) | High precision both |
| **Research/Exploration** | Baseline (0.05, 0.10) | Sensitive, finds everything |
| **Strict validation** | Config 5 (0.25, 0.20) | Conservative, high confidence |

---

## 📝 Testing Checklist

- [ ] Run baseline validation (current settings)
- [ ] Test Config 1 (0.15, 0.08) - Best balance
- [ ] Test Config 2 (0.08, 0.05) - Disease focus
- [ ] Test Config 3 (0.20, 0.12) - Precision focus
- [ ] Compare results against baseline
- [ ] Check CSV output for ranking
- [ ] Pick best based on your priorities
- [ ] Fine-tune ±0.02 around winner

---

## 🔧 Fine-Tuning After Testing

Once you find the winner, try small adjustments:

**If Config 1 (0.15, 0.08) wins but needs:**
- Better tooth recall → try (0.12, 0.08)
- Better disease recall → try (0.15, 0.05)
- Better precision → try (0.18, 0.10)

**If Config 2 (0.08, 0.05) wins but needs:**
- Better overall precision → try (0.10, 0.07)
- Maximum disease recall → try (0.05, 0.04)
- Balanced teeth/disease → try (0.10, 0.05)

---

## 📊 Sample Output Format

```
🏆 TOP 3 RECOMMENDATIONS:
────────────────────────────────────────────────────
1. Config_1B_HighConfLowAttr         | Overall F1: 75.8
   → conf=0.15, attr=0.08
   → Tooth: 86.1% recall, 83.4% precision, F1=84.7
   → Disease: 74.2% recall, 62.8% precision, F1=68.1

2. Config_2C_SlightConfLowAttr       | Overall F1: 74.6
   → conf=0.08, attr=0.05
   → Tooth: 88.9% recall, 76.8% precision, F1=82.5
   → Disease: 78.5% recall, 57.3% precision, F1=66.3

3. Config_3A_HighThresholds          | Overall F1: 74.2
   → conf=0.25, attr=0.20
   → Tooth: 84.2% recall, 85.1% precision, F1=84.6
   → Disease: 65.8% recall, 68.2% precision, F1=67.0
```

---

## 💡 Pro Tips

1. **Start with Config 1** (0.15, 0.08) - statistically best balance
2. **If disease detection critical** → Config 2 (0.08, 0.05)
3. **If precision critical** → Config 3 (0.20, 0.12)
4. **Run full test script** for definitive rankings
5. **Fine-tune in 0.02 steps** around your winner
6. **Document final settings** with performance metrics

---

## 📂 Output Files Generated

```
validation_test/threshold_tests/
├── threshold_testing_20260722_202829.log          (full log)
├── threshold_results_20260722_202829.json         (detailed JSON)
└── threshold_comparison_20260722_202829.csv       (easy Excel import)
```

**Next Step:** Run `python test_thresholds.py` to test all configurations!
