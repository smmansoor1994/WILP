# Dental X-ray Analysis - Comprehensive Validation Report

## Executive Summary

This report presents detailed per-tooth metrics for teeth identification and disease detection,
along with comprehensive performance metrics including Precision, Recall, Specificity, and Sensitivity.

**Validation Dataset:** 470 images
**Analysis Date:** 2026-07-11 20:22:02

---

## 1. OVERALL PERFORMANCE METRICS

### 1.1 Teeth Identification (Overall)

| Metric | Value |
|--------|-------|
| **True Positives (TP)** | 2255 |
| **False Positives (FP)** | 599 |
| **False Negatives (FN)** | 217 |
| **Total Detected** | 2854 |
| **Total Ground Truth** | 2472 |

#### Performance Indicators

| Metric | Value |
|--------|-------|
| **Precision** | 79.01% |
| **Recall** | 91.22% |
| **Sensitivity** | 91.22% |
| **Specificity** | 0.00% |
| **F1 Score** | 84.68 |

### 1.2 Disease Identification (Overall)

| Metric | Value |
|--------|-------|
| **True Positives (TP)** | 1793 |
| **False Positives (FP)** | 1265 |
| **False Negatives (FN)** | 725 |
| **Total Detected** | 3058 |
| **Total Ground Truth** | 2518 |

#### Performance Indicators

| Metric | Value |
|--------|-------|
| **Precision** | 58.63% |
| **Recall** | 71.21% |
| **Sensitivity** | 71.21% |
| **Specificity** | 0.00% |
| **F1 Score** | 64.31 |

---

## 2. PER-TOOTH IDENTIFICATION METRICS

### Confusion Matrix and Performance for Each Tooth

| FDI Tooth | Missed | TP | FP | FN | Precision | Recall | Sensitivity | F1 Score |
|-----------|--------|----|----|----|-----------|--------|-------------|----------|
| **FDI11** | ❌ | 20 | 14 | 2 | 58.8% | 90.9% | 90.9% | 71.43 |
| **FDI12** | ❌ | 21 | 12 | 2 | 63.6% | 91.3% | 91.3% | 75.00 |
| **FDI13** | ❌ | 7 | 3 | 6 | 70.0% | 53.8% | 53.8% | 60.87 |
| **FDI14** | ✓ | 40 | 14 | 0 | 74.1% | 100.0% | 100.0% | 85.11 |
| **FDI15** | ❌ | 51 | 7 | 8 | 87.9% | 86.4% | 86.4% | 87.18 |
| **FDI16** | ❌ | 137 | 25 | 13 | 84.6% | 91.3% | 91.3% | 87.82 |
| **FDI17** | ❌ | 110 | 35 | 4 | 75.9% | 96.5% | 96.5% | 84.94 |
| **FDI18** | ✓ | 166 | 100 | 0 | 62.4% | 100.0% | 100.0% | 76.85 |
| **FDI21** | ❌ | 15 | 12 | 1 | 55.6% | 93.8% | 93.8% | 69.77 |
| **FDI22** | ❌ | 23 | 7 | 3 | 76.7% | 88.5% | 88.5% | 82.14 |
| **FDI23** | ❌ | 8 | 7 | 10 | 53.3% | 44.4% | 44.4% | 48.48 |
| **FDI24** | ❌ | 28 | 11 | 4 | 71.8% | 87.5% | 87.5% | 78.87 |
| **FDI25** | ❌ | 69 | 13 | 8 | 84.1% | 89.6% | 89.6% | 86.79 |
| **FDI26** | ❌ | 132 | 18 | 20 | 88.0% | 86.8% | 86.8% | 87.42 |
| **FDI27** | ❌ | 94 | 16 | 25 | 85.5% | 79.0% | 79.0% | 82.10 |
| **FDI28** | ❌ | 149 | 22 | 2 | 87.1% | 98.7% | 98.7% | 92.55 |
| **FDI31** | ❌ | 0 | 3 | 6 | 0.0% | 0.0% | 0.0% | 0.00 |
| **FDI32** | ❌ | 3 | 4 | 3 | 42.9% | 50.0% | 50.0% | 46.15 |
| **FDI33** | ❌ | 13 | 1 | 4 | 92.9% | 76.5% | 76.5% | 83.87 |
| **FDI34** | ❌ | 52 | 44 | 1 | 54.2% | 98.1% | 98.1% | 69.80 |
| **FDI35** | ❌ | 75 | 15 | 12 | 83.3% | 86.2% | 86.2% | 84.75 |
| **FDI36** | ❌ | 180 | 34 | 3 | 84.1% | 98.4% | 98.4% | 90.68 |
| **FDI37** | ❌ | 154 | 55 | 4 | 73.7% | 97.5% | 97.5% | 83.92 |
| **FDI38** | ❌ | 168 | 15 | 5 | 91.8% | 97.1% | 97.1% | 94.38 |
| **FDI41** | ❌ | 3 | 5 | 6 | 37.5% | 33.3% | 33.3% | 35.29 |
| **FDI42** | ❌ | 7 | 5 | 3 | 58.3% | 70.0% | 70.0% | 63.64 |
| **FDI43** | ❌ | 6 | 3 | 11 | 66.7% | 35.3% | 35.3% | 46.15 |
| **FDI44** | ❌ | 40 | 12 | 12 | 76.9% | 76.9% | 76.9% | 76.92 |
| **FDI45** | ❌ | 72 | 20 | 5 | 78.3% | 93.5% | 93.5% | 85.21 |
| **FDI46** | ❌ | 152 | 18 | 7 | 89.4% | 95.6% | 95.6% | 92.40 |
| **FDI47** | ❌ | 149 | 32 | 13 | 82.3% | 92.0% | 92.0% | 86.88 |
| **FDI48** | ❌ | 153 | 9 | 18 | 94.4% | 89.5% | 89.5% | 91.89 |

---

## 3. DETAILED PER-TOOTH ANALYSIS

### Tooth FDI11

#### Confusion Matrix

| Metric | Count |
|--------|-------|
| True Positives (TP) - Correctly identified | 20 |
| False Positives (FP) - Incorrectly identified | 14 |
| False Negatives (FN) - Missed teeth | 2 |
| True Negatives (TN) | 0 |
| **Total Detected** | 34 |
| **Total Ground Truth** | 22 |

#### Performance Metrics

| Metric | Formula | Value |
|--------|---------|-------|
| **Precision** | TP / (TP + FP) | 58.82% |
| **Recall** | TP / (TP + FN) | 90.91% |
| **Sensitivity** | TP / (TP + FN) | 90.91% |
| **Specificity** | TN / (TN + FP) | 0.00% |
| **F1 Score** | 2 × (P × R) / (P + R) | 71.43 |

**Quality Assessment:** 🟡 NEEDS IMPROVEMENT

### Tooth FDI12

#### Confusion Matrix

| Metric | Count |
|--------|-------|
| True Positives (TP) - Correctly identified | 21 |
| False Positives (FP) - Incorrectly identified | 12 |
| False Negatives (FN) - Missed teeth | 2 |
| True Negatives (TN) | 0 |
| **Total Detected** | 33 |
| **Total Ground Truth** | 23 |

#### Performance Metrics

| Metric | Formula | Value |
|--------|---------|-------|
| **Precision** | TP / (TP + FP) | 63.64% |
| **Recall** | TP / (TP + FN) | 91.30% |
| **Sensitivity** | TP / (TP + FN) | 91.30% |
| **Specificity** | TN / (TN + FP) | 0.00% |
| **F1 Score** | 2 × (P × R) / (P + R) | 75.00 |

**Quality Assessment:** 🟡 NEEDS IMPROVEMENT

### Tooth FDI13

#### Confusion Matrix

| Metric | Count |
|--------|-------|
| True Positives (TP) - Correctly identified | 7 |
| False Positives (FP) - Incorrectly identified | 3 |
| False Negatives (FN) - Missed teeth | 6 |
| True Negatives (TN) | 0 |
| **Total Detected** | 10 |
| **Total Ground Truth** | 13 |

#### Performance Metrics

| Metric | Formula | Value |
|--------|---------|-------|
| **Precision** | TP / (TP + FP) | 70.00% |
| **Recall** | TP / (TP + FN) | 53.85% |
| **Sensitivity** | TP / (TP + FN) | 53.85% |
| **Specificity** | TN / (TN + FP) | 0.00% |
| **F1 Score** | 2 × (P × R) / (P + R) | 60.87 |

**Quality Assessment:** 🟡 NEEDS IMPROVEMENT

### Tooth FDI14

#### Confusion Matrix

| Metric | Count |
|--------|-------|
| True Positives (TP) - Correctly identified | 40 |
| False Positives (FP) - Incorrectly identified | 14 |
| False Negatives (FN) - Missed teeth | 0 |
| True Negatives (TN) | 0 |
| **Total Detected** | 54 |
| **Total Ground Truth** | 40 |

#### Performance Metrics

| Metric | Formula | Value |
|--------|---------|-------|
| **Precision** | TP / (TP + FP) | 74.07% |
| **Recall** | TP / (TP + FN) | 100.00% |
| **Sensitivity** | TP / (TP + FN) | 100.00% |
| **Specificity** | TN / (TN + FP) | 0.00% |
| **F1 Score** | 2 × (P × R) / (P + R) | 85.11 |

**Quality Assessment:** 🟢 EXCELLENT

### Tooth FDI15

#### Confusion Matrix

| Metric | Count |
|--------|-------|
| True Positives (TP) - Correctly identified | 51 |
| False Positives (FP) - Incorrectly identified | 7 |
| False Negatives (FN) - Missed teeth | 8 |
| True Negatives (TN) | 0 |
| **Total Detected** | 58 |
| **Total Ground Truth** | 59 |

#### Performance Metrics

| Metric | Formula | Value |
|--------|---------|-------|
| **Precision** | TP / (TP + FP) | 87.93% |
| **Recall** | TP / (TP + FN) | 86.44% |
| **Sensitivity** | TP / (TP + FN) | 86.44% |
| **Specificity** | TN / (TN + FP) | 0.00% |
| **F1 Score** | 2 × (P × R) / (P + R) | 87.18 |

**Quality Assessment:** 🟢 EXCELLENT

### Tooth FDI16

#### Confusion Matrix

| Metric | Count |
|--------|-------|
| True Positives (TP) - Correctly identified | 137 |
| False Positives (FP) - Incorrectly identified | 25 |
| False Negatives (FN) - Missed teeth | 13 |
| True Negatives (TN) | 0 |
| **Total Detected** | 162 |
| **Total Ground Truth** | 150 |

#### Performance Metrics

| Metric | Formula | Value |
|--------|---------|-------|
| **Precision** | TP / (TP + FP) | 84.57% |
| **Recall** | TP / (TP + FN) | 91.33% |
| **Sensitivity** | TP / (TP + FN) | 91.33% |
| **Specificity** | TN / (TN + FP) | 0.00% |
| **F1 Score** | 2 × (P × R) / (P + R) | 87.82 |

**Quality Assessment:** 🟢 EXCELLENT

### Tooth FDI17

#### Confusion Matrix

| Metric | Count |
|--------|-------|
| True Positives (TP) - Correctly identified | 110 |
| False Positives (FP) - Incorrectly identified | 35 |
| False Negatives (FN) - Missed teeth | 4 |
| True Negatives (TN) | 0 |
| **Total Detected** | 145 |
| **Total Ground Truth** | 114 |

#### Performance Metrics

| Metric | Formula | Value |
|--------|---------|-------|
| **Precision** | TP / (TP + FP) | 75.86% |
| **Recall** | TP / (TP + FN) | 96.49% |
| **Sensitivity** | TP / (TP + FN) | 96.49% |
| **Specificity** | TN / (TN + FP) | 0.00% |
| **F1 Score** | 2 × (P × R) / (P + R) | 84.94 |

**Quality Assessment:** 🟢 EXCELLENT

### Tooth FDI18

#### Confusion Matrix

| Metric | Count |
|--------|-------|
| True Positives (TP) - Correctly identified | 166 |
| False Positives (FP) - Incorrectly identified | 100 |
| False Negatives (FN) - Missed teeth | 0 |
| True Negatives (TN) | 0 |
| **Total Detected** | 266 |
| **Total Ground Truth** | 166 |

#### Performance Metrics

| Metric | Formula | Value |
|--------|---------|-------|
| **Precision** | TP / (TP + FP) | 62.41% |
| **Recall** | TP / (TP + FN) | 100.00% |
| **Sensitivity** | TP / (TP + FN) | 100.00% |
| **Specificity** | TN / (TN + FP) | 0.00% |
| **F1 Score** | 2 × (P × R) / (P + R) | 76.85 |

**Quality Assessment:** 🟡 NEEDS IMPROVEMENT

### Tooth FDI21

#### Confusion Matrix

| Metric | Count |
|--------|-------|
| True Positives (TP) - Correctly identified | 15 |
| False Positives (FP) - Incorrectly identified | 12 |
| False Negatives (FN) - Missed teeth | 1 |
| True Negatives (TN) | 0 |
| **Total Detected** | 27 |
| **Total Ground Truth** | 16 |

#### Performance Metrics

| Metric | Formula | Value |
|--------|---------|-------|
| **Precision** | TP / (TP + FP) | 55.56% |
| **Recall** | TP / (TP + FN) | 93.75% |
| **Sensitivity** | TP / (TP + FN) | 93.75% |
| **Specificity** | TN / (TN + FP) | 0.00% |
| **F1 Score** | 2 × (P × R) / (P + R) | 69.77 |

**Quality Assessment:** 🟡 NEEDS IMPROVEMENT

### Tooth FDI22

#### Confusion Matrix

| Metric | Count |
|--------|-------|
| True Positives (TP) - Correctly identified | 23 |
| False Positives (FP) - Incorrectly identified | 7 |
| False Negatives (FN) - Missed teeth | 3 |
| True Negatives (TN) | 0 |
| **Total Detected** | 30 |
| **Total Ground Truth** | 26 |

#### Performance Metrics

| Metric | Formula | Value |
|--------|---------|-------|
| **Precision** | TP / (TP + FP) | 76.67% |
| **Recall** | TP / (TP + FN) | 88.46% |
| **Sensitivity** | TP / (TP + FN) | 88.46% |
| **Specificity** | TN / (TN + FP) | 0.00% |
| **F1 Score** | 2 × (P × R) / (P + R) | 82.14 |

**Quality Assessment:** 🟢 EXCELLENT

### Tooth FDI23

#### Confusion Matrix

| Metric | Count |
|--------|-------|
| True Positives (TP) - Correctly identified | 8 |
| False Positives (FP) - Incorrectly identified | 7 |
| False Negatives (FN) - Missed teeth | 10 |
| True Negatives (TN) | 0 |
| **Total Detected** | 15 |
| **Total Ground Truth** | 18 |

#### Performance Metrics

| Metric | Formula | Value |
|--------|---------|-------|
| **Precision** | TP / (TP + FP) | 53.33% |
| **Recall** | TP / (TP + FN) | 44.44% |
| **Sensitivity** | TP / (TP + FN) | 44.44% |
| **Specificity** | TN / (TN + FP) | 0.00% |
| **F1 Score** | 2 × (P × R) / (P + R) | 48.48 |

**Quality Assessment:** 🔴 POOR

### Tooth FDI24

#### Confusion Matrix

| Metric | Count |
|--------|-------|
| True Positives (TP) - Correctly identified | 28 |
| False Positives (FP) - Incorrectly identified | 11 |
| False Negatives (FN) - Missed teeth | 4 |
| True Negatives (TN) | 0 |
| **Total Detected** | 39 |
| **Total Ground Truth** | 32 |

#### Performance Metrics

| Metric | Formula | Value |
|--------|---------|-------|
| **Precision** | TP / (TP + FP) | 71.79% |
| **Recall** | TP / (TP + FN) | 87.50% |
| **Sensitivity** | TP / (TP + FN) | 87.50% |
| **Specificity** | TN / (TN + FP) | 0.00% |
| **F1 Score** | 2 × (P × R) / (P + R) | 78.87 |

**Quality Assessment:** 🟢 EXCELLENT

### Tooth FDI25

#### Confusion Matrix

| Metric | Count |
|--------|-------|
| True Positives (TP) - Correctly identified | 69 |
| False Positives (FP) - Incorrectly identified | 13 |
| False Negatives (FN) - Missed teeth | 8 |
| True Negatives (TN) | 0 |
| **Total Detected** | 82 |
| **Total Ground Truth** | 77 |

#### Performance Metrics

| Metric | Formula | Value |
|--------|---------|-------|
| **Precision** | TP / (TP + FP) | 84.15% |
| **Recall** | TP / (TP + FN) | 89.61% |
| **Sensitivity** | TP / (TP + FN) | 89.61% |
| **Specificity** | TN / (TN + FP) | 0.00% |
| **F1 Score** | 2 × (P × R) / (P + R) | 86.79 |

**Quality Assessment:** 🟢 EXCELLENT

### Tooth FDI26

#### Confusion Matrix

| Metric | Count |
|--------|-------|
| True Positives (TP) - Correctly identified | 132 |
| False Positives (FP) - Incorrectly identified | 18 |
| False Negatives (FN) - Missed teeth | 20 |
| True Negatives (TN) | 0 |
| **Total Detected** | 150 |
| **Total Ground Truth** | 152 |

#### Performance Metrics

| Metric | Formula | Value |
|--------|---------|-------|
| **Precision** | TP / (TP + FP) | 88.00% |
| **Recall** | TP / (TP + FN) | 86.84% |
| **Sensitivity** | TP / (TP + FN) | 86.84% |
| **Specificity** | TN / (TN + FP) | 0.00% |
| **F1 Score** | 2 × (P × R) / (P + R) | 87.42 |

**Quality Assessment:** 🟢 EXCELLENT

### Tooth FDI27

#### Confusion Matrix

| Metric | Count |
|--------|-------|
| True Positives (TP) - Correctly identified | 94 |
| False Positives (FP) - Incorrectly identified | 16 |
| False Negatives (FN) - Missed teeth | 25 |
| True Negatives (TN) | 0 |
| **Total Detected** | 110 |
| **Total Ground Truth** | 119 |

#### Performance Metrics

| Metric | Formula | Value |
|--------|---------|-------|
| **Precision** | TP / (TP + FP) | 85.45% |
| **Recall** | TP / (TP + FN) | 78.99% |
| **Sensitivity** | TP / (TP + FN) | 78.99% |
| **Specificity** | TN / (TN + FP) | 0.00% |
| **F1 Score** | 2 × (P × R) / (P + R) | 82.10 |

**Quality Assessment:** 🟢 EXCELLENT

### Tooth FDI28

#### Confusion Matrix

| Metric | Count |
|--------|-------|
| True Positives (TP) - Correctly identified | 149 |
| False Positives (FP) - Incorrectly identified | 22 |
| False Negatives (FN) - Missed teeth | 2 |
| True Negatives (TN) | 0 |
| **Total Detected** | 171 |
| **Total Ground Truth** | 151 |

#### Performance Metrics

| Metric | Formula | Value |
|--------|---------|-------|
| **Precision** | TP / (TP + FP) | 87.13% |
| **Recall** | TP / (TP + FN) | 98.68% |
| **Sensitivity** | TP / (TP + FN) | 98.68% |
| **Specificity** | TN / (TN + FP) | 0.00% |
| **F1 Score** | 2 × (P × R) / (P + R) | 92.55 |

**Quality Assessment:** 🟢 EXCELLENT

### Tooth FDI31

#### Confusion Matrix

| Metric | Count |
|--------|-------|
| True Positives (TP) - Correctly identified | 0 |
| False Positives (FP) - Incorrectly identified | 3 |
| False Negatives (FN) - Missed teeth | 6 |
| True Negatives (TN) | 0 |
| **Total Detected** | 3 |
| **Total Ground Truth** | 6 |

#### Performance Metrics

| Metric | Formula | Value |
|--------|---------|-------|
| **Precision** | TP / (TP + FP) | 0.00% |
| **Recall** | TP / (TP + FN) | 0.00% |
| **Sensitivity** | TP / (TP + FN) | 0.00% |
| **Specificity** | TN / (TN + FP) | 0.00% |
| **F1 Score** | 2 × (P × R) / (P + R) | 0.00 |

**Quality Assessment:** 🔴 POOR

### Tooth FDI32

#### Confusion Matrix

| Metric | Count |
|--------|-------|
| True Positives (TP) - Correctly identified | 3 |
| False Positives (FP) - Incorrectly identified | 4 |
| False Negatives (FN) - Missed teeth | 3 |
| True Negatives (TN) | 0 |
| **Total Detected** | 7 |
| **Total Ground Truth** | 6 |

#### Performance Metrics

| Metric | Formula | Value |
|--------|---------|-------|
| **Precision** | TP / (TP + FP) | 42.86% |
| **Recall** | TP / (TP + FN) | 50.00% |
| **Sensitivity** | TP / (TP + FN) | 50.00% |
| **Specificity** | TN / (TN + FP) | 0.00% |
| **F1 Score** | 2 × (P × R) / (P + R) | 46.15 |

**Quality Assessment:** 🔴 POOR

### Tooth FDI33

#### Confusion Matrix

| Metric | Count |
|--------|-------|
| True Positives (TP) - Correctly identified | 13 |
| False Positives (FP) - Incorrectly identified | 1 |
| False Negatives (FN) - Missed teeth | 4 |
| True Negatives (TN) | 0 |
| **Total Detected** | 14 |
| **Total Ground Truth** | 17 |

#### Performance Metrics

| Metric | Formula | Value |
|--------|---------|-------|
| **Precision** | TP / (TP + FP) | 92.86% |
| **Recall** | TP / (TP + FN) | 76.47% |
| **Sensitivity** | TP / (TP + FN) | 76.47% |
| **Specificity** | TN / (TN + FP) | 0.00% |
| **F1 Score** | 2 × (P × R) / (P + R) | 83.87 |

**Quality Assessment:** 🟢 EXCELLENT

### Tooth FDI34

#### Confusion Matrix

| Metric | Count |
|--------|-------|
| True Positives (TP) - Correctly identified | 52 |
| False Positives (FP) - Incorrectly identified | 44 |
| False Negatives (FN) - Missed teeth | 1 |
| True Negatives (TN) | 0 |
| **Total Detected** | 96 |
| **Total Ground Truth** | 53 |

#### Performance Metrics

| Metric | Formula | Value |
|--------|---------|-------|
| **Precision** | TP / (TP + FP) | 54.17% |
| **Recall** | TP / (TP + FN) | 98.11% |
| **Sensitivity** | TP / (TP + FN) | 98.11% |
| **Specificity** | TN / (TN + FP) | 0.00% |
| **F1 Score** | 2 × (P × R) / (P + R) | 69.80 |

**Quality Assessment:** 🟡 NEEDS IMPROVEMENT

### Tooth FDI35

#### Confusion Matrix

| Metric | Count |
|--------|-------|
| True Positives (TP) - Correctly identified | 75 |
| False Positives (FP) - Incorrectly identified | 15 |
| False Negatives (FN) - Missed teeth | 12 |
| True Negatives (TN) | 0 |
| **Total Detected** | 90 |
| **Total Ground Truth** | 87 |

#### Performance Metrics

| Metric | Formula | Value |
|--------|---------|-------|
| **Precision** | TP / (TP + FP) | 83.33% |
| **Recall** | TP / (TP + FN) | 86.21% |
| **Sensitivity** | TP / (TP + FN) | 86.21% |
| **Specificity** | TN / (TN + FP) | 0.00% |
| **F1 Score** | 2 × (P × R) / (P + R) | 84.75 |

**Quality Assessment:** 🟢 EXCELLENT

### Tooth FDI36

#### Confusion Matrix

| Metric | Count |
|--------|-------|
| True Positives (TP) - Correctly identified | 180 |
| False Positives (FP) - Incorrectly identified | 34 |
| False Negatives (FN) - Missed teeth | 3 |
| True Negatives (TN) | 0 |
| **Total Detected** | 214 |
| **Total Ground Truth** | 183 |

#### Performance Metrics

| Metric | Formula | Value |
|--------|---------|-------|
| **Precision** | TP / (TP + FP) | 84.11% |
| **Recall** | TP / (TP + FN) | 98.36% |
| **Sensitivity** | TP / (TP + FN) | 98.36% |
| **Specificity** | TN / (TN + FP) | 0.00% |
| **F1 Score** | 2 × (P × R) / (P + R) | 90.68 |

**Quality Assessment:** 🟢 EXCELLENT

### Tooth FDI37

#### Confusion Matrix

| Metric | Count |
|--------|-------|
| True Positives (TP) - Correctly identified | 154 |
| False Positives (FP) - Incorrectly identified | 55 |
| False Negatives (FN) - Missed teeth | 4 |
| True Negatives (TN) | 0 |
| **Total Detected** | 209 |
| **Total Ground Truth** | 158 |

#### Performance Metrics

| Metric | Formula | Value |
|--------|---------|-------|
| **Precision** | TP / (TP + FP) | 73.68% |
| **Recall** | TP / (TP + FN) | 97.47% |
| **Sensitivity** | TP / (TP + FN) | 97.47% |
| **Specificity** | TN / (TN + FP) | 0.00% |
| **F1 Score** | 2 × (P × R) / (P + R) | 83.92 |

**Quality Assessment:** 🟢 EXCELLENT

### Tooth FDI38

#### Confusion Matrix

| Metric | Count |
|--------|-------|
| True Positives (TP) - Correctly identified | 168 |
| False Positives (FP) - Incorrectly identified | 15 |
| False Negatives (FN) - Missed teeth | 5 |
| True Negatives (TN) | 0 |
| **Total Detected** | 183 |
| **Total Ground Truth** | 173 |

#### Performance Metrics

| Metric | Formula | Value |
|--------|---------|-------|
| **Precision** | TP / (TP + FP) | 91.80% |
| **Recall** | TP / (TP + FN) | 97.11% |
| **Sensitivity** | TP / (TP + FN) | 97.11% |
| **Specificity** | TN / (TN + FP) | 0.00% |
| **F1 Score** | 2 × (P × R) / (P + R) | 94.38 |

**Quality Assessment:** 🟢 EXCELLENT

### Tooth FDI41

#### Confusion Matrix

| Metric | Count |
|--------|-------|
| True Positives (TP) - Correctly identified | 3 |
| False Positives (FP) - Incorrectly identified | 5 |
| False Negatives (FN) - Missed teeth | 6 |
| True Negatives (TN) | 0 |
| **Total Detected** | 8 |
| **Total Ground Truth** | 9 |

#### Performance Metrics

| Metric | Formula | Value |
|--------|---------|-------|
| **Precision** | TP / (TP + FP) | 37.50% |
| **Recall** | TP / (TP + FN) | 33.33% |
| **Sensitivity** | TP / (TP + FN) | 33.33% |
| **Specificity** | TN / (TN + FP) | 0.00% |
| **F1 Score** | 2 × (P × R) / (P + R) | 35.29 |

**Quality Assessment:** 🔴 POOR

### Tooth FDI42

#### Confusion Matrix

| Metric | Count |
|--------|-------|
| True Positives (TP) - Correctly identified | 7 |
| False Positives (FP) - Incorrectly identified | 5 |
| False Negatives (FN) - Missed teeth | 3 |
| True Negatives (TN) | 0 |
| **Total Detected** | 12 |
| **Total Ground Truth** | 10 |

#### Performance Metrics

| Metric | Formula | Value |
|--------|---------|-------|
| **Precision** | TP / (TP + FP) | 58.33% |
| **Recall** | TP / (TP + FN) | 70.00% |
| **Sensitivity** | TP / (TP + FN) | 70.00% |
| **Specificity** | TN / (TN + FP) | 0.00% |
| **F1 Score** | 2 × (P × R) / (P + R) | 63.64 |

**Quality Assessment:** 🟡 NEEDS IMPROVEMENT

### Tooth FDI43

#### Confusion Matrix

| Metric | Count |
|--------|-------|
| True Positives (TP) - Correctly identified | 6 |
| False Positives (FP) - Incorrectly identified | 3 |
| False Negatives (FN) - Missed teeth | 11 |
| True Negatives (TN) | 0 |
| **Total Detected** | 9 |
| **Total Ground Truth** | 17 |

#### Performance Metrics

| Metric | Formula | Value |
|--------|---------|-------|
| **Precision** | TP / (TP + FP) | 66.67% |
| **Recall** | TP / (TP + FN) | 35.29% |
| **Sensitivity** | TP / (TP + FN) | 35.29% |
| **Specificity** | TN / (TN + FP) | 0.00% |
| **F1 Score** | 2 × (P × R) / (P + R) | 46.15 |

**Quality Assessment:** 🔴 POOR

### Tooth FDI44

#### Confusion Matrix

| Metric | Count |
|--------|-------|
| True Positives (TP) - Correctly identified | 40 |
| False Positives (FP) - Incorrectly identified | 12 |
| False Negatives (FN) - Missed teeth | 12 |
| True Negatives (TN) | 0 |
| **Total Detected** | 52 |
| **Total Ground Truth** | 52 |

#### Performance Metrics

| Metric | Formula | Value |
|--------|---------|-------|
| **Precision** | TP / (TP + FP) | 76.92% |
| **Recall** | TP / (TP + FN) | 76.92% |
| **Sensitivity** | TP / (TP + FN) | 76.92% |
| **Specificity** | TN / (TN + FP) | 0.00% |
| **F1 Score** | 2 × (P × R) / (P + R) | 76.92 |

**Quality Assessment:** 🟢 EXCELLENT

### Tooth FDI45

#### Confusion Matrix

| Metric | Count |
|--------|-------|
| True Positives (TP) - Correctly identified | 72 |
| False Positives (FP) - Incorrectly identified | 20 |
| False Negatives (FN) - Missed teeth | 5 |
| True Negatives (TN) | 0 |
| **Total Detected** | 92 |
| **Total Ground Truth** | 77 |

#### Performance Metrics

| Metric | Formula | Value |
|--------|---------|-------|
| **Precision** | TP / (TP + FP) | 78.26% |
| **Recall** | TP / (TP + FN) | 93.51% |
| **Sensitivity** | TP / (TP + FN) | 93.51% |
| **Specificity** | TN / (TN + FP) | 0.00% |
| **F1 Score** | 2 × (P × R) / (P + R) | 85.21 |

**Quality Assessment:** 🟢 EXCELLENT

### Tooth FDI46

#### Confusion Matrix

| Metric | Count |
|--------|-------|
| True Positives (TP) - Correctly identified | 152 |
| False Positives (FP) - Incorrectly identified | 18 |
| False Negatives (FN) - Missed teeth | 7 |
| True Negatives (TN) | 0 |
| **Total Detected** | 170 |
| **Total Ground Truth** | 159 |

#### Performance Metrics

| Metric | Formula | Value |
|--------|---------|-------|
| **Precision** | TP / (TP + FP) | 89.41% |
| **Recall** | TP / (TP + FN) | 95.60% |
| **Sensitivity** | TP / (TP + FN) | 95.60% |
| **Specificity** | TN / (TN + FP) | 0.00% |
| **F1 Score** | 2 × (P × R) / (P + R) | 92.40 |

**Quality Assessment:** 🟢 EXCELLENT

### Tooth FDI47

#### Confusion Matrix

| Metric | Count |
|--------|-------|
| True Positives (TP) - Correctly identified | 149 |
| False Positives (FP) - Incorrectly identified | 32 |
| False Negatives (FN) - Missed teeth | 13 |
| True Negatives (TN) | 0 |
| **Total Detected** | 181 |
| **Total Ground Truth** | 162 |

#### Performance Metrics

| Metric | Formula | Value |
|--------|---------|-------|
| **Precision** | TP / (TP + FP) | 82.32% |
| **Recall** | TP / (TP + FN) | 91.98% |
| **Sensitivity** | TP / (TP + FN) | 91.98% |
| **Specificity** | TN / (TN + FP) | 0.00% |
| **F1 Score** | 2 × (P × R) / (P + R) | 86.88 |

**Quality Assessment:** 🟢 EXCELLENT

### Tooth FDI48

#### Confusion Matrix

| Metric | Count |
|--------|-------|
| True Positives (TP) - Correctly identified | 153 |
| False Positives (FP) - Incorrectly identified | 9 |
| False Negatives (FN) - Missed teeth | 18 |
| True Negatives (TN) | 0 |
| **Total Detected** | 162 |
| **Total Ground Truth** | 171 |

#### Performance Metrics

| Metric | Formula | Value |
|--------|---------|-------|
| **Precision** | TP / (TP + FP) | 94.44% |
| **Recall** | TP / (TP + FN) | 89.47% |
| **Sensitivity** | TP / (TP + FN) | 89.47% |
| **Specificity** | TN / (TN + FP) | 0.00% |
| **F1 Score** | 2 × (P × R) / (P + R) | 91.89 |

**Quality Assessment:** 🟢 EXCELLENT

---

## 4. METRIC INTERPRETATION GUIDE

### Confusion Matrix Terms

- **TP (True Positive):** Tooth correctly identified as present
- **FP (False Positive):** Non-existent tooth incorrectly identified
- **FN (False Negative):** Existing tooth was missed/not identified
- **TN (True Negative):** Non-existent tooth correctly not identified

### Performance Metrics Explained

| Metric | Formula | Meaning | Ideal Value |
|--------|---------|---------|-------------|
| **Precision** | TP/(TP+FP) | Of all detected teeth, how many were correct? | 100% |
| **Recall** | TP/(TP+FN) | Of all actual teeth, how many were detected? | 100% |
| **Sensitivity** | TP/(TP+FN) | Same as Recall - True Positive Rate | 100% |
| **Specificity** | TN/(TN+FP) | How many non-teeth were correctly rejected? | 100% |
| **F1 Score** | 2(P×R)/(P+R) | Harmonic mean of Precision & Recall | 1.00 |

### Performance Thresholds

| Range | Assessment | Recommendation |
|-------|------------|-----------------|
| 90-100% | Excellent | Use for clinical deployment |
| 80-89% | Good | Suitable with monitoring |
| 70-79% | Fair | Needs improvement |
| Below 70% | Poor | Requires significant tuning |

---

## 5. KEY FINDINGS & INSIGHTS

### Top 5 Performing Teeth (by Recall)

1. **FDI14**: 100.0% recall
2. **FDI18**: 100.0% recall
3. **FDI28**: 98.7% recall
4. **FDI36**: 98.4% recall
5. **FDI34**: 98.1% recall

### Top 5 Underperforming Teeth (by Recall)

1. **FDI31**: 0.0% recall ⚠️
2. **FDI41**: 33.3% recall ⚠️
3. **FDI43**: 35.3% recall ⚠️
4. **FDI23**: 44.4% recall ⚠️
5. **FDI32**: 50.0% recall ⚠️

### Average Recall Across All Teeth: **80.3%**

---

## 6. RECOMMENDATIONS

### For System Improvement

1. **High False Positive Rate (FP):** Review detection threshold tuning
2. **False Negative Teeth:** Focus on augmentation for frequently missed teeth
3. **Precision-Recall Trade-off:** Adjust model confidence thresholds based on clinical needs

### For Clinical Deployment

1. Use this model as **screening assistant**, not primary diagnostic tool
2. Implement **human-in-the-loop** validation for teeth with recall < 80%
3. Maintain **confidence score thresholds** for high-stakes decisions

### Priority Improvements

Focus training efforts on these underperforming teeth:
- **FDI31**: Increase training samples, data augmentation (Current recall: 0.0%)
- **FDI41**: Increase training samples, data augmentation (Current recall: 33.3%)
- **FDI43**: Increase training samples, data augmentation (Current recall: 35.3%)

---

## 7. TECHNICAL NOTES

- Analysis based on 470 validation images
- All 32 FDI tooth numbers analyzed
- Metrics calculated independently per tooth
- Performance thresholds based on clinical ML standards

---
*Report Generated Automatically - For detailed analysis, review validation logs*