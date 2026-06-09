# ARCHON — End-to-End Workflow Reference

> **ARCHON: Arch-Contextualized Hierarchical Orthodontic Network for Severity-Aware Dental Radiograph Analysis**
>
> Built on paper: [arxiv.org/abs/2308.05967](https://arxiv.org/abs/2308.05967) (YOLOrtho)  
> Dataset: DENTEX Challenge 2023  
> This document covers **both** the baseline ARCHON pipeline and the new ARCHON hybrid Swin + Cross-Attention improvements.

---

## Table of Contents

1. [Architecture at a Glance](#1-architecture-at-a-glance)
2. [Entry Point: main.py](#2-entry-point-mainpy)
3. [Mode Reference Table](#3-mode-reference-table)
4. [Mode: preprocess](#4-mode-preprocess)
5. [Mode: pseudo_label](#5-mode-pseudo_label)
6. [Mode: train](#6-mode-train)
7. [Mode: train_attr](#7-mode-train_attr)
8. [Mode: train_hybrid](#8-mode-train_hybrid)  ← NEW
9. [Mode: evaluate](#9-mode-evaluate)
10. [Mode: predict](#10-mode-predict)
11. [Mode: full](#11-mode-full)
12. [Key Component Deep-Dives](#12-key-component-deep-dives)
    - [CoordConv](#coordconv)
    - [FDI Numbering System](#fdi-numbering-system)
    - [Hierarchical Loss and data_type](#hierarchical-loss-and-data_type)
    - [Modified FPN and Strides](#modified-fpn-and-strides)
    - [Attribute Heads (Binary Disease Heads)](#attribute-heads-binary-disease-heads)
    - [GlobalContextEncoder — Swin Transformer](#globalcontextencoder--swin-transformer)  ← NEW
    - [MultiScaleFusion — Cross-Attention](#multiscalefusion--cross-attention)  ← NEW
    - [HybridMultiTaskHead — Severity + Quadrant](#hybridmultitaskhead--severity--quadrant)  ← NEW
    - [CLAHE Augmentation](#clahe-augmentation)  ← NEW
    - [Quadrant-Consistency Post-processing](#quadrant-consistency-post-processing)  ← NEW
    - [Linear Sum Assignment](#linear-sum-assignment)
    - [Pseudo Labels](#pseudo-labels)
    - [Flip Augmentation with Quadrant Remapping](#flip-augmentation-with-quadrant-remapping)
13. [Label Format Reference](#13-label-format-reference)
14. [Config Files Reference](#14-config-files-reference)
15. [Data Directory Layout](#15-data-directory-layout)
16. [Output Files Reference](#16-output-files-reference)
17. [Full Pipeline Call Graph](#17-full-pipeline-call-graph)

---

## 1. Architecture at a Glance

### Baseline (foundation paper — arXiv:2308.05967)

```
Input: Panoramic X-ray (1280×640)
         │
         ▼
┌───────────────────────────────────────────────────┐
│  YOLOv8x Backbone  (layers 0–9, CoordConv patched)│
└─────────────────────┬─────────────────────────────┘
                      │  P3, P4, P5 feature maps
                      ▼
┌───────────────────────────────────────────────────┐
│  Modified PANet Neck                              │
│  → strides [4, 8, 16]  (extra upsample for N2)   │
└──────┬──────────────────────────┬────────────────-┘
       │                          │
       ▼                          ▼
  Detection Head (32 FDI)    Attribute Heads ×4
  bbox + class               impacted / caries /
                             deepcaries / lesion
                             (binary BCE per tooth)
       │
       ▼
  Linear Sum Assignment → unique FDI per detection
       │
       ▼
  ToothDetection: fdi, bbox, conf, 4× disease flags
```

### Hybrid Architecture (new improvements)

```
Input: Panoramic X-ray (1280×640)
         │
         ▼
┌───────────────────────────────────────────────────┐
│  YOLOv8x Backbone  (CoordConv)                    │
└─────────────────────┬─────────────────────────────┘
                      │  P3, P4, P5  (FPN features)
          ┌───────────┴───────────┐
          │                       │
          ▼                       ▼
  ┌───────────────┐   ┌───────────────────────────┐
  │  P5 feature   │   │  P5 only                  │
  │  (stride 32)  │──►│  GlobalContextEncoder      │  ← Swin Transformer
  └───────────────┘   │  2× SwinTransformerBlock   │    swin_transformer.py
                      │  (regular + shifted window) │
                      └────────────┬──────────────-┘
                                   │ global context
                                   ▼
               ┌───────────────────────────────────┐
               │  MultiScaleFusion                 │  ← Cross-Attention
               │  Q=CNN features (P3, P4, P5)      │    cross_attention.py
               │  K/V=Swin global context           │
               │  → fused_P3, fused_P4, fused_P5   │
               └───────────────┬───────────────────┘
                               │
          ┌────────────────────┼──────────────────────┐
          ▼                    ▼                       ▼
  Detection Head        Binary Attr Heads       HybridMultiTaskHead
  (unchanged)           (4× legacy BCE)         ├── SeverityHead ×4
                                                │   (Healthy/Mild/Severe)
                                                └── QuadrantAwareFDIHead
                                                    (4-class aux classifier)
          │                                          │
          ▼                                          ▼
  Linear Sum Assignment  ←  quadrant_probs penalty   │
  (with quadrant-consistency bonus)  ◄───────────────┘
          │
          ▼
  ToothDetection + severity_details per tooth
```

---

## 2. Entry Point: main.py

```
python main.py --mode <MODE> [options]
```

All modes flow: `main.py → parse_args() → dispatch[mode](args)`

### CLI Arguments

| Argument | Type | Default | Used By | Purpose |
|---|---|---|---|---|
| `--mode` | str | `full` | All | Pipeline stage to run |
| `--dentex-root` | str | None | `preprocess`, `full` | Path to DENTEX dataset root on disk |
| `--config` | str | `config/train_config.yaml` | `train`, `train_attr`, `train_hybrid`, `full` | Hyperparameters YAML |
| `--weights` | str | None | `evaluate`, `predict`, `train_attr`, `train_hybrid` | Path to `.pt` weights file |
| `--input` | str | None | `predict` | Image file or directory for inference |
| `--output` | str | `outputs/predictions` | `predict` | Where to save results |
| `--device` | str | `cuda` | `train`, `evaluate`, `predict` | `cuda`, `cuda:0`, or `cpu` |
| `--resume` | flag | `False` | `train` | Resume interrupted training |
| `--conf` | float | `0.25` | `evaluate`, `predict` | Detection confidence threshold |

---

## 3. Mode Reference Table

| Mode | Entry Function | Key Module | Weights In | Writes To |
|---|---|---|---|---|
| `preprocess` | `stage_preprocess()` | `src/data/preprocess.py → preprocess_dentex()` | — | `data/processed/` |
| `pseudo_label` | `stage_pseudo_label()` | `src/data/pseudo_label.py → generate_pseudo_labels()` | `outputs/runs/phase1/weights/best.pt` | `data/pseudo/` |
| `train` | `stage_train()` | `src/training/trainer.py → ARCHONTrainer.train()` | `yolov8x.pt` | `outputs/runs/`, `weights/` |
| `train_attr` | `stage_train_attr()` | `trainer.py → train_attr_only()` | `outputs/runs/phase2/weights/best.pt` | `weights/attr_best.pt` |
| `train_hybrid` | `stage_train_hybrid()` | `trainer.py → ARCHONHybridTrainer.train_hybrid()` | `outputs/runs/phase2/weights/best.pt` | `weights/hybrid_best.pt` |
| `evaluate` | `stage_evaluate()` | `src/inference/predictor.py → predictor.evaluate()` | `weights/archon_best.pt` | stdout / log |
| `predict` | `stage_predict()` | `predictor.py → predictor.predict()` | `archon_best.pt` + `attr_best.pt` | `outputs/predictions/` |
| `full` | `run_full_pipeline()` | All stages in order | `yolov8x.pt` (auto) | All above |

---

## 4. Mode: `preprocess`

```bash
python main.py --mode preprocess --dentex-root D:/path/to/DENTEX
```

### Call Chain

```
main.py → stage_preprocess(args)
    └── src/data/preprocess.py → preprocess_dentex(dentex_root, out_dir)
            ├── _process_part1()         → data_type=0 labels (quadrant only)
            ├── _process_part2()         → data_type=1 labels (quadrant + FDI)
            ├── _process_part3()         → data_type=2 labels (FDI + 4 disease flags)
            ├── _process_validation()    → val/ split
            ├── _copy_images()           → test/ split (images only)
            └── _copy_images()           → data/unlabelled/
```

### Extended 10-Column Label Format

```
class_id   cx    cy    w     h     is_imp  has_car  has_dc  has_les  data_type
   0       0.52  0.41  0.08  0.15    0       1        0       0         2
  15       0.30  0.38  0.07  0.14    0       0        0       0         2
```

**Key detail:** Part 3 has one JSON annotation per (tooth, disease). If tooth 36 has both Caries and Deep Caries, it occupies two rows in the JSON. The preprocessor merges them by `(image_id, fdi)` → one label row with `has_caries=1` and `has_deepcaries=1`.

**Disease → column mapping:**

```python
DISEASE_CAT3_TO_ATTR_IDX = {
    0: 0,   # Impacted          → col 5  (is_impacted)
    1: 1,   # Caries            → col 6  (has_caries)
    2: 3,   # Periapical Lesion → col 8  (has_lesion)
    3: 2,   # Deep Caries       → col 7  (has_deepcaries)
}
```

---

## 5. Mode: `pseudo_label`

```bash
python main.py --mode pseudo_label
```

### Call Chain

```
main.py → stage_pseudo_label(args)
    └── src/data/pseudo_label.py → generate_pseudo_labels(...)
            ├── loads Phase 1 YOLO model
            ├── Part 3 images:
            │     predict(conf=0.5) → filter IoU<0.3 with existing labels
            │     → healthy pseudo label (all attrs=0, data_type=2)
            └── Unlabelled images:
                  predict(conf=0.5) → all detections become healthy labels
```

### Why Pseudo Labels Are Needed

DENTEX Part 3 only annotates **diseased** teeth. Unannotated teeth would be treated as background (false positives during training). Pseudo labeling adds explicit healthy-tooth rows so the attribute heads learn "no disease here" correctly.

---

## 6. Mode: `train`

```bash
python main.py --mode train [--resume] [--device cuda]
```

### Call Chain

```
main.py → stage_train(args)
    └── ARCHONTrainer(config_path, resume, device)
            └── .train()
                    ├── _train_phase1()
                    │     ultralytics YOLO("yolov8x.pt").train(phase1_args)
                    │     → outputs/runs/phase1/weights/best.pt
                    │
                    ├── _train_phase2(phase1_best)
                    │     ultralytics YOLO(phase1_best).train(phase2_args)
                    │     → outputs/runs/phase2/weights/best.pt
                    │
                    └── _train_attribute_heads(phase2_best)
                          ├── build_archon_base()
                          │     yolortho.py → inject CoordConv, attach MultiAttributeHead + hooks
                          ├── ARCHONDataset(labels_ext/train/)   src/data/dataset.py
                          ├── AdamW + CosineAnnealingLR
                          ├── AttributeBCELoss (pos_weight auto-computed from dataset)
                          ├── Per-tooth spatial sampling on FPN at each GT bbox center
                          └── → weights/attr_best.pt
```

### Training Phases

#### Phase 1 — Detection Pre-training (100 epochs)

| Detail | Value |
|---|---|
| Data | Parts 1 + 2 (no disease labels) |
| Loss | `bbox×7.5 + cls×0.5 + DFL×1.5` |
| CoordConv | NOT injected (ultralytics rebuilds internally) |
| Purpose | Learn to localize + enumerate teeth; generate quality pseudo labels |
| Output | `outputs/runs/phase1/weights/best.pt` |

#### Phase 2a — Detection Fine-tuning (50 epochs)

| Detail | Value |
|---|---|
| Data | All data incl. Part 3 + pseudo labels |
| Start | Phase 1 `best.pt` |
| Loss | Same as Phase 1 |
| Output | `outputs/runs/phase2/weights/best.pt` |

#### Phase 2b — Attribute Head Training (50 epochs, backbone frozen)

| Detail | Value |
|---|---|
| Model | `build_archon_base()` — Phase 2 weights + CoordConv injection |
| Backbone | Frozen + `.eval()` (BN uses running stats, matching inference) |
| Heads | `MultiAttributeHead` × 4 diseases × 3 FPN scales |
| Sampling | Per-tooth: FPN features sampled at `(cx × W_f, cy × H_f)` |
| Loss | `AttributeBCELoss` — BCE with auto `pos_weight` for imbalance |
| Optimizer | `AdamW(lr=1e-3)` → `CosineAnnealingLR` → `1e-5` |
| Output | `weights/attr_best.pt` |

### Total Loss

$$\mathcal{L} = 7.5 \cdot L_\text{bbox} + 0.5 \cdot L_\text{cls} + 1.5 \cdot L_\text{DFL} + 8.0 \cdot \sum_{d \in \text{diseases}} L_d$$

`data_type` column gates which terms fire:

| `data_type` | bbox | cls | DFL | attr |
|---|---|---|---|---|
| `0` — quadrant | ✓ | ✓ (grouped 4-class) | ✓ | ✗ |
| `1` — FDI | ✓ | ✓ (full 32-class) | ✓ | ✗ |
| `2` — disease | ✓ | ✓ (full 32-class) | ✓ | ✓ |

---

## 7. Mode: `train_attr`

```bash
python main.py --mode train_attr [--weights outputs/runs/phase2/weights/best.pt]
```

Runs only Phase 2b (attribute heads) in isolation. Use this to retrain disease heads without touching the detector, or when `attr_best.pt` is missing.

### Call Chain

```
main.py → stage_train_attr(args)
    └── ARCHONTrainer.train_attr_only(base_weights)
            └── _train_attribute_heads(base_weights)   [same as Phase 2b above]
```

---

## 8. Mode: `train_hybrid`

```bash
python main.py --mode train_hybrid [--weights path/to/phase2/best.pt] [--device cuda]
```

### Purpose

Trains three new components — GlobalContextEncoder (Swin), MultiScaleFusion (Cross-Attention), and HybridMultiTaskHead (Severity + Quadrant) — **on top of an already-trained Phase 2 model**, with the backbone + original attribute heads frozen. This is Phase 3.

### Prerequisites

- Phase 2 training complete: `outputs/runs/phase2/weights/best.pt` must exist
- 10-column extended labels: `data/processed/labels_ext/train/` must exist

### Call Chain

```
main.py → stage_train_hybrid(args)
    └── ARCHONHybridTrainer(config_path, resume=False, device)
            └── .train_hybrid(base_weights)
                    └── _train_hybrid_heads(base_weights)
                            ├── build_archon_model(base_weights)
                            │     yolortho.py
                            │     ├── loads YOLOv8x + CoordConv
                            │     ├── ARCHONModel(base_model, fpn_channels)
                            │     │     ├── GlobalContextEncoder   swin_transformer.py
                            │     │     ├── MultiScaleFusion       cross_attention.py
                            │     │     ├── HybridMultiTaskHead    hybrid_head.py
                            │     │     └── FPN hook registration
                            │     └── attaches feature hooks (layers -8, -5, -2)
                            │
                            ├── Freeze: backbone + attr_heads
                            │    model.base_model.model.eval()   (BN in eval mode)
                            │    requires_grad=False for backbone params
                            │
                            ├── Trainable: global_encoder + fusion + hybrid_head
                            │
                            ├── Optimizer: AdamW(lr=5e-4)
                            │   Scheduler: CosineAnnealingLR → 1e-6
                            │
                            ├── Loss:
                            │     AttributeBCELoss(weight=4.0)   binary attrs
                            │     SeverityLoss(weight=4.0)       3-class severity
                            │     QuadrantAuxLoss(weight=1.0)    4-class quadrant
                            │
                            ├── Per-tooth spatial sampling across P3, P4, P5
                            │
                            └── saves weights/hybrid_best.pt
                                   outputs/runs/phase2/weights/hybrid_best.pt
```

### What Each New Component Does

#### A. GlobalContextEncoder (`src/models/swin_transformer.py`)

```
P5 feature map (stride 32, ~20×40 for 640×1280 input)
         │
         ▼
┌──────────────────────────────────────────────────────┐
│  SwinTransformerBlock #1 (regular window attention)  │
│    window_size=4, num_heads=8                        │
│    Divides the 20×40 feature into 4×4 windows        │
│    Each window attends to itself (local context)     │
│    + relative position bias                          │
└─────────────────────────────┬────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────┐
│  SwinTransformerBlock #2 (shifted window attention)  │
│    shift_size = window_size // 2 = 2                 │
│    Windows shifted by (2,2) → cross-window context   │
│    Cyclic shift for efficient padding-free shifting  │
└─────────────────────────────┬────────────────────────┘
                              │ global context tensor
                              ▼
               Same shape as input P5 feature
```

**Why P5 only?** P5 has the coarsest resolution (stride 32), so each token represents a 32×32 pixel patch — roughly the size of one tooth on a panoramic X-ray. Swin attention at this level captures inter-tooth relationships (e.g., symmetric patterns, quadrant ordering, arch shape) without the memory cost of attending over P3 or P4.

**Window attention example:**

```
Feature map: 20 rows × 40 columns
window_size: 4 → 5 vertical × 10 horizontal windows
Each window: 4×4=16 tokens
Attention: 16×16 per window (vs 800×800 full attention)
Memory saved: ~50× vs full self-attention on P5
```

**Lazy block construction:** Block sizes depend on the actual input resolution (which varies with image size). Blocks are built on the first `forward()` call so no dummy tensors are needed at `__init__` time.

#### B. MultiScaleFusion (`src/models/cross_attention.py`)

```
Fused P3 = CrossAttentionFusion(CNN_P3, context_upsampled_to_P3)
Fused P4 = CrossAttentionFusion(CNN_P4, context)
Fused P5 = CrossAttentionFusion(CNN_P5, context_downsampled_to_P5)
```

**Inside CrossAttentionFusion:**

```
Q = Linear(CNN_feature.flatten(H×W))   ← local spatial detail
K = Linear(context.flatten(H×W))       ← global dental arch context
V = Linear(context.flatten(H×W))

Attention = softmax(QK^T / √d_k) · V

output = LayerNorm(CNN_feature + Attention)    ← residual
output = LayerNorm(output + FFN(output))       ← feed-forward residual
```

**Why cross-attention (Q from CNN, K/V from Swin)?**  
- CNN features are good at local texture (fine caries lines, lesion shadows)
- Swin features know which quadrant/arch-region each feature belongs to
- Cross-attention lets the CNN features *ask* the global context "which quadrant am I in, and what do neighboring teeth look like?" — improving FDI disambiguation and disease co-occurrence detection

**Positional encoding:** Learned 2D positional encoding (max 80×160 grid, interpolated to actual feature size) is added to Q and K so the model can reason about spatial layout.

#### C. HybridMultiTaskHead (`src/models/hybrid_head.py`)

Two parallel heads, both operating on fused FPN features:

**SeverityHead:**

```
Input: fused feature at tooth location (per-tooth sampled)
  │
  ▼
Conv2d(in_ch, 256, 3×3) → BN → ReLU
Conv2d(256, 256, 3×3) → BN → ReLU
Conv2d(256, num_attrs × 3, 1×1)   ← 4 diseases × 3 severity levels = 12 channels
  │
  ▼
Softmax over 3 severity classes per disease
→ P(Healthy) / P(Mild) / P(Severe)
```

**Binary → Severity mapping** (used during training to generate severity labels from the 10-col binary labels):

| Disease | Binary label | Severity mapping |
|---|---|---|
| is_impacted | 0/1 | 0→Healthy, 1→Severe |
| has_caries | 0/1 | 0→Healthy, 1→Mild (unless also deepcaries) |
| has_caries + has_deepcaries | 1+1 | →Severe |
| has_deepcaries only | 0+1 | →Severe |
| has_lesion | 0/1 | 0→Healthy, 1→Severe |

**QuadrantAwareFDIHead:**

```
Input: fused feature at tooth location
  │
  ▼
Conv2d(in_ch, 128, 3×3) → BN → ReLU
Conv2d(128, 4, 1×1)                ← 4 quadrant classes
  │
  ▼
Softmax → P(Q1) / P(Q2) / P(Q3) / P(Q4)
```

These quadrant probabilities are passed to the post-processor as `quadrant_probs`.

---

## 9. Mode: `evaluate`

```bash
python main.py --mode evaluate --weights weights/archon_best.pt
```

### Call Chain

```
main.py → stage_evaluate(args)
    └── ARCHONPredictor(weights, device, conf)
            └── .evaluate(data_yaml="config/dataset.yaml", split="val")
                    → ultralytics YOLO.val()
                    → mAP@0.5, mAP@0.5:0.95, AP per class
```

Dentex metrics:
- **AP-Quadrant** — detection accuracy per quadrant group
- **AP-Enumeration** — tooth-level identity accuracy (all 32 FDI classes)
- **AP-Diagnosis** — disease attribute classification accuracy

---

## 10. Mode: `predict`

```bash
python main.py --mode predict \
    --input data/processed/images/test/ \
    --weights weights/archon_best.pt \
    --conf 0.25
```

### Call Chain (Baseline predictor)

```
main.py → stage_predict(args)
    └── ARCHONPredictor(weights, device, conf)
            └── .predict(input_path, output_dir)
                    ├── _load_models()
                    │     ├── YOLO(weights_path)
                    │     ├── _load_attr_heads(attr_best.pt)
                    │     └── _attach_fpn_hooks()         ← intercept P3/P4/P5 tensors
                    │
                    └── For each image:
                          _predict_single(img_path)
                          ├── det_model.predict()         ← YOLO detection
                          ├── fpn_features populated by hooks
                          ├── _predict_attributes()       ← per-tooth attr head inference
                          └── postprocess_yolo_output()   ← linear sum assignment
                                  → List[ToothDetection]
                                  → save *_vis.jpg + *_result.json
```

### Call Chain (Hybrid predictor — when hybrid_best.pt is found)

```
ARCHONHybridPredictor._predict_single(img_path)
    ├── baseline YOLO forward
    ├── _load_hybrid_components()
    │     loads GlobalContextEncoder + MultiScaleFusion + HybridMultiTaskHead
    │     from hybrid_best.pt checkpoint keys
    ├── global_encoder(P5) → global_context
    ├── fusion(fpn_features, global_context) → fused_features
    ├── hybrid_head(fused_features) → {'severity': [...], 'quadrant': [...]}
    ├── _sample_per_tooth(scale_outputs, cx_lb, cy_lb, H, W, N)
    │     samples severity+quadrant at each detected tooth's location
    │     severity softmax → binary attr_probs = 1 - P(Healthy)
    │     quadrant softmax → quadrant_probs (4-vector per tooth)
    └── postprocess_yolo_output(..., quadrant_probs=quad_np)
              → quadrant-penalized linear sum assignment
              → ToothDetection + severity_details
```

---

## 11. Mode: `full`

```bash
python main.py --mode full --device cuda --dentex-root D:/path/to/DENTEX
```

Runs all stages in order:
1. `stage_preprocess()`
2. `stage_train()` — Phase 1 + 2 + 2b
3. `stage_pseudo_label()`
4. `stage_train()` — Phase 2 again with pseudo labels
5. `stage_evaluate()`

> **Note:** `train_hybrid` is NOT part of `full` — run it separately after `full` completes.

---

## 12. Key Component Deep-Dives

### CoordConv

**Problem:** Standard convolutions are translation-equivariant — they cannot reason about *where* something is. But tooth positions on a panoramic X-ray are highly correlated with their FDI number (upper-right teeth are always on the upper right of the image).

**Solution:** Before each convolution in backbone layers 0–9, append two extra channels:
- Channel C+0: normalized X coordinate (−1 at left edge, +1 at right edge)
- Channel C+1: normalized Y coordinate (−1 at top edge, +1 at bottom edge)

```python
# Example: 640×1280 feature map
x_coords = torch.linspace(-1, 1, W).expand(B, 1, H, W)  # shape (B,1,H,W)
y_coords = torch.linspace(-1, 1, H).unsqueeze(-1).expand(B, 1, H, W)
feat_with_coords = torch.cat([feat, x_coords, y_coords], dim=1)
# Now Conv2d sees: original feature channels + absolute position
```

**Effect:** The network can learn "if I see an incisor at x≈0.5 (center), it is likely FDI 11 or 21; if at x≈0.1 (left), likely FDI 18 or 28."

**Where used:** `src/models/coord_conv.py → CoordConv`, `replace_backbone_conv_with_coordconv()`  
**When injected:** Only during Phase 2b and Phase 3 (custom training loops), NOT during Phase 1/2 ultralytics `.train()`.

---

### FDI Numbering System

```
            Upper Jaw (Maxilla)
  Q2 (left)               Q1 (right)
  28 27 26 25 24 23 22 21 | 11 12 13 14 15 16 17 18
─────────────────────────────────────────────────────  patient's view
  38 37 36 35 34 33 32 31 | 41 42 43 44 45 46 47 48
  Q3 (left)               Q4 (right)
            Lower Jaw (Mandible)
```

**Formula:** `class_id = (quadrant − 1) × 8 + (position − 1)`

| FDI | class_id | tooth |
|---|---|---|
| 11 | 0 | Upper-Right Central Incisor |
| 18 | 7 | Upper-Right Wisdom |
| 21 | 8 | Upper-Left Central Incisor |
| 36 | 20 | Lower-Left First Molar |
| 48 | 31 | Lower-Right Wisdom |

**Files:** `src/utils/fdi.py` — `FDI_TO_CLASS`, `CLASS_TO_FDI`, `fdi_to_class()`, `quadrant_enum_to_fdi()`

---

### Hierarchical Loss and data_type

Three DENTEX dataset parts have different annotation depth:

| Part | data_type | What is labeled | Loss active |
|---|---|---|---|
| Part 1 | 0 | Quadrant (1–4) | bbox + 4-class quadrant cls |
| Part 2 | 1 | FDI number (11–48) | bbox + 32-class FDI cls |
| Part 3 | 2 | FDI + diseases | bbox + 32-class + 4× attr BCE |
| Pseudo | 2 | Healthy teeth (synthetic) | same as Part 3 |

The `data_type` column in every label row controls which loss terms fire. Rows with `data_type=0` never compute attribute loss, preventing healthy (unlabeled) teeth from being mis-counted as "no disease."

---

### Modified FPN and Strides

Standard YOLOv8 FPN produces heads at strides [8, 16, 32]. ARCHON adds an extra upsample layer to produce stride 4:

```
P5 (stride 32, 20×40) → Upsample → concat P4 → P4_new (stride 16, 40×80)
P4_new               → Upsample → concat P3 → P3_new (stride 8, 80×160)
P3_new               → Upsample → concat P2 → N2     (stride 4, 160×320) ← new
```

Stride 4 gives a 160×320 feature map for the 640×1280 input — high resolution for detecting small or overlapping teeth near midline, where standard FPN at stride 8 can miss precise bounding boxes.

---

### Attribute Heads (Binary Disease Heads)

**File:** `src/models/heads.py → AttributeHead`, `MultiAttributeHead`

One `AttributeHead` per (disease, FPN scale) = 4 diseases × 3 scales = 12 heads.

```
FPN feature (B, C, H_s, W_s)
    │
    ▼
Conv2d(C→256, 3×3) → BN → ReLU
Conv2d(256→256, 3×3) → BN → ReLU
Conv2d(256→1, 1×1)
    │
    ▼
Sigmoid → per-pixel binary probability
```

At inference, the probability at the predicted tooth's center pixel `(cx*W_s, cy*H_s)` is taken as the disease probability. This "per-tooth spatial sampling" was the key fix that made disease prediction work correctly — earlier versions used global average pooling over the entire feature map, which diluted tooth-specific signals.

---

### GlobalContextEncoder — Swin Transformer

**File:** `src/models/swin_transformer.py`

**Key classes:**
- `WindowAttention(dim, window_size, num_heads)` — local window-based multi-head self-attention with learned relative position bias table
- `SwinTransformerBlock(dim, input_resolution, num_heads, window_size, shift_size)` — full block: LayerNorm → W-MSA or SW-MSA → residual → LayerNorm → MLP → residual
- `GlobalContextEncoder(in_channels=640, num_heads=8, window_size=4)` — takes P5 feature, runs 2 Swin blocks, projects back to original channel dim

**What shift_size=0 vs shift_size>0 does:**

```
shift_size=0 (Block 1):  standard 4×4 windows, no overlap
  [0][1][2][3]
  [4][5][6][7]       each cell attends only to its 4×4 block

shift_size=2 (Block 2):  windows shifted by (2,2)
  [─  0  1  ─]
  [2  3  4  5]       now cells at block boundaries can attend
  [─  6  7  ─]       to each other → global-ish context
```

**Cyclic shift:** Instead of padding for shifted windows, the feature map is cyclically rolled before partitioning and unrolled after — no wasted computation on padding.

---

### MultiScaleFusion — Cross-Attention

**File:** `src/models/cross_attention.py`

**Key classes:**
- `CrossAttentionFusion(cnn_channels, ctx_channels, num_heads)` — fuses one CNN scale with global context
- `MultiScaleFusion(fpn_channels, ctx_channels, num_heads)` — applies fusion at all 3 FPN scales

**Positional encoding detail:**
A learned 2D positional encoding of shape `(1, max_H, max_W, embed_dim)` is stored as a parameter. At forward time it is bicubically interpolated to match the actual feature size. This lets the model generalize to different input resolutions without retraining.

**Residual structure:**

```
out = LN(cnn_feat + CrossAttn(Q=cnn_feat, K=context, V=context))
out = LN(out + FFN(out))
```

Both residuals preserve the original CNN information, so if the cross-attention does not improve predictions, the network can learn to ignore it (attention weights → 0, identity pass-through).

---

### HybridMultiTaskHead — Severity + Quadrant

**File:** `src/models/hybrid_head.py`

**Severity levels (3-class):**

| Level | Label | Trigger |
|---|---|---|
| 0 | Healthy | no disease flag set |
| 1 | Mild | early caries (has_caries=1, has_deepcaries=0) |
| 2 | Severe | impacted, deep caries, periapical lesion, or advanced caries |

**Biased initialization:** The final linear layer bias is initialized to `[+2.0, 0.0, -2.0]` so before any learning the model strongly predicts "Healthy." This accelerates training on the imbalanced DENTEX dataset where healthy teeth dominate.

**QuadrantAwareFDIHead:** Predicts which quadrant each tooth belongs to (4-class). During inference, these 4-probabilities are passed to `apply_linear_sum_assignment()` as `quadrant_probs`, adding a quadrant-consistency penalty to the cost matrix.

---

### CLAHE Augmentation

**File:** `src/data/augmentation.py → _augment_clahe()`

Contrast Limited Adaptive Histogram Equalization — applied stochastically (p=0.5) during Phase 3 training.

**Why CLAHE for dental X-rays?**
- Panoramic X-rays have **non-uniform exposure**: bone-dense regions (molars) appear very bright; soft-tissue gaps (interdental spaces) very dark
- CLAHE equalizes local contrast in tiles (8×8 by default), making early caries lines visible in high-brightness enamel regions without blowing out dark shadow regions
- Applied on the L-channel of LAB color space (brightness only) to preserve hue — avoids color artifacts on gray-scale X-rays stored as 3-channel BGR

```
BGR image
   │
   ▼ cv2.cvtColor(BGR→LAB)
L, A, B channels
   │
   ▼ CLAHE on L only  (clipLimit=2.0, tileGridSize=8×8)
L_enhanced, A, B
   │
   ▼ cv2.cvtColor(LAB→BGR)
Enhanced BGR
```

**clip_limit=2.0:** Tiles where contrast gain would exceed 2× are clipped and redistributed — prevents artificial amplification of noise in uniform regions.

---

### Quadrant-Consistency Post-processing

**File:** `src/inference/postprocess.py → apply_linear_sum_assignment()`

After hybrid inference, each detected tooth has a `quadrant_probs` vector `[P(Q1), P(Q2), P(Q3), P(Q4)]`. This is used to add a quadrant-consistency penalty to the FDI cost matrix:

```python
slot_quadrants = np.arange(32) // 8   # [0,0,...,0, 1,1,...,1, 2,2,...,2, 3,3,...,3]
quad_match_prob = quadrant_probs[:, slot_quadrants].T   # (32, N_detections)
quad_penalty = ALPHA * (1.0 - quad_match_prob)          # low cost = high match probability
cost_matrix = cost_matrix + quad_penalty                # ALPHA=0.5
```

**Effect:** A detection the quadrant head says "likely Q2 (upper-left)" becomes more expensive to assign to FDI slots in Q1, Q3, Q4. The Hungarian algorithm still assigns each tooth to exactly one FDI, but now prefers quadrant-consistent assignments. This reduces the most common misidentification error: swapping symmetric tooth pairs across the midline (e.g., assigning tooth 16 to slot 26 because their size/shape are similar).

---

### Linear Sum Assignment

**File:** `src/inference/postprocess.py → apply_linear_sum_assignment()`

After YOLO NMS, we have N detections each with a softmax probability over 32 FDI classes. The goal: assign each detection to a unique FDI slot (real patients have at most one tooth #36).

```python
cost_matrix[slot, det] = 1.0 - prob[det, slot]
row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)
```

`linear_sum_assignment` solves the Hungarian algorithm — it finds the globally optimal 1-to-1 assignment minimizing total cost. A detection with high P(36) and moderate P(46) will be assigned to 36 unless another detection has even higher P(36), in which case it gets 46.

**Quadrant penalty** (hybrid mode only): adds `ALPHA × (1 - quadrant_match_prob)` to the cost before the Hungarian step.

---

### Pseudo Labels

**File:** `src/data/pseudo_label.py → generate_pseudo_labels()`

Two sources:
1. **Part 3 images** — disease labels exist but healthy teeth are unlabeled. Run Phase 1 detector; any detection with IoU < 0.3 with existing disease boxes is a candidate healthy tooth.
2. **Unlabelled images** — no labels at all. Every confident detection becomes a healthy pseudo label.

Pseudo label format: same 10-column format with all disease flags = 0, data_type = 2.

```
class_id  cx  cy  w  h  0  0  0  0  2
```

---

### Flip Augmentation with Quadrant Remapping

**File:** `src/data/augmentation.py → _augment_fliplr()`  
**Table:** `src/utils/fdi.py → FLIP_CLASS_TABLE`

When a panoramic X-ray is horizontally flipped:
- Q1 (upper-right) ↔ Q2 (upper-left)
- Q4 (lower-right) ↔ Q3 (lower-left)

All class labels must be remapped. Example: tooth class 5 (FDI 16, upper-right 6th tooth) → class 13 (FDI 26, upper-left 6th tooth).

`FLIP_CLASS_TABLE` is a precomputed list of length 32:  
`FLIP_CLASS_TABLE[old_class] = new_class_after_flip`

---

## 13. Label Format Reference

**10-column extended YOLO format:**

| Col | Name | Range | Description |
|---|---|---|---|
| 0 | `class_id` | 0–31 | YOLO class (maps to FDI via `CLASS_TO_FDI`) |
| 1 | `cx` | 0–1 | Bounding box center X (normalized) |
| 2 | `cy` | 0–1 | Bounding box center Y (normalized) |
| 3 | `w` | 0–1 | Bounding box width (normalized) |
| 4 | `h` | 0–1 | Bounding box height (normalized) |
| 5 | `is_impacted` | 0/1 | Tooth is impacted / embedded |
| 6 | `has_caries` | 0/1 | Dental caries present |
| 7 | `has_deepcaries` | 0/1 | Deep caries (pulp involvement) |
| 8 | `has_lesion` | 0/1 | Periapical lesion |
| 9 | `data_type` | 0/1/2 | Annotation depth (see §6 Hierarchical Loss) |

> Columns 5–8 are only non-zero when `data_type=2`.

---

## 14. Config Files Reference

### `config/model_config.yaml`

| Key | Default | Purpose |
|---|---|---|
| `model_size` | `x` | YOLOv8 variant (n/s/m/l/x) |
| `use_coordconv` | `true` | Inject CoordConv into backbone layers 0–9 |
| `fpn_strides` | `[4,8,16]` | FPN output strides (extra stride-4 added) |
| `num_classes` | `32` | Number of FDI tooth classes |
| `conf_threshold` | `0.25` | Detection confidence threshold |
| `iou_threshold` | `0.45` | NMS IoU threshold |
| `use_hybrid` | `false` | Set `true` to enable hybrid Swin+Cross-Attn |
| `swin_num_heads` | `8` | Number of attention heads in Swin blocks |
| `swin_window_size` | `4` | Window size for Swin local attention |
| `fusion_num_heads` | `4` | Heads in cross-attention fusion |
| `num_severity_levels` | `3` | Severity levels (Healthy/Mild/Severe) |
| `severity_threshold` | `0.4` | Min P(not Healthy) to report disease |
| `quadrant_penalty_alpha` | `0.5` | Quadrant consistency penalty strength |

### `config/train_config.yaml`

| Key | Default | Purpose |
|---|---|---|
| `phase1_epochs` | `100` | Detection pre-training epochs |
| `phase2_epochs` | `50` | Full detection fine-tuning epochs |
| `phase2b_epochs` | `50` | Binary attribute head training epochs |
| `phase3_epochs` | `50` | Hybrid head training epochs (NEW) |
| `batch_size` | `8` | Training batch size |
| `lr0` | `0.01` | Initial learning rate (detection phases) |
| `loss_attr` | `8.0` | Binary attribute BCE loss weight |
| `fliplr` | `0.5` | Horizontal flip augmentation probability |
| `mosaic` | `0.0` | Mosaic augmentation (disabled for X-rays) |
| `clahe_prob` | `0.5` | CLAHE augmentation probability (Phase 3) |
| `pseudo_label_conf` | `0.5` | Min confidence for pseudo labels |

### `config/dataset.yaml`

Controls YOLO dataset paths and class names. 32 class names are the 32 FDI codes in order (11, 12, ..., 18, 21, ..., 48).

---

## 15. Data Directory Layout

```
data/
├── processed/
│   ├── images/
│   │   ├── train/         ← all training X-rays (parts 1+2+3)
│   │   ├── val/           ← official validation X-rays
│   │   └── test/          ← test X-rays (no labels)
│   ├── labels/
│   │   ├── train/         ← 5-col YOLO labels (detection training)
│   │   └── val/
│   └── labels_ext/
│       ├── train/         ← 10-col extended labels (attr head training)
│       └── val/
├── pseudo/
│   ├── images/train/      ← symlinked / copied from processed/
│   └── labels_ext/train/  ← extended labels + pseudo healthy rows
└── unlabelled/            ← unlabelled X-rays (for pseudo labeling)
```

---

## 16. Output Files Reference

### After Training

```
weights/
├── archon_phase1.pt       ← Phase 1 detection checkpoint
├── archon_best.pt         ← Best Phase 2 detection checkpoint
├── attr_best.pt             ← Phase 2b attribute head checkpoint
└── hybrid_best.pt           ← Phase 3 hybrid component checkpoint (NEW)

outputs/runs/
├── phase1/weights/          ← ultralytics training outputs
├── phase2/weights/
│   ├── best.pt
│   ├── attr_best.pt
│   └── hybrid_best.pt       ← also copied here (NEW)
└── phase2/results.csv       ← per-epoch metrics
```

### After Inference

```
outputs/predictions/
├── <name>_vis.jpg           ← annotated X-ray with bboxes + disease flags
└── <name>_result.json       ← per-tooth structured output
    {
      "image": "sample.jpg",
      "detections": [
        {
          "fdi": 36,
          "bbox": [x1, y1, x2, y2],
          "confidence": 0.91,
          "is_impacted": false,
          "has_caries": true,
          "has_deepcaries": false,
          "has_lesion": false
          // severity_details added when hybrid model is active (NEW)
        }
      ]
    }
```

---

## 17. Full Pipeline Call Graph

```
main.py
  │
  ├─ preprocess ────────────────────────────────────────────────────────────────────────
  │   stage_preprocess()
  │     preprocess.preprocess_dentex()
  │       fdi.quadrant_enum_to_fdi()      part2+3 annotations → FDI number
  │       fdi.fdi_to_class()              FDI → YOLO class index
  │       _merge_disease_annotations()   multi-row per tooth → one row with flags
  │       write  data/processed/labels_ext/
  │
  ├─ pseudo_label ──────────────────────────────────────────────────────────────────────
  │   stage_pseudo_label()
  │     pseudo_label.generate_pseudo_labels()
  │       YOLO(phase1_best).predict()    tooth detections
  │       IoU filter < 0.3              remove boxes near existing disease labels
  │       write  data/pseudo/labels_ext/
  │
  ├─ train ─────────────────────────────────────────────────────────────────────────────
  │   stage_train()
  │     ARCHONTrainer.train()
  │       _train_phase1()
  │         YOLO("yolov8x.pt").train()        → phase1/weights/best.pt
  │       _train_phase2(phase1_best)
  │         YOLO(phase1_best).train()          → phase2/weights/best.pt
  │       _train_attribute_heads(phase2_best)
  │         yolortho.build_archon_base()
  │           coord_conv.replace_backbone_conv_with_coordconv()
  │           heads.MultiAttributeHead()
  │           yolortho.ARCHON.attach_feature_hooks()
  │         dataset.ARCHONDataset(labels_ext/)
  │         loss.AttributeBCELoss(pos_weight)
  │         for epoch:
  │           for batch:
  │             model.base_model(x) → FPN hooks capture P3,P4,P5
  │             per-tooth spatial sampling  (cx*W_f, cy*H_f)
  │             attr_heads(features_at_tooth) → disease probs
  │             AttributeBCELoss(preds, labels[:,5:9])
  │             loss.backward() ; optimizer.step()
  │         save  weights/attr_best.pt
  │
  ├─ train_attr ────────────────────────────────────────────────────────────────────────
  │   stage_train_attr()
  │     ARCHONTrainer.train_attr_only()
  │       _train_attribute_heads(base_weights)    [same as above]
  │
  ├─ train_hybrid ─────────────────────────────────────────────────────────────────────
  │   stage_train_hybrid()
  │     ARCHONHybridTrainer.train_hybrid(base_weights)
  │       _train_hybrid_heads(base_weights)
  │         yolortho.build_archon_model()
  │           loads YOLOv8x + CoordConv
  │           swin_transformer.GlobalContextEncoder()
  │           cross_attention.MultiScaleFusion()
  │           hybrid_head.HybridMultiTaskHead()
  │           ARCHONModel.attach_feature_hooks()
  │         freeze backbone + attr_heads
  │         AdamW(lr=5e-4) on global_encoder + fusion + hybrid_head
  │         CosineAnnealingLR → 1e-6
  │         augmentation.ARCHONAugmentor(clahe_prob=0.5)   ← CLAHE active here
  │         for epoch:
  │           for batch:
  │             model(x) → {'det_output', 'severity', 'quadrant', 'fpn_fused'}
  │             per-tooth spatial sampling at all 3 FPN scales
  │             loss = AttributeBCELoss + SeverityLoss + QuadrantAuxLoss
  │             loss.backward() ; optimizer.step()
  │         save  weights/hybrid_best.pt
  │
  ├─ evaluate ─────────────────────────────────────────────────────────────────────────
  │   stage_evaluate()
  │     ARCHONPredictor.evaluate()
  │       YOLO(weights).val(data="config/dataset.yaml")
  │       → AP-Quadrant / AP-Enumeration / AP-Diagnosis
  │
  └─ predict ──────────────────────────────────────────────────────────────────────────
      stage_predict()
        ARCHONHybridPredictor (if hybrid_best.pt found) else ARCHONPredictor
          _load_models()
            YOLO(archon_best.pt)
            _load_attr_heads(attr_best.pt)
            _load_hybrid_components(hybrid_best.pt)   ← loads swin+cross+hybrid_head
          for image:
            _predict_single(img)
              YOLO.predict() → raw detections
              FPN hooks → P3, P4, P5 tensors
              global_encoder(P5) → context
              fusion([P3,P4,P5], context) → fused_features
              hybrid_head(fused_features) → severity + quadrant per pixel
              _sample_per_tooth() → per-tooth severity + quadrant vectors
              postprocess_yolo_output(dets, attr_probs, quadrant_probs)
                apply_linear_sum_assignment(cost_matrix + quad_penalty)
              → List[ToothDetection]
            visualize.draw_teeth_detections() → *_vis.jpg
            json.dump(detections) → *_result.json
```
