# ARCHON — Project Guide

> **ARCHON: Arch-Contextualized Hierarchical Orthodontic Network for Severity-Aware Dental Radiograph Analysis**
>
> Built on: **YOLOrtho: A Unified Framework for Teeth Enumeration and Dental Disease Detection**
> arXiv: https://arxiv.org/abs/2308.05967
>
> Extended with Swin Transformer + Cross-Attention hybrid improvements.

| Document | Purpose |
|---|---|
| `PROJECT_GUIDE.md` | Code walkthrough, JSON schemas, label format (this file) |
| `WORKFLOW.md` | End-to-end call graph for every mode |
| `IMPROVEMENTS.md` | What changed from baseline and why |
| `README.md` | Architecture overview and quick start |

---

## Table of Contents

1. [Project Structure](#1-project-structure)
2. [What Each File Does](#2-what-each-file-does)
3. [Paper → Code Mapping](#3-paper--code-mapping)
4. [Step-by-Step Execution](#4-step-by-step-execution)
5. [Label Format Explained](#5-label-format-explained)
6. [FDI Numbering System](#6-fdi-numbering-system)
7. [Training Phases Explained](#7-training-phases-explained)
8. [Key Hyperparameters](#8-key-hyperparameters)
9. [Output Files](#9-output-files)
10. [Common Issues & Fixes](#10-common-issues--fixes)
11. [Minimum Requirements](#11-minimum-requirements)

---

## 1. Project Structure

```
ARCHON/
│
├── main.py                          ← SINGLE ENTRY POINT — run everything from here
├── requirements.txt                 ← All Python dependencies
├── PROJECT_GUIDE.md                 ← This file
├── README.md                        ← Architecture & paper summary
│
├── config/                          ← All configuration files (edit here, not in code)
│   ├── model_config.yaml            ← Model architecture: CoordConv, FPN strides, heads
│   ├── dataset.yaml                 ← Dataset paths + 32 FDI class names for YOLO
│   └── train_config.yaml            ← Epochs, learning rate, loss weights, augmentation
│
├── data/                            ← Auto-created by the preprocess stage
│   ├── processed/                   ← YOLO-format converted dataset
│   │   ├── images/
│   │   │   ├── train/               ← Training images (from all 3 training parts)
│   │   │   ├── val/                 ← Validation images (from validation_triple.json)
│   │   │   └── test/                ← Test images (from test_data/disease/input/)
│   │   └── labels/
│   │       ├── train/               ← Extended 10-column YOLO label .txt files
│   │       ├── val/                 ← Extended 10-column YOLO label .txt files
│   │       └── test/                ← Empty (test has no labels)
│   ├── pseudo/                      ← Pseudo-labeled merged dataset (Phase 2)
│   │   ├── images/train/
│   │   └── labels/train/
│   └── unlabelled/                  ← Unlabelled xrays (copied by preprocess from training_data/unlabelled/xrays/)
│
├── src/                             ← All source code (modular, separated by function)
│   ├── data/
│   │   ├── __init__.py
│   │   ├── preprocess.py            ← Converts COCO JSON → extended YOLO .txt labels (all parts + val)
│   │   ├── augmentation.py          ← Custom augmentations (quadrant-aware flip)
│   │   ├── pseudo_label.py          ← Healthy-tooth pseudo labels (Part 3 + unlabelled images)
│   │   └── dataset.py               ← PyTorch Dataset class for training
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── coord_conv.py            ← CoordConv: adds (x,y) coordinate channels
│   │   ├── heads.py                 ← 4 binary disease attribute prediction heads
    │   ├── yolortho.py              ← `ARCHON` base model + `ARCHONModel` (hybrid) wrappers
    │   ├── swin_transformer.py      ← GlobalContextEncoder (Swin Transformer) [NEW]
    │   ├── cross_attention.py       ← MultiScaleFusion (Cross-Attention)      [NEW]
    │   └── hybrid_head.py           ← HybridMultiTaskHead (Severity+Quadrant)  [NEW]
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py               ← ARCHONTrainer + ARCHONHybridTrainer
│   │   └── loss.py                  ← Hierarchical loss (bbox + class + attribute BCE)
│   │
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── predictor.py             ← ARCHONPredictor + ARCHONHybridPredictor  [+Hybrid]
│   │   └── postprocess.py           ← Linear sum assignment + quadrant-consistency penalty  [+Penalty]
│   │
│   └── utils/
│       ├── __init__.py
│       ├── fdi.py                   ← FDI number ↔ YOLO class index conversions
│       └── visualize.py             ← Draw bounding boxes + dental chart overlay
│
├── weights/                         ← Saved model checkpoints (.pt files)
│   ├── attr_best.pt                 ← Phase 2b binary attribute heads
│   ├── archon_best.pt             ← Phase 2 detection model
│   └── hybrid_best.pt               ← Phase 3 Swin+Cross-Attn+Severity heads  [NEW]
│
└── outputs/
    ├── runs/                        ← Ultralytics training logs + TensorBoard events
    ├── predictions/                 ← Inference output: *_vis.jpg + *_result.json
    └── archon.log                 ← Main pipeline log file
```

---

## 2. What Each File Does

### Entry Point

| File | Purpose |
|---|---|
| `main.py` | Parses `--mode` argument and dispatches to correct pipeline stage. Always run this. |

### Config Files (`config/`)

| File | What to Edit |
|---|---|
| `model_config.yaml` | Change model size, CoordConv on/off, FPN strides, confidence threshold |
| `dataset.yaml` | Change dataset path (if you move data), class names (don't change unless dataset changes) |
| `train_config.yaml` | Tune epochs, batch size, learning rate, loss weights |

### Data Pipeline (`src/data/`)

| File | Input | Output |
|---|---|---|
| `preprocess.py` | DENTEX folder (`--dentex-root`) | `data/processed/` — YOLO .txt labels (10 cols) + `data/unlabelled/` |
| `augmentation.py` | Image + label tensors | Augmented image + remapped labels |
| `pseudo_label.py` | Trained Phase 1 model + Part 3 images + unlabelled images | Pseudo label .txt files in `data/pseudo/` |
| `dataset.py` | `data/processed/` or `data/pseudo/` | PyTorch DataLoader-compatible items |

### Models (`src/models/`)

| File | Key Class/Function | What It Does |
|---|---|---|
| `coord_conv.py` | `CoordConv`, `replace_backbone_conv_with_coordconv()` | Appends normalized (x,y) maps to feature maps before convolution |
| `heads.py` | `MultiAttributeHead`, `AttributeLoss` | 4 binary disease heads, one per FPN scale × disease type |
| `yolortho.py` | `ARCHON`, `build_archon_base()` | Loads YOLOv8x, injects CoordConv, attaches attribute heads via hooks |
| `yolortho.py` | `ARCHONModel`, `build_archon_model()` | Wraps baseline with GlobalContextEncoder + MultiScaleFusion + HybridMultiTaskHead **[NEW]** |
| `swin_transformer.py` | `GlobalContextEncoder`, `SwinTransformerBlock`, `WindowAttention` | 2-block Swin Transformer on P5 feature — captures full dental arch context **[NEW]** |
| `cross_attention.py` | `CrossAttentionFusion`, `MultiScaleFusion` | Cross-attention Q=CNN, K/V=Swin; applied at all 3 FPN scales **[NEW]** |
| `hybrid_head.py` | `HybridMultiTaskHead`, `SeverityHead`, `QuadrantAwareFDIHead`, `SeverityLoss`, `QuadrantAuxLoss` | 3-class severity per disease + 4-class quadrant auxiliary **[NEW]** |

### Training (`src/training/`)

| File | Key Class | What It Does |
|---|---|---|
| `trainer.py` | `ARCHONTrainer` | Runs Phase 1 (detection), pseudo-labeling, Phase 2 (full), attribute fine-tune |
| `trainer.py` | `ARCHONHybridTrainer` | Phase 3: trains GlobalContextEncoder + MultiScaleFusion + HybridMultiTaskHead **[NEW]** |
| `loss.py` | `ARCHONLoss` | Combines detection loss + hierarchical class loss + attribute BCE; masks by `data_type` |

### Inference (`src/inference/`)

| File | Key Class | What It Does |
|---|---|---|
| `predictor.py` | `ARCHONPredictor` | Loads model, runs YOLO inference, applies attribute heads, saves results |
| `predictor.py` | `ARCHONHybridPredictor` | Adds Swin+CrossAttn+SeverityHead inference; produces severity-annotated detections **[NEW]** |
| `postprocess.py` | `apply_linear_sum_assignment()` | Ensures each FDI number appears at most once using Hungarian algorithm; accepts optional `quadrant_probs` penalty **[+Penalty]** |

### Utilities (`src/utils/`)

| File | Key Contents | What It Does |
|---|---|---|
| `fdi.py` | `FDI_TO_CLASS`, `FLIP_CLASS_TABLE`, `fdi_to_class()` | Bidirectional FDI↔class index conversion; pre-computed flip remapping table |
| `visualize.py` | `draw_teeth_detections()`, `draw_dental_chart()` | Overlays colored bounding boxes + disease flags + dental odontogram chart |

---

## 3. Dataset Structure & JSON Schemas

### On-Disk Layout (`--dentex-root`)

```
DENTEXsample/
├── training_data/
│   ├── quadrant/                         ← Part 1
│   │   ├── train_quadrant.json
│   │   └── xrays/  (*.png)
│   ├── quadrant_enumeration/             ← Part 2  (underscore)
│   │   ├── train_quadrant_enumeration.json
│   │   └── xrays/  (*.png)
│   ├── quadrant-enumeration-disease/     ← Part 3  (hyphen!)
│   │   ├── train_quadrant_enumeration_disease.json
│   │   └── xrays/  (*.png)
│   └── unlabelled/
│       └── xrays/  (*.png)              ← No annotations; used for pseudo labeling
├── validation_data/
│   ├── validation_triple.json
│   └── quadrant_enumeration_disease/
│       └── xrays/  (*.png)
└── test_data/
    └── disease/
        ├── input/   (*.png)             ← Test images (no labels used in training)
        └── label/   (*.json)            ← Per-image LabelMe JSON (not used by this code)
```

> **Note:** `quadrant-enumeration-disease` uses a **hyphen**, while `quadrant_enumeration` and `quadrant_enumeration_disease` (JSON filename) use **underscores**. This is how the original DENTEX dataset is structured.

---

### JSON Schema: Part 1 — `train_quadrant.json`

Top-level keys: `images`, `annotations`, `categories`

**`categories`** — quadrant labels (id ≠ quadrant−1; use name lookup):
```json
[{"id": 0, "name": "2"}, {"id": 1, "name": "1"}, {"id": 2, "name": "3"}, {"id": 3, "name": "4"}]
```

**`annotations[i]`**:
```json
{"id": 1, "image_id": 5, "category_id": 1, "bbox": [x, y, w, h], "area": 0, "iscrowd": 0, "segmentation": []}
```

→ `cat_to_quadrant = {cat["id"]: int(cat["name"]) for cat in coco["categories"]}` in code.

---

### JSON Schema: Part 2 — `train_quadrant_enumeration.json`

Top-level keys: `images`, `annotations`, `categories_1`, `categories_2`

**`categories_1`** — quadrant (1-4):
```json
[{"id": 0, "name": 1}, {"id": 1, "name": 2}, {"id": 2, "name": 3}, {"id": 3, "name": 4}]
```
**`categories_2`** — tooth position within quadrant (1-8):
```json
[{"id": 0, "name": "1"}, {"id": 1, "name": "2"}, ..., {"id": 7, "name": "8"}]
```

**`annotations[i]`**:
```json
{"id": 1, "image_id": 5, "category_id_1": 0, "category_id_2": 3, "bbox": [x, y, w, h], "area": 0, "iscrowd": 0, "segmentation": []}
```

→ FDI = `(quadrant × 10) + position`.

---

### JSON Schema: Part 3 — `train_quadrant_enumeration_disease.json` & `validation_triple.json`

Top-level keys: `images`, `annotations`, `categories_1`, `categories_2`, `categories_3`

**`categories_3`** — disease types:
```json
[{"id": 0, "name": "Impacted"}, {"id": 1, "name": "Caries"}, {"id": 2, "name": "Periapical Lesion"}, {"id": 3, "name": "Deep Caries"}]
```

**`annotations[i]`**:
```json
{"id": 1, "image_id": 5, "category_id_1": 0, "category_id_2": 3, "category_id_3": 1, "bbox": [x, y, w, h], "area": 0, "iscrowd": 0, "segmentation": []}
```

> **Important:** Each annotation row = **one disease for one tooth**. A tooth with multiple diseases has **multiple rows** with the same `(image_id, category_id_1, category_id_2)`. The preprocess code **merges** these by (image_id, FDI) into a single label row with multiple attribute flags set.

---

### Disease → Attribute Column Mapping

| `categories_3.id` | Disease Name | YOLO label column | Attribute flag |
|---|---|---|---|
| 0 | Impacted | col 5 | `is_impacted` |
| 1 | Caries | col 6 | `has_caries` |
| 2 | Periapical Lesion | col 8 | `has_lesion` |
| 3 | Deep Caries | col 7 | `has_deepcaries` |

```python
DISEASE_CAT3_TO_ATTR_IDX = {0: 0, 1: 1, 2: 3, 3: 2}   # cat3 id → attrs list index
```

---

## 4. Paper → Code Mapping

| Paper Contribution | Section | File | Key Symbol |
|---|---|---|---|
| Coordinate Convolution | §2.2 | `src/models/coord_conv.py` | `CoordConv`, `replace_backbone_conv_with_coordconv()` |
| Modified FPN (strides 4,8,16) | §2.1 | `src/models/yolortho.py` | `build_archon_base()`, `config/model_config.yaml` → `fpn_strides` |
| Disease attribute heads | §2.2 | `src/models/heads.py` | `MultiAttributeHead`, `AttributeHead` |
| Hierarchical loss | §2.2 | `src/training/loss.py` | `ARCHONLoss`, `data_type` column masking |
| Quadrant-aware flip augmentation | §2.1 | `src/data/augmentation.py` | `_augment_fliplr()`, `FLIP_CLASS_TABLE` |
| Two-phase training | §2.1 | `src/training/trainer.py` | `_train_phase1()`, `_train_phase2()`, `_train_attribute_heads()` |
| Pseudo-labeling healthy teeth | §2.1 | `src/data/pseudo_label.py` | `generate_pseudo_labels()` |
| Linear sum assignment post-proc | §2.3 | `src/inference/postprocess.py` | `apply_linear_sum_assignment()` |
| FDI tooth numbering | §2.1 | `src/utils/fdi.py` | `FDI_TO_CLASS`, `CLASS_TO_FDI`, `quadrant_enum_to_fdi()` |
| Multi-label disease output | §2.2 | `src/inference/predictor.py` | `_predict_attributes()`, `ToothDetection` |

### Architecture Diagram (as implemented)

```
Input X-ray (1280 × 640)
        │
        ▼
┌─────────────────────────────────────────┐
│  YOLOv8x Backbone                       │
│  + CoordConv (first 10 layers patched)  │  ← coord_conv.py
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│  Modified PANet FPN                     │
│  Extra upsampling → strides [4, 8, 16] │  ← yolortho.py / model_config.yaml
└─────────────────────────────────────────┘
        │
        ├──────────────────┬──────────────────┐
        ▼                  ▼                  ▼
  [stride 4]          [stride 8]         [stride 16]
  Large teeth         Medium teeth       Small teeth
        │                  │                  │
        ▼                  ▼                  ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│ Detection     │  │ Detection     │  │ Detection     │  ← YOLOv8 Detect head
│ Head (32 cls) │  │ Head (32 cls) │  │ Head (32 cls) │
└───────────────┘  └───────────────┘  └───────────────┘
        │                  │                  │
        ▼                  ▼                  ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│ Attr Head ×4  │  │ Attr Head ×4  │  │ Attr Head ×4  │  ← heads.py
│ (4 diseases)  │  │ (4 diseases)  │  │ (4 diseases)  │
└───────────────┘  └───────────────┘  └───────────────┘
        │
        ▼
Linear Sum Assignment Post-processing   ← postprocess.py
(Hungarian algorithm: 1 FDI per tooth)
        │
        ▼
Output: List of ToothDetection objects
  - fdi: int (e.g. 36)
  - bbox: [x1, y1, x2, y2]
  - confidence: float
  - is_impacted: bool
  - has_caries: bool
  - has_deepcaries: bool
  - has_lesion: bool
```

---

## 4. Step-by-Step Execution

### Prerequisites

```bash
# 1. Install all dependencies
pip install -r requirements.txt

# 2. Point to the DENTEX dataset folder on disk
#    No download needed — the dataset is already available locally.
#    Pass --dentex-root to any stage that needs raw data.
#    Example:  --dentex-root D:/WILP/sem-4/Dataset/DENTEX/DENTEXsample
```

---

### Option A — Full Pipeline (One Command)

```bash
python main.py --mode full --device cuda --dentex-root "D:/WILP/sem-4/Dataset/DENTEX/DENTEXsample"
```

This runs all baseline stages automatically: preprocess → train → pseudo_label → train → evaluate.

Then run Phase 3 separately:

```bash
python main.py --mode train_hybrid --device cuda
```

---

### Option B — Step-by-Step (Recommended for First Run)

```bash
# Stage 1: Convert COCO JSON annotations → extended YOLO format
python main.py --mode preprocess --dentex-root "D:/WILP/sem-4/Dataset/DENTEX/DENTEXsample"
# → creates data/processed/images/{train,val,test}/ and data/processed/labels/
# → creates data/unlabelled/  (unlabelled xrays copied here for pseudo labeling)
# Sources used:
#   training_data/quadrant/                      → data_type=0 → train/
#   training_data/quadrant_enumeration/          → data_type=1 → train/
#   training_data/quadrant-enumeration-disease/  → data_type=2 → train/
#   validation_data/validation_triple.json       → data_type=2 → val/
#   test_data/disease/input/                     → images only → test/
#   training_data/unlabelled/xrays/              → data/unlabelled/

# Stage 2: Phase 1 training — detection only (no attribute heads yet)
python main.py --mode train --device cuda
# → saves weights/archon_phase1.pt
# → training logs at outputs/runs/

# Stage 3: Generate pseudo labels for healthy teeth
python main.py --mode pseudo_label --weights weights/archon_phase1.pt
# Part 3 images: detections not overlapping existing disease boxes → healthy pseudo labels
# Unlabelled images: all detected teeth → healthy pseudo labels
# → creates data/pseudo/ with merged labels

# Stage 4: Phase 2 training — full model with attribute heads
python main.py --mode train --device cuda --phase 2
# → saves weights/archon_best.pt
# → saves weights/attr_best.pt (Phase 2b attribute heads)

# Stage 5 [NEW]: Phase 3 — Hybrid components (Swin + Cross-Attention + Severity)
python main.py --mode train_hybrid --device cuda
# Requires: outputs/runs/phase2/weights/best.pt
# Freezes backbone + attr_heads; trains GlobalContextEncoder + MultiScaleFusion + HybridMultiTaskHead
# → saves weights/hybrid_best.pt (best validation loss across 50 epochs)

# Stage 6: Evaluate on official validation set (validation_triple.json)
python main.py --mode evaluate --weights weights/archon_best.pt
# → prints AP-Quadrant, AP-Enumeration, AP-Diagnosis

# Stage 7: Run inference on a new panoramic X-ray
#          auto-loads hybrid_best.pt for severity output when present
python main.py --mode predict --input path/to/xray.jpg --weights weights/archon_best.pt
# → saves outputs/predictions/xray_vis.jpg (annotated image)
# → saves outputs/predictions/xray_result.json (structured detection data with severity_details)
```

---

### Useful Flags

| Flag | Values | Description |
|---|---|---|
| `--mode` | `full`, `preprocess`, `train`, `pseudo_label`, `train_attr`, `train_hybrid`, `evaluate`, `predict` | Pipeline stage to run |
| `--dentex-root` | folder path | Root of DENTEX dataset on disk (**required** for preprocess / full) |
| `--device` | `cuda`, `cpu`, `0`, `0,1` | Training device (default: auto-detect) |
| `--weights` | path to `.pt` file | Model checkpoint for evaluate/predict/train_hybrid |
| `--input` | file or directory path | Input for predict mode |
| `--resume` | flag (no value) | Resume interrupted training from last checkpoint |
| `--phase` | `1`, `2` | Training phase (default: 1) |

---

## 5. Label Format Explained

Each `.txt` label file contains one row per tooth annotation.
Unlike standard YOLO (5 columns), this implementation uses **10 columns**:

```
class_id  cx    cy    w     h     is_imp  has_car  has_dc  has_les  data_type
   0      0.52  0.41  0.08  0.15    0       1        0       0         2
  15      0.30  0.38  0.07  0.14    0       0        0       0         2
```

| Column | Name | Range | Description |
|---|---|---|---|
| 0 | `class_id` | 0–31 | YOLO class index (maps to FDI via `CLASS_TO_FDI`) |
| 1 | `cx` | 0.0–1.0 | Bounding box center X (normalized) |
| 2 | `cy` | 0.0–1.0 | Bounding box center Y (normalized) |
| 3 | `w` | 0.0–1.0 | Bounding box width (normalized) |
| 4 | `h` | 0.0–1.0 | Bounding box height (normalized) |
| 5 | `is_impacted` | 0 or 1 | Tooth is impacted |
| 6 | `has_caries` | 0 or 1 | Tooth has caries |
| 7 | `has_deepcaries` | 0 or 1 | Tooth has deep caries |
| 8 | `has_lesion` | 0 or 1 | Periapical lesion present |
| 9 | `data_type` | 0, 1, or 2 | Annotation level (see below) |

**`data_type` values:**
- `0` = Quadrant-only annotation (Part 1) — only quadrant class loss is computed
- `1` = FDI enumeration annotation (Part 2) — FDI class loss computed
- `2` = Disease annotation (Part 3) — FDI class + all 4 attribute losses computed

> Columns 5–8 are **only meaningful (non-zero) when `data_type == 2`**.
> For `data_type 0` and `1`, attribute columns are filled with 0 but ignored during loss computation.

---

## 6. FDI Numbering System

The FDI system encodes tooth identity as a 2-digit number:
- **Tens digit** = Quadrant (1=Upper-Right, 2=Upper-Left, 3=Lower-Left, 4=Lower-Right)
- **Units digit** = Position in quadrant (1=Central Incisor → 8=Wisdom Tooth)

```
            Upper Jaw (Maxilla)
  Q2 (left)               Q1 (right)
  28 27 26 25 24 23 22 21 | 11 12 13 14 15 16 17 18
─────────────────────────────────────────────────────   Patient's perspective
  38 37 36 35 34 33 32 31 | 41 42 43 44 45 46 47 48
  Q3 (left)               Q4 (right)
            Lower Jaw (Mandible)
```

**FDI ↔ YOLO class index mapping:**

| YOLO Class | FDI | Tooth |
|---|---|---|
| 0 | 11 | Upper-Right Central Incisor |
| 1 | 12 | Upper-Right Lateral Incisor |
| ... | ... | ... |
| 7 | 18 | Upper-Right Wisdom Tooth |
| 8 | 21 | Upper-Left Central Incisor |
| ... | ... | ... |
| 15 | 28 | Upper-Left Wisdom Tooth |
| 16 | 31 | Lower-Left Central Incisor |
| ... | ... | ... |
| 23 | 38 | Lower-Left Wisdom Tooth |
| 24 | 41 | Lower-Right Central Incisor |
| ... | ... | ... |
| 31 | 48 | Lower-Right Wisdom Tooth |

**Formula:** `class_id = (quadrant - 1) × 8 + (position - 1)`

**Flip augmentation remapping:**
When an image is horizontally flipped, Q1↔Q2 and Q3↔Q4 swap.
`FLIP_CLASS_TABLE` in `src/utils/fdi.py` pre-computes the new class index for each of the 32 classes.

---

## 7. Training Phases Explained

### Phase 1 — Detection Training

- **Data used**: Parts 1 + 2 (quadrant + enumeration labels only)
- **Model**: YOLOv8x + modified FPN (CoordConv NOT injected during ultralytics training)
- **Losses computed**:
  - `data_type=0`: bbox regression + quadrant class loss (4 classes)
  - `data_type=1`: bbox regression + FDI class loss (32 classes)
- **Attribute heads**: NOT trained yet
- **Goal**: Learn to localize teeth and enumerate their FDI numbers; generate quality pseudo labels
- **Output**: `weights/archon_phase1.pt`

### Pseudo-Labeling (Between Phases)

- Run Phase 1 model on Part 3 images
- For each image, generate bounding boxes for teeth that appear healthy (no existing disease label)
- Assign `data_type=2`, all attribute flags = 0 (healthy)
- Merge with existing Part 3 disease labels → `data/pseudo/`

### Phase 2 — Full Model Training

- **Data used**: Parts 1 + 2 + 3 + pseudo labels
- **Model**: Phase 1 checkpoint + newly initialized attribute heads
- **Losses computed**: All of Phase 1 + attribute BCE for `data_type=2` samples
- **Attribute head fine-tuning (Phase 2b)**: Backbone frozen, attribute heads trained with per-tooth spatial sampling
- **Goal**: Learn to classify diseases per detected tooth
- **Output**: `weights/archon_best.pt` + `weights/attr_best.pt`

### Phase 3 — Hybrid Components Training (NEW)

- **Mode**: `python main.py --mode train_hybrid`
- **Requires**: `outputs/runs/phase2/weights/best.pt`
- **Frozen**: YOLOv8x backbone + binary attribute heads (backbone in `.eval()` for stable BN)
- **Trainable**: `GlobalContextEncoder` + `MultiScaleFusion` + `HybridMultiTaskHead`
- **Loss**: `AttributeBCELoss(w=4.0) + SeverityLoss(w=4.0) + QuadrantAuxLoss(w=1.0)`
- **Optimizer**: `AdamW(lr=5e-4)` with `CosineAnnealingLR` → `1e-6`
- **Augmentation**: CLAHE active (p=0.5) for improved early lesion detection
- **Epochs**: `phase3_epochs` (default 50)
- **Goal**: Add severity grading + quadrant consistency to the validated baseline model
- **Output**: `weights/hybrid_best.pt`

**What Phase 3 adds clinically:**
- Binary `has_caries` becomes `caries_severity: Healthy/Mild/Severe`
- FDI assignment benefits from quadrant-consistency penalty (fewer symmetric tooth swaps)
- CLAHE training improves detection of early caries in high-density enamel regions

---

## 8. Key Hyperparameters

All in `config/train_config.yaml` and `config/model_config.yaml`:

| Parameter | Value | Why |
|---|---|---|
| `phase1_epochs` | 100 | Enough epochs to learn stable tooth localization + FDI enumeration |
| `phase2_epochs` | 50 | Fine-tune detection with all data including disease images |
| `phase2b_epochs` | 50 | Attribute head training; more epochs → better disease recall |
| `phase3_epochs` | 50 | Hybrid head training (Swin + Cross-Attn + Severity) **[NEW]** |
| `batch_size` | 8 | Fits panoramic X-rays on 8–16GB GPU; Phase 3 uses half (4) |
| `input_size` | 1280×640 | Panoramic X-rays are wide-format |
| `loss_attr` | 8.0 | Higher weight: disease detection is the hard task |
| `fliplr` | 0.5 | 50% horizontal flip (quadrant-aware class remapping) |
| `mosaic` | 0.0 | Disabled — panoramic X-rays must not be mixed |
| `hsv_s` | 0.0 | X-rays are grayscale; saturation augmentation off |
| `lr0` | 0.01 | Initial learning rate for detection phases |
| `optimizer` | AdamW | Paper uses AdamW |
| `clahe_prob` | 0.5 | CLAHE augmentation probability during Phase 3 **[NEW]** |
| `swin_num_heads` | 8 | Attention heads in Swin blocks **[NEW]** |
| `swin_window_size` | 4 | Window size for local attention on P5 feature **[NEW]** |
| `fusion_num_heads` | 4 | Heads in cross-attention fusion **[NEW]** |
| `num_severity_levels` | 3 | Healthy / Mild / Severe **[NEW]** |
| `quadrant_penalty_alpha` | 0.5 | Strength of quadrant-consistency penalty in post-processing **[NEW]** |

---

## 9. Output Files

### After Training

```
weights/
├── archon_phase1.pt          ← Phase 1 checkpoint (detection only)
├── archon_best.pt            ← Best Phase 2 detection checkpoint
├── attr_best.pt                ← Phase 2b binary disease attribute heads
└── hybrid_best.pt              ← Phase 3 Swin + Cross-Attn + Severity checkpoint  [NEW]

outputs/runs/
└── phase1/weights/
└── phase2/weights/
    ├── best.pt
    ├── attr_best.pt
    ├── hybrid_best.pt          ← also saved here as backup  [NEW]
    └── results.csv             ← per-epoch metrics
```

### After Inference (`--mode predict`)

```
outputs/predictions/
├── <image_name>_vis.jpg        ← Annotated X-ray with colored bboxes + disease flags
└── <image_name>_result.json    ← Structured JSON:
    {
      "image": "xray.jpg",
      "detections": [
        {
          "fdi": 36,
          "bbox": [x1, y1, x2, y2],
          "confidence": 0.91,
          "is_impacted": false,
          "has_caries": true,
          "has_deepcaries": false,
          "has_lesion": false,
          "severity_details": {         ← only when hybrid_best.pt loaded  [NEW]
            "caries": "Mild",
            "impacted": "Healthy",
            "deepcaries": "Healthy",
            "lesion": "Healthy"
          }
        },
        ...
      ]
    }
```

---

## 10. Common Issues & Fixes

### `ModuleNotFoundError: No module named 'ultralytics'`
```bash
pip install ultralytics>=8.0.120
```

### `kaggle.ApiError: 403 Forbidden`
- You must accept the Dentex Challenge 2023 competition rules on Kaggle first.
- Go to: https://www.kaggle.com/competitions/dentex-challenge-2023 → Rules → Accept

### `CUDA out of memory`
Reduce batch size in `config/train_config.yaml`:
```yaml
batch_size: 4      # reduce from 8
```
Or use gradient accumulation:
```yaml
accumulate: 4      # effective batch = 4 × 4 = 16
```

### `RuntimeError: Expected all tensors to be on the same device`
Set device explicitly:
```bash
python main.py --mode train --device 0    # use GPU 0
```

### Training loss is NaN
- Check label files: all values must be in [0, 1] range for normalized coords
- Verify no empty label files exist in `data/processed/labels/`
- Lower learning rate: edit `lr0: 0.001` in `train_config.yaml`

### Pseudo-label step produces no labels
- Ensure Phase 1 training completed and `weights/archon_phase1.pt` exists
- Lower detection threshold: edit `conf_threshold: 0.2` in `model_config.yaml`

---

## 11. Minimum Requirements

| Component | Minimum | Recommended |
|---|---|---|
| Python | 3.9 | 3.10+ |
| CUDA | 11.8 | 12.1 |
| GPU VRAM | 8 GB | 16 GB |
| RAM | 16 GB | 32 GB |
| Disk space | 20 GB | 40 GB |
| PyTorch | 2.0 | 2.1+ |
| Ultralytics | 8.0.120 | latest |

**Estimated training time** (on RTX 3080 10GB):
- Phase 1 (200 epochs): ~6–8 hours
- Pseudo-label generation: ~20 minutes
- Phase 2 (200 epochs): ~8–10 hours
- Attribute fine-tuning: ~1–2 hours
