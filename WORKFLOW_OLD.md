# YOLOrtho — End-to-End Workflow Reference

> Paper: [arxiv.org/abs/2308.05967](https://arxiv.org/abs/2308.05967)  
> Dataset: DENTEX Challenge 2023 (Kaggle)

---

## Table of Contents

1. [Architecture at a Glance](#architecture-at-a-glance)
2. [Entry Point: `main.py`](#entry-point-mainpy)
3. [Mode Reference Table](#mode-reference-table)
4. [Mode: `preprocess`](#mode-preprocess)
5. [Mode: `pseudo_label`](#mode-pseudo_label)
6. [Mode: `train`](#mode-train)
7. [Mode: `train_attr`](#mode-train_attr)
8. [Mode: `evaluate`](#mode-evaluate)
9. [Mode: `predict`](#mode-predict)
10. [Mode: `full`](#mode-full)
11. [Key Component Deep-Dives](#key-component-deep-dives)
    - [CoordConv — Why and How](#coordconv--why-and-how)
    - [FDI Numbering System](#fdi-numbering-system)
    - [Hierarchical Loss and data_type](#hierarchical-loss-and-data_type)
    - [Modified FPN and Strides](#modified-fpn-and-strides)
    - [Attribute Heads (Disease Heads)](#attribute-heads-disease-heads)
    - [Linear Sum Assignment (Post-processing)](#linear-sum-assignment-post-processing)
    - [Flip Augmentation with Quadrant Remapping](#flip-augmentation-with-quadrant-remapping)
    - [Pseudo Labels](#pseudo-labels)
12. [Label Format Reference](#label-format-reference)
13. [Config Files Reference](#config-files-reference)
14. [Data Directory Layout](#data-directory-layout)
15. [Output Files Reference](#output-files-reference)
16. [Full Pipeline — Step-by-Step Call Graph](#full-pipeline--step-by-step-call-graph)

---

## Architecture at a Glance

```
Input: Panoramic X-ray (1280×640)
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  YOLOv8x Backbone (layers 0–9)  ← CoordConv injected here      │
│    Each Conv2d → CoordConv (adds x,y channels before convolution)│
└─────────────────────────┬───────────────────────────────────────┘
                          │  P3, P4, P5 feature maps
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  Modified PANet Neck (layers 10–21)                             │
│    Standard strides [8,16,32] → YOLOrtho strides [4,8,16]      │
│    Extra upsample layer added to create N2 (stride 4)           │
└──────────────┬──────────────────────────┬───────────────────────┘
               │                          │
               ▼                          ▼
  ┌────────────────────┐      ┌──────────────────────────────────┐
  │  Detection Head    │      │  Disease Attribute Heads (×4)    │
  │  32 FDI classes    │      │  is_impacted  (BCE, weight=8.0)  │
  │  (classification   │      │  has_caries   (BCE, weight=8.0)  │
  │  + bbox regression)│      │  has_deepcaries(BCE, weight=8.0) │
  └────────────────────┘      │  has_lesion   (BCE, weight=8.0)  │
                              └──────────────────────────────────┘
               │                          │
               └──────────┬───────────────┘
                          ▼
          Linear Sum Assignment (scipy)
          → Unique FDI per detection
          → ToothDetection objects
          → JSON + Visualized JPEG
```

---

## Entry Point: `main.py`

```
python main.py --mode <MODE> [options]
```

**All `--mode` values trigger `main.py` → `parse_args()` → corresponding `stage_*()` function.**

### CLI Arguments

| Argument | Type | Default | Used By | Purpose |
|---|---|---|---|---|
| `--mode` | str (choice) | `full` | All | Which pipeline stage to run |
| `--dentex-root` | str | `None` | `preprocess`, `full` | Path to the DENTEX dataset folder on disk |
| `--config` | str | `config/train_config.yaml` | `train`, `train_attr`, `full` | Training hyperparameters YAML |
| `--weights` | str | `None` | `evaluate`, `predict`, `train_attr` | Path to a `.pt` weights file |
| `--input` | str | `None` | `predict` | Image file or directory to run inference on |
| `--output` | str | `outputs/predictions` | `predict` | Where to write result JSON + visualized images |
| `--device` | str | `cuda` | `train`, `evaluate`, `predict`, `pseudo_label` | `cuda`, `cuda:0`, or `cpu` |
| `--resume` | flag | `False` | `train` | Resume from last ultralytics checkpoint |
| `--conf` | float | `0.25` | `evaluate`, `predict` | Detection confidence threshold |
| `--val-ratio` | float | `0.2` | (deprecated) | Previously split train→val; now official val set is used |

---

## Mode Reference Table

| Mode | Entry Function | Key Module Called | Weights In | Writes To |
|---|---|---|---|---|
| `preprocess` | `stage_preprocess()` | `src/data/preprocess.py → preprocess_dentex()` | — | `data/processed/` |
| `pseudo_label` | `stage_pseudo_label()` | `src/data/pseudo_label.py → generate_pseudo_labels()` | `outputs/runs/phase1/weights/best.pt` | `data/pseudo/` |
| `train` | `stage_train()` | `src/training/trainer.py → ARCHONTrainer.train()` | `yolov8x.pt` (COCO) | `outputs/runs/`, `weights/` |
| `train_attr` | `stage_train_attr()` | `src/training/trainer.py → ARCHONTrainer.train_attr_only()` | `outputs/runs/phase2/weights/best.pt` | `weights/attr_best.pt` |
| `evaluate` | `stage_evaluate()` | `src/inference/predictor.py → ARCHONPredictor.evaluate()` | `weights/archon_best.pt` | stdout/log |
| `predict` | `stage_predict()` | `src/inference/predictor.py → ARCHONPredictor.predict()` | `weights/archon_best.pt` + `weights/attr_best.pt` | `outputs/predictions/` |
| `full` | `run_full_pipeline()` | All stages above in order | `yolov8x.pt` (auto-download) | All above |

---

## Mode: `preprocess`

```bash
python main.py --mode preprocess --dentex-root D:/path/to/DENTEX
```

### Call Chain

```
main.py → stage_preprocess(args)
    └── src/data/preprocess.py → preprocess_dentex(dentex_root, out_dir)
            ├── _process_part1()    → data_type=0 labels  (quadrant only)
            ├── _process_part2()    → data_type=1 labels  (quadrant + FDI enum)
            ├── _process_part3()    → data_type=2 labels  (quadrant + FDI + disease)
            ├── _process_validation() → val split
            ├── _copy_images()      → test split (images only)
            └── _copy_images()      → data/unlabelled/ (for pseudo labeling later)
```

### What It Does

Converts COCO JSON annotations (from three DENTEX dataset parts) into a single **extended YOLO 10-column label format**:

```
class_id  cx  cy  w  h  is_impacted  has_caries  has_deepcaries  has_lesion  data_type
```

| Source JSON | data_type | Columns populated |
|---|---|---|
| `train_quadrant.json` (Part 1) | `0` | class (quadrant 0-3), bbox, attrs all 0 |
| `train_quadrant_enumeration.json` (Part 2) | `1` | class (FDI 0-31), bbox, attrs all 0 |
| `train_quadrant_enumeration_disease.json` (Part 3) | `2` | class (FDI 0-31), bbox, attrs from JSON |
| `validation_triple.json` | `2` | same as Part 3, → val split |

**Key detail:** Part 3 has one annotation row per (tooth, disease). If tooth 16 has both Caries and Deep Caries, it has **two rows** in the JSON. The preprocessor merges them by `(image_id, fdi)` so the label file has **one row per tooth** with multiple disease flags set.

**Disease category ID → attribute column mapping** (hardcoded in `preprocess.py`):

```python
DISEASE_CAT3_TO_ATTR_IDX = {
    0: 0,   # 'Impacted'          → is_impacted    col 5
    1: 1,   # 'Caries'            → has_caries     col 6
    2: 3,   # 'Periapical Lesion' → has_lesion     col 8
    3: 2,   # 'Deep Caries'       → has_deepcaries col 7
}
```

**FDI class computation:**  
`src/utils/fdi.py → fdi_to_class(fdi)` and `quadrant_enum_to_fdi(quadrant, enum)` are called by the preprocessor to convert `(category_id_1, category_id_2)` → FDI number → YOLO class index.

### Output

```
data/processed/
  images/train/    ← all training X-rays (parts 1+2+3)
  images/val/      ← official validation X-rays
  images/test/     ← test X-rays (no labels)
  labels/train/    ← 5-column YOLO labels (class cx cy w h)
  labels/val/
  labels_ext/train/← 10-column extended labels (used by attr head training)
  labels_ext/val/
data/unlabelled/   ← unlabelled X-rays for pseudo labeling
```

---

## Mode: `pseudo_label`

```bash
python main.py --mode pseudo_label
# Requires Phase 1 training to have been completed first
```

### Call Chain

```
main.py → stage_pseudo_label(args)
    └── src/data/pseudo_label.py → generate_pseudo_labels(
            data_dir, pseudo_dir, unlabelled_dir, weights_path, device, conf)
            ├── loads Phase 1 YOLO model (ultralytics)
            ├── For each Part 3 image (data_type==2 stems):
            │     model.predict(image, conf=0.5)
            │     _extract_pseudo_labels() ← IoU filter against existing disease labels
            │     merges → pseudo/labels/ and pseudo/labels_ext/
            └── _process_unlabelled() ← all detections become healthy pseudo labels
```

### Why Pseudo Labels Are Needed

Part 3 of DENTEX only annotates **diseased** teeth. If a tooth is healthy, it has no annotation. During training, the model is penalized for detecting healthy teeth (treating them as false positives). Pseudo labeling fixes this:

1. Run Phase 1 detector on Part 3 images
2. Any detection that does **not** overlap an existing disease label (IoU < 0.3) is a **healthy tooth candidate**
3. These become pseudo labels: `class_id cx cy w h 0 0 0 0 2`  
   (all disease flags = 0, data_type = 2 so the attr loss fires)

For **unlabelled** images (no annotations at all), every detection becomes a healthy pseudo label.

### Key Parameters

| Parameter | Value | Meaning |
|---|---|---|
| `OVERLAP_IOU_THRESHOLD` | `0.3` | Max IoU with existing disease box to be considered a new healthy tooth |
| `MIN_CONFIDENCE` | `0.5` | Min detection confidence for a pseudo label to be accepted |

### Output

```
data/pseudo/
  images/train/        ← same images as processed/
  labels/train/        ← 5-col labels (detection training)
  labels_ext/train/    ← 10-col labels (attr head training) with pseudo healthy rows added
```

---

## Mode: `train`

```bash
python main.py --mode train [--resume]
```

### Call Chain

```
main.py → stage_train(args)
    └── src/training/trainer.py → ARCHONTrainer(config_path, resume, device)
            └── ARCHONTrainer.train()
                    ├── [if phase1 weights missing] _train_phase1()
                    │       └── ultralytics YOLO("yolov8x.pt").train(phase1_args)
                    │           → saves outputs/runs/phase1/weights/best.pt
                    │
                    └── _train_phase2(pretrain_weights=phase1/best.pt)
                            ├── ultralytics YOLO(phase1_best).train(phase2_args)
                            │   → saves outputs/runs/phase2/weights/best.pt
                            └── _train_attribute_heads(base_weights=phase2_best)
                                    ├── build_archon_base()     ← injects CoordConv
                                    │     src/models/yolortho.py
                                    │     src/models/coord_conv.py
                                    │     src/models/heads.py → MultiAttributeHead
                                    ├── ARCHONDataset(labels_ext/train/)
                                    │     src/data/dataset.py
                                    ├── AttributeBCELoss      src/training/loss.py
                                    ├── Per-tooth spatial sampling of FPN features
                                    └── saves weights/attr_best.pt
```

### Two-Phase Training Strategy

#### Phase 1 — Detection Pre-training

- **Data:** Parts 1 + 2 only (quadrant and FDI labels; no disease labels)
- **Model:** Standard YOLOv8x (CoordConv NOT injected — ultralytics rebuilds the model internally and would reject modified weights)
- **Loss:** Standard YOLOv8 loss: `bbox (7.5) + cls (0.5) + DFL (1.5)` — no attribute loss
- **Purpose:** Train a capable tooth detector before adding disease complexity. This checkpoint is also used for pseudo labeling.
- **Epochs:** `phase1_epochs` (default 100)
- **Saves to:** `outputs/runs/phase1/weights/best.pt`

#### Phase 2 — Full Fine-tuning with Disease Attributes

**Step 2a — Detection head fine-tuning:**
- **Data:** All data including Part 3 and pseudo labels
- **Starting point:** Phase 1 `best.pt`
- **Loss:** Same standard YOLOv8 loss
- **Epochs:** `phase2_epochs` (default 50)
- **Saves to:** `outputs/runs/phase2/weights/best.pt`

**Step 2b — Attribute head training (backbone frozen):**
- **Model:** `build_archon_base()` from `src/models/yolortho.py`
  - Loads Phase 2 `best.pt`
  - **Injects CoordConv** into backbone layers 0–9 (`replace_backbone_conv_with_coordconv()`)
  - Attaches `MultiAttributeHead` and FPN hooks
- **Backbone:** frozen (`requires_grad = False`), put in `.eval()` mode (important: BN uses running stats, not per-batch stats, matching inference)
- **Attribute heads:** trainable (`requires_grad = True`)
- **Loss:** `AttributeBCELoss` — BCE with per-attribute `pos_weight` to counteract class imbalance
- **Supervision:** **Per-tooth spatial sampling** — FPN features are sampled at each GT tooth's center coordinate (`cx * W_f`, `cy * H_f`) rather than global average. This teaches each head to respond to *that specific tooth's location*.
- **Epochs:** `phase2b_epochs` (default 50)
- **Optimizer:** AdamW, `lr=1e-3`, CosineAnnealingLR → 1e-5
- **Saves to:** `weights/attr_best.pt` and `outputs/runs/phase2/weights/attr_best.pt`

### Total Loss Formula

$$\mathcal{L}_{total} = 7.5 \cdot L_{bbox} + 0.5 \cdot L_{cls} + 1.5 \cdot L_{DFL} + 8.0 \cdot (L_{impacted} + L_{caries} + L_{deepcaries} + L_{lesion})$$

The `data_type` column in labels controls which loss terms fire:

| `data_type` | bbox | cls | DFL | attr |
|---|---|---|---|---|
| `0` — quadrant only | ✓ | ✓ (quadrant grouped) | ✓ | ✗ |
| `1` — enumeration | ✓ | ✓ (full FDI) | ✓ | ✗ |
| `2` — disease | ✓ | ✓ (full FDI) | ✓ | ✓ |

### Key Training Config Parameters

| Parameter | Default | Impact |
|---|---|---|
| `phase1_epochs` | 100 | Too few → poor tooth localization → bad pseudo labels |
| `phase2_epochs` | 50 | Fine-tunes detection with all data |
| `phase2b_epochs` | 50 | More epochs = better disease recall (backbone frozen, no regression) |
| `batch_size` | 8 | Reduce to 4 if OOM on 8GB GPU |
| `lr0` | 0.01 | Initial SGD learning rate for detection phases |
| `loss_attr` | 8.0 | High weight makes disease prediction dominant; matches paper |
| `attr_pos_weight` | null (auto) | `null` = computed from dataset; ~ratio of healthy:diseased per attribute |
| `pseudo_label_conf` | 0.5 | Min confidence threshold in pseudo labeling |
| `val_interval` | 5 | Evaluate every 5 epochs |
| `patience` | 50 | Early stopping if no improvement |
| `fliplr` | 0.5 | 50% chance of horizontal flip (with quadrant remapping) |
| `mosaic` | 0.0 | Disabled — panoramic X-rays must not be mixed |

---

## Mode: `train_attr`

```bash
python main.py --mode train_attr [--weights path/to/phase2/best.pt]
```

### Call Chain

```
main.py → stage_train_attr(args)
    └── ARCHONTrainer.train_attr_only(base_weights)
            └── _train_attribute_heads(base_weights)
                    [same as Phase 2b above]
```

### When to Use

Use this mode when:
- `attr_best.pt` is missing
- All disease predictions show ~0.001 probability (everything healthy) — this indicates the **no_grad bug**: the old code wrapped attribute head computation in `torch.no_grad()`, so gradients never flowed and weights never updated
- You want to retrain only disease heads without touching the detector

---

## Mode: `evaluate`

```bash
python main.py --mode evaluate --weights weights/archon_best.pt
```

### Call Chain

```
main.py → stage_evaluate(args)
    └── src/inference/predictor.py → ARCHONPredictor(weights, device, conf)
            └── predictor.evaluate(data_yaml="config/dataset.yaml", split="val")
                    → ultralytics YOLO.val()
                    → returns mAP@0.5, mAP@0.5:0.95, AP per class
```

Reports COCO detection metrics as defined by the Dentex challenge:
- **AP-Quadrant** — detection accuracy at quadrant level
- **AP-Enumeration** — tooth-number-level accuracy
- **AP-Diagnosis** — disease detection accuracy

---

## Mode: `predict`

```bash
python main.py --mode predict \
    --input data/processed/images/test/sample.jpg \
    --weights weights/archon_best.pt \
    --output outputs/predictions \
    --conf 0.25
```

### Call Chain

```
main.py → stage_predict(args)
    └── src/inference/predictor.py → ARCHONPredictor(weights, device, conf)
            └── predictor.predict(input_path, output_dir, save_json, save_vis)
                    ├── _load_models()
                    │     ├── YOLO(weights_path)           ← detection model
                    │     ├── _load_attr_heads(attr_best.pt) ← disease heads
                    │     └── _attach_fpn_hooks()           ← capture features
                    │
                    ├── For each image:
                    │     det_model.predict() → raw YOLO output
                    │     _fpn_features populated by hooks
                    │     _predict_attributes() → disease probabilities
                    │
                    └── postprocess_yolo_output()
                            src/inference/postprocess.py
                            ├── apply_linear_sum_assignment() ← unique FDI per tooth
                            └── returns List[ToothDetection]
                    │
                    ├── draw_teeth_detections()   src/utils/visualize.py
                    │     → <stem>_vis.jpg
                    └── saves <stem>_result.json
```

### Output Per Image

**`<stem>_result.json`:**
```json
{
  "train_5": [
    {
      "fdi": 16,
      "fdi_name": "Upper Right First Molar",
      "conf": 0.912,
      "bbox_xyxy": [320.1, 120.4, 450.8, 240.2],
      "is_impacted": false,
      "has_caries": true,
      "has_deepcaries": false,
      "has_lesion": false,
      "diseases": ["Caries"]
    }
  ]
}
```

**`<stem>_vis.jpg`:** Annotated X-ray with bounding boxes, labels in format `Q: {quad} N: {pos} D: {disease}`, colored by quadrant.

---

## Mode: `full`

```bash
python main.py --mode full --dentex-root D:/path/to/DENTEX
```

Runs all stages in order:

```
run_full_pipeline(args)
    1. stage_preprocess(args)
    2. ARCHONTrainer.train_phase1_only()    ← Phase 1 only, needed for pseudo labels
    3. stage_pseudo_label(args)
    4. stage_train(args)                       ← Phase 2 (Phase 1 auto-skipped if done)
    5. stage_evaluate(args)
```

---

## Key Component Deep-Dives

### CoordConv — Why and How

**File:** `src/models/coord_conv.py`  
**Functions:** `AddCoords`, `CoordConv`, `replace_backbone_conv_with_coordconv()`

#### The Problem with Standard Convolution

Standard `nn.Conv2d` is **translation invariant** — it produces the same output regardless of where in the image the feature appears. This is great for object classification, but a problem for **teeth enumeration**: a tooth's FDI number depends entirely on its position:

- Tooth 16 (Upper Right First Molar) is always top-right
- Tooth 46 (Lower Right First Molar) is always bottom-right
- A standard conv cannot distinguish these from position alone

#### The CoordConv Solution

Before each convolution in the backbone, append two extra channels that encode pixel coordinates, normalized to `[-1, 1]`:

```
x_channel[b, 0, i, j] = j / (W-1) * 2 - 1   ← horizontal position
y_channel[b, 0, i, j] = i / (H-1) * 2 - 1   ← vertical position
```

**Example:** For a 4×4 feature map, the x-channel looks like:

```
-1.0  -0.33  +0.33  +1.0
-1.0  -0.33  +0.33  +1.0
-1.0  -0.33  +0.33  +1.0
-1.0  -0.33  +0.33  +1.0
```

Now the backbone "knows where it is" and can learn position-dependent filters like "if there's a tooth feature here AND the x-coordinate channel says +0.9, this is likely a Q1/Q2 tooth."

#### Injection

`replace_backbone_conv_with_coordconv()` walks backbone layers 0–9 (only backbone, not neck/head) and replaces each `nn.Conv2d` with a `CoordConv`. The existing pretrained weights are preserved by copying them into `new_conv.conv.weight[:, :original_in_channels]`. The extra coordinate channels' weights are **zeroed** so the model initially behaves identically to the pre-CoordConv version (clean initialization, no random noise).

```
CoordConv(in=64, out=128):
   AddCoords()  →  x: (B,64,H,W) + xx + yy = (B,66,H,W)
   Conv2d(66→128, ...)
```

**Note:** CoordConv is only injected during Phase 2b (attribute head training), not Phase 1 or Phase 2 detection training — because ultralytics rebuilds the model internally from its yaml config during `.train()`, causing a state-dict key mismatch if CoordConv is pre-injected.

---

### FDI Numbering System

**File:** `src/utils/fdi.py`

The FDI system assigns a 2-digit code to each tooth:
- **First digit:** quadrant (1=upper-right, 2=upper-left, 3=lower-left, 4=lower-right)
- **Second digit:** position (1=central incisor ... 8=wisdom tooth)

```
Panoramic X-ray (patient view):
  Upper jaw: [18 17 16 15 14 13 12 11] | [21 22 23 24 25 26 27 28]
             ←── Q1 Upper Right ──────   ────── Q2 Upper Left ──→
  Lower jaw: [48 47 46 45 44 43 42 41] | [31 32 33 34 35 36 37 38]
             ←── Q4 Lower Right ──────   ────── Q3 Lower Left ──→
```

**YOLO class mapping** (0-indexed, 32 classes):

| FDI | YOLO class | Quadrant |
|---|---|---|
| 11 | 0 | Q1 Upper Right |
| 12 | 1 | Q1 |
| ... | ... | |
| 18 | 7 | Q1 |
| 21 | 8 | Q2 Upper Left |
| ... | ... | |
| 48 | 31 | Q4 Lower Right |

**Key functions:**

| Function | Input → Output | Example |
|---|---|---|
| `fdi_to_class(fdi)` | FDI number → class index 0-31 | `fdi_to_class(16)` → `5` |
| `class_to_fdi(idx)` | class index → FDI number | `class_to_fdi(5)` → `16` |
| `quadrant_enum_to_fdi(q, e)` | quadrant + position → FDI | `quadrant_enum_to_fdi(2,3)` → `23` |
| `fdi_to_name(fdi)` | FDI → human name | `fdi_to_name(23)` → `"Upper Left Canine"` |
| `flip_fdi(fdi)` | FDI → mirrored FDI | `flip_fdi(16)` → `26` (Q1→Q2) |

---

### Hierarchical Loss and data_type

**File:** `src/training/loss.py`

The DENTEX dataset has three annotation levels (Parts 1, 2, 3). The `data_type` flag in the label (column 9) tells the loss function which supervision is available for each sample:

```
data_type = 0  →  only quadrant boundaries known
data_type = 1  →  full FDI tooth number known
data_type = 2  →  FDI + disease attributes known
```

**`AttributeBCELoss`** only computes loss on rows where `data_type == 2`:
```python
mask = (data_types == 2)  # only disease-annotated teeth
if mask.sum() == 0:
    return pred_attrs.sum() * 0.0  # differentiable zero
```

**`HierarchicalClassLoss`** groups the 32 FDI classes into 4 quadrants for `data_type == 0` samples. So even partial annotations contribute useful gradients.

**`pos_weight` (class imbalance):**  
Most teeth are healthy. Without counterweighting, BCE loss collapses to always predicting "healthy":

| Attribute | Disease prevalence | pos_weight default |
|---|---|---|
| `is_impacted` | ~15% | 5.0 |
| `has_caries` | ~25% | 3.0 |
| `has_deepcaries` | ~8% | 8.0 |
| `has_lesion` | ~15% | 5.0 |

Formula for each element: $\text{loss} = -(pw \cdot y \cdot \log\sigma(x) + (1-y)\cdot\log(1-\sigma(x)))$  
where $pw$ is the pos_weight for that attribute.

---

### Modified FPN and Strides

**Config:** `model_config.yaml → use_modified_fpn: true`, `fpn_strides: [4, 8, 16]`

Standard YOLOv8 detects at strides `[8, 16, 32]`. This means the smallest feature map cells are `8×8` pixels, which can miss fine-grained tooth boundaries in panoramic X-rays (teeth are large relative to the image, benefit from fine resolution).

YOLOrtho adds an extra upsampling stage in the PANet neck, producing a stride-4 feature map (N2). Detection then happens at `[4, 8, 16]` instead of `[8, 16, 32]`. The attribute heads also operate at these three scales, pooling their multi-scale predictions before the final classification decision.

---

### Attribute Heads (Disease Heads)

**File:** `src/models/heads.py`  
**Classes:** `AttributeHead`, `MultiAttributeHead`, `AttributeLoss`

Each of the 4 disease attributes has its own independent head (they share no weights).

**`AttributeHead` architecture (single attribute, single scale):**
```
FPN feature map (B, C, H, W)
    │
    Conv(C → C//4, 3×3, BN, SiLU)
    │
    Conv(C//4 → C//4, 3×3, BN, SiLU)
    │
    Conv(C//4 → 1, 1×1)   ← output: 1 logit per spatial position
```

Output bias is initialized to `-log(99) ≈ -4.6` so `sigmoid(bias) ≈ 0.01` (disease is rare at start).

**`MultiAttributeHead`** manages a `ModuleList[attrs × scales]`:
```
self.heads[attr_idx][scale_idx]  → AttributeHead
```

At inference, features are sampled at each detected box centre across all 3 scales, then averaged:
```python
for feat in fpn_features:         # 3 scales
    xf = (cx * W_f).long()
    yf = (cy * H_f).long()
    sampled = feat[b, :, yf, xf]  # (4, N_teeth)
tooth_logits = mean(scale_samples)  # (N_teeth, 4)
```

This **per-tooth spatial sampling** (vs. earlier global average) is a critical bug fix: the model now learns that a disease feature at *this box's location* predicts disease for *this tooth*, not for the entire image.

---

### Linear Sum Assignment (Post-processing)

**File:** `src/inference/postprocess.py`  
**Function:** `apply_linear_sum_assignment()`

Problem: YOLOv8 might assign the same FDI class to two adjacent teeth (both tooth 16 and tooth 17 predicted as "16").

Solution: Formulate FDI assignment as a **linear sum assignment problem** (Hungarian algorithm):
- **Rows:** 32 possible FDI slots (one per tooth number)
- **Columns:** N detected bounding boxes
- **Cost:** `cost[i,j] = -log(prob[j,i] + ε)` — low cost = high probability of box j being FDI slot i
- **Constraint:** Each FDI slot is assigned at most one box, and each box assigned to at most one FDI slot

```python
from scipy.optimize import linear_sum_assignment
cost_matrix = -np.log(valid_probs.T + 1e-7)  # (32, M)
row_ind, col_ind = linear_sum_assignment(cost_matrix)
```

**Result:** At most 32 `ToothDetection` objects returned, one per FDI number, with guaranteed uniqueness.

---

### Flip Augmentation with Quadrant Remapping

**File:** `src/data/augmentation.py`  
**File:** `src/utils/fdi.py → FLIP_CLASS_TABLE`

When a panoramic X-ray is horizontally flipped, the **physical tooth positions swap quadrants**:
- Q1 (upper-right) ↔ Q2 (upper-left)
- Q3 (lower-left) ↔ Q4 (lower-right)

Standard augmentation libraries just flip the image and mirror the bounding box coordinates — they do **not** remap the class labels. `ARCHONAugmentor._augment_fliplr()` flips both the image and the class ID:

```
Original: tooth 16 (Q1 first molar) at x=0.8
  → Flipped: bounding box x = 1.0 - 0.8 = 0.2
             class remapped: FDI 16 → FDI 26 (Q2 first molar)
```

This doubles the effective training set while maintaining anatomical correctness.

---

### Pseudo Labels

**File:** `src/data/pseudo_label.py`

**Why needed:**  
Part 3 images only annotate diseased teeth. Healthy teeth are unannotated → the model is penalized for detecting them as false positives.

**Strategy:**  
Run Phase 1 detector (trained only on clean Parts 1+2) on Part 3 images. Any high-confidence detection (`conf ≥ 0.5`) that does NOT overlap an existing disease label (`IoU < 0.3`) is labeled as a healthy tooth:

```
pseudo row: class_id cx cy w h 0 0 0 0 2
```

The `data_type=2` flag means the attribute loss will fire on these pseudo labels during Phase 2b — the model learns "no disease here" from these examples, preventing false positives.

---

## Label Format Reference

**Standard YOLO (5 columns) — `labels/` folder:**
```
class_id  cx  cy  w  h
```

**Extended YOLOrtho (10 columns) — `labels_ext/` folder:**
```
col 0: class_id       — int 0-31 (FDI class); 0-3 for data_type=0
col 1: cx             — float [0,1] normalized center x
col 2: cy             — float [0,1] normalized center y
col 3: w              — float [0,1] normalized width
col 4: h              — float [0,1] normalized height
col 5: is_impacted    — int {0, 1}
col 6: has_caries     — int {0, 1}
col 7: has_deepcaries — int {0, 1}
col 8: has_lesion     — int {0, 1}
col 9: data_type      — int {0, 1, 2}
```

**Example rows:**
```
# data_type=0: quadrant label only (class 0 = Q1), no FDI, no disease
0  0.850  0.320  0.080  0.120  0  0  0  0  0

# data_type=1: full FDI (class 5 = FDI 16), no disease
5  0.850  0.320  0.080  0.120  0  0  0  0  1

# data_type=2: FDI + disease (class 5 = FDI 16, has caries)
5  0.850  0.320  0.080  0.120  0  1  0  0  2

# Pseudo label: healthy tooth (class 8 = FDI 21), all diseases = 0
8  0.200  0.300  0.075  0.110  0  0  0  0  2
```

---

## Config Files Reference

### `config/train_config.yaml`

| Section | Key | Default | Role |
|---|---|---|---|
| `training` | `phase1_epochs` | 100 | Phase 1 detection epochs |
| | `phase2_epochs` | 50 | Phase 2 detection fine-tune epochs |
| | `phase2b_epochs` | 50 | Attribute head training epochs |
| | `batch_size` | 8 | Samples per GPU batch |
| | `optimizer` | `SGD` | `SGD` or `AdamW` |
| | `lr0` | 0.01 | Initial learning rate |
| | `lrf` | 0.01 | Final LR factor (cosine: `lr0 * lrf`) |
| | `loss_box` | 7.5 | Bbox regression loss weight |
| | `loss_cls` | 0.5 | Classification loss weight |
| | `loss_dfl` | 1.5 | Distribution Focal Loss weight |
| | `loss_attr` | 8.0 | Disease attribute BCE loss weight |
| | `hierarchical` | true | Enable data_type-based loss masking |
| | `pseudo_label_conf` | 0.5 | Confidence threshold for pseudo labels |
| | `attr_pos_weight` | null | Per-attr BCE positive weight. `null` = auto |
| | `patience` | 50 | Early stopping epochs without improvement |
| `augmentation` | `hsv_v` | 0.4 | Brightness jitter (X-ray safe) |
| | `fliplr` | 0.5 | Horizontal flip probability |
| | `mosaic` | 0.0 | Disabled (panoramic X-rays must not be mixed) |

### `config/model_config.yaml`

| Key | Value | Role |
|---|---|---|
| `model.base` | `yolov8x` | Backbone architecture |
| `model.num_classes` | 32 | FDI tooth classes |
| `model.attributes` | 4 items | Disease head definitions + weights |
| `model.use_coordconv` | true | Inject CoordConv into backbone |
| `model.coordconv_with_r` | false | Add radial coordinate channel (optional) |
| `model.use_modified_fpn` | true | Stride [4,8,16] instead of [8,16,32] |
| `model.input_width` | 1280 | Image width (panoramic X-rays are wide) |
| `model.input_height` | 640 | Image height |
| `model.conf_threshold` | 0.25 | Detection confidence threshold at inference |
| `model.iou_threshold` | 0.45 | NMS IoU threshold |
| `model.max_det` | 32 | Max detections per image (max 32 teeth) |

### `config/dataset.yaml`

Defines the YOLO dataset paths and class names used by ultralytics' `YOLO.train()` and `YOLO.val()`. The 32 class names are the FDI strings (`"11"`, `"12"`, ..., `"48"`). The extended label columns (attrs, data_type) are documented here but handled by the custom dataset/loss code, not by ultralytics natively.

---

## Data Directory Layout

```
data/
├── processed/          ← output of --mode preprocess
│   ├── images/
│   │   ├── train/      ← all training X-rays (Parts 1+2+3 combined)
│   │   ├── val/        ← official validation X-rays
│   │   └── test/       ← test X-rays (no labels)
│   ├── labels/
│   │   ├── train/      ← 5-col YOLO labels (used by ultralytics detection training)
│   │   └── val/
│   └── labels_ext/
│       ├── train/      ← 10-col extended labels (used by attr head training)
│       └── val/
├── pseudo/             ← output of --mode pseudo_label
│   ├── images/train/
│   ├── labels/train/   ← 5-col with pseudo healthy rows added
│   └── labels_ext/train/ ← 10-col with pseudo healthy rows added
└── unlabelled/         ← unlabelled X-rays (copied from DENTEX by preprocess)
```

---

## Output Files Reference

```
outputs/
├── archon.log           ← all run logs (appended)
├── predictions/
│   ├── <stem>_vis.jpg     ← annotated X-ray (bboxes + Q/N/D labels)
│   └── <stem>_result.json ← structured detections
└── runs/
    ├── phase1/
    │   ├── args.yaml
    │   └── weights/
    │       ├── best.pt    ← Phase 1 best detection model
    │       └── last.pt
    └── phase2/
        ├── args.yaml
        ├── results.csv
        └── weights/
            ├── best.pt    ← Phase 2 best detection model
            ├── last.pt
            └── attr_best.pt ← trained attribute heads

weights/                   ← top-level convenience copies
├── archon_best.pt       ← copy of phase2/best.pt
└── attr_best.pt           ← copy of phase2/attr_best.pt
```

---

## Full Pipeline — Step-by-Step Call Graph

```
python main.py --mode full --dentex-root D:/DENTEX
│
│ main.py
├── parse_args()
│     --mode full, --dentex-root set
│
└── run_full_pipeline(args)
    │
    ├─── [STAGE 1] stage_preprocess(args)
    │     └── src/data/preprocess.py
    │           preprocess_dentex(dentex_root, "data/processed")
    │           ├── _process_part1()  →  data/processed/labels/train/ (data_type=0)
    │           ├── _process_part2()  →  data/processed/labels/train/ (data_type=1)
    │           ├── _process_part3()  →  data/processed/labels_ext/train/ (data_type=2)
    │           ├── _process_validation() → data/processed/labels_ext/val/
    │           └── _copy_images() → data/unlabelled/
    │                 calls: src/utils/fdi.py → fdi_to_class(), quadrant_enum_to_fdi()
    │
    ├─── [STAGE 2] ARCHONTrainer.train_phase1_only()
    │     └── src/training/trainer.py
    │           _train_phase1()
    │           ├── ultralytics YOLO("yolov8x.pt")    ← downloads from COCO
    │           └── YOLO.train(data=dataset.yaml, epochs=100, imgsz=1280, ...)
    │                 uses: data/processed/labels/train/ (data_type 0+1)
    │                 saves: outputs/runs/phase1/weights/best.pt
    │
    ├─── [STAGE 3] stage_pseudo_label(args)
    │     └── src/data/pseudo_label.py
    │           generate_pseudo_labels(data_dir, pseudo_dir, unlabelled_dir, ...)
    │           ├── YOLO(phase1/best.pt).predict(Part3_images, conf=0.5)
    │           ├── _extract_pseudo_labels() → filter IoU < 0.3 with disease boxes
    │           ├── merge disease + pseudo healthy rows
    │           ├── write data/pseudo/labels/ and data/pseudo/labels_ext/
    │           └── _process_unlabelled() → all detections → healthy pseudo labels
    │
    ├─── [STAGE 4] stage_train(args)
    │     └── src/training/trainer.py
    │           ARCHONTrainer.train()
    │           │
    │           ├── [Phase 1 weights found → skip]
    │           │
    │           ├── _train_phase2(pretrain_weights=phase1/best.pt)
    │           │     ├── YOLO(phase1/best.pt)
    │           │     ├── YOLO.train(data=dataset.yaml, epochs=50, ...)
    │           │     │     uses: data/pseudo/ (all data with pseudo labels)
    │           │     └── saves: outputs/runs/phase2/weights/best.pt
    │           │
    │           └── _train_attribute_heads(base_weights=phase2/best.pt)
    │                 ├── build_archon_base(base_weights, use_coordconv=True)
    │                 │     src/models/yolortho.py
    │                 │     ├── YOLO(phase2/best.pt)
    │                 │     ├── replace_backbone_conv_with_coordconv()
    │                 │     │     src/models/coord_conv.py
    │                 │     │     replaces Conv2d[0..9] with CoordConv
    │                 │     ├── MultiAttributeHead(fpn_channels=[320,640,640])
    │                 │     │     src/models/heads.py
    │                 │     │     4 × 3 = 12 AttributeHead instances
    │                 │     └── attach_feature_hooks() [layers 15,18,21]
    │                 │
    │                 ├── Freeze backbone (requires_grad=False)
    │                 ├── ARCHONDataset(pseudo/labels_ext/train/)
    │                 │     src/data/dataset.py
    │                 │     └── ARCHONAugmentor()
    │                 │           src/data/augmentation.py
    │                 │           _augment_fliplr() with FLIP_CLASS_TABLE
    │                 │
    │                 ├── AttributeBCELoss(weight=8.0, pos_weight=auto)
    │                 │     src/training/loss.py
    │                 │
    │                 └── Training loop (50 epochs):
    │                       backbone.eval() → populate _fpn_features (no_grad)
    │                       attr_heads.train() → sample at (cx*W_f, cy*H_f)
    │                       loss.backward() → update attr_heads only
    │                       saves: weights/attr_best.pt
    │
    └─── [STAGE 5] stage_evaluate(args)
          └── src/inference/predictor.py
                ARCHONPredictor(weights/archon_best.pt)
                predictor.evaluate(config/dataset.yaml, split="val")
                → ultralytics YOLO.val()
                → logs mAP@0.5, mAP@0.5:0.95

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

After training, inference with --mode predict:

python main.py --mode predict --input xray.jpg

    src/inference/predictor.py → ARCHONPredictor
    ├── YOLO(archon_best.pt).predict(image)
    │     hooks fire → _fpn_features = [P3_feat, P4_feat, P5_feat]
    ├── MultiAttributeHead(_fpn_features)
    │     → attr_probs (N_det, 4) via per-tooth sampling
    ├── postprocess_yolo_output()
    │     src/inference/postprocess.py
    │     apply_linear_sum_assignment(class_probs, boxes, confs, attr_probs)
    │     └── scipy.linear_sum_assignment on 32×N cost matrix
    │         → List[ToothDetection]
    ├── draw_teeth_detections(image, teeth)
    │     src/utils/visualize.py
    │     → annotated JPEG (colored by quadrant, labeled Q:N:D)
    └── JSON result saved
```

---

*Generated from source analysis of the YOLOrtho codebase.*  
*Paper: Liu R. et al., "YOLOrtho — Unified Teeth Enumeration and Dental Disease Detection", arXiv:2308.05967*
