# ARCHON — Complete Technical Guide (Viva-Ready)

> **ARCHON**: **A**rch-**C**ontextualized **H**ierarchical **O**rthodontic **N**etwork for Severity-Aware Dental Radiograph Analysis.
>
> Built on top of the paper: [arXiv:2308.05967](https://arxiv.org/abs/2308.05967) (Dentex Challenge 2023).
>
> Dataset: [DENTEX Challenge 2023 on Kaggle](https://www.kaggle.com/competitions/dentex-challenge-2023).

---

## Table of Contents

1. [Project Overview — The Big Picture](#1-project-overview--the-big-picture)
2. [Dataset & FDI Tooth Numbering](#2-dataset--fdi-tooth-numbering)
3. [File-by-File Breakdown](#3-file-by-file-breakdown)
4. [File Connections & Function Call Flow](#4-file-connections--function-call-flow)
5. [Technical Concepts A–Z](#5-technical-concepts-az)
   - [5.1 YOLOv8 — The Base Detector](#51-yolov8--the-base-detector)
   - [5.2 Backbone, Neck, Head — The Pipeline Inside a Detector](#52-backbone-neck-head--the-pipeline-inside-a-detector)
   - [5.3 Feature Pyramid Network (FPN) & P3/P4/P5](#53-feature-pyramid-network-fpn--p3p4p5)
   - [5.4 CoordConv vs Normal Convolution](#54-coordconv-vs-normal-convolution)
   - [5.5 Swin Transformer — Global Context Encoder](#55-swin-transformer--global-context-encoder)
   - [5.6 Cross-Attention Fusion](#56-cross-attention-fusion)
   - [5.7 Hybrid Multi-Task Head (Severity + Quadrant)](#57-hybrid-multi-task-head-severity--quadrant)
   - [5.8 Linear Sum Assignment — Post-Processing](#58-linear-sum-assignment--post-processing)
   - [5.9 Hierarchical Loss & Data Types](#59-hierarchical-loss--data-types)
   - [5.10 Pseudo Labeling](#510-pseudo-labeling)
   - [5.11 Learning Rate, Optimizer & Scheduler](#511-learning-rate-optimizer--scheduler)
   - [5.12 Thresholds Explained](#512-thresholds-explained)
   - [5.13 Augmentation — Flip Mapping & CLAHE](#513-augmentation--flip-mapping--clahe)
   - [5.14 Letterboxing](#514-letterboxing)
   - [5.15 Anchor-Free Detection (YOLOv8)](#515-anchor-free-detection-yolov8)
   - [5.16 Distribution Focal Loss (DFL)](#516-distribution-focal-loss-dfl)
   - [5.17 BCE with pos_weight — Class Imbalance](#517-bce-with-pos_weight--class-imbalance)
   - [5.18 Label Smoothing](#518-label-smoothing)
   - [5.19 Flash Attention / Scaled Dot-Product Attention](#519-flash-attention--scaled-dot-product-attention)
   - [5.20 Forward Hooks (FPN Feature Capture)](#520-forward-hooks-fpn-feature-capture)
   - [5.21 Warmup Epochs](#521-warmup-epochs)
   - [5.22 Early Stopping (Patience)](#522-early-stopping-patience)
   - [5.23 Batch Normalization in Frozen Backbone](#523-batch-normalization-in-frozen-backbone)
   - [5.24 Depthwise Separable Convolution](#524-depthwise-separable-convolution)
   - [5.25 SiLU / Swish Activation](#525-silu--swish-activation)
   - [5.26 Multi-Scale Detection](#526-multi-scale-detection)
   - [5.27 NMS (Non-Maximum Suppression)](#527-nms-non-maximum-suppression)
   - [5.28 mAP (Mean Average Precision)](#528-map-mean-average-precision)
6. [Technical Decisions — Why This, Not That?](#6-technical-decisions--why-this-not-that)
7. [Alternatives & Their Problems](#7-alternatives--their-problems)
8. [What We Could Do Better (Latest Techniques)](#8-what-we-could-do-better-latest-techniques)
9. [Training Pipeline — Phase-by-Phase](#9-training-pipeline--phase-by-phase)
10. [Quick-Fire Viva Q&A](#10-quick-fire-viva-qa)

---

## 1. Project Overview — The Big Picture

### What does this project do?

Given a **panoramic dental X-ray** (OPG — Orthopantomogram), the system:

1. **Detects** every tooth (bounding box around it)
2. **Numbers** each tooth using the international FDI system (32 teeth, e.g., tooth 16 = Upper Right First Molar)
3. **Diagnoses** four diseases per tooth: Impaction, Caries, Deep Caries, Periapical Lesion
4. (Hybrid extension) **Grades severity**: Healthy / Mild / Severe per disease

**Real-life analogy**: Imagine a post office sorting machine that:
- Finds all letters in a pile (= detection)
- Reads the address on each letter (= FDI numbering)
- Flags damaged letters (= disease classification)
- Rates how damaged each letter is (= severity grading)

### Architecture at 30,000 feet

```
Panoramic X-ray (1280×640)
       │
   ┌───┴───┐
   │ YOLOv8x Backbone (with CoordConv) │  ← Extracts features at multiple scales
   └───┬───┘
       │
   ┌───┴───┐
   │ FPN Neck (PANet)                   │  ← Merges fine + coarse features → P3, P4, P5
   └───┬───┘
       │
       ├──── Detection Head ──── Bounding boxes + 32-class tooth ID
       │
       ├──── Attribute Heads ──── 4 binary disease flags per tooth (Baseline)
       │
       └──── (Hybrid additions) ──┬── Swin Transformer on P5 → global jaw context
                                  ├── Cross-Attention → fuse local+global
                                  └── Severity Head → 3-level grading + Quadrant aux
```

---

## 2. Dataset & FDI Tooth Numbering

### DENTEX Challenge 2023

The dataset has **3 parts** with increasing annotation richness:

| Part | Annotation Level | data_type | Teeth Count | What's Labeled |
|------|-----------------|-----------|-------------|----------------|
| Part 1 | Quadrant only | 0 | ~700 images | Bounding box + which jaw quadrant (1–4) |
| Part 2 | Quadrant + Enumeration | 1 | ~700 images | Bounding box + exact FDI tooth number |
| Part 3 | Quadrant + Enum + Disease | 2 | ~700 images | Bounding box + FDI + which diseases |
| Unlabelled | None | — | ~1000 images | No labels at all |
| Validation | Full (like Part 3) | 2 | ~100 images | Everything |

**Key problem**: Part 3 only labels **diseased teeth**. Healthy teeth in those images have no annotation. This is why we need pseudo-labeling (Section 5.10).

### FDI (Fédération Dentaire Internationale) System

```
          Upper Jaw
  ┌──────────┬──────────┐
  │  Q1 (UR) │  Q2 (UL) │
  │ 18←──→11 │ 21←──→28 │
  ├──────────┼──────────┤
  │  Q4 (LR) │  Q3 (LL) │
  │ 48←──→41 │ 31←──→38 │
  └──────────┴──────────┘
          Lower Jaw
```

- **First digit** = Quadrant (1=Upper Right, 2=Upper Left, 3=Lower Left, 4=Lower Right)
- **Second digit** = Position (1=Central Incisor ... 8=Wisdom Tooth)
- **Example**: FDI 36 = Lower Left First Molar

**YOLO class mapping**: FDI 11→class 0, FDI 12→class 1, ..., FDI 48→class 31.
So `class_idx // 8` gives the quadrant (0-3), and `class_idx % 8` gives the position (0-7).

### YOLO Extended Label Format (10 columns)

Each line in a `.txt` label file:
```
class_id  cx  cy  w  h  is_impacted  has_caries  has_deepcaries  has_lesion  data_type
   0     0.45 0.32 0.05 0.08    0          1           0             0          2
```
- `cx, cy` = normalized center coordinates (0 to 1)
- `w, h` = normalized width and height (0 to 1)
- Disease flags = 0 or 1
- `data_type` = 0 (quadrant only), 1 (enumeration), 2 (disease)

Standard YOLO uses only 5 columns (`class_id cx cy w h`). The extra 5 columns are our custom extension for disease attributes and hierarchical training.

---

## 3. File-by-File Breakdown

### Entry Point

| File | Purpose | Key Functions |
|------|---------|---------------|
| `main.py` | CLI entry point — dispatches to pipeline stages | `parse_args()`, `stage_preprocess()`, `stage_train()`, `stage_predict()`, `run_full_pipeline()` |

### Configuration Files

| File | Purpose | Key Settings |
|------|---------|-------------|
| `config/dataset.yaml` | Dataset paths, 32 class names, attribute names | `nc: 32`, `names: {0: "11", ...}`, `attribute_names` |
| `config/train_config.yaml` | All training hyperparameters | `lr0`, `phase1_epochs`, `batch_size`, `loss_box`, `loss_attr`, `augmentation` |
| `config/model_config.yaml` | Model architecture settings | `use_coordconv`, `swin_num_heads`, `fusion_num_heads`, `num_severity_levels` |

### Data Pipeline (`src/data/`)

| File | Purpose | Key Functions |
|------|---------|---------------|
| `preprocess.py` | Converts DENTEX COCO JSON → YOLO 10-column format | `preprocess_dentex()`, `_process_part1/2/3()` |
| `pseudo_label.py` | Generates healthy-tooth pseudo labels | `generate_pseudo_labels()`, `_extract_pseudo_labels()` |
| `dataset.py` | PyTorch Dataset with letterboxing + extended labels | `ARCHONDataset`, `_letterbox()`, `collate_fn()` |
| `augmentation.py` | Flip-mapping + CLAHE + geometric augmentation | `ARCHONAugmentor.__call__()`, `_augment_fliplr()`, `_augment_clahe()` |
| `download.py` | Dataset download utility | `download_dentex()` |

### Models (`src/models/`)

| File | Purpose | Key Classes |
|------|---------|------------|
| `yolortho.py` | Main model definitions — wraps YOLOv8 + all extensions | `ARCHON` (baseline), `ARCHONModel` (hybrid), `build_archon_base()`, `build_archon_model()` |
| `coord_conv.py` | CoordConv — position-aware convolution | `CoordConv`, `AddCoords`, `replace_backbone_conv_with_coordconv()` |
| `heads.py` | Binary disease attribute prediction heads | `AttributeHead`, `MultiAttributeHead`, `AttributeLoss` |
| `swin_transformer.py` | Swin Transformer for global jaw context | `GlobalContextEncoder`, `SwinTransformerBlock`, `WindowAttention` |
| `cross_attention.py` | Cross-attention fusion of CNN + Swin features | `CrossAttentionFusion`, `MultiScaleFusion` |
| `hybrid_head.py` | Severity grading + quadrant auxiliary head | `SeverityHead`, `HybridMultiTaskHead`, `SeverityLoss`, `QuadrantAuxLoss` |

### Training (`src/training/`)

| File | Purpose | Key Classes |
|------|---------|------------|
| `trainer.py` | 3-phase training pipeline | `ARCHONTrainer` (Phase 1+2+2b), `ARCHONHybridTrainer` (Phase 3) |
| `loss.py` | Hierarchical loss functions | `AttributeBCELoss`, `HierarchicalClassLoss`, `ARCHONLoss` |

### Inference (`src/inference/`)

| File | Purpose | Key Classes |
|------|---------|------------|
| `predictor.py` | Inference pipeline — loads model, runs prediction | `ARCHONPredictor`, `ARCHONHybridPredictor` |
| `postprocess.py` | Linear Sum Assignment for unique FDI assignment | `apply_linear_sum_assignment()`, `ToothDetection` dataclass |

### Utilities (`src/utils/`)

| File | Purpose | Key Functions |
|------|---------|---------------|
| `fdi.py` | FDI number ↔ YOLO class conversions | `fdi_to_class()`, `class_to_fdi()`, `flip_fdi()`, `FLIP_CLASS_TABLE` |
| `visualize.py` | Draw bounding boxes with Q/N/D labels on X-rays | `draw_teeth_detections()`, `draw_dental_chart()` |

---

## 4. File Connections & Function Call Flow

### Full Pipeline Call Graph

```
main.py: run_full_pipeline()
  │
  ├── stage_preprocess()
  │     └── src/data/preprocess.py: preprocess_dentex()
  │           ├── _process_part1() → reads train_quadrant.json → writes YOLO labels
  │           ├── _process_part2() → reads train_quadrant_enumeration.json
  │           ├── _process_part3() → reads train_quadrant_enumeration_disease.json
  │           ├── _process_validation() → reads validation_triple.json
  │           └── src/utils/fdi.py: quadrant_enum_to_fdi(), fdi_to_class()
  │
  ├── stage_train() → ARCHONTrainer.train()
  │     ├── _train_phase1()
  │     │     └── ultralytics YOLO("yolov8x.pt").train() → standard detection training
  │     │
  │     ├── _train_phase2()
  │     │     ├── ultralytics YOLO(phase1_best.pt).train() → fine-tune detection
  │     │     └── _train_attribute_heads()  ← Phase 2b
  │     │           ├── src/models/yolortho.py: build_archon_base()
  │     │           │     ├── ultralytics YOLO(base_weights)
  │     │           │     ├── src/models/coord_conv.py: replace_backbone_conv_with_coordconv()
  │     │           │     ├── src/models/heads.py: MultiAttributeHead()
  │     │           │     └── ARCHON.attach_feature_hooks()
  │     │           ├── src/data/dataset.py: ARCHONDataset()
  │     │           │     └── src/data/augmentation.py: ARCHONAugmentor()
  │     │           └── src/training/loss.py: AttributeBCELoss()
  │     │
  │     └── _promote_weights() → copies best.pt to weights/
  │
  ├── stage_pseudo_label()
  │     └── src/data/pseudo_label.py: generate_pseudo_labels()
  │           └── YOLO.predict() → detect teeth → filter by IoU → write pseudo labels
  │
  ├── stage_train_hybrid() → ARCHONHybridTrainer.train_hybrid()
  │     ├── src/models/yolortho.py: build_archon_model()
  │     │     ├── build_archon_base() first
  │     │     ├── src/models/swin_transformer.py: GlobalContextEncoder()
  │     │     ├── src/models/cross_attention.py: MultiScaleFusion()
  │     │     └── src/models/hybrid_head.py: HybridMultiTaskHead()
  │     └── Trains with SeverityLoss + QuadrantAuxLoss + AttributeBCELoss
  │
  └── stage_evaluate()
        └── src/inference/predictor.py: ARCHONPredictor.evaluate()
              ├── _load_models() → loads YOLO + attr heads
              ├── predict() → runs detection + attribute inference
              └── src/inference/postprocess.py: apply_linear_sum_assignment()
                    └── scipy.optimize.linear_sum_assignment()
```

### Key Inter-File Dependencies

```
coord_conv.py ──used by──→ yolortho.py (backbone injection)
heads.py ──used by──→ yolortho.py (attribute heads in ARCHON class)
swin_transformer.py ──used by──→ yolortho.py (GlobalContextEncoder in ARCHONModel)
cross_attention.py ──used by──→ yolortho.py (MultiScaleFusion in ARCHONModel)
hybrid_head.py ──used by──→ yolortho.py (HybridMultiTaskHead in ARCHONModel)
loss.py ──used by──→ trainer.py (AttributeBCELoss, SeverityLoss, QuadrantAuxLoss)
augmentation.py ──used by──→ dataset.py (ARCHONAugmentor inside ARCHONDataset)
fdi.py ──used by──→ preprocess.py, postprocess.py, visualize.py, augmentation.py
postprocess.py ──used by──→ predictor.py (linear sum assignment)
visualize.py ──used by──→ predictor.py (draw_teeth_detections for saved images)
dataset.py ──used by──→ trainer.py (Phase 2b and Phase 3 custom training loops)
```

---

## 5. Technical Concepts A–Z

### 5.1 YOLOv8 — The Base Detector

**What is it?**
YOLO (You Only Look Once) is a one-stage object detector. It processes the entire image in one pass and outputs all detections simultaneously.

**Real-life analogy**: Imagine looking at a group photo and instantly pointing at every person's face, saying their name, and noting if they're wearing glasses — all in one glance. That's YOLO. Older detectors (like R-CNN) would be like carefully scanning each face one by one.

**Why YOLOv8x specifically?**
- "v8" = version 8, the latest stable YOLO at time of development
- "x" = extra-large variant (68M parameters) — biggest and most accurate
- Variants: n (nano) < s (small) < m (medium) < l (large) < **x (extra-large)**
- We need the largest because 32-class fine-grained tooth classification requires more capacity than typical 80-class COCO detection

**YOLOv8 key features used by us:**
- Anchor-free detection (see 5.15)
- Decoupled head (separate branches for box regression and classification)
- Distribution Focal Loss (see 5.16)
- C2f modules (cross-stage partial bottleneck with 2 convolutions)
- SiLU activation (see 5.25)

### 5.2 Backbone, Neck, Head — The Pipeline Inside a Detector

Every modern object detector has three parts. Think of it like a factory:

```
Image → [BACKBONE] → [NECK] → [HEAD] → Detections
         "eyes"      "brain"   "mouth"
```

**Backbone** (layers 0–9 in YOLOv8):
- Extracts features from the raw image
- Like your eyes: sees edges, textures, shapes at different zoom levels
- Produces feature maps at decreasing spatial resolution but increasing semantic richness
- **Example**: At layer 3, it sees "there's a bright edge here". At layer 9, it sees "this is probably a molar tooth"
- We inject **CoordConv** here (Section 5.4) to add position awareness

**Neck** (layers 10–21 in YOLOv8):
- Merges features from different layers via FPN + PANet
- Like your brain: combines "I see a white blob" (fine detail) with "this is in the lower-left jaw" (big picture context)
- Produces P3, P4, P5 feature maps (Section 5.3)

**Head** (layer 22 in YOLOv8):
- Makes final predictions: bounding box + class
- Like your mouth: says "Tooth 36, here, with caries"
- We add custom attribute heads and hybrid heads on top of this

### 5.3 Feature Pyramid Network (FPN) & P3/P4/P5

**What is FPN?**

When a CNN processes an image, each layer reduces the spatial size but increases the "understanding". FPN takes features from multiple layers and merges them into a pyramid.

**Real-life analogy**: Imagine reading a map at three zoom levels:
- **P3** (stride 8, 160×80 at 1280×640 input): Street-level view — sees individual tooth edges, root tips, small cavities. **Fine detail, small receptive field.**
- **P4** (stride 16, 80×40): Neighborhood view — sees full teeth, gaps between teeth. **Medium detail.**
- **P5** (stride 32, 40×20): City view — sees the entire jaw, quadrant layout, overall tooth arrangement. **Coarse detail, large receptive field.**

**What do "stride" and "P" mean?**

- **Stride** = how much the image has been downsampled. Stride 8 means every pixel in P3 represents an 8×8 patch of the original image.
- **P** = Pyramid level. Higher P = more downsampled = sees bigger picture but less detail.

```
Original: 1280×640
P3 (stride 8):  160×80  ← sees small things (root tips, early caries)
P4 (stride 16):  80×40  ← sees medium things (full tooth crowns)
P5 (stride 32):  40×20  ← sees big things (jaw quadrants, tooth ordering)
```

**Why P3/P4/P5 matter for our attribute heads:**
Each attribute head operates at ALL three scales and the predictions are averaged. This way:
- A small periapical lesion at the root tip is best seen at P3
- A large impacted wisdom tooth is best seen at P5
- Caries (medium-sized) is seen at P4

**What is PANet?**
PANet (Path Aggregation Network) is the specific FPN variant used in YOLOv8. It adds a bottom-up path on top of the standard top-down FPN:
- **Top-down** (FPN): P5 → P4 → P3 (passes global context downward)
- **Bottom-up** (PANet): P3 → P4 → P5 (passes fine details upward)

This bidirectional merging ensures every scale has both fine details AND global context.

### 5.4 CoordConv vs Normal Convolution

**The Problem with Normal Convolution:**

Standard Conv2d is **translation-invariant** — it produces the same output regardless of WHERE in the image a feature appears. The same filter detects "tooth edge" identically whether it's in the top-left corner or bottom-right corner.

**Why this is bad for teeth:**
For teeth numbering, **position IS the answer**:
- A molar in the upper-right jaw is tooth 16
- The exact same-looking molar in the lower-left jaw is tooth 36
- Without knowing position, the model can't distinguish them

**Real-life analogy**: Normal convolution is like a security guard who can recognize faces but doesn't know where in the building they are. CoordConv is like giving that guard a GPS — now they can say "this person is at Desk 42 on Floor 3."

**How CoordConv works:**
```
Normal Conv input:  (B, C, H, W)        ← just pixel values
CoordConv input:    (B, C+2, H, W)      ← pixel values + x-coordinate + y-coordinate
```

Before the convolution, two extra channels are concatenated:
- **x-channel**: Each pixel gets its horizontal position, normalized from -1 (left) to +1 (right)
- **y-channel**: Each pixel gets its vertical position, normalized from -1 (top) to +1 (bottom)

```python
# Simplified view of what happens
xx = [-1.0, -0.5, 0.0, 0.5, 1.0]   # left → right
yy = [-1.0, -0.5, 0.0, 0.5, 1.0]   # top → bottom
input_with_coords = concat(original_features, xx_channel, yy_channel)
output = conv2d(input_with_coords)   # now the conv "knows where it is"
```

**Where is it applied?**
Only in the **backbone** (layers 0–9). Not in the neck or head. This is because:
1. The backbone extracts low-level features where position awareness helps most
2. The detection head structure should stay unchanged to preserve compatibility with ultralytics

**Optional radial channel (`with_r`)**:
A third coordinate channel `r = sqrt(x² + y²)` — distance from center. Useful if "proximity to jaw center" matters. We set `with_r: false` because horizontal/vertical position is more relevant than radial distance for FDI numbering.

**Reference**: Liu et al., *"An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution"*, NeurIPS 2018.

### 5.5 Swin Transformer — Global Context Encoder

**The Problem (Limitation 1 — FDI Numbering Conflicts):**
Even with CoordConv, the CNN backbone has a **limited receptive field**. At P5 (stride 32), each feature "sees" a region of ~32×32 pixels. A tooth at the midline of the jaw could be classified as tooth 11 (Q1) or 21 (Q2) — the CNN can't resolve this because it can't see the full jaw layout to count teeth.

**Real-life analogy**: Imagine trying to identify which house number you're at by looking through a keyhole. You can see the door, but you can't see the full street to count which house number it is. A Swin Transformer is like climbing up to a rooftop — now you can see the entire street and count.

**What is a Swin Transformer?**
A transformer that processes images using **shifted windows**.

Standard Vision Transformer (ViT):
- Divides image into patches
- Every patch attends to every other patch
- Cost: O(N²) where N = number of patches — **very expensive**

Swin Transformer:
- Divides image into **windows** (small groups of patches)
- Attention happens **only within each window** → O(window_size²) per window
- Alternates between **regular windows** and **shifted windows** to allow cross-window information flow

```
Regular Window:              Shifted Window (half-window offset):
┌────┬────┬────┬────┐       ┌──┬─────┬─────┬──┐
│ W1 │ W2 │ W3 │ W4 │       │  │ W1' │ W2' │  │
├────┼────┼────┼────┤       ├──┼─────┼─────┼──┤
│ W5 │ W6 │ W7 │ W8 │  →→   │  │ W3' │ W4' │  │
├────┼────┼────┼────┤       ├──┼─────┼─────┼──┤
│ W9 │W10 │W11 │W12 │       │  │ W5' │ W6' │  │
└────┴────┴────┴────┘       └──┴─────┴─────┴──┘

By shifting, the "edges" of block 1 overlap with parts of blocks 2 and 3.
```

**Our Usage — GlobalContextEncoder:**
- Operates **only on P5** (the coarsest, 20×40 at stride 32)
- Two Swin Transformer blocks: one regular (shift=0), one shifted (shift=window_size//2)
- Window size = 4 (4×4 patches per window)
- 8 attention heads
- Output: same shape as input (B, 640, H, W) — ready for cross-attention fusion

**Why P5 only?**
- P5 has the smallest spatial size (20×40 = 800 tokens) — affordable for attention
- P3 has 160×80 = 12,800 tokens — running full attention there would be too expensive
- P5 already captures the "big picture" (jaw quadrants, tooth ordering)

**What does the Swin branch learn?**
- Which quadrant each region belongs to
- How many teeth are on each side
- The spatial ordering of teeth (incisors near midline, molars at back)
- Whether there's a gap (missing tooth) that affects numbering

**Relative Position Bias:**
The Swin attention includes a learnable relative position bias table. This tells the model "a token 2 positions to the right is more related than one 5 positions away" — encoding spatial locality without absolute coordinates.

### 5.6 Cross-Attention Fusion

**What is Cross-Attention?**

In self-attention, a token attends to other tokens from the **same** source. In cross-attention, tokens from one source (Query) attend to tokens from a **different** source (Key/Value).

```
Standard Self-Attention:     Cross-Attention:
Q, K, V all from same input  Q from source A, K/V from source B

Q ──┐                        Q (CNN local features) ──┐
K ──┤── Attention             K (Swin global context) ──┤── Attention
V ──┘                        V (Swin global context) ──┘
```

**Real-life analogy**: Self-attention is like a group of students discussing among themselves. Cross-attention is like students (CNN features) asking a teacher (Swin features) questions. The students know the local details ("this pixel looks like a cavity"), and the teacher knows the big picture ("we're looking at the upper-right jaw, and there should be 8 teeth here").

**The formula:**
```
Q = W_q × CNN_features       (what each tooth is "asking about")
K = W_k × Swin_features      (what the jaw context "offers")
V = W_v × Swin_features      (the actual information to retrieve)
Attention = softmax(Q × K^T / √d_k) × V
```

**Multi-Scale Fusion:**
Cross-attention happens at ALL three FPN scales:
1. **P5 → P5**: Direct fusion (same resolution)
2. **P4 → P5 upsampled**: Swin output upsampled 2× to match P4
3. **P3 → P5 upsampled**: Swin output upsampled 4× to match P3

This way, even the finest features (P3) benefit from global jaw context.

**Residual Connection:**
The fused output is added back to the original CNN features:
```python
fused = CNN_features + CrossAttention(CNN_features, Swin_features)
```
This ensures the model never loses the original local features — the cross-attention only adds global context on top.

**Pre-LN (Pre-Layer Normalization):**
LayerNorm is applied **before** the attention, not after. This is more stable for training deep transformers (prevents gradient explosion).

### 5.7 Hybrid Multi-Task Head (Severity + Quadrant)

**Improvement C** replaces the simple binary attribute heads with a richer prediction system.

#### Severity Head (replaces binary AttributeHead)

**Problem**: Binary "has_caries = yes/no" misses the clinical spectrum. Early caries (demineralization) is very different from deep caries reaching the pulp.

**Solution**: 3-class severity per disease attribute:
```
Level 0: Healthy (no disease)
Level 1: Mild (early stage)
Level 2: Severe (advanced)
```

**Mapping from DENTEX binary labels to severity:**

| Attribute | Binary Label | Severity |
|-----------|-------------|----------|
| is_impacted | 0 | 0 (healthy) |
| is_impacted | 1 | 2 (severe — impaction is always a major finding) |
| has_caries | 1, deepcaries=0 | 1 (mild — just surface caries) |
| has_caries | 1, deepcaries=1 | 2 (severe — caries reached pulp) |
| has_deepcaries | 1 | 2 (severe by definition) |
| has_lesion | 0 | 0 |
| has_lesion | 1 | 2 (severe — periapical infection is serious) |

**Output bias initialization:**
```python
self.out.bias[0] = log(80)   # healthy is most common (~80%)
self.out.bias[1] = log(15)   # mild (~15%)
self.out.bias[2] = log(5)    # severe (~5%)
```
This ensures the model starts by predicting "healthy" for most teeth (which is the truth) and gradually learns to detect diseases.

#### Quadrant Auxiliary Head

**Purpose**: Gives the model an extra training signal specifically for quadrant classification.

**How it helps**: During post-processing (linear sum assignment), if the quadrant head predicts a tooth is in Q2 with high confidence, the cost of assigning it to any Q1/Q3/Q4 FDI slot is increased → fewer cross-quadrant assignment errors.

**Architecture**: Simple 2-conv + 1×1 output → 4 classes (one per quadrant).

### 5.8 Linear Sum Assignment — Post-Processing

**The Problem:**
YOLOv8 might predict:
- Detection A: 90% confidence it's tooth 16, 85% it's tooth 17
- Detection B: 88% confidence it's tooth 16, 80% it's tooth 17

Both detections want to be tooth 16! But each tooth number can only be assigned once.

**Real-life analogy**: A school assigns lockers to students. Each student ranks their preferred lockers. The school must find the assignment that maximizes overall happiness — no two students share a locker.

**The solution — Hungarian Algorithm (linear_sum_assignment):**

1. Build a **cost matrix** (32 rows × N columns):
   - Rows = FDI positions (0-31)
   - Columns = detections
   - `cost[i, j] = -log(prob[j, i] + ε)` (high prob → low cost → preferred)

2. Solve with `scipy.optimize.linear_sum_assignment` — finds the assignment that minimizes total cost.

3. Result: each FDI position gets at most one detection. No duplicates.

**Quadrant penalty (hybrid extension):**
If the hybrid quadrant head says detection j is in Q2 with 95% confidence, and FDI slot i belongs to Q1, add a penalty:
```
penalty[i, j] = alpha × (1 - quadrant_prob[j, quadrant_of_slot_i])
alpha = 0.5 × confidence_scale   (0 when uncertain, 0.5 when confident)
```

This prevents "midline confusion" where teeth near the jaw center get assigned to the wrong quadrant.

### 5.9 Hierarchical Loss & Data Types

**The core training challenge**: Different parts of the dataset have different levels of annotation. We can't compute disease loss on Part 1 data (which has no disease labels).

**Solution — Hierarchical loss with data_type masking:**

```
data_type = 0 (Part 1): Only compute bbox loss + quadrant-level class loss
                         (32 classes grouped into 4 quadrants → 4-class CE)
data_type = 1 (Part 2): Compute bbox loss + full 32-class FDI loss
data_type = 2 (Part 3): Compute bbox loss + FDI loss + disease attribute loss
```

**Real-life analogy**: A cooking exam with 3 groups:
- Group 0: Only graded on "did they make a soup?" (broad category)
- Group 1: Graded on "did they make tomato soup?" (specific recipe)
- Group 2: Graded on recipe + seasoning + presentation (everything)

**Total Loss formula:**
```
L_total = w_box × L_bbox + w_cls × L_class + w_dfl × L_DFL + w_attr × L_attr

Where:
  L_bbox  = CIoU loss (bounding box regression)     weight = 7.5
  L_class = Cross-Entropy (tooth class)              weight = 0.5
  L_DFL   = Distribution Focal Loss (box refinement) weight = 1.5
  L_attr  = Binary Cross-Entropy (disease attrs)     weight = 8.0
```

**Why is attr weight so high (8.0)?**
Disease detection is the primary clinical contribution. Without a high weight, the model focuses on detection accuracy (which is already good from COCO pretraining) and ignores the harder disease classification task.

### 5.10 Pseudo Labeling

**The Problem:**
Part 3 images only have annotations for **diseased teeth**. A Part 3 image might have 28 teeth visible, but only 3 are labeled (the diseased ones). If we train on this, the model is **penalized for correctly detecting the other 25 healthy teeth** (they're treated as false positives).

**The Solution:**
1. Train Phase 1 detector on Part 1+2 (which label ALL teeth)
2. Run this detector on Part 3 images
3. For each detection that **doesn't overlap** with an existing disease label (IoU < 0.3): create a pseudo label with all disease attributes = 0 (healthy)
4. Also run on unlabelled images → all detections become healthy pseudo labels
5. Merge pseudo labels with existing labels → complete dataset for Phase 2

**Real-life analogy**: A teacher gives students an incomplete answer key (only wrong answers marked). A senior student (Phase 1 model) fills in the remaining correct answers so that future students (Phase 2 model) can learn from a complete key.

**IoU threshold for overlap (0.3):**
If a new detection overlaps more than 30% with an existing disease annotation, we skip it (it's probably the same tooth, already labeled as diseased).

**Confidence threshold (0.5):**
Only accept pseudo labels with ≥50% detection confidence. Low-confidence detections might be false positives — we don't want to teach the model from wrong examples.

### 5.11 Learning Rate, Optimizer & Scheduler

#### Learning Rate (lr)

**What is it?**
The learning rate controls how big each step of learning is.

**Real-life analogy**: Walking toward a target in the dark:
- **lr = 0.1** (high): Big steps → reach the target area fast, but might step past it
- **lr = 0.001** (low): Tiny steps → precise, but takes forever to get there
- **lr = 0.01** (our Phase 1 default): Balanced — reasonably fast, reasonably precise

**Our learning rates:**
```
Phase 1: lr0 = 0.01    (training from scratch → need bigger steps to escape random weights)
Phase 2: lr0 = 0.002   (fine-tuning from Phase 1 → smaller steps to preserve what was learned)
Phase 2b: lr = 1e-3     (attribute heads only → medium steps for new heads)
Phase 3: lr = 5e-4      (hybrid components only → careful steps to not disturb backbone)
```

**Why Phase 2 lr is lower (0.002 vs 0.01)?**
Phase 2 starts from Phase 1's best weights (already a good detector). Using lr=0.01 would "forget" what was learned — the model would reach best mAP at epoch 2 then oscillate. With lr=0.002, it gently improves without catastrophic forgetting.

#### Optimizer: SGD vs AdamW

**SGD (Stochastic Gradient Descent)** — Our choice for Phase 1 & 2:
- Simple: `weight_new = weight_old - lr × gradient`
- With momentum (0.937): adds "inertia" from previous updates, smoothing out noisy gradients
- Better generalization on vision tasks (proven by YOLO authors)

**AdamW** — Our choice for Phase 2b & 3:
- Adaptive per-parameter learning rates
- Better for fine-tuning small modules (attribute heads, hybrid heads)
- `weight_decay = 1e-4` prevents overfitting

**Why SGD for detection, AdamW for heads?**
- Detection training benefits from SGD's generalization properties (well-established in YOLO literature)
- Small head modules with few parameters converge faster with Adam's adaptive rates
- AdamW's per-parameter adaptation is especially useful when different parts of the model need different effective learning rates

#### Cosine Annealing Scheduler

**What it does**: Smoothly decreases the learning rate following a cosine curve.

```
lr(t) = lr_min + 0.5 × (lr_max - lr_min) × (1 + cos(π × t / T_max))
```

```
lr
 │ ╲
 │   ╲
 │    ╲___              ← starts high, decays smoothly
 │        ╲__
 │            ╲____
 └──────────────────→ epochs
```

**Why cosine and not step-decay?**
- Step-decay (e.g., divide lr by 10 at epoch 50, 80) causes sudden jumps that can destabilize training
- Cosine is smooth — the model gracefully transitions from exploration (high lr) to refinement (low lr)

**Our settings:**
```
Phase 1/2: lrf = 0.01 → final lr = lr0 × 0.01 (e.g., 0.01 → 0.0001)
Phase 2b: CosineAnnealingLR(T_max=epochs, eta_min=1e-5) → 1e-3 → 1e-5
Phase 3: CosineAnnealingLR(T_max=50, eta_min=1e-6) → 5e-4 → 1e-6
```

### 5.12 Thresholds Explained

| Threshold | Value | Where Used | What It Controls |
|-----------|-------|-----------|-----------------|
| `conf_threshold` | 0.25 (default) | Inference | Minimum detection confidence to keep a prediction. Below this → discarded. |
| `iou_threshold` | 0.45 | NMS | If two boxes overlap by >45%, the weaker one is suppressed (they're probably the same tooth). |
| `attr_threshold` | 0.3 | Disease prediction | Minimum sigmoid probability to flag a disease as present. Below 0.3 → "healthy". |
| `pseudo_label_conf` | 0.5 | Pseudo labeling | Minimum confidence for a pseudo-labeled tooth. Higher = fewer but more reliable labels. |
| `overlap_iou_threshold` | 0.3 | Pseudo labeling | If detection overlaps >30% with existing disease label, skip it (already labeled). |
| `severity_threshold` | 0.4 | Hybrid severity | Minimum P(non-healthy) to flag disease via severity head. |
| `quadrant_penalty_alpha` | 0.5 | Hybrid post-process | Max penalty weight for cross-quadrant FDI assignment in cost matrix. |

**How to tune thresholds (trade-offs):**

```
                    ↑ Sensitivity (find more diseases)
                    │
   conf=0.05 ●─────┤
   attr=0.05        │
                    │     conf=0.15 ●
                    │     attr=0.08
                    │
                    │              conf=0.25 ●
                    │              attr=0.30
                    │
                    └────────────────────────→ Specificity (fewer false alarms)
```

- **Lower thresholds** = more detections, more disease flags → high sensitivity (good for screening)
- **Higher thresholds** = fewer detections, fewer false alarms → high specificity (good for diagnosis)
- For clinical screening: use conf=0.05, attr=0.05 (catch everything)
- For clinical diagnosis: use conf=0.25, attr=0.30 (be precise)

### 5.13 Augmentation — Flip Mapping & CLAHE

#### Flip Mapping (Baseline — Section 2.1 of paper)

**The Problem:**
When you horizontally flip a dental X-ray, the left side becomes the right side. A tooth that was FDI 16 (Upper Right First Molar) becomes FDI 26 (Upper Left First Molar). Standard augmentation libraries flip the image and bounding boxes but DON'T change the class labels.

**The Solution:**
When flipping horizontally:
1. Flip the image and bounding box x-coordinates: `cx_new = 1 - cx`
2. **Remap the class labels**: Q1 ↔ Q2, Q3 ↔ Q4
   - FDI 16 (class 5) → FDI 26 (class 13)
   - FDI 31 (class 16) → FDI 41 (class 24)

```python
FLIP_QUADRANT_MAP = {1: 2, 2: 1, 3: 4, 4: 3}
# FDI 16 → quadrant 1, position 6 → flip quadrant → quadrant 2, position 6 → FDI 26
```

**Why no vertical flip (flipud=0.0)?**
The skull anatomy dictates that upper jaw is always up and lower jaw is always down. Flipping vertically would create unrealistic images that confuse the model.

**Why no mosaic augmentation (mosaic=0.0)?**
Mosaic stitches 4 images into a grid. Panoramic X-rays are a continuous view of the jaw — stitching pieces from different patients creates nonsensical anatomy. The model would learn to detect teeth at junctions that don't exist.

#### CLAHE (Improvement D)

**What is CLAHE?**
Contrast Limited Adaptive Histogram Equalization — a technique that enhances local contrast.

**Real-life analogy**: Imagine a photo where the left half is too dark and the right half is too bright. Global brightness adjustment would fix one side but ruin the other. CLAHE divides the image into small tiles and adjusts each tile independently, making dark areas brighter and bright areas darker — locally.

**Why for dental X-rays?**
X-rays have **non-uniform exposure**:
- Bone areas (jawbone) appear very bright
- Soft tissue areas appear dark
- Root tips (where lesions occur) are often in the transition zone

Without CLAHE, early caries (subtle brightness change) and periapical lesions (dark halos at root tips) are barely visible. CLAHE makes these subtle features stand out.

**Settings:**
```python
clipLimit = 2.0    # Max amplification factor (2.0 is conservative — prevents noise amplification)
tileGridSize = 8×8 # Each 8×8 tile gets its own histogram (balances local vs global equalization)
prob = 0.5         # Applied randomly to 50% of training images (prevents over-sharpening)
```

**Applied on the L (luminance) channel only** — X-rays are grayscale, so we convert to LAB color space, apply CLAHE to the L channel, and convert back. This avoids color artifacts.

### 5.14 Letterboxing

**What is it?**
Resizing an image to fit a target size while preserving aspect ratio, then padding the remaining space with gray pixels (value 114).

**Real-life analogy**: Fitting a wide-screen movie on a square TV — you get black bars at top and bottom. The movie keeps its proportions.

**Why not just stretch?**
Panoramic X-rays are wide (2400×1200, ~2:1 ratio). Stretching to 1280×640 (2:1) works, but some X-rays have slightly different ratios. Stretching would distort tooth shapes. Letterboxing maintains anatomical proportions.

**Critically important for our project:**
Phase 2b training uses a custom dataset loader with letterboxing. If inference uses different preprocessing (ultralytics default), the FPN features at each spatial position represent a different anatomical location → attribute heads predict wrong diseases. We match letterboxing in both training and inference to ensure consistency.

```python
# Example: 2400×1200 image → 1280×640 target
# Scale = min(640/1200, 1280/2400) = min(0.533, 0.533) = 0.533
# New size: 1279×640 → pad 1px on left side → centered
```

### 5.15 Anchor-Free Detection (YOLOv8)

**What are anchors (old approach)?**
In YOLOv5 and earlier, the model uses predefined "anchor boxes" — templates of different sizes and aspect ratios placed at each grid cell. The model predicts offsets from these templates.

**Problem with anchors:**
- Need to manually choose anchor sizes (usually via k-means clustering on dataset)
- Different tooth types have different aspect ratios — 32 classes × multiple scales = many anchors
- Anchor mismatch hurts performance

**YOLOv8's anchor-free approach:**
- Each grid cell directly predicts the bounding box center offset and width/height
- No predefined templates — the model learns the entire box from scratch
- Simpler, more flexible, works better for our 32-class tooth detection

### 5.16 Distribution Focal Loss (DFL)

**What is it?**
Instead of predicting a single bounding box edge coordinate, YOLOv8 predicts a **probability distribution over possible edge positions**.

**Real-life analogy**: Instead of saying "the tooth edge is exactly at pixel 127", the model says "there's a 60% chance it's at pixel 127, 30% at pixel 128, 10% at pixel 126". The distribution captures uncertainty.

**Why it helps teeth:**
Tooth boundaries are often blurry on X-rays (especially at root tips and areas overlapping with adjacent teeth). DFL lets the model express this ambiguity rather than being forced to pick a single position.

**Weight: 1.5** (from paper). Balances DFL against the main CIoU box loss (7.5).

### 5.17 BCE with pos_weight — Class Imbalance

**The Problem:**
In dental X-rays, most teeth are healthy. If 85% of teeth are healthy:
- A model that always predicts "healthy" gets 85% accuracy!
- But it's clinically useless — it misses all diseases

**Real-life analogy**: A fire alarm that never goes off has 99.99% accuracy (fires are rare). But it's useless.

**The Solution — pos_weight in BCEWithLogitsLoss:**

`pos_weight` upweights false negatives (missed diseases):
```
Normal BCE:     L = -[y × log(p) + (1-y) × log(1-p)]
With pos_weight: L = -[W × y × log(p) + (1-y) × log(1-p)]
```

When `W > 1`, missing a diseased tooth (false negative) is penalized W× more than falsely flagging a healthy tooth.

**Our pos_weight values:**
```
is_impacted:   5.0  (15% prevalence → ratio ≈ 5.7)
has_caries:    3.0  (25% prevalence → ratio ≈ 3.0)
has_deepcaries: 8.0  (8% prevalence → ratio ≈ 11.5)
has_lesion:    5.0  (15% prevalence → ratio ≈ 5.7)
```

Deep caries has the highest pos_weight (8.0) because it's the rarest condition — without upweighting, the model would almost never predict it.

### 5.18 Label Smoothing

**What is it?**
Instead of training with hard labels (0 or 1), soften them slightly:
```
Hard: target = [0, 0, 1, 0]   (class 2 is correct)
Smooth (ε=0.1): target = [0.025, 0.025, 0.925, 0.025]
```

**Why?**
- Prevents overconfident predictions
- Improves generalization (model doesn't memorize training labels)
- Especially important for severity classification where the boundary between "mild" and "severe" is fuzzy

**Used in**: `SeverityLoss` with `label_smoothing=0.1`.

### 5.19 Flash Attention / Scaled Dot-Product Attention

**What is it?**
PyTorch's `F.scaled_dot_product_attention` automatically uses **Flash Attention** when available on the GPU. Flash Attention computes the attention output without materializing the full N×N attention matrix in memory.

**Why it matters:**
At P3 scale, N = 160×80 = 12,800 tokens. The full attention matrix would be 12,800² = ~164 million entries × 4 bytes = 654 MB — per batch, per head! Flash Attention avoids this by computing attention in tiles.

**Used in**: `CrossAttentionFusion.forward()` for the cross-attention computation.

### 5.20 Forward Hooks (FPN Feature Capture)

**The Problem:**
We need the intermediate FPN features (P3, P4, P5) to feed into our attribute heads. But ultralytics' YOLO model doesn't expose them — it only returns the final detection output.

**The Solution — PyTorch forward hooks:**
```python
def hook(module, input, output):
    captured_features.append(output)

layer.register_forward_hook(hook)
```

A hook is a callback function that runs every time a specific layer produces its output. We attach hooks to layers 15, 18, and 21 (the FPN output layers in YOLOv8's 23-layer architecture) to capture P3, P4, P5.

**Real-life analogy**: Putting a camera at three points along a factory assembly line to photograph the product at each stage, without stopping the line.

### 5.21 Warmup Epochs

**What is it?**
During the first few epochs, the learning rate starts very low and gradually increases to the target lr.

```
lr
 │          ╱──────────────╲
 │        ╱                  ╲
 │      ╱                      ╲
 │    ╱  ← warmup                ╲ ← cosine decay
 │  ╱     (3 epochs)               ╲
 └──────────────────────────────────→ epochs
```

**Why?**
At the start of training, the weights are random. Large gradients from random weights + high learning rate = chaotic updates that can diverge. Warmup gives the model a "gentle start".

**Settings:**
```
warmup_epochs: 3.0
warmup_momentum: 0.8       (momentum ramps up from 0.8 to 0.937)
warmup_bias_lr: 0.1        (bias parameters use higher lr during warmup)
```

### 5.22 Early Stopping (Patience)

**What is it?**
Stop training if the validation metric (mAP) hasn't improved for `patience` consecutive epochs.

**Our setting: patience = 50**

**Why 50 (and not lower)?**
- Detection models often plateau for 20-30 epochs before finding a new improvement
- The cosine scheduler naturally reduces lr over time, which can unlock new improvements later
- Setting patience too low (e.g., 10) might stop training right before a breakthrough

### 5.23 Batch Normalization in Frozen Backbone

**Critical detail in Phase 2b & 3:**

When we freeze the backbone and only train attribute/hybrid heads, we set the backbone to `eval()` mode:
```python
model.base_model.model.eval()   # backbone BN uses running statistics
model.attr_heads.train()         # only attr heads in training mode
```

**Why?**
BatchNorm in training mode normalizes using **per-batch statistics** (mean/var of current batch). In eval mode, it uses **running statistics** (accumulated over all training batches).

If the backbone is in training mode with small batches (batch_size=4), the per-batch statistics are noisy → FPN features fluctuate → attribute heads learn from inconsistent features → poor disease prediction at inference time (when BN always uses running statistics).

**Rule**: Frozen layers must be in eval mode. Only trainable layers should be in train mode.

### 5.24 Depthwise Separable Convolution

Used in the attribute heads for efficiency.

**Normal 3×3 Conv**: For C_in=256 → C_out=64: parameters = 256 × 64 × 3 × 3 = 147,456

**Depthwise Separable**:
1. Depthwise: 256 separate 3×3 filters (one per channel) = 256 × 3 × 3 = 2,304
2. Pointwise: 1×1 conv 256→64 = 256 × 64 = 16,384
3. Total = 18,688 — **8× fewer parameters**

Our attribute heads use standard Conv2d (not depthwise) because they're already small — the computational savings aren't worth the accuracy loss for such lightweight heads.

### 5.25 SiLU / Swish Activation

**What is it?**
```
SiLU(x) = x × sigmoid(x)
```

**Why not ReLU?**
- ReLU kills all negative values (`max(0, x)`) — information lost
- SiLU smoothly diminishes negative values — keeps some information
- Better gradient flow for deep networks (no "dead neurons")
- YOLOv8's default activation; we keep it for consistency

### 5.26 Multi-Scale Detection

YOLOv8 detects objects at three different scales simultaneously:

| Scale | Stride | Feature Map Size (1280×640 input) | Best For |
|-------|--------|----------------------------------|----------|
| P3 | 8 | 160×80 | Small objects (root tips, small cavities) |
| P4 | 16 | 80×40 | Medium objects (full teeth) |
| P5 | 32 | 40×20 | Large objects (wisdom teeth, full arch context) |

Each scale's feature map runs through the same detection head but at different resolutions. A small periapical lesion might only be visible at P3, while a large impacted wisdom tooth is best detected at P5.

### 5.27 NMS (Non-Maximum Suppression)

**What is it?**
After the model produces thousands of candidate detections, many overlap. NMS removes duplicate detections by:
1. Sort all detections by confidence (highest first)
2. Keep the top detection
3. Remove any other detection that overlaps with it by more than `iou_threshold` (0.45)
4. Repeat for the next remaining detection

**Real-life analogy**: A photo contest — if two nearly-identical photos (>45% overlap) are submitted, keep only the better-scored one.

### 5.28 mAP (Mean Average Precision)

**What is it?**
The standard metric for object detection quality.

1. For each class, compute **Average Precision (AP)**: the area under the precision-recall curve
2. Average all 32 per-class APs → **mAP**

**mAP@0.5**: IoU threshold of 0.5 (detection counts as correct if box overlaps GT by ≥50%)
**mAP@0.5:0.95**: Average over IoU thresholds from 0.5 to 0.95 (stricter — box must be very precise)

---

## 6. Technical Decisions — Why This, Not That?

### Why YOLOv8x and not a smaller variant?

| Variant | Params | mAP@0.5 on COCO | Our choice? |
|---------|--------|-----------------|-------------|
| YOLOv8n | 3.2M | 37.3 | ❌ Too small for 32-class fine-grained |
| YOLOv8s | 11.2M | 44.9 | ❌ |
| YOLOv8m | 25.9M | 50.2 | ❌ |
| YOLOv8l | 43.7M | 52.9 | ❌ |
| **YOLOv8x** | **68.2M** | **53.9** | ✅ Best accuracy; 32 visually similar classes need more capacity |

**Trade-off**: Slower inference (~10ms vs ~3ms per image on GPU). But dental X-ray analysis is not real-time — 10ms is perfectly acceptable.

### Why CoordConv and not Positional Encoding?

| Approach | How it adds position info | Pros | Cons |
|----------|--------------------------|------|------|
| **CoordConv** (our choice) | Concatenates (x,y) channels before conv | Simple, efficient, works in CNN backbone | Only 2 extra channels |
| Positional Encoding | Adds sinusoidal or learned vectors to features | Rich position info | Designed for transformers, not CNNs |
| Absolute Position Embedding | Learned per-position vector | High capacity | Doesn't generalize to different image sizes |

We chose CoordConv because it's a **drop-in replacement** for Conv2d — no architecture changes needed. It just adds 2 extra input channels and the rest is handled by the learnable weights.

### Why Swin Transformer and not ViT or DeiT?

| Model | Attention Type | Complexity | Our choice? |
|-------|---------------|-----------|-------------|
| ViT | Global (all patches attend to all) | O(N²) — expensive | ❌ Too expensive at high resolution |
| DeiT | Global + distillation token | O(N²) | ❌ Same complexity issue |
| **Swin** | Windowed (local attention + shifting) | O(N × window²) | ✅ Efficient, hierarchical, perfect for our P5 feature map |

At P5, N=20×40=800 tokens, so even ViT would work. But Swin's window-based attention is:
1. More efficient if we ever want to apply it at P3/P4
2. Naturally captures local spatial relationships (nearby teeth interact more)
3. The shifted window mechanism provides cross-window communication

### Why Cross-Attention and not Self-Attention?

| Method | What it does | Our choice? |
|--------|-------------|-------------|
| Self-Attention on CNN features | CNN features attend to each other | ❌ Doesn't bring in NEW information |
| Self-Attention on Swin features | Global features attend to each other | Already done inside Swin |
| **Cross-Attention (CNN→Swin)** | CNN features query Swin for global context | ✅ Each local tooth feature gets enriched with jaw-wide context |

Self-attention on CNN features alone can't fix FDI conflicts because the CNN features don't contain sufficient global context. Cross-attention is the bridge between "local tooth knowledge" (CNN) and "global jaw knowledge" (Swin).

### Why 3-level Severity and not 5 or continuous?

- **Binary (2-level)**: Too coarse — early caries and deep caries are very different clinically
- **3-level**: Matches clinical practice (healthy/mild/severe) and available annotation granularity
- **5-level**: Would need finer-grained annotations that DENTEX doesn't provide
- **Continuous regression**: Requires continuous severity scores in the dataset — not available

### Why SGD for detection and AdamW for heads?

- YOLOv8 is heavily optimized and benchmarked with SGD + cosine LR
- Changing to Adam for the full model would diverge from the well-tested ultralytics pipeline
- For our small custom heads (attribute/severity), Adam converges faster and is more stable
- Using different optimizers for different components = best of both worlds

### Why batch_size = 4?

- Our input is 1280×640 (larger than typical 640×640 YOLO)
- YOLOv8x is the largest variant → more memory per sample
- T4 GPU (16GB VRAM): batch=4 fits; batch=8 often OOM
- Batch size affects BatchNorm statistics — batch=4 is the minimum for stable BN

### Why imgsz = 1280×640 and not 640×640?

Panoramic X-rays have ~2:1 aspect ratio. Using 640×640 would either:
- **Stretch** the image → distort tooth shapes
- **Crop** to square → lose half the teeth
- **Letterbox to 640×640** → waste 50% of pixels on padding

1280×640 preserves the natural aspect ratio. Teeth look correct; features at each position match anatomical reality.

---

## 7. Alternatives & Their Problems

### Alternative 1: Two-Stage Detectors (Faster R-CNN, Cascade R-CNN)

**How they work**: Stage 1 proposes regions of interest (RoI), Stage 2 classifies each region.

**Pros**:
- Generally higher accuracy for small objects
- Cascade R-CNN can have even higher precision with multi-stage refinement

**Problems for our case**:
- **Slow**: 2-5× slower than YOLO for inference
- **Complex pipeline**: Region proposal → RoI pooling → classification is harder to extend with custom heads
- **Overkill**: Teeth are large, clearly visible objects — we don't need the extra accuracy of RPN for finding them
- **Not well-supported**: YOLO's ultralytics framework gives us training, evaluation, export in one package

### Alternative 2: DETR (DEtection TRansformer)

**How it works**: End-to-end transformer — no NMS, no anchors. Uses Hungarian matching during training.

**Pros**:
- Elegant — no post-processing needed
- Good at avoiding duplicate detections (built-in)

**Problems for our case**:
- **Slow to converge**: DETR needs 500+ epochs to train (we use 100)
- **Poor on small objects**: Teeth at the edge of the X-ray can be small — DETR struggles
- **Hard to extend**: Adding custom attribute heads to DETR's decoder is non-trivial
- **Memory hungry**: Full attention over all image tokens at all times

### Alternative 3: Segmentation-based (Mask R-CNN, U-Net)

**How it works**: Instead of bounding boxes, predict pixel-level masks for each tooth.

**Pros**:
- More precise tooth boundaries
- Can separate overlapping teeth

**Problems for our case**:
- **DENTEX provides bounding box annotations**, not masks
- **Labeling masks is expensive** — manual per-pixel annotation for 32 tooth types × disease attributes
- **Overkill for disease classification** — we just need to know which tooth has caries, not the exact caries boundary
- **Slower inference** than bounding box detection

### Alternative 4: Classification-only (ResNet, EfficientNet)

**How it works**: Classify the entire image as "has caries" or crop each tooth and classify individually.

**Problems**:
- **No localization** — doesn't tell you WHICH tooth has the disease
- **Pre-cropping requires a detector anyway** — so you'd need YOLO + ResNet, which is more complex than YOLO + attribute heads
- **Loses spatial context** — a cropped tooth image doesn't show neighboring teeth, jaw structure, etc.

### Alternative 5: Using ConvNeXt instead of Swin Transformer

**ConvNeXt** is a pure CNN architecture that achieves Swin-level performance.

**Why we didn't use it**:
- ConvNeXt still has limited receptive field (just bigger kernels, not true global attention)
- Swin's shifted window attention explicitly models long-range dependencies (essential for jaw-wide context)
- ConvNeXt would need deeper stacking to achieve the same global coverage → more parameters

### Alternative 6: YOLOv9 or YOLOv10

**YOLOv9** (GELAN, PGI) and **YOLOv10** (NMS-free) are newer versions.

**Why we used v8:**
- v8 was the latest stable version when the project started
- v8 has the best ultralytics framework support and community
- v9/v10 improvements (GELAN for efficiency, NMS-free) don't address our core challenges (position awareness, disease classification)
- Upgrading would require re-validating the entire pipeline without guaranteed benefit

---

## 8. What We Could Do Better (Latest Techniques)

### 8.1 YOLOv11 / RT-DETR v2

**What**: Latest YOLO iteration with improved C3k2 modules, or Real-Time DETR with deformable attention.

**How it helps**: Better accuracy-speed trade-off.

**Problem**: Would require rebuilding the entire training pipeline and re-validating all custom extensions.

### 8.2 SAM (Segment Anything Model) for Tooth Segmentation

**What**: Meta's SAM can segment any object with a single prompt (click or box).

**How it helps**: Use YOLO detections as prompts → get precise tooth masks → better feature sampling for disease classification.

**Problem**: SAM is a general model — not trained on X-rays. Fine-tuning SAM requires segmentation masks we don't have. Also adds significant inference time.

### 8.3 Knowledge Distillation (Teacher-Student)

**What**: Train a large "teacher" model (YOLOv8x), then train a smaller "student" model (YOLOv8s) to mimic it.

**How it helps**: Smaller model with similar accuracy → faster inference, lower memory.

**Problem**: Distillation for multi-task models (detection + attributes + severity) is complex. The loss function needs careful balancing between detection distillation and attribute distillation.

### 8.4 Test-Time Augmentation (TTA)

**What**: Run inference on multiple augmented versions of the same image (original, flipped, scaled) and merge predictions.

**How it helps**: Can improve mAP by 1-3%.

**Problem**: 3-5× slower inference. For clinical use, the accuracy gain isn't worth the latency increase.

### 8.5 Self-Supervised Pre-training (MAE, DINOv2)

**What**: Pre-train the backbone on a large corpus of unlabeled dental X-rays using self-supervised learning.

**How it helps**: The backbone learns dental-specific features (tooth shapes, bone patterns, root structures) before any supervised training → better starting point than ImageNet pre-training.

**Problem**: Requires a large unlabeled dental X-ray dataset (>10,000 images). DENTEX provides only ~1,000 unlabeled images — not enough for effective self-supervised pre-training.

### 8.6 Deformable Attention (instead of Window Attention)

**What**: Attention that attends to a learned sparse set of positions (not all positions or a fixed window).

**How it helps**: More flexible than Swin's fixed windows — can attend to anatomically relevant positions regardless of where they fall in the window grid.

**Problem**: More complex implementation, harder to debug, and the benefit over Swin is marginal for our structured dental X-ray layout (teeth are always in a predictable arc).

### 8.7 Vision-Language Models (ClinicalBERT + Detection)

**What**: Combine visual detection with text understanding — generate dental reports from X-rays.

**How it helps**: End-to-end radiograph analysis → clinical report.

**Problem**: Requires paired (X-ray, report) training data. DENTEX only provides structured annotations, not free-text reports. Would be a different project entirely.

---

## 9. Training Pipeline — Phase-by-Phase

### Phase 1: Detection Pre-training

```
Goal:    Teach the model to FIND and NUMBER teeth (no disease yet)
Data:    Part 1 (quadrant) + Part 2 (enumeration)
Model:   Vanilla YOLOv8x (no CoordConv yet)
Loss:    CIoU bbox (w=7.5) + CE class (w=0.5) + DFL (w=1.5)
Epochs:  100
LR:      0.01 → 0.0001 (cosine decay)
Output:  phase1_best.pt (good tooth detector)
```

**Why no CoordConv in Phase 1?**
Ultralytics internally rebuilds the model from its YAML config during `.train()`. If we inject CoordConv before training, the state dict keys change (`model.0.conv.weight` → `model.0.conv.conv.weight`), causing a KeyError. CoordConv is injected in Phase 2b where we control the training loop.

### Pseudo Labeling (between Phase 1 and Phase 2)

```
Goal:    Fill in healthy tooth annotations for Part 3 images
Tool:    Phase 1 model (phase1_best.pt)
Method:  Detect teeth in Part 3 images → filter by IoU with existing labels → add as healthy pseudo labels
Output:  data/pseudo/ (merged dataset with complete annotations)
```

### Phase 2: Full Detection Training

```
Goal:    Train on ALL data including pseudo-labeled images
Data:    Part 1 + Part 2 + Part 3 (with pseudo labels) + unlabelled (pseudo)
Model:   YOLOv8x initialized from phase1_best.pt
Loss:    Same as Phase 1, but applied to all data
Epochs:  100
LR:      0.002 (lower — fine-tuning)
Output:  phase2_best.pt (better tooth detector, trained on more data)
```

### Phase 2b: Attribute Head Training

```
Goal:    Train disease classification heads (backbone frozen)
Data:    Same as Phase 2 (10-column labels with disease flags)
Model:   ARCHON = YOLOv8x + CoordConv + AttributeHeads
         Backbone: FROZEN (from phase2_best.pt, eval mode)
         Only attr_heads: TRAINABLE
Loss:    AttributeBCELoss(w=8.0, pos_weight=[5, 3, 8, 5])
Optim:   AdamW(lr=1e-3, weight_decay=1e-4)
Schedule: CosineAnnealingLR(T_max=50, eta_min=1e-5)
Epochs:  50
Output:  attr_best.pt (disease classification weights)

Training loop per batch:
  1. Forward pass through FROZEN backbone → capture FPN features via hooks
  2. Detach FPN features (no gradient flows to backbone)
  3. Run attr_heads on detached features → attribute logits
  4. Sample logits at each GT tooth's center (per-tooth supervision)
  5. Compute AttributeBCELoss only on data_type=2 samples
  6. Backpropagate through attr_heads only
```

**Per-tooth supervision** (critical fix):
Previously, training used global spatial average → image-level labels ("any tooth in this image has caries"). This caused every tooth to be predicted as diseased.
Fix: Sample the attribute feature map at each GT tooth's (cx, cy) position in the feature map, exactly as inference does at each detected box center.

### Phase 3: Hybrid Training

```
Goal:    Train Swin Transformer, Cross-Attention, Severity Head, Quadrant Head
Data:    Same as Phase 2
Model:   ARCHONModel = ARCHON + GlobalContextEncoder + MultiScaleFusion + HybridMultiTaskHead
         Backbone + Detection head: FROZEN
         Swin + CrossAttention + SeverityHead + QuadrantHead: TRAINABLE
Loss:    AttributeBCE(w=4) + SeverityLoss(w=4) + QuadrantAuxLoss(w=1)
Optim:   AdamW(lr=5e-4, weight_decay=1e-4)
Schedule: CosineAnnealingLR(T_max=50, eta_min=1e-6)
Epochs:  50
Output:  hybrid_best.pt
```

### Weight Hierarchy

```
weights/
├── phase1_best.pt       ← Phase 1 (detection pre-train)
├── phase1_last.pt       ← Phase 1 last epoch
├── phase2_best.pt       ← Phase 2 (full detection)
├── phase2_last.pt       ← Phase 2 last epoch
├── attr_best.pt         ← Phase 2b (disease attribute heads)
├── hybrid_best.pt       ← Phase 3 (Swin + cross-attention + severity)
└── archon_best.pt       ← Unified checkpoint (all of the above merged)
```

---

## 10. Quick-Fire Viva Q&A

### Q: What is ARCHON?
**A**: ARCHON (Arch-Contextualized Hierarchical Orthodontic Network) is a deep learning system that simultaneously detects teeth, assigns FDI numbers, and diagnoses four diseases with severity grading from panoramic dental X-rays. It extends the YOLOv8x detector with CoordConv for position awareness, a Swin Transformer for global jaw context, cross-attention fusion, and hierarchical multi-task prediction heads.

### Q: Why not just use a classification model (ResNet)?
**A**: Classification alone doesn't tell you WHERE the disease is — which specific tooth has caries. We need both localization (bounding box) and classification (disease). A pure classifier would require pre-cropping each tooth, which itself needs a detector — so you'd end up with a 2-model pipeline more complex than our single-model approach.

### Q: What does CoordConv do and why is it needed?
**A**: Standard convolution is translation-invariant — it can't distinguish a molar in the upper-right jaw (tooth 16) from the same-looking molar in the lower-left (tooth 36). CoordConv concatenates normalized (x, y) coordinate channels to the input, breaking translation invariance. The model learns to use position information for FDI numbering.

### Q: Explain P3, P4, P5 to me.
**A**: These are feature maps at different resolutions from the Feature Pyramid Network. P3 (stride 8, 160×80) captures fine details like small root lesions. P4 (stride 16, 80×40) captures medium features like full tooth shapes. P5 (stride 32, 40×20) captures coarse, big-picture information like jaw quadrant layout. All three are used for detection and disease classification.

### Q: Why did you use pseudo labeling?
**A**: Part 3 of the DENTEX dataset only labels diseased teeth — healthy teeth in those images have no annotation. Without pseudo labels, the model is penalized for correctly detecting healthy teeth (treated as false positives). We use the Phase 1 detector to label the remaining healthy teeth, completing the annotations.

### Q: What is the linear sum assignment and why is it needed?
**A**: The model might assign the same FDI number to two different teeth (e.g., both detected as tooth 16). In reality, each tooth number can only appear once. The Hungarian algorithm finds the globally optimal one-to-one assignment that minimizes total cost (= maximizes total class probability). This ensures no duplicate FDI assignments.

### Q: Why not use DETR instead of YOLO + post-processing?
**A**: DETR naturally avoids duplicate detections via its learned set prediction. However, DETR needs 500+ epochs to converge (we use 100), is poor on small objects (teeth at X-ray edges), uses more memory (full attention on all tokens), and is harder to extend with custom disease classification heads.

### Q: What is the Swin Transformer used for in your model?
**A**: The Swin Transformer processes the coarsest P5 feature map to capture jaw-wide context — things like quadrant layout, tooth ordering, and total tooth count. This global context is then fused into local CNN features via cross-attention, helping resolve FDI numbering conflicts where two teeth look identical but are in different quadrants.

### Q: Explain your training phases.
**A**: Phase 1 trains tooth detection on Part 1+2 (no disease labels). Phase 2 fine-tunes on all data including pseudo-labeled healthy teeth. Phase 2b trains disease attribute heads (backbone frozen). Phase 3 trains the hybrid components — Swin Transformer, cross-attention, severity heads (everything else frozen). This progressive training prevents later components from destabilizing earlier ones.

### Q: Why is the attribute loss weight 8.0?
**A**: Disease detection is the primary clinical contribution. The standard detection losses (bbox=7.5, class=0.5, DFL=1.5) sum to ~9.5. Without a high attribute weight, the optimizer focuses on improving detection metrics (already good from COCO pretraining) and ignores the harder disease classification. 8.0 per attribute gives disease classification enough gradient signal.

### Q: What is hierarchical loss masking?
**A**: Different parts of the dataset have different annotation levels. Part 1 has only quadrant labels (data_type=0), Part 2 has FDI numbers (data_type=1), Part 3 has disease labels (data_type=2). The loss function only computes disease loss on data_type=2 samples, FDI loss on data_type≥1, and quadrant loss on all. This prevents the model from trying to learn disease classification from data that has no disease annotations.

### Q: What is CLAHE and why did you add it?
**A**: CLAHE (Contrast Limited Adaptive Histogram Equalization) enhances local contrast in dental X-rays. X-rays have non-uniform exposure — bright bone, dark soft tissue. CLAHE makes subtle features visible: early caries (slight brightness change), periapical lesions (dark halos at root tips). Applied randomly during training (50% probability) to prevent over-sharpening.

### Q: Why no mosaic augmentation?
**A**: Mosaic stitches 4 different images into a grid. Panoramic X-rays show a continuous jaw — stitching pieces from different patients creates unrealistic anatomy (e.g., jaw bones that don't connect). The model would learn to detect teeth at artificial junctions. Similarly, mixup is disabled because blending two jaw X-rays creates ghost teeth.

### Q: What is the flip mapping?
**A**: When a panoramic X-ray is horizontally flipped, the quadrants swap: Q1↔Q2, Q3↔Q4. A tooth labeled as FDI 16 (upper right) becomes FDI 26 (upper left). Standard augmentation libraries flip the image and box coordinates but don't remap the class label. Our custom augmentation does both — it's the key augmentation from the paper.

### Q: How does pos_weight prevent the "always predict healthy" problem?
**A**: In DENTEX, ~85% of teeth are healthy. A naive model can achieve 85% accuracy by always predicting healthy. pos_weight makes missed diseases (false negatives) much more costly than false alarms (false positives). For deep caries (8% prevalence), pos_weight=8.0 means missing a case of deep caries costs 8× more than a false alarm. The model is forced to learn the rare disease patterns.

### Q: Why freeze the backbone in Phase 2b?
**A**: The backbone (YOLOv8x) is already well-trained for tooth detection from Phase 2. If we also update backbone weights during attribute training, the small attribute loss signal could subtly corrupt the larger detection pipeline. Freezing preserves detection quality while allowing attribute heads to specialize independently. Also: BatchNorm in frozen mode uses stable running statistics instead of noisy batch statistics.

### Q: What's the difference between the baseline and hybrid model?
**A**: The baseline (ARCHON) is YOLOv8x + CoordConv + 4 binary disease attribute heads. The hybrid (ARCHONModel) adds three components: (A) Swin Transformer for global jaw context, (B) Cross-attention to fuse local CNN + global Swin features, (C) 3-level severity heads + quadrant auxiliary head. The hybrid addresses FDI numbering conflicts and provides richer disease grading.

### Q: What is DFL (Distribution Focal Loss)?
**A**: Instead of predicting a single bounding box edge coordinate, DFL predicts a probability distribution over possible positions. This captures uncertainty — tooth boundaries on X-rays are often blurry (overlapping teeth, root tips fading into bone). DFL lets the model say "I'm 60% sure the edge is here, 30% it's 1 pixel right" instead of committing to a single location.

### Q: How does the quadrant consistency penalty work in post-processing?
**A**: When the hybrid quadrant head is confident a detection is in Q2 (say, 95% probability), the cost matrix for linear sum assignment is modified: FDI slots in Q1, Q3, Q4 get a penalty proportional to (1 - P(Q2)). The penalty is confidence-gated — when the quadrant prediction is uncertain (each quadrant ~25%), the penalty is zero. This prevents wrong quadrant assignments for midline teeth where the boundary between Q1/Q2 or Q3/Q4 is ambiguous.

### Q: What would you do differently if you started this project today?
**A**: (1) Use YOLOv11 or RT-DETR v2 for better baseline accuracy. (2) Pre-train the backbone on unlabeled dental X-rays using DINOv2 self-supervised learning instead of ImageNet. (3) Add deformable attention instead of window-based Swin for more flexible global context. (4) Use knowledge distillation to create a smaller deployable model. (5) If tooth segmentation masks were available, use SAM-based refinement for more precise feature sampling.

---

## 11. Additional Deep-Dive Topics

### 11.1 IoU (Intersection over Union) — Detailed Explanation

**What is IoU?**

IoU measures how much two bounding boxes overlap. It's the core metric for evaluating detection quality.

```
IoU = Area of Overlap / Area of Union

   ┌──────────┐
   │  Box A   │
   │    ┌─────┼────┐
   │    │/////│    │
   └────┼─────┘    │
        │  Box B   │
        └──────────┘

   Overlap = shaded region
   Union   = total area covered by both boxes
```

**Real-life analogy**: Two pizza slices on a plate. IoU is the fraction of the plate covered by BOTH slices divided by the total plate covered by EITHER slice. If they're perfectly stacked: IoU = 1.0. If they don't touch: IoU = 0.0.

**IoU Variants Used in Our Project:**

| Variant | Formula | Used Where | Why |
|---------|---------|-----------|-----|
| **IoU** | Overlap / Union | NMS, mAP evaluation | Basic overlap metric |
| **GIoU** | IoU − (C − Union) / C | Early YOLO versions | Adds penalty for non-overlapping area in enclosing box C |
| **DIoU** | IoU − d²(centers) / c² | Intermediate YOLO | Adds center distance penalty for faster convergence |
| **CIoU** (our choice) | DIoU − α × v | **YOLOv8 bbox loss** | Adds aspect ratio consistency penalty |

**CIoU explained in detail:**
```
CIoU = IoU − (ρ²(b, b_gt)) / c² − α × v

Where:
  ρ(b, b_gt) = Euclidean distance between predicted and GT box centers
  c           = diagonal length of smallest enclosing box
  v           = (4/π²) × (arctan(w_gt/h_gt) − arctan(w/h))²   (aspect ratio consistency)
  α           = v / ((1 − IoU) + v)                             (trade-off parameter)
```

**Why CIoU and not plain IoU for the loss?**

Plain IoU loss has a **gradient = 0 when boxes don't overlap** (IoU = 0). The model can't learn to move boxes closer. CIoU's center distance term (ρ²/c²) always provides a gradient, even when boxes are far apart.

| Scenario | Plain IoU | CIoU |
|----------|-----------|------|
| Boxes don't overlap | gradient = 0 (stuck!) | gradient exists (center distance) |
| Boxes overlap, wrong aspect ratio | only penalizes overlap | also penalizes aspect ratio mismatch |
| Boxes overlap, centers offset | only penalizes overlap | also penalizes center distance |

**IoU thresholds in our project:**
- **NMS IoU = 0.45**: Two detections overlapping >45% → suppress the weaker one (same tooth)
- **mAP@0.5**: A detection is "correct" if IoU with ground truth ≥ 0.5
- **mAP@0.5:0.95**: Average over IoU thresholds [0.5, 0.55, 0.60, ..., 0.95] — much stricter
- **Pseudo label overlap IoU = 0.3**: A new detection overlapping >30% with existing disease label is skipped

### 11.2 COCO Pretraining — Transfer Learning

**What is COCO?**
COCO (Common Objects in Context) is a massive dataset of 330K images with 80 object classes (person, car, dog, chair, etc.). It has NOTHING to do with dentistry.

**Why pretrain on COCO for dental X-rays?**

**Real-life analogy**: A chef trained in French cuisine can learn Japanese cooking faster than someone who's never cooked. The French skills (knife technique, timing, flavor balance) transfer — even though the cuisines are different.

Similarly, a COCO-pretrained YOLOv8x has already learned:
1. **Low-level features** (edges, textures, gradients) — universal across all images
2. **Object detection skills** (finding contiguous regions, separating foreground from background)
3. **Multi-scale feature extraction** (recognizing objects at different sizes)
4. **Bounding box regression** (predicting precise box coordinates)

These skills directly apply to dental X-rays. The model just needs to re-learn WHAT to detect (teeth instead of cars) while keeping HOW to detect.

**The Transfer Learning Pipeline:**

```
ImageNet pretrained backbone (generic features)
         │
         ▼
COCO pretrained YOLOv8x (detection-specific features) ← "yolov8x.pt"
         │
         ▼
Phase 1: Fine-tune on DENTEX Part 1+2 (tooth detection features)
         │
         ▼
Phase 2: Fine-tune on all DENTEX data (tooth + disease features)
```

**Why not train from scratch?**
- DENTEX has only ~2,000 labeled images — not enough for training a 68M parameter model from random weights
- From scratch requires 500+ epochs; with pretraining, 100 epochs suffice
- From scratch on small data → severe overfitting

**Why not pretrain on dental X-rays specifically?**
- No large public dental X-ray dataset with detection annotations exists (DENTEX is the biggest at ~2,000 images)
- Self-supervised pretraining (MAE, DINO) on unlabeled dental X-rays could help, but DENTEX only provides ~1,000 unlabeled images — not enough for effective self-supervised learning

### 11.3 GPU Memory Estimation Breakdown

Understanding GPU memory is critical for choosing batch size and image size.

**Memory components during training:**

| Component | Size (approx) | Scales With |
|-----------|---------------|-------------|
| Model weights (YOLOv8x) | ~280 MB (68M params × 4 bytes) | Fixed |
| CoordConv overhead | ~5 MB (extra channels) | Fixed |
| Attribute heads | ~2 MB | Fixed |
| Swin Transformer | ~12 MB | Fixed |
| Cross-Attention | ~8 MB | Fixed |
| **Activations (forward pass)** | **~3 GB per image at 1280×640** | Batch size × Image size |
| **Gradients** | **~280 MB** (same as weights) | Fixed |
| **Optimizer state (SGD momentum)** | **~280 MB** | Fixed |
| **Optimizer state (AdamW)** | **~560 MB** (2× for m and v) | Fixed |

**Memory per batch size (estimated for 1280×640, YOLOv8x):**

| Batch Size | Total Memory | Fits on T4 (16GB)? | Fits on A100 (40GB)? |
|-----------|-------------|---------------------|---------------------|
| 1 | ~4.5 GB | ✅ | ✅ |
| 2 | ~7.5 GB | ✅ | ✅ |
| **4** | **~13 GB** | ✅ (tight) | ✅ |
| 8 | ~24 GB | ❌ OOM | ✅ |
| 16 | ~46 GB | ❌ | ❌ (needs 80GB H100) |

**Why batch_size=4?**
- Fits on T4/16GB GPU (Google Colab free tier)
- Provides 4 samples for BatchNorm statistics (minimum reasonable)
- Each image is 1280×640 = 819,200 pixels — 3.2× larger than standard 640×640
- Doubling to batch=8 would need ~24GB VRAM

**Memory optimization techniques we use:**
- Frozen backbone in Phase 2b/3 → no gradient storage for backbone weights → saves ~200 MB
- Flash Attention in cross-attention → avoids O(N²) attention matrix → saves ~500 MB at P3 scale
- `.detach()` on FPN features in Phase 2b → prevents gradient graph from growing into the backbone

### 11.4 Complete Loss Function Comparison Table

| Loss | Type | Weight | data_type Mask | Phase Used | Formula | Purpose |
|------|------|--------|----------------|------------|---------|---------|
| **CIoU Box Loss** | Regression | 7.5 | All (0,1,2) | 1, 2 | CIoU = IoU − ρ²/c² − αv | Bounding box localization |
| **Classification CE** | Cross-Entropy | 0.5 | ≥1 (full FDI), 0 (quadrant-grouped) | 1, 2 | −Σ y_i log(p_i) | Tooth class (FDI number) prediction |
| **DFL (Distribution Focal)** | Cross-Entropy on discretized distribution | 1.5 | All (0,1,2) | 1, 2 | CE between predicted and target distributions over box edge positions | Box edge refinement with uncertainty |
| **Attribute BCE** | Binary Cross-Entropy | 8.0 | Only 2 | 2b, 3 | −[W×y×log(σ(x)) + (1−y)×log(1−σ(x))] | Binary disease classification per tooth |
| **Severity CE** | Cross-Entropy (3 classes) | 4.0 | Only 2 | 3 | −Σ y_i log(softmax(x)_i), with label_smoothing=0.1 | 3-level disease severity per attribute |
| **Quadrant Aux CE** | Cross-Entropy (4 classes) | 1.0 | All (0,1,2) | 3 | −Σ y_i log(softmax(x)_i), target = class_idx // 8 | Auxiliary quadrant classification |
| **Hierarchical Class CE** | Conditional CE | 0.5 | 0 → 4-class grouped, ≥1 → 32-class | 2 (custom) | Sum of quadrant CE + FDI CE | Class loss respecting annotation hierarchy |

**Loss weight rationale summary:**

```
Detection losses:   7.5 (box) + 0.5 (class) + 1.5 (DFL) = 9.5 total
Attribute loss:     8.0 per attribute × 4 attributes = 32.0 total potential
                    (but only fires on data_type=2 samples, so effective weight is lower)
Severity loss:      4.0 (half of attribute weight — severity has more gradient signal per class)
Quadrant aux loss:  1.0 (lightweight auxiliary signal, shouldn't dominate)
```

**Why attribute weight (8.0) is higher than severity weight (4.0)?**
Severity classification into 3 classes provides richer gradient per sample than binary classification. Each severity CE loss backpropagates through 3 logits instead of 1. So even at half the weight, severity heads receive comparable gradient signal.

### 11.5 FPN Channel Counts — Why 320, 640, 640?

**What determines FPN channel counts?**

YOLOv8x has a specific architecture with fixed channel widths at each layer. The channel count doubles as we go deeper:

```
YOLOv8x Backbone:
  Layer 0-1:  Input (3ch) → 80 channels
  Layer 2-3:  80 → 160 channels
  Layer 4-5:  160 → 320 channels        ← C3 (feeds into P3)
  Layer 6-7:  320 → 640 channels         ← C4 (feeds into P4)
  Layer 8-9:  640 → 640 channels         ← C5 (feeds into P5, no doubling — already large)
```

**After the FPN Neck (PANet):**

The neck merges top-down and bottom-up features. The output channels at each scale are determined by the C2f modules in the neck:

```
P3 (stride 8):  320 channels  ← from C3 (320) merged with upsampled C4
P4 (stride 16): 640 channels  ← from C4 (640) merged with upsampled C5 and P3
P5 (stride 32): 640 channels  ← from C5 (640), no merging needed at coarsest scale
```

**Why does this matter?**
- Our attribute heads need to know the input channel count at each scale
- The `_infer_fpn_channels()` function reads these from the model's Detect head (`detect.ch` attribute)
- If inference fails, it falls back to the hardcoded `[320, 640, 640]`

**How channels affect parameter count:**
```
AttributeHead at P3 (in=320):  320→80 (conv1) + 80→80 (conv2) + 80→1 (out) = ~46K params
AttributeHead at P5 (in=640):  640→160 (conv1) + 160→160 (conv2) + 160→1 (out) = ~180K params
Total for 4 attrs × 3 scales: ~1.4M parameters (vs 68M for the full model)
```

This is why training only the attribute heads (Phase 2b) is fast — they're tiny compared to the backbone.

**YOLOv8 variant comparison:**

| Variant | P3 channels | P4 channels | P5 channels | Total params |
|---------|-------------|-------------|-------------|-------------|
| YOLOv8n | 64 | 128 | 256 | 3.2M |
| YOLOv8s | 128 | 256 | 512 | 11.2M |
| YOLOv8m | 192 | 384 | 576 | 25.9M |
| YOLOv8l | 256 | 512 | 512 | 43.7M |
| **YOLOv8x** | **320** | **640** | **640** | **68.2M** |

We use YOLOv8x because wider channels (320/640/640) give the model more capacity to represent 32 visually similar tooth classes + disease attributes.

---

*This guide covers every technical concept, file, function, and decision in the ARCHON project. After studying it, you should be able to answer any viva question about the architecture, training pipeline, alternatives considered, and future improvements.*
