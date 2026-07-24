# ARCHON — Hybrid Architecture Improvements

> **ARCHON: Arch-Contextualized Hierarchical Orthodontic Network for Severity-Aware Dental Radiograph Analysis**
>
> What changed from the ARCHON baseline implementation, why each change matters clinically,
> and what diagnostic improvement each component is designed to deliver.

---

## Table of Contents

1. [Quick Summary Table](#1-quick-summary-table)
2. [Improvement A — GlobalContextEncoder (Swin Transformer)](#2-improvement-a--globalcontextencoder-swin-transformer)
3. [Improvement B — MultiScaleFusion (Cross-Attention)](#3-improvement-b--multiscalefusion-cross-attention)
4. [Improvement C — HybridMultiTaskHead (Severity + Quadrant)](#4-improvement-c--hybridmultitaskhead-severity--quadrant)
5. [Improvement D — CLAHE Augmentation](#5-improvement-d--clahe-augmentation)
6. [Improvement E — Quadrant-Consistency Post-processing](#6-improvement-e--quadrant-consistency-post-processing)
7. [How the Improvements Interact](#7-how-the-improvements-interact)
8. [New Training Mode: train_hybrid](#8-new-training-mode-train_hybrid)
9. [New Config Keys](#9-new-config-keys)
10. [New Output Fields](#10-new-output-fields)
11. [File Change Index](#11-file-change-index)

---

## 1. Quick Summary Table

| # | Component | File(s) | Baseline Problem | Improvement | Clinical Benefit |
|---|---|---|---|---|---|
| A | `GlobalContextEncoder` | `swin_transformer.py` | CNN sees only local patches; no jaw-arch context | Swin Transformer on P5 captures inter-tooth relationships across the full arch | Better FDI disambiguation for symmetric teeth; fewer midline swap errors |
| B | `MultiScaleFusion` | `cross_attention.py` | Local CNN features and global context are computed independently | Cross-attention fuses CNN local texture (Q) with Swin global context (K/V) at all 3 FPN scales | Disease features guided by anatomical position; improved specificity |
| C | `HybridMultiTaskHead` | `hybrid_head.py` | Binary healthy/diseased output only; no severity; no quadrant auxiliary | 3-level severity (Healthy/Mild/Severe) + quadrant auxiliary classifier | Clinically actionable severity grading; assists triage and treatment planning |
| D | `CLAHE augmentation` | `augmentation.py` | Raw X-rays have non-uniform exposure; early lesions invisible in bright enamel regions | CLAHE on L-channel (p=0.5) during Phase 3 training | Model trained on enhanced contrast → better detection of subtle early caries and periapical lesions |
| E | Quadrant-consistency penalty | `postprocess.py` | Pure confidence-based FDI assignment can swap symmetric teeth | Cross-attention quadrant probs added as penalty in Hungarian cost matrix | Fewer left/right symmetric tooth misidentification errors |

---

## 2. Improvement A — GlobalContextEncoder (Swin Transformer)

### What the baseline does

The ARCHON baseline uses a CNN (YOLOv8x) where every convolution operates on a local receptive field. Even the deepest backbone features (P5) effectively "see" roughly a 127×127 pixel patch after accounting for the cascade of 3×3 convolutions. On a 1280×640 panoramic X-ray, each tooth is about 50–100 pixels wide — so P5 can see 1–2 neighboring teeth at best, but **not the entire dental arch at once**.

**Consequence:** When two teeth have similar size, shape, and pathology (e.g., premolars 14 and 24 are near-identical), the CNN must guess the FDI number from local information alone. It has no "map" of where the jaw midline is or what the adjacent teeth look like.

### What GlobalContextEncoder adds

A two-block Swin Transformer hierarchically processes the P5 feature map (stride 32, ~20×40 spatial resolution for a 640×1280 input):

1. **Block 1 — Regular Window Attention (shift_size=0):** Divides the feature into 4×4 windows. Within each window, every P5 cell (each representing one tooth-sized patch) attends to all other cells in the window. Captures fine inter-tooth detail.

2. **Block 2 — Shifted Window Attention (shift_size=2):** The 4×4 windows are shifted by (2,2). This causes cells that were at the *boundary* of two different Block-1 windows to be in the *same* Block-2 window. **Cross-window information flows.** After two blocks, every P5 cell has attended to cells from across the arch.

**Output:** Same shape as input P5 — a "globally-aware" feature that still has P5 spatial resolution but now encodes dental-arch context.

### Why Swin over full self-attention

| | Full self-attention on P5 | Swin Transformer |
|---|---|---|
| Sequence length | 20×40 = 800 | 16 (per window) |
| Attention complexity | O(800²) = 640,000 ops | O(16²) × 50 windows = 12,800 ops |
| Memory | ~50× more | Efficient |
| Cross-window connectivity | Full | Via shifted windows |

### Clinical benefit

- Symmetric teeth on opposite sides of the arch (11↔21, 16↔26, etc.) can be distinguished by their arch-level context — e.g., "this premolar is flanked by a canine on the right and a molar on the left" maps uniquely to Q1 vs Q2
- Rare whole-arch patterns (e.g., oligodontia — absence of multiple teeth) can be flagged when the global encoder perceives large empty P5 regions
- Reduces the most common failure: **midline swap errors** where symmetric teeth are assigned wrong FDI numbers

---

## 3. Improvement B — MultiScaleFusion (Cross-Attention)

### What the baseline does

After the backbone and neck, the baseline keeps **two separate prediction pathways**: the YOLO detection head (uses FPN features directly) and the attribute heads (also use FPN features via hooks). Neither pathway tells the other what it found. There is no mechanism for the disease classifier to know "I am currently looking at an upper-left tooth" vs "lower-right tooth" — it only sees the local texture.

### What MultiScaleFusion adds

Cross-attention at each FPN scale (P3, P4, P5):

```
Query  Q = CNN feature at scale s   (local texture, precise spatial detail)
Key    K = Swin global context      (arch-level anatomy, quadrant information)
Value  V = Swin global context      (same)

fused = Attention(Q, K, V) = softmax(QK^T / √d) · V
output = LN(CNN + fused) + FFN residual
```

**Intuition:** Each CNN feature at a given spatial location can "query" the global context to ask "given the whole arch context, what tooth location am I at, and what disease patterns are common there?"

**Multi-scale:** Fusion is applied at P3 (stride 8, finest), P4 (stride 16), and P5 (stride 32). The global context tensor is resampled to match each scale:
- P3 fusion: context upsampled × 4 → gives fine-grained disease features arch-level context
- P4 fusion: context upsampled × 2
- P5 fusion: context as-is

### Why this matters for diagnosis

- **Periapical lesions** appear as subtle low-density shadow at a tooth root — almost indistinguishable from jaw anatomy without knowing you are looking at a tooth apex. Cross-attention lets the lesion head ask "is this an apex region (P5 knows)?" before deciding.
- **Impacted teeth** are often partially erupted and have unusual orientations. Knowing the neighboring teeth's positions (from global context) helps the head determine if a tooth is crowded/impacted vs simply positioned at an unusual angle.
- **Deep Caries** vs Caries disambiguation: Deep caries affects the pulp. Pulp chamber location is predictable per tooth type — cross-attention gives the severity head tooth-type context from the arch-level encoder.

### Residual safety

Both cross-attention and feed-forward sublayers use residuals:
```
output = LN(CNN + CrossAttn(...))
output = LN(output + FFN(output))
```
If the cross-attention adds no value for a particular location, the network can learn attention weights ≈ 0, effectively bypassing the fusion. The CNN baseline performance is therefore a lower bound — fusion can only add, not subtract.

---

## 4. Improvement C — HybridMultiTaskHead (Severity + Quadrant)

### Severity classification (Healthy / Mild / Severe)

**Baseline output:** Binary — either the disease flag is 0 or 1.

**New output:** 3-class per disease per tooth:

| Severity | Caries | Impacted | Deep Caries | Lesion |
|---|---|---|---|---|
| 0 Healthy | no caries | not impacted | no deep caries | no lesion |
| 1 Mild | early caries (has_caries=1 only) | — | — | — |
| 2 Severe | deep caries (both flags set) | impacted | deep caries | periapical lesion |

**Clinical benefit:**
- **Mild caries** may be managed with fluoride treatment; **Severe caries** requires restorative or endodontic intervention. A binary flag gives the same output for both — forcing the clinician to re-examine the X-ray manually to decide severity.
- **Triage:** In mass screening (e.g., DENTEX challenge's target use case of underserved populations), an automated severity flag can prioritize patients: Severe → urgent referral; Mild → schedule next visit; Healthy → no action.
- **Longitudinal tracking:** Re-scanning the same patient months later and comparing Mild→Severe transitions can objectively measure disease progression.

**Biased initialization:** The output head bias is set to `[+2.0, 0.0, −2.0]` — strongly predicting Healthy before any training. This prevents the common early-training collapse where the model predicts random severity for all teeth, causing unstable gradients on the imbalanced DENTEX dataset.

**SeverityLoss (label smoothing=0.1):** Soft targets prevent the model from becoming overconfident on the noisy binary-derived severity labels. A tooth labeled "Severe" gets target `[0.033, 0.033, 0.933]` instead of `[0, 0, 1]`.

### Quadrant auxiliary classifier

A lightweight head predicts which quadrant each tooth occupies (Q1/Q2/Q3/Q4).

**This is an auxiliary task** — its primary purpose is to force the fused features to encode quadrant-discriminative information. Secondary purpose: the 4-class output is passed to the post-processor as a penalty on FDI assignment (see Improvement E).

**Why auxiliary classification helps the main task:**  
Adding a supervised auxiliary loss on quadrant forces the shared fused features to carry quadrant-discriminative information. This acts as a form of multi-task regularization — the main disease classifier benefits from features that also encode "which quadrant is this tooth in?" even if the quadrant prediction itself is not used by the clinician.

---

## 5. Improvement D — CLAHE Augmentation

### Problem

Panoramic X-ray images have a characteristic exposure distribution:
- **Bright regions** (dense bone, posterior molars): histogram concentrated in high intensities; fine caries lines may be invisible because there is little contrast to separate them from surrounding enamel
- **Dark regions** (soft tissue gaps, incisors on some machines): low dynamic range; periapical shadow may blend with natural jaw translucency

Standard data augmentation (brightness shifts, gamma) applies a global transformation — it cannot fix local contrast issues.

### Solution

**CLAHE (Contrast Limited Adaptive Histogram Equalization)** divides the image into non-overlapping tiles (8×8 by default) and equalizes the histogram *within each tile separately*, then bilinearly interpolates across tile boundaries. The `clipLimit` parameter caps the maximum allowable contrast gain to prevent noise amplification.

**Applied on the L-channel of LAB color space:**
```
BGR → LAB: L = luminance, A/B = color channels
CLAHE on L only → enhanced luminance
LAB → BGR: reassemble with original color channels
```

This is important for X-rays stored as 3-channel BGR (grayscale replicated to 3 channels) — applying CLAHE on all 3 channels would distort the gray balance.

**Training only (p=0.5):** CLAHE is applied stochastically during Phase 3 training. At inference, the original raw X-ray is used — this creates a training set diversity where the model must recognize disease patterns both with and without contrast enhancement, making it more robust to exposure variation in real clinical scanners.

### Clinical benefit

- Early interproximal caries (between teeth) — very thin, low-contrast shadows that are the most commonly missed finding in routine radiographic reads — become visible in the CLAHE-enhanced view
- Periapical lesions in high-density molar bone become detectable earlier
- Model trained on CLAHE-augmented images learns to recognize the low-contrast pattern *without* needing CLAHE at inference (implicit knowledge transfer)

---

## 6. Improvement E — Quadrant-Consistency Post-processing

### Problem

The Linear Sum Assignment (Hungarian algorithm) assigns each detection to an FDI slot based on the softmax class probability from the detection head. The class probabilities from the detection head can confuse symmetric teeth:

```
Detection: high molar in left side of image
Class probability:  P(FDI 16) = 0.52   (correct: upper-right 6th)
                    P(FDI 26) = 0.41   (wrong: upper-left 6th)
                    P(FDI 46) = 0.07
```

If FDI 16 is already taken by another detection, the Hungarian algorithm assigns this to 26 — which could be wrong if the tooth truly is FDI 16 (the image was just ambiguous).

### Solution

The quadrant auxiliary head (from Improvement C) produces a 4-vector `[P(Q1), P(Q2), P(Q3), P(Q4)]` per detection. This is used to add a **quadrant-consistency penalty** to the cost matrix before Hungarian solving:

```
slot_quadrants = [0]*8 + [1]*8 + [2]*8 + [3]*8    # which quadrant each FDI slot belongs to
quad_penalty[slot, det] = ALPHA × (1 - P(correct_quadrant_for_slot))
cost_matrix += quad_penalty                         # ALPHA = 0.5
```

**Effect:**
- Assigning a detection to FDI 16 (Q1 slot): costs extra `0.5 × (1 - P(Q1))` — if the head confidently says Q1, this penalty is near 0
- Assigning to FDI 26 (Q2 slot): costs extra `0.5 × (1 - P(Q2))` — if head says Q1 not Q2, this penalty is ~0.5, making 26 less attractive

The Hungarian algorithm now jointly optimizes detection confidence + quadrant consistency, reducing symmetric-tooth assignment errors.

**Alpha=0.5 balance:** Too high → quadrant probability dominates and detection score is ignored; too low → quadrant signal has no effect. 0.5 means both the detection class score and the quadrant score contribute roughly equally when they disagree.

---

## 7. How the Improvements Interact

```
Input X-ray
    │
    ▼
YOLOv8x (backbone, frozen in Phase 3)
    │
    ├── P3, P4 features (local detail)
    └── P5 feature (coarse, tooth-level patches)
                │
                ▼
          A. GlobalContextEncoder
             (Swin Transformer on P5)
                │
                ▼ global_context tensor
                │
    ┌───────────┴──────────────────────────┐
    │  B. MultiScaleFusion                 │
    │     Q = CNN(P3) + context → fused P3 │
    │     Q = CNN(P4) + context → fused P4 │
    │     Q = CNN(P5) + context → fused P5 │
    └───────────┬──────────────────────────┘
                │ fused features (globally-informed local detail)
                ▼
    ┌───────────────────────────────────────┐
    │  C. HybridMultiTaskHead               │
    │     SeverityHead: 3-class per disease  │
    │     QuadrantHead: 4-class per tooth    │
    └────────────────┬──────────────────────┘
                     │ severity_probs + quadrant_probs
                     ▼
    ┌───────────────────────────────────────┐
    │  E. Quadrant-Consistency Post-proc    │
    │     Hungarian(cost + quad_penalty)    │
    └────────────────┬──────────────────────┘
                     │
                     ▼
    Final output: ToothDetection per tooth
      - fdi (Hungarian-assigned, quad-consistent)
      - has_caries / is_impacted / has_deepcaries / has_lesion (binary)
      - severity_details: {caries: "Mild", lesion: "Severe", ...}
```

**D (CLAHE augmentation)** acts at training time, making the above chain more robust to exposure variation at inference.

---

## 8. New Training Mode: `train_hybrid`

### When to run

Run after `--mode train` (Phases 1 + 2 + 2b) completes. Phase 3 requires:
- `outputs/runs/phase2/weights/best.pt` (or `--weights` override)
- `data/processed/labels_ext/train/` (10-column disease labels)

### What it trains

Only three new modules are updated. Everything else is frozen:

| Module | Frozen? | Reason |
|---|---|---|
| YOLOv8x backbone | Yes | Already well-trained on all dental images; fine-tuning would destabilize detection |
| Binary attribute heads | Yes | Already trained in Phase 2b; re-training would overwrite validated disease detection |
| `GlobalContextEncoder` | **No** | New in Phase 3; must learn from scratch |
| `MultiScaleFusion` | **No** | New in Phase 3; must learn |
| `HybridMultiTaskHead` | **No** | New in Phase 3; must learn |

### Training details

| Detail | Value |
|---|---|
| Epochs | `phase3_epochs` (default 50) |
| Optimizer | AdamW (lr=5e-4) |
| Scheduler | CosineAnnealingLR → 1e-6 |
| Loss | `AttributeBCE(w=4.0) + SeverityLoss(w=4.0) + QuadrantAuxLoss(w=1.0)` |
| Batch size | Half of Phase 2b (extra memory for Swin + cross-attn) |
| CLAHE | Active (p=0.5) |
| Backbone BN | `.eval()` mode — uses running statistics, not batch statistics |
| Checkpoint saved | `weights/hybrid_best.pt` (best validation loss) |

### Commands

```bash
# Standard (auto-finds phase2 best.pt)
python main.py --mode train_hybrid

# Custom backbone weights
python main.py --mode train_hybrid --weights weights/archon_best.pt

# Full pipeline from scratch including Phase 3
python main.py --mode full       # phases 1+2+2b
python main.py --mode train_hybrid   # phase 3
```

---

## 9. New Config Keys

Added to `config/model_config.yaml`:

```yaml
# Hybrid Architecture (Improvements A, B, C)
use_hybrid: false            # set true to load hybrid components at inference
swin_num_heads: 8            # attention heads in Swin blocks (must divide embed_dim)
swin_window_size: 4          # local window size (4×4 on P5 ~20×40)
fusion_num_heads: 4          # attention heads in cross-attention fusion
num_severity_levels: 3       # 3 = Healthy / Mild / Severe
severity_threshold: 0.4      # P(not Healthy) threshold to report a severity finding
quadrant_penalty_alpha: 0.5  # weight of quadrant-consistency penalty in post-processing
```

Added to `config/train_config.yaml`:

```yaml
phase3_epochs: 50            # Phase 3 (Hybrid) training epochs
```

---

## 10. New Output Fields

When `hybrid_best.pt` is present and loaded, the `_result.json` output gains a `severity_details` object per tooth:

```json
{
  "fdi": 36,
  "bbox": [245, 310, 390, 420],
  "confidence": 0.91,
  "is_impacted": false,
  "has_caries": true,
  "has_deepcaries": false,
  "has_lesion": false,
  "severity_details": {
    "caries": "Mild",
    "impacted": "Healthy",
    "deepcaries": "Healthy",
    "lesion": "Healthy"
  }
}
```

Severity levels: `"Healthy"` / `"Mild"` / `"Severe"`

---

## 11. File Change Index

| File | Status | What Changed |
|---|---|---|
| `src/models/swin_transformer.py` | **NEW** | `AddCoords`, `CoordConv`, `WindowAttention`, `SwinTransformerBlock`, `GlobalContextEncoder` |
| `src/models/cross_attention.py` | **NEW** | `CrossAttentionFusion`, `MultiScaleFusion` |
| `src/models/hybrid_head.py` | **NEW** | `SeverityHead`, `MultiScaleSeverityHead`, `QuadrantAwareFDIHead`, `HybridMultiTaskHead`, `SeverityLoss`, `QuadrantAuxLoss` |
| `src/models/yolortho.py` | Modified | Added `ARCHONModel` class + `build_archon_model()` function; existing `ARCHON` base class unchanged |
| `src/training/trainer.py` | Modified | Added `ARCHONHybridTrainer` subclass + `train_hybrid()` + `_train_hybrid_heads()`; existing trainer unchanged |
| `src/inference/predictor.py` | Modified | Added `ARCHONHybridPredictor` subclass + `_load_hybrid_components()` + `_sample_per_tooth()`; existing predictor unchanged |
| `src/inference/postprocess.py` | Modified | `apply_linear_sum_assignment()` and `postprocess_yolo_output()` gained optional `quadrant_probs` param + penalty block |
| `src/data/augmentation.py` | Modified | `ARCHONAugmentor` gained `clahe_prob`, `clahe_clip`, `clahe_tile` params + `_augment_clahe()` method |
| `config/model_config.yaml` | Modified | New `use_hybrid` + 6 hybrid configuration keys |
| `config/train_config.yaml` | Modified | New `phase3_epochs: 50` key |
| `main.py` | Modified | New `--mode train_hybrid` choice + `stage_train_hybrid()` function + dispatch table entry |
| `WORKFLOW.md` | **NEW** | Complete end-to-end workflow reference covering all modes and new components |
| `IMPROVEMENTS.md` | **NEW** | This file |

### What was NOT changed

The following files are **unchanged from the baseline** — all improvements are additive:

- `src/models/coord_conv.py` — CoordConv unchanged
- `src/models/heads.py` — Binary attribute heads unchanged
- `src/training/loss.py` — Hierarchical detection/attribute loss unchanged
- `src/data/preprocess.py` — Label conversion unchanged
- `src/data/pseudo_label.py` — Pseudo labeling unchanged
- `src/data/dataset.py` — Dataset class unchanged
- `src/utils/fdi.py` — FDI utilities unchanged
- `src/utils/visualize.py` — Visualization unchanged
- `config/dataset.yaml` — YOLO dataset config unchanged
- `verify_local_image.py` — Local inference script unchanged
