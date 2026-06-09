# ARCHON — Arch-Contextualized Hierarchical Orthodontic Network

> **ARCHON: Arch-Contextualized Hierarchical Orthodontic Network for Severity-Aware Dental Radiograph Analysis**

Built on top of the baseline paper:
> **YOLOrtho: A Unified Framework for Teeth Enumeration and Dental Disease Detection**  
> Shenxiao Mei, Chenglong Ma, Feihong Shen, Huikai Wu, Kaidi Shen — Chohotech Inc.  
> arXiv: https://arxiv.org/abs/2308.05967

---

## Why ARCHON?

| Letter | Stands for | What it refers to in the model |
|---|---|---|
| **A** | Arch-Contextualized | The Swin Transformer `GlobalContextEncoder` operates on the P5 feature map where each token covers one tooth-sized patch — capturing the full **dental arch** (jaw curve) in a single attention pass |
| **R** | Radiograph | The input modality: panoramic dental X-rays (orthopantomograms / OPGs) |
| **C** | Cross-Attention | `MultiScaleFusion` uses cross-attention (Q = CNN local features, K/V = Swin global context) to let disease classifiers query anatomical position before making predictions |
| **H** | Hierarchical | Three-level disease severity (Healthy → Mild → Severe) output by `HybridMultiTaskHead`, and the multi-level FPN (strides 4 / 8 / 16) that processes teeth at multiple scales |
| **O** | Orthodontic | The clinical domain: FDI-numbered tooth enumeration, a core task in orthodontic and restorative treatment planning |
| **N** | Network | The end-to-end deep learning model |

**Full paper title:**
> *ARCHON: Arch-Contextualized Hierarchical Orthodontic Network for Severity-Aware Dental Radiograph Analysis*

**One-sentence description:**
ARCHON extends the ARCHON detection backbone with a Swin Transformer encoder that captures the full dental arch context, cross-attention fusion that guides disease features with global anatomical knowledge, and a hierarchical multi-task head that outputs clinically actionable three-level severity grades alongside FDI-consistent tooth enumeration on panoramic radiographs.

---

| Document | Purpose |
|---|---|
| `README.md` | Architecture overview, setup, and quick start (this file) |
| `PROJECT_GUIDE.md` | Detailed code walkthrough, label format, training phases, JSON schemas |
| `WORKFLOW.md` | End-to-end call graph for every mode including `train_hybrid` |
| `IMPROVEMENTS.md` | What changed from the baseline, why, and clinical benefit |

---

## What This Implements

### Baseline (foundation paper — arXiv:2308.05967)

| Paper Section | Code Location | Description |
|---|---|---|
| §2.1 Model Overall | `src/models/yolortho.py` | YOLOv8x + modifications |
| §2.2 CoordConv | `src/models/coord_conv.py` | Position-aware convolution |
| §2.1 Modified FPN | `src/models/yolortho.py` | Extra upsampling: strides 4,8,16 |
| §2.2 Attribute Heads | `src/models/heads.py` | 4 disease binary classifiers |
| §2.2 Hierarchical Loss | `src/training/loss.py` | Data-type–masked loss computation |
| §2.1 Flip Augmentation | `src/data/augmentation.py` | Quadrant-aware horizontal flip |
| §2.1 Data Preprocessing | `src/data/preprocess.py` | COCO JSON → YOLO format |
| §2.1 Pseudo Labeling | `src/data/pseudo_label.py` | Healthy teeth labeling for Part 3 |
| §2.3 Post-Processing | `src/inference/postprocess.py` | Linear sum assignment for FDI |

### ARCHON Improvements (this work)

| Improvement | Code Location | Description |
|---|---|---|
| A. GlobalContextEncoder | `src/models/swin_transformer.py` | Swin Transformer on P5 for full dental arch context |
| B. MultiScaleFusion | `src/models/cross_attention.py` | Cross-attention: CNN local features guided by global context |
| C. HybridMultiTaskHead | `src/models/hybrid_head.py` | 3-level severity heads + quadrant auxiliary classifier |
| D. CLAHE Augmentation | `src/data/augmentation.py` | Local contrast enhancement for early lesion visibility |
| E. Quadrant-Consistency | `src/inference/postprocess.py` | Quadrant penalty in Hungarian FDI assignment |

---

## Architecture

### Baseline

```
Input: Panoramic X-ray (1280×640)
  │
  ▼
Backbone: YOLOv8x + CoordConv          (strides 4, 8, 16, 32)
  │
  ▼
Neck: Modified PANet FPN               (extra upsampling → strides 4, 8, 16)
  │
  ├──► Detection Head                  (32 FDI tooth classes + bbox)
  └──► 4 × Attribute Heads             (is_impacted, has_caries, has_deepcaries, has_lesion)
  │
  ▼
Post-Processing: Linear Sum Assignment  (1 unique FDI per detection)
  │
  ▼
Output: Structured tooth detections with disease flags
```

### Hybrid (Phase 3, additive on top of baseline)

```
Backbone (frozen) → P3, P4, P5 features
                              │
              ┌───────────────┴────────────────────┐
              │                                    │
              ▼                                    ▼
     P5 only (stride 32)             P3, P4, P5 (all scales)
              │                                    │
              ▼                                    │
  GlobalContextEncoder                             │
  (2× Swin Transformer blocks)                     │
  window=4, heads=8                                │
              │                                    │
              │ global_context ───────────────────►│
                                                   ▼
                                       MultiScaleFusion
                                       Cross-Attention Q=CNN, K/V=Swin
                                       → fused P3, fused P4, fused P5
                                                   │
                                                   ▼
                                       HybridMultiTaskHead
                                       ├── SeverityHead ×4 diseases
                                       │   (Healthy / Mild / Severe)
                                       └── QuadrantAwareFDIHead
                                           (P(Q1) / P(Q2) / P(Q3) / P(Q4))
                                                   │
                                                   ▼
                              Linear Sum Assignment + quadrant-consistency penalty
                                                   │
                                                   ▼
                              Output: FDI + disease flags + severity_details
```

---

## Dataset

- **Source**: Dentex Challenge 2023 (local folder — no download required)
- **Dataset root**: `D:\WILP\sem-4\Dataset\DENTEX\DENTEXsample` (or your local path)
- **3 annotated training parts + unlabelled images + official validation set**:

| Folder | JSON Annotation File | Schema | data_type |
|---|---|---|---|
| `training_data/quadrant/xrays/` | `train_quadrant.json` | `images, annotations, categories` | 0 |
| `training_data/quadrant_enumeration/xrays/` | `train_quadrant_enumeration.json` | `images, annotations, categories_1, categories_2` | 1 |
| `training_data/quadrant-enumeration-disease/xrays/` | `train_quadrant_enumeration_disease.json` | `images, annotations, categories_1, categories_2, categories_3` | 2 |
| `training_data/unlabelled/xrays/` | *(none)* | Raw images, pseudo-labeled at runtime | 2 |
| `validation_data/quadrant_enumeration_disease/xrays/` | `validation_triple.json` | Same as Part 3 | 2 |
| `test_data/disease/input/` | per-image LabelMe JSONs | LabelMe polygon format | — |

**JSON key differences across parts:**
- Part 1: uses `categories` (single), `category_id` in annotations
- Part 2: uses `categories_1` (quadrant 1-4) + `categories_2` (position 1-8), `category_id_1`/`category_id_2`
- Part 3 + Validation: adds `categories_3` (disease names), `category_id_3`

**`categories_3` disease mapping:**

| id | name | Attribute flag |
|---|---|---|
| 0 | Impacted | `is_impacted` |
| 1 | Caries | `has_caries` |
| 2 | Periapical Lesion | `has_lesion` |
| 3 | Deep Caries | `has_deepcaries` |

---

## Setup & Installation

### 1. Clone and install dependencies
```bash
pip install -r requirements.txt
```

### 2. Point to the DENTEX dataset folder
No download needed. The dataset is expected on local disk. Pass `--dentex-root` when running.
```bash
# Example path (adjust to your local path)
--dentex-root D:/WILP/sem-4/Dataset/DENTEX/DENTEXsample
```

---

## Running the Pipeline

### Available modes

| Mode | Purpose |
|---|---|
| `preprocess` | Convert DENTEX COCO JSON → extended YOLO labels |
| `pseudo_label` | Generate healthy-tooth pseudo labels for Part 3 + unlabelled images |
| `train` | Phase 1 detection + Phase 2 fine-tuning + Phase 2b attribute heads |
| `train_attr` | Re-train binary disease attribute heads only (Phase 2b in isolation) |
| `train_hybrid` | Phase 3: train Swin encoder + cross-attention + severity heads **[NEW]** |
| `evaluate` | Run YOLO val on official validation set |
| `predict` | Run inference on new X-rays (auto-loads hybrid weights when available) |
| `full` | Run all baseline stages end-to-end |

### Full pipeline (recommended)
```bash
python main.py --mode full --device cuda --dentex-root D:/WILP/sem-4/Dataset/DENTEX/DENTEXsample
```

> **No GPU?** Replace `--device cuda` with `--device cpu`, or omit `--device` entirely (auto-detects and falls back to CPU):
> ```bash
> python main.py --mode full --device cpu --dentex-root D:/WILP/sem-4/Dataset/DENTEX/DENTEXsample
> ```

### Step-by-step
```bash
# Step 1: Preprocess — convert COCO JSON → YOLO format
python main.py --mode preprocess --dentex-root D:/WILP/sem-4/Dataset/DENTEX/DENTEXsample
# Creates: data/processed/ (train/ + val/ from validation_triple.json + test/)
# Also copies: data/unlabelled/ for pseudo labeling

# Step 2: Train Phase 1 (detection only)
python main.py --mode train --device cuda   # or --device cpu if no GPU

# Step 3: Generate pseudo labels for healthy teeth
python main.py --mode pseudo_label
# Processes: Part 3 images (adds non-disease detections)
# Processes: data/unlabelled/ images (all detections → healthy pseudo labels)

# Step 4: Re-train with pseudo labels + disease attribute heads
python main.py --mode train --device cuda   # or --device cpu if no GPU

# Step 5: Train hybrid components — Swin encoder + cross-attention + severity heads
#         (runs on top of the Phase 2 checkpoint; backbone stays frozen)
python main.py --mode train_hybrid --device cuda

# Step 6: Evaluate on official validation set (validation_triple.json)
python main.py --mode evaluate --weights weights/archon_best.pt

# Step 7: Run inference on new X-rays
#         (auto-loads hybrid_best.pt when present for severity-annotated output)
python main.py --mode predict --input path/to/xray.jpg --weights weights/archon_best.pt
```

### Resume interrupted training
```bash
python main.py --mode train --resume
```

---

## Verify on a Local Panoramic Image

Use `verify_local_image.py` to run inference on any local dental X-ray and display results.

### Syntax
```powershell
python verify_local_image.py `
  --image          <path to panoramic X-ray (.jpg / .png)> `
  --weights        <path to archon_best.pt> `
  --attr-weights   <path to attr_best.pt> `
  --device         cpu `
  --conf           0.25 `
  --iou            0.45 `
  --attr-threshold 0.3 `
  --attr-mode      per_tooth `
  --save-dir       <folder to save results> `
  --show
```

### Example — full run with disease detection (recommended after retraining)
```powershell
python verify_local_image.py `
  --image "D:\WILP\sem-4\Dataset\DENTEX\DENTEX\training_data\quadrant-enumeration-disease\xrays\train_10.png" `
  --weights "D:\WILP\Workingcode\Baseline\WILP\weights\archon_best.pt" `
  --attr-weights "D:\WILP\Workingcode\Baseline\WILP\weights\attr_best.pt" `
  --device cuda `
  --attr-threshold 0.3 `
  --save-dir "C:\Users\Z0046KUF\Downloads\archon_results" `
  --show
```

> **train_10 sanity check**: this image has 5 teeth with `has_caries=1` in the ground truth.  
> After a correct retrain you should see those 5 teeth labeled `D: Caries`.  
> If all teeth still show `D: Healthy`, check the training log for `Phase 2b` lines with `num_attr_samples > 0`.

### Example — basic run (detection only, no attr-weights)
```powershell
python verify_local_image.py `
  --image "C:\Users\Z0046KUF\Downloads\pnmc.jpg" `
  --weights "D:\WILP\Workingcode\Baseline\WILP\weights\archon_best.pt" `
  --device cpu `
  --show
```

### Example — lower confidence to detect more teeth
```powershell
python verify_local_image.py `
  --image "C:\Users\Z0046KUF\Downloads\pnmc.jpg" `
  --weights "D:\WILP\Workingcode\Baseline\WILP\weights\archon_best.pt" `
  --attr-weights "D:\WILP\Workingcode\Baseline\WILP\weights\attr_best.pt" `
  --device cpu `
  --conf 0.10 `
  --attr-threshold 0.3 `
  --save-dir "C:\Users\Z0046KUF\Downloads\archon_results" `
  --show
```

### All available options

| Option | Default | Description |
|---|---|---|
| `--image` | *(required)* | Path to panoramic X-ray (`.jpg` / `.png`) |
| `--weights` | auto-detected | `archon_best.pt` — detection + FDI enumeration model |
| `--attr-weights` | auto-searched | `attr_best.pt` — disease attribute head weights |
| `--device` | `cpu` | `cpu` or `cuda` (GPU) |
| `--conf` | `0.25` | Detection confidence threshold — lower = more teeth detected |
| `--iou` | `0.45` | NMS IoU threshold |
| `--attr-threshold` | `0.3` | Disease probability threshold — lower = more sensitive to disease; raise to `0.45` if too many false positives |
| `--attr-mode` | `per_tooth` | `per_tooth` for retrained models; `global_avg` for pre-June-2026 weights |
| `--save-dir` | `outputs/predictions/` | Folder to save annotated image + JSON results |
| `--output-dir` | `outputs/predictions/` | Fallback save folder (used if `--save-dir` not set) |
| `--show` | off | Display annotated image via matplotlib |
| `--no-save` | off | Skip saving all output files |

### `--attr-threshold` tuning guide

| Scenario | Recommended value |
|---|---|
| Initial verification on training images | `0.3` |
| Too many false positives (healthy teeth flagged as diseased) | `0.40`–`0.45` |
| Still missing diseases on known-diseased training images | `0.20` |
| Final evaluation / paper reporting | Tune on validation set |

### Auto weight search (if `--weights` is omitted)
The script searches for weights in this order:
1. `weights/archon_best.pt`
2. `outputs/runs/phase2/weights/best.pt`
3. `outputs/runs/phase1/weights/best.pt`

### Output files
| File | Description |
|---|---|
| `<name>_vis.jpg` | Annotated image with bounding boxes, FDI labels, and disease flags |
| `<name>_result.json` | Per-tooth JSON: FDI number, confidence, bbox, disease attributes |

Both files are written to `--save-dir` (or `--output-dir` if not set).  
Use `--no-save` to suppress all file output.

---

## Project Structure

```
ARCHON/
├── main.py                     # Entry point — runs all pipeline stages
├── requirements.txt
├── PROJECT_GUIDE.md            # Detailed step-by-step guide
│
├── config/
│   ├── model_config.yaml       # Model architecture settings
│   ├── dataset.yaml            # YOLO dataset paths + 32 FDI class names
│   └── train_config.yaml       # Training hyperparameters
│
├── data/                        # Auto-created by preprocess stage
│   ├── processed/               # YOLO-format converted dataset
│   │   ├── images/{train,val,test}/
│   │   └── labels/{train,val,test}/  # 10-column extended YOLO .txt files
│   ├── pseudo/                  # After pseudo_label stage
│   └── unlabelled/              # Unlabelled images (copied by preprocess)
│
├── src/
│   ├── data/
│   │   ├── preprocess.py       # COCO JSON → YOLO label conversion (all 3 parts + val)
│   │   ├── pseudo_label.py     # Pseudo-labeling: Part 3 healthy + unlabelled images
│   │   ├── augmentation.py     # Flip + CLAHE + quadrant remapping augmentation  [+CLAHE]
│   │   └── dataset.py          # PyTorch Dataset with extended labels
│   │
│   ├── models/
│   │   ├── coord_conv.py       # CoordConv: add (x,y) coordinate channels
│   │   ├── yolortho.py         # `ARCHON` base + `ARCHONModel` hybrid        [+Hybrid]
│   │   ├── heads.py            # Binary disease attribute prediction heads
│   │   ├── swin_transformer.py # GlobalContextEncoder (Swin Transformer)      [NEW]
│   │   ├── cross_attention.py  # MultiScaleFusion (Cross-Attention)           [NEW]
│   │   └── hybrid_head.py      # HybridMultiTaskHead (Severity + Quadrant)    [NEW]
│   │
│   ├── training/
│   │   ├── trainer.py          # ARCHONTrainer + ARCHONHybridTrainer      [+Hybrid]
│   │   └── loss.py             # Hierarchical BCE + detection loss
│   │
│   ├── inference/
│   │   ├── predictor.py        # ARCHONPredictor + ARCHONHybridPredictor  [+Hybrid]
│   │   └── postprocess.py      # Linear sum assignment + quadrant penalty     [+Penalty]
│   │
│   └── utils/
│       ├── fdi.py              # FDI tooth numbering system utilities
│       └── visualize.py        # Bounding box + dental chart visualization
│
├── weights/
│   ├── archon_best.pt        # Phase 2 detection checkpoint
│   ├── attr_best.pt            # Phase 2b binary attribute head checkpoint
│   └── hybrid_best.pt          # Phase 3 Swin+Cross-Attn+Severity checkpoint  [NEW]
└── outputs/
    ├── runs/                   # Training logs + checkpoints
    └── predictions/            # Inference results (images + JSON)
```

---

## FDI Tooth Numbering System

The FDI (Fédération Dentaire Internationale) system uses 2-digit codes:
- **First digit**: Quadrant (1=Upper-Right, 2=Upper-Left, 3=Lower-Left, 4=Lower-Right)
- **Second digit**: Tooth position (1=Central Incisor … 8=Wisdom Tooth)

| Quadrant | FDI Range | YOLO Classes |
|---|---|---|
| Q1 (Upper-Right) | 11–18 | 0–7 |
| Q2 (Upper-Left)  | 21–28 | 8–15 |
| Q3 (Lower-Left)  | 31–38 | 16–23 |
| Q4 (Lower-Right) | 41–48 | 24–31 |

---

## Results (from the paper)

| Model | AP-Quadrant | AP-Diagnosis | AP-Enumeration |
|---|---|---|---|
| Vanilla YOLO | 0.395 | 0.330 | 0.286 |
| **YOLOrtho** | **0.414** | **0.357** | **0.337** |
| HierarchicalDet | 0.365 | 0.341 | 0.221 |

Ablation study:
| Upsampling | CoordConv | Post-process | AP-Q | AP-D | AP-E |
|---|---|---|---|---|---|
| – | – | – | 0.469 | 0.410 | 0.359 |
| ✓ | – | – | 0.522 | 0.475 | 0.417 |
| ✓ | ✓ | – | 0.545 | 0.494 | 0.438 |
| ✓ | ✓ | ✓ | 0.546 | 0.494 | **0.446** |

---

## Key Technical Contributions

### 1. CoordConv Backbone
Standard convolutions are **translation-invariant** — they don't know WHERE in the image they're processing. For teeth enumeration, position is the primary signal (upper vs. lower, left vs. right). CoordConv appends normalized (x, y) coordinate channels before each convolution in the backbone.

### 2. Modified FPN (Extra Upsampling)
Standard YOLOv8 detects at strides [8, 16, 32]. Panoramic X-rays are wide (2400×1200px+), and teeth are relatively large. Adding an extra upsampling layer shifts detection to strides [4, 8, 16], improving localization for large/medium objects.

### 3. Disease as Attributes (Multi-label)
Instead of treating diseases as separate object classes (requiring two models), each tooth detection carries 4 binary attribute predictions. This enables end-to-end training for both tasks simultaneously.

### 4. Hierarchical Training
The 3 data parts have different annotation completeness. Losses are computed selectively:
- `data_type=0` (quadrant only): bbox + quadrant class loss
- `data_type=1` (enumeration): bbox + FDI class loss  
- `data_type=2` (disease): bbox + FDI class + attribute loss

### 5. Linear Sum Assignment Post-processing
The Hungarian algorithm ensures each FDI tooth number is assigned to at most one detection, correcting the common error of assigning the same number to adjacent teeth.

---

## Citation

```bibtex
@article{mei2023yolortho,
  title={YOLOrtho: A Unified Framework for Teeth Enumeration and Dental Disease Detection},
  author={Mei, Shenxiao and Ma, Chenglong and Shen, Feihong and Wu, Huikai and Shen, Kaidi},
  journal={arXiv preprint arXiv:2308.05967},
  year={2023}
}
```
