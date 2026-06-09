"""
src/data/pseudo_label.py
=========================
ARCHON / ARCHON — Generate pseudo labels for healthy teeth using a Phase 1 detector.

Background (Section 2.1 of the paper):
  Part 3 of the Dentex dataset only annotates DISEASED teeth.
  Healthy teeth in those images have no annotation. If trained as-is,
  the model is penalised for correctly detecting healthy teeth.

  Additionally, the dataset includes UNLABELLED images (training_data/unlabelled/)
  with no annotations at all — these are used as extra training data via
  pseudo labels for ALL detected teeth.

Solution (this module):
  1. Run the Phase 1 detector on Part 3 training images.
     Keep detections that don’t overlap existing disease labels → healthy pseudo labels.
  2. Run the Phase 1 detector on unlabelled images.
     All detected teeth → pseudo labels (healthy, all attrs=0).
  3. Merge pseudo labels with existing disease labels for Part 3.
  4. Write merged dataset to pseudo_dir/.

Pseudo label format: class_id cx cy w h 0 0 0 0 2
  - All attribute flags = 0 (healthy)
  - data_type = 2 (enables attribute loss with zero targets during training)
"""

import logging
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

# IoU threshold — if a detection overlaps this much with an existing label, skip it
OVERLAP_IOU_THRESHOLD = 0.3
# Minimum confidence for a pseudo label to be accepted
MIN_CONFIDENCE = 0.5


def generate_pseudo_labels(
    data_dir: Path = Path("data/processed"),
    pseudo_dir: Path = Path("data/pseudo"),
    unlabelled_dir: Optional[Path] = None,
    weights_path: Optional[Path] = None,
    device: str = "cuda",
    conf: float = MIN_CONFIDENCE,
) -> None:
    """Generate pseudo labels for healthy teeth.

    Processes two image sources:
      1. Part 3 images in data_dir (data_type=2): detections that don’t
         overlap existing disease labels become healthy pseudo labels.
      2. Unlabelled images in unlabelled_dir: all detections become
         healthy pseudo labels (no existing labels to check against).

    Args:
        data_dir:        Processed dataset directory (output of preprocess.py).
        pseudo_dir:      Output directory for merged labels with pseudo labels.
        unlabelled_dir:  Directory containing unlabelled images
                         (data/unlabelled/ — copied there by preprocess.py).
                         Pass None to skip.
        weights_path:    Path to trained Phase 1 detection model (.pt).
                         Auto-detected from outputs/runs/ if None.
        device:          Torch device string ('cuda' or 'cpu').
        conf:            Minimum detection confidence threshold.
    """
    data_dir = Path(data_dir)
    pseudo_dir = Path(pseudo_dir)
    pseudo_dir.mkdir(parents=True, exist_ok=True)

    # ── Find model weights ────────────────────────────────────────────────────
    if weights_path is None:
        weights_path = _find_latest_weights()
        if weights_path is None:
            logger.warning(
                "No model weights found for pseudo labeling. "
                "Train the model on Part 1+2 first (python main.py --mode train), "
                "then re-run pseudo labeling."
            )
            return

    logger.info("Using model weights: '%s'", weights_path)

    # ── Load model ────────────────────────────────────────────────────────────
    try:
        from ultralytics import YOLO
        model = YOLO(str(weights_path))
        model.to(device)
        logger.info("Model loaded on device: %s", device)
    except Exception as e:
        logger.error("Failed to load model: %s", e)
        return

    # ── Process Part 3 train images ───────────────────────────────────────────
    for split in ("train", "val"):
        img_dir = data_dir / "images" / split
        lbl_dir = data_dir / "labels" / split          # 5-col standard YOLO labels
        ext_lbl_dir = data_dir / "labels_ext" / split  # 10-col extended labels
        out_lbl_dir = pseudo_dir / "labels" / split          # 5-col output
        out_ext_lbl_dir = pseudo_dir / "labels_ext" / split  # 10-col output
        out_img_dir = pseudo_dir / "images" / split
        out_lbl_dir.mkdir(parents=True, exist_ok=True)
        out_ext_lbl_dir.mkdir(parents=True, exist_ok=True)
        out_img_dir.mkdir(parents=True, exist_ok=True)

        # Find Part 3 label files (data_type == 2 in at least one row).
        # MUST search labels_ext/ (10-col) — labels/ only has 5 columns so
        # the data_type column (col 9) is never present there.
        part3_stems = _find_part3_stems(ext_lbl_dir)
        logger.info(
            "Found %d Part 3 (disease-labeled) images in '%s' split "
            "(searched labels_ext/).",
            len(part3_stems), split,
        )

        for stem in part3_stems:
            ext_lbl_path = ext_lbl_dir / f"{stem}.txt"
            lbl_path = lbl_dir / f"{stem}.txt"
            img_path = _find_image(img_dir, stem)
            if img_path is None:
                continue

            # Load 10-col labels from labels_ext/ if available; else pad from labels/
            existing_labels = _load_labels(ext_lbl_path if ext_lbl_path.exists() else lbl_path)

            image = cv2.imread(str(img_path))
            if image is None:
                continue

            results = model.predict(
                source=image,
                conf=conf,
                iou=0.45,
                max_det=32,
                verbose=False,
            )

            pseudo = _extract_pseudo_labels(
                results=results,
                existing_labels=existing_labels,
                iou_threshold=OVERLAP_IOU_THRESHOLD,
            )

            merged_10col = list(existing_labels) + pseudo

            # Write 5-col to labels/ (for YOLO detection training)
            out_lbl = out_lbl_dir / f"{stem}.txt"
            with open(out_lbl, "w") as f:
                for row in merged_10col:
                    f.write(" ".join(str(v) for v in list(row)[:5]) + "\n")

            # Write 10-col to labels_ext/ (for attr head training)
            out_ext_lbl = out_ext_lbl_dir / f"{stem}.txt"
            with open(out_ext_lbl, "w") as f:
                for row in merged_10col:
                    row_10 = list(row) + [0] * max(0, 10 - len(list(row)))
                    f.write(" ".join(str(v) for v in row_10[:10]) + "\n")

            out_img = out_img_dir / img_path.name
            if not out_img.exists():
                shutil.copy2(img_path, out_img)

        # For non-Part 3 images: copy 5-col labels and 10-col labels_ext if available
        for lbl_path in lbl_dir.glob("*.txt"):
            stem = lbl_path.stem
            if stem not in part3_stems:
                shutil.copy2(lbl_path, out_lbl_dir / lbl_path.name)
                ext_src = ext_lbl_dir / lbl_path.name
                if ext_src.exists():
                    shutil.copy2(ext_src, out_ext_lbl_dir / lbl_path.name)
                img_path = _find_image(img_dir, stem)
                if img_path:
                    dest = out_img_dir / img_path.name
                    if not dest.exists():
                        shutil.copy2(img_path, dest)

    # ── Process unlabelled images ─────────────────────────────────────────────
    # All detections become healthy pseudo labels (no existing labels to check)
    if unlabelled_dir is not None:
        unlabelled_dir = Path(unlabelled_dir)
        if unlabelled_dir.exists():
            _process_unlabelled(
                unlabelled_dir=unlabelled_dir,
                pseudo_dir=pseudo_dir,
                model=model,
                conf=conf,
            )
        else:
            logger.warning(
                "Unlabelled image dir not found: '%s'. Skipping.", unlabelled_dir
            )

    logger.info(
        "Pseudo labeling complete. Merged dataset written to '%s'.", pseudo_dir
    )


# ─── Helper: Process unlabelled images ───────────────────────────────────────

def _process_unlabelled(
    unlabelled_dir: Path,
    pseudo_dir: Path,
    model,
    conf: float,
) -> None:
    """Generate pseudo labels for all images in unlabelled_dir.

    Since these images have no ground-truth annotations, EVERY detected tooth
    becomes a healthy pseudo label (all attribute flags = 0, data_type = 2).

    Writes:
      - 5-col format to pseudo_dir/labels/train/       (for YOLO detection training)
      - 10-col format to pseudo_dir/labels_ext/train/  (for attr head training)
    """
    out_img_dir = pseudo_dir / "images" / "train"
    out_lbl_dir = pseudo_dir / "labels" / "train"
    out_ext_lbl_dir = pseudo_dir / "labels_ext" / "train"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)
    out_ext_lbl_dir.mkdir(parents=True, exist_ok=True)

    image_files = [
        p for p in unlabelled_dir.glob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    ]
    logger.info(
        "Processing %d unlabelled images for pseudo labels ...", len(image_files)
    )

    for img_path in image_files:
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        results = model.predict(
            source=image,
            conf=conf,
            iou=0.45,
            max_det=32,
            verbose=False,
        )

        # All detected teeth are healthy (no existing labels)
        pseudo = _extract_pseudo_labels(
            results=results,
            existing_labels=[],          # no existing labels
            iou_threshold=0.0,           # keep all detections
        )

        # Only write if at least one tooth was detected
        if not pseudo:
            continue

        stem = img_path.stem

        # Write 5-col to labels/ (for YOLO detection training)
        out_lbl = out_lbl_dir / f"{stem}.txt"
        with open(out_lbl, "w") as f:
            for row in pseudo:
                f.write(" ".join(str(v) for v in row[:5]) + "\n")

        # Write 10-col to labels_ext/ (for attr head training)
        out_ext_lbl = out_ext_lbl_dir / f"{stem}.txt"
        with open(out_ext_lbl, "w") as f:
            for row in pseudo:
                f.write(" ".join(str(v) for v in row[:10]) + "\n")

        dest_img = out_img_dir / img_path.name
        if not dest_img.exists():
            shutil.copy2(img_path, dest_img)

    logger.info(
        "Unlabelled pseudo labeling done: results in '%s'.", out_lbl_dir
    )


# ─── Helper Functions ─────────────────────────────────────────────────────────

def _find_part3_stems(lbl_dir: Path) -> set:
    """Return stems (as a set) of label files that contain at least one data_type=2 row.

    Returns a set instead of a list for O(1) membership tests in the calling loop
    (avoids O(n²) scan when checking `stem not in part3_stems` for every label file).
    """
    part3: set = set()
    for lbl_path in lbl_dir.glob("*.txt"):
        labels = _load_labels(lbl_path)
        for row in labels:
            if len(row) >= 10 and int(row[9]) == 2:
                part3.add(lbl_path.stem)
                break
    return part3


def _load_labels(lbl_path: Path) -> List[List[float]]:
    """Load a YOLO-format label file. Returns list of rows (each a list of floats)."""
    if not lbl_path.exists():
        return []
    rows = []
    with open(lbl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append([float(v) for v in line.split()])
    return rows


def _find_image(img_dir: Path, stem: str) -> Optional[Path]:
    """Find image file with matching stem, trying common extensions."""
    for ext in (".jpg", ".jpeg", ".png", ".bmp"):
        p = img_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def _extract_pseudo_labels(
    results,
    existing_labels: List[List[float]],
    iou_threshold: float,
) -> List[List[float]]:
    """Extract pseudo labels from model predictions.

    Only keeps detections that:
    1. Don't significantly overlap with existing disease labels (new teeth).
    2. Are for classes 0-31 (valid FDI tooth numbers).

    Pseudo labels get:
    - All disease attributes = 0 (healthy)
    - data_type = 2 (same as Part 3 so disease-attribute loss is applied,
                     but targets are all-zero = healthy)
    """
    pseudo = []

    if not results or results[0].boxes is None:
        return pseudo

    boxes = results[0].boxes
    if boxes.xywhn is None or len(boxes.xywhn) == 0:
        return pseudo

    # Normalized [cx, cy, w, h] for all detections
    detected = boxes.xywhn.cpu().numpy()  # shape (N, 4)
    detected_cls = boxes.cls.cpu().numpy().astype(int)  # shape (N,)

    for i, (det_box, det_cls) in enumerate(zip(detected, detected_cls)):
        # Skip invalid class indices
        if det_cls < 0 or det_cls > 31:
            continue

        # Check overlap with existing disease labels
        overlaps_existing = False
        for existing in existing_labels:
            if len(existing) < 5:
                continue
            ex_box = existing[1:5]  # cx, cy, w, h
            iou = _compute_iou_xywh(det_box, ex_box)
            if iou > iou_threshold:
                overlaps_existing = True
                break

        if not overlaps_existing:
            # Healthy tooth pseudo label: all attributes = 0, data_type = 2
            cx, cy, w, h = det_box
            pseudo.append([
                det_cls,           # class_id (FDI)
                float(cx),         # cx
                float(cy),         # cy
                float(w),          # w
                float(h),          # h
                0,                 # is_impacted = 0 (healthy)
                0,                 # has_caries  = 0
                0,                 # has_deepcaries = 0
                0,                 # has_lesion  = 0
                2,                 # data_type = 2 (disease labels available)
            ])

    return pseudo


def _compute_iou_xywh(
    box1: np.ndarray, box2: np.ndarray
) -> float:
    """Compute IoU between two boxes in [cx, cy, w, h] normalized format."""
    # Convert to [x1, y1, x2, y2]
    b1 = _xywh_to_xyxy(box1)
    b2 = _xywh_to_xyxy(box2)

    # Intersection
    ix1 = max(b1[0], b2[0])
    iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2])
    iy2 = min(b1[3], b2[3])

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    inter = (ix2 - ix1) * (iy2 - iy1)
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = area1 + area2 - inter

    return inter / (union + 1e-6)


def _xywh_to_xyxy(box: np.ndarray) -> Tuple[float, float, float, float]:
    cx, cy, w, h = box
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


def _find_latest_weights() -> Optional[Path]:
    """Search outputs/runs for the most recently saved best.pt.

    Uses the pseudo_label.py module location to resolve the project root,
    avoiding CWD-dependent relative paths that break when main.py is invoked
    from a directory other than the WILP project root.
    """
    # Resolve project root from this file's location:
    # src/data/pseudo_label.py → src/data/ → src/ → project_root/
    _module_root = Path(__file__).resolve().parent.parent.parent
    runs_dir = _module_root / "outputs" / "runs"
    if not runs_dir.exists():
        return None
    weight_files = sorted(runs_dir.rglob("best.pt"), key=lambda p: p.stat().st_mtime)
    return weight_files[-1] if weight_files else None
