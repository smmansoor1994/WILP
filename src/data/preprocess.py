"""
src/data/preprocess.py
======================
Convert Dentex Challenge 2023 COCO JSON annotations → YOLO extended format.

Dataset Folder Layout (DENTEX root provided at runtime via --dentex-root):

  DENTEX/
  ├── training_data/
  │   ├── quadrant/
  │   │   ├── train_quadrant.json          ← Part 1: quadrant bbox only
  │   │   └── xrays/                        ← Part 1 images
  │   ├── quadrant_enumeration/
  │   │   ├── train_quadrant_enumeration.json  ← Part 2: quadrant + tooth number
  │   │   └── xrays/
  │   ├── quadrant-enumeration-disease/    ← NOTE: hyphen, not underscore
  │   │   ├── train_quadrant_enumeration_disease.json  ← Part 3: + disease
  │   │   └── xrays/
  │   └── unlabelled/
  │       └── xrays/                        ← Unlabelled images for pseudo labels
  ├── validation_data/
  │   ├── validation_triple.json            ← Val annotations (same schema as Part 3)
  │   └── quadrant_enumeration_disease/
  │       └── xrays/                        ← Validation images
  └── test_data/
      └── disease/
          ├── input/                         ← Test images
          └── label/                         ← LabelMe JSONs (not used for training)

JSON Schema:
  Part 1 (train_quadrant.json):
    keys: images, annotations, categories
    annotations[i]: {image_id, bbox: [x,y,w,h], category_id, ...}
    categories:     [{id, name: str (quadrant number 1-4), supercategory}]
    NOTE: category_id is NOT necessarily quadrant-1 — always look up via categories[]

  Part 2 (train_quadrant_enumeration.json):
    keys: images, annotations, categories_1, categories_2
    annotations[i]: {image_id, bbox, category_id_1 (quadrant), category_id_2 (position)}
    categories_1:   [{id, name: int/str (1-4)}]   ← quadrant number
    categories_2:   [{id, name: str (1-8)}]        ← tooth position in quadrant

  Part 3 (train_quadrant_enumeration_disease.json) & validation_triple.json:
    keys: images, annotations, categories_1, categories_2, categories_3
    annotations[i]: {image_id, bbox, category_id_1, category_id_2, category_id_3}
    categories_3:   [{id:0,name:'Impacted'}, {id:1,name:'Caries'},
                     {id:2,name:'Periapical Lesion'}, {id:3,name:'Deep Caries'}]
    NOTE: ONE annotation per (tooth, disease). Teeth with multiple diseases have
          multiple annotation rows — this script MERGES them by (image_id, fdi).

YOLO Extended Label Format (10 columns per row):
  class_id  cx  cy  w  h  is_impacted  has_caries  has_deepcaries  has_lesion  data_type
  data_type: 0=quadrant_only, 1=enumeration_only, 2=enumeration+disease

Output splits:
  train/ ← All training_data images (Parts 1+2+3)
  val/   ← All validation_data images (from validation_triple.json)
  test/  ← All test_data/disease/input images

Unlabelled images are copied to data/unlabelled/ for use in pseudo_label.py.
"""

import json
import shutil
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from src.utils.fdi import fdi_to_class, quadrant_enum_to_fdi

logger = logging.getLogger(__name__)

# ─── Disease mapping: categories_3 id → attribute column index ───────────────
# Sourced directly from the Dentex JSON categories_3 field:
#   id=0 → 'Impacted'          → is_impacted    (label column 5, attr idx 0)
#   id=1 → 'Caries'            → has_caries     (label column 6, attr idx 1)
#   id=2 → 'Periapical Lesion' → has_lesion     (label column 8, attr idx 3)
#   id=3 → 'Deep Caries'       → has_deepcaries (label column 7, attr idx 2)
DISEASE_CAT3_TO_ATTR_IDX = {
    0: 0,   # Impacted          → is_impacted   [attrs[0]]
    1: 1,   # Caries            → has_caries    [attrs[1]]
    2: 3,   # Periapical Lesion → has_lesion    [attrs[3]]
    3: 2,   # Deep Caries       → has_deepcaries [attrs[2]]
}

RANDOM_SEED = 42


# ─── Main Entry Point ─────────────────────────────────────────────────────────

def preprocess_dentex(
    dentex_root: Path,
    out_dir: Path = Path("data/processed"),
) -> None:
    """Convert all Dentex Challenge 2023 data to YOLO extended format.

    Args:
        dentex_root: Root of the DENTEX dataset folder. Must contain
                     training_data/, validation_data/, and test_data/.
        out_dir:     Output directory for the processed YOLO-format dataset.
    """
    dentex_root = Path(dentex_root)
    out_dir = Path(out_dir)

    if not dentex_root.exists():
        raise FileNotFoundError(
            f"DENTEX root not found: '{dentex_root}'. "
            f"Pass the correct path via --dentex-root."
        )

    # Create output directory structure
    for split in ("train", "val", "test"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels_ext" / split).mkdir(parents=True, exist_ok=True)

    random.seed(RANDOM_SEED)

    # ── Part 1: Quadrant-only annotations (data_type=0) ───────────────────────
    _process_part1(
        images_dir=dentex_root / "training_data" / "quadrant" / "xrays",
        json_path=dentex_root / "training_data" / "quadrant" / "train_quadrant.json",
        out_dir=out_dir,
    )

    # ── Part 2: Quadrant + Enumeration annotations (data_type=1) ─────────────
    _process_part2(
        images_dir=dentex_root / "training_data" / "quadrant_enumeration" / "xrays",
        json_path=dentex_root / "training_data" / "quadrant_enumeration" / "train_quadrant_enumeration.json",
        out_dir=out_dir,
    )

    # ── Part 3: Quadrant + Enumeration + Disease annotations (data_type=2) ────
    # NOTE: This folder uses a hyphen: quadrant-enumeration-disease
    _process_part3(
        images_dir=dentex_root / "training_data" / "quadrant-enumeration-disease" / "xrays",
        json_path=dentex_root / "training_data" / "quadrant-enumeration-disease" / "train_quadrant_enumeration_disease.json",
        out_dir=out_dir,
    )

    # ── Validation: official validation_triple.json → val split ──────────────
    _process_validation(
        images_dir=dentex_root / "validation_data" / "quadrant_enumeration_disease" / "xrays",
        json_path=dentex_root / "validation_data" / "validation_triple.json",
        out_dir=out_dir,
    )

    # ── Test images: copy for inference ──────────────────────────────────────
    test_img_dir = dentex_root / "test_data" / "disease" / "input"
    if test_img_dir.exists():
        _copy_images(test_img_dir, out_dir / "images" / "test")
        logger.info("Copied test images from '%s'.", test_img_dir)
    else:
        logger.warning("Test image dir not found: '%s'. Skipping.", test_img_dir)

    # ── Unlabelled images: copy to data/unlabelled/ for pseudo labeling ───────
    unlabelled_src = dentex_root / "training_data" / "unlabelled" / "xrays"
    if unlabelled_src.exists():
        unlabelled_dst = out_dir.parent / "unlabelled"
        _copy_images(unlabelled_src, unlabelled_dst)
        logger.info(
            "Copied unlabelled images to '%s' (used in pseudo labeling).",
            unlabelled_dst,
        )
    else:
        logger.warning("Unlabelled dir not found: '%s'. Skipping.", unlabelled_src)

    # Remove stale ultralytics cache files so labels are re-verified with new format
    for _cache in (out_dir / "labels").rglob("*.cache"):
        _cache.unlink(missing_ok=True)

    logger.info("Preprocessing complete. Dataset written to '%s'.", out_dir)
    _log_stats(out_dir)


# ─── Part 1: Quadrant annotations ─────────────────────────────────────────────

def _process_part1(
    images_dir: Path,
    json_path: Path,
    out_dir: Path,
) -> None:
    """Convert Part 1 quadrant annotations (data_type=0).

    JSON schema:
      categories: [{id: int, name: str (quadrant number 1-4), supercategory}]
      annotations[i]: {image_id, bbox: [x,y,w,h], category_id: int, ...}

    class_id = quadrant - 1  (0-3, used for quadrant-level loss only)
    NOTE: category_id is NOT always quadrant-1; always look up via categories[].
    """
    if not json_path.exists():
        logger.warning("Part 1 JSON not found at '%s'. Skipping.", json_path)
        return

    logger.info("Processing Part 1 (quadrant) from '%s' ...", json_path)
    coco = _load_json(json_path)

    # Build category_id → quadrant number (1-4) from the categories list
    cat_to_quadrant: Dict[int, int] = {
        cat["id"]: int(cat["name"]) for cat in coco["categories"]
    }

    id_to_image = {img["id"]: img for img in coco["images"]}
    id_to_anns: Dict[int, List[dict]] = {}
    for ann in coco["annotations"]:
        id_to_anns.setdefault(ann["image_id"], []).append(ann)

    count = 0
    for image_id, img_info in id_to_image.items():
        img_w, img_h = img_info["width"], img_info["height"]
        src_image = images_dir / img_info["file_name"]
        stem = Path(img_info["file_name"]).stem

        labels = []
        for ann in id_to_anns.get(image_id, []):
            quadrant = cat_to_quadrant.get(ann["category_id"])
            if quadrant is None:
                logger.debug(
                    "Part 1 annotation %d: unknown category_id=%d. Skipping.",
                    ann["id"], ann["category_id"],
                )
                continue
            # class_id 0-3 (quadrant 1-4 → 0-indexed)
            # All attribute flags = 0; data_type = 0 (quadrant-only supervision)
            class_id = quadrant - 1
            cx, cy, bw, bh = _coco_bbox_to_yolo(ann["bbox"], img_w, img_h)
            labels.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} 0 0 0 0 0")

        _save_sample(src_image, labels, stem, "train", out_dir)
        count += 1

    logger.info("Part 1: processed %d images -> 'train'.", count)


# ─── Part 2: Quadrant + Enumeration annotations ────────────────────────────────

def _process_part2(
    images_dir: Path,
    json_path: Path,
    out_dir: Path,
) -> None:
    """Convert Part 2 quadrant+enumeration annotations (data_type=1).

    JSON schema:
      categories_1: [{id, name: int/str (quadrant 1-4)}]  ← quadrant number
      categories_2: [{id, name: str (position 1-8)}]       ← tooth number in quadrant
      annotations[i]: {image_id, bbox, category_id_1, category_id_2, ...}

    FDI = quadrant * 10 + position
    YOLO class_id = (quadrant - 1) * 8 + (position - 1)
    data_type = 1 (enumeration, no disease attributes)
    """
    if not json_path.exists():
        logger.warning("Part 2 JSON not found at '%s'. Skipping.", json_path)
        return

    logger.info("Processing Part 2 (quadrant+enumeration) from '%s' ...", json_path)
    coco = _load_json(json_path)

    # Build category_id → quadrant/position value from categories_1 and categories_2
    cat1_to_quadrant: Dict[int, int] = {
        cat["id"]: int(cat["name"]) for cat in coco["categories_1"]
    }
    cat2_to_position: Dict[int, int] = {
        cat["id"]: int(cat["name"]) for cat in coco["categories_2"]
    }

    id_to_image = {img["id"]: img for img in coco["images"]}
    id_to_anns: Dict[int, List[dict]] = {}
    for ann in coco["annotations"]:
        id_to_anns.setdefault(ann["image_id"], []).append(ann)

    count = 0
    for image_id, img_info in id_to_image.items():
        img_w, img_h = img_info["width"], img_info["height"]
        src_image = images_dir / img_info["file_name"]
        stem = Path(img_info["file_name"]).stem

        labels = []
        for ann in id_to_anns.get(image_id, []):
            quadrant = cat1_to_quadrant.get(ann.get("category_id_1"))
            position = cat2_to_position.get(ann.get("category_id_2"))

            if quadrant is None or position is None:
                logger.debug(
                    "Part 2 annotation %d: missing category_id_1/2. Skipping.",
                    ann["id"],
                )
                continue

            try:
                fdi = quadrant_enum_to_fdi(quadrant, position)
                class_id = fdi_to_class(fdi)
            except ValueError as e:
                logger.debug("FDI conversion error ann %d: %s", ann["id"], e)
                continue

            cx, cy, bw, bh = _coco_bbox_to_yolo(ann["bbox"], img_w, img_h)
            # data_type = 1 (enumeration only; disease attributes not available)
            labels.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} 0 0 0 0 1")

        _save_sample(src_image, labels, stem, "train", out_dir)
        count += 1

    logger.info("Part 2: processed %d images -> 'train'.", count)


# ─── Part 3: Quadrant + Enumeration + Disease annotations ─────────────────────

def _process_part3(
    images_dir: Path,
    json_path: Path,
    out_dir: Path,
) -> None:
    """Convert Part 3 disease annotations (data_type=2) → train split.

    JSON schema (same as validation_triple.json):
      categories_1: quadrant (1-4)
      categories_2: tooth position (1-8)
      categories_3: [{id:0,'Impacted'}, {id:1,'Caries'},
                     {id:2,'Periapical Lesion'}, {id:3,'Deep Caries'}]
      annotations[i]: {image_id, bbox, category_id_1, category_id_2, category_id_3}

    IMPORTANT: Part 3 only annotates DISEASED teeth (one annotation per disease
    per tooth). Healthy teeth have NO annotation here. The script merges multiple
    disease annotations for the same tooth into one YOLO row.
    Pseudo labels for healthy teeth are generated later in pseudo_label.py.
    """
    if not json_path.exists():
        logger.warning("Part 3 JSON not found at '%s'. Skipping.", json_path)
        return

    logger.info("Processing Part 3 (disease) from '%s' ...", json_path)
    _process_triple_json(images_dir, json_path, out_dir, split="train")


# ─── Validation ───────────────────────────────────────────────────────────────

def _process_validation(
    images_dir: Path,
    json_path: Path,
    out_dir: Path,
) -> None:
    """Convert validation_triple.json → val split.

    Schema is identical to Part 3 (categories_1, categories_2, categories_3).
    All images go to 'val' (no further splitting — this is the official val set).
    """
    if not json_path.exists():
        logger.warning("Validation JSON not found at '%s'. Skipping.", json_path)
        return

    logger.info("Processing validation data from '%s' ...", json_path)
    _process_triple_json(images_dir, json_path, out_dir, split="val")


# ─── Shared: process any JSON with categories_1/2/3 schema ───────────────────

def _process_triple_json(
    images_dir: Path,
    json_path: Path,
    out_dir: Path,
    split: str,
) -> None:
    """Process a COCO JSON with categories_1, categories_2, categories_3 fields.

    Merges multiple per-disease annotations for the same (image_id, fdi) key
    into a single YOLO row with OR-ed disease attribute flags and unioned bbox.
    """
    coco = _load_json(json_path)

    # Build category_id → value lookup tables
    cat1_to_quadrant: Dict[int, int] = {
        cat["id"]: int(cat["name"]) for cat in coco["categories_1"]
    }
    cat2_to_position: Dict[int, int] = {
        cat["id"]: int(cat["name"]) for cat in coco["categories_2"]
    }

    id_to_image = {img["id"]: img for img in coco["images"]}

    # Group annotations by (image_id, fdi) — merge multiple disease rows per tooth
    # tooth_data[key] = {"image_id", "fdi", "bbox": [x,y,w,h], "attrs": [0,0,0,0]}
    tooth_data: Dict[Tuple[int, int], dict] = {}

    for ann in coco["annotations"]:
        img_info = id_to_image.get(ann["image_id"])
        if img_info is None:
            continue

        quadrant = cat1_to_quadrant.get(ann.get("category_id_1"))
        position = cat2_to_position.get(ann.get("category_id_2"))

        if quadrant is None or position is None:
            logger.debug("Annotation %d: missing category_id_1/2. Skipping.", ann["id"])
            continue

        try:
            fdi = quadrant_enum_to_fdi(quadrant, position)
            fdi_to_class(fdi)  # validate range
        except ValueError as e:
            logger.debug("FDI error ann %d: %s", ann["id"], e)
            continue

        key = (ann["image_id"], fdi)
        disease_cat3 = ann.get("category_id_3")

        if key not in tooth_data:
            tooth_data[key] = {
                "image_id": ann["image_id"],
                "fdi": fdi,
                "bbox": list(ann["bbox"]),   # absolute [x, y, w, h]
                "attrs": [0, 0, 0, 0],       # [is_impacted, has_caries, has_deepcaries, has_lesion]
            }
        else:
            # Union bounding boxes for same tooth with multiple disease annotations
            tooth_data[key]["bbox"] = _merge_bboxes(
                tooth_data[key]["bbox"], ann["bbox"]
            )

        # Set disease attribute flag using categories_3 id → attr index mapping
        if disease_cat3 is not None:
            attr_idx = DISEASE_CAT3_TO_ATTR_IDX.get(disease_cat3)
            if attr_idx is not None:
                tooth_data[key]["attrs"][attr_idx] = 1

    # Group merged tooth data by image_id
    image_to_teeth: Dict[int, List[dict]] = {}
    for (image_id, fdi), data in tooth_data.items():
        image_to_teeth.setdefault(image_id, []).append(data)

    count = 0
    for image_id, img_info in id_to_image.items():
        img_w, img_h = img_info["width"], img_info["height"]
        src_image = images_dir / img_info["file_name"]
        stem = Path(img_info["file_name"]).stem

        teeth = image_to_teeth.get(image_id, [])
        labels = []
        for tooth in teeth:
            class_id = fdi_to_class(tooth["fdi"])
            cx, cy, bw, bh = _coco_bbox_to_yolo(tooth["bbox"], img_w, img_h)
            a = tooth["attrs"]
            # data_type = 2 (full supervision: FDI + disease attributes)
            labels.append(
                f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} "
                f"{a[0]} {a[1]} {a[2]} {a[3]} 2"
            )

        _save_sample(src_image, labels, stem, split, out_dir)
        count += 1

    logger.info(
        "'%s': processed %d images -> '%s' split.", json_path.name, count, split
    )


# ─── Helper Functions ─────────────────────────────────────────────────────────

def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _coco_bbox_to_yolo(
    bbox: List[float], img_w: int, img_h: int
) -> Tuple[float, float, float, float]:
    """Convert COCO bbox [x_min, y_min, w, h] to YOLO [cx, cy, w, h] normalized.

    COCO format: top-left corner + width/height (absolute pixels)
    YOLO format: center + dimensions (normalized 0-1)
    """
    x, y, w, h = bbox
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    # Clamp to [0, 1] to handle any annotation overflow
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    nw = max(0.0, min(1.0, nw))
    nh = max(0.0, min(1.0, nh))
    return cx, cy, nw, nh


def _merge_bboxes(
    b1: List[float], b2: List[float]
) -> List[float]:
    """Return the union bounding box of two COCO-format boxes [x,y,w,h]."""
    x1 = min(b1[0], b2[0])
    y1 = min(b1[1], b2[1])
    x2 = max(b1[0] + b1[2], b2[0] + b2[2])
    y2 = max(b1[1] + b1[3], b2[1] + b2[3])
    return [x1, y1, x2 - x1, y2 - y1]


def _save_sample(
    src_image: Path,
    labels: List[str],
    stem: str,
    split: str,
    out_dir: Path,
) -> None:
    """Copy image and write label file to the correct split directory.

    Args:
        src_image: Source image path.
        labels:    List of YOLO-format label strings.
        stem:      Filename stem (no extension).
        split:     "train", "val", or "test".
        out_dir:   Dataset root output directory.
    """
    # Determine image extension
    ext = src_image.suffix if src_image.exists() else ".png"

    dst_img = out_dir / "images" / split / f"{stem}{ext}"
    dst_lbl = out_dir / "labels" / split / f"{stem}.txt"
    dst_lbl_ext = out_dir / "labels_ext" / split / f"{stem}.txt"

    # Copy image
    if src_image.exists():
        shutil.copy2(src_image, dst_img)
    else:
        logger.warning("Source image not found: '%s'", src_image)

    # Write standard 5-column YOLO labels for ultralytics (class cx cy w h)
    yolo5_labels = [" ".join(lbl.split()[:5]) for lbl in labels]
    with open(dst_lbl, "w") as f:
        f.write("\n".join(yolo5_labels) + ("\n" if yolo5_labels else ""))

    # Write full 10-column extended labels for attribute head training
    dst_lbl_ext.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_lbl_ext, "w") as f:
        f.write("\n".join(labels) + ("\n" if labels else ""))


def _copy_images(src_dir: Path, dst_dir: Path) -> None:
    """Copy all images from src_dir to dst_dir."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    for img in src_dir.glob("*"):
        if img.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            shutil.copy2(img, dst_dir / img.name)


def _log_stats(out_dir: Path) -> None:
    """Log the number of images and labels in each split."""
    for split in ("train", "val", "test"):
        img_count = len(list((out_dir / "images" / split).glob("*")))
        lbl_count = len(list((out_dir / "labels" / split).glob("*.txt")))
        logger.info(
            "  %s: %d images, %d label files", split.upper(), img_count, lbl_count
        )
