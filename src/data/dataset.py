"""
src/data/dataset.py
===================
PyTorch Dataset for YOLOrtho with extended label format.

Label file format (10 columns per detection):
  class_id  cx  cy  w  h  is_impacted  has_caries  has_deepcaries  has_lesion  data_type

  class_id : int 0-31 (FDI tooth number, or 0-3 for quadrant-only data)
  cx, cy   : float [0,1] normalized center coordinates
  w, h     : float [0,1] normalized width and height
  attr0-3  : int {0,1} disease attribute flags
  data_type: int {0,1,2} annotation completeness flag

Usage:
    dataset = YOLOrthoDataset(
        images_dir="data/processed/images/train",
        labels_dir="data/processed/labels/train",
        img_size=(640, 1280),
        augment=True,
    )
    img, labels = dataset[0]
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.augmentation import YOLOrthoAugmentor

logger = logging.getLogger(__name__)

# Number of columns in each label row
LABEL_COLS = 10   # class, cx, cy, w, h, attr×4, data_type

# Image extensions to search for
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


class YOLOrthoDataset(Dataset):
    """Dataset loading panoramic X-ray images + extended YOLO labels.

    Args:
        images_dir: Directory containing image files.
        labels_dir: Directory containing .txt label files.
        img_size:   Output image size as (height, width).
        augment:    Whether to apply augmentation.
        augmentor:  Optional custom YOLOrthoAugmentor instance.
    """

    def __init__(
        self,
        images_dir: Path,
        labels_dir: Path,
        img_size: Tuple[int, int] = (640, 1280),
        augment: bool = False,
        augmentor: Optional[YOLOrthoAugmentor] = None,
    ):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.img_size = img_size   # (H, W)
        self.augment = augment
        self.augmentor = augmentor or YOLOrthoAugmentor()

        # Collect all image paths
        self.image_paths = sorted([
            p for p in self.images_dir.glob("*")
            if p.suffix.lower() in IMG_EXTENSIONS
        ])

        if len(self.image_paths) == 0:
            logger.warning("No images found in '%s'.", images_dir)

        logger.info(
            "YOLOrthoDataset: %d images in '%s'",
            len(self.image_paths),
            images_dir,
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load and return (image, labels) for index idx.

        Returns:
            image:  Float tensor (3, H, W) normalized to [0, 1].
            labels: Float tensor (N, LABEL_COLS) — N detections in this image.
                    Empty tensor shape (0, LABEL_COLS) if no annotations.
        """
        img_path = self.image_paths[idx]
        lbl_path = self.labels_dir / (img_path.stem + ".txt")

        # ── Load image ────────────────────────────────────────────────────────
        image = cv2.imread(str(img_path))
        if image is None:
            logger.warning("Could not read image '%s'. Using blank.", img_path)
            h, w = self.img_size
            image = np.full((h, w, 3), 114, dtype=np.uint8)

        # Convert BGR (OpenCV default) to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # ── Load labels ───────────────────────────────────────────────────────
        labels = self._load_labels(lbl_path)

        # ── Resize image to target size ───────────────────────────────────────
        h, w = self.img_size
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)

        # ── Augmentation ──────────────────────────────────────────────────────
        if self.augment and len(labels) > 0:
            image, labels = self.augmentor(image, labels)

        # ── Normalize + convert to tensor ────────────────────────────────────
        # Normalize to [0, 1], then (H, W, C) → (C, H, W)
        img_tensor = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0

        if len(labels) > 0:
            lbl_tensor = torch.from_numpy(labels).float()
        else:
            lbl_tensor = torch.zeros((0, LABEL_COLS), dtype=torch.float32)

        return img_tensor, lbl_tensor

    def _load_labels(self, lbl_path: Path) -> np.ndarray:
        """Read a label file and return Nx10 float array.

        Handles:
        - Missing files → empty array
        - Short rows (< 10 cols) → padded with zeros
        - Invalid rows → skipped with warning
        """
        if not lbl_path.exists():
            return np.zeros((0, LABEL_COLS), dtype=np.float32)

        rows = []
        with open(lbl_path) as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    vals = [float(v) for v in line.split()]
                except ValueError:
                    logger.debug(
                        "Invalid value in '%s' line %d — skipped.", lbl_path, lineno
                    )
                    continue

                if len(vals) < 5:
                    logger.debug(
                        "Short row in '%s' line %d (< 5 cols) — skipped.", lbl_path, lineno
                    )
                    continue

                # Pad to LABEL_COLS if short (e.g. older label format without attrs)
                if len(vals) < LABEL_COLS:
                    vals = vals + [0.0] * (LABEL_COLS - len(vals))

                rows.append(vals[:LABEL_COLS])

        if rows:
            return np.array(rows, dtype=np.float32)
        return np.zeros((0, LABEL_COLS), dtype=np.float32)

    @staticmethod
    def collate_fn(
        batch: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Custom collate to handle variable-length label tensors.

        PyTorch's default collate requires same-length tensors in each batch.
        Images are stacked normally; labels are kept as a list.

        Returns:
            images: (B, 3, H, W) stacked image batch.
            labels: List[Tensor] of length B, each of shape (N_i, LABEL_COLS).
        """
        images, labels = zip(*batch)
        return torch.stack(images, dim=0), list(labels)
