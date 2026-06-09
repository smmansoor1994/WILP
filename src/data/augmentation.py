"""
src/data/augmentation.py
========================
ARCHON / ARCHON custom augmentation pipeline.

Contains two augmentation types:
  • Flip Mapping (baseline)  — quadrant-aware horizontal flip remapping
  • CLAHE (ARCHON Improvement D) — local contrast enhancement for early
    caries and periapical lesion visibility on non-uniform X-ray exposures

Key contribution from Section 2.1 of the paper:
  "Flip Mapping" — when a panoramic image is horizontally flipped,
  the quadrant information of bounding boxes also changes:
    Q1 (upper-right) ↔ Q2 (upper-left)
    Q3 (lower-left)  ↔ Q4 (lower-right)

Standard augmentation libraries (torchvision, albumentations) do NOT handle
this remapping. This module wraps albumentations transforms with the correct
post-flip class remapping.

Usage:
    aug = ARCHONAugmentor(config)
    img_aug, labels_aug = aug(image, labels)

Label format (per row):
    class_id  cx  cy  w  h  is_impacted  has_caries  has_deepcaries  has_lesion  data_type
"""

import random
import logging
from typing import Tuple, Optional

import cv2
import numpy as np

from src.utils.fdi import FLIP_CLASS_TABLE

logger = logging.getLogger(__name__)


class ARCHONAugmentor:
    """Augmentation pipeline for ARCHON.

    Applies standard augmentations (brightness, blur, rotation, scale)
    plus flip-with-quadrant-remapping (the paper's key augmentation).

    Hybrid improvement — CLAHE (Contrast Limited Adaptive Histogram Equalization):
      Dental X-rays often have non-uniform exposure (brighter around jawbone,
      darker at edges). CLAHE performs local histogram equalization within small
      tiles (clipLimit controls max amplification), enhancing subtle structural
      details (early caries, thin root boundaries, early periapical changes)
      that global brightness jitter cannot reveal. This addresses "Pathology
      Sensitivity" limitation: the model now sees better-contrast training images.

    Args:
        hsv_v:         Brightness jitter magnitude (0-1).
        degrees:       Max rotation angle in degrees.
        translate:     Max translation fraction.
        scale:         Max scale jitter factor (1 ± scale).
        blur_prob:     Probability of applying Gaussian blur.
        fliplr:        Probability of horizontal flip.
        clahe_prob:    Probability of applying CLAHE preprocessing (default 0.5).
        clahe_clip:    CLAHE clip limit (default 2.0 — conservative for X-rays).
        clahe_tile:    CLAHE tile grid size (default 8×8).
    """

    def __init__(
        self,
        hsv_v: float = 0.4,
        degrees: float = 5.0,
        translate: float = 0.1,
        scale: float = 0.5,
        blur_prob: float = 0.1,
        fliplr: float = 0.5,
        clahe_prob: float = 0.5,
        clahe_clip: float = 2.0,
        clahe_tile: int = 8,
    ):
        self.hsv_v = hsv_v
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.blur_prob = blur_prob
        self.fliplr = fliplr
        self.clahe_prob = clahe_prob
        # Build CLAHE object once (thread-safe to reuse)
        self._clahe = cv2.createCLAHE(
            clipLimit=clahe_clip,
            tileGridSize=(clahe_tile, clahe_tile),
        )

    def __call__(
        self,
        image: np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply augmentations to an image and its YOLO-format labels.

        Args:
            image:  HxWxC uint8 image (BGR or grayscale-as-BGR).
            labels: Nx10 float array with columns:
                    [class_id, cx, cy, w, h, attr0, attr1, attr2, attr3, data_type]

        Returns:
            Augmented (image, labels) pair.
        """
        # 1. CLAHE preprocessing — enhance local contrast (hybrid improvement)
        #    Increases visibility of subtle structures (early caries, thin root lines)
        #    before any destructive augmentation (blur, brightness jitter).
        image = self._augment_clahe(image)

        # 2. Brightness / exposure augmentation (X-ray-safe: no color change)
        image = self._augment_brightness(image)

        # 3. Gaussian blur (simulate noise / motion blur)
        image = self._augment_blur(image)

        # 4. Geometric: rotation + translation + scale
        image, labels = self._augment_affine(image, labels)

        # 5. Horizontal flip WITH quadrant remapping (key contribution)
        image, labels = self._augment_fliplr(image, labels)

        return image, labels

    # ─── Individual augmentation methods ──────────────────────────────────────

    def _augment_clahe(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

        Dental X-rays have non-uniform exposure — bone is bright, soft tissue
        is dark. CLAHE improves local contrast, making early caries (demineralisation
        appears as subtle brightness change) and periapical lesions (dark halos at
        root tips) more visible during training.

        Applied stochastically (prob=clahe_prob) to prevent over-sharpening.
        Always applied to grayscale channel(s); safely skips colour images.
        """
        if random.random() > self.clahe_prob:
            return image

        # X-rays stored as BGR (3-channel) but are effectively grayscale.
        # Apply CLAHE on the luminance channel only to avoid color artifacts.
        if image.ndim == 2:
            return self._clahe.apply(image)

        # Check if image is actually grayscale stored as BGR (all channels equal)
        if image.shape[2] == 3:
            # Convert to LAB, apply CLAHE on L channel, convert back
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_ch, a_ch, b_ch = cv2.split(lab)
            l_eq = self._clahe.apply(l_ch)
            lab_eq = cv2.merge([l_eq, a_ch, b_ch])
            return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

        return image

    def _augment_brightness(self, image: np.ndarray) -> np.ndarray:
        """Random brightness/contrast adjustment for X-ray images.

        X-rays are grayscale — we skip hue/saturation and only vary value (V).
        """
        if self.hsv_v <= 0:
            return image
        # Convert to float, apply random gain
        gain = 1.0 + random.uniform(-self.hsv_v, self.hsv_v)
        image = np.clip(image.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        return image

    def _augment_blur(self, image: np.ndarray) -> np.ndarray:
        """Random Gaussian blur."""
        if random.random() > self.blur_prob:
            return image
        # Kernel size must be odd
        k = random.choice([3, 5, 7])
        return cv2.GaussianBlur(image, (k, k), 0)

    def _augment_affine(
        self, image: np.ndarray, labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Random rotation + translation + scale affine transform."""
        h, w = image.shape[:2]

        # Build transformation matrix
        angle = random.uniform(-self.degrees, self.degrees)
        tx = random.uniform(-self.translate, self.translate) * w
        ty = random.uniform(-self.translate, self.translate) * h
        scale = random.uniform(1 - self.scale, 1 + self.scale)

        # Rotation + scale about image center
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        # Add translation
        M[0, 2] += tx
        M[1, 2] += ty

        # Warp image
        image = cv2.warpAffine(image, M, (w, h), borderValue=(114, 114, 114))

        if labels is None or len(labels) == 0:
            return image, labels

        # Transform bounding box corners
        new_labels = []
        for row in labels:
            cls, cx, cy, bw, bh = row[0], row[1], row[2], row[3], row[4]
            rest = row[5:]  # attributes + data_type

            # Convert to absolute pixel corners (4 corners of the box)
            x1 = (cx - bw / 2) * w
            y1 = (cy - bh / 2) * h
            x2 = (cx + bw / 2) * w
            y2 = (cy + bh / 2) * h

            corners = np.array([[x1, y1, 1], [x2, y1, 1],
                                 [x1, y2, 1], [x2, y2, 1]], dtype=np.float32)
            transformed = (M @ corners.T).T  # shape (4, 2)

            # Get new axis-aligned bounding box
            nx1 = np.clip(transformed[:, 0].min(), 0, w)
            ny1 = np.clip(transformed[:, 1].min(), 0, h)
            nx2 = np.clip(transformed[:, 0].max(), 0, w)
            ny2 = np.clip(transformed[:, 1].max(), 0, h)

            # Convert back to normalized cx,cy,bw,bh
            ncx = ((nx1 + nx2) / 2) / w
            ncy = ((ny1 + ny2) / 2) / h
            nbw = (nx2 - nx1) / w
            nbh = (ny2 - ny1) / h

            # Skip degenerate boxes
            if nbw < 1e-3 or nbh < 1e-3:
                continue

            new_labels.append(np.concatenate([[cls, ncx, ncy, nbw, nbh], rest]))

        if new_labels:
            return image, np.array(new_labels, dtype=labels.dtype)
        return image, np.zeros((0, labels.shape[1]), dtype=labels.dtype)

    def _augment_fliplr(
        self, image: np.ndarray, labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Horizontal flip WITH quadrant-aware class remapping.

        This is the key augmentation from the paper (Section 2.1).
        Simply flipping the image and mirroring cx is not enough:
        the tooth class (FDI) must also change to reflect the new
        left-right orientation.

        Example: Tooth '16' (Q1 first molar) after horizontal flip
                 becomes '26' (Q2 first molar) because Q1↔Q2.
        """
        if random.random() > self.fliplr:
            return image, labels

        # Flip image horizontally
        image = cv2.flip(image, 1)

        if labels is None or len(labels) == 0:
            return image, labels

        new_labels = labels.copy()

        for i, row in enumerate(new_labels):
            cls_id = int(row[0])

            # Mirror center x: new_cx = 1.0 - old_cx
            new_labels[i, 1] = 1.0 - row[1]

            # Remap class (tooth FDI) to its mirror quadrant
            # Only valid for enumeration labels (data_type >= 1)
            data_type = int(row[9]) if row.shape[0] > 9 else 1
            if data_type >= 1 and 0 <= cls_id < 32:
                new_labels[i, 0] = FLIP_CLASS_TABLE[cls_id]
            # For quadrant-only data (data_type = 0, class 0-3):
            # Q0↔Q1 (class 0↔1) and Q2↔Q3 (class 2↔3)
            elif data_type == 0 and cls_id in (0, 1, 2, 3):
                flip_quadrant_class = {0: 1, 1: 0, 2: 3, 3: 2}
                new_labels[i, 0] = flip_quadrant_class.get(cls_id, cls_id)

        return image, new_labels
