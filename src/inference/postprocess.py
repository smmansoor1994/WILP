"""
src/inference/postprocess.py
=============================
Post-processing for ARCHON / ARCHON: Linear Sum Assignment for teeth enumeration.

ARCHON extension: optional quadrant-consistency penalty (Improvement E) can be
passed as `quadrant_probs` to bias FDI assignment toward quadrant-consistent slots.

Paper Section 2.3 — Post-Process Strategy:
  "Each FDI is associated with one tooth only. We notice that deep learning
  models sometimes produce same teeth enumeration for two teeth close to
  each other. To solve this problem, we formulate the enumeration post-process
  as a linear-sum-assignment problem: each FDI is matched with one prediction
  only, and the cost of objects is constructed by their probability of each class."

Algorithm:
  1. Run ARCHON → get N detections with class probabilities (32 FDI classes)
  2. Build a cost matrix: rows = FDI positions (0-31), cols = detections
     cost[i, j] = 1 - prob[j, i]  (lower cost = higher probability)
  3. Solve the assignment problem using scipy.optimize.linear_sum_assignment
  4. Each FDI position is assigned at most one detection

This ensures:
  - No two detections are assigned the same tooth number
  - Each tooth number appears at most once in the final output

Additionally:
  - After matching, decode disease attributes from the attribute head outputs
  - Combine: FDI number + disease flags → structured per-tooth result
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.utils.fdi import class_to_fdi, fdi_to_name

logger = logging.getLogger(__name__)

# Disease attribute names in order
ATTR_NAMES = ["is_impacted", "has_caries", "has_deepcaries", "has_lesion"]

# Number of tooth classes
NUM_CLASSES = 32


@dataclass
class ToothDetection:
    """A single tooth detection result after post-processing.

    Attributes:
        fdi:           FDI tooth number (e.g. 16 for upper-right first molar).
        fdi_name:      Human-readable tooth name (e.g. 'Upper Right First Molar').
        class_idx:     YOLO class index (0-31).
        conf:          Detection confidence score.
        bbox_xyxy:     Bounding box in [x1, y1, x2, y2] absolute pixel format.
        bbox_xywhn:    Normalized [cx, cy, w, h] bounding box.
        is_impacted:   Disease attribute: tooth impaction.
        has_caries:    Disease attribute: caries (cavity).
        has_deepcaries: Disease attribute: deep caries.
        has_lesion:    Disease attribute: periapical lesion.
        diseases:      List of active disease names (for display).
    """
    fdi: int
    fdi_name: str
    class_idx: int
    conf: float
    bbox_xyxy: List[float]
    bbox_xywhn: List[float]
    is_impacted: bool = False
    has_caries: bool = False
    has_deepcaries: bool = False
    has_lesion: bool = False
    diseases: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fdi": self.fdi,
            "fdi_name": self.fdi_name,
            "conf": round(self.conf, 4),
            "bbox_xyxy": [round(v, 2) for v in self.bbox_xyxy],
            "is_impacted": self.is_impacted,
            "has_caries": self.has_caries,
            "has_deepcaries": self.has_deepcaries,
            "has_lesion": self.has_lesion,
            "diseases": self.diseases,
        }


def apply_linear_sum_assignment(
    class_probs: np.ndarray,    # (N, 32) — class probability for each detection
    boxes_xywhn: np.ndarray,    # (N, 4)  — normalized bounding boxes
    boxes_xyxy: np.ndarray,     # (N, 4)  — absolute bounding boxes
    confidences: np.ndarray,    # (N,)    — detection confidences
    attr_probs: Optional[np.ndarray] = None,    # (N, 4) — attribute probs [0,1]
    conf_threshold: float = 0.25,
    attr_threshold: float = 0.3,
    quadrant_probs: Optional[np.ndarray] = None,  # (N, 4) — quadrant softmax probs
) -> List[ToothDetection]:
    """Apply the linear sum assignment to enforce unique FDI assignment.

    When quadrant_probs are provided (from the hybrid quadrant auxiliary head),
    the cost matrix is augmented with a quadrant-consistency penalty: if the
    model predicts a detection is in quadrant Q with high confidence, the cost
    of assigning it to an FDI slot from a DIFFERENT quadrant is increased.
    This directly addresses FDI numbering conflicts across jaw sides.

    Args:
        class_probs:    Raw class probability scores for each detection.
        boxes_xywhn:    Normalized [cx,cy,w,h] bounding boxes.
        boxes_xyxy:     Absolute [x1,y1,x2,y2] bounding boxes.
        confidences:    Per-detection confidence scores.
        attr_probs:     Disease attribute probabilities [0-1] per detection.
        conf_threshold: Minimum confidence to include a detection.
        attr_threshold: Threshold for binary disease attribute decision.
        quadrant_probs: (N, 4) — softmax probabilities over quadrants 0-3,
                        from the hybrid quadrant head. When provided, adds a
                        quadrant-consistency penalty (alpha=0.5) to cost matrix.

    Returns:
        List of ToothDetection results (at most 32, one per FDI position).
    """
    N = len(class_probs)
    if N == 0:
        return []

    # ── Filter by confidence ──────────────────────────────────────────────────
    valid_mask = confidences >= conf_threshold
    valid_idx = np.where(valid_mask)[0]

    if len(valid_idx) == 0:
        return []

    # Work with valid detections only
    valid_probs = class_probs[valid_idx]       # (M, 32)
    valid_boxes_xywhn = boxes_xywhn[valid_idx] # (M, 4)
    valid_boxes_xyxy = boxes_xyxy[valid_idx]   # (M, 4)
    valid_confs = confidences[valid_idx]        # (M,)
    M = len(valid_idx)

    if attr_probs is not None:
        valid_attrs = attr_probs[valid_idx]    # (M, 4)
    else:
        valid_attrs = np.zeros((M, 4), dtype=np.float32)

    # ── Build cost matrix ─────────────────────────────────────────────────────
    # Cost matrix: (NUM_CLASSES, M) — rows = FDI slots (0-31), cols = detections
    # cost[i, j] = 1 - prob[j, i]: high probability → low cost → preferred assignment
    # We use the negative log-probability for numerical stability
    # cost[i, j] = -log(prob[j, i] + eps)
    eps = 1e-7
    cost_matrix = -np.log(valid_probs.T + eps)  # (32, M)

    # ── Quadrant-consistency penalty (hybrid improvement) ─────────────────────
    # When quadrant_probs are provided by the hybrid quadrant head:
    # For each FDI slot i (belongs to quadrant q_i = i // 8) and each detection j,
    # add a penalty proportional to the probability that detection j is NOT in q_i.
    #   penalty[i, j] = alpha * (1 - quadrant_probs[j, q_i])
    # This lowers the cost of same-quadrant assignments and raises cross-quadrant costs.
    if quadrant_probs is not None:
        valid_quad = quadrant_probs[valid_idx]  # (M, 4)  softmax over quadrants 0-3
        ALPHA = 0.5  # penalty weight — keeps FDI class probs dominant
        # slot_quadrants[i] = quadrant index for FDI slot i (0-3)
        slot_quadrants = np.arange(NUM_CLASSES) // 8  # (32,)
        # quad_match_prob[i, j] = valid_quad[j, slot_quadrants[i]]
        quad_match_prob = valid_quad[:, slot_quadrants].T  # (32, M)
        quad_penalty = ALPHA * (1.0 - quad_match_prob)     # (32, M)
        cost_matrix = cost_matrix + quad_penalty

    # ── Solve the assignment ──────────────────────────────────────────────────
    # linear_sum_assignment minimizes total cost
    # Returns: row_indices (FDI slots), col_indices (detection indices)
    if M < NUM_CLASSES:
        # More FDI slots than detections: some slots will be unassigned
        # We pad with dummy detections at very high cost
        padding = np.full((NUM_CLASSES, NUM_CLASSES - M), 1e6)
        padded_cost = np.concatenate([cost_matrix, padding], axis=1)
        row_ind, col_ind = linear_sum_assignment(padded_cost)
        # Only keep assignments to real detections (col < M)
        valid_assign = col_ind < M
        row_ind = row_ind[valid_assign]
        col_ind = col_ind[valid_assign]
    else:
        # More detections than FDI slots (can happen with false positives)
        # Transpose: minimize over 32 FDI slots assigned to M detections
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # ── Build output detections ───────────────────────────────────────────────
    results = []
    for fdi_slot, det_idx in zip(row_ind, col_ind):
        # Get FDI number and class index
        class_idx = int(fdi_slot)
        fdi_num = class_to_fdi(class_idx)

        # Use the DETECTION confidence (from NMS), not the reassigned class prob.
        # The class-prob check was a double-filter that silently dropped teeth
        # reassigned to a different FDI slot by the linear sum assignment.
        det_conf = float(valid_confs[det_idx])
        if det_conf < conf_threshold:
            continue

        # Bounding box
        bbox_xywhn = valid_boxes_xywhn[det_idx].tolist()
        bbox_xyxy = valid_boxes_xyxy[det_idx].tolist()

        # Disease attributes (threshold sigmoid outputs)
        attrs = valid_attrs[det_idx]
        is_impacted = bool(attrs[0] >= attr_threshold)
        has_caries = bool(attrs[1] >= attr_threshold)
        has_deepcaries = bool(attrs[2] >= attr_threshold)
        has_lesion = bool(attrs[3] >= attr_threshold)

        # Build disease name list
        diseases = []
        if is_impacted:
            diseases.append("Impacted")
        if has_caries:
            diseases.append("Caries")
        if has_deepcaries:
            diseases.append("Deep Caries")
        if has_lesion:
            diseases.append("Periapical Lesion")

        tooth = ToothDetection(
            fdi=fdi_num,
            fdi_name=fdi_to_name(fdi_num),
            class_idx=class_idx,
            conf=det_conf,
            bbox_xyxy=bbox_xyxy,
            bbox_xywhn=bbox_xywhn,
            is_impacted=is_impacted,
            has_caries=has_caries,
            has_deepcaries=has_deepcaries,
            has_lesion=has_lesion,
            diseases=diseases,
        )
        results.append(tooth)

    # Sort by FDI number for consistent output
    results.sort(key=lambda t: t.fdi)

    logger.debug(
        "Linear sum assignment: %d detections → %d unique FDI assignments",
        M,
        len(results),
    )

    return results


def postprocess_yolo_output(
    results,                         # ultralytics Results object
    attr_probs: Optional[np.ndarray] = None,   # (N, 4) attribute probabilities
    conf_threshold: float = 0.25,
    attr_threshold: float = 0.3,
    img_shape: Optional[tuple] = None,
    quadrant_probs: Optional[np.ndarray] = None,  # (N, 4) quadrant probs (hybrid)
) -> List[ToothDetection]:
    """Convert ultralytics YOLO output to ARCHON tooth detections.

    Args:
        results:        ultralytics Results object from model.predict().
        attr_probs:     Disease attribute probabilities from attribute heads.
        conf_threshold: Minimum detection confidence.
        attr_threshold: Threshold for binary attribute prediction.
        img_shape:      Original image (H, W) for absolute box conversion.
        quadrant_probs: (N, 4) quadrant probabilities from hybrid head; when
                        provided, adds quadrant-consistency penalty to linear
                        sum assignment cost matrix.

    Returns:
        List of ToothDetection results.
    """
    if results is None or results.boxes is None or len(results.boxes) == 0:
        return []

    boxes = results.boxes

    # Normalized xywh boxes
    boxes_xywhn = boxes.xywhn.cpu().numpy()   # (N, 4)

    # Absolute xyxy boxes
    boxes_xyxy = boxes.xyxy.cpu().numpy()     # (N, 4)

    # Detection confidences
    confs = boxes.conf.cpu().numpy()           # (N,)

    # Class probability matrix — from softmax of raw classification scores
    # ultralytics stores per-class probabilities in boxes.data[:, 5:-1] or similar
    if hasattr(boxes, "prob") and boxes.prob is not None:
        class_probs = boxes.prob.cpu().numpy()  # (N, num_classes)
    else:
        # Extract per-class probabilities from the raw detection tensor.
        # boxes.data shape: (N, 6+nc) where columns are [x1,y1,x2,y2,conf,cls,p0..pnc-1]
        # or in newer ultralytics: boxes.data[:, 6:] holds class scores.
        # Fall back to soft one-hot scaled by confidence when not available.
        n = len(confs)
        class_ids = boxes.cls.cpu().numpy().astype(int)
        try:
            # ultralytics >= 8.x stores raw cls logits in boxes.data cols 6:
            raw = boxes.data.cpu().numpy()        # (N, 6+nc) or (N, 5+nc)
            if raw.shape[1] >= 6 + NUM_CLASSES:
                class_probs = raw[:, 6 : 6 + NUM_CLASSES].astype(np.float32)
            elif raw.shape[1] >= 5 + NUM_CLASSES:
                class_probs = raw[:, 5 : 5 + NUM_CLASSES].astype(np.float32)
            else:
                raise ValueError("unexpected boxes.data width")
            # Normalize rows to sum to 1 (softmax was applied upstream)
            row_sums = class_probs.sum(axis=1, keepdims=True).clip(1e-7)
            class_probs = class_probs / row_sums
        except Exception:
            # Last-resort: soft one-hot — assign full confidence to predicted class
            class_probs = np.zeros((n, NUM_CLASSES), dtype=np.float32)
            for i, (c, conf) in enumerate(zip(class_ids, confs)):
                if 0 <= c < NUM_CLASSES:
                    class_probs[i, c] = conf

    return apply_linear_sum_assignment(
        class_probs=class_probs,
        boxes_xywhn=boxes_xywhn,
        boxes_xyxy=boxes_xyxy,
        confidences=confs,
        attr_probs=attr_probs,
        conf_threshold=conf_threshold,
        attr_threshold=attr_threshold,
        quadrant_probs=quadrant_probs,
    )


def summarize_detections(teeth: List[ToothDetection]) -> Dict[str, Any]:
    """Generate a summary dict from a list of tooth detections.

    Returns:
        dict with counts of teeth, diseases per quadrant, etc.
    """
    summary = {
        "total_teeth": len(teeth),
        "diseased_teeth": sum(1 for t in teeth if t.diseases),
        "healthy_teeth": sum(1 for t in teeth if not t.diseases),
        "by_disease": {
            "impacted": sum(1 for t in teeth if t.is_impacted),
            "caries": sum(1 for t in teeth if t.has_caries),
            "deep_caries": sum(1 for t in teeth if t.has_deepcaries),
            "lesion": sum(1 for t in teeth if t.has_lesion),
        },
        "by_quadrant": {q: [] for q in range(1, 5)},
        "detections": [t.to_dict() for t in teeth],
    }

    for tooth in teeth:
        q = tooth.fdi // 10
        if 1 <= q <= 4:
            summary["by_quadrant"][q].append(tooth.fdi)

    return summary
