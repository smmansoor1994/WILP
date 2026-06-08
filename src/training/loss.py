"""
src/training/loss.py
====================
Hierarchical Loss for YOLOrtho.

Paper Section 2.2 — Loss:
  Total loss = wb * Loss_bbox + wc * Loss_class + wd * Loss_DFL
             + w1 * Loss_attr1 + ... + wn * Loss_attrn

  Weights from paper:
    bbox loss:      7.5
    class loss:     0.5
    DFL loss:       1.5
    disease attrs:  8.0 (each)

  Hierarchical computation:
    data_type = 0 (quadrant only) → compute bbox + quadrant-level class loss
    data_type = 1 (enumeration)   → compute bbox + full FDI class loss
    data_type = 2 (disease)       → compute bbox + class + attribute loss

This file provides:
  - YOLOrthoLoss: wraps the ultralytics v8 detection loss + attribute BCE
  - HierarchicalClassLoss: computes class loss with quadrant grouping for type-0 data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


# ─── Disease Attribute Indices (column positions in label after bbox) ─────────
# Label format: [class_id, cx, cy, w, h, is_impacted, has_caries, has_deepcaries, has_lesion, data_type]
ATTR_COLS = [5, 6, 7, 8]    # attribute columns in label tensor
DATA_TYPE_COL = 9            # data_type column in label tensor


class AttributeBCELoss(nn.Module):
    """Binary Cross Entropy loss for disease attribute heads.

    Computed ONLY on samples where data_type == 2 (disease annotations available).
    This is the key hierarchical training mechanism from the paper.

    Args:
        num_attrs:     Number of attributes (4).
        loss_weight:   Weight applied to the total attribute loss (default 8.0).
        pos_weight:    Per-attribute positive-class weight for BCEWithLogitsLoss.
                       Counteracts class imbalance (most teeth are healthy → label=0).
                       A value of W means a false negative is penalised W× more than
                       a false positive.  Rule of thumb: W ≈ (#negatives / #positives).
                       If None, defaults to [5, 3, 8, 5] for
                       [impacted, caries, deepcaries, lesion] based on DENTEX prevalence.
    """

    def __init__(
        self,
        num_attrs: int = 4,
        loss_weight: float = 8.0,
        pos_weight: Optional[List[float]] = None,
    ):
        super().__init__()
        self.num_attrs = num_attrs
        self.loss_weight = loss_weight
        # Default pos_weight estimated from DENTEX disease set prevalence:
        #   impacted  ~15%  → ratio ≈ 5.7 → use 5
        #   caries    ~25%  → ratio ≈ 3.0 → use 3
        #   deepcaries ~8%  → ratio ≈ 11.5 → use 8
        #   lesion    ~15%  → ratio ≈ 5.7 → use 5
        # These prevent the trivial "always predict healthy" collapse.
        _pw = pos_weight if pos_weight is not None else [5.0, 3.0, 8.0, 5.0]
        pw_tensor = torch.tensor(_pw, dtype=torch.float32)
        # Register as buffer so it moves to the correct device automatically
        # when loss_fn.to(device) is called.
        self.register_buffer("_pos_weight", pw_tensor)
        # Reduction='none' so we can apply masking manually;
        # pos_weight is passed at forward() time after device placement.
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(
        self,
        pred_attrs: torch.Tensor,      # (N, num_attrs)  predicted logits
        target_attrs: torch.Tensor,    # (N, num_attrs)  ground truth {0, 1}
        data_types: torch.Tensor,      # (N,)            0/1/2
    ) -> torch.Tensor:
        """
        Args:
            pred_attrs:   Flattened attribute predictions.
            target_attrs: Ground truth attribute labels.
            data_types:   Data type per sample.

        Returns:
            Scalar loss. Zero if no disease samples in batch.
        """
        # Mask: only compute loss on disease-annotated samples
        mask = (data_types == 2).float()  # (N,)

        if mask.sum() == 0:
            # Return a differentiable zero so that loss.backward() does not crash
            # when a batch contains no disease-annotated (data_type=2) samples.
            return pred_attrs.sum() * 0.0

        # BCE elementwise with pos_weight applied manually so the weight tensor
        # stays on the correct device (registered as buffer via _pos_weight).
        # pos_weight: (num_attrs,) → broadcast to (N, num_attrs)
        # Effective loss per element:
        #   label=1: pos_weight * -log(sigmoid(logit))
        #   label=0:             -log(1 - sigmoid(logit))
        # This upweights false negatives (missed diseases), preventing the
        # trivial "always predict healthy" collapse caused by class imbalance.
        target_f = target_attrs.float()
        pw = self._pos_weight.to(pred_attrs.device)          # (num_attrs,)
        bce_loss = (
            (1 - target_f) * pred_attrs
            - (1 + (pw - 1) * target_f) * F.logsigmoid(pred_attrs)
        )   # numerically equivalent to BCEWithLogitsLoss(pos_weight=pw), shape (N, num_attrs)

        # Apply per-sample mask (broadcast over attribute dim)
        masked = bce_loss * mask.unsqueeze(1)  # (N, num_attrs)

        # Mean over all valid (masked) elements
        n_valid = mask.sum() * self.num_attrs
        loss = masked.sum() / (n_valid + 1e-6)

        return self.loss_weight * loss


class HierarchicalClassLoss(nn.Module):
    """Class loss that respects the annotation hierarchy.

    data_type = 0: Only quadrant-level supervision.
                   Groups the 32 FDI classes into 4 quadrants for CE loss.
    data_type ≥ 1: Full FDI class (0-31) CE loss.

    Args:
        num_classes: Total number of tooth classes (32 for FDI).
        loss_weight: Classification loss weight (0.5 from paper).
    """

    def __init__(self, num_classes: int = 32, loss_weight: float = 0.5):
        super().__init__()
        self.num_classes = num_classes
        self.loss_weight = loss_weight

        # Map each FDI class (0-31) to quadrant index (0-3)
        # FDI 11-18 → Q1 (index 0), FDI 21-28 → Q2 (index 1), etc.
        quadrant_map = [i // 8 for i in range(32)]  # [0]*8 + [1]*8 + [2]*8 + [3]*8
        self.register_buffer(
            "quadrant_map", torch.tensor(quadrant_map, dtype=torch.long)
        )

    def forward(
        self,
        pred_cls: torch.Tensor,    # (N, num_classes)  class logits
        target_cls: torch.Tensor,  # (N,)              class indices (0-31)
        data_types: torch.Tensor,  # (N,)              0/1/2
    ) -> torch.Tensor:
        """
        Returns:
            Scalar class CE loss.
        """
        if len(pred_cls) == 0:
            return torch.tensor(0.0, device=pred_cls.device)

        total_loss = torch.tensor(0.0, device=pred_cls.device)
        n = 0

        # ── Full FDI loss (data_type >= 1) ────────────────────────────────────
        fdi_mask = (data_types >= 1)
        if fdi_mask.sum() > 0:
            fdi_loss = F.cross_entropy(
                pred_cls[fdi_mask], target_cls[fdi_mask], reduction="mean"
            )
            total_loss = total_loss + fdi_loss
            n += 1

        # ── Quadrant-level loss (data_type == 0) ──────────────────────────────
        q_mask = (data_types == 0)
        if q_mask.sum() > 0:
            # Map FDI targets → quadrant targets (0-3)
            q_targets = self.quadrant_map[target_cls[q_mask]]
            # Sum logits within each quadrant for 4-class CE
            q_pred = pred_cls[q_mask].view(-1, 4, 8).sum(dim=2)  # (M, 4)
            q_loss = F.cross_entropy(q_pred, q_targets, reduction="mean")
            total_loss = total_loss + q_loss
            n += 1

        if n > 0:
            total_loss = total_loss / n

        return self.loss_weight * total_loss


class YOLOrthoLoss(nn.Module):
    """Combined detection + attribute loss for YOLOrtho.

    Combines:
    - Standard YOLOv8 detection loss (bbox + class + DFL)
      via the ultralytics loss machinery (we call it from the trainer)
    - Disease attribute BCE loss (this module)
    - Hierarchical class loss (this module)

    In practice, during training:
    1. The ultralytics Trainer computes bbox+class+DFL loss.
    2. We add attribute loss on top after extracting attribute predictions.

    Args:
        num_classes:     Number of tooth classes.
        num_attrs:       Number of disease attributes.
        weight_bbox:     Bounding box loss weight.
        weight_cls:      Classification loss weight.
        weight_dfl:      Distribution Focal Loss weight.
        weight_attr:     Disease attribute loss weight.
    """

    def __init__(
        self,
        num_classes: int = 32,
        num_attrs: int = 4,
        weight_bbox: float = 7.5,
        weight_cls: float = 0.5,
        weight_dfl: float = 1.5,
        weight_attr: float = 8.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_attrs = num_attrs

        # Attribute BCE loss (handles data_type masking internally)
        self.attr_loss_fn = AttributeBCELoss(
            num_attrs=num_attrs,
            loss_weight=weight_attr,
        )

    def compute_attribute_loss(
        self,
        pred_attr_logits: torch.Tensor,  # (N, num_attrs)
        labels: torch.Tensor,            # (N, 10) — full label rows
    ) -> torch.Tensor:
        """Compute attribute BCE loss from flattened predictions and labels.

        Args:
            pred_attr_logits: Attribute logits per matched anchor.
            labels:           Full label rows containing attr cols and data_type.

        Returns:
            Scalar attribute loss.
        """
        if labels.shape[0] == 0 or pred_attr_logits.shape[0] == 0:
            return torch.tensor(0.0, device=pred_attr_logits.device)

        # Extract attribute targets and data types
        # Labels: [class_id, cx, cy, w, h, attr0, attr1, attr2, attr3, data_type]
        target_attrs = labels[:, ATTR_COLS].float()   # (N, 4)
        data_types = labels[:, DATA_TYPE_COL].long()  # (N,)

        return self.attr_loss_fn(pred_attr_logits, target_attrs, data_types)

    def forward(
        self,
        det_loss: torch.Tensor,              # bbox+cls+dfl loss from ultralytics
        pred_attr_logits: torch.Tensor,      # (N, num_attrs) attribute logits
        labels: torch.Tensor,                # (N, 10) label rows
    ) -> Dict[str, torch.Tensor]:
        """Combine detection and attribute losses.

        Returns:
            dict with keys: 'det', 'attr', 'total'
        """
        attr_loss = self.compute_attribute_loss(pred_attr_logits, labels)
        total = det_loss + attr_loss

        return {
            "det": det_loss,
            "attr": attr_loss,
            "total": total,
        }
