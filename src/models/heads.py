"""
src/models/heads.py
===================
ARCHON / ARCHON — Disease Attribute Prediction Heads.

Paper Section 2.2:
  "We construct four binary classification heads:
   is_impacted, has_caries, has_deepcaries and has_lesion.
   These attributes prediction heads are independent of
   regular bounding box and classification heads."

Each attribute head:
  - Takes multi-scale feature maps from the FPN neck
  - For each anchor/detection, predicts a binary probability (sigmoid output)
  - Loss: Binary Cross Entropy (BCE)
  - Loss weight: 8.0 (from the paper)

The attribute predictions are APPENDED to the detection head output tensor.
Final output per anchor: [cx, cy, w, h, cls_0...cls_31, attr_0...attr_3]
                          4    +   32            +    4  = 40 values
"""

import math
import torch
import torch.nn as nn
from typing import List, Tuple


class AttributeHead(nn.Module):
    """Single binary attribute prediction head (e.g. 'has_caries').

    Architecture (same as YOLOv8 decoupled classification branch):
      - Two 3×3 depthwise-separable convolutions
      - 1×1 output convolution → 1 logit per anchor

    Args:
        in_channels: Number of input feature channels.
        num_anchors: Number of anchors (for output reshaping — use 1 for anchor-free).
    """

    def __init__(self, in_channels: int, num_anchors: int = 1):
        super().__init__()
        hidden = max(in_channels // 4, 16)

        # Two 3×3 convs with BN + SiLU (same pattern as YOLOv8 class head)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        )
        # Output: 1 logit per spatial position (anchor-free, like YOLOv8)
        self.out = nn.Conv2d(hidden, 1, 1, bias=True)

        # Initialize output bias to -log((1-p)/p) where p = 0.01
        # This ensures sigmoid(bias) ≈ 0.01 at start (disease is rare)
        nn.init.constant_(self.out.bias, -math.log(99))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature map of shape (B, C, H, W)
        Returns:
            Attribute logits of shape (B, 1, H, W)
        """
        x = self.conv1(x)
        x = self.conv2(x)
        return self.out(x)


class MultiAttributeHead(nn.Module):
    """Multi-scale attribute prediction for all 4 disease attributes.

    Operates on the same 3 scale feature maps as the detection head.
    Produces attribute logits for each scale, which are then associated
    with the corresponding bounding box predictions.

    Args:
        in_channels_list: List of channel counts per FPN scale.
                          e.g. [256, 512, 1024] for YOLOv8x P3, P4, P5
        num_attrs:        Number of disease attributes (default 4).
    """

    def __init__(
        self,
        in_channels_list: List[int],
        num_attrs: int = 4,
    ):
        super().__init__()
        self.num_attrs = num_attrs

        # One AttributeHead per attribute per scale
        # self.heads[attr_idx][scale_idx]
        self.heads = nn.ModuleList([
            nn.ModuleList([
                AttributeHead(ch) for ch in in_channels_list
            ])
            for _ in range(num_attrs)
        ])

    def forward(
        self, features: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Predict all attributes at all scales.

        Args:
            features: List of feature maps from FPN, one per scale.
                      Shape: [(B, C_i, H_i, W_i) for each scale i]

        Returns:
            List of attribute tensors, one per scale.
            Each tensor shape: (B, num_attrs, H_i, W_i)
        """
        scale_outputs = []
        for scale_idx, feat in enumerate(features):
            # Stack all attribute predictions for this scale
            attr_preds = torch.cat(
                [self.heads[attr_idx][scale_idx](feat)
                 for attr_idx in range(self.num_attrs)],
                dim=1,  # (B, num_attrs, H, W)
            )
            scale_outputs.append(attr_preds)
        return scale_outputs


# ─── Attribute Loss ───────────────────────────────────────────────────────────

class AttributeLoss(nn.Module):
    """BCE loss for disease attribute predictions.

    Only computes loss on samples where data_type == 2 (disease annotations
    available). This implements the hierarchical training from Section 2 of the paper.

    Args:
        weight: Loss weight for attribute predictions (default 8.0 from paper).
    """

    def __init__(self, weight: float = 8.0):
        super().__init__()
        self.weight = weight
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(
        self,
        pred_logits: torch.Tensor,   # (N, num_attrs)  — predicted logits
        targets: torch.Tensor,        # (N, num_attrs)  — GT attribute labels {0, 1}
        data_types: torch.Tensor,     # (N,)            — 0, 1, or 2
    ) -> torch.Tensor:
        """Compute hierarchical attribute BCE loss.

        Args:
            pred_logits: Predicted attribute logits per anchor.
            targets:     Ground truth attribute labels (0 or 1).
            data_types:  Data type per anchor (0=quadrant, 1=enum, 2=disease).

        Returns:
            Scalar loss value.
        """
        # Only compute attribute loss for disease-annotated samples (data_type == 2)
        mask = data_types == 2   # bool tensor of shape (N,)

        if mask.sum() == 0:
            # No disease-annotated samples in this batch
            return torch.tensor(0.0, device=pred_logits.device, requires_grad=True)

        # Apply mask
        pred_masked = pred_logits[mask]   # (M, num_attrs)
        tgt_masked = targets[mask].float()  # (M, num_attrs)

        # BCE per element, then mean
        loss = self.bce(pred_masked, tgt_masked)   # (M, num_attrs)
        return self.weight * loss.mean()
