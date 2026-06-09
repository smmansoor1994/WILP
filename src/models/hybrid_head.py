"""
src/models/hybrid_head.py
=========================
ARCHON — Improvement C: HybridMultiTaskHead (Severity + Quadrant)

Hierarchical Multi-Task Prediction Head for ARCHON (ARCHON).
Outputs 3-level disease severity (Healthy/Mild/Severe) per attribute and
a quadrant auxiliary classifier used for FDI assignment consistency.

Architecture diagram reference (from the proposed hybrid design):
  Unified Multi-Task Prediction Head
       ├── Detection Head
       │     · Tooth Localization
       │     · Disease Bounding Boxes
       │     · Object Confidence
       │
       └── Hierarchical Classification Head
             · Quadrant Prediction       (4 classes)
             · FDI Tooth Numbering       (32 classes)
             · Pathology Classification  (4 binary attrs)
             · Severity Estimation       (mild/moderate/severe per disease)

Improvements over baseline:
  1. Severity estimation per disease attribute (Limitation 2: "Binary classification
     heads may miss subtle diseases such as early vs. deep caries").
     Instead of binary is_diseased, we predict 3-class severity:
       0 = healthy,  1 = mild,  2 = severe
     For caries this maps to: healthy / caries / deep caries (jointly).
     For impacted/lesion: healthy / mild / severe.

  2. Hierarchical confidence: quadrant prediction guides FDI assignment,
     not just the linear sum assignment cost matrix alone.

  3. Context-aware attribute head: uses cross-attention-fused features
     (not just raw FPN features) for the disease classification branches.
"""

import math
import logging
from typing import List, Optional, Dict

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ─── Severity Head ────────────────────────────────────────────────────────────

class SeverityHead(nn.Module):
    """3-class severity prediction for a single disease attribute.

    Outputs:
      0 → healthy
      1 → mild / early stage
      2 → severe / advanced

    Architecture: same depthwise-separable pattern as AttributeHead but
    with a 3-way softmax output instead of binary sigmoid.

    Args:
        in_channels: FPN feature channels for this scale.
        num_severity_levels: Number of severity classes (default 3).
    """

    def __init__(self, in_channels: int, num_severity_levels: int = 3):
        super().__init__()
        hidden = max(in_channels // 4, 16)
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
        self.out = nn.Conv2d(hidden, num_severity_levels, 1, bias=True)
        self.num_severity_levels = num_severity_levels

        # Initialize: mild bias slightly higher than severe (most diseases start mild)
        nn.init.zeros_(self.out.bias)
        with torch.no_grad():
            self.out.bias[0] = math.log(80)   # healthy is most common
            self.out.bias[1] = math.log(15)   # mild
            self.out.bias[2] = math.log(5)    # severe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            logits: (B, num_severity_levels, H, W)
        """
        x = self.conv1(x)
        x = self.conv2(x)
        return self.out(x)


# ─── Multi-Scale Severity Head ────────────────────────────────────────────────

class MultiScaleSeverityHead(nn.Module):
    """Multi-scale severity heads for all disease attributes.

    Each of the 4 disease attributes gets its own SeverityHead per FPN scale.
    At inference, logits are sampled at each detected tooth's center and
    averaged across scales before argmax.

    Args:
        in_channels_list:   Channel count per FPN scale (e.g. [320, 640, 640]).
        num_attrs:          Number of disease attributes (default 4).
        num_severity_levels: Severity classes per attribute (default 3).
    """

    ATTR_NAMES = ["is_impacted", "has_caries", "has_deepcaries", "has_lesion"]

    def __init__(
        self,
        in_channels_list: List[int],
        num_attrs: int = 4,
        num_severity_levels: int = 3,
    ):
        super().__init__()
        self.num_attrs = num_attrs
        self.num_severity_levels = num_severity_levels

        # heads[attr_idx][scale_idx] → SeverityHead
        self.heads = nn.ModuleList([
            nn.ModuleList([
                SeverityHead(ch, num_severity_levels)
                for ch in in_channels_list
            ])
            for _ in range(num_attrs)
        ])

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            features: [(B, C_i, H_i, W_i) for each scale i]

        Returns:
            List of severity logit tensors per scale.
            Each shape: (B, num_attrs * num_severity_levels, H_i, W_i)
        """
        scale_outputs = []
        for scale_idx, feat in enumerate(features):
            per_attr_logits = [
                self.heads[attr_idx][scale_idx](feat)   # (B, 3, H, W)
                for attr_idx in range(self.num_attrs)
            ]
            # Stack along channel: (B, num_attrs*3, H, W)
            scale_out = torch.cat(per_attr_logits, dim=1)
            scale_outputs.append(scale_out)
        return scale_outputs


# ─── Quadrant-Aware FDI Classifier ───────────────────────────────────────────

class QuadrantAwareFDIHead(nn.Module):
    """Two-stage hierarchical FDI classification head.

    Stage 1 — Quadrant prediction: 4-class CE loss (predicts which quadrant).
    Stage 2 — FDI prediction: 32-class CE loss.

    The quadrant prediction is used during post-processing to reinforce the
    linear sum assignment: if the model is confident a tooth is in Q2, the
    cost matrix is penalized for assigning it to any Q1/Q3/Q4 FDI slot.

    Args:
        in_channels_list: FPN channel sizes.
    """

    def __init__(self, in_channels_list: List[int]):
        super().__init__()
        # Lightweight quadrant head (4 classes)
        self.quadrant_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, max(ch // 8, 32), 3, padding=1, bias=False),
                nn.BatchNorm2d(max(ch // 8, 32)),
                nn.SiLU(inplace=True),
                nn.Conv2d(max(ch // 8, 32), 4, 1, bias=True),
            )
            for ch in in_channels_list
        ])
        # Note: the 32-class FDI classification is handled by the YOLOv8
        # detection head natively; this quadrant head only adds auxiliary loss.

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            features: FPN feature maps.

        Returns:
            List of quadrant logits per scale, each (B, 4, H, W).
        """
        return [head(feat) for head, feat in zip(self.quadrant_heads, features)]


# ─── Hybrid Multi-Task Head ───────────────────────────────────────────────────

class HybridMultiTaskHead(nn.Module):
    """Combined prediction head integrating all improvements.

    Manages:
      - MultiScaleSeverityHead (replaces + extends binary AttributeHead)
      - QuadrantAwareFDIHead (auxiliary quadrant classification)

    This head operates on cross-attention-fused FPN features for improved
    context-awareness.

    Args:
        in_channels_list:    FPN channel counts [P3, P4, P5].
        num_attrs:           Number of disease attributes (4).
        num_severity_levels: Severity classes per attribute (3).
    """

    def __init__(
        self,
        in_channels_list: List[int],
        num_attrs: int = 4,
        num_severity_levels: int = 3,
    ):
        super().__init__()
        self.severity_head = MultiScaleSeverityHead(
            in_channels_list=in_channels_list,
            num_attrs=num_attrs,
            num_severity_levels=num_severity_levels,
        )
        self.quadrant_head = QuadrantAwareFDIHead(in_channels_list=in_channels_list)

    def forward(
        self, features: List[torch.Tensor]
    ) -> Dict[str, List[torch.Tensor]]:
        """
        Args:
            features: [P3_fused, P4_fused, P5_fused] — cross-attention fused.

        Returns:
            dict with keys:
              'severity':  list of (B, num_attrs*3, H_i, W_i) per scale
              'quadrant':  list of (B, 4, H_i, W_i) per scale
        """
        return {
            "severity": self.severity_head(features),
            "quadrant": self.quadrant_head(features),
        }


# ─── Severity Loss ────────────────────────────────────────────────────────────

class SeverityLoss(nn.Module):
    """Cross-entropy loss for severity-level disease classification.

    Maps the binary attribute labels to 3-class severity labels using
    available label information:
      - For caries/deepcaries: if has_caries=1 AND has_deepcaries=1 → severity=2
                               if has_caries=1 AND has_deepcaries=0 → severity=1
      - For impacted/lesion: binary → severity 0 or 2 (no mild label in DENTEX)

    Only fires on data_type=2 samples (same hierarchical mask as attribute loss).

    Args:
        num_attrs:         Number of disease attributes.
        loss_weight:       Total loss weight (default 4.0 — half of attribute weight
                           since severity carries more gradient signal per class).
        label_smoothing:   Smooth targets to prevent overconfidence.
    """

    def __init__(
        self,
        num_attrs: int = 4,
        num_severity_levels: int = 3,
        loss_weight: float = 4.0,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.num_attrs = num_attrs
        self.num_severity_levels = num_severity_levels
        self.loss_weight = loss_weight
        self.ce = nn.CrossEntropyLoss(
            reduction="none",
            label_smoothing=label_smoothing,
        )

    @staticmethod
    def binary_to_severity(attrs: torch.Tensor) -> torch.Tensor:
        """Convert 4-column binary attrs to 4-column severity indices.

        Columns:  [is_impacted, has_caries, has_deepcaries, has_lesion]
        Severity: 0=healthy, 1=mild, 2=severe

        Mapping per attribute:
          is_impacted:   0→0, 1→2         (impaction is always a major finding)
          has_caries:    0→0, 1→1          (mild caries)
          has_deepcaries:0→0, 1→2          (advanced caries → severe)
          has_lesion:    0→0, 1→2          (periapical lesion → severe)

        Combined caries logic: if tooth has both caries and deepcaries,
        the caries severity is promoted to severe (= 2) as a joint indicator.

        Args:
            attrs: (N, 4) tensor of binary labels {0,1}

        Returns:
            severity: (N, 4) tensor of severity indices {0, 1, 2}
        """
        sev = torch.zeros_like(attrs, dtype=torch.long)
        sev[:, 0] = (attrs[:, 0] > 0).long() * 2   # impacted → 0 or 2
        # Caries: mild if caries only, severe if deepcaries present
        sev[:, 1] = ((attrs[:, 1] > 0) & (attrs[:, 2] == 0)).long() * 1 + \
                    ((attrs[:, 1] > 0) & (attrs[:, 2] > 0)).long() * 2
        sev[:, 2] = (attrs[:, 2] > 0).long() * 2   # deepcaries → 0 or 2
        sev[:, 3] = (attrs[:, 3] > 0).long() * 2   # lesion → 0 or 2
        return sev

    def forward(
        self,
        pred_severity: torch.Tensor,  # (N, num_attrs*3) logits
        target_attrs: torch.Tensor,   # (N, 4) binary GT attrs
        data_types: torch.Tensor,     # (N,) 0/1/2
    ) -> torch.Tensor:
        """
        Args:
            pred_severity: Severity logits per tooth.
            target_attrs:  Binary disease labels from DENTEX annotations.
            data_types:    Annotation type mask.

        Returns:
            Scalar loss.
        """
        mask = (data_types == 2)
        if mask.sum() == 0:
            return pred_severity.sum() * 0.0

        pred_m = pred_severity[mask]          # (M, num_attrs*3)
        tgt_m = target_attrs[mask]            # (M, 4)
        sev_labels = self.binary_to_severity(tgt_m)   # (M, 4)

        total_loss = torch.tensor(0.0, device=pred_severity.device, requires_grad=True)
        for attr_idx in range(self.num_attrs):
            start = attr_idx * self.num_severity_levels
            end = start + self.num_severity_levels
            logits_attr = pred_m[:, start:end]           # (M, 3)
            labels_attr = sev_labels[:, attr_idx]        # (M,)
            loss_attr = self.ce(logits_attr, labels_attr).mean()
            total_loss = total_loss + loss_attr

        return self.loss_weight * (total_loss / self.num_attrs)


class QuadrantAuxLoss(nn.Module):
    """Auxiliary cross-entropy loss for quadrant prediction.

    This provides an additional gradient signal specifically for quadrant
    classification, helping the model form spatially-aware features early.

    Args:
        loss_weight: Weight for the auxiliary quadrant loss (default 1.0).
    """

    def __init__(self, loss_weight: float = 1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(
        self,
        pred_quadrant: torch.Tensor,  # (N, 4) logits
        target_classes: torch.Tensor, # (N,) FDI class indices 0-31
    ) -> torch.Tensor:
        """
        Args:
            pred_quadrant:  Quadrant logits per anchor.
            target_classes: GT FDI class (0-31); quadrant = class // 8.

        Returns:
            Scalar loss.
        """
        quadrant_labels = target_classes // 8  # 0-31 → 0-3
        loss = self.ce(pred_quadrant, quadrant_labels).mean()
        return self.loss_weight * loss
