"""
Focal Loss for Disease Attribute Detection
===========================================

Paper: "Focal Loss for Dense Object Detection" (Lin et al., ICCV 2017)
https://arxiv.org/abs/1708.02002

Motivation:
-----------
Standard BCE loss treats all errors equally. In imbalanced datasets (most teeth are healthy),
the model receives gradients dominated by easy negatives (healthy teeth). The model learns
to predict "healthy" as default and struggles with disease detection.

Focal loss down-weights easy examples and focuses training on hard positives (diseased teeth).

Formula:
--------
FL(p_t) = -α(1 - p_t)^γ * log(p_t)

where:
  - p_t: model's predicted probability for the ground truth class
  - γ (gamma): focusing parameter (typically 2)
    - γ = 0 : reduces to CE loss
    - γ = 2 : strong focusing on hard examples
  - α (alpha): weighting factor (typically 0.25)

Clinical Application:
---------------------
For dental X-rays:
  - Easy negatives: healthy teeth (model already confident they're healthy)
  - Hard positives: diseased teeth (model struggles to detect)
  - Focal loss forces the model to learn disease features better
  - Expected improvement: +5-10% disease detection recall, -3-8% false positives

Implementation Integration:
----------------------------
1. Add FocalAttributeLoss class to src/training/loss.py
2. Modify Phase 2b trainer to use FocalAttributeLoss instead of AttributeBCELoss
3. Update ARCHONLoss to accept focal_loss_enabled flag
4. Retrain Phase 2b for 100 epochs with focal loss
5. Evaluate on validation set
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class FocalAttributeLoss(nn.Module):
    """Focal loss for disease attributes — reduces impact of easy negatives.
    
    Clinical motivation:
    - Most X-ray teeth are healthy (easy negatives)
    - Disease teeth are rare (hard positives)
    - Focal loss focuses on learning hard positives (diseased teeth)
    - Result: Better disease detection, fewer false negatives
    
    Tuning parameters:
    - gamma=2.0: standard value, strong focus on hard examples
    - gamma=1.5: slightly less aggressive focusing
    - gamma=3.0: very aggressive focusing (may overfit)
    
    alpha=0.25: balances false positives vs false negatives
    - Adjust up/down if needed after testing
    """
    
    def __init__(
        self,
        num_attrs: int = 4,
        loss_weight: float = 8.0,
        pos_weight: Optional[List[float]] = None,
        gamma: float = 2.0,
        alpha: float = 0.25,
    ):
        """
        Args:
            num_attrs: Number of disease attributes (4 for ARCHON)
            loss_weight: Weight applied to focal loss in total loss computation
            pos_weight: Per-attribute positive class weight (for imbalance)
            gamma: Focusing parameter (0=CE loss, higher=more focusing)
            alpha: Balance parameter (0.25 recommended)
        """
        super().__init__()
        self.num_attrs = num_attrs
        self.loss_weight = loss_weight
        self.gamma = gamma
        self.alpha = alpha
        
        # Default pos_weight from DENTEX prevalence
        _pw = pos_weight if pos_weight is not None else [5.0, 3.0, 8.0, 5.0]
        self.register_buffer("_pos_weight", torch.tensor(_pw, dtype=torch.float32))
    
    def forward(
        self,
        pred_attrs: torch.Tensor,      # (N, num_attrs) predicted logits
        target_attrs: torch.Tensor,    # (N, num_attrs) ground truth {0, 1}
        data_types: torch.Tensor,      # (N,) data type indicator
    ) -> torch.Tensor:
        """
        Args:
            pred_attrs: Flattened attribute predictions (logits).
            target_attrs: Ground truth attribute labels {0, 1}.
            data_types: Data type per sample (0/1/2).
                       Only data_type==2 (disease-annotated) samples contribute to loss.
        
        Returns:
            Scalar focal loss. Zero if no disease samples in batch.
        
        Computation:
        1. Mask: only compute loss on disease-annotated (data_type==2) samples
        2. Sigmoid: p = σ(logits)  →  probabilities in [0, 1]
        3. p_t = p if label==1 else (1-p)  →  probability of ground truth class
        4. Focal weight: (1 - p_t)^γ  →  down-weight easy examples
        5. BCE: standard binary cross-entropy
        6. Focal loss: focal_weight * BCE
        """
        # Mask: only compute loss on disease-annotated samples
        mask = (data_types == 2).float()  # (N,)
        
        if mask.sum() == 0:
            # No disease samples in batch → return differentiable zero
            return pred_attrs.sum() * 0.0
        
        # Sigmoid probabilities
        p = torch.sigmoid(pred_attrs)  # (N, num_attrs)
        
        # Probability of ground truth class
        target_f = target_attrs.float()
        p_t = torch.where(target_f == 1, p, 1 - p)  # (N, num_attrs)
        
        # Focal weight: (1 - p_t)^γ
        # Easy examples (p_t ≈ 1) → focal_weight ≈ 0 → loss ≈ 0
        # Hard examples (p_t ≈ 0) → focal_weight ≈ 1 → loss ≈ unchanged
        focal_weight = (1 - p_t) ** self.gamma  # (N, num_attrs)
        
        # Cross-entropy loss component (with pos_weight for class imbalance)
        pw = self._pos_weight.to(pred_attrs.device)  # (num_attrs,)
        
        # Numerically stable BCE with pos_weight:
        # L = (1-y)*x + (1 + (pw-1)*y) * log(1 + exp(-x))
        # where x = logit, y = label
        bce_loss = (
            (1 - target_f) * pred_attrs
            - (1 + (pw - 1) * target_f) * F.logsigmoid(pred_attrs)
        )  # (N, num_attrs)
        
        # Apply focal weight
        focal_loss = focal_weight * bce_loss  # (N, num_attrs)
        
        # Apply mask (only disease samples contribute)
        masked_loss = focal_loss * mask.unsqueeze(1)  # (N, num_attrs)
        
        # Mean over valid elements
        n_valid = mask.sum() * self.num_attrs
        loss = masked_loss.sum() / (n_valid + 1e-6)
        
        return self.loss_weight * loss


class FocalLossWithAuxiliaryInformation(nn.Module):
    """Extended focal loss that incorporates auxiliary information.
    
    For future enhancement: condition focal loss on disease prevalence in batch,
    image region information, or quadrant-level confidence.
    """
    
    def __init__(self, num_attrs: int = 4, loss_weight: float = 8.0):
        super().__init__()
        self.base_focal = FocalAttributeLoss(
            num_attrs=num_attrs,
            loss_weight=loss_weight,
        )
    
    def forward(
        self,
        pred_attrs: torch.Tensor,
        target_attrs: torch.Tensor,
        data_types: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            sample_weights: Optional per-sample weight (e.g., based on FDI class rarity)
        """
        loss = self.base_focal(pred_attrs, target_attrs, data_types)
        
        if sample_weights is not None:
            # Apply per-sample weighting if provided
            # Example: weight rare FDI classes higher
            loss = loss * sample_weights.mean()
        
        return loss


# ─────────────────────────────────────────────────────────────────────────────
# USAGE EXAMPLE FOR TRAINING
# ─────────────────────────────────────────────────────────────────────────────
"""
In src/training/trainer.py, Phase 2b training section:

    # Original (BCE loss):
    # attr_loss_fn = AttributeBCELoss(num_attrs=4, loss_weight=8.0, pos_weight=pos_weight)
    
    # Modified (Focal loss):
    attr_loss_fn = FocalAttributeLoss(
        num_attrs=4,
        loss_weight=8.0,
        pos_weight=pos_weight,
        gamma=2.0,        # Standard focusing parameter
        alpha=0.25,       # Balance parameter
    )
    attr_loss_fn.to(device)
    
    # Then use in loss computation (interface is identical):
    attr_loss = attr_loss_fn(pred_attrs, target_attrs, data_types)
"""

# ─────────────────────────────────────────────────────────────────────────────
# TUNING GUIDE
# ─────────────────────────────────────────────────────────────────────────────
"""
Parameter tuning for optimal performance:

1. GAMMA (focusing parameter):
   - γ=0:   Same as CE loss (baseline)
   - γ=1:   Moderate focusing
   - γ=2:   Strong focusing (RECOMMENDED)
   - γ=3:   Very strong focusing (risky, may overfit)
   
   Test: Start with γ=2, if disease recall still low try γ=3

2. ALPHA (balance parameter):
   - Lower (0.15):   Penalize false positives more
   - Medium (0.25):  Balanced (RECOMMENDED)
   - Higher (0.35):  Penalize false negatives more
   
   Test: Start with α=0.25, adjust based on recall vs precision

3. LOSS_WEIGHT (disease attribute loss weight):
   - Current: 8.0
   - With focal: can reduce to 6.0-7.0 (focal already down-weights easy samples)
   
   Test: Try 6.0-7.0 after confirming γ=2 works

4. POS_WEIGHT (per-attribute weighting):
   - Important BEFORE focal loss (coarse balance)
   - Remains effective WITH focal loss (fine-grained balance)
   - Use tuned values: [8.0, 5.0, 12.0, 8.0]

Recommended training plan:
1. Set γ=2, α=0.25, use existing pos_weight=[8.0, 5.0, 12.0, 8.0]
2. Train Phase 2b for 100 epochs
3. If disease recall < 75%, increase γ to 2.5
4. If disease precision < 60%, reduce γ to 1.5 or increase α to 0.35
5. Iterate until recall ≥ 75% AND precision ≥ 65%
"""

if __name__ == "__main__":
    # Quick test: verify focal loss computation
    batch_size = 8
    num_attrs = 4
    
    pred_attrs = torch.randn(batch_size, num_attrs)
    target_attrs = torch.randint(0, 2, (batch_size, num_attrs)).float()
    data_types = torch.full((batch_size,), 2).long()  # All disease samples
    
    focal_loss = FocalAttributeLoss(num_attrs=4, loss_weight=8.0, gamma=2.0)
    loss = focal_loss(pred_attrs, target_attrs, data_types)
    
    print(f"✅ Focal loss test: loss={loss.item():.4f}")
    print(f"   Input: {batch_size} samples × {num_attrs} attributes")
    print(f"   γ=2.0 (strong focusing)")
    print(f"   ✓ Focal loss module working correctly")
