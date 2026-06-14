"""
src/models/cross_attention.py
==============================
ARCHON — Improvement B: MultiScaleFusion (Cross-Attention)

Cross-Attention Fusion Layer for ARCHON (ARCHON).
Fuses CNN local features (Q) with Swin global dental-arch context (K/V)
at every FPN scale (P3, P4, P5).

Architecture diagram reference (from the proposed hybrid design):
  CNN branch (local features: tooth texture, boundary, lesion patterns)
       │
       ├──────────────────────────────────┐
       │                                  │
  (Query)                          Global Context Embeddings
                                   from Swin Transformer
                                   (jaw-wide, quadrant awareness)
                                         │
                                      (Key, Value)
       │                                  │
       └──────────── Cross-Attention ─────┘
                          │
              Fused: local tooth detail + global arch context
                          │
              Unified Multi-Task Prediction Head

Motivation (Limitation 1 in the proposed improvement):
  CNN features (local) know about tooth texture, root shape, and lesion
  appearance. Swin features (global) know where in the dental arch we are
  (quadrant, jaw side, ordering). Cross-attention lets each local tooth
  detection "attend to" the full jaw context to resolve FDI numbering
  conflicts, e.g.: "this tooth looks like a molar AND the jaw context says
  there are 3 molars already on the right → this is tooth 18, not 17."

Cross-attention formula (standard transformer):
  Q = W_q * local_features   (CNN / local branch)
  K = W_k * global_features  (Swin / global branch)
  V = W_v * global_features
  Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

Design:
  - Operates at P5 scale (coarsest; 20×40 for 640×1280 input).
  - Fused output is then projected and added back to the CNN P5 features
    (residual connection) before the detection/attribute head.
  - Lightweight (single-head or multi-head, configurable).
"""

import logging
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class CrossAttentionFusion(nn.Module):
    """Cross-Attention Fusion: maps local CNN features with global Swin context.

    Implements cross-attention where:
      - Query (Q) comes from the LOCAL CNN feature map (P5 from YOLOv8 backbone).
      - Key and Value (K, V) come from the GLOBAL context features
        (output of GlobalContextEncoder / Swin Transformer branch).

    The fused output is the same shape as the CNN P5 input, so it can be
    used as a drop-in replacement without changing downstream modules.

    Args:
        cnn_channels:    Number of channels in the CNN (query) feature map.
        ctx_channels:    Number of channels in the Swin (key/value) feature map.
                         If different from cnn_channels, a projection is applied.
        num_heads:       Number of attention heads.
        dropout:         Dropout on the attention weights.
        use_pos_enc:     Add learnable 2D position encoding to Q before attention.
    """

    def __init__(
        self,
        cnn_channels: int = 640,
        ctx_channels: int = 640,
        num_heads: int = 8,
        dropout: float = 0.0,
        use_pos_enc: bool = True,
    ):
        super().__init__()
        self.cnn_channels = cnn_channels
        self.ctx_channels = ctx_channels
        self.num_heads = num_heads
        self.head_dim = cnn_channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_pos_enc = use_pos_enc

        assert cnn_channels % num_heads == 0, (
            f"cnn_channels ({cnn_channels}) must be divisible by num_heads ({num_heads})"
        )

        # Projections for Q, K, V
        self.q_proj = nn.Linear(cnn_channels, cnn_channels, bias=False)
        self.k_proj = nn.Linear(ctx_channels, cnn_channels, bias=False)
        self.v_proj = nn.Linear(ctx_channels, cnn_channels, bias=False)
        self.out_proj = nn.Linear(cnn_channels, cnn_channels, bias=True)

        self.attn_drop = nn.Dropout(dropout)
        self.out_drop = nn.Dropout(dropout)

        # Layer norms (Pre-LN architecture for stability)
        self.norm_q = nn.LayerNorm(cnn_channels)
        self.norm_kv = nn.LayerNorm(ctx_channels)

        # Optional learned 2D positional encoding (added to Q)
        # Max resolution: 80×160 (stride-4 on 640×1280) — sized for P5 (stride-32 = 20×40)
        # Using a fixed maximum and cropping avoids reallocation at different scales.
        self._pos_enc_cache: Optional[nn.Parameter] = None
        self._pos_max_h = 80
        self._pos_max_w = 160

        if use_pos_enc:
            self._pos_enc_cache = nn.Parameter(
                torch.zeros(1, self._pos_max_h * self._pos_max_w, cnn_channels)
            )
            nn.init.trunc_normal_(self._pos_enc_cache, std=0.02)

        # Feed-forward sublayer after cross-attention (residual)
        hidden = cnn_channels * 2
        self.ffn = nn.Sequential(
            nn.LayerNorm(cnn_channels),
            nn.Linear(cnn_channels, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, cnn_channels),
            nn.Dropout(dropout),
        )

    def _get_pos_enc(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Get positional encoding slice for spatial size (H, W)."""
        if self._pos_enc_cache is None:
            return torch.zeros(1, H * W, self.cnn_channels, device=device)
        # Reshape cache to 2D, interpolate to (H, W), flatten back to (1, H*W, C)
        full = self._pos_enc_cache.reshape(
            1, self._pos_max_h, self._pos_max_w, self.cnn_channels
        ).permute(0, 3, 1, 2)  # (1, C, H_max, W_max)
        resized = F.interpolate(full, size=(H, W), mode="bilinear", align_corners=False)
        return resized.permute(0, 2, 3, 1).reshape(1, H * W, self.cnn_channels)

    def forward(
        self,
        cnn_feat: torch.Tensor,
        ctx_feat: torch.Tensor,
    ) -> torch.Tensor:
        """Cross-attention fusion.

        Args:
            cnn_feat: (B, C_cnn, H, W) — local CNN feature map (query source).
            ctx_feat: (B, C_ctx, H, W) — global context feature map (key/value source).
                      Must have the same spatial size as cnn_feat.

        Returns:
            fused: (B, C_cnn, H, W) — fused feature map, same shape as cnn_feat.
        """
        B, C, H, W = cnn_feat.shape
        N = H * W

        # Flatten spatial dims: (B, C, H, W) → (B, N, C)
        q_in = cnn_feat.flatten(2).transpose(1, 2)   # (B, N, C_cnn)
        kv_in = ctx_feat.flatten(2).transpose(1, 2)  # (B, N, C_ctx)

        # Pre-LN
        q_in = self.norm_q(q_in)
        kv_in = self.norm_kv(kv_in)

        # Add positional encoding to Q
        if self.use_pos_enc:
            pos = self._get_pos_enc(H, W, cnn_feat.device)  # (1, N, C)
            q_in = q_in + pos

        # Project Q, K, V
        Q = self.q_proj(q_in)   # (B, N, C_cnn)
        K = self.k_proj(kv_in)  # (B, N, C_cnn)  [projected to same dim as Q]
        V = self.v_proj(kv_in)  # (B, N, C_cnn)

        # Multi-head split: (B, N, C) → (B, num_heads, N, head_dim)
        def split_heads(t):
            return t.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        Q, K, V = split_heads(Q), split_heads(K), split_heads(V)

        # Scaled dot-product attention
        attn = (Q @ K.transpose(-2, -1)) * self.scale  # (B, heads, N, N)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ V  # (B, heads, N, head_dim)
        out = out.transpose(1, 2).reshape(B, N, C)  # (B, N, C_cnn)
        out = self.out_proj(out)
        out = self.out_drop(out)

        # Residual connection from CNN feature (skip cross-attention noise early in training)
        x = cnn_feat.flatten(2).transpose(1, 2) + out  # (B, N, C)

        # Feed-forward sublayer
        x = x + self.ffn(x)

        # Reshape back to spatial feature map
        fused = x.transpose(1, 2).reshape(B, C, H, W)
        return fused


class MultiScaleFusion(nn.Module):
    """Apply cross-attention fusion at multiple FPN scales independently.

    The Swin Transformer naturally produces global context at the P5 scale.
    To enrich P3 and P4 features as well, this module upsamples the Swin
    output to each scale's resolution and fuses it there too.

    This implements the "Hierarchical Multi-Task Attention Mapping" from
    the proposed improvement strategy.

    Args:
        fpn_channels: List of [P3_channels, P4_channels, P5_channels].
        ctx_channels: Channel count of the Swin global context (matches P5).
        num_heads:    Number of attention heads for each scale.
        dropout:      Dropout rate.
    """

    def __init__(
        self,
        fpn_channels: List[int],
        ctx_channels: int = 640,
        num_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.fpn_channels = fpn_channels

        # Cross-attention fusion at each FPN scale
        self.fusions = nn.ModuleList([
            CrossAttentionFusion(
                cnn_channels=ch,
                ctx_channels=ctx_channels,
                num_heads=max(1, min(num_heads, ch // 64)),  # at least 1 head
                dropout=dropout,
            )
            for ch in fpn_channels
        ])

        # Project Swin context to each FPN scale's channel count if needed
        self.ctx_projections = nn.ModuleList([
            nn.Conv2d(ctx_channels, ch, 1) if ch != ctx_channels else nn.Identity()
            for ch in fpn_channels
        ])

    def forward(
        self,
        fpn_features: List[torch.Tensor],
        global_context: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Fuse global context into each FPN feature map.

        Args:
            fpn_features:   [P3, P4, P5] feature maps from YOLOv8 FPN.
            global_context: (B, C_ctx, H5, W5) — Swin output at P5 resolution.

        Returns:
            fused_features: [P3_fused, P4_fused, P5_fused]
        """
        fused = []
        for i, (feat, fusion) in enumerate(zip(fpn_features, self.fusions)):
            H_f, W_f = feat.shape[2], feat.shape[3]

            # Upsample/downsample global context to match this scale's spatial size
            if global_context.shape[2:] != (H_f, W_f):
                ctx_resized = F.interpolate(
                    global_context, size=(H_f, W_f), mode="bilinear", align_corners=False
                )
            else:
                ctx_resized = global_context

            # Cross-attention fusion: CrossAttentionFusion handles the channel
            # mismatch internally via k_proj/v_proj (ctx_channels → cnn_channels),
            # so pass ctx_resized directly at the original Swin channel width.
            fused_feat = fusion(feat, ctx_resized)
            fused.append(fused_feat)

        return fused
