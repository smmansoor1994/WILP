"""
src/models/swin_transformer.py
==============================
ARCHON — Improvement A: GlobalContextEncoder (Swin Transformer)

Lightweight Hierarchical Swin Transformer block for ARCHON (ARCHON).
Captures full dental arch context from the P5 coarse feature map.

Architecture diagram reference (from the proposed hybrid design):
  P5 feature maps (global semantic, high-level)
      │
  Patch Embedding Layer   ← flatten spatial → tokens
      │
  Swin Transformer Blocks (Shifted Window Self-Attention)
      │
  Global Context Embeddings
      ├── Jaw-wide relationships
      ├── Tooth ordering logic
      └── Quadrant awareness

Motivation (Limitation 1 in the proposed improvement):
  Standard CNN convolutions have a limited receptive field — even after many
  layers the "context window" is bounded by the kernel stack. In a panoramic
  dental X-ray, whether a given feature is tooth 16 or 26 depends on the
  FULL jaw layout, not just the local patch. A Swin Transformer with shifted
  windows captures these long-range dependencies efficiently.

Design choices:
  - Operates only on the P5 (stride 32 / coarsest) feature map to keep
    computation low (small spatial size: 20×40 for a 640×1280 input).
  - Single Swin stage (2 blocks: one regular, one shifted window).
  - Output projected back to the same channel dimension as P5, so it can
    be fused with CNN features via the CrossAttentionFusion layer without
    changing any existing module sizes.
  - Does NOT replace the backbone — it augments it as a parallel branch,
    preserving all existing detection capability.

References:
  Liu et al., "Swin Transformer: Hierarchical Vision Transformer using
  Shifted Windows", ICCV 2021. https://arxiv.org/abs/2103.14030
"""

import math
import logging
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ─── Utilities ────────────────────────────────────────────────────────────────

def _to_2tuple(x):
    return (x, x) if isinstance(x, int) else tuple(x)


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Partition feature map into non-overlapping windows.

    Args:
        x:           (B, H, W, C)
        window_size: Window size (square).

    Returns:
        windows: (num_windows * B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(-1, window_size, window_size, C)
    return windows


def window_reverse(
    windows: torch.Tensor, window_size: int, H: int, W: int
) -> torch.Tensor:
    """Reverse window partition back to feature map.

    Args:
        windows:     (num_windows * B, window_size, window_size, C)
        window_size: Window size.
        H, W:        Original feature map spatial dims.

    Returns:
        x: (B, H, W, C)
    """
    B_total = windows.shape[0]
    B = int(B_total / (H * W / window_size / window_size))
    x = windows.view(
        B, H // window_size, W // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


# ─── Window Multi-Head Self-Attention ─────────────────────────────────────────

class WindowAttention(nn.Module):
    """Window-based Multi-Head Self-Attention (W-MSA / SW-MSA).

    Supports both regular and shifted-window attention.

    Args:
        dim:         Number of input channels.
        window_size: (Wh, Ww) window dimensions.
        num_heads:   Number of attention heads.
        qkv_bias:    Add learnable bias to Q, K, V.
        attn_drop:   Attention dropout probability.
        proj_drop:   Output projection dropout probability.
    """

    def __init__(
        self,
        dim: int,
        window_size: Tuple[int, int],
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (Wh, Ww)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads
            )
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # Compute relative position index
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # (2, Wh, Ww)
        coords_flatten = torch.flatten(coords, 1)  # (2, Wh*Ww)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # (Wh*Ww, Wh*Ww)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x:    (num_windows*B, N, C) where N = window_size^2.
            mask: (num_windows, N, N) or None.

        Returns:
            (num_windows*B, N, C)
        """
        B_, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)  # each: (B_, num_heads, N, head_dim)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # (B_, num_heads, N, N)

        # Relative position bias
        rel_pos_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )
        rel_pos_bias = rel_pos_bias.permute(2, 0, 1).contiguous()  # (num_heads, N, N)
        attn = attn + rel_pos_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ─── Swin Transformer Block ───────────────────────────────────────────────────

class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block with optional shifted window.

    Args:
        dim:         Number of channels.
        input_resolution: (H, W) of the input feature map.
        num_heads:   Number of attention heads.
        window_size: Window size (square, default 4 for small feature maps).
        shift_size:  Shift amount (0 = regular, window_size//2 = shifted).
        mlp_ratio:   MLP hidden dim / dim.
        drop:        Dropout rate.
        attn_drop:   Attention dropout rate.
    """

    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        num_heads: int,
        window_size: int = 4,
        shift_size: int = 0,
        mlp_ratio: float = 2.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        # Clip shift_size and window_size to not exceed feature map dimensions
        H, W = input_resolution
        if min(H, W) <= window_size:
            self.shift_size = 0
            self.window_size = min(H, W)
        else:
            self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim=dim,
            window_size=_to_2tuple(self.window_size),
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(drop),
        )

        # Compute attention mask for SW-MSA
        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros(1, H, W, 1)
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(
                attn_mask == 0, 0.0
            )
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, H*W, C)
        Returns:
            (B, H*W, C)
        """
        H, W = self.input_resolution
        B, L, C = x.shape

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Cyclic shift for shifted window attention
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Partition into windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # Attention
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

        # Reverse cyclic shift
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # Residual + MLP
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


# ─── Global Context Encoder (the Swin "branch") ───────────────────────────────

class GlobalContextEncoder(nn.Module):
    """Hierarchical Swin Transformer branch for global dental arch context.

    Takes the coarsest CNN feature map (P5) and applies two Swin Transformer
    blocks (regular + shifted) to model jaw-wide spatial relationships.

    Outputs a global context tensor of the same spatial and channel size as
    the input, ready for cross-attention fusion.

    Args:
        in_channels:  Number of channels of the P5 feature map.
        num_heads:    Number of attention heads (default 8).
        window_size:  Swin window size (default 4 — suitable for 20×40 P5).
        mlp_ratio:    MLP expansion ratio inside each Swin block.
        drop:         Dropout rate.
    """

    def __init__(
        self,
        in_channels: int = 640,
        num_heads: int = 8,
        window_size: int = 4,
        mlp_ratio: float = 2.0,
        drop: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels

        # Patch embedding: project to transformer dim
        self.patch_embed = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, in_channels),
        )

        # Two Swin blocks: block 0 = regular (shift=0), block 1 = shifted
        # Input resolution is set lazily on first forward to handle variable sizes.
        self._window_size = window_size
        self._num_heads = num_heads
        self._mlp_ratio = mlp_ratio
        self._drop = drop
        self._blocks: Optional[nn.ModuleList] = None
        self._last_resolution: Optional[Tuple[int, int]] = None

        # Output projection back to original channel space
        self.proj_out = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, in_channels),
        )

    def _build_blocks(self, H: int, W: int) -> None:
        """Lazy-build Swin blocks once spatial resolution is known."""
        self._last_resolution = (H, W)
        ws = min(self._window_size, min(H, W))
        shift = ws // 2 if min(H, W) > ws else 0
        self._blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=self.in_channels,
                input_resolution=(H, W),
                num_heads=self._num_heads,
                window_size=ws,
                shift_size=0,
                mlp_ratio=self._mlp_ratio,
                drop=self._drop,
            ),
            SwinTransformerBlock(
                dim=self.in_channels,
                input_resolution=(H, W),
                num_heads=self._num_heads,
                window_size=ws,
                shift_size=shift,
                mlp_ratio=self._mlp_ratio,
                drop=self._drop,
            ),
        ])
        # Move to same device as existing parameters
        if next(self.parameters(), None) is not None:
            device = next(self.parameters()).device
            self._blocks = self._blocks.to(device)

    def forward(self, p5: torch.Tensor) -> torch.Tensor:
        """Encode global context from P5 feature map.

        Args:
            p5: (B, C, H, W) — coarsest FPN feature map.

        Returns:
            (B, C, H, W) — global context-enriched features, same shape.
        """
        B, C, H, W = p5.shape

        # Build or rebuild blocks if resolution changed
        if self._blocks is None or self._last_resolution != (H, W):
            self._build_blocks(H, W)
            # Ensure new blocks are on the right device
            self._blocks = self._blocks.to(p5.device)

        # (B, C, H, W) → (B, H*W, C) for transformer
        x = p5.flatten(2).transpose(1, 2)  # (B, H*W, C)
        x = self.patch_embed(x)

        for block in self._blocks:
            x = block(x)

        x = self.proj_out(x)

        # (B, H*W, C) → (B, C, H, W)
        out = x.transpose(1, 2).reshape(B, C, H, W)
        return out
