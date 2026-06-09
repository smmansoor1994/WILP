"""
src/models/coord_conv.py
========================
ARCHON / ARCHON — CoordConv: Coordinate-Aware Convolution for position-sensitive tasks.

Reference:
  "An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution"
  Liu et al., NeurIPS 2018
  https://arxiv.org/abs/1807.03247

Motivation for ARCHON (Section 2.2 of the paper):
  Standard convolution is translation-invariant — it produces the same output
  regardless of WHERE in the image the feature appears.
  For teeth enumeration, POSITION is the primary signal:
    - Upper teeth vs. lower teeth
    - Left side vs. right side
  CoordConv appends normalized (x, y) coordinate channels to the feature map
  before the convolution, breaking translation equivariance and enabling the
  network to "know where it is" in the panoramic X-ray.
"""

import torch
import torch.nn as nn
from typing import Optional


class AddCoords(nn.Module):
    """Appends 2D coordinate channels (and optionally a radial channel) to a tensor.

    The coordinate channels are normalized to the range [-1, 1]:
      x_channel[b, 0, i, j] = j / (W-1) * 2 - 1   (horizontal position)
      y_channel[b, 0, i, j] = i / (H-1) * 2 - 1   (vertical position)

    Optionally, a radius channel is added:
      r_channel = sqrt(x^2 + y^2)

    Args:
        with_r: If True, also append the radial distance channel.
    """

    def __init__(self, with_r: bool = False):
        super().__init__()
        self.with_r = with_r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Concatenate coordinate channels to input tensor.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Tensor of shape (B, C+2, H, W) or (B, C+3, H, W) if with_r=True.
        """
        batch, _, h, w = x.shape
        device = x.device
        dtype = x.dtype

        # ── Horizontal coordinate channel (x-axis) ────────────────────────────
        # Shape: (1, 1, 1, W) → broadcast to (B, 1, H, W)
        xx = torch.linspace(-1.0, 1.0, steps=w, device=device, dtype=dtype)
        xx = xx.view(1, 1, 1, w).expand(batch, 1, h, w)

        # ── Vertical coordinate channel (y-axis) ──────────────────────────────
        # Shape: (1, 1, H, 1) → broadcast to (B, 1, H, W)
        yy = torch.linspace(-1.0, 1.0, steps=h, device=device, dtype=dtype)
        yy = yy.view(1, 1, h, 1).expand(batch, 1, h, w)

        # Concatenate along channel dimension
        out = torch.cat([x, xx, yy], dim=1)

        if self.with_r:
            # Radial distance channel: sqrt(x^2 + y^2), normalized to [0, 1]
            rr = torch.sqrt(xx ** 2 + yy ** 2)
            rr = rr / (2.0 ** 0.5)  # max possible value is sqrt(2)
            out = torch.cat([out, rr], dim=1)

        return out


class CoordConv(nn.Module):
    """CoordConv: drop-in replacement for nn.Conv2d with coordinate channels.

    Appends (x, y) coordinate channels to input before the convolution.
    All other arguments are identical to nn.Conv2d.

    Args:
        in_channels:  Number of input channels (same as nn.Conv2d).
        out_channels: Number of output channels.
        kernel_size:  Convolution kernel size.
        stride:       Convolution stride.
        padding:      Padding amount.
        dilation:     Dilation factor.
        groups:       Number of groups for grouped convolution.
        bias:         Whether to add a bias term.
        padding_mode: Padding mode.
        with_r:       Whether to add radial coordinate channel (default False).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        with_r: bool = False,
    ):
        super().__init__()
        self.add_coords = AddCoords(with_r=with_r)
        # Number of extra channels added by coordinate appending
        extra = 3 if with_r else 2
        # Internal conv2d receives in_channels + extra coordinate channels
        self.conv = nn.Conv2d(
            in_channels=in_channels + extra,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )
        # Store original in_channels for repr
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: append coords, then convolve.

        Args:
            x: Input of shape (B, in_channels, H, W).

        Returns:
            Output of shape (B, out_channels, H, W).
        """
        x = self.add_coords(x)    # (B, in_channels+2, H, W)
        return self.conv(x)        # (B, out_channels, H, W)

    def __repr__(self) -> str:
        return (
            f"CoordConv({self.in_channels}, {self.out_channels}, "
            f"kernel_size={self.conv.kernel_size}, "
            f"stride={self.conv.stride}, "
            f"padding={self.conv.padding})"
        )


# ─── Utility: inject CoordConv into a model's backbone ───────────────────────

def replace_backbone_conv_with_coordconv(
    model: nn.Module,
    backbone_attr: str = "model",
    with_r: bool = False,
    verbose: bool = False,
) -> int:
    """Replace all nn.Conv2d layers in the backbone portion with CoordConv.

    This function walks the backbone sub-module and replaces every Conv2d
    with its CoordConv equivalent, preserving all other parameters.

    Args:
        model:          The full YOLOv8 model.
        backbone_attr:  Attribute name of the backbone sub-module.
                        For ultralytics models, the backbone layers are
                        model.model[0..9] (indices 0-9 in model.model).
        with_r:         Pass to CoordConv (add radial channel).
        verbose:        Print each replaced layer.

    Returns:
        Number of Conv2d layers replaced.
    """
    count = 0

    # In ultralytics YOLOv8, model.model is a nn.Sequential of layers.
    # Backbone corresponds to the first 10 layers (indices 0-9).
    # Neck and head are the remaining layers.
    # We only patch backbone layers to preserve detection head behavior.

    backbone_layers = None
    if hasattr(model, "model") and hasattr(model.model, "__getitem__"):
        # ultralytics YOLOv8 structure
        # Total layers: backbone (0-9) + neck (10-21) + head (22+)
        # Only replace in backbone (first 10 layers)
        backbone_layers = nn.Sequential(*list(model.model.children())[:10])
        target_modules = list(model.model.children())[:10]
    else:
        # Fallback: replace in entire model
        target_modules = [model]

    for layer_module in target_modules:
        count += _replace_conv_recursive(layer_module, with_r, verbose)

    if verbose:
        print(f"CoordConv injection complete: {count} Conv2d layers replaced in backbone.")

    return count


def _replace_conv_recursive(
    module: nn.Module, with_r: bool, verbose: bool
) -> int:
    """Recursively replace all nn.Conv2d in module with CoordConv."""
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv2d) and not isinstance(child, CoordConv):
            # Create CoordConv with same parameters
            new_conv = CoordConv(
                in_channels=child.in_channels,
                out_channels=child.out_channels,
                kernel_size=child.kernel_size[0],
                stride=child.stride[0],
                padding=child.padding[0],
                dilation=child.dilation[0],
                groups=child.groups,
                bias=child.bias is not None,
                with_r=with_r,
            )
            # Copy existing weights for the original input channels.
            # IMPORTANT: zero the extra coord-channel weights so the CoordConv
            # behaves identically to the original Conv2d at initialisation.
            # PyTorch's default kaiming_uniform_ init for those channels would
            # otherwise inject random noise into every FPN feature map, causing
            # the attr heads to learn on corrupted features during Phase 2b
            # while seeing clean features at inference (no CoordConv).
            with torch.no_grad():
                new_conv.conv.weight[:, :child.in_channels] = child.weight
                new_conv.conv.weight[:, child.in_channels:].zero_()  # BUG FIX: zero coord channels
                if child.bias is not None:
                    new_conv.conv.bias.copy_(child.bias)
            setattr(module, name, new_conv)
            count += 1
            if verbose:
                print(f"  Replaced {name}: {child} → CoordConv")
        else:
            count += _replace_conv_recursive(child, with_r, verbose)
    return count
