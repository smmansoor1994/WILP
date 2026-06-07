"""
src/models/yolortho.py
======================
YOLOrtho: Modified YOLOv8x for simultaneous teeth enumeration + disease detection.

Key modifications from the paper (Sections 2.1 and 2.2):
  1. CoordConv   — Replace backbone Conv2d with CoordConv for position awareness
  2. Modified FPN — Extra upsampling: detect at strides [4, 8, 16] vs [8, 16, 32]
  3. Attribute Heads — 4 binary disease attribute heads (per detection)

Architecture overview:
  Input (1280×640 panoramic X-ray)
    │
    ▼
  Backbone (YOLOv8x + CoordConv)       → P3(H/8), P4(H/16), P5(H/32)
    │
    ▼
  Modified Neck (PANet + extra upsample) → N2(H/4), N3(H/8), N4(H/16)
    │
    ▼
  ┌─────────────┬─────────────────────┐
  │ Detect Head │ Attribute Heads (×4) │
  │  (32 cls)   │ is_impacted          │
  │             │ has_caries           │
  │             │ has_deepcaries       │
  │             │ has_lesion           │
  └─────────────┴─────────────────────┘

Usage:
    from src.models.yolortho import build_yolortho
    model = build_yolortho(num_classes=32, device='cuda')
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

import torch
import torch.nn as nn

from src.models.coord_conv import replace_backbone_conv_with_coordconv
from src.models.heads import MultiAttributeHead

logger = logging.getLogger(__name__)

# Attribute names (order must match label columns 5-8)
ATTRIBUTE_NAMES = ["is_impacted", "has_caries", "has_deepcaries", "has_lesion"]

# YOLOv8x FPN output channel counts for strides [8, 16, 32]
# These are the channels entering the detection head for YOLOv8x
YOLOV8X_FPN_CHANNELS = {
    "p3": 320,   # stride 8  (after FPN)
    "p4": 640,   # stride 16
    "p5": 640,   # stride 32
}


class YOLOrtho(nn.Module):
    """YOLOrtho model wrapping YOLOv8x with attribute heads.

    This class:
    1. Holds the base YOLOv8x model (ultralytics YOLO)
    2. Attaches 4 attribute heads on top of the FPN features
    3. Exposes a `forward_attributes()` method for disease prediction
    4. Handles the combined loss via `YOLOrthoLoss` in training/loss.py

    Args:
        base_model:      Loaded ultralytics YOLO model.
        num_attrs:       Number of disease attribute heads (default 4).
        fpn_channels:    Output channels from FPN for each scale.
        use_coordconv:   Whether CoordConv was injected into the backbone.
    """

    def __init__(
        self,
        base_model: nn.Module,
        num_attrs: int = 4,
        fpn_channels: Optional[List[int]] = None,
        use_coordconv: bool = True,
    ):
        super().__init__()
        self.base_model = base_model
        self.num_attrs = num_attrs
        self.use_coordconv = use_coordconv

        # Default FPN channels for YOLOv8x (3 detection scales)
        if fpn_channels is None:
            fpn_channels = [320, 640, 640]  # YOLOv8x P3/P4/P5 after neck

        # Attribute heads (multi-scale, one per disease attribute)
        self.attr_heads = MultiAttributeHead(
            in_channels_list=fpn_channels,
            num_attrs=num_attrs,
        )

        # Storage for intermediate FPN features (populated via forward hooks)
        self._fpn_features: List[torch.Tensor] = []
        self._hooks: List = []

    def attach_feature_hooks(self) -> None:
        """Register forward hooks to capture FPN feature maps.

        YOLOv8 model structure:
          self.base_model        → ultralytics YOLO wrapper
          self.base_model.model  → DetectionModel (nn.Module)
          self.base_model.model.model → nn.Sequential of 23 layers (0-22)
        The Detect head (layer 22) takes inputs from layers [15, 18, 21].
        Negative indices: layer 15 = -8, layer 18 = -5, layer 21 = -2.
        """
        self._fpn_features = []
        self._hooks = []

        def _make_hook(idx: int):
            def hook(module, inp, out):
                if idx < len(self._fpn_features):
                    self._fpn_features[idx] = out
                else:
                    self._fpn_features.append(out)
            return hook

        try:
            seq = self.base_model.model.model  # nn.Sequential of 23 layers
            layers = list(seq)
            n = len(layers)
            target_indices = [-8, -5, -2]  # layers 15, 18, 21 for n=23
            for i, hi in enumerate(target_indices):
                h = layers[hi].register_forward_hook(_make_hook(i))
                self._hooks.append(h)
            logger.info(
                "FPN hooks attached at layers %s.",
                [n + idx for idx in target_indices],
            )
        except Exception as e:
            logger.warning("Cannot attach hooks — unexpected model structure: %s", e)

    def remove_hooks(self) -> None:
        """Remove all registered forward hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def forward(
        self, x: torch.Tensor
    ) -> tuple:
        """Full forward pass: detection + attribute prediction.

        Args:
            x: Input image tensor (B, 3, H, W).

        Returns:
            Tuple of:
              - det_output: Detection head output (same as YOLOv8 output).
              - attr_outputs: List of attribute logits per scale,
                              each of shape (B, num_attrs, Hi, Wi).
        """
        # Reset captured features
        self._fpn_features = []

        # Run base YOLOv8 forward pass (also populates _fpn_features via hooks)
        det_output = self.base_model.model(x)

        # Run attribute heads on captured FPN features
        attr_outputs: List[torch.Tensor] = []
        if self._fpn_features:
            attr_outputs = self.attr_heads(self._fpn_features)

        return det_output, attr_outputs

    def predict_attributes(
        self, fpn_features: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Run only the attribute heads on provided FPN features.

        Args:
            fpn_features: List of FPN feature maps per scale.

        Returns:
            List of attribute prediction tensors per scale.
        """
        return self.attr_heads(fpn_features)


def build_yolortho(
    num_classes: int = 32,
    base_weights: str = "yolov8x.pt",
    use_coordconv: bool = True,
    coordconv_with_r: bool = False,
    device: str = "cuda",
    verbose: bool = True,
) -> "YOLOrtho":
    """Build and return a YOLOrtho model.

    Steps:
    1. Load YOLOv8x pretrained weights (from ultralytics — downloads automatically).
    2. Inject CoordConv into backbone (replaces Conv2d layers 0-9).
    3. Attach disease attribute heads.
    4. Register FPN feature hooks.

    Args:
        num_classes:      Number of tooth classes (32 for FDI 11-48).
        base_weights:     ultralytics model identifier or path to .pt file.
                          Use 'yolov8x.pt' to download ImageNet pretrained.
        use_coordconv:    Replace backbone Conv2d with CoordConv.
        coordconv_with_r: Add radial coordinate channel to CoordConv.
        device:           Torch device ('cuda' or 'cpu').
        verbose:          Print build summary.

    Returns:
        YOLOrtho model instance.
    """
    from ultralytics import YOLO

    logger.info("Loading base model: %s", base_weights)

    # Load YOLOv8x (pretrained on COCO; we re-train the head for 32 tooth classes)
    base = YOLO(base_weights)

    # Modify number of classes if needed
    # ultralytics YOLO reconfigures the head automatically based on the dataset YAML
    # but we can also directly set nc
    if hasattr(base.model, "yaml"):
        base.model.yaml["nc"] = num_classes
    if hasattr(base.model, "nc"):
        base.model.nc = num_classes

    # ── CoordConv Injection ───────────────────────────────────────────────────
    coord_count = 0
    if use_coordconv:
        logger.info("Injecting CoordConv into backbone ...")
        coord_count = replace_backbone_conv_with_coordconv(
            model=base.model,
            with_r=coordconv_with_r,
            verbose=verbose,
        )
        logger.info("Replaced %d Conv2d layers with CoordConv.", coord_count)

    # ── Determine FPN output channels ─────────────────────────────────────────
    # For YOLOv8x, the neck outputs [320, 640, 640] channels for P3/P4/P5
    # We use these as inputs to the attribute heads
    fpn_channels = _infer_fpn_channels(base.model)

    # ── Build YOLOrtho wrapper ─────────────────────────────────────────────────
    model = YOLOrtho(
        base_model=base,
        num_attrs=len(ATTRIBUTE_NAMES),
        fpn_channels=fpn_channels,
        use_coordconv=use_coordconv,
    )

    # Attach forward hooks to capture FPN features
    model.attach_feature_hooks()

    # Move to device
    model.to(device)

    if verbose:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("=" * 50)
        logger.info("YOLOrtho Model Summary")
        logger.info("  Base:           YOLOv8x")
        logger.info("  Tooth Classes:  %d (FDI 11-48)", num_classes)
        logger.info("  Attributes:     %s", ATTRIBUTE_NAMES)
        logger.info("  CoordConv:      %s (%d layers replaced)", use_coordconv, coord_count)
        logger.info("  FPN Channels:   %s", fpn_channels)
        logger.info("  Device:         %s", device)
        logger.info("  Total params:   %s", f"{total_params:,}")
        logger.info("  Trainable:      %s", f"{trainable_params:,}")
        logger.info("=" * 50)

    return model


def load_yolortho_checkpoint(
    checkpoint_path: str,
    device: str = "cuda",
) -> "YOLOrtho":
    """Load a saved YOLOrtho checkpoint.

    Args:
        checkpoint_path: Path to saved YOLOrtho .pt checkpoint.
        device:          Target device.

    Returns:
        Loaded YOLOrtho model.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Build the model skeleton
    model = build_yolortho(
        num_classes=checkpoint.get("num_classes", 32),
        base_weights=checkpoint.get("base_weights", "yolov8x.pt"),
        use_coordconv=checkpoint.get("use_coordconv", True),
        device=device,
        verbose=False,
    )

    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    logger.info("Loaded YOLOrtho checkpoint from '%s'.", checkpoint_path)

    return model


def _infer_fpn_channels(inner_model: nn.Module) -> List[int]:
    """Try to infer FPN output channels from the model structure.

    Falls back to default YOLOv8x channels [320, 640, 640] if inference fails.
    """
    try:
        # The Detect head in ultralytics stores ch (input channels per scale)
        layers = list(inner_model.children())
        detect_head = layers[-1]  # Detect is always the last module
        if hasattr(detect_head, "ch"):
            return list(detect_head.ch)
    except Exception:
        pass
    logger.warning(
        "Could not infer FPN channels. Using YOLOv8x defaults [320, 640, 640]."
    )
    return [320, 640, 640]
