"""
src/models/archon.py
======================
ARCHON model definitions — ARCHON baseline + ARCHONModel (ARCHON).

ARCHON: Modified YOLOv8x for simultaneous teeth enumeration + disease detection.
  (Baseline implementation of arXiv:2308.05967)

ARCHONModel (ARCHON): Extends ARCHON with three additional components:
  A. GlobalContextEncoder — Swin Transformer on P5 for dental arch context
  B. MultiScaleFusion     — Cross-Attention fusion at P3/P4/P5 scales
  C. HybridMultiTaskHead  — 3-level severity + quadrant auxiliary classifier

Key modifications from the paper (Sections 2.1 and 2.2):
  1. CoordConv   — Replace backbone Conv2d with CoordConv for position awareness
  2. Modified FPN — Extra upsampling: detect at strides [4, 8, 16] vs [8, 16, 32]
  3. Attribute Heads — 4 binary disease attribute heads (per detection)

Hybrid extension (ARCHONModel) — three improvements over baseline:
  A. GlobalContextEncoder (Swin Transformer) — long-range jaw context for FDI accuracy
  B. CrossAttentionFusion — maps local CNN tooth features with global arch context
  C. HybridMultiTaskHead — severity-level disease classification + quadrant auxiliary loss

Architecture overview (baseline ARCHON):
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

Architecture overview (ARCHONModel — proposed improvement):
  Input (1280×640 panoramic X-ray)
    │
    ▼
  Backbone (YOLOv8x + CoordConv)       → P3, P4, P5 (CNN local features)
    │                    │
    │              P5 (coarsest)
    │                    │
    │        GlobalContextEncoder       ← Swin Transformer branch
    │        (shifted-window attention)    captures jaw-wide context
    │                    │
    │           Global Context Embeddings
    │           (jaw layout, quadrant order, tooth count)
    │                    │
    ▼                    ▼
  MultiScaleFusion (CrossAttentionFusion at P3, P4, P5)
  Q=CNN local features, K/V=Swin global context
    │
    ▼  Fused features (local tooth detail + global arch understanding)
    │
  ┌────────────────────────────────────────────────────────────┐
  │  Unified Multi-Task Prediction Head                        │
  │  ┌──────────────────┐  ┌────────────────────────────────┐ │
  │  │  Detection Head  │  │  Hierarchical Classification   │ │
  │  │  · Localization  │  │  · Quadrant Prediction (4 cls) │ │
  │  │  · BBox          │  │  · FDI Numbering    (32 cls)   │ │
  │  │  · Confidence    │  │  · Severity per Disease (3 lvl)│ │
  │  └──────────────────┘  └────────────────────────────────┘ │
  └────────────────────────────────────────────────────────────┘

Usage:
    from src.models.yolortho import build_archon_base, build_archon_model
    model = build_archon_base(num_classes=32, device='cuda')              # baseline
    model = build_archon_model(num_classes=32, device='cuda')       # hybrid
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

import torch
import torch.nn as nn

from src.models.coord_conv import replace_backbone_conv_with_coordconv
from src.models.heads import MultiAttributeHead
from src.models.swin_transformer import GlobalContextEncoder
from src.models.cross_attention import MultiScaleFusion
from src.models.hybrid_head import HybridMultiTaskHead

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


class ARCHON(nn.Module):
    """ARCHON model wrapping YOLOv8x with attribute heads.

    This class:
    1. Holds the base YOLOv8x model (ultralytics YOLO)
    2. Attaches 4 attribute heads on top of the FPN features
    3. Exposes a `forward_attributes()` method for disease prediction
    4. Handles the combined loss via `ARCHONLoss` in training/loss.py

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


def build_archon_base(
    num_classes: int = 32,
    base_weights: str = "yolov8x.pt",
    use_coordconv: bool = True,
    coordconv_with_r: bool = False,
    device: str = "cuda",
    verbose: bool = True,
) -> "ARCHON":
    """Build and return a ARCHON model.

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
        ARCHON model instance.
    """
    import torch
    from ultralytics import YOLO

    logger.info("Loading base model: %s", base_weights)

    # ─── Custom checkpoint loader for ARCHON architectures ────────────────────
    # Ultralytics YOLO strict format checking fails on custom ARCHON checkpoints
    # (CoordConv, custom heads, etc.). Use torch.load with weights_only=False.
    base = None
    try:
        # Try standard YOLO loading first
        base = YOLO(base_weights)
    except (TypeError, RuntimeError) as e:
        error_msg = str(e)
        if "references types outside" in error_msg or "pickle" in error_msg.lower():
            logger.warning(
                "YOLO format check failed (custom architecture detected). "
                "Loading with torch.load (weights_only=False)..."
            )
            # Load ARCHON checkpoint directly
            try:
                ckpt = torch.load(base_weights, map_location="cpu", weights_only=False)
                logger.info("✓ Loaded checkpoint with custom types support")
                
                # Create vanilla YOLOv8x as template for the YOLO wrapper
                base = YOLO("yolov8x.pt")
                
                # Extract model from checkpoint
                if isinstance(ckpt, dict) and "model" in ckpt:
                    loaded_model = ckpt["model"]
                else:
                    loaded_model = ckpt
                
                # If it's a state dict, load it directly
                if isinstance(loaded_model, dict):
                    try:
                        base.model.load_state_dict(loaded_model, strict=False)
                    except RuntimeError:
                        logger.warning("Strict loading failed, trying with strict=False...")
                else:
                    # It's a full model object — extract state dict
                    try:
                        base.model.load_state_dict(loaded_model.state_dict(), strict=False)
                    except (AttributeError, RuntimeError):
                        pass
                        
            except Exception as load_err:
                logger.error("Failed to load checkpoint: %s", load_err)
                raise
        else:
            raise

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

    # ── Build ARCHON wrapper ─────────────────────────────────────────────────
    model = ARCHON(
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
        logger.info("ARCHON Model Summary")
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


def load_archon_checkpoint(
    checkpoint_path: str,
    device: str = "cuda",
) -> "ARCHON":
    """Load a saved ARCHON checkpoint.

    Args:
        checkpoint_path: Path to saved ARCHON .pt checkpoint.
        device:          Target device.

    Returns:
        Loaded ARCHON model.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Build the model skeleton
    model = build_archon_base(
        num_classes=checkpoint.get("num_classes", 32),
        base_weights=checkpoint.get("base_weights", "yolov8x.pt"),
        use_coordconv=checkpoint.get("use_coordconv", True),
        device=device,
        verbose=False,
    )

    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    logger.info("Loaded ARCHON checkpoint from '%s'.", checkpoint_path)

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


# ─────────────────────────────────────────────────────────────────────────────
# ARCHONModel: CNN + Swin Transformer + Cross-Attention + Hybrid Head
# ─────────────────────────────────────────────────────────────────────────────

class ARCHONModel(nn.Module):
    """ARCHON: augments ARCHON with three architectural improvements.

    Improvement A — GlobalContextEncoder (Swin Transformer branch):
      Processes the coarsest P5 feature map through two Swin Transformer blocks
      (regular + shifted-window attention) to produce global context embeddings
      that encode jaw-wide spatial relationships, quadrant ordering, and
      long-range tooth counting. Addresses the FDI numbering conflict limitation.

    Improvement B — MultiScaleFusion (Cross-Attention):
      At each FPN scale (P3, P4, P5), performs cross-attention where local CNN
      features (Query) attend to the global Swin context (Key, Value). This lets
      each individual tooth detection incorporate full dental-arch awareness.

    Improvement C — HybridMultiTaskHead:
      Replaces the binary disease attribute heads with 3-level severity heads
      (healthy / mild / severe per disease attribute) plus an auxiliary quadrant
      classification head for richer gradient signal.

    The YOLOv8 detection head (tooth localization + 32-class FDI) is unchanged —
    all improvements are additive, preserving the core detection capability.

    Args:
        base_model:        Loaded ultralytics YOLO model.
        fpn_channels:      [P3_ch, P4_ch, P5_ch] from YOLOv8x neck.
        use_coordconv:     Whether CoordConv was injected into the backbone.
        swin_num_heads:    Attention heads in the Swin Transformer blocks.
        swin_window_size:  Window size for Swin (should be ≤ P5 spatial dims).
        fusion_num_heads:  Attention heads in each CrossAttentionFusion layer.
        num_severity_lvls: Severity levels per disease attribute (default 3).
    """

    def __init__(
        self,
        base_model: nn.Module,
        fpn_channels: Optional[List[int]] = None,
        use_coordconv: bool = True,
        swin_num_heads: int = 8,
        swin_window_size: int = 4,
        fusion_num_heads: int = 4,
        num_severity_lvls: int = 3,
    ):
        super().__init__()
        self.base_model = base_model
        self.use_coordconv = use_coordconv

        if fpn_channels is None:
            fpn_channels = [320, 640, 640]
        self.fpn_channels = fpn_channels

        # P5 channel count (coarsest scale; input to Swin)
        p5_channels = fpn_channels[-1]

        # ── Improvement A: Global Context Encoder (Swin branch on P5) ────────
        self.global_encoder = GlobalContextEncoder(
            in_channels=p5_channels,
            num_heads=swin_num_heads,
            window_size=swin_window_size,
        )

        # ── Improvement B: Multi-Scale Cross-Attention Fusion ─────────────────
        self.fusion = MultiScaleFusion(
            fpn_channels=fpn_channels,
            ctx_channels=p5_channels,
            num_heads=fusion_num_heads,
        )

        # ── Improvement C: Hybrid Multi-Task Head ─────────────────────────────
        self.hybrid_head = HybridMultiTaskHead(
            in_channels_list=fpn_channels,
            num_attrs=4,
            num_severity_levels=num_severity_lvls,
        )

        # Keep the baseline binary attribute heads for backward-compat inference
        # (used when attr_best.pt is available and hybrid_best.pt is not yet trained)
        self.attr_heads = MultiAttributeHead(
            in_channels_list=fpn_channels,
            num_attrs=4,
        )

        # FPN feature storage (populated via hooks on base_model)
        self._fpn_features: List[torch.Tensor] = []
        self._hooks: List = []

    def attach_feature_hooks(self) -> None:
        """Register forward hooks on the YOLOv8 FPN layers."""
        self._fpn_features = []
        self._hooks = []

        def _make_hook(idx: int):
            def hook(module, inp, out):
                while len(self._fpn_features) <= idx:
                    self._fpn_features.append(None)
                self._fpn_features[idx] = out
            return hook

        try:
            seq = self.base_model.model.model
            layers = list(seq)
            n = len(layers)
            for i, hi in enumerate([-8, -5, -2]):  # layers 15, 18, 21
                h = layers[hi].register_forward_hook(_make_hook(i))
                self._hooks.append(h)
            logger.info("Hybrid FPN hooks attached at layers %s.", [n + hi for hi in [-8, -5, -2]])
        except Exception as e:
            logger.warning("Cannot attach hybrid hooks: %s", e)

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        """Full hybrid forward pass.

        Args:
            x: (B, 3, H, W) input image tensor.

        Returns:
            dict with keys:
              'det_output'    — YOLOv8 detection head output.
              'severity'      — List of severity logit tensors per scale.
              'quadrant'      — List of quadrant logit tensors per scale.
              'attr_outputs'  — Legacy binary attribute logits per scale.
              'fpn_fused'     — Fused FPN features (for loss computation).
        """
        self._fpn_features = []

        # ── Run YOLOv8 backbone + neck + detect head ──────────────────────────
        det_output = self.base_model.model(x)

        if not self._fpn_features or any(f is None for f in self._fpn_features[:3]):
            # Hooks may not have fired (e.g. ultralytics training mode bypasses hooks)
            return {
                "det_output": det_output,
                "severity": [],
                "quadrant": [],
                "attr_outputs": [],
                "fpn_fused": [],
            }

        fpn = self._fpn_features[:3]  # [P3, P4, P5]

        # ── Improvement A: Swin global context from P5 ────────────────────────
        p5 = fpn[2]
        global_ctx = self.global_encoder(p5)  # (B, C5, H5, W5)

        # ── Improvement B: Cross-attention fusion at all three scales ─────────
        fpn_fused = self.fusion(fpn, global_ctx)  # [P3_fused, P4_fused, P5_fused]

        # ── Improvement C: Hybrid multi-task head on fused features ──────────
        hybrid_out = self.hybrid_head(fpn_fused)

        # ── Legacy binary attribute heads (on fused features) ─────────────────
        attr_outputs = self.attr_heads(fpn_fused)

        return {
            "det_output": det_output,
            "severity": hybrid_out["severity"],
            "quadrant": hybrid_out["quadrant"],
            "attr_outputs": attr_outputs,
            "fpn_fused": fpn_fused,
        }


def build_archon_model(
    num_classes: int = 32,
    base_weights: str = "yolov8x.pt",
    use_coordconv: bool = True,
    coordconv_with_r: bool = False,
    swin_num_heads: int = 8,
    swin_window_size: int = 4,
    fusion_num_heads: int = 4,
    num_severity_lvls: int = 3,
    device: str = "cuda",
    verbose: bool = True,
) -> "ARCHONModel":
    """Build and return a ARCHONModel model.

    Steps:
    1. Load YOLOv8x pretrained weights.
    2. Inject CoordConv into backbone layers 0-9.
    3. Attach GlobalContextEncoder (Swin Transformer) on P5.
    4. Attach MultiScaleFusion (Cross-Attention) at all FPN scales.
    5. Attach HybridMultiTaskHead (severity + quadrant auxiliary).
    6. Register FPN feature hooks.

    Args:
        num_classes:       Number of tooth classes (32 for FDI 11-48).
        base_weights:      ultralytics model id or path to .pt weights.
        use_coordconv:     Inject CoordConv into backbone.
        coordconv_with_r:  Add radial coordinate channel.
        swin_num_heads:    Attention heads in Swin Transformer.
        swin_window_size:  Window size for Swin blocks.
        fusion_num_heads:  Attention heads in CrossAttentionFusion.
        num_severity_lvls: Severity levels per disease (3).
        device:            Target device.
        verbose:           Print build summary.

    Returns:
        ARCHONModel instance ready for training or inference.
    """
    from ultralytics import YOLO

    logger.info("Building ARCHONModel — base: %s", base_weights)

    base = YOLO(base_weights)
    if hasattr(base.model, "yaml"):
        base.model.yaml["nc"] = num_classes
    if hasattr(base.model, "nc"):
        base.model.nc = num_classes

    coord_count = 0
    if use_coordconv:
        coord_count = replace_backbone_conv_with_coordconv(
            model=base.model,
            with_r=coordconv_with_r,
            verbose=verbose,
        )
        logger.info("CoordConv: replaced %d Conv2d layers in backbone.", coord_count)

    fpn_channels = _infer_fpn_channels(base.model)

    model = ARCHONModel(
        base_model=base,
        fpn_channels=fpn_channels,
        use_coordconv=use_coordconv,
        swin_num_heads=swin_num_heads,
        swin_window_size=swin_window_size,
        fusion_num_heads=fusion_num_heads,
        num_severity_lvls=num_severity_lvls,
    )
    model.attach_feature_hooks()
    model.to(device)

    if verbose:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        swin_params = sum(p.numel() for p in model.global_encoder.parameters())
        fusion_params = sum(p.numel() for p in model.fusion.parameters())
        head_params = sum(p.numel() for p in model.hybrid_head.parameters())
        logger.info("=" * 60)
        logger.info("ARCHONModel Model Summary")
        logger.info("  Base:                YOLOv8x")
        logger.info("  Tooth Classes:       %d (FDI 11-48)", num_classes)
        logger.info("  CoordConv:           %s (%d layers)", use_coordconv, coord_count)
        logger.info("  FPN Channels:        %s", fpn_channels)
        logger.info("  Swin heads/window:   %d / %d", swin_num_heads, swin_window_size)
        logger.info("  Fusion heads:        %d", fusion_num_heads)
        logger.info("  Severity levels:     %d", num_severity_lvls)
        logger.info("  --- Parameter counts ---")
        logger.info("  Swin encoder:        %s", f"{swin_params:,}")
        logger.info("  Cross-Attn fusion:   %s", f"{fusion_params:,}")
        logger.info("  Hybrid head:         %s", f"{head_params:,}")
        logger.info("  Total:               %s", f"{total_params:,}")
        logger.info("  Trainable:           %s", f"{trainable_params:,}")
        logger.info("=" * 60)

    return model
