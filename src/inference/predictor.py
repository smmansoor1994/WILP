"""
src/inference/predictor.py
===========================
YOLOrtho inference pipeline.

Runs the full prediction pipeline:
  1. Load image (file, directory, or URL)
  2. Run YOLOv8 detection head → N bounding boxes + class probabilities
  3. Run attribute heads → 4 disease attribute probabilities per detection
  4. Apply linear sum assignment post-processing
  5. Return structured ToothDetection results
  6. Optionally save visualized output

Usage:
    predictor = YOLOrthoPredictor(weights_path="weights/yolortho_best.pt")

    # Single image
    results = predictor.predict("path/to/xray.jpg")

    # Directory
    predictor.predict("path/to/directory/", output_dir="outputs/predictions/")

    # Evaluation on validation set
    predictor.evaluate(data_yaml="config/dataset.yaml")
"""

import json
import logging
from pathlib import Path
from typing import List, Optional, Union, Dict, Any

import cv2
import numpy as np
import torch

from src.inference.postprocess import (
    ToothDetection,
    postprocess_yolo_output,
    summarize_detections,
)
from src.utils.visualize import draw_teeth_detections

logger = logging.getLogger(__name__)

# Supported image extensions
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


class YOLOrthoPredictor:
    """Inference class for YOLOrtho model.

    Args:
        weights_path:    Path to YOLOrtho .pt model weights.
        device:          Torch device ('cuda' or 'cpu').
        conf_threshold:  Minimum detection confidence.
        iou_threshold:   NMS IoU threshold.
        attr_threshold:  Threshold for binary attribute classification.
        img_size:        Input image size (H, W) — should match training.
        attr_inference_mode: How to aggregate FPN features for disease prediction.
            'per_tooth'  — sample feature map at each detected box centre (default,
                           matches trainer.py after the per-tooth supervision fix).
            'global_avg' — global spatial average across the whole feature map
                           (use this for models trained BEFORE the per-tooth fix,
                           e.g. baseline-disease-100-main).
    """

    def __init__(
        self,
        weights_path: str,
        device: str = "cuda",
        conf_threshold: float = 0.1,
        iou_threshold: float = 0.45,
        attr_threshold: float = 0.3,
        img_size: tuple = (640, 1280),
        attr_weights_path: Optional[str] = None,
        attr_inference_mode: str = "per_tooth",
    ):
        self.weights_path = Path(weights_path)
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.attr_threshold = attr_threshold
        self.img_size = img_size
        self.attr_weights_path = attr_weights_path
        self.attr_inference_mode = attr_inference_mode  # 'per_tooth' | 'global_avg'

        # Models loaded lazily
        self._det_model = None       # ultralytics YOLO detection model
        self._attr_heads = None      # attribute head module
        self._attr_hooks = []        # feature capture hooks

        # FPN features captured during forward pass
        self._fpn_features: List[torch.Tensor] = []

        logger.info(
            "YOLOrthoPredictor initialized. Weights: '%s'", self.weights_path
        )

    def _load_models(self) -> None:
        """Lazy-load detection model and attribute heads."""
        if self._det_model is not None:
            return

        if not self.weights_path.exists():
            raise FileNotFoundError(
                f"Model weights not found: '{self.weights_path}'\n"
                f"Run 'python main.py --mode train' first to train the model."
            )

        logger.info("Loading model from '%s' on device '%s' ...", self.weights_path, self.device)

        from ultralytics import YOLO
        self._det_model = YOLO(str(self.weights_path))

        # Try to load attribute head weights if available.
        # Search order: same dir → parent dir → user-supplied path (set via attr_weights_path)
        _attr_candidates = [
            self.weights_path.parent / "attr_best.pt",       # sibling of main weights
            self.weights_path.parent.parent / "attr_best.pt",  # one level up
        ]
        if hasattr(self, "attr_weights_path") and self.attr_weights_path:
            _attr_candidates.insert(0, Path(self.attr_weights_path))

        attr_path = next((p for p in _attr_candidates if p.exists()), None)
        if attr_path:
            logger.info("Found attribute weights at '%s'.", attr_path)
            self._load_attr_heads(attr_path)
        else:
            logger.warning(
                "Attribute weights (attr_best.pt) not found. "
                "Disease attributes will be unavailable (all teeth shown as healthy). "
                "Searched: %s",
                ", ".join(str(p) for p in _attr_candidates),
            )

        logger.info("Models loaded.")

    def _load_attr_heads(self, attr_path: Path) -> None:
        """Load attribute head weights from a checkpoint file."""
        try:
            from src.models.heads import MultiAttributeHead
            from src.models.yolortho import _infer_fpn_channels
            import math

            checkpoint = torch.load(str(attr_path), map_location=self.device)
            fpn_channels = [320, 640, 640]  # YOLOv8x defaults

            self._attr_heads = MultiAttributeHead(
                in_channels_list=fpn_channels,
                num_attrs=4,
            )
            self._attr_heads.load_state_dict(
                checkpoint.get("attr_heads_state", {}), strict=False
            )
            self._attr_heads.to(self.device)
            self._attr_heads.eval()

            # ── Sanity check: warn if heads are still at bias initialisation ──
            state = checkpoint.get("attr_heads_state", {})
            init_bias = -math.log(99)  # ≈ -4.595 — what nn.init sets at startup
            bias_vals = [
                v.item() for k, v in state.items()
                if "out.bias" in k
            ]
            if bias_vals and all(abs(b - init_bias) < 0.01 for b in bias_vals):
                logger.warning(
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    "  UNTRAINED ATTR HEADS DETECTED in '%s'\n"
                    "  All output biases are still at initialisation value\n"
                    "  (sigmoid=0.01).  This happens when the training loop\n"
                    "  had loss=0.0 every epoch (no_grad bug or missing\n"
                    "  labels_ext data) and no gradients ever updated the\n"
                    "  attribute heads.\n"
                    "  ► Disease diagnosis will show EVERYTHING AS HEALTHY.\n"
                    "  ► Re-run Phase 2b training with the current fixed code.\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                    attr_path,
                )
            else:
                trained_epochs = checkpoint.get("epoch", "?")
                best_loss = checkpoint.get("loss", "?")
                logger.info(
                    "Attr heads appear trained: saved epoch=%s, best_loss=%s",
                    trained_epochs, best_loss,
                )

            # Attach hooks to capture FPN features
            self._attach_fpn_hooks()
            logger.info("Attribute heads loaded from '%s'.", attr_path)
        except Exception as e:
            logger.warning(
                "Could not load attribute heads: %s. Disease attributes will be unavailable.", e
            )

    def _attach_fpn_hooks(self) -> None:
        """Attach hooks to capture intermediate FPN feature maps.

        YOLOv8 model structure (23 layers, indices 0-22):
          self._det_model.model        → ultralytics DetectionModel
          self._det_model.model.model  → nn.Sequential of 23 layers
        The Detect head (layer 22) takes inputs from layers [15, 18, 21].
        Negative indices: layer 15 = -8, layer 18 = -5, layer 21 = -2.
        """
        self._fpn_features = []

        def _make_hook(idx):
            def hook(module, inp, out):
                while len(self._fpn_features) <= idx:
                    self._fpn_features.append(None)
                self._fpn_features[idx] = out
            return hook

        try:
            # Navigate to the nn.Sequential containing the individual layers
            seq = self._det_model.model.model  # nn.Sequential of 23 layers
            layers = list(seq)
            n = len(layers)
            # Detect head inputs: layers 15, 18, 21 (P3/P4/P5 scale outputs)
            target_indices = [-8, -5, -2]  # = [n-8, n-5, n-2] for n=23 → 15,18,21
            for i, layer_idx in enumerate(target_indices):
                h = layers[layer_idx].register_forward_hook(_make_hook(i))
                self._attr_hooks.append(h)
            logger.info(
                "FPN hooks attached at layers %s.",
                [n + idx for idx in target_indices],
            )
        except Exception as e:
            logger.warning("Could not attach FPN hooks: %s", e)

    def predict(
        self,
        input_path: Union[str, Path],
        output_dir: Optional[str] = None,
        save_json: bool = True,
        save_vis: bool = True,
    ) -> Dict[str, List[ToothDetection]]:
        """Run prediction on image(s).

        Args:
            input_path: Path to an image file or directory of images.
            output_dir: Directory to save visualizations and JSON results.
            save_json:  Save per-image JSON results.
            save_vis:   Save annotated visualization images.

        Returns:
            Dict mapping image filename → list of ToothDetection results.
        """
        self._load_models()
        input_path = Path(input_path)
        all_results = {}

        # Collect image paths
        if input_path.is_dir():
            img_paths = sorted([
                p for p in input_path.glob("*")
                if p.suffix.lower() in IMG_EXTENSIONS
            ])
        elif input_path.is_file():
            img_paths = [input_path]
        else:
            logger.error("Input path does not exist: '%s'", input_path)
            return {}

        if not img_paths:
            logger.warning("No images found at '%s'.", input_path)
            return {}

        # Prepare output directory
        if output_dir:
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
        else:
            out_dir = None

        logger.info("Running inference on %d image(s) ...", len(img_paths))

        for img_path in img_paths:
            logger.info("  Processing: %s", img_path.name)

            try:
                teeth = self._predict_single(img_path)
            except Exception as e:
                logger.error("Error processing '%s': %s", img_path.name, e)
                teeth = []

            all_results[img_path.name] = teeth

            # Save outputs
            if out_dir and teeth:
                if save_json:
                    self._save_json(teeth, out_dir / f"{img_path.stem}_result.json")
                if save_vis:
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        vis = draw_teeth_detections(img, teeth)
                        cv2.imwrite(
                            str(out_dir / f"{img_path.stem}_vis.jpg"), vis
                        )

        # Print summary
        total_teeth = sum(len(v) for v in all_results.values())
        total_diseased = sum(
            sum(1 for t in v if t.diseases) for v in all_results.values()
        )
        logger.info(
            "Inference complete: %d images, %d teeth detected (%d diseased).",
            len(img_paths),
            total_teeth,
            total_diseased,
        )

        return all_results

    def _predict_single(self, img_path: Path) -> List[ToothDetection]:
        """Run full prediction pipeline on a single image.

        Returns:
            List of ToothDetection after linear sum assignment.
        """
        # Clear captured FPN features
        self._fpn_features = []

        # Run YOLOv8 detection
        det_results = self._det_model.predict(
            source=str(img_path),
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            max_det=32,
            imgsz=list(self.img_size),
            verbose=False,
            device=self.device,
        )

        if not det_results:
            return []

        result = det_results[0]

        # Run attribute heads if available
        attr_probs = None
        if self._attr_heads is not None and self._fpn_features:
            attr_probs = self._predict_attributes(result)

        # Post-process with linear sum assignment
        teeth = postprocess_yolo_output(
            results=result,
            attr_probs=attr_probs,
            conf_threshold=self.conf_threshold,
            attr_threshold=self.attr_threshold,
            img_shape=(result.orig_shape[0], result.orig_shape[1])
            if result.orig_shape
            else None,
        )

        return teeth

    def _predict_attributes(self, det_result) -> Optional[np.ndarray]:
        """Run attribute heads on captured FPN features.

        Returns:
            (N, 4) numpy array of sigmoid attribute probabilities,
            or None if attribute heads are not available.

        Two modes (set via self.attr_inference_mode):
          'per_tooth'  — Sample feature map at each detected box centre, average
                         logits across 3 FPN scales, then apply sigmoid.
                         Use for models trained with per-tooth spatial supervision
                         (trainer.py after the June 2026 fix).
          'global_avg' — Average logits over ALL spatial positions and ALL scales,
                         then apply sigmoid, then tile to every tooth.
                         Use for models trained with image-level global pooling
                         (e.g. baseline-disease-100-main).
        """
        if not self._fpn_features or self._attr_heads is None:
            return None

        try:
            with torch.no_grad():
                valid_features = [f for f in self._fpn_features if f is not None]
                if not valid_features:
                    return None

                # attr_outputs: list of 3 tensors, each (1, 4, H_f, W_f)
                attr_outputs = self._attr_heads(valid_features)

                if not attr_outputs or det_result.boxes is None:
                    return None

                N = len(det_result.boxes)
                if N == 0:
                    return None

                # ── Image-level diagnostic (always logged regardless of mode) ─────────
                img_logit = torch.stack(
                    [o.mean(dim=[2, 3]) for o in attr_outputs], dim=0
                ).mean(dim=0)  # (1, 4)
                img_prob = torch.sigmoid(img_logit)[0].cpu().numpy()  # (4,)
                logger.info(
                    "Attr image-level probs  [is_impacted, has_caries, has_deepcaries, has_lesion]: "
                    "[%.3f, %.3f, %.3f, %.3f]",
                    *img_prob.tolist(),
                )

                if self.attr_inference_mode == "global_avg":
                    # ── Mode: global average (matches old image-level training) ────────
                    # Apply sigmoid to the global average logit, then tile to N teeth
                    per_tooth_probs = np.tile(img_prob, (N, 1))
                    logger.info(
                        "Attr mode=global_avg  probs: [%.3f, %.3f, %.3f, %.3f]",
                        *img_prob.tolist(),
                    )
                else:
                    # ── Mode: per_tooth (matches per-tooth spatial training) ───────────
                    # Compute letterbox parameters used by ultralytics.predict():
                    # ultralytics resizes preserving aspect ratio (scale = min)
                    # then pads to a multiple of stride (32) — so the actual
                    # letterboxed image size depends on the input aspect ratio.
                    # We recover this by inspecting the largest feature map (P3,
                    # stride 8) which gives the letterboxed image dimensions.
                    P3_STRIDE = 8
                    H_f_p3 = attr_outputs[0].shape[2]
                    W_f_p3 = attr_outputs[0].shape[3]
                    effective_h = H_f_p3 * P3_STRIDE
                    effective_w = W_f_p3 * P3_STRIDE

                    target_h_default, target_w_default = self.img_size
                    orig_h, orig_w = (
                        det_result.orig_shape
                        if det_result.orig_shape is not None
                        else (effective_h, effective_w)
                    )

                    # Same scaling factor ultralytics uses
                    lb_scale = min(
                        target_h_default / orig_h, target_w_default / orig_w
                    )
                    new_h_unpad = orig_h * lb_scale
                    new_w_unpad = orig_w * lb_scale
                    pad_top_inf = (effective_h - new_h_unpad) / 2
                    pad_left_inf = (effective_w - new_w_unpad) / 2

                    # boxes.xywhn is normalised to ORIGINAL image dimensions.
                    # Convert each detection's centre to its anatomical pixel
                    # position in the letterboxed image, then sample features.
                    boxes_xywhn = det_result.boxes.xywhn.cpu().numpy()  # (N, 4)
                    cx_norm = boxes_xywhn[:, 0]
                    cy_norm = boxes_xywhn[:, 1]

                    cx_lb_pixel = cx_norm * orig_w * lb_scale + pad_left_inf
                    cy_lb_pixel = cy_norm * orig_h * lb_scale + pad_top_inf

                    per_tooth_logits = []
                    for i in range(N):
                        scale_samples = []
                        for feat in attr_outputs:
                            _, _, H_f, W_f = feat.shape
                            # feature_idx = pixel * (feat_size / effective_size)
                            xf = int(cx_lb_pixel[i] * W_f / effective_w)
                            yf = int(cy_lb_pixel[i] * H_f / effective_h)
                            xf = max(0, min(xf, W_f - 1))
                            yf = max(0, min(yf, H_f - 1))
                            scale_samples.append(feat[0, :, yf, xf])  # (4,) logits
                        tooth_logit = torch.stack(scale_samples, dim=0).mean(dim=0)
                        per_tooth_logits.append(tooth_logit)

                    per_tooth_logits_t = torch.stack(per_tooth_logits, dim=0)  # (N, 4)
                    per_tooth_probs = torch.sigmoid(per_tooth_logits_t).cpu().numpy()  # (N, 4)

                logger.info(
                    "Attr per-tooth probs range  min=[%.3f, %.3f, %.3f, %.3f]"
                    "  max=[%.3f, %.3f, %.3f, %.3f]",
                    *per_tooth_probs.min(axis=0).tolist(),
                    *per_tooth_probs.max(axis=0).tolist(),
                )

                return per_tooth_probs

        except Exception as e:
            logger.warning("Attribute head inference failed: %s", e)
            return None

    def evaluate(
        self,
        data_yaml: str = "config/dataset.yaml",
        split: str = "val",
    ) -> Dict[str, float]:
        """Evaluate model on validation/test set using ultralytics metrics.

        Args:
            data_yaml: Path to dataset YAML config.
            split:     Dataset split to evaluate ('val' or 'test').

        Returns:
            Dict of metric names → values.
        """
        self._load_models()

        logger.info("Running evaluation on '%s' split ...", split)

        metrics = self._det_model.val(
            data=data_yaml,
            split=split,
            imgsz=list(self.img_size),
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            max_det=32,
            device=self.device,
            verbose=True,
        )

        # Extract key metrics
        result = {}
        if hasattr(metrics, "box"):
            result["mAP50"] = float(metrics.box.map50)
            result["mAP50-95"] = float(metrics.box.map)
            logger.info("mAP@0.5 = %.4f", result["mAP50"])
            logger.info("mAP@0.5:0.95 = %.4f", result["mAP50-95"])

        return result

    @staticmethod
    def _save_json(teeth: List[ToothDetection], path: Path) -> None:
        """Save detection results as JSON."""
        summary = summarize_detections(teeth)
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
