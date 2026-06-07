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
    """

    def __init__(
        self,
        weights_path: str,
        device: str = "cuda",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        attr_threshold: float = 0.5,
        img_size: tuple = (640, 1280),
        attr_weights_path: Optional[str] = None,
    ):
        self.weights_path = Path(weights_path)
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.attr_threshold = attr_threshold
        self.img_size = img_size
        self.attr_weights_path = attr_weights_path

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

            # Attach hooks to capture FPN features
            self._attach_fpn_hooks()
            logger.info("Attribute heads loaded from '%s'.", attr_path)
        except Exception as e:
            logger.warning(
                "Could not load attribute heads: %s. Disease attributes will be unavailable.", e
            )

    def _attach_fpn_hooks(self) -> None:
        """Attach hooks to capture intermediate FPN feature maps."""
        self._fpn_features = []

        def _make_hook(idx):
            def hook(module, inp, out):
                while len(self._fpn_features) <= idx:
                    self._fpn_features.append(None)
                self._fpn_features[idx] = out
            return hook

        # Attach to the 3 neck output layers before Detect head
        try:
            inner = self._det_model.model
            layers = list(inner.children())
            for i, layer_idx in enumerate([-4, -3, -2]):
                h = layers[layer_idx].register_forward_hook(_make_hook(i))
                self._attr_hooks.append(h)
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
        """
        if not self._fpn_features or self._attr_heads is None:
            return None

        try:
            with torch.no_grad():
                valid_features = [f for f in self._fpn_features if f is not None]
                if not valid_features:
                    return None

                # Get attribute predictions per scale
                attr_outputs = self._attr_heads(valid_features)

                if not attr_outputs or det_result.boxes is None:
                    return None

                N = len(det_result.boxes)
                # Use first scale (highest resolution) averaged over spatial dims
                attr_map = torch.sigmoid(attr_outputs[0])  # (B, 4, H, W)
                attr_avg = attr_map.mean(dim=[2, 3])       # (B, 4)

                # Expand to one row per detection
                attr_np = attr_avg[0].cpu().numpy()        # (4,)
                # Broadcast across all detections (simplified; full version matches per anchor)
                return np.tile(attr_np, (N, 1))

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
