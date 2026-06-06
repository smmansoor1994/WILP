"""
src/training/trainer.py
=======================
YOLOrtho Training Pipeline.

Training strategy (from the paper, Section 2):
  Phase 1 — Pre-train on Part 1 + Part 2 data (no disease labels):
    • Full bbox + class loss, no attribute loss
    • Saves a "phase1" checkpoint used for pseudo labeling

  Phase 2 — Train on ALL data (Part 1 + 2 + 3 + pseudo-labeled):
    • Full hierarchical loss (bbox + class + attribute)
    • Loads Phase 1 weights as starting point

This trainer wraps ultralytics' YOLO.train() but adds:
  - Custom attribute head training loop on top of standard detection training
  - Hierarchical loss masking via data_type field in labels
  - Proper logging of all loss components

Usage:
    from src.training.trainer import YOLOrthoTrainer
    trainer = YOLOrthoTrainer("config/train_config.yaml")
    trainer.train()
"""

import logging
import shutil
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import yaml

logger = logging.getLogger(__name__)


class YOLOrthoTrainer:
    """Two-phase YOLOrtho trainer.

    Args:
        config_path: Path to train_config.yaml.
        resume:      Resume training from last checkpoint.
        device:      Torch device string ('cuda' or 'cpu').
    """

    def __init__(
        self,
        config_path: str = "config/train_config.yaml",
        resume: bool = False,
        device: str = "cuda",
    ):
        self.config_path = Path(config_path)
        self.resume = resume
        self.device = device

        # Load training configuration
        with open(self.config_path) as f:
            self.cfg = yaml.safe_load(f)

        self.train_cfg = self.cfg.get("training", {})
        self.aug_cfg = self.cfg.get("augmentation", {})

        # Project root: two levels up from config/train_config.yaml → WILP/
        self.project_root = self.config_path.resolve().parent.parent

        # Output directory — always absolute so ultralytics doesn't nest it
        # inside its own default 'runs/detect/' folder.
        save_dir_rel = self.train_cfg.get("save_dir", "outputs/runs")
        self.save_dir = (self.project_root / save_dir_rel).resolve()
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def train(self) -> None:
        """Run the full two-phase training pipeline."""

        # ── Phase 1: Pre-train on Part 1+2 (detection only, no disease) ───────
        phase1_weights = self.save_dir / "phase1" / "weights" / "best.pt"

        if not phase1_weights.exists() or self.resume is False:
            logger.info("=" * 60)
            logger.info("PHASE 1: Pre-training on quadrant + enumeration data")
            logger.info("(No disease attribute loss — data_type 0 and 1 only)")
            logger.info("=" * 60)
            self._train_phase1()
        else:
            logger.info("Phase 1 weights found at '%s'. Skipping.", phase1_weights)

        # ── Phase 2: Full training on all data (with disease attributes) ───────
        logger.info("=" * 60)
        logger.info("PHASE 2: Full training with disease attribute heads")
        logger.info("(Hierarchical loss: bbox + class + attribute)")
        logger.info("=" * 60)
        self._train_phase2(pretrain_weights=str(phase1_weights))

        logger.info("Training complete. Best weights saved to '%s'.", self.save_dir)

    def _train_phase1(self) -> None:
        """Phase 1: standard YOLOv8 training on Parts 1+2 only.

        This gives us a good tooth detector that we can use for:
        1. Pseudo-labeling healthy teeth in Part 3
        2. Initializing Phase 2 training

        Uses ultralytics YOLO.train() directly — the attribute heads
        are not trained yet in this phase.
        """
        from ultralytics import YOLO

        # NOTE: CoordConv is NOT injected here.
        # Ultralytics rebuilds the model internally from its yaml config inside
        # model.train(), so any pre-injection causes a KeyError when it tries to
        # load the CoordConv state dict (model.0.conv.conv.weight) into a fresh
        # standard model that only has model.0.conv.weight.
        # CoordConv is applied in the custom Phase 2b attribute-head loop.
        logger.info("Loading YOLOv8x pretrained weights ...")
        model = YOLO("yolov8x.pt")

        # Build training arguments for ultralytics
        train_args = self._build_ultralytics_args(
            phase=1,
            project=str(self.save_dir),
            name="phase1",
        )

        logger.info("Starting Phase 1 training ...")
        model.train(**train_args)

        best = self.save_dir / "phase1" / "weights" / "best.pt"
        if best.exists():
            logger.info("Phase 1 best weights: '%s'", best)
        else:
            logger.warning("Phase 1 best weights not found at expected path.")

    def _train_phase2(self, pretrain_weights: str) -> None:
        """Phase 2: fine-tune with disease attribute heads on all data.

        Loads Phase 1 weights, then runs a custom training loop that adds
        the attribute BCE loss on top of the standard detection loss.

        Since ultralytics' built-in trainer doesn't know about our custom
        attribute heads, we use a two-stage approach:
          a) Let ultralytics train the detection head normally.
          b) After each epoch, run an attribute head fine-tuning pass.

        In practice, for a full custom training loop, we wrap ultralytics
        using a subclassed Trainer. Here we use the simpler approach of
        running standard training first, then fine-tuning attribute heads.
        """
        from ultralytics import YOLO

        # Load Phase 1 weights
        # NOTE: CoordConv is NOT injected here for the same reason as Phase 1 —
        # Ultralytics rebuilds the model internally, causing a state-dict key
        # mismatch if CoordConv is pre-injected. CoordConv is applied in Phase 2b.
        if Path(pretrain_weights).exists():
            logger.info("Loading Phase 1 weights from '%s'", pretrain_weights)
            model = YOLO(pretrain_weights)
        else:
            logger.warning(
                "Phase 1 weights not found. Starting Phase 2 from pretrained COCO weights."
            )
            model = YOLO("yolov8x.pt")

        # Build training arguments for Phase 2 (full dataset)
        train_args = self._build_ultralytics_args(
            phase=2,
            project=str(self.save_dir),
            name="phase2",
        )

        logger.info("Starting Phase 2 training (detection head) ...")
        model.train(**train_args)

        # ── Phase 2b: Train attribute heads ───────────────────────────────────
        logger.info("Starting Phase 2b: Training attribute heads ...")
        phase2_best = self.save_dir / "phase2" / "weights" / "best.pt"
        self._train_attribute_heads(
            base_weights=str(phase2_best) if phase2_best.exists() else pretrain_weights
        )

        # Copy final weights to top-level weights/
        final_dst = Path("weights") / "yolortho_best.pt"
        final_dst.parent.mkdir(exist_ok=True)
        if phase2_best.exists():
            shutil.copy2(phase2_best, final_dst)
            logger.info("Final weights saved to '%s'.", final_dst)

    def _train_attribute_heads(self, base_weights: str) -> None:
        """Fine-tune only the attribute heads with the detection backbone frozen.

        The detection backbone is kept frozen; only attribute head weights are updated.
        This decouples disease classification from tooth localization.

        Args:
            base_weights: Path to trained detection model weights.
        """
        import torch
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import CosineAnnealingLR
        from src.models.yolortho import build_yolortho, ATTRIBUTE_NAMES
        from src.training.loss import AttributeBCELoss
        from src.data.dataset import YOLOrthoDataset

        device = self.device
        epochs = max(20, self.train_cfg.get("epochs", 200) // 5)

        logger.info(
            "Attribute head fine-tuning: %d epochs, device=%s", epochs, device
        )

        # Build model
        model = build_yolortho(
            num_classes=32,
            base_weights=base_weights,
            use_coordconv=True,
            device=device,
            verbose=False,
        )

        # Freeze detection backbone — only train attribute heads
        for param in model.base_model.parameters():
            param.requires_grad = False
        for param in model.attr_heads.parameters():
            param.requires_grad = True

        n_attr_params = sum(p.numel() for p in model.attr_heads.parameters())
        logger.info("Attribute head parameters: %s", f"{n_attr_params:,}")

        # Data — use pseudo-labeled data if available, else processed
        # Use absolute paths anchored to project root to avoid CWD-dependent resolution
        pseudo_root = self.project_root / "data" / "pseudo"
        processed_root = self.project_root / "data" / "processed"
        data_root = pseudo_root if (pseudo_root / "images").exists() else processed_root
        # Use extended 10-column labels (labels_ext/) for attribute head training
        dataset = YOLOrthoDataset(
            images_dir=data_root / "images" / "train",
            labels_dir=data_root / "labels_ext" / "train",
            img_size=(self.train_cfg.get("input_height", 640),
                      self.train_cfg.get("input_width", 1280)),
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.train_cfg.get("batch_size", 8),
            shuffle=True,
            num_workers=self.train_cfg.get("workers", 4),
            collate_fn=YOLOrthoDataset.collate_fn,
        )

        optimizer = AdamW(model.attr_heads.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
        attr_loss_fn = AttributeBCELoss(num_attrs=4, loss_weight=8.0)

        # Set train mode only on pure nn.Module components, bypassing the
        # ultralytics YOLO wrapper whose .train() method is overridden and
        # does not accept the bool `mode` argument that PyTorch passes internally.
        model.base_model.model.train()
        model.attr_heads.train()
        best_loss = float("inf")
        best_path = self.save_dir / "phase2" / "weights" / "attr_best.pt"

        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0

            for batch_imgs, batch_labels in loader:
                batch_imgs = batch_imgs.to(device)
                # batch_labels: list of (N_i, 10) tensors

                optimizer.zero_grad()

                # Extract FPN features by running backbone
                with torch.no_grad():
                    _, attr_outputs = model(batch_imgs)

                # Flatten attribute predictions and labels for loss computation
                all_preds = []
                all_targets = []
                all_dtypes = []

                for scale_pred, lbl in zip(attr_outputs, batch_labels):
                    if lbl is None or len(lbl) == 0:
                        continue
                    lbl = lbl.to(device)
                    B, num_attrs, H, W = scale_pred.shape
                    # Flatten spatial dims: (B, num_attrs, H*W) → (B*H*W, num_attrs)
                    flat_pred = scale_pred.permute(0, 2, 3, 1).reshape(-1, num_attrs)
                    # For now, repeat each label across spatial positions (simplified)
                    # In full implementation, we match predictions to GT anchors
                    data_types = lbl[:, 9].long()
                    attrs = lbl[:, 5:9]
                    # Use mean-pooled prediction for label-level supervision (simplified)
                    pooled_pred = scale_pred.mean(dim=[2, 3])  # (B, num_attrs)
                    all_preds.append(pooled_pred[:len(lbl)])
                    all_targets.append(attrs)
                    all_dtypes.append(data_types)
                    break  # Use first scale only for attribute head training

                if not all_preds:
                    continue

                pred_tensor = torch.cat(all_preds, dim=0)
                tgt_tensor = torch.cat(all_targets, dim=0).to(device)
                dt_tensor = torch.cat(all_dtypes, dim=0).to(device)

                loss = attr_loss_fn(pred_tensor, tgt_tensor, dt_tensor)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()
            avg_loss = epoch_loss / max(n_batches, 1)
            logger.info(
                "Attr head epoch [%d/%d]  loss=%.4f", epoch + 1, epochs, avg_loss
            )

            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(
                    {
                        "epoch": epoch,
                        "attr_heads_state": model.attr_heads.state_dict(),
                        "loss": best_loss,
                    },
                    best_path,
                )

        logger.info("Attribute head training complete. Best loss: %.4f", best_loss)

    def _build_ultralytics_args(
        self, phase: int, project: str, name: str
    ) -> Dict[str, Any]:
        """Build kwargs for ultralytics YOLO.train().

        Args:
            phase:   1 (detection pre-train) or 2 (full training).
            project: Output project directory.
            name:    Run name.

        Returns:
            dict of ultralytics training arguments.
        """
        t = self.train_cfg
        a = self.aug_cfg

        # Dataset: use pseudo-labeled data for phase 2 if available
        if phase == 2 and Path("data/pseudo/images").exists():
            # Update dataset.yaml to point to pseudo-labeled data
            data_yaml = self._create_pseudo_dataset_yaml()
        else:
            data_yaml = str(self.project_root / "config" / "dataset.yaml")

        return dict(
            data=data_yaml,
            epochs=t.get("epochs", 200),
            imgsz=[t.get("input_height", 640), t.get("input_width", 1280)],
            batch=t.get("batch_size", 8),
            workers=t.get("workers", 4),
            device=self.device,
            optimizer=t.get("optimizer", "SGD"),
            lr0=t.get("lr0", 0.01),
            lrf=t.get("lrf", 0.01),
            momentum=t.get("momentum", 0.937),
            weight_decay=t.get("weight_decay", 0.0005),
            warmup_epochs=t.get("warmup_epochs", 3.0),
            warmup_momentum=t.get("warmup_momentum", 0.8),
            warmup_bias_lr=t.get("warmup_bias_lr", 0.1),
            box=t.get("loss_box", 7.5),
            cls=t.get("loss_cls", 0.5),
            dfl=t.get("loss_dfl", 1.5),
            project=project,
            name=name,
            exist_ok=True,
            patience=t.get("patience", 50),
            save=True,
            save_period=10,
            seed=t.get("seed", 42),
            # Augmentation
            hsv_h=a.get("hsv_h", 0.015),
            hsv_s=a.get("hsv_s", 0.0),
            hsv_v=a.get("hsv_v", 0.4),
            degrees=a.get("degrees", 5.0),
            translate=a.get("translate", 0.1),
            scale=a.get("scale", 0.5),
            shear=a.get("shear", 2.0),
            flipud=a.get("flipud", 0.0),
            fliplr=a.get("fliplr", 0.5),   # Note: quadrant remapping is our custom aug
            mosaic=a.get("mosaic", 0.0),
            mixup=a.get("mixup", 0.0),
            # Resume
            resume=self.resume if phase == 2 else False,
        )

    def _create_pseudo_dataset_yaml(self) -> str:
        """Create a dataset.yaml pointing to the pseudo-labeled data directory."""
        import yaml as pyyaml

        # Load original dataset config
        with open("config/dataset.yaml") as f:
            orig = pyyaml.safe_load(f)

        # Update paths to point to pseudo directory
        pseudo_yaml = dict(orig)
        pseudo_yaml["path"] = "../data/pseudo"

        out_path = Path("config/dataset_pseudo.yaml")
        with open(out_path, "w") as f:
            pyyaml.dump(pseudo_yaml, f, default_flow_style=False)

        return str(out_path)
