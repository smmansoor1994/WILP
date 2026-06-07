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

        # Skip Phase 1 if weights already exist (e.g. second call in full pipeline
        # after pseudo-labeling). Only re-run if explicitly resuming or weights missing.
        if not phase1_weights.exists():
            logger.info("=" * 60)
            logger.info("PHASE 1: Pre-training on quadrant + enumeration data")
            logger.info("(No disease attribute loss — data_type 0 and 1 only)")
            logger.info("=" * 60)
            self._train_phase1()
        else:
            logger.info("Phase 1 weights found at '%s'. Skipping Phase 1.", phase1_weights)

        # ── Phase 2: Full training on all data (with disease attributes) ───────
        logger.info("=" * 60)
        logger.info("PHASE 2: Full training with disease attribute heads")
        logger.info("(Hierarchical loss: bbox + class + attribute)")
        logger.info("=" * 60)
        self._train_phase2(pretrain_weights=str(phase1_weights))

        logger.info("Training complete. Best weights saved to '%s'.", self.save_dir)

    def train_phase1_only(self) -> None:
        """Run Phase 1 only (detection pre-training). Used by the full pipeline
        before pseudo-labeling so Phase 2 can use pseudo-labels on the second call.
        Phase 2 is intentionally skipped here.
        """
        phase1_weights = self.save_dir / "phase1" / "weights" / "best.pt"
        if phase1_weights.exists():
            logger.info("Phase 1 weights already exist at '%s'. Skipping.", phase1_weights)
            return
        logger.info("=" * 60)
        logger.info("PHASE 1 ONLY: Pre-training on quadrant + enumeration data")
        logger.info("=" * 60)
        self._train_phase1()
        self._promote_weights(phase=1)

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

        self._promote_weights(phase=1)

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

        self._promote_weights(phase=2)

        # Copy final weights to top-level weights/
        final_dst = self.project_root / "weights" / "yolortho_best.pt"
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

        # Use extended 10-column labels (labels_ext/) for attribute head training.
        # Pseudo labels_ext is created by pseudo_label.py (must re-run if missing).
        # Fall back to processed/labels_ext if pseudo/labels_ext is absent or empty.
        labels_ext_dir = data_root / "labels_ext" / "train"
        if data_root == pseudo_root and (
            not labels_ext_dir.exists()
            or not any(labels_ext_dir.glob("*.txt"))
        ):
            logger.warning(
                "pseudo/labels_ext/train/ is missing or empty. "
                "Falling back to processed/labels_ext/train/ for attr head training. "
                "Re-run pseudo labeling to fix this."
            )
            labels_ext_dir = processed_root / "labels_ext" / "train"
            # Also align images to processed so filenames match
            images_dir = processed_root / "images" / "train"
        else:
            images_dir = data_root / "images" / "train"

        dataset = YOLOrthoDataset(
            images_dir=images_dir,
            labels_dir=labels_ext_dir,
            img_size=(self.train_cfg.get("input_height", 640),
                      self.train_cfg.get("input_width", 1280)),
        )
        # Cap workers to avoid Colab/low-CPU warnings (system may only support 2)
        import os
        max_workers = min(self.train_cfg.get("workers", 4), os.cpu_count() or 2, 4)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.train_cfg.get("batch_size", 8),
            shuffle=True,
            num_workers=max_workers,
            collate_fn=YOLOrthoDataset.collate_fn,
        )

        optimizer = AdamW(model.attr_heads.parameters(), lr=1e-3, weight_decay=1e-4)
        attr_loss_fn = AttributeBCELoss(num_attrs=4, loss_weight=8.0)
        # Scheduler initialized after first optimizer.step() to avoid PyTorch warning
        # about scheduler.step() being called before optimizer.step()
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
        optimizer_has_stepped = False

        # Set train mode only on pure nn.Module components, bypassing the
        # ultralytics YOLO wrapper whose .train() method is overridden and
        # does not accept the bool `mode` argument that PyTorch passes internally.
        model.base_model.model.train()
        model.attr_heads.train()
        best_loss = float("inf")
        # Save attr_best.pt both alongside phase2 weights and in top-level weights/
        best_path = self.save_dir / "phase2" / "weights" / "attr_best.pt"
        weights_attr_path = self.project_root / "weights" / "attr_best.pt"
        weights_attr_path.parent.mkdir(exist_ok=True)

        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0

            for batch_imgs, batch_labels in loader:
                batch_imgs = batch_imgs.to(device)
                # batch_labels: list of (N_i, 10) tensors

                optimizer.zero_grad()

                # Step 1: Run frozen backbone to populate FPN hooks (no gradients needed)
                model._fpn_features = []
                with torch.no_grad():
                    model.base_model.model(batch_imgs)

                # Step 2: Detach FPN features (backbone is frozen; attrs must be separate)
                fpn_features = [f.detach() for f in model._fpn_features if f is not None]
                if not fpn_features:
                    continue

                # Step 3: Run attr_heads OUTSIDE no_grad so gradients flow for training
                attr_outputs = model.attr_heads(fpn_features)

                if not attr_outputs:
                    continue

                # Pool all 3 scales spatially → (B, 4), then average across scales
                # attr_outputs: list of 3 tensors each (B, num_attrs, H, W)
                pooled_pred = torch.stack(
                    [o.mean(dim=[2, 3]) for o in attr_outputs], dim=0
                ).mean(dim=0)  # (B, 4)

                # Build per-image supervision from labels
                # batch_labels: list of B tensors each (N_teeth, 10)
                all_preds = []
                all_targets = []
                all_dtypes = []

                for img_idx, lbl in enumerate(batch_labels):
                    if lbl is None or len(lbl) == 0 or img_idx >= pooled_pred.shape[0]:
                        continue
                    lbl = lbl.to(device)
                    attrs = lbl[:, 5:9]        # (N_teeth, 4) disease flags
                    data_types = lbl[:, 9].long()

                    # Image-level supervision: tooth is diseased if any annotation says so
                    img_attrs = attrs.max(dim=0).values.unsqueeze(0).float()  # (1, 4)
                    img_dtype = data_types.max().unsqueeze(0)                  # (1,)

                    all_preds.append(pooled_pred[img_idx : img_idx + 1])       # (1, 4)
                    all_targets.append(img_attrs)
                    all_dtypes.append(img_dtype)

                if not all_preds:
                    continue

                pred_tensor = torch.cat(all_preds, dim=0)
                tgt_tensor = torch.cat(all_targets, dim=0).to(device)
                dt_tensor = torch.cat(all_dtypes, dim=0).to(device)

                loss = attr_loss_fn(pred_tensor, tgt_tensor, dt_tensor)
                loss.backward()
                optimizer.step()
                optimizer_has_stepped = True

                epoch_loss += loss.item()
                n_batches += 1

            # Only step scheduler after optimizer has been called at least once
            if optimizer_has_stepped:
                scheduler.step()
            avg_loss = epoch_loss / max(n_batches, 1)
            logger.info(
                "Attr head epoch [%d/%d]  loss=%.4f", epoch + 1, epochs, avg_loss
            )

            if avg_loss < best_loss:
                best_loss = avg_loss
                ckpt = {
                    "epoch": epoch,
                    "attr_heads_state": model.attr_heads.state_dict(),
                    "loss": best_loss,
                }
                torch.save(ckpt, best_path)
                torch.save(ckpt, weights_attr_path)

        logger.info("Attribute head training complete. Best loss: %.4f", best_loss)
        logger.info("attr_best.pt saved to '%s'.", weights_attr_path)

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
        if phase == 2 and (self.project_root / "data" / "pseudo" / "images").exists():
            # Update dataset.yaml to point to pseudo-labeled data
            data_yaml = self._create_pseudo_dataset_yaml()
        else:
            data_yaml = str(self.project_root / "config" / "dataset.yaml")

        # Use phase-specific epoch count if defined, else fall back to shared 'epochs'
        default_epochs = t.get("epochs", 200)
        if phase == 1:
            num_epochs = t.get("phase1_epochs", default_epochs)
        else:
            num_epochs = t.get("phase2_epochs", default_epochs)

        return dict(
            data=data_yaml,
            epochs=num_epochs,
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

    def _promote_weights(self, phase: int) -> None:
        """After a training phase completes:
        - Copy best.pt and last.pt to top-level weights/ with phase prefix.
        - Move epoch*.pt files out of outputs/runs/ into epoch_checkpoints/.

        Args:
            phase: 1 or 2.
        """
        phase_name = f"phase{phase}"
        phase_weights_dir = self.save_dir / phase_name / "weights"
        dest_weights = self.project_root / "weights"
        dest_epochs = self.project_root / "epoch_checkpoints" / phase_name
        dest_weights.mkdir(parents=True, exist_ok=True)
        dest_epochs.mkdir(parents=True, exist_ok=True)

        if not phase_weights_dir.exists():
            logger.warning("Phase %d weights dir not found: '%s'", phase, phase_weights_dir)
            return

        for pt_file in phase_weights_dir.glob("*.pt"):
            if pt_file.name.startswith("epoch"):
                # Move intermediate epoch checkpoints out of outputs/
                shutil.move(str(pt_file), str(dest_epochs / pt_file.name))
                logger.info("Moved %s → epoch_checkpoints/%s/", pt_file.name, phase_name)
            elif pt_file.name in ("best.pt", "last.pt"):
                # Copy valued weights to weights/ with phase prefix
                dst_name = f"{phase_name}_{pt_file.name}"
                shutil.copy2(pt_file, dest_weights / dst_name)
                logger.info("Copied %s → weights/%s", pt_file.name, dst_name)

    def _create_pseudo_dataset_yaml(self) -> str:
        """Create a dataset.yaml pointing to the pseudo-labeled data directory."""
        import yaml as pyyaml

        # Load original dataset config using absolute path
        orig_yaml_path = self.project_root / "config" / "dataset.yaml"
        with open(orig_yaml_path) as f:
            orig = pyyaml.safe_load(f)

        # Pseudo labeling only produces a 'train' split.
        # Use absolute paths so ultralytics resolves them correctly regardless of CWD.
        pseudo_yaml = dict(orig)
        pseudo_train = str(self.project_root / "data" / "pseudo" / "images" / "train")
        processed_val = str(self.project_root / "data" / "processed" / "images" / "val")
        # Override path/train/val with absolute values; drop relative 'path' key
        pseudo_yaml["path"] = str(self.project_root)
        pseudo_yaml["train"] = pseudo_train
        pseudo_yaml["val"] = processed_val
        pseudo_yaml["test"] = str(self.project_root / "data" / "processed" / "images" / "test")

        out_path = self.project_root / "config" / "dataset_pseudo.yaml"
        with open(out_path, "w") as f:
            pyyaml.dump(pseudo_yaml, f, default_flow_style=False)

        return str(out_path)
