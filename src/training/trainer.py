"""
src/training/trainer.py
=======================
ARCHON Training Pipeline.

Contains two trainers:
  ARCHONTrainer      — Baseline two-phase training (paper arXiv:2308.05967)
  ARCHONHybridTrainer — ARCHON Phase 3: trains GlobalContextEncoder +
                          MultiScaleFusion + HybridMultiTaskHead on frozen backbone.

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
    from src.training.trainer import ARCHONTrainer
    trainer = ARCHONTrainer("config/train_config.yaml")
    trainer.train()
"""

import logging
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List

import torch
import yaml

logger = logging.getLogger(__name__)


class ARCHONTrainer:
    """Two-phase ARCHON trainer.

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

    def train_attr_only(self, base_weights: Optional[str] = None) -> None:
        """Re-run Phase 2b (attribute head training) only, using existing backbone weights.

        Use this to retrain the attribute heads after a crash or when the existing
        attr_best.pt was trained with the no_grad bug (all predictions = healthy).

        Args:
            base_weights: Path to the detection backbone weights to freeze during
                          attribute head training. Defaults to the phase2 best.pt,
                          falling back to phase1 best.pt.
        """
        if base_weights is None:
            # Prefer phase2 best, fall back to phase1
            candidates = [
                self.save_dir / "phase2" / "weights" / "best.pt",
                self.save_dir / "phase1" / "weights" / "best.pt",
                self.project_root / "weights" / "archon_best.pt",
            ]
            base_weights_path = next((p for p in candidates if p.exists()), None)
            if base_weights_path is None:
                raise FileNotFoundError(
                    "No backbone weights found. Tried: "
                    + ", ".join(str(p) for p in candidates)
                    + "\nProvide --weights or run --mode train first."
                )
            base_weights = str(base_weights_path)
        logger.info("=" * 60)
        logger.info("PHASE 2b ONLY: Training attribute heads")
        logger.info("Backbone: %s", base_weights)
        logger.info("=" * 60)
        self._train_attribute_heads(base_weights=base_weights)

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
        final_dst = self.project_root / "weights" / "archon_best.pt"
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
        from src.models.yolortho import build_archon_base, ATTRIBUTE_NAMES
        from src.training.loss import AttributeBCELoss
        from src.data.dataset import ARCHONDataset

        device = self.device
        # Phase 2b epochs: use dedicated key > fall back to epochs//5 > minimum 20
        epochs = int(
            self.train_cfg.get(
                "phase2b_epochs",
                max(20, self.train_cfg.get("epochs", 200) // 5),
            )
        )

        logger.info(
            "Attribute head fine-tuning: %d epochs, device=%s", epochs, device
        )

        # Build model
        model = build_archon_base(
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

        dataset = ARCHONDataset(
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
            collate_fn=ARCHONDataset.collate_fn,
        )

        optimizer = AdamW(model.attr_heads.parameters(), lr=1e-3, weight_decay=1e-4)

        # Compute per-attribute pos_weight from dataset to counteract class imbalance.
        # Most teeth are healthy (label=0); without weighting, BCE drives the model to
        # always predict 0 (trivial solution, best_loss → 0).  pos_weight = neg/pos
        # ensures each positive (diseased) sample receives proportionally more gradient.
        # Use config override if provided, else compute from data.
        _pw_cfg = self.train_cfg.get("attr_pos_weight", None)
        if _pw_cfg and len(_pw_cfg) == 4:
            pos_weight = [float(v) for v in _pw_cfg]
            logger.info("attr pos_weight from config: %s", pos_weight)
        else:
            pos_weight = _compute_attr_pos_weight(dataset)
            logger.info("attr pos_weight computed from dataset: %s", pos_weight)

        attr_loss_fn = AttributeBCELoss(num_attrs=4, loss_weight=8.0, pos_weight=pos_weight)
        attr_loss_fn.to(device)
        # Scheduler initialized after first optimizer.step() to avoid PyTorch warning
        # about scheduler.step() being called before optimizer.step()
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
        optimizer_has_stepped = False

        # Keep the frozen backbone in EVAL mode so that BatchNorm layers use
        # their accumulated running_mean/running_var (same statistics used at
        # inference time).  If the backbone were in train() mode its BN layers
        # would normalise using per-batch statistics, producing FPN features that
        # differ from inference features → attr heads learn wrong thresholds.
        # Only the trainable attribute heads are put into train mode.
        model.base_model.model.eval()
        model.attr_heads.train()
        best_loss = float("inf")
        # Save attr_best.pt both alongside phase2 weights and in top-level weights/
        best_path = self.save_dir / "phase2" / "weights" / "attr_best.pt"
        # Ensure the directory exists — it may be absent when running --mode train_attr
        # on a machine where Phase 2 detection training hasn't written files yet
        # (e.g. using pre-downloaded weights from Colab).
        best_path.parent.mkdir(parents=True, exist_ok=True)
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

                # ── Per-tooth supervision (matches inference spatial sampling) ──────
                # Training previously used global spatial average per image → image-level
                # labels, which caused the model to predict "any tooth in image has
                # disease X" rather than "this specific tooth has disease X".
                #
                # Fix: sample attr features at each GT tooth's centre in the feature
                # map, exactly as inference does at each detected box centre.
                # attr_outputs: list of 3 tensors, each (B, 4, H_f, W_f) — logits

                all_preds = []
                all_targets = []
                all_dtypes = []

                for img_idx, lbl in enumerate(batch_labels):
                    if lbl is None or len(lbl) == 0:
                        continue
                    lbl = lbl.to(device)
                    cx = lbl[:, 1]              # (N_teeth,) normalised centre-x [0,1]
                    cy = lbl[:, 2]              # (N_teeth,) normalised centre-y [0,1]
                    attrs = lbl[:, 5:9].float() # (N_teeth, 4) binary disease flags
                    data_types = lbl[:, 9].long()

                    # Sample each scale at each GT tooth's location
                    scale_samples = []
                    for feat in attr_outputs:
                        # feat: (B, 4, H_f, W_f)
                        _, _, H_f, W_f = feat.shape
                        xf = (cx * W_f).long().clamp(0, W_f - 1)   # (N_teeth,)
                        yf = (cy * H_f).long().clamp(0, H_f - 1)   # (N_teeth,)
                        # feat[img_idx, :, yf, xf] → (4, N_teeth) → transpose → (N_teeth, 4)
                        sampled = feat[img_idx, :, yf, xf].T        # (N_teeth, 4)
                        scale_samples.append(sampled)

                    # Average logits across scales → (N_teeth, 4) — matches inference
                    tooth_logits = torch.stack(scale_samples, dim=0).mean(dim=0)

                    all_preds.append(tooth_logits)   # (N_teeth, 4)
                    all_targets.append(attrs)        # (N_teeth, 4)
                    all_dtypes.append(data_types)    # (N_teeth,)

                if not all_preds:
                    continue

                pred_tensor = torch.cat(all_preds, dim=0)             # (total_teeth, 4)
                tgt_tensor = torch.cat(all_targets, dim=0).to(device) # (total_teeth, 4)
                dt_tensor = torch.cat(all_dtypes, dim=0).to(device)   # (total_teeth,)

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


# ─── Module-level helpers ─────────────────────────────────────────────────────

def _compute_attr_pos_weight(dataset) -> List[float]:
    """Compute per-attribute positive-class weights from dataset label statistics.

    For each of the 4 binary disease attributes (impacted, caries, deepcaries, lesion),
    counts positives and negatives across all data_type==2 samples and returns
    neg_count / pos_count (clamped to [1.0, 20.0]) as the pos_weight.

    A pos_weight of W means a false negative (missed disease) is penalised W× more
    than a false positive, counteracting the class imbalance where most teeth are
    healthy (label=0).

    Falls back to [5, 3, 8, 5] if dataset scan fails or has no disease samples.
    """
    FALLBACK = [5.0, 3.0, 8.0, 5.0]
    try:
        import numpy as np
        pos_counts = np.zeros(4, dtype=np.float64)
        neg_counts = np.zeros(4, dtype=np.float64)
        for img_path in dataset.image_paths:
            lbl_path = dataset.labels_dir / (img_path.stem + ".txt")
            if not lbl_path.exists():
                continue
            if lbl_path.stat().st_size == 0:
                continue  # healthy tooth — no labels, not an error
            rows = np.loadtxt(str(lbl_path), ndmin=2)
            if rows.shape[1] < 10:
                continue
            data_types = rows[:, 9]
            mask = data_types == 2
            if not mask.any():
                continue
            attrs = rows[mask, 5:9]   # (N, 4) — impacted, caries, deepcaries, lesion
            pos_counts += (attrs == 1).sum(axis=0)
            neg_counts += (attrs == 0).sum(axis=0)

        total = pos_counts + neg_counts
        if total.sum() == 0:
            logger.warning("No data_type==2 samples found; using fallback pos_weight=%s", FALLBACK)
            return FALLBACK

        pw = []
        for i, (p, n) in enumerate(zip(pos_counts, neg_counts)):
            if p == 0:
                pw.append(FALLBACK[i])   # attribute never seen → use fallback
            else:
                pw.append(float(np.clip(n / p, 1.0, 20.0)))
        logger.info(
            "Disease label counts  pos=%s  neg=%s  → pos_weight=%s",
            pos_counts.astype(int).tolist(),
            neg_counts.astype(int).tolist(),
            [round(v, 2) for v in pw],
        )
        return pw
    except Exception as exc:
        logger.warning("pos_weight computation failed (%s); using fallback %s", exc, FALLBACK)
        return FALLBACK


# ─── Hybrid Trainer ───────────────────────────────────────────────────────────

class ARCHONHybridTrainer(ARCHONTrainer):
    """Extended trainer for ARCHONModel.

    Adds Phase 3 — hybrid component training on top of the baseline two-phase
    pipeline. Only the new components (Swin encoder, cross-attention fusion,
    hybrid head) are trained; the YOLOv8 backbone + detection head remain frozen.

    Phase 3 training strategy:
      - Load Phase 2 detection best.pt as backbone.
      - Inject CoordConv, attach Swin + CrossAttention + HybridHead.
      - Freeze backbone + binary attribute heads.
      - Train: Swin encoder + cross-attention fusion + HybridMultiTaskHead.
      - Loss: AttributeBCELoss (binary, for compat) + SeverityLoss + QuadrantAuxLoss.

    Usage:
        trainer = ARCHONHybridTrainer("config/train_config.yaml")
        trainer.train()          # runs phases 1 + 2 + 3
        trainer.train_hybrid()   # phase 3 only (backbone already trained)
    """

    def train(self) -> None:
        """Run full pipeline: phase 1 + 2 (baseline) + phase 3 (hybrid)."""
        super().train()  # runs baseline Phase 1 + 2 + 2b
        self.train_hybrid()

    def train_hybrid(self, base_weights: Optional[str] = None) -> None:
        """Phase 3: Train hybrid components (Swin + CrossAttn + HybridHead).

        Args:
            base_weights: Detection backbone weights. Defaults to
                          outputs/runs/phase2/weights/best.pt.
        """
        if base_weights is None:
            candidates = [
                self.save_dir / "phase2" / "weights" / "best.pt",
                self.project_root / "weights" / "archon_best.pt",
            ]
            base_weights_path = next((p for p in candidates if p.exists()), None)
            if base_weights_path is None:
                raise FileNotFoundError(
                    "No Phase 2 backbone weights found for hybrid training. "
                    "Run --mode train first."
                )
            base_weights = str(base_weights_path)

        logger.info("=" * 60)
        logger.info("PHASE 3: Hybrid component training")
        logger.info("  Swin Transformer + Cross-Attention + Severity Head")
        logger.info("  Backbone: %s", base_weights)
        logger.info("=" * 60)
        self._train_hybrid_heads(base_weights=base_weights)

    def _train_hybrid_heads(self, base_weights: str) -> None:
        """Train hybrid components with frozen YOLOv8 backbone."""
        import os
        import torch
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import CosineAnnealingLR
        from src.models.yolortho import build_archon_model
        from src.training.loss import AttributeBCELoss
        from src.models.hybrid_head import SeverityLoss, QuadrantAuxLoss
        from src.data.dataset import ARCHONDataset

        device = self.device
        epochs = int(self.train_cfg.get(
            "phase3_epochs",
            self.train_cfg.get("phase2b_epochs", 50)
        ))
        logger.info("Hybrid training: %d epochs on %s", epochs, device)

        # Build hybrid model
        model = build_archon_model(
            num_classes=32,
            base_weights=base_weights,
            use_coordconv=True,
            device=device,
            verbose=True,
        )

        # Freeze backbone + detection head + binary attribute heads
        for param in model.base_model.parameters():
            param.requires_grad = False
        for param in model.attr_heads.parameters():
            param.requires_grad = False

        # Only train the three new hybrid components
        hybrid_trainable = list(model.global_encoder.parameters()) + \
                           list(model.fusion.parameters()) + \
                           list(model.hybrid_head.parameters())
        for p in hybrid_trainable:
            p.requires_grad = True

        n_hybrid = sum(p.numel() for p in hybrid_trainable)
        logger.info("Hybrid trainable parameters: %s", f"{n_hybrid:,}")

        # Dataset (same as attr head training — labels_ext for disease supervision)
        pseudo_root = self.project_root / "data" / "pseudo"
        processed_root = self.project_root / "data" / "processed"
        data_root = pseudo_root if (pseudo_root / "images").exists() else processed_root
        labels_ext_dir = data_root / "labels_ext" / "train"
        if not labels_ext_dir.exists() or not any(labels_ext_dir.glob("*.txt")):
            labels_ext_dir = processed_root / "labels_ext" / "train"
            images_dir = processed_root / "images" / "train"
        else:
            images_dir = data_root / "images" / "train"

        dataset = ARCHONDataset(
            images_dir=images_dir,
            labels_dir=labels_ext_dir,
            img_size=(self.train_cfg.get("input_height", 640),
                      self.train_cfg.get("input_width", 1280)),
        )
        max_workers = min(self.train_cfg.get("workers", 4), os.cpu_count() or 2, 4)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=max(1, self.train_cfg.get("batch_size", 8) // 2),  # half batch for memory
            shuffle=True,
            num_workers=max_workers,
            collate_fn=ARCHONDataset.collate_fn,
        )

        optimizer = AdamW(hybrid_trainable, lr=5e-4, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

        # Loss functions
        _pw_cfg = self.train_cfg.get("attr_pos_weight", None)
        pos_weight = [float(v) for v in _pw_cfg] if _pw_cfg and len(_pw_cfg) == 4 \
                     else _compute_attr_pos_weight(dataset)
        attr_loss_fn = AttributeBCELoss(num_attrs=4, loss_weight=4.0, pos_weight=pos_weight)
        severity_loss_fn = SeverityLoss(num_attrs=4, num_severity_levels=3, loss_weight=4.0)
        quadrant_loss_fn = QuadrantAuxLoss(loss_weight=1.0)
        attr_loss_fn.to(device)
        severity_loss_fn.to(device)

        # Freeze backbone in eval mode (BN uses running stats, not batch stats)
        model.base_model.model.eval()
        model.global_encoder.train()
        model.fusion.train()
        model.hybrid_head.train()

        best_loss = float("inf")
        best_path = self.save_dir / "phase2" / "weights" / "hybrid_best.pt"
        best_path.parent.mkdir(parents=True, exist_ok=True)
        weights_hybrid_path = self.project_root / "weights" / "hybrid_best.pt"
        weights_hybrid_path.parent.mkdir(exist_ok=True)
        optimizer_stepped = False

        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0

            for batch_imgs, batch_labels in loader:
                batch_imgs = batch_imgs.to(device)
                optimizer.zero_grad()

                # Run frozen backbone to populate FPN hooks
                model._fpn_features = []
                with torch.no_grad():
                    model.base_model.model(batch_imgs)

                fpn_raw = [f.detach() for f in model._fpn_features[:3] if f is not None]
                if len(fpn_raw) < 3:
                    continue

                # Swin global context (trainable)
                p5 = fpn_raw[2]
                global_ctx = model.global_encoder(p5)

                # Cross-attention fusion (trainable)
                fpn_fused = model.fusion(fpn_raw, global_ctx)

                # Hybrid head outputs (trainable)
                hybrid_out = model.hybrid_head(fpn_fused)
                # Also run legacy binary attr heads (frozen, for compat)
                with torch.no_grad():
                    attr_outputs = model.attr_heads(fpn_fused)

                # ── Per-tooth supervision ──────────────────────────────────────
                all_attr_preds, all_sev_preds = [], []
                all_quad_preds, all_quad_targets = [], []
                all_targets, all_dtypes = [], []

                for img_idx, lbl in enumerate(batch_labels):
                    if lbl is None or len(lbl) == 0:
                        continue
                    lbl = lbl.to(device)
                    cx = lbl[:, 1]
                    cy = lbl[:, 2]
                    cls_ids = lbl[:, 0].long()
                    attrs = lbl[:, 5:9].float()
                    data_types = lbl[:, 9].long()

                    # Sample severity logits at each tooth center across all scales
                    sev_scale_samples = []
                    attr_scale_samples = []
                    quad_scale_samples = []

                    for s_idx, (sev_feat, quad_feat, attr_feat) in enumerate(zip(
                        hybrid_out["severity"],
                        hybrid_out["quadrant"],
                        attr_outputs,
                    )):
                        _, C_sev, H_f, W_f = sev_feat.shape
                        xf = (cx * W_f).long().clamp(0, W_f - 1)
                        yf = (cy * H_f).long().clamp(0, H_f - 1)

                        # Severity: (C_sev=12, N_teeth) → (N_teeth, 12)
                        sev_scale_samples.append(sev_feat[img_idx, :, yf, xf].T)
                        # Quadrant: (4, N_teeth) → (N_teeth, 4)
                        quad_scale_samples.append(quad_feat[img_idx, :, yf, xf].T)
                        # Attr: (4, N_teeth) → (N_teeth, 4)
                        attr_scale_samples.append(attr_feat[img_idx, :, yf, xf].T)

                    if not sev_scale_samples:
                        continue

                    tooth_sev = torch.stack(sev_scale_samples, dim=0).mean(dim=0)    # (N, 12)
                    tooth_quad = torch.stack(quad_scale_samples, dim=0).mean(dim=0)  # (N, 4)
                    tooth_attr = torch.stack(attr_scale_samples, dim=0).mean(dim=0)  # (N, 4)

                    all_sev_preds.append(tooth_sev)
                    all_quad_preds.append(tooth_quad)
                    all_attr_preds.append(tooth_attr)
                    all_quad_targets.append(cls_ids)
                    all_targets.append(attrs)
                    all_dtypes.append(data_types)

                if not all_sev_preds:
                    continue

                pred_sev = torch.cat(all_sev_preds, dim=0)
                pred_quad = torch.cat(all_quad_preds, dim=0)
                pred_attr = torch.cat(all_attr_preds, dim=0)
                tgt_attr = torch.cat(all_targets, dim=0).to(device)
                quad_tgt = torch.cat(all_quad_targets, dim=0).to(device)
                dt = torch.cat(all_dtypes, dim=0).to(device)

                loss_attr = attr_loss_fn(pred_attr, tgt_attr, dt)
                loss_sev = severity_loss_fn(pred_sev, tgt_attr, dt)
                loss_quad = quadrant_loss_fn(pred_quad, quad_tgt)
                loss = loss_attr + loss_sev + loss_quad

                loss.backward()
                optimizer.step()
                optimizer_stepped = True

                epoch_loss += loss.item()
                n_batches += 1

            if optimizer_stepped:
                scheduler.step()

            avg_loss = epoch_loss / max(n_batches, 1)
            logger.info(
                "Hybrid epoch [%d/%d]  loss=%.4f", epoch + 1, epochs, avg_loss
            )

            if avg_loss < best_loss:
                best_loss = avg_loss
                ckpt = {
                    "epoch": epoch,
                    "global_encoder_state": model.global_encoder.state_dict(),
                    "fusion_state": model.fusion.state_dict(),
                    "hybrid_head_state": model.hybrid_head.state_dict(),
                    "loss": best_loss,
                }
                torch.save(ckpt, best_path)
                torch.save(ckpt, weights_hybrid_path)

        logger.info("Hybrid training complete. Best loss: %.4f", best_loss)
        logger.info("hybrid_best.pt saved to '%s'.", weights_hybrid_path)
