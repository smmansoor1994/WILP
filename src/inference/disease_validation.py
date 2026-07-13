"""
Disease Classification Validation & Improvement Pipeline

Orchestrates 4-stage workflow:
  1. Enumerate teeth + crop disease regions
  2. Train disease classifiers on crops  
  3. Validate & compare with stage-1
  4. Generate improvement recommendations
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime

import cv2

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DiseaseValidationPipeline:
    """Orchestrate two-stage disease classification validation."""
    
    def __init__(
        self,
        model_weights: str,
        device: str = "cuda",
        margin_ratio: float = 0.15,
        batch_size: int = 16,
    ):
        self.model_weights = Path(model_weights)
        self.device = device
        self.margin_ratio = margin_ratio
        self.batch_size = batch_size
        self.stage1_crops: List[Any] = []
        self.stage2_metrics: Dict[str, Any] = {}
        self.validation_report: Optional[Any] = None
        
        logger.info(
            f"DiseaseValidationPipeline initialized: "
            f"weights={model_weights}, device={device}"
        )
    
    def stage_1_enumerate_and_crop(
        self,
        images_dir: Path,
        labels_dir: Path,
        output_crops_dir: Path,
        sample_images: Optional[int] = None,
    ) -> Dict[str, int]:
        """Stage 1: Use enumeration to crop teeth and associate with labels."""
        
        logger.info("=" * 80)
        logger.info("STAGE 1: ENUMERATE & CROP".center(80))
        logger.info("=" * 80)
        
        from src.inference.predictor import ARCHONHybridPredictor
        from src.inference.disease_cropper import DiseaseCropper
        
        images_dir = Path(images_dir)
        labels_dir = Path(labels_dir)
        output_crops_dir = Path(output_crops_dir)
        
        logger.info(f"Loading model: {self.model_weights}")
        predictor = ARCHONHybridPredictor(
            weights_path=str(self.model_weights),
            device=self.device,
            conf_threshold=0.05,
        )
        
        cropper = DiseaseCropper(margin_ratio=self.margin_ratio)
        
        image_files = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
        if sample_images:
            image_files = image_files[:sample_images]
        
        logger.info(f"Processing {len(image_files)} images...")
        
        total_crops = 0
        total_images = 0
        crops_with_gt = 0
        
        for img_idx, img_path in enumerate(image_files):
            if (img_idx + 1) % 50 == 0:
                logger.info(f"Progress: {img_idx + 1}/{len(image_files)}")
            
            try:
                image_id = img_path.stem
                image = cv2.imread(str(img_path))
                if image is None:
                    logger.warning(f"Failed to load image: {img_path}")
                    continue
                
                pred_results = predictor.predict(
                    input_path=str(img_path),
                    save_vis=False,
                    save_json=False,
                )
                
                # The predictor returns results keyed by img_path.name (with extension)
                # not by img_path.stem, so we need to look up using the filename
                teeth_detections = pred_results.get(img_path.name, [])
                
                crops = cropper.extract_crops_from_prediction(
                    image=image,
                    teeth_detections=teeth_detections,
                    image_id=image_id,
                )
                
                # Load ground truth labels
                # Try labels_ext first (with disease attributes), then fall back to labels
                label_path_ext = labels_dir.parent / "labels_ext" / labels_dir.name / f"{image_id}.txt"
                label_path = labels_dir / f"{image_id}.txt"
                
                if label_path_ext.exists():
                    gt_labels = self._load_yolo_labels(label_path_ext)
                elif label_path.exists():
                    gt_labels = self._load_yolo_labels(label_path)
                else:
                    gt_labels = {}
                
                for crop in crops:
                    if crop.fdi in gt_labels:
                        gt = gt_labels[crop.fdi]
                        crop.has_caries = gt.get("has_caries", False)
                        crop.has_deepcaries = gt.get("has_deepcaries", False)
                        crop.has_lesion = gt.get("has_lesion", False)
                        crop.has_impacted = gt.get("has_impacted", False)
                        crop.diseases = gt.get("diseases", [])
                        crops_with_gt += 1
                
                cropper.save_crops_batch(crops, output_crops_dir, include_metadata=True)
                
                self.stage1_crops.extend(crops)
                total_crops += len(crops)
                total_images += 1
                
            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")
                continue
        
        stats = {
            "images_processed": total_images,
            "total_crops": total_crops,
            "crops_with_gt": crops_with_gt,
            "avg_crops_per_image": total_crops / max(total_images, 1),
        }
        
        logger.info(f"Stage 1 Complete: {stats}")
        return stats
    
    def stage_2_train_classifiers(
        self,
        crops_dir: Path,
        output_models_dir: Path,
        epochs: int = 30,
        val_split: float = 0.2,
    ) -> Dict[str, Any]:
        """Stage 2: Train independent disease classifiers on cropped regions."""
        
        logger.info("=" * 80)
        logger.info("STAGE 2: TRAIN CLASSIFIERS".center(80))
        logger.info("=" * 80)
        
        from src.inference.disease_classifier import DiseaseClassifier
        
        output_models_dir = Path(output_models_dir)
        
        if not self.stage1_crops:
            from src.inference.disease_cropper import DiseaseCropper
            cropper = DiseaseCropper()
            self.stage1_crops = cropper.load_crops_from_dir(crops_dir, load_images=True)
        
        logger.info(f"Loaded {len(self.stage1_crops)} crops")
        
        num_train = int(len(self.stage1_crops) * (1 - val_split))
        train_crops = self.stage1_crops[:num_train]
        val_crops = self.stage1_crops[num_train:]
        
        logger.info(f"Train: {len(train_crops)}, Val: {len(val_crops)}")
        
        classifier = DiseaseClassifier(
            backbone="resnet50",
            device=self.device,
            batch_size=self.batch_size,
        )
        
        classifier.train(
            train_crops=train_crops,
            val_crops=val_crops,
            epochs=epochs,
        )
        
        classifier.save(output_models_dir)
        
        self.stage2_metrics = {
            "model_dir": str(output_models_dir),
            "epochs": epochs,
            "train_samples": len(train_crops),
            "val_samples": len(val_crops),
        }
        
        logger.info(f"Stage 2 Complete: Models saved to {output_models_dir}")
        return self.stage2_metrics
    
    def stage_3_validate_and_compare(
        self,
        model_dir: Path,
    ) -> Dict[str, Any]:
        """Stage 3: Validate stage-2 predictions and compare with stage-1."""
        
        logger.info("=" * 80)
        logger.info("STAGE 3: VALIDATE & COMPARE".center(80))
        logger.info("=" * 80)
        
        from src.inference.disease_classifier import DiseaseClassifier
        
        model_dir = Path(model_dir)
        
        if not self.stage1_crops:
            logger.error("No crops loaded. Run stage 1 first.")
            return {}
        
        classifier = DiseaseClassifier(device=self.device)
        classifier.load(model_dir)
        
        report = classifier.validate_against_ground_truth(self.stage1_crops)
        self.validation_report = report
        
        logger.info(f"Validation Report: {report.to_dict()}")
        
        return report.to_dict()
    
    def stage_4_generate_improvement_report(
        self,
        output_path: Path,
    ) -> Dict[str, Any]:
        """Stage 4: Generate comprehensive improvement recommendations."""
        
        logger.info("=" * 80)
        logger.info("STAGE 4: GENERATE REPORT".center(80))
        logger.info("=" * 80)
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not self.validation_report:
            logger.error("No validation report. Run stage 3 first.")
            return {}
        
        report_dict = self.validation_report.to_dict()
        report_dict["timestamp"] = datetime.now().isoformat()
        report_dict["improvements_recommendations"] = self._generate_recommendations()
        
        with open(output_path, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        logger.info(f"Report saved to: {output_path}")
        
        return report_dict
    
    def _generate_recommendations(self) -> Dict[str, Any]:
        """Generate improvement recommendations based on validation results."""
        recommendations = {
            "short_term": [],
            "medium_term": [],
            "long_term": [],
        }
        
        if not self.validation_report:
            return recommendations
        
        for disease_name, metrics in self.validation_report.diseases.items():
            if metrics.recall < 0.70:
                recommendations["short_term"].append({
                    "action": "Lower threshold",
                    "disease": disease_name,
                    "reason": f"Recall too low: {metrics.recall:.2%}",
                    "suggested_threshold": metrics.threshold - 0.1,
                })
            if metrics.precision < 0.60:
                recommendations["short_term"].append({
                    "action": "Raise threshold",
                    "disease": disease_name,
                    "reason": f"Precision too low: {metrics.precision:.2%}",
                    "suggested_threshold": metrics.threshold + 0.1,
                })
        
        if self.validation_report.agreement_with_stage1 < 0.80:
            recommendations["medium_term"].append({
                "action": "Collect hard examples",
                "reason": f"Low agreement with stage-1: {self.validation_report.agreement_with_stage1:.2%}",
                "suggestion": "Focus retraining on hard negatives and positives",
            })
        
        recommendations["long_term"].append({
            "action": "Build ensemble",
            "suggestion": "Combine stage-1 and stage-2 predictions via voting or weighted fusion",
        })
        
        return recommendations
    
    def run_full_pipeline(
        self,
        images_dir: Path,
        labels_dir: Path,
        output_dir: Path,
        epochs: int = 30,
        sample_images: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run complete pipeline."""
        output_dir = Path(output_dir)
        
        crops_dir = output_dir / "stage1_crops"
        models_dir = output_dir / "stage2_models"
        report_path = output_dir / "stage4_validation_report.json"
        
        logger.info("STARTING FULL PIPELINE")
        logger.info(f"Output directory: {output_dir}\n")
        
        s1_stats = self.stage_1_enumerate_and_crop(
            images_dir, labels_dir, crops_dir, sample_images
        )
        
        s2_stats = self.stage_2_train_classifiers(
            crops_dir, models_dir, epochs
        )
        
        s3_results = self.stage_3_validate_and_compare(models_dir)
        
        s4_report = self.stage_4_generate_improvement_report(report_path)
        
        logger.info("=" * 80)
        logger.info("PIPELINE COMPLETE".center(80))
        logger.info("=" * 80)
        logger.info(f"Report: {report_path}\n")
        
        return {
            "stage_1": s1_stats,
            "stage_2": s2_stats,
            "stage_3": s3_results,
            "stage_4": s4_report,
        }
    
    @staticmethod
    def _load_yolo_labels(label_path: Path) -> Dict[int, Dict[str, Any]]:
        """Load YOLO format labels with disease attributes."""
        labels_dict = {}
        
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    
                    fdi_class = int(parts[0])
                    
                    attributes = {
                        "has_caries": False,
                        "has_deepcaries": False,
                        "has_lesion": False,
                        "has_impacted": False,
                    }
                    
                    if len(parts) >= 9:
                        attributes["has_caries"] = int(parts[5]) > 0
                        attributes["has_deepcaries"] = int(parts[6]) > 0
                        attributes["has_lesion"] = int(parts[7]) > 0
                        attributes["has_impacted"] = int(parts[8]) > 0
                    
                    attributes["diseases"] = [
                        k for k, v in attributes.items() if v
                    ]
                    
                    labels_dict[fdi_class] = attributes
        
        except Exception as e:
            logger.error(f"Error loading labels from {label_path}: {e}")
        
        return labels_dict
