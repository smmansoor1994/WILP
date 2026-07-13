"""
Disease Ensemble Model: Fuses Stage-1 (ARCHON enumeration) + Stage-2 (disease classifiers)

This module combines predictions from two stages:
  - Stage 1: ARCHON model for tooth enumeration + initial disease detection
  - Stage 2: Independent ResNet50 binary classifiers for each disease attribute

The ensemble uses weighted voting to produce robust disease predictions.
"""

import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import json

import torch
import numpy as np
from PIL import Image
import cv2

from src.inference.predictor import ARCHONPredictor, ToothDetection
from src.inference.disease_cropper import DiseaseCropper, DiseaseCrop
from src.inference.disease_classifier import DiseaseClassifier, DiseaseDataset

logger = logging.getLogger(__name__)


@dataclass
class EnsemblePrediction:
    """Prediction result from ensemble model"""
    image_path: Path
    fdi_number: int
    stage1_predictions: Dict[str, float]  # Raw Stage-1 scores
    stage2_predictions: Dict[str, float]  # Raw Stage-2 scores
    fused_predictions: Dict[str, float]   # Weighted fusion
    fused_probabilities: Dict[str, float] # After softmax
    final_predictions: Dict[str, bool]    # After thresholding
    confidence: float  # Min confidence across diseases


@dataclass
class EnsembleReport:
    """Summary report from ensemble evaluation"""
    total_crops: int
    total_images: int
    agreement_stage1_stage2: float
    per_disease_metrics: Dict
    predictions: List[Dict]
    timestamp: str


class DiseaseEnsemble:
    """
    Ensemble model combining Stage-1 enumeration and Stage-2 disease classification.
    
    Architecture:
    ┌─────────────────────────────────────────────────────────┐
    │  Input Image                                            │
    └─────────────────────────────────────────────────────────┘
                         │
         ┌───────────────┴───────────────┐
         ▼                               ▼
    ┌─────────────┐              ┌──────────────┐
    │  Stage 1    │              │              │
    │  ARCHON     │──────────────▶ Extract      │
    │  Detection  │              │ Crops        │
    └─────────────┘              └──────────────┘
         │                            │
         │ Disease scores            │ Crop images
         │ (raw)                     │
         │                           ▼
         │                      ┌──────────────┐
         │                      │  Stage 2     │
         │                      │  4 Binary    │
         │                      │  Classifiers │
         │                      └──────────────┘
         │                           │
         │                     Disease scores
         │                     (raw)
         │                           │
         └───────────────┬───────────┘
                         ▼
                   ┌──────────────┐
                   │ Weighted     │
                   │ Fusion       │
                   │ (α*S1 +      │
                   │  (1-α)*S2)   │
                   └──────────────┘
                         │
                         ▼
                   ┌──────────────┐
                   │ Apply        │
                   │ Thresholds   │
                   └──────────────┘
                         │
                         ▼
                   ┌──────────────┐
                   │ Final        │
                   │ Predictions  │
                   │ (bool)       │
                   └──────────────┘
    """
    
    def __init__(
        self,
        archon_weights: Path,
        stage2_models_dir: Path,
        device: str = "cuda",
        stage1_weight: float = 0.3,
        stage2_weight: float = 0.7,
    ):
        """
        Initialize ensemble with both stages.
        
        Args:
            archon_weights: Path to ARCHON model weights
            stage2_models_dir: Directory containing 4 trained disease classifiers
            device: 'cuda' or 'cpu'
            stage1_weight: Weight for Stage-1 predictions (0.0-1.0)
            stage2_weight: Weight for Stage-2 predictions (0.0-1.0)
        """
        self.device = device
        self.archon_weights = Path(archon_weights)
        self.stage2_models_dir = Path(stage2_models_dir)
        
        # Normalize weights
        total_weight = stage1_weight + stage2_weight
        self.stage1_weight = stage1_weight / total_weight
        self.stage2_weight = stage2_weight / total_weight
        
        logger.info(f"DiseaseEnsemble initialized: stage1_weight={self.stage1_weight:.2f}, "
                   f"stage2_weight={self.stage2_weight:.2f}")
        
        # Initialize Stage 1: ARCHON predictor
        self.predictor = ARCHONPredictor(
            weights_path=self.archon_weights,
            device=device
        )
        
        # Initialize crop extractor
        self.cropper = DiseaseCropper(margin=0.15)
        
        # Initialize Stage 2: Disease classifiers
        self.classifier = DiseaseClassifier(
            backbone="resnet50",
            lr=0.0001,
            batch_size=16,
            device=device
        )
        
        # Load trained disease classifiers
        self._load_stage2_models()
        
        # Disease list
        self.diseases = ['has_caries', 'has_deepcaries', 'has_lesion', 'has_impacted']
        
        # Thresholds for final predictions (can be tuned)
        self.thresholds = {
            'has_caries': 0.45,
            'has_deepcaries': 0.45,
            'has_lesion': 0.40,
            'has_impacted': 0.50
        }
        
        logger.info("✓ DiseaseEnsemble fully initialized")
    
    def _load_stage2_models(self):
        """Load 4 trained disease classifiers from disk"""
        logger.info("Loading Stage-2 disease classifiers...")
        
        model_files = {
            'has_caries': self.stage2_models_dir / 'has_caries_model.pt',
            'has_deepcaries': self.stage2_models_dir / 'has_deepcaries_model.pt',
            'has_lesion': self.stage2_models_dir / 'has_lesion_model.pt',
            'has_impacted': self.stage2_models_dir / 'has_impacted_model.pt',
        }
        
        for disease, model_path in model_files.items():
            if not model_path.exists():
                logger.warning(f"Model not found: {model_path}")
                continue
            
            self.classifier.load(model_path, disease)
            logger.info(f"  ✓ Loaded {disease}")
    
    def predict_single(self, image_path: Path) -> Dict[str, EnsemblePrediction]:
        """
        Predict diseases for all teeth in a single image.
        
        Args:
            image_path: Path to X-ray image
            
        Returns:
            Dictionary mapping FDI number → EnsemblePrediction
        """
        image_path = Path(image_path)
        
        # Stage 1: Enumerate teeth and detect diseases
        logger.info(f"Stage 1: Enumerating teeth in {image_path.name}")
        stage1_results = self.predictor.predict([image_path])
        
        if not stage1_results or image_path.name not in stage1_results:
            logger.warning(f"No teeth detected in {image_path.name}")
            return {}
        
        detections = stage1_results[image_path.name]
        if not detections:
            logger.info(f"No teeth detected in {image_path.name}")
            return {}
        
        # Extract crops for Stage 2
        logger.info(f"Stage 2: Extracting crops and classifying diseases")
        crops = self.cropper.extract_crops_from_prediction(detections)
        
        if not crops:
            logger.warning("No crops extracted")
            return {}
        
        # Get Stage-2 predictions for each crop
        predictions = {}
        
        for crop in crops:
            fdi = crop.fdi_number
            
            # Get Stage-1 disease scores
            stage1_scores = self._extract_stage1_scores(crop)
            
            # Get Stage-2 disease scores
            stage2_scores = self._extract_stage2_scores(crop)
            
            # Fuse predictions
            fused_scores = self._fuse_predictions(stage1_scores, stage2_scores)
            
            # Apply thresholds
            final_preds = {
                disease: (score > self.thresholds[disease])
                for disease, score in fused_scores.items()
            }
            
            # Compute confidence
            confidence = min(fused_scores.values())
            
            predictions[fdi] = EnsemblePrediction(
                image_path=image_path,
                fdi_number=fdi,
                stage1_predictions=stage1_scores,
                stage2_predictions=stage2_scores,
                fused_predictions=fused_scores,
                fused_probabilities=self._softmax_scores(fused_scores),
                final_predictions=final_preds,
                confidence=confidence
            )
        
        return predictions
    
    def predict_batch(self, image_paths: List[Path]) -> Dict[str, Dict]:
        """
        Predict diseases for multiple images.
        
        Args:
            image_paths: List of paths to X-ray images
            
        Returns:
            Dictionary mapping image name → predictions
        """
        all_predictions = {}
        
        for i, img_path in enumerate(image_paths, 1):
            logger.info(f"Processing {i}/{len(image_paths)}: {Path(img_path).name}")
            preds = self.predict_single(img_path)
            all_predictions[Path(img_path).name] = preds
        
        return all_predictions
    
    def _extract_stage1_scores(self, crop: DiseaseCrop) -> Dict[str, float]:
        """Extract disease scores from Stage-1 ARCHON predictions"""
        scores = {}
        
        # Get Stage-1 raw predictions from crop metadata
        if crop.pred_labels:
            for disease in self.diseases:
                # pred_labels contains Stage-1 predictions
                scores[disease] = float(crop.pred_labels.get(disease, 0.0))
        else:
            # Default to zero if not available
            scores = {disease: 0.0 for disease in self.diseases}
        
        return scores
    
    def _extract_stage2_scores(self, crop: DiseaseCrop) -> Dict[str, float]:
        """
        Extract disease scores from Stage-2 classifiers.
        
        Args:
            crop: DiseaseCrop object containing tooth image
            
        Returns:
            Dictionary of disease → confidence score (0.0-1.0)
        """
        scores = {}
        
        # Prepare image for Stage-2 classifiers
        image = crop.crop_image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Get predictions from each disease classifier
        for disease in self.diseases:
            try:
                # Preprocess image
                img_tensor = self._preprocess_image(image)
                
                # Get model prediction
                with torch.no_grad():
                    output = self.classifier.models[disease](img_tensor.to(self.device))
                    prob = torch.softmax(output, dim=1)[0, 1].item()
                
                scores[disease] = prob
            except Exception as e:
                logger.warning(f"Error predicting {disease}: {e}")
                scores[disease] = 0.5  # Default to uncertain
        
        return scores
    
    def _fuse_predictions(
        self,
        stage1_scores: Dict[str, float],
        stage2_scores: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Fuse Stage-1 and Stage-2 predictions using weighted average.
        
        Args:
            stage1_scores: Disease scores from ARCHON
            stage2_scores: Disease scores from ResNet50 classifiers
            
        Returns:
            Fused disease scores (0.0-1.0)
        """
        fused = {}
        
        for disease in self.diseases:
            s1 = stage1_scores.get(disease, 0.0)
            s2 = stage2_scores.get(disease, 0.5)  # Default to 0.5 if missing
            
            # Weighted average
            fused[disease] = (
                self.stage1_weight * s1 +
                self.stage2_weight * s2
            )
        
        return fused
    
    def _softmax_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        """
        Convert raw scores to softmax probabilities.
        
        Args:
            scores: Dictionary of disease → score
            
        Returns:
            Softmax normalized probabilities
        """
        # For binary classification per disease, just clip to [0, 1]
        return {
            disease: np.clip(score, 0.0, 1.0)
            for disease, score in scores.items()
        }
    
    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess image for Stage-2 classifiers.
        
        Args:
            image: PIL Image
            
        Returns:
            Torch tensor (1, 3, 224, 224)
        """
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        tensor = transform(image).unsqueeze(0)  # Add batch dimension
        return tensor
    
    def set_thresholds(self, thresholds: Dict[str, float]):
        """
        Set prediction thresholds for each disease.
        
        Args:
            thresholds: Dictionary mapping disease → threshold (0.0-1.0)
        """
        for disease in self.diseases:
            if disease in thresholds:
                self.thresholds[disease] = thresholds[disease]
        
        logger.info(f"Thresholds updated: {self.thresholds}")
    
    def save_config(self, output_path: Path):
        """Save ensemble configuration to JSON"""
        config = {
            'archon_weights': str(self.archon_weights),
            'stage2_models_dir': str(self.stage2_models_dir),
            'device': self.device,
            'stage1_weight': float(self.stage1_weight),
            'stage2_weight': float(self.stage2_weight),
            'thresholds': self.thresholds
        }
        
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Configuration saved to {output_path}")
    
    def load_config(self, config_path: Path):
        """Load ensemble configuration from JSON"""
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        self.set_thresholds(config.get('thresholds', self.thresholds))
        logger.info(f"Configuration loaded from {config_path}")


# ============================================================================
# Quick-start functions for easy usage
# ============================================================================

def create_ensemble(
    archon_weights: str = "weights/archon_best.pt",
    stage2_models_dir: str = "outputs/disease_validation_option2/stage2_models",
    device: str = "cuda"
) -> DiseaseEnsemble:
    """
    Convenience function to create ensemble with defaults.
    
    Args:
        archon_weights: Path to ARCHON weights
        stage2_models_dir: Directory with trained disease classifiers
        device: 'cuda' or 'cpu'
        
    Returns:
        Initialized DiseaseEnsemble
    """
    return DiseaseEnsemble(
        archon_weights=Path(archon_weights),
        stage2_models_dir=Path(stage2_models_dir),
        device=device,
        stage1_weight=0.3,
        stage2_weight=0.7
    )


def predict_on_image(
    image_path: str,
    ensemble: Optional[DiseaseEnsemble] = None
) -> Dict:
    """
    Simple function to get ensemble predictions on a single image.
    
    Args:
        image_path: Path to X-ray image
        ensemble: DiseaseEnsemble instance (created if None)
        
    Returns:
        Dictionary of predictions
    """
    if ensemble is None:
        ensemble = create_ensemble()
    
    results = ensemble.predict_single(Path(image_path))
    
    # Format output
    output = {
        'image': Path(image_path).name,
        'teeth': []
    }
    
    for fdi, pred in sorted(results.items()):
        output['teeth'].append({
            'fdi': fdi,
            'diseases': pred.final_predictions,
            'confidence': pred.confidence,
            'raw_scores': pred.fused_predictions
        })
    
    return output


if __name__ == "__main__":
    # Example usage
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Create ensemble
    ensemble = create_ensemble()
    
    # Predict on single image
    test_image = Path("data/processed/images/test/test_0.png")
    if test_image.exists():
        results = ensemble.predict_single(test_image)
        
        print("\n" + "="*80)
        print("ENSEMBLE PREDICTIONS")
        print("="*80)
        
        for fdi, pred in results.items():
            print(f"\nTooth FDI {fdi}:")
            print(f"  Confidence: {pred.confidence:.2%}")
            print(f"  Predictions:")
            for disease, result in pred.final_predictions.items():
                score = pred.fused_predictions[disease]
                print(f"    • {disease}: {result} (score: {score:.3f})")
    else:
        print(f"Test image not found: {test_image}")
