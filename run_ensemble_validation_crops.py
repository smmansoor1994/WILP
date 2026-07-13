"""
Ensemble Validation on Extracted Crops (has ground truth labels)

This script evaluates the ensemble by:
1. Loading pre-extracted crops from Stage 1 (which have GT disease labels)
2. Running Stage 2 disease classifiers on each crop
3. Comparing Stage-1 vs Stage-2 vs Ensemble predictions
4. Generating metrics against ground truth
"""

import logging
from pathlib import Path
import json
from typing import Dict, List
import numpy as np
from datetime import datetime
import pickle

from src.inference.disease_classifier import DiseaseClassifier
from src.inference.disease_cropper import DiseaseCrop

# ============================================================================
# Configuration
# ============================================================================

CONFIG = {
    "crops_dir": Path("outputs/disease_validation_option2/stage1_crops"),
    "stage2_models_dir": Path("outputs/disease_validation_option2/stage2_models"),
    
    # Ensemble weights
    "stage1_weight": 0.3,
    "stage2_weight": 0.7,
    
    # Thresholds for final predictions
    "thresholds": {
        "has_caries": 0.45,
        "has_deepcaries": 0.45,
        "has_lesion": 0.40,
        "has_impacted": 0.50
    },
    
    # Output
    "output_dir": Path("outputs/ensemble_validation_on_crops"),
    "device": "cuda",
}

# ============================================================================
# Setup logging
# ============================================================================

def setup_logging():
    """Configure logging"""
    log_format = "[%(asctime)s] [%(levelname)s] %(name)s — %(message)s"
    log_dir = CONFIG["output_dir"] / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(log_dir / f"ensemble_crops_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
    
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=handlers
    )
    
    return logging.getLogger(__name__)

logger = setup_logging()

# ============================================================================
# Load and evaluate
# ============================================================================

def load_crops(crops_dir: Path) -> List[DiseaseCrop]:
    """Load all extracted crops with their ground truth labels"""
    logger.info(f"Loading crops from {crops_dir}")
    crops = []
    
    for json_file in sorted(crops_dir.glob("*.json")):
        try:
            with open(json_file, 'r') as f:
                meta = json.load(f)
            
            # Load the associated image
            img_file = json_file.with_suffix('.jpg')
            if not img_file.exists():
                logger.warning(f"Image not found for {json_file.name}")
                continue
            
            # Reconstruct DiseaseCrop object
            import cv2
            crop_img = cv2.imread(str(img_file))
            crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
            
            crop = DiseaseCrop(
                image_id=meta['image_id'],
                fdi=meta['fdi'],
                fdi_name=meta['fdi_name'],
                crop_image=crop_img,
                bbox_original=tuple(meta['bbox_original']),
                bbox_crop=tuple(meta['bbox_crop']),
                has_caries=meta['ground_truth']['has_caries'],
                has_deepcaries=meta['ground_truth']['has_deepcaries'],
                has_lesion=meta['ground_truth']['has_lesion'],
                has_impacted=meta['ground_truth']['has_impacted'],
                tooth_confidence=meta['tooth_confidence'],
                image_height=meta['image_size']['height'],
                image_width=meta['image_size']['width'],
            )
            
            # Store Stage-1 predictions if available
            if 'prediction' in meta:
                crop.pred_has_caries = meta['prediction'].get('has_caries')
                crop.pred_has_deepcaries = meta['prediction'].get('has_deepcaries')
                crop.pred_has_lesion = meta['prediction'].get('has_lesion')
                crop.pred_has_impacted = meta['prediction'].get('has_impacted')
            
            crops.append(crop)
        
        except Exception as e:
            logger.warning(f"Error loading crop {json_file.name}: {e}")
    
    logger.info(f"✓ Loaded {len(crops)} crops with ground truth labels")
    return crops


def evaluate_ensemble_on_crops(
    crops: List[DiseaseCrop],
    classifier: DiseaseClassifier,
    stage1_weight: float = 0.3,
    stage2_weight: float = 0.7,
    thresholds: Dict[str, float] = None
) -> Dict:
    """
    Evaluate ensemble predictions on crops.
    
    Args:
        crops: List of DiseaseCrop objects with GT labels
        classifier: Initialized DiseaseClassifier with loaded models
        stage1_weight: Weight for Stage-1 predictions
        stage2_weight: Weight for Stage-2 predictions
        thresholds: Decision thresholds per disease
        
    Returns:
        Evaluation report
    """
    if thresholds is None:
        thresholds = {d: 0.5 for d in classifier.DISEASES}
    
    logger.info(f"\nEvaluating ensemble on {len(crops)} crops")
    
    # Initialize metrics
    diseases = classifier.DISEASES
    metrics = {
        disease: {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0}
        for disease in diseases
    }
    
    stage1_agreement = 0
    stage2_agreement = 0
    
    # Process each crop
    for crop_idx, crop in enumerate(crops):
        if (crop_idx + 1) % 100 == 0:
            logger.info(f"  Processed {crop_idx + 1}/{len(crops)}")
        
        # Get Stage-2 predictions for this crop
        try:
            import torch
            from PIL import Image
            from torchvision import transforms
            
            # Preprocess crop image
            if isinstance(crop.crop_image, np.ndarray):
                pil_img = Image.fromarray(crop.crop_image)
            else:
                pil_img = crop.crop_image
            
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            img_tensor = transform(pil_img).unsqueeze(0).to(CONFIG["device"])
            
            # Get predictions from each disease classifier
            stage2_scores = {}
            for disease in diseases:
                if disease in classifier.models:
                    with torch.no_grad():
                        output = classifier.models[disease](img_tensor)
                        prob = torch.softmax(output, dim=1)[0, 1].item()
                    stage2_scores[disease] = prob
                else:
                    stage2_scores[disease] = 0.5
        
        except Exception as e:
            logger.warning(f"Error evaluating crop {crop.fdi}: {e}")
            stage2_scores = {d: 0.5 for d in diseases}
        
        # Get Stage-1 scores (convert bool to float)
        stage1_scores = {
            disease: float(getattr(crop, f'pred_{disease}', 0.0))
            for disease in diseases
        }
        
        # Fuse predictions
        fused_scores = {
            disease: (stage1_weight * (stage1_scores[disease] or 0.0) +
                     stage2_weight * stage2_scores[disease])
            for disease in diseases
        }
        
        # Apply thresholds
        for disease in diseases:
            gt = getattr(crop, disease, False)
            pred_s1 = bool(stage1_scores[disease] > 0.5) if stage1_scores[disease] else False
            pred_s2 = stage2_scores[disease] > thresholds[disease]
            pred_fused = fused_scores[disease] > thresholds[disease]
            
            # Compare ground truth vs predictions
            if pred_fused and gt:
                metrics[disease]['TP'] += 1
            elif pred_fused and not gt:
                metrics[disease]['FP'] += 1
            elif not pred_fused and gt:
                metrics[disease]['FN'] += 1
            else:
                metrics[disease]['TN'] += 1
            
            # Track agreement rates
            if pred_s1 == pred_fused:
                stage1_agreement += 1
            if pred_s2 == pred_fused:
                stage2_agreement += 1
    
    # Calculate metrics per disease
    results = {}
    for disease in diseases:
        counts = metrics[disease]
        tp, fp, fn, tn = counts['TP'], counts['FP'], counts['FN'], counts['TN']
        total = tp + fp + fn + tn
        
        accuracy = (tp + tn) / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        results[disease] = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'tn': int(tn),
            'total': int(total)
        }
    
    return {
        'total_crops': len(crops),
        'per_disease_metrics': results,
        'stage1_agreement': float(stage1_agreement / (len(crops) * len(diseases))),
        'stage2_agreement': float(stage2_agreement / (len(crops) * len(diseases))),
        'timestamp': datetime.now().isoformat(),
        'config': {
            'stage1_weight': stage1_weight,
            'stage2_weight': stage2_weight,
            'thresholds': thresholds
        }
    }


# ============================================================================
# Main
# ============================================================================

def main():
    """Main execution"""
    
    logger.info("="*80)
    logger.info("ENSEMBLE VALIDATION ON EXTRACTED CROPS")
    logger.info("="*80)
    
    # Validate paths
    logger.info("\nValidating paths...")
    if not CONFIG["crops_dir"].exists():
        logger.error(f"✗ Crops directory not found: {CONFIG['crops_dir']}")
        return
    if not CONFIG["stage2_models_dir"].exists():
        logger.error(f"✗ Models directory not found: {CONFIG['stage2_models_dir']}")
        return
    
    logger.info("✓ All paths valid")
    
    # Create output directory
    CONFIG["output_dir"].mkdir(parents=True, exist_ok=True)
    
    # Load crops with GT labels
    logger.info("\n" + "="*80)
    logger.info("LOADING CROPS")
    logger.info("="*80)
    crops = load_crops(CONFIG["crops_dir"])
    
    if not crops:
        logger.error("No crops loaded!")
        return
    
    # Count ground truth positives
    logger.info("\nGround truth distribution:")
    gt_counts = {'has_caries': 0, 'has_deepcaries': 0, 'has_lesion': 0, 'has_impacted': 0}
    for crop in crops:
        for disease in gt_counts:
            if getattr(crop, disease, False):
                gt_counts[disease] += 1
    
    for disease, count in gt_counts.items():
        pct = 100 * count / len(crops)
        logger.info(f"  • {disease}: {count} ({pct:.1f}%)")
    
    # Initialize and load classifiers
    logger.info("\n" + "="*80)
    logger.info("LOADING STAGE-2 CLASSIFIERS")
    logger.info("="*80)
    classifier = DiseaseClassifier(
        backbone="resnet50",
        learning_rate=0.0001,
        batch_size=16,
        device=CONFIG["device"]
    )
    classifier.load(CONFIG["stage2_models_dir"])
    logger.info(f"✓ Loaded {len(classifier.models)} classifiers")
    
    # Evaluate ensemble
    logger.info("\n" + "="*80)
    logger.info("EVALUATING ENSEMBLE")
    logger.info("="*80)
    
    report = evaluate_ensemble_on_crops(
        crops=crops,
        classifier=classifier,
        stage1_weight=CONFIG["stage1_weight"],
        stage2_weight=CONFIG["stage2_weight"],
        thresholds=CONFIG["thresholds"]
    )
    
    # Print results
    logger.info("\n" + "="*80)
    logger.info("RESULTS")
    logger.info("="*80)
    
    avg_f1 = np.mean([m['f1'] for m in report['per_disease_metrics'].values()])
    avg_accuracy = np.mean([m['accuracy'] for m in report['per_disease_metrics'].values()])
    
    logger.info(f"\nOverall Metrics:")
    logger.info(f"  Average Accuracy: {avg_accuracy:.1%}")
    logger.info(f"  Average F1 Score: {avg_f1:.3f}")
    logger.info(f"  Stage-1 Agreement: {report['stage1_agreement']:.1%}")
    logger.info(f"  Stage-2 Agreement: {report['stage2_agreement']:.1%}")
    
    logger.info(f"\nPer-Disease Metrics:")
    for disease, metrics in report['per_disease_metrics'].items():
        logger.info(f"\n  {disease}:")
        logger.info(f"    Accuracy:  {metrics['accuracy']:.1%}")
        logger.info(f"    Precision: {metrics['precision']:.1%}")
        logger.info(f"    Recall:    {metrics['recall']:.1%}")
        logger.info(f"    F1 Score:  {metrics['f1']:.3f}")
        logger.info(f"    Confusion: TP={metrics['tp']}, FP={metrics['fp']}, "
                   f"FN={metrics['fn']}, TN={metrics['tn']}")
    
    # Save report
    output_file = CONFIG["output_dir"] / f"ensemble_crops_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"\n✓ Report saved to {output_file}")
    
    logger.info("\n" + "="*80)
    logger.info("VALIDATION COMPLETE ✓")
    logger.info("="*80)


if __name__ == "__main__":
    main()
