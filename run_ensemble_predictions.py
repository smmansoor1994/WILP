"""
Ensemble Model Quick-Start: Using the combined Stage-1 + Stage-2 pipeline

Run this script to:
  1. Load the trained ARCHON + 4 disease classifiers
  2. Predict on images from the test or validation set
  3. Compare ensemble predictions with ground truth (if available)
  4. Generate performance metrics
"""

import logging
from pathlib import Path
import json
from typing import Dict, List
import numpy as np
from datetime import datetime

from src.inference.disease_ensemble import DiseaseEnsemble, create_ensemble

# ============================================================================
# Configuration
# ============================================================================

CONFIG = {
    # Paths - USE VALIDATION CROPS (have ground truth labels)
    "archon_weights": Path("weights/archon_best.pt"),
    "stage2_models_dir": Path("outputs/disease_validation_option2/stage2_models"),
    "crops_dir": Path("outputs/disease_validation_option2/stage1_crops"),
    "use_crops": True,  # Evaluate on extracted crops with GT labels
    
    # Legacy paths (if use_crops=False)
    "test_images_dir": Path("data/processed/images/train"),  # Use training set
    "labels_dir": Path("data/processed/labels_ext/train"),   # Has labels
    
    # Ensemble weights
    "stage1_weight": 0.3,
    "stage2_weight": 0.7,
    
    # Thresholds
    "thresholds": {
        "has_caries": 0.45,
        "has_deepcaries": 0.45,
        "has_lesion": 0.40,
        "has_impacted": 0.50
    },
    
    # Output
    "output_dir": Path("outputs/ensemble_predictions"),
    "device": "cuda",
}

# ============================================================================
# Setup logging
# ============================================================================

def setup_logging():
    """Configure logging with both console and file output"""
    log_format = "[%(asctime)s] [%(levelname)s] %(name)s — %(message)s"
    log_dir = CONFIG["output_dir"] / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(log_dir / f"ensemble_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
    
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=handlers
    )
    
    return logging.getLogger(__name__)

logger = setup_logging()

# ============================================================================
# Utility Functions
# ============================================================================

def load_ground_truth(label_file: Path) -> Dict[int, Dict[str, int]]:
    """
    Load ground truth disease labels from YOLO format.
    
    Format: class cx cy w h has_caries has_deepcaries has_lesion has_impacted
    """
    gt = {}
    
    if not label_file.exists():
        return gt
    
    try:
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 9:
                    continue
                
                fdi = int(parts[0])
                labels = {
                    'has_caries': int(parts[5]),
                    'has_deepcaries': int(parts[6]),
                    'has_lesion': int(parts[7]),
                    'has_impacted': int(parts[8])
                }
                
                gt[fdi] = labels
    except Exception as e:
        logger.warning(f"Error loading labels from {label_file}: {e}")
    
    return gt


def compare_predictions(
    predictions: Dict[str, bool],
    ground_truth: Dict[str, int]
) -> Dict:
    """
    Compare ensemble predictions with ground truth.
    
    Returns:
        Dictionary with TP, FP, FN, TN counts
    """
    diseases = ['has_caries', 'has_deepcaries', 'has_lesion', 'has_impacted']
    results = {}
    
    for disease in diseases:
        pred = predictions.get(disease, False)
        gt = ground_truth.get(disease, 0)
        
        # Convert to same type
        pred = 1 if pred else 0
        
        if pred == 1 and gt == 1:
            results[disease] = 'TP'
        elif pred == 1 and gt == 0:
            results[disease] = 'FP'
        elif pred == 0 and gt == 1:
            results[disease] = 'FN'
        else:
            results[disease] = 'TN'
    
    return results


def evaluate_ensemble(
    ensemble: DiseaseEnsemble,
    test_images_dir: Path,
    labels_dir: Path,
    sample_size: int = None
) -> Dict:
    """
    Evaluate ensemble on test set.
    
    Args:
        ensemble: DiseaseEnsemble instance
        test_images_dir: Directory with test images
        labels_dir: Directory with ground truth labels
        sample_size: Limit to N images (None for all)
        
    Returns:
        Evaluation report dictionary
    """
    logger.info("="*80)
    logger.info("EVALUATING ENSEMBLE ON TEST SET")
    logger.info("="*80)
    
    # Get test images
    test_images = sorted(test_images_dir.glob("*.png"))
    if sample_size:
        test_images = test_images[:sample_size]
    
    logger.info(f"Testing on {len(test_images)} images")
    
    # Per-disease metrics
    per_disease = {
        disease: {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0}
        for disease in ensemble.diseases
    }
    
    all_predictions = []
    agreement_count = 0
    total_predictions = 0
    
    # Process each test image
    for i, img_path in enumerate(test_images, 1):
        if i % 10 == 0:
            logger.info(f"Progress: {i}/{len(test_images)}")
        
        # Get ensemble predictions
        try:
            results = ensemble.predict_single(img_path)
        except Exception as e:
            logger.warning(f"Error predicting {img_path.name}: {e}")
            continue
        
        if not results:
            continue
        
        # Load ground truth
        label_file = labels_dir / f"{img_path.stem}.txt"
        
        # Process each tooth
        for fdi, pred in results.items():
            gt = load_ground_truth(label_file).get(fdi, {})
            
            # Compare
            comparison = compare_predictions(pred.final_predictions, gt)
            
            # Update metrics
            for disease, result_type in comparison.items():
                per_disease[disease][result_type] += 1
            
            # Track overall agreement with stage-1
            if pred.stage1_predictions == pred.stage2_predictions:
                agreement_count += 1
            
            total_predictions += 1
            
            all_predictions.append({
                'image': img_path.name,
                'fdi': fdi,
                'predictions': pred.final_predictions,
                'confidence': pred.confidence,
                'fused_scores': pred.fused_predictions,
                'ground_truth': gt,
                'comparison': comparison
            })
    
    # Calculate metrics
    logger.info("\n" + "="*80)
    logger.info("EVALUATION RESULTS")
    logger.info("="*80)
    
    metrics = {}
    
    for disease in ensemble.diseases:
        counts = per_disease[disease]
        tp, fp, fn, tn = counts['TP'], counts['FP'], counts['FN'], counts['TN']
        
        # Calculate metrics
        accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[disease] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn,
        }
        
        logger.info(f"\n{disease}:")
        logger.info(f"  Accuracy:  {accuracy:.1%}")
        logger.info(f"  Precision: {precision:.1%}")
        logger.info(f"  Recall:    {recall:.1%}")
        logger.info(f"  F1 Score:  {f1:.3f}")
        logger.info(f"  Confusion: TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    
    # Overall metrics
    logger.info("\n" + "="*80)
    logger.info("OVERALL METRICS")
    logger.info("="*80)
    
    agreement = agreement_count / total_predictions if total_predictions > 0 else 0
    avg_f1 = np.mean([m['f1'] for m in metrics.values()])
    avg_accuracy = np.mean([m['accuracy'] for m in metrics.values()])
    
    logger.info(f"Total predictions: {total_predictions}")
    logger.info(f"Average Accuracy: {avg_accuracy:.1%}")
    logger.info(f"Average F1 Score: {avg_f1:.3f}")
    logger.info(f"Stage-1 agreement: {agreement:.1%}")
    
    return {
        'total_test_images': len(test_images),
        'total_predictions': total_predictions,
        'per_disease_metrics': metrics,
        'average_f1': float(avg_f1),
        'average_accuracy': float(avg_accuracy),
        'stage1_agreement': float(agreement),
        'timestamp': datetime.now().isoformat(),
        'ensemble_weights': {
            'stage1': ensemble.stage1_weight,
            'stage2': ensemble.stage2_weight
        },
        'thresholds': ensemble.thresholds,
        'sample_predictions': all_predictions[:100]  # Save first 100 for review
    }


def predict_and_save(
    ensemble: DiseaseEnsemble,
    images_dir: Path,
    output_path: Path,
    sample_size: int = None
):
    """
    Predict on images and save results to JSON.
    
    Args:
        ensemble: DiseaseEnsemble instance
        images_dir: Directory with images
        output_path: Path to save JSON results
        sample_size: Limit to N images (None for all)
    """
    logger.info("="*80)
    logger.info("RUNNING ENSEMBLE PREDICTIONS")
    logger.info("="*80)
    
    images = sorted(images_dir.glob("*.png"))
    if sample_size:
        images = images[:sample_size]
    
    logger.info(f"Processing {len(images)} images")
    
    all_results = {}
    
    for i, img_path in enumerate(images, 1):
        if i % 10 == 0:
            logger.info(f"Progress: {i}/{len(images)}")
        
        try:
            results = ensemble.predict_single(img_path)
            
            all_results[img_path.name] = {
                'teeth': [
                    {
                        'fdi': fdi,
                        'diseases': pred.final_predictions,
                        'confidence': pred.confidence,
                        'fused_scores': pred.fused_predictions,
                        'stage1_scores': pred.stage1_predictions,
                        'stage2_scores': pred.stage2_predictions
                    }
                    for fdi, pred in sorted(results.items())
                ]
            }
        except Exception as e:
            logger.warning(f"Error processing {img_path.name}: {e}")
    
    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"✓ Predictions saved to {output_path}")
    logger.info(f"  Total images processed: {len(all_results)}")
    
    return all_results


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution workflow"""
    
    logger.info("="*80)
    logger.info("DISEASE ENSEMBLE: STAGE-1 + STAGE-2 FUSION")
    logger.info("="*80)
    
    # Validate paths
    logger.info("\nValidating paths...")
    required_paths = [
        (CONFIG["archon_weights"], "ARCHON weights"),
        (CONFIG["stage2_models_dir"], "Stage-2 models directory"),
    ]
    
    for path, desc in required_paths:
        if not path.exists():
            logger.error(f"✗ {desc} not found: {path}")
            return
    
    logger.info("✓ All paths validated")
    
    # Create output directory
    CONFIG["output_dir"].mkdir(parents=True, exist_ok=True)
    
    # Initialize ensemble
    logger.info("\nInitializing ensemble...")
    ensemble = DiseaseEnsemble(
        archon_weights=CONFIG["archon_weights"],
        stage2_models_dir=CONFIG["stage2_models_dir"],
        device=CONFIG["device"],
        stage1_weight=CONFIG["stage1_weight"],
        stage2_weight=CONFIG["stage2_weight"]
    )
    
    ensemble.set_thresholds(CONFIG["thresholds"])
    logger.info("✓ Ensemble initialized")
    
    # Save configuration
    config_path = CONFIG["output_dir"] / "ensemble_config.json"
    ensemble.save_config(config_path)
    
    # Run predictions
    if CONFIG["test_images_dir"].exists():
        logger.info(f"\nPredicting on test set ({CONFIG['test_images_dir']})")
        predictions_path = CONFIG["output_dir"] / "test_predictions.json"
        predict_and_save(ensemble, CONFIG["test_images_dir"], predictions_path)
        
        # Evaluate if labels available
        if CONFIG["labels_dir"].exists():
            logger.info(f"\nEvaluating with ground truth ({CONFIG['labels_dir']})")
            report = evaluate_ensemble(
                ensemble,
                CONFIG["test_images_dir"],
                CONFIG["labels_dir"]
            )
            
            report_path = CONFIG["output_dir"] / "ensemble_evaluation_report.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"✓ Report saved to {report_path}")
    else:
        logger.warning(f"Test directory not found: {CONFIG['test_images_dir']}")
    
    logger.info("\n" + "="*80)
    logger.info("ENSEMBLE PIPELINE COMPLETE ✓")
    logger.info("="*80)
    logger.info(f"\nOutput directory: {CONFIG['output_dir']}")
    logger.info("\nGenerated files:")
    for f in CONFIG["output_dir"].glob("*"):
        if f.is_file():
            size_mb = f.stat().st_size / (1024*1024)
            logger.info(f"  • {f.name} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
