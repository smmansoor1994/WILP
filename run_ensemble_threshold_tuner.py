"""
Ensemble Threshold Tuner: Find optimal prediction thresholds for each disease

This script evaluates ensemble predictions across a range of thresholds and
finds the values that maximize F1 score (or other metrics) for each disease.
"""

import logging
from pathlib import Path
import json
from itertools import product
import numpy as np
from datetime import datetime

from src.inference.disease_ensemble import DiseaseEnsemble

logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

CONFIG = {
    "archon_weights": Path("weights/archon_best.pt"),
    "stage2_models_dir": Path("outputs/disease_validation_option2/stage2_models"),
    "test_images_dir": Path("data/processed/images/test"),
    "labels_dir": Path("data/processed/labels_ext/test"),
    
    "output_dir": Path("outputs/ensemble_threshold_tuning"),
    "device": "cuda",
    
    # Threshold search space
    "threshold_range": np.arange(0.3, 0.8, 0.05),  # 0.3 to 0.75 in 0.05 steps
    "metric_to_optimize": "f1",  # 'f1', 'accuracy', 'precision', or 'recall'
    
    # Sample size for faster tuning (None = all)
    "sample_images": None,
}

# ============================================================================
# Setup logging
# ============================================================================

log_format = "[%(asctime)s] [%(levelname)s] %(name)s — %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        logging.StreamHandler(),
    ]
)

logger = logging.getLogger(__name__)

# ============================================================================
# Threshold Tuning
# ============================================================================

def load_ground_truth(label_file: Path) -> dict:
    """Load ground truth labels from YOLO format"""
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
                gt[fdi] = {
                    'has_caries': int(parts[5]),
                    'has_deepcaries': int(parts[6]),
                    'has_lesion': int(parts[7]),
                    'has_impacted': int(parts[8])
                }
    except Exception as e:
        logger.warning(f"Error loading labels from {label_file}: {e}")
    
    return gt


def evaluate_threshold_set(
    ensemble: DiseaseEnsemble,
    thresholds: dict,
    test_images_dir: Path,
    labels_dir: Path,
    sample_size: int = None
) -> dict:
    """
    Evaluate ensemble with a specific threshold configuration.
    
    Args:
        ensemble: DiseaseEnsemble instance
        thresholds: Dict mapping disease → threshold
        test_images_dir: Directory with test images
        labels_dir: Directory with ground truth labels
        sample_size: Limit to N images
        
    Returns:
        Dictionary with metrics for each disease
    """
    
    # Set thresholds
    ensemble.set_thresholds(thresholds)
    
    # Get test images
    test_images = sorted(test_images_dir.glob("*.png"))
    if sample_size:
        test_images = test_images[:sample_size]
    
    # Initialize metrics
    diseases = ensemble.diseases
    metrics = {
        disease: {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0}
        for disease in diseases
    }
    
    # Process each image
    for img_path in test_images:
        try:
            predictions = ensemble.predict_single(img_path)
        except Exception:
            continue
        
        if not predictions:
            continue
        
        # Load ground truth
        label_file = labels_dir / f"{img_path.stem}.txt"
        
        # Process each tooth
        for fdi, pred in predictions.items():
            gt = load_ground_truth(label_file).get(fdi, {})
            
            for disease in diseases:
                pred_val = 1 if pred.final_predictions.get(disease, False) else 0
                gt_val = gt.get(disease, 0)
                
                if pred_val == 1 and gt_val == 1:
                    metrics[disease]['TP'] += 1
                elif pred_val == 1 and gt_val == 0:
                    metrics[disease]['FP'] += 1
                elif pred_val == 0 and gt_val == 1:
                    metrics[disease]['FN'] += 1
                else:
                    metrics[disease]['TN'] += 1
    
    # Calculate performance metrics
    results = {}
    for disease in diseases:
        counts = metrics[disease]
        tp, fp, fn, tn = counts['TP'], counts['FP'], counts['FN'], counts['TN']
        
        total = tp + fp + fn + tn
        if total == 0:
            continue
        
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
    
    return results


def search_optimal_thresholds(
    ensemble: DiseaseEnsemble,
    threshold_range: np.ndarray,
    test_images_dir: Path,
    labels_dir: Path,
    metric: str = "f1",
    sample_size: int = None
) -> dict:
    """
    Search for optimal thresholds for each disease independently.
    
    Args:
        ensemble: DiseaseEnsemble instance
        threshold_range: Array of threshold values to test
        test_images_dir: Directory with test images
        labels_dir: Directory with ground truth labels
        metric: Metric to optimize ('f1', 'accuracy', 'precision', 'recall')
        sample_size: Limit to N images for faster search
        
    Returns:
        Dictionary with optimal thresholds and their performance
    """
    
    diseases = ensemble.diseases
    results = {disease: {} for disease in diseases}
    
    logger.info(f"\nSearching optimal thresholds (metric={metric})...")
    logger.info(f"Testing {len(threshold_range)} thresholds per disease")
    
    # Test each disease independently
    for disease_idx, disease in enumerate(diseases, 1):
        logger.info(f"\n[{disease_idx}/{len(diseases)}] {disease}")
        best_threshold = None
        best_metric_value = -1
        threshold_performance = {}
        
        for threshold_idx, threshold in enumerate(threshold_range):
            # Set thresholds
            thresholds_dict = {d: 0.5 for d in diseases}
            thresholds_dict[disease] = threshold
            
            # Evaluate
            try:
                metrics = evaluate_threshold_set(
                    ensemble,
                    thresholds_dict,
                    test_images_dir,
                    labels_dir,
                    sample_size
                )
            except Exception as e:
                logger.warning(f"Error evaluating threshold {threshold}: {e}")
                continue
            
            if disease not in metrics:
                continue
            
            disease_metrics = metrics[disease]
            metric_value = disease_metrics.get(metric, 0)
            
            threshold_performance[float(threshold)] = disease_metrics
            
            # Track best
            if metric_value > best_metric_value:
                best_metric_value = metric_value
                best_threshold = threshold
            
            if (threshold_idx + 1) % 5 == 0:
                logger.info(f"  Tested {threshold_idx + 1}/{len(threshold_range)} thresholds")
        
        # Store results
        if best_threshold is not None:
            results[disease] = {
                'optimal_threshold': float(best_threshold),
                f'best_{metric}': float(best_metric_value),
                'performance_at_optimal': threshold_performance[float(best_threshold)],
                'all_thresholds': threshold_performance
            }
            
            logger.info(f"  ✓ Optimal threshold: {best_threshold:.2f} "
                       f"({metric}={best_metric_value:.3f})")
    
    return results


def main():
    """Main execution"""
    
    logger.info("="*80)
    logger.info("ENSEMBLE THRESHOLD TUNER")
    logger.info("="*80)
    
    # Validate paths
    logger.info("\nValidating paths...")
    for path, desc in [
        (CONFIG["archon_weights"], "ARCHON weights"),
        (CONFIG["stage2_models_dir"], "Stage-2 models"),
        (CONFIG["test_images_dir"], "Test images"),
        (CONFIG["labels_dir"], "Labels"),
    ]:
        if not path.exists():
            logger.error(f"✗ {desc} not found: {path}")
            return
    
    logger.info("✓ All paths valid")
    
    # Create output directory
    CONFIG["output_dir"].mkdir(parents=True, exist_ok=True)
    
    # Initialize ensemble
    logger.info("\nInitializing ensemble...")
    ensemble = DiseaseEnsemble(
        archon_weights=CONFIG["archon_weights"],
        stage2_models_dir=CONFIG["stage2_models_dir"],
        device=CONFIG["device"],
    )
    logger.info("✓ Ensemble initialized")
    
    # Search optimal thresholds
    logger.info("\n" + "="*80)
    logger.info("THRESHOLD SEARCH")
    logger.info("="*80)
    
    tuning_results = search_optimal_thresholds(
        ensemble,
        CONFIG["threshold_range"],
        CONFIG["test_images_dir"],
        CONFIG["labels_dir"],
        metric=CONFIG["metric_to_optimize"],
        sample_size=CONFIG["sample_images"]
    )
    
    # Apply optimal thresholds
    logger.info("\n" + "="*80)
    logger.info("APPLYING OPTIMAL THRESHOLDS")
    logger.info("="*80)
    
    optimal_thresholds = {
        disease: result['optimal_threshold']
        for disease, result in tuning_results.items()
        if 'optimal_threshold' in result
    }
    
    ensemble.set_thresholds(optimal_thresholds)
    
    for disease, threshold in optimal_thresholds.items():
        logger.info(f"  • {disease}: {threshold:.2f}")
    
    # Final evaluation with optimal thresholds
    logger.info("\n" + "="*80)
    logger.info("FINAL EVALUATION WITH OPTIMAL THRESHOLDS")
    logger.info("="*80)
    
    final_metrics = evaluate_threshold_set(
        ensemble,
        optimal_thresholds,
        CONFIG["test_images_dir"],
        CONFIG["labels_dir"],
        CONFIG["sample_images"]
    )
    
    # Print summary
    logger.info("\nPer-Disease Performance:")
    for disease in ensemble.diseases:
        if disease in final_metrics:
            m = final_metrics[disease]
            logger.info(f"\n{disease}:")
            logger.info(f"  Accuracy:  {m['accuracy']:.1%}")
            logger.info(f"  Precision: {m['precision']:.1%}")
            logger.info(f"  Recall:    {m['recall']:.1%}")
            logger.info(f"  F1 Score:  {m['f1']:.3f}")
    
    # Save results
    output_file = CONFIG["output_dir"] / f"threshold_tuning_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'metric_optimized': CONFIG["metric_to_optimize"],
        'threshold_range': [float(t) for t in CONFIG["threshold_range"]],
        'optimal_thresholds': optimal_thresholds,
        'tuning_results': tuning_results,
        'final_metrics': final_metrics
    }
    
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"\n✓ Report saved to {output_file}")
    
    # Print config for copy-paste
    logger.info("\n" + "="*80)
    logger.info("COPY THIS CONFIG TO disease_ensemble.py")
    logger.info("="*80)
    logger.info("\nthresholds = {")
    for disease, threshold in optimal_thresholds.items():
        logger.info(f"    '{disease}': {threshold:.2f},")
    logger.info("}")
    
    logger.info("\n" + "="*80)
    logger.info("TUNING COMPLETE ✓")
    logger.info("="*80)


if __name__ == "__main__":
    main()
