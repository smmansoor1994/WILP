#!/usr/bin/env python
"""
Option 2: Quick Fine-tune on Crops
=====================================
Train lightweight disease classifiers on cropped regions (Stage 2)
without full retraining. Uses existing enumeration model + trains on crops only.

Time: ~45-60 minutes on GPU

Stages:
  1. Extract crops from existing enumeration
  2. Train disease classifiers on crops (light fine-tuning)
  3. Validate & compare stage-1 vs stage-2
  4. Generate improvement report
"""

import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    """Run Option 2: Quick fine-tune on crops."""
    
    logger.info("=" * 80)
    logger.info("OPTION 2: QUICK FINE-TUNE ON CROPS".center(80))
    logger.info("=" * 80)
    logger.info("Using existing enumeration model + train disease classifiers\n")
    
    # ─────────────────────────────────────────────────────────────────────────
    # CONFIGURATION
    # ─────────────────────────────────────────────────────────────────────────
    
    CONFIG = {
        # Existing model (no retraining)
        "model_weights": Path("weights/archon_best.pt"),
        
        # Data paths
        "images_dir": Path("data/processed/images/train"),
        "labels_dir": Path("data/processed/labels_ext/train"),  # Extended labels with disease attributes
        
        # Output directories
        "output_dir": Path("outputs/disease_validation_option2"),
        "crops_dir": Path("outputs/disease_validation_option2/stage1_crops"),
        "models_dir": Path("outputs/disease_validation_option2/stage2_models"),
        "report_path": Path("outputs/disease_validation_option2/stage4_validation_report.json"),
        
        # Training config
        "device": "cuda",  # Use CPU since CUDA not available
        "batch_size": 16,
        "epochs": 50,  # Full training for production models
        "val_split": 0.2,
        "sample_images": None,  # Use ALL training images
    }
    
    logger.info("Configuration:")
    logger.info(f"  • Model: {CONFIG['model_weights']}")
    logger.info(f"  • Images: {CONFIG['images_dir']}")
    logger.info(f"  • Device: {CONFIG['device']}")
    logger.info(f"  • Epochs: {CONFIG['epochs']}")
    logger.info(f"  • Sample images: {CONFIG['sample_images'] or 'ALL (FULL DATASET)'}")
    logger.info(f"  • Output: {CONFIG['output_dir']}\n")
    logger.info("⚠️  RUNNING FULL DATASET - This will take ~2-3 hours on GPU\n")
    
    # Validate configuration
    if not CONFIG["model_weights"].exists():
        logger.error(f"Model not found: {CONFIG['model_weights']}")
        return False
    
    if not CONFIG["images_dir"].exists():
        logger.error(f"Images directory not found: {CONFIG['images_dir']}")
        return False
    
    if not CONFIG["labels_dir"].exists():
        logger.error(f"Labels directory not found: {CONFIG['labels_dir']}")
        return False
    
    logger.info("✓ All paths validated\n")
    
    # ─────────────────────────────────────────────────────────────────────────
    # IMPORT & INITIALIZE
    # ─────────────────────────────────────────────────────────────────────────
    
    logger.info("Importing modules...")
    
    try:
        from src.inference.disease_validation import DiseaseValidationPipeline
        logger.info("✓ DiseaseValidationPipeline imported\n")
    except ImportError as e:
        logger.error(f"Import failed: {e}")
        return False
    
    pipeline = DiseaseValidationPipeline(
        model_weights=str(CONFIG["model_weights"]),
        device=CONFIG["device"],
        margin_ratio=0.15,
        batch_size=CONFIG["batch_size"],
    )
    
    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 1: ENUMERATE & CROP
    # ─────────────────────────────────────────────────────────────────────────
    
    logger.info("STAGE 1: ENUMERATE & CROP")
    logger.info("-" * 80)
    
    try:
        stage_1_stats = pipeline.stage_1_enumerate_and_crop(
            images_dir=CONFIG["images_dir"],
            labels_dir=CONFIG["labels_dir"],
            output_crops_dir=CONFIG["crops_dir"],
            sample_images=CONFIG["sample_images"],
        )
        
        logger.info("\n✓ Stage 1 Complete")
        logger.info(f"  • Images processed: {stage_1_stats['images_processed']}")
        logger.info(f"  • Total crops: {stage_1_stats['total_crops']}")
        logger.info(f"  • Crops with GT: {stage_1_stats['crops_with_gt']}")
        logger.info(f"  • Avg/image: {stage_1_stats['avg_crops_per_image']:.1f}\n")
        
    except Exception as e:
        logger.error(f"Stage 1 failed: {e}", exc_info=True)
        return False
    
    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 2: TRAIN DISEASE CLASSIFIERS
    # ─────────────────────────────────────────────────────────────────────────
    
    logger.info("STAGE 2: TRAIN DISEASE CLASSIFIERS")
    logger.info("-" * 80)
    
    try:
        stage_2_stats = pipeline.stage_2_train_classifiers(
            crops_dir=CONFIG["crops_dir"],
            output_models_dir=CONFIG["models_dir"],
            epochs=CONFIG["epochs"],
            val_split=CONFIG["val_split"],
        )
        
        logger.info("\n✓ Stage 2 Complete")
        logger.info(f"  • Models saved to: {CONFIG['models_dir']}")
        logger.info(f"  • Training samples: {stage_2_stats['train_samples']}")
        logger.info(f"  • Validation samples: {stage_2_stats['val_samples']}")
        logger.info(f"  • Epochs: {stage_2_stats['epochs']}\n")
        
    except Exception as e:
        logger.error(f"Stage 2 failed: {e}", exc_info=True)
        return False
    
    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 3: VALIDATE & COMPARE
    # ─────────────────────────────────────────────────────────────────────────
    
    logger.info("STAGE 3: VALIDATE & COMPARE")
    logger.info("-" * 80)
    
    try:
        stage_3_results = pipeline.stage_3_validate_and_compare(
            model_dir=CONFIG["models_dir"]
        )
        
        logger.info("\n✓ Stage 3 Complete\n")
        
        if stage_3_results:
            summary = stage_3_results.get("summary", {})
            
            logger.info("Validation Summary:")
            logger.info(
                f"  • Agreement with Stage-1: {summary.get('agreement_with_stage1', 0):.2%}"
            )
            logger.info(
                f"  • Improvement in Recall: {summary.get('improvement_recall', 0):+.2%}"
            )
            logger.info(
                f"  • Improvement in Precision: {summary.get('improvement_precision', 0):+.2%}"
            )
            logger.info(
                f"  • Hard FPs in Stage-1: {stage_3_results.get('hard_examples', {}).get('false_positives_in_stage1', 0)}"
            )
            logger.info(
                f"  • Hard FNs in Stage-1: {stage_3_results.get('hard_examples', {}).get('false_negatives_in_stage1', 0)}"
            )
            
            # Per-disease metrics
            logger.info("\nPer-Disease Performance:")
            per_disease = stage_3_results.get("per_disease", {})
            for disease_name, metrics in per_disease.items():
                logger.info(
                    f"  • {disease_name:20s} | "
                    f"Acc={metrics.get('accuracy', 0):.3f} | "
                    f"Prec={metrics.get('precision', 0):.3f} | "
                    f"Recall={metrics.get('recall', 0):.3f} | "
                    f"F1={metrics.get('f1', 0):.3f}"
                )
        
    except Exception as e:
        logger.error(f"Stage 3 failed: {e}", exc_info=True)
        return False
    
    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 4: GENERATE REPORT
    # ─────────────────────────────────────────────────────────────────────────
    
    logger.info("STAGE 4: GENERATE IMPROVEMENT REPORT")
    logger.info("-" * 80)
    
    try:
        stage_4_report = pipeline.stage_4_generate_improvement_report(
            output_path=CONFIG["report_path"]
        )
        
        logger.info("\n✓ Stage 4 Complete\n")
        
        if stage_4_report and "improvements_recommendations" in stage_4_report:
            recs = stage_4_report["improvements_recommendations"]
            
            logger.info("Top Improvement Recommendations:")
            
            if recs.get("short_term"):
                logger.info("\nShort-term (Immediate - Threshold Tuning):")
                for i, rec in enumerate(recs["short_term"][:3], 1):
                    logger.info(
                        f"  {i}. {rec.get('action', 'Action')}")
                    logger.info(
                        f"     Disease: {rec.get('disease', 'N/A')}")
                    logger.info(
                        f"     Reason: {rec.get('reason', 'N/A')}")
                    if 'suggested_threshold' in rec:
                        logger.info(
                            f"     Suggested threshold: {rec['suggested_threshold']:.2f}")
            
            if recs.get("medium_term"):
                logger.info("\nMedium-term (1-2 weeks - Data & Training):")
                for i, rec in enumerate(recs["medium_term"], 1):
                    logger.info(
                        f"  {i}. {rec.get('action', 'Action')}")
                    logger.info(
                        f"     {rec.get('suggestion', rec.get('reason', 'N/A'))}")
            
            if recs.get("long_term"):
                logger.info("\nLong-term (Ensemble & Architecture):")
                for i, rec in enumerate(recs["long_term"], 1):
                    logger.info(
                        f"  {i}. {rec.get('action', 'Action')}")
                    logger.info(
                        f"     {rec.get('suggestion', rec.get('reason', 'N/A'))}")
        
    except Exception as e:
        logger.error(f"Stage 4 failed: {e}", exc_info=True)
        return False
    
    # ─────────────────────────────────────────────────────────────────────────
    # FINAL SUMMARY
    # ─────────────────────────────────────────────────────────────────────────
    
    logger.info("\n" + "=" * 80)
    logger.info("OPTION 2 PIPELINE COMPLETE ✓".center(80))
    logger.info("=" * 80)
    
    logger.info("\nOutput Files:")
    logger.info(f"  • Crops: {CONFIG['crops_dir']}")
    logger.info(f"  • Models: {CONFIG['models_dir']}")
    logger.info(f"  • Report: {CONFIG['report_path']}\n")
    
    logger.info("Next Steps:")
    logger.info("  1. Review stage4_validation_report.json for recommendations")
    logger.info("  2. Check hard examples in stage1_crops/")
    logger.info("  3. Apply threshold tuning suggestions per disease")
    logger.info("  4. Consider collecting more hard examples for retraining")
    logger.info("  5. Build ensemble: combine Stage-1 + Stage-2 predictions")
    logger.info("  6. Validate improvements on held-out test set\n")
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)
