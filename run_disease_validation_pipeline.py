#!/usr/bin/env python
"""
Quick Start Script for Two-Stage Disease Classification Pipeline

This script demonstrates how to run the complete pipeline with a small sample.
Modify paths and parameters as needed for your setup.

Usage:
    python run_disease_validation_pipeline.py
"""

import logging
from pathlib import Path
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    \"\"\"Run the complete disease validation pipeline.\"\"\"
    
    logger.info("=" * 80)
    logger.info("TWO-STAGE DISEASE CLASSIFICATION PIPELINE - QUICK START".center(80))
    logger.info("=" * 80 + "\\n")
    
    # ─────────────────────────────────────────────────────────────────────────
    # CONFIGURATION
    # ─────────────────────────────────────────────────────────────────────────
    
    # Paths to modify based on your setup
    CONFIG = {
        \"model_weights\": Path(\"weights/archon_best.pt\"),
        \"images_dir\": Path(\"data/processed/images/train\"),
        \"labels_dir\": Path(\"data/processed/labels/train\"),
        \"output_dir\": Path(\"outputs/disease_validation\"),
        \"device\": \"cuda\",  # Change to \"cpu\" if no GPU
        \"epochs\": 30,
        \"sample_images\": None,  # Set to small number (e.g., 50) for testing
        \"batch_size\": 16,
    }
    
    # Validate configuration
    logger.info(\"Validating configuration...\\n\")
    
    if not CONFIG[\"model_weights\"].exists():
        logger.error(f\"Model weights not found: {CONFIG['model_weights']}\")
        logger.error(\"Run training first or download pre-trained weights\")
        return False
    
    if not CONFIG[\"images_dir\"].exists():
        logger.error(f\"Images directory not found: {CONFIG['images_dir']}\")
        return False
    
    if not CONFIG[\"labels_dir\"].exists():
        logger.error(f\"Labels directory not found: {CONFIG['labels_dir']}\")
        return False
    
    logger.info(f\"✓ Model weights: {CONFIG['model_weights']}\")
    logger.info(f\"✓ Images: {CONFIG['images_dir']}\")
    logger.info(f\"✓ Labels: {CONFIG['labels_dir']}\")
    logger.info(f\"✓ Output: {CONFIG['output_dir']}\")
    logger.info(f\"✓ Device: {CONFIG['device']}\\n\")
    
    # ─────────────────────────────────────────────────────────────────────────
    # IMPORT MODULES
    # ─────────────────────────────────────────────────────────────────────────
    
    logger.info(\"Importing modules...\\n\")
    
    try:
        from src.inference.disease_validation import DiseaseValidationPipeline
        logger.info(\"✓ DiseaseValidationPipeline imported\\n\")
    except ImportError as e:
        logger.error(f\"Failed to import modules: {e}\")
        logger.error(\"Ensure src/ is in PYTHONPATH\")
        return False
    
    # ─────────────────────────────────────────────────────────────────────────
    # INITIALIZE PIPELINE
    # ─────────────────────────────────────────────────────────────────────────
    
    logger.info(\"Initializing pipeline...\\n\")
    
    pipeline = DiseaseValidationPipeline(
        model_weights=str(CONFIG[\"model_weights\"]),
        device=CONFIG[\"device\"],
        margin_ratio=0.15,
        batch_size=CONFIG[\"batch_size\"],
    )
    
    # ─────────────────────────────────────────────────────────────────────────
    # RUN FULL PIPELINE
    # ─────────────────────────────────────────────────────────────────────────
    
    logger.info(\"Starting full pipeline...\\n\")
    
    try:\n        result = pipeline.run_full_pipeline(\n            images_dir=CONFIG[\"images_dir\"],\n            labels_dir=CONFIG[\"labels_dir\"],\n            output_dir=CONFIG[\"output_dir\"],\n            epochs=CONFIG[\"epochs\"],\n            sample_images=CONFIG[\"sample_images\"],\n        )\n        \n        logger.info(\"\\n\" + \"=\" * 80)\n        logger.info(\"PIPELINE COMPLETED SUCCESSFULLY\".center(80))\n        logger.info(\"=\" * 80 + \"\\n\")\n        \n        # ───────────────────────────────────────────────────────────────────\n        # PRINT RESULTS SUMMARY\n        # ───────────────────────────────────────────────────────────────────\n        \n        logger.info(\"RESULTS SUMMARY:\")\n        logger.info(\"-\" * 80)\n        \n        # Stage 1 results\n        s1 = result[\"stage_1\"]\n        logger.info(f\"\\nStage 1 - Enumeration & Cropping:\")\n        logger.info(f\"  • Images processed: {s1['images_processed']}\")\n        logger.info(f\"  • Total crops extracted: {s1['total_crops']}\")\n        logger.info(f\"  • Crops with ground truth: {s1['crops_with_gt']}\")\n        logger.info(f\"  • Avg crops per image: {s1['avg_crops_per_image']:.1f}\")\n        \n        # Stage 3 results\n        s3 = result[\"stage_3\"]\n        if s3:\n            logger.info(f\"\\nStage 3 - Validation Results:\")\n            logger.info(f\"  • Agreement with Stage 1: {s3['summary']['agreement_with_stage1']:.2%}\")\n            logger.info(f\"  • Improvement in Recall: {s3['summary']['improvement_recall']:+.2%}\")\n            logger.info(f\"  • Improvement in Precision: {s3['summary']['improvement_precision']:+.2%}\")\n            logger.info(f\"  • Hard examples (FP in stage1): {s3['hard_examples']['false_positives_in_stage1']}\")\n            logger.info(f\"  • Hard examples (FN in stage1): {s3['hard_examples']['false_negatives_in_stage1']}\")\n            \n            # Per-disease metrics\n            logger.info(f\"\\nPer-Disease Metrics:\")\n            for disease_name, metrics in s3[\"per_disease\"].items():\n                logger.info(\n                    f\"  • {disease_name}: \"\n                    f\"Acc={metrics['accuracy']:.3f}, \"\n                    f\"Prec={metrics['precision']:.3f}, \"\n                    f\"Recall={metrics['recall']:.3f}, \"\n                    f\"F1={metrics['f1']:.3f}\"\n                )\n        \n        # Stage 4 recommendations\n        s4 = result[\"stage_4\"]\n        if s4 and \"improvements_recommendations\" in s4:\n            logger.info(f\"\\nTop Recommendations:\")\n            recs = s4[\"improvements_recommendations\"]\n            \n            if recs[\"short_term\"]:\n                logger.info(f\"  Short-term (Immediate):\")\n                for rec in recs[\"short_term\"][:3]:\n                    logger.info(f\"    - {rec['action']}: {rec['reason']}\")\n            \n            if recs[\"medium_term\"]:\n                logger.info(f\"  Medium-term (1-2 weeks):\")\n                for rec in recs[\"medium_term\"][:2]:\n                    logger.info(f\"    - {rec['action']}: {rec.get('suggestion', rec['reason'])}\")\n        \n        logger.info(\"-\" * 80)\n        logger.info(f\"\\nOutput directory: {CONFIG['output_dir']}\")\n        logger.info(f\"Report file: {CONFIG['output_dir'] / 'stage4_validation_report.json'}\")\n        logger.info(\"\\nNext steps:\")\n        logger.info(\"  1. Review stage4_validation_report.json\")\n        logger.info(\"  2. Examine hard examples in stage1_crops/\")\n        logger.info(\"  3. Consider threshold tuning per disease\")\n        logger.info(\"  4. Collect hard examples for retraining\")\n        logger.info(\"  5. Build ensemble: Stage 1 + Stage 2 predictions\")\n        \n        return True\n    \n    except Exception as e:\n        logger.error(f\"Pipeline failed with error: {e}\", exc_info=True)\n        return False\n\n\nif __name__ == \"__main__\":\n    success = main()\n    sys.exit(0 if success else 1)\n