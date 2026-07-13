"""
Disease Cropping and Region Extraction Module

This module extracts disease-relevant regions from detected teeth bounding boxes.
It creates training data for a separate disease classifier by:
  1. Cropping regions around each detected tooth (with expansion margin)
  2. Extracting multiple patches (center + corners) if detailed analysis needed
  3. Associating crops with ground truth disease labels
  4. Preparing dataset for disease-specific classification

Usage:
    from src.inference.disease_cropper import DiseaseCropper
    
    cropper = DiseaseCropper(
        margin_ratio=0.15,  # 15% expansion around tooth bbox
        patch_size=(256, 256)
    )
    
    # Extract crops for a single prediction
    crops = cropper.extract_crops_from_prediction(
        image_path="path/to/xray.jpg",
        teeth_detections=prediction_results,
    )
    
    # Extract crops for training dataset
    dataset = cropper.build_disease_training_dataset(
        images_dir="data/processed/images/train",
        labels_dir="data/processed/labels/train",
        output_dir="data/disease_crops"
    )
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict

import cv2
import numpy as np
from dataclasses import field

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class DiseaseCrop:
    """Single disease crop with metadata."""
    image_id: str                      # e.g., "train_0"
    fdi: int                          # FDI tooth number
    fdi_name: str                      # Human-readable name
    crop_image: np.ndarray             # Cropped image array
    bbox_original: Tuple[float, ...]   # Original [x1, y1, x2, y2] in full image
    bbox_crop: Tuple[int, int, int, int]  # [x1, y1, x2, y2] in crop
    
    # Ground truth disease labels (from annotation)
    has_caries: bool = False
    has_deepcaries: bool = False
    has_lesion: bool = False
    has_impacted: bool = False
    diseases: List[str] = field(default_factory=list)
    
    # Prediction-time labels (from model)
    pred_has_caries: Optional[bool] = None
    pred_has_deepcaries: Optional[bool] = None
    pred_has_lesion: Optional[bool] = None
    pred_has_impacted: Optional[bool] = None
    pred_diseases: List[str] = field(default_factory=list)
    
    # Metadata
    tooth_confidence: float = 0.0      # Detection confidence
    image_height: int = 0
    image_width: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excludes raw image array)."""
        result = {
            "image_id": self.image_id,
            "fdi": self.fdi,
            "fdi_name": self.fdi_name,
            "bbox_original": self.bbox_original,
            "bbox_crop": self.bbox_crop,
            "ground_truth": {
                "has_caries": self.has_caries,
                "has_deepcaries": self.has_deepcaries,
                "has_lesion": self.has_lesion,
                "has_impacted": self.has_impacted,
                "diseases": self.diseases,
            },
            "prediction": {
                "has_caries": self.pred_has_caries,
                "has_deepcaries": self.pred_has_deepcaries,
                "has_lesion": self.pred_has_lesion,
                "has_impacted": self.pred_has_impacted,
                "diseases": self.pred_diseases,
            },
            "tooth_confidence": round(self.tooth_confidence, 4),
            "image_size": {
                "height": self.image_height,
                "width": self.image_width,
            }
        }
        return result


class DiseaseCropper:
    """Extract disease-relevant crops from detected teeth."""
    
    # Disease attribute names
    ATTR_NAMES = ["has_caries", "has_deepcaries", "has_lesion", "has_impacted"]
    
    def __init__(
        self,
        margin_ratio: float = 0.15,
        patch_size: Optional[Tuple[int, int]] = None,
        enable_multi_patch: bool = False,
    ):
        """
        Initialize DiseaseCropper.
        
        Args:
            margin_ratio: Expand bbox by this ratio (e.g., 0.15 = 15% expansion)
            patch_size: Target size for cropped patches. If None, keep original.
            enable_multi_patch: If True, extract center + corner patches
        """
        self.margin_ratio = margin_ratio
        self.patch_size = patch_size
        self.enable_multi_patch = enable_multi_patch
        logger.info(
            f"DiseaseCropper initialized: margin={margin_ratio}, "
            f"size={patch_size}, multi_patch={enable_multi_patch}"
        )
    
    def _expand_bbox(
        self,
        bbox_xyxy: Tuple[float, float, float, float],
        img_h: int,
        img_w: int,
    ) -> Tuple[int, int, int, int]:
        """Expand bounding box by margin_ratio, clipped to image boundaries."""
        x1, y1, x2, y2 = bbox_xyxy
        w, h = x2 - x1, y2 - y1
        
        # Expand by margin_ratio
        expand_w = w * self.margin_ratio
        expand_h = h * self.margin_ratio
        
        x1_exp = max(0, int(x1 - expand_w))
        y1_exp = max(0, int(y1 - expand_h))
        x2_exp = min(img_w - 1, int(x2 + expand_w))
        y2_exp = min(img_h - 1, int(y2 + expand_h))
        
        return x1_exp, y1_exp, x2_exp, y2_exp
    
    def extract_crop(
        self,
        image: np.ndarray,
        bbox_xyxy: Tuple[float, float, float, float],
        expand: bool = True,
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Extract a single crop from image.
        
        Args:
            image: Image array (H, W, 3)
            bbox_xyxy: Bounding box [x1, y1, x2, y2]
            expand: If True, expand bbox by margin_ratio
            
        Returns:
            (crop_image, crop_bbox_xyxy)
        """
        img_h, img_w = image.shape[:2]
        
        if expand:
            x1, y1, x2, y2 = self._expand_bbox(bbox_xyxy, img_h, img_w)
        else:
            x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w - 1, x2), min(img_h - 1, y2)
        
        crop = image[y1:y2+1, x1:x2+1].copy()
        
        # Resize if patch_size specified
        if self.patch_size is not None:
            crop = cv2.resize(crop, self.patch_size, interpolation=cv2.INTER_LINEAR)
        
        return crop, (x1, y1, x2, y2)
    
    def extract_crops_from_prediction(
        self,
        image: np.ndarray,
        teeth_detections: List[Any],  # ToothDetection objects
        include_gt: bool = False,
        image_id: str = "unknown",
    ) -> List[DiseaseCrop]:
        """
        Extract disease crops from a list of tooth detections.
        
        Args:
            image: Input image array (H, W, 3)
            teeth_detections: List of ToothDetection objects
            include_gt: If True, populate ground truth fields
            image_id: Identifier for the image
            
        Returns:
            List of DiseaseCrop objects
        """
        img_h, img_w = image.shape[:2]
        crops = []
        
        for tooth in teeth_detections:
            try:
                # Extract crop
                crop_img, crop_bbox = self.extract_crop(
                    image,
                    tuple(tooth.bbox_xyxy),
                    expand=True
                )
                
                # Create DiseaseCrop object
                dc = DiseaseCrop(
                    image_id=image_id,
                    fdi=tooth.fdi,
                    fdi_name=tooth.fdi_name,
                    crop_image=crop_img,
                    bbox_original=tuple(tooth.bbox_xyxy),
                    bbox_crop=crop_bbox,
                    # Copy current predictions
                    pred_has_caries=tooth.has_caries,
                    pred_has_deepcaries=tooth.has_deepcaries,
                    pred_has_lesion=tooth.has_lesion,
                    pred_has_impacted=tooth.is_impacted,
                    pred_diseases=tooth.diseases.copy() if tooth.diseases else [],
                    tooth_confidence=tooth.conf,
                    image_height=img_h,
                    image_width=img_w,
                )
                
                crops.append(dc)
                
            except Exception as e:
                logger.warning(f"Failed to extract crop for FDI {tooth.fdi}: {e}")
                continue
        
        return crops
    
    def save_crop(
        self,
        crop: DiseaseCrop,
        output_dir: Path,
        include_metadata: bool = True,
    ) -> Path:
        """
        Save a single crop to disk.
        
        Args:
            crop: DiseaseCrop object
            output_dir: Output directory
            include_metadata: If True, also save JSON metadata
            
        Returns:
            Path to saved image
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Filename: {image_id}_FDI{fdi:02d}.jpg
        img_filename = f"{crop.image_id}_FDI{crop.fdi:02d}.jpg"
        img_path = output_dir / img_filename
        
        cv2.imwrite(str(img_path), crop.crop_image)
        logger.debug(f"Saved crop: {img_path}")
        
        if include_metadata:
            meta_filename = f"{crop.image_id}_FDI{crop.fdi:02d}.json"
            meta_path = output_dir / meta_filename
            with open(meta_path, 'w') as f:
                json.dump(crop.to_dict(), f, indent=2)
        
        return img_path
    
    def save_crops_batch(
        self,
        crops: List[DiseaseCrop],
        output_dir: Path,
        include_metadata: bool = True,
        max_crops: Optional[int] = None,
    ) -> Dict[str, int]:
        """
        Save a batch of crops to disk.
        
        Args:
            crops: List of DiseaseCrop objects
            output_dir: Output directory
            include_metadata: If True, also save JSON metadata
            max_crops: Limit number of crops saved (for testing)
            
        Returns:
            Statistics dict with counts
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_count = 0
        error_count = 0
        
        for i, crop in enumerate(crops):
            if max_crops and i >= max_crops:
                break
            
            try:
                self.save_crop(crop, output_dir, include_metadata)
                saved_count += 1
            except Exception as e:
                logger.error(f"Error saving crop {i}: {e}")
                error_count += 1
        
        stats = {
            "saved": saved_count,
            "errors": error_count,
            "total": len(crops),
        }
        
        logger.info(
            f"Saved {saved_count}/{len(crops)} crops "
            f"({error_count} errors)"
        )
        
        return stats
    
    def load_crops_from_dir(
        self,
        crop_dir: Path,
        load_images: bool = True,
    ) -> List[DiseaseCrop]:
        """
        Load crops and metadata from directory.
        
        Args:
            crop_dir: Directory containing crops and .json metadata files
            load_images: If True, load image arrays; else set to None
            
        Returns:
            List of DiseaseCrop objects
        """
        crop_dir = Path(crop_dir)
        crops = []
        
        json_files = sorted(crop_dir.glob("*.json"))
        
        for json_path in json_files:
            try:
                with open(json_path, 'r') as f:
                    meta = json.load(f)
                
                # Load image if requested
                img_path = json_path.with_suffix('.jpg')
                if load_images and img_path.exists():
                    crop_img = cv2.imread(str(img_path))
                else:
                    crop_img = np.zeros((256, 256, 3), dtype=np.uint8)
                
                # Reconstruct DiseaseCrop
                dc = DiseaseCrop(
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
                    diseases=meta['ground_truth']['diseases'],
                    pred_has_caries=meta['prediction']['has_caries'],
                    pred_has_deepcaries=meta['prediction']['has_deepcaries'],
                    pred_has_lesion=meta['prediction']['has_lesion'],
                    pred_has_impacted=meta['prediction']['has_impacted'],
                    pred_diseases=meta['prediction']['diseases'],
                    tooth_confidence=meta['tooth_confidence'],
                    image_height=meta['image_size']['height'],
                    image_width=meta['image_size']['width'],
                )
                
                crops.append(dc)
                
            except Exception as e:
                logger.error(f"Error loading crop {json_path}: {e}")
                continue
        
        logger.info(f"Loaded {len(crops)} crops from {crop_dir}")
        return crops
