"""
Validate model on training images with ground truth comparison
Compares model predictions against DENTEX ground truth annotations
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Set
from src.inference.predictor import ARCHONHybridPredictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("validation")

# Test images
TEST_IMAGES = [
    "train_165.png",
    "train_127.png", 
    "train_645.png",
    "train_9.png",
    "train_546.png",
    "train_185.png",
    "train_373.png",
    "train_192.png",
    "train_133.png",
    "train_201.png",
]

TEST_IMAGES = [f"train_{i}.png" for i in range(501)]

IMG_DIR = Path(r"D:\WILP\sem-4\Dataset\DENTEX\DENTEX\training_data\quadrant-enumeration-disease\xrays")
GT_JSON = Path(r"D:\WILP\sem-4\Dataset\DENTEX\DENTEX\training_data\quadrant-enumeration-disease\train_quadrant_enumeration_disease.json")
WEIGHTS = Path(r"D:\WILP\Workingcode\models\archon-100-main-hybrid\weights_20260704_163459\content\WILP\weights\archon_best.pt")
OUTPUT_DIR = Path(r"D:\WILP\validation_test")

# FDI conversion from DENTEX format
# category_id_1 (0-3): quadrant (UR, UL, LL, LR)
# category_id_2 (0-7): tooth position within quadrant
# FDI = 10 * (quadrant + 1) + (tooth + 1)
def dentex_to_fdi(quad_id: int, tooth_id: int) -> int:
    """Convert DENTEX quadrant/tooth to FDI number."""
    return 10 * (quad_id + 1) + (tooth_id + 1)

# Disease names
DISEASE_NAMES = {
    0: "Impacted",
    1: "Caries",
    2: "Periapical Lesion",
    3: "Deep Caries",
}


def load_ground_truth(gt_json_path: Path) -> Dict[str, List[Dict]]:
    """Load ground truth annotations from DENTEX JSON."""
    with open(gt_json_path) as f:
        data = json.load(f)
    
    # Map image names to their annotations
    image_map = {img["file_name"]: img["id"] for img in data["images"]}
    
    gt_by_image = {}
    for img_name in TEST_IMAGES:
        gt_by_image[img_name] = []
        if img_name not in image_map:
            continue
        
        img_id = image_map[img_name]
        for ann in data["annotations"]:
            if ann["image_id"] == img_id:
                quad_id = ann.get("category_id_1", -1)
                tooth_id = ann.get("category_id_2", -1)
                disease = ann.get("category_id_3", -1)
                
                # Convert DENTEX quadrant/tooth to FDI
                if quad_id >= 0 and tooth_id >= 0:
                    fdi = dentex_to_fdi(quad_id, tooth_id)
                else:
                    fdi = -1
                
                gt_by_image[img_name].append({
                    "fdi": fdi,
                    "disease": disease,
                    "bbox": ann["bbox"],
                })
    
    return gt_by_image


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load ground truth
    logger.info("Loading ground truth from %s", GT_JSON)
    gt_by_image = load_ground_truth(GT_JSON)
    
    predictor = ARCHONHybridPredictor(
        weights_path=str(WEIGHTS),
        device="cpu",
        conf_threshold=0.05,
        iou_threshold=0.45,
        attr_threshold=0.1,
    )
    
    results = []
    
    # Aggregate metrics
    total_gt_teeth = 0
    total_pred_teeth = 0
    total_tp_teeth = 0  # True positive teeth
    total_fp_teeth = 0  # False positive teeth
    total_fn_teeth = 0  # False negative teeth
    
    total_gt_diseases = 0
    total_pred_diseases = 0
    total_tp_diseases = 0
    total_fp_diseases = 0
    total_fn_diseases = 0
    
    print("\n" + "="*100)
    print("VALIDATION WITH GROUND TRUTH COMPARISON".center(100))
    print("="*100 + "\n")
    
    for img_name in TEST_IMAGES:
        img_path = IMG_DIR / img_name
        if not img_path.exists():
            logger.warning(f"Image not found: {img_name}")
            continue
        
        # Get ground truth
        gt_teeth = gt_by_image.get(img_name, [])
        gt_fdi_set = {t["fdi"] for t in gt_teeth}
        gt_disease_set = {(t["fdi"], DISEASE_NAMES.get(t["disease"], "Unknown")) 
                          for t in gt_teeth if t["disease"] >= 0}
        
        logger.info(f"Processing: {img_name}")
        pred_results = predictor.predict(input_path=str(img_path), save_vis=False, save_json=False)
        pred_teeth = pred_results.get(img_name, [])
        
        pred_fdi_set = {t.fdi for t in pred_teeth}
        pred_disease_set = set()
        for t in pred_teeth:
            for disease_name in t.diseases:
                pred_disease_set.add((t.fdi, disease_name))
        
        # Calculate tooth detection metrics
        tp_teeth = pred_fdi_set & gt_fdi_set
        fp_teeth = pred_fdi_set - gt_fdi_set
        fn_teeth = gt_fdi_set - pred_fdi_set
        
        # Calculate disease metrics
        tp_diseases = pred_disease_set & gt_disease_set
        fp_diseases = pred_disease_set - gt_disease_set
        fn_diseases = gt_disease_set - pred_disease_set
        
        # Update aggregates
        total_gt_teeth += len(gt_teeth)
        total_pred_teeth += len(pred_teeth)
        total_tp_teeth += len(tp_teeth)
        total_fp_teeth += len(fp_teeth)
        total_fn_teeth += len(fn_teeth)
        
        total_gt_diseases += len(gt_disease_set)
        total_pred_diseases += len(pred_disease_set)
        total_tp_diseases += len(tp_diseases)
        total_fp_diseases += len(fp_diseases)
        total_fn_diseases += len(fn_diseases)
        
        # Store result
        result = {
            "image": img_name,
            "gt_teeth": len(gt_teeth),
            "pred_teeth": len(pred_teeth),
            "tp_teeth": len(tp_teeth),
            "fp_teeth": len(fp_teeth),
            "fn_teeth": len(fn_teeth),
            "gt_diseases": len(gt_disease_set),
            "pred_diseases": len(pred_disease_set),
            "tp_diseases": len(tp_diseases),
            "fp_diseases": len(fp_diseases),
            "fn_diseases": len(fn_diseases),
        }
        results.append(result)
        
        # Print per-image summary
        print(f"\n{img_name:25}")
        print("-" * 100)
        print(f"  Ground Truth: {len(gt_teeth):2} teeth | {len(gt_disease_set):2} diseases")
        print(f"  Predictions:  {len(pred_teeth):2} teeth | {len(pred_disease_set):2} diseases")
        print(f"  Teeth Detection: TP={len(tp_teeth):<2} FP={len(fp_teeth):<2} FN={len(fn_teeth):<2}", end="")
        if len(gt_teeth) > 0:
            recall = len(tp_teeth) / len(gt_teeth) * 100
            precision = len(tp_teeth) / max(1, len(pred_teeth)) * 100
            print(f" | Recall={recall:5.1f}% Precision={precision:5.1f}%")
        else:
            print()
        
        print(f"  Disease Detection: TP={len(tp_diseases):<2} FP={len(fp_diseases):<2} FN={len(fn_diseases):<2}", end="")
        if len(gt_disease_set) > 0:
            recall = len(tp_diseases) / len(gt_disease_set) * 100
            precision = len(tp_diseases) / max(1, len(pred_disease_set)) * 100
            print(f" | Recall={recall:5.1f}% Precision={precision:5.1f}%")
        else:
            print()
        
        # Show ground truth teeth
        if gt_teeth:
            gt_fdi_str = ', '.join([f"FDI{t['fdi']}" for t in gt_teeth])
            print(f"  Ground Truth Teeth: {gt_fdi_str}")
        
        # Show detected teeth
        if pred_teeth:
            pred_fdi_str = ', '.join([f"FDI{t.fdi}" for t in sorted(pred_teeth, key=lambda x: x.fdi)])
            print(f"  Detected Teeth:     {pred_fdi_str}")
        
        # Show missed teeth
        if fn_teeth:
            missed_str = ', '.join([f"FDI{fdi}" for fdi in sorted(fn_teeth)])
            print(f"  ❌ MISSED TEETH:     {missed_str}")
        
        # Show false positives
        if fp_teeth:
            fp_str = ', '.join([f"FDI{fdi}" for fdi in sorted(fp_teeth)])
            print(f"  ⚠️  FALSE POSITIVES:  {fp_str}")
        
        # Show disease mismatches
        if fn_diseases:
            missed_dis = ', '.join([f"FDI{fdi}:{disease}" for fdi, disease in sorted(fn_diseases)])
            print(f"  ❌ MISSED DISEASES:  {missed_dis}")
        
        if fp_diseases:
            fp_dis = ', '.join([f"FDI{fdi}:{disease}" for fdi, disease in sorted(fp_diseases)])
            print(f"  ⚠️  FALSE DISEASES:   {fp_dis}")
    
    # Calculate aggregate metrics
    print("\n" + "="*100)
    print("AGGREGATE METRICS".center(100))
    print("="*100)
    
    print(f"\n📊 TOOTH DETECTION METRICS")
    print(f"  Ground Truth Teeth:  {total_gt_teeth}")
    print(f"  Predicted Teeth:     {total_pred_teeth}")
    print(f"  True Positives (TP): {total_tp_teeth}")
    print(f"  False Positives:     {total_fp_teeth}")
    print(f"  False Negatives:     {total_fn_teeth}")
    
    if total_gt_teeth > 0:
        tooth_recall = total_tp_teeth / total_gt_teeth * 100
        print(f"  Recall (Sensitivity): {tooth_recall:.1f}% ({total_tp_teeth}/{total_gt_teeth})")
    
    if total_pred_teeth > 0:
        tooth_precision = total_tp_teeth / total_pred_teeth * 100
        print(f"  Precision: {tooth_precision:.1f}% ({total_tp_teeth}/{total_pred_teeth})")
    
    print(f"\n🦷 DISEASE DETECTION METRICS")
    print(f"  Ground Truth Diseases: {total_gt_diseases}")
    print(f"  Predicted Diseases:    {total_pred_diseases}")
    print(f"  True Positives (TP):   {total_tp_diseases}")
    print(f"  False Positives:       {total_fp_diseases}")
    print(f"  False Negatives:       {total_fn_diseases}")
    
    if total_gt_diseases > 0:
        disease_recall = total_tp_diseases / total_gt_diseases * 100
        print(f"  Recall (Sensitivity): {disease_recall:.1f}% ({total_tp_diseases}/{total_gt_diseases})")
    
    if total_pred_diseases > 0:
        disease_precision = total_tp_diseases / total_pred_diseases * 100
        print(f"  Precision: {disease_precision:.1f}% ({total_tp_diseases}/{total_pred_diseases})")
    
    print("\n" + "="*100 + "\n")
    
    # Save detailed results
    json_path = OUTPUT_DIR / "validation_with_gt_results.json"
    with open(json_path, "w") as f:
        json.dump({
            "summary": {
                "total_images": len(TEST_IMAGES),
                "total_gt_teeth": total_gt_teeth,
                "total_pred_teeth": total_pred_teeth,
                "total_tp_teeth": total_tp_teeth,
                "total_fp_teeth": total_fp_teeth,
                "total_fn_teeth": total_fn_teeth,
                "tooth_recall": round(total_tp_teeth / max(1, total_gt_teeth) * 100, 2),
                "tooth_precision": round(total_tp_teeth / max(1, total_pred_teeth) * 100, 2),
                "total_gt_diseases": total_gt_diseases,
                "total_pred_diseases": total_pred_diseases,
                "total_tp_diseases": total_tp_diseases,
                "total_fp_diseases": total_fp_diseases,
                "total_fn_diseases": total_fn_diseases,
                "disease_recall": round(total_tp_diseases / max(1, total_gt_diseases) * 100, 2),
                "disease_precision": round(total_tp_diseases / max(1, total_pred_diseases) * 100, 2),
            },
            "results": results
        }, f, indent=2)
    
    logger.info(f"Results saved to: {json_path}")


if __name__ == "__main__":
    main()
