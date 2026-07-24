"""
Validate model on training images with ground truth comparison
Compares model predictions against DENTEX ground truth annotations
"""
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Set
from src.inference.predictor import ARCHONHybridPredictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("validation")
THREAD_LOCAL = threading.local()

# Test images
TEST_IMAGES = [f"train_{i}.png" for i in range(700)]

IMG_DIR = Path(r"D:\WILP\sem-4\Dataset\DENTEX\DENTEX\training_data\quadrant-enumeration-disease\xrays")
GT_JSON = Path(r"D:\WILP\sem-4\Dataset\DENTEX\DENTEX\training_data\quadrant-enumeration-disease\train_quadrant_enumeration_disease.json")
WEIGHTS = Path(r"D:\WILP\Workingcode\Baseline\WILP\weights\archon_best.pt")
OUTPUT_DIR = Path(r"D:\WILP\validation_test")

LINE_WIDTH = 100

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


def format_banner(title: str) -> List[str]:
    return ["", "=" * LINE_WIDTH, title.center(LINE_WIDTH), "=" * LINE_WIDTH, ""]


def format_image_block(
    img_name: str,
    gt_teeth: List[Dict],
    gt_disease_set: Set,
    pred_teeth,
    pred_disease_set: Set,
    tp_teeth: Set,
    fp_teeth: Set,
    fn_teeth: Set,
    tp_diseases: Set,
    fp_diseases: Set,
    fn_diseases: Set,
) -> List[str]:
    lines = [
        "",
        img_name,
        "-" * LINE_WIDTH,
        f"  Ground Truth: {len(gt_teeth):2} teeth | {len(gt_disease_set):2} diseases",
        f"  Predictions:  {len(pred_teeth):2} teeth | {len(pred_disease_set):2} diseases",
        f"  Teeth Detection: TP={len(tp_teeth):<2} FP={len(fp_teeth):<2} FN={len(fn_teeth):<2}",
    ]

    if len(gt_teeth) > 0:
        recall = len(tp_teeth) / len(gt_teeth) * 100
        precision = len(tp_teeth) / max(1, len(pred_teeth)) * 100
        lines[-1] += f"  | Recall={recall:5.1f}% Precision={precision:5.1f}%"

    lines.append(f"  Disease Detection: TP={len(tp_diseases):<2} FP={len(fp_diseases):<2} FN={len(fn_diseases):<2}")
    if len(gt_disease_set) > 0:
        recall = len(tp_diseases) / len(gt_disease_set) * 100
        precision = len(tp_diseases) / max(1, len(pred_disease_set)) * 100
        lines[-1] += f"  | Recall={recall:5.1f}% Precision={precision:5.1f}%"

    if gt_teeth:
        gt_fdi_str = ", ".join([f"FDI{t['fdi']}" for t in gt_teeth])
        lines.append(f"  Ground Truth Teeth: {gt_fdi_str}")

    if pred_teeth:
        pred_fdi_str = ", ".join([f"FDI{t.fdi}" for t in sorted(pred_teeth, key=lambda x: x.fdi)])
        lines.append(f"  Detected Teeth:     {pred_fdi_str}")

    if fn_teeth:
        missed_str = ", ".join([f"FDI{fdi}" for fdi in sorted(fn_teeth)])
        lines.append(f"  MISSED TEETH:       {missed_str}")

    if fp_teeth:
        fp_str = ", ".join([f"FDI{fdi}" for fdi in sorted(fp_teeth)])
        lines.append(f"  FALSE POSITIVES:    {fp_str}")

    if fn_diseases:
        missed_dis = ", ".join([f"FDI{fdi}:{disease}" for fdi, disease in sorted(fn_diseases)])
        lines.append(f"  MISSED DISEASES:    {missed_dis}")

    if fp_diseases:
        fp_dis = ", ".join([f"FDI{fdi}:{disease}" for fdi, disease in sorted(fp_diseases)])
        lines.append(f"  FALSE DISEASES:     {fp_dis}")

    return lines


def format_overall_block(
    total_gt_teeth: int,
    total_pred_teeth: int,
    total_tp_teeth: int,
    total_fp_teeth: int,
    total_fn_teeth: int,
    total_gt_diseases: int,
    total_pred_diseases: int,
    total_tp_diseases: int,
    total_fp_diseases: int,
    total_fn_diseases: int,
) -> List[str]:
    tooth_recall = total_tp_teeth / max(1, total_gt_teeth) * 100
    tooth_precision = total_tp_teeth / max(1, total_pred_teeth) * 100
    disease_recall = total_tp_diseases / max(1, total_gt_diseases) * 100
    disease_precision = total_tp_diseases / max(1, total_pred_diseases) * 100

    return [
        "",
        "=" * LINE_WIDTH,
        "AGGREGATE METRICS".center(LINE_WIDTH),
        "=" * LINE_WIDTH,
        "",
        "📊 TOOTH DETECTION METRICS",
        f"  Ground Truth Teeth:  {total_gt_teeth}",
        f"  Predicted Teeth:     {total_pred_teeth}",
        f"  True Positives (TP): {total_tp_teeth}",
        f"  False Positives:     {total_fp_teeth}",
        f"  False Negatives:     {total_fn_teeth}",
        f"  Recall (Sensitivity): {tooth_recall:.1f}% ({total_tp_teeth}/{total_gt_teeth})",
        f"  Precision: {tooth_precision:.1f}% ({total_tp_teeth}/{total_pred_teeth})",
        "",
        "🦷 DISEASE DETECTION METRICS",
        f"  Ground Truth Diseases: {total_gt_diseases}",
        f"  Predicted Diseases:    {total_pred_diseases}",
        f"  True Positives (TP):   {total_tp_diseases}",
        f"  False Positives:       {total_fp_diseases}",
        f"  False Negatives:       {total_fn_diseases}",
        f"  Recall (Sensitivity): {disease_recall:.1f}% ({total_tp_diseases}/{total_gt_diseases})",
        f"  Precision: {disease_precision:.1f}% ({total_tp_diseases}/{total_pred_diseases})",
        "",
        "=" * LINE_WIDTH,
        "",
    ]


def format_threshold_tag(conf_threshold: float, attr_threshold: float, iou_threshold: float) -> str:
    return (
        f"conf{conf_threshold:.2f}"
        f"_attr{attr_threshold:.2f}"
        f"_iou{iou_threshold:.2f}"
    )


def get_predictor() -> ARCHONHybridPredictor:
    predictor = getattr(THREAD_LOCAL, "predictor", None)
    if predictor is None:
        predictor = ARCHONHybridPredictor(
            weights_path=str(WEIGHTS),
            device="cpu",
            conf_threshold=0.15,
            iou_threshold=0.45,
            attr_threshold=0.08,
        )
        THREAD_LOCAL.predictor = predictor
    return predictor


def process_image(img_name: str) -> Dict:
    img_path = IMG_DIR / img_name
    if not img_path.exists():
        return {
            "image": img_name,
            "missing": True,
        }

    gt_teeth = GT_BY_IMAGE_CACHE.get(img_name, [])
    gt_fdi_set = {t["fdi"] for t in gt_teeth}
    gt_disease_set = {
        (t["fdi"], DISEASE_NAMES.get(t["disease"], "Unknown"))
        for t in gt_teeth
        if t["disease"] >= 0
    }

    logger.info("Processing: %s", img_name)
    pred_results = get_predictor().predict(input_path=str(img_path), save_vis=False, save_json=False)
    pred_teeth = pred_results.get(img_name, [])

    pred_fdi_set = {t.fdi for t in pred_teeth}
    pred_disease_set = set()
    for t in pred_teeth:
        for disease_name in t.diseases:
            pred_disease_set.add((t.fdi, disease_name))

    tp_teeth = pred_fdi_set & gt_fdi_set
    fp_teeth = pred_fdi_set - gt_fdi_set
    fn_teeth = gt_fdi_set - pred_fdi_set

    tp_diseases = pred_disease_set & gt_disease_set
    fp_diseases = pred_disease_set - gt_disease_set
    fn_diseases = gt_disease_set - pred_disease_set

    image_lines = format_image_block(
        img_name,
        gt_teeth,
        gt_disease_set,
        pred_teeth,
        pred_disease_set,
        tp_teeth,
        fp_teeth,
        fn_teeth,
        tp_diseases,
        fp_diseases,
        fn_diseases,
    )

    return {
        "image": img_name,
        "missing": False,
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
        "log_lines": image_lines,
        "ground_truth_teeth": [
            {"fdi": tooth["fdi"], "disease": tooth["disease"], "bbox": tooth["bbox"]}
            for tooth in gt_teeth
        ],
        "predicted_teeth": [
            {
                "fdi": tooth.fdi,
                "diseases": list(tooth.diseases),
                "score": getattr(tooth, "score", None),
            }
            for tooth in sorted(pred_teeth, key=lambda x: x.fdi)
        ],
        "ground_truth_diseases": [
            {"fdi": fdi, "disease": disease}
            for fdi, disease in sorted(gt_disease_set)
        ],
        "predicted_diseases": [
            {"fdi": fdi, "disease": disease}
            for fdi, disease in sorted(pred_disease_set)
        ],
    }


GT_BY_IMAGE_CACHE = {}


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
    global GT_BY_IMAGE_CACHE
    GT_BY_IMAGE_CACHE = gt_by_image
    threshold_tag = format_threshold_tag(0.25, 0.20, 0.45)
    
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

    report_lines = []
    report_data = {
        "title": "Validation with Ground Truth Comparison",
        "total_images_requested": len(TEST_IMAGES),
        "overall": {},
        "images": [],
        "console_lines": [],
    }
    
    for line in format_banner("VALIDATION WITH GROUND TRUTH COMPARISON"):
        print(line)
        report_lines.append(line)
        report_data["console_lines"].append(line)
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        for result in executor.map(process_image, TEST_IMAGES):
            if result.get("missing"):
                logger.warning("Image not found: %s", result["image"])
                continue

            total_gt_teeth += result["gt_teeth"]
            total_pred_teeth += result["pred_teeth"]
            total_tp_teeth += result["tp_teeth"]
            total_fp_teeth += result["fp_teeth"]
            total_fn_teeth += result["fn_teeth"]

            total_gt_diseases += result["gt_diseases"]
            total_pred_diseases += result["pred_diseases"]
            total_tp_diseases += result["tp_diseases"]
            total_fp_diseases += result["fp_diseases"]
            total_fn_diseases += result["fn_diseases"]

            report_data["images"].append(result)
            results.append(result)

            for line in result["log_lines"]:
                print(line)
                report_lines.append(line)
                report_data["console_lines"].append(line)
    
    overall_lines = format_overall_block(
        total_gt_teeth,
        total_pred_teeth,
        total_tp_teeth,
        total_fp_teeth,
        total_fn_teeth,
        total_gt_diseases,
        total_pred_diseases,
        total_tp_diseases,
        total_fp_diseases,
        total_fn_diseases,
    )

    report_data["overall"] = {
        "tooth_detection": {
            "ground_truth": total_gt_teeth,
            "predicted": total_pred_teeth,
            "true_positives": total_tp_teeth,
            "false_positives": total_fp_teeth,
            "false_negatives": total_fn_teeth,
            "recall": round(total_tp_teeth / max(1, total_gt_teeth) * 100, 2),
            "precision": round(total_tp_teeth / max(1, total_pred_teeth) * 100, 2),
        },
        "disease_detection": {
            "ground_truth": total_gt_diseases,
            "predicted": total_pred_diseases,
            "true_positives": total_tp_diseases,
            "false_positives": total_fp_diseases,
            "false_negatives": total_fn_diseases,
            "recall": round(total_tp_diseases / max(1, total_gt_diseases) * 100, 2),
            "precision": round(total_tp_diseases / max(1, total_pred_diseases) * 100, 2),
        },
        "summary": {
            "total_images_processed": len(results),
            "total_images_requested": len(TEST_IMAGES),
        },
        "console_lines": overall_lines,
    }

    for line in overall_lines:
        print(line)
        report_lines.append(line)
        report_data["console_lines"].append(line)
    
    # Save detailed results
    json_path = OUTPUT_DIR / f"validation_with_gt_report_{threshold_tag}.json"
    with open(json_path, "w") as f:
        json.dump(report_data, f, indent=2)
    
    logger.info(f"Results saved to: {json_path}")
    logger.info("Report saved to: %s", json_path)


if __name__ == "__main__":
    main()
