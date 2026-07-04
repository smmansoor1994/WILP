"""
Validate model on random training images and generate performance report
"""
import json
import logging
from pathlib import Path
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

IMG_DIR = Path(r"D:\WILP\sem-4\Dataset\DENTEX\DENTEX\training_data\quadrant-enumeration-disease\xrays")
WEIGHTS = Path(r"D:\WILP\Workingcode\models\archon-100-main-hybrid\weights_20260704_163459\content\WILP\weights\archon_best.pt")
OUTPUT_DIR = Path(r"D:\WILP\validation_test")

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    predictor = ARCHONHybridPredictor(
        weights_path=str(WEIGHTS),
        device="cpu",
        conf_threshold=0.05,
        iou_threshold=0.45,
        attr_threshold=0.3,
    )
    
    results = []
    total_teeth = 0
    total_diseased = 0
    total_healthy = 0
    
    print("\n" + "="*80)
    print("VALIDATION ON TRAINING DATA".center(80))
    print("="*80 + "\n")
    
    for img_name in TEST_IMAGES:
        img_path = IMG_DIR / img_name
        if not img_path.exists():
            logger.warning(f"Image not found: {img_name}")
            continue
        
        logger.info(f"Processing: {img_name}")
        pred_results = predictor.predict(input_path=str(img_path), save_vis=False, save_json=False)
        teeth = pred_results.get(img_name, [])
        
        diseased = [t for t in teeth if t.diseases]
        healthy = [t for t in teeth if not t.diseases]
        
        total_teeth += len(teeth)
        total_diseased += len(diseased)
        total_healthy += len(healthy)
        
        result = {
            "image": img_name,
            "teeth_detected": len(teeth),
            "healthy": len(healthy),
            "diseased": len(diseased),
            "teeth": []
        }
        
        for tooth in sorted(teeth, key=lambda t: t.fdi):
            disease_str = ", ".join(tooth.diseases) if tooth.diseases else "Healthy"
            result["teeth"].append({
                "fdi": tooth.fdi,
                "status": disease_str,
                "conf": tooth.conf,
            })
        
        results.append(result)
        
        print(f"\n{img_name:20} | Teeth: {len(teeth):2} | Healthy: {len(healthy):2} | Diseased: {len(diseased):2}")
        for tooth in sorted(teeth, key=lambda t: t.fdi):
            disease_str = ", ".join(tooth.diseases) if tooth.diseases else "Healthy"
            print(f"  └─ FDI {tooth.fdi:2}  {disease_str:30}  conf={tooth.conf:.3f}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY STATISTICS".center(80))
    print("="*80)
    print(f"Total Images Tested:    {len(TEST_IMAGES)}")
    print(f"Total Teeth Detected:   {total_teeth}")
    print(f"  ├─ Healthy Teeth:     {total_healthy} ({100*total_healthy/total_teeth:.1f}%)")
    print(f"  └─ Diseased Teeth:    {total_diseased} ({100*total_diseased/total_teeth:.1f}%)")
    print(f"Avg Teeth per Image:    {total_teeth/len(TEST_IMAGES):.1f}")
    print(f"Avg Diseased per Image: {total_diseased/len(TEST_IMAGES):.1f}")
    print("="*80 + "\n")
    
    # Save JSON
    json_path = OUTPUT_DIR / "validation_results.json"
    with open(json_path, "w") as f:
        json.dump({
            "summary": {
                "total_images": len(TEST_IMAGES),
                "total_teeth": total_teeth,
                "total_healthy": total_healthy,
                "total_diseased": total_diseased,
                "pct_diseased": round(100*total_diseased/total_teeth, 2),
                "avg_teeth_per_image": round(total_teeth/len(TEST_IMAGES), 2),
            },
            "results": results
        }, f, indent=2)
    
    logger.info(f"Results saved to: {json_path}")

if __name__ == "__main__":
    main()
