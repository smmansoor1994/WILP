from src.inference.predictor import ARCHONHybridPredictor
from src.inference.disease_cropper import DiseaseCropper

# Load existing model (no retraining)
predictor = ARCHONHybridPredictor(
    weights_path='weights/archon_best.pt',
    device='cpu'
)

# Run on test images
results = predictor.predict('data/processed/images/test')

# Crop results for manual inspection
cropper = DiseaseCropper(margin_ratio=0.15)
for image_id, teeth in results.items():
    crops = cropper.extract_crops_from_prediction(
        image=cv2.imread(...),
        teeth_detections=teeth,
        image_id=image_id
    )
    cropper.save_crops_batch(crops, 'outputs/analysis_crops')