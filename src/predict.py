from pathlib import Path
from ultralytics import YOLO

# Path to our trained model
model_path = Path("runs/detect/runs/detect/protein_nucleus_test/weights/best.pt")

# Test images
source = Path("dataset/images/test")

# Where prediction results will be saved
output_dir = Path("results/test_predictions")

# Load the trained model
model = YOLO(str(model_path))

# Run prediction
results = model.predict(
    source=str(source),
    conf=0.50,
    save=True,
    show_labels=False,
    show_conf=False,
    project=str(output_dir.parent),
    name=output_dir.name
)

print("Prediction complete!")
print(f"Results saved to: {results[0].save_dir}")