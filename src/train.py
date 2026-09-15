from ultralytics import YOLO

# Load the small pretrained YOLO11 model
model = YOLO("yolo11n.pt")

# Train the model on our nucleus dataset
model.train(
    data="dataset/data.yaml",
    epochs=10,
    imgsz=640,
    batch=4,
    device="cpu",
    workers=0,
    patience=5,
    project="runs/detect",
    name="protein_nucleus_test",
    pretrained=True
)