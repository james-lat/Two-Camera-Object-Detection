from ultralytics import YOLO

# Load base pretrained detection model
model = YOLO("yolo11n.pt")

# Fine-tune the model on YOUR tomato datasetif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
device = "mps"

# Assuming you're running this from the muib_proj folder
model.train(
    data="data.yaml",  # <-- your dataset YAML
    epochs=100
)

# Use the updated model for inference on a new image
# Replace this with an actual image filename from your val set
model.predict("content/data/val/images/IMG_0988.jpg", device=device)
