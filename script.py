# train_cls.py
from ultralytics import YOLO
from pathlib import Path

#Run this if you wanna do eval, Muib
# yolo classify val \
#   model=runs/classify/train16/weights/best.pt \
#   data=content/ieee-mbl-cls


# === PRESET HYPERPARAMETERS ===
DATA_DIR = "content/ieee-mbl-cls"   # friend will create this folder
MODEL    = "yolo11n-cls.pt"
EPOCHS   = 50
IMGSZ    = 224
BATCH    = 32
RUN_NAME = "muib_script"
DEVICE   = ""  # "", "0", "cpu", etc.

def main():
    data_path = Path(DATA_DIR)
    if not data_path.exists():
        raise SystemExit(f"[ERR] Dataset path does not exist: {data_path}")

    model = YOLO(MODEL)

    results = model.train(
        data=str(data_path),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        name=RUN_NAME,
        device=DEVICE if DEVICE else None,
        task="classify",
    )

    print("Training finished.")
    print("Results directory:", results.save_dir)

if __name__ == "__main__":
    main()
