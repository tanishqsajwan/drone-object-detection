from ultralytics import YOLO
from ultralytics import settings
import torch
import os

def main():
    print("GPU:", torch.cuda.get_device_name(0))
    BASE = os.path.dirname(os.path.abspath(__file__))
    settings.update({"runs_dir": os.path.join(BASE, "runs")})

    model = YOLO("yolov8n.pt")
    model.train(
        data="VisDrone.yaml",
        epochs=50,
        imgsz=416,
        device=0,
        batch=16,
        patience=15,
        dropout=0.2,
        weight_decay=0.0005,
        cls=2.0,
        flipud=0.5,
        copy_paste=0.0,
        workers=4,
        cache="disk",
        amp=True,
        fraction=0.5,
        project="detects", 
        name="train",
    )

if __name__ == "__main__":
    main()