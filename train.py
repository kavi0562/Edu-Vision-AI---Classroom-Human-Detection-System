from ultralytics import YOLO

def train_model():
    # Use extra-large model for maximum accuracy
    model = YOLO("yolov8x.pt")  

    model.train(
        data="dataset.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        optimizer="AdamW",
        lr0=0.001,
        augment=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        patience=50,
        project="classroom_ai",
        name="max_accuracy_run"
    )

if __name__ == "__main__":
    train_model()