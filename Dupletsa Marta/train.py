from ultralytics import YOLO
import torch


def main():
    """
    Training script for YOLOv8n on the PASCAL VOC0712 dataset.
    """

    # Select device automatically
    device = 0 if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    # Load pretrained YOLOv8n model
    model = YOLO("yolov8n.pt")

    # Train the model
    model.train(
        data="voc_local.yaml",
        epochs=50,
        imgsz=416,
        batch=16,
        device=device,
        project="runs_diploma",
        name="yolov8n_voc_416_50ep",
        exist_ok=True
    )


if __name__ == "__main__":
    main()
