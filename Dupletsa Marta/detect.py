from ultralytics import YOLO
from pathlib import Path


def main():
    """
    Inference script for YOLOv8 object detection.
    The source can be an image, video file, folder, or webcam index.
    """

    # Path to trained weights.
    # Change this path if your best.pt is stored elsewhere.
    model_path = "runs_diploma/yolov8n_voc_416_50ep/weights/best.pt"

    # If trained weights are not available, use pretrained YOLOv8n weights.
    if not Path(model_path).exists():
        print("Trained weights were not found. Using yolov8n.pt instead.")
        model_path = "yolov8n.pt"

    model = YOLO(model_path)

    # Change source depending on what you want to test:
    # source = "example.jpg"      # image
    # source = "video.mp4"        # video
    # source = 0                  # webcam
    source = "example.jpg"

    results = model.predict(
        source=source,
        imgsz=416,
        conf=0.25,
        save=True,
        project="runs_diploma",
        name="inference_results",
        exist_ok=True
    )

    print("Inference completed.")
    print("Results saved to runs_diploma/inference_results")


if __name__ == "__main__":
    main()
