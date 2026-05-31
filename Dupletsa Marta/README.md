# YOLOv8 Object Detection Thesis

This repository contains the source code and experimental materials for the bachelor's thesis:

**Object Detection and Recognition Algorithms Based on Modern Deep Learning Neural Networks**

Author: Marta Dupletsa  
University: Ivan Franko National University of Lviv  
Faculty: Faculty of Applied Mathematics and Informatics  
Department: Department of Applied Mathematics  
Model: YOLOv8n  
Dataset: PASCAL VOC0712  

## Project Description

The project implements an object detection pipeline based on the YOLOv8 neural network model.  
The work includes dataset preparation, annotation conversion, model training, inference, evaluation, and visual analysis of detection results.

The main goal of the project is to study modern deep learning approaches for object detection and to implement a practical system for detecting and recognizing objects in images and video data.

## Main Tasks

The project includes the following stages:

- analysis of modern object detection architectures;
- preparation of the PASCAL VOC0712 dataset;
- conversion of XML annotations to YOLO format;
- training of the YOLOv8n model;
- evaluation using Precision, Recall and mAP metrics;
- visual analysis of successful and problematic detections;
- analysis of model errors in complex scenes.

## Repository Structure

```text
.
├── prepare_voc_yolo.py      # Dataset preparation and annotation conversion
├── train.py                 # YOLOv8 training script
├── detect.py                # Inference script
├── voc_local.yaml           # Dataset configuration file
├── requirements.txt         # Python dependencies
├── figures/                 # Figures used in the thesis
└── results/                 # Experimental plots and evaluation results
```

## Dataset

The experiments were conducted on the PASCAL VOC0712 dataset.

The dataset itself is not included in this repository because of its large size.  
To reproduce the experiments, the dataset should be downloaded separately and prepared in YOLO format.

Expected dataset structure:

```text
datasets/
└── VOC/
    ├── images/
    └── labels/
```

## Installation

Install the required Python libraries:

```bash
pip install -r requirements.txt
```

## Training

The model can be trained using the training script:

```bash
python train.py
```

or using the Ultralytics YOLO command line interface:

```bash
yolo detect train model=yolov8n.pt data=voc_local.yaml imgsz=416 epochs=50 batch=16
```

## Inference on Images

```bash
yolo detect predict model=best.pt source=example.jpg imgsz=416 conf=0.25 save=True
```

## Inference on Video

```bash
yolo detect predict model=best.pt source=video.mp4 imgsz=416 conf=0.25 save=True
```

## Webcam Detection

```bash
yolo detect predict model=best.pt source=0 imgsz=416 conf=0.25 show=True
```

## Evaluation

```bash
yolo detect val model=best.pt data=voc_local.yaml imgsz=416
```

## Evaluation Metrics

The model was evaluated using the following metrics:

- Precision;
- Recall;
- mAP@0.5;
- mAP@0.5:0.95;
- Confusion Matrix;
- Precision-Recall Curve.

## Notes

Large files such as datasets, trained weights and full training runs are not included in this repository.

The repository contains the source code, configuration files, selected figures and experimental materials required to describe and reproduce the main stages of the bachelor's thesis project.
