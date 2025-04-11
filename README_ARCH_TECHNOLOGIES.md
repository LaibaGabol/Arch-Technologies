# Brain Tumor Segmentation using YOLOv11 & SAM2

## Team Members

- Priya Lohana (Intern ID: ARCH-2504-0358, Phone: 03332287838)
- Laiba Gabol (Intern ID: ARCH-2504-0355, Phone: 03030286202)

---

## Project Overview

This project detects and segments brain tumors in brain MRI images using advanced deep learning models — YOLOv11 for tumor detection and SAM2 for tumor segmentation.

---

## Objectives

- Detect tumors in MRI images using YOLOv11.
- Segment tumors precisely using SAM2.
- Visualize and analyze the results.

---

## Dataset

Brain MRI Dataset with tumor labels.

### Folder Structure:

Dataset/
│── Train/
│── Test/
│── Labels/

---

## Technologies Used

- Python
- OpenCV
- YOLOv11
- SAM2 (Segment Anything Model)
- Torch
- Pandas

---

## How to Run

### 1. Install Dependencies:

pip install -r requirements.txt

### 2. Train YOLOv11 Model:

from ultralytics import YOLO
model = YOLO('yolov11.pt')
model.train(data='config.yaml', epochs=50)

### 3. Segment Tumor using SAM2:

from segment_anything import SamPredictor
predictor = SamPredictor('sam2_model.pth')
segmentation = predictor.predict(image)

---

## References

- YOLOv11 Documentation: https://docs.ultralytics.com/
- SAM2 GitHub Repository: https://github.com/facebookresearch/segment-anything
- YouTube Tutorials Used:
  - YOLOv11 Object Detection
  - Brain Tumor Segmentation

---

## GitHub Repository Link

https://github.com/LaibaGabol/Arch-Technologies.git

---
