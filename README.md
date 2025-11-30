# Object Detection API (YOLOv8 + FastAPI + Docker)

This project implements an end-to-end **object detection API** using **YOLOv8**, **FastAPI**, and **Docker**.  
Given an input image, the API returns the detected objects with their class labels, confidence scores, and bounding box coordinates in JSON format.

> **Use case:** Real-time or batch inference for wildlife object detection (buffalo, elephant, rhino, zebra), easily extendable to other domains and datasets.

---

## 🚀 Features

- **YOLOv8**-based object detection (Ultralytics)
- Trained on a **wildlife dataset** with 4 classes:
  - `buffalo`, `elephant`, `rhino`, `zebra`
- **FastAPI** backend with:
  - `/` health-check endpoint
  - `/detect` endpoint for image upload and inference
- Interactive API documentation via **Swagger UI** (`/docs`)
- **Dockerized** application for portable deployment

---

## 🧱 Project Structure

```text
.
├── app/
│   └── main.py                  # FastAPI application (loads YOLO model and exposes endpoints)
├── data/
│   └── final_data/
│       ├── data_wl.yaml         # YOLO dataset configuration (paths, classes)
│       └── ...                  # (train/valid/test images & labels - NOT included in the repo)
├── models/
│   └── wildlife_yolo22/
│       └── weights/
│           └── best.pt          # Trained YOLOv8 model weights (ignored in Git)
├── Dockerfile                   # Docker image definition
├── requirements.txt             # Python dependencies
├── .dockerignore
├── .gitignore
└── README.md



Note:

Dataset images/labels and model weights (*.pt) are not tracked by Git (ignored via .gitignore).

You can plug in your own YOLOv8 model and dataset by adjusting paths in data_wl.yaml and app/main.py.
