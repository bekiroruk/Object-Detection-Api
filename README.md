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


## 🧠 Model Training (YOLOv8)

The model is trained using Ultralytics YOLOv8.
An example training command:

yolo detect train \
  model=yolov8n.pt \
  data=data/final_data/data_wl.yaml \
  epochs=30 \
  imgsz=640 \
  project=models \
  name=wildlife_yolo22


After training, the best model weights are saved under:

models/wildlife_yolo22/weights/best.pt


These weights are then loaded by the FastAPI application.

Example Dataset Config (data_wl.yaml)
train: path/to/train/images
val: path/to/valid/images
test: path/to/test/images

nc: 4

names:
  0: buffalo
  1: elephant
  2: rhino
  3: zebra


You can replace paths and class names to adapt the project to a different dataset.

## 🌐 FastAPI Application

The core API logic lives in app/main.py.

Model Loading (example)
from ultralytics import YOLO
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "wildlife_yolo22" / "weights" / "best.pt"

model = YOLO(str(MODEL_PATH))

Endpoints
GET /

Simple health-check endpoint:

{
  "message": "Object Detection API is running!"
}

POST /detect

Accepts an image file (multipart/form-data) and returns detection results:

Request:

Method: POST

URL: /detect

Body: file (image, e.g. .jpg, .png)

Response (example):

{
  "filename": "example.jpg",
  "num_detections": 2,
  "detections": [
    {
      "label": "zebra",
      "confidence": 0.94,
      "bbox": {
        "x1": 120.3,
        "y1": 80.5,
        "x2": 360.7,
        "y2": 290.1
      }
    },
    {
      "label": "elephant",
      "confidence": 0.90,
      "bbox": {
        "x1": 50.2,
        "y1": 60.1,
        "x2": 220.4,
        "y2": 250.8
      }
    }
  ]
}

## 🧪 Local Development (without Docker)
1. Create and activate a virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

2. Install dependencies
pip install -r requirements.txt

3. Start the FastAPI server
uvicorn app.main:app --reload


The API will be available at:

http://localhost:8000/ – health-check

http://localhost:8000/docs – Swagger UI

## 🐳 Docker Deployment
1. Build the Docker image
docker build -t object-detection-api .

2. Run the container
docker run -p 8000:8000 object-detection-api


Now the API is accessible at:

http://localhost:8000/

http://localhost:8000/docs

You can upload images via /docs → POST /detect.
