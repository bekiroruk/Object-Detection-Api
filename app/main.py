from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
from pathlib import Path
import io

# -------------------------
# 1) FastAPI uygulaması
# -------------------------
app = FastAPI(
    title="Wildlife Object Detection API",
    description="YOLOv8 ile buffalo, elephant, rhino, zebra tespiti yapan API",
    version="1.0.0",
)

# -------------------------
# 2) Modeli yükle
# -------------------------
# Bu dosya: app/main.py
# Bir üst klasöre çık -> project root -> models/wildlife_yolo22/weights/best.pt
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "wildlife_yolo22" / "weights" / "best.pt"

model = YOLO(str(MODEL_PATH))


# -------------------------
# 3) Basit sağlık kontrolü
# -------------------------
@app.get("/")
def root():
    return {"message": "Nesne Tespiti API çalışıyor!"}


# -------------------------
# 4) Detect endpoint'i
# -------------------------
@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    # 1) Dosyayı byte olarak oku
    image_bytes = await file.read()

    # 2) Byte -> PIL Image
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # 3) YOLO ile tahmin al
    results = model(image)

    detections = []

    # 4) Sonuçları parse et
    for r in results:
        for box in r.boxes:
            # Kutu koordinatları (xyxy)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0].item())
            cls_id = int(box.cls[0].item())
            label = model.names[cls_id]

            detections.append({
                "label": label,
                "confidence": conf,
                "bbox": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2
                }
            })

    return JSONResponse(content={
        "filename": file.filename,
        "num_detections": len(detections),
        "detections": detections
    })
