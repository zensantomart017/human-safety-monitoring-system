from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import cv2
import numpy as np
import base64
import io
from contextlib import asynccontextmanager
from src.pipeline import SafetyPipeline
from src.api.schemas.detection import DetectionResponse
import os
from dotenv import load_dotenv

load_dotenv()

pipeline: SafetyPipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    pipeline = SafetyPipeline(
        person_model="yolov8n.pt",
        ppe_model=os.getenv("PPE_WEIGHTS_PATH", "models/best.pt"),
        conf_person=float(os.getenv("CONFIDENCE_THRESHOLD", 0.45)),
        device=os.getenv("DEVICE", "auto"),
    )
    print("[API] Pipeline initialized.")
    yield
    print("[API] Shutting down.")

app = FastAPI(
    title="Human Safety Monitoring API",
    version="1.0.0",
    lifespan=lifespan,
)

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": pipeline is not None}

@app.post("/api/detect", response_model=DetectionResponse)
async def detect(
    file: UploadFile = File(...),
    return_image: bool = True,
):
    """
    Upload satu frame (JPEG/PNG), kembalikan deteksi + annotated image.
    """
    if pipeline is None:
        raise HTTPException(503, "Pipeline not ready")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(400, "Invalid image file")

    result = pipeline.process_frame(frame, return_annotated=return_image)

    # encode annotated frame ke base64
    annotated_b64 = None
    if return_image and "annotated_frame" in result:
        _, buf = cv2.imencode(".jpg", result["annotated_frame"], [cv2.IMWRITE_JPEG_QUALITY, 85])
        annotated_b64 = base64.b64encode(buf.tobytes()).decode()

    return DetectionResponse(
        frame_id=result["frame_id"],
        latency_ms=result["latency_ms"],
        fps=result["fps"],
        person_count=result["person_count"],
        persons=result["persons"],
        violations=result["violations"],
        annotated_image_b64=annotated_b64,
    )

@app.post("/api/detect/base64")
async def detect_base64(payload: dict):
    """Alternatif: kirim frame sebagai base64 string dalam JSON body."""
    if pipeline is None:
        raise HTTPException(503, "Pipeline not ready")

    img_data = base64.b64decode(payload["image_b64"])
    nparr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "Invalid base64 image")

    result = pipeline.process_frame(frame, return_annotated=True)
    _, buf = cv2.imencode(".jpg", result.pop("annotated_frame"))
    result["annotated_image_b64"] = base64.b64encode(buf.tobytes()).decode()
    return JSONResponse(result)

# Run: uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload