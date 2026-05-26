import io
import base64
import cv2
import numpy as np
import subprocess
import tempfile
import shutil
import os
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import FileResponse
from PIL import Image
from typing import List

from api.schemas.detection import DetectionResponse, DetectionResult, BoundingBox, FullPipelineResponse
from inference.yolo_pipeline import SafetyPipeline
from visualization.annotator import Annotator

router = APIRouter()
pipeline = SafetyPipeline()
annotator = Annotator()


# =========================
# HELPERS
# =========================

def is_video(filename: str) -> bool:
    return filename.lower().endswith((".mp4", ".avi", ".mov"))


def encode_image(image: np.ndarray) -> str:
    """Encode BGR frame ke base64 JPEG string."""
    _, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buffer).decode("utf-8")


async def parse_image(file: UploadFile) -> np.ndarray:
    """Decode uploaded image file ke BGR numpy array."""
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def reencode_to_h264(input_path: str, output_path: str) -> bool:
    """
    Re-encode video ke H.264 menggunakan ffmpeg.
    Wajib agar browser bisa memutar video (HTML5 butuh H.264 + yuv420p).
    Return True jika berhasil.
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vcodec", "libx264",
        "-crf", "23",
        "-preset", "fast",
        "-pix_fmt", "yuv420p",      # browser compatibility
        "-movflags", "+faststart",  # streaming: moov atom di awal file
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def process_video_frames(
    input_path: str,
    output_path: str,
    mode: str = "full_pipeline",
) -> int:
    """
    Baca video frame by frame, proses dengan pipeline, tulis ke output.
    Return: jumlah frame yang diproses.
    """
    cap = cv2.VideoCapture(input_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Tulis ke raw mp4v dulu — akan di-reencode setelahnya
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if mode == "detect":
            results  = pipeline.detect_person(frame)
            annotated = annotator.draw_detections(frame.copy(), results)

        elif mode == "track":
            results  = pipeline.track_persons(frame)
            annotated = annotator.draw_detections(frame.copy(), results)

        elif mode == "ppe":
            results  = pipeline.detect_ppe(frame)
            annotated = annotator.draw_detections(frame.copy(), results)

        else:  # full_pipeline
            person_results = pipeline.track_persons(frame)
            ppe_results    = pipeline.detect_ppe(frame)
            annotated      = annotator.draw_detections(frame.copy(), person_results)
            annotated      = annotator.draw_detections(annotated, ppe_results)

        out.write(annotated)
        frame_count += 1

    cap.release()
    out.release()
    return frame_count


# =========================
# IMAGE ENDPOINTS
# =========================

@router.post("/detect")
async def detect_person(file: UploadFile = File(...)):
    image = await parse_image(file)
    results = pipeline.detect_person(image)
    annotated = annotator.draw_detections(image.copy(), results)
    return {
        "status": "success",
        "annotated_image_base64": encode_image(annotated)
    }


@router.post("/track")
async def track_person(file: UploadFile = File(...)):
    image = await parse_image(file)
    results = pipeline.track_persons(image)
    annotated = annotator.draw_detections(image.copy(), results)
    return {
        "status": "success",
        "annotated_image_base64": encode_image(annotated)
    }


@router.post("/ppe")
async def detect_ppe(file: UploadFile = File(...)):
    image = await parse_image(file)
    results = pipeline.detect_ppe(image)
    annotated = annotator.draw_detections(image.copy(), results)
    return {
        "status": "success",
        "annotated_image_base64": encode_image(annotated)
    }


# =========================
# FULL PIPELINE ENDPOINT
# =========================

@router.post("/full_pipeline")
async def full_pipeline(file: UploadFile = File(...)):

    # =========================
    # VIDEO PROCESSING
    # =========================
    if is_video(file.filename):

        # Simpan upload ke tempfile input
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_in:
            shutil.copyfileobj(file.file, tmp_in)
            input_path = tmp_in.name

        # Tempfile untuk raw output (mp4v)
        with tempfile.NamedTemporaryFile(delete=False, suffix="_raw.mp4") as tmp_raw:
            raw_path = tmp_raw.name

        # Tempfile untuk output H.264 final
        with tempfile.NamedTemporaryFile(
            delete=False, suffix="_h264.mp4", dir="outputs"
        ) as tmp_final:
            final_path = tmp_final.name

        os.makedirs("outputs", exist_ok=True)

        # Proses frame
        process_video_frames(input_path, raw_path, mode="full_pipeline")

        # Re-encode ke H.264
        success = reencode_to_h264(raw_path, final_path)

        # Cleanup tempfiles
        for p in [input_path, raw_path]:
            try:
                os.remove(p)
            except Exception:
                pass

        if not success:
            # Fallback: kembalikan raw jika ffmpeg tidak tersedia
            return FileResponse(
                raw_path,
                media_type="video/mp4",
                filename="processed_video.mp4"
            )

        # --- FIX: kembalikan H.264 video yang bisa diputar browser ---
        return FileResponse(
            final_path,
            media_type="video/mp4",
            filename="processed_video.mp4",
            headers={"Content-Disposition": "attachment; filename=processed_video.mp4"}
        )

    # =========================
    # IMAGE PROCESSING
    # =========================
    image = await parse_image(file)

    person_results = pipeline.track_persons(image)
    ppe_results    = pipeline.detect_ppe(image)

    annotated = annotator.draw_detections(image.copy(), person_results)
    annotated = annotator.draw_detections(annotated, ppe_results)

    return {
        "status": "success",
        "annotated_image_base64": encode_image(annotated)
    }