import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import streamlit as st
import requests
from PIL import Image
import cv2
import numpy as np
import base64
import tempfile
import subprocess

from inference.yolo_pipeline import SafetyPipeline
from visualization.annotator import Annotator

# =========================
# INIT
# =========================
pipeline = SafetyPipeline()
annotator = Annotator()

API_BASE_URL = "https://mascot-tiny-reggae.ngrok-free.dev/api/v1"

st.set_page_config(page_title="Human Safety Monitoring System", layout="wide")

# =========================
# TITLE
# =========================
st.title("🦺 Human Safety Monitoring System")
st.markdown("AI-Based Workplace Safety Monitoring")

# =========================
# SIDEBAR
# =========================
mode = st.sidebar.selectbox(
    "Select Monitoring Mode",
    ["Person Detection", "Person Tracking", "PPE Detection", "Full Pipeline"]
)

# =========================
# UPLOAD
# =========================
uploaded_file = st.file_uploader(
    "Upload Image or Video",
    type=["jpg", "jpeg", "png", "mp4", "avi", "mov"]
)

# =========================
# ENDPOINT MAP
# =========================
endpoint_map = {
    "Person Detection": "/detect",
    "Person Tracking": "/track",
    "PPE Detection": "/ppe",
    "Full Pipeline": "/full_pipeline"
}

# =========================
# HELPER: re-encode ke H.264
# =========================
def reencode_to_h264(input_path: str) -> str:
    """
    Re-encode video dari mp4v (MPEG-4 Part 2) ke H.264 menggunakan ffmpeg.
    Browser / Streamlit st.video() butuh H.264 agar bisa diputar.
    Kembalikan path file output yang sudah di-encode.
    """
    output_path = input_path.replace(".mp4", "_h264.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vcodec", "libx264",
        "-crf", "23",          # quality: 0=lossless, 51=worst; 23 adalah default
        "-preset", "fast",     # fast = balance speed vs compression
        "-pix_fmt", "yuv420p", # wajib agar kompatibel semua browser
        "-movflags", "+faststart",  # streaming-friendly: moov atom di depan
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        st.warning(f"ffmpeg re-encode failed: {result.stderr}")
        return input_path  # fallback ke file asli jika ffmpeg gagal
    return output_path


# =========================
# MAIN
# =========================
if uploaded_file is not None:

    extension = uploaded_file.name.split(".")[-1].lower()
    col1, col2 = st.columns(2)

    # =========================
    # IMAGE MODE
    # =========================
    if extension in ["jpg", "jpeg", "png"]:

        image = Image.open(uploaded_file)

        with col1:
            st.subheader("Input Image")
            st.image(image, use_container_width=True)

        if st.button("🚀 Run Monitoring"):
            with st.spinner("Processing Image..."):

                endpoint = endpoint_map[mode]
                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type
                    )
                }
                response = requests.post(f"{API_BASE_URL}{endpoint}", files=files)

                if response.status_code == 200:
                    data = response.json()
                    if "annotated_image_base64" in data:
                        img_data = base64.b64decode(data["annotated_image_base64"])
                        nparr = np.frombuffer(img_data, np.uint8)
                        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                        with col2:
                            st.subheader("Detection Result")
                            st.image(img, use_container_width=True)
                else:
                    st.error(response.text)

    # =========================
    # VIDEO MODE
    # =========================
    elif extension in ["mp4", "avi", "mov"]:

        # Simpan upload ke tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{extension}") as tmp_in:
            tmp_in.write(uploaded_file.read())
            input_video_path = tmp_in.name

        with col1:
            st.subheader("Input Video")
            st.video(input_video_path)

        if st.button("🚀 Run Video Monitoring"):
            with st.spinner("Processing Video..."):

                cap = cv2.VideoCapture(input_video_path)
                width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0

                # --- FIX 1: Tulis ke tempfile dulu, bukan path relatif ---
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix="_raw.mp4"
                ) as tmp_out:
                    raw_output_path = tmp_out.name

                # --- FIX 2: Tetap pakai mp4v untuk penulisan frame ---
                # (mp4v lebih kompatibel untuk VideoWriter lintas OS)
                # Re-encode ke H.264 dilakukan setelah selesai via ffmpeg.
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out = cv2.VideoWriter(raw_output_path, fourcc, fps, (width, height))

                if not out.isOpened():
                    st.error("Failed to initialize VideoWriter. Periksa path dan codec.")
                    st.stop()

                frame_count = 0
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                progress = st.progress(0, text="Processing frames...")

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # --- FIX 3: Unified processing — semua mode tulis ke `out` ---
                    if mode == "Person Detection":
                        results  = pipeline.detect_person(frame)
                        annotated = annotator.draw_detections(frame.copy(), results)

                    elif mode == "Person Tracking":
                        results  = pipeline.track_persons(frame)
                        annotated = annotator.draw_detections(frame.copy(), results)

                    elif mode == "PPE Detection":
                        results  = pipeline.detect_ppe(frame)
                        annotated = annotator.draw_detections(frame.copy(), results)

                    else:  # Full Pipeline
                        person_results = pipeline.track_persons(frame)
                        ppe_results    = pipeline.detect_ppe(frame)
                        annotated      = annotator.draw_detections(frame.copy(), person_results)
                        annotated      = annotator.draw_detections(annotated, ppe_results)

                    out.write(annotated)

                    frame_count += 1
                    if total_frames > 0:
                        progress.progress(
                            min(frame_count / total_frames, 1.0),
                            text=f"Frame {frame_count}/{total_frames}"
                        )

                cap.release()
                out.release()
                progress.empty()

                # --- FIX 4: Re-encode ke H.264 agar bisa diputar browser ---
                st.info("Re-encoding ke H.264 untuk kompatibilitas browser...")
                final_output_path = reencode_to_h264(raw_output_path)

                # Cleanup raw file
                try:
                    os.remove(raw_output_path)
                except Exception:
                    pass

                with col2:
                    st.subheader("Processed Video")
                    # --- FIX 5: Baca sebagai bytes agar st.video() andal ---
                    with open(final_output_path, "rb") as f:
                        video_bytes = f.read()
                    st.video(video_bytes)

                st.success(
                    f"✅ Video Processing Complete! "
                    f"({frame_count} frames processed)"
                )

                # Cleanup final file
                try:
                    os.remove(final_output_path)
                except Exception:
                    pass