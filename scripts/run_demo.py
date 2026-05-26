import cv2
import time
import sys
import asyncio
from src.pipeline import SafetyPipeline
from scripts.alert import AlertService
from dotenv import load_dotenv

load_dotenv()

def run_live(source=0, record_output: bool = True):
    """
    source: 0 = webcam, atau path ke video file / RTSP stream.
    """
    pipeline = SafetyPipeline()
    alert = AlertService()

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {source}")
        return

    fps_cap = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if record_output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter("demo/output_demo.mp4", fourcc, fps_cap, (w, h))

    ALERT_COOLDOWN = 10.0   # detik antar notifikasi per track_id
    last_alert: dict = {}

    print("[DEMO] Press 'q' to quit, 's' to save screenshot.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = pipeline.process_frame(frame)
        annotated = result.get("annotated_frame", frame)

        # FPS overlay
        cv2.putText(annotated, f"FPS: {result['fps']:.1f}",
                    (w-130, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (255,255,255), 2, cv2.LINE_AA)

        # alert throttle
        for viol in result["violations"]:
            tid = viol["track_id"]
            now = time.time()
            if now - last_alert.get(tid, 0) > ALERT_COOLDOWN:
                last_alert[tid] = now
                asyncio.run(alert.send_violation_alert(tid, viol["violations"], annotated))

        if writer:
            writer.write(annotated)

        cv2.imshow("Human Safety Monitor", annotated)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite(f"demo/screenshot_{int(time.time())}.jpg", annotated)
            print("[Screenshot saved]")

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("[DONE] Session ended.")

if __name__ == "__main__":
    source = sys.argv[1] if len(sys.argv) > 1 else 0
    run_live(source)