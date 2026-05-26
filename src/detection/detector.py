from ultralytics import YOLO
import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import List, Optional
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Detection:
    bbox: List[float]       # [x1, y1, x2, y2] normalized
    bbox_px: List[int]      # [x1, y1, x2, y2] pixel
    confidence: float
    class_id: int
    class_name: str
    track_id: Optional[int] = None

class PersonDetector:
    """
    YOLOv8-based person detector.
    Menggunakan model pretrained COCO — class 0 = person.
    """
    PERSON_CLASS_ID = 0

    def __init__(
        self,
        model_path: str = "yolov8n.pt",  # ganti yolov8s.pt untuk akurasi lebih
        conf_threshold: float = 0.45,
        iou_threshold: float = 0.50,
        device: str = "auto",
    ):
        self.model = YOLO(model_path)
        self.conf = conf_threshold
        self.iou = iou_threshold
        self.device = device
        print(f"[PersonDetector] Loaded: {model_path} | device: {device}")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Input : BGR frame (H, W, 3)
        Output: List[Detection] — hanya class person
        """
        h, w = frame.shape[:2]
        results = self.model(
            frame,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            classes=[self.PERSON_CLASS_ID],  # filter person only
            verbose=False,
        )[0]

        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls  = int(box.cls[0])
            detections.append(Detection(
                bbox=[x1/w, y1/h, x2/w, y2/h],
                bbox_px=[int(x1), int(y1), int(x2), int(y2)],
                confidence=conf,
                class_id=cls,
                class_name="person",
            ))
        return detections

    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        """Batch inference untuk efisiensi pada multi-frame."""
        return [self.detect(f) for f in frames]


# Quick test
if __name__ == "__main__":
    import sys
    detector = PersonDetector(model_path="yolov8n.pt", conf_threshold=0.45)
    cap = cv2.VideoCapture(sys.argv[1] if len(sys.argv) > 1 else 0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        dets = detector.detect(frame)
        for d in dets:
            x1,y1,x2,y2 = d.bbox_px
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, f"person {d.confidence:.2f}",
                        (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        cv2.imshow("Person Detection", frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()