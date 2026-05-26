import cv2
import numpy as np
from typing import List, Dict, Optional
from src.detection.detector import PersonDetector, Detection
from src.detection.ppe_detector import PPEDetector, PPEStatus
from src.tracking.tracker import PersonTracker
from src.utils.visualizer import Visualizer

class SafetyPipeline:
    """
    End-to-end pipeline: frame → detection → tracking → PPE check → annotated output.
    Ini adalah class utama yang dipanggil oleh FastAPI.
    """
    def __init__(
        self,
        person_model: str = "yolov8n.pt",
        ppe_model: str = "models/weights/ppe_best.pt",
        conf_person: float = 0.45,
        conf_ppe: float = 0.40,
        device: str = "auto",
    ):
        self.detector = PersonDetector(person_model, conf_person, device=device)
        self.ppe_detector = PPEDetector(ppe_model, conf_ppe, device=device)
        self.tracker = PersonTracker()
        self.visualizer = Visualizer()
        self.frame_count = 0
        self.violation_log: List[Dict] = []

    def process_frame(
        self,
        frame: np.ndarray,
        return_annotated: bool = True,
    ) -> Dict:
        """
        Input : BGR numpy frame
        Output: dict {persons, ppe_status, violations, annotated_frame, fps}
        """
        import time
        t0 = time.perf_counter()
        self.frame_count += 1

        # 1. Detect persons
        persons_raw = self.detector.detect(frame)

        # 2. Track persons (persistent ID)
        persons = self.tracker.update(persons_raw, frame.shape)

        # 3. PPE check per person
        ppe_map = self.ppe_detector.check_ppe(frame, persons)

        # 4. Collect violations
        violations = []
        for tid, status in ppe_map.items():
            if not status.is_compliant:
                violations.append({
                    "track_id": tid,
                    "violations": status.violations,
                    "frame": self.frame_count,
                })
        self.violation_log.extend(violations)

        latency_ms = (time.perf_counter() - t0) * 1000

        result = {
            "frame_id": self.frame_count,
            "latency_ms": round(latency_ms, 2),
            "fps": round(1000 / latency_ms, 1) if latency_ms > 0 else 0,
            "person_count": len(persons),
            "persons": [
                {
                    "track_id": p.track_id,
                    "bbox": p.bbox,
                    "bbox_px": p.bbox_px,
                    "confidence": round(p.confidence, 3),
                    "ppe_status": ppe_map.get(p.track_id, PPEStatus(p.track_id)).to_dict()
                    if p.track_id else None,
                }
                for p in persons
            ],
            "violations": violations,
        }

        if return_annotated:
            annotated = self.visualizer.draw(frame.copy(), persons, ppe_map)
            result["annotated_frame"] = annotated

        return result

    def reset(self):
        self.tracker.reset()
        self.frame_count = 0
        self.violation_log.clear()