from ultralytics import YOLO
import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from src.detection.detector import Detection

PPE_CLASSES = {0: "helmet", 1: "safety_vest", 2: "safety_glasses"}

@dataclass
class PPEStatus:
    track_id: int
    has_helmet: bool = False
    has_vest: bool = False
    has_glasses: bool = False
    violations: List[str] = field(default_factory=list)

    @property
    def is_compliant(self) -> bool:
        return self.has_helmet and self.has_vest

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "has_helmet": self.has_helmet,
            "has_vest": self.has_vest,
            "has_glasses": self.has_glasses,
            "is_compliant": self.is_compliant,
            "violations": self.violations,
        }

class PPEDetector:
    """
    Mendeteksi APD (helmet, safety vest, safety glasses) pada crop
    area setiap person yang telah di-track.
    """
    def __init__(
        self,
        model_path: str = "models/weights/ppe_best.pt",
        conf_threshold: float = 0.40,
        device: str = "auto",
    ):
        self.model = YOLO(model_path)
        self.conf = conf_threshold
        self.device = device
        print(f"[PPEDetector] Loaded: {model_path}")

    def check_ppe(
        self,
        frame: np.ndarray,
        persons: List[Detection],
        expand_ratio: float = 0.1,  # expand bbox sedikit untuk tangkap APD di tepi
    ) -> Dict[int, PPEStatus]:
        """
        Untuk setiap person yang di-track, crop region-nya lalu
        jalankan inference PPE detection.
        """
        h, w = frame.shape[:2]
        ppe_map: Dict[int, PPEStatus] = {}

        for person in persons:
            if person.track_id is None:
                continue
            
            x1, y1, x2, y2 = person.bbox_px
            # expand crop
            pw, ph = x2-x1, y2-y1
            ex = int(pw * expand_ratio)
            ey = int(ph * expand_ratio)
            cx1 = max(0, x1-ex); cy1 = max(0, y1-ey)
            cx2 = min(w, x2+ex); cy2 = min(h, y2+ey)
            
            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                continue

            results = self.model(crop, conf=self.conf, verbose=False)[0]
            status = PPEStatus(track_id=person.track_id)

            for box in results.boxes:
                cls = int(box.cls[0])
                if cls == 0: status.has_helmet = True
                elif cls == 1: status.has_vest = True
                elif cls == 2: status.has_glasses = True

            # tentukan violations
            if not status.has_helmet:
                status.violations.append("no_helmet")
            if not status.has_vest:
                status.violations.append("no_vest")

            ppe_map[person.track_id] = status

        return ppe_map