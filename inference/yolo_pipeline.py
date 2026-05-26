import os
from pathlib import Path
from typing import List, Dict, Any, Union, Optional
from collections import deque
import numpy as np
import cv2
import torch
from ultralytics import YOLO


class _AppearanceRecord:
    def __init__(self, stable_id: int):
        self.stable_id  = stable_id
        self.histograms = deque(maxlen=30)
        self.last_frame = 0

    def update(self, crop: np.ndarray):
        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            return
        hsv    = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h_hist = cv2.normalize(cv2.calcHist([hsv], [0], None, [50], [0, 180]), None).flatten()
        s_hist = cv2.normalize(cv2.calcHist([hsv], [1], None, [60], [0, 256]), None).flatten()
        self.histograms.append(np.concatenate([h_hist, s_hist]))

    def mean_hist(self) -> Optional[np.ndarray]:
        if not self.histograms:
            return None
        return np.mean(self.histograms, axis=0)

    def similarity(self, other_hist: np.ndarray) -> float:
        ref = self.mean_hist()
        if ref is None:
            return 0.0
        dist = cv2.compareHist(
            ref.reshape(-1, 1).astype(np.float32),
            other_hist.reshape(-1, 1).astype(np.float32),
            cv2.HISTCMP_BHATTACHARYYA,
        )
        return float(1.0 - dist)


class StableIDMapper:
    def __init__(self, reid_threshold: float = 0.72, max_lost_age: int = 90):
        self.reid_threshold = reid_threshold
        self.max_lost_age   = max_lost_age
        self._active: Dict[int, _AppearanceRecord] = {}
        self._lost:   Dict[int, _AppearanceRecord] = {}
        self._yolo_to_stable: Dict[int, int] = {}
        self._frame_idx   = 0
        self._next_stable = 1

    def update(self, yolo_results, frame: np.ndarray) -> Dict[int, int]:
        self._frame_idx += 1
        current_yolo_ids = set()
        id_to_bbox: Dict[int, List[int]] = {}

        boxes = yolo_results[0].boxes if yolo_results else None
        if boxes is not None and boxes.id is not None:
            for box in boxes:
                yolo_id = int(box.id[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                current_yolo_ids.add(yolo_id)
                id_to_bbox[yolo_id] = [x1, y1, x2, y2]

        disappeared = set(self._active.keys()) - current_yolo_ids
        for yid in disappeared:
            rec = self._active.pop(yid)
            rec.last_frame = self._frame_idx
            self._lost[yid] = rec

        for yolo_id in current_yolo_ids:
            bbox = id_to_bbox[yolo_id]
            crop = self._crop(frame, *bbox)
            if yolo_id in self._active:
                self._active[yolo_id].update(crop)
            else:
                stable_id = self._reid_or_new(crop)
                self._yolo_to_stable[yolo_id] = stable_id
                rec = _AppearanceRecord(stable_id)
                rec.update(crop)
                self._active[yolo_id] = rec

        stale = [
            yid for yid, rec in self._lost.items()
            if (self._frame_idx - rec.last_frame) > self.max_lost_age
        ]
        for yid in stale:
            del self._lost[yid]

        return {
            yid: self._yolo_to_stable[yid]
            for yid in current_yolo_ids
            if yid in self._yolo_to_stable
        }

    def reset(self):
        self._active.clear()
        self._lost.clear()
        self._yolo_to_stable.clear()
        self._frame_idx   = 0
        self._next_stable = 1

    def _reid_or_new(self, crop: np.ndarray) -> int:
        if not self._lost or crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            return self._new_id()

        hsv    = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h_hist = cv2.normalize(cv2.calcHist([hsv], [0], None, [50], [0, 180]), None).flatten()
        s_hist = cv2.normalize(cv2.calcHist([hsv], [1], None, [60], [0, 256]), None).flatten()
        query_hist = np.concatenate([h_hist, s_hist])

        best_score, best_yid = 0.0, None
        for lost_yid, rec in self._lost.items():
            score = rec.similarity(query_hist)
            if score > best_score:
                best_score, best_yid = score, lost_yid

        if best_score >= self.reid_threshold and best_yid is not None:
            return self._lost.pop(best_yid).stable_id

        return self._new_id()

    def _new_id(self) -> int:
        sid = self._next_stable
        self._next_stable += 1
        return sid

    @staticmethod
    def _crop(frame: np.ndarray, x1, y1, x2, y2, expand=0.05) -> np.ndarray:
        h, w = frame.shape[:2]
        ex = int((x2 - x1) * expand)
        ey = int((y2 - y1) * expand)
        return frame[max(0, y1-ey):min(h, y2+ey), max(0, x1-ex):min(w, x2+ex)]


class SafetyPipeline:
    def __init__(self, config_path: str = "configs/system_config.yaml"):
        self.person_model = YOLO("yolov8s.pt")

        ppe_weights = "models/best.pt"
        if os.path.exists(ppe_weights):
            self.ppe_model       = YOLO(ppe_weights)
            self.has_ppe_weights = True
        else:
            print(f"[Warning] {ppe_weights} not found — using yolov8s fallback for PPE.")
            self.ppe_model       = YOLO("yolov8s.pt")
            self.has_ppe_weights = False

        self.tracker_config = "src/tracking/custom_bytetrack.yaml"
        self._id_mapper = StableIDMapper(reid_threshold=0.72, max_lost_age=90)

    def detect_person(self, source: Union[str, np.ndarray], conf: float = 0.25) -> Any:
        return self.person_model.predict(source=source, conf=conf, classes=[0], verbose=False)

    def track_persons(self, source: Union[str, np.ndarray], conf: float = 0.25) -> Any:
        results = self.person_model.track(
            source=source,
            tracker=self.tracker_config,
            conf=conf,
            persist=True,
            classes=[0],
            verbose=False,
        )

        if (
            isinstance(source, np.ndarray)
            and results
            and results[0].boxes is not None
            and results[0].boxes.id is not None
        ):
            stable_map = self._id_mapper.update(results, source)

            new_ids = [
                stable_map.get(int(box.id[0]), int(box.id[0]))
                for box in results[0].boxes
            ]

            if new_ids and results[0].boxes.data.shape[1] >= 7:
                data_clone = results[0].boxes.data.clone()
                data_clone[:, -3] = torch.tensor(
                    new_ids,
                    dtype=data_clone.dtype,
                    device=data_clone.device,
                )
                results[0].boxes.data = data_clone

        return results

    def detect_ppe(self, source: Union[str, np.ndarray], conf: float = 0.25) -> Any:
        if self.has_ppe_weights:
            return self.ppe_model.predict(source=source, conf=conf, verbose=False)
        else:
            demo_classes = [24, 27, 31]
            results = self.ppe_model.predict(source=source, conf=conf, classes=demo_classes, verbose=False)
            if results:
                results[0].names = {24: "helmet", 27: "vest", 31: "glasses"}
            return results

    def reset_tracker(self):
        self._id_mapper.reset()
        self.person_model.predictor = None