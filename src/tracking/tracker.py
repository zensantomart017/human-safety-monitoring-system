import supervision as sv
import numpy as np
from typing import List
from src.detection.detector import Detection

class PersonTracker:
    """
    ByteTrack wrapper menggunakan supervision library.
    Mempertahankan track_id stabil meskipun deteksi confidence rendah.
    """
    def __init__(
        self,
        track_thresh: float = 0.45,
        track_buffer: int = 30,    # frame sebelum track dihapus
        match_thresh: float = 0.8, # IoU matching threshold
        frame_rate: int = 30,
    ):
        self.tracker = sv.ByteTracker(
            track_thresh=track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            frame_rate=frame_rate,
        )
        self._id_switches = 0

    def update(self, detections: List[Detection], frame_shape: tuple) -> List[Detection]:
        """
        Input : List[Detection] dari PersonDetector
        Output: List[Detection] dengan track_id terisi
        """
        if not detections:
            return []

        h, w = frame_shape[:2]
        
        # konversi ke sv.Detections
        xyxy = np.array([d.bbox_px for d in detections], dtype=float)
        confs = np.array([d.confidence for d in detections], dtype=float)
        sv_dets = sv.Detections(xyxy=xyxy, confidence=confs)

        # update tracker
        tracked = self.tracker.update_with_detections(sv_dets)

        # map balik ke Detection dengan track_id
        result = []
        for i, tracker_id in enumerate(tracked.tracker_id):
            x1, y1, x2, y2 = tracked.xyxy[i]
            result.append(Detection(
                bbox=[x1/w, y1/h, x2/w, y2/h],
                bbox_px=[int(x1), int(y1), int(x2), int(y2)],
                confidence=float(tracked.confidence[i]),
                class_id=0,
                class_name="person",
                track_id=int(tracker_id),
            ))
        return result

    def reset(self):
        self.tracker.reset()
        self._id_switches = 0