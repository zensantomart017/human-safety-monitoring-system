import supervision as sv
import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque


@dataclass
class TrackState:
    """Menyimpan state sebuah track: history bbox dan appearance histogram."""
    track_id: int
    # Ring buffer 30 frame terakhir untuk smooth appearance model
    histograms: deque = field(default_factory=lambda: deque(maxlen=30))
    last_seen_frame: int = 0
    bbox_history: deque = field(default_factory=lambda: deque(maxlen=10))

    def update_appearance(self, crop: np.ndarray):
        """Hitung dan simpan HSV histogram dari crop person."""
        if crop.size == 0:
            return
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        # H: 50 bins, S: 60 bins — cukup diskriminatif, cukup cepat
        h_hist = cv2.calcHist([hsv], [0], None, [50], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [60], [0, 256])
        hist = np.concatenate([
            cv2.normalize(h_hist, h_hist).flatten(),
            cv2.normalize(s_hist, s_hist).flatten(),
        ])
        self.histograms.append(hist)

    def mean_histogram(self) -> Optional[np.ndarray]:
        """Rata-rata histogram dari semua frame yang tersimpan."""
        if not self.histograms:
            return None
        return np.mean(self.histograms, axis=0)


class StablePersonTracker:
    """
    ByteTrack + appearance-based ReID layer.

    Cara kerja:
    1. ByteTrack assign track_id per frame (bisa ID switch saat occlusion).
    2. ReID layer menyimpan appearance model (HSV histogram) tiap track.
    3. Saat track baru muncul, dibandingkan ke track yang recently lost.
    4. Jika similarity tinggi → restore ID lama, bukan assign ID baru.
    """

    def __init__(
        self,
        # --- ByteTrack params ---
        track_thresh: float = 0.35,   # ↓ dari 0.45: tangkap deteksi low-conf
        track_buffer: int  = 60,      # ↑ dari 30: simpan track 2 detik (30fps)
        match_thresh: float = 0.85,   # ↑ dari 0.80: IoU matching lebih ketat
        frame_rate: int = 30,

        # --- ReID params ---
        reid_similarity_thresh: float = 0.75,  # min similarity untuk restore ID
        reid_max_age: int = 90,                # max frame track "hilang" sebelum dihapus
    ):
        self.tracker = sv.ByteTracker(
            track_thresh=track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            frame_rate=frame_rate,
        )

        self.reid_similarity_thresh = reid_similarity_thresh
        self.reid_max_age = reid_max_age

        # State aktif: {bytetrack_id → TrackState}
        self._active: Dict[int, TrackState] = {}

        # State hilang: {bytetrack_id → TrackState} — kandidat restore
        self._lost: Dict[int, TrackState] = {}

        # Mapping ByteTrack ID → stable ID yang kita expose ke luar
        self._id_map: Dict[int, int] = {}

        self._frame_idx: int = 0
        self._next_stable_id: int = 1

    # ------------------------------------------------------------------
    # PUBLIC
    # ------------------------------------------------------------------

    def update(
        self,
        detections_xyxy: np.ndarray,   # shape (N, 4) pixel coords
        confidences: np.ndarray,        # shape (N,)
        frame: np.ndarray,              # BGR frame untuk ekstrak appearance
    ) -> List[Dict]:
        """
        Input:
            detections_xyxy : bbox hasil PersonDetector [[x1,y1,x2,y2], ...]
            confidences      : confidence score tiap bbox
            frame            : frame BGR lengkap

        Output:
            List[dict] dengan key: stable_id, bbox_px, confidence
        """
        self._frame_idx += 1

        if len(detections_xyxy) == 0:
            self._age_lost_tracks()
            return []

        sv_dets = sv.Detections(
            xyxy=detections_xyxy.astype(float),
            confidence=confidences,
        )
        tracked = self.tracker.update_with_detections(sv_dets)

        if len(tracked) == 0:
            self._age_lost_tracks()
            return []

        # ID ByteTrack yang aktif frame ini
        current_bt_ids = set(tracked.tracker_id.tolist())

        # Tandai track yang tidak muncul frame ini sebagai lost
        disappeared = set(self._active.keys()) - current_bt_ids
        for bt_id in disappeared:
            self._lost[bt_id] = self._active.pop(bt_id)

        results = []
        for i, bt_id in enumerate(tracked.tracker_id):
            x1, y1, x2, y2 = tracked.xyxy[i].astype(int)
            conf = float(tracked.confidence[i])

            # Crop person untuk appearance
            crop = self._safe_crop(frame, x1, y1, x2, y2)

            if bt_id not in self._active:
                # Track baru dari ByteTrack → coba match ke lost track
                stable_id = self._try_reid(crop, bt_id)
                self._active[bt_id] = TrackState(
                    track_id=stable_id,
                    last_seen_frame=self._frame_idx,
                )
                self._id_map[bt_id] = stable_id
            else:
                stable_id = self._active[bt_id].track_id

            # Update appearance model
            self._active[bt_id].update_appearance(crop)
            self._active[bt_id].last_seen_frame = self._frame_idx
            self._active[bt_id].bbox_history.append([x1, y1, x2, y2])

            results.append({
                "stable_id": stable_id,
                "bbox_px": [x1, y1, x2, y2],
                "confidence": conf,
            })

        self._age_lost_tracks()
        return results

    def reset(self):
        self.tracker.reset()
        self._active.clear()
        self._lost.clear()
        self._id_map.clear()
        self._frame_idx = 0
        self._next_stable_id = 1

    # ------------------------------------------------------------------
    # PRIVATE
    # ------------------------------------------------------------------

    def _try_reid(self, crop: np.ndarray, new_bt_id: int) -> int:
        """
        Coba cocokkan crop baru ke salah satu lost track via histogram similarity.
        Jika cocok → kembalikan stable_id lama (restore).
        Jika tidak → assign stable_id baru.
        """
        if not self._lost or crop.size == 0:
            return self._assign_new_id()

        # Hitung histogram crop baru
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h_hist = cv2.normalize(
            cv2.calcHist([hsv], [0], None, [50], [0, 180]), None
        ).flatten()
        s_hist = cv2.normalize(
            cv2.calcHist([hsv], [1], None, [60], [0, 256]), None
        ).flatten()
        new_hist = np.concatenate([h_hist, s_hist])

        best_score = 0.0
        best_bt_id = None

        for lost_bt_id, state in self._lost.items():
            ref_hist = state.mean_histogram()
            if ref_hist is None:
                continue
            # Bhattacharyya distance → similarity
            # cv2.HISTCMP_BHATTACHARYYA: 0 = identical, 1 = totally different
            dist = cv2.compareHist(
                new_hist.reshape(-1, 1).astype(np.float32),
                ref_hist.reshape(-1, 1).astype(np.float32),
                cv2.HISTCMP_BHATTACHARYYA,
            )
            similarity = 1.0 - dist

            if similarity > best_score:
                best_score = similarity
                best_bt_id = lost_bt_id

        if best_score >= self.reid_similarity_thresh and best_bt_id is not None:
            # Restore ID lama
            restored_state = self._lost.pop(best_bt_id)
            stable_id = restored_state.track_id
            return stable_id

        return self._assign_new_id()

    def _assign_new_id(self) -> int:
        sid = self._next_stable_id
        self._next_stable_id += 1
        return sid

    def _age_lost_tracks(self):
        """Hapus lost track yang sudah terlalu lama tidak muncul."""
        to_delete = [
            bt_id for bt_id, state in self._lost.items()
            if (self._frame_idx - state.last_seen_frame) > self.reid_max_age
        ]
        for bt_id in to_delete:
            del self._lost[bt_id]

    @staticmethod
    def _safe_crop(
        frame: np.ndarray,
        x1: int, y1: int, x2: int, y2: int,
        expand: float = 0.05,
    ) -> np.ndarray:
        """Crop frame dengan sedikit expand untuk tangkap konteks penampilan."""
        h, w = frame.shape[:2]
        pw, ph = x2 - x1, y2 - y1
        ex, ey = int(pw * expand), int(ph * expand)
        cx1 = max(0, x1 - ex)
        cy1 = max(0, y1 - ey)
        cx2 = min(w, x2 + ex)
        cy2 = min(h, y2 + ey)
        return frame[cy1:cy2, cx1:cx2]