import cv2
import numpy as np
from typing import List, Dict
from src.detection.detector import Detection
from src.detection.ppe_detector import PPEStatus

COMPLIANT_COLOR   = (0, 200, 80)   # green
VIOLATION_COLOR   = (0, 60, 220)   # red (BGR)
UNKNOWN_COLOR     = (180, 180, 180)
FONT = cv2.FONT_HERSHEY_SIMPLEX

class Visualizer:
    def draw(
        self,
        frame: np.ndarray,
        persons: List[Detection],
        ppe_map: Dict[int, PPEStatus],
    ) -> np.ndarray:
        overlay = frame.copy()

        for person in persons:
            x1, y1, x2, y2 = person.bbox_px
            tid = person.track_id
            status = ppe_map.get(tid) if tid else None

            color = (
                COMPLIANT_COLOR if (status and status.is_compliant)
                else VIOLATION_COLOR if (status and not status.is_compliant)
                else UNKNOWN_COLOR
            )

            # bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # filled label background
            label = f"ID:{tid}" if tid else "person"
            conf_txt = f" {person.confidence:.2f}"
            (lw, lh), _ = cv2.getTextSize(label+conf_txt, FONT, 0.45, 1)
            cv2.rectangle(frame, (x1, y1-lh-8), (x1+lw+4, y1), color, -1)
            cv2.putText(frame, label+conf_txt,
                        (x1+2, y1-4), FONT, 0.45, (255,255,255), 1, cv2.LINE_AA)

            # violation badges
            if status and status.violations:
                badge_y = y1 + 4
                for viol in status.violations:
                    badge = viol.replace("no_", "NO ").upper()
                    (bw, bh), _ = cv2.getTextSize(badge, FONT, 0.38, 1)
                    cv2.rectangle(frame, (x1+2, badge_y), (x1+bw+6, badge_y+bh+4),
                                  VIOLATION_COLOR, -1)
                    cv2.putText(frame, badge, (x1+4, badge_y+bh),
                                FONT, 0.38, (255,255,255), 1, cv2.LINE_AA)
                    badge_y += bh + 8

        # HUD: person count
        cv2.putText(frame, f"Persons: {len(persons)}",
                    (10, 24), FONT, 0.65, (255,255,255), 2, cv2.LINE_AA)
        violations_total = sum(1 for p in persons
                               if ppe_map.get(p.track_id) and not ppe_map[p.track_id].is_compliant)
        cv2.putText(frame, f"Violations: {violations_total}",
                    (10, 50), FONT, 0.65, VIOLATION_COLOR, 2, cv2.LINE_AA)

        return cv2.addWeighted(overlay, 0.05, frame, 0.95, 0)