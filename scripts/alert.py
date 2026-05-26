import httpx
import cv2
import numpy as np
import os
from typing import List
from dotenv import load_dotenv

load_dotenv()

class AlertService:
    def __init__(self):
        self.token   = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.enabled = bool(self.token and self.chat_id)
        if not self.enabled:
            print("[AlertService] Telegram not configured — alerts disabled.")

    async def send_violation_alert(
        self,
        track_id: int,
        violations: List[str],
        frame: np.ndarray = None,
    ):
        if not self.enabled:
            print(f"[ALERT] Track ID {track_id}: {violations}")
            return

        viol_str = ", ".join(v.replace("_", " ").upper() for v in violations)
        caption = (
            f"⚠️ *Safety Violation Detected*\n"
            f"Track ID : `{track_id}`\n"
            f"Violation: `{viol_str}`"
        )

        async with httpx.AsyncClient() as client:
            if frame is not None:
                _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                await client.post(
                    f"https://api.telegram.org/bot{self.token}/sendPhoto",
                    data={"chat_id": self.chat_id, "caption": caption, "parse_mode": "Markdown"},
                    files={"photo": ("frame.jpg", buf.tobytes(), "image/jpeg")},
                    timeout=10.0,
                )
            else:
                await client.post(
                    f"https://api.telegram.org/bot{self.token}/sendMessage",
                    json={"chat_id": self.chat_id, "text": caption, "parse_mode": "Markdown"},
                    timeout=10.0,
                )