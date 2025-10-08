import cv2
import numpy as np
from typing import Tuple

class PhotoBoxDetector:
    """Detects black photo box within the template image."""
    @staticmethod
    def detect_black_box(template_bgr) -> Tuple[int, int, int, int]:
        hsv = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 0, 0], np.uint8)
        upper = np.array([179, 80, 60], np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            raise RuntimeError("No black rectangle found in template.")
        x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
        return x + 2, y + 2, max(1, w - 4), max(1, h - 4)
