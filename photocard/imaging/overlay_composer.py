import cv2
import numpy as np

class OverlayComposer:
    """Build overlay mask and composite with template."""
    @staticmethod
    def build_overlay_mask(template_bgr):
        hsv = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2HSV)
        black_mask = cv2.inRange(
            hsv, np.array([0, 0, 0], np.uint8), np.array([179, 80, 60], np.uint8)
        )
        nonblack = cv2.bitwise_not(black_mask)
        return cv2.merge([nonblack, nonblack, nonblack])

    @staticmethod
    def composite_with_template(base_bgr, template_bgr, mask3):
        return (base_bgr & (~mask3)) + (template_bgr & mask3)
