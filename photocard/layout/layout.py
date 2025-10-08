from dataclasses import dataclass
from typing import Tuple
from PIL import Image

@dataclass
class BoxPerc:
    l: float
    t: float
    r: float
    b: float
    def to_px(self, W: int, H: int) -> Tuple[int, int, int, int]:
        return (int(self.l * W), int(self.t * H), int(self.r * W), int(self.b * H))

class Layout:
    """Relative→pixel conversion and center paste helpers."""
    @staticmethod
    def to_px(box, W: int, H: int):
        return BoxPerc(*box).to_px(W, H)

    @staticmethod
    def paste_center(base_rgba: Image.Image, overlay_rgba: Image.Image,
                     box_px: Tuple[int, int, int, int], dy_px: int = 0):
        l, t, r, b = box_px
        t += dy_px
        b += dy_px
        bw, bh = max(1, r - l), max(1, b - t)
        x = l + (bw - overlay_rgba.width) // 2
        y = t + (bh - overlay_rgba.height) // 2
        base_rgba.alpha_composite(overlay_rgba, (x, y))
