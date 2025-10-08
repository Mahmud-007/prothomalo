from __future__ import annotations
import io, re, os
from pathlib import Path
import numpy as np
import cv2
import requests
from PIL import Image

# Optional AVIF support (no-op if unavailable)
try:
    import pillow_avif  # noqa: F401
except Exception:
    pass

class ImageIO:
    """Loading remote/local images and filename utilities."""

    IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".avif"}

    @staticmethod
    def load_cv(path_or_url: str):
        if path_or_url.startswith(("http://", "https://")):
            r = requests.get(path_or_url, timeout=60)
            r.raise_for_status()
            data = r.content
        else:
            with open(path_or_url, "rb") as f:
                data = f.read()
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None:
            return img
        pil = Image.open(io.BytesIO(data)).convert("RGB")
        rgb = np.array(pil)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    @staticmethod
    def safe_slug(text: str, max_len: int = 60) -> str:
        s = re.sub(r"[^\w\-]+", "_", text, flags=re.UNICODE).strip("_")
        return s[:max_len] or "untitled"
