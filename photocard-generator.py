#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import io
import re
import csv
import cv2
import requests
import numpy as np
from pathlib import Path
from PIL import Image


try:
    import pillow_avif  # noqa: F401  # optional
except Exception:
    pass


# ---------------- CONFIG ----------------
template_path = Path("./templates/version-1.png")  # <-- single template file
csv_path = Path("./articles/prothomalo_130920251242.csv")
out_dir = Path("./photocards/prothomalo_130920251242-photocard")
naming_mode = "index"  # "index" or "title"
# ----------------------------------------

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".avif"}


def load_cv(path_or_url: str) -> np.ndarray:
    """Load an image (URL or local) into BGR np.ndarray."""
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

    # Fallback via PIL if OpenCV decode failed (e.g., for some AVIF/WebP edge cases)
    pil = Image.open(io.BytesIO(data)).convert("RGB")
    rgb = np.array(pil)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def detect_black_box(template_bgr: np.ndarray):
    """Find the main black rectangle (mask area) on the template."""
    hsv = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 0], np.uint8)
    upper = np.array([179, 80, 60], np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        raise RuntimeError("No black rectangle found in template.")
    x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    # inset a couple pixels so seams aren’t visible
    return x + 2, y + 2, max(1, w - 4), max(1, h - 4)


def fit_cover(src_bgr: np.ndarray, tw: int, th: int) -> np.ndarray:
    """Resize/crop like CSS object-fit: cover to exactly (tw, th)."""
    sh, sw = src_bgr.shape[:2]
    src_as = sw / sh
    tgt_as = tw / th
    if src_as < tgt_as:
        new_w = tw
        new_h = int(new_w / src_as)
    else:
        new_h = th
        new_w = int(new_h * src_as)
    resized = cv2.resize(src_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    x = (new_w - tw) // 2
    y = (new_h - th) // 2
    return resized[y : y + th, x : x + tw]


def build_overlay_mask(template_bgr: np.ndarray) -> np.ndarray:
    """Return 3-channel mask of non-black areas to keep from the template."""
    hsv = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2HSV)
    black_mask = cv2.inRange(hsv, np.array([0, 0, 0], np.uint8), np.array([179, 80, 60], np.uint8))
    nonblack_mask = cv2.bitwise_not(black_mask)
    return cv2.merge([nonblack_mask, nonblack_mask, nonblack_mask])


def composite_with_template(base_bgr: np.ndarray, template_bgr: np.ndarray, mask3: np.ndarray) -> np.ndarray:
    """Keep non-black regions from the template on top of base."""
    return (base_bgr & (~mask3)) + (template_bgr & mask3)


def safe_slug(text: str, max_len: int = 60) -> str:
    s = re.sub(r"[^\w\-]+", "_", text, flags=re.UNICODE).strip("_")
    return s[: max_len] or "untitled"


def read_csv_rows(csv_path: Path):
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def main():
    if not template_path.is_file():
        raise FileNotFoundError(f"Template file not found: {template_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Load template once and precompute masks/coordinates
    template_bgr = load_cv(str(template_path))
    overlay_src = template_bgr.copy()  # the decorative bits we’ll keep on top
    mask3 = build_overlay_mask(template_bgr)
    x, y, w, h = detect_black_box(template_bgr)

    rows = read_csv_rows(csv_path)
    total = len(rows)

    for i, row in enumerate(rows, start=1):
        # Try common column names for image
        img_url = (
            (row.get("article_image") or row.get("image") or row.get("img") or "").strip()
        )
        if not img_url:
            print(f"[{i}/{total}] skipped (no image url)")
            continue

        if naming_mode == "title":
            title = (row.get("title") or "").strip()
            base_name = safe_slug(title) if title else str(i)
        else:
            base_name = str(i)

        out_path = out_dir / f"{base_name}_final.png"

        try:
            photo_bgr = load_cv(img_url)
            fitted = fit_cover(photo_bgr, w, h)

            base = template_bgr.copy()
            base[y : y + h, x : x + w] = fitted

            final_bgr = composite_with_template(base, overlay_src, mask3)
            cv2.imwrite(str(out_path), final_bgr)
            print(f"[{i}/{total}] OK  -> {out_path.name}")
        except Exception as e:
            print(f"[{i}/{total}] FAIL ({img_url}): {e}")

    print(f"✅ Done. Results saved in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
