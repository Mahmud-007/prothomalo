#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Photocard generator (image only)
- Places each article_image into the black box area of the template
- No text or category rendering
"""

import os
import cv2
import csv
import requests
import numpy as np
from pathlib import Path
import sys

# --- Configuration ---
def get_site():
    # 1️⃣ try command-line arg
    if "--site" in sys.argv:
        idx = sys.argv.index("--site")
        if idx + 1 < len(sys.argv):
            return sys.argv[idx + 1].strip().lower()
    # 2️⃣ fallback to environment variable
    return os.getenv("SITE", "").strip().lower()
SITE = get_site()
print(f"INFO: SITE={SITE!r}")
TEMPLATE_PATH = Path("./templates/version-1.png")

if SITE == "prothomalo":
	CSV_PATH = Path("./articles/prothomalo.csv")
	OUT_DIR  = Path("./photocards/prothomalo-photocard")
elif SITE == "kalbela":
	CSV_PATH = Path("./articles/kalbela.csv")
	OUT_DIR  = Path("./photocards/kalbela-photocard")
else:
    CSV_PATH = Path("./articles/prothomalo.csv")
    OUT_DIR = Path("./photocards/prothomalo-photocard")

# Column name in CSV that contains the image URL
IMAGE_KEY = "article_image"

# Auto-detect black box where the image should go
def detect_black_box(template_bgr: np.ndarray):
    hsv = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 0], np.uint8)
    upper = np.array([179, 80, 60], np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        raise RuntimeError("No black rectangle found in template.")
    x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    return x + 2, y + 2, max(1, w - 4), max(1, h - 4)

# Load image from URL or local path
def load_cv(path_or_url: str) -> np.ndarray:
    if path_or_url.startswith(("http://", "https://")):
        r = requests.get(path_or_url, timeout=60)
        r.raise_for_status()
        data = r.content
    else:
        with open(path_or_url, "rb") as f:
            data = f.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image.")
    return img

# Fit image into the target window (cover style)
def fit_cover(src_bgr: np.ndarray, tw: int, th: int) -> np.ndarray:
    sh, sw = src_bgr.shape[:2]
    src_as = sw / sh
    tgt_as = tw / th
    if src_as < tgt_as:
        new_w, new_h = tw, int(tw / src_as)
    else:
        new_h, new_w = th, int(th * src_as)
    resized = cv2.resize(src_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    x = (new_w - tw) // 2
    y = (new_h - th) // 2
    return resized[y:y + th, x:x + tw]

# --- Main function ---
def main():
    if not TEMPLATE_PATH.exists():
        raise FileNotFoundError("Template image not found.")
    if not CSV_PATH.exists():
        raise FileNotFoundError("CSV file not found.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    template_bgr = cv2.imread(str(TEMPLATE_PATH))
    px, py, pw, ph = detect_black_box(template_bgr)

    with open(CSV_PATH, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    total = len(rows)
    for i, row in enumerate(rows, 1):
        img_url = (row.get(IMAGE_KEY) or "").strip()
        if not img_url:
            print(f"[{i}/{total}] Skipped (no image URL)")
            continue

        try:
            photo_bgr = load_cv(img_url)
            fitted = fit_cover(photo_bgr, pw, ph)
            base = template_bgr.copy()
            base[py:py + ph, px:px + pw] = fitted

            out_path = OUT_DIR / f"{i}.png"
            cv2.imwrite(str(out_path), base)
            print(f"[{i}/{total}] OK -> {out_path.name}")
        except Exception as e:
            print(f"[{i}/{total}] FAIL ({img_url}): {e}")

    print(f"✅ Done. Images saved in {OUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
