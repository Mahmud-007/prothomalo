#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Photocard generator (single template, CTA removed)
- Places each article_image into template photo box (auto-detected black rect)
- Renders Bengali category pill, title, date, source
- Easy position control: change SHIFT_*_DY_PERC (± moves element up/down)
"""

import io
import os
import re
import csv
import html
import atexit
import cv2
import requests
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from PIL import Image, ImageDraw
from playwright.sync_api import sync_playwright, Browser

# Optional AVIF
try:
    import pillow_avif  # noqa: F401
except Exception:
    pass


# =========================
# ======  CONFIG  =========
# =========================

TEMPLATE_PATH = Path("./templates/version-1.png")
CSV_PATH      = Path("./articles/prothomalo_120920251152.csv")
OUT_DIR       = Path("./photocards/prothomalo_120920251152-photocard")

# Save names: "index" or "title"
NAMING_MODE = "index"

# EXACT CSV KEYS you requested
CSV_KEYS: Dict[str, List[str]] = {
    "image":    ["article_image"],
    "title":    ["article_title"],
    "date":     ["published_date_bn"],
    "source":   ["source"],
    "category": ["category_bn"],
    "cta":      [],  # not used
}

# Photo window detection (auto). If you prefer fixed pixels, set to False and edit PHOTO_BOX_PIXELS.
DETECT_PHOTO_BOX = True
PHOTO_BOX_PIXELS: Tuple[int, int, int, int] = (120, 160, 1840, 1000)  # (l, t, r, b) used when DETECT_PHOTO_BOX=False

# Brand colors (tuned to your sample)
RED  = "#C4161C"
DARK = "#222222"
MID  = "#4A4A4A"

# ---------- Layout: base positions (percent of template W/H) ----------
@dataclass
class BoxPerc:
    l: float; t: float; r: float; b: float
    def to_px(self, W: int, H: int) -> Tuple[int, int, int, int]:
        return (int(self.l*W), int(self.t*H), int(self.r*W), int(self.b*H))

# Measured to your example; tweak these for coarse moves
TITLE_BOX_PERC = BoxPerc(0.085, 0.66, 0.915, 0.76)   # red title
DATE_BOX_PERC  = BoxPerc(0.40, 0.84, 0.60, 0.875)    # date
SRC_BOX_PERC   = BoxPerc(0.36, 0.895, 0.64, 0.93)    # source

# Divider line (centered)
DIVIDER_Y_PERC     = 0.885
DIVIDER_WIDTH_PERC = 0.28
DIVIDER_THICK_PERC = 0.004

# Category pill geometry (percent of W/H)
PILL_TOP_OFFSET_PERC = -0.025  # relative to photo bottom; negative overlaps
PILL_MIN_WIDTH_PERC  = 0.12
PILL_H_PERC          = 0.045
PILL_H_PAD_PERC      = 0.020
PILL_RADIUS_PERC     = 0.022

# ---------- Fine control: quick UP/DOWN nudges (percent of template height) ----------
# Positive moves element DOWN; negative moves UP
SHIFT_TITLE_DY_PERC   = -0.022   # e.g., +0.01 moves title down by 1% of H
SHIFT_DATE_DY_PERC    = -0.03
SHIFT_SOURCE_DY_PERC  = -0.05
SHIFT_PILL_DY_PERC    = 0.000   # category pill (extra on top of PILL_TOP_OFFSET_PERC)

# ---------- Font sizing (start; will auto-shrink to fit boxes) ----------
TITLE_FONT_START_PERC = 0.079; TITLE_FONT_MIN_PERC = 0.045
DATE_FONT_START_PERC  = 0.018; DATE_FONT_MIN_PERC  = 0.006
SRC_FONT_START_PERC   = 0.018; SRC_FONT_MIN_PERC   = 0.006
PILL_FONT_START_PERC  = 0.018; PILL_FONT_MIN_PERC  = 0.006

DEVICE_SCALE_FACTOR   = 1
DEBUG_KEEP_TEXT_PNGS  = False


# =========================
# =====  STYLES  ==========
# =========================

@dataclass
class TextStyle:
    font_path: Path
    color: str
    font_px_start: int
    font_px_min: int
    line_height: float = 1.22
    padding_px: int = 2
    text_align: str = "center"
    text_shadow: str = "0 2px 6px rgba(0,0,0,.25)"


# =========================
# ======  CORE  ===========
# =========================

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".avif"}

_browser_singleton: Optional[Browser] = None
_playwright_ctx = None


def get_browser() -> Browser:
    """Single Chromium reused for speed."""
    global _browser_singleton, _playwright_ctx
    if _browser_singleton is None:
        _playwright_ctx = sync_playwright().start()
        _browser_singleton = _playwright_ctx.chromium.launch()
        atexit.register(_shutdown_browser)
    return _browser_singleton


def _shutdown_browser():
    global _browser_singleton, _playwright_ctx
    try:
        if _browser_singleton:
            _browser_singleton.close()
    except Exception:
        pass
    try:
        if _playwright_ctx:
            _playwright_ctx.stop()
    except Exception:
        pass
    _browser_singleton = None


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
    if img is not None:
        return img
    pil = Image.open(io.BytesIO(data)).convert("RGB")
    rgb = np.array(pil)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def detect_black_box(template_bgr: np.ndarray) -> Tuple[int, int, int, int]:
    hsv = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 0], np.uint8)
    upper = np.array([179, 80, 60], np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        raise RuntimeError("No black rectangle found in template.")
    x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    return x + 2, y + 2, max(1, w - 4), max(1, h - 4)


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
    return resized[y:y+th, x:x+tw]


def build_overlay_mask(template_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2HSV)
    black_mask = cv2.inRange(hsv, np.array([0, 0, 0], np.uint8), np.array([179, 80, 60], np.uint8))
    nonblack = cv2.bitwise_not(black_mask)
    return cv2.merge([nonblack, nonblack, nonblack])


def composite_with_template(base_bgr: np.ndarray, template_bgr: np.ndarray, mask3: np.ndarray) -> np.ndarray:
    return (base_bgr & (~mask3)) + (template_bgr & mask3)


def safe_slug(text: str, max_len: int = 60) -> str:
    s = re.sub(r"[^\w\-]+", "_", text, flags=re.UNICODE).strip("_")
    return s[:max_len] or "untitled"


def read_csv_rows(csv_path: Path) -> List[dict]:
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def get_val(row: dict, keys: List[str]) -> str:
    # exact keys as provided; just return the first non-empty
    for k in keys:
        v = (row.get(k) or "").strip()
        if v:
            return v
    return ""


# ---------- Playwright text rendering (copy to RAM) ----------

def _render_text_png_to_file(text: str, max_width: int, style: TextStyle, out_path: Path):
    font_url = style.font_path.resolve().as_uri()
    text_html = html.escape(text)
    html_doc = f"""<!doctype html>
<html><head><meta charset="utf-8">
<style>
@font-face {{
  font-family: 'BanglaFont';
  src: url('{font_url}') format('truetype');
}}
html,body {{ margin:0; padding:0; background:rgba(0,0,0,0); }}
.wrap {{
  width:{max_width - style.padding_px}px;
  padding:{style.padding_px}px;
  font-family:'BanglaFont','Noto Sans Bengali','Hind Siliguri',sans-serif;
  font-size:{style.font_px_start}px;
  line-height:{style.line_height};
  color:{style.color};
  text-align:{style.text_align};
  white-space:normal;
  word-break:break-word;
}}
</style></head>
<body><div class="wrap">{text_html}</div></body></html>"""

    browser = get_browser()
    page = browser.new_page(viewport={"width": max_width, "height": 1600}, device_scale_factor=DEVICE_SCALE_FACTOR)
    page.set_content(html_doc)
    page.wait_for_load_state("networkidle")
    page.locator(".wrap").screenshot(path=str(out_path), omit_background=True)
    page.close()


def render_text_fit_box(text: str, box_w: int, box_h: int, style: TextStyle, tmp_name: str) -> Image.Image:
    font_px = style.font_px_start
    tmp_path = Path(f"_{tmp_name}.png")
    while True:
        style_pass = TextStyle(
            font_path=style.font_path,
            color=style.color,
            font_px_start=font_px,
            font_px_min=style.font_px_min,
            line_height=style.line_height,
            padding_px=style.padding_px,
            text_align=style.text_align,
            text_shadow=style.text_shadow,
        )
        _render_text_png_to_file(text, box_w, style_pass, tmp_path)
        with Image.open(tmp_path).convert("RGBA") as on_disk:
            in_mem = on_disk.copy()
        if in_mem.height <= box_h or font_px <= style.font_px_min:
            if not DEBUG_KEEP_TEXT_PNGS:
                try: os.remove(tmp_path)
                except OSError: pass
            return in_mem
        font_px = max(style.font_px_min, font_px - 6)


def paste_center(base_rgba: Image.Image, overlay_rgba: Image.Image, box_px: Tuple[int, int, int, int], dy_px: int = 0):
    l, t, r, b = box_px
    # apply vertical shift
    t += dy_px; b += dy_px
    bw, bh = max(1, r - l), max(1, b - t)
    x = l + (bw - overlay_rgba.width) // 2
    y = t + (bh - overlay_rgba.height) // 2
    base_rgba.alpha_composite(overlay_rgba, (x, y))


# ---------- Category pill ----------
def render_category_pill(text: str, W: int, H: int, tmp_name: str = "pill") -> Image.Image:
    pill_h = int(PILL_H_PERC * H)
    font_px_start = int(PILL_FONT_START_PERC * W)
    font_px_min   = int(PILL_FONT_MIN_PERC * W)
    min_w         = int(PILL_MIN_WIDTH_PERC * W)
    hpad          = int(PILL_H_PAD_PERC * W)
    radius        = max(6, int(PILL_RADIUS_PERC * W))

    style = TextStyle(
        font_path=Path("./fonts/NotoSansBengali_Condensed-Bold.ttf"),
        color="#FFFFFF",
        font_px_start=font_px_start,
        font_px_min=font_px_min,
        line_height=1.0,
        padding_px=0,
        text_align="center",
        text_shadow="0 1px 2px rgba(0,0,0,.25)",
    )

    tmp_text = Path(f"_{tmp_name}_text.png")
    _render_text_png_to_file(text, max_width=W, style=style, out_path=tmp_text)
    with Image.open(tmp_text).convert("RGBA") as on_disk:
        text_img = on_disk.copy()
    try: os.remove(tmp_text)
    except OSError: pass

    pill_w = max(min_w, text_img.width + 2*hpad)

    font_url = style.font_path.resolve().as_uri()
    html_doc = f"""<!doctype html>
<html><head><meta charset="utf-8">
<style>
@font-face {{ font-family:'BanglaFont'; src:url('{font_url}') format('truetype'); }}
html,body {{ margin:0; padding:0; background:rgba(0,0,0,0); }}
.pill {{
  display:inline-flex; align-items:center; justify-content:center;
  width:{100}px; height:{50}px; border-radius:{radius}px;
  background:{RED}; color:#fff; font-family:'BanglaFont','Noto Sans Bengali','Hind Siliguri',sans-serif;
  font-size:{style.font_px_start}px; line-height:1; padding:0;
}}
</style></head>
<body><div class="pill">{html.escape(text)}</div></body></html>"""

    tmp_pill = Path(f"_{tmp_name}.png")
    browser = get_browser()
    page = browser.new_page(viewport={"width": pill_w, "height": pill_h}, device_scale_factor=DEVICE_SCALE_FACTOR)
    page.set_content(html_doc)
    page.wait_for_load_state("networkidle")
    page.locator(".pill").screenshot(path=str(tmp_pill), omit_background=True)
    page.close()

    with Image.open(tmp_pill).convert("RGBA") as on_disk2:
        pill_img = on_disk2.copy()
    try: os.remove(tmp_pill)
    except OSError: pass
    return pill_img


# =========================
# ========= MAIN ==========
# =========================

def main():
    if not TEMPLATE_PATH.is_file():
        raise FileNotFoundError(f"Template not found: {TEMPLATE_PATH}")
    font_path = Path("./fonts/NotoSansBengali_Condensed-Bold.ttf")
    if not font_path.is_file():
        raise FileNotFoundError(f"Bangla TTF not found: {font_path}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    template_bgr = load_cv(str(TEMPLATE_PATH))
    H, W = template_bgr.shape[:2]

    # Convert perc boxes to px
    title_box_px = TITLE_BOX_PERC.to_px(W, H)
    date_box_px  = DATE_BOX_PERC.to_px(W, H)
    src_box_px   = SRC_BOX_PERC.to_px(W, H)

    # Divider px
    line_len = int(DIVIDER_WIDTH_PERC * W)
    line_y   = int(DIVIDER_Y_PERC * H)
    thick    = max(2, int(DIVIDER_THICK_PERC * H))
    x1 = W//2 - line_len//2
    x2 = W//2 + line_len//2

    # Styles (start font size scales with W)
    title_style = TextStyle(font_path, RED,  int(TITLE_FONT_START_PERC*W), int(TITLE_FONT_MIN_PERC*W), 1.18, 4, "center", "0 2px 6px rgba(0,0,0,.25)")
    date_style  = TextStyle(font_path, MID,  int(DATE_FONT_START_PERC*W),  int(DATE_FONT_MIN_PERC*W),  1.12,  4, "center", "0 1px 2px rgba(0,0,0,.10)")
    src_style   = TextStyle(font_path, DARK, int(SRC_FONT_START_PERC*W),   int(SRC_FONT_MIN_PERC*W),   1.12,  4, "center", "0 1px 2px rgba(0,0,0,.10)")

    # Overlays & photo box
    overlay_src = template_bgr.copy()
    mask3 = build_overlay_mask(template_bgr)

    if DETECT_PHOTO_BOX:
        px, py, pw, ph = detect_black_box(template_bgr)
    else:
        l, t, r, b = PHOTO_BOX_PIXELS
        px, py, pw, ph = l, t, r - l, b - t

    rows = read_csv_rows(CSV_PATH)
    total = len(rows)

    for i, row in enumerate(rows, start=1):
        img_url    = get_val(row, CSV_KEYS["image"])
        title_text = get_val(row, CSV_KEYS["title"])
        date_text  = get_val(row, CSV_KEYS["date"])
        src_text   = get_val(row, CSV_KEYS["source"])
        cat_text   = get_val(row, CSV_KEYS["category"])

        if not img_url:
            print(f"[{i}/{total}] skipped (no image url)")
            continue

        # file name
        if NAMING_MODE == "title" and title_text:
            base_name = safe_slug(title_text)
        else:
            base_name = str(i)
        out_path = OUT_DIR / f"{base_name}_final.png"

        try:
            # 1) photo into window
            photo_bgr = load_cv(img_url)
            fitted = fit_cover(photo_bgr, pw, ph)
            base = template_bgr.copy()
            base[py:py+ph, px:px+pw] = fitted

            # 2) template overlay
            composed_bgr = composite_with_template(base, overlay_src, mask3)

            # 3) draw texts
            card_rgba = Image.fromarray(cv2.cvtColor(composed_bgr, cv2.COLOR_BGR2RGBA))

            # Category pill: bottom-center of photo; add global shift
            if cat_text:
                pill = render_category_pill(cat_text, W, H, tmp_name="pill")
                pill_x = W//2 - pill.width//2
                pill_y = int(py + ph + (PILL_TOP_OFFSET_PERC + SHIFT_PILL_DY_PERC) * H)
                card_rgba.alpha_composite(pill, (pill_x, pill_y))

            # Title
            if title_text:
                tw, th = title_box_px[2]-title_box_px[0], title_box_px[3]-title_box_px[1]
                title_png = render_text_fit_box(title_text, tw, th, title_style, "title_tmp")
                dy_px = int(SHIFT_TITLE_DY_PERC * H)
                paste_center(card_rgba, title_png, title_box_px, dy_px=dy_px)

            # Date
            if date_text:
                dw, dh = date_box_px[2]-date_box_px[0], date_box_px[3]-date_box_px[1]
                date_png = render_text_fit_box(date_text, dw, dh, date_style, "date_tmp")
                dy_px = int(SHIFT_DATE_DY_PERC * H)
                paste_center(card_rgba, date_png, date_box_px, dy_px=dy_px)

            # Divider line (between date and source)
            # draw = ImageDraw.Draw(card_rgba)
            # draw.rectangle([x1, line_y, x2, line_y + thick], fill=RED)

            # Source
            if src_text:
                sw, sh = src_box_px[2]-src_box_px[0], src_box_px[3]-src_box_px[1]
                src_png = render_text_fit_box(src_text, sw, sh, src_style, "src_tmp")
                dy_px = int(SHIFT_SOURCE_DY_PERC * H)
                paste_center(card_rgba, src_png, src_box_px, dy_px=dy_px)

            # 4) save
            final_bgr = cv2.cvtColor(np.array(card_rgba), cv2.COLOR_RGBA2BGR)
            cv2.imwrite(str(out_path), final_bgr)
            print(f"[{i}/{total}] OK -> {out_path.name} | title: {title_text[:50]!r}")

        except Exception as e:
            print(f"[{i}/{total}] FAIL ({img_url}): {e}")

    print(f"✅ Done. Results saved in {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
