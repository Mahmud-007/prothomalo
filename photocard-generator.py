#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Photocard generator (single template, measured to sample layout)

Features
- Places each article_image into the template's photo window (auto-detected black rect)
- Category pill centered at photo bottom
- Title, CTA, Date, Source in lower white area (with red divider line)
- Bengali text rendered via Playwright + TTF, auto-shrinks to fit boxes
- Percentage-based layout → scales to any template size
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

# Optional AVIF (safe if missing)
try:
    import pillow_avif  # noqa: F401
except Exception:
    pass


# =========================
# ======  CONFIG  =========
# =========================

# Single template file
TEMPLATE_PATH = Path("./templates/version-1.png")

# Input CSV (needs at least image + title; others optional)
CSV_PATH = Path("./articles/prothomalo_120920251152.csv")

# Output folder
OUT_DIR = Path("./photocards/prothomalo_120920251152-photocard")

# Save names: "index" or "title"
NAMING_MODE = "index"

# CSV field mapping (first non-empty wins)
CSV_KEYS: Dict[str, List[str]] = {
    "image":    ["article_image",],
    "title":    ["article_title",],
    "date":     ["published_date_bn",],
    "source":   ["source"],
    "category": ["category_bn"],
    "cta":[]
}

# If your template has a black rectangle where the photo goes, keep this True.
# Otherwise set to False and provide PHOTO_BOX_PIXELS (left, top, right, bottom).
DETECT_PHOTO_BOX = True
PHOTO_BOX_PIXELS: Tuple[int, int, int, int] = (120, 160, 1840, 1000)  # only used if DETECT_PHOTO_BOX=False

# Brand colors (tuned to your sample)
RED = "#C4161C"
DARK = "#222222"
MID  = "#4A4A4A"


# =========================
# =====  LAYOUT  ==========
# =========================
# All below are **ratios** of template width/height so this scales with any template size.

@dataclass
class BoxPerc:
    """A box in percentage: left, top, right, bottom (fractions of W/H)."""
    l: float
    t: float
    r: float
    b: float

    def to_px(self, W: int, H: int) -> Tuple[int, int, int, int]:
        return (int(self.l*W), int(self.t*H), int(self.r*W), int(self.b*H))

# Measured to your mock (square layout). Tweak these 4 boxes if you want micro-adjustments.
TITLE_BOX_PERC = BoxPerc(0.085, 0.66, 0.915, 0.76)   # big red title
CTA_BOX_PERC   = BoxPerc(0.36, 0.79, 0.64, 0.83)     # "বিস্তারিত কমেন্ট"
DATE_BOX_PERC  = BoxPerc(0.40, 0.84, 0.60, 0.875)    # date line
SRC_BOX_PERC   = BoxPerc(0.36, 0.895, 0.64, 0.93)    # source line (under red divider)

# Red divider line (x center + length + thickness), as fractions of W/H
DIVIDER_Y_PERC      = 0.885
DIVIDER_WIDTH_PERC  = 0.28
DIVIDER_THICK_PERC  = 0.004

# Category pill dimensions relative to template width/height
PILL_TOP_OFFSET_PERC    = -0.025  # place pill slightly overlapping photo bottom
PILL_MIN_WIDTH_PERC     = 0.12
PILL_H_PERC             = 0.045
PILL_H_PAD_PERC         = 0.020   # extra horiz padding around text inside pill
PILL_RADIUS_PERC        = 0.022

# Font size (starting) as fraction of template width (will auto-shrink to fit)
TITLE_FONT_START_PERC = 0.085
CTA_FONT_START_PERC   = 0.042
DATE_FONT_START_PERC  = 0.036
SRC_FONT_START_PERC   = 0.038
PILL_FONT_START_PERC  = 0.036

# Minimum font size as fraction of W
TITLE_FONT_MIN_PERC = 0.035
CTA_FONT_MIN_PERC   = 0.026
DATE_FONT_MIN_PERC  = 0.024
SRC_FONT_MIN_PERC   = 0.026
PILL_FONT_MIN_PERC  = 0.024

DEVICE_SCALE_FACTOR = 1
DEBUG_KEEP_TEXT_PNGS = False


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
    padding_px: int = 12
    text_align: str = "center"
    text_shadow: str = "0 2px 6px rgba(0,0,0,.25)"


# =========================
# ======  CORE  ===========
# =========================

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".avif"}

_browser_singleton: Optional[Browser] = None
_playwright_ctx = None  # keep ref to stop() cleanly


def get_browser() -> Browser:
    """Launch a single Chromium instance; reuse across rows."""
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
    """Load image (URL or local) as BGR ndarray."""
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
    """Find largest black-ish rectangle as the photo box."""
    hsv = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 0], np.uint8)
    upper = np.array([179, 80, 60], np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        raise RuntimeError("No black rectangle found in template.")
    x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    return x + 2, y + 2, max(1, w - 4), max(1, h - 4)  # inset for clean edge


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
    """3-channel mask of non-black template areas (to keep on top)."""
    hsv = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2HSV)
    black_mask = cv2.inRange(hsv, np.array([0, 0, 0], np.uint8), np.array([179, 80, 60], np.uint8))
    nonblack = cv2.bitwise_not(black_mask)
    return cv2.merge([nonblack, nonblack, nonblack])


def composite_with_template(base_bgr: np.ndarray, template_bgr: np.ndarray, mask3: np.ndarray) -> np.ndarray:
    """Keep non-black template pixels overlaid on top of base."""
    return (base_bgr & (~mask3)) + (template_bgr & mask3)


def safe_slug(text: str, max_len: int = 60) -> str:
    s = re.sub(r"[^\w\-]+", "_", text, flags=re.UNICODE).strip("_")
    return s[:max_len] or "untitled"


def read_csv_rows(csv_path: Path) -> List[dict]:
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def first_nonempty(row: dict, keys: List[str], default: str = "") -> str:
    for k in keys:
        v = (row.get(k) or "").strip()
        if v:
            return v
    return default


# ---------- Playwright text rendering (with in-memory copy) ----------

def _render_text_png_to_file(text: str, max_width: int, style: TextStyle, out_path: Path):
    """Render a single text block to a transparent PNG via Chromium."""
    font_url = style.font_path.resolve().as_uri()
    text_html = html.escape(text)
    html_doc = f"""<!doctype html>
<html><head><meta charset="utf-8">
<style>
@font-face {{
  font-family: 'BanglaFont';
  src: url('{font_url}') format('truetype');
  font-weight: normal; font-style: normal;
}}
html,body {{
  margin:0; padding:0; background:rgba(0,0,0,0);
}}
.wrap {{
  width:{max_width - 2*style.padding_px}px;
  padding:{style.padding_px}px;
  font-family:'BanglaFont','Noto Sans Bengali','Hind Siliguri',sans-serif;
  font-size:{style.font_px_start}px;
  line-height:{style.line_height};
  color:{style.color};
  text-align:{style.text_align};
  white-space:normal;
  word-break:break-word;
  text-shadow:{style.text_shadow};
}}
</style></head>
<body><div class="wrap">{text_html}</div></body></html>"""

    browser = get_browser()
    page = browser.new_page(
        viewport={"width": max_width, "height": 1600},
        device_scale_factor=DEVICE_SCALE_FACTOR,
    )
    page.set_content(html_doc)
    page.wait_for_load_state("networkidle")
    page.locator(".wrap").screenshot(path=str(out_path), omit_background=True)
    page.close()


def render_text_fit_box(text: str, box_w: int, box_h: int, style: TextStyle, tmp_name: str) -> Image.Image:
    """
    Render text and auto-shrink font until it fits within (box_w, box_h).
    Returns an RGBA Image **in memory** (safe to composite).
    """
    font_px = style.font_px_start
    tmp_path = Path(f"_{tmp_name}.png")

    while True:
        # clone style with updated font size
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
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            return in_mem

        font_px = max(style.font_px_min, font_px - 6)


def paste_center(base_rgba: Image.Image, overlay_rgba: Image.Image, box_px: Tuple[int, int, int, int]):
    l, t, r, b = box_px
    bw, bh = max(1, r - l), max(1, b - t)
    x = l + (bw - overlay_rgba.width) // 2
    y = t + (bh - overlay_rgba.height) // 2
    base_rgba.alpha_composite(overlay_rgba, (x, y))


# ---------- Category pill (HTML to PNG) ----------

def render_category_pill(text: str, W: int, H: int, tmp_name: str = "pill") -> Image.Image:
    """Render a red rounded pill with white text, sizing from W/H ratios."""
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

    # Render the label once to measure width, then wrap in a rounded rect via CSS.
    tmp_text = Path(f"_{tmp_name}_text.png")
    _render_text_png_to_file(text, max_width=W, style=style, out_path=tmp_text)
    with Image.open(tmp_text).convert("RGBA") as on_disk:
        text_img = on_disk.copy()
    try:
        os.remove(tmp_text)
    except OSError:
        pass

    pill_w = max(min_w, text_img.width + 2*hpad)

    # Build HTML for pill (rounded red background)
    font_url = style.font_path.resolve().as_uri()
    html_doc = f"""<!doctype html>
<html><head><meta charset="utf-8">
<style>
@font-face {{
  font-family: 'BanglaFont';
  src: url('{font_url}') format('truetype');
}}
html,body {{
  margin:0; padding:0; background:rgba(0,0,0,0);
}}
.pill {{
  display:inline-flex;
  align-items:center; justify-content:center;
  width:{pill_w}px; height:{pill_h}px;
  border-radius:{radius}px;
  background:{RED};
  color:#fff;
  font-family:'BanglaFont','Noto Sans Bengali','Hind Siliguri',sans-serif;
  font-size:{style.font_px_start}px;
  line-height:1;
  padding:0;
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
    try:
        os.remove(tmp_pill)
    except OSError:
        pass
    return pill_img


# =========================
# ========= MAIN ==========
# =========================

def main():
    # sanity
    if not TEMPLATE_PATH.is_file():
        raise FileNotFoundError(f"Template not found: {TEMPLATE_PATH}")
    font_path = Path("./fonts/NotoSansBengali_Condensed-Bold.ttf")
    if not font_path.is_file():
        raise FileNotFoundError(f"Bangla TTF not found: {font_path}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load template
    template_bgr = load_cv(str(TEMPLATE_PATH))
    H, W = template_bgr.shape[:2]

    # Compute pixel boxes from ratios
    title_box_px = TITLE_BOX_PERC.to_px(W, H)
    cta_box_px   = CTA_BOX_PERC.to_px(W, H)
    date_box_px  = DATE_BOX_PERC.to_px(W, H)
    src_box_px   = SRC_BOX_PERC.to_px(W, H)

    # Font sizes scaled from W (auto-shrink will handle overflow)
    title_style = TextStyle(font_path, RED,  int(TITLE_FONT_START_PERC*W), int(TITLE_FONT_MIN_PERC*W), 1.18, 12, "center", "0 2px 6px rgba(0,0,0,.25)")
    cta_style   = TextStyle(font_path, RED,  int(CTA_FONT_START_PERC*W),   int(CTA_FONT_MIN_PERC*W),   1.18,  6, "center", "0 1px 3px rgba(0,0,0,.15)")
    date_style  = TextStyle(font_path, MID,  int(DATE_FONT_START_PERC*W),  int(DATE_FONT_MIN_PERC*W),  1.12,  4, "center", "0 1px 2px rgba(0,0,0,.10)")
    src_style   = TextStyle(font_path, DARK, int(SRC_FONT_START_PERC*W),   int(SRC_FONT_MIN_PERC*W),   1.12,  4, "center", "0 1px 2px rgba(0,0,0,.10)")

    # Precompute overlay mask & photo box
    overlay_src = template_bgr.copy()
    mask3 = build_overlay_mask(template_bgr)
    if DETECT_PHOTO_BOX:
        px, py, pw, ph = detect_black_box(template_bgr)
    else:
        lx, ty, rx, by = PHOTO_BOX_PIXELS
        px, py, pw, ph = lx, ty, rx - lx, by - ty

    rows = read_csv_rows(CSV_PATH)
    total = len(rows)

    for i, row in enumerate(rows, start=1):
        img_url = first_nonempty(row, CSV_KEYS["image"])
        if not img_url:
            print(f"[{i}/{total}] skipped (no image url)")
            continue

        title_text   = first_nonempty(row, CSV_KEYS["title"])
        date_text    = first_nonempty(row, CSV_KEYS["date"])
        source_text  = first_nonempty(row, CSV_KEYS["source"])
        category_txt = first_nonempty(row, CSV_KEYS["category"])
        cta_text     = first_nonempty(row, CSV_KEYS["cta"], default="বিস্তারিত কমেন্ট")

        # output filename
        if NAMING_MODE == "title" and title_text:
            base_name = safe_slug(title_text)
        else:
            base_name = str(i)
        out_path = OUT_DIR / f"{base_name}_final.png"

        try:
            # 1) place article photo
            photo_bgr = load_cv(img_url)
            fitted = fit_cover(photo_bgr, pw, ph)
            base = template_bgr.copy()
            base[py:py+ph, px:px+pw] = fitted

            # 2) keep non-black template overlaid
            composed_bgr = composite_with_template(base, overlay_src, mask3)

            # 3) RGBA pipeline for text overlays
            card_rgba = Image.fromarray(cv2.cvtColor(composed_bgr, cv2.COLOR_BGR2RGBA))

            # 3a) Category pill overlapping photo bottom center
            if category_txt:
                pill_img = render_category_pill(category_txt, W, H, tmp_name="pill")
                pill_x = W//2 - pill_img.width//2
                pill_y = int(py + ph + PILL_TOP_OFFSET_PERC*H)  # overlap a bit
                card_rgba.alpha_composite(pill_img, (pill_x, pill_y))

            # 3b) Title (big red)
            if title_text:
                tw, th = (title_box_px[2]-title_box_px[0], title_box_px[3]-title_box_px[1])
                title_png = render_text_fit_box(title_text, tw, th, title_style, "title_tmp")
                paste_center(card_rgba, title_png, title_box_px)

            # 3c) CTA (red)
            if cta_text:
                cw, ch = (cta_box_px[2]-cta_box_px[0], cta_box_px[3]-cta_box_px[1])
                cta_png = render_text_fit_box(cta_text, cw, ch, cta_style, "cta_tmp")
                paste_center(card_rgba, cta_png, cta_box_px)

            # 3d) Date (mid gray)
            if date_text:
                dw, dh = (date_box_px[2]-date_box_px[0], date_box_px[3]-date_box_px[1])
                date_png = render_text_fit_box(date_text, dw, dh, date_style, "date_tmp")
                paste_center(card_rgba, date_png, date_box_px)

            # 3e) Red divider line
            line_len = int(DIVIDER_WIDTH_PERC * W)
            line_y   = int(DIVIDER_Y_PERC * H)
            thick    = max(2, int(DIVIDER_THICK_PERC * H))
            x1 = W//2 - line_len//2
            x2 = W//2 + line_len//2
            draw = ImageDraw.Draw(card_rgba)
            draw.rectangle([x1, line_y, x2, line_y + thick], fill=RED)

            # 3f) Source (dark)
            if source_text:
                sw, sh = (src_box_px[2]-src_box_px[0], src_box_px[3]-src_box_px[1])
                src_png = render_text_fit_box(source_text, sw, sh, src_style, "src_tmp")
                paste_center(card_rgba, src_png, src_box_px)

            # 4) save
            final_bgr = cv2.cvtColor(np.array(card_rgba), cv2.COLOR_RGBA2BGR)
            cv2.imwrite(str(out_path), final_bgr)
            print(f"[{i}/{total}] OK  -> {out_path.name} | title: {title_text[:50]!r}")

        except Exception as e:
            print(f"[{i}/{total}] FAIL ({img_url}): {e}")

    print(f"✅ Done. Results saved in {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
