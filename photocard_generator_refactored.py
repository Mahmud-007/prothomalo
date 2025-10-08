#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Photocard generator (SRP refactor)
- Behavior preserved from the original script.
- Each class has a single, clear responsibility.
"""

from __future__ import annotations

import io
import os
import re
import csv
import html
import atexit
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import cv2
import requests
import numpy as np
from PIL import Image

from playwright.sync_api import sync_playwright, Browser

# Optional AVIF support (no-op if unavailable)
try:
    import pillow_avif  # noqa: F401
except Exception:
    pass


# =========================
# ======= CONFIG ==========
# =========================

@dataclass(frozen=True)
class Config:
    # Paths
    TEMPLATE_PATH: Path = Path("./templates/version-1.png")
    CSV_PATH: Path = Path("./articles/prothomalo.csv")
    OUT_DIR: Path = Path("./photocards/prothomalo-photocard")
    FONT_PATH: Path = Path("./fonts/HindSiliguri-Bold.ttf")

    # CSV keys map
    CSV_KEYS: Dict[str, List[str]] = None
    NAMING_MODE: str = "index"  # "index" | "title"

    # Photo window
    DETECT_PHOTO_BOX: bool = True
    PHOTO_BOX_PIXELS: Tuple[int, int, int, int] = (120, 160, 1840, 1000)  # l, t, r, b

    # Colors
    RED: str = "#C4161C"
    DARK: str = "#222222"
    MID: str = "#4A4A4A"

    # Layout (percentages)
    TITLE_BOX_PERC: Tuple[float, float, float, float] = (0.060, 0.62, 0.940, 0.80)
    DATE_BOX_PERC: Tuple[float, float, float, float] = (0.40, 0.84, 0.60, 0.875)
    SRC_BOX_PERC: Tuple[float, float, float, float] = (0.36, 0.895, 0.64, 0.93)

    DIVIDER_Y_PERC: float = 0.885
    DIVIDER_WIDTH_PERC: float = 0.28
    DIVIDER_THICK_PERC: float = 0.004

    # Category pill
    PILL_TOP_OFFSET_PERC: float = -0.025
    PILL_MIN_WIDTH_PERC: float = 0.12
    PILL_H_PERC: float = 0.045
    PILL_H_PAD_PERC: float = 0.020
    PILL_RADIUS_PERC: float = 0.022

    # Fine shift (DY as fraction of H)
    SHIFT_TITLE_DY_PERC: float = -0.022
    SHIFT_DATE_DY_PERC: float = -0.03
    SHIFT_SOURCE_DY_PERC: float = -0.05
    SHIFT_PILL_DY_PERC: float = 0.000

    # Font sizing (start/min as fraction of W)
    TITLE_FONT_START_PERC: float = 0.055
    TITLE_FONT_MIN_PERC: float = 0.035
    DATE_FONT_START_PERC: float = 0.018
    DATE_FONT_MIN_PERC: float = 0.006
    SRC_FONT_START_PERC: float = 0.018
    SRC_FONT_MIN_PERC: float = 0.006
    PILL_FONT_START_PERC: float = 0.018
    PILL_FONT_MIN_PERC: float = 0.006

    DEVICE_SCALE_FACTOR: int = 1
    DEBUG_KEEP_TEXT_PNGS: bool = False

    def __post_init__(self):
        if self.CSV_KEYS is None:
            object.__setattr__(self, "CSV_KEYS", {
                "image": ["article_image"],
                "title": ["article_title"],
                "date": ["published_date_bn"],
                "source": ["source"],
                "category": ["category_bn"],
                "cta": [],
            })


# =========================
# ======= SERVICES ========
# =========================

class CSVService:
    """Only reads CSV rows and returns dicts; knows how to pick values by key list."""

    def __init__(self, csv_path: Path, keys_map: Dict[str, List[str]]):
        self.csv_path = csv_path
        self.keys_map = keys_map

    def read_rows(self) -> List[dict]:
        with open(self.csv_path, "r", encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))

    @staticmethod
    def _first_non_empty(row: dict, keys: List[str]) -> str:
        for k in keys:
            v = (row.get(k) or "").strip()
            if v:
                return v
        return ""

    def extract_fields(self, row: dict) -> Dict[str, str]:
        return {
            "image": self._first_non_empty(row, self.keys_map["image"]),
            "title": self._first_non_empty(row, self.keys_map["title"]),
            "date": self._first_non_empty(row, self.keys_map["date"]),
            "source": self._first_non_empty(row, self.keys_map["source"]),
            "category": self._first_non_empty(row, self.keys_map["category"]),
        }


class BrowserManager:
    """Manages a single Chromium browser instance (lifecycle)."""

    _browser: Optional[Browser] = None
    _ctx = None

    def get(self) -> Browser:
        if self._browser is None:
            self._ctx = sync_playwright().start()
            self._browser = self._ctx.chromium.launch()
            atexit.register(self._shutdown)
        return self._browser

    def _shutdown(self):
        try:
            if self._browser:
                self._browser.close()
        except Exception:
            pass
        try:
            if self._ctx:
                self._ctx.stop()
        except Exception:
            pass
        self._browser = None
        self._ctx = None


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


class TextRenderer:
    """Renders text (and pill) as transparent PNG using Playwright."""

    def __init__(self, cfg: Config, browser_mgr: BrowserManager):
        self.cfg = cfg
        self.browser_mgr = browser_mgr

    def _render_html_to_png(self, html_doc: str, width: int, height: int, out_path: Path):
        browser = self.browser_mgr.get()
        page = browser.new_page(
            viewport={"width": width, "height": height},
            device_scale_factor=self.cfg.DEVICE_SCALE_FACTOR,
        )
        page.set_content(html_doc)
        page.wait_for_load_state("networkidle")
        page.locator("body > *").screenshot(path=str(out_path), omit_background=True)
        page.close()

    def render_text_fit_box(self, text: str, box_w: int, box_h: int, style: TextStyle, tmp_name: str) -> Image.Image:
        """Binary-search-like iterative shrink (simple decrement step to preserve original behavior)."""
        font_px = style.font_px_start
        tmp_path = Path(f"_{tmp_name}.png")
        while True:
            html_doc = self._text_html_doc(text, style, max_width=box_w)
            self._render_html_to_png(html_doc, width=box_w, height=1600, out_path=tmp_path)
            with Image.open(tmp_path).convert("RGBA") as on_disk:
                img = on_disk.copy()

            if img.height <= box_h or font_px <= style.font_px_min:
                if not self.cfg.DEBUG_KEEP_TEXT_PNGS:
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass
                return img

            # reduce and try again
            font_px = max(style.font_px_min, font_px - 6)
            style = TextStyle(
                font_path=style.font_path,
                color=style.color,
                font_px_start=font_px,
                font_px_min=style.font_px_min,
                line_height=style.line_height,
                padding_px=style.padding_px,
                text_align=style.text_align,
                text_shadow=style.text_shadow,
            )

    def render_category_pill(self, text: str, W: int, H: int, tmp_name: str = "pill") -> Image.Image:
        pill_h = int(self.cfg.PILL_H_PERC * H)
        font_px_start = int(self.cfg.PILL_FONT_START_PERC * W)
        min_w = int(self.cfg.PILL_MIN_WIDTH_PERC * W)
        hpad = int(self.cfg.PILL_H_PAD_PERC * W)
        radius = max(6, int(self.cfg.PILL_RADIUS_PERC * W))

        font_url = self.cfg.FONT_PATH.resolve().as_uri()
        # first render text width
        text_style = TextStyle(
            font_path=self.cfg.FONT_PATH,
            color=self.cfg.RED,
            font_px_start=font_px_start,
            font_px_min=int(self.cfg.PILL_FONT_MIN_PERC * W),
            line_height=1.0,
            padding_px=0,
            text_align="center",
            text_shadow="0 1px 2px rgba(0,0,0,.25)",
        )
        text_png = self.render_text_fit_box(text, W, pill_h, text_style, f"{tmp_name}_text")
        pill_w = max(min_w, text_png.width + 2 * hpad)

        html_doc = f"""<!doctype html>
<html><head><meta charset="utf-8">
<style>
@font-face {{ font-family:'BanglaFont'; src:url('{font_url}') format('truetype'); }}
html,body {{ margin:0; padding:0; background:rgba(0,0,0,0); }}
.pill {{
  display:flex; align-items:center; justify-content:center;
  width:{pill_w}px; height:{pill_h}px; border-radius:{radius}px;
  background:{self.cfg.RED}; color:#fff; font-family:'BanglaFont','Noto Sans Bengali','Hind Siliguri',sans-serif;
  font-size:{text_style.font_px_start}px; line-height:1;
}}
</style></head>
<body><div class="pill">{html.escape(text)}</div></body></html>"""
        tmp_pill = Path(f"_{tmp_name}.png")
        self._render_html_to_png(html_doc, pill_w, pill_h, tmp_pill)
        with Image.open(tmp_pill).convert("RGBA") as on_disk2:
            pill_img = on_disk2.copy()
        try:
            os.remove(tmp_pill)
        except OSError:
            pass
        return pill_img

    @staticmethod
    def _text_html_doc(text: str, style: TextStyle, max_width: int) -> str:
        font_url = style.font_path.resolve().as_uri()
        text_html = html.escape(text)
        return f"""<!doctype html>
<html><head><meta charset="utf-8">
<style>
@font-face {{
  font-family: 'HindSiliguriBold';
  src: url('{font_url}') format('truetype');
}}
html,body {{ margin:0; padding:0; background:rgba(0,0,0,0); }}
.wrap {{
  width:{max_width - 2*style.padding_px}px;
  padding:{style.padding_px}px;
  font-size:{style.font_px_start}px;
  line-height:{style.line_height};
  color:{style.color};
  text-align:{style.text_align};
  white-space:normal;
  font-style: normal;
  font-weight: 500;
  word-break:break-word;
}}
</style></head>
<body><div class="wrap">{text_html}</div></body></html>"""


class ImageIO:
    """Loading remote/local images and simple filename utils."""

    IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".avif"}

    @staticmethod
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
        # PIL fallback
        pil = Image.open(io.BytesIO(data)).convert("RGB")
        rgb = np.array(pil)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    @staticmethod
    def safe_slug(text: str, max_len: int = 60) -> str:
        s = re.sub(r"[^\w\-]+", "_", text, flags=re.UNICODE).strip("_")
        return s[:max_len] or "untitled"


class PhotoBoxDetector:
    """Detects black photo box or returns configured fixed box."""

    @staticmethod
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


class ImageFitter:
    """Aspect-fill (cover) resizing and crop."""

    @staticmethod
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


class OverlayComposer:
    """Builds overlay mask from template and composites with base."""

    @staticmethod
    def build_overlay_mask(template_bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2HSV)
        black_mask = cv2.inRange(
            hsv, np.array([0, 0, 0], np.uint8), np.array([179, 80, 60], np.uint8)
        )
        nonblack = cv2.bitwise_not(black_mask)
        return cv2.merge([nonblack, nonblack, nonblack])

    @staticmethod
    def composite_with_template(base_bgr: np.ndarray, template_bgr: np.ndarray, mask3: np.ndarray) -> np.ndarray:
        return (base_bgr & (~mask3)) + (template_bgr & mask3)


@dataclass
class BoxPerc:
    l: float
    t: float
    r: float
    b: float

    def to_px(self, W: int, H: int) -> Tuple[int, int, int, int]:
        return (int(self.l * W), int(self.t * H), int(self.r * W), int(self.b * H))


class Layout:
    """Converts relative boxes to pixels and places overlays."""

    @staticmethod
    def to_px(box: Tuple[float, float, float, float], W: int, H: int) -> Tuple[int, int, int, int]:
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


# =========================
# ======= PIPELINE ========
# =========================

class PhotocardPipeline:
    """Coordinates services to produce final photocard images."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.csv = CSVService(cfg.CSV_PATH, cfg.CSV_KEYS)
        self.browser_mgr = BrowserManager()
        self.renderer = TextRenderer(cfg, self.browser_mgr)

    def run(self):
        # Validate inputs
        if not self.cfg.TEMPLATE_PATH.is_file():
            raise FileNotFoundError(f"Template not found: {self.cfg.TEMPLATE_PATH}")
        if not self.cfg.FONT_PATH.is_file():
            raise FileNotFoundError(f"Bangla TTF not found: {self.cfg.FONT_PATH}")

        self.cfg.OUT_DIR.mkdir(parents=True, exist_ok=True)

        # Template
        template_bgr = ImageIO.load_cv(str(self.cfg.TEMPLATE_PATH))
        H, W = template_bgr.shape[:2]

        # Layout: compute once
        title_box_px = Layout.to_px(self.cfg.TITLE_BOX_PERC, W, H)
        date_box_px = Layout.to_px(self.cfg.DATE_BOX_PERC, W, H)
        src_box_px = Layout.to_px(self.cfg.SRC_BOX_PERC, W, H)

        # Styles
        title_style = TextStyle(
            self.cfg.FONT_PATH, self.cfg.RED,
            int(self.cfg.TITLE_FONT_START_PERC * W),
            int(self.cfg.TITLE_FONT_MIN_PERC * W),
            1.18, 4, "center", "0 2px 6px rgba(0,0,0,.25)"
        )
        date_style = TextStyle(
            self.cfg.FONT_PATH, self.cfg.MID,
            int(self.cfg.DATE_FONT_START_PERC * W),
            int(self.cfg.DATE_FONT_MIN_PERC * W),
            1.12, 4, "center", "0 1px 2px rgba(0,0,0,.10)"
        )
        src_style = TextStyle(
            self.cfg.FONT_PATH, self.cfg.DARK,
            int(self.cfg.SRC_FONT_START_PERC * W),
            int(self.cfg.SRC_FONT_MIN_PERC * W),
            1.12, 4, "center", "0 1px 2px rgba(0,0,0,.10)"
        )

        # Overlay
        overlay_src = template_bgr.copy()
        mask3 = OverlayComposer.build_overlay_mask(template_bgr)

        # Photo box
        if self.cfg.DETECT_PHOTO_BOX:
            px, py, pw, ph = PhotoBoxDetector.detect_black_box(template_bgr)
        else:
            l, t, r, b = self.cfg.PHOTO_BOX_PIXELS
            px, py, pw, ph = l, t, r - l, b - t

        # Data
        rows = self.csv.read_rows()
        total = len(rows)

        for i, row in enumerate(rows, start=1):
            fields = self.csv.extract_fields(row)
            img_url = fields["image"]
            title_text = fields["title"]
            date_text = fields["date"]
            src_text = fields["source"]
            cat_text = fields["category"]

            if not img_url:
                print(f"[{i}/{total}] skipped (no image url)")
                continue

            # output filename
            if self.cfg.NAMING_MODE == "title" and title_text:
                base_name = ImageIO.safe_slug(title_text)
            else:
                base_name = str(i)
            out_path = self.cfg.OUT_DIR / f"{base_name}_final.png"

            try:
                # 1) Fit photo into window
                photo_bgr = ImageIO.load_cv(img_url)
                fitted = ImageFitter.fit_cover(photo_bgr, pw, ph)
                base = template_bgr.copy()
                base[py:py + ph, px:px + pw] = fitted

                # 2) Apply template overlay (preserve non-black parts)
                composed_bgr = OverlayComposer.composite_with_template(base, overlay_src, mask3)

                # 3) Draw texts
                card_rgba = Image.fromarray(cv2.cvtColor(composed_bgr, cv2.COLOR_BGR2RGBA))

                # Category pill (bottom-center of photo)
                if cat_text:
                    pill = self.renderer.render_category_pill(cat_text, W, H, tmp_name="pill")
                    pill_x = W // 2 - pill.width // 2
                    pill_y = int(py + ph + (self.cfg.PILL_TOP_OFFSET_PERC + self.cfg.SHIFT_PILL_DY_PERC) * H)
                    card_rgba.alpha_composite(pill, (pill_x, pill_y))

                # Title
                if title_text:
                    tw, th = title_box_px[2] - title_box_px[0], title_box_px[3] - title_box_px[1]
                    title_png = self.renderer.render_text_fit_box(title_text, tw, th, title_style, "title_tmp")
                    dy_px = int(self.cfg.SHIFT_TITLE_DY_PERC * H)
                    Layout.paste_center(card_rgba, title_png, title_box_px, dy_px=dy_px)

                # Date
                if date_text:
                    dw, dh = date_box_px[2] - date_box_px[0], date_box_px[3] - date_box_px[1]
                    date_png = self.renderer.render_text_fit_box(date_text, dw, dh, date_style, "date_tmp")
                    dy_px = int(self.cfg.SHIFT_DATE_DY_PERC * H)
                    Layout.paste_center(card_rgba, date_png, date_box_px, dy_px=dy_px)

                # Source
                if src_text:
                    sw, sh = src_box_px[2] - src_box_px[0], src_box_px[3] - src_box_px[1]
                    src_png = self.renderer.render_text_fit_box(src_text, sw, sh, src_style, "src_tmp")
                    dy_px = int(self.cfg.SHIFT_SOURCE_DY_PERC * H)
                    Layout.paste_center(card_rgba, src_png, src_box_px, dy_px=dy_px)

                # 4) Save
                final_bgr = cv2.cvtColor(np.array(card_rgba), cv2.COLOR_RGBA2BGR)
                cv2.imwrite(str(out_path), final_bgr)
                print(f"[{i}/{total}] OK -> {out_path.name} | title: {title_text[:50]!r}")

            except Exception as e:
                print(f"[{i}/{total}] FAIL ({img_url}): {e}")

        print(f"✅ Done. Results saved in {self.cfg.OUT_DIR.resolve()}")


# =========================
# ========= MAIN ==========
# =========================

def main():
    cfg = Config()
    PhotocardPipeline(cfg).run()


if __name__ == "__main__":
    main()
