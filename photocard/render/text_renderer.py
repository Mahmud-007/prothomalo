from __future__ import annotations
import html
import os
from dataclasses import dataclass
from pathlib import Path
from PIL import Image
from photocard.config import Config
from photocard.services.browser_manager import BrowserManager

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
    """Renders text and category pill to PNG using Playwright (transparent bg)."""
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
        font_px = style.font_px_start
        tmp_path = Path(f"_{tmp_name}.png")
        while True:
            html_doc = self._text_html_doc(text, style, max_width=box_w)
            self._render_html_to_png(html_doc, width=box_w, height=1600, out_path=tmp_path)
            with Image.open(tmp_path).convert("RGBA") as on_disk:
                img = on_disk.copy()

            if img.height <= box_h or font_px <= style.font_px_min:
                if not self.cfg.DEBUG_KEEP_TEXT_PNGS:
                    try: os.remove(tmp_path)
                    except OSError: pass
                return img

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
@font-face {{ font-family:'HindSiliguriBold'; src:url('{font_url}') format('truetype'); }}
html,body {{ margin:0; padding:0; background:rgba(0,0,0,0); }}
.pill {{
  display:flex; align-items:center; justify-content:center;
  width:{pill_w}px; height:{pill_h}px; border-radius:{radius}px;
  background:{self.cfg.RED}; color:#fff; font-family:'HindSiliguriBold';
  font-size:{text_style.font_px_start}px; line-height:1;
}}
</style></head>
<body><div class="pill">{html.escape(text)}</div></body></html>"""
        tmp_pill = Path(f"_{tmp_name}.png")
        self._render_html_to_png(html_doc, pill_w, pill_h, tmp_pill)
        with Image.open(tmp_pill).convert("RGBA") as on_disk2:
            pill_img = on_disk2.copy()
        try: os.remove(tmp_pill)
        except OSError: pass
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
  font-weight: 600;
  word-break:break-word;
}}
</style></head>
<body><div class="wrap">{text_html}</div></body></html>"""
