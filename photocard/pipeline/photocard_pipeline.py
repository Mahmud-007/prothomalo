from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

from photocard.config import Config
from photocard.services.csv_service import CSVService
from photocard.services.browser_manager import BrowserManager
from photocard.render.text_renderer import TextRenderer, TextStyle
from photocard.imaging.image_io import ImageIO
from photocard.imaging.image_fitter import ImageFitter
from photocard.imaging.overlay_composer import OverlayComposer
from photocard.imaging.photo_box_detector import PhotoBoxDetector
from photocard.layout.layout import Layout

class PhotocardPipeline:
    """Coordinates services to produce final photocard images."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.csv = CSVService(cfg.CSV_PATH, cfg.CSV_KEYS)
        self.browser_mgr = BrowserManager()
        self.renderer = TextRenderer(cfg, self.browser_mgr)

    def run(self):
        if not self.cfg.TEMPLATE_PATH.is_file():
            raise FileNotFoundError(f"Template not found: {self.cfg.TEMPLATE_PATH}")
        if not self.cfg.FONT_PATH.is_file():
            raise FileNotFoundError(f"Bangla TTF not found: {self.cfg.FONT_PATH}")

        self.cfg.OUT_DIR.mkdir(parents=True, exist_ok=True)

        template_bgr = ImageIO.load_cv(str(self.cfg.TEMPLATE_PATH))
        H, W = template_bgr.shape[:2]

        title_box_px = Layout.to_px(self.cfg.TITLE_BOX_PERC, W, H)
        date_box_px  = Layout.to_px(self.cfg.DATE_BOX_PERC,  W, H)
        src_box_px   = Layout.to_px(self.cfg.SRC_BOX_PERC,   W, H)

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

        overlay_src = template_bgr.copy()
        mask3 = OverlayComposer.build_overlay_mask(template_bgr)

        if self.cfg.DETECT_PHOTO_BOX:
            px, py, pw, ph = PhotoBoxDetector.detect_black_box(template_bgr)
        else:
            l, t, r, b = self.cfg.PHOTO_BOX_PIXELS
            px, py, pw, ph = l, t, r - l, b - t

        rows = self.csv.read_rows()
        total = len(rows)

        for i, row in enumerate(rows, start=1):
            fields = self.csv.extract_fields(row)
            img_url   = fields["image"]
            title_txt = fields["title"]
            date_txt  = fields["date"]
            src_txt   = fields["source"]
            cat_txt   = fields["category"]

            if not img_url:
                print(f"[{i}/{total}] skipped (no image url)")
                continue

            base_name = ImageIO.safe_slug(title_txt) if (self.cfg.NAMING_MODE == "title" and title_txt) else str(i)
            out_path = self.cfg.OUT_DIR / f"{base_name}_final.png"

            try:
                photo_bgr = ImageIO.load_cv(img_url)
                fitted = ImageFitter.fit_cover(photo_bgr, pw, ph)
                base = template_bgr.copy()
                base[py:py+ph, px:px+pw] = fitted

                composed_bgr = OverlayComposer.composite_with_template(base, overlay_src, mask3)
                card_rgba = Image.fromarray(cv2.cvtColor(composed_bgr, cv2.COLOR_BGR2RGBA))

                if cat_txt:
                    pill = self.renderer.render_category_pill(cat_txt, W, H, tmp_name="pill")
                    pill_x = W // 2 - pill.width // 2
                    pill_y = int(py + ph + (self.cfg.PILL_TOP_OFFSET_PERC + self.cfg.SHIFT_PILL_DY_PERC) * H)
                    card_rgba.alpha_composite(pill, (pill_x, pill_y))

                if title_txt:
                    tw, th = title_box_px[2]-title_box_px[0], title_box_px[3]-title_box_px[1]
                    title_png = self.renderer.render_text_fit_box(title_txt, tw, th, title_style, "title_tmp")
                    dy_px = int(self.cfg.SHIFT_TITLE_DY_PERC * H)
                    Layout.paste_center(card_rgba, title_png, title_box_px, dy_px=dy_px)

                if date_txt:
                    dw, dh = date_box_px[2]-date_box_px[0], date_box_px[3]-date_box_px[1]
                    date_png = self.renderer.render_text_fit_box(date_txt, dw, dh, date_style, "date_tmp")
                    dy_px = int(self.cfg.SHIFT_DATE_DY_PERC * H)
                    Layout.paste_center(card_rgba, date_png, date_box_px, dy_px=dy_px)

                if src_txt:
                    sw, sh = src_box_px[2]-src_box_px[0], src_box_px[3]-src_box_px[1]
                    src_png = self.renderer.render_text_fit_box(src_txt, sw, sh, src_style, "src_tmp")
                    dy_px = int(self.cfg.SHIFT_SOURCE_DY_PERC * H)
                    Layout.paste_center(card_rgba, src_png, src_box_px, dy_px=dy_px)

                final_bgr = cv2.cvtColor(np.array(card_rgba), cv2.COLOR_RGBA2BGR)
                cv2.imwrite(str(out_path), final_bgr)
                print(f"[{i}/{total}] OK -> {out_path.name} | title: {title_txt[:50]!r}")

            except Exception as e:
                print(f"[{i}/{total}] FAIL ({img_url}): {e}")

        print(f"✅ Done. Results saved in {self.cfg.OUT_DIR.resolve()}")
