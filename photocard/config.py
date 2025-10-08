from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

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
    TITLE_BOX_PERC: Tuple[float, float, float, float] = (0.04, 0.62, 0.940, 0.80)
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
    TITLE_FONT_START_PERC: float = 0.042
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
