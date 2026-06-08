"""
src/utils/visualize.py
=======================
Visualization utilities for YOLOrtho dental X-ray detections.

Draws:
  - Bounding boxes colored by quadrant
  - Labels in the form  "Q: {quad} N: {pos} D: {disease}"
  - Filled label background matching the box color (white text)
"""

import logging
from typing import List, Optional, Tuple, Dict

import cv2
import numpy as np

from src.utils.fdi import fdi_to_quadrant_enum

logger = logging.getLogger(__name__)

# ─── Color Scheme (BGR for OpenCV) ────────────────────────────────────────────
# One distinct color per quadrant, matching the reference visualization style.
# Q1 Upper-Right → cyan-blue, Q2 Upper-Left → navy/purple,
# Q3 Lower-Left  → red,       Q4 Lower-Right → orange
QUADRANT_COLORS = {
    1: (210, 160,  50),   # Q1 Upper-Right — steel blue (BGR)
    2: (180,  60,  60),   # Q2 Upper-Left  — slate purple/dark-blue (BGR)
    3: ( 40,  40, 200),   # Q3 Lower-Left  — red (BGR)
    4: ( 30, 140, 210),   # Q4 Lower-Right — orange (BGR)
}

# Disease name abbreviation used in the label
DISEASE_ABBREV = {
    "Impacted":          "Impacted",
    "Caries":            "Caries",
    "Deep Caries":       "Deep Caries",
    "Periapical Lesion": "Periapical Lesion",
}

HEALTHY_COLOR = (100, 180, 100)  # muted green — healthy tooth
DISEASE_COLOR = (40, 40, 200)    # red — diseased tooth (used in dental chart)


def draw_teeth_detections(
    image: np.ndarray,
    teeth,                        # List[ToothDetection]
    show_fdi: bool = True,
    show_diseases: bool = True,
    show_conf: bool = False,
    line_thickness: int = 2,
) -> np.ndarray:
    """Draw tooth detections with  "Q: x N: y D: disease"  labels.

    Args:
        image:           BGR image (H, W, 3) — the panoramic X-ray.
        teeth:           List of ToothDetection results.
        show_fdi:        Include tooth number in label (kept for API compat, always on).
        show_diseases:   Show disease name in label.
        show_conf:       Show confidence score (disabled by default to match reference).
        line_thickness:  Bounding box stroke width.

    Returns:
        Annotated BGR image copy.
    """
    canvas = image.copy()
    h, w = canvas.shape[:2]

    # Scale font / thickness with image width so it looks right at any resolution
    scale   = w / 2400          # reference width ≈ 2400 px
    font_sc = max(0.40, min(0.65, 0.50 * scale * 2.5))
    thick   = max(1, int(1 * scale * 2.5))
    box_t   = max(2, int(line_thickness * scale * 2.5))
    font    = cv2.FONT_HERSHEY_SIMPLEX

    # Collect label placements and shift overlapping ones downward
    label_tops: list = []   # (x1, x2, y) of placed labels so far

    for tooth in sorted(teeth, key=lambda t: t.bbox_xyxy[0]):  # left-to-right
        x1, y1, x2, y2 = [int(v) for v in tooth.bbox_xyxy]
        x1, x2 = int(np.clip(x1, 0, w - 1)), int(np.clip(x2, 0, w - 1))
        y1, y2 = int(np.clip(y1, 0, h - 1)), int(np.clip(y2, 0, h - 1))

        quadrant, position = fdi_to_quadrant_enum(tooth.fdi)
        box_color = QUADRANT_COLORS.get(quadrant, (180, 180, 180))

        # ── Bounding box ──────────────────────────────────────────────────────
        cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, box_t)

        # ── Label text  "Q: 3 N: 6 D: Caries" ───────────────────────────────
        disease_str = tooth.diseases[0] if tooth.diseases else "Healthy"
        disease_str = DISEASE_ABBREV.get(disease_str, disease_str)
        label = f"Q: {quadrant} N: {position} D: {disease_str}"

        (lw, lh), baseline = cv2.getTextSize(label, font, font_sc, thick)
        pad = 4

        # Default: just above box top
        ly = y1 - pad
        if ly - lh - pad < 0:
            ly = y1 + lh + pad

        # Nudge down if it overlaps a previously placed label at same x range
        for (px1, px2, py) in label_tops:
            # x overlap?
            if x1 < px2 and x2 > px1 and abs(ly - py) < lh + pad * 2:
                ly = py + lh + pad * 2

        label_tops.append((x1, x1 + lw + pad, ly))

        # Filled background rectangle
        cv2.rectangle(
            canvas,
            (x1 - 1, ly - lh - pad),
            (x1 + lw + pad, ly + baseline),
            box_color,
            -1,
        )
        # White text
        cv2.putText(
            canvas, label,
            (x1 + 2, ly - baseline),
            font, font_sc, (255, 255, 255), thick, cv2.LINE_AA,
        )

    return canvas


def _draw_legend(canvas: np.ndarray) -> np.ndarray:
    """No-op — legend removed to match clean reference style."""
    return canvas


def draw_dental_chart(
    teeth,                         # List[ToothDetection]
    chart_size: Tuple[int, int] = (600, 300),
) -> np.ndarray:
    """Draw a standard dental chart (odontogram) with detected teeth highlighted.

    The chart shows a top-down view of 32 tooth positions in standard layout:
      Upper arch (left to right): Q2(28→21) | Q1(11→18)
      Lower arch (left to right): Q3(38→31) | Q4(41→48)

    Args:
        teeth:      List of ToothDetection results.
        chart_size: (width, height) of the output chart image.

    Returns:
        BGR chart image.
    """
    cw, ch = chart_size
    chart = np.full((ch, cw, 3), 40, dtype=np.uint8)  # dark background

    # Map FDI numbers to grid positions
    # Upper arch: Q2 right-to-left then Q1 left-to-right = 16 teeth across top
    upper_fdi = [20 + (8 - p) for p in range(1, 9)] + [10 + p for p in range(1, 9)]
    # Lower arch: Q3 right-to-left then Q4 left-to-right
    lower_fdi = [30 + (8 - p) for p in range(1, 9)] + [40 + p for p in range(1, 9)]

    detected_fdi = {t.fdi: t for t in teeth}

    cell_w = cw // 16
    cell_h = ch // 4

    def draw_tooth_cell(x: int, y: int, fdi: int) -> None:
        tooth = detected_fdi.get(fdi)
        # Cell rectangle
        rect_x1, rect_y1 = x + 2, y + 2
        rect_x2, rect_y2 = x + cell_w - 2, y + cell_h - 2
        if tooth:
            if tooth.diseases:
                color = DISEASE_COLOR
            else:
                color = HEALTHY_COLOR
            cv2.rectangle(chart, (rect_x1, rect_y1), (rect_x2, rect_y2), color, -1)
        cv2.rectangle(chart, (rect_x1, rect_y1), (rect_x2, rect_y2), (200, 200, 200), 1)
        # FDI label
        label = str(fdi)
        cv2.putText(
            chart, label,
            (rect_x1 + 2, rect_y2 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1
        )

    # Draw upper teeth
    for i, fdi in enumerate(upper_fdi):
        draw_tooth_cell(i * cell_w, cell_h, fdi)

    # Draw lower teeth
    for i, fdi in enumerate(lower_fdi):
        draw_tooth_cell(i * cell_w, 2 * cell_h, fdi)

    # Labels
    cv2.putText(chart, "UPPER", (cw // 2 - 30, cell_h - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    cv2.putText(chart, "LOWER", (cw // 2 - 30, 2 * cell_h + cell_h + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    return chart


def save_detection_mosaic(
    image: np.ndarray,
    teeth,
    output_path: str,
) -> None:
    """Save a mosaic image with X-ray annotation and dental chart.

    Layout:
      [ Annotated X-ray (full width)     ]
      [ Dental chart (bottom, centered)  ]

    Args:
        image:       Original BGR panoramic X-ray.
        teeth:       List of ToothDetection results.
        output_path: Output file path.
    """
    # Annotated X-ray
    vis = draw_teeth_detections(image, teeth)

    # Dental chart
    chart = draw_dental_chart(teeth, chart_size=(vis.shape[1], vis.shape[0] // 4))

    # Stack vertically
    mosaic = np.vstack([vis, chart])
    cv2.imwrite(output_path, mosaic)
    logger.info("Saved detection mosaic to '%s'.", output_path)
