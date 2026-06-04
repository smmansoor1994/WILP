"""
src/utils/visualize.py
=======================
Visualization utilities for YOLOrtho dental X-ray detections.

Draws:
  - Bounding boxes colored by quadrant (Q1=blue, Q2=green, Q3=red, Q4=yellow)
  - FDI tooth number labels
  - Disease attribute icons/text
  - Dental chart overlay showing detected teeth on a standard dental diagram
"""

import logging
from typing import List, Optional, Tuple, Dict

import cv2
import numpy as np

from src.utils.fdi import fdi_to_quadrant_enum

logger = logging.getLogger(__name__)

# ─── Color Scheme (BGR for OpenCV) ────────────────────────────────────────────
# Quadrant colors
QUADRANT_COLORS = {
    1: (255, 100, 50),    # Q1 Upper-Right — blue-ish
    2: (50, 200, 50),     # Q2 Upper-Left  — green
    3: (50, 50, 255),     # Q3 Lower-Left  — red
    4: (50, 200, 255),    # Q4 Lower-Right — yellow
}

# Disease colors
DISEASE_COLORS = {
    "Impacted":          (0, 165, 255),    # Orange
    "Caries":            (0, 255, 255),    # Yellow
    "Deep Caries":       (0, 0, 255),      # Red
    "Periapical Lesion": (147, 20, 255),   # Magenta
}

HEALTHY_COLOR = (0, 220, 0)      # Green — healthy tooth
DISEASE_COLOR = (0, 0, 220)      # Red — diseased tooth


def draw_teeth_detections(
    image: np.ndarray,
    teeth,                        # List[ToothDetection]
    show_fdi: bool = True,
    show_diseases: bool = True,
    show_conf: bool = True,
    line_thickness: int = 2,
) -> np.ndarray:
    """Draw all tooth detections on the image.

    Args:
        image:           BGR image (H, W, 3) — the panoramic X-ray.
        teeth:           List of ToothDetection results.
        show_fdi:        Draw FDI tooth number label.
        show_diseases:   Highlight diseased teeth and list disease names.
        show_conf:       Show detection confidence score.
        line_thickness:  Bounding box line thickness.

    Returns:
        Annotated BGR image.
    """
    canvas = image.copy()
    h, w = canvas.shape[:2]

    for tooth in teeth:
        # ── Determine bounding box coordinates ───────────────────────────────
        x1, y1, x2, y2 = [int(v) for v in tooth.bbox_xyxy]
        x1, x2 = np.clip([x1, x2], 0, w - 1)
        y1, y2 = np.clip([y1, y2], 0, h - 1)

        # ── Choose color based on quadrant and disease status ─────────────────
        quadrant, _ = fdi_to_quadrant_enum(tooth.fdi)
        base_color = QUADRANT_COLORS.get(quadrant, (200, 200, 200))

        if tooth.diseases and show_diseases:
            # Diseased: draw with disease-specific color (use first disease)
            box_color = DISEASE_COLOR
        else:
            box_color = base_color

        # ── Draw bounding box ─────────────────────────────────────────────────
        cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, line_thickness)

        # ── Build label text ──────────────────────────────────────────────────
        label_parts = []
        if show_fdi:
            label_parts.append(str(tooth.fdi))
        if show_conf:
            label_parts.append(f"{tooth.conf:.2f}")
        if show_diseases and tooth.diseases:
            label_parts.append("|".join(d[:3] for d in tooth.diseases))  # abbreviated

        label = " ".join(label_parts)

        # ── Draw label background ─────────────────────────────────────────────
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.35, min(0.55, w / 2400))  # scale to image width
        thickness = 1

        (lw, lh), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        label_y = max(y1 - 4, lh + 4)

        cv2.rectangle(
            canvas,
            (x1, label_y - lh - baseline),
            (x1 + lw, label_y),
            box_color,
            -1,
        )
        # White text for readability
        cv2.putText(
            canvas, label, (x1, label_y - baseline),
            font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA
        )

    # ── Add legend ────────────────────────────────────────────────────────────
    canvas = _draw_legend(canvas)

    return canvas


def _draw_legend(canvas: np.ndarray) -> np.ndarray:
    """Draw a small legend on the bottom-left of the image."""
    h, w = canvas.shape[:2]
    legend_items = [
        ("Q1 Upper-Right", QUADRANT_COLORS[1]),
        ("Q2 Upper-Left",  QUADRANT_COLORS[2]),
        ("Q3 Lower-Left",  QUADRANT_COLORS[3]),
        ("Q4 Lower-Right", QUADRANT_COLORS[4]),
        ("Diseased",       DISEASE_COLOR),
    ]

    x0, y0 = 10, h - 10 - len(legend_items) * 20
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = max(0.3, min(0.45, w / 2400))

    for i, (text, color) in enumerate(legend_items):
        y = y0 + i * 20
        cv2.rectangle(canvas, (x0, y - 12), (x0 + 15, y), color, -1)
        cv2.putText(canvas, text, (x0 + 20, y - 2), font, fs, (255, 255, 255), 1)

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
