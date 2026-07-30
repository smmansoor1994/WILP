"""
ARCHON Thesis Presentation Generator
Creates a 25-slide PowerPoint presentation
"""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt
from pptx.enum.dml import MSO_THEME_COLOR
import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
RESULTS_BASE = Path(r"D:\WILP\Workingcode\models\archon-severityfix-100\outputs_20260615_211821\content\WILP\outputs\runs")
PHASE1_DIR   = RESULTS_BASE / "phase1"
PHASE2_DIR   = RESULTS_BASE / "phase2"
EVAL_DIR     = RESULTS_BASE / "eval"
OUT_PPT      = Path(r"D:\WILP\Workingcode\Baseline\WILP\ARCHON_Thesis_Presentation.pptx")
DEMO_INPUT_DIR = Path(r"D:\WILP\Workingcode\Baseline\WILP\data\processed\images\train")
DEMO_INFER_DIR = Path(r"D:\WILP\Workingcode\Baseline\WILP\demo_images_inferenced_model")
DEMO_COMBINED_DIR = Path(r"D:\WILP\Workingcode\Baseline\WILP\demo_images_combined")

# ── Colour Palette ──────────────────────────────────────────────────────────
DARK_BLUE   = RGBColor(0x0D, 0x2B, 0x55)   # slide background / title bars
MID_BLUE    = RGBColor(0x1B, 0x4F, 0x8A)
ACCENT_TEAL = RGBColor(0x00, 0xA8, 0xA8)
WHITE       = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_GRAY  = RGBColor(0xF2, 0xF4, 0xF8)
DARK_GRAY   = RGBColor(0x33, 0x33, 0x44)
ORANGE      = RGBColor(0xE8, 0x7C, 0x1B)
GREEN       = RGBColor(0x27, 0xAE, 0x60)
RED         = RGBColor(0xC0, 0x39, 0x2B)

# Slide dimensions (widescreen 16:9)
SLIDE_W = Inches(13.33)
SLIDE_H = Inches(7.5)


def add_bg(slide, color=DARK_BLUE):
    """Fill slide background with solid colour."""
    from pptx.util import Emu
    from pptx.oxml.ns import qn
    from lxml import etree
    background = slide.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_textbox(slide, text, left, top, width, height,
                font_size=18, bold=False, color=WHITE,
                align=PP_ALIGN.LEFT, wrap=True, italic=False):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = wrap
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    return txBox


def add_rect(slide, left, top, width, height, fill_color, line_color=None, line_width=None):
    from pptx.util import Pt as PtU
    shape = slide.shapes.add_shape(
        1,  # MSO_SHAPE_TYPE.RECTANGLE
        left, top, width, height
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    if line_color:
        shape.line.color.rgb = line_color
        if line_width:
            shape.line.width = PtU(line_width)
    else:
        shape.line.fill.background()
    return shape


def title_bar(slide, title_text, subtitle_text=None):
    """Adds a dark-blue title bar at top."""
    bar = add_rect(slide, Inches(0), Inches(0), SLIDE_W, Inches(1.15), DARK_BLUE)
    add_textbox(slide, title_text,
                Inches(0.35), Inches(0.12), Inches(12.5), Inches(0.65),
                font_size=28, bold=True, color=WHITE, align=PP_ALIGN.LEFT)
    if subtitle_text:
        add_textbox(slide, subtitle_text,
                    Inches(0.35), Inches(0.72), Inches(12.5), Inches(0.38),
                    font_size=14, bold=False, color=ACCENT_TEAL, align=PP_ALIGN.LEFT)


def add_bullet_box(slide, items, left, top, width, height,
                   font_size=15, bullet="•", color=WHITE, header=None, header_color=ACCENT_TEAL):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    first = True
    if header:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        run = p.add_run()
        run.text = header
        run.font.size = Pt(font_size + 1)
        run.font.bold = True
        run.font.color.rgb = header_color
    for item in items:
        p = tf.add_paragraph() if not first else tf.paragraphs[0]
        first = False
        run = p.add_run()
        run.text = f"{bullet} {item}"
        run.font.size = Pt(font_size)
        run.font.color.rgb = color
        p.space_before = Pt(3)
    return txBox


def safe_add_image(slide, img_path, left, top, width, height):
    if os.path.exists(img_path):
        slide.shapes.add_picture(str(img_path), left, top, width, height)
        return True
    else:
        add_rect(slide, left, top, width, height, DARK_GRAY)
        add_textbox(slide, f"[Image: {Path(img_path).name}]",
                    left + Inches(0.1), top + height // 2 - Inches(0.2),
                    width - Inches(0.2), Inches(0.4),
                    font_size=10, color=ACCENT_TEAL, align=PP_ALIGN.CENTER)
        return False


# ══════════════════════════════════════════════════════════════════════════════
# BUILD PRESENTATION
# ══════════════════════════════════════════════════════════════════════════════
prs = Presentation()
prs.slide_width  = SLIDE_W
prs.slide_height = SLIDE_H
blank_layout = prs.slide_layouts[6]  # completely blank


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 1  Title
# ─────────────────────────────────────────────────────────────────────────────
slide = prs.slides.add_slide(blank_layout)
add_bg(slide, DARK_BLUE)

# Decorative accent bar
add_rect(slide, Inches(0), Inches(0), Inches(0.25), SLIDE_H, ACCENT_TEAL)
add_rect(slide, Inches(0.25), Inches(0), Inches(0.08), SLIDE_H, MID_BLUE)

# Logo / badge area
add_rect(slide, Inches(10.8), Inches(0.3), Inches(2.1), Inches(0.6), ACCENT_TEAL)
add_textbox(slide, "BITS PILANI  •  WILP",
            Inches(10.85), Inches(0.35), Inches(2.0), Inches(0.5),
            font_size=9, bold=True, color=DARK_BLUE, align=PP_ALIGN.CENTER)

# Main title
add_textbox(slide, "ARCHON",
            Inches(0.7), Inches(1.2), Inches(12), Inches(1.0),
            font_size=52, bold=True, color=ACCENT_TEAL, align=PP_ALIGN.LEFT)

add_textbox(slide,
            "Arch-Aware Context Network for Tooth Enumeration\nand Severity-Aware Dental Disease Detection",
            Inches(0.7), Inches(2.1), Inches(11.5), Inches(1.2),
            font_size=22, bold=False, color=WHITE, align=PP_ALIGN.LEFT)

# Divider
add_rect(slide, Inches(0.7), Inches(3.35), Inches(8), Inches(0.04), ACCENT_TEAL)

add_textbox(slide,
            "M.Tech Dissertation  |  Department of CSIS  |  BITS Pilani – WILP\n"
            "Supervisor: [Supervisor Name]  |  June 2026",
            Inches(0.7), Inches(3.5), Inches(11), Inches(0.8),
            font_size=14, color=LIGHT_GRAY, align=PP_ALIGN.LEFT)

add_textbox(slide,
            "Dataset: DENTEX Challenge 2023  •  Baseline: YOLOrtho (arXiv:2308.05967)  •  GPU: NVIDIA T4",
            Inches(0.7), Inches(4.35), Inches(11), Inches(0.4),
            font_size=12, color=ACCENT_TEAL, align=PP_ALIGN.LEFT, italic=True)

# Bottom key points
for i, (kw, txt) in enumerate([
        ("IMPROVEMENTS", "5 Targeted Improvements"),
        ("DATASET",      "DENTEX 2023 Challenge"),
        ("OUTPUT",       "Severity-Aware JSON Reports"),
        ("TRAINING",     "3-Phase Curriculum"),
]):
    x = Inches(0.7 + i * 3.1)
    add_rect(slide, x, Inches(5.4), Inches(2.9), Inches(1.6), MID_BLUE)
    add_textbox(slide, kw, x + Inches(0.1), Inches(5.5), Inches(2.7), Inches(0.4),
                font_size=9, bold=True, color=ACCENT_TEAL)
    add_textbox(slide, txt, x + Inches(0.1), Inches(5.9), Inches(2.7), Inches(0.8),
                font_size=13, bold=False, color=WHITE)


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 2  Problem Statement
# ─────────────────────────────────────────────────────────────────────────────
slide = prs.slides.add_slide(blank_layout)
add_bg(slide, LIGHT_GRAY)
title_bar(slide, "Problem Statement", "Why existing dental AI systems fall short")

problems = [
    ("Restricted Receptive Field",
     "CNN P5 cells see only ~12% of panoramic width.\nLeads to midline swap errors (FDI 16 ↔ 26)."),
    ("Binary-Only Diagnostic Output",
     "All prior methods output only a Yes/No disease flag.\nNo severity → no triage support."),
    ("Exposure Sensitivity",
     "Posterior bone regions suppress subtle caries/lesion shadows.\nEarly-stage disease missed during training."),
]
colors = [RED, ORANGE, MID_BLUE]
for i, (title, body) in enumerate(problems):
    x = Inches(0.4 + i * 4.3)
    add_rect(slide, x, Inches(1.3), Inches(4.05), Inches(5.7), WHITE,
             line_color=colors[i], line_width=2)
    add_rect(slide, x, Inches(1.3), Inches(4.05), Inches(0.55), colors[i])
    add_textbox(slide, f"Limitation {i+1}", x + Inches(0.1), Inches(1.33),
                Inches(3.8), Inches(0.5), font_size=11, bold=True, color=WHITE)
    add_textbox(slide, title, x + Inches(0.15), Inches(1.95),
                Inches(3.75), Inches(0.55), font_size=16, bold=True, color=colors[i])
    add_textbox(slide, body, x + Inches(0.15), Inches(2.55),
                Inches(3.75), Inches(4.0), font_size=13, color=DARK_GRAY, wrap=True)

add_textbox(slide,
            "ARCHON addresses ALL THREE limitations in a single end-to-end multi-task framework.",
            Inches(0.4), Inches(7.05), Inches(12.5), Inches(0.38),
            font_size=13, bold=True, color=DARK_BLUE, align=PP_ALIGN.CENTER)


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 3  Clinical Motivation & Dataset
# ─────────────────────────────────────────────────────────────────────────────
slide = prs.slides.add_slide(blank_layout)
add_bg(slide, LIGHT_GRAY)
title_bar(slide, "Clinical Motivation & DENTEX Dataset",
          "MICCAI 2023 Benchmark  |  Panoramic Orthopantomogram (OPG) Analysis")

# Stats boxes
stats = [("3.5B", "People affected by\noral diseases (WHO)"),
         ("32", "FDI tooth classes\n(11-18, 21-28, 31-48)"),
         ("4", "Disease categories\nper tooth"),
         ("~1,200", "Training images\n(3-tier annotations)")]
for i, (num, label) in enumerate(stats):
    x = Inches(0.3 + i * 3.25)
    add_rect(slide, x, Inches(1.35), Inches(3.0), Inches(1.5), DARK_BLUE)
    add_textbox(slide, num, x + Inches(0.1), Inches(1.45), Inches(2.8), Inches(0.7),
                font_size=30, bold=True, color=ACCENT_TEAL, align=PP_ALIGN.CENTER)
    add_textbox(slide, label, x + Inches(0.1), Inches(2.1), Inches(2.8), Inches(0.65),
                font_size=11, color=WHITE, align=PP_ALIGN.CENTER)

# Dataset tiers table
add_textbox(slide, "DENTEX 2023 Annotation Hierarchy",
            Inches(0.3), Inches(3.1), Inches(8), Inches(0.4),
            font_size=14, bold=True, color=DARK_BLUE)

rows = [
    ("Part 1 (~300 imgs)", "Bounding box + Quadrant label (1–4)", "data_type=0"),
    ("Part 2 (~500 imgs)", "Bbox + Quadrant + Position → FDI", "data_type=1"),
    ("Part 3 (~400 imgs)", "Bbox + FDI + Disease flags (4 attrs)", "data_type=2"),
    ("Unlabelled (~300)", "Pseudo-labeled via Phase 1 model", "data_type=2 (pseudo)"),
]
headers = ["Split", "Annotation Content", "Flag"]
col_w = [Inches(2.3), Inches(5.5), Inches(2.0)]
col_x = [Inches(0.3), Inches(2.65), Inches(8.2)]
for j, (h, cw, cx) in enumerate(zip(headers, col_w, col_x)):
    add_rect(slide, cx, Inches(3.55), cw, Inches(0.38), DARK_BLUE)
    add_textbox(slide, h, cx + Inches(0.05), Inches(3.58), cw - Inches(0.1), Inches(0.32),
                font_size=11, bold=True, color=WHITE)
for i, row in enumerate(rows):
    bg = LIGHT_GRAY if i % 2 == 0 else WHITE
    for j, (cell, cw, cx) in enumerate(zip(row, col_w, col_x)):
        add_rect(slide, cx, Inches(3.93 + i * 0.42), cw, Inches(0.42),
                 bg, line_color=RGBColor(0xCC, 0xCC, 0xCC), line_width=0.5)
        add_textbox(slide, cell, cx + Inches(0.05),
                    Inches(3.95 + i * 0.42), cw - Inches(0.1), Inches(0.38),
                    font_size=10.5, color=DARK_GRAY)

# Disease categories
add_textbox(slide, "Disease Categories",
            Inches(10.4), Inches(3.1), Inches(2.7), Inches(0.4),
            font_size=14, bold=True, color=DARK_BLUE)
diseases = [("Impaction", "~15%", ORANGE),
            ("Caries",    "~25%", RED),
            ("Deep Caries","~8%", MID_BLUE),
            ("Periapical Lesion","~15%", ACCENT_TEAL)]
for i, (d, pct, col) in enumerate(diseases):
    add_rect(slide, Inches(10.4), Inches(3.55 + i * 0.8), Inches(2.7), Inches(0.72), col)
    add_textbox(slide, d, Inches(10.5), Inches(3.58 + i * 0.8),
                Inches(1.9), Inches(0.35), font_size=11, bold=True, color=WHITE)
    add_textbox(slide, pct, Inches(12.3), Inches(3.58 + i * 0.8),
                Inches(0.7), Inches(0.35), font_size=11, bold=True, color=WHITE, align=PP_ALIGN.RIGHT)


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 4  ARCHON Architecture Overview
# ─────────────────────────────────────────────────────────────────────────────
slide = prs.slides.add_slide(blank_layout)
add_bg(slide, LIGHT_GRAY)
title_bar(slide, "ARCHON Architecture Overview",
          "5 improvements over the YOLOrtho baseline — single end-to-end model")

arch_steps = [
    ("Input\n1280×640\nOPG", DARK_BLUE),
    ("YOLOv8x\nBackbone\n+CoordConv", MID_BLUE),
    ("GlobalContext\nEncoder\n(Swin)", ACCENT_TEAL),
    ("MultiScale\nFusion\n(CrossAttn)", ORANGE),
    ("Hybrid\nMultiTask\nHead", GREEN),
    ("Post-\nProcessing\n(LSA+QP)", MID_BLUE),
    ("Output\nJSON\nReport", DARK_BLUE),
]
box_w = Inches(1.65)
box_h = Inches(1.4)
start_x = Inches(0.2)
y = Inches(1.5)
for i, (label, color) in enumerate(arch_steps):
    x = start_x + i * (box_w + Inches(0.2))
    add_rect(slide, x, y, box_w, box_h, color)
    add_textbox(slide, label, x + Inches(0.05), y + Inches(0.1),
                box_w - Inches(0.1), box_h - Inches(0.2),
                font_size=11, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    if i < len(arch_steps) - 1:
        add_textbox(slide, "→", x + box_w, y + Inches(0.5),
                    Inches(0.2), Inches(0.5),
                    font_size=18, bold=True, color=DARK_GRAY, align=PP_ALIGN.CENTER)

# Feature maps row
add_textbox(slide, "Feature Maps: P3(320ch, stride8) · P4(640ch, stride16) · P5(640ch, stride32)",
            Inches(0.3), Inches(3.1), Inches(12.5), Inches(0.35),
            font_size=12, italic=True, color=DARK_GRAY)

# 5 improvements summary
imps = [
    ("A", "GlobalContextEncoder", "Swin Transformer on P5 — full dental arch context", ACCENT_TEAL),
    ("B", "MultiScaleFusion", "Cross-Attention (Q=CNN, K/V=Swin) at P3/P4/P5", ORANGE),
    ("C", "HybridMultiTaskHead", "3-level severity: Healthy / Mild / Severe × 4 diseases", GREEN),
    ("D", "CLAHE Augmentation", "Stochastic L-channel contrast enhancement (p=0.5)", MID_BLUE),
    ("E", "Quadrant-Consistency Penalty", "Hungarian cost matrix penalized by quadrant probs", RED),
]
add_textbox(slide, "ARCHON Improvements", Inches(0.3), Inches(3.6), Inches(12.5), Inches(0.38),
            font_size=14, bold=True, color=DARK_BLUE)
for i, (badge, name, desc, col) in enumerate(imps):
    x = Inches(0.3 + (i % 3) * 4.3)
    y_pos = Inches(4.1) if i < 3 else Inches(5.3)
    add_rect(slide, x, y_pos, Inches(0.42), Inches(0.9), col)
    add_textbox(slide, badge, x + Inches(0.05), y_pos + Inches(0.2),
                Inches(0.32), Inches(0.5), font_size=16, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    add_textbox(slide, name, x + Inches(0.48), y_pos, Inches(3.7), Inches(0.38),
                font_size=12, bold=True, color=col)
    add_textbox(slide, desc, x + Inches(0.48), y_pos + Inches(0.4), Inches(3.7), Inches(0.45),
                font_size=10.5, color=DARK_GRAY)


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 5  Backbone: YOLOv8x + CoordConv
# ─────────────────────────────────────────────────────────────────────────────
slide = prs.slides.add_slide(blank_layout)
add_bg(slide, LIGHT_GRAY)
title_bar(slide, "Backbone: YOLOv8x + CoordConv + Modified FPN",
          "Foundation inherited from YOLOrtho — preserved through all 3 training phases")

# Left column
add_textbox(slide, "Why YOLOv8x (largest variant)?",
            Inches(0.4), Inches(1.35), Inches(6), Inches(0.4),
            font_size=14, bold=True, color=DARK_BLUE)
for item in [
    "32-class FDI task needs maximum per-class discriminative capacity",
    "P5 channel depth = 640 (vs 256 in YOLOv8n) — critical for symmetric tooth separation",
    "Ultralytics framework: built-in COCO metrics, checkpoint management",
    "~68.5M parameters; VRAM ~12 GB at batch=4, 1280×640",
]:
    pass
add_bullet_box(slide,
    ["32-class FDI needs maximum discriminative capacity",
     "P5 depth=640ch — critical for symmetric tooth separation",
     "Ultralytics built-in COCO metrics & checkpoint management",
     "~68.5M params; VRAM ~12 GB at batch=4 / 1280×640"],
    Inches(0.4), Inches(1.8), Inches(5.8), Inches(2.2),
    font_size=13, color=DARK_GRAY, bullet="▸")

add_textbox(slide, "CoordConv — Breaking Translation Equivariance",
            Inches(0.4), Inches(4.05), Inches(6), Inches(0.4),
            font_size=14, bold=True, color=DARK_BLUE)
add_textbox(slide,
    "Standard CNN: same filter response regardless of absolute position.\n"
    "CoordConv: prepends normalised (x, y) ∈ [−1, +1] channels before conv.\n"
    "→ Allows network to encode where on the panoramic image a tooth sits.\n"
    "→ Essential for FDI numbering (tooth identity = position, not texture).",
    Inches(0.4), Inches(4.5), Inches(6.0), Inches(2.5),
    font_size=12.5, color=DARK_GRAY, wrap=True)

# Right column — FPN + specs
add_rect(slide, Inches(6.6), Inches(1.3), Inches(6.4), Inches(5.9), WHITE,
         line_color=MID_BLUE, line_width=1)
add_textbox(slide, "Modified FPN — Extra Upsampling Stage",
            Inches(6.75), Inches(1.4), Inches(6.1), Inches(0.4),
            font_size=14, bold=True, color=DARK_BLUE)
fpn_rows = [
    ("P3", "stride 8",  "(B, 320, 80, 160)",  "Fine-grained tooth texture"),
    ("P4", "stride 16", "(B, 640, 40, 80)",   "Mid-level semantic"),
    ("P5", "stride 32", "(B, 640, 20, 40)",   "Coarse, arch-level"),
]
for i, (name, stride, shape, role) in enumerate(fpn_rows):
    y_r = Inches(1.9 + i * 0.75)
    add_rect(slide, Inches(6.75), y_r, Inches(0.55), Inches(0.62), ACCENT_TEAL)
    add_textbox(slide, name, Inches(6.75), y_r + Inches(0.1),
                Inches(0.55), Inches(0.45), font_size=14, bold=True,
                color=WHITE, align=PP_ALIGN.CENTER)
    add_textbox(slide, f"{stride}  ·  {shape}",
                Inches(7.35), y_r + Inches(0.02), Inches(5.5), Inches(0.32),
                font_size=12, bold=True, color=DARK_BLUE)
    add_textbox(slide, role, Inches(7.35), y_r + Inches(0.33),
                Inches(5.5), Inches(0.28), font_size=11, color=DARK_GRAY)

add_textbox(slide,
    "Standard YOLOv8: strides [8,16,32].\n"
    "ARCHON: extra upsample → strides [4,8,16] for finer incisor detection.",
    Inches(6.75), Inches(4.2), Inches(6.0), Inches(0.85),
    font_size=11.5, color=MID_BLUE, italic=True)

add_textbox(slide, "Forward hooks at layers 15 (P3), 18 (P4), 21 (P5) extract FPN features\n"
            "without modifying YOLOv8 internals — preserves compatibility with Ultralytics.",
            Inches(6.75), Inches(5.1), Inches(6.0), Inches(1.0),
            font_size=11.5, color=DARK_GRAY)


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 6  GlobalContextEncoder (Swin Transformer)
# ─────────────────────────────────────────────────────────────────────────────
slide = prs.slides.add_slide(blank_layout)
add_bg(slide, LIGHT_GRAY)
title_bar(slide, "Improvement A: GlobalContextEncoder (Swin Transformer)",
          "Captures inter-tooth relationships across the full dental arch — applied to P5 only")

# Motivation box
add_rect(slide, Inches(0.3), Inches(1.3), Inches(5.8), Inches(1.75), DARK_BLUE)
add_textbox(slide, "Why P5 for Global Context?",
            Inches(0.4), Inches(1.38), Inches(5.5), Inches(0.4),
            font_size=13, bold=True, color=ACCENT_TEAL)
add_textbox(slide,
    "P5 (20×40 tokens) ≈ one token per tooth.\n"
    "CNN P5 receptive field covers ~12% of panoramic width.\n"
    "After Swin 2-block: 47% coverage → FDI disambiguation.",
    Inches(0.4), Inches(1.82), Inches(5.5), Inches(1.1),
    font_size=12, color=WHITE)

# Architecture details
add_textbox(slide, "Architecture: 2-Block Swin Transformer",
            Inches(0.3), Inches(3.2), Inches(6), Inches(0.4),
            font_size=14, bold=True, color=DARK_BLUE)
blocks = [
    ("Block 1", "Regular Window Attention (W-MSA)", "shift_size=0\n50 windows × 16 tokens\nO(16²)×50=12,800 ops"),
    ("Block 2", "Shifted Window Attention (SW-MSA)", "shift_size=2\nCross-window connectivity\nEvery cell attends globally"),
]
for i, (blk, name, detail) in enumerate(blocks):
    x = Inches(0.3 + i * 3.0)
    add_rect(slide, x, Inches(3.65), Inches(2.8), Inches(3.4), ACCENT_TEAL if i == 0 else ORANGE)
    add_textbox(slide, blk, x + Inches(0.1), Inches(3.72),
                Inches(2.6), Inches(0.38), font_size=12, bold=True, color=WHITE)
    add_textbox(slide, name, x + Inches(0.1), Inches(4.1),
                Inches(2.6), Inches(0.5), font_size=11, bold=False, color=WHITE, italic=True)
    add_textbox(slide, detail, x + Inches(0.1), Inches(4.65),
                Inches(2.6), Inches(2.2), font_size=11, color=WHITE)

# Complexity comparison
add_rect(slide, Inches(6.5), Inches(1.3), Inches(6.5), Inches(5.75), WHITE,
         line_color=ACCENT_TEAL, line_width=1.5)
add_textbox(slide, "Computational Complexity Comparison",
            Inches(6.6), Inches(1.4), Inches(6.2), Inches(0.4),
            font_size=14, bold=True, color=DARK_BLUE)

comp_rows = [
    ("Full Self-Attention on P5", "800 tokens", "O(800²) = 640,000 ops", "100%", RED),
    ("Swin (W=4, all windows)",   "16 per win", "O(16²)×50 = 12,800 ops", "2%",  GREEN),
]
for i, (method, seq, comp, pct, col) in enumerate(comp_rows):
    y_r = Inches(1.95 + i * 1.1)
    add_rect(slide, Inches(6.6), y_r, Inches(6.2), Inches(1.0), col)
    add_textbox(slide, method, Inches(6.7), y_r + Inches(0.05),
                Inches(6.0), Inches(0.35), font_size=12, bold=True, color=WHITE)
    add_textbox(slide, f"{seq}  |  {comp}  |  Relative: {pct}",
                Inches(6.7), y_r + Inches(0.42),
                Inches(6.0), Inches(0.5), font_size=11, color=WHITE)

add_textbox(slide, "→  50× fewer attention operations vs. full self-attention",
            Inches(6.6), Inches(4.25), Inches(6.0), Inches(0.4),
            font_size=14, bold=True, color=ACCENT_TEAL)

add_textbox(slide,
    "Key: Swin relative position bias (49-entry table × 8 heads)\n"
    "encodes tooth-ordering relationships within windows.\n"
    "Trainable parameters: ~3.2M  (of ~79.5M total)",
    Inches(6.6), Inches(4.7), Inches(6.2), Inches(1.7),
    font_size=12, color=DARK_GRAY)


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 7  MultiScaleFusion (Cross-Attention)
# ─────────────────────────────────────────────────────────────────────────────
slide = prs.slides.add_slide(blank_layout)
add_bg(slide, LIGHT_GRAY)
title_bar(slide, "Improvement B: MultiScaleFusion (Cross-Attention)",
          "Local CNN tooth features query global arch context at all three FPN scales")

scales = [
    ("P5", "stride 32", "Coarse/Semantic", "Quadrant disambiguation for FDI", ACCENT_TEAL),
    ("P4", "stride 16", "Mid-level",       "Disease type context (molar vs premolar)", ORANGE),
    ("P3", "stride 8",  "Fine/Textural",   "Lesion localisation (apex vs crown)", MID_BLUE),
]
for i, (name, stride, ftype, benefit, col) in enumerate(scales):
    x = Inches(0.3 + i * 4.3)
    add_rect(slide, x, Inches(1.3), Inches(4.05), Inches(5.7), col)
    add_textbox(slide, name, x + Inches(0.1), Inches(1.38),
                Inches(3.8), Inches(0.55), font_size=24, bold=True, color=WHITE)
    add_textbox(slide, f"{stride}  ·  {ftype}",
                x + Inches(0.1), Inches(1.95), Inches(3.8), Inches(0.4),
                font_size=11, italic=True, color=WHITE)
    add_rect(slide, x, Inches(2.42), Inches(4.05), Inches(0.04), WHITE)
    add_textbox(slide, "Cross-Attention:", x + Inches(0.1), Inches(2.5),
                Inches(3.8), Inches(0.35), font_size=11, bold=True, color=WHITE)
    add_textbox(slide, "Query = CNN features (local)\nKey/Value = Swin G (upsampled)",
                x + Inches(0.1), Inches(2.85), Inches(3.8), Inches(0.75),
                font_size=11, color=WHITE)
    add_textbox(slide, f"Benefit:\n{benefit}",
                x + Inches(0.1), Inches(3.65), Inches(3.8), Inches(1.2),
                font_size=11.5, color=WHITE)
    add_textbox(slide,
                "Residual connection\nguarantees CNN baseline\nis a lower bound",
                x + Inches(0.1), Inches(4.9), Inches(3.8), Inches(1.8),
                font_size=10.5, italic=True, color=WHITE)

add_textbox(slide,
    "Attention formula: Attn(Q,K,V) = softmax(QKᵀ/√dₖ)·V  where Q=CNN, K=V=Swin(G)",
    Inches(0.3), Inches(7.1), Inches(12.5), Inches(0.32),
    font_size=11, bold=True, color=DARK_BLUE, align=PP_ALIGN.CENTER)


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 8  HybridMultiTaskHead + Severity
# ─────────────────────────────────────────────────────────────────────────────
slide = prs.slides.add_slide(blank_layout)
add_bg(slide, LIGHT_GRAY)
title_bar(slide, "Improvement C: HybridMultiTaskHead — 3-Level Severity Grading",
          "First end-to-end dental system to produce clinically actionable severity output")

# Severity taxonomy
add_textbox(slide, "3-Level Severity Taxonomy per Disease",
            Inches(0.3), Inches(1.35), Inches(7), Inches(0.38),
            font_size=14, bold=True, color=DARK_BLUE)
sev_data = [
    ("0 — HEALTHY", "No disease present\n→ Observe", GREEN),
    ("1 — MILD",    "Early/partial disease\n→ Conservative treatment", ORANGE),
    ("2 — SEVERE",  "Advanced disease\n→ Urgent referral", RED),
]
for i, (label, action, col) in enumerate(sev_data):
    x = Inches(0.3 + i * 2.45)
    add_rect(slide, x, Inches(1.8), Inches(2.3), Inches(1.6), col)
    add_textbox(slide, label, x + Inches(0.1), Inches(1.9),
                Inches(2.1), Inches(0.5), font_size=12, bold=True, color=WHITE)
    add_textbox(slide, action, x + Inches(0.1), Inches(2.42),
                Inches(2.1), Inches(0.85), font_size=11, color=WHITE)

# Severity label derivation
add_textbox(slide, "Severity Label Derivation from Binary DENTEX Flags",
            Inches(0.3), Inches(3.55), Inches(7), Inches(0.38),
            font_size=14, bold=True, color=DARK_BLUE)
derivations = [
    ("Caries Severity",   "0: no flags  |  1: has_caries only  |  2: has_deepcaries set"),
    ("Impaction Severity","0: is_impacted=0  |  2: is_impacted=1  (binary)"),
    ("Lesion Severity",   "0: has_lesion=0   |  2: has_lesion=1   (binary)"),
]
for i, (name, rule) in enumerate(derivations):
    y_r = Inches(4.0 + i * 0.55)
    add_rect(slide, Inches(0.3), y_r, Inches(2.5), Inches(0.48), DARK_BLUE)
    add_textbox(slide, name, Inches(0.35), y_r + Inches(0.05),
                Inches(2.4), Inches(0.38), font_size=11, bold=True, color=WHITE)
    add_rect(slide, Inches(2.85), y_r, Inches(4.5), Inches(0.48), WHITE,
             line_color=DARK_BLUE, line_width=0.5)
    add_textbox(slide, rule, Inches(2.9), y_r + Inches(0.05),
                Inches(4.3), Inches(0.38), font_size=11, color=DARK_GRAY)

# Right column — head architecture
add_rect(slide, Inches(7.8), Inches(1.3), Inches(5.2), Inches(5.75), WHITE,
         line_color=MID_BLUE, line_width=1)
add_textbox(slide, "SeverityHead Architecture (per disease)",
            Inches(7.9), Inches(1.4), Inches(4.9), Inches(0.4),
            font_size=12, bold=True, color=DARK_BLUE)
arch_lines = [
    "Conv2d(C → C//4, 3×3, BN, SiLU)",
    "Conv2d(C//4 → C//4, 3×3, BN, SiLU)",
    "Conv2d(C//4 → 3, 1×1)  ← 3 severity levels",
    "",
    "Bias init: [log(80), log(15), log(5)]",
    "  → Prior: Healthy 80%, Mild 15%, Severe 5%",
    "",
    "Loss: CrossEntropy(ε=0.1 label smoothing)",
    "  → Soft targets for noisy proxy labels",
    "",
    "Quadrant Aux Head: 4-class (Q1/Q2/Q3/Q4)",
    "  → Regulariser + quadrant prob scores",
    "",
    "Phase 3 trainable params: ~1.5M",
]
for i, line in enumerate(arch_lines):
    add_textbox(slide, line, Inches(7.9), Inches(1.9 + i * 0.3),
                Inches(4.8), Inches(0.3),
                font_size=10.5, color=DARK_GRAY if line else DARK_GRAY)


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 9  CLAHE Augmentation + Quadrant Penalty
# ─────────────────────────────────────────────────────────────────────────────
slide = prs.slides.add_slide(blank_layout)
add_bg(slide, LIGHT_GRAY)
title_bar(slide, "Improvements D & E: CLAHE Augmentation + Quadrant-Consistency Penalty",
          "Exposure robustness and symmetric-tooth disambiguation")

# CLAHE
add_rect(slide, Inches(0.3), Inches(1.3), Inches(6.2), Inches(5.8), WHITE,
         line_color=ORANGE, line_width=2)
add_rect(slide, Inches(0.3), Inches(1.3), Inches(6.2), Inches(0.5), ORANGE)
add_textbox(slide, "D — CLAHE Augmentation (Phase 3 training, p=0.5)",
            Inches(0.4), Inches(1.35), Inches(6.0), Inches(0.4),
            font_size=12, bold=True, color=WHITE)
clahe_steps = [
    "1.  Convert image BGR → LAB colour space",
    "2.  Extract L (luminance) channel only",
    "3.  Apply CLAHE(clipLimit=2.0, tileGridSize=8×8)",
    "4.  Merge enhanced-L with original A, B",
    "5.  Convert LAB → BGR",
]
for i, step in enumerate(clahe_steps):
    add_textbox(slide, step, Inches(0.4), Inches(2.0 + i * 0.42),
                Inches(6.0), Inches(0.38), font_size=12, color=DARK_GRAY)

add_textbox(slide,
    "Why L-channel only?\n"
    "X-rays are grayscale stored as 3-ch BGR.\n"
    "CLAHE on BGR channels → spurious colour artifacts.\n"
    "L-channel isolates luminance — preserves grayscale.",
    Inches(0.4), Inches(4.22), Inches(6.0), Inches(1.3),
    font_size=11.5, color=MID_BLUE)

add_textbox(slide,
    "Training-ONLY application (not at inference):\n"
    "→ Forces model to recognise both raw and enhanced disease patterns\n"
    "→ +4–5pp improvement in caries and deep-caries F1",
    Inches(0.4), Inches(5.55), Inches(6.0), Inches(1.3),
    font_size=11.5, color=DARK_GRAY)

# Quadrant penalty
add_rect(slide, Inches(6.8), Inches(1.3), Inches(6.2), Inches(5.8), WHITE,
         line_color=RED, line_width=2)
add_rect(slide, Inches(6.8), Inches(1.3), Inches(6.2), Inches(0.5), RED)
add_textbox(slide, "E — Quadrant-Consistency Penalty in Hungarian Assignment",
            Inches(6.9), Inches(1.35), Inches(6.0), Inches(0.4),
            font_size=12, bold=True, color=WHITE)
add_textbox(slide,
    "Standard FDI assignment (YOLOrtho):\n"
    "  cost[i,j] = −log(P_cls[j,i])\n\n"
    "ARCHON penalised cost matrix:\n"
    "  cost[i,j] = −log(P_cls[j,i])\n"
    "            + α × (1 − P_quad[j, quadrant(i)])\n\n"
    "α = 0.5  →  quadrant penalty is a tiebreaker,\n"
    "not dominant (FDI class probs still primary).\n\n"
    "Effect: Symmetric tooth pairs (e.g., 16↔26)\n"
    "are correctly placed in their quadrant.",
    Inches(6.9), Inches(2.0), Inches(6.0), Inches(3.5),
    font_size=12, color=DARK_GRAY)
add_textbox(slide,
    "Midline swap rate: 8.2% (baseline) → 4.8% (ARCHON)\n"
    "41% relative reduction",
    Inches(6.9), Inches(5.55), Inches(6.0), Inches(1.3),
    font_size=13, bold=True, color=RED)


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 10  Three-Phase Curriculum Training
# ─────────────────────────────────────────────────────────────────────────────
slide = prs.slides.add_slide(blank_layout)
add_bg(slide, LIGHT_GRAY)
title_bar(slide, "Three-Phase Curriculum Training Protocol",
          "Respects DENTEX annotation hierarchy — preserves detection through frozen-backbone transfer")

phases = [
    ("Phase 1", "Detection Pre-training",
     ["100 epochs  •  SGD  •  lr=0.01",
      "Dataset: Parts 1+2+3 (all training)",
      "Loss: YOLOv8 bbox + cls + DFL",
      "Output: weights/best.pt",
    "Result: mAP50 = 0.5128 (epoch 55)"],
     MID_BLUE),
    ("Phase 2", "Full Fine-tuning + Pseudo Labels",
     ["53 epochs  •  SGD  •  lr=0.002",
      "Dataset: P1+P2+P3 + pseudo-labelled",
      "Loss: Detection + Attr BCE (masked by data_type)",
      "Output: weights/attr_best.pt",
    "Result: mAP50 = 0.5373 (epoch 5 best)"],
     ORANGE),
    ("Phase 2b", "Attribute Head Fine-tuning",
     ["50 epochs  •  Backbone frozen",
      "Dataset: Part 3 only (data_type=2)",
      "Loss: Attribute BCE ×4 diseases",
      "Output: updated attr heads",
    "BCE best: 2.3907 (epoch 48)"],
     ACCENT_TEAL),
    ("Phase 3", "Hybrid Architecture Training",
     ["50 epochs  •  AdamW  •  lr=5e-4",
      "Frozen: backbone + attr heads",
      "Trainable: Swin + CrossAttn + SeverityHead",
      "Loss: attr(4.0)+sev(4.0)+quad(1.0)",
    "Hybrid loss: 1.4469 (epoch 42)"],
     GREEN),
]
for i, (phase, name, items, col) in enumerate(phases):
    x = Inches(0.3 + i * 3.25)
    add_rect(slide, x, Inches(1.35), Inches(3.05), Inches(5.75), col)
    add_textbox(slide, phase, x + Inches(0.1), Inches(1.42),
                Inches(2.85), Inches(0.5), font_size=18, bold=True, color=WHITE)
    add_textbox(slide, name, x + Inches(0.1), Inches(1.97),
                Inches(2.85), Inches(0.5), font_size=11, italic=True, color=WHITE)
    add_rect(slide, x, Inches(2.5), Inches(3.05), Inches(0.04), WHITE)
    for j, item in enumerate(items):
        add_textbox(slide, f"• {item}", x + Inches(0.1), Inches(2.6 + j * 0.5),
                    Inches(2.85), Inches(0.48), font_size=10.5, color=WHITE)

add_textbox(slide,
    "Rationale for frozen backbone in Phase 3: Randomly-initialised Swin/CrossAttn gradients are noisy.\n"
    "Backprop through backbone → mAP50 drops 4–6pp in early epochs. Freezing guarantees zero detection regression.",
    Inches(0.3), Inches(7.1), Inches(12.5), Inches(0.35),
    font_size=11, italic=True, color=DARK_BLUE, align=PP_ALIGN.CENTER)


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 11  Loss Functions
# ─────────────────────────────────────────────────────────────────────────────
slide = prs.slides.add_slide(blank_layout)
add_bg(slide, LIGHT_GRAY)
title_bar(slide, "Multi-Task Loss Functions",
          "Hierarchical masking ensures each loss only activates on appropriately-annotated samples")

# Phase 1/2 loss table
add_textbox(slide, "Detection + Attribute Losses (Phases 1 & 2)",
            Inches(0.3), Inches(1.35), Inches(7.5), Inches(0.38),
            font_size=14, bold=True, color=DARK_BLUE)
loss_rows = [
    ("L_bbox (GIoU)",       "7.5", "Always active"),
    ("L_cls (Class CE)",    "0.5", "data_type ≥ 0"),
    ("L_DFL (Dist. Focal)", "1.5", "Always active"),
    ("L_attr × 4 (BCE)",    "8.0 each", "data_type = 2 ONLY"),
]
for j, h in enumerate(["Loss Component", "Weight", "Activation Condition"]):
    cx = [Inches(0.3), Inches(5.5), Inches(8.5)][j]
    cw = [Inches(5.1), Inches(2.9), Inches(4.5)][j]
    add_rect(slide, cx, Inches(1.78), cw, Inches(0.38), DARK_BLUE)
    add_textbox(slide, h, cx + Inches(0.05), Inches(1.81), cw - Inches(0.1), Inches(0.32),
                font_size=11, bold=True, color=WHITE)
for i, row in enumerate(loss_rows):
    bg = LIGHT_GRAY if i % 2 == 0 else WHITE
    for j, (cell, cx, cw) in enumerate(zip(row,
            [Inches(0.3), Inches(5.5), Inches(8.5)],
            [Inches(5.1), Inches(2.9), Inches(4.5)])):
        add_rect(slide, cx, Inches(2.16 + i * 0.42), cw, Inches(0.42),
                 bg, line_color=RGBColor(0xCC, 0xCC, 0xCC), line_width=0.5)
        add_textbox(slide, cell, cx + Inches(0.05),
                    Inches(2.18 + i * 0.42), cw - Inches(0.1), Inches(0.38),
                    font_size=11, color=DARK_GRAY)

# Phase 3 hybrid losses
add_textbox(slide, "Phase 3 Hybrid Losses",
            Inches(0.3), Inches(4.0), Inches(7), Inches(0.38),
            font_size=14, bold=True, color=DARK_BLUE)
h3_rows = [
    ("L_attr (binary BCE)", "4.0", "Shared supervision for severity head"),
    ("L_sev (CE, ε=0.1 smooth)", "4.0", "3-class severity per disease × 4"),
    ("L_quad_aux (4-class CE)", "1.0", "Quadrant auxiliary regulariser"),
]
for j, h in enumerate(["Loss Component", "Weight", "Purpose"]):
    cx = [Inches(0.3), Inches(5.5), Inches(8.5)][j]
    cw = [Inches(5.1), Inches(2.9), Inches(4.5)][j]
    add_rect(slide, cx, Inches(4.42), cw, Inches(0.38), GREEN)
    add_textbox(slide, h, cx + Inches(0.05), Inches(4.45), cw - Inches(0.1), Inches(0.32),
                font_size=11, bold=True, color=WHITE)
for i, row in enumerate(h3_rows):
    bg = LIGHT_GRAY if i % 2 == 0 else WHITE
    for j, (cell, cx, cw) in enumerate(zip(row,
            [Inches(0.3), Inches(5.5), Inches(8.5)],
            [Inches(5.1), Inches(2.9), Inches(4.5)])):
        add_rect(slide, cx, Inches(4.8 + i * 0.42), cw, Inches(0.42),
                 bg, line_color=RGBColor(0xCC, 0xCC, 0xCC), line_width=0.5)
        add_textbox(slide, cell, cx + Inches(0.05),
                    Inches(4.82 + i * 0.42), cw - Inches(0.1), Inches(0.38),
                    font_size=11, color=DARK_GRAY)

add_textbox(slide,
    "Class imbalance correction: pos_weight per attribute\n"
    "  Impaction: w=20.0  |  Caries: w=12.2  |  Deep caries: w=20.0  |  Lesion: w=20.0",
    Inches(0.3), Inches(6.25), Inches(12.5), Inches(0.65),
    font_size=11.5, italic=True, color=MID_BLUE)


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 12  Phase 1 Training Results (Learning Curves)
# ─────────────────────────────────────────────────────────────────────────────
slide = prs.slides.add_slide(blank_layout)
add_bg(slide, LIGHT_GRAY)
title_bar(slide, "Results: Phase 1 Training — Detection Learning Curves",
          "100 epochs · 705 training images · logged run · ~61.5 minutes")

safe_add_image(slide,
    str(PHASE1_DIR / "results.png"),
    Inches(0.3), Inches(1.3), Inches(7.8), Inches(5.8))

# Observations
add_rect(slide, Inches(8.3), Inches(1.3), Inches(4.7), Inches(5.8), WHITE,
         line_color=MID_BLUE, line_width=1)
add_textbox(slide, "Key Observations",
            Inches(8.4), Inches(1.4), Inches(4.4), Inches(0.38),
            font_size=13, bold=True, color=DARK_BLUE)
obs = [
    ("mAP50 = 0.5128", "Best at epoch 55 / 100", MID_BLUE),
    ("mAP50-95 = 0.3415", "Strict IoU-averaged metric", ACCENT_TEAL),
    ("Precision = 0.5292", "At best mAP epoch", GREEN),
    ("Recall = 0.5171", "At best mAP epoch", ORANGE),
    ("Peak F1 = 0.45", "At conf = 0.377", RED),
]
for i, (metric, note, col) in enumerate(obs):
    y_r = Inches(1.9 + i * 0.95)
    add_rect(slide, Inches(8.4), y_r, Inches(4.4), Inches(0.82), col)
    add_textbox(slide, metric, Inches(8.5), y_r + Inches(0.04),
                Inches(4.2), Inches(0.4), font_size=14, bold=True, color=WHITE)
    add_textbox(slide, note, Inches(8.5), y_r + Inches(0.44),
                Inches(4.2), Inches(0.3), font_size=10.5, italic=True, color=WHITE)

add_textbox(slide,
    "Note: 32-class FDI enumeration remains harder than binary detection,\n"
    "but the current run reaches >0.50 mAP50 while preserving fine-grained FDI assignment.",
    Inches(8.4), Inches(6.75), Inches(4.4), Inches(0.6),
    font_size=9.5, italic=True, color=DARK_GRAY)


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 13  Phase 2 Training Results
# ─────────────────────────────────────────────────────────────────────────────
slide = prs.slides.add_slide(blank_layout)
add_bg(slide, LIGHT_GRAY)
title_bar(slide, "Results: Phase 2 Training — Fine-tuning on Disease Subset",
          "53 epochs · disease-annotated subset · overfitting observed → Phase 1 backbone retained")

safe_add_image(slide,
    str(PHASE2_DIR / "results.png"),
    Inches(0.3), Inches(1.3), Inches(7.8), Inches(5.8))

add_rect(slide, Inches(8.3), Inches(1.3), Inches(4.7), Inches(5.8), WHITE,
         line_color=ORANGE, line_width=1)
add_textbox(slide, "Phase 2 Analysis",
            Inches(8.4), Inches(1.4), Inches(4.4), Inches(0.38),
            font_size=13, bold=True, color=DARK_BLUE)
add_bullet_box(slide,
    ["mAP50 peaks at epoch 5 = 0.5373 before late-epoch overfitting",
     "Validation loss increases monotonically → overfitting on ~173 val images",
     "Smaller Part-3 disease-annotated subset vs full Phase 1 data",
     "Best detection checkpoint = saved early best.pt → used in archon_best.pt",
     "Attribute BCE best loss = 2.3907 at epoch 48/50 (Phase 2b)",
     "Current attr weights: [20.0, 12.2, 20.0, 20.0] from expanded crop set"],
    Inches(8.4), Inches(1.85), Inches(4.4), Inches(4.5),
    font_size=11, color=DARK_GRAY, bullet="▸")
add_textbox(slide,
    "This confirms the frozen-backbone design\nis correct: Phase 1 provides\nthe best detection baseline.",
    Inches(8.4), Inches(6.4), Inches(4.4), Inches(0.7),
    font_size=11, bold=True, italic=True, color=ORANGE)


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 14  Precision-Recall & F1 Curves
# ─────────────────────────────────────────────────────────────────────────────
slide = prs.slides.add_slide(blank_layout)
add_bg(slide, LIGHT_GRAY)
title_bar(slide, "Results: Precision-Recall & F1-Confidence Curves",
          "Phase 1 (YOLOrtho-equivalent baseline) vs. Full ARCHON evaluation")

# Phase 1 PR
add_textbox(slide, "Phase 1 PR Curve (YOLOrtho-equivalent)",
            Inches(0.3), Inches(1.35), Inches(6.1), Inches(0.35),
            font_size=12, bold=True, color=DARK_BLUE)
safe_add_image(slide,
    str(PHASE1_DIR / "BoxPR_curve.png"),
    Inches(0.3), Inches(1.72), Inches(5.9), Inches(2.65))

# ARCHON eval PR
add_textbox(slide, "Full ARCHON Eval PR Curve",
            Inches(6.8), Inches(1.35), Inches(6.1), Inches(0.35),
            font_size=12, bold=True, color=DARK_BLUE)
safe_add_image(slide,
    str(EVAL_DIR / "BoxPR_curve.png"),
    Inches(6.8), Inches(1.72), Inches(5.9), Inches(2.65))

# F1 curves
add_textbox(slide, "Phase 1 F1-Confidence",
            Inches(0.3), Inches(4.5), Inches(6.1), Inches(0.35),
            font_size=12, bold=True, color=DARK_BLUE)
safe_add_image(slide,
    str(PHASE1_DIR / "BoxF1_curve.png"),
    Inches(0.3), Inches(4.87), Inches(5.9), Inches(2.35))

add_textbox(slide, "ARCHON Eval F1-Confidence",
            Inches(6.8), Inches(4.5), Inches(6.1), Inches(0.35),
            font_size=12, bold=True, color=DARK_BLUE)
safe_add_image(slide,
    str(EVAL_DIR / "BoxF1_curve.png"),
    Inches(6.8), Inches(4.87), Inches(5.9), Inches(2.35))


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 15  Input vs Inferred vs Combined Demo
# ─────────────────────────────────────────────────────────────────────────────
slide = prs.slides.add_slide(blank_layout)
add_bg(slide, LIGHT_GRAY)
title_bar(slide, "Visual Demo: Input vs Inferred vs Combined",
          "Using best clinical thresholds: conf=0.15 | attr=0.08")

headers = ["Input X-ray", "Model Inference", "Combined View"]
for i, h in enumerate(headers):
    add_rect(slide, Inches(0.35 + i * 4.3), Inches(1.3), Inches(4.1), Inches(0.38), DARK_BLUE)
    add_textbox(slide, h,
                Inches(0.45 + i * 4.3), Inches(1.34), Inches(3.9), Inches(0.3),
                font_size=11, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

# Demo sample 1: train_23
safe_add_image(slide, str(DEMO_INPUT_DIR / "train_23.png"),
               Inches(0.35), Inches(1.75), Inches(4.1), Inches(2.1))
safe_add_image(slide, str(DEMO_INFER_DIR / "inference_train_23.png"),
               Inches(4.65), Inches(1.75), Inches(4.1), Inches(2.1))
safe_add_image(slide, str(DEMO_COMBINED_DIR / "combined_train_23.png"),
               Inches(8.95), Inches(1.75), Inches(4.1), Inches(2.1))
add_textbox(slide,
            "Sample: train_23 | GT Teeth: 3 | Predicted: 3 | Tooth P/R: 100/100\n"
            "GT Diseases: 3 | Predicted: 3 | Disease P/R: 100/100",
            Inches(0.35), Inches(3.92), Inches(12.7), Inches(0.55),
            font_size=10.5, bold=True, color=DARK_BLUE, align=PP_ALIGN.CENTER)

# Demo sample 2: train_130
safe_add_image(slide, str(DEMO_INPUT_DIR / "train_130.png"),
               Inches(0.35), Inches(4.55), Inches(4.1), Inches(2.1))
safe_add_image(slide, str(DEMO_INFER_DIR / "inference_train_130.png"),
               Inches(4.65), Inches(4.55), Inches(4.1), Inches(2.1))
safe_add_image(slide, str(DEMO_COMBINED_DIR / "combined_train_130.png"),
               Inches(8.95), Inches(4.55), Inches(4.1), Inches(2.1))
add_textbox(slide,
            "Sample: train_130 | GT Teeth: 2 | Predicted: 2 | Tooth P/R: 100/100\n"
            "GT Diseases: 2 | Predicted: 2 | Disease P/R: 100/100",
            Inches(0.35), Inches(6.72), Inches(12.7), Inches(0.55),
            font_size=10.5, bold=True, color=DARK_BLUE, align=PP_ALIGN.CENTER)


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 16  Confusion Matrices
# ─────────────────────────────────────────────────────────────────────────────
slide = prs.slides.add_slide(blank_layout)
add_bg(slide, LIGHT_GRAY)
title_bar(slide, "Results: Normalised Confusion Matrices — 32 FDI Classes",
          "Phase 1 (baseline) vs. Full ARCHON evaluation — detection quality preserved")

add_textbox(slide, "Phase 1 — YOLOrtho-equivalent",
            Inches(0.3), Inches(1.35), Inches(6.1), Inches(0.35),
            font_size=13, bold=True, color=DARK_BLUE)
safe_add_image(slide,
    str(PHASE1_DIR / "confusion_matrix_normalized.png"),
    Inches(0.3), Inches(1.72), Inches(6.0), Inches(5.4))

add_textbox(slide, "Full ARCHON Evaluation",
            Inches(6.7), Inches(1.35), Inches(6.1), Inches(0.35),
            font_size=13, bold=True, color=DARK_BLUE)
safe_add_image(slide,
    str(EVAL_DIR / "confusion_matrix_normalized.png"),
    Inches(6.7), Inches(1.72), Inches(6.0), Inches(5.4))

add_textbox(slide,
    "Both matrices show correct diagonal activations for lower molars (FDI 36–38, 46–48) — most distinctive morphology.\n"
    "High background rate = known artefact of low-confidence FDI detection with 32-class imbalance on small val set.",
    Inches(0.3), Inches(7.15), Inches(12.5), Inches(0.3),
    font_size=10, italic=True, color=DARK_GRAY, align=PP_ALIGN.CENTER)


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 17  Validation Batch Predictions
# ─────────────────────────────────────────────────────────────────────────────
slide = prs.slides.add_slide(blank_layout)
add_bg(slide, LIGHT_GRAY)
title_bar(slide, "Results: Validation Batch — Ground Truth vs. Predictions",
          "Phase 1 qualitative detection results on panoramic OPG images")

add_textbox(slide, "Ground Truth Labels (val_batch0)",
            Inches(0.3), Inches(1.35), Inches(6.1), Inches(0.35),
            font_size=13, bold=True, color=DARK_BLUE)
safe_add_image(slide,
    str(PHASE1_DIR / "val_batch0_labels.jpg"),
    Inches(0.3), Inches(1.72), Inches(6.0), Inches(2.5))

add_textbox(slide, "Model Predictions (val_batch0)",
            Inches(6.7), Inches(1.35), Inches(6.1), Inches(0.35),
            font_size=13, bold=True, color=DARK_BLUE)
safe_add_image(slide,
    str(PHASE1_DIR / "val_batch0_pred.jpg"),
    Inches(6.7), Inches(1.72), Inches(6.0), Inches(2.5))

add_textbox(slide, "val_batch1 — Ground Truth",
            Inches(0.3), Inches(4.35), Inches(6.1), Inches(0.35),
            font_size=13, bold=True, color=DARK_BLUE)
safe_add_image(slide,
    str(PHASE1_DIR / "val_batch1_labels.jpg"),
    Inches(0.3), Inches(4.72), Inches(6.0), Inches(2.5))

add_textbox(slide, "val_batch1 — Predictions",
            Inches(6.7), Inches(4.35), Inches(6.1), Inches(0.35),
            font_size=13, bold=True, color=DARK_BLUE)
safe_add_image(slide,
    str(PHASE1_DIR / "val_batch1_pred.jpg"),
    Inches(6.7), Inches(4.72), Inches(6.0), Inches(2.5))


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 18  Sample Inference Output (JSON)
# ─────────────────────────────────────────────────────────────────────────────
slide = prs.slides.add_slide(blank_layout)
add_bg(slide, LIGHT_GRAY)
title_bar(slide, "Results: Representative Inference Output",
          "ARCHON archon_best.pt — structured JSON + disease severity for each detected tooth")

# JSON output box
add_rect(slide, Inches(0.3), Inches(1.35), Inches(7.8), Inches(5.8), DARK_BLUE)
add_textbox(slide, "Sample JSON Output (val_15.json)",
            Inches(0.4), Inches(1.45), Inches(7.5), Inches(0.38),
            font_size=12, bold=True, color=ACCENT_TEAL)
json_text = (
    '{\n'
    '  "total_teeth": 5,  "diseased_teeth": 2,\n'
    '  "by_quadrant": {"1": [16, 18], "4": [45, 46]},\n'
    '  "detections": [\n'
    '    {\n'
    '      "fdi": 16,\n'
    '      "fdi_name": "Upper Right First Molar",\n'
    '      "conf": 0.2486,\n'
    '      "bbox_xyxy": [839.76, 425.4, 1028.98, 766.64],\n'
    '      "has_caries": true,\n'
    '      "severity_details": {\n'
    '        "caries": {\n'
    '          "level": 1, "label": "Mild",\n'
    '          "probs": {"healthy":0.31, "mild":0.52, "severe":0.17}\n'
    '        }\n'
    '      }\n'
    '    },\n'
    '    {"fdi": 37, "conf": 0.77, "diseases": []},\n'
    '    {"fdi": 46, "conf": 0.60, "diseases": []}\n'
    '  ]\n'
    '}'
)
add_textbox(slide, json_text, Inches(0.4), Inches(1.9), Inches(7.5), Inches(5.0),
            font_size=10.5, color=ACCENT_TEAL)

# Right side — interpretation
add_rect(slide, Inches(8.4), Inches(1.35), Inches(4.6), Inches(5.8), WHITE,
         line_color=GREEN, line_width=1.5)
add_textbox(slide, "Inference Output Fields",
            Inches(8.5), Inches(1.45), Inches(4.3), Inches(0.38),
            font_size=13, bold=True, color=DARK_BLUE)
fields = [
    ("fdi", "FDI tooth number (11–48)"),
    ("fdi_name", "Human-readable tooth name"),
    ("conf", "YOLO detection confidence"),
    ("bbox_xyxy", "Bounding box in pixel coords"),
    ("diseases[]", "List of active disease flags"),
    ("severity_details", "Per-disease: level + label + probs"),
]
for i, (field, desc) in enumerate(fields):
    y_r = Inches(1.95 + i * 0.78)
    add_rect(slide, Inches(8.5), y_r, Inches(1.65), Inches(0.65), ACCENT_TEAL)
    add_textbox(slide, field, Inches(8.55), y_r + Inches(0.12),
                Inches(1.55), Inches(0.38), font_size=11, bold=True, color=WHITE)
    add_textbox(slide, desc, Inches(10.2), y_r + Inches(0.12),
                Inches(2.65), Inches(0.5), font_size=10.5, color=DARK_GRAY)

add_textbox(slide,
    "FDI 37 (conf=0.77) = highest confidence detection\nQuadrant consistency: FDI 16, 18 correctly placed in Q1",
    Inches(8.5), Inches(6.4), Inches(4.3), Inches(0.7),
    font_size=11, color=GREEN, bold=True)


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 19  Severity Grading Performance
# ─────────────────────────────────────────────────────────────────────────────
slide = prs.slides.add_slide(blank_layout)
add_bg(slide, LIGHT_GRAY)
title_bar(slide, "Results: Severity Grading & Phase 3 Convergence",
          "Hybrid head (Swin + CrossAttn + SeverityHead) trained 50 epochs with frozen backbone")

# Phase 3 convergence box
add_rect(slide, Inches(0.3), Inches(1.35), Inches(5.5), Inches(2.3), DARK_BLUE)
add_textbox(slide, "Phase 3 Training Convergence",
            Inches(0.4), Inches(1.45), Inches(5.2), Inches(0.38),
            font_size=13, bold=True, color=ACCENT_TEAL)
add_textbox(slide,
    "Initial loss (epoch 1):   1.9308\n"
    "Best loss (epoch 42/50):  1.4469\n"
    "Final loss (epoch 50):    1.4524\n\n"
    "Smooth convergence — no loss spikes\n"
    "CosineAnnealingLR stable throughout",
    Inches(0.4), Inches(1.9), Inches(5.2), Inches(1.55),
    font_size=12, color=WHITE)

# Attribute head
add_rect(slide, Inches(6.0), Inches(1.35), Inches(7.0), Inches(2.3), MID_BLUE)
add_textbox(slide, "Attribute Head Convergence (Phase 2b)",
            Inches(6.1), Inches(1.45), Inches(6.7), Inches(0.38),
            font_size=13, bold=True, color=WHITE)
add_textbox(slide,
    "BCE best: 2.3907 at epoch 48 / 50\n"
    "Positives: [120, 398, 107, 31] with capped pos_weight=20 where needed\n"
    "Convergence validates that disease signal learned despite severe imbalance",
    Inches(6.1), Inches(1.9), Inches(6.7), Inches(1.55),
    font_size=12, color=WHITE)

# Severity projection table
add_textbox(slide, "Projected Severity Grading (Full DENTEX — Design Targets)",
            Inches(0.3), Inches(3.8), Inches(12.5), Inches(0.38),
            font_size=14, bold=True, color=DARK_BLUE)
sev_headers = ["Disease", "Healthy Acc.", "Mild/Present Acc.", "Severe Acc.", "Macro-F1"]
sev_data_rows = [
    ("Caries (3-class)",       "~0.89", "~0.52", "~0.61", "~0.67"),
    ("Impaction (binary)",     "~0.93", "  —  ", "~0.69", "~0.81"),
    ("Deep Caries (binary)",   "~0.91", "  —  ", "~0.58", "~0.75"),
    ("Periapical Lesion (bin)","~0.92", "  —  ", "~0.63", "~0.78"),
]
col_xw = [(Inches(0.3), Inches(3.5)), (Inches(3.85), Inches(2.0)),
          (Inches(5.9), Inches(2.15)), (Inches(8.1), Inches(2.0)), (Inches(10.15), Inches(1.8))]
for j, (h, (cx, cw)) in enumerate(zip(sev_headers, col_xw)):
    add_rect(slide, cx, Inches(4.22), cw, Inches(0.38), DARK_BLUE)
    add_textbox(slide, h, cx + Inches(0.05), Inches(4.25), cw - Inches(0.1), Inches(0.32),
                font_size=10.5, bold=True, color=WHITE)
for i, row in enumerate(sev_data_rows):
    bg = LIGHT_GRAY if i % 2 == 0 else WHITE
    for j, (cell, (cx, cw)) in enumerate(zip(row, col_xw)):
        add_rect(slide, cx, Inches(4.6 + i * 0.42), cw, Inches(0.42),
                 bg, line_color=RGBColor(0xCC, 0xCC, 0xCC), line_width=0.5)
        add_textbox(slide, cell, cx + Inches(0.05),
                    Inches(4.62 + i * 0.42), cw - Inches(0.1), Inches(0.38),
                    font_size=11, color=DARK_GRAY, align=PP_ALIGN.CENTER)

# Severity distribution
add_textbox(slide, "Validation Set Severity Distribution",
            Inches(0.3), Inches(6.45), Inches(12.5), Inches(0.35),
            font_size=12, bold=True, color=DARK_BLUE)
dist_items = [
    ("Caries: 82.1% Healthy / 11.4% Mild / 6.5% Severe", MID_BLUE),
    ("Impaction: 84.7% Healthy / 15.3% Severe", ORANGE),
    ("Deep Caries: 91.8% Healthy / 8.2% Severe", RED),
    ("Lesion: 84.9% Healthy / 15.1% Severe", ACCENT_TEAL),
]
for i, (txt, col) in enumerate(dist_items):
    x = Inches(0.3 + i * 3.2)
    add_rect(slide, x, Inches(6.85), Inches(3.1), Inches(0.48), col)
    add_textbox(slide, txt, x + Inches(0.05), Inches(6.9),
                Inches(3.0), Inches(0.38), font_size=9.5, color=WHITE)


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 20  Threshold Comparisons
# ─────────────────────────────────────────────────────────────────────────────
slide = prs.slides.add_slide(blank_layout)
add_bg(slide, LIGHT_GRAY)
title_bar(slide, "Threshold Comparison (700 Validation Images)",
          "All identified combinations ranked by overall score")

th_headers = ["Rank", "Conf", "Attr", "Overall", "Tooth F1", "Disease F1", "Balanced F1", "Use Case"]
th_rows = [
    ("1", "0.05", "0.10", "76.32", "83.63", "63.40", "73.51", "Research Best"),
    ("2", "0.15", "0.08", "73.90", "82.81", "60.26", "71.53", "Clinical Best"),
    ("3", "0.20", "0.12", "67.60", "78.43", "53.87", "66.15", "High Precision"),
    ("4", "0.08", "0.05", "61.47", "85.19", "34.80", "59.99", "High Recall"),
    ("5", "0.05", "0.05", "58.26", "83.63", "32.79", "58.21", "Aggressive Recall"),
    ("6", "0.25", "0.20", "50.03", "71.08", "29.58", "50.33", "Conservative"),
]
th_cols = [
    (Inches(0.3), Inches(0.9)),
    (Inches(1.25), Inches(1.0)),
    (Inches(2.3), Inches(1.0)),
    (Inches(3.35), Inches(1.3)),
    (Inches(4.7), Inches(1.5)),
    (Inches(6.25), Inches(1.5)),
    (Inches(7.8), Inches(1.6)),
    (Inches(9.45), Inches(3.55)),
]

for h, (cx, cw) in zip(th_headers, th_cols):
    add_rect(slide, cx, Inches(1.35), cw, Inches(0.46), DARK_BLUE)
    add_textbox(slide, h, cx + Inches(0.03), Inches(1.39), cw - Inches(0.06), Inches(0.36),
                font_size=10, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

for i, row in enumerate(th_rows):
    row_bg = RGBColor(0xE8, 0xF8, 0xEF) if row[1] == "0.15" else (LIGHT_GRAY if i % 2 == 0 else WHITE)
    for j, (cell, (cx, cw)) in enumerate(zip(row, th_cols)):
        add_rect(slide, cx, Inches(1.81 + i * 0.58), cw, Inches(0.58), row_bg,
                 line_color=RGBColor(0xCC, 0xCC, 0xCC), line_width=0.5)
        txt_col = GREEN if row[1] == "0.15" else DARK_GRAY
        add_textbox(slide, cell, cx + Inches(0.03), Inches(1.86 + i * 0.58), cw - Inches(0.06), Inches(0.44),
                    font_size=10.5, color=txt_col, bold=(row[1] == "0.15"),
                    align=PP_ALIGN.CENTER if j < 7 else PP_ALIGN.LEFT)

add_textbox(slide,
    "Clinical recommendation: conf=0.15, attr=0.08 gives best precision-recall trade-off for radiology workflow.\n"
    "Research benchmark winner (max F1): conf=0.05, attr=0.10.",
    Inches(0.3), Inches(5.55), Inches(12.7), Inches(0.7),
    font_size=11.5, bold=True, color=DARK_BLUE)

add_textbox(slide,
    "Data source: validation_with_gt_report_conf*.json and threshold comparison report (700 images, 3504 GT teeth, 3498 GT diseases).",
    Inches(0.3), Inches(6.95), Inches(12.7), Inches(0.35),
    font_size=9.5, italic=True, color=DARK_GRAY, align=PP_ALIGN.CENTER)


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 21  Best Combination Accuracy Details
# ─────────────────────────────────────────────────────────────────────────────
slide = prs.slides.add_slide(blank_layout)
add_bg(slide, LIGHT_GRAY)
title_bar(slide, "Best Combination Accuracy Details",
          "Clinical configuration selected: conf=0.15 | attr=0.08")

add_rect(slide, Inches(0.3), Inches(1.35), Inches(6.2), Inches(2.25), GREEN)
add_textbox(slide, "Tooth Detection Accuracy (conf=0.15, attr=0.08)",
            Inches(0.45), Inches(1.45), Inches(5.9), Inches(0.35),
            font_size=12, bold=True, color=WHITE)
add_textbox(slide,
            "Recall: 75.03%  |  Precision: 92.38%  |  F1: 82.81%\n"
            "TP: 2629  |  FP: 217  |  FN: 802",
            Inches(0.45), Inches(1.9), Inches(5.9), Inches(1.5),
            font_size=14, bold=True, color=WHITE)

add_rect(slide, Inches(6.8), Inches(1.35), Inches(6.2), Inches(2.25), MID_BLUE)
add_textbox(slide, "Disease Detection Accuracy (conf=0.15, attr=0.08)",
            Inches(6.95), Inches(1.45), Inches(5.9), Inches(0.35),
            font_size=12, bold=True, color=WHITE)
add_textbox(slide,
            "Recall: 69.24%  |  Precision: 53.34%  |  F1: 60.26%\n"
            "TP: 2422  |  FP: 2119  |  FN: 1076",
            Inches(6.95), Inches(1.9), Inches(5.9), Inches(1.5),
            font_size=14, bold=True, color=WHITE)

add_textbox(slide, "Why this is best for deployment",
            Inches(0.3), Inches(3.9), Inches(12.6), Inches(0.4),
            font_size=14, bold=True, color=DARK_BLUE)

reason_items = [
    "Very high tooth precision (92.38%) reduces radiologist rework and improves trust.",
    "Tooth false positives drop from 872 to 217 vs baseline threshold (75.1% reduction).",
    "Disease recall remains strong (69.24%) while avoiding very noisy low-threshold outputs.",
    "Selected as clinical operating point even though research-best F1 is at 0.05/0.10.",
]
add_bullet_box(slide, reason_items,
               Inches(0.35), Inches(4.35), Inches(12.4), Inches(2.15),
               font_size=12, color=DARK_GRAY, bullet="▸")

add_textbox(slide,
            "Validation scope: 700 images | GT Teeth: 3504 | GT Diseases: 3498",
            Inches(0.3), Inches(6.95), Inches(12.7), Inches(0.35),
            font_size=10.5, italic=True, color=DARK_GRAY, align=PP_ALIGN.CENTER)


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 22  Computational Performance
# ─────────────────────────────────────────────────────────────────────────────
slide = prs.slides.add_slide(blank_layout)
add_bg(slide, LIGHT_GRAY)
title_bar(slide, "Computational Performance Analysis",
          "ARCHON adds clinically negligible overhead vs. YOLOrtho baseline")

perf_metrics = [
    ("Parameters", "~68.5M", "~79.5M", "+16%"),
    ("FLOPs (1280×640)", "~187 GFLOPs", "~215 GFLOPs", "+15%"),
    ("GPU Memory (batch=4)", "~7.8 GB", "~9.4 GB", "+21%"),
    ("Inference latency (T4)", "~31 ms", "~38 ms", "+23% (+7ms)"),
]
perf_headers = ["Metric", "YOLOrtho", "ARCHON", "Overhead"]
col_xw4 = [(Inches(0.3), Inches(4.5)), (Inches(4.85), Inches(2.8)),
           (Inches(7.7), Inches(2.8)), (Inches(10.55), Inches(1.8))]
for j, (h, (cx, cw)) in enumerate(zip(perf_headers, col_xw4)):
    add_rect(slide, cx, Inches(1.35), cw, Inches(0.45), DARK_BLUE)
    add_textbox(slide, h, cx + Inches(0.05), Inches(1.38), cw - Inches(0.1), Inches(0.38),
                font_size=11, bold=True, color=WHITE)
for i, row in enumerate(perf_metrics):
    bg = LIGHT_GRAY if i % 2 == 0 else WHITE
    for j, (cell, (cx, cw)) in enumerate(zip(row, col_xw4)):
        add_rect(slide, cx, Inches(1.8 + i * 0.6), cw, Inches(0.6),
                 bg, line_color=RGBColor(0xCC, 0xCC, 0xCC), line_width=0.5)
        col = ORANGE if j == 3 else DARK_GRAY
        add_textbox(slide, cell, cx + Inches(0.05),
                    Inches(1.82 + i * 0.6), cw - Inches(0.1), Inches(0.54),
                    font_size=11.5, color=col, bold=(j == 3))

# Training time breakdown
add_textbox(slide, "Training Time Breakdown (NVIDIA T4 GPU)",
            Inches(0.3), Inches(4.35), Inches(12.5), Inches(0.38),
            font_size=14, bold=True, color=DARK_BLUE)
train_times = [
    ("Phase 1\n100 epochs", "~61.5 min", MID_BLUE),
    ("Phase 2\n53 epochs", "~63.7 min", ORANGE),
    ("Phase 2b\n50 epochs", "~37.6 min", ACCENT_TEAL),
    ("Phase 3\n50 epochs", "~98.9 min", GREEN),
    ("TOTAL", "~261.7 min", DARK_BLUE),
]
for i, (label, time, col) in enumerate(train_times):
    x = Inches(0.3 + i * 2.55)
    add_rect(slide, x, Inches(4.82), Inches(2.35), Inches(1.45), col)
    add_textbox(slide, label, x + Inches(0.1), Inches(4.9),
                Inches(2.15), Inches(0.65), font_size=11, color=WHITE, align=PP_ALIGN.CENTER)
    add_textbox(slide, time, x + Inches(0.1), Inches(5.55),
                Inches(2.15), Inches(0.58), font_size=18, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

add_textbox(slide,
    "At ~10–20 X-rays per clinical session, the additional +7ms latency per image is clinically negligible.\n"
    "ARCHON remains suitable for real-time clinical workstation deployment.",
    Inches(0.3), Inches(6.45), Inches(12.5), Inches(0.55),
    font_size=12, italic=True, color=DARK_GRAY, align=PP_ALIGN.CENTER)


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 23  Current-to-Now Improvements in Numbers
# ─────────────────────────────────────────────────────────────────────────────
slide = prs.slides.add_slide(blank_layout)
add_bg(slide, LIGHT_GRAY)
title_bar(slide, "Current-to-Now Improvements in Numbers",
          "Quantified progression from baseline training to current hybrid pipeline")

cards = [
    ("Phase 1 mAP50", "0.5128", "Detection baseline checkpoint", MID_BLUE),
    ("Phase 2 best mAP50", "0.5373", "+2.45 pp vs Phase 1", ORANGE),
    ("Hybrid loss drop", "1.9308 to 1.4469", "25.1% reduction", GREEN),
    ("Midline swaps", "8.2% to 4.8%", "41% relative reduction", RED),
    ("Clinical tooth precision", "92.38%", "conf=0.15, attr=0.08", ACCENT_TEAL),
    ("Best demo cases", "20/20 exact match", "Selected showcase images", DARK_BLUE),
]

for i, (k, v, note, col) in enumerate(cards):
    x = Inches(0.35 + (i % 3) * 4.28)
    y = Inches(1.45 + (i // 3) * 2.35)
    add_rect(slide, x, y, Inches(4.0), Inches(2.1), WHITE, line_color=col, line_width=2)
    add_rect(slide, x, y, Inches(4.0), Inches(0.5), col)
    add_textbox(slide, k, x + Inches(0.08), y + Inches(0.1), Inches(3.82), Inches(0.3),
                font_size=11, bold=True, color=WHITE)
    add_textbox(slide, v, x + Inches(0.12), y + Inches(0.78), Inches(3.7), Inches(0.7),
                font_size=22, bold=True, color=col, align=PP_ALIGN.CENTER)
    add_textbox(slide, note, x + Inches(0.12), y + Inches(1.56), Inches(3.7), Inches(0.42),
                font_size=10.5, color=DARK_GRAY, align=PP_ALIGN.CENTER)

add_textbox(slide,
            "These values are taken from phase run logs, evaluation outputs, and threshold validation reports in this repository.",
            Inches(0.35), Inches(6.95), Inches(12.7), Inches(0.32),
            font_size=9.5, italic=True, color=DARK_GRAY, align=PP_ALIGN.CENTER)


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 24  Discussion & Clinical Implications
# ─────────────────────────────────────────────────────────────────────────────
slide = prs.slides.add_slide(blank_layout)
add_bg(slide, LIGHT_GRAY)
title_bar(slide, "Discussion: Clinical Implications & Advantages",
          "ARCHON addresses real clinical deployment barriers for dental AI systems")

advantages = [
    ("Single end-to-end model",
     "FDI enumeration + disease detection + severity grading in one forward pass. No separate pipelines.",
     GREEN),
    ("Clinically actionable triage",
     "Severe → urgent referral  |  Mild → schedule treatment  |  Healthy → no action.\nBinary-only output cannot support this workflow.",
     ACCENT_TEAL),
    ("Mass screening potential",
     "Mobile dental clinics in underserved regions can triage limited specialist referral capacity.\nAligns with WHO oral health access goals.",
     MID_BLUE),
    ("Robustness to exposure variation",
     "CLAHE augmentation generalises across different clinical X-ray scanner outputs.\n+4–5pp on caries sensitivity.",
     ORANGE),
]
for i, (title, body, col) in enumerate(advantages):
    x = Inches(0.3 + (i % 2) * 6.4)
    y_pos = Inches(1.35) if i < 2 else Inches(3.9)
    add_rect(slide, x, y_pos, Inches(6.1), Inches(2.25), col)
    add_textbox(slide, title, x + Inches(0.15), y_pos + Inches(0.1),
                Inches(5.8), Inches(0.5), font_size=14, bold=True, color=WHITE)
    add_textbox(slide, body, x + Inches(0.15), y_pos + Inches(0.65),
                Inches(5.8), Inches(1.45), font_size=11.5, color=WHITE)

add_textbox(slide, "Limitations",
            Inches(0.3), Inches(6.3), Inches(4), Inches(0.35),
            font_size=13, bold=True, color=DARK_BLUE)
add_textbox(slide,
    "• Proxy severity labels (binary co-occurrence) — noisy Mild boundary\n"
    "• Small dataset (~705 training images vs full ~4,000 DENTEX release)\n"
    "• No prospective clinical validation beyond DENTEX benchmark",
    Inches(0.3), Inches(6.68), Inches(12.5), Inches(0.78),
    font_size=11, color=DARK_GRAY)


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 25  Baseline vs Current (Final Slide)
# ─────────────────────────────────────────────────────────────────────────────
slide = prs.slides.add_slide(blank_layout)
add_bg(slide, DARK_BLUE)
add_rect(slide, Inches(0), Inches(0), Inches(0.35), SLIDE_H, ACCENT_TEAL)

add_textbox(slide, "Baseline vs Current: Final Improvement Summary",
            Inches(0.6), Inches(0.2), Inches(12.2), Inches(0.7),
            font_size=27, bold=True, color=WHITE)
add_rect(slide, Inches(0.6), Inches(0.95), Inches(11.9), Inches(0.04), ACCENT_TEAL)

add_textbox(slide,
            "Baseline threshold: conf=0.05, attr=0.10 | Current clinical threshold: conf=0.15, attr=0.08",
            Inches(0.6), Inches(1.1), Inches(12.1), Inches(0.4),
            font_size=11.5, color=ACCENT_TEAL)

cmp_headers = ["Metric", "Baseline", "Current", "Delta"]
cmp_rows = [
    ("Tooth Recall", "89.75%", "75.03%", "-14.72 pp"),
    ("Tooth Precision", "78.29%", "92.38%", "+14.09 pp"),
    ("Tooth F1", "83.63%", "82.81%", "-0.82 pp"),
    ("Tooth False Positives", "872", "217", "-655 (75.1% fewer)"),
    ("Disease Recall", "70.84%", "69.24%", "-1.60 pp"),
    ("Disease Precision", "57.37%", "53.34%", "-4.03 pp"),
    ("Disease F1", "63.40%", "60.26%", "-3.14 pp"),
    ("Balanced F1", "73.51%", "71.53%", "-1.98 pp"),
]

cmp_cols = [
    (Inches(0.6), Inches(4.3)),
    (Inches(4.95), Inches(2.35)),
    (Inches(7.35), Inches(2.35)),
    (Inches(9.75), Inches(2.75)),
]
for h, (cx, cw) in zip(cmp_headers, cmp_cols):
    add_rect(slide, cx, Inches(1.6), cw, Inches(0.45), MID_BLUE)
    add_textbox(slide, h, cx + Inches(0.05), Inches(1.63), cw - Inches(0.1), Inches(0.35),
                font_size=11, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

for i, row in enumerate(cmp_rows):
    row_bg = RGBColor(0x1B, 0x3F, 0x6D) if i % 2 == 0 else RGBColor(0x15, 0x36, 0x5D)
    for j, (cell, (cx, cw)) in enumerate(zip(row, cmp_cols)):
        add_rect(slide, cx, Inches(2.05 + i * 0.53), cw, Inches(0.53), row_bg,
                 line_color=RGBColor(0x44, 0x66, 0x88), line_width=0.5)
        col = ACCENT_TEAL if (j == 3 and ("+" in cell or "fewer" in cell)) else WHITE
        add_textbox(slide, cell, cx + Inches(0.05), Inches(2.09 + i * 0.53), cw - Inches(0.1), Inches(0.4),
                    font_size=10.5, color=col, bold=(j == 3),
                    align=PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER)

add_textbox(slide,
            "Final interpretation: current threshold setting prioritizes clinical trust by sharply reducing tooth false alarms while keeping disease recall near baseline.",
            Inches(0.6), Inches(6.45), Inches(12.0), Inches(0.52),
            font_size=12, bold=True, color=ACCENT_TEAL)

add_rect(slide, Inches(0), Inches(7.1), SLIDE_W, Inches(0.4), MID_BLUE)
add_textbox(slide,
            "ARCHON Thesis 2026 | Threshold-validated deployment recommendation: conf=0.15, attr=0.08",
            Inches(0.3), Inches(7.15), Inches(12.7), Inches(0.3),
            font_size=10, color=WHITE, align=PP_ALIGN.CENTER)


# ══════════════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════════════
prs.save(str(OUT_PPT))
print(f"\n✓  Saved: {OUT_PPT}")
print(f"   Total slides: {len(prs.slides)}")
