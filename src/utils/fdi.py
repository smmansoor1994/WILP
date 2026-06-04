"""
src/utils/fdi.py
================
FDI (Fédération Dentaire Internationale) Tooth Numbering System Utilities.

The FDI system assigns a 2-digit code to each tooth:
  - First digit: quadrant  (1=upper-right, 2=upper-left, 3=lower-left, 4=lower-right)
  - Second digit: position (1=central incisor ... 8=wisdom tooth)

Valid FDI numbers: 11-18, 21-28, 31-38, 41-48  (32 teeth total)

YOLO class mapping (0-indexed):
  FDI 11→0, 12→1, ..., 18→7
  FDI 21→8, 22→9, ..., 28→15
  FDI 31→16, 32→17, ..., 38→23
  FDI 41→24, 42→25, ..., 48→31

Reference: https://www.fdiworlddental.org/
"""

from typing import Tuple, List, Optional


# ─── Constants ────────────────────────────────────────────────────────────────

QUADRANT_NAMES = {
    1: "Upper Right",
    2: "Upper Left",
    3: "Lower Left",
    4: "Lower Right",
}

POSITION_NAMES = {
    1: "Central Incisor",
    2: "Lateral Incisor",
    3: "Canine",
    4: "First Premolar",
    5: "Second Premolar",
    6: "First Molar",
    7: "Second Molar",
    8: "Wisdom Tooth (Third Molar)",
}

# All valid FDI numbers in canonical order
ALL_FDI = [q * 10 + p for q in range(1, 5) for p in range(1, 9)]  # 32 teeth

# Mapping: FDI number → YOLO class index (0-31)
FDI_TO_CLASS = {fdi: idx for idx, fdi in enumerate(ALL_FDI)}

# Mapping: YOLO class index → FDI number
CLASS_TO_FDI = {idx: fdi for fdi, idx in FDI_TO_CLASS.items()}

# Mapping: quadrant number → list of YOLO class indices
QUADRANT_TO_CLASSES = {
    1: list(range(0, 8)),    # FDI 11-18
    2: list(range(8, 16)),   # FDI 21-28
    3: list(range(16, 24)),  # FDI 31-38
    4: list(range(24, 32)),  # FDI 41-48
}

# Horizontal flip remaps quadrant pairs: Q1↔Q2, Q3↔Q4
# (when image is flipped left-right, upper-right becomes upper-left, etc.)
FLIP_QUADRANT_MAP = {1: 2, 2: 1, 3: 4, 4: 3}


# ─── Conversion Functions ─────────────────────────────────────────────────────

def fdi_to_class(fdi: int) -> int:
    """Convert FDI tooth number to YOLO class index (0-31).

    Args:
        fdi: FDI number e.g. 11, 28, 46

    Returns:
        YOLO class index 0-31

    Raises:
        ValueError: if FDI number is not valid
    """
    if fdi not in FDI_TO_CLASS:
        raise ValueError(
            f"Invalid FDI number: {fdi}. "
            f"Valid range: 11-18, 21-28, 31-38, 41-48"
        )
    return FDI_TO_CLASS[fdi]


def class_to_fdi(class_idx: int) -> int:
    """Convert YOLO class index to FDI tooth number.

    Args:
        class_idx: YOLO class index 0-31

    Returns:
        FDI number (e.g. 11, 28, 46)
    """
    if class_idx not in CLASS_TO_FDI:
        raise ValueError(f"Invalid class index: {class_idx}. Valid: 0-31")
    return CLASS_TO_FDI[class_idx]


def quadrant_enum_to_fdi(quadrant: int, enumeration: int) -> int:
    """Convert quadrant + enumeration to FDI number.

    Args:
        quadrant:    Quadrant number 1-4
        enumeration: Tooth position within quadrant 1-8

    Returns:
        FDI number (e.g. quadrant=2, enumeration=3 → FDI 23)
    """
    if quadrant not in range(1, 5):
        raise ValueError(f"Quadrant must be 1-4, got {quadrant}")
    if enumeration not in range(1, 9):
        raise ValueError(f"Enumeration must be 1-8, got {enumeration}")
    return quadrant * 10 + enumeration


def fdi_to_quadrant_enum(fdi: int) -> Tuple[int, int]:
    """Decompose FDI number into (quadrant, enumeration).

    Args:
        fdi: FDI number e.g. 23

    Returns:
        (quadrant, enumeration) e.g. (2, 3)
    """
    quadrant = fdi // 10
    enumeration = fdi % 10
    return quadrant, enumeration


def fdi_to_name(fdi: int) -> str:
    """Get human-readable tooth name from FDI number.

    Example: fdi_to_name(23) → 'Upper Left Canine'
    """
    quadrant, position = fdi_to_quadrant_enum(fdi)
    return f"{QUADRANT_NAMES.get(quadrant, '?')} {POSITION_NAMES.get(position, '?')}"


def flip_fdi(fdi: int) -> int:
    """Return the mirrored FDI number when panoramic X-ray is flipped horizontally.

    When flipping left-right:
      Q1 (upper-right) ↔ Q2 (upper-left)
      Q3 (lower-left)  ↔ Q4 (lower-right)

    Args:
        fdi: Original FDI number

    Returns:
        FDI number after horizontal flip
    """
    quadrant, enumeration = fdi_to_quadrant_enum(fdi)
    new_quadrant = FLIP_QUADRANT_MAP[quadrant]
    return quadrant_enum_to_fdi(new_quadrant, enumeration)


def flip_class(class_idx: int) -> int:
    """Return the mirrored YOLO class index for horizontal flip.

    Used in flip augmentation to preserve correct tooth labels.
    """
    fdi = class_to_fdi(class_idx)
    flipped_fdi = flip_fdi(fdi)
    return fdi_to_class(flipped_fdi)


def get_class_names() -> List[str]:
    """Return list of class names in YOLO order (index 0-31).

    Returns:
        List of FDI strings: ['11', '12', ..., '48']
    """
    return [str(CLASS_TO_FDI[i]) for i in range(32)]


def get_quadrant_from_class(class_idx: int) -> int:
    """Get quadrant (1-4) from YOLO class index."""
    fdi = class_to_fdi(class_idx)
    return fdi // 10


# ─── Flip Mapping Table ───────────────────────────────────────────────────────

# Pre-computed flip mapping for all 32 YOLO classes
# flip_class_table[class_idx] = flipped_class_idx
FLIP_CLASS_TABLE = [flip_class(i) for i in range(32)]
