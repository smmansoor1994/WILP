"""
src/data/download.py
====================
Download the Dentex Challenge 2023 dataset from Kaggle.

Dataset source:
  https://www.kaggle.com/competitions/dentex-challenge-2023

Requirements:
  - kaggle Python package:  pip install kaggle
  - Kaggle API credentials placed at:
      Windows:  C:\\Users\\<user>\\.kaggle\\kaggle.json
      Linux:    ~/.kaggle/kaggle.json
  - Format of kaggle.json:
      {"username": "YOUR_USERNAME", "key": "YOUR_API_KEY"}

How to get credentials:
  1. Go to https://www.kaggle.com → Account → Create New API Token
  2. Download kaggle.json and place at the path above

Alternatively, set environment variables:
  KAGGLE_USERNAME and KAGGLE_KEY
"""

import os
import zipfile
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

# Kaggle competition identifier
COMPETITION_NAME = "dentex-challenge-2023"

# Expected sub-directories after extraction
EXPECTED_DIRS = [
    "quadrant",
    "quadrant_enumeration",
    "quadrant_enumeration_disease",
    "test",
]


# ─── Main Download Function ───────────────────────────────────────────────────

def download_dentex_dataset(
    raw_dir: Path = Path("data/raw"),
    competition: str = COMPETITION_NAME,
    overwrite: bool = False,
) -> None:
    """Download and extract the Dentex Challenge 2023 dataset from Kaggle.

    Args:
        raw_dir:     Directory to store raw downloaded data.
        competition: Kaggle competition identifier.
        overwrite:   Re-download even if data already present.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    # ── Check if already downloaded ──────────────────────────────────────────
    already_present = all((raw_dir / d).exists() for d in EXPECTED_DIRS)
    if already_present and not overwrite:
        logger.info(
            "Dataset already present at '%s'. "
            "Pass overwrite=True to re-download.",
            raw_dir,
        )
        return

    # ── Verify Kaggle credentials ─────────────────────────────────────────────
    _check_kaggle_credentials()

    # ── Import kaggle (deferred to avoid import error if not installed) ───────
    try:
        import kaggle  # noqa: F401
    except ImportError:
        raise ImportError(
            "The 'kaggle' package is required.\n"
            "Install it with:  pip install kaggle\n"
            "Then set up API credentials as described in this file's docstring."
        )

    # ── Download via Kaggle API ───────────────────────────────────────────────
    logger.info("Downloading '%s' from Kaggle to '%s' ...", competition, raw_dir)

    # kaggle competitions download -c dentex-challenge-2023 -p data/raw
    import kaggle.api as kapi

    kapi.authenticate()
    kapi.competition_download_files(
        competition=competition,
        path=str(raw_dir),
        force=overwrite,
        quiet=False,
    )

    # ── Extract zip files ────────────────────────────────────────────────────
    zip_files = list(raw_dir.glob("*.zip"))
    if not zip_files:
        logger.warning(
            "No .zip files found in '%s' after download. "
            "Check that the competition download succeeded.",
            raw_dir,
        )
        return

    for zf in zip_files:
        logger.info("Extracting '%s' ...", zf.name)
        with zipfile.ZipFile(zf, "r") as z:
            z.extractall(raw_dir)
        # Remove zip after extraction to save disk space
        zf.unlink()
        logger.info("Extracted and removed '%s'.", zf.name)

    # ── Flatten nested directories if needed ─────────────────────────────────
    _flatten_extracted_dirs(raw_dir)

    logger.info("Dataset download complete. Files in '%s':", raw_dir)
    for item in sorted(raw_dir.iterdir()):
        logger.info("  %s", item.name)


def _check_kaggle_credentials() -> None:
    """Validate that Kaggle API credentials are available.

    Checks for:
    1. Environment variables KAGGLE_USERNAME + KAGGLE_KEY
    2. ~/.kaggle/kaggle.json file
    """
    # Option 1: environment variables
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        logger.info("Using Kaggle credentials from environment variables.")
        return

    # Option 2: kaggle.json file
    kaggle_json_paths = [
        Path.home() / ".kaggle" / "kaggle.json",
        Path(os.environ.get("KAGGLE_CONFIG_DIR", "~/.kaggle")) / "kaggle.json",
    ]
    for p in kaggle_json_paths:
        if p.expanduser().exists():
            logger.info("Using Kaggle credentials from '%s'.", p)
            return

    raise FileNotFoundError(
        "Kaggle API credentials not found.\n"
        "Setup steps:\n"
        "  1. Visit https://www.kaggle.com → Account → Create New API Token\n"
        "  2. Download 'kaggle.json'\n"
        "  3. Place it at:  C:\\Users\\<YourUser>\\.kaggle\\kaggle.json\n"
        "  4. Set file permissions (Linux only): chmod 600 ~/.kaggle/kaggle.json\n"
        "\nAlternatively set environment variables:\n"
        "  KAGGLE_USERNAME=<your_username>\n"
        "  KAGGLE_KEY=<your_api_key>"
    )


def _flatten_extracted_dirs(raw_dir: Path) -> None:
    """Move dataset files from a nested sub-directory to raw_dir if needed.

    Some Kaggle downloads extract into a sub-folder named after the competition.
    This function flattens that one extra level.
    """
    # Look for a single sub-directory that contains all expected dirs
    subdirs = [d for d in raw_dir.iterdir() if d.is_dir()]
    if len(subdirs) == 1:
        inner = subdirs[0]
        inner_expected = [inner / d for d in EXPECTED_DIRS]
        if all(p.exists() for p in inner_expected):
            logger.info(
                "Flattening nested directory: '%s' → '%s'", inner, raw_dir
            )
            for item in inner.iterdir():
                dest = raw_dir / item.name
                if not dest.exists():
                    shutil.move(str(item), str(dest))
            inner.rmdir()
