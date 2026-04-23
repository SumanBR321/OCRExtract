"""
ocr_engine.py — Tesseract OCR wrapper.

Runs pytesseract on each preprocessed page image and combines the
results into a single string for downstream extraction.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from PIL import Image

from backend.utils.logger import get_logger

logger = get_logger("ocr_engine")

# Tesseract config optimised for printed academic text
# --oem 3  → LSTM + legacy (best accuracy)
# --psm 1  → Automatic page segmentation with OSD
TESS_CONFIG = "--oem 3 --psm 1 -l eng"

# Separator inserted between pages in the combined string
PAGE_SEPARATOR = "\n\n--- PAGE BREAK ---\n\n"


def run_ocr_on_images(images: List[Image.Image], tesseract_cmd: Optional[str] = None) -> str:
    """
    Run Tesseract OCR on a list of PIL images and combine the output.

    Parameters
    ----------
    images : List[Image.Image]
        Preprocessed page images.
    tesseract_cmd : str | None
        Optional path to the tesseract binary (e.g. on Windows).
        If None, pytesseract uses whatever is on PATH.

    Returns
    -------
    str
        Combined OCR text from all pages.
    """
    import pytesseract

    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    page_texts: List[str] = []
    for page_num, img in enumerate(images, start=1):
        logger.debug("  OCR page %d / %d …", page_num, len(images))
        try:
            text = pytesseract.image_to_string(img, config=TESS_CONFIG)
            page_texts.append(text.strip())
        except Exception as exc:
            logger.warning("  OCR failed on page %d: %s", page_num, exc)
            page_texts.append("")

    combined = PAGE_SEPARATOR.join(page_texts)
    logger.info("OCR complete — %d chars extracted across %d page(s).",
                len(combined), len(images))
    return combined


def run_ocr_on_file(pdf_path: str | Path, tesseract_cmd: Optional[str] = None) -> str:
    """
    Convenience wrapper: preprocess PDF pages inline and return OCR text.
    Useful for lightweight local testing without the full pipeline.
    """
    from backend.processing.pdf_to_image import pdf_to_images
    from backend.processing.preprocess import preprocess_image

    images = pdf_to_images(pdf_path)
    processed = [preprocess_image(img) for img in images]
    return run_ocr_on_images(processed, tesseract_cmd=tesseract_cmd)
