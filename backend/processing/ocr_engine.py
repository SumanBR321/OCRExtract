"""
ocr_engine.py — Parallelised Tesseract OCR wrapper.
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Optional
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import pytesseract
from backend.utils.logger import get_logger

logger = get_logger("ocr_engine")

# Tesseract config optimized for printed academic text
TESS_CONFIG = "--oem 3 --psm 1 -l eng"
PAGE_SEPARATOR = "\n\n--- PAGE BREAK ---\n\n"

def run_ocr_on_images(images: List[Image.Image], tesseract_cmd: Optional[str] = None) -> str:
    """
    Run Tesseract OCR on a list of images in parallel using 8 workers.
    """
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    def ocr_single(img_tuple):
        page_num, img = img_tuple
        logger.debug("  OCR page %d / %d starting...", page_num, len(images))
        try:
            return pytesseract.image_to_string(img, config=TESS_CONFIG).strip()
        except Exception as exc:
            logger.warning("  OCR failed on page %d: %s", page_num, exc)
            return ""

    # Use 8 workers to process pages simultaneously
    # ThreadPoolExecutor is best here because we're calling an external .exe
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(ocr_single, enumerate(images, start=1)))

    combined = PAGE_SEPARATOR.join(results)
    logger.info("OCR complete — %d chars extracted across %d page(s).",
                len(combined), len(images))
    return combined

def run_ocr_on_file(pdf_path: str | Path, tesseract_cmd: Optional[str] = None) -> str:
    from backend.processing.pdf_to_image import pdf_to_images
    from backend.processing.preprocess import preprocess_image

    images = pdf_to_images(pdf_path)
    processed = [preprocess_image(img) for img in images]
    return run_ocr_on_images(processed, tesseract_cmd=tesseract_cmd)
