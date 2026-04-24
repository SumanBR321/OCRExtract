"""
ocr_engine.py — Parallelised Tesseract OCR wrapper.
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Optional
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import pytesseract
import numpy as np
from backend.utils.logger import get_logger

logger = get_logger("ocr_engine")

# Lazy-loaded EasyOCR reader
_reader = None

def _get_easyocr_reader():
    global _reader
    if _reader is None:
        try:
            import easyocr
            import torch
            use_gpu = torch.cuda.is_available()
            if use_gpu:
                logger.info("🚀 CUDA detected! Using RTX 3050 (GPU) for OCR.")
            else:
                logger.info("⚠️ CUDA not found. Using EasyOCR on CPU.")
            
            _reader = easyocr.Reader(['en'], gpu=use_gpu)
        except ImportError:
            logger.warning("easyocr or torch not installed. Falling back to Tesseract.")
            return None
    return _reader

# Tesseract config (fallback)
TESS_CONFIG = "--oem 3 --psm 1 -l eng"
PAGE_SEPARATOR = "\n\n--- PAGE BREAK ---\n\n"

def run_ocr_on_images(images: List[Image.Image], tesseract_cmd: Optional[str] = None) -> str:
    """
    Run OCR on a list of images.
    Uses EasyOCR (GPU-accelerated) by default, falls back to Tesseract.
    """
    reader = _get_easyocr_reader()
    
    # --- Strategy A: EasyOCR (GPU) ---
    if reader:
        try:
            logger.info("  🔍 OCR starting using EasyOCR...")
            results = []
            for i, img in enumerate(images, start=1):
                # Convert PIL to numpy array for EasyOCR
                img_np = np.array(img)
                # detail=0 returns just the text strings
                text_parts = reader.readtext(img_np, detail=0)
                results.append(" ".join(text_parts))
                logger.debug("    Processed page %d/%d", i, len(images))
                
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            return PAGE_SEPARATOR.join(results)
        except Exception as e:
            logger.error("EasyOCR failed, trying Tesseract fallback: %s", e)

    # --- Strategy B: Tesseract (CPU) ---
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    def ocr_single(img_tuple):
        page_num, img = img_tuple
        try:
            return pytesseract.image_to_string(img, config=TESS_CONFIG).strip()
        except Exception as exc:
            logger.warning("  Tesseract failed on page %d: %s", page_num, exc)
            return ""

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(ocr_single, enumerate(images, start=1)))

    return PAGE_SEPARATOR.join(results)

def run_ocr_on_file(pdf_path: str | Path, tesseract_cmd: Optional[str] = None) -> str:
    from backend.processing.pdf_to_image import pdf_to_images
    from backend.processing.preprocess import preprocess_image

    images = pdf_to_images(pdf_path)
    processed = [preprocess_image(img) for img in images]
    return run_ocr_on_images(processed, tesseract_cmd=tesseract_cmd)
