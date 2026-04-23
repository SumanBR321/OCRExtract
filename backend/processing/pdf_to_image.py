"""
pdf_to_image.py — Convert a PDF file into a list of PIL Images.
Uses pdf2image (poppler wrapper) at 300 DPI for best OCR quality.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from PIL import Image

from backend.config import settings
from backend.utils.logger import get_logger

logger = get_logger("pdf_to_image")

# Target DPI — 300 is the sweet spot for OCR accuracy on scanned docs
DEFAULT_DPI = 300


def pdf_to_images(pdf_path: str | Path, dpi: int = DEFAULT_DPI) -> List[Image.Image]:
    """
    Convert every page of *pdf_path* into a PIL Image.

    Parameters
    ----------
    pdf_path : str | Path
        Absolute path to the PDF file.
    dpi : int
        Rendering resolution (default 300).

    Returns
    -------
    List[Image.Image]
        One Image per page.

    Raises
    ------
    FileNotFoundError
        If *pdf_path* does not exist.
    RuntimeError
        If conversion fails (e.g. Poppler not installed).
    """
    from pdf2image import convert_from_path
    from pdf2image.exceptions import (
        PDFInfoNotInstalledError,
        PDFPageCountError,
        PDFSyntaxError,
    )

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    logger.info("Converting '%s' → images @ %d DPI …", pdf_path.name, dpi)

    try:
        images = convert_from_path(
            str(pdf_path),
            dpi=dpi,
            fmt="png",
            thread_count=2,          # parallel page rendering
            use_cropbox=True,
            strict=False,            # tolerate slightly malformed PDFs
            poppler_path=settings.poppler_path,
        )
    except PDFInfoNotInstalledError as exc:
        raise RuntimeError(
            "Poppler is not installed or not on PATH. "
            "Install it via: https://github.com/oschwartz10612/poppler-windows"
        ) from exc
    except (PDFPageCountError, PDFSyntaxError) as exc:
        raise RuntimeError(f"Could not read PDF '{pdf_path.name}': {exc}") from exc

    logger.info("  → %d page(s) extracted from '%s'", len(images), pdf_path.name)
    return images
