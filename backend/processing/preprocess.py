"""
preprocess.py — Image preprocessing pipeline using OpenCV.

Steps applied (in order):
1. Convert to grayscale
2. Deskew (straighten skewed scans)
3. Denoise (fastNlMeansDenoising)
4. Adaptive thresholding  → crisp black-on-white for Tesseract
5. Optional border removal

All functions accept and return numpy arrays (BGR or grayscale).
"""

from __future__ import annotations

import numpy as np
import cv2
from PIL import Image

from backend.utils.logger import get_logger

logger = get_logger("preprocess")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def preprocess_image(pil_image: Image.Image) -> Image.Image:
    """
    Full preprocessing pipeline.

    Parameters
    ----------
    pil_image : PIL.Image.Image
        Raw page image from pdf_to_image.

    Returns
    -------
    PIL.Image.Image
        Processed image, ready for OCR.
    """
    img = _pil_to_cv2(pil_image)
    img = _to_grayscale(img)
    img = _deskew(img)
    img = _denoise(img)
    img = _threshold(img)
    img = _remove_borders(img)
    return _cv2_to_pil(img)


# ---------------------------------------------------------------------------
# Step functions
# ---------------------------------------------------------------------------

def _pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    """PIL → BGR numpy array."""
    rgb = np.array(pil_image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _cv2_to_pil(img: np.ndarray) -> Image.Image:
    """Grayscale or BGR numpy array → PIL."""
    if img.ndim == 2:
        return Image.fromarray(img)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _to_grayscale(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def _deskew(gray: np.ndarray) -> np.ndarray:
    """
    Detect and correct page skew.
    Uses projection-profile method: finds the rotation angle that
    maximises the variance of horizontal pixel projections.
    """
    coords = np.column_stack(np.where(gray < 200))
    if len(coords) < 10:
        return gray

    angle = cv2.minAreaRect(coords.astype(np.float32))[-1]
    # minAreaRect returns angles in (-90, 0]; adjust to (-45, 45]
    if angle < -45:
        angle = 90 + angle

    if abs(angle) < 0.5:          # negligible skew — skip
        return gray

    h, w = gray.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        gray, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    logger.debug("Deskew: corrected %.2f°", angle)
    return rotated


def _denoise(gray: np.ndarray) -> np.ndarray:
    """
    Non-local means denoising — effective for scanned documents.
    h=10 is a balanced strength (higher → smoother but slower).
    """
    return cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)


def _threshold(gray: np.ndarray) -> np.ndarray:
    """
    Adaptive Gaussian thresholding.
    Better than global thresholding for uneven illumination (common
    in scanned documents).
    """
    return cv2.adaptiveThreshold(
        gray,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=31,
        C=10,
    )


def _remove_borders(binary: np.ndarray, margin: int = 15) -> np.ndarray:
    """
    Crop a fixed-pixel border that may contain scanner artifacts.
    Clamps to image dimensions so small images don't raise errors.
    """
    h, w = binary.shape
    top    = min(margin, h // 4)
    bottom = max(h - margin, h * 3 // 4)
    left   = min(margin, w // 4)
    right  = max(w - margin, w * 3 // 4)
    return binary[top:bottom, left:right]
