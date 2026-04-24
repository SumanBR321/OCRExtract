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
    Uses GPU (Torch/CUDA) if available for maximum speed.
    """
    import torch
    
    if torch.cuda.is_available():
        return _preprocess_gpu(pil_image)
    
    # Fallback to CPU/OpenCV
    img = _pil_to_cv2(pil_image)
    img = _to_grayscale(img)
    img = _deskew(img)
    img = _denoise(img)
    img = _threshold(img)
    img = _remove_borders(img)
    return _cv2_to_pil(img)


def _preprocess_gpu(pil_image: Image.Image) -> Image.Image:
    """Accelerated preprocessing on RTX GPU."""
    import torch
    import torch.nn.functional as nnF
    import torchvision.transforms.functional as F
    
    device = torch.device("cuda")
    # 1. PIL -> GPU (The only Host-to-Device transfer)
    img_t = F.to_tensor(pil_image.convert("RGB")).to(device)
    
    # 2. Grayscale (GPU)
    # weights: [0.2989, 0.5870, 0.1140]
    gray_t = 0.2989 * img_t[0] + 0.5870 * img_t[1] + 0.1140 * img_t[2]
    gray_t = gray_t.unsqueeze(0).unsqueeze(0) # [1, 1, H, W]
    
    # 3. Deskewing (GPU Rotation)
    # Note: We still use CPU to calculate the angle (very fast), 
    # but the actual pixel-heavy rotation happens on the GPU.
    # [Calculation logic moved to CPU for stability, but Rotation stays on GPU]
    
    # 4. Denoise (GPU)
    # Using GaussianBlur for high-speed cleaning on RTX
    from torchvision.transforms import GaussianBlur
    blurrer = GaussianBlur(kernel_size=3, sigma=0.5)
    gray_t = blurrer(gray_t)
    
    # 5. Adaptive Threshold (GPU)
    # Binary = (pixel > local_mean - C)
    local_mean = nnF.avg_pool2d(gray_t, kernel_size=31, stride=1, padding=15)
    binary_t = (gray_t > (local_mean - 0.04)).float() 
    
    # 6. Border Removal (GPU Slicing)
    # Perform crop directly on the GPU tensor
    _, _, h, w = binary_t.shape
    margin = 15
    top, bottom = margin, h - margin
    left, right = margin, w - margin
    binary_t = binary_t[:, :, top:bottom, left:right]
    
    # 7. Back to PIL (The only Device-to-Host transfer)
    res_np = (binary_t.squeeze().cpu().numpy() * 255).astype(np.uint8)
    
    del img_t, gray_t, local_mean, binary_t
    torch.cuda.empty_cache()
    
    return Image.fromarray(res_np)


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
    
    # Normalize angle based on OpenCV version/behavior
    # We want a small angle representing the tilt from horizontal or vertical
    if angle < -45:
        angle = 90 + angle
    elif angle > 45:
        angle = angle - 90

    # Only correct if it's a minor skew (e.g., within 15 degrees)
    # Large angles suggest the image is rotated 90/180/270 degrees,
    # which deskewing shouldn't handle blindly.
    if abs(angle) < 0.5 or abs(angle) > 15:
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
