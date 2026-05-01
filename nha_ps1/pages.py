"""PDF/image splitting + per-page quality estimation.

Fixes the missing-import bug from full(1).py (fitz, PIL, io were never imported).
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, List

import fitz  # PyMuPDF
import numpy as np
from PIL import Image

try:
    import cv2  # type: ignore
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

from .config import BLUR_THRESHOLD, PDF_RENDER_ZOOM


def extract_pages(file_path: Path) -> List[Dict[str, Any]]:
    """Split a PDF into per-page PIL images, or wrap a single image as page 1.

    Returns: [{"page_number": int, "image": PIL.Image, "file_name": str}, ...]
    Multi-page PDFs preserve sequential page_number (HI.txt §A.1).
    """
    ext = file_path.suffix.lower()
    file_name = file_path.name
    pages: List[Dict[str, Any]] = []

    if ext == ".pdf":
        try:
            doc = fitz.open(str(file_path))
            for page_idx in range(len(doc)):
                page = doc[page_idx]
                mat = fitz.Matrix(PDF_RENDER_ZOOM, PDF_RENDER_ZOOM)
                pix = page.get_pixmap(matrix=mat)
                img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
                pages.append({
                    "page_number": page_idx + 1,
                    "image": img,
                    "file_name": file_name,
                })
            doc.close()
        except Exception as e:
            print(f"[pages] PDF extract failed for {file_name}: {e}")
    elif ext in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
        try:
            img = Image.open(file_path)
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            pages.append({
                "page_number": 1,
                "image": img,
                "file_name": file_name,
            })
        except Exception as e:
            print(f"[pages] image load failed for {file_name}: {e}")

    return pages


def estimate_page_quality(image: Image.Image, text: str = "") -> Dict[str, Any]:
    """Lightweight quality heuristics — used for MG006A's `poor_quality` field.

    Combines a Laplacian blur score (cv2 when available, np fallback) with
    text-density. Returns is_blurry/is_poor_quality booleans.
    """
    arr = np.array(image.convert("L"))

    if _HAS_CV2:
        blur_score = float(cv2.Laplacian(arr, cv2.CV_64F).var())
    else:
        # Pure-NumPy Laplacian fallback (3x3 discrete Laplacian).
        a = arr.astype(np.float32)
        lap = (
            -4 * a[1:-1, 1:-1]
            + a[:-2, 1:-1] + a[2:, 1:-1]
            + a[1:-1, :-2] + a[1:-1, 2:]
        )
        blur_score = float(lap.var())

    text_len = len(text.strip())
    is_blurry = blur_score < BLUR_THRESHOLD
    low_text = text_len < 40

    return {
        "blur_score": blur_score,
        "text_density": text_len,
        "is_blurry": bool(is_blurry),
        "is_poor_quality": bool(is_blurry or low_text),
    }
