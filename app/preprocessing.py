from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageOps

try:  # OpenCV is optional at import time to make the module importable in minimal envs.
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

from .config import settings


def load_image_bgr(path: str | Path) -> np.ndarray:
    """Load an image file into a BGR numpy array (OpenCV convention)."""
    path = Path(path)
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.array(img)
    # RGB -> BGR
    return arr[:, :, ::-1].copy()


def save_thumbnail(src_path: str | Path, dest_path: str | Path, size: Optional[int] = None) -> None:
    size = size or settings.thumbnail_size
    img = Image.open(src_path)
    img = ImageOps.exif_transpose(img)
    img.thumbnail((size, size))
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(dest_path, format="JPEG", quality=85)


def preprocess_for_detection(bgr: np.ndarray) -> np.ndarray:
    """CLAHE contrast enhancement on luminance channel for faded historical scans."""
    if cv2 is None or not settings.preprocess_clahe:
        return bgr
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    merged = cv2.merge((l, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def upscale_crop_if_small(crop_bgr: np.ndarray, min_side: Optional[int] = None) -> np.ndarray:
    if cv2 is None:
        return crop_bgr
    min_side = min_side or settings.upscale_small_face_px
    h, w = crop_bgr.shape[:2]
    short = min(h, w)
    if short >= min_side:
        return crop_bgr
    scale = max(1.5, min_side / max(short, 1))
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(crop_bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


def face_quality_score(crop_bgr: np.ndarray) -> float:
    """Crude quality proxy: Laplacian variance + size. 0..1 range (clipped)."""
    if cv2 is None:
        h, w = crop_bgr.shape[:2]
        return min(1.0, (h * w) / (160 * 160))
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    h, w = gray.shape
    size_score = min(1.0, (h * w) / (160 * 160))
    sharp_score = min(1.0, lap_var / 200.0)
    return round(0.6 * size_score + 0.4 * sharp_score, 4)


def render_pdf_first_page(pdf_path: str | Path, out_path: str | Path, dpi: int = 200) -> None:
    """Render the first page of a PDF as JPEG. Requires poppler installed at runtime."""
    from pdf2image import convert_from_path

    images = convert_from_path(str(pdf_path), dpi=dpi, first_page=1, last_page=1)
    if not images:
        raise RuntimeError("PDF rendered no pages")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    images[0].save(out_path, format="JPEG", quality=90)


def embedding_to_bytes(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def embedding_from_bytes(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
