"""Face detection + embedding backend.

Tries InsightFace (buffalo_l — 512-d ArcFace embeddings) first; on import or
init failure, falls back to ``face_recognition`` (128-d dlib embeddings). The
active backend is reported by ``backend_name()`` and embeddings are always
returned as float32 numpy arrays — dimension is stable per backend.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .config import settings
from .preprocessing import (
    face_quality_score,
    preprocess_for_detection,
    upscale_crop_if_small,
)

log = logging.getLogger(__name__)


@dataclass
class DetectedFace:
    bbox: tuple[int, int, int, int]  # x, y, w, h
    embedding: np.ndarray
    crop_bgr: np.ndarray
    quality: float


class _BaseEngine:
    name = "none"
    embedding_dim = 0

    def detect(self, bgr: np.ndarray) -> List[DetectedFace]:  # pragma: no cover
        raise NotImplementedError


class _InsightFaceEngine(_BaseEngine):
    name = "insightface"
    embedding_dim = 512

    def __init__(self) -> None:
        from insightface.app import FaceAnalysis  # type: ignore

        self.app = FaceAnalysis(name=settings.face_model, providers=["CPUExecutionProvider"])
        self.app.prepare(ctx_id=-1, det_size=(640, 640))

    def detect(self, bgr: np.ndarray) -> List[DetectedFace]:
        prepped = preprocess_for_detection(bgr)
        results = self.app.get(prepped)
        out: List[DetectedFace] = []
        h_img, w_img = bgr.shape[:2]
        for f in results:
            x1, y1, x2, y2 = [int(v) for v in f.bbox]
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(w_img, x2); y2 = min(h_img, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = bgr[y1:y2, x1:x2].copy()
            quality = face_quality_score(crop)
            # Embedding is L2-normalized already for buffalo_l, but normalize to be safe.
            emb = np.asarray(f.normed_embedding, dtype=np.float32)
            out.append(
                DetectedFace(
                    bbox=(x1, y1, x2 - x1, y2 - y1),
                    embedding=emb,
                    crop_bgr=crop,
                    quality=quality,
                )
            )
        return out


class _FaceRecognitionEngine(_BaseEngine):
    name = "face_recognition"
    embedding_dim = 128

    def __init__(self) -> None:
        import face_recognition  # type: ignore

        self._fr = face_recognition

    def detect(self, bgr: np.ndarray) -> List[DetectedFace]:
        rgb = bgr[:, :, ::-1]
        locations = self._fr.face_locations(rgb, model="hog")
        if not locations:
            return []
        encodings = self._fr.face_encodings(rgb, known_face_locations=locations, num_jitters=1)
        out: List[DetectedFace] = []
        for (top, right, bottom, left), enc in zip(locations, encodings):
            crop = bgr[top:bottom, left:right].copy()
            crop = upscale_crop_if_small(crop)
            quality = face_quality_score(crop)
            emb = np.asarray(enc, dtype=np.float32)
            out.append(
                DetectedFace(
                    bbox=(left, top, right - left, bottom - top),
                    embedding=emb,
                    crop_bgr=crop,
                    quality=quality,
                )
            )
        return out


_engine_lock = threading.Lock()
_engine: Optional[_BaseEngine] = None
_engine_error: Optional[str] = None


def _build_engine() -> _BaseEngine:
    preferred = settings.face_backend.lower()
    attempts: list[str] = []
    if preferred == "face_recognition":
        attempts = ["face_recognition", "insightface"]
    else:
        attempts = ["insightface", "face_recognition"]

    last_err: Optional[Exception] = None
    for candidate in attempts:
        try:
            if candidate == "insightface":
                return _InsightFaceEngine()
            if candidate == "face_recognition":
                return _FaceRecognitionEngine()
        except Exception as exc:
            last_err = exc
            log.warning("Face backend %s failed to init: %s", candidate, exc)
    raise RuntimeError(f"No face recognition backend available: {last_err}")


def get_engine() -> _BaseEngine:
    global _engine, _engine_error
    if _engine is not None:
        return _engine
    with _engine_lock:
        if _engine is None:
            try:
                _engine = _build_engine()
                _engine_error = None
            except Exception as exc:
                _engine_error = str(exc)
                raise
    return _engine


def engine_status() -> dict:
    return {
        "active": _engine.name if _engine else None,
        "error": _engine_error,
    }


def backend_name() -> str:
    return _engine.name if _engine else settings.face_backend
