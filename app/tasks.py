"""Background tasks: face detection + matching for newly uploaded photos."""
from __future__ import annotations

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from sqlalchemy.orm import Session

try:  # OpenCV is only required when actually processing photos.
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

from . import models
from .config import settings
from .db import session_scope
from .face_engine import DetectedFace, get_engine
from .preprocessing import (
    cosine_similarity,
    embedding_from_bytes,
    embedding_to_bytes,
    load_image_bgr,
)

log = logging.getLogger(__name__)


def _save_face_crop(crop_bgr: np.ndarray, photo_id: int) -> str:
    """Save the cropped face as a JPEG under settings.faces_dir. Returns relative filename."""
    fname = f"photo{photo_id}_{uuid.uuid4().hex[:8]}.jpg"
    out = settings.faces_dir / fname
    out.parent.mkdir(parents=True, exist_ok=True)
    if cv2 is not None:
        cv2.imwrite(str(out), crop_bgr)
    else:
        # Fallback path: write via Pillow (expects RGB).
        from PIL import Image

        rgb = crop_bgr[:, :, ::-1]
        Image.fromarray(rgb).save(out, format="JPEG", quality=90)
    return fname


def _all_seed_faces(db: Session) -> list[models.Face]:
    return (
        db.query(models.Face)
        .filter(models.Face.is_seed.is_(True), models.Face.embedding.is_not(None))
        .all()
    )


def _best_seed_match(
    embedding: np.ndarray, seeds: list[models.Face]
) -> tuple[Optional[models.Face], float]:
    best: Optional[models.Face] = None
    best_score = 0.0
    for seed in seeds:
        if seed.embedding is None:
            continue
        seed_vec = embedding_from_bytes(seed.embedding)
        if seed_vec.shape != embedding.shape:
            # Embedding dims must match — skip mismatched backends.
            continue
        score = cosine_similarity(embedding, seed_vec)
        if score > best_score:
            best_score = score
            best = seed
    return best, best_score


def process_photo(photo_id: int) -> None:
    """Detect faces in a photo, store crops + embeddings, suggest matches vs. seeds."""
    with session_scope() as db:
        photo = db.get(models.Photo, photo_id)
        if photo is None:
            log.warning("process_photo: photo %s not found", photo_id)
            return
        photo.processing_status = "processing"
        db.flush()
        filepath = photo.filepath

    try:
        engine = get_engine()
        bgr = load_image_bgr(filepath)
        detected: list[DetectedFace] = engine.detect(bgr)
    except Exception as exc:
        log.exception("process_photo: detection failed for photo %s: %s", photo_id, exc)
        with session_scope() as db:
            photo = db.get(models.Photo, photo_id)
            if photo is not None:
                photo.processing_status = "failed"
        return

    with session_scope() as db:
        photo = db.get(models.Photo, photo_id)
        if photo is None:
            return
        seeds = _all_seed_faces(db)
        for det in detected:
            crop_name = _save_face_crop(det.crop_bgr, photo_id)
            face = models.Face(
                photo_id=photo_id,
                bbox_x=det.bbox[0],
                bbox_y=det.bbox[1],
                bbox_w=det.bbox[2],
                bbox_h=det.bbox[3],
                crop_path=crop_name,
                embedding=embedding_to_bytes(det.embedding),
                quality=det.quality,
            )
            best_seed, score = _best_seed_match(det.embedding, seeds)
            if best_seed is not None and best_seed.person_id is not None:
                face.confidence = float(score)
                if score >= settings.auto_match_threshold:
                    face.person_id = best_seed.person_id
                    face.is_confirmed = True
                elif score >= settings.suggest_threshold:
                    face.person_id = best_seed.person_id
                    face.is_confirmed = False
            db.add(face)
        photo.processing_status = "done"
        photo.processed_at = datetime.utcnow()


def rematch_face(face_id: int) -> tuple[Optional[int], float]:
    """Recompute the best seed match for a single face. Returns (person_id, score)."""
    with session_scope() as db:
        face = db.get(models.Face, face_id)
        if face is None or face.embedding is None:
            return None, 0.0
        vec = embedding_from_bytes(face.embedding)
        seeds = _all_seed_faces(db)
        best, score = _best_seed_match(vec, seeds)
        pid = best.person_id if best else None
        face.confidence = float(score)
        if not face.is_confirmed:
            face.person_id = pid if score >= settings.suggest_threshold else None
        return pid, score


def rematch_all_unconfirmed() -> int:
    """Re-run matching for every unconfirmed face. Returns count updated."""
    with session_scope() as db:
        seeds = _all_seed_faces(db)
        faces = (
            db.query(models.Face)
            .filter(
                models.Face.is_confirmed.is_(False),
                models.Face.is_seed.is_(False),
                models.Face.embedding.is_not(None),
            )
            .all()
        )
        count = 0
        for face in faces:
            vec = embedding_from_bytes(face.embedding)
            best, score = _best_seed_match(vec, seeds)
            face.confidence = float(score)
            if best is not None and score >= settings.suggest_threshold and not face.is_rejected:
                face.person_id = best.person_id
            else:
                face.person_id = None
            count += 1
        return count
