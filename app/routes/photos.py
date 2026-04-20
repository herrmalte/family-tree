from __future__ import annotations

import mimetypes
import re
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.responses import RedirectResponse
from sqlalchemy import func
from sqlalchemy.orm import Session

from .. import models
from ..config import settings
from ..db import get_db
from ..preprocessing import render_pdf_first_page, save_thumbnail
from ..tasks import process_photo
from ..templating import templates

router = APIRouter()

ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".pdf"}


def _sanitize(name: str) -> str:
    name = Path(name).name  # strip any directory components
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return name[:180] or "upload"


def _decade_from_estimate(date_estimate: str | None) -> int | None:
    if not date_estimate:
        return None
    m = re.search(r"(\d{4})", date_estimate)
    if not m:
        return None
    year = int(m.group(1))
    return (year // 10) * 10


@router.get("/")
def home(request: Request, db: Session = Depends(get_db)):
    total_photos = db.query(func.count(models.Photo.id)).scalar() or 0
    total_faces = db.query(func.count(models.Face.id)).scalar() or 0
    total_people = db.query(func.count(models.Person.id)).scalar() or 0
    unidentified = (
        db.query(func.count(models.Face.id))
        .filter(
            models.Face.person_id.is_(None),
            models.Face.is_seed.is_(False),
            models.Face.is_rejected.is_(False),
        )
        .scalar()
        or 0
    )
    suggested = (
        db.query(func.count(models.Face.id))
        .filter(
            models.Face.person_id.is_not(None),
            models.Face.is_confirmed.is_(False),
            models.Face.is_seed.is_(False),
        )
        .scalar()
        or 0
    )
    recent = (
        db.query(models.Photo)
        .order_by(models.Photo.uploaded_at.desc())
        .limit(6)
        .all()
    )
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "totals": {
                "photos": total_photos,
                "faces": total_faces,
                "people": total_people,
                "unidentified": unidentified,
                "suggested": suggested,
            },
            "recent": recent,
        },
    )


@router.get("/upload")
def upload_form(request: Request, db: Session = Depends(get_db)):
    albums = [a for (a,) in db.query(models.Photo.album).distinct() if a]
    branches = [b for (b,) in db.query(models.Photo.family_branch).distinct() if b]
    return templates.TemplateResponse(
        "upload.html",
        {"request": request, "albums": sorted(albums), "branches": sorted(branches)},
    )


@router.post("/upload")
async def upload_submit(
    background: BackgroundTasks,
    files: list[UploadFile] = File(...),
    date_estimate: str = Form(""),
    description: str = Form(""),
    source: str = Form(""),
    tags: str = Form(""),
    album: str = Form(""),
    family_branch: str = Form(""),
    db: Session = Depends(get_db),
):
    accepted_ids: list[int] = []
    skipped: list[str] = []

    for upload in files:
        if not upload.filename:
            continue
        ext = Path(upload.filename).suffix.lower()
        if ext not in ALLOWED_EXTS:
            skipped.append(upload.filename)
            continue

        safe_name = _sanitize(upload.filename)
        stored_name = f"{uuid.uuid4().hex[:10]}_{safe_name}"
        dest = settings.photos_dir / stored_name
        dest.parent.mkdir(parents=True, exist_ok=True)

        with dest.open("wb") as out:
            shutil.copyfileobj(upload.file, out)

        size_bytes = dest.stat().st_size
        if size_bytes > settings.max_upload_mb * 1024 * 1024:
            dest.unlink(missing_ok=True)
            skipped.append(upload.filename)
            continue

        if ext == ".pdf":
            rendered_name = stored_name.rsplit(".", 1)[0] + ".jpg"
            rendered_path = settings.photos_dir / rendered_name
            try:
                render_pdf_first_page(dest, rendered_path)
            except Exception:
                skipped.append(upload.filename)
                dest.unlink(missing_ok=True)
                continue
            image_path = rendered_path
            image_name = rendered_name
        else:
            image_path = dest
            image_name = stored_name

        thumb_name = image_name.rsplit(".", 1)[0] + ".jpg"
        thumb_path = settings.thumbs_dir / thumb_name
        try:
            save_thumbnail(image_path, thumb_path)
        except Exception:
            thumb_name = None  # type: ignore[assignment]

        mime, _ = mimetypes.guess_type(image_name)
        try:
            from PIL import Image, ImageOps

            with Image.open(image_path) as img:
                img = ImageOps.exif_transpose(img)
                width, height = img.size
        except Exception:
            width = height = None

        photo = models.Photo(
            filename=upload.filename,
            filepath=str(image_path),
            thumbnail_path=thumb_name,
            mime_type=mime or "image/jpeg",
            width=width,
            height=height,
            date_estimate=date_estimate.strip() or None,
            description=description.strip() or None,
            source=source.strip() or None,
            tags=tags.strip() or None,
            album=album.strip() or None,
            family_branch=family_branch.strip() or None,
            decade=_decade_from_estimate(date_estimate),
        )
        # Store a path relative to photos_dir for portability.
        photo.filepath = str(image_path)
        db.add(photo)
        db.commit()
        db.refresh(photo)
        accepted_ids.append(photo.id)
        background.add_task(process_photo, photo.id)

    if not accepted_ids:
        raise HTTPException(status_code=400, detail="No valid files uploaded")

    return RedirectResponse(url="/gallery", status_code=303)


@router.get("/gallery")
def gallery(
    request: Request,
    album: Optional[str] = None,
    decade: Optional[int] = None,
    branch: Optional[str] = None,
    tag: Optional[str] = None,
    person_id: Optional[int] = None,
    db: Session = Depends(get_db),
):
    q = db.query(models.Photo)
    if album:
        q = q.filter(models.Photo.album == album)
    if decade is not None:
        q = q.filter(models.Photo.decade == decade)
    if branch:
        q = q.filter(models.Photo.family_branch == branch)
    if tag:
        q = q.filter(models.Photo.tags.ilike(f"%{tag}%"))
    if person_id:
        photo_ids = [
            pid for (pid,) in db.query(models.Face.photo_id)
            .filter(models.Face.person_id == person_id, models.Face.is_confirmed.is_(True))
            .distinct()
        ]
        if not photo_ids:
            q = q.filter(models.Photo.id.in_([-1]))
        else:
            q = q.filter(models.Photo.id.in_(photo_ids))

    photos = q.order_by(models.Photo.uploaded_at.desc()).limit(500).all()

    albums = sorted({a for (a,) in db.query(models.Photo.album).distinct() if a})
    decades = sorted({d for (d,) in db.query(models.Photo.decade).distinct() if d})
    branches = sorted({b for (b,) in db.query(models.Photo.family_branch).distinct() if b})
    people = db.query(models.Person).order_by(models.Person.name).all()
    active_person = db.get(models.Person, person_id) if person_id else None

    return templates.TemplateResponse(
        "gallery.html",
        {
            "request": request,
            "photos": photos,
            "albums": albums,
            "decades": decades,
            "branches": branches,
            "people": people,
            "filters": {
                "album": album,
                "decade": decade,
                "branch": branch,
                "tag": tag,
                "person_id": person_id,
            },
            "active_person": active_person,
        },
    )


@router.get("/photos/{photo_id}")
def photo_detail(photo_id: int, request: Request, db: Session = Depends(get_db)):
    photo = db.get(models.Photo, photo_id)
    if photo is None:
        raise HTTPException(status_code=404)
    people = db.query(models.Person).order_by(models.Person.name).all()
    return templates.TemplateResponse(
        "photo.html",
        {
            "request": request,
            "photo": photo,
            "people": people,
        },
    )


@router.post("/photos/{photo_id}/edit")
def photo_edit(
    photo_id: int,
    date_estimate: str = Form(""),
    description: str = Form(""),
    source: str = Form(""),
    tags: str = Form(""),
    album: str = Form(""),
    family_branch: str = Form(""),
    db: Session = Depends(get_db),
):
    photo = db.get(models.Photo, photo_id)
    if photo is None:
        raise HTTPException(status_code=404)
    photo.date_estimate = date_estimate.strip() or None
    photo.description = description.strip() or None
    photo.source = source.strip() or None
    photo.tags = tags.strip() or None
    photo.album = album.strip() or None
    photo.family_branch = family_branch.strip() or None
    photo.decade = _decade_from_estimate(date_estimate)
    db.commit()
    return RedirectResponse(url=f"/photos/{photo_id}", status_code=303)


@router.post("/photos/{photo_id}/delete")
def photo_delete(photo_id: int, db: Session = Depends(get_db)):
    photo = db.get(models.Photo, photo_id)
    if photo is None:
        raise HTTPException(status_code=404)

    for face in photo.faces:
        try:
            (settings.faces_dir / face.crop_path).unlink(missing_ok=True)
        except Exception:
            pass
    try:
        Path(photo.filepath).unlink(missing_ok=True)
    except Exception:
        pass
    if photo.thumbnail_path:
        try:
            (settings.thumbs_dir / photo.thumbnail_path).unlink(missing_ok=True)
        except Exception:
            pass

    db.delete(photo)
    db.commit()
    return RedirectResponse(url="/gallery", status_code=303)


@router.post("/photos/{photo_id}/reprocess")
def photo_reprocess(
    photo_id: int,
    background: BackgroundTasks,
    db: Session = Depends(get_db),
):
    photo = db.get(models.Photo, photo_id)
    if photo is None:
        raise HTTPException(status_code=404)
    # Drop existing, unconfirmed faces before reprocessing so we don't duplicate.
    for face in list(photo.faces):
        if not face.is_confirmed and not face.is_seed:
            (settings.faces_dir / face.crop_path).unlink(missing_ok=True)
            db.delete(face)
    photo.processing_status = "pending"
    photo.processed_at = None
    db.commit()
    background.add_task(process_photo, photo_id)
    return RedirectResponse(url=f"/photos/{photo_id}", status_code=303)
