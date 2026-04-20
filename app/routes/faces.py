from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Form, HTTPException, Query, Request
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from .. import models
from ..config import settings
from ..db import get_db
from ..face_engine import engine_status
from ..tasks import rematch_all_unconfirmed, rematch_face
from ..templating import templates

router = APIRouter()


@router.get("/faces/seed")
def seed_list(request: Request, db: Session = Depends(get_db)):
    """Seed labeling UI — shows unseeded confirmed faces and a people picker."""
    people = db.query(models.Person).order_by(models.Person.name).all()
    # Candidate faces: unassigned, high quality, show highest quality first.
    candidates = (
        db.query(models.Face)
        .filter(models.Face.is_seed.is_(False), models.Face.is_rejected.is_(False))
        .order_by(models.Face.quality.desc(), models.Face.id.desc())
        .limit(60)
        .all()
    )
    from sqlalchemy import func
    seed_counts = dict(
        db.query(models.Face.person_id, func.count(models.Face.id))
        .filter(models.Face.is_seed.is_(True))
        .group_by(models.Face.person_id)
        .all()
    )
    return templates.TemplateResponse(
        "seed.html",
        {
            "request": request,
            "people": people,
            "candidates": candidates,
            "seed_counts": seed_counts,
            "engine": engine_status(),
        },
    )


@router.post("/faces/{face_id}/assign")
def face_assign(
    face_id: int,
    person_id: int = Form(...),
    as_seed: bool = Form(False),
    confirm: bool = Form(True),
    next_url: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    face = db.get(models.Face, face_id)
    if face is None:
        raise HTTPException(status_code=404)
    person = db.get(models.Person, person_id)
    if person is None:
        raise HTTPException(status_code=400, detail="Unknown person")
    face.person_id = person_id
    face.is_rejected = False
    if confirm:
        face.is_confirmed = True
    if as_seed:
        face.is_seed = True
        face.is_confirmed = True
    db.commit()
    # When a new seed is added, suggestions for unconfirmed faces may change.
    if as_seed:
        rematch_all_unconfirmed()
    return RedirectResponse(url=next_url or "/faces/seed", status_code=303)


@router.post("/faces/{face_id}/unseed")
def face_unseed(
    face_id: int,
    next_url: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    face = db.get(models.Face, face_id)
    if face is None:
        raise HTTPException(status_code=404)
    face.is_seed = False
    db.commit()
    rematch_all_unconfirmed()
    return RedirectResponse(url=next_url or "/faces/seed", status_code=303)


@router.post("/faces/{face_id}/reject")
def face_reject(
    face_id: int,
    next_url: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    face = db.get(models.Face, face_id)
    if face is None:
        raise HTTPException(status_code=404)
    face.is_rejected = True
    face.person_id = None
    face.is_confirmed = False
    face.is_seed = False
    db.commit()
    return RedirectResponse(url=next_url or "/faces/unidentified", status_code=303)


@router.post("/faces/{face_id}/unreject")
def face_unreject(
    face_id: int,
    next_url: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    face = db.get(models.Face, face_id)
    if face is None:
        raise HTTPException(status_code=404)
    face.is_rejected = False
    db.commit()
    rematch_face(face_id)
    return RedirectResponse(url=next_url or "/faces/unidentified", status_code=303)


@router.post("/faces/{face_id}/confirm")
def face_confirm(
    face_id: int,
    next_url: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    face = db.get(models.Face, face_id)
    if face is None:
        raise HTTPException(status_code=404)
    if face.person_id is None:
        raise HTTPException(status_code=400, detail="No suggestion to confirm")
    face.is_confirmed = True
    face.is_rejected = False
    db.commit()
    return RedirectResponse(url=next_url or "/faces/review", status_code=303)


@router.post("/faces/{face_id}/unconfirm")
def face_unconfirm(
    face_id: int,
    next_url: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    face = db.get(models.Face, face_id)
    if face is None:
        raise HTTPException(status_code=404)
    face.is_confirmed = False
    face.person_id = None
    db.commit()
    return RedirectResponse(url=next_url or "/faces/review", status_code=303)


@router.get("/faces/review")
def review_queue(request: Request, db: Session = Depends(get_db)):
    """Queue of faces with suggested matches (score between suggest and auto thresholds)."""
    faces = (
        db.query(models.Face)
        .filter(
            models.Face.person_id.is_not(None),
            models.Face.is_confirmed.is_(False),
            models.Face.is_seed.is_(False),
            models.Face.is_rejected.is_(False),
        )
        .order_by(models.Face.confidence.desc())
        .limit(200)
        .all()
    )
    people = db.query(models.Person).order_by(models.Person.name).all()
    return templates.TemplateResponse(
        "review.html",
        {
            "request": request,
            "faces": faces,
            "people": people,
            "thresholds": {
                "auto": settings.auto_match_threshold,
                "suggest": settings.suggest_threshold,
            },
        },
    )


@router.get("/faces/unidentified")
def unidentified_queue(
    request: Request,
    sort: str = Query("quality", pattern="^(quality|recent)$"),
    db: Session = Depends(get_db),
):
    q = db.query(models.Face).filter(
        models.Face.person_id.is_(None),
        models.Face.is_seed.is_(False),
        models.Face.is_rejected.is_(False),
    )
    if sort == "recent":
        q = q.order_by(models.Face.id.desc())
    else:
        q = q.order_by(models.Face.quality.desc(), models.Face.id.desc())
    faces = q.limit(300).all()
    people = db.query(models.Person).order_by(models.Person.name).all()
    return templates.TemplateResponse(
        "unidentified.html",
        {
            "request": request,
            "faces": faces,
            "people": people,
            "sort": sort,
        },
    )


@router.post("/faces/rematch-all")
def rematch_all_action():
    rematch_all_unconfirmed()
    return RedirectResponse(url="/faces/review", status_code=303)
