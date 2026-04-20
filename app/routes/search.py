from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy import or_
from sqlalchemy.orm import Session

from .. import models
from ..db import get_db
from ..templating import templates

router = APIRouter()


@router.get("/search")
def search(
    request: Request,
    q: Optional[str] = Query(None),
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
    tag: Optional[str] = Query(None),
    person_id: Optional[int] = Query(None),
    db: Session = Depends(get_db),
):
    photos_q = db.query(models.Photo)
    people_q = db.query(models.Person)

    if q:
        like = f"%{q}%"
        photos_q = photos_q.filter(
            or_(
                models.Photo.description.ilike(like),
                models.Photo.tags.ilike(like),
                models.Photo.source.ilike(like),
                models.Photo.album.ilike(like),
                models.Photo.family_branch.ilike(like),
                models.Photo.date_estimate.ilike(like),
                models.Photo.filename.ilike(like),
            )
        )
        people_q = people_q.filter(
            or_(
                models.Person.name.ilike(like),
                models.Person.bio.ilike(like),
                models.Person.relationship_tags.ilike(like),
            )
        )
    else:
        people_q = people_q.filter(models.Person.id == -1)

    if tag:
        photos_q = photos_q.filter(models.Photo.tags.ilike(f"%{tag}%"))

    if date_from:
        photos_q = photos_q.filter(models.Photo.date_estimate >= date_from)
    if date_to:
        photos_q = photos_q.filter(models.Photo.date_estimate <= date_to)

    if person_id:
        photo_ids = [
            pid
            for (pid,) in db.query(models.Face.photo_id)
            .filter(
                models.Face.person_id == person_id,
                models.Face.is_confirmed.is_(True),
            )
            .distinct()
        ]
        if photo_ids:
            photos_q = photos_q.filter(models.Photo.id.in_(photo_ids))
        else:
            photos_q = photos_q.filter(models.Photo.id == -1)

    photos = photos_q.order_by(models.Photo.uploaded_at.desc()).limit(200).all()
    people = people_q.order_by(models.Person.name).limit(50).all()

    all_people = db.query(models.Person).order_by(models.Person.name).all()

    return templates.TemplateResponse(
        "search.html",
        {
            "request": request,
            "photos": photos,
            "people": people,
            "all_people": all_people,
            "query": {
                "q": q or "",
                "date_from": date_from or "",
                "date_to": date_to or "",
                "tag": tag or "",
                "person_id": person_id,
            },
        },
    )
