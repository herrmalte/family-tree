from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import RedirectResponse
from sqlalchemy import or_
from sqlalchemy.orm import Session

from .. import models
from ..db import get_db
from ..templating import templates

router = APIRouter()

RELATIONSHIP_TYPES = ["parent", "child", "spouse", "sibling"]


@router.get("/people")
def people_list(request: Request, db: Session = Depends(get_db)):
    people = db.query(models.Person).order_by(models.Person.name).all()
    counts: dict[int, dict] = {}
    for person in people:
        confirmed = sum(1 for f in person.faces if f.is_confirmed)
        seeds = sum(1 for f in person.faces if f.is_seed)
        counts[person.id] = {"confirmed": confirmed, "seeds": seeds}
    return templates.TemplateResponse(
        "people.html",
        {"request": request, "people": people, "counts": counts},
    )


@router.post("/people")
def person_create(
    name: str = Form(...),
    birth_date: str = Form(""),
    death_date: str = Form(""),
    bio: str = Form(""),
    relationship_tags: str = Form(""),
    db: Session = Depends(get_db),
):
    name = name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Name is required")
    person = models.Person(
        name=name,
        birth_date=birth_date.strip() or None,
        death_date=death_date.strip() or None,
        bio=bio.strip() or None,
        relationship_tags=relationship_tags.strip() or None,
    )
    db.add(person)
    db.commit()
    db.refresh(person)
    return RedirectResponse(url=f"/people/{person.id}", status_code=303)


@router.get("/people/{person_id}")
def person_detail(person_id: int, request: Request, db: Session = Depends(get_db)):
    person = db.get(models.Person, person_id)
    if person is None:
        raise HTTPException(status_code=404)
    confirmed_faces = [f for f in person.faces if f.is_confirmed]
    seed_faces = [f for f in person.faces if f.is_seed]
    suggested_faces = [f for f in person.faces if not f.is_confirmed and not f.is_seed]

    links = (
        db.query(models.FamilyLink)
        .filter(models.FamilyLink.person_id == person_id)
        .all()
    )
    related = {
        link.related_person_id: (link, db.get(models.Person, link.related_person_id))
        for link in links
    }

    all_people = (
        db.query(models.Person)
        .filter(models.Person.id != person_id)
        .order_by(models.Person.name)
        .all()
    )
    return templates.TemplateResponse(
        "person.html",
        {
            "request": request,
            "person": person,
            "confirmed_faces": confirmed_faces,
            "seed_faces": seed_faces,
            "suggested_faces": suggested_faces,
            "related": related,
            "all_people": all_people,
            "relationship_types": RELATIONSHIP_TYPES,
        },
    )


@router.post("/people/{person_id}/edit")
def person_edit(
    person_id: int,
    name: str = Form(...),
    birth_date: str = Form(""),
    death_date: str = Form(""),
    bio: str = Form(""),
    relationship_tags: str = Form(""),
    db: Session = Depends(get_db),
):
    person = db.get(models.Person, person_id)
    if person is None:
        raise HTTPException(status_code=404)
    person.name = name.strip() or person.name
    person.birth_date = birth_date.strip() or None
    person.death_date = death_date.strip() or None
    person.bio = bio.strip() or None
    person.relationship_tags = relationship_tags.strip() or None
    db.commit()
    return RedirectResponse(url=f"/people/{person_id}", status_code=303)


@router.post("/people/{person_id}/delete")
def person_delete(person_id: int, db: Session = Depends(get_db)):
    person = db.get(models.Person, person_id)
    if person is None:
        raise HTTPException(status_code=404)
    # Clear assignments on faces; preserve the faces themselves.
    for face in list(person.faces):
        face.person_id = None
        face.is_confirmed = False
        face.is_seed = False
    db.query(models.FamilyLink).filter(
        or_(
            models.FamilyLink.person_id == person_id,
            models.FamilyLink.related_person_id == person_id,
        )
    ).delete(synchronize_session=False)
    db.delete(person)
    db.commit()
    return RedirectResponse(url="/people", status_code=303)


@router.post("/people/{person_id}/links")
def person_add_link(
    person_id: int,
    related_person_id: int = Form(...),
    relationship_type: str = Form(...),
    db: Session = Depends(get_db),
):
    if relationship_type not in RELATIONSHIP_TYPES:
        raise HTTPException(status_code=400, detail="Unknown relationship type")
    if related_person_id == person_id:
        raise HTTPException(status_code=400, detail="Cannot relate a person to themselves")

    person = db.get(models.Person, person_id)
    related = db.get(models.Person, related_person_id)
    if person is None or related is None:
        raise HTTPException(status_code=404)

    existing = (
        db.query(models.FamilyLink)
        .filter(
            models.FamilyLink.person_id == person_id,
            models.FamilyLink.related_person_id == related_person_id,
            models.FamilyLink.relationship_type == relationship_type,
        )
        .first()
    )
    if existing:
        return RedirectResponse(url=f"/people/{person_id}", status_code=303)

    db.add(
        models.FamilyLink(
            person_id=person_id,
            related_person_id=related_person_id,
            relationship_type=relationship_type,
        )
    )
    # Add the symmetric link so the tree is consistent.
    inverse_map = {
        "parent": "child",
        "child": "parent",
        "spouse": "spouse",
        "sibling": "sibling",
    }
    inverse_type = inverse_map[relationship_type]
    existing_inverse = (
        db.query(models.FamilyLink)
        .filter(
            models.FamilyLink.person_id == related_person_id,
            models.FamilyLink.related_person_id == person_id,
            models.FamilyLink.relationship_type == inverse_type,
        )
        .first()
    )
    if not existing_inverse:
        db.add(
            models.FamilyLink(
                person_id=related_person_id,
                related_person_id=person_id,
                relationship_type=inverse_type,
            )
        )
    db.commit()
    return RedirectResponse(url=f"/people/{person_id}", status_code=303)


@router.post("/people/{person_id}/links/{link_id}/delete")
def person_delete_link(person_id: int, link_id: int, db: Session = Depends(get_db)):
    link = db.get(models.FamilyLink, link_id)
    if link is None or link.person_id != person_id:
        raise HTTPException(status_code=404)
    # Remove symmetric pair too.
    inverse_map = {
        "parent": "child",
        "child": "parent",
        "spouse": "spouse",
        "sibling": "sibling",
    }
    inverse_type = inverse_map.get(link.relationship_type)
    db.query(models.FamilyLink).filter(
        models.FamilyLink.person_id == link.related_person_id,
        models.FamilyLink.related_person_id == link.person_id,
        models.FamilyLink.relationship_type == inverse_type,
    ).delete(synchronize_session=False)
    db.delete(link)
    db.commit()
    return RedirectResponse(url=f"/people/{person_id}", status_code=303)


@router.get("/tree")
def family_tree(request: Request, root_id: Optional[int] = None, db: Session = Depends(get_db)):
    people = db.query(models.Person).order_by(models.Person.name).all()
    root = db.get(models.Person, root_id) if root_id else (people[0] if people else None)

    nodes: dict[int, dict] = {}
    edges: list[dict] = []
    if root is not None:
        # BFS up to depth 3 from root.
        visited = {root.id}
        frontier = [(root, 0)]
        while frontier:
            person, depth = frontier.pop(0)
            nodes[person.id] = {
                "id": person.id,
                "name": person.name,
                "birth": person.birth_date,
                "death": person.death_date,
            }
            if depth >= 3:
                continue
            links = (
                db.query(models.FamilyLink)
                .filter(models.FamilyLink.person_id == person.id)
                .all()
            )
            for link in links:
                edges.append(
                    {
                        "from": link.person_id,
                        "to": link.related_person_id,
                        "type": link.relationship_type,
                    }
                )
                if link.related_person_id not in visited:
                    visited.add(link.related_person_id)
                    related = db.get(models.Person, link.related_person_id)
                    if related is not None:
                        frontier.append((related, depth + 1))

    return templates.TemplateResponse(
        "tree.html",
        {
            "request": request,
            "people": people,
            "root": root,
            "nodes": list(nodes.values()),
            "edges": edges,
        },
    )
