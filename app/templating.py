from __future__ import annotations

from pathlib import Path

from fastapi.templating import Jinja2Templates

from .config import settings

templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


def _media_url(kind: str, filename: str | None) -> str:
    if not filename:
        return ""
    return f"/media/{kind}/{filename}"


def _split_tags(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [t.strip() for t in raw.split(",") if t.strip()]


templates.env.globals["settings"] = settings
templates.env.filters["media_photo"] = lambda f: _media_url("photos", f)
templates.env.filters["media_thumb"] = lambda f: _media_url("thumbs", f)
templates.env.filters["media_face"] = lambda f: _media_url("faces", f)
templates.env.filters["split_tags"] = _split_tags
