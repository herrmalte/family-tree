from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .config import settings
from .db import init_db
from .routes import faces as faces_routes
from .routes import people as people_routes
from .routes import photos as photos_routes
from .routes import search as search_routes

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


def create_app() -> FastAPI:
    app = FastAPI(title="Family Tree Photo Archive", version="0.1.0")

    init_db()

    # Static assets (CSS/JS).
    static_dir = Path(__file__).parent / "static"
    static_dir.mkdir(exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Uploaded photos, thumbnails, face crops — served straight from disk.
    app.mount(
        "/media/photos",
        StaticFiles(directory=str(settings.photos_dir)),
        name="photos_media",
    )
    app.mount(
        "/media/thumbs",
        StaticFiles(directory=str(settings.thumbs_dir)),
        name="thumbs_media",
    )
    app.mount(
        "/media/faces",
        StaticFiles(directory=str(settings.faces_dir)),
        name="faces_media",
    )

    app.include_router(photos_routes.router)
    app.include_router(faces_routes.router)
    app.include_router(people_routes.router)
    app.include_router(search_routes.router)

    return app


app = create_app()
