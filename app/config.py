from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "sqlite:///./data/family_tree.db"
    data_dir: Path = Path("./data")
    photos_dir_name: str = "photos"
    thumbs_dir_name: str = "thumbs"
    faces_dir_name: str = "faces"

    thumbnail_size: int = 360
    max_upload_mb: int = 50

    # Confidence thresholds for cosine similarity (1.0 = identical).
    auto_match_threshold: float = 0.85
    suggest_threshold: float = 0.55

    # Historical photo preprocessing toggles.
    preprocess_clahe: bool = True
    preprocess_grayscale_fallback: bool = True
    upscale_small_face_px: int = 80

    # Face detector — "insightface" or "face_recognition".
    face_backend: str = "insightface"

    @property
    def photos_dir(self) -> Path:
        return self.data_dir / self.photos_dir_name

    @property
    def thumbs_dir(self) -> Path:
        return self.data_dir / self.thumbs_dir_name

    @property
    def faces_dir(self) -> Path:
        return self.data_dir / self.faces_dir_name

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.photos_dir, self.thumbs_dir, self.faces_dir):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
