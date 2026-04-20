# Family Tree Photo Archive

A self-hosted web app for scanning, organising, and identifying people in a
historical family photo collection. Faces are detected automatically, matched
against a small set of user-labeled *seed* faces, and proposed for review.

## Features

- **Photo management** – upload JPEG / PNG / TIFF / PDF, with metadata (date,
  description, source, tags), grouping by album, decade, and family branch.
- **Gallery** – thumbnailed grid with filters and a full-resolution detail view.
- **AI face recognition** – InsightFace `buffalo_s` (MobileFaceNet + SCRFD-500MF,
  ~17 MB, 512-d) by default for a small deploy footprint. Swap to `buffalo_l`
  (~326 MB, R100 + RetinaFace-R50) for best accuracy on historical scans.
  Automatic fallback to `face_recognition` (dlib, 128-d) if InsightFace fails
  to install. Embeddings are stored as blobs in the database.
- **Seed labeling** – tag 2–3 clean photos per person; the system uses them
  as references for automatic matching.
- **Review queue** – suggested matches ranked by cosine similarity. Confirm
  or reject per face; configurable auto / suggest / unidentified thresholds.
- **Unidentified queue** – all unmatched faces, sorted by image quality.
- **People directory** – profiles with birth/death years, bio, relationship
  tags, and a confirmed-photo grid.
- **Family tree** – simple SVG tree for parent / child / spouse / sibling
  relationships, rooted at any person.
- **Search** – full-text search across photos and people; filter by
  date range, tag, or "who's in this photo".
- **Historical preprocessing** – CLAHE contrast enhancement and small-face
  upscaling to help with faded / low-resolution scans.
- **Async processing** – face detection runs in a FastAPI `BackgroundTask`
  after upload, so the UI never blocks.

## Tech stack

- Python 3.11, FastAPI, SQLAlchemy 2
- Jinja2 templates (no JS framework)
- SQLite by default (set `DATABASE_URL` to a Postgres URL to migrate)
- InsightFace (`buffalo_l`) via `onnxruntime`, with `face_recognition` fallback
- Pillow, OpenCV (headless), `pdf2image` for PDF ingestion
- Single-container Docker Compose deployment

## Running with Docker

```bash
cp .env.example .env        # optional
docker compose up --build
```

Then open http://localhost:8000.

The InsightFace model pack (`buffalo_s`, ~17 MB) is downloaded on first run
and cached in the `insightface_models` volume so subsequent starts are fast.
For maximum accuracy on low-quality historical photos, set
`FACE_MODEL=buffalo_l` (and raise the thresholds by ~0.10).

## Running locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

System packages required for local runs:
- `poppler-utils` (for PDF ingestion)
- `libgl1`, `libglib2.0-0` (OpenCV runtime)

## Typical workflow

1. **Upload** a batch of photos on `/upload`.
2. Watch the gallery — faces are detected in the background.
3. Create a couple of **people** (`/people`), then head to `/faces/seed`
   and tag 2–3 clear faces of each person as seeds.
4. The system re-runs matching automatically after seeds are added.
5. Use `/faces/review` to confirm or reject suggested matches, and
   `/faces/unidentified` to work through faces that weren't matched.
6. Record relationships on each person's profile; view the tree at `/tree`.

## Configuration

Environment variables (also via `.env`):

| Variable                 | Default                               | Description                              |
|--------------------------|---------------------------------------|------------------------------------------|
| `DATABASE_URL`           | `sqlite:///./data/family_tree.db`     | SQLAlchemy URL. Use Postgres in prod.    |
| `DATA_DIR`               | `./data`                              | Root for photos/thumbs/face crops.       |
| `FACE_BACKEND`           | `insightface`                         | `insightface` or `face_recognition`.     |
| `FACE_MODEL`             | `buffalo_s`                           | `buffalo_s` (17 MB), `buffalo_m`, `buffalo_l` (326 MB). |
| `AUTO_MATCH_THRESHOLD`   | `0.80`                                | Auto-confirm above this cosine sim.      |
| `SUGGEST_THRESHOLD`      | `0.45`                                | Suggest between suggest–auto threshold.  |
| `THUMBNAIL_SIZE`         | `360`                                 | Max edge (px) for gallery thumbnails.    |
| `MAX_UPLOAD_MB`          | `50`                                  | Per-file upload size limit.              |
| `PREPROCESS_CLAHE`       | `true`                                | Apply CLAHE contrast on detection.       |
| `UPSCALE_SMALL_FACE_PX`  | `80`                                  | Upscale face crops below this size.      |

## Data model

- `photos` — id, filepath, thumbnail_path, date_estimate, description, tags,
  album, family_branch, decade, uploaded_at, processing_status
- `faces` — id, photo_id, bbox (x,y,w,h), crop_path, embedding (blob),
  person_id?, quality, confidence, is_confirmed, is_seed, is_rejected
- `people` — id, name, birth_date, death_date, bio, relationship_tags
- `family_links` — id, person_id, related_person_id, relationship_type

Embeddings are serialized with `np.float32.tobytes()`. Similarity is cosine
(vectors normalized on read).

## Build order followed

1. Upload + gallery + database ✓
2. Face detection pipeline (detect → crop → store) ✓
3. Seed labeling UI ✓
4. Matching engine + review queue ✓
5. People profiles + search ✓

Each step is usable end-to-end before the next layers on top.
