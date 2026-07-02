# Painting Similarity Search

Local web app to find visually similar paintings. Ingest a folder of images once, then upload a painting in the browser to get the closest matches.

## Requirements

- Python 3.10+
- ~8 GB RAM (CPU-only)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Download WikiArt dataset (optional)

The [huggan/wikiart](https://huggingface.co/datasets/huggan/wikiart) dataset has ~81k paintings with artist, genre, and style labels. **Non-commercial research only.**

Requires ~34 GB disk space for the full download.

```bash
# Test with 100 images first
python download_wikiart.py --limit 100

# Full download (resumable)
python download_wikiart.py

# Resume after interruption
python download_wikiart.py --resume
```

Output layout:

```
wikiart_data/
  images/          # 000000.jpg, 000001.jpg, ...
  manifest.csv     # id, filename, artist, genre, style
```

Then ingest:

```bash
python ingest.py wikiart_data/images
```

## Ingest paintings

```bash
python ingest.py /path/to/paintings
```

Walks the folder recursively, embeds each image with DINOv2, and writes the FAISS index, SQLite metadata, and thumbnails to `data/`.

## Run the search UI

```bash
uvicorn app:app --reload
```

Open http://localhost:8000 — drag and drop a painting to search.

## Project layout

- `config.py` — paths and model settings
- `embedder.py` — DINOv2 image embeddings
- `index_store.py` — FAISS index + SQLite metadata
- `ingest.py` — batch ingestion CLI
- `app.py` — FastAPI server
- `static/index.html` — drag-drop search UI
- `data/` — persisted index, database, thumbnails (gitignored)
