from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from config import BASE_DIR, THUMBNAILS_DIR, TOP_K
from embedder import Embedder
from index_store import IndexStore

embedder: Embedder | None = None
store: IndexStore | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedder, store
    embedder = Embedder()
    store = IndexStore()
    if not store.load():
        print("Warning: no index found. Run ingest.py first.")
    yield
    if store:
        store.close()


app = FastAPI(title="Painting Similarity Search", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def index_page():
    html_path = BASE_DIR / "static" / "index.html"
    return HTMLResponse(html_path.read_text())


@app.post("/search")
async def search(file: UploadFile = File(...), k: int = TOP_K):
    if embedder is None or store is None:
        raise HTTPException(503, "Server not ready")

    if store.count() == 0:
        raise HTTPException(503, "No paintings indexed. Run ingest.py first.")

    try:
        image = Image.open(file.file)
        vector = embedder.embed(image)
    except Exception:
        raise HTTPException(400, "Could not read image")

    results = store.search(vector, k)
    matches = []
    for row_id, score in results:
        meta = store.get_by_id(row_id)
        if meta is None:
            continue
        matches.append(
            {
                "id": row_id,
                "score": round(score, 4),
                "path": meta["path"],
                "title": meta["title"],
                "artist": meta["artist"],
                "thumbnail_url": f"/thumb/{row_id}",
            }
        )

    return {"matches": matches}


@app.get("/thumb/{row_id}")
async def thumbnail(row_id: int):
    thumb_path = THUMBNAILS_DIR / f"{row_id}.jpg"
    if not thumb_path.exists():
        raise HTTPException(404, "Thumbnail not found")
    return FileResponse(thumb_path, media_type="image/jpeg")


@app.get("/health")
async def health():
    return {"indexed": store.count() if store else 0}
