from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
INDEX_PATH = DATA_DIR / "index.faiss"
DB_PATH = DATA_DIR / "meta.sqlite"
THUMBNAILS_DIR = DATA_DIR / "thumbnails"

MODEL_NAME = "facebook/dinov2-base"
EMBED_DIM = 768

INGEST_MAX_SIDE = 512
THUMBNAIL_MAX_SIDE = 256
TOP_K = 12

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}
