import argparse
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from config import IMAGE_EXTENSIONS, THUMBNAILS_DIR
from embedder import Embedder
from index_store import IndexStore


def find_images(root: Path) -> list[Path]:
    images = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(path)
    return sorted(images)


def save_thumbnail(image: Image.Image, row_id: int, max_side: int = 256):
    from config import THUMBNAIL_MAX_SIDE

    max_side = max_side or THUMBNAIL_MAX_SIDE
    image = image.convert("RGB")
    w, h = image.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        image = image.resize(
            (int(w * scale), int(h * scale)), Image.Resampling.LANCZOS
        )
    thumb_path = THUMBNAILS_DIR / f"{row_id}.jpg"
    image.save(thumb_path, "JPEG", quality=85)


def ingest(folder: Path, rebuild: bool = False):
    images = find_images(folder)
    if not images:
        print(f"No images found in {folder}")
        sys.exit(1)

    store = IndexStore()
    if rebuild or not store.load():
        store.create_index()
        conn = store._connect()
        conn.execute("DELETE FROM paintings")
        conn.commit()
        if THUMBNAILS_DIR.exists():
            for thumb in THUMBNAILS_DIR.glob("*.jpg"):
                thumb.unlink()
    else:
        existing = {row["path"] for row in store._connect().execute("SELECT path FROM paintings")}
        images = [p for p in images if str(p.resolve()) not in existing]
        if not images:
            print("All images already ingested.")
            return

    embedder = Embedder()
    print(f"Ingesting {len(images)} images from {folder}")

    for path in tqdm(images, unit="img"):
        try:
            image = Image.open(path)
            vector = embedder.embed(image)
            row_id = store.add(vector, str(path.resolve()))
            save_thumbnail(image, row_id)
        except Exception as e:
            tqdm.write(f"Skipping {path}: {e}")

    count = store.count()
    store.save()
    store.close()
    print(f"Done. Indexed {count} paintings.")


def main():
    parser = argparse.ArgumentParser(description="Ingest paintings into the search index")
    parser.add_argument("folder", type=Path, help="Folder containing painting images")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild index from scratch (ignores existing index)",
    )
    args = parser.parse_args()

    if not args.folder.is_dir():
        print(f"Not a directory: {args.folder}")
        sys.exit(1)

    ingest(args.folder, rebuild=args.rebuild)


if __name__ == "__main__":
    main()
