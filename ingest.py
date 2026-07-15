import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from config import (
    IMAGE_EXTENSIONS,
    INGEST_BATCH_SIZE,
    INGEST_CHECKPOINT_INTERVAL,
    THUMBNAIL_MAX_SIDE,
    THUMBNAILS_DIR,
)
from embedder import Embedder
from index_store import IndexStore


def find_images(root: Path) -> list[Path]:
    images = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(path)
    return sorted(images)


def save_thumbnail(
    image: Image.Image, row_id: int, max_side: int = THUMBNAIL_MAX_SIDE
):
    image = image.convert("RGB")
    w, h = image.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        image = image.resize(
            (int(w * scale), int(h * scale)), Image.Resampling.LANCZOS
        )
    thumb_path = THUMBNAILS_DIR / f"{row_id}.jpg"
    image.save(thumb_path, "JPEG", quality=85)


def ingest(
    folder: Path,
    rebuild: bool = False,
    batch_size: int = INGEST_BATCH_SIZE,
    checkpoint_interval: int = INGEST_CHECKPOINT_INTERVAL,
):
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
        # Replace any older on-disk index immediately, so an interruption
        # before the first periodic checkpoint resumes from a clean baseline.
        store.save()
    else:
        existing = {row["path"] for row in store._connect().execute("SELECT path FROM paintings")}
        images = [p for p in images if str(p.resolve()) not in existing]
        if not images:
            print("All images already ingested.")
            return

    embedder = Embedder()
    print(
        f"Ingesting {len(images)} images from {folder} "
        f"(batch={batch_size}, checkpoint={checkpoint_interval})"
    )
    last_checkpoint = store.count()

    with tqdm(total=len(images), unit="img") as progress:
        for offset in range(0, len(images), batch_size):
            batch_paths = images[offset : offset + batch_size]
            loaded: list[tuple[Path, Image.Image]] = []
            for path in batch_paths:
                try:
                    with Image.open(path) as source:
                        loaded.append((path, source.convert("RGB")))
                except Exception as error:
                    tqdm.write(f"Skipping {path}: {error}")
                    progress.update(1)

            if not loaded:
                continue

            vectors = embedder.embed_batch([image for _, image in loaded])
            first_id = store.count()
            accepted_vectors = []
            metadata = []
            for vector, (path, image) in zip(vectors, loaded):
                row_id = first_id + len(metadata)
                try:
                    save_thumbnail(image, row_id)
                except Exception as error:
                    tqdm.write(f"Skipping {path}: thumbnail failed: {error}")
                    progress.update(1)
                    continue
                accepted_vectors.append(vector)
                metadata.append((str(path.resolve()), None, None))

            if metadata:
                store.add_batch(
                    np.stack(accepted_vectors),
                    metadata,
                )
            progress.update(len(metadata))

            if store.count() - last_checkpoint >= checkpoint_interval:
                store.save()
                last_checkpoint = store.count()
                tqdm.write(f"Checkpoint saved at {last_checkpoint} paintings")

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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=INGEST_BATCH_SIZE,
        help=f"Images per model batch (default: {INGEST_BATCH_SIZE})",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=INGEST_CHECKPOINT_INTERVAL,
        help=(
            "Save the FAISS index every N paintings "
            f"(default: {INGEST_CHECKPOINT_INTERVAL})"
        ),
    )
    args = parser.parse_args()

    if not args.folder.is_dir():
        print(f"Not a directory: {args.folder}")
        sys.exit(1)
    if args.batch_size < 1 or args.checkpoint_interval < 1:
        parser.error("--batch-size and --checkpoint-interval must be positive")

    ingest(
        args.folder,
        rebuild=args.rebuild,
        batch_size=args.batch_size,
        checkpoint_interval=args.checkpoint_interval,
    )


if __name__ == "__main__":
    main()
