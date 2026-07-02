"""
Download huggan/wikiart from Hugging Face.

Saves images to disk and writes a CSV manifest with artist, genre, and style labels.
Downloads parquet shards first (with HF progress), then extracts images locally.

License: WikiArt is for non-commercial research only.
https://huggingface.co/datasets/huggan/wikiart
"""

import os

# Must be set before huggingface_hub is imported (Xet path stalls on some networks)
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")

import argparse
import csv
import sys
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import snapshot_download
from tqdm import tqdm

MANIFEST_FIELDS = ["id", "filename", "artist", "genre", "style"]
DATASET_REPO = "huggan/wikiart"
TOTAL_IMAGES = 81444


def load_manifest(path: Path) -> set[int]:
    if not path.exists():
        return set()
    done = set()
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            done.add(int(row["id"]))
    return done


def append_manifest(path: Path, row: dict, write_header: bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def fetch_parquet_shards() -> Path:
    print("Step 1/2: Downloading parquet shards from Hugging Face (~34 GB)...")
    print("         (Uses HF_TOKEN if set — progress shown per file)\n")
    cache_path = snapshot_download(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        allow_patterns=["data/*.parquet"],
        max_workers=1,
    )
    shard_dir = Path(cache_path) / "data"
    shards = sorted(shard_dir.glob("*.parquet"))
    print(f"\n{len(shards)} parquet shards ready at {shard_dir}\n")
    return shard_dir


def extract_images(
    shard_dir: Path,
    output_dir: Path,
    limit: int | None = None,
    resume: bool = False,
):
    images_dir = output_dir / "images"
    manifest_path = output_dir / "manifest.csv"
    images_dir.mkdir(parents=True, exist_ok=True)

    done_ids: set[int] = set()
    if resume:
        done_ids = load_manifest(manifest_path)
        if done_ids:
            print(f"Resuming: {len(done_ids)} images already extracted")

    write_header = not manifest_path.exists() or manifest_path.stat().st_size == 0
    if resume and done_ids:
        write_header = False
    elif not resume and manifest_path.exists():
        manifest_path.unlink()
        write_header = True

    parquet_files = sorted(shard_dir.glob("*.parquet"))
    total = limit or TOTAL_IMAGES
    print(f"Step 2/2: Extracting images (0/{total})...\n")

    saved = 0
    skipped = 0
    global_idx = 0

    for parquet_file in parquet_files:
        dataset = load_dataset(
            "parquet", data_files=str(parquet_file), split="train", streaming=True
        )
        for row in dataset:
            if limit is not None and global_idx >= limit:
                break

            if resume and global_idx in done_ids:
                skipped += 1
                global_idx += 1
                continue

            filename = f"{global_idx:06d}.jpg"
            try:
                image = row["image"]
                if image.mode != "RGB":
                    image = image.convert("RGB")
                image.save(images_dir / filename, "JPEG", quality=92)
            except Exception as e:
                tqdm.write(f"Skipping row {global_idx}: {e}")
                global_idx += 1
                continue

            append_manifest(
                manifest_path,
                {
                    "id": global_idx,
                    "filename": filename,
                    "artist": row["artist"],
                    "genre": row["genre"],
                    "style": row["style"],
                },
                write_header=write_header,
            )
            write_header = False
            saved += 1
            global_idx += 1

            if saved % 100 == 0 or (limit and saved == limit):
                print(f"  Extracted {saved} images ({global_idx} rows processed)")

        if limit is not None and global_idx >= limit:
            break

    print(f"\nDone. Saved {saved} images to {images_dir}")
    if skipped:
        print(f"Skipped {skipped} already-extracted images")
    print(f"Manifest: {manifest_path}")
    print(f"\nNext step — ingest into the search index:")
    print(f"  python ingest.py {images_dir}")


def download(output_dir: Path, limit: int | None = None, resume: bool = False):
    shard_dir = fetch_parquet_shards()
    extract_images(shard_dir, output_dir, limit=limit, resume=resume)


def main():
    parser = argparse.ArgumentParser(
        description="Download WikiArt dataset (images + metadata) from Hugging Face"
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("wikiart_data"),
        help="Output directory (default: wikiart_data/)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Extract only the first N images (for testing)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip images already listed in manifest.csv",
    )
    args = parser.parse_args()

    print("Note: WikiArt is licensed for non-commercial research only.")
    print("      Expect ~34 GB disk space for the full dataset.\n")

    try:
        download(args.output_dir, limit=args.limit, resume=args.resume)
    except KeyboardInterrupt:
        print("\nInterrupted. Re-run with --resume to continue.")
        sys.exit(1)


if __name__ == "__main__":
    main()
