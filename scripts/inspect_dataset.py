"""Create a reproducible, read-only inventory of the supplied dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from PIL import Image


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=Path, default=root / "data" / "raw" / "images")
    parser.add_argument("--output", type=Path, default=root / "artifacts" / "inventory" / "dataset_inventory.json")
    arguments = parser.parse_args()
    images = sorted(path for path in arguments.image_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    records = []
    hashes: Counter[str] = Counter()
    resolutions: Counter[str] = Counter()
    for image_path in images:
        with Image.open(image_path) as image:
            width, height = image.size
            image.verify()
        digest = sha256(image_path)
        hashes[digest] += 1
        resolutions[f"{width}x{height}"] += 1
        records.append({"file": image_path.name, "width": width, "height": height, "bytes": image_path.stat().st_size, "sha256": digest})
    manifest = {
        "image_directory": str(arguments.image_dir),
        "image_count": len(records),
        "resolution_counts": dict(resolutions),
        "exact_duplicate_groups": [digest for digest, count in hashes.items() if count > 1],
        "annotation_status": "No annotations supplied; do not start detector training.",
        "images": records,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {arguments.output} ({len(records)} images; {len(manifest['exact_duplicate_groups'])} exact duplicate groups).")


if __name__ == "__main__":
    main()
