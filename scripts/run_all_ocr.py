"""Run the OCR baseline over every raw image and write auditable JSON artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from nepali_banknote_scanner.pipeline import scan_image


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    input_dir = root / "data" / "raw" / "images"
    output_dir = root / "artifacts" / "ocr"
    output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(
        path for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    print(f"Found {len(images)} images in {input_dir}")

    total_words = 0
    total_candidates = 0
    failed: list[tuple[str, str]] = []
    for index, image_path in enumerate(images, 1):
        output_path = output_dir / f"{image_path.stem}.json"
        try:
            result = scan_image(image_path)
            output_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            word_count = len(result["ocr"]["words"])
            candidate_count = len(result["date_candidates"])
            total_words += word_count
            total_candidates += candidate_count
            print(
                f"[{index:02d}/{len(images)}] {image_path.name}: "
                f"{word_count} words, {candidate_count} date candidates, "
                f"confidence={result['ocr']['mean_word_confidence']}"
            )
        except (RuntimeError, ValueError, OSError) as error:
            failed.append((image_path.name, str(error)))
            print(f"[{index:02d}/{len(images)}] {image_path.name}: FAILED: {error}")

    summary = {
        "image_count": len(images),
        "processed": len(images) - len(failed),
        "failed": len(failed),
        "total_words": total_words,
        "total_date_candidates": total_candidates,
        "failed_files": [{"filename": name, "error": error} for name, error in failed],
    }
    (output_dir / "_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
PY