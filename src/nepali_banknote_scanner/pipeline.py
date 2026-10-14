"""The review-first application pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .dates import CalendarHint, extract_date_candidates
from .ocr import read_image, run_ocr


def scan_image(path: Path, calendar_hint: CalendarHint = "auto") -> dict[str, Any]:
    image = read_image(path)
    ocr = run_ocr(image)
    candidates = [item.to_dict() for item in extract_date_candidates(ocr["raw_text"], calendar_hint)]
    # A candidate is never auto-accepted in this baseline: calendar identity
    # and field semantics must be verified from real labelled note examples.
    return {
        "filename": path.name,
        "image": {"width": int(image.shape[1]), "height": int(image.shape[0])},
        "ocr": ocr,
        "date_candidates": candidates,
        "normalized_bs_date": None,
        "source_calendar": "unknown",
        "valid": False,
        "requires_review": True,
        "review_reason": "No confirmed BS date is produced until target-field and calendar validation are established.",
    }
