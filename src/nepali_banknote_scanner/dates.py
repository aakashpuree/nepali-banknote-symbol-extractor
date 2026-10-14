"""Conservative date extraction and normalisation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
import re
from typing import Literal

CalendarHint = Literal["auto", "ad", "bs"]
DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
DATE_RE = re.compile(r"(?<!\d)(\d{4})\s*[-/.]\s*(\d{1,2})\s*[-/.]\s*(\d{1,2})(?!\d)")


@dataclass(frozen=True)
class DateAssessment:
    original: str
    ascii_value: str
    normalized_date: str | None
    source_calendar: str
    valid: bool
    requires_review: bool
    reason: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def to_ascii_digits(value: str) -> str:
    """Convert Nepali digits while retaining all other OCR evidence."""
    return value.translate(DEVANAGARI_DIGITS)


def _format(year: int, month: int, day: int) -> str:
    return f"{year:04d}/{month:02d}/{day:02d}"


def _assess(year: int, month: int, day: int, original: str, hint: CalendarHint) -> DateAssessment:
    ascii_value = _format(year, month, day)
    if hint == "auto":
        return DateAssessment(original, ascii_value, None, "unknown", False, True, "Calendar was not identified; AD and BS must not be guessed.")
    if hint == "ad":
        try:
            date(year, month, day)
        except ValueError as error:
            return DateAssessment(original, ascii_value, None, "ad", False, True, str(error))
        return DateAssessment(original, ascii_value, None, "ad", True, True, "Valid AD date found. Configure a verified local BS calendar adapter before accepting conversion.")
    if not 1 <= month <= 12 or not 1 <= day <= 32:
        return DateAssessment(original, ascii_value, None, "bs", False, True, "Month or day is outside the possible BS range.")
    return DateAssessment(original, ascii_value, ascii_value, "bs", True, True, "Structurally valid BS candidate; confirm it against a verified calendar.")


def extract_date_candidates(text: str, calendar_hint: CalendarHint = "auto") -> list[DateAssessment]:
    """Return only explicitly separated YYYY/MM/DD-like values, in OCR order."""
    if calendar_hint not in {"auto", "ad", "bs"}:
        raise ValueError("calendar_hint must be one of: auto, ad, bs")
    assessments: list[DateAssessment] = []
    for match in DATE_RE.finditer(to_ascii_digits(text)):
        year, month, day = (int(part) for part in match.groups())
        assessments.append(_assess(year, month, day, match.group(0), calendar_hint))
    return assessments
