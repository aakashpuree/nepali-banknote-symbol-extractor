"""Offline OCR with transparent preprocessing and evidence preservation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytesseract

ALLOWED_SUFFIXES = {".jpeg", ".jpg", ".png"}
MAX_UPLOAD_BYTES = 15 * 1024 * 1024


def configure_tesseract() -> None:
    """Prefer an explicit executable, then the isolated project-local engine."""
    command = os.environ.get("TESSERACT_CMD")
    if not command:
        local_engine = Path(__file__).resolve().parents[2] / ".tools" / "tesseract" / "tesseract.exe"
        if local_engine.is_file():
            command = str(local_engine)
    if command:
        pytesseract.pytesseract.tesseract_cmd = command


def read_image(path: Path) -> np.ndarray:
    if path.suffix.lower() not in ALLOWED_SUFFIXES:
        raise ValueError("Only JPEG and PNG images are accepted.")
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("The file is not a readable image.")
    return image


def preprocess(image: np.ndarray) -> np.ndarray:
    """Create one conservative grayscale OCR input without altering the source."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    longest_side = max(gray.shape)
    if longest_side < 1600:
        scale = 1600 / longest_side
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return cv2.fastNlMeansDenoising(gray, None, h=7, templateWindowSize=7, searchWindowSize=21)


def available_languages() -> list[str]:
    configure_tesseract()
    try:
        return sorted(pytesseract.get_languages(config=""))
    except pytesseract.TesseractNotFoundError as error:
        raise RuntimeError("Tesseract was not found. Install it or set TESSERACT_CMD.") from error


def run_ocr(image: np.ndarray, languages: str = "nep+eng") -> dict[str, Any]:
    configure_tesseract()
    installed = available_languages()
    missing = [language for language in languages.split("+") if language not in installed]
    if missing:
        raise RuntimeError(f"Missing Tesseract language data: {', '.join(missing)}")
    prepared = preprocess(image)
    try:
        data = pytesseract.image_to_data(prepared, lang=languages, config="--oem 1 --psm 6", output_type=pytesseract.Output.DICT)
    except pytesseract.TesseractNotFoundError as error:
        raise RuntimeError("Tesseract was not found. Install it or set TESSERACT_CMD.") from error
    words = []
    for index, word in enumerate(data["text"]):
        word = word.strip()
        try:
            confidence = float(data["conf"][index])
        except (ValueError, TypeError):
            confidence = -1.0
        if word:
            words.append({"text": word, "confidence": confidence})
    valid_confidences = [word["confidence"] for word in words if word["confidence"] >= 0]
    return {
        "raw_text": " ".join(word["text"] for word in words),
        "words": words,
        "mean_word_confidence": round(sum(valid_confidences) / len(valid_confidences), 2) if valid_confidences else None,
        "languages": languages,
        "preprocess": "grayscale + conditional upscaling + light denoising",
    }
