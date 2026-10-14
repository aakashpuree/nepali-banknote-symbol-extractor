"""FastAPI entry point for local website integration."""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .dates import CalendarHint
from .ocr import ALLOWED_SUFFIXES, MAX_UPLOAD_BYTES, available_languages
from .pipeline import scan_image


def _origins() -> list[str]:
    value = os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
    return [origin.strip() for origin in value.split(",") if origin.strip()]


app = FastAPI(title="Nepali Banknote Scanner", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=_origins(), allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["Content-Type"])


@app.get("/health")
def health() -> dict[str, object]:
    try:
        languages = available_languages()
        return {"status": "ok", "tesseract_languages": languages, "required_languages_ready": {"eng", "nep"}.issubset(languages)}
    except RuntimeError as error:
        return {"status": "degraded", "detail": str(error), "required_languages_ready": False}


@app.post("/scan")
async def scan(image: UploadFile = File(...), calendar_hint: CalendarHint = Query("auto")) -> dict[str, object]:
    suffix = Path(image.filename or "upload").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=415, detail="Only JPEG and PNG images are accepted.")
    content = await image.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 15 MB upload limit.")
    if not content:
        raise HTTPException(status_code=422, detail="Image is empty.")
    temporary = NamedTemporaryFile(prefix="banknote-", suffix=suffix, delete=False)
    try:
        temporary.write(content)
        temporary.close()
        return scan_image(Path(temporary.name), calendar_hint)
    except (RuntimeError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    finally:
        Path(temporary.name).unlink(missing_ok=True)
