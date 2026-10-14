# Nepali Banknote Scanner

A local, review-first OCR baseline for Nepali banknote photographs. It accepts
JPEG/PNG uploads, uses local Tesseract OCR with Nepali and English language
data, preserves word-level evidence, and returns only explicit date-shaped
values. It does **not** assume serial numbers, denominations, or decorative
symbols are dates.

## Current data status

The supplied dataset has 55 unmodified JPEGs in `data/raw/images/`:

- 22 camera captures at 3120×4160;
- 33 gallery-derived images at 720×1600;
- no exact SHA-256 duplicate files;
- no bounding-box/text annotations or verified calendar labels.

The images depict multiple note denominations, but the samples do not establish
that a date field exists. The API consequently returns `normalized_bs_date:
null` and `requires_review: true` unless a date is explicitly found and its
calendar workflow has been verified.

## Safe Windows setup

NVIDIA reports an RTX 5050 Laptop GPU, driver 610.74, and 8151 MiB VRAM. The
project drive has roughly 83 GiB free. The development setup keeps Python,
Tesseract, language data, package cache, and `.venv` under this project and
does not alter global Python installations.

1. For a fresh clone, install [uv](https://docs.astral.sh/uv/getting-started/installation/) and run the following from PowerShell. It downloads a managed Python 3.13 and makes a project-local environment.

   ```powershell
   uv venv .venv --python 3.13 --managed-python
   uv pip install --python .\.venv\Scripts\python.exe -r requirements-dev.txt
   uv pip install --python .\.venv\Scripts\python.exe -e .
   .\.venv\Scripts\python.exe -m pytest -q
   .\.venv\Scripts\python.exe scripts\inspect_dataset.py
   .\.venv\Scripts\python.exe scripts\environment_report.py
   ```

2. Install 64-bit Tesseract for Windows, then place `eng` and `nep` traineddata in its `tessdata` directory. This workspace already has a local engine at `.tools/tesseract`; confirm it with `.\.tools\tesseract\tesseract.exe --list-langs`.

3. Baseline a real note entirely offline:

   ```powershell
   .\.venv\Scripts\python.exe scripts\run_ocr.py data\raw\images\IMG_20250808_075449.jpg
   ```

The local engine is discovered automatically. To use a different one, set
`TESSERACT_CMD` in the launching shell. The baseline does not require PyTorch
or a GPU. Do not install an object detector until OCR has been measured against
reviewed labels.

## Run the local API

```powershell
.\.venv\Scripts\python.exe -m uvicorn nepali_banknote_scanner.api:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` for the local interactive API. A website can
send multipart form data to `POST /scan` with the image file field named
`image`. The optional `calendar_hint` accepts `auto`, `ad`, or `bs`; leave it
as `auto` unless the website has separately verified the source calendar. JSON
includes original OCR words, confidence values, date candidates, a BS date only
when justified, and a review flag.

## Troubleshooting

- **Tesseract not found:** install it, add its folder to PATH, or set
  `TESSERACT_CMD` to its full executable path.
- **Missing `nep`:** add `nep.traineddata` to Tesseract's `tessdata` folder and
  verify `tesseract --list-langs` shows both `eng` and `nep`.
- **CUDA unavailable:** the baseline stays usable on CPU. Run
  `python scripts/environment_report.py` after any PyTorch installation; add a
  detector only when it reports CUDA available and reviewed labels require it.
- **Memory or disk pressure:** keep originals in `data/raw/images`, do not cache
  large models, and use a small detector only after a labelled evaluation
  requires it.
- **Dependency conflict:** delete only this project's `.venv`, recreate it with
  Python 3.13, and reinstall `requirements-dev.txt`; do not modify a global
  Python installation.
