"""Run the local baseline on one image and print auditable JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from nepali_banknote_scanner.pipeline import scan_image


def main() -> None:
    # PowerShell may still use a legacy code page; OCR evidence can be Devanagari.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--calendar-hint", choices=("auto", "ad", "bs"), default="auto")
    arguments = parser.parse_args()
    print(json.dumps(scan_image(arguments.image, arguments.calendar_hint), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
