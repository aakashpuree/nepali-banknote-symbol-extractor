"""Run the local baseline on one image and print auditable JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nepali_banknote_scanner.pipeline import scan_image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--calendar-hint", choices=("auto", "ad", "bs"), default="auto")
    arguments = parser.parse_args()
    print(json.dumps(scan_image(arguments.image, arguments.calendar_hint), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
