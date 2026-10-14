"""Report the local environment without modifying it."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT, timeout=10).strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        return None


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    disk = shutil.disk_usage(root)
    report: dict[str, object] = {
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "project_free_gib": round(disk.free / 1024**3, 2),
        "tesseract": command_output(["tesseract", "--version"]),
        "nvidia_smi": command_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]),
        "packages": {name: package_version(name) for name in ("fastapi", "numpy", "opencv-python-headless", "Pillow", "pytesseract", "torch", "torchvision")},
    }
    try:
        import torch
        report["torch_cuda"] = {"available": torch.cuda.is_available(), "version": torch.version.cuda, "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    except ImportError:
        report["torch_cuda"] = {"available": False, "reason": "PyTorch is not installed; expected for the OCR-first baseline."}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
