"""Scan data/raw/images, auto-crop the banknote, save to data/cropped_raw.

Usage (PowerShell, from repo root):
    .\\.venv\\Scripts\\python.exe scripts\\crop_banknotes.py
    .\\.venv\\Scripts\\python.exe scripts\\crop_banknotes.py --input-dir data\\raw\\images --output-dir data\\cropped_raw --overwrite

How it works:
  1. Trims black letterbox bars (gallery screenshots have them).
  2. Builds two masks: Canny edges + HSV-saturation (pink border vs white paper).
  3. Scores every 4-point contour by area + banknote aspect ratio (~1.5-2.8).
  4. Perspective-warps the best quad to a straight landscape rectangle.
  5. Falls back to axis-aligned saturation bbox (no bad rotation) if no quad.
  6. Saves cropped JPGs with the same filenames into the output folder.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def trim_black_bars(gray: np.ndarray, thresh: int = 20) -> tuple[int, int]:
    h = gray.shape[0]
    row_mean = gray.mean(axis=1)
    mask = row_mean > thresh
    top = 0
    while top < h and not mask[top]:
        top += 1
    bottom = h
    while bottom > top and not mask[bottom - 1]:
        bottom -= 1
    if (top > int(h * 0.02) or bottom < int(h * 0.98)) and (bottom - top) > int(h * 0.5):
        return top, bottom
    return 0, h


def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def expand_quad(quad: np.ndarray, img_w: int, img_h: int, pad: float) -> np.ndarray:
    if pad <= 0:
        return quad
    center = quad.mean(axis=0)
    expanded = center + (quad - center) * (1.0 + pad)
    expanded[:, 0] = np.clip(expanded[:, 0], 0, img_w - 1)
    expanded[:, 1] = np.clip(expanded[:, 1], 0, img_h - 1)
    return expanded.astype("float32")


def quad_score(contour_area: float, img_area: float, quad: np.ndarray) -> float:
    """Higher is better. Rejects tiny / wrong-aspect quads with low score."""
    area_ratio = contour_area / max(img_area, 1)
    if area_ratio < 0.07:  # note fills at least ~15-50%; tiny = interior detail
        return 0.0
    rect = cv2.minAreaRect(quad.astype(np.float32))
    rw, rh = rect[1]
    if min(rw, rh) < 1:
        return 0.0
    ratio = max(rw, rh) / min(rw, rh)  # want ~1.6-2.4 for banknotes
    if 1.4 <= ratio <= 2.9:
        aspect = 1.0
    elif 1.2 <= ratio <= 3.5:
        aspect = 0.5
    else:
        aspect = 0.15
    # Prefer quads that fill more of their bounding rect (solid, not diagonal slivers)
    x, y, bw, bh = cv2.boundingRect(quad.astype(np.int32))
    fill = contour_area / max(float(bw * bh), 1.0)
    return area_ratio * aspect * (0.5 + 0.5 * fill)


def collect_quad_candidates(mask: np.ndarray, img_area: float) -> list[tuple[float, np.ndarray]]:
    out: list[tuple[float, np.ndarray]] = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < img_area * 0.05:
            continue
        found = False
        for eps in (0.02, 0.035, 0.06):
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, eps * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                quad = approx.reshape(4, 2).astype(np.float32)
                score = quad_score(area, img_area, quad)
                if score > 0:
                    out.append((score, quad))
                found = True
                break  # coarsest successful epsilon is enough per contour
        # Rescue wavy/curved borders (approx not convex): use rotated rect box.
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect).astype(np.float32)
        score_box = quad_score(area, img_area, box) * 0.85
        if score_box > 0:
            out.append((score_box, box))
    return out


def find_note_quad(image: np.ndarray) -> tuple[np.ndarray | None, str]:
    orig_h, orig_w = image.shape[:2]
    max_side = 1000
    scale = min(1.0, max_side / max(orig_h, orig_w))
    small = cv2.resize(image, (int(orig_w * scale), int(orig_h * scale))) if scale < 1.0 else image.copy()
    sh, sw = small.shape[:2]
    img_area = float(sh * sw)

    # Mask A: edges (good for sharp close-ups on white paper).
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 40, 120)
    k9 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    mask_a = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k9, iterations=2)
    mask_a = cv2.dilate(mask_a, k9, iterations=1)

    # Mask B: saturation (pink printed border vs white background).
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    sat = cv2.medianBlur(sat, 5)
    _, mask_b = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Otsu can flip on mostly-white images; ensure foreground is the smaller part.
    if (mask_b > 0).mean() > 0.5:
        mask_b = cv2.bitwise_not(mask_b)
    k15 = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mask_b = cv2.morphologyEx(mask_b, cv2.MORPH_CLOSE, k15, iterations=2)
    k5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask_b = cv2.morphologyEx(mask_b, cv2.MORPH_OPEN, k5, iterations=1)

    candidates = collect_quad_candidates(mask_a, img_area)
    candidates += collect_quad_candidates(mask_b, img_area)
    if not candidates:
        return None, "none"
    candidates.sort(key=lambda t: t[0], reverse=True)
    best_score, best_quad = candidates[0]
    best_quad = best_quad / scale
    return order_points(best_quad), f"quad({best_score:.3f})"


def saturation_bbox(image: np.ndarray, scale_max: int = 800) -> tuple[int, int, int, int] | None:
    """Axis-aligned fallback box from saturation mask. Returns (x, y, w, h) or None."""
    h, w = image.shape[:2]
    scale = min(1.0, scale_max / max(h, w))
    small = cv2.resize(image, (int(w * scale), int(h * scale))) if scale < 1.0 else image
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    sat = cv2.medianBlur(hsv[:, :, 1], 5)
    _, mask = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if (mask > 0).mean() > 0.5:
        mask = cv2.bitwise_not(mask)
    k15 = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k15, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    area_ratio = cv2.contourArea(cnt) / float(small.shape[0] * small.shape[1])
    if area_ratio < 0.05:
        return None
    x, y, bw, bh = cv2.boundingRect(cnt)
    return int(x / scale), int(y / scale), int(bw / scale), int(bh / scale)


def warp_quad(image: np.ndarray, quad: np.ndarray) -> np.ndarray:
    (tl, tr, br, bl) = quad
    max_w = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    max_h = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    max_w, max_h = max(max_w, 10), max(max_h, 10)
    dst = np.array([[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]], dtype="float32")
    matrix = cv2.getPerspectiveTransform(quad.astype("float32"), dst)
    return cv2.warpPerspective(image, matrix, (max_w, max_h))


def crop_one(src: Path, dst: Path, padding: float) -> str:
    image = cv2.imread(str(src))
    if image is None:
        return "unreadable"
    h, w = image.shape[:2]

    gray_full = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    top, bottom = trim_black_bars(gray_full)
    if bottom - top < h:
        image = image[top:bottom, :]
        h, w = image.shape[:2]

    quad, _info = find_note_quad(image)
    if quad is not None:
        quad = expand_quad(quad, w, h, pad=padding)
        warped = warp_quad(image, quad)
        # Only rotate true perspective warps: note photographed sideways.
        # quad is ordered TL,TR,BR,BL so top-edge vs left-edge lengths tell orientation.
        top_len = float(np.linalg.norm(quad[1] - quad[0]))
        left_len = float(np.linalg.norm(quad[3] - quad[0]))
        if left_len > top_len * 1.15:
            warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
        dst.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst), warped, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        return "quad"

    # Fallback: axis-aligned saturation box (keeps horizontal notes horizontal).
    box = saturation_bbox(image)
    if box is not None:
        x, y, bw, bh = box
        pad_px_w, pad_px_h = int(bw * padding), int(bh * padding)
        x0, y0 = max(0, x - pad_px_w), max(0, y - pad_px_h)
        x1, y1 = min(w, x + bw + pad_px_w), min(h, y + bh + pad_px_h)
        crop = image[y0:y1, x0:x1]
        if crop.size > 0:
            dst.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dst), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            return "bbox"

    m = 0.04
    crop = image[int(h * m): int(h * (1 - m)), int(w * m): int(w * (1 - m))]
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return "fallback-center"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Auto-crop banknotes from raw photos.")
    parser.add_argument("--input-dir", type=Path, default=root / "data" / "raw" / "images")
    parser.add_argument("--output-dir", type=Path, default=root / "data" / "cropped_raw")
    parser.add_argument("--padding", type=float, default=0.02)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    files = sorted(p for p in args.input_dir.iterdir() if p.suffix.lower() in SUPPORTED)
    if not files:
        print(f"No images found in {args.input_dir}")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats: dict[str, int] = {}
    for src in files:
        dst = args.output_dir / (src.stem + ".jpg")
        if dst.exists() and not args.overwrite:
            stats["skipped"] = stats.get("skipped", 0) + 1
            continue
        try:
            status = crop_one(src, dst, padding=args.padding)
        except Exception as exc:
            print(f"FAIL {src.name}: {exc}")
            stats["unreadable"] = stats.get("unreadable", 0) + 1
            continue
        stats[status] = stats.get(status, 0) + 1
        print(f"{status:>15}  {src.name} -> {dst.name}")

    print(f"\nDone: {len(files)} scanned | {stats}\nOutput: {args.output_dir}")


if __name__ == "__main__":
    main()
