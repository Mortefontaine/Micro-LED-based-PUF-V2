"""Per-device template-based registration for raw micro-LED frames.

This is a stronger non-neural alternative to the first coarse registration
prototype. It uses the previously YOLO-selected 256x256 frames only to build a
reference style/template for each device, then registers raw 1280x1024 frames
with a small similarity-transform search.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import re
from collections import deque
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
RAW_ROOT = DATA_ROOT / "00_raw_rgb_M1_M6"
TARGET_ROOT = DATA_ROOT / "02_stn_pairs_M1_M6" / "target"
OUT_DIR = REPO_ROOT / "work" / "registration_preview"
IMAGE_SIZE = 256
DETECT_SIZE = 320
DEVICE_RE = re.compile(r"^(M\d+)_", re.IGNORECASE)


def natural_key(text: str) -> List[object]:
    parts: List[object] = []
    for part in re.split(r"(\d+)", text):
        parts.append(int(part) if part.isdigit() else part.lower())
    return parts


def get_font(size: int) -> ImageFont.ImageFont:
    for candidate in [r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\segoeui.ttf"]:
        try:
            return ImageFont.truetype(candidate, size)
        except Exception:
            pass
    return ImageFont.load_default()


def draw_text(draw: ImageDraw.ImageDraw, xy: Tuple[int, int], text: str, size: int = 14, fill=(30, 30, 30)) -> None:
    draw.text(xy, text, font=get_font(size), fill=fill)


def device_from_condition(name: str) -> str:
    match = DEVICE_RE.match(name)
    if not match:
        return "UNKNOWN"
    return match.group(1).upper()


def list_images(root: Path) -> List[Path]:
    return sorted((p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS), key=lambda p: natural_key(str(p)))


def normalize01(x: np.ndarray, lo_pct: float = 1, hi_pct: float = 99) -> np.ndarray:
    lo, hi = np.percentile(x, [lo_pct, hi_pct])
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0, 1).astype(np.float32)


def largest_component(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    best: List[Tuple[int, int]] = []
    for y in range(h):
        for x in range(w):
            if seen[y, x] or not mask[y, x]:
                continue
            q: deque[Tuple[int, int]] = deque([(y, x)])
            seen[y, x] = True
            comp: List[Tuple[int, int]] = []
            while q:
                cy, cx = q.popleft()
                comp.append((cy, cx))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and mask[ny, nx]:
                        seen[ny, nx] = True
                        q.append((ny, nx))
            if len(comp) > len(best):
                best = comp
    out = np.zeros_like(mask, dtype=bool)
    for y, x in best:
        out[y, x] = True
    return out


def active_component_from_blue(blue: np.ndarray) -> Tuple[np.ndarray, float]:
    p995 = float(np.percentile(blue, 99.5))
    p50 = float(np.percentile(blue, 50))
    # Use a conservative threshold for the bright emitting body, not only the
    # very brightest pixels.
    threshold = max(p50 + 0.10 * (p995 - p50), p995 * 0.12, 0.025)
    comp = largest_component(blue >= threshold)
    if comp.sum() < 8:
        threshold = max(p50 + 0.07 * (p995 - p50), p995 * 0.08, 0.02)
        comp = largest_component(blue >= threshold)
    return comp, threshold


def radial_residual(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    h, w = arr.shape
    yy, xx = np.indices((h, w))
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    radius = np.floor(np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)).astype(np.int32)
    flat_r = radius.ravel()
    flat = arr.ravel()
    sums = np.bincount(flat_r, weights=flat)
    counts = np.bincount(flat_r)
    means = sums / np.maximum(counts, 1)
    return (flat - means[flat_r]).reshape(h, w).astype(np.float32)


def norm_corr_image(img: Image.Image) -> np.ndarray:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    blue = normalize01(arr[:, :, 2], 1, 99)
    residual = radial_residual(blue)
    residual = residual - float(residual.mean())
    std = float(residual.std())
    return residual / std if std > 1e-9 else np.zeros_like(residual)


def corr_score(img: Image.Image, ref: np.ndarray) -> float:
    return float((norm_corr_image(img) * ref).mean())


def wrap_angle(angle: float) -> float:
    while angle <= -180:
        angle += 360
    while angle > 180:
        angle -= 360
    return angle


def dark_axis_angle(img: Image.Image) -> float:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    blue = normalize01(arr[:, :, 2], 1, 99)
    active, _ = active_component_from_blue(blue)
    if active.sum() < 20:
        return 0.0
    active_vals = blue[active]
    dark_level = float(np.percentile(active_vals, 65))
    weights = np.clip(dark_level - blue, 0, None) * active.astype(np.float32)
    if float(weights.sum()) <= 1e-6:
        return 0.0
    yy, xx = np.indices(blue.shape)
    wsum = float(weights.sum())
    cx = float((weights * xx).sum() / wsum)
    cy = float((weights * yy).sum() / wsum)
    x = (xx - cx).ravel()
    y = (yy - cy).ravel()
    ww = weights.ravel()
    cxx = float((ww * x * x).sum() / wsum)
    cyy = float((ww * y * y).sum() / wsum)
    cxy = float((ww * x * y).sum() / wsum)
    cov = np.array([[cxx, cxy], [cxy, cyy]], dtype=np.float32)
    vals, vecs = np.linalg.eigh(cov)
    vx, vy = vecs[:, int(np.argmax(vals))]
    return wrap_angle(math.degrees(math.atan2(float(vy), float(vx))))


def build_reference_templates(target_root: Path, max_per_device: int = 180) -> Dict[str, dict]:
    grouped: Dict[str, List[Path]] = {}
    for path in list_images(target_root):
        device = device_from_condition(path.parent.name)
        if device != "UNKNOWN":
            grouped.setdefault(device, []).append(path)
    refs: Dict[str, dict] = {}
    for device, files in sorted(grouped.items(), key=lambda kv: natural_key(kv[0])):
        step = max(1, len(files) // max_per_device)
        selected = files[::step][:max_per_device]
        blues = []
        for path in selected:
            img = Image.open(path).convert("RGB")
            if img.size != (IMAGE_SIZE, IMAGE_SIZE):
                img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
            arr = np.asarray(img, dtype=np.float32) / 255.0
            blues.append(arr[:, :, 2])
        mean_blue = np.mean(np.stack(blues, axis=0), axis=0)
        mean_img = Image.fromarray(np.uint8(normalize01(mean_blue, 1, 99) * 255), "L").convert("RGB")
        ref = norm_corr_image(mean_img)
        comp, _ = active_component_from_blue(normalize01(mean_blue, 1, 99))
        ys, xs = np.where(comp)
        target_side = float(max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)) if xs.size else 198.0
        refs[device] = {
            "device": device,
            "ref": ref,
            "mean_img": mean_img,
            "dark_axis_angle": dark_axis_angle(mean_img),
            "target_active_side": target_side,
            "n_images": len(selected),
        }
    return refs


def detect_raw_active(img: Image.Image) -> Tuple[dict, np.ndarray]:
    w, h = img.size
    detect_h = int(round(h * DETECT_SIZE / w))
    small = img.resize((DETECT_SIZE, detect_h), Image.Resampling.BILINEAR).convert("RGB")
    arr = np.asarray(small, dtype=np.float32) / 255.0
    blue = arr[:, :, 2]
    comp, threshold = active_component_from_blue(blue)
    ys, xs = np.where(comp)
    if xs.size == 0:
        raise RuntimeError("active component not found")
    weights = np.maximum(blue[ys, xs] - threshold, 1e-4)
    cx_small = float(np.average(xs, weights=weights))
    cy_small = float(np.average(ys, weights=weights))
    sx = w / DETECT_SIZE
    sy = h / detect_h
    bbox_w = float(xs.max() - xs.min() + 1) * sx
    bbox_h = float(ys.max() - ys.min() + 1) * sy
    info = {
        "cx": cx_small * sx,
        "cy": cy_small * sy,
        "bbox_w": bbox_w,
        "bbox_h": bbox_h,
        "threshold": threshold,
        "component_pixels_detect": int(comp.sum()),
    }
    return info, comp


def square_crop_resize(img: Image.Image, cx: float, cy: float, side: float) -> Image.Image:
    w, h = img.size
    left = cx - side / 2
    top = cy - side / 2
    right = cx + side / 2
    bottom = cy + side / 2
    canvas_side = int(math.ceil(side))
    canvas = Image.new("RGB", (canvas_side, canvas_side), (0, 0, 0))
    src_left = max(0, int(math.floor(left)))
    src_top = max(0, int(math.floor(top)))
    src_right = min(w, int(math.ceil(right)))
    src_bottom = min(h, int(math.ceil(bottom)))
    crop = img.crop((src_left, src_top, src_right, src_bottom))
    canvas.paste(crop, (src_left - int(math.floor(left)), src_top - int(math.floor(top))))
    return canvas.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)


def shift_image(img: Image.Image, dx: int, dy: int) -> Image.Image:
    out = Image.new("RGB", img.size, (0, 0, 0))
    src_x0 = max(0, -dx)
    src_y0 = max(0, -dy)
    src_x1 = min(img.size[0], img.size[0] - dx)
    src_y1 = min(img.size[1], img.size[1] - dy)
    if src_x1 <= src_x0 or src_y1 <= src_y0:
        return out
    patch = img.crop((src_x0, src_y0, src_x1, src_y1))
    out.paste(patch, (max(0, dx), max(0, dy)))
    return out


def register_image(img: Image.Image, ref_payload: dict) -> Tuple[Image.Image, dict]:
    raw_info, _ = detect_raw_active(img)
    raw_side = max(float(raw_info["bbox_w"]), float(raw_info["bbox_h"]))
    target_side = float(ref_payload["target_active_side"])
    base_crop_side = raw_side * IMAGE_SIZE / max(1.0, target_side)

    ref = ref_payload["ref"]
    target_dark_angle = float(ref_payload.get("dark_axis_angle", 0.0))
    best = {"score": -1e9}
    scale_factors = (0.92, 1.00, 1.08)
    for sf in scale_factors:
        side = base_crop_side * sf
        coarse = square_crop_resize(img, float(raw_info["cx"]), float(raw_info["cy"]), side)
        raw_dark_angle = dark_axis_angle(coarse)
        deltas = [
            wrap_angle(target_dark_angle - raw_dark_angle),
            wrap_angle(raw_dark_angle - target_dark_angle),
            wrap_angle(target_dark_angle - raw_dark_angle + 180),
            wrap_angle(raw_dark_angle - target_dark_angle + 180),
        ]
        candidate_angles = set()
        for delta in deltas:
            for offset in range(-30, 31, 10):
                candidate_angles.add(int(round(wrap_angle(delta + offset))))
        # Keep a sparse global fallback for cases where dark-axis PCA is weak.
        candidate_angles.update(range(-180, 180, 36))
        for angle in sorted(candidate_angles):
            rot = coarse.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0))
            score = corr_score(rot, ref)
            if score > best["score"]:
                best = {"score": score, "side": side, "scale_factor": sf, "angle": float(angle), "dx": 0, "dy": 0, "image": rot}

    # Refine scale, rotation, and a small output-plane translation.
    refine_sides = [best["side"] * sf for sf in (0.98, 1.0, 1.02)]
    refine_angles = range(int(best["angle"] - 8), int(best["angle"] + 9), 4)
    translations = (-6, 0, 6)
    for side in refine_sides:
        coarse = square_crop_resize(img, float(raw_info["cx"]), float(raw_info["cy"]), side)
        for angle in refine_angles:
            rot = coarse.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0))
            for dy in translations:
                for dx in translations:
                    shifted = shift_image(rot, dx, dy)
                    score = corr_score(shifted, ref)
                    if score > best["score"]:
                        best = {
                            "score": score,
                            "side": side,
                            "scale_factor": side / base_crop_side,
                            "angle": float(angle),
                            "dx": dx,
                            "dy": dy,
                            "image": shifted,
                        }
    info = {
        **raw_info,
        "base_crop_side": base_crop_side,
        "crop_side": float(best["side"]),
        "scale_factor": float(best["scale_factor"]),
        "angle_deg": float(best["angle"]),
        "dx": int(best["dx"]),
        "dy": int(best["dy"]),
        "corr_score": float(best["score"]),
        "target_active_side": target_side,
    }
    return best["image"], info


def process_one(path: Path, raw_root: Path, out_root: Path, refs: Dict[str, dict]) -> dict:
    device = device_from_condition(path.parent.name)
    ref_payload = refs.get(device) or next(iter(refs.values()))
    img = Image.open(path).convert("RGB")
    aligned, info = register_image(img, ref_payload)
    rel = path.relative_to(raw_root)
    out_path = out_root / "registered_256" / rel.with_suffix(".png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    aligned.save(out_path)
    return {
        "input_path": str(path),
        "output_path": str(out_path),
        "condition": rel.parent.as_posix(),
        "frame": rel.name,
        "device": device,
        **info,
    }


def make_preview(rows: Sequence[dict], refs: Dict[str, dict], path: Path) -> None:
    selected = list(rows)[:24]
    cols = 4
    tile_w, tile_h = 620, 232
    canvas = Image.new("RGB", (cols * tile_w, math.ceil(len(selected) / cols) * tile_h + 72), "white")
    draw = ImageDraw.Draw(canvas)
    draw_text(draw, (28, 20), "Level-2 per-device template registration preview", 26)
    for i, row in enumerate(selected):
        raw = Image.open(row["input_path"]).convert("RGB")
        raw.thumbnail((170, 170))
        reg = Image.open(row["output_path"]).convert("RGB").resize((170, 170), Image.Resampling.NEAREST)
        ref = refs[row["device"]]["mean_img"].resize((170, 170), Image.Resampling.NEAREST)
        x = (i % cols) * tile_w + 20
        y = (i // cols) * tile_h + 75
        canvas.paste(raw, (x, y))
        canvas.paste(reg, (x + 190, y))
        canvas.paste(ref, (x + 390, y))
        draw_text(draw, (x, y + 176), "raw", 12)
        draw_text(draw, (x + 190, y + 176), f"registered a={row['angle_deg']:.0f}, s={row['scale_factor']:.2f}", 12)
        draw_text(draw, (x + 390, y + 176), f"{row['device']} reference", 12)
        draw_text(draw, (x, y + 194), Path(row["input_path"]).parent.name + "/" + Path(row["input_path"]).name, 10, (80, 80, 80))
    canvas.save(path, quality=95)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--target-root", type=Path, default=TARGET_ROOT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--max-total", type=int, default=240)
    parser.add_argument("--per-condition", type=int, default=3)
    parser.add_argument("--sample-seed", type=int, default=20260706)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    refs = build_reference_templates(args.target_root)
    ref_dir = args.out_dir / "device_references"
    ref_dir.mkdir(exist_ok=True)
    for device, payload in refs.items():
        payload["mean_img"].save(ref_dir / f"{device}_reference_style.png")

    files = list_images(args.raw_root)
    if args.per_condition > 0:
        grouped: Dict[Path, List[Path]] = {}
        for path in files:
            grouped.setdefault(path.parent, []).append(path)
        capped: List[Path] = []
        for _, group in sorted(grouped.items(), key=lambda kv: natural_key(str(kv[0]))):
            capped.extend(group[: args.per_condition])
        files = capped
    if args.max_total and len(files) > args.max_total:
        rng = random.Random(args.sample_seed)
        files = sorted(rng.sample(files, args.max_total), key=lambda p: natural_key(str(p)))

    rows = []
    for idx, path in enumerate(files, 1):
        try:
            rows.append(process_one(path, args.raw_root, args.out_dir, refs))
        except Exception as exc:
            rows.append({
                "input_path": str(path),
                "output_path": "",
                "condition": path.parent.name,
                "frame": path.name,
                "device": device_from_condition(path.parent.name),
                "error": str(exc),
            })
        if idx % 20 == 0 or idx == len(files):
            print(f"processed {idx}/{len(files)}")

    fieldnames = [
        "input_path",
        "output_path",
        "condition",
        "frame",
        "device",
        "angle_deg",
        "scale_factor",
        "dx",
        "dy",
        "corr_score",
        "base_crop_side",
        "crop_side",
        "target_active_side",
        "component_pixels_detect",
        "bbox_w",
        "bbox_h",
        "cx",
        "cy",
        "error",
    ]
    with (args.out_dir / "registration_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    ok_rows = [row for row in rows if row.get("output_path")]
    if ok_rows:
        make_preview(ok_rows, refs, args.out_dir / "registration_preview_montage.jpg")

    report = args.out_dir / "registration_report.md"
    with report.open("w", encoding="utf-8") as f:
        f.write("# Level-2 Per-Device Template Registration Preview\n\n")
        f.write("## Method\n\n")
        f.write("```text\n")
        f.write("raw image\n")
        f.write("-> blue active-component detection\n")
        f.write("-> crop scale initialized from per-device target active size\n")
        f.write("-> coarse scale/rotation search\n")
        f.write("-> refined scale/rotation/translation search\n")
        f.write("-> 256x256 registered frame\n")
        f.write("```\n\n")
        f.write("## Summary\n\n")
        f.write(f"- Processed frames: {len(rows)}\n")
        f.write(f"- Successful frames: {len(ok_rows)}\n")
        f.write(f"- Devices with references: {', '.join(sorted(refs.keys(), key=natural_key))}\n")
        if ok_rows:
            angles = np.asarray([float(row["angle_deg"]) for row in ok_rows], dtype=np.float32)
            scores = np.asarray([float(row["corr_score"]) for row in ok_rows], dtype=np.float32)
            scales = np.asarray([float(row["scale_factor"]) for row in ok_rows], dtype=np.float32)
            f.write(f"- Rotation angle mean/std: {angles.mean():.2f} / {angles.std():.2f} deg\n")
            f.write(f"- Scale factor mean/std: {scales.mean():.3f} / {scales.std():.3f}\n")
            f.write(f"- Correlation score mean/std: {scores.mean():.4f} / {scores.std():.4f}\n")
        f.write("\n## Files\n\n")
        f.write("- `registered_256/`: normalized output frames\n")
        f.write("- `registration_preview_montage.jpg`: raw / registered / per-device reference comparison\n")
        f.write("- `device_references/`: per-device reference styles built from previous YOLO-selected frames\n")
        f.write("- `registration_summary.csv`: transform diagnostics\n")
    print(f"Report: {report}")


if __name__ == "__main__":
    main()
