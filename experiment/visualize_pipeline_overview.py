from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


COLORS = {
    "bg": (190, 190, 190),
    "os0": (220, 45, 45),
    "os1": (40, 170, 75),
    "merged": (65, 105, 225),
    "foreground": (210, 35, 210),
    "axis": (40, 40, 40),
}


def read_pcd(path: Path) -> tuple[list[str], np.ndarray]:
    header = []
    with path.open("rb") as f:
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"PCD header has no DATA line: {path}")
            text = line.decode("ascii", errors="replace").strip()
            header.append(text)
            if text.startswith("DATA "):
                break
        payload = f.read()

    fields = next(x for x in header if x.startswith("FIELDS ")).split()[1:]
    sizes = [int(x) for x in next(x for x in header if x.startswith("SIZE ")).split()[1:]]
    types = next(x for x in header if x.startswith("TYPE ")).split()[1:]
    points = int(next(x for x in header if x.startswith("POINTS ")).split()[1])

    if any(size != 4 for size in sizes) or any(tp != "F" for tp in types):
        raise ValueError(f"Only float32 PCD fields are supported: {path}")

    arr = np.frombuffer(payload, dtype=np.float32).reshape(points, len(fields)).copy()
    return fields, arr


def xyz(path: Path) -> np.ndarray:
    fields, arr = read_pcd(path)
    idx = [fields.index(k) for k in ("x", "y", "z")]
    pts = arr[:, idx]
    return pts[np.isfinite(pts).all(axis=1)]


def sample_points(points: np.ndarray, max_points: int = 60000) -> np.ndarray:
    if len(points) <= max_points:
        return points
    step = max(1, len(points) // max_points)
    return points[::step][:max_points]


def world_to_image(points: np.ndarray, bounds: tuple[float, float, float, float], size: int, pad: int) -> np.ndarray:
    xmin, xmax, ymin, ymax = bounds
    x = (points[:, 0] - xmin) / (xmax - xmin)
    y = (points[:, 1] - ymin) / (ymax - ymin)
    px = pad + x * (size - 2 * pad)
    py = size - pad - y * (size - 2 * pad)
    return np.stack([px, py], axis=1).astype(np.int32)


def draw_points(draw: ImageDraw.ImageDraw, points: np.ndarray, bounds, size, pad, color, radius=1):
    if len(points) == 0:
        return
    mask = (
        (points[:, 0] >= bounds[0]) & (points[:, 0] <= bounds[1]) &
        (points[:, 1] >= bounds[2]) & (points[:, 1] <= bounds[3])
    )
    pix = world_to_image(sample_points(points[mask]), bounds, size, pad)
    for x, y in pix:
        draw.rectangle((x - radius, y - radius, x + radius, y + radius), fill=color)


def draw_frame(draw: ImageDraw.ImageDraw, bounds, size, pad, title: str, subtitle: str):
    font = ImageFont.load_default()
    draw.rectangle((0, 0, size - 1, size - 1), outline=(210, 210, 210))
    draw.text((18, 14), title, fill=(15, 15, 15), font=font)
    draw.text((18, 30), subtitle, fill=(70, 70, 70), font=font)

    # Coordinate axes through origin if visible.
    xmin, xmax, ymin, ymax = bounds
    if xmin <= 0 <= xmax:
        x0 = int(pad + (0 - xmin) / (xmax - xmin) * (size - 2 * pad))
        draw.line((x0, pad, x0, size - pad), fill=(225, 225, 225))
    if ymin <= 0 <= ymax:
        y0 = int(size - pad - (0 - ymin) / (ymax - ymin) * (size - 2 * pad))
        draw.line((pad, y0, size - pad, y0), fill=(225, 225, 225))

    draw.text((pad, size - pad + 8), "x/y top view, meters", fill=COLORS["axis"], font=font)
    draw.text((pad, size - pad + 22), f"x [{xmin}, {xmax}], y [{ymin}, {ymax}]", fill=COLORS["axis"], font=font)


def save_scene(out_path: Path, layers, bounds, title: str, subtitle: str, size: int = 980):
    pad = 70
    img = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)
    draw_frame(draw, bounds, size, pad, title, subtitle)
    for points, color, radius in layers:
        draw_points(draw, points, bounds, size, pad, color, radius)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def foreground_by_voxel(scene: np.ndarray, background: np.ndarray, voxel: float = 0.05) -> np.ndarray:
    # Approximation of pcd_annotate.py background removal: keep voxels not seen
    # in the background frame. This is for visualization, not label generation.
    scene_q = np.floor(scene / voxel).astype(np.int32)
    bg_q = np.floor(background / voxel).astype(np.int32)
    bg_keys = set(map(tuple, bg_q))
    keep = np.fromiter((tuple(k) not in bg_keys for k in scene_q), dtype=bool, count=len(scene_q))
    return scene[keep]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="1_experiment_car_1")
    ap.add_argument("--timestamp", default="1760002143141061019")
    ap.add_argument("--base", default="data/2025_10_09/Experiment-Data/experiments")
    ap.add_argument("--out-dir", default="experiment/visualizations/pipeline_overview")
    args = ap.parse_args()

    exp = Path(args.base) / args.experiment
    ts = args.timestamp
    out = Path(args.out_dir)

    raw_os0 = xyz(exp / "os0_pcd" / f"{ts}.pcd")
    raw_os1 = xyz(exp / "os1_pcd" / f"{ts}.pcd")
    tr_os0 = xyz(exp / "os0_pcd_transform" / f"{ts}.pcd")
    tr_os1 = xyz(exp / "os1_pcd_transform" / f"{ts}.pcd")
    merged = xyz(exp / "merged_pcd" / f"{ts}.pcd")
    bg = xyz(Path(args.base) / "bg-frame-merged.pcd")
    fg = foreground_by_voxel(merged, bg)

    common_bounds = (-8, 24, -6, 24)
    raw_bounds = (-8, 8, -8, 8)

    save_scene(
        out / "01_raw_sensor_frames.png",
        [(raw_os0, COLORS["os0"], 1), (raw_os1, COLORS["os1"], 1)],
        raw_bounds,
        "01 raw PCD export: local sensor frames",
        "red=os0, green=os1; both still live in their own LiDAR coordinate frames",
    )
    save_scene(
        out / "02_transformed_common_frame.png",
        [(tr_os0, COLORS["os0"], 1), (tr_os1, COLORS["os1"], 1)],
        common_bounds,
        "02 transformed PCDs: common garage frame",
        "manual calibration + ICP moved both sensors into one coordinate system",
    )
    save_scene(
        out / "03_merged_common_frame.png",
        [(merged, COLORS["merged"], 1)],
        common_bounds,
        "03 merged point cloud",
        "this is the model input candidate for the multi-sensor condition",
    )
    save_scene(
        out / "04_background_removal.png",
        [(bg, COLORS["bg"], 1), (fg, COLORS["foreground"], 1)],
        common_bounds,
        "04 background removal overview",
        "gray=background frame, magenta=points not seen in background voxels",
    )

    print(f"Wrote overview PNGs to {out.resolve()}")
    print(f"Frame: {args.experiment}/{ts}")
    print(f"raw os0={len(raw_os0)} raw os1={len(raw_os1)} transformed os0={len(tr_os0)} transformed os1={len(tr_os1)} merged={len(merged)} foreground~={len(fg)}")


if __name__ == "__main__":
    main()
