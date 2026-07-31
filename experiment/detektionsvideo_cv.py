"""Präsentationsvideo aus einem kompletten ungesehenen CV-Experiment.

Gezeigt werden alle 105 gemeinsamen gültigen Frames von Experiment 1 aus
Fold 1 der experiment-held-out Cross-Validation. Eine große isometrische
Übersicht zeigt die komplette fusionierte Szene; ein mitfahrender Zoom macht
das bewegte Auto und seine Box deutlich erkennbar.

Voraussetzung: `experiment/export_cv_video_assets.py` wurde auf dem Server
für merged/CV1/Experiment 1 ausgeführt und die Assets liegen unter
`results/crossval/exp1_merged_cv1*`.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parent.parent
ASSET_JSON = ROOT / "results" / "crossval" / "exp1_merged_cv1.json"
POINTS = ROOT / "results" / "crossval" / "exp1_merged_cv1" / "points"
OUT = ROOT / "results" / "videos"

VIDEO = OUT / "exp1_cv_full_merged_oblique.mp4"
POSTER = OUT / "exp1_cv_full_merged_oblique_poster.png"
PREVIEW = OUT / "exp1_cv_full_merged_oblique_preview.png"

INK = "#1B1E24"
SUBTLE = "#5F6672"
FAINT = "#9AA1AC"
HAIR = "#D8DBE0"
ACCENT = "#6E56CF"
GT_STATIC = "#A6ACB8"
CLASS_HEX = {0: "#3B82F6", 1: "#CA8A04", 2: "#DB2777"}
CLASS_NAME = {0: "Person", 1: "Fahrrad", 2: "Auto"}
POINT_CMAP = LinearSegmentedColormap.from_list(
    "height", ["#D9DDE3", "#A3AAB5", "#343B46"])

FPS = 10
SCORE_MIN = 0.3
AZIMUTH = -62.0
ELEVATION = 27.0
OVERVIEW_BOUNDS = ((7.0, 53.0), (-7.0, 22.0), (-2.2, 1.5))

plt.rcParams.update({"font.family": "Segoe UI", "text.color": INK})


def load_frames() -> list[dict]:
    payload = json.loads(ASSET_JSON.read_text(encoding="utf-8"))
    frames = payload["frames"]
    for frame in frames:
        frame["points"] = np.fromfile(
            POINTS / frame["lidar_path"], dtype=np.float32).reshape(-1, 4)
        frame["predictions"] = [
            pred for pred in frame["predictions"]
            if pred["score"] >= SCORE_MIN
        ]
    return frames


def identify_moving_car(frames: list[dict]) -> None:
    """Markiert die nicht ortsfeste Auto-GT und speichert ihren Verlauf."""
    car_centers = [
        [np.asarray(item["box"][:2], dtype=float)
         for item in frame["gt"] if item["label"] == 2]
        for frame in frames
    ]
    for frame, centers in zip(frames, car_centers):
        target = None
        for item, center in zip(
                [item for item in frame["gt"] if item["label"] == 2],
                centers):
            stable = sum(
                any(np.linalg.norm(center - other) < 0.25
                    for other in other_centers)
                for other_centers in car_centers)
            item["moving"] = stable < 0.75 * len(frames)
            if item["moving"]:
                target = item
        if target is None:
            raise RuntimeError(
                f"Kein bewegtes Auto in Frame {frame['sample_id']} gefunden")
        frame["target"] = target


def camera_basis() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    azimuth = np.deg2rad(AZIMUTH)
    elevation = np.deg2rad(ELEVATION)
    forward = np.array([
        np.cos(elevation) * np.cos(azimuth),
        np.cos(elevation) * np.sin(azimuth),
        np.sin(elevation),
    ])
    right = np.array([-np.sin(azimuth), np.cos(azimuth), 0.0])
    up = np.cross(forward, right)
    return right, up, forward


RIGHT, UP, FORWARD = camera_basis()


def project(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xyz = np.asarray(points, dtype=float)
    return np.column_stack((xyz @ RIGHT, xyz @ UP)), xyz @ FORWARD


def box_corners(box) -> np.ndarray:
    x, y, z, dx, dy, dz, yaw = map(float, box[:7])
    local_xy = np.array([
        [-dx / 2, -dy / 2], [dx / 2, -dy / 2],
        [dx / 2, dy / 2], [-dx / 2, dy / 2],
    ])
    c, s = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[c, -s], [s, c]])
    xy = local_xy @ rotation.T + np.array([x, y])
    bottom = np.column_stack((xy, np.full(4, z)))
    top = np.column_stack((xy, np.full(4, z + dz)))
    return np.vstack((bottom, top))


BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


def bounds_mask(points: np.ndarray, bounds) -> np.ndarray:
    return (
        (points[:, 0] >= bounds[0][0]) & (points[:, 0] <= bounds[0][1]) &
        (points[:, 1] >= bounds[1][0]) & (points[:, 1] <= bounds[1][1]) &
        (points[:, 2] >= bounds[2][0]) & (points[:, 2] <= bounds[2][1])
    )


def points_in_box(points: np.ndarray, box) -> np.ndarray:
    """Maske für Punkte innerhalb einer rotierten Bottom-Center-3D-Box."""
    x, y, z, dx, dy, dz, yaw = map(float, box[:7])
    relative = points[:, :2] - np.array([x, y])
    c, s = np.cos(yaw), np.sin(yaw)
    local_x = relative[:, 0] * c + relative[:, 1] * s
    local_y = -relative[:, 0] * s + relative[:, 1] * c
    return (
        (np.abs(local_x) <= dx / 2) &
        (np.abs(local_y) <= dy / 2) &
        (points[:, 2] >= z) & (points[:, 2] <= z + dz)
    )


def projected_limits(bounds) -> tuple[tuple[float, float], tuple[float, float]]:
    corners = np.array([
        [x, y, z]
        for x in bounds[0] for y in bounds[1] for z in bounds[2]
    ])
    screen, _ = project(corners)
    x_margin = max(0.5, np.ptp(screen[:, 0]) * 0.03)
    y_margin = max(0.5, np.ptp(screen[:, 1]) * 0.05)
    return (
        (screen[:, 0].min() - x_margin, screen[:, 0].max() + x_margin),
        (screen[:, 1].min() - y_margin, screen[:, 1].max() + y_margin),
    )


def draw_box(ax, box, color, linewidth, linestyle="-", alpha=1.0,
             zorder=5) -> None:
    corners = box_corners(box)
    screen, _ = project(corners)
    for start, end in BOX_EDGES:
        ax.plot(screen[[start, end], 0], screen[[start, end], 1],
                color=color, linewidth=linewidth, linestyle=linestyle,
                alpha=alpha, zorder=zorder, solid_capstyle="round")


def draw_view(ax, frame: dict, bounds, title: str, max_points: int,
              point_size: float, label_mode: str = "none",
              label_fontsize: float = 9.0,
              trail: np.ndarray | None = None) -> None:
    ax.set_facecolor("white")
    points = frame["points"][:, :3]
    points = points[bounds_mask(points, bounds)]
    step = max(1, math.ceil(len(points) / max_points))
    points = points[::step]
    screen, depth = project(points)
    order = np.argsort(depth)
    points, screen = points[order], screen[order]
    height = np.clip((points[:, 2] + 1.85) / 2.7, 0.0, 1.0)
    point_labels = np.full(len(points), -1, dtype=int)
    point_scores = np.full(len(points), -np.inf, dtype=float)
    for pred in frame["predictions"]:
        inside = points_in_box(points, pred["box"])
        replace = inside & (pred["score"] > point_scores)
        point_labels[replace] = pred["label"]
        point_scores[replace] = pred["score"]

    background = point_labels < 0
    ax.scatter(screen[background, 0], screen[background, 1],
               c=height[background], cmap=POINT_CMAP, s=point_size,
               linewidths=0, alpha=0.72, zorder=1)
    for label in (0, 1, 2):
        selected = point_labels == label
        if selected.any():
            ax.scatter(screen[selected, 0], screen[selected, 1],
                       color=CLASS_HEX[label], s=point_size * 3.2,
                       linewidths=0, alpha=0.92, zorder=2)

    if trail is not None and len(trail) > 1:
        trail_xyz = np.column_stack((trail, np.full(len(trail), -1.65)))
        trail_screen, _ = project(trail_xyz)
        ax.plot(trail_screen[:, 0], trail_screen[:, 1], color=ACCENT,
                linewidth=2.0, alpha=0.72, zorder=3)

    for item in frame["gt"]:
        moving = bool(item.get("moving", False))
        draw_box(ax, item["box"], INK if moving else GT_STATIC,
                 2.2 if moving else 0.9, linestyle="--",
                 alpha=1.0 if moving else 0.58, zorder=4)

    target_xy = np.asarray(frame["target"]["box"][:2])
    for pred in frame["predictions"]:
        center = np.asarray(pred["box"][:2])
        if not (bounds[0][0] <= center[0] <= bounds[0][1] and
                bounds[1][0] <= center[1] <= bounds[1][1]):
            continue
        color = CLASS_HEX[pred["label"]]
        near_target = pred["label"] == 2 and np.linalg.norm(
            center - target_xy) < 3.0
        draw_box(ax, pred["box"], color, 2.7 if near_target else 1.6,
                 alpha=1.0 if near_target else 0.82, zorder=6)
        show_label = (
            label_mode == "all" or
            (label_mode == "target" and near_target)
        )
        if show_label:
            top = box_corners(pred["box"])[4:].mean(axis=0)
            label_pos, _ = project(top[None, :])
            ax.text(label_pos[0, 0], label_pos[0, 1] + 0.35,
                    f"{CLASS_NAME[pred['label']]} {pred['score']:.2f}",
                    color=color, fontsize=label_fontsize,
                    fontweight="bold", ha="center", va="bottom", zorder=7,
                    bbox=dict(facecolor="white", edgecolor="none",
                              alpha=0.78, pad=1.5))

    xlim, ylim = projected_limits(bounds)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=16, fontweight="bold", color=INK,
                 loc="left", pad=9)
    for spine in ax.spines.values():
        spine.set_color(HAIR)
        spine.set_linewidth(1.0)


def render_frame(frames: list[dict], index: int, path: Path) -> None:
    frame = frames[index]
    target = np.asarray(frame["target"]["box"], dtype=float)
    zoom_bounds = (
        (target[0] - 6.5, target[0] + 6.5),
        (target[1] - 5.0, target[1] + 5.0),
        (-2.2, 1.5),
    )
    trail = np.asarray([
        previous["target"]["box"][:2]
        for previous in frames[max(0, index - 24):index + 1]
    ], dtype=float)

    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    fig.patch.set_facecolor("white")
    fig.text(0.04, 0.948,
             "Fusion auf einem vollständig ungesehenen Auto-Experiment",
             fontsize=24, fontweight="bold", color=INK)
    fig.patches.append(Rectangle((0.04, 0.908), 0.055, 0.007,
                                 transform=fig.transFigure, color=ACCENT,
                                 zorder=5))
    fig.text(0.04, 0.872,
             "Experiment 1 · Cross-Validation Fold 1 · 105 gemeinsame "
             "gültige Frames · PointPillars + GT-Sampling · Score ≥ 0,3",
             fontsize=13.2, color=SUBTLE)
    fig.text(0.96, 0.948, f"Frame {index + 1} / {len(frames)}",
             fontsize=15, color=SUBTLE, ha="right")

    overview = fig.add_axes((0.04, 0.145, 0.60, 0.67))
    zoom = fig.add_axes((0.675, 0.145, 0.285, 0.67))
    draw_view(overview, frame, OVERVIEW_BOUNDS,
              "Schrägansicht der fusionierten Szene", 70000, 0.45,
              label_mode="all", label_fontsize=8.5, trail=trail)
    draw_view(zoom, frame, zoom_bounds, "Mitfahrender Zoom: Zielauto",
              45000, 1.25, label_mode="target", label_fontsize=13,
              trail=trail)

    handles = [
        Line2D([], [], color=CLASS_HEX[label], linewidth=2.5,
               label=CLASS_NAME[label]) for label in (0, 1, 2)
    ] + [
        Line2D([], [], color=INK, linewidth=2.0, linestyle="--",
               label="Ground Truth"),
        Line2D([], [], color=ACCENT, linewidth=2.0,
               label="bisheriger Fahrweg"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               fontsize=12.5, bbox_to_anchor=(0.5, 0.035))
    fig.text(0.5, 0.014,
             "Nur Testvorhersagen eines Modells, das Experiment 1 weder "
             "im Training noch zur Modellauswahl gesehen hat.",
             fontsize=10.8, color=FAINT, ha="center")
    fig.savefig(path, dpi=100, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true",
                        help="rendert nur den mittleren Frame")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    frames = load_frames()
    identify_moving_car(frames)
    middle = len(frames) // 2
    if args.preview:
        render_frame(frames, middle, PREVIEW)
        print(f"Preview: {PREVIEW}")
        return

    with tempfile.TemporaryDirectory(prefix="detvideo_cv_") as tmp:
        tmp_path = Path(tmp)
        for index in range(len(frames)):
            render_frame(frames, index,
                         tmp_path / f"frame_{index:03d}.png")
            if (index + 1) % 10 == 0 or index + 1 == len(frames):
                print(f"Gerendert: {index + 1}/{len(frames)}", flush=True)
        POSTER.write_bytes((tmp_path / f"frame_{middle:03d}.png").read_bytes())
        subprocess.run([
            "ffmpeg", "-y", "-framerate", str(FPS),
            "-i", str(tmp_path / "frame_%03d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "19",
            "-movflags", "+faststart", str(VIDEO),
        ], check=True)
    print(f"Video: {VIDEO}")
    print(f"Poster: {POSTER}")


if __name__ == "__main__":
    main()
