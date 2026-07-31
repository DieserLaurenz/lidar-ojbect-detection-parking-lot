"""Detektionsvideo für die Abschlusspräsentation — von Grund auf neu.

Rendert die 11 offiziellen Test-Frames von Experiment 1 (alter temporaler
Split, fahrendes Auto) als Zwei-Panel-BEV-Vergleich os1 vs. merged im
hellen Design des neuen Decks. Ground Truth aus den offiziellen
Test-Infos (lokaler Abzug: official_test_exp1_gt.json), Vorhersagen aus
den Full-Inference-JSONs der finetunten GT-Sampling-Modelle.

Erzeugt in results/videos/:
  exp1_testframes_os1_vs_merged.mp4   (1920x1080, 2 fps, H.264)
  exp1_testframes_poster.png          (Posterframe)
  exp1_scene_os1.png / exp1_scene_merged.png  (Einzel-Panels für Folien)

Aufruf: python experiment/detektionsvideo.py
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "mmdetection3d" / "data" / "exp"
PRED = ROOT / "results" / "predictions"
OUT = ROOT / "results" / "videos"

INK = "#1B1E24"
SUBTLE = "#5F6672"
FAINT = "#9AA1AC"
ACCENT = "#6E56CF"
POINTS = "#454C59"
GT_STATIC = "#B9BEC7"
# Klassenfarben (CVD-validiert; bewusst andere Töne als die Sicht-Farben
# der Deck-Charts)
CLASS_HEX = {0: "#3B82F6", 1: "#CA8A04", 2: "#DB2777"}
CLASS_NAME = {0: "Person", 1: "Fahrrad", 2: "Auto"}

SCORE_MIN = 0.3
FPS = 2
XLIM = (6.0, 40.0)
YLIM = (-7.0, 19.0)

plt.rcParams.update({"font.family": "Segoe UI", "text.color": INK})


# ------------------------------------------------------------------ Daten

def load_frames() -> list[dict]:
    gt = json.loads((DATA / "official_test_exp1_gt.json").read_text())
    preds = {
        "os1": json.loads(
            (PRED / "predictions_os1_full.json").read_text())["frames"],
        "merged": json.loads(
            (PRED / "predictions_merged_full.json").read_text())["frames"],
    }
    view_code = {"os1": "1", "merged": "2"}
    ts_list = sorted(gt["merged"])
    frames = []
    for ts in ts_list:
        entry = {"ts": ts}
        for view in ("os1", "merged"):
            key = "1" + view_code[view] + "0000" + ts
            pts = np.fromfile(DATA / "points_kitti" / f"{key}.bin",
                              dtype=np.float32).reshape(-1, 4)
            pf = preds[view][key]
            entry[view] = {
                "points": pts,
                "gt": gt[view][ts],
                "pred": [
                    (box, label, score)
                    for box, label, score in zip(pf["boxes"], pf["labels"],
                                                 pf["scores"])
                    if score >= SCORE_MIN
                ],
            }
        frames.append(entry)
    return frames


def mark_moving(frames: list[dict]) -> None:
    """GT-Boxen als bewegt/statisch klassifizieren (Positionsstabilität)."""
    for view in ("os1", "merged"):
        centers = [
            [(b["bbox_3d"][0], b["bbox_3d"][1], b["label"])
             for b in f[view]["gt"]] for f in frames
        ]
        for frame in frames:
            for box in frame[view]["gt"]:
                x, y = box["bbox_3d"][:2]
                stable = sum(
                    any(abs(x - cx) < 0.3 and abs(y - cy) < 0.3
                        and box["label"] == cl for cx, cy, cl in other)
                    for other in centers)
                box["moving"] = stable < len(frames) - 1


# ------------------------------------------------------------------ Malen

def bev_corners(box) -> np.ndarray:
    x, y, sx, sy, yaw = box[0], box[1], box[3], box[4], box[6]
    c, s = np.cos(yaw), np.sin(yaw)
    local = np.array([[sx, sy], [sx, -sy], [-sx, -sy], [-sx, sy]]) / 2.0
    rot = np.array([[c, -s], [s, c]])
    return local @ rot.T + np.array([x, y])


def draw_panel(ax, data, title):
    ax.set_facecolor("white")
    pts = data["points"]
    ax.scatter(pts[:, 0], pts[:, 1], s=0.9, color=POINTS, alpha=0.5,
               linewidths=0)
    for box in data["gt"]:
        corners = bev_corners(box["bbox_3d"])
        moving = box.get("moving", False)
        ax.add_patch(Polygon(
            corners, closed=True, fill=False, linestyle=(0, (4, 3)),
            edgecolor=INK if moving else GT_STATIC,
            linewidth=2.4 if moving else 1.2, zorder=4 if moving else 2))
    for box, label, score in data["pred"]:
        corners = bev_corners(box)
        color = CLASS_HEX[label]
        ax.add_patch(Polygon(corners, closed=True, fill=False,
                             edgecolor=color, linewidth=2.2, zorder=5))
        top = corners[np.argmax(corners[:, 1])]
        ax.text(top[0], top[1] + 0.35, f"{score:.2f}", color=color,
                fontsize=8.5, ha="center", va="bottom", zorder=6,
                fontweight="bold")
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=17, fontweight="bold", color=INK,
                 loc="left", pad=10)
    ax.tick_params(labelsize=9, colors=SUBTLE, length=0)
    ax.set_xlabel("x [m]", fontsize=9.5, color=SUBTLE)
    ax.set_ylabel("y [m]", fontsize=9.5, color=SUBTLE)
    for spine in ax.spines.values():
        spine.set_color("#D8DBE0")


def render_frame(frame, index, total, path: Path) -> None:
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    fig.patch.set_facecolor("white")
    fig.text(0.045, 0.945, "Ein Blickwinkel gegen die Fusion — dieselben "
                           "elf Test-Momente", fontsize=23,
             fontweight="bold", color=INK)
    fig.patches.append(Rectangle((0.045, 0.905), 0.052, 0.007,
                                 transform=fig.transFigure, color=ACCENT,
                                 zorder=5))
    fig.text(0.045, 0.868, "Experiment 1, fahrendes Auto · PointPillars "
                           "finetuned (GT-Sampling) · Vorhersagen mit "
                           "Score ≥ 0,3 · offizieller Test-Split des "
                           "alten temporalen Protokolls, verlangsamt auf "
                           "2 fps", fontsize=13, color=SUBTLE)
    fig.text(0.955, 0.945, f"Testframe {index + 1} / {total}",
             fontsize=15, color=SUBTLE, ha="right")
    ax_l = fig.add_axes((0.045, 0.115, 0.44, 0.68))
    ax_r = fig.add_axes((0.535, 0.115, 0.44, 0.68))
    draw_panel(ax_l, frame["os1"], "Einzelsensor os1")
    draw_panel(ax_r, frame["merged"], "Fusion os0 + os1 (merged)")
    handles = [
        Line2D([], [], color=CLASS_HEX[i], linewidth=2.4,
               label=CLASS_NAME[i]) for i in (0, 1, 2)
    ] + [
        Line2D([], [], color=INK, linewidth=2.2, linestyle=(0, (4, 3)),
               label="Ground Truth (bewegt)"),
        Line2D([], [], color=GT_STATIC, linewidth=1.4,
               linestyle=(0, (4, 3)), label="Ground Truth (statisch)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               fontsize=12.5, bbox_to_anchor=(0.5, 0.012))
    fig.savefig(path, dpi=100, facecolor="white")
    plt.close(fig)


def render_scene(data, title, path: Path) -> None:
    """Einzel-Panel als Folienbild (ohne Header/Zähler)."""
    fig = plt.figure(figsize=(9.0, 6.4), dpi=200)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes((0.08, 0.10, 0.89, 0.86))
    draw_panel(ax, data, "")
    ax.set_title("")
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------------ Main

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frames = load_frames()
    mark_moving(frames)
    total = len(frames)
    with tempfile.TemporaryDirectory(prefix="detvideo_") as tmp:
        tmp_path = Path(tmp)
        for i, frame in enumerate(frames):
            render_frame(frame, i, total, tmp_path / f"frame_{i:03d}.png")
        # letzten Frame kurz halten
        last = (tmp_path / f"frame_{total - 1:03d}.png").read_bytes()
        (tmp_path / f"frame_{total:03d}.png").write_bytes(last)
        video = OUT / "exp1_testframes_os1_vs_merged.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-framerate", str(FPS),
            "-i", str(tmp_path / "frame_%03d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
            str(video),
        ], check=True, capture_output=True)
        poster = OUT / "exp1_testframes_poster.png"
        poster.write_bytes(
            (tmp_path / "frame_000.png").read_bytes())
    mid = frames[total // 2]
    render_scene(mid["os1"], "os1", OUT / "exp1_scene_os1.png")
    render_scene(mid["merged"], "merged", OUT / "exp1_scene_merged.png")
    print(f"Video: {video}")
    print(f"Poster + Szenenbilder: {OUT}")


if __name__ == "__main__":
    main()
