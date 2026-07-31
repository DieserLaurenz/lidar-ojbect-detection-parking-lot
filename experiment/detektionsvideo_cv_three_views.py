"""Synchronisiertes Präsentationsvideo für os0, os1 und merged.

Phase 1 zeigt drei große, identisch skalierte vollständige Schrägansichten
ohne Klassen- und Konfidenztexte. Phase 2 wiederholt dieselben 105 Zeitpunkte
als drei Zielauto-Zooms mit Klassen- und Konfidenzwerten. Alle Frames stammen
aus dem vollständig ungesehenen Experiment 1 der Fold-1-Cross-Validation.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from detektionsvideo_cv import (
    ACCENT,
    CLASS_HEX,
    CLASS_NAME,
    FAINT,
    FPS,
    INK,
    SCORE_MIN,
    SUBTLE,
    draw_view,
    identify_moving_car,
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "results" / "crossval"
OUT = ROOT / "results" / "videos"

VIDEO = OUT / "exp1_cv_full_three_views_oblique.mp4"
POSTER = OUT / "exp1_cv_full_three_views_oblique_poster.png"
PREVIEW = OUT / "exp1_cv_full_three_views_oblique_preview.png"
ZOOM_PREVIEW = OUT / "exp1_cv_full_three_views_oblique_zoom_preview.png"

VIEW_TITLE = {
    "os0": "os0 · Einzelsensor",
    "os1": "os1 · Einzelsensor",
    "merged": "merged · Fusion",
}

# Enger Präsentationsausschnitt der vollständigen relevanten Kreuzungsszene.
FULL_VIEW_BOUNDS = ((12.0, 48.0), (-4.5, 18.5), (-2.2, 1.5))

plt.rcParams.update({"font.family": "Segoe UI", "text.color": INK})


def load_view(view: str) -> dict[str, dict]:
    payload = json.loads(
        (DATA / f"exp1_{view}_cv1.json").read_text(encoding="utf-8"))
    points_root = DATA / f"exp1_{view}_cv1" / "points"
    frames = payload["frames"]
    for frame in frames:
        frame["points"] = np.fromfile(
            points_root / frame["lidar_path"], dtype=np.float32
        ).reshape(-1, 4)
        frame["predictions"] = [
            pred for pred in frame["predictions"]
            if pred["score"] >= SCORE_MIN
        ]
    identify_moving_car(frames)
    return {frame["sample_id"][2:]: frame for frame in frames}


def load_frames() -> list[dict[str, dict]]:
    by_view = {view: load_view(view) for view in ("os0", "os1", "merged")}
    timestamps = sorted(set.intersection(*(set(data) for data in by_view.values())))
    if len(timestamps) != 105:
        raise RuntimeError(f"Erwartet: 105 gemeinsame Frames, gefunden: {len(timestamps)}")
    return [
        {view: by_view[view][timestamp] for view in by_view}
        for timestamp in timestamps
    ]


def render_frame(frames: list[dict[str, dict]], index: int, path: Path,
                 mode: str) -> None:
    frame = frames[index]
    if mode not in ("overview", "zoom"):
        raise ValueError(mode)
    target = np.asarray(frame["merged"]["target"]["box"], dtype=float)
    zoom_bounds = (
        (target[0] - 6.5, target[0] + 6.5),
        (target[1] - 5.0, target[1] + 5.0),
        (-2.2, 1.5),
    )
    trail = np.asarray([
        previous["merged"]["target"]["box"][:2]
        for previous in frames[max(0, index - 24):index + 1]
    ], dtype=float)

    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    fig.patch.set_facecolor("white")
    fig.text(0.04, 0.95,
             "Drei Sichten — Inferenz auf demselben Experiment",
             fontsize=23.5, fontweight="bold", color=INK)
    fig.patches.append(Rectangle((0.04, 0.91), 0.055, 0.007,
                                 transform=fig.transFigure, color=ACCENT,
                                 zorder=5))
    fig.text(0.04, 0.875,
             "Experiment 1 · Cross-Validation Fold 1 · synchronisierte "
             "Testframes · PointPillars + GT-Sampling · Score ≥ 0,3",
             fontsize=13.0, color=SUBTLE)
    phase = "Gesamtansichten" if mode == "overview" else "Zielauto-Zooms"
    fig.text(0.96, 0.95,
             f"{phase} · Frame {index + 1} / {len(frames)}",
             fontsize=15, color=SUBTLE, ha="right")

    panel_x = (0.01, 0.34, 0.67)
    for x, view in zip(panel_x, ("os0", "os1", "merged")):
        overview = fig.add_axes((x, 0.17, 0.32, 0.66))
        if mode == "overview":
            draw_view(overview, frame[view], FULL_VIEW_BOUNDS,
                      VIEW_TITLE[view], 60000, 0.82,
                      label_mode="none", trail=trail)
        else:
            draw_view(overview, frame[view], zoom_bounds,
                      f"{view} · Zielauto-Zoom", 42000, 1.25,
                      label_mode="all", label_fontsize=10.0,
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
               fontsize=11.0, bbox_to_anchor=(0.5, 0.025))
    fig.text(0.5, 0.012,
             "Jede Sicht stammt von ihrem eigenen Fold-1-Modell; keines "
             "davon sah Experiment 1 im Training oder zur Modellauswahl.",
             fontsize=10.6, color=FAINT, ha="center")
    fig.savefig(path, dpi=100, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    frames = load_frames()
    middle = len(frames) // 2
    if args.preview:
        render_frame(frames, middle, PREVIEW, "overview")
        render_frame(frames, middle, ZOOM_PREVIEW, "zoom")
        print(f"Previews: {PREVIEW} · {ZOOM_PREVIEW}")
        return

    with tempfile.TemporaryDirectory(prefix="detvideo_cv_three_") as tmp:
        tmp_path = Path(tmp)
        output_index = 0
        for mode, phase_name in (("overview", "Gesamt"), ("zoom", "Zoom")):
            for index in range(len(frames)):
                render_frame(frames, index,
                             tmp_path / f"frame_{output_index:03d}.png",
                             mode)
                output_index += 1
                if (index + 1) % 10 == 0 or index + 1 == len(frames):
                    print(f"{phase_name}: {index + 1}/{len(frames)}",
                          flush=True)
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
