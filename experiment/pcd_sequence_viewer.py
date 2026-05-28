import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import open3d as o3d


CLASS_COLORS = {
    "car": (1.0, 0.2, 0.1),
    "bike": (1.0, 0.65, 0.0),
    "bicycle": (1.0, 0.65, 0.0),
    "person": (0.1, 0.35, 1.0),
    "Pedestrian": (0.1, 0.35, 1.0),
    "Cyclist": (1.0, 0.65, 0.0),
    "Car": (1.0, 0.2, 0.1),
}


def sorted_pcds(path: Path) -> list[Path]:
    files = sorted(path.glob("*.pcd"))
    if not files:
        raise FileNotFoundError(f"No .pcd files found in {path}")
    return files


def read_pcd(path: Path, color: tuple[float, float, float] | None = None) -> o3d.geometry.PointCloud:
    pcd = o3d.io.read_point_cloud(str(path))
    pcd.remove_non_finite_points()
    if color is not None:
        pcd.paint_uniform_color(color)
    return pcd


def make_bbox(label: dict) -> o3d.geometry.OrientedBoundingBox:
    bbox = label["bbox"]
    center = bbox[0:3]
    extent = bbox[3:6]
    yaw = bbox[6]
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    rot = np.array([
        [cos_yaw, -sin_yaw, 0.0],
        [sin_yaw, cos_yaw, 0.0],
        [0.0, 0.0, 1.0],
    ])
    obj = o3d.geometry.OrientedBoundingBox(center, rot, extent)
    label_name = label.get("label") or label.get("class") or label.get("name")
    obj.color = CLASS_COLORS.get(label_name, (0.0, 0.0, 1.0))
    return obj


def load_labels(path: Path) -> list[o3d.geometry.OrientedBoundingBox]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    labels = data if isinstance(data, list) else data.get("instances", [])
    return [make_bbox(item) for item in labels if "bbox" in item]


def subtract_background(
    pcd: o3d.geometry.PointCloud,
    background: o3d.geometry.PointCloud,
    voxel_size: float,
    threshold: float,
) -> o3d.geometry.PointCloud:
    pcd_ds = pcd.voxel_down_sample(voxel_size)
    bg_ds = background.voxel_down_sample(voxel_size)
    dists = np.asarray(pcd_ds.compute_point_cloud_distance(bg_ds))
    pts = np.asarray(pcd_ds.points)
    fg = o3d.geometry.PointCloud()
    fg.points = o3d.utility.Vector3dVector(pts[dists > threshold])
    fg.paint_uniform_color((1.0, 0.0, 1.0))
    return fg


def make_geometries(args, frame_path: Path, background=None) -> tuple[list[o3d.geometry.Geometry], dict[str, int]]:
    exp = Path(args.base) / args.experiment
    ts = frame_path.stem
    geoms: list[o3d.geometry.Geometry] = [
        o3d.geometry.TriangleMesh.create_coordinate_frame(size=args.axis_size)
    ]
    stats: dict[str, int] = {}

    if args.mode == "raw":
        os0 = read_pcd(exp / "os0_pcd" / f"{ts}.pcd", (1.0, 0.0, 0.0))
        os1 = read_pcd(exp / "os1_pcd" / f"{ts}.pcd", (0.0, 0.75, 0.0))
        stats["os0_points"] = len(os0.points)
        stats["os1_points"] = len(os1.points)
        geoms.extend([os0, os1])
    elif args.mode == "transformed":
        os0 = read_pcd(exp / "os0_pcd_transform" / f"{ts}.pcd", (1.0, 0.0, 0.0))
        os1 = read_pcd(exp / "os1_pcd_transform" / f"{ts}.pcd", (0.0, 0.75, 0.0))
        stats["os0_points"] = len(os0.points)
        stats["os1_points"] = len(os1.points)
        geoms.extend([os0, os1])
    elif args.mode == "merged":
        merged = read_pcd(exp / "merged_pcd" / f"{ts}.pcd", None)
        if args.force_color:
            merged.paint_uniform_color((0.15, 0.65, 1.0))
        stats["merged_points"] = len(merged.points)
        geoms.append(merged)
    elif args.mode == "foreground":
        current = read_pcd(exp / "merged_pcd" / f"{ts}.pcd", (0.7, 0.7, 0.7))
        if background is None:
            raise ValueError("--mode foreground needs a background frame")
        bg_vis = o3d.geometry.PointCloud(background)
        bg_vis.paint_uniform_color((0.55, 0.55, 0.55))
        foreground = subtract_background(current, background, args.voxel_size, args.bg_threshold)
        stats["merged_points"] = len(current.points)
        stats["foreground_points"] = len(foreground.points)
        geoms.append(bg_vis)
        geoms.append(foreground)
    else:
        raise ValueError(args.mode)

    if args.show_labels:
        label_dir = args.label_dir or {
            "raw": None,
            "transformed": "merged_labels",
            "merged": "merged_labels",
            "foreground": "merged_labels",
        }[args.mode]
        if label_dir:
            boxes = load_labels(exp / label_dir / f"{ts}.json")
            stats["boxes"] = len(boxes)
            geoms.extend(boxes)

    return geoms, stats


def run(args) -> None:
    if os.name == "posix":
        os.environ["XDG_SESSION_TYPE"] = "x11"

    exp = Path(args.base) / args.experiment
    frame_dir = {
        "raw": exp / "os0_pcd",
        "transformed": exp / "os0_pcd_transform",
        "merged": exp / "merged_pcd",
        "foreground": exp / "merged_pcd",
    }[args.mode]
    files = sorted_pcds(frame_dir)[args.start:args.end:args.step]
    if not files:
        raise ValueError("Selected frame range is empty")

    background = None
    if args.mode == "foreground":
        background = read_pcd(Path(args.base) / "bg-frame-merged.pcd", None)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(
        window_name=f"{args.experiment} | {args.mode}",
        width=args.width,
        height=args.height,
    )

    state = {"idx": 0, "running": args.play, "view_initialized": False}
    current_geoms: list[o3d.geometry.Geometry] = []

    def render_current():
        nonlocal current_geoms
        old_camera = None
        should_preserve_view = state["view_initialized"] and not args.reset_view_each_frame
        if should_preserve_view:
            old_camera = vis.get_view_control().convert_to_pinhole_camera_parameters()

        vis.clear_geometries()
        frame_path = files[state["idx"]]
        current_geoms, stats = make_geometries(args, frame_path, background)
        for geom in current_geoms:
            vis.add_geometry(
                geom,
                reset_bounding_box=not should_preserve_view,
            )

        opt = vis.get_render_option()
        opt.background_color = np.asarray(args.background)
        opt.point_size = args.point_size
        opt.line_width = args.line_width

        stat_text = " ".join(f"{k}={v}" for k, v in stats.items())
        print(f"[{state['idx'] + 1}/{len(files)}] {frame_path.stem} {stat_text}")
        if args.reset_view_each_frame or state["idx"] == 0:
            reset_camera()
        elif old_camera is not None:
            try:
                vis.get_view_control().convert_from_pinhole_camera_parameters(
                    old_camera,
                    allow_arbitrary=True,
                )
            except TypeError:
                vis.get_view_control().convert_from_pinhole_camera_parameters(old_camera)
        state["view_initialized"] = True
        vis.poll_events()
        vis.update_renderer()

    def reset_camera():
        ctr = vis.get_view_control()
        ctr.set_front([0.25, -0.75, 0.6])
        ctr.set_lookat([3.0, 6.0, 0.0])
        ctr.set_up([0.0, 0.0, 1.0])
        ctr.set_zoom(0.28)

    def toggle_play(_vis):
        state["running"] = not state["running"]
        print("play" if state["running"] else "pause")
        return False

    def next_frame(_vis):
        state["idx"] = min(len(files) - 1, state["idx"] + 1)
        render_current()
        return False

    def prev_frame(_vis):
        state["idx"] = max(0, state["idx"] - 1)
        render_current()
        return False

    def reset_view(_vis):
        reset_camera()
        vis.update_renderer()
        return False

    def animation(_vis):
        if state["running"]:
            state["idx"] = (state["idx"] + 1) % len(files)
            render_current()
            time.sleep(args.wait)
        return False

    vis.register_key_callback(ord(" "), toggle_play)
    vis.register_key_callback(ord("N"), next_frame)
    vis.register_key_callback(ord("B"), prev_frame)
    vis.register_key_callback(ord("R"), reset_view)
    vis.register_animation_callback(animation)

    print("Controls: Space play/pause, N next frame, B previous frame, R reset view, mouse rotate/zoom/pan")
    render_current()
    vis.run()
    vis.destroy_window()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Play experiment PCD frames in 3D with Open3D.")
    parser.add_argument("--base", default="data/2025_10_09/Experiment-Data/experiments")
    parser.add_argument("--experiment", default="1_experiment_car_1")
    parser.add_argument(
        "--mode",
        choices=["raw", "transformed", "merged", "foreground"],
        default="transformed",
        help=(
            "raw: local os0/os1 frames; transformed: common coordinate frame; "
            "merged: fused cloud; foreground: background removal view"
        ),
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--wait", type=float, default=0.08)
    parser.add_argument("--play", action="store_true")
    parser.add_argument("--show-labels", action="store_true")
    parser.add_argument(
        "--label-dir",
        help=(
            "Label directory inside the experiment folder. Useful for comparing "
            "merged_labels and merged_labels_clean."
        ),
    )
    parser.add_argument("--axis-size", type=float, default=3.0)
    parser.add_argument("--point-size", type=float, default=2.0)
    parser.add_argument("--line-width", type=float, default=2.0)
    parser.add_argument("--background", type=float, nargs=3, default=(0.02, 0.02, 0.025))
    parser.add_argument("--force-color", action="store_true",
                        help="Paint merged clouds blue instead of using stored PCD colors.")
    parser.add_argument("--reset-view-each-frame", action="store_true")
    parser.add_argument("--voxel-size", type=float, default=0.05)
    parser.add_argument("--bg-threshold", type=float, default=0.05)
    parser.add_argument("--width", type=int, default=1400)
    parser.add_argument("--height", type=int, default=900)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
