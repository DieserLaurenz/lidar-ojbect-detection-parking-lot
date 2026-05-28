import argparse
import json
import os
from pathlib import Path

import numpy as np
import open3d as o3d


TEMPLATES = {
    "car": [4.5, 1.8, 1.5],
    "bike": [2.0, 0.6, 1.1],
    "person": [0.5, 0.5, 1.5],
}

CLASS_COLORS = {
    "car": (1.0, 0.2, 0.1),
    "bike": (1.0, 0.65, 0.0),
    "person": (0.1, 0.35, 1.0),
}

IGNORE_FRAME_KEY = "ignore_frame"


def sorted_pcds(path: Path) -> list[Path]:
    files = sorted(path.glob("*.pcd"))
    if not files:
        raise FileNotFoundError(f"No .pcd files found in {path}")
    return files


def read_pcd(path: Path, force_color=None) -> o3d.geometry.PointCloud:
    pcd = o3d.io.read_point_cloud(str(path))
    pcd.remove_non_finite_points()
    if force_color:
        pcd.paint_uniform_color(force_color)
    return pcd


def filter_pcd_for_display(
    pcd: o3d.geometry.PointCloud,
    z_min: float | None = None,
    z_max: float | None = None,
) -> o3d.geometry.PointCloud:
    if z_min is None and z_max is None:
        return pcd
    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        return pcd
    mask = np.ones(len(pts), dtype=bool)
    if z_min is not None:
        mask &= pts[:, 2] >= z_min
    if z_max is not None:
        mask &= pts[:, 2] <= z_max
    colors = np.asarray(pcd.colors)
    filtered = o3d.geometry.PointCloud()
    filtered.points = o3d.utility.Vector3dVector(pts[mask])
    if len(colors) == len(pts):
        filtered.colors = o3d.utility.Vector3dVector(colors[mask])
    return filtered


def bbox_to_obb(bbox: list[float], color=(1.0, 0.2, 0.1)) -> o3d.geometry.OrientedBoundingBox:
    x, y, z, dx, dy, dz, yaw = bbox
    c = np.cos(yaw)
    s = np.sin(yaw)
    rot = np.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0],
    ])
    obb = o3d.geometry.OrientedBoundingBox([x, y, z], rot, [dx, dy, dz])
    obb.color = color
    return obb


def bbox_from_points(
    points: np.ndarray,
    class_name: str,
    template_dims: list[float] | None = None,
) -> list[float]:
    dims = template_dims or TEMPLATES[class_name]
    if len(points) == 0:
        return [0.0, 0.0, dims[2] / 2.0, *dims, 0.0]
    center = np.median(points, axis=0)
    return [float(center[0]), float(center[1]), float(center[2]), *dims, 0.0]


def default_bbox_from_current_view(
    pcd: o3d.geometry.PointCloud,
    class_name: str,
    template_dims: list[float],
) -> list[float]:
    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        return bbox_from_points(pts, class_name, template_dims=template_dims)

    # Use a robust center from points close to the middle of the visible cloud.
    # This is only a starting point; the user can move/rotate the box manually.
    q_low = np.percentile(pts[:, :2], 35, axis=0)
    q_high = np.percentile(pts[:, :2], 65, axis=0)
    mask = (
        (pts[:, 0] >= q_low[0]) & (pts[:, 0] <= q_high[0]) &
        (pts[:, 1] >= q_low[1]) & (pts[:, 1] <= q_high[1])
    )
    center_points = pts[mask] if np.any(mask) else pts
    center = np.median(center_points, axis=0)
    return [
        float(center[0]),
        float(center[1]),
        float(center[2]),
        *template_dims,
        0.0,
    ]


def load_label(
    path: Path,
    class_name: str,
    point_cloud: o3d.geometry.PointCloud,
    template_dims: list[float] | None = None,
    force_template_dims: bool = False,
    preserve_manual_dims: bool = True,
) -> tuple[list[float], bool]:
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        labels = data if isinstance(data, list) else data.get("instances", [])
        for item in labels:
            if item.get("label") == class_name and "bbox" in item:
                bbox = [float(x) for x in item["bbox"]]
                manual_dims = bool(item.get("manual_dims", False))
                if (force_template_dims and template_dims is not None and
                        (not manual_dims or not preserve_manual_dims)):
                    bbox[3:6] = template_dims
                    manual_dims = False
                return bbox, manual_dims
        if labels and "bbox" in labels[0]:
            bbox = [float(x) for x in labels[0]["bbox"]]
            manual_dims = bool(labels[0].get("manual_dims", False))
            if (force_template_dims and template_dims is not None and
                    (not manual_dims or not preserve_manual_dims)):
                bbox[3:6] = template_dims
                manual_dims = False
            return bbox, manual_dims
    pts = np.asarray(point_cloud.points)
    return bbox_from_points(pts, class_name, template_dims=template_dims), False


def is_ignore_frame_label(path: Path) -> bool:
    if not path.exists():
        return False
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return bool(data.get(IGNORE_FRAME_KEY, False))
    if isinstance(data, list):
        return any(bool(item.get(IGNORE_FRAME_KEY, False)) for item in data if isinstance(item, dict))
    return False


def read_label_bbox(path: Path, class_name: str) -> list[float]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    labels = data if isinstance(data, list) else data.get("instances", [])
    for item in labels:
        if item.get("label") == class_name and "bbox" in item:
            return [float(x) for x in item["bbox"]]
    for item in labels:
        if "bbox" in item:
            return [float(x) for x in item["bbox"]]
    raise ValueError(f"No bbox found in {path}")


def load_template_bbox(
    args,
    all_files: list[Path],
    label_dir: Path,
    class_name: str,
) -> list[float] | None:
    if args.template_from_index is None:
        return None
    if args.template_from_index < 0 or args.template_from_index >= len(all_files):
        raise IndexError(
            f"--template-from-index {args.template_from_index} outside "
            f"available frame range 0:{len(all_files)}")
    frame = all_files[args.template_from_index]
    label_path = label_dir / f"{frame.stem}.json"
    if not label_path.exists():
        raise FileNotFoundError(
            f"Template label does not exist for frame index "
            f"{args.template_from_index}: {label_path}")
    bbox = read_label_bbox(label_path, class_name)
    dims = bbox[3:6]
    print(
        f"Using template geometry from frame index "
        f"{args.template_from_index} ({frame.stem}): "
        f"z={bbox[2]}, dims={dims}")
    return bbox


def apply_template_geometry(
    bbox: list[float],
    template_bbox: list[float] | None,
    force_dims: bool = False,
    force_z: bool = False,
) -> list[float]:
    if template_bbox is None:
        return bbox
    adjusted = list(bbox)
    if force_dims:
        adjusted[3:6] = template_bbox[3:6]
    if force_z:
        adjusted[2] = template_bbox[2]
    return adjusted


def count_points_in_bbox(pcd: o3d.geometry.PointCloud, bbox: list[float]) -> int:
    obb = bbox_to_obb(bbox)
    return len(obb.get_point_indices_within_bounding_box(pcd.points))


def save_label(
    path: Path,
    class_name: str,
    bbox: list[float],
    num_points: int,
    manual_dims: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump([{
            "label": class_name,
            "bbox": [float(x) for x in bbox],
            "num_lidar_pts": int(num_points),
            "manual_dims": bool(manual_dims),
        }], f, indent=2)
    print(f"Saved {path}")


def save_ignore_frame(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump({IGNORE_FRAME_KEY: True}, f, indent=2)
    print(f"Saved ignore marker {path}")


def format_bbox(bbox: list[float]) -> str:
    x, y, z, dx, dy, dz, yaw = bbox
    return (
        f"center=({x:.2f},{y:.2f},{z:.2f}) "
        f"dim=({dx:.2f},{dy:.2f},{dz:.2f}) yaw={yaw:.2f}"
    )


def run(args) -> None:
    if os.name == "posix":
        os.environ["XDG_SESSION_TYPE"] = "x11"

    exp = Path(args.base) / args.experiment
    pcd_dir = exp / args.pcd_dir
    label_dir = exp / args.label_dir
    all_files = sorted_pcds(pcd_dir)
    files = all_files[args.start:args.end:args.step]
    if not files:
        raise ValueError("Selected frame range is empty")

    class_name = args.class_name
    template_bbox = load_template_bbox(args, all_files, label_dir, class_name)
    template_dims = template_bbox[3:6] if template_bbox is not None else TEMPLATES[class_name]
    force_template_dims = args.force_template_dims or args.template_from_index is not None
    force_template_z = args.force_template_z or args.template_from_index is not None
    default_box_for_empty = (
        args.template_from_index is not None and
        not args.no_default_box_for_empty
    )
    color = CLASS_COLORS[class_name]

    state = {
        "idx": 0,
        "bbox": [0.0, 0.0, template_dims[2] / 2.0, *template_dims, 0.0],
        "pcd": None,
        "box_visible": True,
        "ignored": False,
        "manual_dims": False,
        "view_initialized": False,
        "step_scale": 1.0,
    }

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(
        window_name=f"Manual bbox editor | {args.experiment}",
        width=args.width,
        height=args.height,
    )

    def current_paths():
        frame = files[state["idx"]]
        return frame, label_dir / f"{frame.stem}.json"

    def reset_camera():
        ctr = vis.get_view_control()
        ctr.set_front([0.25, -0.75, 0.6])
        ctr.set_lookat([3.0, 6.0, 0.0])
        ctr.set_up([0.0, 0.0, 1.0])
        ctr.set_zoom(0.28)

    def set_free_edit_camera():
        ctr = vis.get_view_control()
        ctr.set_front([0.25, -0.75, 0.6])
        ctr.set_lookat([state["bbox"][0], state["bbox"][1], state["bbox"][2]])
        ctr.set_up([0.0, 0.0, 1.0])
        ctr.set_zoom(0.45)

    def set_top_down_camera():
        ctr = vis.get_view_control()
        ctr.set_front([0.0, 0.0, -1.0])
        ctr.set_up([0.0, 1.0, 0.0])
        ctr.set_lookat(args.lookat)
        ctr.set_zoom(args.topdown_zoom)

    def render(load_existing: bool = True):
        old_camera = None
        if state["view_initialized"]:
            old_camera = vis.get_view_control().convert_to_pinhole_camera_parameters()

        frame_path, label_path = current_paths()
        pcd = read_pcd(frame_path, force_color=(0.15, 0.65, 1.0))
        pcd = filter_pcd_for_display(
            pcd,
            z_min=args.display_z_min,
            z_max=args.display_z_max,
        )
        state["pcd"] = pcd
        if load_existing:
            has_label = label_path.exists()
            state["ignored"] = is_ignore_frame_label(label_path)
            if state["ignored"]:
                state["box_visible"] = False
                state["manual_dims"] = False
                state["bbox"] = bbox_from_points(
                    np.asarray(pcd.points),
                    class_name,
                    template_dims=template_dims,
                )
            else:
                state["box_visible"] = has_label or default_box_for_empty
                state["bbox"], state["manual_dims"] = load_label(
                    label_path,
                    class_name,
                    pcd,
                    template_dims=template_dims,
                    force_template_dims=force_template_dims,
                    preserve_manual_dims=False,
                )
                state["bbox"] = apply_template_geometry(
                    state["bbox"],
                    template_bbox,
                    force_dims=False,
                    force_z=force_template_z,
                )

        vis.clear_geometries()
        vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=args.axis_size))
        vis.add_geometry(pcd)
        if state["box_visible"]:
            vis.add_geometry(bbox_to_obb(state["bbox"], color))

        opt = vis.get_render_option()
        opt.background_color = np.asarray(args.background)
        opt.point_size = args.point_size
        opt.line_width = args.line_width

        pts_in_box = count_points_in_bbox(pcd, state["bbox"])
        if state["ignored"]:
            visible_text = "ignored"
        else:
            visible_text = "visible" if state["box_visible"] else "hidden/no-label"
        print(
            f"[{state['idx'] + 1}/{len(files)}] {frame_path.stem} "
            f"{visible_text} points_in_box={pts_in_box} {format_bbox(state['bbox'])}"
        )

        if old_camera is not None:
            try:
                vis.get_view_control().convert_from_pinhole_camera_parameters(
                    old_camera,
                    allow_arbitrary=True,
                )
            except TypeError:
                vis.get_view_control().convert_from_pinhole_camera_parameters(old_camera)
        else:
            if args.topdown:
                set_top_down_camera()
            else:
                reset_camera()
        state["view_initialized"] = True
        vis.poll_events()
        vis.update_renderer()

    def mutate(delta=None, dim_delta=None, yaw_delta=0.0):
        state["ignored"] = False
        state["box_visible"] = True
        if delta is not None:
            state["bbox"][0] += delta[0]
            state["bbox"][1] += delta[1]
            state["bbox"][2] += delta[2]
        if dim_delta is not None:
            state["bbox"][3] = max(0.05, state["bbox"][3] + dim_delta[0])
            state["bbox"][4] = max(0.05, state["bbox"][4] + dim_delta[1])
            state["bbox"][5] = max(0.05, state["bbox"][5] + dim_delta[2])
            state["manual_dims"] = True
        state["bbox"][6] += yaw_delta
        render(load_existing=False)
        return False

    def save_current(_vis=None):
        frame_path, label_path = current_paths()
        if state["ignored"]:
            save_ignore_frame(label_path)
            return False
        if not state["box_visible"]:
            print("Box hidden; skip autosave for current frame")
            return False
        state["box_visible"] = True
        pts = count_points_in_bbox(state["pcd"], state["bbox"])
        manual_dims = state["manual_dims"] or not np.allclose(
            np.asarray(state["bbox"][3:6], dtype=float),
            np.asarray(template_dims, dtype=float),
            atol=1e-6,
        )
        save_label(label_path, class_name, state["bbox"], pts, manual_dims)
        return False

    def ignore_current(_vis=None):
        _frame_path, label_path = current_paths()
        state["ignored"] = True
        state["box_visible"] = False
        save_ignore_frame(label_path)
        render(load_existing=False)
        return False

    def maybe_autosave():
        if args.no_autosave_on_frame_change:
            return
        save_current()

    def delete_current(_vis=None):
        _frame_path, label_path = current_paths()
        if label_path.exists():
            label_path.unlink()
            print(f"Deleted {label_path}")
        else:
            print(f"No label file to delete: {label_path}")
        state["ignored"] = False
        state["box_visible"] = False
        render(load_existing=False)
        return False

    def next_frame(_vis=None):
        maybe_autosave()
        state["idx"] = min(len(files) - 1, state["idx"] + 1)
        render(load_existing=True)
        return False

    def prev_frame(_vis=None):
        maybe_autosave()
        state["idx"] = max(0, state["idx"] - 1)
        render(load_existing=True)
        return False

    def copy_previous(_vis=None):
        if state["idx"] == 0:
            print("No previous frame to copy")
            return False
        state["ignored"] = False
        prev_label = label_dir / f"{files[state['idx'] - 1].stem}.json"
        state["bbox"], state["manual_dims"] = load_label(
            prev_label,
            class_name,
            state["pcd"],
            template_dims=template_dims,
            force_template_dims=False,
        )
        if not np.allclose(
                np.asarray(state["bbox"][3:6], dtype=float),
                np.asarray(template_dims, dtype=float),
                atol=1e-6):
            state["manual_dims"] = True
        state["bbox"] = apply_template_geometry(
            state["bbox"],
            template_bbox,
            force_dims=False,
            force_z=force_template_z,
        )
        state["box_visible"] = True
        render(load_existing=False)
        return False

    def reset_template(_vis=None):
        state["ignored"] = False
        state["bbox"][3:6] = template_dims
        if force_template_z and template_bbox is not None:
            state["bbox"][2] = template_bbox[2]
        state["manual_dims"] = False
        state["box_visible"] = True
        render(load_existing=False)
        return False

    def new_box(_vis=None):
        state["ignored"] = False
        state["bbox"] = default_bbox_from_current_view(
            state["pcd"],
            class_name,
            template_dims,
        )
        if force_template_z and template_bbox is not None:
            state["bbox"][2] = template_bbox[2]
        state["box_visible"] = True
        state["manual_dims"] = False
        render(load_existing=False)
        set_free_edit_camera()
        vis.update_renderer()
        print("Created new box. Free 3D edit camera enabled.")
        return False

    def print_help(_vis=None):
        print(HELP_TEXT)
        return False

    def move_step():
        return args.move_step * state["step_scale"]

    def z_step():
        return args.z_step * state["step_scale"]

    def dim_step():
        return args.dim_step * state["step_scale"]

    def yaw_step():
        return args.yaw_step * state["step_scale"]

    def set_step_scale(scale: float):
        state["step_scale"] = scale
        print(
            "Step scale set to "
            f"{scale:g}: move={move_step():.3f}, z={z_step():.3f}, "
            f"dim={dim_step():.3f}, yaw={yaw_step():.3f}"
        )
        return False

    keymap = {
        "W": lambda v: mutate(delta=(0.0, move_step(), 0.0)),
        "S": lambda v: mutate(delta=(0.0, -move_step(), 0.0)),
        "A": lambda v: mutate(delta=(-move_step(), 0.0, 0.0)),
        "D": lambda v: mutate(delta=(move_step(), 0.0, 0.0)),
        "Q": lambda v: mutate(delta=(0.0, 0.0, z_step())),
        "E": lambda v: mutate(delta=(0.0, 0.0, -z_step())),
        "J": lambda v: mutate(yaw_delta=yaw_step()),
        "L": lambda v: mutate(yaw_delta=-yaw_step()),
        "U": lambda v: mutate(dim_delta=(dim_step(), 0.0, 0.0)),
        "O": lambda v: mutate(dim_delta=(-dim_step(), 0.0, 0.0)),
        "I": lambda v: mutate(dim_delta=(0.0, dim_step(), 0.0)),
        "K": lambda v: mutate(dim_delta=(0.0, -dim_step(), 0.0)),
        "G": lambda v: mutate(dim_delta=(0.0, 0.0, dim_step())),
        "H": lambda v: mutate(dim_delta=(0.0, 0.0, -dim_step())),
        "1": lambda v: set_step_scale(0.25),
        "2": lambda v: set_step_scale(1.0),
        "3": lambda v: set_step_scale(2.0),
        "N": next_frame,
        "B": prev_frame,
        "C": new_box,
        "P": copy_previous,
        "T": reset_template,
        "V": lambda v: (state.update({"ignored": False, "box_visible": not state["box_visible"]}), render(load_existing=False), False)[-1],
        "Y": ignore_current,
        "Z": delete_current,
        "R": lambda v: (reset_camera(), vis.update_renderer(), False)[-1],
        "F": lambda v: (set_free_edit_camera(), vis.update_renderer(), False)[-1],
        "M": lambda v: (set_top_down_camera(), vis.update_renderer(), False)[-1],
        "X": save_current,
        "?": print_help,
    }

    for key, callback in keymap.items():
        vis.register_key_callback(ord(key), callback)

    print(HELP_TEXT)
    render(load_existing=True)
    vis.run()
    vis.destroy_window()


HELP_TEXT = """
Manual bbox editor controls
---------------------------
Mouse       rotate / zoom / pan view
1/2/3       fine / normal / coarse edit steps
W/S         move bbox +Y / -Y
A/D         move bbox -X / +X
Q/E         move bbox up / down
J/L         rotate yaw left / right
U/O         length + / -
I/K         width  + / -
G/H         height + / -
P           copy previous frame label; template dims/z stay locked
C           create a new visible box and switch to free 3D edit camera
T           reset dimensions and template z to class/frame template
V           show / hide current box without deleting file
Y           mark current frame as ignored for dataset creation
Z           delete current frame label JSON and hide box
X           save current bbox JSON
N/B         next / previous frame
R           reset camera
F           free 3D edit camera centered on current box
M           top-down camera
?           print this help
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Manual 3D bbox editor for this project's label JSON format.")
    parser.add_argument("--base", default="data/2025_10_09/Experiment-Data/experiments")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--pcd-dir", default="merged_pcd")
    parser.add_argument("--label-dir", required=True)
    parser.add_argument("--class-name", choices=sorted(TEMPLATES), default="car")
    parser.add_argument(
        "--template-from-index",
        type=int,
        help=(
            "Use bbox dimensions and center z from this absolute frame "
            "index as class template for loaded/new boxes and the T key."
        ),
    )
    parser.add_argument(
        "--force-template-dims",
        action="store_true",
        help=(
            "Apply template dimensions to every loaded box. This is enabled "
            "automatically when --template-from-index is set; the flag remains "
            "for backwards compatibility."
        ),
    )
    parser.add_argument(
        "--force-template-z",
        action="store_true",
        help=(
            "Apply the template bbox center z to every loaded/new/copied box. "
            "This is enabled automatically when --template-from-index is set."
        ),
    )
    parser.add_argument(
        "--no-default-box-for-empty",
        action="store_true",
        help=(
            "When using --template-from-index, do not show a provisional "
            "template-sized box for frames without an existing label."
        ),
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument(
        "--no-autosave-on-frame-change",
        action="store_true",
        help="Disable the default behavior of saving before N/B frame changes.",
    )
    parser.add_argument("--move-step", type=float, default=0.05)
    parser.add_argument("--z-step", type=float, default=0.025)
    parser.add_argument("--dim-step", type=float, default=0.05)
    parser.add_argument("--yaw-step", type=float, default=0.025)
    parser.add_argument("--axis-size", type=float, default=3.0)
    parser.add_argument("--point-size", type=float, default=3.0)
    parser.add_argument("--line-width", type=float, default=2.0)
    parser.add_argument("--background", type=float, nargs=3, default=(0.02, 0.02, 0.025))
    parser.add_argument("--topdown", action="store_true",
                        help="Start in top-down view. Press M to return to top-down later.")
    parser.add_argument("--topdown-zoom", type=float, default=0.36)
    parser.add_argument("--lookat", type=float, nargs=3, default=(3.0, 6.0, 0.0))
    parser.add_argument(
        "--display-z-min",
        type=float,
        help="Only display points with z >= this value. Labels are still saved normally.",
    )
    parser.add_argument(
        "--display-z-max",
        type=float,
        help="Only display points with z <= this value. Useful to hide the ceiling.",
    )
    parser.add_argument("--width", type=int, default=1400)
    parser.add_argument("--height", type=int, default=900)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
