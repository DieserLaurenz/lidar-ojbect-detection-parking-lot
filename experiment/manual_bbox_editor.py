import argparse
import json
import math
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
STATIC_BOX_COLOR = (0.1, 1.0, 0.25)
SELECTED_BOX_COLOR = (1.0, 1.0, 0.0)

IGNORE_FRAME_KEY = "ignore_frame"
INVALID_FRAME_KEY = "invalid_frame"


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


def normalize_static_car_bbox_axes(bbox: list[float]) -> list[float]:
    normalized = [float(x) for x in bbox]
    if len(normalized) >= 7 and normalized[3] < normalized[4]:
        normalized[3], normalized[4] = normalized[4], normalized[3]
        normalized[6] = ((normalized[6] + math.pi / 2.0 + math.pi) % (2.0 * math.pi)) - math.pi
    return normalized


def normalize_label_bbox_axes(label: dict) -> dict:
    item = dict(label)
    if item.get("static") is True and item.get("label") == "car" and "bbox" in item:
        item["bbox"] = normalize_static_car_bbox_axes(item["bbox"])
    return item


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
        if labels and "bbox" in labels[0] and "label" not in labels[0]:
            bbox = [float(x) for x in labels[0]["bbox"]]
            manual_dims = bool(labels[0].get("manual_dims", False))
            if (force_template_dims and template_dims is not None and
                    (not manual_dims or not preserve_manual_dims)):
                bbox[3:6] = template_dims
                manual_dims = False
            return bbox, manual_dims
    pts = np.asarray(point_cloud.points)
    return bbox_from_points(pts, class_name, template_dims=template_dims), False


def load_display_labels(path: Path, class_name: str | None = None) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if is_invalid_frame_data(data):
        return []
    labels = data if isinstance(data, list) else data.get("instances", [])
    display_labels = []
    for item in labels:
        if not isinstance(item, dict) or "bbox" not in item:
            continue
        if (
            class_name is not None
            and item.get("label") != class_name
            and not item.get("static", False)
        ):
            continue
        display_labels.append(item)
    return display_labels


def is_invalid_frame_data(data: object) -> bool:
    if isinstance(data, dict):
        return bool(data.get(INVALID_FRAME_KEY, False) or data.get(IGNORE_FRAME_KEY, False))
    if isinstance(data, list):
        return any(
            bool(item.get(INVALID_FRAME_KEY, False) or item.get(IGNORE_FRAME_KEY, False))
            for item in data
            if isinstance(item, dict)
        )
    return False


def is_invalid_frame_label(path: Path) -> bool:
    if not path.exists():
        return False
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return is_invalid_frame_data(data)


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


def save_labels(path: Path, labels: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = []
    for label in labels:
        item = normalize_label_bbox_axes(label)
        if "bbox" in item:
            item["bbox"] = [float(x) for x in item["bbox"]]
        if "num_lidar_pts" in item:
            item["num_lidar_pts"] = int(item["num_lidar_pts"])
        serializable.append(item)
    with path.open("w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
    print(f"Saved {path}")


def invalid_frame_marker() -> dict:
    return {INVALID_FRAME_KEY: True}


def save_invalid_frame(path: Path, labels: list[dict] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if labels:
        data = [dict(label) for label in labels]
        data.append(invalid_frame_marker())
    else:
        data = invalid_frame_marker()
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Saved invalid-frame marker {path}")


def read_label_data(path: Path) -> object:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def remove_invalid_frame_marker(path: Path) -> list[dict]:
    data = read_label_data(path)
    if not isinstance(data, list):
        return []
    labels = [
        item for item in data
        if isinstance(item, dict)
        and not item.get(INVALID_FRAME_KEY, False)
        and not item.get(IGNORE_FRAME_KEY, False)
    ]
    save_labels(path, labels)
    return labels


def strip_static_and_invalid_labels(data: object) -> list[dict]:
    if not isinstance(data, list):
        return []
    return [
        dict(item) for item in data
        if isinstance(item, dict)
        and "bbox" in item
        and not item.get("static", False)
        and not item.get(INVALID_FRAME_KEY, False)
        and not item.get(IGNORE_FRAME_KEY, False)
    ]


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
        "label_items": [],
        "selected_label_idx": 0,
        "view_initialized": False,
        "step_scale": 1.0,
    }

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(
        window_name=(
            f"{'Label viewer' if args.view_only else 'Manual bbox editor'} | "
            f"{args.experiment}"
        ),
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

    def selected_label() -> dict | None:
        labels = state["label_items"]
        if not labels:
            return None
        state["selected_label_idx"] = max(
            0,
            min(state["selected_label_idx"], len(labels) - 1),
        )
        return labels[state["selected_label_idx"]]

    def sync_selected_label_from_bbox() -> None:
        label = selected_label()
        if label is not None:
            label["bbox"] = [float(x) for x in state["bbox"]]

    def sync_bbox_from_selected_label() -> None:
        label = selected_label()
        if label is None:
            return
        state["bbox"] = [float(x) for x in label["bbox"]]
        state["manual_dims"] = bool(label.get("manual_dims", False))
        state["box_visible"] = True

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
            state["ignored"] = is_invalid_frame_label(label_path)
            if args.view_only:
                display_labels = load_display_labels(label_path, class_name)
                state["label_items"] = [dict(item) for item in display_labels]
                state["box_visible"] = bool(display_labels)
                state["manual_dims"] = False
                if display_labels:
                    state["bbox"] = [float(x) for x in display_labels[0]["bbox"]]
                else:
                    state["bbox"] = bbox_from_points(
                        np.asarray(pcd.points),
                        class_name,
                        template_dims=template_dims,
                    )
            elif state["ignored"]:
                state["label_items"] = []
                state["box_visible"] = False
                state["manual_dims"] = False
                state["bbox"] = bbox_from_points(
                    np.asarray(pcd.points),
                    class_name,
                    template_dims=template_dims,
                )
            else:
                edit_labels = load_display_labels(label_path, class_name)
                state["label_items"] = [dict(item) for item in edit_labels]
                state["selected_label_idx"] = min(
                    state["selected_label_idx"],
                    max(0, len(state["label_items"]) - 1),
                )
                if state["label_items"]:
                    sync_bbox_from_selected_label()
                    if force_template_dims and template_dims is not None:
                        state["bbox"][3:6] = template_dims
                    state["bbox"] = apply_template_geometry(
                        state["bbox"],
                        template_bbox,
                        force_dims=False,
                        force_z=force_template_z,
                    )
                    sync_selected_label_from_bbox()
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
        display_labels = []
        if args.view_only and not state["ignored"]:
            display_labels = load_display_labels(label_path, class_name)
            for label in display_labels:
                bbox = [float(x) for x in label["bbox"]]
                label_color = STATIC_BOX_COLOR if label.get("static") else color
                vis.add_geometry(bbox_to_obb(bbox, label_color))
        elif state["label_items"] and not state["ignored"]:
            for idx, label in enumerate(state["label_items"]):
                bbox = [float(x) for x in label["bbox"]]
                if idx == state["selected_label_idx"]:
                    label_color = SELECTED_BOX_COLOR
                else:
                    label_color = STATIC_BOX_COLOR if label.get("static") else color
                vis.add_geometry(bbox_to_obb(bbox, label_color))
        elif state["box_visible"]:
            vis.add_geometry(bbox_to_obb(state["bbox"], color))

        opt = vis.get_render_option()
        opt.background_color = np.asarray(args.background)
        opt.point_size = args.point_size
        opt.line_width = args.line_width

        pts_in_box = count_points_in_bbox(pcd, state["bbox"])
        if state["ignored"]:
            visible_text = "invalid"
        elif args.view_only:
            static_count = sum(1 for item in display_labels if item.get("static"))
            visible_text = (
                f"view-only labels={len(display_labels)} static={static_count}"
            )
        else:
            if state["label_items"]:
                selected = selected_label()
                selected_kind = "static" if selected and selected.get("static") else "label"
                visible_text = (
                    f"visible labels={len(state['label_items'])} "
                    f"selected={state['selected_label_idx'] + 1}/"
                    f"{len(state['label_items'])} {selected_kind}"
                )
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
        sync_selected_label_from_bbox()
        render(load_existing=False)
        return False

    def save_current(_vis=None, propagate_static: bool = False):
        if args.view_only:
            print("View-only mode: save disabled")
            return False
        frame_path, label_path = current_paths()
        if state["ignored"]:
            save_invalid_frame(label_path)
            return False
        if not state["box_visible"]:
            print("Box hidden; skip autosave for current frame")
            return False
        state["box_visible"] = True
        if state["label_items"]:
            sync_selected_label_from_bbox()
            for label in state["label_items"]:
                if "bbox" not in label:
                    continue
                label["num_lidar_pts"] = count_points_in_bbox(
                    state["pcd"],
                    [float(x) for x in label["bbox"]],
                )
            save_labels(label_path, state["label_items"])
            if propagate_static:
                propagate_static_labels_from_current()
            return False
        pts = count_points_in_bbox(state["pcd"], state["bbox"])
        manual_dims = state["manual_dims"] or not np.allclose(
            np.asarray(state["bbox"][3:6], dtype=float),
            np.asarray(template_dims, dtype=float),
            atol=1e-6,
        )
        save_label(label_path, class_name, state["bbox"], pts, manual_dims)
        return False

    def save_current_and_propagate(_vis=None):
        return save_current(_vis, propagate_static=True)

    def propagate_static_labels_from_current(allow_empty: bool = False):
        static_templates = [
            dict(label) for label in state["label_items"]
            if isinstance(label, dict) and label.get("static") and "bbox" in label
        ]
        if not static_templates and not allow_empty:
            return

        updated = 0
        skipped_invalid = 0
        for frame_path in all_files:
            label_path = label_dir / f"{frame_path.stem}.json"
            data = read_label_data(label_path)
            if is_invalid_frame_data(data):
                skipped_invalid += 1
                continue

            base_labels = strip_static_and_invalid_labels(data)
            pcd = read_pcd(frame_path, force_color=None)
            propagated = []
            for template in static_templates:
                item = dict(template)
                bbox = [float(x) for x in item["bbox"]]
                num_points = count_points_in_bbox(pcd, bbox)
                reference_points = float(item.get("num_lidar_pts", num_points))
                required_points = min(reference_points * 0.5, 120.0)
                if num_points < required_points:
                    continue
                item["bbox"] = bbox
                item["num_lidar_pts"] = num_points
                item["static"] = True
                propagated.append(item)

            save_labels(label_path, base_labels + propagated)
            updated += 1

        print(
            f"Propagated {len(static_templates)} static boxes to "
            f"{updated} frames; skipped_invalid={skipped_invalid}"
        )

    def unmark_invalid_current(label_path: Path):
        labels = remove_invalid_frame_marker(label_path)
        state["ignored"] = False
        state["label_items"] = [dict(label) for label in labels]
        state["selected_label_idx"] = 0
        if state["label_items"]:
            sync_bbox_from_selected_label()
        else:
            state["bbox"] = default_bbox_from_current_view(
                state["pcd"],
                class_name,
                template_dims,
            )
            state["box_visible"] = True
            state["manual_dims"] = False
        print(f"Frame unmarked invalid: {label_path}")
        render(load_existing=False)
        return False

    def toggle_invalid_current(_vis=None):
        if args.view_only:
            print("View-only mode: invalid marking disabled")
            return False
        _frame_path, label_path = current_paths()
        if state["ignored"]:
            return unmark_invalid_current(label_path)
        state["ignored"] = True
        state["box_visible"] = False
        save_invalid_frame(label_path, state["label_items"])
        render(load_existing=False)
        return False

    def mark_invalid_current(_vis=None):
        return toggle_invalid_current(_vis)

    def maybe_autosave():
        if args.view_only:
            return
        if args.no_autosave_on_frame_change:
            return
        save_current()

    def delete_current(_vis=None):
        if args.view_only:
            print("View-only mode: delete/invalid disabled")
            return False
        return delete_selected_static_box(_vis)

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
        if args.view_only:
            print("View-only mode: copy disabled")
            return False
        if state["idx"] == 0:
            print("No previous frame to copy")
            return False
        state["ignored"] = False
        state["label_items"] = []
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
        if args.view_only:
            print("View-only mode: template reset disabled")
            return False
        state["ignored"] = False
        state["bbox"][3:6] = template_dims
        if force_template_z and template_bbox is not None:
            state["bbox"][2] = template_bbox[2]
        state["manual_dims"] = False
        sync_selected_label_from_bbox()
        state["box_visible"] = True
        render(load_existing=False)
        return False

    def new_box(_vis=None):
        if args.view_only:
            print("View-only mode: new box disabled")
            return False
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
        state["label_items"].append({
            "label": class_name,
            "bbox": [float(x) for x in state["bbox"]],
            "num_lidar_pts": count_points_in_bbox(state["pcd"], state["bbox"]),
            "manual_dims": False,
        })
        state["selected_label_idx"] = len(state["label_items"]) - 1
        render(load_existing=False)
        set_free_edit_camera()
        vis.update_renderer()
        print("Created new box. Free 3D edit camera enabled.")
        return False

    def new_static_car_box(_vis=None):
        if args.view_only:
            print("View-only mode: new static car box disabled")
            return False
        state["ignored"] = False
        car_dims = TEMPLATES["car"]
        state["bbox"] = default_bbox_from_current_view(
            state["pcd"],
            "car",
            car_dims,
        )
        state["manual_dims"] = False
        state["box_visible"] = True
        state["label_items"].append({
            "label": "car",
            "bbox": [float(x) for x in state["bbox"]],
            "num_lidar_pts": count_points_in_bbox(state["pcd"], state["bbox"]),
            "manual_dims": False,
            "static": True,
        })
        state["selected_label_idx"] = len(state["label_items"]) - 1
        render(load_existing=False)
        set_free_edit_camera()
        vis.update_renderer()
        print("Created new static car box. Save with X to propagate it to all frames.")
        return False

    def select_label(delta: int):
        labels = state["label_items"]
        if not labels:
            print("No multi-label boxes in this frame")
            return False
        state["selected_label_idx"] = (state["selected_label_idx"] + delta) % len(labels)
        sync_bbox_from_selected_label()
        selected = selected_label()
        selected_kind = "static" if selected and selected.get("static") else "label"
        print(
            f"Selected box {state['selected_label_idx'] + 1}/{len(labels)} "
            f"({selected_kind}) {format_bbox(state['bbox'])}"
        )
        render(load_existing=False)
        return False

    def delete_selected_static_box(_vis=None):
        labels = state["label_items"]
        selected = selected_label()
        if selected is None:
            print("No selected box to delete")
            return False
        if not selected.get("static"):
            print("Refusing to delete a non-static/manual box")
            return False
        _frame_path, label_path = current_paths()
        removed = labels.pop(state["selected_label_idx"])
        state["selected_label_idx"] = min(
            state["selected_label_idx"],
            max(0, len(labels) - 1),
        )
        if labels:
            sync_bbox_from_selected_label()
            state["box_visible"] = True
        else:
            state["box_visible"] = False
        save_labels(label_path, labels)
        print(f"Deleted selected static box: {format_bbox(removed['bbox'])}")
        propagate_static_labels_from_current(allow_empty=True)
        render(load_existing=False)
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

    edit_keymap = {
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
        "4": lambda v: select_label(-1),
        "5": lambda v: select_label(1),
        "6": new_static_car_box,
        "N": next_frame,
        "B": prev_frame,
        "C": new_box,
        "P": copy_previous,
        "T": reset_template,
        "V": lambda v: (state.update({"ignored": False, "box_visible": not state["box_visible"]}), render(load_existing=False), False)[-1],
        "0": toggle_invalid_current,
        "Z": delete_current,
        "R": lambda v: (reset_camera(), vis.update_renderer(), False)[-1],
        "F": lambda v: (set_free_edit_camera(), vis.update_renderer(), False)[-1],
        "M": lambda v: (set_top_down_camera(), vis.update_renderer(), False)[-1],
        "X": save_current_and_propagate,
        "?": print_help,
    }
    view_keymap = {
        "4": lambda v: select_label(-1),
        "5": lambda v: select_label(1),
        "N": next_frame,
        "B": prev_frame,
        "R": lambda v: (reset_camera(), vis.update_renderer(), False)[-1],
        "F": lambda v: (set_free_edit_camera(), vis.update_renderer(), False)[-1],
        "M": lambda v: (set_top_down_camera(), vis.update_renderer(), False)[-1],
        "?": print_help,
    }
    keymap = view_keymap if args.view_only else edit_keymap

    for key, callback in keymap.items():
        vis.register_key_callback(ord(key), callback)
    if not args.view_only:
        for key_code in (48, 96):
            vis.register_key_callback(key_code, mark_invalid_current)
        vis.register_key_callback(ord("Y"), toggle_invalid_current)

    print(VIEW_HELP_TEXT if args.view_only else HELP_TEXT)
    render(load_existing=True)
    vis.run()
    vis.destroy_window()


HELP_TEXT = """
Manual bbox editor controls
---------------------------
Mouse       rotate / zoom / pan view
1/2/3       fine / normal / coarse edit steps
4/5         select previous / next box in multi-label frames
6           create a new static car box
W/S         move bbox +Y / -Y
A/D         move bbox -X / +X
Q/E         move bbox up / down
J/L         rotate yaw left / right
U/O         length + / -
I/K         width  + / -
G/H         height + / -
P           copy previous frame label; template dims/z stay locked
C           create a new class box and switch to free 3D edit camera
T           reset dimensions and template z to class/frame template
V           show / hide current box without deleting file
0/Y         toggle current frame invalid/uninvalid for dataset creation
Z           delete selected static box in this frame and all other frames
X           save current bbox JSON; static boxes are propagated to all frames
N/B         next / previous frame
R           reset camera
F           free 3D edit camera centered on current box
M           top-down camera
?           print this help
"""


VIEW_HELP_TEXT = """
Label viewer controls
---------------------
Mouse       rotate / zoom / pan view
N/B         next / previous frame
R           reset camera
F           free 3D camera centered on first box
M           top-down camera
?           print this help

View-only mode does not save, edit, delete or mark frames invalid.
Static labels are shown in green; regular labels use the class color.
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
        "--view-only",
        action="store_true",
        help="Show all labels in each frame without editing or writing files.",
    )
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
