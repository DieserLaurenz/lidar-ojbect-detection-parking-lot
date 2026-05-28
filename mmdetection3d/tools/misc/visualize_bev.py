"""Bird's-eye-view visualization of predictions vs. ground truth.

Renders top-down PNGs: gray dots = LiDAR points, green = GT boxes,
colored = predictions (red=Car, blue=Pedestrian, orange=Cyclist).
"""
import argparse
import glob
import json
import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPoly
from matplotlib.collections import PatchCollection


CLS_NAMES = ["Pedestrian", "Cyclist", "Car"]
CLS_COLORS = {"Pedestrian": "tab:blue", "Cyclist": "tab:orange", "Car": "tab:red"}


def bbox_to_corners_2d(bbox):
    """7-DoF bbox (x,y,z,dx,dy,dz,yaw) -> 4 BEV corners."""
    x, y, _, dx, dy, _, yaw = bbox[:7]
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    half = np.array([[ dx/2,  dy/2],
                     [ dx/2, -dy/2],
                     [-dx/2, -dy/2],
                     [-dx/2,  dy/2]])
    return (half @ R.T) + np.array([x, y])


def draw_box(ax, corners, color, label=None, score=None, lw=1.2):
    poly = MplPoly(corners, closed=True, fill=False, ec=color, lw=lw)
    ax.add_patch(poly)
    if label is not None:
        text = label if score is None else f"{label} {score:.2f}"
        ax.text(corners[0, 0], corners[0, 1], text,
                color=color, fontsize=6, alpha=0.9)


def render_frame(points, gt_boxes, gt_labels, pred_boxes, pred_labels,
                 pred_scores, out_path, score_thr=0.3, x_range=(-30, 30),
                 y_range=(-30, 30)):
    fig, ax = plt.subplots(figsize=(8, 8), dpi=110)
    if points is not None and len(points):
        m = (points[:, 0] >= x_range[0]) & (points[:, 0] <= x_range[1]) & \
            (points[:, 1] >= y_range[0]) & (points[:, 1] <= y_range[1])
        p = points[m]
        ax.scatter(p[:, 0], p[:, 1], s=0.15, c="0.6", linewidths=0)

    for box, lbl in zip(gt_boxes, gt_labels):
        c = bbox_to_corners_2d(box)
        draw_box(ax, c, "lime", label=f"GT:{CLS_NAMES[int(lbl)]}", lw=1.4)

    for box, lbl, sc in zip(pred_boxes, pred_labels, pred_scores):
        if sc < score_thr:
            continue
        cls = CLS_NAMES[int(lbl)]
        col = CLS_COLORS.get(cls, "magenta")
        draw_box(ax, bbox_to_corners_2d(box), col, label=cls, score=float(sc))

    ax.set_aspect("equal")
    ax.set_xlim(*x_range)
    ax.set_ylim(*y_range)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title(os.path.basename(out_path))
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def extract_pred(r):
    p = getattr(r, "pred_instances_3d", r)
    boxes = getattr(p, "bboxes_3d", None)
    if boxes is None and isinstance(p, dict):
        boxes = p.get("bboxes_3d")
    if hasattr(boxes, "tensor"):
        boxes = boxes.tensor.cpu().numpy()
    elif hasattr(boxes, "numpy"):
        boxes = boxes.numpy()
    scores = getattr(p, "scores_3d", None)
    labels = getattr(p, "labels_3d", None)
    if hasattr(scores, "cpu"): scores = scores.cpu().numpy()
    if hasattr(labels, "cpu"): labels = labels.cpu().numpy()
    sample_idx = None
    meta = getattr(r, "metainfo", None)
    if meta:
        sample_idx = meta.get("sample_idx") or meta.get("sample_id") or \
                     meta.get("lidar_path", "").split("/")[-1].split(".")[0]
    return sample_idx, np.asarray(boxes), np.asarray(scores), np.asarray(labels)


def load_gt(pkl_path):
    """Returns dict: sample_idx -> (boxes, labels)"""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict) and "data_list" in data:
        data = data["data_list"]
    out = {}
    for item in data:
        sid = item.get("sample_id") or item.get("sample_idx")
        boxes, labels = [], []
        for inst in item.get("instances", []):
            boxes.append(inst["bbox_3d"])
            labels.append(inst.get("bbox_label_3d", inst.get("label", -1)))
        out[str(sid)] = (np.asarray(boxes), np.asarray(labels))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True, help="path to results pkl")
    ap.add_argument("--gt", default="data/exp/exp_kitti_infos_merged.pkl")
    ap.add_argument("--points-dir", default="data/exp/points_kitti")
    ap.add_argument("--out-dir", default="results/exp/vis_bev")
    ap.add_argument("--score-thr", type=float, default=0.3)
    ap.add_argument("--limit", type=int, default=20,
                    help="max frames to render (use 0 for all)")
    ap.add_argument("--stride", type=int, default=1,
                    help="render every Nth frame")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.result, "rb") as f:
        results = pickle.load(f)
    gt = load_gt(args.gt)
    print(f"Results: {len(results)} frames | GT samples: {len(gt)}")

    rendered = 0
    for i, r in enumerate(results):
        if args.stride > 1 and i % args.stride != 0:
            continue
        sid, boxes, scores, labels = extract_pred(r)
        sid_s = str(sid)
        gt_boxes, gt_labels = gt.get(sid_s, (np.empty((0, 7)), np.empty((0,))))
        pts_path = os.path.join(args.points_dir, f"{sid_s}.bin")
        points = None
        if os.path.exists(pts_path):
            points = np.fromfile(pts_path, dtype=np.float32).reshape(-1, 4)
        out = os.path.join(args.out_dir, f"{i:04d}_{sid_s}.png")
        render_frame(points, gt_boxes, gt_labels,
                     boxes, labels, scores, out, score_thr=args.score_thr)
        rendered += 1
        if args.limit and rendered >= args.limit:
            break
    print(f"Wrote {rendered} PNGs to {args.out_dir}")


if __name__ == "__main__":
    main()
