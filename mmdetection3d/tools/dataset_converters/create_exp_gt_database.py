"""Create a GT database for ObjectSample from an exp KITTI train PKL.

Crops the points inside every GT box of the given train split and stores
them relative to the box position, in the format expected by
``mmdet3d.datasets.transforms.dbsampler.DataBaseSampler``.

Only ever run this on a *train* split: sampling objects from val/test
frames into training would leak evaluation data.

Usage:
    python tools/dataset_converters/create_exp_gt_database.py \
        --root data/exp --info exp_kitti_infos_merged_train.pkl
"""
import argparse
import pickle
from collections import Counter
from pathlib import Path

import numpy as np

from mmdet3d.structures.ops import box_np_ops

CLASSES = ["person", "bicycle", "car"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/exp")
    parser.add_argument("--info", default="exp_kitti_infos_merged_train.pkl")
    parser.add_argument("--points-dir", default="points_kitti")
    args = parser.parse_args()

    assert "_train" in args.info, (
        "GT database must be built from a train split only "
        "(val/test would leak into training)")

    root = Path(args.root)
    view_split = args.info.replace("exp_kitti_infos_", "").replace(".pkl", "")
    db_dir = root / f"exp_gt_database_{view_split}"
    db_dir.mkdir(parents=True, exist_ok=True)
    out_info = root / f"exp_kitti_dbinfos_{view_split}.pkl"

    with (root / args.info).open("rb") as f:
        data = pickle.load(f)

    db_infos = {name: [] for name in CLASSES}
    group_counter = 0
    stats = Counter()
    pts_per_class = {name: [] for name in CLASSES}

    for sample in data["data_list"]:
        instances = sample.get("instances", [])
        if not instances:
            continue
        image_idx = sample["sample_idx"]
        bin_path = root / args.points_dir / sample["lidar_points"]["lidar_path"]
        points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)

        boxes = np.array(
            [inst["bbox_3d"][:7] for inst in instances], dtype=np.float32)
        labels = [inst["bbox_label_3d"] for inst in instances]

        # bottom-center convention, matching LiDARInstance3DBoxes
        masks = box_np_ops.points_in_rbbox(points[:, :3], boxes)

        for i, (label, box) in enumerate(zip(labels, boxes)):
            name = CLASSES[label]
            gt_points = points[masks[:, i]].copy()
            if gt_points.shape[0] == 0:
                stats[f"skipped_empty_{name}"] += 1
                continue
            gt_points[:, :3] -= box[:3]

            filename = f"{image_idx}_{name}_{i}.bin"
            with (db_dir / filename).open("wb") as f:
                gt_points.tofile(f)

            db_infos[name].append({
                "name": name,
                "path": f"{db_dir.name}/{filename}",
                "image_idx": image_idx,
                "gt_idx": i,
                "box3d_lidar": box,
                "num_points_in_gt": int(gt_points.shape[0]),
                "difficulty": 0,
                "group_id": group_counter,
            })
            group_counter += 1
            stats[name] += 1
            pts_per_class[name].append(gt_points.shape[0])

    with out_info.open("wb") as f:
        pickle.dump(db_infos, f)

    print(f"Wrote {out_info}")
    for name in CLASSES:
        pts = pts_per_class[name]
        med = int(np.median(pts)) if pts else 0
        print(f"  {name}: {stats[name]} instances "
              f"(median pts {med}, skipped empty {stats[f'skipped_empty_{name}']})")


if __name__ == "__main__":
    main()
