"""Exportiert einen kompletten CV-Testlauf als portable Video-Assets.

Das Skript läuft in der mmdet3d-Umgebung auf dem GPU-Server. Es verbindet
Test-Infos und Vorhersagen über die Sample-ID, beschränkt sie auf ein komplett
ungesehenes Experiment und packt nur die dafür benötigten v2-Punktwolken.
Es startet weder Training noch Inferenz.
"""

from __future__ import annotations

import argparse
import json
import pickle
import tarfile
from pathlib import Path

import numpy as np


def array(value) -> np.ndarray:
    if hasattr(value, "tensor"):
        value = value.tensor
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--info", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--points", type=Path, required=True)
    parser.add_argument("--view", choices=("merged", "os0", "os1"),
                        required=True)
    parser.add_argument("--experiment", default="1")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-tar", type=Path, required=True)
    args = parser.parse_args()

    with args.info.open("rb") as handle:
        samples = pickle.load(handle)["data_list"]
    with args.predictions.open("rb") as handle:
        predictions = pickle.load(handle)

    samples_by_id = {str(sample["sample_id"]): sample for sample in samples}
    frames = []
    point_paths = []
    for result in predictions:
        sample_id = str(result.metainfo.get("sample_id", ""))
        if sample_id[:1] != args.experiment:
            continue
        sample = samples_by_id[sample_id]
        lidar_path = sample["lidar_points"]["lidar_path"]
        boxes = array(result.bboxes_3d)[:, :7]
        labels = array(result.labels_3d).astype(int)
        scores = array(result.scores_3d)
        frames.append({
            "sample_id": sample_id,
            "lidar_path": lidar_path,
            "gt": [
                {
                    "box": instance["bbox_3d"][:7],
                    "label": int(instance["bbox_label_3d"]),
                }
                for instance in sample.get("instances", [])
            ],
            "predictions": [
                {
                    "box": box.tolist(),
                    "label": int(label),
                    "score": float(score),
                }
                for box, label, score in zip(boxes, labels, scores)
            ],
        })
        point_paths.append(args.points / lidar_path)

    frames.sort(key=lambda frame: frame["sample_id"])
    if not frames:
        raise RuntimeError(f"Keine Frames für Experiment {args.experiment}")
    missing = [str(path) for path in point_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} Punktwolken fehlen")

    payload = {
        "protocol": "experiment-held-out cross-validation",
        "experiment": args.experiment,
        "view": args.view,
        "fold": 1,
        "frames": frames,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    args.output_tar.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.output_tar, "w:gz") as archive:
        for frame in frames:
            point_path = args.points / frame["lidar_path"]
            archive.add(point_path, arcname=f"points/{point_path.name}")

    print(f"Frames: {len(frames)}")
    print(f"JSON: {args.output_json}")
    print(f"Punktwolken: {args.output_tar}")


if __name__ == "__main__":
    main()
