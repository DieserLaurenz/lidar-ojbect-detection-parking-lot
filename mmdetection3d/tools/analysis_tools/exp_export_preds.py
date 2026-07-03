"""Export test-split predictions to plain JSON for the local viewer."""
import glob
import json
import os
import pickle
import numpy as np

RES = os.path.expanduser(
    "~/projects/dcaiti_masterarbeit/mmdetection3d/results/exp")
OUT = "/tmp/pred_export"
os.makedirs(OUT, exist_ok=True)

RUNS = {
    "merged": "merged_gtsample_test_results",
    "os0": "os0_gtsample_test_results",
    "os1": "os1_gtsample_test_results",
}

for view, prefix in RUNS.items():
    dump = sorted(glob.glob(os.path.join(RES, prefix + "_*.pkl")),
                  key=os.path.getmtime)[-1]
    with open(dump, "rb") as f:
        results = pickle.load(f)
    out = {}
    for r in results:
        sid = str(r.metainfo.get("sample_id", ""))
        out[sid] = {
            "experiment": sid[0],
            "ts": sid[6:],
            "boxes": np.round(r.bboxes_3d.tensor.numpy()[:, :7], 4).tolist(),
            "labels": np.asarray(r.labels_3d).tolist(),
            "scores": np.round(np.asarray(r.scores_3d), 4).tolist(),
        }
    fp = os.path.join(OUT, f"predictions_{view}.json")
    with open(fp, "w") as f:
        json.dump({"source": os.path.basename(dump),
                   "box_frame": "kitti (x,y,z_bottom,dx,dy,dz,yaw)",
                   "classes": ["person", "bicycle", "car"],
                   "frames": out}, f)
    print(fp, len(out), "frames")
