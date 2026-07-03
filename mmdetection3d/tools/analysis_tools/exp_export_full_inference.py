"""Inferenz ueber ALLE Frames (nicht nur Test) fuer den Prediction-Viewer.

GT-Sampling-Best-Checkpoints je View; Ausgabe wie export_preds.py
(predictions_{view}_full.json), Score-Cut 0.2 gegen Dateigroesse.
"""
import glob
import json
import os
import numpy as np
from mmdet3d.apis import LidarDet3DInferencer

DATA = os.path.expanduser("~/data/exp/points_kitti")
CFG = os.path.expanduser(
    "~/projects/dcaiti_masterarbeit/mmdetection3d/configs/pointpillars")
RUNS = os.path.expanduser("~/runs/pointpillars")
OUT = "/tmp/pred_export"
os.makedirs(OUT, exist_ok=True)

VIEWS = {  # view -> (type-digit, config, run-dir)
    "merged": ("2", "exp-merged-gtsample", "merged_ft_v2_gtsample"),
    "os0": ("0", "exp-os0-gtsample", "os0_ft_v2_gtsample"),
    "os1": ("1", "exp-os1-gtsample", "os1_ft_v2_gtsample"),
}
SCORE_CUT = 0.2

for view, (digit, cfg_suffix, run_dir) in VIEWS.items():
    cfg = os.path.join(
        CFG, f"pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class-{cfg_suffix}.py")
    ckpt = sorted(glob.glob(os.path.join(RUNS, run_dir, "best_*.pth")))[-1]
    print(f"== {view}: {os.path.basename(ckpt)}")
    inf = LidarDet3DInferencer(model=cfg, weights=ckpt, device="cuda:0")
    bins = sorted(glob.glob(os.path.join(DATA, "*.bin")))
    bins = [b for b in bins if os.path.basename(b)[1] == digit]
    out = {}
    B = 32
    for i in range(0, len(bins), B):
        batch = [dict(points=b) for b in bins[i:i + B]]
        res = inf(batch, batch_size=len(batch), return_datasamples=True,
                  show=False, print_result=False)
        for fp, ds in zip(bins[i:i + B], res["predictions"]):
            sid = os.path.basename(fp)[:-4]
            inst = ds.pred_instances_3d
            boxes = inst.bboxes_3d.tensor.cpu().numpy()[:, :7]
            scores = inst.scores_3d.cpu().numpy()
            labels = inst.labels_3d.cpu().numpy()
            m = scores >= SCORE_CUT
            out[sid] = {
                "experiment": sid[0],
                "ts": sid[6:],
                "boxes": np.round(boxes[m], 4).tolist(),
                "labels": labels[m].tolist(),
                "scores": np.round(scores[m], 4).tolist(),
            }
        if (i // B) % 20 == 0:
            print(f"  {i + len(batch)}/{len(bins)}")
    fp = os.path.join(OUT, f"predictions_{view}_full.json")
    with open(fp, "w") as f:
        json.dump({"source": f"{run_dir}/{os.path.basename(ckpt)} (full inference)",
                   "box_frame": "kitti (x,y,z_bottom,dx,dy,dz,yaw)",
                   "classes": ["person", "bicycle", "car"],
                   "frames": out}, f)
    print(fp, len(out), "frames")
