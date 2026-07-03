"""Why is dynamic-car AP60 low on merged?

Three checks on the merged test split:
1. Cross-view label offset: same frame timestamp, compare the labeled
   center of the dynamic car in os0 vs os1 vs merged raw labels.
   A systematic os0-os1 offset = the two sweeps captured the car at
   different times -> the merged cloud is smeared.
2. Point-spread: points inside the (expanded) dynamic-car GT box in the
   merged cloud, spread along the heading axis vs. box length; compared
   to static cars in the same frames.
3. Prediction error decomposition: best pred (by IoU) per dynamic-car GT
   from the merged gtsample dump: longitudinal/lateral/z center offset,
   size and yaw error, IoU.
"""
import glob
import json
import os
import pickle
import numpy as np
import torch
from mmcv.ops import diff_iou_rotated_3d
from mmdet3d.structures.ops import box_np_ops

RAW = os.path.expanduser("~/data/experiments")
DATA = os.path.expanduser("~/data/exp")
RES = os.path.expanduser(
    "~/projects/dcaiti_masterarbeit/mmdetection3d/results/exp")

THETA = np.deg2rad(90)
c_, s_ = np.cos(THETA), np.sin(THETA)
R = np.array([[c_, -s_, 0], [s_, c_, 0], [0, 0, 1]])
T = np.array([30, 0.0, -1.6])
LUMPI_IDX = {"person": 0, "car": 1, "bicycle": 2, "bike": 2}
KITTI_MAP = {0: 0, 1: 2, 2: 1}
DEVICE = "cuda:0"


def load_raw(view, exp_dir, ts):
    for cand in [f"{view}_labels_manual_correct", f"{view}_labels_manual_static",
                 f"{view}_labels_manual", f"{view}_labels"]:
        fp = os.path.join(exp_dir, cand, f"{ts}.json")
        if os.path.exists(fp):
            with open(fp) as f:
                data = json.load(f)
            return data if isinstance(data, list) else data.get("instances", [])
    return None


def tf(b):
    bb = np.array(b[:7], dtype=np.float64)
    bb[:3] = R @ bb[:3] + T
    bb[6] = (b[6] + THETA + np.pi) % (2 * np.pi) - np.pi
    return bb  # gravity-center z


def dyn_cars(items):
    out = []
    for it in items or []:
        if not isinstance(it, dict) or "bbox" not in it:
            continue
        if KITTI_MAP[LUMPI_IDX[it["label"]]] == 2 and not it.get("static", False):
            out.append(tf(it["bbox"]))
    return out


# ---- collect merged test frames with a dynamic car -------------------
with open(os.path.join(DATA, "exp_kitti_infos_merged_test.pkl"), "rb") as f:
    gtd = pickle.load(f)

frames = []  # (sample_id, ts, exp_dir, gt_box7_bottom)
for s in gtd["data_list"]:
    sid = str(s["sample_id"])
    meas, ts = sid[0], sid[6:]
    exp_dir = glob.glob(os.path.join(RAW, f"{meas}_experiment_*"))[0]
    dcs = dyn_cars(load_raw("merged", exp_dir, ts))
    if not dcs:
        continue
    # match to PKL instance (bottom-center) for the exact GT used in eval
    best = None
    for inst in s.get("instances", []):
        if inst["bbox_label_3d"] != 2:
            continue
        g = np.array(inst["bbox_3d"][:7])
        gc = g.copy()
        gc[2] += gc[5] / 2
        d = np.linalg.norm(gc[:3] - dcs[0][:3])
        if d < 0.1:
            best = g
    if best is not None:
        frames.append((sid, ts, exp_dir, best))

print(f"merged-Testframes mit dynamischem Auto: {len(frames)}")

# ---- 1. cross-view label offset --------------------------------------
d01, d0m, d1m, speeds = [], [], [], []
prev = None
for sid, ts, exp_dir, g in frames:
    c_m = dyn_cars(load_raw("merged", exp_dir, ts))
    c_0 = dyn_cars(load_raw("os0", exp_dir, ts))
    c_1 = dyn_cars(load_raw("os1", exp_dir, ts))
    if c_0 and c_1:
        d01.append(np.linalg.norm(c_0[0][:2] - c_1[0][:2]))
    if c_0 and c_m:
        d0m.append(np.linalg.norm(c_0[0][:2] - c_m[0][:2]))
    if c_1 and c_m:
        d1m.append(np.linalg.norm(c_1[0][:2] - c_m[0][:2]))
    if prev is not None and c_m:
        dt = (int(ts) - int(prev[0])) / 1e9
        if 0 < dt < 0.5:
            speeds.append(np.linalg.norm(c_m[0][:2] - prev[1][:2]) / dt)
    if c_m:
        prev = (ts, c_m[0])

def stats(a):
    a = np.asarray(a)
    return f"n={len(a)} median={np.median(a):.3f} p90={np.percentile(a, 90):.3f} max={a.max():.3f}" if len(a) else "n=0"

print("\n[1] Label-Versatz desselben dyn. Autos zum selben Timestamp (BEV, m):")
print(f"  os0 vs os1   : {stats(d01)}")
print(f"  os0 vs merged: {stats(d0m)}")
print(f"  os1 vs merged: {stats(d1m)}")
print(f"  Geschwindigkeit (merged-Label, m/s): {stats(speeds)}")

# ---- 2. point spread along heading ------------------------------------
def spread(sid, box_bottom, expand=1.4):
    pts = np.fromfile(os.path.join(DATA, "points_kitti", sid + ".bin"),
                      dtype=np.float32).reshape(-1, 4)[:, :3]
    b = box_bottom.copy()
    b[3] *= expand
    b[4] *= expand
    m = box_np_ops.points_in_rbbox(pts, b[None])[:, 0]
    p = pts[m]
    if len(p) < 20:
        return None
    h = np.array([np.cos(box_bottom[6]), np.sin(box_bottom[6])])
    proj = (p[:, :2] - box_bottom[:2]) @ h
    return np.percentile(proj, 99) - np.percentile(proj, 1)

dyn_sp, dyn_len = [], []
for sid, ts, exp_dir, g in frames:
    sp = spread(sid, g)
    if sp is not None:
        dyn_sp.append(sp)
        dyn_len.append(g[3])

stat_sp, stat_len = [], []
rng = np.random.default_rng(0)
smp = [s for s in gtd["data_list"] if str(s["sample_id"])[0] in "123"]
for s in rng.choice(smp, size=min(40, len(smp)), replace=False):
    sid = str(s["sample_id"])
    meas, ts = sid[0], sid[6:]
    exp_dir = glob.glob(os.path.join(RAW, f"{meas}_experiment_*"))[0]
    raw = load_raw("merged", exp_dir, ts) or []
    statics = [tf(it["bbox"]) for it in raw
               if isinstance(it, dict) and "bbox" in it
               and KITTI_MAP[LUMPI_IDX[it["label"]]] == 2 and it.get("static", False)]
    for inst in s.get("instances", []):
        if inst["bbox_label_3d"] != 2:
            continue
        g = np.array(inst["bbox_3d"][:7])
        gc = g.copy(); gc[2] += gc[5] / 2
        if not any(np.linalg.norm(gc[:3] - st[:3]) < 0.1 for st in statics):
            continue
        sp = spread(sid, g)
        if sp is not None:
            stat_sp.append(sp); stat_len.append(g[3])

dyn_ratio = np.array(dyn_sp) / np.array(dyn_len)
stat_ratio = np.array(stat_sp) / np.array(stat_len)
print("\n[2] Punkt-Ausdehnung entlang Fahrzeugachse / GT-Laenge (merged):")
print(f"  dynamisch: {stats(dyn_ratio)}")
print(f"  statisch : {stats(stat_ratio)}")

# ---- 3. prediction error decomposition --------------------------------
dump = sorted(glob.glob(os.path.join(RES, "merged_gtsample_test_results_*.pkl")),
              key=os.path.getmtime)[-1]
with open(dump, "rb") as f:
    results = pickle.load(f)
res_by_sid = {str(r.metainfo.get("sample_id", "")): r for r in results}

rows = []
for sid, ts, exp_dir, g in frames:
    r = res_by_sid.get(sid)
    if r is None:
        continue
    pb = r.bboxes_3d.tensor.numpy()[:, :7]
    pl = np.asarray(r.labels_3d)
    ps = np.asarray(r.scores_3d)
    cars = pb[pl == 2]
    scs = ps[pl == 2]
    if not len(cars):
        rows.append(dict(iou=0.0)); continue
    gt_t = torch.tensor(g[None], dtype=torch.float32, device=DEVICE).clone()
    pd_t = torch.tensor(cars, dtype=torch.float32, device=DEVICE).clone()
    gt_t[:, 2] += gt_t[:, 5] / 2
    pd_t[:, 2] += pd_t[:, 5] / 2
    iou = diff_iou_rotated_3d(pd_t.unsqueeze(0),
                              gt_t.expand(len(cars), -1).unsqueeze(0))[0]
    iou = iou.cpu().numpy()
    i = int(np.argmax(iou))
    p = cars[i]
    h = np.array([np.cos(g[6]), np.sin(g[6])])
    n = np.array([-h[1], h[0]])
    dxy = p[:2] - g[:2]
    dyaw = (p[6] - g[6] + np.pi / 2) % np.pi - np.pi / 2
    rows.append(dict(iou=float(iou[i]), score=float(scs[i]),
                     dlong=float(dxy @ h), dlat=float(dxy @ n),
                     dz=float((p[2] + p[5] / 2) - (g[2] + g[5] / 2)),
                     dl=float(p[3] - g[3]), dw=float(p[4] - g[4]),
                     dh=float(p[5] - g[5]), dyaw=float(np.rad2deg(dyaw))))

print(f"\n[3] Bestes Pred je dyn. car-GT (merged gtsample, n={len(rows)}):")
ious = np.array([r["iou"] for r in rows])
print(f"  IoU: median={np.median(ious):.3f}  <0.6: {(ious < 0.6).sum()}/{len(ious)}"
      f"  <0.3: {(ious < 0.3).sum()}/{len(ious)}")
for k in ["dlong", "dlat", "dz", "dl", "dw", "dh", "dyaw", "score"]:
    v = np.array([r[k] for r in rows if k in r])
    if len(v):
        print(f"  {k:5s}: median={np.median(v):+.3f}  |median|={np.median(np.abs(v)):.3f}"
              f"  p90(|.|)={np.percentile(np.abs(v), 90):.3f}")
print("\nGT-Groesse (median l/w/h):",
      np.median([f[3][3] for f in frames]),
      np.median([f[3][4] for f in frames]),
      np.median([f[3][5] for f in frames]))
