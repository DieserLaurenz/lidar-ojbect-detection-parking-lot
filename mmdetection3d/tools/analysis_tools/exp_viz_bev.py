"""BEV figures for the thesis: GT vs predictions.

Panels:
  A) os1 (gtsample), frame with dynamic car -> systematic mislocalization
  B) merged (gtsample), SAME timestamp -> fusion fixes it
  C) merged (gtsample), bicycle frame
  D) merged (gtsample), dynamic person frame
Output: /tmp/thesis_figs/*.png
"""
import glob
import json
import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

RAW = os.path.expanduser("~/data/experiments")
DATA = os.path.expanduser("~/data/exp")
RES = os.path.expanduser(
    "~/projects/dcaiti_masterarbeit/mmdetection3d/results/exp")
OUT = "/tmp/thesis_figs"
os.makedirs(OUT, exist_ok=True)

THETA = np.deg2rad(90)
c_, s_ = np.cos(THETA), np.sin(THETA)
R = np.array([[c_, -s_, 0], [s_, c_, 0], [0, 0, 1]])
T = np.array([30, 0.0, -1.6])
LUMPI_IDX = {"person": 0, "car": 1, "bicycle": 2, "bike": 2}
KITTI_MAP = {0: 0, 1: 2, 2: 1}
CLASSES = ["person", "bicycle", "car"]
SCORE_THR = 0.3


def load_dump(prefix):
    f = sorted(glob.glob(os.path.join(RES, prefix + "_test_results_*.pkl")),
               key=os.path.getmtime)[-1]
    with open(f, "rb") as fh:
        return {str(r.metainfo.get("sample_id", "")): r for r in pickle.load(fh)}


def load_infos(view):
    with open(os.path.join(DATA, f"exp_kitti_infos_{view}_test.pkl"), "rb") as f:
        d = pickle.load(f)
    return {str(s["sample_id"]): s for s in d["data_list"]}


def raw_items(view, exp_dir, ts):
    for cand in [f"{view}_labels_manual_correct", f"{view}_labels_manual_static",
                 f"{view}_labels_manual", f"{view}_labels"]:
        fp = os.path.join(exp_dir, cand, f"{ts}.json")
        if os.path.exists(fp):
            with open(fp) as f:
                data = json.load(f)
            return data if isinstance(data, list) else data.get("instances", [])
    return []


def static_flags(view, sid, smp):
    meas, ts = sid[0], sid[6:]
    exp_dir = glob.glob(os.path.join(RAW, f"{meas}_experiment_*"))[0]
    raw = []
    for it in raw_items(view, exp_dir, ts):
        if not isinstance(it, dict) or "bbox" not in it:
            continue
        b = np.array(it["bbox"][:7], dtype=np.float64)
        bb = b.copy()
        bb[:3] = R @ b[:3] + T
        raw.append((KITTI_MAP[LUMPI_IDX[it["label"]]], bb,
                    bool(it.get("static", False))))
    flags = []
    for inst in smp.get("instances", []):
        g = np.array(inst["bbox_3d"][:7])
        gc = g.copy(); gc[2] += gc[5] / 2
        best, bd = False, 1e9
        for lb, bb, st in raw:
            if lb != inst["bbox_label_3d"]:
                continue
            d = np.linalg.norm(bb[:3] - gc[:3])
            if d < bd:
                bd, best = d, st
        flags.append(best if bd < 0.1 else False)
    return flags


def corners(b):
    x, y, l, w, yaw = b[0], b[1], b[3], b[4], b[6]
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s], [s, c]])
    pts = np.array([[l, w], [l, -w], [-l, -w], [-l, w]]) / 2
    return pts @ rot.T + [x, y]


def draw_panel(ax, sid, smp, r, center, half=11.0, title="", target=None):
    pts = np.fromfile(os.path.join(DATA, "points_kitti", sid + ".bin"),
                      dtype=np.float32).reshape(-1, 4)[:, :3]
    m = ((np.abs(pts[:, 0] - center[0]) < half) &
         (np.abs(pts[:, 1] - center[1]) < half))
    p = pts[m]
    ax.scatter(p[:, 0], p[:, 1], s=0.4, c=p[:, 2], cmap="Greys",
               vmin=-2.2, vmax=1.5, rasterized=True)
    for inst in smp.get("instances", []):
        g = np.array(inst["bbox_3d"][:7])
        if np.abs(g[0] - center[0]) > half or np.abs(g[1] - center[1]) > half:
            continue
        ax.add_patch(Polygon(corners(g), closed=True, fill=False,
                             edgecolor="tab:green", lw=1.8, zorder=5))
    pb = r.bboxes_3d.tensor.numpy()[:, :7]
    ps = np.asarray(r.scores_3d)
    for b, s in zip(pb, ps):
        if s < SCORE_THR:
            continue
        if np.abs(b[0] - center[0]) > half or np.abs(b[1] - center[1]) > half:
            continue
        ax.add_patch(Polygon(corners(b), closed=True, fill=False,
                             edgecolor="tab:red", lw=1.5, ls="--", zorder=6))
        ax.annotate(f"{s:.2f}", (b[0], b[1]), color="tab:red", fontsize=7,
                    ha="center", va="bottom", zorder=7)
    if target is not None:
        off = max(2.5, 0.25 * half)
        ax.annotate("Zielobjekt", xy=(target[0], target[1]),
                    xytext=(target[0] - off, target[1] + off),
                    color="tab:orange", fontsize=9, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="tab:orange",
                                    lw=1.6), zorder=8)
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")


# ---- pick frames ------------------------------------------------------
infos_m = load_infos("merged")
infos_1 = load_infos("os1")
dump_m = load_dump("merged_gtsample")
dump_1 = load_dump("os1_gtsample")

# A/B: dynamic-car frame where os1 pred is far off but GT has many points
best_pick, best_score = None, -1
for sid1, smp1 in infos_1.items():
    if sid1[0] not in "123":
        continue
    r1 = dump_1.get(sid1)
    if r1 is None:
        continue
    flags = static_flags("os1", sid1, smp1)
    for inst, st in zip(smp1.get("instances", []), flags):
        if inst["bbox_label_3d"] != 2 or st:
            continue
        g = np.array(inst["bbox_3d"][:7])
        pb = r1.bboxes_3d.tensor.numpy()[:, :7]
        pl = np.asarray(r1.labels_3d)
        ps = np.asarray(r1.scores_3d)
        cars = pb[(pl == 2) & (ps >= SCORE_THR)]
        dmin = (np.linalg.norm(cars[:, :2] - g[:2], axis=1).min()
                if len(cars) else 99)
        sid_m = sid1[0] + "2" + sid1[2:]
        if sid_m not in infos_m or sid_m not in dump_m:
            continue
        score = min(dmin, 3.0) + inst.get("num_lidar_pts", 0) / 1000.0
        if score > best_score:
            best_score = score
            best_pick = (sid1, sid_m, g)

sid1, sid_m, g_car = best_pick
print("Panel A/B Frame:", sid1, sid_m, "os1-GT-center:", g_car[:2])

# C: bicycle frame (merged) with bike GT far from origin clutter
pick_c = None
for sid, smp in infos_m.items():
    if sid[0] not in "456":
        continue
    if sid not in dump_m:
        continue
    for inst in smp.get("instances", []):
        if inst["bbox_label_3d"] == 1 and inst.get("num_lidar_pts", 0) > 400:
            pick_c = (sid, np.array(inst["bbox_3d"][:7]))
if pick_c is None:
    raise SystemExit("kein bike frame")
print("Panel C Frame:", pick_c[0])

# D: dynamic person frame (merged)
pick_d = None
for sid, smp in infos_m.items():
    if sid[0] not in "789":
        continue
    if sid not in dump_m:
        continue
    flags = static_flags("merged", sid, smp)
    for inst, st in zip(smp.get("instances", []), flags):
        if inst["bbox_label_3d"] == 0 and not st \
                and inst.get("num_lidar_pts", 0) > 150:
            pick_d = (sid, np.array(inst["bbox_3d"][:7]))
if pick_d is None:
    raise SystemExit("kein person frame")
print("Panel D Frame:", pick_d[0])

# ---- render -----------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(11, 10.5))
draw_panel(axes[0, 0], sid1, infos_1[sid1], dump_1[sid1], g_car[:2],
           title="(a) os1: dynamisches Auto — keine gültige Detektion",
           target=g_car[:2])
g_m = None
for inst in infos_m[sid_m].get("instances", []):
    if inst["bbox_label_3d"] == 2:
        gg = np.array(inst["bbox_3d"][:7])
        if np.linalg.norm(gg[:2] - g_car[:2]) < 2.5:
            g_m = gg
center_b = g_m[:2] if g_m is not None else g_car[:2]
draw_panel(axes[0, 1], sid_m, infos_m[sid_m], dump_m[sid_m], center_b,
           title="(b) merged: gleicher Zeitpunkt — korrekte Detektion",
           target=center_b)
draw_panel(axes[1, 0], pick_c[0], infos_m[pick_c[0]], dump_m[pick_c[0]],
           pick_c[1][:2], half=9.0,
           title="(c) merged: Fahrrad", target=pick_c[1][:2])
draw_panel(axes[1, 1], pick_d[0], infos_m[pick_d[0]], dump_m[pick_d[0]],
           pick_d[1][:2], half=9.0,
           title="(d) merged: dynamische Person", target=pick_d[1][:2])
import matplotlib.lines as mlines
handles = [mlines.Line2D([], [], color="tab:green", lw=1.8, label="Ground Truth"),
           mlines.Line2D([], [], color="tab:red", lw=1.5, ls="--",
                         label=f"Prediction (Score ≥ {SCORE_THR})")]
fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=10)
fig.suptitle("PointPillars-Detektionen (BEV), Test-Split, GT-Sampling-Modelle",
             fontsize=12)
fig.tight_layout(rect=[0, 0.03, 1, 0.97])
fig.savefig(os.path.join(OUT, "qualitative_bev.png"), dpi=170)
print("gespeichert:", os.path.join(OUT, "qualitative_bev.png"))
