"""Thesis-Abbildungen: PR-Kurven + Score-Verteilungen (merged Test-Split).

Fig 1: Precision-Recall je Klasse bei IoU 0.3 und 0.6,
       gtsample (durchgezogen) vs. Baseline (gestrichelt).
Fig 2: Score-Histogramme TP vs FP je Klasse (gtsample, IoU 0.3)
       + Panel: bester Pred-Score dynamisches vs. statisches Auto.
"""
import glob
import json
import os
import pickle
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mmcv.ops import diff_iou_rotated_3d

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
CLASSES = ["person", "bicycle", "car"]
COLORS = {"person": "tab:blue", "bicycle": "tab:orange", "car": "tab:red"}
DEVICE = "cuda:1"
LUMPI = {"person": 0, "car": 2, "bicycle": 1, "bike": 1}


def iou3d(pred, gt):
    if len(pred) == 0 or len(gt) == 0:
        return np.zeros((len(pred), len(gt)))
    p = torch.tensor(np.asarray(pred), dtype=torch.float32, device=DEVICE).clone()
    g = torch.tensor(np.asarray(gt), dtype=torch.float32, device=DEVICE).clone()
    p[:, 2] += p[:, 5] / 2
    g[:, 2] += g[:, 5] / 2
    P = p.unsqueeze(1).expand(-1, g.shape[0], -1)
    G = g.unsqueeze(0).expand(p.shape[0], -1, -1)
    return diff_iou_rotated_3d(P.contiguous(), G.contiguous()).cpu().numpy()


def pool(dump_path, thresholds=(0.3, 0.6)):
    with open(os.path.join(DATA, "exp_kitti_infos_merged_test.pkl"), "rb") as f:
        gtd = pickle.load(f)
    gt_by_sid = {str(s["sample_id"]): s for s in gtd["data_list"]}
    with open(dump_path, "rb") as f:
        results = pickle.load(f)
    pooled = {c: {t: {"s": [], "tp": [], "n": 0} for t in thresholds}
              for c in range(3)}
    for r in results:
        sid = str(r.metainfo.get("sample_id", ""))
        smp = gt_by_sid.get(sid)
        if smp is None:
            continue
        pb = r.bboxes_3d.tensor.numpy()[:, :7]
        pl = np.asarray(r.labels_3d)
        ps = np.asarray(r.scores_3d)
        gts = [(inst["bbox_label_3d"], np.array(inst["bbox_3d"][:7]))
               for inst in smp.get("instances", [])]
        for c in range(3):
            gb = [g for lb, g in gts if lb == c]
            idx = np.where(pl == c)[0]
            order = idx[np.argsort(-ps[idx])]
            iou = iou3d(pb[order], gb)
            for t in thresholds:
                pooled[c][t]["n"] += len(gb)
                matched = [False] * len(gb)
                for k in range(len(order)):
                    if len(gb):
                        mi = int(np.argmax(iou[k]))
                        mv = iou[k][mi]
                    else:
                        mi, mv = -1, 0
                    ok = mv >= t and not matched[mi]
                    if ok:
                        matched[mi] = True
                    pooled[c][t]["s"].append(float(ps[order][k]))
                    pooled[c][t]["tp"].append(1 if ok else 0)
    return pooled


def pr_curve(d):
    order = np.argsort(-np.asarray(d["s"]))
    tp = np.asarray(d["tp"], dtype=np.float64)[order]
    tp_c = np.cumsum(tp)
    fp_c = np.cumsum(1.0 - tp)
    rec = tp_c / max(d["n"], 1)
    prec = tp_c / np.maximum(tp_c + fp_c, 1e-9)
    return rec, prec


def latest(prefix):
    return sorted(glob.glob(os.path.join(RES, prefix + "_*.pkl")),
                  key=os.path.getmtime)[-1]


gts = pool(latest("merged_gtsample_test_results"))
base = pool(latest("merged_test_results"))

# --- Fig 1: PR-Kurven ---------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
for ax, t in zip(axes, (0.3, 0.6)):
    for c in range(3):
        r1, p1 = pr_curve(gts[c][t])
        r0, p0 = pr_curve(base[c][t])
        ax.plot(r1, p1, color=COLORS[CLASSES[c]], lw=1.8, label=CLASSES[c])
        ax.plot(r0, p0, color=COLORS[CLASSES[c]], lw=1.2, ls="--", alpha=0.6)
    ax.set_title(f"IoU ≥ {t}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
axes[0].legend(loc="lower left", fontsize=9,
               title="durchgezogen: GT-Sampling\ngestrichelt: Baseline")
fig.suptitle("Precision-Recall, merged Test-Split (datensatzweites Matching)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "pr_curves_merged.png"), dpi=170)

# --- Fig 2: Score-Verteilungen ------------------------------------------
# TP/FP-Scores bei IoU 0.3 (gtsample)
fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
bins = np.linspace(0.3, 1.0, 15)
for c in range(3):
    d = gts[c][0.3]
    s = np.asarray(d["s"]); tp = np.asarray(d["tp"], bool)
    axes[c].hist(s[tp], bins=bins, alpha=0.65, label="TP",
                 color="tab:green")
    axes[c].hist(s[~tp], bins=bins, alpha=0.65, label="FP",
                 color="tab:red")
    axes[c].set_title(CLASSES[c])
    axes[c].set_xlabel("Score")
    axes[c].legend(fontsize=8)
axes[0].set_ylabel("Anzahl Predictions")

# dyn vs stat car best-pred-scores (aus raw labels)
dyn, stat = [], []
with open(os.path.join(DATA, "exp_kitti_infos_merged_test.pkl"), "rb") as f:
    gtd = pickle.load(f)
with open(latest("merged_gtsample_test_results"), "rb") as f:
    results = {str(r.metainfo.get("sample_id", "")): r for r in pickle.load(f)}
for smp in gtd["data_list"]:
    sid = str(smp["sample_id"])
    meas, ts = sid[0], sid[6:]
    exp_dir = glob.glob(os.path.join(RAW, f"{meas}_experiment_*"))[0]
    raw_items = []
    for cand in ["merged_labels_manual_correct", "merged_labels_manual_static",
                 "merged_labels_manual", "merged_labels"]:
        fp = os.path.join(exp_dir, cand, f"{ts}.json")
        if os.path.exists(fp):
            with open(fp) as f:
                dd = json.load(f)
            raw_items = dd if isinstance(dd, list) else dd.get("instances", [])
            break
    r = results.get(sid)
    if r is None:
        continue
    pb = r.bboxes_3d.tensor.numpy()[:, :7]
    pl = np.asarray(r.labels_3d)
    ps = np.asarray(r.scores_3d)
    cars = pb[pl == 2][:, :2]
    scs = ps[pl == 2]
    for it in raw_items:
        if not isinstance(it, dict) or "bbox" not in it:
            continue
        if LUMPI.get(it["label"], -1) != 2:
            continue
        gt = (R @ np.array(it["bbox"][:3]) + T)[:2]
        best = 0.0
        for p, sc in zip(cars, scs):
            if np.linalg.norm(p - gt) < 2.0 and sc > best:
                best = float(sc)
        (stat if it.get("static", False) else dyn).append(best)
b21 = np.linspace(0, 1, 21)
axes[3].hist(stat, bins=b21, density=True, alpha=0.55,
             label=f"statisch (n={len(stat)})", color="tab:gray")
axes[3].hist(dyn, bins=b21, density=True, alpha=0.6,
             label=f"dynamisch (n={len(dyn)})", color="tab:purple")
axes[3].set_title("car: bester Score je GT (normiert)")
axes[3].set_xlabel("Score")
axes[3].set_ylabel("Dichte")
axes[3].legend(fontsize=8)
fig.suptitle("Score-Verteilungen, merged Test-Split, GT-Sampling-Modell (IoU 0.3)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "score_distributions_merged.png"), dpi=170)
print("saved pr_curves_merged.png + score_distributions_merged.png")
