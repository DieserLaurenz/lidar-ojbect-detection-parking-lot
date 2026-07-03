"""Dynamic-vs-static test evaluation.

Splits the test GT by the `static` flag of the raw manual labels and
computes dataset-level AP11 separately:
- dynamic AP: static GT act as ignore regions (predictions matched to
  them are dropped from the ranking, neither TP nor FP)
- static AP: vice versa
Matching mirrors the fixed OSDaR23Metric (frame-local greedy by score,
3D rotated IoU with gravity-center z).
"""
import glob
import json
import os
import pickle
import numpy as np
import torch
from mmcv.ops import diff_iou_rotated_3d

RAW = os.path.expanduser("~/data/experiments")
DATA = os.path.expanduser("~/data/exp")
RES = os.path.expanduser(
    "~/projects/dcaiti_masterarbeit/mmdetection3d/results/exp")

THETA = np.deg2rad(90)
c, s = np.cos(THETA), np.sin(THETA)
R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
T = np.array([30, 0.0, -1.6])
CLASSES = ["person", "bicycle", "car"]
LUMPI_IDX = {"person": 0, "car": 1, "bicycle": 2, "bike": 2}
KITTI_MAP = {0: 0, 1: 2, 2: 1}
TYPE_NUM = {"merged": 2, "os0": 0, "os1": 1}
THRESHOLDS = [0.3, 0.6]
DEVICE = "cuda:0"


def build_static_lookup(view):
    """sample_id -> list of (label, box7_kitti, static_flag)"""
    lookup = {}
    for exp_dir in sorted(glob.glob(os.path.join(RAW, "*_experiment_*"))):
        meas = os.path.basename(exp_dir)[0]
        raw_dir = None
        for cand in [f"{view}_labels_manual_correct",
                     f"{view}_labels_manual_static",
                     f"{view}_labels_manual", f"{view}_labels"]:
            p = os.path.join(exp_dir, cand)
            if glob.glob(os.path.join(p, "*.json")):
                raw_dir = p
                break
        if raw_dir is None:
            continue
        for fp in glob.glob(os.path.join(raw_dir, "*.json")):
            with open(fp) as f:
                data = json.load(f)
            items = data if isinstance(data, list) else data.get("instances", [])
            frame = int(os.path.basename(fp).split(".")[0])
            sid = f"{meas}{TYPE_NUM[view]}0000{frame:06d}"
            entries = []
            for it in items:
                if not isinstance(it, dict) or "bbox" not in it:
                    continue
                b = np.array(it["bbox"][:7], dtype=np.float64)
                bb = b.copy()
                bb[:3] = R @ b[:3] + T
                bb[2] -= bb[5] / 2.0
                bb[6] = (b[6] + THETA + np.pi) % (2 * np.pi) - np.pi
                lb = KITTI_MAP[LUMPI_IDX[it["label"]]]
                entries.append((lb, bb, bool(it.get("static", False))))
            lookup[sid] = entries
    return lookup


def iou_matrix(pred, gt):
    if len(pred) == 0 or len(gt) == 0:
        return np.zeros((len(pred), len(gt)))
    p = torch.tensor(np.asarray(pred), dtype=torch.float32, device=DEVICE)
    g = torch.tensor(np.asarray(gt), dtype=torch.float32, device=DEVICE)
    p = p.clone(); g = g.clone()
    p[:, 2] += p[:, 5] / 2.0
    g[:, 2] += g[:, 5] / 2.0
    P = p.unsqueeze(1).expand(-1, g.shape[0], -1)
    G = g.unsqueeze(0).expand(p.shape[0], -1, -1)
    return diff_iou_rotated_3d(P, G).cpu().numpy()


def ap11(scores, tps, n_gt):
    if n_gt == 0:
        return float("nan")
    if len(scores) == 0:
        return 0.0
    order = np.argsort(-np.asarray(scores))
    tp = np.asarray(tps, dtype=np.float64)[order]
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(1.0 - tp)
    rec = tp_cum / n_gt
    prec = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    ap = 0.0
    for level in np.linspace(0, 1, 11):
        m = rec >= level - 1e-9
        ap += prec[m].max() if m.any() else 0.0
    return ap / 11.0


def evaluate(view, dump_path, target="dynamic"):
    lookup = LOOKUPS[view]
    with open(dump_path, "rb") as f:
        results = pickle.load(f)
    pooled = {c_: {t: {"s": [], "tp": [], "n": 0} for t in THRESHOLDS}
              for c_ in range(3)}
    unmatched_gt = 0
    with open(os.path.join(DATA, f"exp_kitti_infos_{view}_test.pkl"), "rb") as f:
        gtd = pickle.load(f)
    gt_by_sid = {str(s["sample_id"]): s for s in gtd["data_list"]}

    for r in results:
        sid = str(r.metainfo.get("sample_id", ""))
        smp = gt_by_sid.get(sid)
        raw = lookup.get(sid)
        if smp is None or raw is None:
            continue
        pb = r.bboxes_3d.tensor.numpy()[:, :7]
        pl = np.asarray(r.labels_3d)
        ps = np.asarray(r.scores_3d)
        order = np.argsort(-ps)
        pb, pl, ps = pb[order], pl[order], ps[order]

        # tag each pkl GT instance with static flag via nearest raw center
        gts, flags = [], []
        for inst in smp.get("instances", []):
            g = np.array(inst["bbox_3d"][:7])
            best, bd = None, 1e9
            for lb, bb, st in raw:
                if lb != inst["bbox_label_3d"]:
                    continue
                d = np.linalg.norm(bb[:3] - g[:3])
                if d < bd:
                    bd, best = d, st
            if best is None or bd > 0.1:
                unmatched_gt += 1
                best = False
            gts.append((inst["bbox_label_3d"], g))
            flags.append(best)

        for c_ in range(3):
            g_idx = [i for i, (lb, _) in enumerate(gts) if lb == c_]
            if target == "dynamic":
                is_target = [not flags[i] for i in g_idx]
            else:
                is_target = [flags[i] for i in g_idx]
            gboxes = [gts[i][1] for i in g_idx]
            p_idx = np.where(pl == c_)[0]
            pboxes = pb[p_idx]
            pscores = ps[p_idx]
            iou = iou_matrix(pboxes, gboxes)
            for t in THRESHOLDS:
                pooled[c_][t]["n"] += int(sum(is_target))
                matched = [False] * len(gboxes)
                for pi in range(len(pboxes)):
                    if len(gboxes):
                        mi = np.argmax(iou[pi])
                        mv = iou[pi][mi]
                    else:
                        mi, mv = -1, 0
                    if mv >= t and not matched[mi]:
                        matched[mi] = True
                        if is_target[mi]:
                            pooled[c_][t]["s"].append(pscores[pi])
                            pooled[c_][t]["tp"].append(1)
                        # else: ignore region -> drop from ranking
                    else:
                        pooled[c_][t]["s"].append(pscores[pi])
                        pooled[c_][t]["tp"].append(0)
    out = {}
    for c_ in range(3):
        for t in THRESHOLDS:
            d = pooled[c_][t]
            out[f"AP{int(t*100)}_{CLASSES[c_]}"] = ap11(d["s"], d["tp"], d["n"])
            out[f"n_{CLASSES[c_]}"] = d["n"]
    if unmatched_gt:
        print(f"  [WARN] {unmatched_gt} GT ohne raw-Match (als dynamisch gezaehlt)")
    return out


def latest(prefix):
    files = sorted(glob.glob(os.path.join(RES, prefix + "*.pkl")),
                   key=os.path.getmtime)
    return files[-1] if files else None


RUNS = [
    ("merged baseline", "merged", latest("merged_test_results")),
    ("merged gtsample", "merged", latest("merged_gtsample_test_results")),
    ("merged bikeover", "merged", latest("merged_bikeover_test_results")),
    ("os0", "os0", latest("os0_test_results")),
    ("os1", "os1", latest("os1_test_results")),
]

LOOKUPS = {}
for _, view, _p in RUNS:
    if view not in LOOKUPS:
        LOOKUPS[view] = build_static_lookup(view)

for name, view, dump in RUNS:
    if dump is None:
        print(f"=== {name}: kein Dump gefunden ===")
        continue
    print(f"=== {name} ({os.path.basename(dump)}) ===")
    for target in ["dynamic", "static"]:
        r = evaluate(view, dump, target=target)
        row = " ".join(
            f"{c}: AP30={r[f'AP30_{c}']:.3f} AP60={r[f'AP60_{c}']:.3f} (n={r[f'n_{c}']})"
            for c in CLASSES)
        print(f"  {target:8s} {row}")
