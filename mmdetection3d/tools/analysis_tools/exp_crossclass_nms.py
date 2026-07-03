"""Ablation cross-class NMS als Nachverarbeitung (auf Test-Dumps).

Regel: Eine Prediction wird verworfen, wenn ihr BEV-Footprint zu
>= COVER in einer hoeher gescorten Prediction einer ANDEREN Klasse
liegt (Flaechenanteil der Schnittflaeche an der eigenen Flaeche).
Kein Retrain - nur Filterung der Dumps, dann Neuberechnung der
datensatzweiten AP (Matching identisch zur korrigierten Metrik).
"""
import glob
import json
import os
import pickle
import numpy as np
import torch
from mmcv.ops import diff_iou_rotated_3d, box_iou_rotated

RAW = os.path.expanduser("~/data/experiments")
DATA = os.path.expanduser("~/data/exp")
RES = os.path.expanduser(
    "~/projects/dcaiti_masterarbeit/mmdetection3d/results/exp")

CLASSES = ["person", "bicycle", "car"]
THRESHOLDS = [0.3, 0.6]
COVER = 0.8
DEVICE = "cuda:1"


def bev_cover(boxes):
    """cover[i,j] = Anteil der BEV-Flaeche von Box i, der in Box j liegt."""
    n = len(boxes)
    if n == 0:
        return np.zeros((0, 0))
    b = torch.tensor(boxes[:, [0, 1, 3, 4, 6]], dtype=torch.float32,
                     device=DEVICE)
    iou = box_iou_rotated(b, b).cpu().numpy()  # IoU der BEV-Rechtecke
    area = boxes[:, 3] * boxes[:, 4]
    inter = iou * (area[:, None] + area[None, :]) / (1.0 + iou)
    return inter / area[:, None]


def crossclass_filter(boxes, labels, scores):
    keep = np.ones(len(boxes), dtype=bool)
    if len(boxes) < 2:
        return keep
    cov = bev_cover(boxes)
    order = np.argsort(-scores)
    for i in order:
        if not keep[i]:
            continue
        for j in order:
            if j == i or not keep[j]:
                continue
            if scores[j] < scores[i] and labels[j] != labels[i] \
                    and cov[j, i] >= COVER:
                keep[j] = False
    return keep


def ap11(scores, tps, n_gt):
    if n_gt == 0:
        return float("nan")
    if len(scores) == 0:
        return 0.0
    order = np.argsort(-np.asarray(scores))
    tp = np.asarray(tps, dtype=np.float64)[order]
    tp_c = np.cumsum(tp)
    fp_c = np.cumsum(1.0 - tp)
    rec = tp_c / n_gt
    prec = tp_c / np.maximum(tp_c + fp_c, 1e-9)
    ap = 0.0
    for level in np.linspace(0, 1, 11):
        m = rec >= level - 1e-9
        ap += prec[m].max() if m.any() else 0.0
    return ap / 11.0


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


def evaluate(results, gt_by_sid, use_filter):
    pooled = {c: {t: {"s": [], "tp": [], "n": 0} for t in THRESHOLDS}
              for c in range(3)}
    removed = 0
    for r in results:
        sid = str(r.metainfo.get("sample_id", ""))
        smp = gt_by_sid.get(sid)
        if smp is None:
            continue
        pb = r.bboxes_3d.tensor.numpy()[:, :7]
        pl = np.asarray(r.labels_3d)
        ps = np.asarray(r.scores_3d)
        if use_filter:
            keep = crossclass_filter(pb, pl, ps)
            removed += int((~keep).sum())
            pb, pl, ps = pb[keep], pl[keep], ps[keep]
        gts = [(inst["bbox_label_3d"], np.array(inst["bbox_3d"][:7]))
               for inst in smp.get("instances", [])]
        for c in range(3):
            gb = [g for lb, g in gts if lb == c]
            idx = np.where(pl == c)[0]
            order = idx[np.argsort(-ps[idx])]
            iou = iou3d(pb[order], gb)
            for t in THRESHOLDS:
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
                    pooled[c][t]["s"].append(ps[order][k])
                    pooled[c][t]["tp"].append(1 if ok else 0)
    out = {}
    for c in range(3):
        for t in THRESHOLDS:
            d = pooled[c][t]
            out[f"AP{int(t * 100)}_{CLASSES[c]}"] = ap11(d["s"], d["tp"], d["n"])
    return out, removed


with open(os.path.join(DATA, "exp_kitti_infos_merged_test.pkl"), "rb") as f:
    gtd = pickle.load(f)
gt_by_sid = {str(s["sample_id"]): s for s in gtd["data_list"]}

dump = sorted(glob.glob(os.path.join(RES, "merged_gtsample_test_results_*.pkl")),
              key=os.path.getmtime)[-1]
with open(dump, "rb") as f:
    results = pickle.load(f)

base, _ = evaluate(results, gt_by_sid, use_filter=False)
filt, removed = evaluate(results, gt_by_sid, use_filter=True)
n_pred = sum(len(np.asarray(r.scores_3d)) for r in results)
print(f"merged gtsample, cross-class NMS (COVER>={COVER}): "
      f"{removed}/{n_pred} Predictions entfernt")
print(f"{'Metrik':16s} {'ohne':>8s} {'mit':>8s}")
for k in sorted(base):
    print(f"{k:16s} {base[k]:8.3f} {filt[k]:8.3f}")
