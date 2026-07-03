"""Ablation 'Unterboden-Filter': merged-Bins ohne Punkte unter z=-2.

Boden liegt (Daten v2) bei z ~ -1.74; Spiegelreflexionen bei ~ -2.0
und tiefer. Filter: behalte z >= -2.0. Nur merged-Frames (sid[1]=='2'),
Ausgabe nach points_kitti_zfilt/ (Original bleibt unangetastet).
"""
import glob
import os
import numpy as np

SRC = os.path.expanduser("~/data/exp/points_kitti")
DST = os.path.expanduser("~/data/exp/points_kitti_zfilt")
os.makedirs(DST, exist_ok=True)

Z_MIN = -2.0
n, dropped_tot, pts_tot = 0, 0, 0
for fp in sorted(glob.glob(os.path.join(SRC, "*.bin"))):
    sid = os.path.basename(fp)[:-4]
    if sid[1] != "2":
        continue
    out = os.path.join(DST, sid + ".bin")
    if os.path.exists(out):
        continue
    pts = np.fromfile(fp, dtype=np.float32).reshape(-1, 4)
    m = pts[:, 2] >= Z_MIN
    pts[m].tofile(out)
    n += 1
    dropped_tot += int((~m).sum())
    pts_tot += len(pts)
print(f"{n} merged-Bins gefiltert; entfernt: {dropped_tot} von {pts_tot} "
      f"Punkten ({dropped_tot / max(pts_tot, 1) * 100:.2f}%)")
