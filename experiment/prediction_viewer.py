"""Interaktiver Viewer: Experiment-Punktwolken + NN-Predictions (+ GT).

Zeigt die rohen .pcd-Frames eines Experiments zusammen mit den
Bounding-Box-Predictions der finetuned PointPillars-Modelle
(GT-Sampling-Runs, Test-Split) und optional den manuellen Labels.

Die Predictions liegen im KITTI-Trainingskoordinatensystem und werden
hier zurueck ins rohe Sensor-Koordinatensystem transformiert
(Inverse von exp_to_kitti.py: Rz(90deg), T=[30,0,-1.6], z bottom->center).

Voraussetzung: predictions/predictions_<view>.json im Projekt-Root
(Export vom GPU-Server, tools-Skript export_preds.py). Predictions
existieren nur fuer Test-Frames (letzte ~10% jedes Experiments).

Benutzung:
    python experiment/prediction_viewer.py --experiment 1 --view merged
    python experiment/prediction_viewer.py -e 4 -v os0 --score-thr 0.5
    python experiment/prediction_viewer.py -e 1 --full   # ALLE Frames
                       # (braucht predictions_<view>_full.json)

Tasten:
    N / Pfeil rechts   naechster Frame
    P / Pfeil links    voriger Frame
    G                  Ground-Truth-Boxen ein/aus
    B                  Predictions ein/aus
    +/-                Score-Schwelle +-0.05 (fuer gewaehlte Klasse)
    1 / 2 / 3          +/- wirkt nur auf person / bicycle / car
    0                  +/- wirkt auf alle Klassen (Standard)
    Q / Esc            beenden

Tipp: Die Geisterraeder (Score 0.3-0.6) und das schwach gescorte
dynamische Auto ueberlappen im Score — eine globale Schwelle entfernt
beide. Stattdessen: Taste 2, dann zweimal +  (nur bicycle auf 0.4+).

Farben:  Prediction: person=blau, bicycle=orange, car=rot
         Ground Truth: gruen (statisch: grau)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS = ROOT / "data" / "2025_10_09" / "Experiment-Data" / "experiments"
PREDICTIONS = ROOT / "predictions"

THETA = np.deg2rad(90)
_c, _s = np.cos(THETA), np.sin(THETA)
R_KITTI = np.array([[_c, -_s, 0], [_s, _c, 0], [0, 0, 1]])
T_KITTI = np.array([30.0, 0.0, -1.6])

CLASSES = ["person", "bicycle", "car"]
PRED_COLORS = {0: (0.1, 0.35, 1.0), 1: (1.0, 0.65, 0.0), 2: (1.0, 0.15, 0.1)}
GT_COLOR = (0.1, 0.8, 0.1)
GT_STATIC_COLOR = (0.55, 0.55, 0.55)


def kitti_to_raw(box7):
    """[x,y,z_bottom,dx,dy,dz,yaw] (KITTI) -> [x,y,z_center,dx,dy,dz,yaw] roh."""
    b = np.asarray(box7, dtype=np.float64).copy()
    b[2] += b[5] / 2.0                      # bottom -> gravity center
    b[:3] = R_KITTI.T @ (b[:3] - T_KITTI)   # inverse Transformation
    b[6] = (b[6] - THETA + np.pi) % (2 * np.pi) - np.pi
    return b


def obb_lineset(box7_center, color):
    x, y, z, dx, dy, dz, yaw = box7_center
    rot = o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, yaw])
    obb = o3d.geometry.OrientedBoundingBox([x, y, z], rot, [dx, dy, dz])
    ls = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb)
    ls.paint_uniform_color(color)
    return ls


def find_exp_dir(exp_id):
    hits = sorted(EXPERIMENTS.glob(f"{exp_id}_experiment_*"))
    if not hits:
        raise SystemExit(f"Kein Experiment-Ordner {exp_id}_experiment_* unter {EXPERIMENTS}")
    return hits[0]


def load_gt(exp_dir, view, ts):
    for cand in [f"{view}_labels_manual_correct", f"{view}_labels_manual_static",
                 f"{view}_labels_manual", f"{view}_labels"]:
        fp = exp_dir / cand / f"{ts}.json"
        if fp.exists():
            with open(fp) as f:
                data = json.load(f)
            items = data if isinstance(data, list) else data.get("instances", [])
            return [it for it in items if isinstance(it, dict) and "bbox" in it]
    return []


class Viewer:
    def __init__(self, exp_id, view, score_thr, show_gt, full=False):
        self.view = view
        self.thr = {0: score_thr, 1: score_thr, 2: score_thr}
        self.thr_class = None  # None = +/- wirkt auf alle Klassen
        self.show_gt = show_gt
        self.show_pred = True
        self.exp_dir = find_exp_dir(exp_id)

        suffix = "_full" if full else ""
        pred_file = PREDICTIONS / f"predictions_{view}{suffix}.json"
        if not pred_file.exists():
            raise SystemExit(f"{pred_file} fehlt — erst vom Server exportieren.")
        with open(pred_file) as f:
            data = json.load(f)
        self.source = data["source"]
        self.frames = sorted(
            [v for v in data["frames"].values() if v["experiment"] == str(exp_id)],
            key=lambda v: int(v["ts"]))
        if not self.frames:
            raise SystemExit(f"Keine Predictions fuer Experiment {exp_id} in {pred_file}")
        self.idx = 0
        print(f"{len(self.frames)} Test-Frames mit Predictions "
              f"(Experiment {exp_id}, {view}, Modell-Dump: {self.source})")

        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(
            f"Predictions — Experiment {exp_id} ({view})", 1600, 900)
        opt = self.vis.get_render_option()
        opt.point_size = 1.5
        opt.background_color = np.array([0.05, 0.05, 0.08])
        for key, fn in [(ord("N"), self.next), (262, self.next),
                        (ord("P"), self.prev), (263, self.prev),
                        (ord("G"), self.toggle_gt), (ord("B"), self.toggle_pred),
                        (ord("+"), self.thr_up), (ord("-"), self.thr_down),
                        (61, self.thr_up), (45, self.thr_down),
                        (ord("0"), lambda _: self.set_thr_class(None)),
                        (ord("1"), lambda _: self.set_thr_class(0)),
                        (ord("2"), lambda _: self.set_thr_class(1)),
                        (ord("3"), lambda _: self.set_thr_class(2))]:
            self.vis.register_key_callback(key, fn)
        self.geoms = []
        self.load_frame(first=True)

    # --- key callbacks (return True = redraw) ---------------------------
    def next(self, _=None):
        self.idx = (self.idx + 1) % len(self.frames)
        self.load_frame(); return True

    def prev(self, _=None):
        self.idx = (self.idx - 1) % len(self.frames)
        self.load_frame(); return True

    def toggle_gt(self, _=None):
        self.show_gt = not self.show_gt
        self.load_frame(); return True

    def toggle_pred(self, _=None):
        self.show_pred = not self.show_pred
        self.load_frame(); return True

    def set_thr_class(self, c_):
        self.thr_class = c_
        name = "alle Klassen" if c_ is None else CLASSES[c_]
        print(f"+/- wirkt jetzt auf: {name}")
        return False

    def _shift_thr(self, delta):
        keys = [self.thr_class] if self.thr_class is not None else [0, 1, 2]
        for k in keys:
            self.thr[k] = min(0.95, max(0.0, self.thr[k] + delta))
        self.load_frame(); return True

    def thr_up(self, _=None):
        return self._shift_thr(+0.05)

    def thr_down(self, _=None):
        return self._shift_thr(-0.05)

    # --------------------------------------------------------------------
    def load_frame(self, first=False):
        fr = self.frames[self.idx]
        ts = fr["ts"]
        pcd_path = self.exp_dir / f"{self.view}_pcd" / f"{ts}.pcd"
        if not pcd_path.exists():
            print(f"[WARN] {pcd_path} fehlt"); return

        cam = None
        if not first:
            cam = self.vis.get_view_control().convert_to_pinhole_camera_parameters()
        for g in self.geoms:
            self.vis.remove_geometry(g, reset_bounding_box=False)
        self.geoms = []

        pcd = o3d.io.read_point_cloud(str(pcd_path))
        pts = np.asarray(pcd.points)
        z = pts[:, 2] if len(pts) else np.zeros(0)
        col = np.empty((len(pts), 3))
        t = np.clip((z - z.min()) / max(np.ptp(z), 1e-6), 0, 1)[:, None] if len(pts) else z
        col = 0.25 + 0.65 * np.repeat(t, 3, axis=1) if len(pts) else col
        pcd.colors = o3d.utility.Vector3dVector(col)
        self.vis.add_geometry(pcd, reset_bounding_box=first)
        self.geoms.append(pcd)

        n_pred = 0
        counts = {c: 0 for c in CLASSES}
        if self.show_pred:
            for b, lb, sc in zip(fr["boxes"], fr["labels"], fr["scores"]):
                if sc < self.thr[int(lb)]:
                    continue
                ls = obb_lineset(kitti_to_raw(b), PRED_COLORS[int(lb)])
                self.vis.add_geometry(ls, reset_bounding_box=False)
                self.geoms.append(ls)
                n_pred += 1
                counts[CLASSES[int(lb)]] += 1

        n_gt = 0
        if self.show_gt:
            for it in load_gt(self.exp_dir, self.view, ts):
                color = GT_STATIC_COLOR if it.get("static", False) else GT_COLOR
                ls = obb_lineset(np.asarray(it["bbox"][:7], dtype=np.float64), color)
                self.vis.add_geometry(ls, reset_bounding_box=False)
                self.geoms.append(ls)
                n_gt += 1

        if cam is not None:
            self.vis.get_view_control().convert_from_pinhole_camera_parameters(cam)
        cnt = ", ".join(f"{k}:{v}" for k, v in counts.items() if v)
        thr_s = "/".join(f"{self.thr[i]:.2f}" for i in range(3))
        print(f"[{self.idx + 1}/{len(self.frames)}] ts={ts}  "
              f"preds (thr p/b/c {thr_s}): {n_pred} ({cnt})  GT: {n_gt}"
              f"{'' if self.show_gt else ' (aus)'}")

    def run(self):
        self.vis.run()
        self.vis.destroy_window()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("-e", "--experiment", required=True,
                    help="Experiment-Nr. 1-9 (1-3 car, 4-6 bike, 7-9 person)")
    ap.add_argument("-v", "--view", default="merged",
                    choices=["merged", "os0", "os1"])
    ap.add_argument("--score-thr", type=float, default=0.3)
    ap.add_argument("--no-gt", action="store_true",
                    help="Ground-Truth-Boxen nicht anzeigen")
    ap.add_argument("--full", action="store_true",
                    help="alle Frames statt nur Test-Split "
                         "(predictions_<view>_full.json)")
    ap.add_argument("--check", action="store_true",
                    help="Nur Daten pruefen, kein Fenster oeffnen")
    args = ap.parse_args()

    if args.check:
        exp_dir = find_exp_dir(args.experiment)
        suffix = "_full" if args.full else ""
        with open(PREDICTIONS / f"predictions_{args.view}{suffix}.json") as f:
            data = json.load(f)
        frames = [v for v in data["frames"].values()
                  if v["experiment"] == str(args.experiment)]
        frames.sort(key=lambda v: int(v["ts"]))
        ok = sum((exp_dir / f"{args.view}_pcd" / f"{f['ts']}.pcd").exists()
                 for f in frames)
        gt = sum(bool(load_gt(exp_dir, args.view, f["ts"])) for f in frames)
        b = kitti_to_raw(frames[0]["boxes"][0]) if frames and frames[0]["boxes"] else None
        print(f"Experiment {args.experiment} / {args.view}: "
              f"{len(frames)} Pred-Frames, {ok} pcd vorhanden, {gt} mit GT")
        if b is not None:
            print("Beispiel-Predbox (roh, x,y,z,dx,dy,dz,yaw):", np.round(b, 2))
        return

    Viewer(args.experiment, args.view, args.score_thr,
           show_gt=not args.no_gt, full=args.full).run()


if __name__ == "__main__":
    main()
