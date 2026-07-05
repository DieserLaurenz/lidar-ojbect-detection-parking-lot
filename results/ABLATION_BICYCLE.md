# Ablation: Maßnahmen gegen die bicycle-Klassenimbalance

**Datum:** 2026-07-02
**Baseline:** `merged_ft_v2` (Daten v2, test mAP 0.6532; bicycle AP30 nur 0.239
bei car:bicycle ≈ 22:1 im Train). Diagnose laut `DATA_AUDIT.md` §7: Die
bicycle-AP ist über die IoU-Schwellen nahezu konstant → Recall-Problem
(zu wenige Positives im Training), kein Lokalisierungsproblem. Die
Anchor-Größen passen (bike-Median 2.0×0.73×1.7 vs. Anchor 1.76×0.6×1.73).

## Versuchsaufbau

Zwei unabhängige Runs gegen die unveränderte v2-Baseline, identische
Hyperparameter (50 Epochen, lr 1e-4, KITTI-Checkpoint, `save_best`):

### A) GT-Sampling (`merged_ft_v2_gtsample`)

Copy-Paste-Augmentierung (SECOND/PointPillars-Standard): GT-Database
**ausschließlich aus dem merged-Train-Split** gebaut (kein Leakage aus
val/test) mit `tools/dataset_converters/create_exp_gt_database.py`:

```bash
python tools/dataset_converters/create_exp_gt_database.py \
    --root data/exp --info exp_kitti_infos_merged_train.pkl
# → exp_kitti_dbinfos_merged_train.pkl + exp_gt_database_merged_train/
#   person: 3256, bicycle: 562 (median 675 pts), car: 12676 Instanzen
```

Config `...exp-merged-gtsample.py`: `ObjectSample` nach `LoadAnnotations3D`,
`sample_groups=dict(person=8, bicycle=10)` (car wird nicht gesampelt),
`filter_by_min_points=10`. Der Sampler pastet Instanzen an ihren
Original-Szenenpositionen aus anderen Train-Frames (BEV-Kollisionscheck
gegen vorhandene GT inkl. der statischen Autos) — geometrisch konsistent,
da alle Frames dieselbe Szene teilen.

### B) Frame-Oversampling (`merged_ft_v2_bikeover`)

Repeat-Factor-Sampling (Gupta et al., LVIS) via `ClassBalancedDataset`
mit `oversample_thr=1.0` (Config `...exp-merged-bikeover.py`).
Wiederholungsfaktor pro Frame: max_c √(1/f(c)); mit f(bicycle) ≈ 0.26
werden bike-Frames ~2× pro Epoche gesehen (Kappungsgrenze der Methode —
mehr als ~2× ist mit Repeat-Factor-Sampling bei dieser Frequenz nicht
möglich). Effektiv 710 statt 355 Iterationen/Epoche.

## Einschränkung (fürs Thesis-Kapitel)

Der Datensatz enthält nur **3 physische Fahrräder** (Experimente 4–6).
Beide Maßnahmen erhöhen Anzahl/Positionsdiversität der Trainings-Positives,
nicht die Formdiversität. Val/Test enthalten dieselben Räder an späteren
Zeitpunkten (temporaler Split) — gemessen wird also die Erkennung bekannter
Radtypen an neuen Positionen.

## Wichtige Neubewertung während der Ablation

Die Diagnose "bicycle AP30 = 0.239" entpuppte sich als **Metrik-Artefakt**
(Details in `DATA_AUDIT.md` §3.5): Die alte Metrik mittelte Pro-Frame-APs
über alle Frames, auch solche ohne bicycle-GT — Obergrenze für bicycle auf
dem merged-Test-Split war dadurch 70/266 × 0.909 = 0.2392. Eine Analyse der
Prediction-Dumps zeigte, dass bereits die Baseline **70/70 Test-Bikes
erkennt**. Die Metrik wurde auf datensatzweites AP korrigiert; alle Zahlen
unten stammen aus der korrigierten Metrik.

## Ergebnisse (Test-Split, korrigierte Metrik)

| Run | test mAP | AP30 bicycle | AP60 bicycle | AP30 person | AP30 car |
|---|---|---|---|---|---|
| Baseline (`merged_ft_v2`) | 0.8880 | 0.935 | 0.856 | 0.900 | 0.909 |
| GT-Sampling (`merged_ft_v2_gtsample`) | **0.8916** | **0.972** | 0.883 | 0.903 | 0.909 |
| Oversampling (`merged_ft_v2_bikeover`) | 0.8788 | 0.890 | 0.881 | 0.901 | 0.909 |

### Dynamisch/statisch getrennt (Skript `tools/analysis_tools/exp_eval_dynamic_static.py`)

| Run | dyn. bicycle AP30/AP60 (n=70) | dyn. car AP30/AP60 (n=35) | dyn. person AP30/AP60 (n=229) |
|---|---|---|---|
| Baseline | 0.935 / 0.856 | 0.697 / 0.125 | 0.892 / 0.710 |
| GT-Sampling | **0.972** / **0.883** | **0.902** / **0.191** | 0.898 / 0.706 |
| Oversampling | 0.890 / 0.881 | 0.619 / 0.086 | 0.894 / **0.736** |

### Anmerkung zum bikeover-Checkpoint (Transparenz)

Der Run crashte bei Epoche 37 (Metrik-Code wurde während des Laufs
deployt; die spawn-Worker der Validierung re-importieren Module von der
Platte → Formatkonflikt) und wurde mit `--resume` ab `epoch_37.pth`
fortgesetzt. Val-Epochen 1–37 liegen damit auf der alten Metrik-Skala,
38–50 auf der neuen; der `save_best`-Hook wählte zwangsläufig aus der
neuen Phase (Ep. 49, val mAP 0.8606). Der Alt-Phasen-Kandidat
`best_epoch_28` wurde von mmengine beim Speichern des neuen Best
automatisch gelöscht, ein Nachtest war nicht mehr möglich. Risiko gering:
val mAP plateaut über die letzten 6 Epochen bei 0.859–0.861 (konvergiert,
lr→0). Getestet wurde `best_osdar23_mAP_epoch_49.pth`.

## Fazit

**GT-Sampling ist der klare Sieger** und wird als Standard-Maßnahme
übernommen: bestes test mAP (0.8916), bicycle AP30 +3.7 Pkt. und AP60
+2.7 Pkt. gegenüber Baseline, und als einziger Run eine deutliche
Verbesserung beim dynamischen Zielauto (AP30 0.70 → 0.90) — ohne
person/car zu verschlechtern.

**Frame-Oversampling lohnt sich nicht:** test mAP unter Baseline
(0.8788), bicycle AP30 sogar schlechter (0.890 vs. 0.935; nur AP60
leicht besser), dynamisches car deutlich schlechter (AP30 0.62). Das
Wiederholen ganzer Frames (~2× für Bike-Frames) verschiebt die
Datenverteilung, ohne neue Positiv-Konstellationen zu erzeugen — die
Kappungsgrenze des Repeat-Factor-Samplings bei dieser Klassenfrequenz
ist zu niedrig, um zu wirken, und das Übergewicht der Bike-Frames geht
zulasten anderer Szenenteile. Einziger Pluspunkt: person AP60
(0.74 dyn. / 0.81 stat.) — für die Thesis als Nebenbefund erwähnbar.

Der ursprüngliche Leidensdruck (vermeintliche bicycle-Schwäche AP 0.24)
bestand ohnehin nicht — er war ein Messfehler (§ oben).

## Nachtrag (2026-07-03): GT-Sampling auf os0/os1 übertragen

Für einen methodisch einheitlichen View-Vergleich wurde die
Gewinner-Konfiguration auch für die Einzelsensor-Views trainiert
(GT-Databases strikt aus dem jeweiligen Train-Split; os0: 496,
os1: 478 bicycle-Instanzen). Best-Checkpoint-Wahl durchgängig auf der
korrigierten Metrik.

| Run | test mAP | AP30 bicycle | AP60 bicycle | dyn. car AP30 |
|---|---|---|---|---|
| os0 Baseline | 0.8514 | 0.837 | 0.551 | 0.634 |
| os0 + GT-Sampling | 0.8412 | **0.982** | 0.496 | 0.735 |
| os1 Baseline | 0.8457 | 0.804 | 0.681 | 0.021 |
| os1 + GT-Sampling | **0.8570** | 0.863 | **0.791** | 0.091 |

Befunde:
- **bicycle AP30 steigt überall** (os0 +14.5, os1 +5.9 Punkte), auf os1
  auch AP60 (+11.0) und das mAP (+1.1). Auf os0 kippt der Trade-off:
  Detektion ↑, Lokalisierungsschärfe ↓ (bicycle AP60 −5.5, person AP60
  −3.2, pred/GT-Ratio 1.17) → mAP leicht unter Baseline. Bei der dünner
  besetzten os0-Punktwolke erzeugen die eingefügten Instanzen offenbar
  mehr unpräzise Zusatzdetektionen.
- **Das dynamische Auto bleibt auf os1 kaputt (0.02 → 0.09)** — auch die
  beste Trainingskonfiguration behebt die Fehllokalisierung bei
  einseitiger Sicht nicht. Das stärkt das Fusionsargument: merged +
  GT-Sampling erreicht 0.90.
- View-Ranking bei einheitlicher Konfiguration:
  **merged (0.8916) > os1 (0.8570) > os0 (0.8412)**.
