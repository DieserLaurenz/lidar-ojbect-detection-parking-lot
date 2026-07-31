# Multisensor LiDAR 3D Object Detection — Tiefgaragen-Experiment

Finetuning eines vortrainierten PointPillars-Modells (MMDetection3D, KITTI-3-Klassen-Checkpoint)
auf einem Zwei-Sensor-LiDAR-Datensatz (Ouster OS0 + OS1) aus einer Tiefgarage.
Evaluiert werden die Einzelsensor-Sichten (`os0`, `os1`) und die fusionierte
Sicht (`merged`) auf den Klassen **person, bicycle, car**.
Aufgabenstellung: `docs/Project Topic.txt`.

**Abgabe:** [`report/`](report/) enthält den Projektbericht (PDF und LaTeX-Quellen)
sowie die Abschlusspräsentation.

## Kernergebnisse (Experiment-Cross-Validation, Protokoll eingefroren 11.07.2026)

Der frühere temporale 80/10/10-Split war mit der Objektposition vermischt und
ist als Hauptbewertung abgelöst. Maßgeblich ist eine gepaarte
3-Fold-Cross-Validation mit je einem vollständig ungesehenen Auto-, Fahrrad-
und Personenexperiment pro Fold.

**Finetuning ist der entscheidende Schritt.** Ausschließlich vortrainierte
Modelle erreichen auf denselben Aufnahmen höchstens 0.058 mAP (Vorgängerarbeit
T. Pagel 2025), das finetunete Modell 0.798 — ein Abstand von 0.740 mAP und
damit sechs- bis sechzehnmal größer als der Abstand zwischen den beiden
untersuchten Architekturen.

**Fusion liefert die beste ausgewogene Gesamtleistung** (Fold-Mittel ± Std.):

| View | alle Objekte | bewegt, offiziell | bewegt, labelbasiert |
|---|---:|---:|---:|
| **merged** | **0.798 ± 0.037** | **0.784 ± 0.060** | **0.832 ± 0.038** |
| os0 | 0.777 ± 0.018 | 0.736 ± 0.037 | 0.746 ± 0.031 |
| os1 | 0.770 ± 0.027 | 0.744 ± 0.049 | 0.763 ± 0.049 |

Beschränkt auf bewegte Objekte wächst der Vorsprung gegenüber der besseren
Einzelsicht von 0.021 auf 0.040 (offiziell) bzw. 0.069 (labelbasiert). Der
Gesamtwert bewertet zugleich die Wiedererkennung der vielen geparkten Fahrzeuge.

**Der Vorteil liegt in der Lokalisierung, nicht im Finden.** Bei AP30 findet
jede Sicht jede bewegte Klasse; auf einzelnen Testexperimenten ist `os0` dort
sogar am stärksten (Mittel 0.914 ± 0.023 gegen 0.871 ± 0.063 für `merged`).
Bei AP60 führt `merged` bei Person und Fahrrad in **beiden** Auswertungsvarianten
deutlich. Nur beim Auto dreht die offizielle Variante die Rangfolge — Ursache ist
ein aufgeklärtes Zuordnungsartefakt an einem sensornahen geparkten Fahrzeug.

**Laufzeit:** Modellinferenz 85 ms (merged) / 45 ms (os0) / 43 ms (os1) auf einer
V100 — 10-Hz-fähig. Der Offline-Merge mit per-Frame-ICP braucht median 2.25 s;
ein Online-Merge mit einmalig kalibrierter Extrinsik ist noch nicht gemessen.

Vollständige Ergebnisse: [`results/CROSS_VALIDATION_RESULTS.md`](results/CROSS_VALIDATION_RESULTS.md).

**Trainierte Modelle:** Die bisherige GitHub Release v1.0 enthält die alten
Temporal-Split-Modelle. Die neun Cross-Validation-Checkpoints liegen derzeit
auf dem DCAITI-Server unter `~/runs/pointpillars_crossval/`.

## Projektstruktur

```
report/                Abgabe: Projektbericht (PDF + LaTeX) und Abschlusspräsentation
results/               Alle Ergebnisse: Tabellen, Abbildungen, Predictions, Audits
docs/                  Aufgabenstellung, Anleitungen, Vorgängerarbeit (T. Pagel 2025)
experiment/            Datenaufbereitung: ROS2-Export, PCD-Merge, Labeling-Tools, Viewer
mmdetection3d/         Angepasstes MMDetection3D (Converter, Configs, Metrik, Analyse-Skripte)
data/                  Experiment-Rohdaten und Labels (9 Experimente, 3 Sichten)
archive/               Überholte Skripte und Zwischenstände (nicht mehr in Verwendung)
```

## Einstiegspunkte

- **Bericht lesen:** [`report/Projektbericht.pdf`](report/Projektbericht.pdf)
- **Ergebnisse ansehen:** [`results/README.md`](results/README.md)
- **Detektionen interaktiv inspizieren:** `python experiment/prediction_viewer.py --view merged`
- **Training/Evaluation reproduzieren:** Configs unter
  `mmdetection3d/configs/pointpillars/*-exp*.py`, Datenkonvertierung über
  `mmdetection3d/tools/create_data.py exp ...`; Cross-Validation-Splits über
  `mmdetection3d/tools/dataset_converters/create_exp_crossval_splits.py`,
  Trainingssteuerung `mmdetection3d/tools/run_exp_crossval.sh`
- **Bericht neu bauen:** [`report/README.md`](report/README.md)
