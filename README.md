# Multisensor LiDAR 3D Object Detection — Tiefgaragen-Experiment

Finetuning eines vortrainierten PointPillars-Modells (MMDetection3D, KITTI-3-Klassen-Checkpoint)
auf einem Zwei-Sensor-LiDAR-Datensatz (Ouster OS0 + OS1) aus einer Tiefgarage.
Evaluiert werden die Einzelsensor-Sichten (`os0`, `os1`) und die fusionierte
Sicht (`merged`) auf den Klassen **person, bicycle, car**.
Aufgabenstellung: `docs/Project Topic.txt`.

## Kernergebnis

| View | Test-mAP (AP30–60) | Konfiguration |
|---|---|---|
| **merged** | **0.8916** | Finetuning + GT-Sampling |
| os1 | 0.8570 | Finetuning + GT-Sampling |
| os0 | 0.8514 | Finetuning (Baseline) |

Die Sensorfusion gewinnt durchgängig; beim dynamischen Fahrzeug ist sie
entscheidend (os1 allein: AP30 0.09 — merged: 0.90). Details, alle
AP30/40/50/60-Tabellen und Fehleranalysen: [`results/RESULTS.md`](results/RESULTS.md).

**Trainierte Modelle (Checkpoints):** [GitHub Release v1.0](https://github.com/DieserLaurenz/lidar-ojbect-detection-parking-lot/releases/tag/v1.0)

## Projektstruktur

```
PROJEKTBERICHT.md      Ausführlicher Projektbericht (Deutsch)
PROJEKTBERICHT_ZH.md   项目报告（中文版）
results/               Alle Ergebnisse: Tabellen, Abbildungen, Predictions, Audits
docs/                  Aufgabenstellung, Anleitungen, Vorgängerarbeit (T. Pagel 2025)
experiment/            Datenaufbereitung: ROS2-Export, PCD-Merge, Labeling-Tools, Viewer
mmdetection3d/         Angepasstes MMDetection3D (Converter, Configs, Metrik, Analyse-Skripte)
data/                  Experiment-Rohdaten und Labels (9 Experimente, 3 Sichten)
archive/               Überholte Skripte und Zwischenstände (nicht mehr in Verwendung)
```

## Einstiegspunkte

- **Ergebnisse ansehen:** `results/README.md`
- **Detektionen interaktiv inspizieren:** `python experiment/prediction_viewer.py --view merged`
- **Training/Evaluation reproduzieren:** Configs unter
  `mmdetection3d/configs/pointpillars/*-exp*.py`, Datenkonvertierung über
  `mmdetection3d/tools/create_data.py exp ...` (Details: `PROJEKTBERICHT.md` §4–§6)
