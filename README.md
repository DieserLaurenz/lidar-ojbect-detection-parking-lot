# Multisensor LiDAR 3D Object Detection — Tiefgaragen-Experiment

Finetuning eines vortrainierten PointPillars-Modells (MMDetection3D, KITTI-3-Klassen-Checkpoint)
auf einem Zwei-Sensor-LiDAR-Datensatz (Ouster OS0 + OS1) aus einer Tiefgarage.
Evaluiert werden die Einzelsensor-Sichten (`os0`, `os1`) und die fusionierte
Sicht (`merged`) auf den Klassen **person, bicycle, car**.
Aufgabenstellung: `docs/Project Topic.txt`.

## Kernergebnis (Experiment-Cross-Validation, 11.07.2026)

Der frühere temporale 80/10/10-Split war mit der Objektposition vermischt und
ist als Hauptbewertung abgelöst. Maßgeblich ist jetzt eine gepaarte
3-Fold-Cross-Validation mit je einem vollständig ungesehenen Auto-, Fahrrad-
und Personenexperiment pro Fold.

| View | Fold-mAP Mittel ± Std. | Gepooltes OOF-mAP | Konfiguration |
|---|---:|---:|---|
| **merged** | **0.7979 ± 0.0371** | **0.789** | Finetuning + GT-Sampling |
| os0 | 0.7772 ± 0.0181 | 0.759 | Finetuning + GT-Sampling |
| os1 | 0.7703 ± 0.0272 | 0.765 | Finetuning + GT-Sampling |

Fusion liefert die beste ausgewogene Gesamtleistung, ist aber nicht für die
Autoerkennung notwendig. Dynamisches Auto AP30/AP60 (gepoolt): merged
0.877/0.739, os0 0.885/0.847, os1 0.831/0.783. Der alte Befund
`os1=0.09 → merged=0.90` war ein Splitartefakt. Vollständige Ergebnisse:
[`results/CROSS_VALIDATION_RESULTS.md`](results/CROSS_VALIDATION_RESULTS.md).

**Trainierte Modelle:** Die bisherige GitHub Release v1.0 enthält die alten
Temporal-Split-Modelle. Die neun Cross-Validation-Checkpoints liegen derzeit
auf dem DCAITI-Server unter `~/runs/pointpillars_crossval/`.

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
