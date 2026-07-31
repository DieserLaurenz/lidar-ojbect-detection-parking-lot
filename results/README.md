# Ergebnisse

Alle Endergebnisse des Projekts an einem Ort.

## Dokumente

| Datei | Inhalt |
|---|---|
| [`RESULTS.md`](RESULTS.md) | Historischer Temporal-Split-Report: per-Klasse AP30/40/50/60, Ablationen, Laufzeiten und qualitative Analyse; als Hauptbewertung durch Cross-Validation abgelöst |
| [`CROSS_VALIDATION_RESULTS.md`](CROSS_VALIDATION_RESULTS.md) | **Aktueller Haupt-Ergebnisreport**: gepaarte experiment-held-out 3-Fold-Cross-Validation, Fold-Streuung, gepoolte OOF-Metriken, dynamisch/statisch und OOF-False-Positives |
| [`CROSS_VALIDATION.md`](CROSS_VALIDATION.md) | Vorab festgelegtes Cross-Validation-Protokoll, Trainingsdetails und wissenschaftliche Einordnung |
| [`CROSS_VALIDATION_STATIC_AWARE.md`](CROSS_VALIDATION_STATIC_AWARE.md) | Benannte Auswertungsvariante: Bewegt-/Statisch-Wertung nutzt die manuellen Labels (Ignore-Floor 0.25) — Grundlage der Bewegt-Folien der Abschlusspräsentation |
| [`CENTERPOINT_CV_STATIC_AWARE.md`](CENTERPOINT_CV_STATIC_AWARE.md) | Dieselbe label-basierte Auswertungsvariante für den CenterPoint-Vergleichslauf |
| [`DATA_AUDIT.md`](DATA_AUDIT.md) | Technisches Audit der Datenpipeline: Geometrie-Fixes (Daten v2), Metrik-Korrektur (datensatzweites AP11), Ursachenanalysen (dyn. car, Geisterräder) |
| [`ABLATION_BICYCLE.md`](ABLATION_BICYCLE.md) | Klassenimbalance-Ablationen: GT-Sampling (übernommen) vs. Oversampling und Unterboden-Filter (verworfen) |

## Abbildungen (`figures/`)

| Datei | Zeigt |
|---|---|
| `qualitative_bev.png` | Qualitative BEV-Beispielszenen (Erfolge und typische Fehler) |
| `pr_curves_merged.png` | Precision-Recall-Kurven, merged View |
| `score_distributions_merged.png` | Score-Verteilungen TP vs. FP |
| `fp_bike_spots.png` | Räumliche Verteilung der bicycle-False-Positives |

## Predictions (`predictions/`)

Exportierte Test-Split-Detektionen als JSON, pro View (`merged`, `os0`, `os1`):
`_baseline` = Finetuning ohne GT-Sampling, ohne Suffix = beste Variante,
`_full` = alle Frames (für den Viewer), `_zfilt` = Unterboden-Filter-Ablation.
Interaktiv ansehen: `python experiment/prediction_viewer.py --view merged` (vom Projekt-Root).

## Trainierte Modelle

Die Checkpoints (`.pth`) aller 8 Läufe samt Configs und Test-Logs liegen im
[GitHub Release v1.0](https://github.com/DieserLaurenz/lidar-ojbect-detection-parking-lot/releases/tag/v1.0).

## Kurzübersicht Test-mAP (AP30–60, IoU-gemittelt)

| Lauf | merged | os0 | os1 |
|---|---|---|---|
| Finetuning (Baseline) | 0.8880 | 0.8514 | 0.8457 |
| Finetuning + GT-Sampling | **0.8916** | 0.8412 | 0.8570 |
| Ablation Oversampling¹ | 0.8788 | — | — |
| Ablation GT-Sampling + z-Filter¹ | 0.8881 | — | — |

¹ nicht übernommen, Begründung in `ABLATION_BICYCLE.md` bzw. `DATA_AUDIT.md` §11.
