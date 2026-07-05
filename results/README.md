# Ergebnisse

Alle Endergebnisse des Projekts an einem Ort.

## Dokumente

| Datei | Inhalt |
|---|---|
| [`RESULTS.md`](RESULTS.md) | **Haupt-Ergebnisreport**: per-Klasse AP30/40/50/60 aller 8 Läufe, View-Vergleich (merged/os0/os1), dynamisch/statisch-Analyse, Laufzeiten, Vergleich mit der Vorgängerarbeit, qualitative Analyse |
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
