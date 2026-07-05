# Daten-Audit & Korrekturen (Daten-Version v2)

**Datum:** 2026-07-02
**Anlass:** Verifikation, dass die Trainingsdatenbasis (Experiment-Daten → MMDetection3D/KITTI-Format) sauber ist und das Finetuning aussagekräftige Ergebnisse liefert.

---

## 1. Audit-Umfang und Methode

Geprüft wurde die komplette Datenkette:

```
manuelle Labels (*_labels_manual_correct/, Editor-Konvention)
  → tools/dataset_converters/exp.py          (Umbenennung/Sammlung, LUMPI-Klassen)
  → tools/dataset_converters/exp_to_kitti.py (Welt-Transformation, KITTI-Klassen)
  → tools/create_data.py exp --target-dataset kitti  (Full-View-PKLs)
  → tools/dataset_converters/split_exp_infos.py      (temporale 80/10/10-Splits)
  → Training/Eval (LUMPIDataset + OSDaR23Metric)
```

Prüfskript (`audit_data.py`, auf dem GPU-Server ausgeführt) verifizierte:
Split-Disjunktheit, Klassenverteilungen, Label-Aktualität (Stichprobe von 156
Roh-Labels unabhängig durch die Transformation gerechnet und mit den PKLs
verglichen), z-Konvention (Punkte-in-Box-Zählung gegen gespeicherte
`num_lidar_pts`), Box-Positionen gegen die `point_cloud_range`,
Intensitätswertebereiche sowie MD5-Gleichheit der Converter lokal/Server.

## 2. Ergebnis: Was nachweislich sauber ist

| Prüfung | Ergebnis |
|---|---|
| train/val/test disjunkt (alle 3 Views) | ✓ 0 Überlappungen, 0 Duplikate |
| Temporale Ordnung der Splits | ✓ val/test zeitlich nach train, pro Experiment |
| Alle 9 Experimente in jedem Split | ✓ |
| PKLs entsprechen finalen `_manual_correct`-Labels | ✓ 156 Stichproben, 0 Abweichungen |
| Klassen-Mapping Editor→LUMPI→KITTI | ✓ person=0, bicycle=1, car=2 |
| Boxen mit 0 Punkten / degenerierte Dimensionen | ✓ keine |
| Box-Zentren in `point_cloud_range` | ✓ 54 221/54 221 |
| Converter-Skripte lokal = Server | ✓ MD5-identisch |
| Ignorierte Frames (`invalid_frame: true`) ausgeschlossen | ✓ 9 981 Roh-Frames − 2 521 markierte = 7 460 konvertiert = 7 460 PKL-Samples (exakt) |

**Zu den ignorierten Frames:** Der Label-Editor speichert seinen
"ignored"-Zustand als `invalid_frame: true` in der Label-JSON (frame- oder
instanz-level). `exp.py::is_invalid_frame_annotation()` prüft genau diese
Flags und überspringt die gesamte Datei — auch die 1 148 markierten Frames,
die noch Boxen enthalten, wurden vollständig ausgeschlossen. Zusätzlich
löscht der Converter zuvor konvertierte Ausgaben, wenn ein Frame nachträglich
als invalid markiert wurde. Die Zählung geht ohne Rest auf; ignorierte Labels
sind nachweislich **nicht** ins Training gelangt.

## 3. Gefundene Fehler und Korrekturen (→ Daten v2)

### 3.1 z-Konvention: Gravity-Center statt Bottom-Center (Fehler, behoben)

Der manuelle Label-Editor speichert `bbox[2]` als **Box-Zentrum**
(Open3D-`OrientedBoundingBox`-Konvention). Beweis: Die gespeicherten
`num_lidar_pts` korrelieren mit der Zentrum-Interpretation zu **1.000**
(Unterkante: 0.936). MMDetection3D (`LiDARInstance3DBoxes`, Origin z=0)
erwartet jedoch die **Unterkante**.

**Fix:** `exp_to_kitti.convert_labels()` verschiebt jetzt `bbox[2] -= dz/2`
(Gravity-Center → Bottom-Center). Nach Regenerierung: `corr(num_lidar_pts,
Bottom-Interpretation) = 1.000`. ✓

### 3.2 Boden lag unterhalb der z-Range → Punkte wurden abgeschnitten (Fehler, behoben)

Gemessene Bodenebene im Experiment-Rohframe: z ≈ −0.14. Mit der alten
Translation `T_z = −3.3` lag der Boden in KITTI-Koordinaten bei ≈ −3.44 —
**unterhalb** der `point_cloud_range` z = [−3, 1]. Der `PointsRangeFilter`
entfernte dadurch im Training den Boden und die **unteren ~0.4 m aller
Objekte** aus der Eingabe.

**Fix:** `T_z = −1.6` → Boden bei z ≈ −1.74, wie in KITTI (LiDAR-Höhe
~1.73 m). Damit passen auch die aus KITTI übernommenen Anchor-z-Werte
(−0.6 Fußgänger/Radfahrer, −1.78 Auto) und der Pretrained-Checkpoint
geometrisch exakt. Nach Regenerierung: Punkte-z-Spanne −1.82..0.87
(vollständig in der Range), Box-Unterkanten −1.79..−1.30. ✓

### 3.3 Intensitätsnormierung pro Frame (Fehler, behoben)

Alt: `(i + min) · 1/max` — formal falsche Min-Max-Normierung, und der
Frame-Maximalwert schwankt zwischen ~2 900 und ~5 800 (Faktor 2), wodurch
dieselbe Oberfläche je Frame unterschiedliche Intensität bekam.

**Fix:** fester globaler Divisor `INTENSITY_SCALE = 5000` mit Clipping auf
[0, 1] (gewählt anhand gemessener Frame-Maxima 2 876..5 754, p99 ≈ 4 742).

### 3.4 IoU-Berechnung der Metrik (Präzisierung, behoben)

`mmcv.ops.diff_iou_rotated_3d` erwartet z am **Zentrum**. Nach dem
Bottom-Center-Fix (3.1) konvertiert `OSDaR23Metric.compute_mmcv_iou()` GT
und Prediction jetzt explizit mit `z += dz/2`, sodass die 3D-IoU exakt ist.

### 3.5 Metrik verwässerte seltene Klassen massiv (Fehler, behoben 2026-07-02 abends)

Die `OSDaR23Metric` berechnete AP **pro Frame** (AP11 über die
Predictions eines einzelnen Frames) und mittelte dann über **alle** Frames
des Splits — auch über Frames, in denen die Klasse gar nicht vorkommt
(diese zählten als AP = 0). Damit war die AP einer Klasse nach oben durch
`(Frames mit Klasse / alle Frames) × 0.909` begrenzt (0.909 statt 1.0 wegen
eines zweiten Fehlers: strikte `>`-Interpolation, Level 1.0 unerreichbar).

Konkret auf dem merged-Test-Split: bicycle kommt in 70 von 266 Frames vor
→ Obergrenze 70/266 × 0.909 = **0.2392** — exakt der gemeldete Wert. Eine
unabhängige Analyse der Prediction-Dumps zeigte: **alle 70/70 Test-Bikes
wurden erkannt** (score > 0.3). Die vermeintliche bicycle-Schwäche war
vollständig ein Metrik-Artefakt; auch car war mit 0.9078 bereits an der
Sättigungsgrenze (0.909) und konnte nichts mehr differenzieren.

**Fix:** Umstellung auf datensatzweites AP (Standard wie KITTI/COCO):
Matching bleibt frame-lokal (greedy, score-absteigend, IoU-Schwelle),
aber die (Score, TP)-Paare aller Frames werden gepoolt, nach Score
gerankt und die PR-Kurve gegen die Gesamt-GT-Zahl des Splits integriert
(AP11 mit korrektem `>=`). Nebeneffekt: IoU-Matrix wird nur noch einmal
pro Klasse/Frame berechnet statt einmal pro Schwelle (~4× schnellere Eval).

Hinweis: Die Best-Checkpoint-Auswahl der bisherigen Runs erfolgte noch
mit der alten Metrik; da die Verwässerung pro Klasse ein konstanter
Faktor ist, bleibt das Epochen-Ranking davon weitgehend unberührt.

### 3.6 Kaputtes `create_data.py` im Projekt-Root (behoben)

Die Kopie im Projekt-Root rief ein nicht existierendes
`exp.create_train_val_test_split(seed=42, 70/15/15)` auf. Maßgeblich ist
`mmdetection3d/tools/create_data.py` (create_test_split → update_pkl_infos;
Splits separat via `split_exp_infos.py`, 80/10/10). Die Root-Kopie wurde
durch die funktionierende Version ersetzt.

## 4. Einschränkungen für die Interpretation (kein Fehler, dokumentieren!)

1. **Klassenimbalance:** merged-Train enthält 12 676 car-, 3 256 person-,
   aber nur 562 bicycle-Instanzen (~22:1 car:bicycle). Die niedrige
   bicycle-AP ist primär hierdurch erklärt. Möglicher nächster Schritt:
   klassenbalanciertes Sampling oder Loss-Gewichtung (separates Experiment,
   um Attribution sauber zu halten).
2. **Statische Objekte über Splits hinweg:** Die geparkten Autos derselben
   Szene erscheinen identisch in train, val und test (temporaler Split,
   gleiche Szene). Die car-AP misst daher überwiegend Wiedererkennung
   bekannter statischer Objekte, keine Generalisierung auf neue Szenen.
   Person und bicycle (bewegte Objekte an neuen Positionen) sind die
   aussagekräftigeren Klassen.
3. **Skip-if-exists in allen Convertern:** `exp.py` und `exp_to_kitti.py`
   überspringen vorhandene Ausgabedateien. Bei jeder Label-Änderung müssen
   `labels/` bzw. `labels_kitti/`, `points_kitti/` auf dem Server **vorher
   geleert** werden, sonst bleiben veraltete Konvertierungen bestehen.

## 5. Regenerierung Daten v2 (Server, durchgeführt 2026-07-02)

```bash
# Backup v1 (z-Center-Konvention):  ~/data/exp_v1_zcenter/
mkdir -p ~/data/exp_v1_zcenter
mv ~/data/exp/points_kitti ~/data/exp/labels_kitti ~/data/exp/exp_kitti_infos_* ~/data/exp_v1_zcenter/
mkdir ~/data/exp/points_kitti ~/data/exp/labels_kitti

cd ~/projects/dcaiti_masterarbeit/mmdetection3d
python tools/create_data.py exp --root-path ~/data/experiments \
    --out-dir data/exp --workers 8 --target-dataset kitti --only-annotation
python tools/dataset_converters/split_exp_infos.py --root data/exp
```

Splits v2 sind deterministisch identisch zu v1 (gleiche Sample- und
Klassenzahlen pro Split); nur Geometrie (z, T_z) und Intensitäten änderten sich.

## 6. Konsequenz für alte Ergebnisse

Der Lauf `~/runs/pointpillars/merged_ft` (v1-Daten, best val mAP 0.6415)
ist **nicht mehr referenzierbar**: Er trainierte mit abgeschnittenem Boden,
schwebenden Boxen (Halbe-Höhe-Versatz in mmdet3d-Semantik) und
frame-abhängigen Intensitäten. Alle Views wurden auf v2 neu trainiert:

| Run | Work-Dir | GPU |
|---|---|---|
| merged | `~/runs/pointpillars/merged_ft_v2` | 0 |
| os1 (nach merged) | `~/runs/pointpillars/os1_ft_v2` | 0 |
| os0 | `~/runs/pointpillars/os0_ft_v2` | 1 |

Config unverändert: 50 Epochen, lr = 1e-4, Finetuning von
`checkpoints/pointpillars_kitti_3class.pth`, `save_best='osdar23/mAP'`.
Metrik-Zahlen v2 sind wegen 3.1–3.4 nicht mit v1 vergleichbar.

## 7. Ergebnisse v2 mit korrigierter Metrik (Best-Checkpoint je Lauf, Test-Split)

Alle Werte mit der **datensatzweiten AP** (Fix 3.5). Die zuvor hier
dokumentierten Zahlen (merged test mAP 0.6532 usw.) stammten aus der
verwässernden Pro-Frame-Metrik und sind nicht mehr referenzierbar.

| Run | Best-Ep. | test mAP | AP30 person | AP30 car | AP30 bicycle | AP60 bicycle |
|---|---|---|---|---|---|---|
| merged (Baseline) | 25 | **0.8880** | 0.900 | 0.909 | 0.935 | 0.856 |
| merged + GT-Sampling | 29 | **0.8916** | 0.903 | 0.909 | **0.972** | **0.883** |
| merged + Oversampling | 49 | 0.8788 | 0.901 | 0.909 | 0.890 | 0.881 |
| os0 | 38 | **0.8514** | 0.873 | 0.997 | 0.838 | 0.551 |
| os0 + GT-Sampling | 37 | 0.8412 | 0.863 | 0.997 | **0.982** | 0.496 |
| os1 | 43 | **0.8457** | 0.907 | 0.909 | 0.804 | 0.681 |
| os1 + GT-Sampling | 28 | **0.8570** | 0.902 | 0.909 | 0.863 | 0.791 |

(GT-Sampling-Runs für os0/os1 vom 2026-07-03; deren Best-Checkpoint-Wahl
lief durchgängig auf der korrigierten Metrik.)

Beobachtungen:
- **bicycle war nie schwach** — die frühere AP30 von 0.239 war das
  Metrik-Artefakt aus 3.5. Real: 0.80–0.97 je nach View.
- GT-Sampling (siehe `ABLATION_BICYCLE.md`) verbessert bicycle überall
  bei AP30 (merged +3.7, os0 +14.5, os1 +5.9 Punkte). Auf merged und os1
  steigt auch das mAP; auf os0 sinkt es leicht (0.8514→0.8412), weil die
  Lokalisierungsschärfe leidet (bicycle AP60 −5.5, mehr Predictions:
  pred/GT-Ratio 1.17). Bei dünn besetzten Einzelsensor-Punktwolken ist
  der Trade-off Detektion↔Präzision also sichtbar.
- Schwächste Punkte: person bei strengem IoU (AP60 0.62–0.77) und
  bicycle auf os0 bei AP60 — Lokalisierungspräzision kleiner
  Objekte in den Einzelsensor-Views.
- **merged > os1 > os0** bei einheitlicher GT-Sampling-Konfiguration
  (0.8916 / 0.8570 / 0.8412), konsistent mit der Punktdichte pro Objekt.

Test-Kommando (je View):
```bash
python tools/test.py configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class-exp-<view>.py \
    ~/runs/pointpillars/<view>_ft_v2/best_osdar23_mAP_epoch_*.pth \
    --work-dir ~/runs/pointpillars/<view>_ft_v2/test
```

## 8. Gültigkeitsbewertung (verifiziert 2026-07-02)

**Kein Leakage im ML-Sinn — alle Punkte einzeln geprüft:**

| Prüfung | Ergebnis |
|---|---|
| Frame-IDs train/val/test disjunkt (alle 3 Views, v2-PKLs) | ✓ 0 Überlappungen |
| Punktwolken-Dateien train ∩ test | ✓ 0 gemeinsame .bin |
| GT-Sampling-DB nur aus Train-Frames | ✓ 0/16 494 Einträge verletzen das (+ Assert im Skript) |
| Checkpoint-Wahl nur auf val; test einmalig mit fixem Checkpoint | ✓ |
| Keine auf Test-Daten gefittete Vorverarbeitung | ✓ nur feste Konstanten (R, T, Intensitäts-Skala) |
| Keine Score-Filterung in der Eval (threshold 0.0) | ✓ alle FPs gehen in die Präzision ein |
| Zielobjekte bewegen sich zwischen Splits | ✓ Test-Positionen = andere Frames/Zeitpunkte |

**Verbleibende Gültigkeits-Einschränkungen (kein Leakage, aber Geltungsbereich):**

1. **Ein-Szenen-Design:** Alle 9 Experimente teilen dieselbe Örtlichkeit;
   die statischen Autos sind in train/val/test identisch. Die Ergebnisse
   belegen die Erkennung *bekannter Objektinstanzen in bekannter Szene zu
   ungesehenen Zeitpunkten/Positionen* — keine Generalisierung auf neue
   Szenen oder neue Objektexemplare (nur 3 physische Fahrräder).
2. **Temporale Nachbarschaft an Split-Grenzen:** val beginnt unmittelbar
   nach train (~0.1 s Abstand); die ersten val-Frames sind mit den letzten
   train-Frames hoch korreliert. **test liegt hinter val** und ist damit
   die konservativere Zahl — für Kernaussagen test verwenden.
3. **Kleine bicycle-Stichprobe:** 70 Test-Instanzen → AP-Differenzen von
   wenigen Punkten entsprechen 2–3 Objekten; Ablation-Unterschiede
   entsprechend vorsichtig interpretieren.
4. **Checkpoint-Wahl der bisherigen Runs mit alter Metrik** (konstanter
   Verwässerungsfaktor pro Klasse → Epochen-Ranking weitgehend unberührt);
   künftige Runs selektieren mit der korrigierten Metrik.

## 9. Dynamisch vs. statisch: die aussagekräftige Auswertung

Die Roh-Labels tragen ein `static: true`-Flag (im Test-Split: 1 548 von
1 583 car-GT statisch, 172 von 401 person-GT; alle 70 bicycle-GT dynamisch).
Aggregierte Metriken werden von den trivial wiedererkennbaren statischen
Parkern dominiert. `tools/analysis_tools/exp_eval_dynamic_static.py`
evaluiert getrennt (statische GT als Ignore-Regionen beim Dynamik-AP und
umgekehrt; Matching identisch zur korrigierten Metrik).

**Test-Split, nur dynamische Objekte (AP30 / AP60):**

| Run | person (n=229*) | bicycle (n=70) | car (n=34–35) |
|---|---|---|---|
| merged (Baseline) | 0.892 / 0.710 | 0.935 / 0.856 | 0.697 / 0.125 |
| merged + GT-Sampling | 0.898 / 0.706 | **0.972 / 0.883** | **0.902** / 0.191 |
| merged + Oversampling | 0.894 / 0.736 | 0.890 / 0.881 | 0.619 / 0.086 |
| os0 | 0.737 / 0.408 (n=126) | 0.837 / 0.551 | 0.634 / 0.634 |
| os0 + GT-Sampling | 0.725 / 0.348 (n=126) | 0.982 / 0.496 | 0.735 / 0.679 |
| os1 | 0.905 / 0.559 (n=209) | 0.804 / 0.681 | **0.021 / 0.012** |
| os1 + GT-Sampling | 0.895 / 0.455 (n=209) | 0.863 / 0.791 | 0.091 / 0.091 |

*Statische Referenz: car AP30 ≈ 1.0 in allen Views — bestätigt die
Vermutung, dass die Aggregatwerte von den Parkern getragen werden.*

Zentrale Befunde:
1. **Das dynamische (Ziel-)Auto ist der wahre Schwachpunkt**, vom
   Aggregat (car ≈ 0.91) vollständig maskiert. In merged wird es zu 97%
   gefunden (median IoU 0.66), aber mit niedriger Konfidenz (median Score
   0.61 vs. ~0.98 bei Parkern) und mäßiger Boxgenauigkeit → AP60 0.125.
2. **os1 versagt am dynamischen Auto fast völlig** (AP30 0.02), obwohl es
   klar sichtbar ist (median 420 Punkte im GT): Die nächste car-Prediction
   liegt median 1.52 m daneben (IoU 0.21) — systematische Fehllokalisierung
   bei einseitiger Sensorsicht. os0: 0.63. **Merged-Fusion löst das
   Problem** (97% < 1 m Distanz) — ein Kernargument für den
   Multisensor-Ansatz der Arbeit. Bekräftigt durch die GT-Sampling-Runs:
   Auch mit der besten Trainingskonfiguration bleibt os1 beim dynamischen
   Auto bei 0.09 — das Versagen ist ein Geometrie-/Sichtproblem des
   Einzelsensors, kein Trainingsproblem, und nur die Fusion (0.90) behebt es.
3. Dynamische person/bicycle sind in merged solide (0.89/0.94); die
   Einzelsensoren fallen v. a. bei strengem IoU ab.
4. Vorsicht bei n=34–35 (dynamisches Auto): kleine Stichprobe; das
   Zielauto bewegt sich im Test-Segment mit median ≈ 2.2 m/s (max 3.5,
   aus den Label-Positionen aufeinanderfolgender Frames) — langsame
   Fahrt, keine schnelle Durchfahrt.

## 10. Ursachenanalyse: Warum ist AP60 beim dynamischen Auto niedrig?

Skript: `tools/analysis_tools/exp_analyze_dyncar.py` (merged-Test-Split,
35 Frames mit dynamischem Auto, Predictions aus `merged_gtsample`).

**Hypothese "Bewegungsverschmierung durch Sweep-Fusion": WIDERLEGT.**
- Die Label aller drei Views teilen dieselben Frame-Timestamps. Der
  BEV-Versatz desselben dynamischen Autos zwischen os0- und os1-Label
  im selben Frame beträgt median nur **0.09 m** (max 0.17 m) — bei
  2.2 m/s entspricht das ≤ ~50 ms Sweep-Versatz. Zu klein, um IoU 0.6
  zu verhindern. (merged-Label sind identisch mit os1-Label.)
- Die Punkte des dynamischen Autos sind entlang der Fahrzeugachse
  nicht verlängert, sondern decken nur **~60 % der GT-Boxlänge** ab
  (statische Autos: ~95 %) — das Gegenteil von Verschmierung:
  Teilsichtbarkeit.

**Tatsächliche Ursache: systematische Größenunterschätzung.**
Bestes Pred je GT (median über 35 Frames): IoU 0.614 (10/35 unter 0.6,
0/35 unter 0.3 — gefunden wird es praktisch immer). Fehlerzerlegung:
- Länge **−0.95 m**, Breite **−0.41 m** (GT 4.95×1.95×1.35; Pred
  ≈ 4.0×1.55 — fast exakt der KITTI-car-Prior 3.9×1.6)
- Zentrum quer nur 0.09 m daneben, längs +0.35 m (zur sichtbaren
  Seite hin), z 0.02 m, Gierwinkel 3.6° — alles unkritisch.

Geometrische Konsequenz: Selbst bei perfekter Zentrierung ergibt
4.0×1.55 in 4.95×1.95 nur IoU ≈ 0.62 — die AP60-Schwäche ist fast
vollständig der Boxgröße geschuldet. Das Netz regressiert bei nur
~60 % sichtbarer Fahrzeuglänge auf sein KITTI-Größen-Prior.

**Score-Verteilung (Betriebspunkt-Relevanz):** Bester car-Pred-Score
am dynamischen Auto median 0.74 (2/35 Frames < 0.45, 10/35 < 0.6,
0 Misses); statische Autos median 0.98 (0.3 % < 0.45). Die Scores der
bicycle-Geisterboxen (§11: 0.3–0.6) **überlappen** mit denen des
dynamischen Autos — eine globale Score-Schwelle, die alle Geisterboxen
entfernt und das Zielauto in jedem Frame behält, existiert nicht.
Praxis-Konsequenz: Schwellen pro Klasse wählen (bicycle streng, car
locker; im Prediction-Viewer per Tasten 1/2/3 + `+`/`−`) bzw. im
Realeinsatz zeitliche Verfolgung (Tracking) über die schwachen
Einzelframes hinweg.
**Offener Prüfpunkt:** die realen Maße des Zielfahrzeugs nachmessen —
GT 4.95×1.95 m ist groß; wäre die GT-Box großzügig gelabelt, wäre ein
Teil der "Schwäche" ein Label-Artefakt.

**Qualitative Belege:** `figures/qualitative_bev.png`
(Skript `tools/analysis_tools/exp_viz_bev.py`): os1 ohne gültige
Detektion am dynamischen Auto vs. merged mit Treffer im selben Frame,
dazu Fahrrad- und Person-Beispiele.

## 11. Analyse der bicycle-False-Positives ("Geisterboxen" im Viewer)

Auf dem merged-Test-Split gibt es 42 bicycle-Predictions (score ≥ 0.3)
ohne GT-Rad im Umkreis von 2 m. Untersuchung
(`figures/fp_bike_spots.png`):

- **Kein GT-Sampling-Artefakt:** Die Baseline (ohne GT-Sampling) hat
  praktisch gleich viele (40) an denselben Orten.
- Die FPs clustern an **wenigen festen Positionen**, die über mehrere
  Experimente wiederkehren und median 0.5–0.9 m neben Positionen liegen,
  an denen im Train-Zeitraum Räder standen/fuhren.
- **Drei Kategorien** (Punktinhalt der Boxen):
  1. **Echte statische Objekte** (29/42 mit >100 Punkten, median 880):
     vertikale Strukturen ~1–2 m hoch (Pfosten/Busch/evtl. tatsächlich
     abgestellte Räder — vor Ort prüfen!), die als bicycle
     fehlklassifiziert werden bzw. schlicht nie gelabelt wurden.
     Falls es echte Räder sind, ist die gemessene bicycle-Precision
     unterschätzt (FPs wären eigentlich TPs ohne Label).
  2. **Leere Boxen über Boden** (die "Geisterboxen" im Viewer): Im
     Seitenschnitt liegt unter mehreren dieser Boxen ein Punktcluster
     **0.3–0.5 m unter dem Boden** (z ≈ −2.0 bei Boden −1.74) —
     Spiegelreflexionen (nasse Fläche/Glas). Die pc_range reicht bis
     z = −3, PointPillars kollabiert die z-Achse pro Pillar → solche
     Unterboden-Cluster können Detektionen an Anchor-Höhe auslösen.
     Möglicher Fix (Future Work): Unterboden-Punkte (z < −2) beim
     Konvertieren filtern oder pc_range-z-Minimum anheben; müsste als
     Ablation geprüft werden.
  3. **Positions-Halluzinationen** (Frame-Forensik exp 1,
     ts 1760002176837492193): bicycle-Pred (Score 0.31) über blankem
     Boden — 286 Punkte in der Box, aber alle in einer 2 cm flachen
     Schicht (reiner Boden), Position nur **0.23 m** neben dem Weg der
     Trainingsräder aus Experimenten 4–6 (gleiche Szene). Das Netz hat
     Ort und Objekt nicht getrennt gelernt → direkte Folge der
     Ein-Szenen-Datenbasis (zentrale Limitation; mit mehr Szenen nicht
     zu erwarten). Im selben Frame: bicycle (0.53) + person (0.38)
     parallel zum car (0.65) auf dem dünn abgetasteten **Heck des
     dynamischen Autos** — klassenübergreifende Mehrfach-Hypothesen,
     die das klassenweise NMS nicht unterdrückt (Abhilfe:
     cross-class NMS oder Regel "Box vollständig in höher gescorter
     Box anderer Klasse → verwerfen").
- Scores der FPs liegen bei 0.3–0.85 (meist < 0.6) — im Viewer mit
  `+` auf 0.6 filterbar; in der AP ranken sie unter den meisten TPs.
- **Gegenmaßnahmen-Einordnung:** Nachlabeln hilft nur bei Kategorie 1
  (falls echte Räder); gegen Kategorie 2 Unterboden-Filter (z < −2);
  gegen Kategorie 3 helfen nur mehr Szenen-Diversität (hier nicht
  verfügbar), eine Statik-/Hintergrundkarte des fest installierten
  Sensors oder klassenweise Betriebspunkte (bicycle ≥ ~0.55).
- **Ablation cross-class NMS (Nachverarbeitung, 2026-07-03):** Regel
  "Prediction verwerfen, wenn ihr BEV-Footprint zu ≥ 80 % in einer
  höher gescorten Prediction anderer Klasse liegt"
  (`tools/analysis_tools/exp_crossclass_nms.py`, auf dem
  merged-gtsample-Dump): entfernt 11/2353 Predictions (u. a. die
  In-Auto-Duplikate), **alle AP-Werte unverändert** — die Duplikate
  ranken zu niedrig, um die Metrik zu berühren. Fazit: kosmetisch
  sinnvoll (saubere Ausgabe), metrisch neutral, risikofrei.
- **Ablation Unterboden-Filter (2026-07-03, Ergebnis: lohnt nicht):**
  `points_kitti_zfilt` (Punkte z < −2.0 entfernt = 0.09 % aller
  Punkte), Config `...exp-merged-gtsample-zfilt.py`, Run
  `merged_ft_v2_gtsample_zfilt` (Best Ep. 26, val 0.8655). Test:
  **mAP 0.8881** (gtsample: 0.8916). Die bicycle-Geisterboxen wurden
  **nicht** weniger (58 vs. 42 — Trainings-Varianz, aber sicher keine
  Reduktion); das dynamische Auto fiel auf AP30 0.63 (gtsample 0.90).
  Einziger Gewinn: bicycle-Lokalisierung (AP60 0.94, konstant über
  alle IoU-Schwellen). **Fazit:** Die Spiegelreflexionen sind nicht
  der treibende Mechanismus hinter den Geisterboxen (konsistent mit
  der Frame-Forensik: Positions-Halluzination + echte Objekte
  dominieren); der Filter wird nicht übernommen, Standard bleibt
  Daten v2 ungefiltert mit GT-Sampling.

## Anhang: Wie die Metrik zu lesen ist — IoU-Schwelle vs. Score

Zwei Schwellen, die man nicht verwechseln darf:

**1. IoU-Schwelle (die Zahl in AP30/40/50/60).** IoU = Intersection
over Union: Überlappungsvolumen der Prediction- und GT-Box geteilt
durch das Volumen ihrer Vereinigung (1.0 = deckungsgleich, 0 = keine
Berührung; bei uns 3D-Volumen rotierter Quader). Die Zahl im AP-Namen
ist die Mindest-IoU, ab der eine Prediction als Treffer zählt:
- **AP30** ≈ "Objekt ungefähr gefunden?" (Detektionsfähigkeit)
- **AP60** ≈ "Sitzt die Box auch präzise?" (Lokalisierungsschärfe)
Diagnostisch: AP30 hoch + AP60 niedrig = Objekt wird gefunden, aber
Box passt nicht (Beispiel dynamisches Auto, §10: Pred zu klein →
IoU-Deckel ~0.62). AP über alle Schwellen konstant = reines
Recall-/Ranking-Problem, kein Lokalisierungsproblem.

**2. Score (Konfidenz, 0–1).** Ausgabe des Klassifikationskopfes pro
Box: wie sicher das Netz ist, dass dort ein Objekt dieser Klasse ist.
Der Prediction-Viewer filtert damit nur die *Anzeige* (`+`/`−`).
**In die AP geht keine feste Score-Schwelle ein** — der Score dient
als Ranking:
1. Alle Predictions des Testsplits nach Score absteigend sortieren.
2. Liste von oben abarbeiten; je Prediction: Treffer (IoU ≥ Schwelle
   mit noch nicht vergebenem GT) oder False Positive.
3. An jeder Listenposition ergibt sich Precision (Anteil Richtige am
   bisher Gemeldeten) und Recall (Anteil gefundener GT-Objekte).
4. AP = über 11 Recall-Stufen gemittelte Precision (Fläche unter der
   Precision-Recall-Kurve, AP11-Interpolation).

Die AP bewertet damit implizit *alle* Betriebspunkte gleichzeitig und
misst auch die Score-Qualität: Fehldetektionen mit niedrigem Score
(z. B. die Geisterboxen aus §11, Score 0.3–0.6, echte Treffer ~0.9+)
stehen hinten im Ranking und kosten kaum AP; dieselben Fehler mit
Score 0.9 würden die AP deutlich drücken. Für einen realen Einsatz
wählt man dagegen einen festen Betriebspunkt (Score-Schwelle) als
Kompromiss aus Falschalarmrate und Empfindlichkeit.

**mAP** in unseren Tabellen = Mittel über die vier IoU-Schwellen
(0.3/0.4/0.5/0.6) und die drei Klassen. Matching frame-lokal, Pooling
über den ganzen Testsplit (datensatzweite AP, §3.5).
