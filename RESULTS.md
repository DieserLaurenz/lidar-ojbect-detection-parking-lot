# Evaluationsreport: PointPillars auf den Kreuzungs-Experimenten

**Stand:** 2026-07-03. Alle Zahlen: Test-Split (temporal, letzte ~10 %
jedes Experiments), korrigierte datensatzweite Metrik (Herleitung und
Lesehilfe: `DATA_AUDIT.md`, Anhang). Modelle: PointPillars, finetuned
vom KITTI-3-Klassen-Checkpoint, 50 Epochen, Best-Checkpoint nach
Val-mAP. "GT-Sampling" = Copy-Paste-Augmentierung seltener Klassen aus
dem jeweiligen Train-Split (`ABLATION_BICYCLE.md`).

## 0. Zusammenfassung und Einordnung (einfache Sprache)

**Worum ging es?** Zwei fest montierte LiDAR-Sensoren beobachten eine
Kreuzung aus verschiedenen Richtungen. Ein neuronales Netz
(PointPillars) soll darin Personen, Fahrräder und Autos finden. Die
Frage: Bringt es etwas, die Daten beider Sensoren zu fusionieren,
statt nur einen zu benutzen?

**Die fünf wichtigsten Erkenntnisse:**

1. **Fusion ist beim bewegten Objekt nicht Luxus, sondern notwendig.**
   Sensor os1 allein verfehlt das fahrende Auto fast komplett
   (Trefferquote-Wert 0.02–0.09 von 1.0): Seine Boxen liegen
   systematisch ~1.5 m daneben, weil er das Auto nur von einer Seite
   sieht. Wichtig: Auch das beste Training ändert daran nichts — es
   ist ein Sichtproblem, kein Trainingsproblem. Die Fusion beider
   Sensoren löst es (0.90). Sie kostet doppelte Rechenzeit, bleibt
   aber schnell genug für Live-Betrieb.

2. **Die beste Trainingsverbesserung ist das "GT-Sampling"** (beim
   Training zusätzliche Fahrrad-/Personen-Beispiele in die Szenen
   hineinkopieren, weil Fahrräder 22× seltener sind als Autos). Es
   verbessert Fahrrad- und Auto-Erkennung deutlich. Die Alternative
   (Fahrrad-Aufnahmen einfach doppelt zeigen) bringt nichts —
   auch das wurde getestet und gehört als Negativergebnis in die Arbeit.

3. **Die Gesamtzahlen schmeicheln — die ehrliche Zahl ist die für
   bewegte Objekte.** Der Testbereich steht voller geparkter Autos
   (1548 von 1583), die das Netz aus dem Training kennt und trivial
   wiedererkennt. Deshalb werten wir bewegte und stehende Objekte
   getrennt aus und berichten die bewegten als Hauptergebnis.

4. **Die verbleibenden Schwächen sind verstanden:** (a) Beim fahrenden
   Auto stimmt die Boxgröße nicht ganz — das Netz schätzt das Auto ~1 m
   zu kurz, weil es nur teilweise sichtbar ist und das Netz auf typische
   (kleinere) Auto-Maße aus dem Vortraining zurückfällt. Position und
   Ausrichtung stimmen dagegen fast perfekt. (b) Das Netz ist sich beim
   bewegten Objekt weniger sicher (Konfidenz ~0.74 statt ~0.98) — für
   den Praxiseinsatz braucht man deshalb klassenweise Schwellwerte oder
   Tracking. (c) Vereinzelte "Geister-Fahrräder" an Stellen, wo im
   Training oft Räder fuhren: Das Netz hat sich Orte gemerkt statt nur
   Formen — eine Folge davon, dass alle Daten aus einer einzigen Szene
   stammen.

5. **Die Zahlen sind belastbar, weil vorher aufgeräumt wurde:** Drei
   Datenfehler (abgeschnittener Boden, verschobene Boxhöhen,
   inkonsistente Intensitäten) und ein Fehler in der Messmethode wurden
   gefunden und behoben — vorher schien z. B. die Fahrrad-Erkennung
   katastrophal (0.24), real lag sie bei 0.94. Trainings-/Test-Daten
   sind sauber getrennt; die bekannten Einschränkungen (wenige
   Zielobjekte, eine Szene, langsames Zielauto) sind dokumentiert.

**Ein Satz:** Ein einzelner Infrastruktur-LiDAR erkennt die statische
Szene fast perfekt, scheitert aber je nach Blickwinkel systematisch am
bewegten Zielobjekt — die Fusion zweier Sensoren behebt genau das bei
weiterhin echtzeitfähiger Verarbeitung.

## 1. Per-Klasse-AP (Gesamttest)

### merged (Fusion beider Sensoren)

| Modell | mAP | AP30 p/b/c | AP60 p/b/c |
|---|---|---|---|
| Baseline | 0.8880 | 0.900 / 0.935 / 0.909 | 0.772 / 0.856 / 0.816 |
| **GT-Sampling** | **0.8916** | 0.903 / **0.972** / 0.909 | 0.758 / 0.883 / 0.814 |
| Frame-Oversampling | 0.8788 | 0.901 / 0.890 / 0.909 | 0.856 / 0.881 / 0.812 |

### Einzelsensoren (GT-Sampling-Konfiguration, einheitlich)

| View | mAP | AP30 p/b/c | AP60 p/b/c |
|---|---|---|---|
| os0 | 0.8412 | 0.863 / 0.982 / 0.997 | 0.654 / 0.496 / 0.908 |
| os1 | 0.8570 | 0.902 / 0.863 / 0.909 | 0.623 / 0.791 / 0.909 |

(p = person, b = bicycle, c = car. os0-car 0.997: kleinerer Testbereich
mit fast nur Parkern — nicht direkt mit merged vergleichbar.)

**View-Ranking bei einheitlicher Konfiguration: merged (0.8916) >
os1 (0.8570) > os0 (0.8412).**

### Nur dynamische Objekte (die eigentliche Zielaufgabe)

| Modell | person AP30/60 | bicycle AP30/60 | car AP30/60 |
|---|---|---|---|
| merged GT-Sampling | 0.898 / 0.706 (n=229) | **0.972 / 0.883** (n=70) | **0.902** / 0.191 (n=35) |
| os0 GT-Sampling | 0.725 / 0.348 (n=126) | 0.982 / 0.496 (n=69) | 0.735 / 0.679 (n=35) |
| os1 GT-Sampling | 0.895 / 0.455 (n=209) | 0.863 / 0.791 (n=49) | **0.091** / 0.091 (n=34) |

Kernbefund: os1 allein verfehlt das bewegte Auto systematisch
(Fehllokalisierung ~1.5 m bei einseitiger Sicht, auch mit bester
Trainingskonfiguration); die Fusion behebt es (0.90). Details und
Ursachenanalysen: `DATA_AUDIT.md` §9–§10.

## 2. Laufzeiten (Evaluations-Setup)

> **Zum Verständnis (einfache Sprache):**
>
> **Inferenzzeit** = wie schnell das Netz "denkt". Inferenz heißt, das
> fertig trainierte Netz auf neue Daten anzuwenden: Punktwolke rein →
> Netz rechnet → Bounding Boxes raus. Die Inferenzzeit ist die Dauer
> dieses einen Durchlaufs. Der LiDAR liefert 10 Aufnahmen pro Sekunde
> (alle 100 ms eine) — bleibt das Netz darunter, kann das System
> **live** mitlaufen. Unser Netz schafft das selbst mit der doppelt so
> großen fusionierten Punktwolke (85 ms).
>
> **Daten-Merge** = zwei Sensor-Aufnahmen zu einem Bild zusammenkleben.
> Beide LiDARs sehen dieselbe Kreuzung aus verschiedenen Richtungen,
> aber jeder in seinem eigenen Koordinatensystem (jeder hält sich
> selbst für den Nullpunkt). Der Merge macht pro Aufnahme: beide
> Punktwolken laden → Messrauschen entfernen → eine Wolke ins
> Koordinatensystem der anderen drehen/verschieben (vorab kalibriert)
> → Feinausrichtung per ICP (schiebt die Wolken millimetergenau
> aufeinander) → Punkte aneinanderhängen. Die **Merge-Zeit** ist die
> Dauer dieses Vorgangs. Unsere ~2.3 s pro Aufnahme klingen langsam,
> aber fast alles davon sind Rauschfilter und die ICP-Feinausrichtung —
> und Letztere müsste man nur **einmal** machen, weil die Sensoren fest
> montiert sind (wir haben sie sicherheitshalber pro Aufnahme
> wiederholt). Live bliebe nur drehen + zusammenkleben übrig: wenige
> Millisekunden.

**Inferenz** (Tesla V100-SXM3-32GB, Batch 1, `tools/analysis_tools/
benchmark.py`, GT-Sampling-Modelle, inkl. Datenladen/Voxelisierung):

| View | Durchsatz | ≈ Zeit/Frame | Punkte/Frame (≈ median) |
|---|---|---|---|
| merged | 11.7 fps | 85 ms | ~210 000 |
| os0 | 22.3 fps | 45 ms | ~105 000 |
| os1 | 23.5 fps | 43 ms | ~105 000 |

Die Fusion kostet also grob den Faktor 2 in der Inferenz (doppelte
Punktzahl → doppelte Voxelisierungs-/Pillar-Arbeit); alle Varianten
liegen über der Sensorrate von 10 Hz.

**Daten-Merge os0+os1 → merged** (`experiment/pcd_merge.py`, gemessen
während der Vorverarbeitung, lokale CPU, 8 parallele Worker; n=3327,
aus `merge_times.csv` je Experiment):

- median **2.25 s/Frame**, mean 2.20 s (min 0.71, max 4.00)
- Enthalten: PCD-Laden, statistischer Ausreißerfilter, feste
  Vorab-Transformation (Blender-Kalibrierung), Voxel-Downsampling,
  Normalenschätzung und **per-Frame-ICP-Registrierung** — Letztere
  dominieren die Zeit.
- Einordnung: Das ist der Offline-Vorverarbeitungspfad. Für einen
  Online-Betrieb würde man die Extrinsik einmalig kalibrieren (ICP
  einmal, nicht pro Frame); der verbleibende Merge (Transformation +
  Konkatenation) läge im Millisekundenbereich. Die 2.25 s sind also
  keine inhärente Latenz des Fusionsansatzes.

## 3. Qualitative Analyse: typische Erfolge und Fehlermodi

Abbildungen in `figures_analysis/`, interaktiv nachvollziehbar mit
`experiment/prediction_viewer.py` (auch `--full` für ganze
Experimente; klassenweise Score-Schwellen mit Tasten 1/2/3 + `+`/`−`).

**Erfolge** (`qualitative_bev.png`):
- Statische Autos werden praktisch perfekt erkannt (Score ~0.98,
  AP30 ≈ 1.0) — allerdings misst das Wiedererkennung derselben Parker
  aus dem Training, keine Generalisierung.
- Dynamische Personen und Fahrräder in merged solide (AP30 0.90/0.97),
  inkl. Radfahrer-Erkennung.
- Der Paradefall der Fusion: os1 ohne gültige Detektion am bewegten
  Auto, merged trifft im selben Frame (Panel a/b).

**Fehlermodus 1 — Größenunterschätzung am bewegten Auto**
(`DATA_AUDIT.md` §10): Prediction median 0.95 m zu kurz / 0.41 m zu
schmal (regressiert Richtung KITTI-Prior), Zentrum/z/Gierwinkel nahezu
fehlerfrei. Deckelt die IoU bei ~0.62 → niedrige AP60 (0.19) trotz
~100 % Recall. Punktabdeckung nur ~60 % der Boxlänge (Teilsichtbarkeit).
Offen: reale Fahrzeugmaße gegen die GT-Box (4.95×1.95 m) prüfen.

**Fehlermodus 2 — niedrige Konfidenz am dynamischen Objekt**
(`score_distributions_merged.png`): bester Score am bewegten Auto
median 0.74 (10/35 Frames < 0.6) vs. 0.98 bei Parkern. Ein globaler
Betriebspunkt, der alle Fehldetektionen entfernt und das Zielauto in
jedem Frame behält, existiert nicht → klassenweise Schwellen bzw.
Tracking nötig.

**Fehlermodus 3 — "Geisterräder"** (`fp_bike_spots.png`,
`DATA_AUDIT.md` §11): 42 bicycle-FPs (Score ≥ 0.3), drei Ursachen:
(a) echte ungelabelte statische Objekte, (b) Spiegelreflexions-Cluster
unter dem Boden (Ablation mit Unterboden-Filter läuft), (c) reine
**Positions-Halluzination** — Boxen über blankem Boden exakt auf dem
Weg der Trainingsräder (0.23 m Abstand), belegt per Frame-Forensik.
(c) ist eine direkte Folge der Ein-Szenen-Datenbasis (zentrale
Limitation). Kein GT-Sampling-Artefakt (Baseline zeigt dieselben FPs).
Cross-class-NMS-Nachverarbeitung entfernt Duplikat-Hypothesen
(Rad/Person "im" Auto), ändert aber keinen AP-Wert.

**Precision-Recall-Kurven** (`pr_curves_merged.png`): GT-Sampling vs.
Baseline je Klasse bei IoU 0.3/0.6 — GT-Sampling verschiebt die
bicycle-Kurve sichtbar nach rechts oben, person/car unverändert.

## Referenzen

- `DATA_AUDIT.md` — Pipeline-Audit, Fixes (Daten v2), Metrik-Korrektur,
  alle Detailanalysen, Metrik-Lesehilfe
- `ABLATION_BICYCLE.md` — Klassenimbalance-Ablation (GT-Sampling vs.
  Oversampling) inkl. os0/os1-Übertragung
- Runs auf dem GPU-Server: `~/runs/pointpillars/{merged,os0,os1}_ft_v2*`
