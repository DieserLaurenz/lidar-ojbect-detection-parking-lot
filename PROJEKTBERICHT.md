# Projektbericht: Multisensor-LiDAR-Objekterkennung an einer Kreuzung

**Masterarbeit — Stand 2026-07-03.** Dieser Bericht beschreibt von
Anfang bis Ende, was gemacht wurde, welche Probleme gefunden und
behoben wurden, welche Ergebnisse herauskamen und was daraus folgt.
Fachbegriffe werden bei der ersten Verwendung *kursiv* eingeführt und
kurz erklärt. Detailbelege stehen in `DATA_AUDIT.md` (Audit + alle
Analysen), `ABLATION_BICYCLE.md` (Vergleichsexperimente) und
`RESULTS.md` (Ergebnisreport mit Tabellen und Laufzeiten).

---

## 1. Ausgangslage und Ziel

Zwei fest montierte **LiDAR-Sensoren** (Ouster; *LiDAR* = Laserscanner,
der die Umgebung mit Lichtpulsen abtastet und daraus eine *Punktwolke*
erzeugt — Millionen von 3D-Messpunkten pro Sekunde) beobachten eine
Kreuzung aus verschiedenen Richtungen. Ein neuronales Netz soll in
diesen Punktwolken **Personen, Fahrräder und Autos** erkennen und mit
*3D-Bounding-Boxes* markieren (Quader, die Position, Größe und
Ausrichtung eines Objekts beschreiben).

Die Leitfrage der Arbeit: **Bringt die Fusion beider Sensoren einen
messbaren Vorteil gegenüber einem Einzelsensor?** Dafür werden drei
"Ansichten" (*Views*) verglichen:

- **os0** — nur Sensor 0
- **os1** — nur Sensor 1
- **merged** — beide Punktwolken zu einer fusioniert

## 2. Datenerhebung

Am 09.10.2025 wurden **9 Experimente** aufgenommen: je dreimal fuhr/
ging ein bewegtes *Zielobjekt* durch die Szene — ein **Auto**
(Experimente 1–3), ein **Fahrrad** (4–6), eine **Person** (7–9).
Zusätzlich stehen in der Szene dauerhaft geparkte Autos und weitere
statische Objekte. Insgesamt entstanden 3327 Frames (Aufnahmen) pro
View bei 10 Hz Aufnahmerate (10 Aufnahmen pro Sekunde).

## 3. Datenaufbereitung: Merge der beiden Sensoren

Jeder Sensor liefert seine Punktwolke im eigenen Koordinatensystem
(jeder hält sich selbst für den Nullpunkt). Der **Merge** legt beide
passgenau übereinander (`experiment/pcd_merge.py`):

1. Beide Punktwolken laden, Messrauschen entfernen (*statistischer
   Ausreißerfilter*: Punkte, die untypisch weit von ihren Nachbarn
   entfernt liegen, werden verworfen)
2. Grobe Ausrichtung mit einer vorab manuell bestimmten
   Transformation (*Extrinsik* = Position/Drehung der Sensoren
   zueinander, hier in Blender kalibriert)
3. Feinausrichtung per **ICP** (*Iterative Closest Point* — ein
   Algorithmus, der zwei Punktwolken schrittweise millimetergenau
   aufeinanderschiebt)
4. Punkte aneinanderhängen → merged-Punktwolke

Gemessene Dauer: **median 2.25 s pro Frame** (Offline-Verarbeitung;
im Live-Betrieb würde man ICP nur einmal kalibrieren — dann kostet der
Merge nur Millisekunden, siehe §11).

## 4. Labeling (Ground Truth)

Für das Training und die Bewertung braucht man die "Wahrheit":
**Ground Truth (GT)** = von Hand gezeichnete Boxen um jedes echte
Objekt. Dafür wurde ein eigener Label-Editor gebaut
(`experiment/manual_bbox_editor.py`, Open3D-basiert). Konventionen:

- Klassen: person, bicycle, car
- Jede Box trägt ein **static-Flag** (steht das Objekt, z. B.
  geparktes Auto, oder bewegt es sich — wichtig für §9)
- Unbrauchbare Frames wurden als *ignored/invalid* markiert (2521
  Stück) und nachweislich von Training und Bewertung ausgeschlossen
- Finale Labelstände liegen in `*_labels_manual_correct/`

## 5. Konvertierung ins Trainingsformat und Datensplits

Das Trainingsframework **MMDetection3D** (Open-Source-Baukasten für
3D-Objekterkennung) erwartet Daten im **KITTI-Format** (KITTI = der
bekannteste Datensatz/Standard für 3D-Erkennung im Straßenverkehr).
Die Konverter (`exp.py`, `exp_to_kitti.py`) drehen und verschieben die
Punktwolken in ein KITTI-ähnliches Koordinatensystem und schreiben
Binärdateien + Index-Dateien (*PKL-Infos*).

**Datensplits:** Die Frames jedes Experiments wurden **zeitlich** in
80 % Training / 10 % Validierung / 10 % Test geteilt (*Split* =
Aufteilung; *zeitlich* heißt: die letzten 10 % der Zeitachse sind
Test — so "sieht" das Training nie in die Zukunft des Tests).
Geprüft: kein Frame liegt in zwei Splits (*Leakage-Check*; *Leakage* =
verbotenes Durchsickern von Testinformation ins Training, das
Ergebnisse künstlich schönt).

## 6. Das Daten-Audit: drei gefundene und behobene Fehler

Auf die Frage "Ist die Datenbasis wirklich sauber?" wurde die gesamte
Pipeline systematisch geprüft (`DATA_AUDIT.md`). Drei echte Fehler:

**Fehler 1 — Der Boden wurde abgeschnitten.** Das Netz verarbeitet nur
Punkte innerhalb eines festen Quaders (*point cloud range*, hier
z ≥ −3 m). Die Konvertierung schob den Boden aber auf z = −3.44 m —
ein Filter warf damit den Boden **und die unteren ~40 cm jedes
Objekts** weg (Menschen ohne Füße, Autos ohne Räder). Fix: Verschiebung
korrigiert, Boden liegt jetzt bei −1.74 m (wie in KITTI üblich).

**Fehler 2 — Alle Boxen schwebten.** Der Label-Editor speichert die
Boxhöhe als **Mittelpunkt** (*gravity center*), das Framework erwartet
die **Unterkante** (*bottom center*). Alle Trainingsboxen saßen eine
halbe Boxhöhe zu hoch (beim Auto ~75 cm). Fix im Konverter
(`z −= Höhe/2`), Korrektheit statistisch verifiziert.

**Fehler 3 — Intensitäten nicht vergleichbar.** Jeder Punkt hat eine
*Intensität* (Stärke der Laserreflexion). Sie wurde pro Frame am
hellsten Punkt normiert — dieselbe Fläche bekam in jedem Frame andere
Werte. Fix: fester Divisor (5000) mit Clipping.

Alle Ergebnisse **vor** diesen Fixes ("v1") sind obsolet. Der Datensatz
nach den Fixes heißt **v2**.

## 7. Training

**Modell: PointPillars** — ein schnelles Standard-Netz für
3D-Erkennung. Es teilt die Szene von oben in ein Raster aus Säulen
(*Pillars*), fasst die Punkte jeder Säule zu einem Merkmalsvektor
zusammen und lässt darauf ein 2D-Faltungsnetz Boxen vorhersagen. Boxen
entstehen aus *Anchors* (vordefinierte Referenzboxen typischer
Objektgröße), die das Netz verschiebt und skaliert.

**Finetuning statt Training von Null:** Startpunkt ist ein auf KITTI
vortrainierter *Checkpoint* (= gespeicherter Netzzustand). Das Netz
kennt also schon "Auto/Fußgänger/Radfahrer" und lernt 50 *Epochen*
(eine Epoche = einmal alle Trainingsdaten sehen) lang unsere Szene.
Wichtige Konvention: Klassenreihenfolge person=0, bicycle=1, car=2
(muss zum KITTI-Checkpoint passen).

**Checkpoint-Auswahl:** Nach jeder Epoche wird auf dem
Validierungs-Split die Erkennungsqualität (mAP, §8) gemessen; der
beste Stand (`save_best`) wird gespeichert. Der Test-Split wird genau
**einmal** am Ende mit diesem besten Checkpoint angefasst — so bleibt
die Testzahl unverfälscht.

## 8. Die Metrik — und der gefundene Metrik-Fehler

**Wie wird "gut" gemessen?** Eine Prediction zählt als Treffer, wenn
ihre **IoU** (*Intersection over Union* = Überlappungsvolumen geteilt
durch Vereinigungsvolumen von Prediction- und GT-Box; 1.0 = perfekt)
über einer Schwelle liegt. **AP30/40/50/60** = *Average Precision* bei
IoU-Schwelle 0.3/0.4/0.5/0.6; **mAP** = Mittel über die vier Schwellen
und drei Klassen. Diagnose-Lesart: AP30 misst "Objekt gefunden?",
AP60 misst "Box sitzt präzise?". Jede Prediction trägt zudem einen
**Score** (Konfidenz 0–1); die AP bewertet das gesamte Score-Ranking,
ohne feste Schwelle (Details: Anhang von `DATA_AUDIT.md`).

**Der Metrik-Fehler:** Die ursprüngliche Implementierung berechnete
die AP **pro Frame** und mittelte über *alle* Frames — auch die ohne
das jeweilige Objekt (die zählten als 0). Da nur 70 von 266 Test-Frames
ein Fahrrad enthalten, war die Fahrrad-AP mathematisch auf
70/266 × 0.909 = **0.239 gedeckelt** — egal wie gut das Netz war.
Tatsächlich wurden **70 von 70 Fahrrädern erkannt**. Die Metrik wurde
auf das Standardverfahren umgebaut (*datensatzweite AP*: alle
Predictions des Testsplits gemeinsam ranken). Aufgefallen war der Bug,
weil zwei verschiedene Modelle exakt dieselbe Fahrrad-AP hatten —
verdächtig genau der theoretische Deckelwert.

## 9. Ergebnisse

### 9.1 Gesamtergebnis (Test-Split, korrigierte Metrik)

| Modell | mAP | AP30 person/bicycle/car | AP60 person/bicycle/car |
|---|---|---|---|
| **merged + GT-Sampling** | **0.8916** | 0.903 / 0.972 / 0.909 | 0.758 / 0.883 / 0.814 |
| merged Baseline | 0.8880 | 0.900 / 0.935 / 0.909 | 0.772 / 0.856 / 0.816 |
| merged + Oversampling | 0.8788 | 0.901 / 0.890 / 0.909 | 0.856 / 0.881 / 0.812 |
| os1 + GT-Sampling | 0.8570 | 0.902 / 0.863 / 0.909 | 0.623 / 0.791 / 0.909 |
| os0 + GT-Sampling | 0.8412 | 0.863 / 0.982 / 0.997 | 0.654 / 0.496 / 0.908 |

View-Ranking bei einheitlicher Konfiguration:
**merged > os1 > os0** — konsistent mit der Punktdichte pro Objekt.

### 9.2 Warum die Gesamtzahlen schmeicheln: dynamisch vs. statisch

Der Testbereich steht voller **geparkter Autos** (1548 von 1583
car-GT!), die in allen Splits identisch vorkommen — das Netz erkennt
sie trivial wieder (AP ≈ 1.0, Score ≈ 0.98). Die Gesamtwerte messen
dort also Wiedererkennung, keine Generalisierung. Deshalb wurde die
Bewertung nach dem static-Flag **getrennt** (eigenes Skript mit
*Ignore-Regionen*: die jeweils andere Gruppe zählt weder als Treffer
noch als Fehler):

**Nur dynamische Objekte (AP30/AP60):**

| Modell | person (n≈229) | bicycle (n=70) | car (n=35) |
|---|---|---|---|
| merged + GT-Sampling | 0.90 / 0.71 | **0.97 / 0.88** | **0.90** / 0.19 |
| merged Baseline | 0.89 / 0.71 | 0.94 / 0.86 | 0.70 / 0.13 |
| os0 + GT-Sampling | 0.73 / 0.35 | 0.98 / 0.50 | 0.74 / 0.68 |
| os1 + GT-Sampling | 0.90 / 0.46 | 0.86 / 0.79 | **0.09 / 0.09** |

### 9.3 Das Hauptergebnis: Fusion rettet das bewegte Auto

**os1 allein verfehlt das fahrende Auto fast vollständig** (AP30
0.02 Baseline / 0.09 mit bester Trainingskonfiguration): Seine Boxen
liegen systematisch ~1.5 m daneben, obwohl das Auto klar sichtbar ist
(median 420 Punkte). Da auch das beste Training nichts ändert, ist es
ein **Geometrieproblem der einseitigen Sicht**, kein Trainingsproblem.
Die **Fusion behebt es: 0.90.** os0 liegt dazwischen (0.74). Das ist
die zentrale, experimentell abgesicherte Aussage der Arbeit.

## 10. Ablationen (gezielte Vergleichsexperimente)

*Ablation* = kontrolliertes Experiment, bei dem genau eine Stellgröße
verändert wird, um ihren Effekt zu isolieren.

**A) GT-Sampling (Gewinner, übernommen).** Gegen die Klassenimbalance
(Auto:Fahrrad ≈ 22:1 im Training) werden beim Training zusätzliche
Fahrrad-/Personen-Instanzen aus anderen Trainingsframes in die Szene
kopiert (*Copy-Paste-Augmentierung*; die Instanzen-Datenbank wurde
strikt nur aus dem Train-Split gebaut — kein Leakage). Effekt:
bicycle AP30 0.935→0.972, dynamisches Auto 0.70→0.90, nichts wird
schlechter. Auf os1 ebenfalls positiv; auf dem dünner besetzten os0
ein Trade-off (Detektion ↑, Präzision ↓).

**B) Frame-Oversampling (verworfen).** Frames mit Fahrrädern doppelt
zeigen (*Repeat-Factor-Sampling*). Ergebnis: mAP 0.8788 < Baseline;
bringt keine neuen Objekt-Konstellationen. (Transparenz: Dieser Run
crashte bei Epoche 37, weil Metrik-Code während des Laufs deployt
wurde — mein Fehler, per Resume beendet; Checkpoint-Caveat in
`ABLATION_BICYCLE.md` dokumentiert. Lehre: nie Code ändern, den ein
laufender Prozess nutzt.)

**C) Cross-class NMS (metrisch neutral).** *NMS* (Non-Maximum
Suppression) entfernt Duplikat-Boxen, aber nur innerhalb derselben
Klasse — deshalb kann das Netz z. B. "Fahrrad" und "Person" gleichzeitig
auf demselben Cluster melden. Eine Nachverarbeitungsregel (Box zu ≥80 %
in höher gescorter Box anderer Klasse → verwerfen) entfernt 11/2353
Predictions, ändert aber keinen AP-Wert. Kosmetisch sinnvoll, mehr nicht.

**D) Unterboden-Filter (verworfen, Hypothese widerlegt).** Verdacht:
Spiegelreflexions-Punkte unter dem Boden (z ≈ −2) lösen Geisterboxen
aus. Test: Punkte z < −2 entfernt (0.09 % aller Punkte), neu trainiert.
Ergebnis: Geisterboxen wurden **nicht** weniger (58 vs. 42), das
dynamische Auto wurde schlechter (0.63). Nicht übernommen — und die
Reflexions-Hypothese damit sauber falsifiziert.

## 11. Laufzeiten und Echtzeitfähigkeit

*Inferenz* = das fertige Netz auf neue Daten anwenden. Gemessen auf
einer Tesla V100 (Batch 1, inkl. Datenladen):

| View | Zeit/Frame | Sensorbudget (10 Hz) | Auslastung |
|---|---|---|---|
| merged | 85 ms | 100 ms | 85 % ✓ |
| os0 / os1 | 45 / 43 ms | 100 ms | ~44 % ✓ |

Rechnung: 10 Aufnahmen/s → 100 ms Budget pro Aufnahme; alle Varianten
bleiben darunter → **echtzeitfähig, auch die Fusion** (Kosten der
Fusion: Faktor 2 wegen doppelter Punktzahl). Der Offline-Merge
(2.25 s, per-Frame-ICP) ist dafür nicht repräsentativ — online genügt
die einmal kalibrierte Transformation (Millisekunden).

## 12. Fehleranalysen (die drei verstandenen Schwächen)

**(1) Größenunterschätzung am bewegten Auto** (erklärt die niedrige
AP60 von 0.19): Das beste Pred pro GT sitzt quer nur 9 cm daneben,
Höhe und Winkel fast perfekt — aber die Box ist median **0.95 m zu
kurz und 0.41 m zu schmal**. Das Netz sieht nur ~60 % der
Fahrzeuglänge (Teilsichtbarkeit) und fällt auf sein KITTI-Größen-Prior
(~3.9×1.6 m) zurück, während die GT-Box 4.95×1.95 m misst. Rechnerisch
deckelt allein die Größe die IoU bei ~0.62. Eine anfängliche
Alternativhypothese (*Bewegungsverschmierung* durch zeitversetzte
Sensor-Sweeps) wurde geprüft und **widerlegt** (Label-Versatz zwischen
den Sensoren median nur 9 cm). Offener Prüfpunkt: reale Fahrzeugmaße
gegen die GT-Box prüfen (evtl. teilweise Label-Artefakt).

**(2) Niedrige Konfidenz am dynamischen Objekt:** bester Score am
bewegten Auto median 0.74 (10/35 Frames < 0.6) vs. 0.98 bei Parkern.
Konsequenz: Ein globaler Score-Schwellwert, der alle Fehldetektionen
entfernt UND das Zielauto in jedem Frame behält, existiert nicht —
in der Praxis braucht es klassenweise Betriebspunkte oder Tracking.

**(3) "Geisterräder"** (42 bicycle-Fehldetektionen, Score 0.3–0.6):
Drei per Punktwolken-Forensik belegte Ursachen: (a) echte, nie
gelabelte statische Objekte (vertikale Strukturen mit hunderten
Punkten an festen Positionen — vor Ort prüfen, evtl. abgestellte
Räder); (b) vereinzelt Reflexionsartefakte (als Hauptursache durch
Ablation D widerlegt); (c) **Positions-Halluzination**: Boxen über
blankem Boden, exakt (0.23 m) auf dem Weg der Trainingsräder — das
Netz hat sich *Orte* gemerkt statt nur Formen, weil alle Daten aus
**einer einzigen Szene** stammen. Kein GT-Sampling-Artefakt (die
Baseline zeigt dieselben FPs). Das ist die zentrale Limitation der
Datenbasis.

## 13. Werkzeuge und Artefakte

- **Prediction-Viewer** (`experiment/prediction_viewer.py`):
  interaktiver 3D-Viewer — Punktwolke + GT + Predictions, Frame-
  Navigation, klassenweise Score-Schwellen (Tasten 1/2/3 + `+`/`−`),
  `--full` für ganze Experimente (Predictions für alle 9981 Frames
  exportiert)
- **Abbildungen** (`figures_analysis/`): qualitative BEV-Beispiele
  (os1-Versagen vs. merged-Erfolg im selben Frame, *BEV* = Bird's Eye
  View, Vogelperspektive), Precision-Recall-Kurven, Score-Verteilungen,
  Geisterbox-Forensik
- **Analyse-Skripte** (`mmdetection3d/tools/analysis_tools/exp_*.py`):
  dynamisch/statisch-Evaluation, Fehlerzerlegung, cross-class NMS,
  PR-Figuren, Prediction-Exporte
- **Doku**: `DATA_AUDIT.md`, `ABLATION_BICYCLE.md`, `RESULTS.md`
- Alles versioniert auf GitHub
  (`DieserLaurenz/lidar-ojbect-detection-parking-lot`, Branch master)

## 14. Schlussfolgerungen

1. **Sensorfusion ist beim bewegten Objekt notwendig, nicht optional:**
   Ein Einzelsensor mit einseitiger Sicht lokalisiert das bewegte Auto
   systematisch falsch (0.02–0.09); die Fusion löst das (0.90) — bei
   weiterhin echtzeitfähiger Inferenz (85 ms < 100 ms Budget).
2. **GT-Sampling ist die wirksamste Trainingsmaßnahme** gegen die
   Klassenimbalance; Frame-Oversampling und Unterboden-Filter lohnen
   nicht (beides sauber belegte Negativergebnisse).
3. **Nur die dynamische Auswertung ist aussagekräftig** — Aggregatwerte
   werden von auswendig gelernten Parkern getragen.
4. **Daten- und Metrikqualität entscheidet:** Ohne die vier Fixes
   (Boden, Boxhöhe, Intensität, Metrik) wären sämtliche Zahlen falsch
   gewesen (Fahrrad schien 0.24, real 0.94).
5. **Restschwächen sind verstanden und benannt:** Boxgröße am
   teilsichtbaren Objekt (KITTI-Prior), Konfidenz am dynamischen
   Objekt (→ klassenweise Betriebspunkte/Tracking), Positions-
   Halluzination (→ Ein-Szenen-Limitation).

## 15. Limitationen und Future Work

**Limitationen** (gehören transparent in die Thesis): eine einzige
Szene; nur 3 physische Fahrräder / 1 Zielauto / wenige Personen;
n=35 dynamische Auto-Beobachtungen im Test; Zielauto im Testsegment
langsam (~2.2 m/s); statische Objekte identisch über alle Splits.

**Future Work:** Statik-/Hintergrundkarte für den fest installierten
Sensor (unterdrückt Positions-Halluzinationen), Tracking über Frames,
Online-Merge mit fester Extrinsik, mehr Szenen/Objektvielfalt,
Motion-Kompensation, ggf. Nachlabeln der statischen "Rad-Verdächtigen".

**Offene Prüfpunkte (nur vor Ort klärbar):** reale Maße des Zielautos
(GT-Box 4.95×1.95 m plausibel?); was steht an den Rohkoordinaten
[3.3, 2.2] und [2.7, 0.2] (abgestellte Räder oder Pfosten/Busch?).
