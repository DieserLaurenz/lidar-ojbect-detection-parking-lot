# Projektbericht: Multisensor-LiDAR-Objekterkennung an einer Kreuzung

> **Nachtrag vom 11. Juli 2026:** Die im Bericht verwendete temporale
> 80/10/10-Auswertung koppelte den Testsplit an die letzten Objektpositionen.
> Die Schlussfolgerung, os1 könne das bewegte Auto praktisch nicht erkennen
> (`AP30=0.09`), ist deshalb als allgemeine Aussage nicht haltbar. Eine neue
> gepaarte 3-Fold-Cross-Validation hält jeweils komplette Experimente zurück.
> Dort erreicht das dynamische Auto gepoolt AP30/AP60: merged 0.877/0.739,
> os0 0.885/0.847, os1 0.831/0.783. Fusion erzielt weiterhin den besten
> Gesamt-mAP (Fold-Mittel 0.798), ihr belastbarer Vorteil ist aber ausgewogene
> Gesamtleistung und bessere Personen-/Fahrrad-Lokalisierung, nicht die
> notwendige Rettung der Autoerkennung. Maßgeblicher Ergebnisreport:
> `results/CROSS_VALIDATION_RESULTS.md`. Der alte Berichtstext bleibt als
> Dokumentation des damaligen Auswertungsstands erhalten.

**Masterarbeit — Stand: 3. Juli 2026.**

Dieser Bericht erzählt von Anfang bis Ende, was in diesem Projekt
gemacht wurde: wie die Daten entstanden sind, welche Fehler auf dem Weg
gefunden und behoben wurden, welche Ergebnisse am Ende herauskamen und
welche Schlüsse man daraus ziehen kann. Fachbegriffe werden bei ihrer
ersten Verwendung immer direkt erklärt, sodass der Bericht auch ohne
Vorwissen verständlich ist. Wer tiefer einsteigen möchte, findet alle
Detailbelege in drei weiteren Dokumenten: `results/DATA_AUDIT.md` enthält die
technische Prüfung der Daten und alle Detailanalysen,
`results/ABLATION_BICYCLE.md` beschreibt die Vergleichsexperimente, und
`results/RESULTS.md` ist der kompakte Ergebnisreport mit allen Tabellen.

---

## 1. Worum geht es in diesem Projekt?

An einer Kreuzung wurden zwei LiDAR-Sensoren fest installiert. Ein
LiDAR ist ein Laserscanner: Er sendet Lichtpulse aus, misst, wie lange
das Licht bis zu einem Hindernis und zurück braucht, und berechnet
daraus für jeden Puls einen Messpunkt im Raum. Aus Millionen solcher
Messpunkte pro Sekunde entsteht eine sogenannte Punktwolke — man kann
sie sich wie ein dreidimensionales Foto der Umgebung vorstellen, das
nur aus einzelnen Punkten besteht.

In diesen Punktwolken soll ein neuronales Netz automatisch Personen,
Fahrräder und Autos erkennen. Das Erkennungsergebnis ist für jedes
gefundene Objekt eine sogenannte Bounding Box: ein gedachter Quader,
der beschreibt, wo das Objekt steht, wie groß es ist und in welche
Richtung es zeigt.

Die zentrale Frage der Arbeit lautet: **Lohnt es sich, die Daten beider
Sensoren zu kombinieren, oder reicht ein einzelner Sensor aus?** Um das
zu beantworten, wurden drei Varianten verglichen. Die Variante "os0"
verwendet nur den ersten Sensor, die Variante "os1" nur den zweiten
Sensor, und die Variante "merged" verwendet die zusammengeführten
Punktwolken beider Sensoren.

## 2. Wie sind die Daten entstanden?

Am 9. Oktober 2025 wurden neun Experimente aufgenommen. In jedem
Experiment bewegte sich genau ein bekanntes Zielobjekt durch die
Kreuzung: In den Experimenten 1 bis 3 fuhr ein Auto durch die Szene,
in den Experimenten 4 bis 6 ein Fahrrad, und in den Experimenten 7 bis
9 ging eine Person. Zusätzlich standen während der gesamten Aufnahmen
geparkte Autos und andere unbewegliche Objekte in der Szene.

Die Sensoren nehmen zehnmal pro Sekunde eine vollständige Punktwolke
auf. Insgesamt entstanden so 3327 Aufnahmen (im Folgenden "Frames"
genannt) pro Sensorvariante.

## 3. Wie werden die beiden Sensoren zu einem Bild kombiniert?

Jeder Sensor misst die Welt aus seiner eigenen Perspektive und in
seinem eigenen Koordinatensystem — das heißt, jeder Sensor betrachtet
sich selbst als Nullpunkt. Bevor man beide Punktwolken kombinieren
kann, muss man sie deshalb passgenau übereinanderlegen. Diesen Vorgang
nennt man Merge (englisch für "zusammenführen"). Er läuft für jede
Aufnahme in vier Schritten ab:

1. Beide Punktwolken werden geladen, und offensichtliches Messrauschen
   wird entfernt. Dafür sorgt ein statistischer Ausreißerfilter: Punkte,
   die auffällig weit von allen ihren Nachbarpunkten entfernt liegen,
   sind mit hoher Wahrscheinlichkeit Messfehler und werden verworfen.
2. Eine der beiden Punktwolken wird grob in das Koordinatensystem der
   anderen gedreht und verschoben. Die dafür nötige Drehung und
   Verschiebung beschreibt, wie die beiden Sensoren zueinander stehen.
   Man nennt sie die Extrinsik; sie wurde vorab einmalig von Hand
   bestimmt.
3. Anschließend übernimmt ein Algorithmus namens ICP (Iterative Closest
   Point) die Feinarbeit: Er verschiebt die beiden Punktwolken in
   kleinen Schritten so lange gegeneinander, bis sie bestmöglich
   aufeinanderliegen.
4. Zum Schluss werden die Punkte beider Wolken einfach zu einer
   gemeinsamen Punktwolke aneinandergehängt.

Dieser Vorgang dauerte in unserer Verarbeitung im Mittel etwa 2,3
Sekunden pro Aufnahme. Das klingt langsam, ist aber kein grundsätzliches
Problem des Ansatzes: Fast die gesamte Zeit entfällt auf den
Rauschfilter und die ICP-Feinausrichtung. Da die Sensoren fest montiert
sind und sich nicht bewegen, müsste man die Feinausrichtung eigentlich
nur ein einziges Mal durchführen. In einem Live-System bliebe pro
Aufnahme nur das Drehen und Aneinanderhängen übrig, was nur wenige
Tausendstelsekunden dauert. Mehr dazu in Abschnitt 11.

## 4. Woher weiß man, was richtig ist? Das Labeling

Damit ein neuronales Netz lernen kann und damit man seine Leistung
bewerten kann, braucht man die "richtige Antwort" zum Vergleichen.
Diese richtige Antwort nennt man Ground Truth: Ein Mensch schaut sich
jede Aufnahme an und zeichnet von Hand um jedes echte Objekt eine
Bounding Box. Diesen Vorgang nennt man Labeling, die einzelnen
Markierungen Labels.

Für dieses Projekt wurde dafür ein eigenes Markierungsprogramm
entwickelt (`experiment/manual_bbox_editor.py`). Beim Labeling wurden
einige wichtige Konventionen eingehalten:

- Es gibt drei Objektklassen: Person, Fahrrad und Auto.
- Jede Box bekommt zusätzlich ein Merkmal namens "static", das angibt,
  ob das Objekt stillsteht (zum Beispiel ein geparktes Auto) oder sich
  bewegt. Diese Unterscheidung wird in Abschnitt 9 noch sehr wichtig.
- Aufnahmen, die aus irgendeinem Grund unbrauchbar waren, wurden als
  ungültig markiert. Das betraf 2521 Frames. Es wurde ausdrücklich
  geprüft, dass diese Frames weder ins Training noch in die Bewertung
  eingeflossen sind.

## 5. Vorbereitung fürs Training: Datenformat und Datenaufteilung

Für das Training wurde das frei verfügbare Software-Framework
MMDetection3D verwendet, ein Baukasten für 3D-Objekterkennung. Dieses
Framework erwartet die Daten in einem bestimmten Format, dem
KITTI-Format. KITTI ist der bekannteste öffentliche Datensatz für
Objekterkennung im Straßenverkehr, und sein Datenformat hat sich als
Standard etabliert. Eigens geschriebene Konvertierungsprogramme drehen
und verschieben unsere Punktwolken dafür in ein KITTI-ähnliches
Koordinatensystem und schreiben sie in das erwartete Dateiformat.

Anschließend wurden die Daten aufgeteilt. Ein neuronales Netz darf
nämlich niemals mit denselben Daten bewertet werden, mit denen es
trainiert wurde — sonst prüft man nur, ob es auswendig gelernt hat.
Üblich ist eine Aufteilung in drei Teile, die man Splits nennt:

- **Trainings-Split (80 %):** Mit diesen Daten lernt das Netz.
- **Validierungs-Split (10 %):** Mit diesen Daten wird während des
  Trainings zwischendurch geprüft, welcher Trainingsstand der beste ist.
- **Test-Split (10 %):** Diese Daten bleiben bis ganz zum Schluss
  unangetastet und liefern die endgültige, unverfälschte Bewertung.

Die Aufteilung erfolgte zeitlich: Von jedem Experiment bilden die
ersten 80 Prozent der Zeitachse das Training und die letzten 10 Prozent
den Test. So kann das Training niemals "in die Zukunft" des Tests
schauen. Es wurde außerdem geprüft, dass keine einzige Aufnahme in
zwei Splits gleichzeitig vorkommt. Ein solches Durchsickern von
Testdaten ins Training nennt man Leakage; es würde die Ergebnisse
künstlich verbessern und wurde hier ausgeschlossen.

## 6. Das Daten-Audit: drei versteckte Fehler gefunden und behoben

Bevor den Ergebnissen vertraut werden konnte, wurde die gesamte
Datenverarbeitungskette systematisch überprüft. Diese Prüfung (das
"Audit") förderte drei echte Fehler zutage, die alle behoben wurden.

**Fehler 1: Der Boden wurde abgeschnitten.** Das Netz verarbeitet nur
Punkte innerhalb eines festgelegten räumlichen Bereichs, unter anderem
nur Punkte, die höher als 3 Meter unter dem Koordinatenursprung liegen.
Durch einen falschen Verschiebungswert in der Konvertierung landete der
Boden unserer Szene aber bei 3,44 Metern unter dem Ursprung — also
außerhalb dieses Bereichs. Ein Filter entfernte deshalb den kompletten
Boden und zusätzlich die unteren etwa 40 Zentimeter jedes Objekts. Das
Netz sah im Training also Menschen ohne Füße und Autos ohne Räder. Nach
der Korrektur liegt der Boden auf der Höhe, die das Netz aus seinem
Vortraining gewohnt ist.

**Fehler 2: Alle Boxen schwebten in der Luft.** Das Markierungsprogramm
speichert die Höhenposition einer Box als deren Mittelpunkt. Das
Trainingsframework interpretiert denselben Zahlenwert aber als
Unterkante der Box. Dadurch saßen alle Trainingsboxen um eine halbe
Boxhöhe zu hoch — bei einem Auto sind das rund 75 Zentimeter. Der
Fehler wurde im Konvertierungsprogramm korrigiert, und die Korrektur
wurde statistisch überprüft.

**Fehler 3: Die Helligkeitswerte waren nicht vergleichbar.** Jeder
Messpunkt trägt neben seiner Position auch eine Intensität — ein Maß
dafür, wie stark die Oberfläche das Laserlicht reflektiert hat. Diese
Werte wurden ursprünglich in jeder Aufnahme am jeweils hellsten Punkt
skaliert. Die Folge: Dieselbe Oberfläche bekam in jeder Aufnahme einen
anderen Zahlenwert. Jetzt wird stattdessen durch eine feste Zahl
geteilt, sodass die Werte überall dieselbe Bedeutung haben.

Alle Ergebnisse, die vor diesen Korrekturen entstanden waren, wurden
für ungültig erklärt. Der bereinigte Datenstand trägt die Bezeichnung
"v2", und alle Zahlen in diesem Bericht beziehen sich darauf.

## 7. Das Training des neuronalen Netzes

Als Erkennungsmodell wurde PointPillars verwendet, ein etabliertes und
besonders schnelles neuronales Netz für die 3D-Objekterkennung. Die
Grundidee: Das Netz betrachtet die Szene von oben und teilt sie in ein
Raster aus senkrechten Säulen ein (englisch "pillars", daher der Name).
Alle Punkte innerhalb einer Säule werden zu einer kompakten Beschreibung
zusammengefasst. Auf diesem Raster arbeitet dann ein Bilderkennungsnetz,
das die Bounding Boxes vorhersagt. Als Startpunkte für die Boxen dienen
sogenannte Anchors: vordefinierte Referenzboxen in der typischen Größe
eines Autos, einer Person oder eines Fahrrads, die das Netz dann nur
noch verschieben und in der Größe anpassen muss.

Das Netz wurde nicht von Null trainiert. Stattdessen wurde ein
Checkpoint als Ausgangspunkt verwendet — so nennt man einen
gespeicherten Zustand eines bereits trainierten Netzes. Unser
Startpunkt war ein auf dem KITTI-Datensatz vortrainiertes PointPillars,
das also bereits grundsätzlich weiß, wie Autos, Fußgänger und Radfahrer
in LiDAR-Daten aussehen. Dieses Vorwissen wurde dann auf unsere Szene
angepasst. Dieses Vorgehen nennt man Finetuning. Trainiert wurde über
50 Epochen; eine Epoche bedeutet, dass das Netz einmal alle
Trainingsdaten gesehen hat.

Nach jeder Epoche wurde auf dem Validierungs-Split gemessen, wie gut
das Netz gerade erkennt. Der beste Zwischenstand wurde automatisch
gespeichert. Erst ganz am Ende wurde dieser beste Stand ein einziges
Mal auf den Test-Split angewendet — das Ergebnis dieser einen Messung
ist die berichtete Testleistung. Dieses strenge Vorgehen stellt sicher,
dass die Testzahl nicht durch wiederholtes Probieren geschönt ist.

## 8. Wie wird die Erkennungsleistung gemessen? Und der Metrik-Fehler

Um zu entscheiden, ob eine vorhergesagte Box ein Treffer ist, vergleicht
man sie mit der zugehörigen Ground-Truth-Box über die sogenannte IoU
(Intersection over Union). Das ist das Verhältnis aus dem Volumen, in
dem sich beide Boxen überlappen, und dem Volumen, das beide Boxen
zusammen einnehmen. Eine IoU von 1,0 bedeutet perfekte Deckung, eine
IoU von 0 bedeutet keinerlei Berührung.

Die Erkennungsleistung wird als Average Precision (AP) angegeben, auf
Deutsch etwa "mittlere Genauigkeit". Die Zahl hinter der AP nennt die
IoU-Schwelle, ab der eine Vorhersage als Treffer zählt: AP30 verlangt
mindestens 30 Prozent Überlappung und misst damit vor allem, ob ein
Objekt überhaupt gefunden wurde. AP60 verlangt 60 Prozent Überlappung
und misst damit, ob die Box auch präzise sitzt. Der Gesamtwert mAP
(mean Average Precision) ist der Mittelwert über die vier verwendeten
Schwellen (0,3 / 0,4 / 0,5 / 0,6) und über alle drei Objektklassen.

Jede Vorhersage des Netzes trägt außerdem einen Score: eine Zahl
zwischen 0 und 1, die ausdrückt, wie sicher sich das Netz bei dieser
Box ist. In die AP-Berechnung fließt keine feste Score-Schwelle ein —
stattdessen werden alle Vorhersagen nach ihrem Score sortiert und die
gesamte Rangliste bewertet. Das bestraft genau das Richtige: Ein Netz,
dessen Fehler niedrige Scores haben und dessen Treffer hohe Scores
haben, verliert kaum Punkte.

**Bei dieser Messung steckte ein schwerwiegender Fehler in der
ursprünglichen Implementierung.** Die AP wurde für jede Aufnahme einzeln
berechnet und dann über alle Aufnahmen gemittelt — auch über solche, in
denen die gesuchte Objektklasse gar nicht vorkommt. Solche Aufnahmen
gingen als "0 Punkte" in den Mittelwert ein. Da nur 70 der 266
Test-Aufnahmen überhaupt ein Fahrrad enthalten, konnte der Fahrrad-Wert
rechnerisch nie über 0,24 steigen — völlig unabhängig davon, wie gut
das Netz war. Tatsächlich hatte das Netz alle 70 Fahrräder gefunden.
Aufgefallen ist der Fehler, weil zwei unterschiedlich trainierte
Modelle exakt denselben Fahrrad-Wert erreichten — und zwar genau den
theoretischen Maximalwert dieser fehlerhaften Rechnung. Die Messung
wurde daraufhin auf das übliche Standardverfahren umgestellt, bei dem
alle Vorhersagen des gesamten Test-Splits gemeinsam bewertet werden.

Dieser Fehler hat das Netz nie schlechter gemacht — aber er hat lange
verdeckt, wie gut es wirklich war.

## 9. Die Ergebnisse

### 9.1 Gesamtergebnis auf dem Test-Split

Die folgende Tabelle zeigt die Testleistung aller trainierten Varianten.
Die Abkürzungen in den Spalten stehen für die drei Klassen Person (p),
Fahrrad (b für bicycle) und Auto (c für car). "GT-Sampling" und
"Oversampling" sind Trainingsvarianten, die in Abschnitt 10 erklärt
werden.

| Modell | mAP | AP30 p/b/c | AP60 p/b/c |
|---|---|---|---|
| **merged mit GT-Sampling** | **0.8916** | 0.903 / 0.972 / 0.909 | 0.758 / 0.883 / 0.814 |
| merged Basisvariante | 0.8880 | 0.900 / 0.935 / 0.909 | 0.772 / 0.856 / 0.816 |
| merged mit Oversampling | 0.8788 | 0.901 / 0.890 / 0.909 | 0.856 / 0.881 / 0.812 |
| os1 mit GT-Sampling | 0.8570 | 0.902 / 0.863 / 0.909 | 0.623 / 0.791 / 0.909 |
| os0 mit GT-Sampling | 0.8412 | 0.863 / 0.982 / 0.997 | 0.654 / 0.496 / 0.908 |

Bei einheitlicher Trainingskonfiguration ergibt sich die Rangfolge
merged vor os1 vor os0. Das passt zu der Erwartung, dass mehr
Messpunkte pro Objekt die Erkennung erleichtern.

### 9.2 Warum diese Zahlen schmeicheln — und die ehrlichere Auswertung

Ein wichtiger Einwand kam beim Betrachten der Szene auf: Der
Testbereich steht voller geparkter Autos. Von den 1583 Auto-Boxen im
Test-Split gehören 1548 zu geparkten Autos — und genau dieselben
geparkten Autos standen auch schon in den Trainingsdaten, am selben
Ort, in derselben Stellung. Das Netz muss sie also nicht wirklich
"erkennen", sondern nur wiedererkennen. Solche trivialen Treffer ziehen
die Gesamtwerte nach oben.

Deshalb wurde die Bewertung zusätzlich getrennt durchgeführt: einmal
nur für die bewegten Objekte und einmal nur für die stillstehenden.
Die bewegten Objekte sind die eigentliche Aufgabe des Systems. Das
Ergebnis für die bewegten Objekte (jeweils AP30 / AP60):

| Modell | Person | Fahrrad | Auto |
|---|---|---|---|
| merged mit GT-Sampling | 0.90 / 0.71 | **0.97 / 0.88** | **0.90** / 0.19 |
| merged Basisvariante | 0.89 / 0.71 | 0.94 / 0.86 | 0.70 / 0.13 |
| os0 mit GT-Sampling | 0.73 / 0.35 | 0.98 / 0.50 | 0.74 / 0.68 |
| os1 mit GT-Sampling | 0.90 / 0.46 | 0.86 / 0.79 | **0.09 / 0.09** |

### 9.3 Das Hauptergebnis der Arbeit

In dieser Tabelle steckt der wichtigste Befund: **Der Sensor os1
verfehlt das fahrende Auto fast vollständig.** Sein Wert von 0,09
bedeutet, dass seine Vorhersagen praktisch nie als Treffer zählen —
die Boxen liegen im Mittel anderthalb Meter neben dem echten Auto,
obwohl das Auto in seinen Daten klar sichtbar ist (im Mittel 420
Messpunkte). Besonders aussagekräftig: Auch mit der besten gefundenen
Trainingskonfiguration verbessert sich os1 kaum (von 0,02 auf 0,09).
Das Problem lässt sich also nicht durch besseres Training lösen — es
liegt an der Geometrie: os1 sieht das Auto nur von einer Seite und
kann seine Position deshalb systematisch nicht richtig bestimmen.

**Die Fusion beider Sensoren löst genau dieses Problem: Der Wert
springt auf 0,90.** Das ist die zentrale, experimentell abgesicherte
Aussage der Arbeit — die Kombination beider Blickwinkel ist beim
bewegten Objekt kein Luxus, sondern notwendig.

### 9.4 Einordnung: der Vergleich mit der Vorgängerarbeit

Zu genau diesem Versuchsaufbau gibt es eine Vorgängerarbeit: die
Masterarbeit von Thea Pagel (2025, als PDF im Projektordner). Sie hat
dieselben Aufnahmen aus der Tiefgarage verwendet — dieselben zwei
Sensoren, dieselbe Szene, dieselbe Software zum Zusammenfügen der
Punktwolken. Der entscheidende Unterschied: Sie hat die neuronalen
Netze **nicht auf den Szenendaten nachtrainiert**, sondern nur fertige,
auf fremden Datensätzen vortrainierte Netze auf die Aufnahmen
angewendet.

Das Ergebnis dieser Vorgehensweise war ernüchternd. Ihr bestes Netz
(vortrainiert auf dem Straßenverkehrsdatensatz KITTI) erreichte auf
den fusionierten Daten einen Gesamtwert von 0,058 — auf einer Skala,
auf der 1,0 perfekt wäre. Unsere Variante desselben Netzes erreicht
nach dem Nachtraining auf denselben Daten 0,8916. Der Unterschied
beträgt also etwa den Faktor 15. Fairerweise muss man dazusagen, dass
die beiden Zahlen nicht bis auf die Nachkommastelle vergleichbar sind:
Die Vorgängerarbeit hat ihre Referenz-Boxen automatisch erzeugt (ohne
manuelle Korrektur, wie wir sie durchgeführt haben) und eine eigene
Variante der Messmethode verwendet. An der Größenordnung des
Unterschieds ändert das aber nichts.

Daraus folgt die vielleicht wichtigste praktische Lehre des
Gesamtprojekts: **Ein vortrainiertes Netz einfach auf eine neue Szene
loszulassen, funktioniert nicht — das Nachtraining auf Daten aus der
Zielszene ist der entscheidende Schritt.** Interessant ist außerdem,
dass beide Arbeiten unabhängig voneinander denselben Fusionsbefund
liefern: Schon bei der Vorgängerarbeit war die Fusion beider Sensoren
um den Faktor 1,4 bis 3,4 besser als jeder Einzelsensor. Nach dem
Nachtraining bleibt dieser Vorteil bestehen und zeigt sich am
deutlichsten dort, wo er am wichtigsten ist: beim bewegten Auto
(Einzelsensor os1: 0,09 — Fusion: 0,90).

## 10. Die Vergleichsexperimente (Ablationen)

Eine Ablation ist ein kontrolliertes Experiment, bei dem man genau
eine Stellschraube verändert und alles andere gleich lässt, um den
Effekt dieser einen Änderung sauber zu messen. Vier solcher
Experimente wurden durchgeführt.

**Experiment A: GT-Sampling — der Gewinner.** In den Trainingsdaten
kommen auf jedes Fahrrad ungefähr 22 Autos. Bei einem so starken
Ungleichgewicht lernt das Netz die seltene Klasse schlechter. Als
Gegenmaßnahme wurden beim Training zusätzliche Fahrrad- und
Personen-Exemplare aus anderen Trainingsaufnahmen in die jeweilige
Szene hineinkopiert — eine Standard-Technik, die man GT-Sampling oder
Copy-Paste-Augmentierung nennt. Wichtig für die Sauberkeit: Die
kopierten Exemplare stammen ausschließlich aus dem Trainings-Split,
niemals aus Validierungs- oder Testdaten. Das Ergebnis: Die
Fahrrad-Erkennung stieg von 0,935 auf 0,972, und die Erkennung des
fahrenden Autos verbesserte sich von 0,70 auf 0,90 — ohne dass
irgendetwas anderes schlechter wurde. Diese Variante wurde als
Standard übernommen.

**Experiment B: Frame-Oversampling — verworfen.** Die naheliegendere
Alternative: Aufnahmen, die Fahrräder enthalten, werden im Training
einfach doppelt so oft gezeigt. Das brachte nichts — die Gesamtleistung
lag sogar unter der Basisvariante. Die Erklärung: Das bloße Wiederholen
derselben Aufnahmen erzeugt keine neuen Situationen, aus denen das Netz
etwas lernen könnte. (Zur Transparenz: Dieser Trainingslauf stürzte bei
Epoche 37 ab, weil während des laufenden Trainings der Messcode auf dem
Server ausgetauscht wurde — ein vermeidbarer Bedienfehler. Der Lauf
wurde vom letzten Zwischenstand fortgesetzt; die Einschränkung ist in
`results/ABLATION_BICYCLE.md` dokumentiert und ändert nichts am negativen
Fazit.)

**Experiment C: Klassenübergreifende Duplikat-Entfernung — wirkungslos,
aber unschädlich.** Das Netz meldet manchmal mehrere Deutungen für
denselben Punkthaufen gleichzeitig, zum Beispiel "Auto" und darin
zusätzlich "Fahrrad". Der übliche Aufräummechanismus (Non-Maximum
Suppression, kurz NMS) entfernt Duplikate nur innerhalb derselben
Klasse. Getestet wurde eine Zusatzregel: Eine Box, die fast vollständig
in einer sichereren Box einer anderen Klasse liegt, wird verworfen.
Diese Regel entfernte 11 von 2353 Vorhersagen und veränderte keinen
einzigen Messwert. Sie schadet nicht und macht die Ausgabe optisch
sauberer, bringt aber messbar nichts.

**Experiment D: Unterboden-Filter — Hypothese widerlegt.** In den Daten
finden sich vereinzelt Punkthaufen unterhalb des Bodens, vermutlich
Spiegelungen an nassen Flächen. Die Hypothese war, dass diese
"Spiegelpunkte" Fehldetektionen auslösen. Zum Test wurden alle Punkte
tiefer als 2 Meter unter dem Ursprung entfernt (das betraf nur 0,09
Prozent aller Punkte) und das Netz neu trainiert. Das Ergebnis
widerlegte die Hypothese: Die Fehldetektionen wurden nicht weniger
(58 statt 42), und die Erkennung des fahrenden Autos wurde schlechter
(0,63 statt 0,90). Der Filter wurde deshalb nicht übernommen. Auch ein
solches Negativergebnis ist wertvoll: Eine plausible Erklärung wurde
aufgestellt, gezielt getestet und sauber ausgeschlossen.

## 11. Geschwindigkeit: Kann das System in Echtzeit arbeiten?

Unter Inferenz versteht man das Anwenden des fertig trainierten Netzes
auf neue Daten: Punktwolke hinein, Bounding Boxes heraus. Die
Inferenzzeit ist die Dauer eines solchen Durchlaufs. Sie wurde auf der
Grafikkarte des Rechenservers (einer Tesla V100) gemessen, einschließlich
des Ladens und Vorverarbeitens der Daten.

Die Rechnung zur Echtzeitfähigkeit geht so: Der Sensor liefert zehn
Aufnahmen pro Sekunde, also kommt alle 100 Millisekunden eine neue
Punktwolke an. Ein System arbeitet in Echtzeit, wenn es mit der
Verarbeitung einer Aufnahme fertig ist, bevor die nächste eintrifft —
es muss also unter 100 Millisekunden bleiben.

| Variante | Zeit pro Aufnahme | Budget | Ergebnis |
|---|---|---|---|
| merged (Fusion) | 85 ms | 100 ms | echtzeitfähig, 15 ms Reserve |
| os0 einzeln | 45 ms | 100 ms | echtzeitfähig |
| os1 einzeln | 43 ms | 100 ms | echtzeitfähig |

Die Fusion kostet ungefähr doppelt so viel Rechenzeit wie ein
Einzelsensor — logisch, denn es sind doppelt so viele Punkte zu
verarbeiten. Entscheidend ist: **Auch die Fusion bleibt unter dem
100-Millisekunden-Budget und ist damit echtzeitfähig.**

Zum vollständigen Bild gehört der Merge-Schritt (Abschnitt 3). In
unserer Offline-Verarbeitung dauerte er 2,3 Sekunden pro Aufnahme —
das läge weit über dem Budget. Diese Zahl ist aber nicht repräsentativ
für einen Live-Betrieb: Sie enthält die ICP-Feinausrichtung, die bei
fest montierten Sensoren nur ein einziges Mal nötig ist. Live bliebe
nur das Drehen und Aneinanderhängen der Punkte übrig, was wenige
Millisekunden kostet. Das Gesamtsystem bliebe damit im Zeitbudget.

## 12. Die drei verbleibenden Schwächen — und warum sie auftreten

Kein Ergebnis ist perfekt. Die drei Schwachstellen des Systems wurden
bis zur Ursache zurückverfolgt.

**Schwäche 1: Die Box um das fahrende Auto ist zu klein.** Das fahrende
Auto wird zwar fast immer gefunden (die AP30 von 0,90 belegt das), aber
der strengere Wert AP60 liegt nur bei 0,19 — die Boxen sitzen also
nicht präzise genug. Die Fehlerzerlegung zeigt: Die Position stimmt
fast perfekt (seitlich nur 9 Zentimeter Abweichung), aber die
vorhergesagte Box ist im Mittel 95 Zentimeter zu kurz und 41 Zentimeter
zu schmal. Der Grund: Die Sensoren sehen von dem fahrenden Auto immer
nur einen Teil (die Messpunkte decken nur etwa 60 Prozent der
Fahrzeuglänge ab), und in dieser Unsicherheit fällt das Netz auf die
typischen Automaße aus seinem KITTI-Vortraining zurück — und
KITTI-Autos sind kleiner als unser Zielfahrzeug laut Label (4,95 m ×
1,95 m). Eine zunächst vermutete andere Erklärung — dass die Bewegung
des Autos die fusionierte Punktwolke "verschmiert", weil die beiden
Sensoren minimal zeitversetzt aufnehmen — wurde geprüft und
ausgeschlossen: Der Versatz beträgt im Mittel nur 9 Zentimeter. Offen
ist noch ein Realitätsabgleich: Sollten die echten Maße des Zielautos
kleiner sein als die Label-Box, wäre ein Teil dieser "Schwäche" in
Wirklichkeit ein Messfehler beim Labeln.

**Schwäche 2: Das Netz ist sich beim bewegten Objekt unsicher.** Beim
fahrenden Auto liegt der Konfidenz-Score im Mittel bei 0,74, bei den
geparkten Autos bei 0,98. In 10 von 35 Fällen liegt er unter 0,6.
Praktische Folge: Es gibt keinen einzelnen Schwellwert, der gleichzeitig
alle Fehldetektionen unterdrückt und das Zielauto in jeder Aufnahme
behält. Ein reales System braucht deshalb entweder unterschiedliche
Schwellwerte pro Objektklasse oder eine zeitliche Verfolgung über
mehrere Aufnahmen hinweg (Tracking), die einzelne unsichere Aufnahmen
überbrückt.

**Schwäche 3: "Geisterfahrräder".** Das Netz meldet vereinzelt
Fahrräder an Stellen, an denen keine sind (42 Fälle im Test, alle mit
niedrigen Scores zwischen 0,3 und 0,6). Die Untersuchung dieser Fälle
bis hinunter auf einzelne Aufnahmen ergab drei Ursachen. Erstens:
Einige dieser "Fehler" sitzen auf echten, unbeweglichen Objekten mit
hunderten Messpunkten, die schlicht nie gelabelt wurden — was dort
wirklich steht (möglicherweise tatsächlich abgestellte Fahrräder!),
lässt sich nur vor Ort klären. Zweitens: Die vermuteten
Spiegelungs-Artefakte — diese Erklärung wurde durch Experiment D
widerlegt. Drittens, und am interessantesten: Einige Boxen stehen über
völlig leerem Boden, aber auf den Zentimeter genau an Stellen, an denen
während des Trainings Fahrräder vorbeifuhren. Das Netz hat sich also
nicht nur gemerkt, wie Fahrräder aussehen, sondern auch, **wo** sie
üblicherweise auftauchen — und meldet an diesen Orten nun gelegentlich
Fahrräder ohne Anlass. Das ist eine direkte Folge davon, dass alle
Daten aus einer einzigen Szene stammen, und damit die wichtigste
Einschränkung dieser Datenbasis.

## 13. Entstandene Werkzeuge

Neben den Trainingsergebnissen sind mehrere wiederverwendbare Werkzeuge
entstanden:

- Ein **interaktiver 3D-Viewer** (`experiment/prediction_viewer.py`),
  mit dem man durch die Aufnahmen blättern und dabei die Punktwolke,
  die von Hand gezeichneten Labels und die Vorhersagen des Netzes
  gleichzeitig sehen kann. Die Anzeigeschwelle lässt sich pro
  Objektklasse einstellen, und mit der Option `--full` lassen sich
  ganze Experimente statt nur der Testabschnitte betrachten.
- **Abbildungen** für die Thesis (im Ordner `results/figures/`):
  Beispielszenen aus der Vogelperspektive (unter anderem der direkte
  Vergleich "os1 verfehlt das Auto / merged trifft es" im selben
  Moment), Genauigkeits-Vollständigkeits-Kurven (Precision-Recall) und
  die Score-Verteilungen.
- **Analyse-Skripte** (in `mmdetection3d/tools/analysis_tools/`) für
  die getrennte Auswertung bewegt/unbewegt, die Fehlerzerlegung, die
  Duplikat-Entfernung und die Erzeugung aller Abbildungen.
- Die gesamte Arbeit ist mit Git versioniert und auf GitHub gesichert
  (Repository `DieserLaurenz/lidar-ojbect-detection-parking-lot`).

## 14. Schlussfolgerungen

Fünf Aussagen lassen sich aus diesem Projekt belegen:

1. **Die Sensorfusion ist beim bewegten Objekt notwendig, nicht nur
   hilfreich.** Ein einzelner Sensor mit ungünstigem Blickwinkel
   bestimmt die Position des fahrenden Autos systematisch falsch, und
   kein noch so gutes Training ändert daran etwas. Die Fusion beider
   Sensoren löst das Problem — und bleibt dabei schnell genug für den
   Live-Betrieb.
2. **GT-Sampling ist die wirksamste Trainingsmaßnahme** gegen das
   Klassenungleichgewicht. Die beiden Alternativen (Aufnahmen
   wiederholen, Unterboden-Punkte filtern) wurden getestet und
   begründet verworfen.
3. **Nur die getrennte Auswertung nach bewegten und unbewegten Objekten
   ist aussagekräftig.** Die Gesamtwerte werden von auswendig gelernten
   geparkten Autos getragen und überzeichnen die wahre Leistung.
4. **Daten- und Messqualität entscheiden über alles.** Ohne die vier
   behobenen Fehler (Boden, Boxhöhe, Intensität, Messmethode) wären
   sämtliche Zahlen falsch gewesen — die Fahrrad-Erkennung schien
   katastrophal (0,24), war aber in Wahrheit ausgezeichnet (0,94).
5. **Die verbleibenden Schwächen sind verstanden und erklärbar** —
   zu kleine Boxen am teilverdeckten Objekt, geringere Konfidenz bei
   Bewegung und ortsgebundene Fehldetektionen aus der Ein-Szenen-
   Datenbasis.

## 15. Grenzen der Arbeit und Ausblick

Zu einer ehrlichen Bewertung gehören die Grenzen: Alle Daten stammen
aus einer einzigen Szene. Es gab nur drei verschiedene physische
Fahrräder, ein Zielauto und wenige Personen. Im Test sind nur 35
Beobachtungen des fahrenden Autos enthalten, und das Auto bewegt sich
im Testabschnitt langsam (etwa 2 Meter pro Sekunde). Die geparkten
Autos sind in Training und Test identisch. Diese Punkte begrenzen, wie
weit sich die Zahlen verallgemeinern lassen, und gehören transparent in
die Thesis.

Für die Zukunft bieten sich an: eine Hintergrundkarte der festen Szene,
die ortsgebundene Fehldetektionen unterdrückt; eine zeitliche Verfolgung
der Objekte über mehrere Aufnahmen (Tracking); der Umbau des Merges auf
eine einmalig kalibrierte Ausrichtung für den Live-Betrieb; und vor
allem Daten aus weiteren Szenen mit mehr Objektvielfalt.

Zwei kleine Fragen können nur vor Ort geklärt werden: Wie groß ist das
Zielauto wirklich (zum Abgleich mit der Label-Box von 4,95 × 1,95
Metern)? Und was steht an den beiden Stellen nahe der Sensoren, an
denen das Netz beharrlich Fahrräder meldet — sind es vielleicht
tatsächlich abgestellte Fahrräder, die nie gelabelt wurden?
