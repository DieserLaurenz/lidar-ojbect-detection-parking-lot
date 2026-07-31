# Multisensor-LiDAR-Objekterkennung in einer Tiefgarage

## Technische und wissenschaftliche Projektdokumentation – erster Entwurf

**Stand:** 16. Juli 2026

**Status:** Entwurf 0.1

**Projekt:** Multisensor LiDAR 3D Object Detection: Training and Evaluation

> Dieser Text ist ein erster zusammenhängender Dokumentationsentwurf. Er
> verwendet die abgeschlossene experiment-held-out Cross-Validation als
> wissenschaftliche Hauptbewertung. Frühere Resultate des temporalen
> 80/10/10-Splits werden nur noch als Teil der Fehler- und Methodenanalyse
> behandelt.

## Kurzfassung

In diesem Projekt wurde untersucht, wie gut vortrainierte neuronale Netze zur
dreidimensionalen Objekterkennung auf eine fest installierte
Zwei-Sensor-LiDAR-Anlage in einer Tiefgarage angepasst werden können. Die
Zielklassen sind Person, Fahrrad und Auto. Verglichen wurden die beiden
Einzelsensoransichten `os0` und `os1` sowie die aus beiden Sensoren erzeugte
fusionierte Punktwolke `merged`.

Als Hauptmodell wurde PointPillars eingesetzt und ausgehend von einem auf KITTI
vortrainierten Drei-Klassen-Checkpoint für 50 Epochen auf den Projektdaten
finetuned. Wegen des starken Klassenungleichgewichts wurde eine ausschließlich
aus Trainingsdaten erzeugte GT-Sampling-Datenbank verwendet. Die abschließende
Bewertung folgt einer vor der Ergebnissichtung eingefrorenen gepaarten
3-Fold-Cross-Validation. In jedem Fold werden drei vollständige Experimente –
je eines für Auto, Fahrrad und Person – als ungesehene Testaufnahmen
zurückgehalten.

PointPillars erreicht mit der fusionierten Ansicht in jedem Fold den höchsten
Gesamt-mAP. Der Fold-Mittelwert beträgt `0,798 ± 0,037` für `merged`,
`0,777 ± 0,018` für `os0` und `0,770 ± 0,027` für `os1`. Der Vorteil der Fusion
ist damit nicht sehr groß, aber über alle Folds konsistent. Besonders deutlich
ist er bei der Lokalisierung von Personen und Fahrrädern. Das bewegte Auto wird
dagegen in allen drei Ansichten zuverlässig erkannt; `os0` ist bei AP30 sogar
leicht besser. Die frühere Aussage, `os1` könne das bewegte Auto grundsätzlich
nicht erkennen, beruhte auf einem positionskonfundierten temporalen Split und
ist nicht mehr gültig.

Die fusionierte Punktwolke benötigt auf der Evaluations-GPU ungefähr `85 ms`
pro Modellinferenz, gegenüber `45 ms` für `os0` und `43 ms` für `os1`. Diese
Messung belegt die 10-Hz-Fähigkeit der Modellinferenz, nicht jedoch die
Echtzeitfähigkeit des vollständigen Fusionssystems. Die vorhandene
Offline-Merge-Pipeline benötigt im Median `2,25 s` pro Frame, weil sie
Rauschfilterung, Normalenschätzung und ICP für jeden Frame erneut ausführt. Ein
optimierter Online-Merge mit einmalig bestimmter Extrinsik wurde noch nicht
gemessen.

## 1. Aufgabenstellung und Forschungsfragen

Die Ausgangssituation ist eine fest installierte Verkehrsszene in einer
Tiefgarage, die gleichzeitig von zwei LiDAR-Sensoren aus unterschiedlichen
Perspektiven erfasst wird. Das Projekt soll ein vortrainiertes
3D-Objekterkennungsmodell auf diese Zielszene anpassen und den Nutzen der
Multisensorfusion quantifizieren.

Aus der Aufgabenstellung ergeben sich vier zentrale Forschungsfragen:

1. Wie stark verbessert das Finetuning auf den projektspezifischen
   Tiefgaragendaten ein vortrainiertes 3D-Detektionsmodell?
2. Welche Unterschiede bestehen zwischen den Einzelsensoransichten `os0` und
   `os1` und der fusionierten Ansicht `merged`?
3. Zeigt sich ein Fusionsvorteil eher beim bloßen Finden eines Objekts oder bei
   der räumlich präzisen Lokalisierung?
4. Welche typischen Fehlerbilder und praktischen Laufzeitgrenzen besitzt die
   entwickelte Verarbeitungskette?

Der Fokus liegt auf drei Klassen: `person`, `bicycle` und `car`. Die
wissenschaftliche Hauptaussage soll nicht aus einzelnen Beispielbildern oder
einem einzigen zufälligen Split abgeleitet werden, sondern aus vollständig
ungesehenen Aufnahmen unter einem für alle drei Ansichten gepaarten Protokoll.

## 2. Datengrundlage

### 2.1 Sensoraufbau und Ansichten

Die Szene wird von zwei Ouster-LiDAR-Sensoren mit einer Aufnahmerate von
ungefähr `10 Hz` erfasst. Jeder Sensor besitzt zunächst ein eigenes
Koordinatensystem. Aus den Rohdaten werden drei Eingabevarianten erzeugt:

- `os0`: transformierte Punktwolke des ersten Sensors,
- `os1`: transformierte Punktwolke des zweiten Sensors,
- `merged`: geometrisch registrierte und zusammengefügte Punktwolke beider
  Sensoren.

Eine Einzelsensor-Punktwolke enthält typischerweise ungefähr `105.000` Punkte,
die fusionierte Punktwolke ungefähr `210.000` Punkte. Die Fusion erhöht damit
die Punktdichte und ergänzt verdeckte Objektseiten, vergrößert aber zugleich
die zu verarbeitende Datenmenge und den sichtbaren Hintergrund.

### 2.2 Aufgezeichnete Experimente

Am 9. Oktober 2025 wurden neun Bewegungsaufnahmen in derselben Tiefgaragenszene
erstellt:

| Experimente | Bewegte Zielklasse |
|---|---|
| 1–3 | Auto |
| 4–6 | Fahrrad |
| 7–9 | Person |

Neben dem jeweiligen bewegten Zielobjekt enthält die Szene zahlreiche
statische Objekte, insbesondere geparkte Fahrzeuge. Diese statischen Objekte
sind für eine realistische Detektion relevant, dürfen bei der Interpretation
der Generalisierungsleistung aber nicht mit vollständig neuen Objektinstanzen
verwechselt werden: Sie stehen über viele Aufnahmen hinweg an denselben
Positionen.

Für die abschließende Cross-Validation wird ausschließlich die Schnittmenge
der gültigen Zeitstempel aller drei Ansichten verwendet. Dadurch erhält jede
Ansicht dieselben `2.122` physischen Frames. Die Annotationen bleiben dennoch
ansichtsspezifisch, weil ein Objekt je nach Sensorabdeckung und Sichtbarkeit
nicht in jeder Ansicht in exakt denselben Frames annotiert ist.

### 2.3 Annotationen

Die Ground-Truth-Annotationen bestehen aus orientierten dreidimensionalen
Bounding Boxes. Für jede Box werden Klasse, Position, Abmessungen und
Orientierung gespeichert. Zusätzlich kennzeichnet ein `static`-Attribut
statische Objekte. Die finalen Labels wurden mit dem projektspezifischen
Editor `experiment/manual_bbox_editor.py` geprüft und korrigiert.

Die ansichtsspezifischen Annotationen sind für den regulären Betrieb sinnvoll,
begrenzen aber einen direkten Klassenvergleich zwischen den Ansichten. Beim
bewegten Auto ist die Vergleichsmenge exakt gepaart (`n=339` in jeder
Ansicht). Bei Person und Fahrrad unterscheiden sich die GT-Zahlen. Beispielsweise
enthält die gepoolte Fahrradauswertung `718` Instanzen für `merged`, `627` für
`os0` und `581` für `os1`.

Diese Differenz ist besonders wichtig für die Interpretation des Fahrrad-AP30:
`os0` wird teilweise nur auf dem kürzeren, gut sichtbaren Abschnitt einer
Trajektorie bewertet, während `merged` durch die größere Abdeckung zusätzliche
und weiter entfernte Objektpositionen enthält. Der AP30-Vorsprung von `os0`
darf deshalb nicht ohne Einschränkung als grundsätzliche Überlegenheit eines
Einzelsensors interpretiert werden.

## 3. Datenaufbereitung und Sensorfusion

### 3.1 Registrierung und Merge

Die Merge-Pipeline ist in `experiment/pcd_merge.py` implementiert. Sie führt
für jedes Sensorpaar folgende Schritte aus:

1. Laden der beiden Punktwolken und ihrer Intensitätswerte,
2. Entfernen statistischer Ausreißer,
3. Anwenden einer manuell bestimmten Vorabtransformation,
4. Voxel-Downsampling und Normalenschätzung,
5. Feinausrichtung durch Iterative Closest Point (ICP),
6. Konkatenation der transformierten Punkte.

Die beiden Sensoren sind fest montiert. Für einen späteren Online-Betrieb wäre
deshalb eine einmalig auf einer geeigneten statischen Szene bestimmte
Extrinsik vorzuziehen. Die aktuelle Implementierung berechnet die
ICP-Korrektur dagegen für jeden Frame neu. Das ist für die Offline-Erzeugung
der Projektdaten verwendbar, aber weder laufzeitoptimal noch als endgültige
Live-Pipeline zu verstehen.

### 3.2 Konvertierung in das Trainingsformat

MMDetection3D verarbeitet die Daten in einer KITTI-ähnlichen Repräsentation.
Eigene Converter übertragen Punktwolken und Labels in dieses Format und
erzeugen die benötigten Info-PKLs. Dabei wurden mehrere geometrische und
metrische Fehler identifiziert und korrigiert:

- Der Boden lag durch eine fehlerhafte z-Transformation außerhalb des
  vorgesehenen Punktwolkenbereichs. Dadurch wurden Boden und untere
  Objektteile abgeschnitten.
- Der Label-Editor speichert die z-Koordinate im Boxzentrum, während die
  Trainingspipeline an einer Stelle die Boxunterkante erwartete. Ohne
  Korrektur schwebten die Boxen um eine halbe Objekthöhe zu hoch.
- Intensitäten wurden ursprünglich frameabhängig normiert. Sie werden nun mit
  einer festen Skala verarbeitet, sodass gleiche Intensitätswerte über Frames
  hinweg dieselbe Bedeutung besitzen.
- Die frühere Metrik mittelte AP pro Frame und wertete Frames ohne Instanz der
  betrachteten Klasse als Null. Dies begrenzte insbesondere die Fahrrad-AP
  künstlich. Die aktuelle Implementierung berechnet datensatzweite AP.

Alle maßgeblichen Trainings- und Cross-Validation-Ergebnisse basieren auf dem
korrigierten Datenstand. Frühere Resultate mit abgeschnittenem Boden,
verschobenen Boxen oder der fehlerhaften Pro-Frame-Metrik sind nicht
vergleichbar.

## 4. Modelle und Training

### 4.1 PointPillars

PointPillars wandelt die unregelmäßige Punktwolke in ein Raster senkrechter
Säulen, sogenannter Pillars, um. Innerhalb jedes Pillars werden die Punkte zu
einem Merkmalsvektor verdichtet. Anschließend verarbeitet ein
zweidimensionales Backbone die entstehende Bird's-Eye-View-Repräsentation und
sagt orientierte 3D-Boxen voraus.

Das Modell wird nicht von Grund auf neu trainiert. Ausgangspunkt ist ein auf
KITTI vortrainierter Drei-Klassen-Checkpoint. Dieses Vorwissen wird durch
Finetuning auf die Tiefgaragenszene angepasst. Für jeden Fold und jede Ansicht
wird ein eigenes Modell trainiert. Insgesamt entstehen damit neun evaluierbare
PointPillars-Foldmodelle.

Die wesentlichen Trainingsparameter sind:

| Parameter | Festgelegter Wert |
|---|---|
| Initialisierung | KITTI-Drei-Klassen-Checkpoint |
| Trainingsdauer | 50 Epochen |
| Zufallsseed | 42 |
| Klassen | Person, Fahrrad, Auto |
| Checkpoint-Auswahl | höchste Validierungs-mAP |
| GT-Sampling | Person 8, Fahrrad 10 |

Für jede Fold-/Ansichtskombination wird die GT-Sampling-Datenbank ausschließlich
aus deren Trainingsframes erzeugt. Test- und Validierungsobjekte können daher
nicht durch Copy-Paste-Augmentierung in das Training gelangen.

### 4.2 Klassenungleichgewicht und GT-Sampling

In den Trainingsdaten kommen deutlich mehr Autos als Fahrräder vor; im
früheren merged-Trainingssplit lag das Verhältnis ungefähr bei `22:1`.
GT-Sampling fügt zusätzliche Personen- und Fahrradinstanzen aus anderen
Trainingsframes in die aktuelle Trainingsszene ein. Dadurch steigt die
Variabilität der seltenen Klassen, ohne Frames aus Validierung oder Test zu
verwenden.

Frühere kontrollierte Ablationen zeigten, dass GT-Sampling die
Fahrrad-Detektion verbessert. Einfaches Frame-Oversampling war weniger
wirksam. Ein Filter für Punkte unterhalb des Bodens reduzierte die
Fehldetektionen nicht und verschlechterte andere Ergebnisse; er wurde deshalb
nicht in die Standardpipeline übernommen.

### 4.3 Architekturvergleich mit CenterPoint

Zusätzlich wurde CenterPoint unter demselben experiment-held-out Protokoll
trainiert. CenterPoint ist im Gegensatz zum anchorbasierten PointPillars ein
anchorfreier Detektor, der Objektzentren über eine Heatmap vorhersagt. Der
Vergleich dient als Robustheitsprüfung der Modellwahl, nicht als nachträgliche
Optimierung des Testprotokolls.

Auch bei CenterPoint ist `merged` in jedem Fold die beste Ansicht. Die
PointPillars-Foldmittel liegen jedoch in allen drei Ansichten höher:

| Ansicht | PointPillars | CenterPoint |
|---|---:|---:|
| merged | **0,798** | 0,751 |
| os0 | **0,777** | 0,659 |
| os1 | **0,770** | 0,723 |

Der Abstand entsteht vor allem durch die schwächere Lokalisierung von
CenterPoint bei strengeren IoU-Schwellen. Für die weitere Ergebnisdiskussion
bleibt PointPillars deshalb das Hauptmodell.

## 5. Evaluationsprotokoll

### 5.1 Warum der frühere temporale Split ersetzt wurde

Die erste Auswertung verwendete innerhalb jedes Experiments eine zeitliche
80/10/10-Aufteilung. Im Fall der Autoaufnahmen lagen die Testframes damit am
Ende der Trajektorie. Splitzugehörigkeit und Objektposition waren gekoppelt.
Der sehr niedrige damalige dynamische Auto-AP30 von `os1=0,09` beschrieb somit
nicht die allgemeine Qualität von `os1`, sondern die Generalisierung auf
diesen letzten räumlichen Trajektorienabschnitt.

Dieser Befund darf nicht mehr als Hauptresultat verwendet werden. Die
Cross-Validation mit vollständig zurückgehaltenen Experimenten ersetzt den
temporalen Split als wissenschaftliche Hauptbewertung.

### 5.2 Experiment-held-out Cross-Validation

Das Protokoll wurde am 11. Juli 2026 vor der Sichtung der
Cross-Validation-Testergebnisse eingefroren. Die drei Folds sind:

| Fold | Vollständig ungesehene Testexperimente |
|---|---|
| 1 | 1 Auto, 4 Fahrrad, 7 Person |
| 2 | 2 Auto, 5 Fahrrad, 8 Person |
| 3 | 3 Auto, 6 Fahrrad, 9 Person |

Innerhalb der jeweils übrigen sechs Experimente werden die letzten zehn
Prozent als Validierungsdaten verwendet. Die zehn davorliegenden Frames pro
Trainingsexperiment werden als zeitlicher Guard ausgeschlossen. Die
vollständigen Testexperimente werden weder für das Training noch für die
Checkpoint-Auswahl verwendet.

| Fold | Training | Validierung | Test | Guard |
|---|---:|---:|---:|---:|
| 1 | 1.239 | 146 | 677 | 60 |
| 2 | 1.172 | 137 | 753 | 60 |
| 3 | 1.227 | 143 | 692 | 60 |

Alle drei Ansichten erhalten pro Fold identische physische Zeitstempel. Damit
ist der View-Vergleich zeitlich gepaart. Die bereits erwähnten
ansichtsspezifischen Labels bleiben jedoch erhalten.

### 5.3 Metriken

Eine Vorhersage gilt als Treffer, wenn ihre dreidimensionale Intersection over
Union (IoU) mit einer noch nicht zugeordneten Ground-Truth-Box derselben Klasse
den betrachteten Schwellwert erreicht. Verwendet werden `AP30`, `AP40`, `AP50`
und `AP60`:

- AP30 bewertet überwiegend, ob das Objekt ungefähr gefunden wurde.
- AP60 verlangt eine deutlich präzisere räumliche Lokalisierung.
- Der Klassen-mAP mittelt über AP30 bis AP60.
- Der Gesamt-mAP mittelt zusätzlich über die drei Klassen.

Die AP wird als datensatzweite AP11 berechnet. Vorhersagen werden nach ihrem
Konfidenzscore sortiert; Matching bleibt frameweise, die Precision-Recall-Kurve
wird jedoch aus der gesamten Testmenge aufgebaut.

### 5.4 Fold-Mittel und gepoolte Out-of-Fold-Auswertung

Die Fold-Ergebnisse sind die primäre Generalisierungsschätzung. Aus ihnen
werden Mittelwert und Stichproben-Standardabweichung berechnet. Zusätzlich
werden die Vorhersagen aller drei Testfolds zu einer gepoolten
Out-of-Fold-Auswertung zusammengeführt. Dabei wurde jeder Frame genau von dem
Modell vorhergesagt, für das sein vollständiges Experiment ungesehen war.

Die gepoolte AP wird einmal auf der gemeinsamen Rangliste aller
Out-of-Fold-Vorhersagen berechnet. Sie ist deshalb nicht identisch mit dem
arithmetischen Mittel dreier AP-Werte. Da die Konfidenzscores der drei
Foldmodelle unterschiedlich kalibriert sein können, ergänzt der gepoolte Wert
die Foldanalyse, ersetzt sie aber nicht.

### 5.5 Offizielle und labelbasierte Bewegt-/Statisch-Auswertung

Die offizielle dynamische Teilbewertung ignoriert eine Vorhersage auf einem
statischen Objekt nur, wenn diese am jeweiligen Evaluationsschwellwert
ausreichend mit der statischen GT-Box überlappt. Eine etwas ungenau
lokalisierte, aber eindeutig einem geparkten Auto zuordenbare Box kann dadurch
bei AP60 als False Positive der dynamischen Rangliste erscheinen.

Deshalb existiert zusätzlich eine benannte labelbasierte Variante. Sie nutzt
das manuelle `static`-Attribut und ignoriert Vorhersagen auf Nichtzielobjekten
bereits ab IoU `0,25`. Die regulären All-Object-Metriken bleiben dabei
bit-identisch; nur die Teilranglisten für bewegte und statische Objekte ändern
sich. Die offizielle Auswertung bleibt die Vergleichsreferenz, während die
labelbasierte Variante die interpretierbare Analyse der bewegten Objekte
liefert.

## 6. Ergebnisse

### 6.1 Gesamtleistung von PointPillars

| Ansicht | Fold 1 | Fold 2 | Fold 3 | Mittel ± Std. | Gepoolt OOF |
|---|---:|---:|---:|---:|---:|
| **merged** | **0,840** | **0,786** | **0,768** | **0,798 ± 0,037** | **0,789** |
| os0 | 0,793 | 0,781 | 0,757 | 0,777 ± 0,018 | 0,759 |
| os1 | 0,801 | 0,751 | 0,759 | 0,770 ± 0,027 | 0,765 |

`merged` erzielt in jedem Fold den höchsten Gesamt-mAP. Der mittlere Vorsprung
beträgt ungefähr `0,021` gegenüber `os0` und `0,028` gegenüber `os1`. Bei nur
drei Folds ist das kein starker Signifikanznachweis. Es ist jedoch ein
konsistenter Hinweis darauf, dass Fusion die ausgewogenste Gesamtleistung
liefert und in keinem Fold von einer Einzelsicht übertroffen wird.

### 6.2 Bewegte Objekte

Die folgende Tabelle verwendet die labelbasierte Bewegt-/Statisch-Variante,
die auch der Abschlusspräsentation zugrunde liegt:

| Ansicht | Person AP30 / AP60 | Fahrrad AP30 / AP60 | Auto AP30 / AP60 |
|---|---:|---:|---:|
| **merged** | **0,901 / 0,729** | 0,800 / **0,668** | **0,891 / 0,883** |
| os0 | 0,653 / 0,370 | **0,895** / 0,378 | 0,885 / 0,856 |
| os1 | 0,896 / 0,426 | 0,744 / 0,529 | 0,832 / 0,789 |

Die Tabelle trennt zwei Aspekte:

- Bei AP30 finden alle Ansichten alle drei bewegten Klassen grundsätzlich.
- Bei AP60 liegt `merged` in allen Klassen vorn. Der Fusionsvorteil zeigt sich
  damit besonders in der räumlichen Präzision.

Beim bewegten Auto liegen die drei Ansichten wesentlich näher zusammen als im
früheren temporalen Split. Alle drei Ansichten erkennen das Auto gut. Die
defensible Schlussfolgerung lautet daher nicht, dass Fusion für die
Autoerkennung notwendig sei, sondern dass Fusion über Klassen und
IoU-Schwellen hinweg die beste Balance bietet.

### 6.3 Warum ist Fahrrad-AP30 bei os0 höher?

`os0` erreicht bei Fahrrädern einen höheren AP30, während `merged` bei AP60
deutlich führt. Ein Teil des AP30-Unterschieds ist ein Sichtbarkeits- und
Label-Effekt. Die Ansichten enthalten nicht dieselbe Fahrrad-GT-Menge.

Besonders deutlich ist Experiment 5:

| Ansicht | Fahrrad-GT | AP30 | AP60 |
|---|---:|---:|---:|
| merged | 310 | 0,727 | **0,611** |
| os0 | 244 | **0,907** | 0,425 |

In den Rohannotationen liegen die zusätzlichen `merged`-Fahrradframes
gegenüber `os0` überwiegend weiter entfernt. `os0` wird damit teilweise nur
auf dem kürzeren, gut sichtbaren Trajektorienabschnitt bewertet. Eine
objektweise vollständig faire Aussage darüber, ob `merged` dieselben
Fahrräder schlechter findet, würde eine zusätzliche Auswertung auf einer
gemeinsamen Referenz-GT-Menge erfordern.

Unabhängig davon zeigt AP60, dass die von `merged` gefundenen Fahrräder
räumlich wesentlich genauer lokalisiert werden. Die kurze Interpretation
lautet daher: `os0` besitzt in seiner eigenen Labelmenge die vollständigere
Groberkennung, die Fusion liefert die präziseren Boxen.

### 6.4 Offizielle dynamische Auswertung und AP60-Artefakt

In der offiziellen dynamischen Auswertung beträgt der Auto-AP30/AP60
`0,877/0,739` für `merged`, `0,885/0,847` für `os0` und `0,831/0,783` für
`os1`. Der niedrige offizielle AP60 von `merged` entsteht nicht primär durch
eine schlechte Box auf dem bewegten Auto. Die beste merged-Box auf dem
bewegten Auto hat im Median sogar die höchste IoU der drei Ansichten
(`0,848`).

Der Haupttreiber ist ein sensornahe geparktes Fahrzeug. Zahlreiche
hochkonfidente merged-Vorhersagen liegen dort bei IoU `0,25–0,60`. Sie werden
bei der offiziellen AP60-Teilbewertung nicht mehr als statische Ignore-Treffer
erkannt und erscheinen als False Positives in der dynamischen Rangliste. Die
labelbasierte Variante ordnet diese Vorhersagen dem statischen Objekt zu; der
merged-Auto-AP60 steigt dadurch von `0,739` auf `0,883`.

### 6.5 Vergleich zu CenterPoint

CenterPoint bestätigt das qualitative View-Ranking: `merged` ist auch dort in
jedem Fold die beste Ansicht. Gleichzeitig liegt CenterPoint im Gesamt-mAP
unter PointPillars. Unter der labelbasierten dynamischen Auswertung ist die
AP60-Lücke besonders bei Fahrrädern sichtbar. Der Fusionsbefund hängt somit
nicht ausschließlich an der anchorbasierten PointPillars-Architektur; die
höhere absolute Leistung in diesem Projekt liefert jedoch PointPillars.

## 7. Qualitative Fehleranalyse

### 7.1 Definition der Geisterboxen

Für die generalisierte OOF-Analyse wird eine Geisterbox als Vorhersage mit
Score mindestens `0,3` definiert, die sich keiner noch unbenutzten GT-Box
derselben Klasse bei 3D-IoU mindestens `0,3` zuordnen lässt. Diese Definition
umfasst nicht nur leere Halluzinationen, sondern auch Duplikate, falsch
klassifizierte reale Strukturen und schlecht lokalisierte Boxen.

Auf den `2.122` gemeinsamen Testframes ergeben sich:

| Ansicht | Person | Fahrrad | Auto | Gesamt |
|---|---:|---:|---:|---:|
| merged | 510 | 315 | 159 | 984 |
| os0 | 948 | 157 | 415 | 1.520 |
| os1 | 329 | 372 | 248 | 949 |

Die Fehler sind nicht auf Fahrräder beschränkt. `os0` erzeugt beispielsweise
deutlich mehr Person-Geisterboxen, während `merged` mehr Fahrrad-Geisterboxen
als `os0` enthält.

### 7.2 Plausible Ursachen

Die detaillierte Fahrradforensik zeigt drei wiederkehrende Gruppen:

1. Reale statische oder nicht gelabelte Strukturen werden als Fahrrad
   interpretiert.
2. Sparse Reflexions- und Unterbodenpunkte können lokale Aktivierungen
   auslösen.
3. Vorhersagen treten wiederholt an Positionen auf, an denen Fahrräder in den
   Trainingsaufnahmen vorkamen. Dies deutet auf eine ortsgebundene
   Szenenmemorierung hin.

Ein Unterbodenfilter reduzierte die Geisterboxen in einer Ablation nicht. Die
Spiegelungshypothese ist daher nicht der dominante Mechanismus.

Teilweise ragen Fahrrad- oder Personenboxen in die Ground-Truth-Box eines
Autos. Das ist als klassenübergreifende Mehrfachhypothese plausibel: Dünn
abgetastete Fahrzeugkanten oder das Heck können gleichzeitig einen
fahrradähnlichen Anchor aktivieren. Die standardmäßige Non-Maximum Suppression
arbeitet klassenweise. Eine korrekte Autobox unterdrückt deshalb keine
zusätzliche Fahrradbox. Da die Zusatzbox keine Fahrrad-GT trifft, zählt sie in
der Fahrradwertung als False Positive, auch wenn sie ein Auto überlappt.

Eine getestete Cross-Class-NMS entfernte vollständig enthaltene
klassenübergreifende Zusatzboxen und veränderte die Metriken nicht. Sie ist als
Ausgabebereinigung vertretbar, sollte aber nicht nachträglich anhand der
Cross-Validation-Testdaten optimiert werden.

### 7.3 Praktische Gegenmaßnahmen

Für eine spätere Anwendung bieten sich folgende Maßnahmen an:

- pro Klasse auf Validierungsdaten gewählte Konfidenzschwellen,
- zeitliche Trackbestätigung statt Entscheidungen aus einzelnen Frames,
- Hintergrundunterstützung oder eine statische Szenenkarte,
- Cross-Class-NMS als konservative Ausgabebereinigung,
- zusätzliche Trainingsszenen zur Verringerung der Ortsmemorierung.

## 8. Laufzeit

### 8.1 Modellinferenz

Die Laufzeiten wurden mit Batchgröße 1 auf einer Tesla V100-SXM3-32GB
gemessen. Die Messung umfasst die Modellpipeline einschließlich
Datenverarbeitung und Voxelisierung auf bereits vorbereiteten
Ansichtsdaten.

| Ansicht | Durchsatz | Zeit pro Frame | Typische Punktzahl |
|---|---:|---:|---:|
| merged | 11,7 fps | 85 ms | ca. 210.000 |
| os0 | 22,3 fps | 45 ms | ca. 105.000 |
| os1 | 23,5 fps | 43 ms | ca. 105.000 |

Bei einer Sensorfrequenz von `10 Hz` steht pro Frame ein Budget von `100 ms`
zur Verfügung. Die Modellinferenz auf der bereits fusionierten Punktwolke
bleibt mit `85 ms` darunter. Daraus folgt aber nur, dass die Modellinferenz
10-Hz-fähig ist.

### 8.2 Merge-Zeit und Ende-zu-Ende-Einordnung

Die vorhandene Offline-Merge-Pipeline wurde auf einer lokalen CPU mit acht
parallelen Workern gemessen. Über `3.327` Frames beträgt die mediane
Bearbeitungszeit `2,25 s` pro Frame, der Mittelwert `2,20 s`. Enthalten sind
PCD-Laden, statistischer Filter, feste Vorabtransformation, Downsampling,
Normalenschätzung und per-Frame-ICP.

Würde diese Pipeline unverändert seriell vor der Inferenz ausgeführt, läge die
Ende-zu-Ende-Latenz grob bei:

```text
2,25 s Merge + 0,085 s Inferenz ≈ 2,34 s pro Frame
```

Das wäre nicht echtzeitfähig. Bei fest installierten Sensoren kann die
Extrinsik jedoch einmalig offline bestimmt werden. Online wären dann nur
Zeitsynchronisation, feste Transformation und Konkatenation erforderlich.
Dieser optimierte Pfad wurde bislang nicht implementiert und gemessen.

Nach der merged-Inferenz verbleiben in einer seriellen 10-Hz-Pipeline nur
ungefähr `15 ms` für Merge und sonstigen Overhead. Eine parallele CPU-/GPU-
Pipeline könnte den Durchsatz verbessern, die Ende-zu-Ende-Latenz muss dennoch
separat erfasst werden. Die korrekte Aussage ist deshalb:

> Die Modellinferenz auf bereits fusionierten Daten ist 10-Hz-fähig; die
> Echtzeitfähigkeit des vollständigen Online-Fusionssystems ist noch nicht
> nachgewiesen.

## 9. Gültigkeit und Grenzen

### 9.1 Interne Gültigkeit

Für die Cross-Validation wurden folgende Schutzmaßnahmen umgesetzt:

- vollständige Testexperimente sind von Training und Checkpoint-Auswahl
  ausgeschlossen,
- Train-, Validierungs- und Testframes sind disjunkt,
- GT-Sampling-Datenbanken enthalten ausschließlich Trainingsobjekte,
- ein zeitlicher Guard trennt Training und Validierung,
- der Zufallsseed ist fixiert,
- alle Ansichten verwenden dieselben gültigen physischen Zeitstempel,
- die offiziellen Fold-mAP-Werte wurden aus den Prediction-Dateien mit einer
  maximalen Abweichung von `2,22e-16` reproduziert.

### 9.2 Einschränkungen der Aussagekraft

Die Ergebnisse besitzen folgende Grenzen:

1. Alle neun Experimente stammen aus derselben Tiefgaragenszene. Die
   Cross-Validation misst Generalisierung auf ungesehene Aufnahmen und
   Trajektorien in dieser Szene, nicht auf neue Orte.
2. Viele statische Fahrzeuge erscheinen in mehreren Folds an denselben
   Positionen. All-Object-Werte enthalten deshalb einen Anteil
   Szenenwiedererkennung.
3. Die Zahl physisch verschiedener Zielobjekte und Personen ist klein.
4. Person- und Fahrradlabels sind zwischen Ansichten nicht vollständig
   objektweise gepaart.
5. Drei Folds erlauben nur eine vorsichtige Interpretation der Streuung und
   keine starke Signifikanzbehauptung.
6. Gepoolte OOF-AP kombiniert Scores verschiedener Foldmodelle, die
   unterschiedlich kalibriert sein können.
7. Die Live-Laufzeit des optimierten Online-Merges wurde noch nicht gemessen.

## 10. Schlussfolgerung

Das Projekt zeigt, dass ein auf KITTI vortrainiertes 3D-Detektionsmodell durch
Finetuning erfolgreich auf eine fest installierte Tiefgaragenszene angepasst
werden kann. PointPillars liefert mit der fusionierten Punktwolke in jedem
Cross-Validation-Fold den höchsten Gesamt-mAP. Der Fusionsvorteil ist moderat,
aber konsistent und zeigt sich besonders in der präzisen Lokalisierung
bewegter Personen und Fahrräder.

Die Fusion ist dagegen nicht notwendig, um das bewegte Auto grundsätzlich zu
erkennen. Alle drei Ansichten erreichen auf vollständig ungesehenen
Autoaufnahmen eine hohe AP30; `os0` ist leicht am besten. Der frühere starke
`os1`-Einbruch war ein Artefakt der zeitlich und räumlich gekoppelten
Testaufteilung. Dieses Ergebnis unterstreicht, wie stark wissenschaftliche
Schlussfolgerungen von einem passenden Evaluationsdesign abhängen.

Die aussagekräftigste Gesamtformulierung lautet:

> Die Multisensorfusion liefert die beste ausgewogene Gesamtleistung und die
> präziseste Lokalisierung, ist aber nicht in jeder Klasse und bei jeder
> Metrik die beste Einzelansicht.

Für einen späteren Einsatz sind vor allem ein finaler Refit des ausgewählten
Modells, ein gemessener Online-Merge, validierungsbasierte Betriebsschwellen
und zeitliches Tracking erforderlich.

## 11. Reproduzierbarkeit und Projektartefakte

### 11.1 Zentrale Implementierungen

- Cross-Validation-Splits:
  `mmdetection3d/tools/dataset_converters/create_exp_crossval_splits.py`
- PointPillars-CV-Konfiguration:
  `mmdetection3d/configs/pointpillars/pointpillars_hv_secfpn_8xb6-50e_exp-crossval-gtsample.py`
- Trainingssteuerung: `mmdetection3d/tools/run_exp_crossval.sh`
- Cross-Validation-Auswertung:
  `mmdetection3d/tools/analysis_tools/exp_eval_crossval.py`
- Geisterboxanalyse:
  `mmdetection3d/tools/analysis_tools/exp_ghost_boxes_crossval.py`
- Punktwolken-Merge: `experiment/pcd_merge.py`
- Label-Editor: `experiment/manual_bbox_editor.py`

### 11.2 Primäre Ergebnisquellen

- `results/CROSS_VALIDATION.md`: eingefrorenes Evaluationsprotokoll
- `results/CROSS_VALIDATION_RESULTS.md`: offizielle PointPillars-Ergebnisse
- `results/CROSS_VALIDATION_STATIC_AWARE.md`: labelbasierte Teilbewertung
- `results/CENTERPOINT_CV_STATIC_AWARE.md`: CenterPoint-Vergleich
- `results/GHOST_BOXES.json`: generalisierte OOF-Fehlerzählung
- `results/DATA_AUDIT.md`: Daten-, Metrik- und Ursachenanalyse
- `results/ABLATION_BICYCLE.md`: frühere kontrollierte Ablationen

### 11.3 Präsentationsmaterial

Das primäre qualitative Video ist
`results/videos/exp1_cv_full_three_views_oblique.mp4`. Es zeigt alle `105`
gemeinsamen gültigen Frames von Experiment 1 mit den drei Fold-1-Modellen, für
die das vollständige Experiment während Training und Checkpoint-Auswahl
ungesehen war.

Ältere Videos des temporalen 11-Frame-Testsegments sind nur historische
Artefakte. Sie dürfen nicht als Beleg dafür verwendet werden, dass `os1` das
bewegte Auto generell nicht erkennen könne.

## 12. Offene Arbeiten

Für die nächste Überarbeitung beziehungsweise einen späteren Deployment-Stand
sind folgende Punkte offen:

- wissenschaftliche Quellen und Literaturverweise ergänzen,
- Abbildungen und Tabellen nummerieren und im Text referenzieren,
- gemeinsame Fahrrad-GT-Auswertung für einen streng objektweise gepaarten
  View-Vergleich durchführen,
- Fehler nach kontinuierlicher Position, Distanz und Punktunterstützung
  aufschlüsseln,
- finalen Refit für die ausgewählte Ansicht und eine vorab festgelegte
  Trainingsdauer auf allen gültigen Daten trainieren,
- Online-Merge mit fester Extrinsik implementieren und Ende-zu-Ende-Latenz
  messen,
- Betriebsschwellen ausschließlich aus Validierungsfolds bestimmen,
- Tracking und Hintergrundmodell als separate, klar benannte Erweiterungen
  evaluieren.

Ein finaler Refit ist nicht mit einem neuen unabhängigen Testergebnis zu
verwechseln. Die Cross-Validation bleibt die wissenschaftliche
Leistungsschätzung; der Refit wäre das spätere Inferenzartefakt.
