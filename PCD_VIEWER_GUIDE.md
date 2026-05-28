# PCD Sequence Viewer Guide

Start im Projektroot:

```bash
cd "$PROJ_DIR"
```

Falls nötig prüfen:

```bash
ls
```

Du solltest `experiment`, `data`, `mmdetection3d` sehen.

## Viewer Controls

```text
Maus links ziehen  = drehen
Mausrad            = zoomen
Maus rechts/mitte  = verschieben
N                  = nächster Frame
B                  = vorheriger Frame
R                  = Kamera resetten
Space              = Play/Pause
```

Die Kamera bleibt beim Framewechsel erhalten.

## 1. PCD-Export Verstehen

```bash
python experiment/pcd_sequence_viewer.py --experiment 1_experiment_car_1 --mode raw --point-size 3
```

Das zeigt die direkt exportierten PCDs.
Rot = `os0`, grün = `os1`. Beide sind noch in ihren lokalen Sensor-Koordinatensystemen.

## 2. Gemeinsames Koordinatensystem

```bash
python experiment/pcd_sequence_viewer.py --experiment 1_experiment_car_1 --mode transformed --point-size 3
```

Hier wurden beide Sensoren transformiert.
Rot und grün sollten jetzt zusammen dieselbe Garage/Szene bilden.

## 3. Gemergte Punktwolke

```bash
python experiment/pcd_sequence_viewer.py --experiment 1_experiment_car_1 --mode merged --force-color --point-size 3
```

Das ist die fusionierte Multi-Sensor-Punktwolke.

## 4. Background Removal

```bash
python experiment/pcd_sequence_viewer.py --experiment 1_experiment_car_1 --mode foreground --point-size 3
```

Grau = Background.
Magenta = Punkte, die nach Background-Removal übrig bleiben.

## 5. Auto-Labels Anzeigen

```bash
python experiment/pcd_sequence_viewer.py --experiment 1_experiment_car_1 --mode merged --force-color --show-labels --point-size 3
```

Das zeigt die gemergte Punktwolke mit automatisch erzeugten 3D-Bounding-Boxes.

## 6. Animation

Bei jedem Kommando kannst du `--play` anhängen:

```bash
python experiment/pcd_sequence_viewer.py --experiment 1_experiment_car_1 --mode transformed --point-size 3 --play
```

Mit `Space` pausieren, dann mit `N`/`B` durchgehen.

## 7. Langsame Wiedergabe / Weniger Frames

Jeden 5. Frame:

```bash
python experiment/pcd_sequence_viewer.py --experiment 1_experiment_car_1 --mode transformed --point-size 3 --step 5
```

Langsamer abspielen:

```bash
python experiment/pcd_sequence_viewer.py --experiment 1_experiment_car_1 --mode transformed --point-size 3 --play --wait 0.25
```

## 8. Andere Experimente

Car:

```bash
python experiment/pcd_sequence_viewer.py --experiment 2_experiment_car_2 --mode transformed --point-size 3
```

Bike:

```bash
python experiment/pcd_sequence_viewer.py --experiment 4_experiment_bike_1 --mode transformed --point-size 3
```

Person:

```bash
python experiment/pcd_sequence_viewer.py --experiment 7_experiment_person_1 --mode transformed --point-size 3
```

Person mit Background-Removal:

```bash
python experiment/pcd_sequence_viewer.py --experiment 7_experiment_person_1 --mode foreground --point-size 3
```

## Empfohlene Reihenfolge

```bash
python experiment/pcd_sequence_viewer.py --experiment 1_experiment_car_1 --mode raw --point-size 3
python experiment/pcd_sequence_viewer.py --experiment 1_experiment_car_1 --mode transformed --point-size 3
python experiment/pcd_sequence_viewer.py --experiment 1_experiment_car_1 --mode merged --force-color --point-size 3
python experiment/pcd_sequence_viewer.py --experiment 1_experiment_car_1 --mode foreground --point-size 3
python experiment/pcd_sequence_viewer.py --experiment 1_experiment_car_1 --mode merged --force-color --show-labels --point-size 3
```

So sieht man nacheinander: Export -> Transformation -> Merge -> Background-Removal -> Auto-Labels.
