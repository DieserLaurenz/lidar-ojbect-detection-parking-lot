# Abgabe: Projektbericht und Abschlusspräsentation

Die beiden finalen Dokumente des Projekts.

| Datei | Inhalt |
|---|---|
| `Projektbericht.pdf` | Projektbericht, 38 Seiten, A4 — die gebaute Fassung der Quellen in `latex/` |
| `Abschlusspraesentation-DCAITI-LiDAR-Projekt.pptx` | Abschlusspräsentation, 22 Folien inkl. Backup-Folie zur Daten- und Messqualität |
| `latex/` | LaTeX-Quellen des Berichts |

Der Bericht enthält einen **Sperrvermerk**: Vervielfältigung und Veröffentlichung
sind auch auszugsweise nicht erlaubt.

## Bericht bauen

Der Bericht verwendet `biblatex` mit BibTeX-Backend, daher die vier Durchläufe:

```bash
cd latex
pdflatex main
bibtex   main
pdflatex main
pdflatex main
```

Ergebnis ist `latex/main.pdf`. Nach inhaltlichen Änderungen bitte die gebaute
Fassung nach `Projektbericht.pdf` kopieren — nur diese ist versioniert, die
Zwischendateien des Laufs sind über `.gitignore` ausgeschlossen.

Getestet mit MiKTeX 25.12. Benötigte Pakete: `babel` (ngerman), `geometry`,
`graphicx`, `subcaption`, `float`, `booktabs`, `multirow`, `amsmath`, `amssymb`,
`biblatex`, `csquotes`, `hyperref`, `placeins`, `listings`, `soul`, `breqn`,
`chngcntr`.

## Aufbau der Quellen

`main.tex` bindet die Kapitel einzeln ein; `Configs.tex` enthält Präambel und
Seitenlayout (A4, 11 pt, 3 cm seitlich).

```
Deckblatt · Sperrvermerk · Abkuerzungen · Zusammenfassung
Aufgabenstellung        Forschungsfragen und Wegweiser
Datengrundlage          Sensoraufbau, Experimente, Annotationen
Datenaufbereitung       Merge-Pipeline, Konvertierung, korrigierte Datenfehler
Modelle                 PointPillars, GT-Sampling, CenterPoint-Vergleich
Evaluationsprotokoll    Cross-Validation, Metriken, Auswertungsvarianten
Ergebnisse              Finetuning, Fusion, Klassen, Experimente, Architektur
Fehleranalyse           Geisterboxen, Ursachen, Gegenmaßnahmen
Laufzeit                Inferenz- und Merge-Zeiten
Gueltigkeit             Interne Gültigkeit und Grenzen
Ausblick · Schlussfolgerung · Reproduzierbarkeit
```

`OffeneArbeiten.tex` ist eine **projektinterne Checkliste** und absichtlich
nicht in `main.tex` eingebunden — sie listet die verbliebenen offenen Punkte
und gehört nicht in die Abgabefassung.

## Datenquellen der Zahlen

Alle Werte im Bericht stammen aus `../results/`:

| Berichtsabschnitt | Quelle |
|---|---|
| Fold- und Bewegt-mAP, alle vier IoU-Schwellen | `CROSS_VALIDATION_RESULTS.json`, `CROSS_VALIDATION_STATIC_AWARE.json` |
| Offizielle gegenüber labelbasierter Variante | `CROSS_VALIDATION_RESULTS.md` |
| Geisterboxen | `GHOST_BOXES.json` |
| CenterPoint-Vergleich | `CENTERPOINT_CV_RESULTS.md` |
| Datenfehler und Metrikkorrektur | `DATA_AUDIT.md` |
| Ablationen | `ABLATION_BICYCLE.md` |
| Laufzeiten | `RESULTS.md` |
