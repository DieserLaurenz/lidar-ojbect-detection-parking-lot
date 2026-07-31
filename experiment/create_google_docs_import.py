"""Create a Word/Google-Docs-friendly HTML version of the documentation.

Microsoft Word can convert the generated HTML to DOCX while preserving the
heading hierarchy, tables, lists, emphasis and code-style spans. Google Drive
can then import the DOCX as an editable Google Docs document.
"""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

import markdown


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "DOKUMENTATION_ENTWURF.md"
DEFAULT_OUTPUT = ROOT / "DOKUMENTATION_GOOGLE_DOCS.html"


WORD_CSS = r"""
@page Section1 {
  size: 595.3pt 841.9pt;
  margin: 58pt 54pt 62pt 54pt;
}

div.Section1 { page: Section1; }

body {
  color: #172733;
  background: white;
  font-family: Aptos, "Segoe UI", Arial, sans-serif;
  font-size: 10.5pt;
  line-height: 1.45;
}

.cover {
  page-break-after: always;
  border-top: 18pt solid #16847f;
  padding: 45pt 36pt 30pt;
  background: #123247;
  color: white;
  min-height: 650pt;
}

.cover-kicker {
  color: #8fd8d5;
  font-size: 10pt;
  font-weight: bold;
  letter-spacing: 1.5pt;
  text-transform: uppercase;
}

.cover-title {
  margin: 52pt 0 22pt;
  color: white;
  font-size: 30pt;
  font-weight: bold;
  line-height: 1.08;
}

.cover-subtitle {
  margin: 0 0 50pt;
  color: #d9e8ed;
  font-size: 15pt;
  line-height: 1.35;
}

.cover-summary {
  margin: 0 0 85pt;
  padding: 12pt 16pt;
  color: #eaf4f5;
  background: #1d5365;
  border-left: 5pt solid #58b9bf;
}

.cover-meta {
  width: 100%;
  border-collapse: collapse;
  color: white;
}

.cover-meta td {
  width: 50%;
  padding: 10pt 14pt 10pt 0;
  border-top: 1pt solid #4f7584;
  vertical-align: top;
}

.meta-label {
  color: #8fd8d5;
  font-size: 8pt;
  font-weight: bold;
  letter-spacing: 1pt;
  text-transform: uppercase;
}

.toc-page {
  page-break-after: always;
}

.toc-page h1 {
  margin-top: 0;
  color: #123247;
  font-size: 24pt;
  border-bottom: 4pt solid #16847f;
  padding-bottom: 8pt;
}

.toc-page ul {
  list-style: none;
  padding-left: 0;
}

.toc-page ul ul {
  padding-left: 18pt;
}

.toc-page li {
  margin: 5pt 0;
}

.toc-page a {
  color: #1e607b;
  text-decoration: none;
}

h1 {
  page-break-before: always;
  margin: 0 0 18pt;
  padding-top: 9pt;
  color: #123247;
  border-top: 4pt solid #16847f;
  font-size: 22pt;
  line-height: 1.15;
}

h2 {
  margin: 24pt 0 8pt;
  color: #1e607b;
  font-size: 15pt;
  line-height: 1.2;
}

h3 {
  margin: 18pt 0 6pt;
  color: #16847f;
  font-size: 12pt;
}

p {
  margin: 0 0 9pt;
}

ul, ol {
  margin: 6pt 0 12pt;
  padding-left: 24pt;
}

li {
  margin: 3pt 0;
}

blockquote {
  margin: 14pt 0;
  padding: 10pt 14pt;
  color: #28414f;
  background: #edf6f6;
  border-left: 5pt solid #16847f;
}

code {
  color: #16485f;
  background: #edf3f5;
  font-family: Consolas, "Courier New", monospace;
  font-size: 9pt;
}

pre {
  margin: 12pt 0;
  padding: 12pt;
  color: white;
  background: #173343;
  border-left: 4pt solid #58b9bf;
  font-family: Consolas, "Courier New", monospace;
  font-size: 8.5pt;
  white-space: pre-wrap;
}

table {
  width: 100%;
  margin: 10pt 0 15pt;
  border-collapse: collapse;
  font-size: 8.5pt;
}

th {
  padding: 7pt;
  color: white;
  background: #123247;
  border: 1pt solid #123247;
  text-align: left;
  vertical-align: bottom;
}

td {
  padding: 6pt 7pt;
  border: 1pt solid #d6e0e5;
  vertical-align: top;
}

tr:nth-child(even) td {
  background: #f2f6f7;
}

strong {
  color: #102c3e;
}

.source-note {
  margin-top: 24pt;
  padding-top: 8pt;
  color: #788a95;
  border-top: 1pt solid #d9e2e7;
  font-size: 8pt;
}
"""


def source_body(source: str) -> str:
    marker = "## Kurzfassung"
    position = source.find(marker)
    if position < 0:
        raise ValueError(f"Expected heading '{marker}' in source document")
    return source[position:]


def promote_headings(fragment: str) -> str:
    """Make Markdown H2 chapters become Google Docs Heading 1 entries."""
    fragment = re.sub(r"<h2([^>]*)>", r"<h1\1>", fragment)
    fragment = fragment.replace("</h2>", "</h1>")
    fragment = re.sub(r"<h3([^>]*)>", r"<h2\1>", fragment)
    fragment = fragment.replace("</h3>", "</h2>")
    return fragment


def render(source: str) -> tuple[str, str]:
    md = markdown.Markdown(
        extensions=["extra", "sane_lists", "toc"],
        extension_configs={"toc": {"toc_depth": "2-3"}},
        output_format="html5",
    )
    body = promote_headings(md.convert(source_body(source)))
    toc = md.toc
    return body, toc


def document_html(source_path: Path) -> str:
    source = source_path.read_text(encoding="utf-8")
    body, toc = render(source)
    source_name = html.escape(source_path.name)
    return f"""<!doctype html>
<html xmlns:o="urn:schemas-microsoft-com:office:office"
      xmlns:w="urn:schemas-microsoft-com:office:word" lang="de">
<head>
  <meta charset="utf-8">
  <meta name="ProgId" content="Word.Document">
  <meta name="Generator" content="Codex Google Docs import generator">
  <title>Multisensor-LiDAR-Objekterkennung – Projektdokumentation</title>
  <style>{WORD_CSS}</style>
</head>
<body>
<div class="Section1">
  <div class="cover">
    <div class="cover-kicker">Technische und wissenschaftliche Dokumentation</div>
    <div class="cover-title">Multisensor-LiDAR-Objekterkennung in einer Tiefgarage</div>
    <div class="cover-subtitle">Training, Cross-Validation und Bewertung von
      PointPillars und CenterPoint auf zwei Einzelsensoransichten und einer
      fusionierten Punktwolke</div>
    <div class="cover-summary">Experiment-held-out 3-Fold-Cross-Validation ·
      Person, Fahrrad und Auto · Vergleich von os0, os1 und merged ·
      Fehleranalyse und Laufzeiteinordnung</div>
    <table class="cover-meta">
      <tr>
        <td><span class="meta-label">Projekt</span><br>Multisensor LiDAR 3D Object Detection</td>
        <td><span class="meta-label">Dokumentstatus</span><br>Erster Entwurf · Version 0.1</td>
      </tr>
      <tr>
        <td><span class="meta-label">Stand</span><br>16. Juli 2026</td>
        <td><span class="meta-label">Hauptmodell</span><br>PointPillars · KITTI-Finetuning</td>
      </tr>
    </table>
  </div>

  <div class="toc-page">
    <h1>Inhaltsverzeichnis</h1>
    {toc}
  </div>

  {body}
  <p class="source-note">Automatisch erzeugt aus <code>{source_name}</code>.
    Die Kapitel sind als echte Überschriftsebenen formatiert und erscheinen
    nach dem Import in der Google-Docs-Dokumentgliederung.</p>
</div>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document_html(source), encoding="utf-8")
    print(f"Google Docs import HTML: {output}")


if __name__ == "__main__":
    main()
