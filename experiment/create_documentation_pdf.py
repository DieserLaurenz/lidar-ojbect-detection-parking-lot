"""Render DOKUMENTATION_ENTWURF.md as a styled A4 HTML/PDF report.

The script intentionally uses only local assets and the Python-Markdown
package. Chromium/Chrome performs the final PDF rendering.
"""

from __future__ import annotations

import argparse
import html
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import markdown


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "DOKUMENTATION_ENTWURF.md"
DEFAULT_HTML = ROOT / "DOKUMENTATION_ENTWURF.html"
DEFAULT_PDF = ROOT / "DOKUMENTATION_ENTWURF.pdf"


CSS = r"""
@page {
  size: A4;
  margin: 19mm 18mm 21mm 18mm;

  @top-left {
    content: "MULTISENSOR-LIDAR-OBJEKTERKENNUNG";
    color: #6c7b86;
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 7.5pt;
    font-weight: 650;
    letter-spacing: 0.08em;
  }

  @top-right {
    content: "PROJEKTDOKUMENTATION · ENTWURF 0.1";
    color: #8a969e;
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 7.5pt;
    letter-spacing: 0.04em;
  }

  @bottom-left {
    content: "DCAITI · 16. JULI 2026";
    color: #8a969e;
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 7.5pt;
    letter-spacing: 0.05em;
  }

  @bottom-right {
    content: counter(page);
    color: #1e607b;
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 8pt;
    font-weight: 700;
  }
}

@page cover {
  margin: 0;

  @top-left { content: none; }
  @top-right { content: none; }
  @bottom-left { content: none; }
  @bottom-right { content: none; }
}

:root {
  --ink: #172733;
  --muted: #60727e;
  --hair: #d9e2e7;
  --soft: #eef4f6;
  --navy: #123247;
  --blue: #1e607b;
  --teal: #16847f;
  --cyan: #58b9bf;
  --amber: #d99032;
  --paper: #ffffff;
}

* {
  box-sizing: border-box;
}

html {
  font-size: 10.4pt;
}

body {
  margin: 0;
  color: var(--ink);
  background: var(--paper);
  font-family: "Segoe UI", "Aptos", Arial, sans-serif;
  font-variant-numeric: tabular-nums;
  line-height: 1.52;
  hyphens: auto;
  -webkit-print-color-adjust: exact;
  print-color-adjust: exact;
}

.cover {
  page: cover;
  position: relative;
  width: 210mm;
  height: 297mm;
  overflow: hidden;
  color: white;
  background:
    radial-gradient(circle at 88% 15%, rgba(88,185,191,.27) 0 13mm, transparent 13.3mm),
    radial-gradient(circle at 88% 15%, transparent 0 24mm, rgba(88,185,191,.13) 24.2mm 24.7mm, transparent 25mm),
    linear-gradient(142deg, #0e2b3e 0%, #174c62 58%, #167d7b 100%);
  break-after: page;
}

.cover::before {
  content: "";
  position: absolute;
  left: 0;
  top: 0;
  width: 10mm;
  height: 100%;
  background: linear-gradient(180deg, var(--cyan), var(--teal) 58%, var(--amber));
}

.cover::after {
  content: "";
  position: absolute;
  right: -27mm;
  bottom: -39mm;
  width: 125mm;
  height: 125mm;
  border: 1px solid rgba(255,255,255,.18);
  transform: rotate(18deg);
}

.cover-inner {
  position: absolute;
  inset: 0;
  padding: 35mm 24mm 24mm 31mm;
  display: flex;
  flex-direction: column;
}

.cover-kicker {
  color: #9de2e0;
  font-size: 10pt;
  font-weight: 700;
  letter-spacing: .15em;
  text-transform: uppercase;
}

.cover-rule {
  width: 33mm;
  height: 1.5mm;
  margin: 9mm 0 14mm;
  background: var(--cyan);
}

.cover h1 {
  max-width: 145mm;
  margin: 0;
  color: white;
  font-size: 31pt;
  font-weight: 720;
  letter-spacing: -.035em;
  line-height: 1.08;
}

.cover-subtitle {
  max-width: 125mm;
  margin-top: 9mm;
  color: rgba(255,255,255,.82);
  font-size: 15pt;
  font-weight: 350;
  line-height: 1.35;
}

.cover-summary {
  max-width: 132mm;
  margin-top: 23mm;
  padding-left: 6mm;
  border-left: 1.2mm solid rgba(88,185,191,.86);
  color: rgba(255,255,255,.78);
  font-size: 10.5pt;
  line-height: 1.55;
}

.cover-meta {
  margin-top: auto;
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 4mm 14mm;
  max-width: 138mm;
  padding-top: 7mm;
  border-top: .25mm solid rgba(255,255,255,.26);
}

.meta-label {
  display: block;
  margin-bottom: 1.2mm;
  color: #9de2e0;
  font-size: 7.2pt;
  font-weight: 700;
  letter-spacing: .12em;
  text-transform: uppercase;
}

.meta-value {
  color: rgba(255,255,255,.9);
  font-size: 9.5pt;
}

.toc-page {
  break-after: page;
}

.toc-title {
  margin: 0 0 8mm;
  color: var(--navy);
  font-size: 23pt;
  line-height: 1.15;
  letter-spacing: -.025em;
}

.toc-title::after {
  content: "";
  display: block;
  width: 25mm;
  height: 1.1mm;
  margin-top: 5mm;
  background: var(--teal);
}

.toc ul {
  margin: 0;
  padding: 0;
  list-style: none;
}

.toc > ul > li {
  margin: 0;
  padding: 2.2mm 0;
  border-bottom: .22mm solid var(--hair);
}

.toc > ul > li > a {
  color: var(--navy);
  font-size: 10.3pt;
  font-weight: 700;
  text-decoration: none;
}

.toc ul ul {
  margin-top: 1.5mm;
  padding-left: 6mm;
}

.toc ul ul li {
  display: inline;
  color: var(--muted);
  font-size: 8.5pt;
}

.toc ul ul li:not(:last-child)::after {
  content: "  ·  ";
  color: #a6b2b9;
}

.toc ul ul a {
  color: var(--muted);
  text-decoration: none;
}

main > h2 {
  margin: 0 0 7mm;
  padding-top: 4mm;
  color: var(--navy);
  border-top: 1.2mm solid var(--teal);
  font-size: 22pt;
  font-weight: 720;
  letter-spacing: -.025em;
  line-height: 1.15;
  break-before: page;
  break-after: avoid;
}

main > h2:first-child {
  break-before: auto;
}

h3 {
  margin: 8mm 0 3mm;
  color: var(--blue);
  font-size: 14.5pt;
  font-weight: 700;
  line-height: 1.25;
  break-after: avoid;
}

h4 {
  margin: 6mm 0 2mm;
  color: var(--teal);
  font-size: 11.5pt;
  font-weight: 700;
  break-after: avoid;
}

p {
  margin: 0 0 3.2mm;
  orphans: 3;
  widows: 3;
}

strong {
  color: #102c3e;
  font-weight: 700;
}

a {
  color: var(--blue);
  text-decoration-color: rgba(30,96,123,.35);
  text-underline-offset: 1.5px;
}

ul, ol {
  margin: 2mm 0 4mm;
  padding-left: 6.5mm;
}

li {
  margin: 1mm 0;
  padding-left: 1mm;
}

li::marker {
  color: var(--teal);
  font-weight: 700;
}

blockquote {
  margin: 5mm 0;
  padding: 4mm 5mm 4mm 6mm;
  color: #28414f;
  background: linear-gradient(90deg, #edf6f6, #f7fafb);
  border-left: 1.2mm solid var(--teal);
  border-radius: 0 2mm 2mm 0;
  break-inside: avoid;
}

blockquote p:last-child {
  margin-bottom: 0;
}

code {
  padding: .15em .38em;
  color: #16485f;
  background: #edf3f5;
  border-radius: 1mm;
  font-family: "Cascadia Mono", Consolas, monospace;
  font-size: .87em;
  overflow-wrap: anywhere;
}

pre {
  margin: 4mm 0 5mm;
  padding: 4mm 5mm;
  color: #eaf2f4;
  background: #173343;
  border-left: 1mm solid var(--cyan);
  border-radius: 1.5mm;
  font-family: "Cascadia Mono", Consolas, monospace;
  font-size: 8.3pt;
  line-height: 1.45;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  break-inside: avoid;
}

pre code {
  padding: 0;
  color: inherit;
  background: transparent;
  font-size: inherit;
}

.table-wrap {
  margin: 4mm 0 5.5mm;
  break-inside: avoid;
}

table {
  width: 100%;
  border-collapse: collapse;
  color: #223642;
  font-size: 8.6pt;
  line-height: 1.34;
}

thead {
  display: table-header-group;
}

tr {
  break-inside: avoid;
}

th {
  padding: 2.5mm 2.2mm;
  color: white;
  background: var(--navy);
  border: .2mm solid var(--navy);
  font-weight: 700;
  text-align: left;
  vertical-align: bottom;
}

td {
  padding: 2.1mm 2.2mm;
  border: .2mm solid #d6e0e5;
  vertical-align: top;
}

tbody tr:nth-child(even) td {
  background: #f2f6f7;
}

tbody tr:first-child td {
  border-top-color: #b9cbd3;
}

hr {
  margin: 8mm 0;
  border: 0;
  border-top: .25mm solid var(--hair);
}

.lead {
  margin-bottom: 7mm;
  color: #3d5664;
  font-size: 11.2pt;
  line-height: 1.58;
}

.source-note {
  margin-top: 10mm;
  padding-top: 3mm;
  color: #84919a;
  border-top: .25mm solid var(--hair);
  font-size: 7.8pt;
}

@media screen {
  body {
    max-width: 210mm;
    margin: 0 auto;
    box-shadow: 0 0 24px rgba(15,38,52,.18);
  }

  .toc-page, main {
    padding: 19mm 18mm 21mm;
  }
}

@media print {
  .toc-page, main {
    padding: 0;
  }
}
"""


def find_chrome(explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit)
        if path.is_file():
            return path
        raise FileNotFoundError(f"Chrome executable not found: {path}")

    candidates = [
        shutil.which("chrome"),
        shutil.which("msedge"),
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    raise FileNotFoundError("Neither Chrome nor Edge could be found")


def content_markdown(source: str) -> str:
    """Drop the Markdown-only title block; the PDF has a custom cover."""
    marker = "## Kurzfassung"
    pos = source.find(marker)
    if pos < 0:
        raise ValueError(f"Expected heading '{marker}' in source document")
    return source[pos:]


def render_markdown(source: str) -> tuple[str, str]:
    md = markdown.Markdown(
        extensions=["extra", "sane_lists", "toc"],
        extension_configs={"toc": {"toc_depth": "2-3"}},
        output_format="html5",
    )
    body = md.convert(content_markdown(source))
    body = re.sub(r"<table>", '<div class="table-wrap"><table>', body)
    body = re.sub(r"</table>", "</table></div>", body)
    body = body.replace(
        "<h2 id=\"kurzfassung\">Kurzfassung</h2>",
        '<h2 id="kurzfassung">Kurzfassung</h2>',
    )
    return body, md.toc


def build_html(source_path: Path) -> str:
    source = source_path.read_text(encoding="utf-8")
    body, toc = render_markdown(source)
    source_name = html.escape(source_path.name)
    return f"""<!doctype html>
<html lang="de">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Multisensor-LiDAR-Objekterkennung – Projektdokumentation</title>
  <style>{CSS}</style>
</head>
<body>
  <section class="cover">
    <div class="cover-inner">
      <div class="cover-kicker">Technische und wissenschaftliche Dokumentation</div>
      <div class="cover-rule"></div>
      <h1>Multisensor-LiDAR-Objekterkennung in einer Tiefgarage</h1>
      <div class="cover-subtitle">Training, Cross-Validation und Bewertung von
        PointPillars und CenterPoint auf zwei Einzelsensoransichten und einer
        fusionierten Punktwolke</div>
      <div class="cover-summary">Experiment-held-out 3-Fold-Cross-Validation ·
        Person, Fahrrad und Auto · Vergleich von os0, os1 und merged ·
        Fehleranalyse und Laufzeiteinordnung</div>
      <div class="cover-meta">
        <div><span class="meta-label">Projekt</span><span class="meta-value">Multisensor LiDAR 3D Object Detection</span></div>
        <div><span class="meta-label">Dokumentstatus</span><span class="meta-value">Erster Entwurf · Version 0.1</span></div>
        <div><span class="meta-label">Stand</span><span class="meta-value">16. Juli 2026</span></div>
        <div><span class="meta-label">Hauptmodell</span><span class="meta-value">PointPillars · KITTI-Finetuning</span></div>
      </div>
    </div>
  </section>

  <section class="toc-page">
    <h1 class="toc-title">Inhaltsverzeichnis</h1>
    <nav class="toc">{toc}</nav>
  </section>

  <main>
    {body}
    <p class="source-note">Automatisch erzeugt aus <code>{source_name}</code>.
      Maßgebliche Ergebnisquellen sind die im Abschnitt „Reproduzierbarkeit und
      Projektartefakte“ aufgeführten Cross-Validation-Reports.</p>
  </main>
</body>
</html>
"""


def render_pdf(chrome: Path, html_path: Path, pdf_path: Path) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    if pdf_path.exists():
        pdf_path.unlink()

    with tempfile.TemporaryDirectory(prefix="lidar-doc-chrome-") as profile:
        command = [
            str(chrome),
            "--headless=new",
            "--disable-gpu",
            "--disable-extensions",
            "--disable-background-networking",
            "--no-first-run",
            "--no-default-browser-check",
            "--allow-file-access-from-files",
            "--no-pdf-header-footer",
            f"--user-data-dir={profile}",
            f"--print-to-pdf={pdf_path}",
            html_path.resolve().as_uri(),
        ]
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
    if completed.returncode != 0 or not pdf_path.is_file():
        details = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(
            f"Chrome PDF rendering failed ({completed.returncode}): {details}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--html", type=Path, default=DEFAULT_HTML)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--chrome", help="Explicit Chrome/Edge executable")
    parser.add_argument(
        "--html-only", action="store_true", help="Generate HTML without PDF"
    )
    args = parser.parse_args()

    source_path = args.source.resolve()
    html_path = args.html.resolve()
    pdf_path = args.pdf.resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)

    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(build_html(source_path), encoding="utf-8")
    print(f"HTML: {html_path}")

    if not args.html_only:
        chrome = find_chrome(args.chrome)
        render_pdf(chrome, html_path, pdf_path)
        print(f"PDF:  {pdf_path} ({pdf_path.stat().st_size / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
