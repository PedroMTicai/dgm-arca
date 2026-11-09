"""Reproducible ingestion of the phase-3 corpus: pharmacovigilance documents.

The corpus is what a pharmacovigilance technician has open while coding a report:

* **Summaries of product characteristics (fichas técnicas)** from CIMA, the public drug
  database of the Spanish agency AEMPS, for every drug that appears in our phase-1 dataset
  (``rlm.generate_meddra.INDICATIONS``). We keep section 4 ("Datos clínicos": indications,
  posology, contraindications, warnings, interactions, pregnancy, driving, adverse reactions,
  overdose), downloaded section by section through the CIMA REST API and written as Markdown.
* **EMA / ICH pharmacovigilance guidelines** (PDF): GVP Module VI (collection and reporting of
  suspected adverse reactions), GVP Module IX (signal management) and GVP Annex I
  Rev 5 (definitions). ICH E2D(R1) was left out: the EMA answers 429 to every request
  for that PDF, so it cannot be downloaded reproducibly.

Optional, added by hand: *MedDRA Term Selection: Points to Consider* (the MSSO website only
serves it through a JavaScript application, so it cannot be scripted). Drop the PDF in
``rag/corpus/manual/`` and ``rag.ingest`` will pick it up.

Licences are documented in ``rag/CORPUS.md``. Nothing downloaded here is committed:
``rag/corpus/`` is in ``.gitignore`` and this script rebuilds it.

Run::

    uv run python -m rag.download_corpus            # skips files already present
    uv run python -m rag.download_corpus --force    # downloads everything again
"""

from __future__ import annotations

import argparse
import html
import json
import re
import time
import unicodedata
from datetime import date
from pathlib import Path

import requests

from rag.ingest import CORPUS_DIR

CIMA_API = "https://cima.aemps.es/cima/rest"
CIMA_SECTIONS = ("4.1", "4.2", "4.3", "4.4", "4.5", "4.6", "4.7", "4.8", "4.9")
EMA = "https://www.ema.europa.eu/en/documents"
GUIDELINES = {
    "gvp_module_vi_adr_reporting.pdf": f"{EMA}/regulatory-procedural-guideline/"
    "guideline-good-pharmacovigilance-practices-gvp-module-vi-collection-management-submission-"
    "reports-suspected-adverse-reactions-medicinal-products-rev-2_en.pdf",
    "gvp_module_ix_signal_management.pdf": f"{EMA}/scientific-guideline/"
    "guideline-good-pharmacovigilance-practices-gvp-module-ix-signal-management-rev-1_en.pdf",
    "gvp_annex_i_definitions.pdf": "https://www.ema.europa.eu/system/files/documents/"
    "scientific-guideline/13-gvp-annex-i-definitions-rev-5-final-published_en_0_0.pdf",
}
HEADERS = {"User-Agent": "dgm-arca corpus builder (academic project)"}


def dataset_drugs() -> list[str]:
    """Every drug mentioned as an indication treatment in the phase-1 generator."""
    from rlm.generate_meddra import INDICATIONS

    return sorted({drug for _, _, drugs in INDICATIONS for drug in drugs})


def slugify(text: str) -> str:
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", ascii_text.lower()).strip("_")


def get(url: str, retries: int = 6, accept: str | None = None, **params) -> requests.Response:
    """GET with exponential back-off: the EMA site answers 429 when hit too fast."""
    headers = {**HEADERS, "Accept": accept} if accept else HEADERS
    wait = 5.0
    for attempt in range(retries):
        response = requests.get(url, params=params or None, headers=headers, timeout=60)
        if response.status_code not in (429, 500, 502, 503, 504):
            response.raise_for_status()
            return response
        if attempt < retries - 1:
            print(f"  {response.status_code} from {url[:80]}..., retrying in {wait:.0f}s")
            time.sleep(wait)
            wait *= 2
    response.raise_for_status()
    return response


# Only real tags: "<" followed by a letter or "/". The sheets contain literal "<1/10" in the
# frequency headers, which a naive "<[^>]+>" would swallow together with the next cells.
_TAG = re.compile(r"</?[a-zA-Z][^>]*>")
# Inline formatting tags are removed without a space: Word-exported HTML splits words across
# them ("frecuente<span>mente</span>"), and a space there would break the word in two.
_INLINE_TAG = re.compile(r"</?(span|em|strong|b|i|u|sup|sub|font|a)(\s[^>]*)?>", re.I)
_TABLE = re.compile(r"<table[^>]*>(?P<body>.*?)</table>", re.DOTALL | re.I)
_ROW = re.compile(r"<tr[^>]*>(?P<body>.*?)</tr>", re.DOTALL | re.I)
_CELL = re.compile(r"<t[dh][^>]*>(?P<body>.*?)</t[dh]>", re.DOTALL | re.I)


def _inline(fragment: str) -> str:
    """Text of a fragment on one line (used for table cells)."""
    fragment = _INLINE_TAG.sub("", fragment)
    return re.sub(r"\s+", " ", html.unescape(_TAG.sub(" ", fragment))).strip()


def _short_header(header: str) -> str:
    """'Frecuentes (≥1/100 a <1/10)' -> 'Frecuentes'."""
    return re.sub(r"\s*\(.*", "", header).strip() or header


def table_to_text(table_html: str) -> str:
    """One line per row. Rows of the header's width become ``first cell: header: value; ...``.

    Section 4.8 tables have one row per organ class and one column per frequency. Writing
    each row as "Trastornos del sistema nervioso: Muy frecuentes: mareo, cefalea, somnolencia;
    Frecuentes: temblor..." keeps the frequency next to each reaction, so a chunk answers
    "¿con qué frecuencia produce somnolencia?" on its own, whatever the chunk boundaries.
    """
    rows = [
        [_inline(c.group("body")) for c in _CELL.finditer(r.group("body"))]
        for r in _ROW.finditer(table_html)
    ]
    rows = [row for row in rows if any(row)]
    width = max((len(r) for r in rows), default=0)
    header = next((r for r in rows if len(r) == width and width > 1), None)
    lines = []
    for row in rows:
        if header is not None and row is not header and len(row) == width and row[0]:
            values = [
                f"{_short_header(h)}: {v}" for h, v in zip(header[1:], row[1:], strict=True) if v
            ]
            lines.append(f"{row[0]}: " + "; ".join(values) if values else row[0])
        else:
            lines.append(" | ".join(cell for cell in row if cell))
    return "\n".join(lines)


def html_to_text(fragment: str) -> str:
    """CIMA section HTML to plain text that keeps paragraphs and table rows.

    ``rag.ingest`` would flatten HTML into one line, which leaves the recursive splitter
    without paragraph boundaries. Here every ``<p>``/``<li>`` becomes a paragraph and every
    table is rewritten by ``table_to_text``.
    """
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", fragment, flags=re.DOTALL | re.I)
    text = _INLINE_TAG.sub("", text)
    text = _TABLE.sub(lambda m: "\n\n" + table_to_text(m.group("body")) + "\n\n", text)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"</(p|li|h\d|div)\s*>", "\n\n", text, flags=re.I)
    text = re.sub(r"<li[^>]*>", "- ", text, flags=re.I)
    text = html.unescape(_TAG.sub(" ", text))
    lines = [re.sub(r"[ \t ]+", " ", line).strip() for line in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def pick_product(drug: str) -> dict | None:
    """A product whose active ingredient is exactly ``drug`` and has a sectioned technical
    sheet (``secc``), preferring marketed ones. Generics share the reference product's text;
    some ingredients (empagliflozina) only have sectioned sheets for non-marketed generics."""
    results = get(f"{CIMA_API}/medicamentos", nombre=drug).json().get("resultados", [])
    target = slugify(drug)

    def has_sheet(product: dict) -> bool:
        return any(d.get("tipo") == 1 and d.get("secc") for d in product.get("docs", []))

    def is_exact(product: dict) -> bool:
        return slugify((product.get("vtm") or {}).get("nombre", "")) == target

    with_sheet = [p for p in results if has_sheet(p)]
    for group in (
        [p for p in with_sheet if p.get("comerc") and is_exact(p)],
        [p for p in with_sheet if is_exact(p)],
        [p for p in with_sheet if p.get("comerc")],
        with_sheet,
    ):
        if group:
            return group[0]
    return None


def download_sheet(drug: str, out_dir: Path) -> dict | None:
    product = pick_product(drug)
    if product is None:
        print(f"  ! no sectioned technical sheet found for {drug}")
        return None
    nregistro = product["nregistro"]
    titles = {
        s["seccion"]: s["titulo"]
        for s in get(f"{CIMA_API}/docSegmentado/secciones/1", nregistro=nregistro).json()
    }
    parts = [
        f"# Ficha técnica de {drug}: {product['nombre']}",
        f"Fuente: AEMPS, CIMA (nregistro {nregistro}). "
        f"https://cima.aemps.es/cima/dochtml/ft/{nregistro}/FT_{nregistro}.html",
    ]
    for section in CIMA_SECTIONS:
        if section not in titles:
            continue
        # Without an explicit Accept header CIMA may answer text/plain, which drops the tables.
        body = get(
            f"{CIMA_API}/docSegmentado/contenido/1",
            accept="text/html",
            nregistro=nregistro,
            seccion=section,
        ).text
        text = html_to_text(body)
        if text:
            parts.append(f"## {section}. {titles[section]} ({drug})\n\n{text}")
        time.sleep(0.2)
    path = out_dir / f"{slugify(drug)}.md"
    path.write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    return {
        "file": str(path.relative_to(CORPUS_DIR)).replace("\\", "/"),
        "drug": drug,
        "product": product["nombre"],
        "nregistro": nregistro,
        "url": f"https://cima.aemps.es/cima/dochtml/ft/{nregistro}/FT_{nregistro}.html",
        "licence": "AEMPS public information (CIMA); reuse with attribution",
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--force", action="store_true", help="download files already present")
    parser.add_argument("--skip-guidelines", action="store_true")
    args = parser.parse_args(argv)

    sheets_dir = CORPUS_DIR / "fichas"
    guides_dir = CORPUS_DIR / "guidelines"
    sheets_dir.mkdir(parents=True, exist_ok=True)
    guides_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = CORPUS_DIR / "MANIFEST.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    )
    entries = {e["file"]: e for e in manifest.get("documents", [])}

    for drug in dataset_drugs():
        path = sheets_dir / f"{slugify(drug)}.md"
        if path.exists() and not args.force:
            print(f"= {path.name} (already there)")
            continue
        print(f"↓ {drug}")
        entry = download_sheet(drug, sheets_dir)
        if entry:
            entries[entry["file"]] = entry

    if not args.skip_guidelines:
        for name, url in GUIDELINES.items():
            path = guides_dir / name
            if path.exists() and not args.force:
                print(f"= {name} (already there)")
                continue
            print(f"↓ {name}")
            try:
                path.write_bytes(get(url).content)
            except requests.HTTPError as exc:
                print(f"  ! could not download {name}: {exc}")
                continue
            entries[f"guidelines/{name}"] = {
                "file": f"guidelines/{name}",
                "url": url,
                "licence": "EMA content, reproduction authorised with acknowledgement of source",
            }
            time.sleep(3)

    manifest = {
        "downloaded": date.today().isoformat(),
        "documents": sorted(entries.values(), key=lambda e: e["file"]),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{len(entries)} documents listed in {manifest_path}")


if __name__ == "__main__":
    main()
