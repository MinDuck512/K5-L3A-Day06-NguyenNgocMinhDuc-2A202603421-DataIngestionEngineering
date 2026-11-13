"""Read, normalize, and prepare the lab corpus while preserving source positions.

HTML preparation uses the standard library and never runs retrieval.
PDF extraction remains a starter TODO because this corpus uses HTML.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import Request, urlopen

from .models import DocumentPage


LAB_ROOT = Path(__file__).resolve().parents[2]
DATA = LAB_ROOT / "starter" / "data"
BASE_URL = "https://docs.python.org/3.14/"
LICENSE_URL = BASE_URL + "license.html"
CHAPTERS = ("introduction", "controlflow", "datastructures", "errors")
MIN_CHARS = 1500
TARGET_CHARS = 2400
LOCAL_TZ = timezone(timedelta(hours=7))


def load_pdf(path: Path, document_id: str, source: str) -> list[DocumentPage]:
    """TODO: extract physical PDF pages, preserve source, and record empty pages."""
    raise NotImplementedError("TODO: load_pdf; the selected corpus is HTML")


def normalize_text(text: str) -> str:
    """Normalize Unicode and whitespace while preserving code indentation."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return re.sub(r"\n(?:[ \t]*\n)+", "\n\n", text).strip()


def load_source_units(path: Path) -> list[DocumentPage]:
    """Load the frozen HTML source units without downloading or changing text."""
    pages: list[DocumentPage] = []
    seen: set[tuple[str, int]] = set()
    with path.open(encoding="utf-8") as source_file:
        for line_number, line in enumerate(source_file, start=1):
            if not line.strip():
                continue
            unit = json.loads(line)
            text = unit["text"]
            key = (unit["document_id"], unit["page"])
            if key in seen:
                raise ValueError(f"Duplicate source unit at line {line_number}: {key}")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Empty source text at line {line_number}")
            if unit["char_count"] != len(text) or unit["char_end"] - unit["char_start"] != len(text):
                raise ValueError(f"Invalid source offsets at line {line_number}")
            metadata = dict(unit["metadata"])
            metadata.update({
                "unit_id": unit["unit_id"],
                "position_type": unit["position_type"],
                "source_char_start": unit["char_start"],
                "source_char_end": unit["char_end"],
            })
            pages.append(DocumentPage(unit["document_id"], unit["source"], unit["page"], text, metadata))
            seen.add(key)
    if not pages:
        raise ValueError("No source units found")
    return pages


class MainTextParser(HTMLParser):
    """Keep Sphinx's main article, headings, paragraphs, and code indentation."""

    VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input",
            "link", "meta", "param", "source", "track", "wbr"}
    BLOCK = {"section", "div", "p", "li", "ul", "ol", "dl", "dt", "dd",
             "pre", "table", "tr", "blockquote", "h1", "h2", "h3", "h4",
             "h5", "h6"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.depth = 0
        self.main_depth: int | None = None
        self.main_seen = False
        self.ignore_depth: int | None = None
        self.pre_depth: int | None = None
        self.h1_depth: int | None = None
        self.in_title = False
        self.parts: list[str] = []
        self.heading_parts: list[str] = []
        self.title_parts: list[str] = []
        self.language: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "html":
            self.language = attributes.get("lang")
        if tag == "title":
            self.in_title = True
        if tag not in self.VOID:
            self.depth += 1
        if (not self.main_seen
                and (tag == "main" or attributes.get("role") == "main")):
            self.main_depth = self.depth
            self.main_seen = True
        if self.main_depth is None:
            return
        if tag in {"script", "style", "nav"} and self.ignore_depth is None:
            self.ignore_depth = self.depth
        if self.ignore_depth is not None:
            return
        if tag in self.BLOCK:
            self.parts.append("\n\n")
        if tag == "br":
            self.parts.append("\n")
        if tag == "pre":
            self.pre_depth = self.depth
        if tag == "h1":
            self.h1_depth = self.depth

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self.in_title = False
        if self.main_depth is not None:
            if self.ignore_depth is None and tag in self.BLOCK:
                self.parts.append("\n\n")
            if self.depth == self.h1_depth:
                self.h1_depth = None
            if self.depth == self.pre_depth:
                self.pre_depth = None
            if self.depth == self.ignore_depth:
                self.ignore_depth = None
            if self.depth == self.main_depth:
                self.main_depth = None
        if tag not in self.VOID:
            self.depth = max(0, self.depth - 1)

    def handle_data(self, data: str) -> None:
        if self.in_title:
            self.title_parts.append(data)
        if self.main_depth is None or self.ignore_depth is not None:
            return
        if data.strip() == "\u00b6":
            return
        if self.h1_depth is not None:
            self.heading_parts.append(data)
        if self.pre_depth is None:
            data = re.sub(r"\s+", " ", data)
        self.parts.append(data)

    def text(self) -> str:
        return normalize_text("".join(self.parts))


def fetch(url: str, destination: Path) -> tuple[bytes, str]:
    request = Request(url, headers={"User-Agent": "Day06-CourseLab/1.0"})
    with urlopen(request, timeout=60) as response:
        payload = response.read()
        if "text/html" not in response.headers.get("Content-Type", ""):
            raise RuntimeError(f"Expected HTML from {url}")
    destination.write_bytes(payload)
    accessed_at = datetime.now(LOCAL_TZ).isoformat(timespec="seconds")
    return payload, accessed_at


def source_boundaries(text: str) -> list[tuple[int, int]]:
    """Partition, without overlap, at paragraph boundaries where possible.

    These are source units used instead of PDF pages, not retrieval chunks.
    A short remainder is merged into the preceding unit.
    """
    bounds: list[tuple[int, int]] = []
    start = 0
    while start < len(text):
        while start < len(text) and text[start].isspace():
            start += 1
        if start == len(text):
            break
        end = min(start + TARGET_CHARS, len(text))
        if end < len(text):
            paragraph_end = text.rfind("\n\n", start + MIN_CHARS, end)
            if paragraph_end >= 0:
                end = paragraph_end
        while end > start and text[end - 1].isspace():
            end -= 1
        if end - start < MIN_CHARS:
            if not bounds:
                raise ValueError("Source has fewer than 1,500 useful characters")
            previous_start, _ = bounds.pop()
            bounds.append((previous_start, len(text)))
            break
        bounds.append((start, end))
        start = end
    return bounds


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prepare_corpus() -> None:
    raw_dir = DATA / "raw"
    processed_dir = DATA / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    sources: list[dict] = []
    all_units: list[dict] = []

    for slug in CHAPTERS:
        document_id = "python314-" + slug
        source_url = BASE_URL + "tutorial/" + slug + ".html"
        raw_path = raw_dir / (document_id + ".html")
        print(f"Downloading {source_url}", flush=True)
        payload, accessed_at = fetch(source_url, raw_path)
        parser = MainTextParser()
        parser.feed(payload.decode("utf-8"))
        parser.close()
        text = parser.text()
        title = "".join(parser.heading_parts).strip()
        page_title = "".join(parser.title_parts)
        version_match = re.search(r"Python\s+(\d+\.\d+\.\d+[^\s]*)\s+documentation", page_title)
        chapter_match = re.match(r"(\d+)\.", title)
        if not title or not version_match or not chapter_match or not parser.language:
            raise RuntimeError(f"Cannot verify title/version/chapter/language: {source_url}")
        metadata = {
            "chapter": int(chapter_match.group(1)),
            "chapter_title": title,
            "version": version_match.group(1),
            "language": parser.language,
        }
        processed_path = processed_dir / (document_id + ".txt")
        processed_path.write_text(text, encoding="utf-8")
        units: list[dict] = []
        for number, (start, end) in enumerate(source_boundaries(text), start=1):
            unit_id = f"{document_id}:unit-{number:03d}"
            unit = {
                "unit_id": unit_id,
                "document_id": document_id,
                "source": source_url,
                "page": number,
                "position_type": "normalized_source_unit_not_pdf_page",
                "char_start": start,
                "char_end": end,
                "char_count": end - start,
                "text": text[start:end],
                "metadata": dict(metadata),
            }
            assert unit["char_count"] >= MIN_CHARS
            assert text[start:end] == unit["text"]
            all_units.append(unit)
            units.append({key: unit[key] for key in
                          ("unit_id", "page", "char_start", "char_end", "char_count")})
        sources.append({
            "source_id": document_id,
            "title": title,
            "source_path_or_url": source_url,
            "local_source_path": raw_path.relative_to(LAB_ROOT).as_posix(),
            "normalized_text_path": processed_path.relative_to(LAB_ROOT).as_posix(),
            "license_or_permission": "PSF-2.0; code examples additionally available under 0BSD",
            "license_url": LICENSE_URL,
            "accessed_at": accessed_at,
            "snapshot_version": metadata["version"],
            "raw_sha256": hashlib.sha256(payload).hexdigest(),
            "normalized_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "physical_page_count": None,
            "normalized_char_count": len(text),
            "page_ids": [unit["unit_id"] for unit in units],
            "units": units,
            "metadata": metadata,
        })
        print(f"  {title}: {len(text):,} characters, {len(units)} source units", flush=True)

    print("Downloading the license snapshot", flush=True)
    license_payload, license_accessed_at = fetch(LICENSE_URL, raw_dir / "python-license.html")
    if len(all_units) < 30:
        raise RuntimeError(f"Corpus only has {len(all_units)} valid source units; at least 30 required")
    versions = sorted({source["snapshot_version"] for source in sources})
    if len(versions) != 1:
        raise RuntimeError(f"Mixed documentation versions: {versions}; download again")
    manifest = {
        "corpus_title": "Python basics: introduction, control flow, data structures, errors",
        "source_format": "HTML",
        "counting_rule": {
            "method": "non-overlapping units of normalized main-article text",
            "minimum_characters_per_unit": MIN_CHARS,
            "target_characters_per_unit": TARGET_CHARS,
            "short_remainder": "merged into preceding unit",
            "offsets": "zero-based Unicode character offsets, end exclusive",
            "excluded": "navigation, footer, scripts, styles, heading permalink markers",
            "note": "Unit numbers are traceable source positions, not physical PDF pages or retrieval chunks.",
        },
        "total_sources": len(sources),
        "total_valid_source_units": len(all_units),
        "total_normalized_characters": sum(source["normalized_char_count"] for source in sources),
        "snapshot_versions": versions,
        "license_snapshot": {
            "path": "starter/data/raw/python-license.html",
            "url": LICENSE_URL,
            "accessed_at": license_accessed_at,
            "sha256": hashlib.sha256(license_payload).hexdigest(),
        },
        "sources": sources,
    }
    write_json(DATA / "manifest.json", manifest)
    with (DATA / "source_units.jsonl").open("w", encoding="utf-8", newline="\n") as output:
        for unit in all_units:
            output.write(json.dumps(unit, ensure_ascii=False) + "\n")
    notice = (
        "# Corpus attribution\n\n"
        "Source: Python documentation, Python Software Foundation.\n\n"
        "Copyright 2001 Python Software Foundation; All Rights Reserved.\n\n"
        f"License and copyright terms: {LICENSE_URL}\n"
        "A complete snapshot is retained in raw/python-license.html.\n"
        "Documentation: PSF License Version 2; examples/recipes/code additionally available under 0BSD.\n\n"
        "## Changes made for the lab\n\n"
        "Selected four tutorial chapters; extracted main-article text; normalized Unicode NFC, "
        "line endings, and whitespace while retaining code indentation; partitioned into "
        "non-overlapping source units of at least 1,500 characters. Original HTML is retained.\n\n"
        "## Sources\n\n"
        + "\n".join(f"- {source['title']}: {source['source_path_or_url']}" for source in sources)
        + "\n"
    )
    (DATA / "ATTRIBUTION.md").write_text(notice, encoding="utf-8")
    print(f"Corpus ready: {len(sources)} documents / {len(all_units)} valid source units")
    print(f"Manifest: {DATA / 'manifest.json'}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-corpus", action="store_true", help="Download a fresh Python documentation snapshot")
    args = parser.parse_args()
    if not args.prepare_corpus:
        parser.error("Use --prepare-corpus only before freezing the benchmark")
    if (DATA / "benchmark.json").exists():
        parser.error("Benchmark already exists; keep its frozen corpus instead of downloading a new snapshot")
    prepare_corpus()
