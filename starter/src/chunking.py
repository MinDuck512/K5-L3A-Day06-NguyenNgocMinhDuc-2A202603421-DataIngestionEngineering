"""Split source units with stable IDs and exact character provenance."""
import hashlib
import json
import math
import re
from collections import Counter
from statistics import mean, median
from typing import Any

from .models import Chunk, DocumentPage


def _validate_size(size: int, overlap: int = 0) -> None:
    if type(size) is not int or size <= 0:
        raise ValueError("Chunk size must be a positive integer")
    if type(overlap) is not int or not 0 <= overlap < size:
        raise ValueError("Overlap must be an integer with 0 <= overlap < size")


def _make_chunk(page: DocumentPage, start: int, end: int, method: str,
                size: int, overlap: int, boundary: str) -> Chunk:
    text = page.text[start:end]
    # Include parameters and content: repeated builds keep IDs, changed text gets new IDs.
    identity = json.dumps([page.document_id, page.source, page.page, method, size,
                           overlap, start, end, text], ensure_ascii=True, separators=(",", ":"))
    chunk_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    metadata = dict(page.metadata)
    document_start = metadata.get("source_char_start", 0)
    metadata.update({
        "unit_char_start": start,
        "unit_char_end": end,
        "document_char_start": document_start + start,
        "document_char_end": document_start + end,
        "chunking": method,
        "chunk_size": size,
        "overlap": overlap,
        "boundary": boundary,
    })
    return Chunk(chunk_id, page.document_id, page.source, page.page, text, metadata)


def fixed_size(page: DocumentPage, size: int, overlap: int) -> list[Chunk]:
    """Use character windows; stride is size - overlap and the last window is shorter."""
    _validate_size(size, overlap)
    chunks: list[Chunk] = []
    if not page.text.strip():
        return chunks
    start = 0
    while start < len(page.text):
        end = min(start + size, len(page.text))
        if page.text[start:end].strip():
            chunks.append(_make_chunk(page, start, end, "fixed_size", size, overlap, "fixed"))
        if end == len(page.text):
            break
        start += size - overlap
    return chunks


def _long_paragraph_spans(text: str, start: int, end: int, size: int):
    """Prefer lines, then sentences, then words; bound even an unbroken token."""
    while end - start > size:
        limit = start + size
        lower = start + max(1, size // 2)
        newline = text.rfind("\n", lower, limit)
        if newline >= 0:
            cut, boundary = newline + 1, "line"
        else:
            sentences = list(re.finditer(r"[.!?](?=\s)", text[lower:limit]))
            spaces = list(re.finditer(r"[ \t]+", text[lower:limit]))
            if sentences:
                cut, boundary = lower + sentences[-1].end(), "sentence"
            elif spaces:
                cut, boundary = lower + spaces[-1].end(), "word"
            else:
                cut, boundary = limit, "hard"
        yield start, cut, boundary
        start = cut
    if start < end:
        yield start, end, "oversized_remainder"


def by_paragraph(page: DocumentPage, max_size: int) -> list[Chunk]:
    """Group whole paragraphs, including intervening blank lines, up to max_size.

    An oversized paragraph is split at a line/sentence/word boundary when
    possible. No artificial overlap is added; code indentation stays intact.
    """
    _validate_size(max_size)
    text = page.text
    if not text.strip():
        return []
    paragraphs: list[tuple[int, int]] = []
    start = 0
    for separator in re.finditer(r"\n[ \t]*\n", text):
        if text[start:separator.start()].strip():
            paragraphs.append((start, separator.start()))
        start = separator.end()
    if text[start:].strip():
        paragraphs.append((start, len(text)))

    chunks: list[Chunk] = []
    pending: tuple[int, int] | None = None

    def flush() -> None:
        nonlocal pending
        if pending is not None:
            chunks.append(_make_chunk(page, *pending, "paragraph", max_size, 0, "paragraph"))
            pending = None

    for start, end in paragraphs:
        if end - start > max_size:
            flush()
            for part_start, part_end, boundary in _long_paragraph_spans(text, start, end, max_size):
                if text[part_start:part_end].strip():
                    chunks.append(_make_chunk(page, part_start, part_end, "paragraph", max_size, 0, boundary))
        elif pending is None:
            pending = (start, end)
        elif end - pending[0] <= max_size:
            pending = (pending[0], end)
        else:
            flush()
            pending = (start, end)
    flush()
    return chunks


def chunk_statistics(pages: list[DocumentPage], chunks: list[Chunk], max_size: int) -> dict[str, Any]:
    """Validate source coverage/provenance and summarize actual character lengths."""
    if not chunks:
        raise ValueError("Chunking produced no chunks")
    source_pages = {(page.document_id, page.page): page for page in pages}
    coverage = {key: bytearray(len(page.text)) for key, page in source_pages.items()}
    if len({chunk.chunk_id for chunk in chunks}) != len(chunks):
        raise ValueError("Duplicate chunk IDs")
    for chunk in chunks:
        key = (chunk.document_id, chunk.page)
        if key not in source_pages:
            raise ValueError("Chunk references an unknown source unit")
        page = source_pages[key]
        start, end = chunk.metadata["unit_char_start"], chunk.metadata["unit_char_end"]
        if not 0 <= start < end <= len(page.text) or page.text[start:end] != chunk.text:
            raise ValueError("Chunk text does not match its source offsets")
        if not chunk.text.strip() or not 0 < len(chunk.text) <= max_size:
            raise ValueError("Empty or oversized chunk")
        if chunk.source != page.source or any(chunk.metadata.get(key) != value for key, value in page.metadata.items()):
            raise ValueError("Chunk lost source metadata")
        origin = page.metadata.get("source_char_start", 0)
        if (chunk.metadata["document_char_start"], chunk.metadata["document_char_end"]) != (origin + start, origin + end):
            raise ValueError("Incorrect document-level offsets")
        coverage[key][start:end] = b"\x01" * (end - start)
    missing = sum(1 for key, page in source_pages.items()
                  for index, character in enumerate(page.text)
                  if not character.isspace() and not coverage[key][index])
    if missing:
        raise ValueError(f"Chunking dropped {missing} non-whitespace source characters")
    lengths = sorted(len(chunk.text) for chunk in chunks)
    input_characters = sum(len(page.text) for page in pages)
    return {
        "chunk_count": len(chunks),
        "length_unit": "Unicode characters",
        "min_characters": lengths[0],
        "max_characters": lengths[-1],
        "mean_characters": round(mean(lengths), 2),
        "median_characters": median(lengths),
        "p10_characters": lengths[max(0, math.ceil(len(lengths) * 0.1) - 1)],
        "p90_characters": lengths[max(0, math.ceil(len(lengths) * 0.9) - 1)],
        "percentile_method": "nearest rank",
        "input_characters": input_characters,
        "total_chunk_characters": sum(lengths),
        "character_expansion_ratio": round(sum(lengths) / input_characters, 4),
        "uncovered_non_whitespace_characters": missing,
        "boundary_counts": dict(Counter(chunk.metadata["boundary"] for chunk in chunks)),
    }
