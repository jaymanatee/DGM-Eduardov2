"""Data indexing, step 1: collect the documents of your knowledge base.

Drop your files in ``rag/corpus/`` (PDF, Markdown, plain text, HTML) and this module
turns them into ``Document`` objects with a stable id and a ``source`` you can cite.
Keep the corpus out of git if it is big or not yours to redistribute; leave a script or
a link in ``rag/corpus/README.md`` so the index can be rebuilt.

Run ``uv run python -m rag.ingest`` to see what gets loaded and how long each document is.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from rag.sources import corpus_index

CORPUS_DIR = Path(__file__).parent / "corpus"
# The header rag/download.py puts on top of every section it cuts out of a legal text.
_MARKDOWN_HEADER = re.compile(
    r"\A# (?P<title>[^\n]+)\n\nSource: (?P<reference>[^\n]+?)\. Licence: [^\n]*\n+"
)
TEXT_SUFFIXES = {".txt", ".md", ".markdown"}


@dataclass
class Document:
    id: str
    source: str
    text: str
    metadata: dict | None = None


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return clean_pdf_pages([page.extract_text() or "" for page in reader.pages])


# A line that starts a new paragraph: bullets, "1.", "(a)", "B.16.", "3.2.1", "Question:".
_PARAGRAPH_START = re.compile(
    r"^(?:[•▪●◦·\-–—]\s|\(?\d{1,3}[a-z]?[.)]\s|\(?[a-z]{1,4}\)\s|[A-Z]\.\d+(?:\.\d+)*\.?\s"
    r"|\d+(?:\.\d+)+\.?\s|(?:Question|Answer|Pregunta|Respuesta)\s*:)"
)
_TABLE_OF_CONTENTS = re.compile(r"(?:\.\s?){5,}\s*\d*\s*$")


def _shape(line: str) -> str:
    """A line with its numbers blanked: "March 2025 Page 7 of 212" -> "March # Page # of #"."""
    return re.sub(r"\d+", "#", re.sub(r"\s+", " ", line))


def clean_pdf_pages(pages: list[str]) -> str:
    """Rebuild paragraphs from the text pypdf extracts, page by page.

    pypdf gives one line per printed line, with the page header and footer on every page and
    the table of contents as dotted lines. This drops the lines that repeat (numbers aside) on
    at least half of the pages, bare page numbers and dotted contents lines, and joins the
    rest into paragraphs. A paragraph ends at a blank line, before a line that starts a list
    item or a numbered heading, and after a line clearly shorter than the text width (the last
    line of a paragraph, or a heading). Words split by a hyphen at the end of a line are
    joined again.
    """
    lines_by_page = [[line.strip() for line in page.split("\n")] for page in pages]
    on_pages = Counter(
        shape for page in lines_by_page for shape in {_shape(line) for line in page if line}
    )
    repeated = (
        {shape for shape, n in on_pages.items() if n >= len(pages) / 2}
        if len(pages) >= 3
        else set()
    )
    lines: list[str] = []
    for page in lines_by_page:
        for line in page:
            if not line:
                lines.append("")
            elif not (
                _shape(line) in repeated
                or re.fullmatch(r"\d{1,4}", line)
                or _TABLE_OF_CONTENTS.search(line)
            ):
                lines.append(re.sub(r"\s+", " ", line))

    widths = sorted(len(line) for line in lines if line)
    text_width = widths[int(0.9 * (len(widths) - 1))] if widths else 0
    paragraphs: list[str] = []
    current, previous = "", ""
    for line in lines:
        if current and (
            not line or _PARAGRAPH_START.match(line) or len(previous) < 0.6 * text_width
        ):
            paragraphs.append(current)
            current = ""
        if not line:
            continue
        if current.endswith("-") and line[:1].islower():
            current = current[:-1] + line
        else:
            current = f"{current} {line}" if current else line
        previous = line
    if current:
        paragraphs.append(current)
    return "\n\n".join(paragraphs)


def _read_html(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    raw = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", raw, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", raw)
    return re.sub(r"\s+", " ", text).strip()


def read_document(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return _read_pdf(path)
    if path.suffix.lower() in {".html", ".htm"}:
        return _read_html(path)
    return path.read_text(encoding="utf-8", errors="ignore")


def load_corpus(corpus_dir: Path = CORPUS_DIR) -> list[Document]:
    """Every supported file under ``corpus_dir``, recursively, as a ``Document``."""
    documents = []
    for path in sorted(corpus_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES | {
            ".pdf",
            ".html",
            ".htm",
        }:
            continue
        text = read_document(path).strip()
        if not text:
            continue
        relative = path.relative_to(corpus_dir).as_posix()
        metadata = {"chars": 0, "suffix": path.suffix.lower(), "title": path.stem}
        known = corpus_index().get(relative)
        if known:
            source, title = known
            metadata.update(title=title, reference=source.reference, language=source.language)
        header = _MARKDOWN_HEADER.match(text)
        if header:  # written by rag/download.py: keep it as metadata, not as text to chunk
            metadata.update(title=header["title"], reference=header["reference"])
            text = text[header.end() :].strip()
        metadata["chars"] = len(text)
        documents.append(
            Document(
                id=hashlib.sha1(relative.encode()).hexdigest()[:10],
                source=relative,
                text=text,
                metadata=metadata,
            )
        )
    return documents


if __name__ == "__main__":
    docs = load_corpus()
    if not docs:
        print(f"No documents found in {CORPUS_DIR}. Add your corpus there (see rag/README.md).")
    for doc in docs:
        print(f"{doc.id}  {doc.metadata['chars']:>8} chars  {doc.source}")
    print(f"{len(docs)} documents, {sum(d.metadata['chars'] for d in docs):,} characters in total")
