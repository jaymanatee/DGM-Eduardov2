"""Data indexing, step 2: chunking.

Three strategies, compared on the same gold set in ``rag/evaluate.py``:

* ``recursive_chunks``: the default of most RAG systems (LangChain's recursive splitter):
  paragraphs, then lines, then sentences, until every piece fits ``chunk_size``.
* ``semantic_chunks``: consecutive sentences stay together while their embeddings are close
  (the algorithm from class); a new chunk starts where the distance jumps.
* ``structural_chunks``: one chunk per unit the documents already mark: an article, a note,
  a CN code, a numbered heading or a question of a guide. Long units are split recursively,
  heading-only units are merged into the next one. This is what the proposal promised.

Every chunk keeps the offsets of its text in the document (``start``, ``end``), so the gold
set marks the evidence once, as a quote, and the relevant chunks of each strategy are the ones
that overlap it. ``add_context`` then puts the same header on the chunks of every strategy:
the document title and the section the chunk starts in. Agreements share most of their
wording, and the header is what tells "Article 3 of the TCA" from "Article 3 of the PEM".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from itertools import pairwise

from rag.ingest import Document


@dataclass
class Chunk:
    id: str
    doc_id: str
    source: str
    text: str
    index: int  # position of the chunk within its document
    start: int = -1  # offsets of the chunk in the document text (before any header)
    end: int = -1
    section: str = ""  # heading the chunk starts under, set by ``add_context``


def _chunk(doc: Document, index: int, start: int, end: int) -> Chunk:
    return Chunk(
        id=f"{doc.id}-{index:04d}",
        doc_id=doc.id,
        source=doc.source,
        text=doc.text[start:end],
        index=index,
        start=start,
        end=end,
    )


def _locate(text: str, pieces: list[str], offset: int = 0) -> list[tuple[int, int]]:
    """Offsets of ``pieces`` in ``text``, in order; pieces may overlap (chunk overlap)."""
    spans, cursor = [], 0
    for piece in pieces:
        start = text.find(piece, cursor)
        if start < 0:  # never happens with the splitters used here; keep the order anyway
            start = cursor
        spans.append((offset + start, offset + start + len(piece)))
        cursor = start + 1
    return spans


# -------------------------------------------------------------------------------- recursive


def _recursive_splitter(chunk_size: int, chunk_overlap: int, separators: tuple[str, ...]):
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    return RecursiveCharacterTextSplitter(
        separators=list(separators),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        keep_separator="end",  # "end of sentence." | "Next one", not "end of sentence" | ". Next"
    )


SEPARATORS = ("\n\n", "\n", ". ", " ", "")


def recursive_chunks(
    documents: list[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    separators: tuple[str, ...] = SEPARATORS,
) -> list[Chunk]:
    """Split by paragraphs, then lines, then sentences... until every chunk fits ``chunk_size``."""
    splitter = _recursive_splitter(chunk_size, chunk_overlap, separators)
    chunks = []
    for doc in documents:
        spans = _locate(doc.text, splitter.split_text(doc.text))
        chunks += [_chunk(doc, i, start, end) for i, (start, end) in enumerate(spans)]
    return chunks


# -------------------------------------------------------------------------------- sentences

# A full stop after these is not the end of a sentence: "e.g. the", "Art. 7", "No. 1".
_ABBREVIATIONS = {
    "e.g", "i.e", "no", "nos", "art", "arts", "p", "pp", "para", "paras", "cf", "ex", "vs",
    "approx", "reg", "dir", "op", "cit", "ibid", "fig", "sr", "dr", "mr", "mrs", "ms", "st",
    "etc", "núm", "pág", "apdo", "sra", "dña",
}  # fmt: skip
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])[\"”’)]?\s+|\n{2,}")


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Offsets of the sentences of ``text``.

    A sentence ends at a blank line, or at ``.``, ``!`` or ``?`` followed by a space, unless
    the word before the stop is a list marker ("1.", "3.2.", "b."), an abbreviation
    ("e.g.", "Art.") or the next word starts in lower case.
    """
    spans, start = [], 0
    for match in _SENTENCE_BREAK.finditer(text):
        if "\n\n" not in match.group():
            word = text[: match.start()].rsplit(None, 1)[-1].rstrip(".!?").lstrip("(").lower()
            following = text[match.end() : match.end() + 1]
            if (
                re.fullmatch(r"\d{1,3}(?:\.\d{1,3})*|[a-z]|[ivx]{1,4}", word)
                or word in _ABBREVIATIONS
                or following.islower()
            ):
                continue
        end = match.start() + len(match.group().rstrip())
        if text[start:end].strip():
            spans.append((start, end))
        start = match.end()
    if text[start:].strip():
        spans.append((start, len(text.rstrip())))
    # Trim the spaces each span may start or end with.
    trimmed = []
    for s, e in spans:
        piece = text[s:e]
        s += len(piece) - len(piece.lstrip())
        e -= len(piece) - len(piece.rstrip())
        if e > s:
            trimmed.append((s, e))
    return trimmed


def split_sentences(text: str) -> list[str]:
    """The sentences of ``text``; good enough to feed the semantic chunker."""
    return [text[s:e] for s, e in sentence_spans(text)]


# --------------------------------------------------------------------------------- semantic


def consecutive_distances(embeddings) -> list[float]:
    """Cosine distance 1 - e_i · e_{i+1} between consecutive normalised embeddings."""
    import numpy as np

    embeddings = np.asarray(embeddings)
    if len(embeddings) < 2:
        return []
    return (1.0 - np.sum(embeddings[:-1] * embeddings[1:], axis=1)).tolist()


def semantic_chunks(
    documents: list[Document],
    embed_fn,
    distance_threshold: float = 0.25,
    max_chunk_chars: int = 1500,
    min_chunk_chars: int = 0,
) -> list[Chunk]:
    """Group consecutive sentences while their embeddings stay close.

    ``embed_fn(list[str]) -> array of shape (n, d)`` with L2-normalised rows (see
    ``rag/embed.py``). The algorithm from the slides:

    1. Split the document into sentences s_1 … s_n.
    2. Embed them: e_1 … e_n.
    3. Start the first chunk with s_1.
    4. For each consecutive pair, compute the cosine distance 1 - e_i · e_{i+1}.
    5. If the distance is <= threshold, append s_{i+1} to the current chunk.
    6. Otherwise close the chunk and start a new one at s_{i+1}.
    7. Also close a chunk when it would exceed ``max_chunk_chars``.

    ``min_chunk_chars`` (0, off, in the slides) keeps a chunk open, whatever the distance,
    until it reaches that size. On this corpus headings, CN codes and list items are
    "sentences" of their own, and without it a third of the chunks are under 100 characters.

    A chunk is the document text from its first sentence to its last, so it keeps the
    original line breaks. A single sentence longer than ``max_chunk_chars`` stays whole.
    """
    chunks = []
    for doc in documents:
        spans = sentence_spans(doc.text)
        if not spans:
            continue
        distances = consecutive_distances(embed_fn([doc.text[s:e] for s, e in spans]))
        groups = [[spans[0]]]
        for span, distance in zip(spans[1:], distances, strict=True):
            too_long = span[1] - groups[-1][0][0] > max_chunk_chars
            too_short = groups[-1][-1][1] - groups[-1][0][0] < min_chunk_chars
            if (distance <= distance_threshold or too_short) and not too_long:
                groups[-1].append(span)
            else:
                groups.append([span])
        chunks += [_chunk(doc, i, g[0][0], g[-1][1]) for i, g in enumerate(groups)]
    return chunks


# ------------------------------------------------------------------------------- structure

# Paragraphs that open a unit of a legal text. "top" units group "unit" ones.
_TOP = re.compile(
    r"(?:CHAPTER|Chapter|TITLE|Title|SECTION|Section|SUB-SECTION|Sub-section|PART|Part"
    r"|ANNEX|Annex|APPENDIX|Appendix)\s+[\w.-]+"
)
_UNIT = re.compile(
    r"(?:ARTICLE|Article|NOTE|Note)\s+[\w.-]+"  # Article 3.5, Note 7
    r"|\d{4}(?: \d{2}){0,3}"  # CN heading or subheading: 8701, 8701 10 00
    r"|General"  # the general notes of a CN chapter
)
# Paragraphs that open a unit of a guide: "B.16. Duty drawback", "3.2.2.2 Mercosur
# exporters reference numbers", "ii. Change of tariff heading (CTH)", "Example B.1: ...",
# "19. Question: ...". Short, and not a sentence.
_GUIDE_HEADING = re.compile(
    r"(?:[A-Z]\.\d+(?:\.\d+)*\.?|\d+(?:\.\d+)*\.?|[ivx]{1,4}\.|[A-Z]\.)\s+[A-ZÁÉÍÓÚÑ¿].{2,100}"
    r"|(?:Example|Ejemplo)\s+[\w.]+\s*:.{0,100}"
    r"|Questions?\s+[IVX]+(?:\.\d+)*\..{0,100}"
)
_QUESTION = re.compile(r"\d{1,3}\.\s*(?:Question|Pregunta)\s*:")


@dataclass(frozen=True)
class Heading:
    start: int  # offset of the heading paragraph
    level: str  # "top" or "unit"
    label: str  # the heading and its title on one line: "Article 37 Objective"


def _lines(text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in re.finditer(r"[^\n]+", text)]


def _is_title(line: str) -> bool:
    """The short line under a legal heading: "Objective", "RULES OF ORIGIN"."""
    return (
        0 < len(line) <= 120
        and not line.endswith((".", ":", ";", ","))
        and not (_TOP.fullmatch(line) or _UNIT.fullmatch(line))
    )


def find_headings(text: str, numbered: bool = False) -> list[Heading]:
    """The headings of a document, in order.

    Legal headings ("Article 37", "ANNEX 3-A", "8701 10 00") are a line of their own, with
    their title on the next line. ``numbered`` also takes the numbered headings and the
    questions of a guide ("B.16. Duty drawback", "19. Question: ..."); in a legal text those
    patterns would catch every numbered paragraph, so they are only used for guides.
    """
    lines = _lines(text)
    headings = []
    for i, (start, end) in enumerate(lines):
        line = text[start:end].strip()
        if _TOP.fullmatch(line) or _UNIT.fullmatch(line):
            title = text[slice(*lines[i + 1])].strip() if i + 1 < len(lines) else ""
            # In a guide, "Annex II" on a line of its own is as likely to be a leftover of the
            # table of contents as a real part: guides get one level of headings only.
            level = "top" if _TOP.fullmatch(line) and not numbered else "unit"
            headings.append(
                Heading(start, level, _short(f"{line} {title}" if _is_title(title) else line))
            )
        elif not numbered:
            continue
        elif _QUESTION.match(line):
            headings.append(Heading(start, "unit", f"Question {line.split('.', 1)[0]}"))
        elif (
            _GUIDE_HEADING.fullmatch(line)
            and len(line) <= 110
            and not line.endswith((".", ",", ";", ":"))
        ):
            headings.append(Heading(start, "unit", _short(line)))
    return headings


def _short(label: str, limit: int = 80) -> str:
    return label if len(label) <= limit else label[: limit - 1].rstrip() + "…"


def headings_of(doc: Document) -> list[Heading]:
    """``find_headings`` with the guide patterns on for PDFs."""
    return find_headings(doc.text, numbered=(doc.metadata or {}).get("suffix") == ".pdf")


def section_at(headings: list[Heading], offset: int) -> str:
    """The heading path in force at ``offset``: "SECTION 1 RULES OF ORIGIN › Article 37"."""
    top, unit = "", ""
    for heading in headings:
        if heading.start > offset:
            break
        if heading.level == "top":
            top, unit = heading.label, ""
        else:
            unit = heading.label
    return " › ".join(part for part in (top, unit) if part)


def structural_chunks(
    documents: list[Document],
    max_chunk_chars: int = 1500,
    min_chunk_chars: int = 200,
    chunk_overlap: int = 100,
) -> list[Chunk]:
    """One chunk per article, note, CN code, numbered heading or question.

    The text from one heading to the next is a unit. Units shorter than ``min_chunk_chars``
    (a heading with nothing under it yet, like "CHAPTER 2 RULES OF ORIGIN") are merged into
    the next one; units longer than ``max_chunk_chars`` are split with the recursive splitter.
    A document without headings is split recursively as a whole.
    """
    splitter = _recursive_splitter(max_chunk_chars, chunk_overlap, SEPARATORS)
    chunks = []
    for doc in documents:
        bounds = sorted({0, *(h.start for h in headings_of(doc))}) + [len(doc.text)]
        units, pending = [], None
        for start, end in pairwise(bounds):
            start = start if pending is None else pending
            if end - start < min_chunk_chars and end < len(doc.text):
                pending = start
                continue
            units.append((start, end))
            pending = None
        spans = []
        for start, end in units:
            body = doc.text[start:end].rstrip()
            if len(body) <= max_chunk_chars:
                spans.append((start, start + len(body)))
            else:
                spans += _locate(body, splitter.split_text(body), offset=start)
        chunks += [_chunk(doc, i, s, e) for i, (s, e) in enumerate(spans)]
    return chunks


# --------------------------------------------------------------------------------- context


def add_context(chunks: list[Chunk], documents: list[Document]) -> list[Chunk]:
    """Prefix every chunk with its document title and the section it starts in.

    The same header for every strategy, so that comparing them measures where they cut and
    not what they say about the chunk. Offsets keep pointing at the document text.
    """
    by_id = {doc.id: doc for doc in documents}
    headings = {doc.id: headings_of(doc) for doc in documents}
    out = []
    for chunk in chunks:
        doc = by_id[chunk.doc_id]
        section = section_at(headings[doc.id], chunk.start)
        title = (doc.metadata or {}).get("title", doc.source)
        header = f"[{title}{' › ' + section if section else ''}]"
        out.append(replace(chunk, text=f"{header}\n{chunk.text}", section=section))
    return out


def describe(chunks: list[Chunk]) -> dict[str, float]:
    """Numbers for your chunking comparison table: lengths of the text taken from the
    document, without the ``add_context`` header (the same for every strategy)."""
    lengths = sorted(c.end - c.start if c.start >= 0 else len(c.text) for c in chunks)
    return {
        "n_chunks": len(chunks),
        "mean_chars": sum(lengths) / max(len(lengths), 1),
        "median_chars": lengths[len(lengths) // 2] if lengths else 0,
        "min_chars": lengths[0] if lengths else 0,
        "max_chars": lengths[-1] if lengths else 0,
    }
