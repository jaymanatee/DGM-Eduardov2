"""Phase 3 corpus: download every source of ``rag/sources.py`` and build ``rag/corpus/``.

Run::

    uv run python -m rag.download            # download what is missing, check, build
    uv run python -m rag.download --pin      # also record the fingerprints in the lock file

Each source is downloaded once to ``rag/downloads/`` (out of git) and checked against its
SHA-256 in ``rag/sources.lock.json`` (in git): if a file changes upstream, or a hand download
is not the one we used, the script says so instead of building a different corpus. Then:

* legal texts (XHTML/HTML) are converted to plain text and the sections listed in
  ``sources.py`` are cut out, one Markdown file each, with a header naming the source;
* guides (PDF) are copied whole; ``rag/ingest.py`` reads PDFs directly.

Sources marked ``manual`` are never requested: their site answers scripts with a bot check.
The script prints where to download them and the name to save them under, and builds the rest.
It exits with 1 while anything is missing, changed or failed, so a clean run means the corpus
is complete.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import time
from html.parser import HTMLParser
from pathlib import Path

import httpx

from rag.sources import SOURCES, Section, Source

RAG_DIR = Path(__file__).parent
DOWNLOADS_DIR = RAG_DIR / "downloads"
CORPUS_DIR = RAG_DIR / "corpus"
LOCK_FILE = RAG_DIR / "sources.lock.json"

USER_AGENT = "arca-rag-download/1.0"
ATTEMPTS = 3
PAUSE_SECONDS = 2.0  # between downloads, to be gentle with the servers


# ------------------------------------------------------------------------- HTML to text


class _TextExtractor(HTMLParser):
    """Paragraphs separated by blank lines; the cells of a table row joined on one line.

    EUR-Lex lays out numbered paragraphs as two-cell tables ("1." | text), so joining cells
    keeps each number next to its text.
    """

    BLOCKS = {
        "p", "div", "li", "table", "h1", "h2", "h3", "h4", "h5", "h6",
        "dt", "dd", "caption", "section", "article", "blockquote", "ul", "ol",
    }  # fmt: skip
    SKIPPED = {"script", "style", "head"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skipping = 0
        self.cells = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in self.SKIPPED:
            self.skipping += 1
        elif tag in ("td", "th"):
            self.cells += 1
        elif tag == "br":
            self.parts.append(" " if self.cells else "\n")

    def handle_startendtag(self, tag: str, attrs) -> None:
        if tag == "br":
            self.parts.append(" " if self.cells else "\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIPPED:
            self.skipping = max(0, self.skipping - 1)
        elif tag in ("td", "th"):
            self.cells = max(0, self.cells - 1)
            self.parts.append(" ")
        elif tag == "tr":
            self.parts.append("\n\n")
        elif tag in self.BLOCKS:
            self.parts.append(" " if self.cells else "\n\n")

    def handle_data(self, data: str) -> None:
        if not self.skipping:
            self.parts.append(data)


# Markers that consolidated texts put where an amendment starts or ends (▼M59, ▼B, ▲).
_CONSOLIDATION_MARK = re.compile(r"[▼►][A-Z]\d*|▲")
_SPACES = re.compile(r"[ \t\xa0    ]+")


def html_to_text(raw: str) -> str:
    parser = _TextExtractor()
    parser.feed(raw)
    parser.close()
    text = _CONSOLIDATION_MARK.sub(" ", "".join(parser.parts))
    lines = [_SPACES.sub(" ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


# ------------------------------------------------------------------------------ sections


class SectionNotFound(ValueError):
    """A ``start`` or ``end`` pattern of ``sources.py`` does not match the downloaded text."""


def cut_section(text: str, section: Section) -> str:
    """From the only match of ``start`` to the first match of ``end`` after it.

    A ``start`` that matches twice is an error: the first match is usually the table of
    contents, and cutting from there would drag in everything up to the real heading.
    """
    starts = list(re.finditer(section.start, text))
    if not starts:
        raise SectionNotFound(f"{section.name}: start pattern {section.start!r} not found")
    if len(starts) > 1:
        raise SectionNotFound(
            f"{section.name}: start pattern {section.start!r} matches {len(starts)} times; "
            "make it longer so that only the heading in the body matches"
        )
    start = starts[0]
    if section.end is None:
        return text[start.start() :].strip()
    end = re.compile(section.end).search(text, start.end())
    if end is None:
        raise SectionNotFound(f"{section.name}: end pattern {section.end!r} not found")
    return text[start.start() : end.start()].strip()


def section_document(source: Source, section: Section, body: str) -> str:
    """The Markdown file for one section: a header naming the source, then the text."""
    return (
        f"# {source.title}: {section.title}\n\n"
        f"Source: {source.reference}. Licence: {source.licence}.\n\n"
        f"{body}\n"
    )


def build(source: Source, raw_path: Path, corpus_dir: Path) -> list[Path]:
    """Turn one downloaded file into its corpus files; returns the paths written."""
    out_dir = corpus_dir / source.group
    out_dir.mkdir(parents=True, exist_ok=True)
    if not source.sections:
        target = out_dir / source.file_name
        shutil.copyfile(raw_path, target)
        return [target]
    text = html_to_text(raw_path.read_text(encoding="utf-8", errors="replace"))
    written = []
    for section in source.sections:
        target = out_dir / f"{section.name}.md"
        target.write_text(
            section_document(source, section, cut_section(text, section)), encoding="utf-8"
        )
        written.append(target)
    return written


# -------------------------------------------------------------------------- downloading


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(client: httpx.Client, url: str, target: Path) -> None:
    """Stream ``url`` to ``target``; retries time-outs, 429 and 5xx, raises on the last one."""
    partial = target.with_name(target.name + ".part")
    for attempt in range(1, ATTEMPTS + 1):
        try:
            with client.stream("GET", url) as response:
                transient = response.status_code == 429 or response.status_code >= 500
                if transient and attempt < ATTEMPTS:
                    time.sleep(5 * attempt)
                    continue
                response.raise_for_status()
                with partial.open("wb") as f:
                    for chunk in response.iter_bytes():
                        f.write(chunk)
            partial.replace(target)
            return
        except (httpx.TimeoutException, httpx.TransportError):
            partial.unlink(missing_ok=True)
            if attempt == ATTEMPTS:
                raise
            time.sleep(5 * attempt)


def load_lock(path: Path = LOCK_FILE) -> dict[str, dict]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save_lock(lock: dict[str, dict], path: Path = LOCK_FILE) -> None:
    path.write_text(json.dumps(dict(sorted(lock.items())), indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--pin",
        action="store_true",
        help="record the SHA-256 of every downloaded file in the lock file (after a deliberate "
        "update, or the first time)",
    )
    parser.add_argument("--only", nargs="+", metavar="NAME", help="only these sources")
    parser.add_argument("--downloads", type=Path, default=DOWNLOADS_DIR)
    parser.add_argument("--corpus", type=Path, default=CORPUS_DIR)
    parser.add_argument("--lock", type=Path, default=LOCK_FILE)
    args = parser.parse_args(argv)

    sources = [s for s in SOURCES if not args.only or s.name in args.only]
    unknown = set(args.only or ()) - {s.name for s in SOURCES}
    if unknown:
        parser.error(f"unknown sources: {', '.join(sorted(unknown))}")
    args.downloads.mkdir(parents=True, exist_ok=True)
    lock = load_lock(args.lock)
    missing: list[Source] = []
    problems: list[str] = []
    built: list[Path] = []

    with httpx.Client(
        follow_redirects=True,
        timeout=httpx.Timeout(180, connect=20),
        headers={"User-Agent": USER_AGENT},
    ) as client:
        for source in sources:
            raw = args.downloads / source.file_name
            if not raw.exists():
                if source.manual:
                    missing.append(source)
                    continue
                print(f"Downloading {source.name} ...", flush=True)
                try:
                    fetch(client, source.url, raw)
                except httpx.HTTPError as error:
                    problems.append(f"{source.name}: download failed ({error}); try again later")
                    continue
                time.sleep(PAUSE_SECONDS)

            digest = sha256(raw)
            pinned = lock.get(source.name, {}).get("sha256")
            if args.pin:
                lock[source.name] = {"file": source.file_name, "sha256": digest}
            elif pinned is None:
                problems.append(f"{source.name}: no fingerprint in the lock file (run with --pin)")
            elif pinned != digest:
                problems.append(
                    f"{source.name}: {raw} is not the file in the lock (sha256 {digest[:12]}..., "
                    f"expected {pinned[:12]}...). Delete it and download it again, or run with "
                    "--pin if the change is deliberate."
                )
                continue

            try:
                paths = build(source, raw, args.corpus)
            except SectionNotFound as error:
                problems.append(f"{source.name}: {error}")
                continue
            built.extend(paths)
            sizes = ", ".join(f"{p.name} ({p.stat().st_size // 1000} kB)" for p in paths)
            print(f"  {source.name}: {sizes}")

    if args.pin:
        save_lock(lock, args.lock)
        print(f"Fingerprints written to {args.lock}")

    # Files in the corpus that this script did not write (renamed sections, stray copies).
    expected = {p.resolve() for p in built}
    if not args.only:
        strays = [
            p
            for p in sorted(args.corpus.rglob("*"))
            if p.is_file() and p.name != ".gitkeep" and p.resolve() not in expected
        ]
        for p in strays:
            problems.append(f"{p} does not come from sources.py; delete it or add its source")

    print(f"\n{len(built)} corpus files from {len(sources) - len(missing)} sources.")
    if missing:
        print(f"\n{len(missing)} sources have to be downloaded by hand (the site blocks scripts):")
        for source in missing:
            print(f"  - {source.title}\n    {source.url}")
            print(f"    save as {args.downloads / source.file_name}")
        print("Then run the script again.")
    for problem in problems:
        print(f"! {problem}")
    return 1 if missing or problems else 0


if __name__ == "__main__":
    sys.exit(main())
