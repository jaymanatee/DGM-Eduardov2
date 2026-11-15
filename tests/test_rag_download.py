"""The corpus downloader, without network: sources on disk, manual ones missing."""

import hashlib
import json
import re

import pytest

from rag import download
from rag.download import SectionNotFound, cut_section, html_to_text
from rag.sources import SOURCES, Section, Source


def test_html_to_text_keeps_paragraphs_and_joins_table_cells():
    raw = """<?xml version="1.0"?><html><head><title>EUR-Lex</title>
    <style>p {color: red}</style></head><body>
    <p class="title">CHAPTER 2</p><p>RULES OF ORIGIN</p>
    <table><tr><td><p>1.</p></td><td><p>This Annex&nbsp;sets out<br/>the rules.</p></td></tr>
    <tr><td><p>2.</p></td><td><p>▼M59 Amended text ▲</p></td></tr></table>
    <script>var x = 1;</script></body></html>"""
    text = html_to_text(raw)
    assert text == (
        "CHAPTER 2\n\nRULES OF ORIGIN\n\n1. This Annex sets out the rules.\n\n2. Amended text"
    )


def test_html_to_text_glues_lone_list_markers_to_their_text():
    raw = (
        "<div><p>1.</p></div><div><p>Products shall be originating if:</p></div>"
        "<div><p>(a)</p></div><div><p>wholly obtained; or</p></div>"
        "<div><p>(ii)</p></div><div><p>sufficiently worked.</p></div>"
        "<p>Article 3.5</p><p>Tolerances</p>"
    )
    assert html_to_text(raw) == (
        "1. Products shall be originating if:\n\n(a) wholly obtained; or\n\n"
        "(ii) sufficiently worked.\n\nArticle 3.5\n\nTolerances"
    )


def test_cut_section_from_start_to_the_first_end_after_it():
    text = "Contents: ANNEX 2 Notes\n\nANNEX 2\n\nNOTES\n\nNote 1 body.\n\nANNEX 3\n\nTABLE"
    section = Section("notes", "Notes", r"ANNEX 2\s+NOTES", r"ANNEX 3\s+TABLE")
    assert cut_section(text, section) == "ANNEX 2\n\nNOTES\n\nNote 1 body."
    assert cut_section(text, Section("tail", "Tail", r"ANNEX 3", None)) == "ANNEX 3\n\nTABLE"


@pytest.mark.parametrize(
    "start, end, message",
    [
        (r"ANNEX 9", None, "start pattern .* not found"),
        (r"ANNEX 2\s+NOTES", r"ANNEX 9", "end pattern .* not found"),
        # The table of contents repeats the heading: cutting from there would be wrong.
        (r"ANNEX 2", r"ANNEX 3", "matches 2 times"),
    ],
)
def test_cut_section_refuses_missing_or_ambiguous_patterns(start, end, message):
    text = "Contents: ANNEX 2 Notes\n\nANNEX 2\n\nNOTES\n\nbody\n\nANNEX 3"
    with pytest.raises(SectionNotFound, match=message):
        cut_section(text, Section("s", "S", start, end))


def test_every_source_is_well_formed():
    names = [s.name for s in SOURCES]
    assert len(names) == len(set(names))
    outputs = [(s.group, sec.name) for s in SOURCES for sec in s.sections]
    outputs += [(s.group, s.name) for s in SOURCES if not s.sections]
    assert len(outputs) == len(set(outputs)), "two sources would write the same corpus file"
    for source in SOURCES:
        assert source.suffix in {".xhtml", ".html", ".pdf"}
        assert source.licence and source.reference
        assert not (source.manual and source.sections), "hand downloads are kept whole"
        for section in source.sections:
            re.compile(section.start)
            if section.end:
                re.compile(section.end)


def test_every_automatic_source_is_pinned_in_the_lock():
    lock = download.load_lock()
    for source in SOURCES:
        if not source.manual:
            assert re.fullmatch(r"[0-9a-f]{64}", lock[source.name]["sha256"]), source.name


# ------------------------------------------------------------------------------- main()

HTML = "<html><body><p>ANNEX 1</p><p>NOTES</p><p>Body text.</p><p>ANNEX 2</p></body></html>"
LEGAL = Source(
    name="legal",
    title="An agreement",
    group="agreement",
    suffix=".html",
    url="http://example.invalid/legal",
    reference="CELEX 0",
    licence="test",
    sections=(Section("annex1", "Annex 1", r"ANNEX 1\s+NOTES", r"ANNEX 2"),),
)
GUIDE = Source(
    name="guide",
    title="A guide",
    group="guides",
    suffix=".pdf",
    url="http://example.invalid/guide.pdf",
    reference="Publisher",
    licence="test",
    manual=True,
)


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(download, "SOURCES", (LEGAL, GUIDE))
    monkeypatch.setattr(download, "fetch", _no_network)
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    (downloads / "legal.html").write_text(HTML, encoding="utf-8")
    argv = [
        "--downloads", str(downloads),
        "--corpus", str(tmp_path / "corpus"),
        "--lock", str(tmp_path / "lock.json"),
    ]  # fmt: skip
    return tmp_path, argv


def _no_network(*args, **kwargs):
    raise AssertionError("the test tried to download")


def test_missing_hand_download_is_listed_and_the_rest_is_built(workspace, capsys):
    tmp_path, argv = workspace
    assert download.main([*argv, "--pin"]) == 1
    out = capsys.readouterr().out
    assert "have to be downloaded by hand" in out
    assert str(tmp_path / "downloads" / "guide.pdf") in out

    built = (tmp_path / "corpus" / "agreement" / "annex1.md").read_text(encoding="utf-8")
    assert built.startswith("# An agreement: Annex 1\n\nSource: CELEX 0. Licence: test.")
    assert built.endswith("ANNEX 1\n\nNOTES\n\nBody text.\n")
    lock = json.loads((tmp_path / "lock.json").read_text())
    assert lock == {
        "legal": {"file": "legal.html", "sha256": hashlib.sha256(HTML.encode()).hexdigest()}
    }

    # Once the guide is in place, the corpus is complete and the run is clean.
    (tmp_path / "downloads" / "guide.pdf").write_bytes(b"%PDF-1.4 guide")
    assert download.main([*argv, "--pin"]) == 0
    assert (tmp_path / "corpus" / "guides" / "guide.pdf").read_bytes() == b"%PDF-1.4 guide"
    assert download.main(argv) == 0


def test_a_file_that_is_not_the_pinned_one_is_not_built(workspace, capsys):
    tmp_path, argv = workspace
    (tmp_path / "downloads" / "guide.pdf").write_bytes(b"%PDF-1.4 guide")
    assert download.main([*argv, "--pin"]) == 0
    (tmp_path / "downloads" / "guide.pdf").write_bytes(b"%PDF-1.4 another guide")
    (tmp_path / "corpus" / "guides" / "guide.pdf").unlink()

    assert download.main(argv) == 1
    assert "guide: " in capsys.readouterr().out
    assert not (tmp_path / "corpus" / "guides" / "guide.pdf").exists()


def test_stray_corpus_files_are_reported(workspace, capsys):
    tmp_path, argv = workspace
    (tmp_path / "downloads" / "guide.pdf").write_bytes(b"%PDF-1.4 guide")
    (tmp_path / "corpus" / "agreement").mkdir(parents=True)
    (tmp_path / "corpus" / "agreement" / "old_name.md").write_text("renamed section")

    assert download.main([*argv, "--pin"]) == 1
    assert "old_name.md does not come from sources.py" in capsys.readouterr().out
