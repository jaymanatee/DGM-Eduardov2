"""Phase 3 text preparation: PDF cleaning, sentences and the three chunking strategies."""

import numpy as np
import pytest

from rag.chunking import (
    add_context,
    consecutive_distances,
    find_headings,
    recursive_chunks,
    section_at,
    semantic_chunks,
    split_sentences,
    structural_chunks,
)
from rag.ingest import Document, clean_pdf_pages, load_corpus

pytest.importorskip("langchain_text_splitters")


# ------------------------------------------------------------------------------ PDF text


def test_clean_pdf_pages_drops_headers_and_rebuilds_paragraphs():
    def page(n: int, who: str) -> str:
        return (
            f"March 2025 Page {n} of 3\n"
            "Contents ........................ 2\n"
            f"The {who} shall keep the proofs of origin and the supporting documents for\n"
            f"{n + 2} years ({who}).\n"
            f"1. The {who} makes out the statement on origin on an invoice or on any oth-\n"
            f"er commercial document of the {who}.\n"
            f"{n}\n"
        )

    text = clean_pdf_pages([page(1, "exporter"), page(2, "producer"), page(3, "supplier")])
    assert "Page" not in text and "Contents" not in text
    paragraphs = text.split("\n\n")
    assert paragraphs[:2] == [
        "The exporter shall keep the proofs of origin and the supporting documents for "
        "3 years (exporter).",
        # A numbered item starts a paragraph; a hyphenated line break is joined.
        "1. The exporter makes out the statement on origin on an invoice or on any other "
        "commercial document of the exporter.",
    ]
    assert len(paragraphs) == 6


def test_markdown_header_becomes_metadata(tmp_path):
    (tmp_path / "tca").mkdir()
    (tmp_path / "tca" / "rules_of_origin.md").write_text(
        "# EU-UK TCA: Rules of origin\n\nSource: CELEX 1, https://x. Licence: reuse.\n\n"
        "Article 37\n\nObjective",
        encoding="utf-8",
    )
    (doc,) = load_corpus(tmp_path)
    assert doc.text == "Article 37\n\nObjective"
    assert doc.metadata["title"] == "EU-UK TCA: Rules of origin"  # the header wins
    assert doc.metadata["reference"] == "CELEX 1, https://x"
    assert doc.metadata["language"] == "en"


# ----------------------------------------------------------------------------- sentences


def test_split_sentences_does_not_break_after_markers_or_abbreviations():
    text = (
        "1. Products shall be originating, e.g. wholly obtained ones. See Art. 7 of the "
        "Convention.\n\n(a) The value is 10 %. This is the limit! Is it? Yes."
    )
    assert split_sentences(text) == [
        "1. Products shall be originating, e.g. wholly obtained ones.",
        "See Art. 7 of the Convention.",
        "(a) The value is 10 %.",
        "This is the limit!",
        "Is it?",
        "Yes.",
    ]


# ----------------------------------------------------------------------------- semantic

TOPICS = {"origin": [1.0, 0.0], "tariff": [0.0, 1.0]}


def _fake_embed(sentences):
    """Each sentence points at the topic it mentions: distance 0 within, 1 across topics."""
    vectors = [TOPICS["origin" if "origin" in s else "tariff"] for s in sentences]
    return np.array(vectors, dtype=float)


def _doc(text, suffix=".md", title="Doc"):
    return Document(id="d1", source="d1.md", text=text, metadata={"suffix": suffix, "title": title})


def test_consecutive_distances():
    assert consecutive_distances(np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])) == [0.0, 1.0]


def test_semantic_chunks_cut_where_the_topic_changes_and_keep_offsets():
    text = (
        "Rules of origin apply. The origin is proved.\n\n"
        "The tariff is 5 %. The tariff rate is reduced. Back to origin now."
    )
    doc = _doc(text)
    chunks = semantic_chunks([doc], _fake_embed, distance_threshold=0.5)
    assert [c.text for c in chunks] == [
        "Rules of origin apply. The origin is proved.",
        "The tariff is 5 %. The tariff rate is reduced.",
        "Back to origin now.",
    ]
    assert all(text[c.start : c.end] == c.text for c in chunks)
    assert [c.id for c in chunks] == ["d1-0000", "d1-0001", "d1-0002"]


def test_semantic_chunks_min_chars_keeps_short_chunks_open():
    text = "One origin. Two tariff. Three origin. Four tariff."
    chunks = semantic_chunks([_doc(text)], _fake_embed, distance_threshold=0.5, min_chunk_chars=20)
    assert [c.text for c in chunks] == ["One origin. Two tariff.", "Three origin. Four tariff."]


def test_semantic_chunks_close_a_chunk_at_max_chars():
    text = " ".join(f"Sentence {i} about origin." for i in range(20))
    chunks = semantic_chunks([_doc(text)], _fake_embed, distance_threshold=0.5, max_chunk_chars=100)
    assert len(chunks) > 1
    assert all(len(c.text) <= 100 for c in chunks)


# --------------------------------------------------------------------------- structural

LEGAL = (
    """SECTION 1

RULES OF ORIGIN

Article 37

Objective

The objective of this Chapter is to lay down the provisions determining the origin of goods.

Article 38

Definitions

1. For the purposes of this Chapter:

(a) "classification" means the classification of a product under the Harmonised System.

Article 39
Tolerances
"""
    + "A long paragraph about tolerances. " * 60
)


def test_find_headings_of_a_legal_text():
    labels = [(h.level, h.label) for h in find_headings(LEGAL)]
    assert labels == [
        ("top", "SECTION 1 RULES OF ORIGIN"),
        ("unit", "Article 37 Objective"),
        ("unit", "Article 38 Definitions"),
        ("unit", "Article 39 Tolerances"),  # heading and title on consecutive lines (PEM)
    ]
    headings = find_headings(LEGAL)
    assert section_at(headings, LEGAL.index("lay down")) == (
        "SECTION 1 RULES OF ORIGIN › Article 37 Objective"
    )
    # Numbered paragraphs are not headings in a legal text.
    assert not any(h.label.startswith("1.") for h in headings)


def test_guide_headings_only_with_numbered():
    text = "B.16. Duty drawback\n\nSome text.\n\n19. Question: Is it valid?\n\nAnswer: Yes."
    assert find_headings(text) == []
    assert [h.label for h in find_headings(text, numbered=True)] == [
        "B.16. Duty drawback",
        "Question 19",
    ]


def test_structural_chunks_one_per_article_merge_headings_split_long_ones():
    doc = _doc(LEGAL)
    chunks = structural_chunks([doc], max_chunk_chars=600, min_chunk_chars=120)
    # The section heading has no text of its own: it is merged into Article 37.
    assert chunks[0].text.startswith("SECTION 1\n\nRULES OF ORIGIN\n\nArticle 37")
    assert chunks[0].text.endswith("origin of goods.")
    assert chunks[1].text.startswith("Article 38") and chunks[1].text.endswith("System.")
    # Article 39 is longer than 600 characters: split, every piece within the limit.
    assert all(c.text.startswith(("Article 39", "A long")) for c in chunks[2:])
    assert len(chunks) > 3 and all(len(c.text) <= 600 for c in chunks)
    assert all(LEGAL[c.start : c.end] == c.text for c in chunks)


def test_recursive_chunks_keep_offsets():
    doc = _doc(LEGAL)
    chunks = recursive_chunks([doc], chunk_size=300, chunk_overlap=50)
    assert all(LEGAL[c.start : c.end] == c.text for c in chunks)


def test_add_context_is_the_same_header_for_every_strategy():
    doc = _doc(LEGAL, title="EU-UK TCA: Rules of origin")
    recursive = add_context(recursive_chunks([doc], chunk_size=300, chunk_overlap=0), [doc])
    structural = add_context(structural_chunks([doc], max_chunk_chars=600), [doc])
    for chunks in (recursive, structural):
        first = chunks[0]
        assert first.text.startswith("[EU-UK TCA: Rules of origin › SECTION 1 RULES OF ORIGIN]\n")
        assert first.text.endswith(LEGAL[first.start : first.end])
    last = structural[-1]
    assert last.section == "SECTION 1 RULES OF ORIGIN › Article 39 Tolerances"
