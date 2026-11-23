"""Phase 3: the agreement filter (``rag/agreements.py`` and ``keep`` in the retrievers)."""

import json
from pathlib import Path

import numpy as np
import pytest

from rag.agreements import agreement_of, agreements_named, keep_for
from rag.chunking import Chunk
from rag.evaluate import GOLD_FILE, load_gold
from rag.retriever import AgreementFilter, DenseRetriever, HybridRetriever

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("How long is a statement on origin valid under the EU-UK TCA?", {"tca"}),
        ("What is the value limit for small packages imported into Korea?", {"korea"}),
        ("¿Cuánto tiempo es válida una declaración de origen en el acuerdo UE-Japón?", {"japan"}),
        ("Does the revised PEM Convention prohibit duty drawback?", {"pem"}),
        ("We export to Switzerland and to Korea", {"pem", "korea"}),
        ("¿Cómo se llama en el CETA el documento...?", {"ceta"}),
        ("Which reference number must Argentinian exporters show?", {"mercosur"}),
        ("¿Caduca el número REX de un exportador registrado?", set()),
        ("How long is a statement on origin valid under the EU-Australia FTA?", set()),
        ("The packaging is checked by the customs office", set()),  # no "PEM", "UK" inside
    ],
)
def test_agreements_named(text, expected):
    assert agreements_named(text) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("tca/rules_of_origin.md", "tca"),
        ("korea/protocol_rules_of_origin.md", "korea"),
        ("guides/tca_origin_questions_answers.pdf", "tca"),
        ("guides/japan_statements_on_origin.pdf", "japan"),
        ("guides/mercosur_origin_guidance_2026.pdf", "mercosur"),
        ("guides/aeat_rex.pdf", None),
        ("guides/guidance_preferential_origin_2025.pdf", None),
        ("cn_notes/chapter87_vehicles.md", None),
    ],
)
def test_agreement_of(source, expected):
    assert agreement_of(source) == expected


def test_keep_for_keeps_the_named_agreement_and_general_documents():
    assert keep_for("REX numbers") is None
    keep = keep_for("small packages into Korea")
    assert keep("korea/protocol_rules_of_origin.md")
    assert keep("guides/aeat_rex.pdf")
    assert not keep("japan/chapter3_rules_of_origin.md")
    assert not keep("guides/tca_guide_for_traders.pdf")


def test_the_filter_never_removes_gold_evidence():
    for item in load_gold(GOLD_FILE):
        keep = keep_for(item["question"])
        if keep is not None:
            lost = [e["source"] for e in item["evidence"] if not keep(e["source"])]
            assert not lost, (item["id"], lost)


def test_held_out_knowledge_tasks_name_their_agreement():
    """The agent's held-out tasks (not used to write the patterns) are recognised too."""
    expected = {
        "kn-01": {"tca"},
        "kn-02": {"japan"},
        "kn-03": {"korea"},
        "kn-04": set(),
        "kn-05": {"japan"},
        "kn-06": {"tca"},
    }
    path = ROOT / "agent" / "bench" / "tasks_knowledge.jsonl"
    tasks = {t["id"]: t for t in map(json.loads, path.read_text(encoding="utf-8").splitlines())}
    assert {tid: agreements_named(tasks[tid]["task"]) for tid in expected} == expected


# ------------------------------------------------------------------------------ retrievers


def _chunk(cid, source, text):
    return Chunk(id=cid, doc_id=cid, source=source, text=text, index=0)


CHUNKS = [
    _chunk("jp", "japan/chapter3_rules_of_origin.md", "small packages origin limit Japan"),
    _chunk("pem", "pem/appendix1_rules_of_origin.md", "small packages origin limit"),
    _chunk("kr", "korea/protocol_rules_of_origin.md", "packages limit"),
    _chunk("gen", "guides/aeat_rex.pdf", "registered exporter"),
]


class FakeEmbedder:
    model_id = "fake"

    def _vector(self, text):
        return [1.0, 0.0] if "small" in text else [0.0, 1.0]

    def embed_passages(self, texts):
        return np.array([self._vector(t) for t in texts])

    def embed_queries(self, texts, instruction=""):
        return np.array([self._vector(t) for t in texts])


@pytest.fixture
def hybrid(tmp_path):
    pytest.importorskip("rank_bm25")
    from rag.retriever import BM25Retriever

    dense = DenseRetriever(CHUNKS, FakeEmbedder(), persist_dir=tmp_path)
    return HybridRetriever(BM25Retriever(CHUNKS), dense, weight_bm25=0.3, pool=20)


def test_keep_fills_the_top_k_with_passages_that_pass(hybrid):
    query = "small packages limit Korea"
    unfiltered = [h.chunk.id for h in hybrid.search(query, top_k=2)]
    assert unfiltered[0] in {"jp", "pem"}  # the other agreements win without the filter
    keep = keep_for(query)
    for retriever in (hybrid.bm25, hybrid.dense, hybrid):
        ids = [h.chunk.id for h in retriever.search(query, top_k=2, keep=keep)]
        assert ids[0] == "kr"
        assert set(ids) <= {"kr", "gen"}
        assert len(ids) == 2  # filled with the next passage that passes


def test_agreement_filter_passes_through_queries_without_an_agreement(hybrid):
    filtered = AgreementFilter(hybrid)
    assert filtered.name == "hybrid+agreement"
    assert filtered.chunks is CHUNKS
    query = "small packages limit"
    assert [h.chunk.id for h in filtered.search(query, 3)] == [
        h.chunk.id for h in hybrid.search(query, 3)
    ]
    assert [h.chunk.id for h in filtered.search("small packages Korea", 1)] == ["kr"]


def test_pipeline_reads_the_filter_from_the_environment(monkeypatch):
    from rag.generate import RagPipeline

    monkeypatch.delenv("ARCA_RAG_AGREEMENT_FILTER", raising=False)
    pipeline = RagPipeline.from_env()
    assert pipeline.agreement_filter  # on by default
    assert "agreement filter" in pipeline.name
    monkeypatch.setenv("ARCA_RAG_AGREEMENT_FILTER", "0")
    assert not RagPipeline.from_env().agreement_filter