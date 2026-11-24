"""Retrieval and generation: augment the prompt with the retrieved chunks and answer with citations.

``RagPipeline.answer`` is what ``POST /rag`` calls:

1. retrieve ``top_k`` chunks with the requested retriever (dense, bm25 or hybrid);
2. build the augmented prompt: ``RAG_PROMPT_TEMPLATE`` (the one from class, with chunk ids
   so the model can cite them) as the user message, and ``SYSTEM_PROMPT`` with the rules:
   only facts from the passages, a citation after each sentence, the provision in force
   today when it depends on dates, a fixed sentence when they cover nothing, and the
   language of the question. Rules in a
   system message, passages in the user one, made Qwen3-1.7B cite in every answer of a
   sample of the gold set; with everything in one message it often cited nothing;
3. generate the answer with the generator model;
4. put the chunk ids back in place of the passage numbers, drop a stray "I don't know"
   after an answer that cites passages, extract the citations (only ids among the
   returned chunks survive) and return everything in a ``RagResponse``.

``RagPipeline.answer_closed_book`` is the baseline the RAG is compared with: the same model
and rules, but no passages, so it answers from what it learned (``rag/evaluate_generation.py
--retriever none``). The API does not serve it.

What is served is what was measured (``reports/phase3_eval.json``): the structural chunking
with the ``[document › section]`` header, the generic query instruction and the hybrid
retriever with λ = 0.3, chosen by MRR on the gold set.

The generator is configured from environment variables, as in phase 2:

* ``ARCA_RAG_MODEL``     model id (default ``Qwen/Qwen3-1.7B``, without adapter). The phase-1
  RLM is a LoRA on this same model, trained to solve origin calculations in a
  ``<think>/<answer>`` format rather than to read passages and cite them; it has not been
  measured as the generator.
* ``ARCA_RAG_ADAPTER``   optional LoRA adapter, merged into the model as in phases 1 and 2
* ``ARCA_RAG_THINKING``  ``0`` (default) answers without thinking, ``1`` lets Qwen3 think.
  On the same sample, thinking took 5 to 10 times longer and cited less.
* ``ARCA_RAG_MAX_NEW_TOKENS`` per answer (default 512, or 2048 when thinking)
* ``ARCA_RAG_RERANK``    empty (default) serves the retrievers as they are; ``generic`` or
  ``agreement`` puts Qwen3-Reranker on top of each of them, with that instruction
  (``rag/rerank.py``): the first ``POOL`` hits reordered by the reranker's P(yes)
* ``ARCA_RAG_AGREEMENT_FILTER`` ``1`` (default) keeps only passages of the agreements the
  question names, plus general documents (``rag/agreements.py``); ``0`` does not filter.
  Applied under the reranker, so the reranker sees the filtered pool

Try it from the command line::

    uv run python -m rag.generate "How long is a proof of origin valid under the PEM rules?"
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import date
from pathlib import Path

from api.schemas import RagResponse, RetrievedChunk
from rag.retriever import Hit

# The sentences the model is told to use when the passages do not answer the question.
NO_ANSWER = "I don't know: the retrieved passages do not answer this question."
NO_ANSWER_ES = "No lo sé: los pasajes recuperados no responden a esta pregunta."

SYSTEM_PROMPT = """You answer questions from traders about EU preferential rules of origin, using only the passages the user gives you. Today is {today}.
Rules:
- Use only facts stated in the passages. Do not add outside knowledge. A passage about one agreement never answers a question about another agreement.
- After each sentence, cite the passages it comes from by their number in square brackets, like [1] or [2][3].
- When a provision applies only between certain dates, use the one in force today.
- If none of the passages answers the question, your whole reply is: "{no_answer}" (to a question in Spanish: "{no_answer_es}").
- Answer in the language of the question, in at most five sentences."""  # noqa: E501 (the prompt's lines)

# The baseline without RAG: the same rules, minus those about passages and citations.
CLOSED_BOOK_NO_ANSWER = "I don't know."
CLOSED_BOOK_NO_ANSWER_ES = "No lo sé."
CLOSED_BOOK_SYSTEM_PROMPT = """You answer questions from traders about EU preferential rules of origin. Today is {today}.
Rules:
- Answer from what you know. Give the figures, periods and articles only if you are sure of them.
- When a provision applies only between certain dates, use the one in force today.
- If you do not know the answer, your whole reply is: "{no_answer}" (to a question in Spanish: "{no_answer_es}").
- Answer in the language of the question, in at most five sentences."""  # noqa: E501 (the prompt's lines)

# The template from class. The passages are numbered rather than shown with their chunk ids:
# Qwen3-1.7B miscopied the ten-character ids (a citation to a chunk that was not returned
# is dropped), and copies [1]..[k] without fault. ``restore_citations`` puts the ids back.
RAG_PROMPT_TEMPLATE = """Use the provided up-to-date information to answer the question.
Cite the passages you rely on by writing their number in square brackets, like [{example_id}].

Retrieved information:
{chunks}

Question: {question}
"""

# The fixed sentences, and other ways the model has of saying that it does not know.
ABSTENTION_PATTERN = re.compile(
    r"I don'?t know|I do not know|no lo s[ée]|no (?:dispongo|tengo) de (?:suficiente )?"
    r"informaci[óo]n|(?:passages|information|documents?|text) (?:provided |retrieved )?"
    r"(?:do(?:es)? not|don'?t) (?:answer|contain|provide|include|mention|specify|address)"
    r"|not (?:\w+ )?(?:specified|mentioned|provided|stated|addressed|given) in the "
    r"(?:provided|retrieved|given)"
    r"|(?:pasajes|documentos) (?:recuperados )?no (?:responden|contienen|mencionan|especifican)",
    re.IGNORECASE,
)

CITATION_PATTERN = re.compile(r"\[([A-Za-z0-9_-]+)\]")
NUMBERED_CITATION = re.compile(r"\[\s*(\d+(?:\s*[,;]\s*\d+)*)\s*\]")
THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)

SERVING_STRATEGY = "structural"
HYBRID_LAMBDA = 0.3
HYBRID_POOL = 20
DEFAULT_MODEL = "Qwen/Qwen3-1.7B"
RETRIEVERS = ("dense", "bm25", "hybrid")


def format_chunks(hits: list[Hit], numbered: bool = False) -> str:
    """The chunks for the prompt, each under its id, or under its rank if ``numbered``."""
    return "\n\n".join(
        f"[{i if numbered else hit.chunk.id}] (from {hit.chunk.source})\n{hit.chunk.text}"
        for i, hit in enumerate(hits, start=1)
    )


def build_prompt(question: str, hits: list[Hit]) -> str:
    """The user message: the retrieved chunks, numbered 1..k, and the question."""
    return RAG_PROMPT_TEMPLATE.format(
        example_id=1, chunks=format_chunks(hits, numbered=True), question=question
    )


STRAY_NO_ANSWER = re.compile(
    r"\s*(?:Therefore,\s*|Por (?:lo )?tanto,\s*)?\[?\"?(?:"
    + "|".join(re.escape(s) for s in (NO_ANSWER, NO_ANSWER_ES))
    + r")\"?\]?",
    re.IGNORECASE,
)


def drop_stray_no_answer(answer: str) -> str:
    """The fixed "I don't know" sentence, removed from an answer that cites passages.

    Qwen3-1.7B appended it to 33 of 55 answers that cited passages on the gold set, however
    the rules were worded, so the answer contradicted itself. A real abstention cites
    nothing and is left as it is."""
    rest = STRAY_NO_ANSWER.sub("", answer).strip()
    return rest if rest and NUMBERED_CITATION.search(rest) else answer


def restore_citations(answer: str, hits: list[Hit]) -> str:
    """``[2]`` -> ``[<id of the 2nd chunk>]``, ``[1, 3]`` -> ``[<id 1>][<id 3>]``.

    A number with no chunk behind it (``[7]`` when five were retrieved) is left as written:
    ``extract_citations`` then ignores it, as it ignores any id that was not returned."""

    def replace(match: re.Match) -> str:
        numbers = [int(n) for n in re.findall(r"\d+", match.group(1))]
        if not all(1 <= n <= len(hits) for n in numbers):
            return match.group(0)
        return "".join(f"[{hits[n - 1].chunk.id}]" for n in numbers)

    return NUMBERED_CITATION.sub(replace, answer)


def build_messages(question: str, hits: list[Hit], today: date | None = None) -> list[dict]:
    """The conversation the generator sees: the rules, then the augmented prompt."""
    system = SYSTEM_PROMPT.format(
        today=(today or date.today()).isoformat(), no_answer=NO_ANSWER, no_answer_es=NO_ANSWER_ES
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": build_prompt(question, hits)},
    ]


def build_closed_book_messages(question: str, today: date | None = None) -> list[dict]:
    """The baseline's conversation: the rules without passages, then the bare question."""
    system = CLOSED_BOOK_SYSTEM_PROMPT.format(
        today=(today or date.today()).isoformat(),
        no_answer=CLOSED_BOOK_NO_ANSWER,
        no_answer_es=CLOSED_BOOK_NO_ANSWER_ES,
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": question}]


def extract_citations(answer: str, valid_ids: set[str]) -> list[str]:
    """Ids cited in the answer that actually exist among the retrieved chunks, in order."""
    seen: list[str] = []
    for match in CITATION_PATTERN.findall(answer):
        if match in valid_ids and match not in seen:
            seen.append(match)
    return seen


def strip_thinking(raw: str) -> str:
    """The answer without the ``<think>`` block, closed or (if the model never closed it)
    open: with thinking on, what follows the tag is then the answer."""
    text = THINK_BLOCK.sub("", raw).strip()
    if text.startswith("<think>"):
        text = text.removeprefix("<think>").strip()
    return text


ABSTENTION_OPENING = 200  # characters where an answer that does not know says so


def is_abstention(answer: str, citations: list[str] | None = None) -> bool:
    """Whether the answer opens by saying it does not know, and cites nothing.

    Only the opening counts: an answer that ends by saying what the passages leave open
    is still an answer."""
    return not citations and bool(ABSTENTION_PATTERN.search(answer[:ABSTENTION_OPENING]))


def to_response(answer: str, hits: list[Hit], retriever: str, model: str = "") -> RagResponse:
    chunks = [
        RetrievedChunk(id=h.chunk.id, source=h.chunk.source, score=h.score, text=h.chunk.text)
        for h in hits
    ]
    return RagResponse(
        answer=answer,
        chunks=chunks,
        citations=extract_citations(answer, {c.id for c in chunks}),
        retriever=retriever,
        model=model,
    )


class RagPipeline:
    """Everything ``POST /rag`` needs, built once and reused."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        adapter_path: str | None = None,
        thinking: bool = False,
        max_new_tokens: int | None = None,
        strategy: str = SERVING_STRATEGY,
        weight_bm25: float = HYBRID_LAMBDA,
        index_dir: str | Path = "rag/index",
        rerank: str | None = None,
        agreement_filter: bool = True,
    ):
        self.model_id, self.adapter_path, self.thinking = model_id, adapter_path, thinking
        self.max_new_tokens = max_new_tokens or (2048 if thinking else 512)
        self.strategy, self.weight_bm25, self.index_dir = strategy, weight_bm25, Path(index_dir)
        self.rerank = rerank or None
        self.agreement_filter = agreement_filter
        self.retrievers: dict = {}
        self.model = self.tokenizer = None

    @classmethod
    def from_env(cls) -> RagPipeline:
        return cls(
            model_id=os.environ.get("ARCA_RAG_MODEL") or DEFAULT_MODEL,
            adapter_path=os.environ.get("ARCA_RAG_ADAPTER") or None,
            thinking=os.environ.get("ARCA_RAG_THINKING", "0") == "1",
            max_new_tokens=int(os.environ.get("ARCA_RAG_MAX_NEW_TOKENS") or 0) or None,
            rerank=os.environ.get("ARCA_RAG_RERANK") or None,
            agreement_filter=os.environ.get("ARCA_RAG_AGREEMENT_FILTER", "1") == "1",
        )

    @property
    def name(self) -> str:
        """Model label for the response: id, adapter and whether it thinks."""
        label = self.model_id + (f"+{self.adapter_path}" if self.adapter_path else "")
        rerank = f", rerank {self.rerank}" if self.rerank else ""
        rerank += ", agreement filter" if self.agreement_filter else ""
        return f"{label} (thinking {'on' if self.thinking else 'off'}{rerank})"

    def load(self, generator: bool = True) -> None:
        """Ingest -> chunk -> the three retrievers -> the generator model.

        The dense index is the one the evaluation persisted under ``rag/index/<strategy>``:
        same chunks, same fingerprint, so nothing is embedded again. ``generator=False``
        loads only the retrievers (enough for ``retrieve``).
        """
        from rag.embed import CachedEmbedder
        from rag.ingest import load_corpus
        from rag.retriever import AgreementFilter, BM25Retriever, DenseRetriever, HybridRetriever
        from rag.strategies import build_chunks

        documents = load_corpus()
        embedder = CachedEmbedder()
        chunks = build_chunks(self.strategy, documents, embedder)
        bm25 = BM25Retriever(chunks)
        dense = DenseRetriever(chunks, embedder, persist_dir=self.index_dir / self.strategy)
        hybrid = HybridRetriever(bm25, dense, weight_bm25=self.weight_bm25, pool=HYBRID_POOL)
        self.retrievers = {"bm25": bm25, "dense": dense, "hybrid": hybrid}
        if self.agreement_filter:
            self.retrievers = {name: AgreementFilter(r) for name, r in self.retrievers.items()}
        if self.rerank:
            from rag.rerank import INSTRUCTIONS, Reranker, RerankingRetriever

            if self.rerank not in INSTRUCTIONS:
                raise ValueError(
                    f"ARCA_RAG_RERANK={self.rerank!r}: one of {', '.join(INSTRUCTIONS)}"
                )
            reranker = Reranker(instruction=INSTRUCTIONS[self.rerank])
            self.retrievers = {
                name: RerankingRetriever(r, reranker) for name, r in self.retrievers.items()
            }
        if generator:
            self._load_generator()

    def _load_generator(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from rlm import disable_cudnn_attention

        disable_cudnn_attention()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        model = AutoModelForCausalLM.from_pretrained(self.model_id, dtype=dtype, device_map=device)
        if self.adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, self.adapter_path).merge_and_unload()
        self.model = model.eval()

    def retrieve(self, question: str, top_k: int, retriever: str) -> list[Hit]:
        if retriever not in self.retrievers:
            raise ValueError(f"unknown retriever {retriever!r}: one of {', '.join(RETRIEVERS)}")
        return self.retrievers[retriever].search(question, top_k=top_k)

    def generate(self, messages: list[dict]) -> str:
        """One greedy generation of the answer, without the thinking block if there is one."""
        import torch

        if self.model is None:
            raise RuntimeError("call load() first")
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.thinking,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        new_tokens = out[0, inputs["input_ids"].shape[1] :]
        return strip_thinking(self.tokenizer.decode(new_tokens, skip_special_tokens=True))

    def answer(self, question: str, top_k: int = 5, retriever: str = "hybrid") -> RagResponse:
        hits = self.retrieve(question, top_k, retriever)
        raw = self.generate(build_messages(question, hits))
        answer = restore_citations(drop_stray_no_answer(raw), hits)
        return to_response(answer, hits, retriever, model=self.name)

    def answer_closed_book(self, question: str) -> str:
        """The baseline: the same generator with no passages. Only the answer is returned:
        there are no chunks to cite, and ``RagResponse`` is the API's, for the three
        retrievers."""
        return self.generate(build_closed_book_messages(question))


_pipeline: RagPipeline | None = None


def build_pipeline() -> RagPipeline:
    global _pipeline
    if _pipeline is None:
        pipeline = RagPipeline.from_env()
        pipeline.load()
        _pipeline = pipeline
    return _pipeline


def answer_with_rag(question: str, top_k: int = 5, retriever: str = "hybrid") -> RagResponse:
    """Convenience entry point for scripts and the evaluation."""
    return build_pipeline().answer(question, top_k=top_k, retriever=retriever)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Answer one question with the RAG pipeline.")
    parser.add_argument("question", nargs="+")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--retriever", choices=RETRIEVERS, default="hybrid")
    args = parser.parse_args(argv)
    response = answer_with_rag(" ".join(args.question), args.top_k, args.retriever)
    print(response.model_dump_json(indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])