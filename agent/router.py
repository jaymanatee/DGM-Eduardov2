"""Knowledge router: search the knowledge base before the first step when the task asks how the
rules work.

Why it exists (EXPERIMENTS.md, 2026-11-23): in 45 attempts at the three knowledge tasks of the
phase-4 bench, the agent called ``search_knowledge_base`` 3 times. The tool-use SFT of phase 2
was trained, before the knowledge base existed, on "a concept question: I answer directly, no
tool", and the base brain goes straight to the calculation when the task also has numbers. The
RAG of phase 3 answers 81 % of its gold questions; inside the agent it was almost never asked.

So the loop decides instead of the model, the way the phase-3 pipeline always retrieves:

1. ``question_clauses`` takes the questions of the task (sentences with "?"), split where a
   second question starts ("Are they originating under CETA, and how long ...?").
2. A small classifier scores each one: does it ask how the rules work (validity, deadlines,
   thresholds, who may do what), or for something the tools answer (a verdict, a rule, a rate)?
   Character and word n-grams, logistic regression, trained on questions that are NOT the
   phase-4 bench: the phase-3 gold questions (minus the ones the bench and the held-out tasks
   were written from) are the positives, and the phase-2 synthetic training requests are the
   negatives, plus its concept questions as positives. Held-out checks: ``python -m
   agent.router evaluate``.
3. For each question above the threshold (two at most), the loop calls ``search_knowledge_base``
   itself before the model's first step. The call is in the trace like any other, and counts
   towards ``max_steps``.

Numpy only, no new dependency; the training examples are in ``agent/router_examples.jsonl``
(``python -m agent.router build`` writes them again) and the model is trained on first use,
which takes a few seconds.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

EXAMPLES = Path(__file__).resolve().parent / "router_examples.jsonl"
GOLD = Path(__file__).resolve().parents[1] / "rag" / "gold" / "gold.jsonl"

# Gold questions the router never trains on: q25, q27 and q32 are the source of the three
# knowledge tasks of the phase-4 bench; the rest, of the held-out tasks in
# agent/bench/tasks_knowledge.jsonl.
HELD_OUT = frozenset({"q25", "q27", "q32", "q01", "q10", "q18", "q26", "q53", "q57"})
THRESHOLD = 0.5
MAX_SEARCHES = 2
DIM = 1 << 20

_SENTENCE = re.compile(r"(?<=[?!])\s+|(?<=\.)\s+(?=[¿¡\"«A-ZÁÉÍÓÚÑ])")
_WH = (
    r"how|what|which|who|whom|when|where|why|can|could|do|does|did|is|are|may|must|should|will|"
    r"cu[aá]nt[oa]s?|cu[aá]l(?:es)?|qu[eé]|qui[eé]n(?:es)?|cu[aá]ndo|d[oó]nde|c[oó]mo|"
    r"puedo|puede|pueden|podemos|hay|debe|deben|debemos|tenemos|tiene|tienen|es|son"
)
_SECOND_QUESTION = re.compile(rf",?\s+(?:and|or|y|o|e)\s+(?=(?:{_WH})\b)|(?=¿)", re.IGNORECASE)

# The agreement a question is about, to add to a query that only says "that agreement".
AGREEMENTS = {
    "EU-UK TCA": r"\bTCA\b|Reino Unido|\bUK\b|United Kingdom|británic|British",
    "EU-Japan EPA": r"Jap[oó]n|Japan|UE-JP|EU-JP",
    "EU-Korea FTA": r"Corea|Korea|UE-KR|EU-KR",
    "CETA": r"\bCETA\b|Canad[aá]",
    "EU-Mercosur": r"Mercosur|Brasil|Brazil|Argentin|Paraguay|Uruguay",
    "PEM Convention": r"\bPEM\b|Suiza|Switzerland|Noruega|Norway|Turqu[ií]a|Turkey",
    "EU-Australia": r"Australia",
}


# A question for an origin verdict ("are they originating under the TCA?", "¿es originario?") is
# the job of evaluate_origin, not of the knowledge base: some gold questions are worded like that
# (q07, q09, q34), so the classifier alone cannot tell them from the verdict question that ends
# most tool tasks. Not "non-originating materials", which is a concept.
_VERDICT = re.compile(
    r"\b(?:is|are|be|count as|qualif(?:y|ies))\b[^?]{0,60}?(?<!non-)\boriginating\b"
    r"|\b(?:es|son|ser|sea|sean)\s+originari[oa]s?\b"
    r"|\btienen?\s+origen\s+preferencial\b",
    re.IGNORECASE,
)
MIN_WORDS = 4
SHORT_QUESTION = 8


def routable(clause: str) -> bool:
    """A question the router may send to the knowledge base: long enough to mean something
    ("¿Es válido?" is not) and not asking for a verdict."""
    return len(re.findall(r"\w+", clause)) >= MIN_WORDS and not _VERDICT.search(clause)


def question_clauses(text: str) -> list[tuple[str, str]]:
    """(clause, whole sentence) for every question of the text. The clause is what gets
    scored; the sentence, which keeps the context ("we are not an approved exporter"), is what
    gets searched."""
    out = []
    sentences = [s.strip() for s in _SENTENCE.split(text.strip()) if s.strip()]
    for i, sentence in enumerate(sentences):
        if "?" not in sentence:
            continue
        # A short question leans on the sentence before ("... how long we have to keep the
        # copies. What does the FTA say?"): that one goes into the search too.
        searched = sentence
        if i and len(re.findall(r"\w+", sentence)) < SHORT_QUESTION:
            searched = f"{sentences[i - 1]} {sentence}"
        parts = [p.strip(" ,;:") for p in _SECOND_QUESTION.split(sentence) if p and p.strip()]
        for part in parts:
            if len(part) > 8:
                out.append((part if searched == sentence else searched, searched))
    return out


def with_agreement(query: str, task: str) -> str:
    """The query, plus the agreement the task names if the query itself does not."""
    if any(re.search(p, query, re.IGNORECASE) for p in AGREEMENTS.values()):
        return query
    named = [n for n, p in AGREEMENTS.items() if re.search(p, task, re.IGNORECASE)]
    return f"{query} ({', '.join(named)})" if named else query


_ANY_AGREEMENT = re.compile(
    "|".join(f"(?:{p})" for p in AGREEMENTS.values()) + r"|\b(?:EU|UE)-\w+", re.IGNORECASE
)


def _normalise(text: str) -> str:
    """Lower case, no accents, digits as 0, and every agreement or partner country as one
    token: the gold questions name the agreement ("Under the EU-UK TCA, ...") far more than the
    tool requests do, and the classifier must not learn the agreement instead of the question."""
    text = _ANY_AGREEMENT.sub(" acuerdo ", text)
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"\d", "0", text)
    return re.sub(r"\s+", " ", text).strip()


def features(text: str) -> tuple[np.ndarray, np.ndarray]:
    """Hashed character 3-5-grams inside words and word 1-2-grams, L2-normalised."""
    norm = _normalise(text)
    words = re.findall(r"\w+|[¿?]", norm)
    grams = [f"w:{w}" for w in words] + [f"b:{a} {b}" for a, b in zip(words, words[1:])]
    for w in words:
        padded = f" {w} "
        grams += [f"c:{padded[i : i + n]}" for n in (3, 4, 5) for i in range(len(padded) - n + 1)]
    counts: dict[int, float] = {}
    for g in grams:
        h = zlib.crc32(g.encode()) % DIM
        counts[h] = counts.get(h, 0.0) + 1.0
    idx = np.fromiter(counts, dtype=np.int64, count=len(counts))
    val = np.sqrt(np.fromiter(counts.values(), dtype=np.float64, count=len(counts)))
    return idx, val / (np.linalg.norm(val) or 1.0)


@dataclass
class KnowledgeRouter:
    weights: np.ndarray
    bias: float
    threshold: float = THRESHOLD

    @classmethod
    def train(
        cls, examples: list[tuple[str, int]], epochs: int = 15, lr: float = 0.5, l2: float = 1e-5
    ) -> KnowledgeRouter:
        """Logistic regression by SGD, classes weighted to the same total, fixed order."""
        data = [(features(t), y) for t, y in examples]
        n_pos = sum(y for _, y in data) or 1
        n_neg = (len(data) - n_pos) or 1
        weight = {1: len(data) / (2 * n_pos), 0: len(data) / (2 * n_neg)}
        w, b = np.zeros(DIM), 0.0
        order = np.random.default_rng(0)
        for _ in range(epochs):
            for i in order.permutation(len(data)):
                (idx, val), y = data[i]
                p = 1 / (1 + np.exp(-(w[idx] @ val + b)))
                g = (p - y) * weight[y]
                w[idx] -= lr * (g * val + l2 * w[idx])
                b -= lr * g
        return cls(w, b)

    @classmethod
    def from_examples(cls, path: Path = EXAMPLES) -> KnowledgeRouter:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        return cls.train([(r["text"], r["label"]) for r in rows])

    def probability(self, text: str) -> float:
        idx, val = features(text)
        return float(1 / (1 + np.exp(-(self.weights[idx] @ val + self.bias))))

    def queries(self, task: str) -> list[tuple[str, float]]:
        """The searches to make before the first step: (query, probability), best first, one
        per sentence, at most MAX_SEARCHES."""
        best: dict[str, float] = {}
        for clause, sentence in question_clauses(task):
            if not routable(clause):
                continue
            p = self.probability(clause)
            if p >= self.threshold:
                best[sentence] = max(p, best.get(sentence, 0.0))
        ranked = sorted(best.items(), key=lambda kv: -kv[1])[:MAX_SEARCHES]
        return [(with_agreement(s, task), p) for s, p in ranked]


ROUTED_NOTE = (
    "\nNote: the loop searched the knowledge base for the user's question «{query}» before "
    "your first step. Answer that part from these passages and cite their ids. A passage about "
    "another agreement does not answer it; if none answers it, say the knowledge base does not "
    "cover it. The rest of the task still needs the tools."
)


# --------------------------------------------------------------------------- examples and checks


def _user_messages(trajectories) -> list[tuple[str, str]]:
    return [
        (t.kind, next(m["content"] for m in t.messages if m["role"] == "user"))
        for t in trajectories
    ]


def build_examples(path: Path = EXAMPLES) -> list[dict]:
    """Gold questions (minus HELD_OUT) and concept requests are 1; tool requests, requests
    outside the covered agreements and off-topic ones are 0. Only their question clauses, as the
    router scores them."""
    from tool_use.sft.build import build_split

    rows, seen = [], set()

    def add(text: str, label: int, source: str) -> None:
        for clause, _ in question_clauses(text) or [(text, text)]:
            if not routable(clause):
                continue  # never routed, so not a training example either
            key = _normalise(clause)
            if key not in seen:
                seen.add(key)
                rows.append({"text": clause, "label": label, "source": source})

    for g in map(json.loads, GOLD.read_text(encoding="utf-8").splitlines()):
        if g["id"] not in HELD_OUT:
            add(g["question"], 1, f"gold:{g['id']}")
    for kind, text in _user_messages(build_split("train", 1.0, 0)):
        add(text, int(kind == "concept"), f"synthetic_train:{kind}")
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8")
    return rows


def _report(name: str, router: KnowledgeRouter, items: list[tuple[str, int]]) -> None:
    items = [(t, y) for t, y in items if routable(t)]
    for t, y in items:
        p = router.probability(t)
        if (p >= router.threshold) != bool(y):
            print(f"    {'falso positivo' if not y else 'no detectada':<15} {p:.2f} {t[:90]}")
    tp = sum(1 for t, y in items if y and router.probability(t) >= router.threshold)
    fp = sum(1 for t, y in items if not y and router.probability(t) >= router.threshold)
    pos = sum(y for _, y in items)
    neg = len(items) - pos
    print(f"{name:<34} recall {tp}/{pos}   false positives {fp}/{neg}")


def evaluate(router: KnowledgeRouter) -> None:
    """Held-out checks: nothing here is in the training examples."""
    from tool_use.sft.build import build_split

    root = Path(__file__).resolve().parents[1]
    gold = {g["id"]: g for g in map(json.loads, GOLD.read_text(encoding="utf-8").splitlines())}
    _report(
        "gold held out (9, all knowledge)",
        router,
        [(c, 1) for i in sorted(HELD_OUT) for c, _ in question_clauses(gold[i]["question"])],
    )
    bench2 = [json.loads(line) for line in (root / "tool_use/bench/cases.jsonl").open()]
    concept2 = {"none-01", "none-02", "none-03", "none-04"}
    _report(
        "phase-2 bench (31 requests)",
        router,
        [
            (c, int(case["id"] in concept2))
            for case in bench2
            for c, _ in question_clauses(case["query"])
        ],
    )
    test = _user_messages(build_split("test", 0.1, 0))  # other products, other concept topics
    _report(
        "phase-2 synthetic test (200)",
        router,
        [(c, int(kind == "concept")) for kind, text in test for c, _ in question_clauses(text)],
    )
    print("\nphase-4 bench: the searches the router would make")
    for line in (root / "agent/bench/tasks.jsonl").open():
        task = json.loads(line)
        queries = router.queries(task["task"])
        shown = "; ".join(f"{p:.2f} «{q[:70]}»" for q, p in queries) or "-"
        print(f"  {task['id']:<24} {shown}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("build", "evaluate"))
    args = parser.parse_args()
    if args.command == "build":
        rows = build_examples()
        print(f"{len(rows)} examples, {sum(r['label'] for r in rows)} positive -> {EXAMPLES}")
    else:
        evaluate(KnowledgeRouter.from_examples())


if __name__ == "__main__":
    main()