"""Which trade agreement a question names, and which agreement a passage belongs to.

The agreements of the corpus share most of their wording: "a statement on origin shall be
valid for 12 months" is in the TCA, Japan and Mercosur texts. The ``[document › section]``
header of each chunk (``rag/chunking.py``) helps the retrievers tell them apart, but not
enough: in the hybrid top 5 of the gold set, 41 passages came from another agreement than
the one the question names, and they were part of the wrong answers to q02, q27, q43 and
q44 ("small packages into Korea" answered with the Japanese limit).

``agreements_named`` finds the agreements a question names, by name or by partner country.
``agreement_of`` gives the agreement of a corpus document from its path: the folder of the
agreement texts (``tca/``, ``japan/``...) or the prefix of a guide (``guides/tca_...``).
Documents of no agreement (the CN explanatory notes, the Commission's general guidance,
the AEAT guides on REX and authorised exporters) belong to ``None`` and are always kept.

``keep_for(query)`` is the filter the retrievers apply (``rag/retriever.py``): when the
query names one or more agreements, only passages of those agreements or of none can be
retrieved; when it names none (or names one that is not in the corpus, such as Australia),
nothing is filtered.
"""

from __future__ import annotations

import re
from collections.abc import Callable

AGREEMENTS = ("tca", "japan", "korea", "ceta", "pem", "mercosur")

# Names and partner countries, in English and Spanish. Case-sensitive where an acronym
# could be part of another word ("UK", "PEM", "TCA", "CETA").
_PATTERNS = {
    "tca": r"\bTCA\b|Trade and Cooperation|\bEU-UK\b|\bUE-RU\b|Reino Unido|\bUK\b"
    r"|United Kingdom|[Bb]ritish|[Bb]ritánic",
    "japan": r"[Jj]apan|[Jj]ap[oó]n|[Jj]apon[eé]s|[Jj]aponesa",
    "korea": r"[Kk]orea|[Cc]orea",
    "ceta": r"\bCETA\b|[Cc]anad",
    "pem": r"\bPEM\b|[Pp]an-Euro|[Pp]aneuromediterr|Switzerland|Suiza|Norway|Noruega"
    r"|Iceland|Islandia",
    "mercosur": r"[Mm]ercosur|[Aa]rgentin|Brazil|Brasil|Paraguay|Urugua",
}
_COMPILED = {name: re.compile(pattern) for name, pattern in _PATTERNS.items()}


def agreements_named(text: str) -> set[str]:
    """The agreements of the corpus that ``text`` names (empty set: none)."""
    return {name for name, pattern in _COMPILED.items() if pattern.search(text)}


def agreement_of(source: str) -> str | None:
    """Agreement of a corpus document from its path, or ``None`` for general documents."""
    folder, _, name = source.partition("/")
    if folder in AGREEMENTS:
        return folder
    if folder == "guides":
        return next((a for a in AGREEMENTS if name.startswith(f"{a}_")), None)
    return None


def keep_for(query: str) -> Callable[[str], bool] | None:
    """A predicate on document sources for ``query``, or ``None`` when nothing is filtered."""
    named = agreements_named(query)
    if not named:
        return None

    def keep(source: str) -> bool:
        agreement = agreement_of(source)
        return agreement is None or agreement in named

    return keep