"""Subconjunto de train para un piloto de destilación dirigido a los casos difíciles.

Entre los N primeros problemas de train elige los k primeros que son originarios solo por un
criterio de cambio de clasificación (CC, CTH o CTSH), que es donde falla el profesor, y los k
primeros del resto. Son los mismos problemas de train (misma semilla), así que el resultado se
compara problema a problema con una destilación anterior sobre esos N.

    uv run python -m rlm.pilot_subset --out rlm/data/piloto_def.jsonl

Imprime qué problemas de train ha elegido: la fila i del fichero es el problema ``train_idx[i]``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rlm.generate_problems import OriginGenerator, Problem, to_jsonl

CLASSIFICATION = {"CC", "CTH", "CTSH"}


def is_classification_only(problem: Problem) -> bool:
    """Originario, y ningún criterio de valor (MaxNOM, RVC) lo hace originario."""
    truth = json.loads(problem.answer)
    return truth["originario"] and set(truth["criterios_validos"]) <= CLASSIFICATION


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--from-first", type=int, default=256, help="N primeros de train")
    parser.add_argument("--k", type=int, default=10, help="problemas de cada grupo")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="rlm/data/piloto_def.jsonl")
    args = parser.parse_args()

    problems = OriginGenerator().generate(args.from_first, "train", args.seed)
    hard = [i for i, p in enumerate(problems) if is_classification_only(p)][: args.k]
    rest = [i for i, p in enumerate(problems) if not is_classification_only(p)][: args.k]
    indices = sorted(hard + rest)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(to_jsonl([problems[i] for i in indices], "train"), encoding="utf-8")
    print(json.dumps({"train_idx": indices, "clasificacion": hard, "resto": rest}))
    print(f"{len(indices)} problemas -> {out}")


if __name__ == "__main__":
    main()
