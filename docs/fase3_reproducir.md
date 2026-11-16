# Fase 3 · Cómo reproducirla

Los comandos, en orden, desde un clon limpio hasta el endpoint `/rag`. Por qué se eligió cada
cosa está en [`EXPERIMENTS.md`](../EXPERIMENTS.md) (entradas de la fase 3, del 12 al 16 de
noviembre); aquí solo está la receta.

Todo se lanza desde la raíz del repositorio. Hacen falta los dos extras: `rag` para los
retrievers y `train` para el generador y el reranker. En la DGX, `source smoke/dgx_env.sh` antes
de nada (ver [`dgx.md`](dgx.md)).

```bash
uv sync --extra train --extra rag
uv run pytest            # todo verde
export HF_HOME=./hf_cache
```

## 1. El corpus (minutos, sin GPU)

El corpus no está en git: lo escribe `rag/download.py` a partir de `rag/sources.py`, y
`rag/sources.lock.json` comprueba que los ficheros son los mismos.

Cuatro guías de la Comisión hay que bajarlas a mano, porque su web responde a los scripts con una
comprobación anti-bots, que no saltamos. Se guardan en `rag/downloads/` con estos nombres:

| Fichero | De dónde |
|---|---|
| `tca_guide_for_traders.pdf` | https://taxation-customs.ec.europa.eu/document/download/2db103b1-08c1-4fdf-9b09-edde7c2dab51_en |
| `tca_origin_questions_answers.pdf` | https://taxation-customs.ec.europa.eu/system/files/2021-03/compendium_on_rules_of_origin_q_a.pdf |
| `japan_statements_on_origin.pdf` | https://taxation-customs.ec.europa.eu/system/files/2019-12/eu-japan-epa-guidance-statements-on-origin.pdf |
| `mercosur_origin_guidance_2026.pdf` | https://taxation-customs.ec.europa.eu/document/download/a620d284-34db-48d8-99e6-c6cd6252d236_en?filename=EU-Mercosur+ITA+Guidance+RoO+Apr2026.pdf |

```bash
uv run python -m rag.download     # baja el resto, comprueba las huellas y escribe rag/corpus/
```

Si falta alguna guía manual, el script dice de dónde bajarla, monta el resto y termina con código 1.
Una pasada que termina sin error quiere decir que el corpus está completo: 29 documentos.

## 2. La recuperación (GPU para embeber el corpus la primera vez)

Los vectores de los fragmentos se guardan en `rag/index/cache/`, y el índice de cada chunking en
`rag/index/<estrategia>/`. Las dos carpetas están fuera de git. Una segunda pasada no vuelve a
embeber nada.

```bash
# Las cuatro estrategias con los tres retrievers y el barrido de λ
#   -> reports/phase3_eval.json y reports/phase3_eval_lambda.png
uv run python -m rag.evaluate

# El umbral del semántico
#   -> reports/phase3_semantic_sweep.json
uv run python -m rag.evaluate --strategies semantic@0.55 semantic@0.6 semantic@0.65 \
    semantic@0.7 semantic@0.75 --out reports/phase3_semantic_sweep.json
#   -> reports/phase3_semantic_thresholds.json y reports/phase3_semantic_distances.png
uv run python -m rag.semantic_analysis

# La ablación de la cabecera y el experimento de la instrucción
#   -> reports/phase3_ablations.json
uv run python -m rag.ablations
```

## 3. Las respuestas (unos 5 minutos por variante en el portátil)

Cada variante responde las 64 preguntas del dorado. Escribe sus respuestas en
`reports/phase3_generation_<variante>.jsonl` y añade su resumen a `reports/phase3_generation.json`.

```bash
uv run python -m rag.evaluate_generation --retriever none     # línea base, sin RAG
uv run python -m rag.evaluate_generation --retriever bm25
uv run python -m rag.evaluate_generation --retriever dense
uv run python -m rag.evaluate_generation --retriever hybrid   # lo que sirve la API

# Tras cambiar answer_contains o verdict_excludes en rag/gold/gold.jsonl (sin GPU)
uv run python -m rag.evaluate_generation --rescore
```

## 4. El reranker (opcional; medido y desactivado)

```bash
#   -> reports/phase3_rerank_generic.json y reports/phase3_rerank_agreement.json
uv run python -m rag.rerank --instruction generic
uv run python -m rag.rerank --instruction agreement

ARCA_RAG_RERANK=generic uv run python -m rag.evaluate_generation --retriever hybrid \
    --label hybrid_rerank_generic
ARCA_RAG_RERANK=agreement uv run python -m rag.evaluate_generation --retriever hybrid \
    --label hybrid_rerank_agreement
```

## 5. El endpoint

Con el `.env` copiado de [`.env.example`](../.env.example), que no activa el reranker:

```bash
uv run arca-api
curl -s localhost:8000/rag -H 'Content-Type: application/json' \
    -d '{"question": "How long is a proof of origin valid under the revised PEM rules?", "top_k": 5, "retriever": "hybrid"}'
```

La primera petición carga el generador y los retrievers (unos 20 s en el portátil), y cada respuesta
tarda unos 4 s. `retriever` admite `bm25`, `dense` o `hybrid`.
