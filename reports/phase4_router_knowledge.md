# Fase 4 · Análisis de las ejecuciones

## Resumen

| Ejecución | Modelo | Éxito | Pasos | Tokens | s/tarea | Argumentos exactos | Acierta con argumentos mal |
|---|---|---|---|---|---|---|---|
| `eval_v1k:base` | Qwen/Qwen3-1.7B (base, thinking off) | 50 % (3/6) | 2.3 | 9230 | 10 | 0/2 | 0 |
| `eval_v1k:thinking` | Qwen/Qwen3-1.7B (thinking, thinking on) | 50 % (3/6) | 2.7 | 10734 | 17 | 0/2 | 0 |
| `eval_v1k:rlm` | Qwen/Qwen3-1.7B+tool_use/weights/sft_think (rlm, thinking on) | 33 % (2/6) | 2.7 | 9856 | 27 | 2/2 | 0 |
| `eval_v2k:base` | Qwen/Qwen3-1.7B (base, thinking off) | 100 % (6/6) | 2.3 | 6112 | 7 | 0/2 | 2 |
| `eval_v2k:thinking` | Qwen/Qwen3-1.7B (thinking, thinking on) | 100 % (6/6) | 4.3 | 16575 | 21 | 1/2 | 1 |
| `eval_v2k:rlm` | Qwen/Qwen3-1.7B+tool_use/weights/sft_think (rlm, thinking on) | 83 % (5/6) | 3.3 | 10684 | 15 | 2/2 | 0 |

*Argumentos exactos*: la llamada a `evaluate_origin` coincide con la de referencia (acuerdo, producto, precios y cada material). *Acierta con argumentos mal*: la tarea pasa su comprobación aunque los argumentos estén mal.

## Éxito por categoría

| Categoría | `eval_v1k:base` | `eval_v1k:thinking` | `eval_v1k:rlm` | `eval_v2k:base` | `eval_v2k:thinking` | `eval_v2k:rlm` |
|---|---|---|---|---|---|---|
| conocimiento | 50 % | 50 % | 33 % | 100 % | 100 % | 83 % |

## Tarea a tarea

| Tarea | `eval_v1k:base` | `eval_v1k:thinking` | `eval_v1k:rlm` | `eval_v2k:base` | `eval_v2k:thinking` | `eval_v2k:rlm` |
|---|---|---|---|---|---|---|
| kn-01 | ✓ 2 | ✓ 2 | ✗ 3 | ✓ 2 | ✓ 3 | ✓ 3 |
| kn-02 | ✓ 2 | ✓ 3 | ✓ 2 | ✓ 2 | ✓ 3 | ✓ 3 |
| kn-03 | ✓ 2 | ✓ 3 | ✓ 3 | ✓ 2 | ✓ 10 | ✓ 3 |
| kn-04 | ✗ 2 | ✗ 2 | ✗ 2 | ✓ 2 | ✓ 3 | ✓ 3 |
| kn-05 | ✗ ⚠ 3 | ✗ ⚠ 3 | ✗ 3 | ✓ ⚠ 3 | ✓ 4 | ✓ 4 |
| kn-06 | ✗ ⚠ 3 | ✗ ⚠ 3 | ✗ 3 | ✓ ⚠ 3 | ✓ ⚠ 3 | ✗ 4 |

✓/✗ éxito y número de pasos; ⚠ argumentos de `evaluate_origin` distintos de la referencia.

## Errores en los argumentos de `evaluate_origin`

- `eval_v1k:base`: precio FOB (1), valor de un material (1)
- `eval_v1k:thinking`: declaración de un material (1), valor de un material (1)
- `eval_v1k:rlm`: ninguno
- `eval_v2k:base`: precio FOB (1), valor de un material (1)
- `eval_v2k:thinking`: valor de un material (1)
- `eval_v2k:rlm`: ninguno

## Cómo falla (tareas en las que aparece cada cosa)

|  | `eval_v1k:base` | `eval_v1k:thinking` | `eval_v1k:rlm` | `eval_v2k:base` | `eval_v2k:thinking` | `eval_v2k:rlm` |
|---|---|---|---|---|---|---|
| alguna herramienta devolvió error | 0 | 0 | 0 | 0 | 1 | 0 |
| llamada sin etiquetas recuperada | 1 | 3 | 0 | 0 | 0 | 0 |
| memoria comprimida | 0 | 0 | 0 | 0 | 1 | 0 |
| recordatorio de final_answer | 0 | 4 | 6 | 0 | 5 | 6 |
| sin final_answer | 0 | 4 | 6 | 0 | 0 | 6 |

## Memoria

| Ejecución | Compresiones | Tokens antes (media) | Después (media) |
|---|---|---|---|
| `eval_v2k:thinking` | 1 | 6221 | 5143 |

