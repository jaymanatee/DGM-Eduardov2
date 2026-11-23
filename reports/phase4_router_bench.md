# Fase 4 · Análisis de las ejecuciones

## Resumen

| Ejecución | Modelo | Éxito | Pasos | Tokens | s/tarea | Argumentos exactos | Acierta con argumentos mal |
|---|---|---|---|---|---|---|---|
| `eval_v1:base` | Qwen/Qwen3-1.7B (base, thinking off) | 55 % (11/20) | 4.0 | 16522 | 20 | 7/17 | 5 |
| `eval_v1:thinking` | Qwen/Qwen3-1.7B (thinking, thinking on) | 50 % (10/20) | 3.5 | 16917 | 35 | 6/17 | 5 |
| `eval_v1:rlm` | Qwen/Qwen3-1.7B+tool_use/weights/sft_think (rlm, thinking on) | 50 % (10/20) | 4.0 | 15872 | 27 | 13/17 | 1 |
| `eval_v1b:base` | Qwen/Qwen3-1.7B (base, thinking off) | 50 % (10/20) | 3.6 | 14682 | 18 | 7/17 | 5 |
| `eval_v1b:thinking` | Qwen/Qwen3-1.7B (thinking, thinking on) | 55 % (11/20) | 3.6 | 17213 | 34 | 6/17 | 5 |
| `eval_v1b:rlm` | Qwen/Qwen3-1.7B+tool_use/weights/sft_think (rlm, thinking on) | 55 % (11/20) | 4.0 | 15839 | 26 | 13/17 | 1 |
| `eval_v1c:base` | Qwen/Qwen3-1.7B (base, thinking off) | 50 % (10/20) | 4.0 | 16535 | 19 | 7/17 | 5 |
| `eval_v1c:thinking` | Qwen/Qwen3-1.7B (thinking, thinking on) | 55 % (11/20) | 3.6 | 17247 | 36 | 6/17 | 6 |
| `eval_v1c:rlm` | Qwen/Qwen3-1.7B+tool_use/weights/sft_think (rlm, thinking on) | 60 % (12/20) | 3.9 | 15614 | 32 | 13/17 | 1 |
| `eval_v2:base` | Qwen/Qwen3-1.7B (base, thinking off) | 55 % (11/20) | 3.5 | 13583 | 17 | 5/17 | 8 |
| `eval_v2:thinking` | Qwen/Qwen3-1.7B (thinking, thinking on) | 55 % (11/20) | 4.1 | 19086 | 36 | 6/17 | 4 |
| `eval_v2:rlm` | Qwen/Qwen3-1.7B+tool_use/weights/sft_think (rlm, thinking on) | 65 % (13/20) | 4.2 | 16218 | 26 | 12/17 | 1 |
| `eval_rlm_sft_nothink_v1:rlm` | Qwen/Qwen3-1.7B+tool_use/weights/sft_nothink (rlm, thinking off) | 60 % (12/20) | 4.2 | 16168 | 26 | 10/17 | 3 |
| `eval_rlm_sft_nothink_v2:rlm` | Qwen/Qwen3-1.7B+tool_use/weights/sft_nothink (rlm, thinking off) | 55 % (11/20) | 3.9 | 14475 | 21 | 10/17 | 1 |

*Argumentos exactos*: la llamada a `evaluate_origin` coincide con la de referencia (acuerdo, producto, precios y cada material). *Acierta con argumentos mal*: la tarea pasa su comprobación aunque los argumentos estén mal.

## Éxito por categoría

| Categoría | `eval_v1:base` | `eval_v1:thinking` | `eval_v1:rlm` | `eval_v1b:base` | `eval_v1b:thinking` | `eval_v1b:rlm` | `eval_v1c:base` | `eval_v1c:thinking` | `eval_v1c:rlm` | `eval_v2:base` | `eval_v2:thinking` | `eval_v2:rlm` | `eval_rlm_sft_nothink_v1:rlm` | `eval_rlm_sft_nothink_v2:rlm` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cambio | 100 % | 100 % | 67 % | 67 % | 100 % | 33 % | 67 % | 100 % | 67 % | 67 % | 67 % | 33 % | 33 % | 67 % |
| codigo_no_material | 100 % | 100 % | 100 % | 100 % | 100 % | 100 % | 100 % | 100 % | 100 % | 100 % | 100 % | 100 % | 100 % | 100 % |
| conocimiento | 0 % | 0 % | 0 % | 0 % | 0 % | 33 % | 0 % | 0 % | 0 % | 67 % | 67 % | 67 % | 33 % | 33 % |
| declaracion | 67 % | 67 % | 67 % | 67 % | 67 % | 100 % | 67 % | 33 % | 100 % | 100 % | 100 % | 100 % | 100 % | 100 % |
| faltan_datos | 50 % | 50 % | 50 % | 50 % | 50 % | 50 % | 50 % | 50 % | 50 % | 50 % | 0 % | 50 % | 50 % | 50 % |
| imposible | 50 % | 50 % | 100 % | 50 % | 50 % | 100 % | 50 % | 50 % | 100 % | 0 % | 50 % | 50 % | 50 % | 0 % |
| larga | 100 % | 100 % | 0 % | 100 % | 100 % | 0 % | 100 % | 100 % | 0 % | 100 % | 0 % | 100 % | 0 % | 0 % |
| veredicto | 50 % | 25 % | 50 % | 50 % | 50 % | 50 % | 50 % | 75 % | 75 % | 25 % | 50 % | 75 % | 75 % | 50 % |
| vies | 0 % | 0 % | 0 % | 0 % | 0 % | 0 % | 0 % | 0 % | 0 % | 0 % | 0 % | 0 % | 100 % | 100 % |

## Tarea a tarea

| Tarea | `eval_v1:base` | `eval_v1:thinking` | `eval_v1:rlm` | `eval_v1b:base` | `eval_v1b:thinking` | `eval_v1b:rlm` | `eval_v1c:base` | `eval_v1c:thinking` | `eval_v1c:rlm` | `eval_v2:base` | `eval_v2:thinking` | `eval_v2:rlm` | `eval_rlm_sft_nothink_v1:rlm` | `eval_rlm_sft_nothink_v2:rlm` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cambio-01 | ✓ 4 | ✓ ⚠ 5 | ✓ 5 | ✓ 4 | ✓ ⚠ 5 | ✓ 5 | ✓ 4 | ✓ ⚠ 5 | ✓ 5 | ✓ ⚠ 4 | ✓ ⚠ 5 | ✓ 5 | ✗ 2 | ✗ 2 |
| cambio-02 | ✓ 5 | ✓ 4 | ✓ 5 | ✓ 5 | ✓ 5 | ✗ 5 | ✓ 5 | ✓ 5 | ✓ 5 | ✓ 5 | ✓ 5 | ✗ ⚠ 6 | ✗ 2 | ✓ 5 |
| cambio-03 | ✓ 4 | ✓ ⚠ 5 | ✗ ⚠ 6 | ✗ 4 | ✓ ⚠ 5 | ✗ ⚠ 6 | ✗ 4 | ✓ ⚠ 5 | ✗ ⚠ 6 | ✗ 4 | ✗ ⚠ 5 | ✗ ⚠ 5 | ✓ 4 | ✓ 4 |
| veredicto-01 | ✓ 2 | ✓ 2 | ✗ 3 | ✓ 2 | ✓ 2 | ✗ 3 | ✓ 2 | ✓ 2 | ✓ 3 | ✓ ⚠ 2 | ✓ 2 | ✓ 3 | ✓ 4 | ✓ 4 |
| veredicto-02 | ✓ 2 | ✗ 4 | ✓ 3 | ✓ 2 | ✓ 4 | ✓ 3 | ✓ 2 | ✓ 4 | ✓ 3 | ✗ 2 | ✓ 2 | ✓ 3 | ✓ ⚠ 4 | ✗ 3 |
| veredicto-03 | ✗ ⚠ 3 | ✗ ⚠ 3 | ✗ ⚠ 5 | ✗ ⚠ 3 | ✗ ⚠ 3 | ✗ ⚠ 5 | ✗ ⚠ 3 | ✗ ⚠ 3 | ✗ ⚠ 4 | ✗ ⚠ 3 | ✗ ⚠ 4 | ✗ ⚠ 4 | ✗ ⚠ 4 | ✗ ⚠ 4 |
| codigo-no-material-01 | ✓ ⚠ 2 | ✓ ⚠ 4 | ✓ 3 | ✓ ⚠ 2 | ✓ ⚠ 4 | ✓ 3 | ✓ ⚠ 2 | ✓ ⚠ 4 | ✓ 3 | ✓ ⚠ 2 | ✓ ⚠ 3 | ✓ 3 | ✓ 3 | ✓ 3 |
| veredicto-04 | ✗ ⚠ 3 | ✗ ⚠ 5 | ✓ ⚠ 3 | ✗ ⚠ 3 | ✗ ⚠ 5 | ✓ ⚠ 3 | ✗ ⚠ 3 | ✓ ⚠ 5 | ✓ ⚠ 3 | ✗ ⚠ 3 | ✗ ⚠ 2 | ✓ ⚠ 3 | ✓ ⚠ 6 | ✓ ⚠ 6 |
| declaracion-01 | ✓ ⚠ 3 | ✗ 2 | ✓ 5 | ✓ ⚠ 3 | ✗ 2 | ✓ 4 | ✓ ⚠ 3 | ✗ 2 | ✓ 4 | ✓ ⚠ 3 | ✓ ⚠ 4 | ✓ 4 | ✓ 4 | ✓ 4 |
| declaracion-02 | ✓ ⚠ 4 | ✓ 3 | ✗ 4 | ✓ ⚠ 4 | ✓ 4 | ✓ 4 | ✓ ⚠ 4 | ✓ 4 | ✓ 4 | ✓ ⚠ 4 | ✓ 4 | ✓ 4 | ✓ 5 | ✓ 5 |
| faltan-datos-01 | ✓ ⚠ 3 | ✓ ⚠ 4 | ✓ 4 | ✓ ⚠ 3 | ✓ ⚠ 4 | ✓ 4 | ✓ ⚠ 3 | ✓ ⚠ 4 | ✓ 4 | ✓ ⚠ 3 | ✗ ⚠ 6 | ✓ 4 | ✓ 5 | ✓ 5 |
| declaracion-03 | ✗ 3 | ✓ 4 | ✓ 4 | ✗ 3 | ✓ 3 | ✓ 4 | ✗ 3 | ✗ 3 | ✓ 3 | ✓ 3 | ✓ 2 | ✓ 3 | ✓ 4 | ✓ 4 |
| conocimiento-01 | ✗ 6 | ✗ 2 | ✗ 3 | ✗ 6 | ✗ 2 | ✗ 3 | ✗ 6 | ✗ 2 | ✗ 3 | ✓ 4 | ✓ 4 | ✗ 3 | ✗ 4 | ✗ 3 |
| conocimiento-02 | ✗ 3 | ✗ 2 | ✗ 3 | ✗ 3 | ✗ 2 | ✓ 3 | ✗ 3 | ✗ 2 | ✗ 3 | ✗ 2 | ✗ 3 | ✓ 4 | ✓ ⚠ 6 | ✗ 3 |
| conocimiento-03 | ✗ ⚠ 2 | ✗ ⚠ 2 | ✗ 3 | ✗ ⚠ 2 | ✗ ⚠ 2 | ✗ 3 | ✗ ⚠ 2 | ✗ ⚠ 2 | ✗ 3 | ✓ ⚠ 3 | ✓ ⚠ 10 | ✓ 5 | ✗ 4 | ✓ 4 |
| vies-01 | ✗ 10 | ✗ ⚠ 4 | ✗ ⚠ 4 | ✗ ⚠ 3 | ✗ ⚠ 4 | ✗ ⚠ 5 | ✗ 10 | ✗ ⚠ 4 | ✗ ⚠ 4 | ✗ ⚠ 3 | ✗ ⚠ 4 | ✗ ⚠ 4 | ✓ 4 | ✓ 4 |
| imposible-01 | ✗ ⚠ 6 | ✗ ⚠ 4 | ✓ 4 | ✗ ⚠ 6 | ✗ ⚠ 4 | ✓ 4 | ✗ ⚠ 6 | ✗ ⚠ 4 | ✓ 4 | ✗ 3 | ✗ 4 | ✓ 4 | ✓ 4 | ✗ 2 |
| imposible-02 | ✓ 2 | ✓ 1 | ✓ 3 | ✓ 2 | ✓ 1 | ✓ 3 | ✓ 2 | ✓ 1 | ✓ 3 | ✗ 4 | ✓ 3 | ✗ 4 | ✗ 2 | ✗ 3 |
| faltan-datos-02 | ✗ 6 | ✗ 3 | ✗ 4 | ✗ 6 | ✗ 3 | ✗ 4 | ✗ 6 | ✗ 3 | ✗ 4 | ✗ 6 | ✗ 2 | ✗ 4 | ✗ 5 | ✗ 2 |
| larga-01 | ✓ ⚠ 7 | ✓ ⚠ 8 | ✗ 7 | ✓ ⚠ 7 | ✓ ⚠ 8 | ✗ 7 | ✓ ⚠ 7 | ✓ ⚠ 8 | ✗ 7 | ✓ ⚠ 7 | ✗ ⚠ 8 | ✓ 8 | ✗ ⚠ 7 | ✗ ⚠ 7 |

✓/✗ éxito y número de pasos; ⚠ argumentos de `evaluate_origin` distintos de la referencia.

## Errores en los argumentos de `evaluate_origin`

- `eval_v1:base`: origen de un material (5), precio franco fábrica (3), declaración de un material (2), materiales que sobran o faltan (2), subdivisión (1), operaciones (1), no llamó a evaluate_origin (1), valor de un material (1)
- `eval_v1:thinking`: origen de un material (10), declaración de un material (3), precio franco fábrica (2), subdivisión (1), operaciones (1), materiales que sobran o faltan (1), no llamó a evaluate_origin (1)
- `eval_v1:rlm`: valor de un material (1), subdivisión (1), precio franco fábrica (1), materiales que sobran o faltan (1), origen de un material (1)
- `eval_v1b:base`: origen de un material (6), precio franco fábrica (3), declaración de un material (2), materiales que sobran o faltan (2), subdivisión (1), operaciones (1), valor de un material (1)
- `eval_v1b:thinking`: origen de un material (10), declaración de un material (3), precio franco fábrica (2), subdivisión (1), operaciones (1), materiales que sobran o faltan (1), no llamó a evaluate_origin (1)
- `eval_v1b:rlm`: valor de un material (1), subdivisión (1), precio franco fábrica (1), materiales que sobran o faltan (1), origen de un material (1), declaración de un material (1)
- `eval_v1c:base`: origen de un material (5), precio franco fábrica (3), declaración de un material (2), materiales que sobran o faltan (2), subdivisión (1), operaciones (1), no llamó a evaluate_origin (1), valor de un material (1)
- `eval_v1c:thinking`: origen de un material (10), declaración de un material (3), precio franco fábrica (2), subdivisión (1), operaciones (1), materiales que sobran o faltan (1), no llamó a evaluate_origin (1)
- `eval_v1c:rlm`: valor de un material (1), subdivisión (1), precio franco fábrica (1), materiales que sobran o faltan (1), origen de un material (1)
- `eval_v2:base`: origen de un material (9), precio franco fábrica (3), subdivisión (1), operaciones (1), materiales que sobran o faltan (1), no llamó a evaluate_origin (1)
- `eval_v2:thinking`: origen de un material (5), precio franco fábrica (3), valor de un material (2), subdivisión (1), operaciones (1), materiales que sobran o faltan (1), no llamó a evaluate_origin (1)
- `eval_v2:rlm`: valor de un material (2), subdivisión (1), precio franco fábrica (1), materiales que sobran o faltan (1), origen de un material (1)
- `eval_rlm_sft_nothink_v1:rlm`: precio franco fábrica (3), no llamó a evaluate_origin (2), declaración de un material (2), subdivisión (1), materiales que sobran o faltan (1), valor de un material (1)
- `eval_rlm_sft_nothink_v2:rlm`: no llamó a evaluate_origin (4), declaración de un material (2), subdivisión (1), precio franco fábrica (1), materiales que sobran o faltan (1), valor de un material (1)

## Cómo falla (tareas en las que aparece cada cosa)

|  | `eval_v1:base` | `eval_v1:thinking` | `eval_v1:rlm` | `eval_v1b:base` | `eval_v1b:thinking` | `eval_v1b:rlm` | `eval_v1c:base` | `eval_v1c:thinking` | `eval_v1c:rlm` | `eval_v2:base` | `eval_v2:thinking` | `eval_v2:rlm` | `eval_rlm_sft_nothink_v1:rlm` | `eval_rlm_sft_nothink_v2:rlm` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| alguna herramienta devolvió error | 13 | 11 | 12 | 12 | 11 | 10 | 13 | 11 | 9 | 12 | 10 | 9 | 11 | 8 |
| llamada repetida rechazada | 4 | 1 | 3 | 4 | 1 | 3 | 4 | 1 | 1 | 2 | 1 | 1 | 1 | 1 |
| llamada sin etiquetas recuperada | 7 | 6 | 2 | 7 | 5 | 1 | 7 | 6 | 2 | 9 | 3 | 0 | 0 | 0 |
| memoria comprimida | 2 | 2 | 0 | 1 | 2 | 0 | 2 | 2 | 0 | 1 | 4 | 0 | 0 | 0 |
| producto como material rechazado | 3 | 1 | 0 | 3 | 1 | 0 | 3 | 1 | 0 | 2 | 1 | 0 | 1 | 1 |
| recordatorio de final_answer | 1 | 11 | 20 | 1 | 12 | 20 | 1 | 12 | 20 | 1 | 12 | 19 | 20 | 20 |
| sin final_answer | 1 | 9 | 18 | 1 | 9 | 19 | 1 | 9 | 19 | 1 | 10 | 19 | 18 | 16 |
| sin respuesta (pasos agotados o atascado) | 5 | 0 | 0 | 4 | 0 | 0 | 5 | 0 | 0 | 2 | 0 | 1 | 0 | 0 |
| valor inventado rechazado | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 2 | 0 | 0 |

## Memoria

| Ejecución | Compresiones | Tokens antes (media) | Después (media) |
|---|---|---|---|
| `eval_v1:base` | 2 | 6162 | 4588 |
| `eval_v1:thinking` | 2 | 7192 | 4876 |
| `eval_v1b:base` | 1 | 6210 | 5154 |
| `eval_v1b:thinking` | 2 | 7180 | 4864 |
| `eval_v1c:base` | 2 | 6194 | 4600 |
| `eval_v1c:thinking` | 2 | 7172 | 4856 |
| `eval_v2:base` | 1 | 6212 | 5156 |
| `eval_v2:thinking` | 6 | 6437 | 4641 |

