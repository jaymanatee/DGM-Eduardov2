# Fase 4 · Análisis de las ejecuciones

## Resumen

| Ejecución | Modelo | Éxito | Pasos | Tokens | s/tarea | Argumentos exactos | Acierta con argumentos mal |
|---|---|---|---|---|---|---|---|
| `eval:base` | Qwen/Qwen3-1.7B (base, thinking off) | 35 % (7/20) | 3.0 | 11927 | 15 | 3/17 | 6 |
| `eval:thinking` | Qwen/Qwen3-1.7B (thinking, thinking on) | 40 % (8/20) | 3.1 | 14235 | 28 | 3/17 | 4 |
| `eval:rlm` | Qwen/Qwen3-1.7B+tool_use/weights/sft_think (rlm, thinking on) | 40 % (8/20) | 3.4 | 12920 | 21 | 10/17 | 1 |
| `eval_rlm_sft_nothink:rlm` | Qwen/Qwen3-1.7B+tool_use/weights/sft_nothink (rlm, thinking off) | 45 % (9/20) | 3.8 | 14316 | 22 | 11/17 | 0 |
| `eval_rlm_grpo_pps3:rlm` | Qwen/Qwen3-1.7B+rlm/weights/grpo_pps3 (rlm, thinking on) | 10 % (2/20) | 2.0 | 9197 | 58 | 0/17 | 1 |
| `eval_v1:base` | Qwen/Qwen3-1.7B (base, thinking off) | 55 % (11/20) | 4.0 | 16522 | 20 | 7/17 | 5 |
| `eval_v1:thinking` | Qwen/Qwen3-1.7B (thinking, thinking on) | 50 % (10/20) | 3.5 | 16917 | 35 | 6/17 | 5 |
| `eval_v1:rlm` | Qwen/Qwen3-1.7B+tool_use/weights/sft_think (rlm, thinking on) | 50 % (10/20) | 4.0 | 15872 | 27 | 13/17 | 1 |
| `eval_rlm_sft_nothink_v1:rlm` | Qwen/Qwen3-1.7B+tool_use/weights/sft_nothink (rlm, thinking off) | 60 % (12/20) | 4.2 | 16168 | 26 | 10/17 | 3 |
| `eval_rlm_grpo_pps3_v1:rlm` | Qwen/Qwen3-1.7B+rlm/weights/grpo_pps3 (rlm, thinking on) | 15 % (3/20) | 2.0 | 9306 | 58 | 1/17 | 1 |

*Argumentos exactos*: la llamada a `evaluate_origin` coincide con la de referencia (acuerdo, producto, precios y cada material). *Acierta con argumentos mal*: la tarea pasa su comprobación aunque los argumentos estén mal.

## Éxito por categoría

| Categoría | `eval:base` | `eval:thinking` | `eval:rlm` | `eval_rlm_sft_nothink:rlm` | `eval_rlm_grpo_pps3:rlm` | `eval_v1:base` | `eval_v1:thinking` | `eval_v1:rlm` | `eval_rlm_sft_nothink_v1:rlm` | `eval_rlm_grpo_pps3_v1:rlm` |
|---|---|---|---|---|---|---|---|---|---|---|
| cambio | 0 % | 0 % | 0 % | 33 % | 33 % | 100 % | 100 % | 67 % | 33 % | 0 % |
| codigo_no_material | 0 % | 0 % | 0 % | 0 % | 0 % | 100 % | 100 % | 100 % | 100 % | 0 % |
| conocimiento | 0 % | 0 % | 0 % | 0 % | 0 % | 0 % | 0 % | 0 % | 33 % | 0 % |
| declaracion | 67 % | 100 % | 67 % | 100 % | 0 % | 67 % | 67 % | 67 % | 100 % | 0 % |
| faltan_datos | 0 % | 50 % | 100 % | 50 % | 0 % | 50 % | 50 % | 50 % | 50 % | 0 % |
| imposible | 50 % | 50 % | 50 % | 0 % | 0 % | 50 % | 50 % | 100 % | 50 % | 0 % |
| larga | 100 % | 100 % | 0 % | 0 % | 0 % | 100 % | 100 % | 0 % | 0 % | 0 % |
| veredicto | 50 % | 50 % | 75 % | 75 % | 25 % | 50 % | 25 % | 50 % | 75 % | 75 % |
| vies | 100 % | 0 % | 0 % | 100 % | 0 % | 0 % | 0 % | 0 % | 100 % | 0 % |

## Tarea a tarea

| Tarea | `eval:base` | `eval:thinking` | `eval:rlm` | `eval_rlm_sft_nothink:rlm` | `eval_rlm_grpo_pps3:rlm` | `eval_v1:base` | `eval_v1:thinking` | `eval_v1:rlm` | `eval_rlm_sft_nothink_v1:rlm` | `eval_rlm_grpo_pps3_v1:rlm` |
|---|---|---|---|---|---|---|---|---|---|---|
| cambio-01 | ✗ ⚠ 2 | ✗ ⚠ 3 | ✗ ⚠ 3 | ✗ ⚠ 3 | ✗ ⚠ 2 | ✓ 4 | ✓ ⚠ 5 | ✓ 5 | ✗ 2 | ✗ 2 |
| cambio-02 | ✗ ⚠ 3 | ✗ ⚠ 3 | ✗ ⚠ 3 | ✗ ⚠ 3 | ✓ ⚠ 2 | ✓ 5 | ✓ 4 | ✓ 5 | ✗ 2 | ✗ 2 |
| cambio-03 | ✗ ⚠ 3 | ✗ ⚠ 3 | ✗ ⚠ 3 | ✓ 4 | ✗ 2 | ✓ 4 | ✓ ⚠ 5 | ✗ ⚠ 6 | ✓ 4 | ✗ 2 |
| veredicto-01 | ✓ ⚠ 2 | ✗ 1 | ✓ 3 | ✓ 4 | ✓ 2 | ✓ 2 | ✓ 2 | ✗ 3 | ✓ 4 | ✓ 3 |
| veredicto-02 | ✓ 2 | ✓ 2 | ✓ 3 | ✓ 3 | ✗ 2 | ✓ 2 | ✗ 4 | ✓ 3 | ✓ ⚠ 4 | ✓ 2 |
| veredicto-03 | ✗ ⚠ 3 | ✓ ⚠ 4 | ✗ ⚠ 4 | ✗ ⚠ 4 | ✗ 2 | ✗ ⚠ 3 | ✗ ⚠ 3 | ✗ ⚠ 5 | ✗ ⚠ 4 | ✓ ⚠ 2 |
| codigo-no-material-01 | ✗ ⚠ 3 | ✗ ⚠ 4 | ✗ 2 | ✗ 3 | ✗ 2 | ✓ ⚠ 2 | ✓ ⚠ 4 | ✓ 3 | ✓ 3 | ✗ 2 |
| veredicto-04 | ✗ ⚠ 4 | ✗ ⚠ 4 | ✓ ⚠ 3 | ✓ 2 | ✗ ⚠ 3 | ✗ ⚠ 3 | ✗ ⚠ 5 | ✓ ⚠ 3 | ✓ ⚠ 6 | ✗ 2 |
| declaracion-01 | ✓ ⚠ 3 | ✓ 4 | ✓ 4 | ✓ 4 | ✗ 1 | ✓ ⚠ 3 | ✗ 2 | ✓ 5 | ✓ 4 | ✗ 2 |
| declaracion-02 | ✓ ⚠ 4 | ✓ ⚠ 3 | ✗ 4 | ✓ 4 | ✗ 2 | ✓ ⚠ 4 | ✓ 3 | ✗ 4 | ✓ 5 | ✗ 2 |
| faltan-datos-01 | ✗ ⚠ 5 | ✗ ⚠ 4 | ✓ 4 | ✓ 5 | ✗ 2 | ✓ ⚠ 3 | ✓ ⚠ 4 | ✓ 4 | ✓ 5 | ✗ 1 |
| declaracion-03 | ✗ 2 | ✓ 2 | ✓ 3 | ✓ 4 | ✗ 2 | ✗ 3 | ✓ 4 | ✓ 4 | ✓ 4 | ✗ 2 |
| conocimiento-01 | ✗ 4 | ✗ 1 | ✗ 3 | ✗ 4 | ✗ 2 | ✗ 6 | ✗ 2 | ✗ 3 | ✗ 4 | ✗ 2 |
| conocimiento-02 | ✗ 2 | ✗ ⚠ 2 | ✗ 3 | ✗ 3 | ✗ 2 | ✗ 3 | ✗ 2 | ✗ 3 | ✓ ⚠ 6 | ✗ 2 |
| conocimiento-03 | ✗ ⚠ 2 | ✗ ⚠ 3 | ✗ 3 | ✗ 3 | ✗ 2 | ✗ ⚠ 2 | ✗ ⚠ 2 | ✗ 3 | ✗ 4 | ✗ 2 |
| vies-01 | ✓ ⚠ 5 | ✗ ⚠ 4 | ✗ ⚠ 5 | ✓ 4 | ✗ 2 | ✗ 10 | ✗ ⚠ 4 | ✗ ⚠ 4 | ✓ 4 | ✗ 2 |
| imposible-01 | ✓ ⚠ 3 | ✓ ⚠ 3 | ✓ 4 | ✗ 4 | ✗ 2 | ✗ ⚠ 6 | ✗ ⚠ 4 | ✓ 4 | ✓ 4 | ✗ 2 |
| imposible-02 | ✗ 3 | ✗ 3 | ✗ 3 | ✗ 5 | ✗ 2 | ✓ 2 | ✓ 1 | ✓ 3 | ✗ 2 | ✗ 2 |
| faltan-datos-02 | ✗ 3 | ✓ 3 | ✓ 4 | ✗ 7 | ✗ 2 | ✗ 6 | ✗ 3 | ✗ 4 | ✗ 5 | ✗ 3 |
| larga-01 | ✓ ⚠ 3 | ✓ ⚠ 6 | ✗ ⚠ 4 | ✗ ⚠ 3 | ✗ 2 | ✓ ⚠ 7 | ✓ ⚠ 8 | ✗ 7 | ✗ ⚠ 7 | ✗ 2 |

✓/✗ éxito y número de pasos; ⚠ argumentos de `evaluate_origin` distintos de la referencia.

## Errores en los argumentos de `evaluate_origin`

- `eval:base`: origen de un material (11), valor de un material (5), precio franco fábrica (3), subdivisión (1), operaciones (1), código del producto (1), materiales que sobran o faltan (1), declaración de un material (1)
- `eval:thinking`: origen de un material (16), valor de un material (5), declaración de un material (3), materiales que sobran o faltan (2), precio franco fábrica (2), no llamó a evaluate_origin (1), subdivisión (1), operaciones (1), código del producto (1)
- `eval:rlm`: valor de un material (5), subdivisión (1), precio franco fábrica (1), materiales que sobran o faltan (1), origen de un material (1)
- `eval_rlm_sft_nothink:rlm`: valor de un material (4), no llamó a evaluate_origin (2), subdivisión (1), declaración de un material (1)
- `eval_rlm_grpo_pps3:rlm`: no llamó a evaluate_origin (14), valor de un material (2), declaración de un material (1), código del producto (1), precio franco fábrica (1), operaciones (1), materiales que sobran o faltan (1)
- `eval_v1:base`: origen de un material (5), precio franco fábrica (3), declaración de un material (2), materiales que sobran o faltan (2), subdivisión (1), operaciones (1), no llamó a evaluate_origin (1), valor de un material (1)
- `eval_v1:thinking`: origen de un material (10), declaración de un material (3), precio franco fábrica (2), subdivisión (1), operaciones (1), materiales que sobran o faltan (1), no llamó a evaluate_origin (1)
- `eval_v1:rlm`: valor de un material (1), subdivisión (1), precio franco fábrica (1), materiales que sobran o faltan (1), origen de un material (1)
- `eval_rlm_sft_nothink_v1:rlm`: precio franco fábrica (3), no llamó a evaluate_origin (2), declaración de un material (2), subdivisión (1), materiales que sobran o faltan (1), valor de un material (1)
- `eval_rlm_grpo_pps3_v1:rlm`: no llamó a evaluate_origin (15), subdivisión (1)

## Cómo falla (tareas en las que aparece cada cosa)

|  | `eval:base` | `eval:thinking` | `eval:rlm` | `eval_rlm_sft_nothink:rlm` | `eval_rlm_grpo_pps3:rlm` | `eval_v1:base` | `eval_v1:thinking` | `eval_v1:rlm` | `eval_rlm_sft_nothink_v1:rlm` | `eval_rlm_grpo_pps3_v1:rlm` |
|---|---|---|---|---|---|---|---|---|---|---|
| alguna herramienta devolvió error | 11 | 8 | 11 | 8 | 2 | 13 | 11 | 12 | 11 | 2 |
| lista de materiales rechazada | 4 | 3 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| llamada repetida rechazada | 6 | 2 | 1 | 2 | 0 | 4 | 1 | 3 | 1 | 0 |
| llamada sin etiquetas recuperada | 8 | 5 | 2 | 2 | 4 | 7 | 6 | 2 | 0 | 4 |
| memoria comprimida | 0 | 0 | 0 | 0 | 0 | 2 | 2 | 0 | 0 | 0 |
| producto como material rechazado | 2 | 0 | 0 | 0 | 1 | 3 | 1 | 0 | 1 | 0 |
| recordatorio de final_answer | 1 | 7 | 18 | 19 | 17 | 1 | 11 | 20 | 20 | 17 |
| sin final_answer | 1 | 5 | 16 | 17 | 17 | 1 | 9 | 18 | 18 | 16 |
| sin respuesta (pasos agotados o atascado) | 6 | 2 | 1 | 1 | 0 | 5 | 0 | 0 | 0 | 0 |
| valor inventado rechazado | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |

## Memoria

| Ejecución | Compresiones | Tokens antes (media) | Después (media) |
|---|---|---|---|
| `eval_v1:base` | 2 | 6162 | 4588 |
| `eval_v1:thinking` | 2 | 7192 | 4876 |

