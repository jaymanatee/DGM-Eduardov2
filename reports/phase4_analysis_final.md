# Fase 4 · Análisis de las ejecuciones

## Resumen

| Ejecución | Modelo | Éxito | Pasos | Tokens | s/tarea | Argumentos exactos | Acierta con argumentos mal |
|---|---|---|---|---|---|---|---|
| `codeagent_v1:code_base` | smolagents CodeAgent, Qwen/Qwen3-1.7B (base) | 15 % (3/20) | 6.5 | 46551 | 58 | 1/17 | 1 |
| `codeagent_v1:code_rlm` | smolagents CodeAgent, Qwen/Qwen3-1.7B+tool_use/weights/sft_think (rlm) | 0 % (0/20) | 10.8 | 70915 | 118 | 2/17 | 0 |
| `eval:base` | Qwen/Qwen3-1.7B (base, thinking off) | 35 % (7/20) | 3.0 | 11927 | 15 | 3/17 | 6 |
| `eval:thinking` | Qwen/Qwen3-1.7B (thinking, thinking on) | 40 % (8/20) | 3.1 | 14235 | 28 | 3/17 | 4 |
| `eval:rlm` | Qwen/Qwen3-1.7B+tool_use/weights/sft_think (rlm, thinking on) | 40 % (8/20) | 3.4 | 12920 | 21 | 10/17 | 1 |
| `eval_rlm_grpo_pps3:rlm` | Qwen/Qwen3-1.7B+rlm/weights/grpo_pps3 (rlm, thinking on) | 10 % (2/20) | 2.0 | 9197 | 58 | 0/17 | 1 |
| `eval_rlm_grpo_pps3_v1:rlm` | Qwen/Qwen3-1.7B+rlm/weights/grpo_pps3 (rlm, thinking on) | 15 % (3/20) | 2.0 | 9306 | 58 | 1/17 | 1 |
| `eval_rlm_grpo_pps3_v1b:rlm` | Qwen/Qwen3-1.7B+rlm/weights/grpo_pps3 (rlm, thinking on) | 15 % (3/20) | 2.0 | 9301 | 57 | 1/17 | 1 |
| `eval_rlm_grpo_pps3_v1c:rlm` | Qwen/Qwen3-1.7B+rlm/weights/grpo_pps3 (rlm, thinking on) | 15 % (3/20) | 2.0 | 9304 | 56 | 1/17 | 1 |
| `eval_rlm_sft_nothink:rlm` | Qwen/Qwen3-1.7B+tool_use/weights/sft_nothink (rlm, thinking off) | 45 % (9/20) | 3.8 | 14316 | 22 | 11/17 | 0 |
| `eval_rlm_sft_nothink_v1:rlm` | Qwen/Qwen3-1.7B+tool_use/weights/sft_nothink (rlm, thinking off) | 60 % (12/20) | 4.2 | 16168 | 26 | 10/17 | 3 |
| `eval_rlm_sft_nothink_v1b:rlm` | Qwen/Qwen3-1.7B+tool_use/weights/sft_nothink (rlm, thinking off) | 50 % (10/20) | 4.2 | 16173 | 24 | 10/17 | 1 |
| `eval_rlm_sft_nothink_v1c:rlm` | Qwen/Qwen3-1.7B+tool_use/weights/sft_nothink (rlm, thinking off) | 60 % (12/20) | 4.1 | 15969 | 24 | 10/17 | 3 |
| `eval_v1:base` | Qwen/Qwen3-1.7B (base, thinking off) | 55 % (11/20) | 4.0 | 16522 | 20 | 7/17 | 5 |
| `eval_v1:thinking` | Qwen/Qwen3-1.7B (thinking, thinking on) | 50 % (10/20) | 3.5 | 16917 | 35 | 6/17 | 5 |
| `eval_v1:rlm` | Qwen/Qwen3-1.7B+tool_use/weights/sft_think (rlm, thinking on) | 50 % (10/20) | 4.0 | 15872 | 27 | 13/17 | 1 |
| `eval_v1b:base` | Qwen/Qwen3-1.7B (base, thinking off) | 50 % (10/20) | 3.6 | 14682 | 18 | 7/17 | 5 |
| `eval_v1b:thinking` | Qwen/Qwen3-1.7B (thinking, thinking on) | 55 % (11/20) | 3.6 | 17213 | 34 | 6/17 | 5 |
| `eval_v1b:rlm` | Qwen/Qwen3-1.7B+tool_use/weights/sft_think (rlm, thinking on) | 55 % (11/20) | 4.0 | 15839 | 26 | 13/17 | 1 |
| `eval_v1c:base` | Qwen/Qwen3-1.7B (base, thinking off) | 50 % (10/20) | 4.0 | 16535 | 19 | 7/17 | 5 |
| `eval_v1c:thinking` | Qwen/Qwen3-1.7B (thinking, thinking on) | 55 % (11/20) | 3.6 | 17247 | 36 | 6/17 | 6 |
| `eval_v1c:rlm` | Qwen/Qwen3-1.7B+tool_use/weights/sft_think (rlm, thinking on) | 60 % (12/20) | 3.9 | 15614 | 32 | 13/17 | 1 |
| `memory_0_v1:rlm` | Qwen/Qwen3-1.7B+tool_use/weights/sft_think (rlm, thinking on) | 70 % (14/20) | 4.1 | 16263 | 27 | 13/17 | 1 |
| `memory_4000_v1:rlm` | Qwen/Qwen3-1.7B+tool_use/weights/sft_think (rlm, thinking on) | 55 % (11/20) | 3.8 | 14933 | 30 | 12/17 | 2 |

*Argumentos exactos*: la llamada a `evaluate_origin` coincide con la de referencia (acuerdo, producto, precios y cada material). *Acierta con argumentos mal*: la tarea pasa su comprobación aunque los argumentos estén mal.

## Éxito por categoría

| Categoría | `codeagent_v1:code_base` | `codeagent_v1:code_rlm` | `eval:base` | `eval:thinking` | `eval:rlm` | `eval_rlm_grpo_pps3:rlm` | `eval_rlm_grpo_pps3_v1:rlm` | `eval_rlm_grpo_pps3_v1b:rlm` | `eval_rlm_grpo_pps3_v1c:rlm` | `eval_rlm_sft_nothink:rlm` | `eval_rlm_sft_nothink_v1:rlm` | `eval_rlm_sft_nothink_v1b:rlm` | `eval_rlm_sft_nothink_v1c:rlm` | `eval_v1:base` | `eval_v1:thinking` | `eval_v1:rlm` | `eval_v1b:base` | `eval_v1b:thinking` | `eval_v1b:rlm` | `eval_v1c:base` | `eval_v1c:thinking` | `eval_v1c:rlm` | `memory_0_v1:rlm` | `memory_4000_v1:rlm` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cambio | 0 % | 0 % | 0 % | 0 % | 0 % | 33 % | 0 % | 0 % | 0 % | 33 % | 33 % | 33 % | 33 % | 100 % | 100 % | 67 % | 67 % | 100 % | 33 % | 67 % | 100 % | 67 % | 67 % | 67 % |
| codigo_no_material | 0 % | 0 % | 0 % | 0 % | 0 % | 0 % | 0 % | 0 % | 0 % | 0 % | 100 % | 100 % | 100 % | 100 % | 100 % | 100 % | 100 % | 100 % | 100 % | 100 % | 100 % | 100 % | 100 % | 100 % |
| conocimiento | 33 % | 0 % | 0 % | 0 % | 0 % | 0 % | 0 % | 0 % | 0 % | 0 % | 33 % | 0 % | 33 % | 0 % | 0 % | 0 % | 0 % | 0 % | 33 % | 0 % | 0 % | 0 % | 33 % | 0 % |
| declaracion | 0 % | 0 % | 67 % | 100 % | 67 % | 0 % | 0 % | 0 % | 0 % | 100 % | 100 % | 100 % | 100 % | 67 % | 67 % | 67 % | 67 % | 67 % | 100 % | 67 % | 33 % | 100 % | 100 % | 67 % |
| faltan_datos | 0 % | 0 % | 0 % | 50 % | 100 % | 0 % | 0 % | 0 % | 0 % | 50 % | 50 % | 50 % | 50 % | 50 % | 50 % | 50 % | 50 % | 50 % | 50 % | 50 % | 50 % | 50 % | 50 % | 50 % |
| imposible | 50 % | 0 % | 50 % | 50 % | 50 % | 0 % | 0 % | 0 % | 0 % | 0 % | 50 % | 50 % | 50 % | 50 % | 50 % | 100 % | 50 % | 50 % | 100 % | 50 % | 50 % | 100 % | 100 % | 100 % |
| larga | 100 % | 0 % | 100 % | 100 % | 0 % | 0 % | 0 % | 0 % | 0 % | 0 % | 0 % | 0 % | 0 % | 100 % | 100 % | 0 % | 100 % | 100 % | 0 % | 100 % | 100 % | 0 % | 100 % | 0 % |
| veredicto | 0 % | 0 % | 50 % | 50 % | 75 % | 25 % | 75 % | 75 % | 75 % | 75 % | 75 % | 50 % | 75 % | 50 % | 25 % | 50 % | 50 % | 50 % | 50 % | 50 % | 75 % | 75 % | 75 % | 75 % |
| vies | 0 % | 0 % | 100 % | 0 % | 0 % | 0 % | 0 % | 0 % | 0 % | 100 % | 100 % | 100 % | 100 % | 0 % | 0 % | 0 % | 0 % | 0 % | 0 % | 0 % | 0 % | 0 % | 0 % | 0 % |

## Tarea a tarea

| Tarea | `codeagent_v1:code_base` | `codeagent_v1:code_rlm` | `eval:base` | `eval:thinking` | `eval:rlm` | `eval_rlm_grpo_pps3:rlm` | `eval_rlm_grpo_pps3_v1:rlm` | `eval_rlm_grpo_pps3_v1b:rlm` | `eval_rlm_grpo_pps3_v1c:rlm` | `eval_rlm_sft_nothink:rlm` | `eval_rlm_sft_nothink_v1:rlm` | `eval_rlm_sft_nothink_v1b:rlm` | `eval_rlm_sft_nothink_v1c:rlm` | `eval_v1:base` | `eval_v1:thinking` | `eval_v1:rlm` | `eval_v1b:base` | `eval_v1b:thinking` | `eval_v1b:rlm` | `eval_v1c:base` | `eval_v1c:thinking` | `eval_v1c:rlm` | `memory_0_v1:rlm` | `memory_4000_v1:rlm` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cambio-01 | ✗ ⚠ 11 | ✗ ⚠ 11 | ✗ ⚠ 2 | ✗ ⚠ 3 | ✗ ⚠ 3 | ✗ ⚠ 2 | ✗ 2 | ✗ 2 | ✗ 2 | ✗ ⚠ 3 | ✗ 2 | ✗ 2 | ✗ 2 | ✓ 4 | ✓ ⚠ 5 | ✓ 5 | ✓ 4 | ✓ ⚠ 5 | ✓ 5 | ✓ 4 | ✓ ⚠ 5 | ✓ 5 | ✓ 5 | ✓ 5 |
| cambio-02 | ✗ ⚠ 11 | ✗ ⚠ 11 | ✗ ⚠ 3 | ✗ ⚠ 3 | ✗ ⚠ 3 | ✓ ⚠ 2 | ✗ 2 | ✗ 2 | ✗ 2 | ✗ ⚠ 3 | ✗ 2 | ✗ 2 | ✗ 2 | ✓ 5 | ✓ 4 | ✓ 5 | ✓ 5 | ✓ 5 | ✗ 5 | ✓ 5 | ✓ 5 | ✓ 5 | ✓ 5 | ✓ 5 |
| cambio-03 | ✗ ⚠ 11 | ✗ 11 | ✗ ⚠ 3 | ✗ ⚠ 3 | ✗ ⚠ 3 | ✗ 2 | ✗ 2 | ✗ 2 | ✗ 2 | ✓ 4 | ✓ 4 | ✓ 4 | ✓ 4 | ✓ 4 | ✓ ⚠ 5 | ✗ ⚠ 6 | ✗ 4 | ✓ ⚠ 5 | ✗ ⚠ 6 | ✗ 4 | ✓ ⚠ 5 | ✗ ⚠ 6 | ✗ ⚠ 6 | ✗ ⚠ 5 |
| veredicto-01 | ✗ ⚠ 2 | ✗ 11 | ✓ ⚠ 2 | ✗ 1 | ✓ 3 | ✓ 2 | ✓ 3 | ✓ 3 | ✓ 3 | ✓ 4 | ✓ 4 | ✓ 3 | ✓ 3 | ✓ 2 | ✓ 2 | ✗ 3 | ✓ 2 | ✓ 2 | ✗ 3 | ✓ 2 | ✓ 2 | ✓ 3 | ✓ 3 | ✗ 3 |
| veredicto-02 | ✗ ⚠ 2 | ✗ 11 | ✓ 2 | ✓ 2 | ✓ 3 | ✗ 2 | ✓ 2 | ✓ 2 | ✓ 2 | ✓ 3 | ✓ ⚠ 4 | ✓ ⚠ 4 | ✓ ⚠ 4 | ✓ 2 | ✗ 4 | ✓ 3 | ✓ 2 | ✓ 4 | ✓ 3 | ✓ 2 | ✓ 4 | ✓ 3 | ✓ 3 | ✓ 3 |
| veredicto-03 | ✗ ⚠ 3 | ✗ 11 | ✗ ⚠ 3 | ✓ ⚠ 4 | ✗ ⚠ 4 | ✗ 2 | ✓ ⚠ 2 | ✓ ⚠ 2 | ✓ ⚠ 2 | ✗ ⚠ 4 | ✗ ⚠ 4 | ✗ ⚠ 4 | ✗ ⚠ 4 | ✗ ⚠ 3 | ✗ ⚠ 3 | ✗ ⚠ 5 | ✗ ⚠ 3 | ✗ ⚠ 3 | ✗ ⚠ 5 | ✗ ⚠ 3 | ✗ ⚠ 3 | ✗ ⚠ 4 | ✗ ⚠ 5 | ✓ ⚠ 5 |
| codigo-no-material-01 | ✗ ⚠ 4 | ✗ ⚠ 11 | ✗ ⚠ 3 | ✗ ⚠ 4 | ✗ 2 | ✗ 2 | ✗ 2 | ✗ 2 | ✗ 2 | ✗ 3 | ✓ 3 | ✓ 3 | ✓ 3 | ✓ ⚠ 2 | ✓ ⚠ 4 | ✓ 3 | ✓ ⚠ 2 | ✓ ⚠ 4 | ✓ 3 | ✓ ⚠ 2 | ✓ ⚠ 4 | ✓ 3 | ✓ 3 | ✓ 3 |
| veredicto-04 | ✗ ⚠ 11 | ✗ ⚠ 11 | ✗ ⚠ 4 | ✗ ⚠ 4 | ✓ ⚠ 3 | ✗ ⚠ 3 | ✗ 2 | ✗ 2 | ✗ 2 | ✓ 2 | ✓ ⚠ 6 | ✗ ⚠ 6 | ✓ ⚠ 6 | ✗ ⚠ 3 | ✗ ⚠ 5 | ✓ ⚠ 3 | ✗ ⚠ 3 | ✗ ⚠ 5 | ✓ ⚠ 3 | ✗ ⚠ 3 | ✓ ⚠ 5 | ✓ ⚠ 3 | ✓ ⚠ 3 | ✓ ⚠ 3 |
| declaracion-01 | ✗ ⚠ 6 | ✗ 11 | ✓ ⚠ 3 | ✓ 4 | ✓ 4 | ✗ 1 | ✗ 2 | ✗ 2 | ✗ 2 | ✓ 4 | ✓ 4 | ✓ 4 | ✓ 4 | ✓ ⚠ 3 | ✗ 2 | ✓ 5 | ✓ ⚠ 3 | ✗ 2 | ✓ 4 | ✓ ⚠ 3 | ✗ 2 | ✓ 4 | ✓ 4 | ✓ 4 |
| declaracion-02 | ✗ ⚠ 5 | ✗ 11 | ✓ ⚠ 4 | ✓ ⚠ 3 | ✗ 4 | ✗ 2 | ✗ 2 | ✗ 2 | ✗ 2 | ✓ 4 | ✓ 5 | ✓ 5 | ✓ 5 | ✓ ⚠ 4 | ✓ 3 | ✗ 4 | ✓ ⚠ 4 | ✓ 4 | ✓ 4 | ✓ ⚠ 4 | ✓ 4 | ✓ 4 | ✓ 4 | ✗ 3 |
| faltan-datos-01 | ✗ ⚠ 11 | ✗ 11 | ✗ ⚠ 5 | ✗ ⚠ 4 | ✓ 4 | ✗ 2 | ✗ 1 | ✗ 1 | ✗ 1 | ✓ 5 | ✓ 5 | ✓ 5 | ✓ 5 | ✓ ⚠ 3 | ✓ ⚠ 4 | ✓ 4 | ✓ ⚠ 3 | ✓ ⚠ 4 | ✓ 4 | ✓ ⚠ 3 | ✓ ⚠ 4 | ✓ 4 | ✓ 4 | ✓ 4 |
| declaracion-03 | ✗ ⚠ 3 | ✗ 11 | ✗ 2 | ✓ 2 | ✓ 3 | ✗ 2 | ✗ 2 | ✗ 2 | ✗ 2 | ✓ 4 | ✓ 4 | ✓ 4 | ✓ 4 | ✗ 3 | ✓ 4 | ✓ 4 | ✗ 3 | ✓ 3 | ✓ 4 | ✗ 3 | ✗ 3 | ✓ 3 | ✓ 3 | ✓ 3 |
| conocimiento-01 | ✓ 4 | ✗ 11 | ✗ 4 | ✗ 1 | ✗ 3 | ✗ 2 | ✗ 2 | ✗ 2 | ✗ 2 | ✗ 4 | ✗ 4 | ✗ 4 | ✗ 4 | ✗ 6 | ✗ 2 | ✗ 3 | ✗ 6 | ✗ 2 | ✗ 3 | ✗ 6 | ✗ 2 | ✗ 3 | ✗ 3 | ✗ 3 |
| conocimiento-02 | ✗ 11 | ✗ 11 | ✗ 2 | ✗ ⚠ 2 | ✗ 3 | ✗ 2 | ✗ 2 | ✗ 2 | ✗ 2 | ✗ 3 | ✓ ⚠ 6 | ✗ ⚠ 6 | ✓ ⚠ 6 | ✗ 3 | ✗ 2 | ✗ 3 | ✗ 3 | ✗ 2 | ✓ 3 | ✗ 3 | ✗ 2 | ✗ 3 | ✓ 3 | ✗ 3 |
| conocimiento-03 | ✗ ⚠ 3 | ✗ ⚠ 11 | ✗ ⚠ 2 | ✗ ⚠ 3 | ✗ 3 | ✗ 2 | ✗ 2 | ✗ 2 | ✗ 2 | ✗ 3 | ✗ 4 | ✗ 5 | ✗ 4 | ✗ ⚠ 2 | ✗ ⚠ 2 | ✗ 3 | ✗ ⚠ 2 | ✗ ⚠ 2 | ✗ 3 | ✗ ⚠ 2 | ✗ ⚠ 2 | ✗ 3 | ✗ 3 | ✗ 4 |
| vies-01 | ✗ ⚠ 3 | ✗ ⚠ 11 | ✓ ⚠ 5 | ✗ ⚠ 4 | ✗ ⚠ 5 | ✗ 2 | ✗ 2 | ✗ 2 | ✗ 2 | ✓ 4 | ✓ 4 | ✓ 4 | ✓ 4 | ✗ 10 | ✗ ⚠ 4 | ✗ ⚠ 4 | ✗ ⚠ 3 | ✗ ⚠ 4 | ✗ ⚠ 5 | ✗ 10 | ✗ ⚠ 4 | ✗ ⚠ 4 | ✗ ⚠ 5 | ✗ ⚠ 4 |
| imposible-01 | ✗ ⚠ 11 | ✗ 11 | ✓ ⚠ 3 | ✓ ⚠ 3 | ✓ 4 | ✗ 2 | ✗ 2 | ✗ 2 | ✗ 2 | ✗ 4 | ✓ 4 | ✓ 4 | ✓ 4 | ✗ ⚠ 6 | ✗ ⚠ 4 | ✓ 4 | ✗ ⚠ 6 | ✗ ⚠ 4 | ✓ 4 | ✗ ⚠ 6 | ✗ ⚠ 4 | ✓ 4 | ✓ 4 | ✓ 4 |
| imposible-02 | ✓ 3 | ✗ 11 | ✗ 3 | ✗ 3 | ✗ 3 | ✗ 2 | ✗ 2 | ✗ 2 | ✗ 2 | ✗ 5 | ✗ 2 | ✗ 2 | ✗ 2 | ✓ 2 | ✓ 1 | ✓ 3 | ✓ 2 | ✓ 1 | ✓ 3 | ✓ 2 | ✓ 1 | ✓ 3 | ✓ 3 | ✓ 3 |
| faltan-datos-02 | ✗ 11 | ✗ 11 | ✗ 3 | ✓ 3 | ✓ 4 | ✗ 2 | ✗ 3 | ✗ 3 | ✗ 3 | ✗ 7 | ✗ 5 | ✗ 5 | ✗ 5 | ✗ 6 | ✗ 3 | ✗ 4 | ✗ 6 | ✗ 3 | ✗ 4 | ✗ 6 | ✗ 3 | ✗ 4 | ✗ 4 | ✗ 4 |
| larga-01 | ✓ ⚠ 3 | ✗ ⚠ 7 | ✓ ⚠ 3 | ✓ ⚠ 6 | ✗ ⚠ 4 | ✗ 2 | ✗ 2 | ✗ 2 | ✗ 2 | ✗ ⚠ 3 | ✗ ⚠ 7 | ✗ ⚠ 7 | ✗ ⚠ 7 | ✓ ⚠ 7 | ✓ ⚠ 8 | ✗ 7 | ✓ ⚠ 7 | ✓ ⚠ 8 | ✗ 7 | ✓ ⚠ 7 | ✓ ⚠ 8 | ✗ 7 | ✓ 9 | ✗ ⚠ 5 |

✓/✗ éxito y número de pasos; ⚠ argumentos de `evaluate_origin` distintos de la referencia.

## Errores en los argumentos de `evaluate_origin`

- `codeagent_v1:code_base`: origen de un material (44), declaración de un material (39), valor de un material (7), precio franco fábrica (4), subdivisión (1), operaciones (1), materiales que sobran o faltan (1)
- `codeagent_v1:code_rlm`: no llamó a evaluate_origin (8), origen de un material (5), valor de un material (4), declaración de un material (3), precio franco fábrica (1), operaciones (1)
- `eval:base`: origen de un material (11), valor de un material (5), precio franco fábrica (3), subdivisión (1), operaciones (1), código del producto (1), materiales que sobran o faltan (1), declaración de un material (1)
- `eval:thinking`: origen de un material (16), valor de un material (5), declaración de un material (3), materiales que sobran o faltan (2), precio franco fábrica (2), no llamó a evaluate_origin (1), subdivisión (1), operaciones (1), código del producto (1)
- `eval:rlm`: valor de un material (5), subdivisión (1), precio franco fábrica (1), materiales que sobran o faltan (1), origen de un material (1)
- `eval_rlm_grpo_pps3:rlm`: no llamó a evaluate_origin (14), valor de un material (2), declaración de un material (1), código del producto (1), precio franco fábrica (1), operaciones (1), materiales que sobran o faltan (1)
- `eval_rlm_grpo_pps3_v1:rlm`: no llamó a evaluate_origin (15), subdivisión (1)
- `eval_rlm_grpo_pps3_v1b:rlm`: no llamó a evaluate_origin (15), subdivisión (1)
- `eval_rlm_grpo_pps3_v1c:rlm`: no llamó a evaluate_origin (15), subdivisión (1)
- `eval_rlm_sft_nothink:rlm`: valor de un material (4), no llamó a evaluate_origin (2), subdivisión (1), declaración de un material (1)
- `eval_rlm_sft_nothink_v1:rlm`: precio franco fábrica (3), no llamó a evaluate_origin (2), declaración de un material (2), subdivisión (1), materiales que sobran o faltan (1), valor de un material (1)
- `eval_rlm_sft_nothink_v1b:rlm`: precio franco fábrica (3), no llamó a evaluate_origin (2), declaración de un material (2), subdivisión (1), materiales que sobran o faltan (1), valor de un material (1)
- `eval_rlm_sft_nothink_v1c:rlm`: precio franco fábrica (3), no llamó a evaluate_origin (2), declaración de un material (2), subdivisión (1), materiales que sobran o faltan (1), valor de un material (1)
- `eval_v1:base`: origen de un material (5), precio franco fábrica (3), declaración de un material (2), materiales que sobran o faltan (2), subdivisión (1), operaciones (1), no llamó a evaluate_origin (1), valor de un material (1)
- `eval_v1:thinking`: origen de un material (10), declaración de un material (3), precio franco fábrica (2), subdivisión (1), operaciones (1), materiales que sobran o faltan (1), no llamó a evaluate_origin (1)
- `eval_v1:rlm`: valor de un material (1), subdivisión (1), precio franco fábrica (1), materiales que sobran o faltan (1), origen de un material (1)
- `eval_v1b:base`: origen de un material (6), precio franco fábrica (3), declaración de un material (2), materiales que sobran o faltan (2), subdivisión (1), operaciones (1), valor de un material (1)
- `eval_v1b:thinking`: origen de un material (10), declaración de un material (3), precio franco fábrica (2), subdivisión (1), operaciones (1), materiales que sobran o faltan (1), no llamó a evaluate_origin (1)
- `eval_v1b:rlm`: valor de un material (1), subdivisión (1), precio franco fábrica (1), materiales que sobran o faltan (1), origen de un material (1), declaración de un material (1)
- `eval_v1c:base`: origen de un material (5), precio franco fábrica (3), declaración de un material (2), materiales que sobran o faltan (2), subdivisión (1), operaciones (1), no llamó a evaluate_origin (1), valor de un material (1)
- `eval_v1c:thinking`: origen de un material (10), declaración de un material (3), precio franco fábrica (2), subdivisión (1), operaciones (1), materiales que sobran o faltan (1), no llamó a evaluate_origin (1)
- `eval_v1c:rlm`: valor de un material (1), subdivisión (1), precio franco fábrica (1), materiales que sobran o faltan (1), origen de un material (1)
- `memory_0_v1:rlm`: valor de un material (1), subdivisión (1), precio franco fábrica (1), materiales que sobran o faltan (1), origen de un material (1)
- `memory_4000_v1:rlm`: valor de un material (2), subdivisión (1), precio franco fábrica (1), materiales que sobran o faltan (1), origen de un material (1)

## Cómo falla (tareas en las que aparece cada cosa)

|  | `codeagent_v1:code_base` | `codeagent_v1:code_rlm` | `eval:base` | `eval:thinking` | `eval:rlm` | `eval_rlm_grpo_pps3:rlm` | `eval_rlm_grpo_pps3_v1:rlm` | `eval_rlm_grpo_pps3_v1b:rlm` | `eval_rlm_grpo_pps3_v1c:rlm` | `eval_rlm_sft_nothink:rlm` | `eval_rlm_sft_nothink_v1:rlm` | `eval_rlm_sft_nothink_v1b:rlm` | `eval_rlm_sft_nothink_v1c:rlm` | `eval_v1:base` | `eval_v1:thinking` | `eval_v1:rlm` | `eval_v1b:base` | `eval_v1b:thinking` | `eval_v1b:rlm` | `eval_v1c:base` | `eval_v1c:thinking` | `eval_v1c:rlm` | `memory_0_v1:rlm` | `memory_4000_v1:rlm` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| alguna herramienta devolvió error | 16 | 11 | 11 | 8 | 11 | 2 | 2 | 2 | 2 | 8 | 11 | 11 | 10 | 13 | 11 | 12 | 12 | 11 | 10 | 13 | 11 | 9 | 10 | 10 |
| lista de materiales rechazada | 0 | 0 | 4 | 3 | 1 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| llamada repetida rechazada | 0 | 0 | 6 | 2 | 1 | 0 | 0 | 0 | 0 | 2 | 1 | 0 | 0 | 4 | 1 | 3 | 4 | 1 | 3 | 4 | 1 | 1 | 3 | 1 |
| llamada sin etiquetas recuperada | 0 | 0 | 8 | 5 | 2 | 4 | 4 | 4 | 4 | 2 | 0 | 0 | 0 | 7 | 6 | 2 | 7 | 5 | 1 | 7 | 6 | 2 | 3 | 1 |
| memoria comprimida | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 2 | 0 | 1 | 2 | 0 | 2 | 2 | 0 | 0 | 6 |
| producto como material rechazado | 0 | 0 | 2 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 3 | 1 | 0 | 3 | 1 | 0 | 3 | 1 | 0 | 0 | 0 |
| recordatorio de final_answer | 0 | 0 | 1 | 7 | 18 | 17 | 17 | 17 | 17 | 19 | 20 | 20 | 20 | 1 | 11 | 20 | 1 | 12 | 20 | 1 | 12 | 20 | 20 | 19 |
| sin final_answer | 0 | 0 | 1 | 5 | 16 | 17 | 16 | 16 | 17 | 17 | 18 | 18 | 18 | 1 | 9 | 18 | 1 | 9 | 19 | 1 | 9 | 19 | 19 | 18 |
| sin respuesta (pasos agotados o atascado) | 8 | 19 | 6 | 2 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 5 | 0 | 0 | 4 | 0 | 0 | 5 | 0 | 0 | 0 | 1 |
| valor inventado rechazado | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |

## Memoria

| Ejecución | Compresiones | Tokens antes (media) | Después (media) |
|---|---|---|---|
| `eval_v1:base` | 2 | 6162 | 4588 |
| `eval_v1:thinking` | 2 | 7192 | 4876 |
| `eval_v1b:base` | 1 | 6210 | 5154 |
| `eval_v1b:thinking` | 2 | 7180 | 4864 |
| `eval_v1c:base` | 2 | 6194 | 4600 |
| `eval_v1c:thinking` | 2 | 7172 | 4856 |
| `memory_4000_v1:rlm` | 7 | 4346 | 4080 |

## JSON frente a código

| Ejecución | Llamadas por paso | Pasos con varias llamadas |
|---|---|---|
| `codeagent_v1:code_base` | 0.84 | 10 |
| `codeagent_v1:code_rlm` | 0.31 | 3 |
| `eval:base` | 0.97 | - |
| `eval:thinking` | 0.81 | - |
| `eval:rlm` | 0.50 | - |
| `eval_rlm_grpo_pps3:rlm` | 0.15 | - |
| `eval_rlm_grpo_pps3_v1:rlm` | 0.20 | - |
| `eval_rlm_grpo_pps3_v1b:rlm` | 0.20 | - |
| `eval_rlm_grpo_pps3_v1c:rlm` | 0.17 | - |
| `eval_rlm_sft_nothink:rlm` | 0.53 | - |
| `eval_rlm_sft_nothink_v1:rlm` | 0.54 | - |
| `eval_rlm_sft_nothink_v1b:rlm` | 0.54 | - |
| `eval_rlm_sft_nothink_v1c:rlm` | 0.54 | - |
| `eval_v1:base` | 0.97 | - |
| `eval_v1:thinking` | 0.72 | - |
| `eval_v1:rlm` | 0.53 | - |
| `eval_v1b:base` | 0.97 | - |
| `eval_v1b:thinking` | 0.71 | - |
| `eval_v1b:rlm` | 0.52 | - |
| `eval_v1c:base` | 0.97 | - |
| `eval_v1c:thinking` | 0.71 | - |
| `eval_v1c:rlm` | 0.50 | - |
| `memory_0_v1:rlm` | 0.52 | - |
| `memory_4000_v1:rlm` | 0.51 | - |

## Repeticiones (v1*): media (mínimo-máximo)

| Configuración | Ejecuciones | Éxito | Aciertos limpios | s/tarea |
|---|---|---|---|---|
| `eval_rlm_grpo_pps3:rlm` | 3 | 15 % (15 %-15 %) | 2.0 (2.0-2.0) | 57 (56-58) |
| `eval_rlm_sft_nothink:rlm` | 3 | 57 % (50 %-60 %) | 9.0 (9.0-9.0) | 25 (24-26) |
| `eval:base` | 3 | 52 % (50 %-55 %) | 5.3 (5.0-6.0) | 19 (18-20) |
| `eval:thinking` | 3 | 53 % (50 %-55 %) | 5.3 (5.0-6.0) | 35 (34-36) |
| `eval:rlm` | 3 | 55 % (50 %-60 %) | 10.0 (9.0-11.0) | 28 (26-32) |

