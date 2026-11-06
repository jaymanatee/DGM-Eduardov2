# Análisis de `reports/phase1_eval_test_ood.json` (base, grpo), `reports/sft_rerun_ood.json` (sft) y `reports/grpo2_ood.json` (grpo2)

Modelos: base, sft (el SFT repetido sin cuDNN), grpo (GRPO 1, `final_rlm_lora`) y grpo2 (GRPO 2, `grpo_pps3`).

## 1. pass@1

| Modelo | n | Aciertos | IC 95 % | Lenient | Formato válido | Cortadas |
|---|---|---|---|---|---|---|
| base | 100 | 33 (33,0 %) | 24,6 % – 42,7 % | 38 (38,0 %) | 21,0 % | 12,0 % |
| sft | 100 | 59 (59,0 %) | 49,2 % – 68,1 % | 60 (60,0 %) | 82,0 % | 18,0 % |
| grpo | 100 | 66 (66,0 %) | 56,3 % – 74,5 % | 67 (67,0 %) | 89,0 % | 11,0 % |
| grpo2 | 100 | 70 (70,0 %) | 60,4 % – 78,1 % | 70 (70,0 %) | 93,0 % | 7,0 % |

Con n = 100, el intervalo llega a medir unos ±10 puntos: diferencias de pass@1 menores que eso entre dos modelos no son concluyentes por sí solas (la comparación por parejas de la sección 2 es más fina).

## 2. sft frente a grpo2 (mismos problemas)

|  | Problemas |
|---|---|
| Aciertan los dos | 50 |
| Solo grpo2 (arreglados) | 20 |
| Solo sft (estropeados) | 9 |
| Fallan los dos | 21 |

Test de McNemar exacto sobre los 29 problemas en que discrepan: p = 0,0614.

## 3. Tipos de fallo

| Tipo | base | sft | grpo | grpo2 |
|---|---|---|---|---|
| veredicto_incorrecto | 14 | 12 | 19 | 19 |
| cortada_sin_answer | 10 | 18 | 11 | 7 |
| porcentaje_incorrecto | 13 | 9 | 3 | 3 |
| tool_call_en_vez_de_answer | 14 | 0 | 0 | 0 |
| sin_answer | 11 | 0 | 0 | 0 |
| criterio_incorrecto | 5 | 2 | 1 | 1 |

`tool_call_en_vez_de_answer`: escribe `<tool_call>` tras `</think>`. `cortada_sin_answer`: llega al límite de tokens sin responder.

## 4. pass@1 por grupo (n ≥ 5)

| Grupo | n | base | sft | grpo | grpo2 | Profesor (aceptación) |
|---|---|---|---|---|---|---|
| todos | 100 | 33,0 % | 59,0 % | 66,0 % | 70,0 % | 64,1 % (n=2000) |
| criterio_aplicado=CC | 18 | 5,6 % | 33,3 % | 50,0 % | 38,9 % | 75,8 % (n=124) |
| criterio_aplicado=CTH | 16 | 50,0 % | 75,0 % | 75,0 % | 75,0 % | 56,3 % (n=504) |
| criterio_aplicado=CTSH | 16 | 50,0 % | 62,5 % | 68,8 % | 93,8 % | 27,8 % (n=144) |
| criterio_aplicado=MAXNOM | 6 | 66,7 % | 100,0 % | 83,3 % | 83,3 % | 97,5 % (n=404) |
| criterio_aplicado=NINGUNO | 30 | 33,3 % | 60,0 % | 66,7 % | 73,3 % | 54,3 % (n=700) |
| criterio_aplicado=RVC | 14 | 14,3 % | 50,0 % | 64,3 % | 64,3 % | 72,6 % (n=124) |
| criterios_validos=CC | 18 | 5,6 % | 33,3 % | 50,0 % | 38,9 % | 75,0 % (n=44) |
| criterios_validos=CTH|MAXNOM | 8 | 50,0 % | 75,0 % | 87,5 % | 87,5 % | 87,0 % (n=108) |
| criterios_validos=CTH|RVC | 6 | 50,0 % | 83,3 % | 83,3 % | 66,7 % | 68,3 % (n=120) |
| criterios_validos=CTSH|MAXNOM | 13 | 46,2 % | 69,2 % | 69,2 % | 92,3 % | — |
| criterios_validos=MAXNOM | 6 | 66,7 % | 100,0 % | 83,3 % | 83,3 % | 97,5 % (n=404) |
| criterios_validos=NINGUNO | 30 | 33,3 % | 60,0 % | 66,7 % | 73,3 % | 54,3 % (n=700) |
| criterios_validos=RVC | 14 | 14,3 % | 50,0 % | 64,3 % | 64,3 % | 72,6 % (n=124) |
| excepcion_partida=False | 80 | 35,0 % | 61,3 % | 67,5 % | 68,8 % | 65,2 % (n=1680) |
| excepcion_partida=True | 20 | 25,0 % | 50,0 % | 60,0 % | 75,0 % | 58,1 % (n=320) |
| material_misma_partida=False | 24 | 12,5 % | 58,3 % | 66,7 % | 62,5 % | 65,8 % (n=708) |
| material_misma_partida=True | 76 | 39,5 % | 59,2 % | 65,8 % | 72,4 % | 63,2 % (n=1292) |
| operacion_insuficiente=False | 89 | 33,7 % | 61,8 % | 67,4 % | 74,2 % | 68,7 % (n=1812) |
| operacion_insuficiente=True | 11 | 27,3 % | 36,4 % | 54,5 % | 36,4 % | 19,7 % (n=188) |
| origen_desconocido=False | 69 | 31,9 % | 56,5 % | 65,2 % | 69,6 % | 65,2 % (n=1424) |
| origen_desconocido=True | 31 | 35,5 % | 64,5 % | 67,7 % | 71,0 % | 61,3 % (n=576) |
| originario=False | 30 | 33,3 % | 60,0 % | 66,7 % | 73,3 % | 54,3 % (n=700) |
| originario=True | 70 | 32,9 % | 58,6 % | 65,7 % | 68,6 % | 69,4 % (n=1300) |
| proveedor_sin_declaracion=False | 77 | 36,4 % | 58,4 % | 68,8 % | 67,5 % | 66,8 % (n=1464) |
| proveedor_sin_declaracion=True | 23 | 21,7 % | 60,9 % | 56,5 % | 78,3 % | 56,7 % (n=536) |
| redaccion=descriptiva | 50 | 34,0 % | 50,0 % | 62,0 % | 68,0 % | — |
| redaccion=siglas | 50 | 32,0 % | 68,0 % | 70,0 % | 72,0 % | — |
| regla=MERCOSUR-8481 | 28 | 28,6 % | 60,7 % | 67,9 % | 75,0 % | — |
| regla=MERCOSUR-8516 | 31 | 51,6 % | 74,2 % | 74,2 % | 83,9 % | — |
| regla=MERCOSUR-8708 | 17 | 41,2 % | 76,5 % | 58,8 % | 70,6 % | — |
| regla=MERCOSUR-9401 | 24 | 8,3 % | 25,0 % | 58,3 % | 45,8 % | — |
| subdivision_distinta=False | 35 | 25,7 % | 68,6 % | 77,1 % | 74,3 % | 66,9 % (n=852) |
| subdivision_distinta=True | 65 | 36,9 % | 53,8 % | 60,0 % | 67,7 % | 62,0 % (n=1148) |
| tipo=criterio_de_valor | 47 | 40,4 % | 70,2 % | 74,5 % | 78,7 % | 86,2 % (n=836) |
| tipo=no_originario | 30 | 33,3 % | 60,0 % | 66,7 % | 73,3 % | 54,3 % (n=700) |
| tipo=solo_cambio_de_clasificacion | 23 | 17,4 % | 34,8 % | 47,8 % | 47,8 % | 39,0 % (n=464) |
| tolerancia_no_amplia_maxnom=False | 89 | 33,7 % | 56,2 % | 67,4 % | 67,4 % | 63,1 % (n=1776) |
| tolerancia_no_amplia_maxnom=True | 11 | 27,3 % | 81,8 % | 54,5 % | 90,9 % | 71,9 % (n=224) |
| tolerancia_usada=False | 88 | 35,2 % | 64,8 % | 73,9 % | 77,3 % | 69,7 % (n=1804) |
| tolerancia_usada=True | 12 | 16,7 % | 16,7 % | 8,3 % | 16,7 % | 12,8 % (n=196) |
| umbral_exacto=False | 89 | 32,6 % | 58,4 % | 68,5 % | 70,8 % | 61,2 % (n=1744) |
| umbral_exacto=True | 11 | 36,4 % | 63,6 % | 45,5 % | 63,6 % | 84,0 % (n=256) |

Profesor: trazas verificadas / generadas en los 500 problemas de destilación (Qwen3-4B, temperatura 0,6, problemas de `train`). Orientativo: no son los mismos problemas ni la misma forma de generar.

## 5. Longitud

| Modelo | Mediana (aciertos) | Mediana (fallos) | p95 | Cortadas |
|---|---|---|---|---|
| base | 1562 | 1488 | 3072 | 12,0 % |
| sft | 1468 | 2516 | 3072 | 18,0 % |
| grpo | 1296 | 1586 | 3072 | 11,0 % |
| grpo2 | 1273 | 1282 | 3072 | 7,0 % |

## 6. ¿Acierta por las razones correctas? (solo respuestas acertadas, heurístico)

| Modelo | Aciertos | Cita la partida | Porcentaje en el razonamiento | Suma en el razonamiento | Aciertos sin el cálculo a la vista |
|---|---|---|---|---|---|
| base | 33 | 100,0 % (n=33) | 97,0 % (n=33) | 97,0 % (n=33) | 1 |
| sft | 59 | 100,0 % (n=59) | 100,0 % (n=59) | 100,0 % (n=59) | 0 |
| grpo | 66 | 100,0 % (n=66) | 100,0 % (n=66) | 100,0 % (n=66) | 0 |
| grpo2 | 70 | 100,0 % (n=70) | 100,0 % (n=70) | 100,0 % (n=70) | 0 |

Se busca en el `<think>` el porcentaje esperado (±0,5 puntos) y la suma esperada de materiales no originarios (porcentaje × precio EXW leído del enunciado, ±0,5 %). Los casos con 0 % no se comprueban. La última columna son aciertos cuyo razonamiento no contiene el porcentaje ni la suma: candidatos a acierto por casualidad, para leer a mano.

## 7. Fallos para leer a mano (grpo2)

**veredicto_incorrecto** (19 problemas): índices [8, 14, 18, 20, 22, 24, 26, 36, 49, 51, 53, 61, 72, 74, 79] …

- Problema 8 [originario=True, tipo=criterio_de_valor, regla=MERCOSUR-8481, criterio_aplicado=RVC, criterios_validos=RVC, material_misma_partida=True, subdivision_distinta=True, excepcion_partida=True, proveedor_sin_declaracion=True, origen_desconocido=False, operacion_insuficiente=False, tolerancia_usada=False, umbral_exacto=True, tolerancia_no_amplia_maxnom=False, redaccion=siglas]
  - esperado: `{"originario": true, "criterio": "RVC", "pct_no_originario": 44.39, "criterios_validos": ["RVC"]}`
  - modelo: `{   "originario": false,   "criterio": "NINGUNO",   "pct_no_originario": "44.44" }`
  - motivo: veredicto incorrecto: False
- Problema 14 [originario=True, tipo=solo_cambio_de_clasificacion, regla=MERCOSUR-9401, criterio_aplicado=CC, criterios_validos=CC, material_misma_partida=True, subdivision_distinta=True, excepcion_partida=False, proveedor_sin_declaracion=False, origen_desconocido=False, operacion_insuficiente=False, tolerancia_usada=True, umbral_exacto=False, tolerancia_no_amplia_maxnom=False, redaccion=siglas]
  - esperado: `{"originario": true, "criterio": "CC", "pct_no_originario": 25.31, "criterios_validos": ["CC"]}`
  - modelo: `{   "originario": false,   "criterio": "NINGUNO",   "pct_no_originario": "25.31" }`
  - motivo: veredicto incorrecto: False
- Problema 18 [originario=True, tipo=solo_cambio_de_clasificacion, regla=MERCOSUR-9401, criterio_aplicado=CC, criterios_validos=CC, material_misma_partida=True, subdivision_distinta=True, excepcion_partida=False, proveedor_sin_declaracion=False, origen_desconocido=False, operacion_insuficiente=False, tolerancia_usada=True, umbral_exacto=False, tolerancia_no_amplia_maxnom=False, redaccion=descriptiva]
  - esperado: `{"originario": true, "criterio": "CC", "pct_no_originario": 53.42, "criterios_validos": ["CC"]}`
  - modelo: `{   "originario": false,   "criterio": "NINGUNO",   "pct_no_originario": "53.42" }`
  - motivo: veredicto incorrecto: False

**cortada_sin_answer** (7 problemas): índices [2, 10, 19, 27, 70, 90, 99]

- Problema 2 [originario=True, tipo=criterio_de_valor, regla=MERCOSUR-8516, criterio_aplicado=CTSH, criterios_validos=CTSH|MAXNOM, material_misma_partida=True, subdivision_distinta=True, excepcion_partida=False, proveedor_sin_declaracion=False, origen_desconocido=False, operacion_insuficiente=False, tolerancia_usada=False, umbral_exacto=True, tolerancia_no_amplia_maxnom=False, redaccion=descriptiva]
  - esperado: `{"originario": true, "criterio": "CTSH", "pct_no_originario": 55.0, "criterios_validos": ["CTSH", "MAXNOM"]}`
  - modelo: `her. But the problem says that the answer should be based on the first applicable. So the answer would be originario, and the criterion could be either. However`
  - motivo: no <answer> block found
- Problema 10 [originario=True, tipo=criterio_de_valor, regla=MERCOSUR-8481, criterio_aplicado=RVC, criterios_validos=RVC, material_misma_partida=False, subdivision_distinta=False, excepcion_partida=True, proveedor_sin_declaracion=False, origen_desconocido=False, operacion_insuficiente=False, tolerancia_usada=False, umbral_exacto=True, tolerancia_no_amplia_maxnom=False, redaccion=descriptiva]
  - esperado: `{"originario": true, "criterio": "RVC", "pct_no_originario": 41.78, "criterios_validos": ["RVC"]}`
  - modelo: `m 8484.10 are not originario. Therefore, the materials from 8484.10 are not originario. Therefore, the materials from 8484.10 are not originario. Therefore, the`
  - motivo: no <answer> block found
- Problema 19 [originario=True, tipo=criterio_de_valor, regla=MERCOSUR-8708, criterio_aplicado=CTH, criterios_validos=CTH|MAXNOM, material_misma_partida=False, subdivision_distinta=False, excepcion_partida=False, proveedor_sin_declaracion=False, origen_desconocido=False, operacion_insuficiente=False, tolerancia_usada=False, umbral_exacto=False, tolerancia_no_amplia_maxnom=False, redaccion=siglas]
  - esperado: `{"originario": true, "criterio": "CTH", "pct_no_originario": 39.0, "criterios_validos": ["CTH", "MAXNOM"]}`
  - modelo: `d on the first one. Therefore, the answer would be "originario" with CTH. But the problem says that the answer should specify which criterion is used. Therefore`
  - motivo: no <answer> block found

**porcentaje_incorrecto** (3 problemas): índices [63, 68, 81]

- Problema 63 [originario=True, tipo=solo_cambio_de_clasificacion, regla=MERCOSUR-9401, criterio_aplicado=CC, criterios_validos=CC, material_misma_partida=True, subdivision_distinta=True, excepcion_partida=False, proveedor_sin_declaracion=False, origen_desconocido=True, operacion_insuficiente=False, tolerancia_usada=True, umbral_exacto=False, tolerancia_no_amplia_maxnom=False, redaccion=descriptiva]
  - esperado: `{"originario": true, "criterio": "CC", "pct_no_originario": 20.29, "criterios_validos": ["CC"]}`
  - modelo: `{   "originario": true,   "criterio": "CC",   "pct_no_originario": "3.42" }`
  - motivo: porcentaje fuera de ±0.5: 3.42 vs 20.29
- Problema 68 [originario=False, tipo=no_originario, regla=MERCOSUR-8516, criterio_aplicado=NINGUNO, criterios_validos=NINGUNO, material_misma_partida=True, subdivision_distinta=True, excepcion_partida=False, proveedor_sin_declaracion=True, origen_desconocido=True, operacion_insuficiente=False, tolerancia_usada=False, umbral_exacto=False, tolerancia_no_amplia_maxnom=False, redaccion=siglas]
  - esperado: `{"originario": false, "criterio": "NINGUNO", "pct_no_originario": 57.67, "criterios_validos": []}`
  - modelo: `{   "originario": false,   "criterio": "NINGUNO",   "pct_no_originario": 74.25 }`
  - motivo: porcentaje fuera de ±0.5: 74.25 vs 57.67
- Problema 81 [originario=True, tipo=criterio_de_valor, regla=MERCOSUR-8481, criterio_aplicado=CTH, criterios_validos=CTH|RVC, material_misma_partida=True, subdivision_distinta=True, excepcion_partida=False, proveedor_sin_declaracion=True, origen_desconocido=True, operacion_insuficiente=False, tolerancia_usada=False, umbral_exacto=False, tolerancia_no_amplia_maxnom=False, redaccion=descriptiva]
  - esperado: `{"originario": true, "criterio": "CTH", "pct_no_originario": 20.68, "criterios_validos": ["CTH", "RVC"]}`
  - modelo: `{   "originario": true,   "criterio": "RVC",   "pct_no_originario": "13.96" }`
  - motivo: porcentaje fuera de ±0.5: 13.96 vs 20.68

**criterio_incorrecto** (1 problemas): índices [9]

- Problema 9 [originario=True, tipo=criterio_de_valor, regla=MERCOSUR-8481, criterio_aplicado=RVC, criterios_validos=RVC, material_misma_partida=False, subdivision_distinta=False, excepcion_partida=True, proveedor_sin_declaracion=False, origen_desconocido=False, operacion_insuficiente=False, tolerancia_usada=False, umbral_exacto=False, tolerancia_no_amplia_maxnom=False, redaccion=siglas]
  - esperado: `{"originario": true, "criterio": "RVC", "pct_no_originario": 13.55, "criterios_validos": ["RVC"]}`
  - modelo: `{   "originario": true,   "criterio": "CTH",   "pct_no_originario": 13.55 }`
  - motivo: criterio 'CTH' no válido (válidos: ['RVC'])

Para ver el razonamiento completo: `uv run --extra train python -m rlm.try_model <índice>` (o leer `raw` en el JSON).
