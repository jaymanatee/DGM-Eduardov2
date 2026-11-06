# Análisis de `reports/phase1_eval_test.json` (base, grpo), `reports/sft_rerun_test.json` (sft) y `reports/grpo2_test.json` (grpo2)

Modelos: base, sft (el SFT repetido sin cuDNN), grpo (GRPO 1, `final_rlm_lora`) y grpo2 (GRPO 2, `grpo_pps3`).

## 1. pass@1

| Modelo | n | Aciertos | IC 95 % | Lenient | Formato válido | Cortadas |
|---|---|---|---|---|---|---|
| base | 200 | 56 (28,0 %) | 22,2 % – 34,6 % | 58 (29,0 %) | 19,0 % | 10,5 % |
| sft | 200 | 117 (58,5 %) | 51,6 % – 65,1 % | 117 (58,5 %) | 85,0 % | 15,0 % |
| grpo | 200 | 125 (62,5 %) | 55,6 % – 68,9 % | 125 (62,5 %) | 89,5 % | 10,5 % |
| grpo2 | 200 | 132 (66,0 %) | 59,2 % – 72,2 % | 132 (66,0 %) | 90,5 % | 9,5 % |

Con n = 200, el intervalo llega a medir unos ±7 puntos: diferencias de pass@1 menores que eso entre dos modelos no son concluyentes por sí solas (la comparación por parejas de la sección 2 es más fina).

## 2. sft frente a grpo2 (mismos problemas)

|  | Problemas |
|---|---|
| Aciertan los dos | 101 |
| Solo grpo2 (arreglados) | 31 |
| Solo sft (estropeados) | 16 |
| Fallan los dos | 52 |

Test de McNemar exacto sobre los 47 problemas en que discrepan: p = 0,04.

## 3. Tipos de fallo

| Tipo | base | sft | grpo | grpo2 |
|---|---|---|---|---|
| veredicto_incorrecto | 39 | 32 | 38 | 38 |
| cortada_sin_answer | 20 | 30 | 21 | 19 |
| porcentaje_incorrecto | 29 | 21 | 16 | 11 |
| tool_call_en_vez_de_answer | 38 | 0 | 0 | 0 |
| criterio_incorrecto | 11 | 0 | 0 | 0 |
| sin_answer | 7 | 0 | 0 | 0 |

`tool_call_en_vez_de_answer`: escribe `<tool_call>` tras `</think>`. `cortada_sin_answer`: llega al límite de tokens sin responder.

## 4. pass@1 por grupo (n ≥ 5)

| Grupo | n | base | sft | grpo | grpo2 | Profesor (aceptación) |
|---|---|---|---|---|---|---|
| todos | 200 | 28,0 % | 58,5 % | 62,5 % | 66,0 % | 64,1 % (n=2000) |
| criterio_aplicado=CC | 13 | 38,5 % | 76,9 % | 76,9 % | 84,6 % | 75,8 % (n=124) |
| criterio_aplicado=CTH | 54 | 16,7 % | 50,0 % | 59,3 % | 64,8 % | 56,3 % (n=504) |
| criterio_aplicado=CTSH | 28 | 25,0 % | 7,1 % | 7,1 % | 21,4 % | 27,8 % (n=144) |
| criterio_aplicado=MAXNOM | 27 | 40,7 % | 96,3 % | 96,3 % | 96,3 % | 97,5 % (n=404) |
| criterio_aplicado=NINGUNO | 65 | 36,9 % | 67,7 % | 70,8 % | 67,7 % | 54,3 % (n=700) |
| criterio_aplicado=RVC | 13 | 0,0 % | 61,5 % | 69,2 % | 76,9 % | 72,6 % (n=124) |
| criterios_validos=CC | 6 | 33,3 % | 66,7 % | 100,0 % | 83,3 % | 75,0 % (n=44) |
| criterios_validos=CC|MAXNOM | 7 | 42,9 % | 85,7 % | 57,1 % | 85,7 % | 76,2 % (n=80) |
| criterios_validos=CTH | 27 | 7,4 % | 33,3 % | 48,1 % | 59,3 % | 39,1 % (n=276) |
| criterios_validos=CTH|MAXNOM | 13 | 30,8 % | 69,2 % | 69,2 % | 69,2 % | 87,0 % (n=108) |
| criterios_validos=CTH|RVC | 14 | 21,4 % | 64,3 % | 71,4 % | 71,4 % | 68,3 % (n=120) |
| criterios_validos=CTSH | 28 | 25,0 % | 7,1 % | 7,1 % | 21,4 % | 27,8 % (n=144) |
| criterios_validos=MAXNOM | 27 | 40,7 % | 96,3 % | 96,3 % | 96,3 % | 97,5 % (n=404) |
| criterios_validos=NINGUNO | 65 | 36,9 % | 67,7 % | 70,8 % | 67,7 % | 54,3 % (n=700) |
| criterios_validos=RVC | 13 | 0,0 % | 61,5 % | 69,2 % | 76,9 % | 72,6 % (n=124) |
| excepcion_partida=False | 169 | 30,2 % | 59,8 % | 60,9 % | 65,7 % | 65,2 % (n=1680) |
| excepcion_partida=True | 31 | 16,1 % | 51,6 % | 71,0 % | 67,7 % | 58,1 % (n=320) |
| material_misma_partida=False | 75 | 29,3 % | 62,7 % | 66,7 % | 76,0 % | 65,8 % (n=708) |
| material_misma_partida=True | 125 | 27,2 % | 56,0 % | 60,0 % | 60,0 % | 63,2 % (n=1292) |
| operacion_insuficiente=False | 183 | 28,4 % | 60,1 % | 65,0 % | 69,4 % | 68,7 % (n=1812) |
| operacion_insuficiente=True | 17 | 23,5 % | 41,2 % | 35,3 % | 29,4 % | 19,7 % (n=188) |
| origen_desconocido=False | 142 | 27,5 % | 56,3 % | 63,4 % | 65,5 % | 65,2 % (n=1424) |
| origen_desconocido=True | 58 | 29,3 % | 63,8 % | 60,3 % | 67,2 % | 61,3 % (n=576) |
| originario=False | 65 | 36,9 % | 67,7 % | 70,8 % | 67,7 % | 54,3 % (n=700) |
| originario=True | 135 | 23,7 % | 54,1 % | 58,5 % | 65,2 % | 69,4 % (n=1300) |
| proveedor_sin_declaracion=False | 150 | 32,0 % | 63,3 % | 64,0 % | 68,7 % | 66,8 % (n=1464) |
| proveedor_sin_declaracion=True | 50 | 16,0 % | 44,0 % | 58,0 % | 58,0 % | 56,7 % (n=536) |
| regla=CETA-8413 | 39 | 30,8 % | 25,6 % | 33,3 % | 38,5 % | 40,8 % (n=316) |
| regla=TCA-8414 | 33 | 6,1 % | 45,5 % | 45,5 % | 45,5 % | 35,0 % (n=320) |
| regla=TCA-8507 | 38 | 39,5 % | 65,8 % | 65,8 % | 71,1 % | 77,8 % (n=316) |
| regla=UE-JP-8501 | 38 | 10,5 % | 55,3 % | 68,4 % | 76,3 % | 64,7 % (n=360) |
| regla=UE-KR-8479 | 31 | 38,7 % | 93,5 % | 90,3 % | 90,3 % | 86,5 % (n=364) |
| regla=UE-KR-9403 | 21 | 52,4 % | 81,0 % | 85,7 % | 85,7 % | 76,2 % (n=324) |
| subdivision_distinta=False | 99 | 30,3 % | 59,6 % | 63,6 % | 71,7 % | 66,9 % (n=852) |
| subdivision_distinta=True | 101 | 25,7 % | 57,4 % | 61,4 % | 60,4 % | 62,0 % (n=1148) |
| tipo=criterio_de_valor | 74 | 28,4 % | 78,4 % | 78,4 % | 82,4 % | 86,2 % (n=836) |
| tipo=no_originario | 65 | 36,9 % | 67,7 % | 70,8 % | 67,7 % | 54,3 % (n=700) |
| tipo=solo_cambio_de_clasificacion | 61 | 18,0 % | 24,6 % | 34,4 % | 44,3 % | 39,0 % (n=464) |
| tolerancia_no_amplia_maxnom=False | 186 | 27,4 % | 58,6 % | 61,3 % | 64,0 % | 63,1 % (n=1776) |
| tolerancia_no_amplia_maxnom=True | 14 | 35,7 % | 57,1 % | 78,6 % | 92,9 % | 71,9 % (n=224) |
| tolerancia_usada=False | 170 | 31,2 % | 67,6 % | 71,2 % | 75,9 % | 69,7 % (n=1804) |
| tolerancia_usada=True | 30 | 10,0 % | 6,7 % | 13,3 % | 10,0 % | 12,8 % (n=196) |
| umbral_exacto=False | 177 | 27,7 % | 55,9 % | 60,5 % | 64,4 % | 61,2 % (n=1744) |
| umbral_exacto=True | 23 | 30,4 % | 78,3 % | 78,3 % | 78,3 % | 84,0 % (n=256) |

Profesor: trazas verificadas / generadas en los 500 problemas de destilación (Qwen3-4B, temperatura 0,6, problemas de `train`). Orientativo: no son los mismos problemas ni la misma forma de generar.

## 5. Longitud

| Modelo | Mediana (aciertos) | Mediana (fallos) | p95 | Cortadas |
|---|---|---|---|---|
| base | 1440 | 1438 | 3072 | 10,5 % |
| sft | 1078 | 1548 | 3072 | 15,0 % |
| grpo | 1181 | 1673 | 3072 | 10,5 % |
| grpo2 | 1200 | 1560 | 3072 | 9,5 % |

## 6. ¿Acierta por las razones correctas? (solo respuestas acertadas, heurístico)

| Modelo | Aciertos | Cita la partida | Porcentaje en el razonamiento | Suma en el razonamiento | Aciertos sin el cálculo a la vista |
|---|---|---|---|---|---|
| base | 56 | 92,9 % (n=56) | 100,0 % (n=54) | 96,3 % (n=54) | 0 |
| sft | 117 | 100,0 % (n=117) | 99,1 % (n=116) | 100,0 % (n=116) | 0 |
| grpo | 125 | 100,0 % (n=125) | 99,2 % (n=125) | 99,2 % (n=125) | 0 |
| grpo2 | 132 | 100,0 % (n=132) | 98,5 % (n=132) | 98,5 % (n=132) | 0 |

Se busca en el `<think>` el porcentaje esperado (±0,5 puntos) y la suma esperada de materiales no originarios (porcentaje × precio EXW leído del enunciado, ±0,5 %). Los casos con 0 % no se comprueban. La última columna son aciertos cuyo razonamiento no contiene el porcentaje ni la suma: candidatos a acierto por casualidad, para leer a mano.

## 7. Fallos para leer a mano (grpo2)

**veredicto_incorrecto** (38 problemas): índices [8, 12, 14, 16, 17, 26, 33, 40, 42, 44, 48, 50, 51, 54, 55] …

- Problema 8 [originario=True, tipo=solo_cambio_de_clasificacion, regla=CETA-8413, criterio_aplicado=CTSH, criterios_validos=CTSH, material_misma_partida=True, subdivision_distinta=False, excepcion_partida=False, proveedor_sin_declaracion=False, origen_desconocido=False, operacion_insuficiente=False, tolerancia_usada=True, umbral_exacto=False, tolerancia_no_amplia_maxnom=False]
  - esperado: `{"originario": true, "criterio": "CTSH", "pct_no_originario": 24.83, "criterios_validos": ["CTSH"]}`
  - modelo: `{   "originario": false,   "criterio": "NINGUNO",   "pct_no_originario": "24.83" }`
  - motivo: veredicto incorrecto: False
- Problema 12 [originario=True, tipo=solo_cambio_de_clasificacion, regla=TCA-8414, criterio_aplicado=CTH, criterios_validos=CTH, material_misma_partida=True, subdivision_distinta=True, excepcion_partida=False, proveedor_sin_declaracion=False, origen_desconocido=False, operacion_insuficiente=False, tolerancia_usada=True, umbral_exacto=False, tolerancia_no_amplia_maxnom=False]
  - esperado: `{"originario": true, "criterio": "CTH", "pct_no_originario": 67.63, "criterios_validos": ["CTH"]}`
  - modelo: `{   "originario": false,   "criterio": "NINGUNO",   "pct_no_originario": "67.80" }`
  - motivo: veredicto incorrecto: False
- Problema 14 [originario=True, tipo=solo_cambio_de_clasificacion, regla=CETA-8413, criterio_aplicado=CTSH, criterios_validos=CTSH, material_misma_partida=True, subdivision_distinta=True, excepcion_partida=False, proveedor_sin_declaracion=False, origen_desconocido=True, operacion_insuficiente=False, tolerancia_usada=False, umbral_exacto=False, tolerancia_no_amplia_maxnom=False]
  - esperado: `{"originario": true, "criterio": "CTSH", "pct_no_originario": 22.73, "criterios_validos": ["CTSH"]}`
  - modelo: `{   "originario": false,   "criterio": "NINGUNO",   "pct_no_originario": "22.73" }`
  - motivo: veredicto incorrecto: False

**cortada_sin_answer** (19 problemas): índices [0, 7, 53, 57, 61, 82, 85, 88, 95, 98, 100, 120, 122, 128, 136] …

- Problema 0 [originario=True, tipo=solo_cambio_de_clasificacion, regla=CETA-8413, criterio_aplicado=CTSH, criterios_validos=CTSH, material_misma_partida=False, subdivision_distinta=False, excepcion_partida=False, proveedor_sin_declaracion=False, origen_desconocido=False, operacion_insuficiente=False, tolerancia_usada=False, umbral_exacto=False, tolerancia_no_amplia_maxnom=False]
  - esperado: `{"originario": true, "criterio": "CTSH", "pct_no_originario": 0.0, "criterios_validos": ["CTSH"]}`
  - modelo: `nario.  But the bearings are in 8482.10, which is part of 8482. But since they are originario, the rule says that CTSH is not applicable if any material is in 8`
  - motivo: no <answer> block found
- Problema 7 [originario=True, tipo=criterio_de_valor, regla=TCA-8507, criterio_aplicado=CTH, criterios_validos=CTH|MAXNOM, material_misma_partida=False, subdivision_distinta=False, excepcion_partida=False, proveedor_sin_declaracion=False, origen_desconocido=False, operacion_insuficiente=False, tolerancia_usada=False, umbral_exacto=False, tolerancia_no_amplia_maxnom=False]
  - esperado: `{"originario": true, "criterio": "CTH", "pct_no_originario": 0.0, "criterios_validos": ["CTH", "MAXNOM"]}`
  - modelo: ` is 0% (since there are no non-originary materials). Therefore, the product meets the MAXNOM criterion. Therefore, the answer is originario: true, criterio: MAX`
  - motivo: no <answer> block found
- Problema 53 [originario=True, tipo=criterio_de_valor, regla=TCA-8507, criterio_aplicado=CTH, criterios_validos=CTH|MAXNOM, material_misma_partida=True, subdivision_distinta=False, excepcion_partida=False, proveedor_sin_declaracion=False, origen_desconocido=True, operacion_insuficiente=False, tolerancia_usada=False, umbral_exacto=False, tolerancia_no_amplia_maxnom=False]
  - esperado: `{"originario": true, "criterio": "CTH", "pct_no_originario": 37.71, "criterios_validos": ["CTH", "MAXNOM"]}`
  - modelo: `er might have intended that only one is considered. But according to the rules, both are valid. However, the problem says that the answer should be the one that`
  - motivo: no <answer> block found

**porcentaje_incorrecto** (11 problemas): índices [13, 24, 31, 32, 47, 71, 93, 96, 119, 127, 162]

- Problema 13 [originario=False, tipo=no_originario, regla=TCA-8414, criterio_aplicado=NINGUNO, criterios_validos=NINGUNO, material_misma_partida=True, subdivision_distinta=True, excepcion_partida=False, proveedor_sin_declaracion=False, origen_desconocido=False, operacion_insuficiente=False, tolerancia_usada=False, umbral_exacto=False, tolerancia_no_amplia_maxnom=False]
  - esperado: `{"originario": false, "criterio": "NINGUNO", "pct_no_originario": 25.05, "criterios_validos": []}`
  - modelo: `{   "originario": false,   "criterio": "NINGUNO",   "pct_no_originario": "20.54" }`
  - motivo: porcentaje fuera de ±0.5: 20.54 vs 25.05
- Problema 24 [originario=True, tipo=solo_cambio_de_clasificacion, regla=CETA-8413, criterio_aplicado=CTSH, criterios_validos=CTSH, material_misma_partida=True, subdivision_distinta=False, excepcion_partida=False, proveedor_sin_declaracion=True, origen_desconocido=False, operacion_insuficiente=False, tolerancia_usada=True, umbral_exacto=False, tolerancia_no_amplia_maxnom=False]
  - esperado: `{"originario": true, "criterio": "CTSH", "pct_no_originario": 54.42, "criterios_validos": ["CTSH"]}`
  - modelo: `{   "originario": true,   "criterio": "CTSH",   "pct_no_originario": "7.17" }`
  - motivo: porcentaje fuera de ±0.5: 7.17 vs 54.42
- Problema 31 [originario=True, tipo=criterio_de_valor, regla=TCA-8507, criterio_aplicado=CTH, criterios_validos=CTH|MAXNOM, material_misma_partida=True, subdivision_distinta=True, excepcion_partida=False, proveedor_sin_declaracion=False, origen_desconocido=False, operacion_insuficiente=False, tolerancia_usada=False, umbral_exacto=False, tolerancia_no_amplia_maxnom=False]
  - esperado: `{"originario": true, "criterio": "CTH", "pct_no_originario": 36.0, "criterios_validos": ["CTH", "MAXNOM"]}`
  - modelo: `{   "originario": true,   "criterio": "CTH",   "pct_no_originario": 35.00 }`
  - motivo: porcentaje fuera de ±0.5: 35.0 vs 36.0

Para ver el razonamiento completo: `uv run --extra train python -m rlm.try_model <índice>` (o leer `raw` en el JSON).
