# Análisis de `phase1_eval_test_ood.json`

Modelos: base, sft.

## 1. pass@1

| Modelo | n | Aciertos | IC 95 % | Lenient | Formato válido | Cortadas |
|---|---|---|---|---|---|---|
| base | 100 | 33 (33,0 %) | 24,6 % – 42,7 % | 38 (38,0 %) | 21,0 % | 12,0 % |
| sft | 100 | 58 (58,0 %) | 48,2 % – 67,2 % | 58 (58,0 %) | 80,0 % | 20,0 % |

Con n = 100, el intervalo llega a medir unos ±10 puntos: diferencias de pass@1 menores que eso entre dos modelos no son concluyentes por sí solas (la comparación por parejas de la sección 2 es más fina).

## 2. base frente a sft (mismos problemas)

|  | Problemas |
|---|---|
| Aciertan los dos | 20 |
| Solo sft (arreglados) | 38 |
| Solo base (estropeados) | 13 |
| Fallan los dos | 29 |

Test de McNemar exacto sobre los 51 problemas en que discrepan: p = 0,000621.

## 3. Tipos de fallo

| Tipo | base | sft |
|---|---|---|
| cortada_sin_answer | 10 | 20 |
| veredicto_incorrecto | 14 | 15 |
| porcentaje_incorrecto | 13 | 7 |
| tool_call_en_vez_de_answer | 14 | 0 |
| sin_answer | 11 | 0 |
| criterio_incorrecto | 5 | 0 |

`tool_call_en_vez_de_answer`: escribe `<tool_call>` tras `</think>`. `cortada_sin_answer`: llega al límite de tokens sin responder.

## 4. pass@1 por grupo (n ≥ 5)

| Grupo | n | base | sft |
|---|---|---|---|
| todos | 100 | 33,0 % | 58,0 % |
| criterio_aplicado=CC | 18 | 5,6 % | 38,9 % |
| criterio_aplicado=CTH | 16 | 50,0 % | 50,0 % |
| criterio_aplicado=CTSH | 16 | 50,0 % | 56,2 % |
| criterio_aplicado=MAXNOM | 6 | 66,7 % | 83,3 % |
| criterio_aplicado=NINGUNO | 30 | 33,3 % | 76,7 % |
| criterio_aplicado=RVC | 14 | 14,3 % | 42,9 % |
| criterios_validos=CC | 18 | 5,6 % | 38,9 % |
| criterios_validos=CTH|MAXNOM | 8 | 50,0 % | 62,5 % |
| criterios_validos=CTH|RVC | 6 | 50,0 % | 50,0 % |
| criterios_validos=CTSH|MAXNOM | 13 | 46,2 % | 53,8 % |
| criterios_validos=MAXNOM | 6 | 66,7 % | 83,3 % |
| criterios_validos=NINGUNO | 30 | 33,3 % | 76,7 % |
| criterios_validos=RVC | 14 | 14,3 % | 42,9 % |
| excepcion_partida=False | 80 | 35,0 % | 60,0 % |
| excepcion_partida=True | 20 | 25,0 % | 50,0 % |
| material_misma_partida=False | 24 | 12,5 % | 62,5 % |
| material_misma_partida=True | 76 | 39,5 % | 56,6 % |
| operacion_insuficiente=False | 89 | 33,7 % | 57,3 % |
| operacion_insuficiente=True | 11 | 27,3 % | 63,6 % |
| origen_desconocido=False | 69 | 31,9 % | 59,4 % |
| origen_desconocido=True | 31 | 35,5 % | 54,8 % |
| originario=False | 30 | 33,3 % | 76,7 % |
| originario=True | 70 | 32,9 % | 50,0 % |
| proveedor_sin_declaracion=False | 77 | 36,4 % | 59,7 % |
| proveedor_sin_declaracion=True | 23 | 21,7 % | 52,2 % |
| redaccion=descriptiva | 50 | 34,0 % | 54,0 % |
| redaccion=siglas | 50 | 32,0 % | 62,0 % |
| regla=MERCOSUR-8481 | 28 | 28,6 % | 53,6 % |
| regla=MERCOSUR-8516 | 31 | 51,6 % | 67,7 % |
| regla=MERCOSUR-8708 | 17 | 41,2 % | 58,8 % |
| regla=MERCOSUR-9401 | 24 | 8,3 % | 50,0 % |
| subdivision_distinta=False | 35 | 25,7 % | 65,7 % |
| subdivision_distinta=True | 65 | 36,9 % | 53,8 % |
| tipo=criterio_de_valor | 47 | 40,4 % | 55,3 % |
| tipo=no_originario | 30 | 33,3 % | 76,7 % |
| tipo=solo_cambio_de_clasificacion | 23 | 17,4 % | 39,1 % |
| tolerancia_no_amplia_maxnom=False | 89 | 33,7 % | 55,1 % |
| tolerancia_no_amplia_maxnom=True | 11 | 27,3 % | 81,8 % |
| tolerancia_usada=False | 88 | 35,2 % | 64,8 % |
| tolerancia_usada=True | 12 | 16,7 % | 8,3 % |
| umbral_exacto=False | 89 | 32,6 % | 61,8 % |
| umbral_exacto=True | 11 | 36,4 % | 27,3 % |

## 5. Longitud

| Modelo | Mediana (aciertos) | Mediana (fallos) | p95 | Cortadas |
|---|---|---|---|---|
| base | 1562 | 1488 | 3072 | 12,0 % |
| sft | 1346 | 2532 | 3072 | 20,0 % |

## 6. ¿Acierta por las razones correctas? (solo respuestas acertadas, heurístico)

| Modelo | Aciertos | Cita la partida | Porcentaje en el razonamiento | Suma en el razonamiento | Aciertos sin el cálculo a la vista |
|---|---|---|---|---|---|
| base | 33 | 100,0 % (n=33) | 97,0 % (n=33) | 97,0 % (n=33) | 1 |
| sft | 58 | 100,0 % (n=58) | 100,0 % (n=58) | 100,0 % (n=58) | 0 |

Se busca en el `<think>` el porcentaje esperado (±0,5 puntos) y la suma esperada de materiales no originarios (porcentaje × precio EXW leído del enunciado, ±0,5 %). Los casos con 0 % no se comprueban. La última columna son aciertos cuyo razonamiento no contiene el porcentaje ni la suma: candidatos a acierto por casualidad, para leer a mano.

## 7. Fallos para leer a mano (sft)

**cortada_sin_answer** (20 problemas): índices [2, 7, 8, 9, 15, 21, 27, 42, 44, 46, 47, 48, 49, 50, 53] …

- Problema 2 [originario=True, tipo=criterio_de_valor, regla=MERCOSUR-8516, criterio_aplicado=CTSH, criterios_validos=CTSH|MAXNOM, material_misma_partida=True, subdivision_distinta=True, excepcion_partida=False, proveedor_sin_declaracion=False, origen_desconocido=False, operacion_insuficiente=False, tolerancia_usada=False, umbral_exacto=True, tolerancia_no_amplia_maxnom=False, redaccion=descriptiva]
  - esperado: `{"originario": true, "criterio": "CTSH", "pct_no_originario": 55.0, "criterios_validos": ["CTSH", "MAXNOM"]}`
  - modelo: ` the correct one. Therefore, the answer is originario with either CTSH or MAXNOM. However, the problem says that the answer should be the correct one. Therefore`
  - motivo: no <answer> block found
- Problema 7 [originario=True, tipo=criterio_de_valor, regla=MERCOSUR-8708, criterio_aplicado=CTH, criterios_validos=CTH|MAXNOM, material_misma_partida=True, subdivision_distinta=False, excepcion_partida=False, proveedor_sin_declaracion=True, origen_desconocido=False, operacion_insuficiente=False, tolerancia_usada=False, umbral_exacto=False, tolerancia_no_amplia_maxnom=False, redaccion=descriptiva]
  - esperado: `{"originario": true, "criterio": "CTH", "pct_no_originario": 39.4, "criterios_validos": ["CTH", "MAXNOM"]}`
  - modelo: `are in the same heading) sum up to 10% of EXW, then the criterion is considered met. Therefore, in this case, the sum is 125.69 €, which is 8.91% of EXW (1410.0`
  - motivo: no <answer> block found
- Problema 8 [originario=True, tipo=criterio_de_valor, regla=MERCOSUR-8481, criterio_aplicado=RVC, criterios_validos=RVC, material_misma_partida=True, subdivision_distinta=True, excepcion_partida=True, proveedor_sin_declaracion=True, origen_desconocido=False, operacion_insuficiente=False, tolerancia_usada=False, umbral_exacto=True, tolerancia_no_amplia_maxnom=False, redaccion=siglas]
  - esperado: `{"originario": true, "criterio": "RVC", "pct_no_originario": 44.39, "criterios_validos": ["RVC"]}`
  - modelo: `erefore, the non-originary materials are body and pipe fittings. The positioner is in 9032.89, which is a different tariff heading, so it's not considered here.`
  - motivo: no <answer> block found

**veredicto_incorrecto** (15 problemas): índices [18, 20, 24, 37, 43, 51, 61, 63, 68, 74, 79, 93, 94, 95, 99]

- Problema 18 [originario=True, tipo=solo_cambio_de_clasificacion, regla=MERCOSUR-9401, criterio_aplicado=CC, criterios_validos=CC, material_misma_partida=True, subdivision_distinta=True, excepcion_partida=False, proveedor_sin_declaracion=False, origen_desconocido=False, operacion_insuficiente=False, tolerancia_usada=True, umbral_exacto=False, tolerancia_no_amplia_maxnom=False, redaccion=descriptiva]
  - esperado: `{"originario": true, "criterio": "CC", "pct_no_originario": 53.42, "criterios_validos": ["CC"]}`
  - modelo: `{   "originario": false,   "criterio": "NINGUNO",   "pct_no_originario": 53.42 }`
  - motivo: veredicto incorrecto: False
- Problema 20 [originario=True, tipo=criterio_de_valor, regla=MERCOSUR-8481, criterio_aplicado=RVC, criterios_validos=RVC, material_misma_partida=True, subdivision_distinta=True, excepcion_partida=True, proveedor_sin_declaracion=False, origen_desconocido=True, operacion_insuficiente=False, tolerancia_usada=False, umbral_exacto=True, tolerancia_no_amplia_maxnom=False, redaccion=siglas]
  - esperado: `{"originario": true, "criterio": "RVC", "pct_no_originario": 41.48, "criterios_validos": ["RVC"]}`
  - modelo: `{   "originario": false,   "criterio": "NINGUNO",   "pct_no_originario": "41.63" }`
  - motivo: veredicto incorrecto: False
- Problema 24 [originario=True, tipo=solo_cambio_de_clasificacion, regla=MERCOSUR-9401, criterio_aplicado=CC, criterios_validos=CC, material_misma_partida=True, subdivision_distinta=True, excepcion_partida=False, proveedor_sin_declaracion=False, origen_desconocido=False, operacion_insuficiente=False, tolerancia_usada=True, umbral_exacto=False, tolerancia_no_amplia_maxnom=False, redaccion=siglas]
  - esperado: `{"originario": true, "criterio": "CC", "pct_no_originario": 51.34, "criterios_validos": ["CC"]}`
  - modelo: `{   "originario": false,   "criterio": "NINGUNO",   "pct_no_originario": "51.34" }`
  - motivo: veredicto incorrecto: False

**porcentaje_incorrecto** (7 problemas): índices [14, 22, 25, 28, 30, 81, 86]

- Problema 14 [originario=True, tipo=solo_cambio_de_clasificacion, regla=MERCOSUR-9401, criterio_aplicado=CC, criterios_validos=CC, material_misma_partida=True, subdivision_distinta=True, excepcion_partida=False, proveedor_sin_declaracion=False, origen_desconocido=False, operacion_insuficiente=False, tolerancia_usada=True, umbral_exacto=False, tolerancia_no_amplia_maxnom=False, redaccion=siglas]
  - esperado: `{"originario": true, "criterio": "CC", "pct_no_originario": 25.31, "criterios_validos": ["CC"]}`
  - modelo: `{   "originario": true,   "criterio": "CC",   "pct_no_originario": "2.38" }`
  - motivo: porcentaje fuera de ±0.5: 2.38 vs 25.31
- Problema 22 [originario=True, tipo=solo_cambio_de_clasificacion, regla=MERCOSUR-9401, criterio_aplicado=CC, criterios_validos=CC, material_misma_partida=True, subdivision_distinta=True, excepcion_partida=False, proveedor_sin_declaracion=False, origen_desconocido=False, operacion_insuficiente=False, tolerancia_usada=True, umbral_exacto=False, tolerancia_no_amplia_maxnom=False, redaccion=descriptiva]
  - esperado: `{"originario": true, "criterio": "CC", "pct_no_originario": 40.6, "criterios_validos": ["CC"]}`
  - modelo: `{   "originario": true,   "criterio": "CC",   "pct_no_originario": "7.22" }`
  - motivo: porcentaje fuera de ±0.5: 7.22 vs 40.6
- Problema 25 [originario=True, tipo=solo_cambio_de_clasificacion, regla=MERCOSUR-8708, criterio_aplicado=CTH, criterios_validos=CTH, material_misma_partida=True, subdivision_distinta=True, excepcion_partida=False, proveedor_sin_declaracion=True, origen_desconocido=False, operacion_insuficiente=False, tolerancia_usada=True, umbral_exacto=False, tolerancia_no_amplia_maxnom=False, redaccion=descriptiva]
  - esperado: `{"originario": true, "criterio": "CTH", "pct_no_originario": 65.96, "criterios_validos": ["CTH"]}`
  - modelo: `{   "originario": true,   "criterio": "CTH",   "pct_no_originario": 7.30 }`
  - motivo: porcentaje fuera de ±0.5: 7.3 vs 65.96

Para ver el razonamiento completo: `uv run --extra train python -m rlm.try_model <índice>` (o leer `raw` en el JSON).
