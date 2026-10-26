# Análisis de `phase1_eval_test.json`

Modelos: base, sft.

## 1. pass@1

| Modelo | n | Aciertos | IC 95 % | Lenient | Formato válido | Cortadas |
|---|---|---|---|---|---|---|
| base | 200 | 56 (28,0 %) | 22,2 % – 34,6 % | 58 (29,0 %) | 19,0 % | 10,5 % |
| sft | 200 | 112 (56,0 %) | 49,1 % – 62,7 % | 113 (56,5 %) | 83,5 % | 16,5 % |

Con n = 200, el intervalo llega a medir unos ±7 puntos: diferencias de pass@1 menores que eso entre dos modelos no son concluyentes por sí solas (la comparación por parejas de la sección 2 es más fina).

## 2. base frente a sft (mismos problemas)

|  | Problemas |
|---|---|
| Aciertan los dos | 43 |
| Solo sft (arreglados) | 69 |
| Solo base (estropeados) | 13 |
| Fallan los dos | 75 |

Test de McNemar exacto sobre los 82 problemas en que discrepan: p = 2,26e-10.

## 3. Tipos de fallo

| Tipo | base | sft |
|---|---|---|
| veredicto_incorrecto | 39 | 32 |
| cortada_sin_answer | 20 | 33 |
| porcentaje_incorrecto | 29 | 22 |
| tool_call_en_vez_de_answer | 38 | 0 |
| criterio_incorrecto | 11 | 1 |
| sin_answer | 7 | 0 |

`tool_call_en_vez_de_answer`: escribe `<tool_call>` tras `</think>`. `cortada_sin_answer`: llega al límite de tokens sin responder.

## 4. pass@1 por grupo (n ≥ 5)

| Grupo | n | base | sft | Profesor (aceptación) |
|---|---|---|---|---|
| todos | 200 | 28,0 % | 56,0 % | 64,1 % (n=2000) |
| criterio_aplicado=CC | 13 | 38,5 % | 69,2 % | 75,8 % (n=124) |
| criterio_aplicado=CTH | 54 | 16,7 % | 48,1 % | 56,3 % (n=504) |
| criterio_aplicado=CTSH | 28 | 25,0 % | 14,3 % | 27,8 % (n=144) |
| criterio_aplicado=MAXNOM | 27 | 40,7 % | 100,0 % | 97,5 % (n=404) |
| criterio_aplicado=NINGUNO | 65 | 36,9 % | 63,1 % | 54,3 % (n=700) |
| criterio_aplicado=RVC | 13 | 0,0 % | 38,5 % | 72,6 % (n=124) |
| criterios_validos=CC | 6 | 33,3 % | 83,3 % | 75,0 % (n=44) |
| criterios_validos=CC|MAXNOM | 7 | 42,9 % | 57,1 % | 76,2 % (n=80) |
| criterios_validos=CTH | 27 | 7,4 % | 25,9 % | 39,1 % (n=276) |
| criterios_validos=CTH|MAXNOM | 13 | 30,8 % | 84,6 % | 87,0 % (n=108) |
| criterios_validos=CTH|RVC | 14 | 21,4 % | 57,1 % | 68,3 % (n=120) |
| criterios_validos=CTSH | 28 | 25,0 % | 14,3 % | 27,8 % (n=144) |
| criterios_validos=MAXNOM | 27 | 40,7 % | 100,0 % | 97,5 % (n=404) |
| criterios_validos=NINGUNO | 65 | 36,9 % | 63,1 % | 54,3 % (n=700) |
| criterios_validos=RVC | 13 | 0,0 % | 38,5 % | 72,6 % (n=124) |
| excepcion_partida=False | 169 | 30,2 % | 58,6 % | 65,2 % (n=1680) |
| excepcion_partida=True | 31 | 16,1 % | 41,9 % | 58,1 % (n=320) |
| material_misma_partida=False | 75 | 29,3 % | 56,0 % | 65,8 % (n=708) |
| material_misma_partida=True | 125 | 27,2 % | 56,0 % | 63,2 % (n=1292) |
| operacion_insuficiente=False | 183 | 28,4 % | 58,5 % | 68,7 % (n=1812) |
| operacion_insuficiente=True | 17 | 23,5 % | 29,4 % | 19,7 % (n=188) |
| origen_desconocido=False | 142 | 27,5 % | 56,3 % | 65,2 % (n=1424) |
| origen_desconocido=True | 58 | 29,3 % | 55,2 % | 61,3 % (n=576) |
| originario=False | 65 | 36,9 % | 63,1 % | 54,3 % (n=700) |
| originario=True | 135 | 23,7 % | 52,6 % | 69,4 % (n=1300) |
| proveedor_sin_declaracion=False | 150 | 32,0 % | 61,3 % | 66,8 % (n=1464) |
| proveedor_sin_declaracion=True | 50 | 16,0 % | 40,0 % | 56,7 % (n=536) |
| regla=CETA-8413 | 39 | 30,8 % | 28,2 % | 40,8 % (n=316) |
| regla=TCA-8414 | 33 | 6,1 % | 36,4 % | 35,0 % (n=320) |
| regla=TCA-8507 | 38 | 39,5 % | 68,4 % | 77,8 % (n=316) |
| regla=UE-JP-8501 | 38 | 10,5 % | 47,4 % | 64,7 % (n=360) |
| regla=UE-KR-8479 | 31 | 38,7 % | 93,5 % | 86,5 % (n=364) |
| regla=UE-KR-9403 | 21 | 52,4 % | 76,2 % | 76,2 % (n=324) |
| subdivision_distinta=False | 99 | 30,3 % | 54,5 % | 66,9 % (n=852) |
| subdivision_distinta=True | 101 | 25,7 % | 57,4 % | 62,0 % (n=1148) |
| tipo=criterio_de_valor | 74 | 28,4 % | 74,3 % | 86,2 % (n=836) |
| tipo=no_originario | 65 | 36,9 % | 63,1 % | 54,3 % (n=700) |
| tipo=solo_cambio_de_clasificacion | 61 | 18,0 % | 26,2 % | 39,0 % (n=464) |
| tolerancia_no_amplia_maxnom=False | 186 | 27,4 % | 55,4 % | 63,1 % (n=1776) |
| tolerancia_no_amplia_maxnom=True | 14 | 35,7 % | 64,3 % | 71,9 % (n=224) |
| tolerancia_usada=False | 170 | 31,2 % | 64,7 % | 69,7 % (n=1804) |
| tolerancia_usada=True | 30 | 10,0 % | 6,7 % | 12,8 % (n=196) |
| umbral_exacto=False | 177 | 27,7 % | 53,7 % | 61,2 % (n=1744) |
| umbral_exacto=True | 23 | 30,4 % | 73,9 % | 84,0 % (n=256) |

Profesor: trazas verificadas / generadas en los 500 problemas de destilación (Qwen3-4B, temperatura 0,6, problemas de `train`). Orientativo: no son los mismos problemas ni la misma forma de generar.

## 5. Longitud

| Modelo | Mediana (aciertos) | Mediana (fallos) | p95 | Cortadas |
|---|---|---|---|---|
| base | 1440 | 1438 | 3072 | 10,5 % |
| sft | 1071 | 1696 | 3072 | 16,5 % |

## 6. ¿Acierta por las razones correctas? (solo respuestas acertadas, heurístico)

| Modelo | Aciertos | Cita la partida | Porcentaje en el razonamiento | Suma en el razonamiento | Aciertos sin el cálculo a la vista |
|---|---|---|---|---|---|
| base | 56 | 92,9 % (n=56) | 100,0 % (n=54) | 96,3 % (n=54) | 0 |
| sft | 112 | 100,0 % (n=112) | 97,3 % (n=111) | 100,0 % (n=111) | 0 |

Se busca en el `<think>` el porcentaje esperado (±0,5 puntos) y la suma esperada de materiales no originarios (porcentaje × precio EXW leído del enunciado, ±0,5 %). Los casos con 0 % no se comprueban. La última columna son aciertos cuyo razonamiento no contiene el porcentaje ni la suma: candidatos a acierto por casualidad, para leer a mano.

## 7. Fallos para leer a mano (sft)

**cortada_sin_answer** (33 problemas): índices [3, 5, 14, 23, 33, 40, 41, 55, 57, 79, 81, 82, 86, 88, 103] …

- Problema 3 [originario=True, tipo=criterio_de_valor, regla=UE-JP-8501, criterio_aplicado=CTH, criterios_validos=CTH|RVC, material_misma_partida=False, subdivision_distinta=False, excepcion_partida=False, proveedor_sin_declaracion=False, origen_desconocido=False, operacion_insuficiente=False, tolerancia_usada=False, umbral_exacto=False, tolerancia_no_amplia_maxnom=False]
  - esperado: `{"originario": true, "criterio": "CTH", "pct_no_originario": 4.79, "criterios_validos": ["CTH", "RVC"]}`
  - modelo: `nd the stator and rotor (36.92 €)? Wait, but the stator and rotor are in 8503, which is excluded. So they are not considered as originarily. Therefore, the non-`
  - motivo: no <answer> block found
- Problema 5 [originario=True, tipo=criterio_de_valor, regla=TCA-8507, criterio_aplicado=CTH, criterios_validos=CTH|MAXNOM, material_misma_partida=False, subdivision_distinta=False, excepcion_partida=False, proveedor_sin_declaracion=True, origen_desconocido=False, operacion_insuficiente=False, tolerancia_usada=False, umbral_exacto=True, tolerancia_no_amplia_maxnom=False]
  - esperado: `{"originario": true, "criterio": "CTH", "pct_no_originario": 50.0, "criterios_validos": ["CTH", "MAXNOM"]}`
  - modelo: ` the problem says that the rule allows either, the answer can be either. However, the user's answer should pick one.   But the problem says that the rule allows`
  - motivo: no <answer> block found
- Problema 14 [originario=True, tipo=solo_cambio_de_clasificacion, regla=CETA-8413, criterio_aplicado=CTSH, criterios_validos=CTSH, material_misma_partida=True, subdivision_distinta=True, excepcion_partida=False, proveedor_sin_declaracion=False, origen_desconocido=True, operacion_insuficiente=False, tolerancia_usada=False, umbral_exacto=False, tolerancia_no_amplia_maxnom=False]
  - esperado: `{"originario": true, "criterio": "CTSH", "pct_no_originario": 22.73, "criterios_validos": ["CTSH"]}`
  - modelo: `refore, since there's a non-originary material (parts of the pump) in the same subpartida, CTSH is not met. Therefore, the product is not originario.  But wait,`
  - motivo: no <answer> block found

**veredicto_incorrecto** (32 problemas): índices [0, 8, 17, 21, 24, 26, 30, 42, 44, 45, 48, 50, 59, 65, 83] …

- Problema 0 [originario=True, tipo=solo_cambio_de_clasificacion, regla=CETA-8413, criterio_aplicado=CTSH, criterios_validos=CTSH, material_misma_partida=False, subdivision_distinta=False, excepcion_partida=False, proveedor_sin_declaracion=False, origen_desconocido=False, operacion_insuficiente=False, tolerancia_usada=False, umbral_exacto=False, tolerancia_no_amplia_maxnom=False]
  - esperado: `{"originario": true, "criterio": "CTSH", "pct_no_originario": 0.0, "criterios_validos": ["CTSH"]}`
  - modelo: `{   "originario": false,   "criterio": "NINGUNO",   "pct_no_originario": "16.51" }`
  - motivo: veredicto incorrecto: False
- Problema 8 [originario=True, tipo=solo_cambio_de_clasificacion, regla=CETA-8413, criterio_aplicado=CTSH, criterios_validos=CTSH, material_misma_partida=True, subdivision_distinta=False, excepcion_partida=False, proveedor_sin_declaracion=False, origen_desconocido=False, operacion_insuficiente=False, tolerancia_usada=True, umbral_exacto=False, tolerancia_no_amplia_maxnom=False]
  - esperado: `{"originario": true, "criterio": "CTSH", "pct_no_originario": 24.83, "criterios_validos": ["CTSH"]}`
  - modelo: `{   "originario": false,   "criterio": "NINGUNO",   "pct_no_originario": "24.83" }`
  - motivo: veredicto incorrecto: False
- Problema 17 [originario=False, tipo=no_originario, regla=UE-JP-8501, criterio_aplicado=NINGUNO, criterios_validos=NINGUNO, material_misma_partida=True, subdivision_distinta=True, excepcion_partida=False, proveedor_sin_declaracion=False, origen_desconocido=False, operacion_insuficiente=True, tolerancia_usada=False, umbral_exacto=False, tolerancia_no_amplia_maxnom=False]
  - esperado: `{"originario": false, "criterio": "NINGUNO", "pct_no_originario": 55.13, "criterios_validos": []}`
  - modelo: `{   "originario": true,   "criterio": "RVC",   "pct_no_originario": 55.17 }`
  - motivo: veredicto incorrecto: True

**porcentaje_incorrecto** (22 problemas): índices [9, 12, 27, 29, 31, 32, 39, 47, 51, 62, 71, 73, 76, 80, 84] …

- Problema 9 [originario=True, tipo=solo_cambio_de_clasificacion, regla=TCA-8414, criterio_aplicado=CTH, criterios_validos=CTH, material_misma_partida=False, subdivision_distinta=False, excepcion_partida=False, proveedor_sin_declaracion=True, origen_desconocido=False, operacion_insuficiente=False, tolerancia_usada=False, umbral_exacto=False, tolerancia_no_amplia_maxnom=False]
  - esperado: `{"originario": true, "criterio": "CTH", "pct_no_originario": 10.74, "criterios_validos": ["CTH"]}`
  - modelo: `{   "originario": true,   "criterio": "CTH",   "pct_no_originario": "0.00" }`
  - motivo: porcentaje fuera de ±0.5: 0.0 vs 10.74
- Problema 12 [originario=True, tipo=solo_cambio_de_clasificacion, regla=TCA-8414, criterio_aplicado=CTH, criterios_validos=CTH, material_misma_partida=True, subdivision_distinta=True, excepcion_partida=False, proveedor_sin_declaracion=False, origen_desconocido=False, operacion_insuficiente=False, tolerancia_usada=True, umbral_exacto=False, tolerancia_no_amplia_maxnom=False]
  - esperado: `{"originario": true, "criterio": "CTH", "pct_no_originario": 67.63, "criterios_validos": ["CTH"]}`
  - modelo: `{   "originario": true,   "criterio": "CTH",   "pct_no_originario": "9.75" }`
  - motivo: porcentaje fuera de ±0.5: 9.75 vs 67.63
- Problema 27 [originario=False, tipo=no_originario, regla=UE-JP-8501, criterio_aplicado=NINGUNO, criterios_validos=NINGUNO, material_misma_partida=True, subdivision_distinta=True, excepcion_partida=True, proveedor_sin_declaracion=True, origen_desconocido=False, operacion_insuficiente=False, tolerancia_usada=False, umbral_exacto=False, tolerancia_no_amplia_maxnom=True]
  - esperado: `{"originario": false, "criterio": "NINGUNO", "pct_no_originario": 60.0, "criterios_validos": []}`
  - modelo: `{   "originario": false,   "criterio": "NINGUNO",   "pct_no_originario": "32.73" }`
  - motivo: porcentaje fuera de ±0.5: 32.73 vs 60.0

**criterio_incorrecto** (1 problemas): índices [61]

- Problema 61 [originario=True, tipo=criterio_de_valor, regla=UE-JP-8501, criterio_aplicado=RVC, criterios_validos=RVC, material_misma_partida=False, subdivision_distinta=False, excepcion_partida=True, proveedor_sin_declaracion=True, origen_desconocido=False, operacion_insuficiente=False, tolerancia_usada=False, umbral_exacto=False, tolerancia_no_amplia_maxnom=False]
  - esperado: `{"originario": true, "criterio": "RVC", "pct_no_originario": 30.02, "criterios_validos": ["RVC"]}`
  - modelo: `{   "originario": true,   "criterio": "CTH",   "pct_no_originario": 30.00 }`
  - motivo: criterio 'CTH' no válido (válidos: ['RVC'])

Para ver el razonamiento completo: `uv run --extra train python -m rlm.try_model <índice>` (o leer `raw` en el JSON).
