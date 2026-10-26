# Análisis de `prueba_eval.json`

Modelos: base.

## 1. pass@1

| Modelo | n | Aciertos | IC 95 % | Lenient | Formato válido | Cortadas |
|---|---|---|---|---|---|---|
| base | 4 | 1 (25,0 %) | 4,6 % – 69,9 % | 1 (25,0 %) | 25,0 % | 25,0 % |

Con n = 4, el intervalo llega a medir unos ±49 puntos: diferencias de pass@1 menores que eso entre dos modelos no son concluyentes por sí solas (la comparación por parejas de la sección 2 es más fina).

## 3. Tipos de fallo

| Tipo | base |
|---|---|
| tool_call_en_vez_de_answer | 2 |
| cortada_sin_answer | 1 |

`tool_call_en_vez_de_answer`: escribe `<tool_call>` tras `</think>`. `cortada_sin_answer`: llega al límite de tokens sin responder.

## 4. pass@1 por grupo (n ≥ 5)

| Grupo | n | base |
|---|---|---|

## 5. Longitud

| Modelo | Mediana (aciertos) | Mediana (fallos) | p95 | Cortadas |
|---|---|---|---|---|
| base | 1514 | 933 | 1514 | 25,0 % |

## 6. ¿Acierta por las razones correctas? (solo respuestas acertadas, heurístico)

| Modelo | Aciertos | Cita la partida | Porcentaje en el razonamiento | Suma en el razonamiento | Aciertos sin el cálculo a la vista |
|---|---|---|---|---|---|
| base | 1 | 100,0 % (n=1) | 100,0 % (n=1) | 100,0 % (n=1) | 0 |

Se busca en el `<think>` el porcentaje esperado (±0,5 puntos) y la suma esperada de materiales no originarios (porcentaje × precio EXW leído del enunciado, ±0,5 %). Los casos con 0 % no se comprueban. La última columna son aciertos cuyo razonamiento no contiene el porcentaje ni la suma: candidatos a acierto por casualidad, para leer a mano.

## 7. Fallos para leer a mano (base)

**tool_call_en_vez_de_answer** (2 problemas): índices [0, 2]

- Problema 0 [originario=True, tipo=solo_cambio_de_clasificacion, regla=CETA-8413, criterio_aplicado=CTSH, criterios_validos=CTSH, material_misma_partida=False, subdivision_distinta=False, excepcion_partida=False, proveedor_sin_declaracion=False, origen_desconocido=False, operacion_insuficiente=False, tolerancia_usada=False, umbral_exacto=False, tolerancia_no_amplia_maxnom=False]
  - esperado: `{"originario": true, "criterio": "CTSH", "pct_no_originario": 0.0, "criterios_validos": ["CTSH"]}`
  - modelo: `answer would be originario, criterio is CTSH, and pct_no_originario is 0.00% because all materials are originarios. </think>  <tool_call> reasoning process here`
  - motivo: no <answer> block found
- Problema 2 [originario=True, tipo=criterio_de_valor, regla=UE-JP-8501, criterio_aplicado=CTH, criterios_validos=CTH|RVC, material_misma_partida=True, subdivision_distinta=True, excepcion_partida=False, proveedor_sin_declaracion=False, origen_desconocido=True, operacion_insuficiente=False, tolerancia_usada=False, umbral_exacto=False, tolerancia_no_amplia_maxnom=False]
  - esperado: `{"originario": true, "criterio": "CTH", "pct_no_originario": 31.48, "criterios_validos": ["CTH", "RVC"]}`
  - modelo: `ay over 10%. So the tolerance doesn't apply here. Therefore, both criteria are valid, so the product is originario. </think>  <tool_call> reasoning process here`
  - motivo: no <answer> block found

**cortada_sin_answer** (1 problemas): índices [3]

- Problema 3 [originario=True, tipo=criterio_de_valor, regla=UE-JP-8501, criterio_aplicado=CTH, criterios_validos=CTH|RVC, material_misma_partida=False, subdivision_distinta=False, excepcion_partida=False, proveedor_sin_declaracion=False, origen_desconocido=False, operacion_insuficiente=False, tolerancia_usada=False, umbral_exacto=False, tolerancia_no_amplia_maxnom=False]
  - esperado: `{"originario": true, "criterio": "CTH", "pct_no_originario": 4.79, "criterios_validos": ["CTH", "RVC"]}`
  - modelo: `h "criterio" as "CTH" and "RVC" both. But the user's instruction might require that. Therefore, the final answer is:  {   "originario": true,   "criterio": "CTH`
  - motivo: no <answer> block found

Para ver el razonamiento completo: `uv run --extra train python -m rlm.try_model <índice>` (o leer `raw` en el JSON).
