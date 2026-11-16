# 5. Fase 3: conocimiento

*Borrador de la sección 5 del informe (plantilla: una página y media). El detalle de cada
experimento está en `EXPERIMENTS.md`, en las entradas de la fase 3, del 12 al 16 de noviembre.*

**El corpus.** Son 29 documentos y 1,36 millones de caracteres, de 14 fuentes:
- el capítulo o protocolo de origen, las notas introductorias y la declaración de origen de seis
  acuerdos (TCA, Japón, Corea, CETA, PEM revisado y Mercosur), descargados de Cellar, el
  repositorio de EUR-Lex;
- las notas explicativas de la NC de los capítulos 84-85, 87 y 94;
- siete guías de la Comisión y de la AEAT.

En cuanto a licencias, los documentos de la UE se reutilizan citando la fuente (Decisión
2011/833/UE) y los de la AEAT según sus condiciones de reutilización.

`rag/download.py` reconstruye el corpus y comprueba las huellas en `rag/sources.lock.json`. Cuatro
guías se bajan a mano, porque su web bloquea los scripts y no saltamos esa comprobación.

Las listas de reglas por partida quedan fuera, en contra de lo que decía la propuesta: son tablas
que se trocean mal, y la herramienta `get_origin_rule` de la fase 2 ya las sirve exactas.

**El conjunto dorado.** Tiene 64 preguntas escritas leyendo el corpus:
- 29 de un dato;
- 18 que se repiten entre acuerdos (`cross`): tolerancia, validez, conservación de registros;
- 3 de dos acuerdos a la vez;
- 8 en español;
- 6 sin respuesta en el corpus.

Cada pregunta guarda la **cita literal** que la responde, no el identificador de un fragmento. Así
el mismo dorado mide cualquier chunking: un fragmento es relevante si contiene al menos la mitad de
la cita. Recall@k se cuenta por evidencia y MRR por el primer fragmento relevante.

**Chunking y retrievers.** Todas las estrategias tienen tope de 1.500 caracteres y la misma cabecera
`[documento › sección]`. El recursivo de 1.500 es un control de tamaño.

| Chunking | Fragmentos | Mediana (car.) | BM25 R@5 / MRR | Denso R@5 / MRR | Híbrido: λ, R@5 / MRR |
|---|---|---|---|---|---|
| Recursivo 800 | 2.266 | 678 | 0,60 / 0,46 | 0,74 / 0,55 | 0,4: 0,78 / 0,60 |
| Recursivo 1.500 | 1.106 | 1.375 | 0,49 / 0,37 | 0,77 / 0,55 | 0,4: 0,73 / 0,59 |
| Semántico (umbral 0,7) | 1.426 | 1.076 | 0,47 / 0,42 | 0,72 / 0,52 | 0,4: 0,76 / 0,60 |
| **Estructural** | 1.523 | 963 | 0,56 / 0,47 | **0,88** / 0,67 | **0,3: 0,84 / 0,72** |

- **Gana el estructural** (una unidad por artículo, nota NC o pregunta de las Q&A), y no por el
  tamaño: el recursivo de 1.500 tiene fragmentos más largos y no lo iguala. La evidencia queda
  entera en un fragmento, y la cabecera nombra el acuerdo y el artículo.
- **El umbral del semántico (0,7)** se eligió con el dorado y con la gráfica de distancias entre
  frases consecutivas. A 0,75 el MRR sube 0,02, pero el 37 % de los fragmentos ya los cierra el
  tope de tamaño y no la distancia.
- **BM25 falla en las `cross`** (MRR de 0,38 frente a 0,73 del denso): todos los acuerdos usan las
  mismas palabras.
- **La ablación más reveladora es quitar la cabecera.** El MRR del denso cae de 0,67 a 0,40, y en
  las `cross` de 0,73 a 0,25. Lo que más pesa en la recuperación es saber de qué acuerdo viene cada
  fragmento.
- **La instrucción de la consulta** de Qwen3-Embedding también mueve los resultados. Una de dominio
  sube el MRR del denso a 0,72, pero baja las preguntas en español (0,73 → 0,62). En el híbrido,
  todas las instrucciones quedan entre 0,70 y 0,73, así que nos quedamos con la genérica.

**Cómo se eligió λ.** Barrido de 0 a 1 en pasos de 0,1, eligiendo por MRR (si empata, por R@5 y
luego el λ menor). Para no ajustarlo al dorado, lo comprobamos en 200 particiones aleatorias en dos
mitades: se elige λ con una mitad y se mide el MRR en la otra. En el estructural, el λ elegido así da
0,69 en la otra mitad, frente a 0,67 del denso. El elegido va de 0,1 a 0,5, y 0,3 sale en 85 de las
200 particiones. Servimos λ = 0,3 (curvas en `reports/phase3_eval_lambda.png`).

**Respuestas y citas.** Qwen3-1.7B, sin thinking, con las reglas en el mensaje de sistema y los 5
pasajes numerados en el de usuario.
- Numeramos los pasajes porque el modelo copiaba mal los identificadores de 10 caracteres. Después,
  el código vuelve a poner los ids.
- `extract_citations` descarta toda cita a un fragmento que no se devolvió: cada cita existe por
  construcción. Que cite el fragmento correcto se mide aparte, en el 71 % de las respuestas en las
  que se recuperó.
- Como en las fases 1 y 2, cada respuesta se puntúa como correcta o incorrecta: cada pregunta lleva
  el dato clave que debe contener (`answer_contains`).

| Variante | Con respuesta (58) | Sin respuesta (6): "no lo sé" |
|---|---|---|
| El mismo modelo sin RAG | 0 | 5 |
| BM25 | 43 | 1 |
| Denso | 50 | 4 |
| **Híbrido** (servido) | 48 | 4 |

- **Sin RAG** dice que no lo sabe en 46 de las 58 preguntas con respuesta, y en las otras 12 se
  inventa plazos e importes.
- **Con la evidencia entre los 5 recuperados,** el generador acierta alrededor del 90 % con los tres
  retrievers. Lo que los separa es la recuperación.
- **El denso y el híbrido empatan en las respuestas** (2 preguntas de diferencia). Se sirve el
  híbrido por su MRR.

**Qué preguntas fallan y por qué.** El fallo que más se repite es confundir acuerdos, el mismo que la
cabecera resolvió en la recuperación:
- en q27 responde con los 500 € de otro régimen, en lugar de los 1.000 $ de Corea;
- en q43, con el 10 % de tolerancia de los demás acuerdos, en lugar del 15 % del PEM;
- en q59 (Australia, que no está en el corpus), con los 12 meses de Japón.

Otros fallos:
- **Reglas que cambian con la fecha:** en q09 (baterías) mezcla las de 2021-2023 con las vigentes.
- **Preguntas de dos acuerdos:** en q46 responde solo una mitad.
- **Evidencia fuera de los 5 primeros:** pasa en 9 preguntas, pero en 5 de ellas responde bien con
  otro fragmento. Las guías de la Comisión repiten muchos datos de los acuerdos, y el dorado anota un
  solo fragmento por pregunta.

**Más allá: un reranker.** Probamos Qwen3-Reranker-0.6B sobre los 20 primeros resultados.
- **Solo mejora BM25** (MRR de 0,47 a 0,61). Al híbrido lo empeora (de 0,72 a 0,64): reconoce el
  tema, pero no de qué acuerdo habla el fragmento.
- **En las respuestas** sube de 48 a 52 de 58, pero solo se abstiene en 2 de las 6 preguntas sin
  respuesta.
- **Lo dejamos desactivado.**
