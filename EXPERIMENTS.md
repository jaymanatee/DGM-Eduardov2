# EXPERIMENTS.md — cuaderno de experimentos

Una entrada cada vez que lanzamos algo o tomamos una decisión, de la más antigua a la más
reciente. Formato de `docs/EXPERIMENTS_plantilla.md`.

## Versiones del enunciado

Las trazas de un enunciado no valen para otro, así que cada fichero de trazas lleva en el nombre
la versión con la que se generó. Los problemas (parámetros y respuesta correcta) son los mismos
en todas las versiones; solo cambia el texto.

| Versión | Qué cambia en el enunciado | Commit / fichero |
|---|---|---|
| E1 | El original | `39cbf04` |
| E2 | + «Si el producto no es originario, "criterio" debe ser exactamente "NINGUNO"» | `4d6d614` |
| E3 | + definiciones genéricas de CC, CTH, CTSH y MaxNOM, y «basta con una alternativa» | sin commit |
| E4 | E3 + «Solo cuentan los criterios que aparecen en la regla de la partida» | sin commit |
| E5 | Las definiciones genéricas se sustituyen por «Cómo se aplica esta regla», solo con los criterios de esa regla; la pregunta lista solo esos criterios | sin commit |
| E6 | E5 + la tolerancia dice qué materiales entran en la suma | **actual** |

## Ficheros de datos

Nombres: `E<versión>_trazas_<problemas>[_<variante>].jsonl`, en `rlm/data/experimentos/`. Los
problemas sin trazas no se guardan: se regeneran con el generador y la semilla.

| Fichero | Problemas de `train` | Notas |
|---|---|---|
| `E1_trazas_train000-009.jsonl` | 0 a 9 | Piloto E1 |
| `E2_trazas_train000-255.jsonl` | 0 a 255 | Tirada E2 interrumpida; los 20 primeros son el piloto E2 |
| `E3_trazas_dirigidos20_max2048.jsonl` | 18 de los 20 dirigidos | Cortado a mano |
| `E4_trazas_dirigidos20_max4096.jsonl` | 12 de los 20 dirigidos | Cortado por falta de memoria |
| `E5_trazas_dirigidos20.jsonl` | Los 20 dirigidos | Completo |
| `rlm/data/sft_traces.jsonl` | 0 a 499 | Tirada E6, la que va al SFT: 1.282 trazas válidas de 2.000 |

Los "20 dirigidos" son los problemas 0 a 10, 12, 15, 22, 26, 31, 37, 39, 47 y 49 de `train`:
los 10 primeros que solo son originarios por un criterio de cambio de clasificación y los 10
primeros del resto. Se obtienen con `uv run python -m rlm.pilot_subset_deprecated` (el script ya
no se usa en la cadena principal, de ahí el nombre).

Referencias fijas con el enunciado E6 (se actualizan solo si cambia el generador a propósito):

| Fichero | Comando | sha256 |
|---|---|---|
| `train.jsonl` | `--n 4000 --split train --seed 0` | `b61dcfc8012aea2d7343a48eed8549f0a07b7a6c6c91bb4526605839cbf4e3d2` |
| `test.jsonl` | `--n 200 --split test --seed 0` | `9e66110dd87cd5bc9d3970ae38795c76d4b1aad649194f6e6c47f0f7d95840be` |
| `test_ood.jsonl` | `--n 100 --split ood --seed 0` | `58787a033551ba6f68a0c92d9375985752f5855ba4de37b7b38223200cee08c8` |
| `sft_problemas.jsonl` | `--n 500 --split train --seed 0` | `3327deb2e5d4a831c458cc7731bddeafc61b70f80b9c6821e9427f1e766ca17d` |

Modelos y evaluaciones. Los pesos no se suben al repositorio: se regeneran con el comando de su
entrada.

| Ruta | Qué es | Entrada |
|---|---|---|
| `rlm/weights/sft_prueba/` | Adaptador LoRA de la prueba corta (0,1 épocas). Solo sirvió para medir memoria y velocidad | 2026-10-16 · Configuración del SFT y prueba corta |
| `rlm/weights/sft_lora/` | Adaptador LoRA del SFT completo (2 épocas). Es el elegido; `checkpoint-75` es el de la época 1 | 2026-10-24 · SFT completo |
| `reports/prueba_eval.json` | Modelo base en los 4 primeros problemas de `test` | 2026-10-16 · Modelo base en 4 problemas de test |
| `rlm/weights/final_rlm_lora/` | Adaptador LoRA del GRPO 1 (400 pasos de 1 problema sobre `sft_lora`), con `log_history.json` y `completions/` | 2026-11-02 · GRPO completo |
| `rlm/weights/grpo_pps3/` | Adaptador LoRA del GRPO 2 (270 pasos de 3 problemas sobre `sft_lora`), con `log_history.json` y `completions/`. Es el modelo final de la fase 1 | 2026-11-05 · GRPO 2 |
| `reports/grpo2_test.json`, `reports/grpo2_ood.json` | Respuestas completas y veredictos del GRPO 2 en `test` y `test_ood` | 2026-11-05 · Evaluación del GRPO 2 |
| `reports/phase1_analysis_grpo2_test.md`, `reports/phase1_analysis_grpo2_test_ood.md` | Análisis de base, SFT repetido, GRPO 1 y GRPO 2 (`rlm/analyze_eval.py --compare sft grpo2`) | 2026-11-05 · Evaluación del GRPO 2 |
| `reports/phase1_grpo_curves.png` | Curvas de recompensa, formato, cortadas y longitud de los dos GRPO | 2026-11-05 · GRPO 2 |
| `reports/phase1_eval_test.json`, `reports/phase1_eval_test_ood.json` | Respuestas completas y veredictos del base, el SFT y el GRPO en `test` y `test_ood` | 2026-10-26 · Evaluación del base y del SFT; 2026-11-03 · Evaluación del GRPO |
| `reports/phase1_analysis_test.md`, `reports/phase1_analysis_test_ood.md` | Análisis de esas evaluaciones (`rlm/analyze_eval.py`) | 2026-10-26 · Evaluación del base y del SFT; 2026-11-03 · Evaluación del GRPO |
| `reports/sft_rerun_test.json`, `reports/sft_rerun_ood.json` | El mismo SFT evaluado de nuevo, en las condiciones del GRPO | 2026-11-03 · Evaluación del GRPO |

---

## Entradas

### 2026-10-11 · Fase 1 · `distill.py` no enlazaba con el cargador de datos

**Qué queríamos saber.** Si la cadena generador → `load_domain_dataset` → `distill.py` →
verificador funcionaba de punta a punta antes de gastar GPU.

**Qué hicimos.** Leer el código y ejecutar `distill.py` con un profesor simulado (un modelo
falso que devuelve siempre la misma traza) sobre 4 problemas del generador.

**Qué pasó.** `load_domain_dataset` crea la columna `prompt` (sistema + usuario) y elimina
`question`. `distill.py` pedía `p["question"]` en dos sitios, así que habría fallado con
`KeyError` en el primer lote, después de cargar el profesor entero. Además, las filas de salida
solo conservaban `question` y `answer` del problema: se perdía `branches`, que es donde el
generador marca la regla y las trampas de cada problema.

**Qué concluimos.** Tres cambios en `distill.py` (commit `39cbf04`): usar `p["prompt"]` para
construir la entrada del profesor, leer el enunciado de `problem["prompt"][-1]["content"]`, y
copiar `branches` a cada fila de `sft_traces.jsonl`. Lo último permite calcular la tasa de
aceptación por regla y por trampa, no solo el total. `data.py` no se toca porque es código de la
plantilla y lo usan también el entrenamiento y la evaluación.

### 2026-10-11 · Fase 1 · Piloto E1: el profesor acierta el 50 %, pero casi nunca en los "no originario"

**Qué queríamos saber.** Tres cosas antes de lanzar la destilación completa: si el profesor
termina de razonar en 2.048 tokens, qué tasa de aceptación tiene en nuestro dominio y cuánto
tarda.

**Qué hicimos.** En la DGX (partición `H200 MIG 1g.18gb`, 17,2 GB visibles; torch 2.11.0+cu128,
transformers 5.17.0), con los 20 primeros problemas de `train`:

```
uv run python -m rlm.generate_problems --n 20 --split train --out rlm/data/piloto.jsonl
uv run python -m rlm.distill --data rlm/data/piloto.jsonl --output rlm/data/piloto_trazas.jsonl
```

Profesor `Qwen/Qwen3-4B` en modo thinking, 4 muestras por problema, temperatura 0,6, lotes de 2
problemas. Lo cortamos con 10 problemas hechos (40 trazas), guardadas en
`rlm/data/experimentos/E1_trazas_train000-009.jsonl`.

**Qué pasó.**

| Resultado | Trazas |
|---|---|
| Verificadas | 20 de 40 (50 %) |
| `wrong_criterion` | 11 |
| `wrong_verdict` | 8 |
| Cortadas a 2.048 tokens | 1 |

- 7 de los 10 problemas tienen al menos una traza válida.
- Longitud de las trazas no cortadas: de 630 a 1.744 tokens, mediana 950. El límite de 2.048
  basta.
- Memoria: 10,6 GB de los 16,4 de la partición.
- Tiempo: el primer lote tardó unos 12 minutos y los siguientes alrededor de 3 (medido a ojo).
  No sabemos explicar la diferencia solo con la longitud de las trazas.
- El reparto por tipo de respuesta es muy desigual: 17 trazas válidas de 24 en los problemas
  cuya respuesta es "originario", y solo 3 de 16 en los "no originario".

La causa del desequilibrio son los `wrong_criterion`. En los 11 casos el profesor acierta el
veredicto (no originario) pero escribe en `criterio` el criterio que ha comprobado y que no se
cumple, en vez de `NINGUNO`. Ejemplo, problema 6:

```
Esperado:  {"originario": false, "criterio": "NINGUNO", "pct_no_originario": 74.6}
Profesor:  {"originario": false, "criterio": "MAXNOM",  "pct_no_originario": 74.60}
```

En 8 de los 11 también el porcentaje es correcto: lo único que falla es la etiqueta. En los
problemas 8 y 9 (regla `UE-KR-8479`, solo MaxNOM) lo hacen las 8 muestras.

Los `wrong_verdict` sí son errores de razonamiento. El más claro es el problema 3 (batería,
regla "CTH o MaxNOM 50 %"): ningún material es de la partida 8507, así que CTH se cumple, pero
las 4 muestras comprueban solo el MaxNOM (56,9 %) y responden que no. En la traza se lee
«maybe CTH isn't applicable here». También hay errores de suma (problema 8: 748,76 € en lugar de
848,76 €). El profesor razona en inglés aunque el enunciado está en español, porque el prompt de
sistema de R1-Zero está en inglés.

**Qué concluimos.** La cadena técnica funciona. El 50 % no mide solo la dificultad del dominio:
una parte es una ambigüedad de nuestro enunciado, que pedía el criterio "que se cumpla" y dejaba
`NINGUNO` en un paréntesis. Si destilamos así, el SFT vería casi solo ejemplos de "originario:
true". El verificador hace bien en rechazarlas; lo que hay que cambiar es el enunciado
(siguiente entrada). Queda como hipótesis, con solo 10 problemas, que el profesor resuelve por
MaxNOM y falla los casos de cambio de clasificación; se medirá por trampa con `branches` en la
tirada completa.

### 2026-10-12 · Fase 1 · Enunciado E2: explícito cuándo poner `NINGUNO`

**Qué queríamos saber.** Si haciendo explícita la instrucción el profesor deja de poner el
criterio fallido en los casos "no originario".

**Qué hicimos.** Añadir una frase al final de la pregunta en `render` (commit `4d6d614`):

> Si el producto no es originario, "criterio" debe ser exactamente "NINGUNO", aunque hayas
> comprobado otros criterios.

No relajamos el verificador (aceptar cualquier criterio cuando no es originario) porque las
trazas enseñarían al alumno un formato distinto del que luego se le exige en `/reasoning`.

**Qué pasó.** Regenerados los 4.000 problemas de `train`: parámetros, respuesta, plantilla y
`branches` son idénticos a los de antes; solo cambia el texto del enunciado. La respuesta sigue
sin aparecer en ningún enunciado. La huella de `train.jsonl` pasa de `b8d69f8a…` (E1) a
`44f70a16…` (E2). Comprobado también que el generador da el mismo fichero en máquinas y
versiones de Python distintas (3.12 y 3.13).

**Qué concluimos.** Las 40 trazas del piloto E1 quedan como referencia del "antes" y no se usan
para SFT. Relanzamos el piloto con los mismos 20 problemas: como son los mismos, lo que cambie
en `wrong_criterion` se debe a la frase. Resultado en la entrada del piloto E2.

### 2026-10-12 · Fase 1 · Split fuera de distribución: UE-Mercosur, en catálogo aparte

**Qué queríamos saber.** Cómo definir el conjunto fuera de distribución sin modificar `train`,
porque las trazas se generan sobre los primeros problemas de `train` y dependen de que no cambie.

**Qué hicimos.** Añadir a `generate_problems.py` un catálogo reservado (`OOD_RULES`,
`OOD_PRODUCTS`), separado de `RULES` y `PRODUCTS`, que solo se usa con `--split ood`:

| Regla | Producto | Criterios | Qué no ha visto el modelo |
|---|---|---|---|
| `MERCOSUR-8708` | Caja de cambios | CTH o MaxNOM 45 % | El umbral |
| `MERCOSUR-9401` | Silla de oficina | CC | CC como único criterio |
| `MERCOSUR-8516` | Calentador eléctrico | CTSH o MaxNOM 55 % | La combinación y el umbral |
| `MERCOSUR-8481` | Válvula de control | CTH salvo 7307 y 8484, o RVC 60 % | Dos partidas excluidas y el umbral |

Además, la mitad de los problemas llevan la regla redactada sin siglas, al estilo de los
protocolos UE-Corea («fabricación a partir de materiales de cualquier partida, excepto la del
producto»). Queda marcado en `branches["redaccion"]` (`siglas` o `descriptiva`).

Va en diccionarios aparte porque `train` elige regla con `rng.choice(sorted(RULES))`: una
entrada más en `RULES` cambiaría el sorteo y la misma semilla dejaría de regenerar el mismo
`train.jsonl`.

**Qué pasó.** `train.jsonl` y `test.jsonl` salen idénticos byte a byte antes y después del
cambio. Los 100 problemas de `ood`: 70 originarios y 30 no, 50 con siglas y 50 descriptivos,
cada trampa presente en al menos 11 problemas, ninguna fuga de la respuesta.
`tests/test_generate_problems.py`, que seguía siendo el de la plantilla y no cargaba, está
reescrito para nuestro generador (10 tests), incluido uno que compara los 200 primeros problemas
de `train` con una huella fija y falla si algo altera el sorteo o el texto.

**Qué concluimos.** Dos diferencias respecto a la propuesta, a propósito:

- Las reglas de UE-Corea se quedan en `train` (la propuesta las ponía fuera). Con solo 6 reglas,
  reservar 2 dejaba el entrenamiento demasiado pobre. Lo que conservamos de aquella idea es la
  redacción sin siglas.
- Las reglas de Mercosur son de relleno, igual que 4 de las 6 de `train`: ejercitan el motor
  pero no son las del acuerdo real.

Las dos redacciones dan dos niveles de dificultad dentro del mismo split; en la evaluación
separaremos el pass@1 por `redaccion`. Nota posterior: desde el enunciado E5, la sección «Cómo se
aplica esta regla» nombra la sigla de cada alternativa, así que la redacción sin siglas es menos
difícil de lo que era al diseñarla.

### 2026-10-12 · Fase 1 · Piloto E2: del 50 % al 82,5 % con la frase de `NINGUNO`

**Qué queríamos saber.** Si la frase de `NINGUNO` corrige los `wrong_criterion` y equilibra las
trazas válidas entre "originario" y "no originario".

**Qué hicimos.** Mismo comando, mismo profesor y mismos problemas que el piloto E1, con el
enunciado E2. Comparamos los 10 primeros problemas (40 trazas), que son los mismos que en E1. El
piloto completo (20 problemas) terminó con un 65 % de trazas verificadas: los 10 segundos
problemas salieron bastante peor que los 10 primeros. Sus trazas están al principio de
`E2_trazas_train000-255.jsonl`.

**Qué pasó.**

| | Piloto E1 | Piloto E2 |
|---|---|---|
| Verificadas | 20 de 40 (50 %) | 33 de 40 (82,5 %) |
| En problemas "originario" | 17 de 24 | 20 de 24 |
| En problemas "no originario" | 3 de 16 | 13 de 16 |
| `wrong_criterion` | 11 | 1 |
| `wrong_verdict` | 8 | 4 |
| `wrong_pct` | 0 | 2 |
| Cortadas a 2.048 tokens | 1 | 0 |
| Problemas con alguna traza válida | 7 de 10 | 9 de 10 |

- Longitud de las trazas: de 585 a 1.497 tokens, mediana 960.
- Tiempo: unos 27 minutos para los 10 problemas en la DGX (medido a ojo), 2,7 minutos por
  problema. Como referencia, en un portátil con una RTX 5070 Ti de 12 GB la misma generación va
  a 14 pasos por segundo con 4 secuencias (`--batch-size 1`, prueba de 256 tokens).
- Los 4 `wrong_verdict` son las 4 muestras del problema 3, el mismo que fallaba entero en E1:
  CTH se cumple y el profesor solo comprueba el MaxNOM.
- Queda un `wrong_criterion` (problema 8) y dos errores de cálculo del porcentaje, uno de ellos
  un 771,81 % por una coma mal puesta.

**Qué concluimos.** La caída de `wrong_criterion` de 11 a 1 se debe a la frase; el fallo era
del enunciado y no del profesor. El resto de la mejora no se le puede atribuir del todo: el
problema 1 pasa de 2 a 4 aciertos y la frase no le afecta, así que parte es azar del muestreo
(temperatura 0,6, 10 problemas). El límite real del profesor que queda a la vista es el cambio
de clasificación cuando el MaxNOM no se cumple. Dos observaciones para más adelante:

- El margen de ±0,5 puntos del verificador deja pasar trazas con la aritmética imprecisa (en el
  problema 9 se aceptan 77,00, 77,05 y 77,13 para un valor correcto de 77,02).
- Errata conocida del generador: en la plantilla de lista, los materiales de origen desconocido
  salen como «origen origen desconocido». No se corrige porque cambiaría el texto de `train` e
  invalidaría las trazas; el profesor lo resuelve igual (problema 5, 4 de 4).

Con la aceptación del piloto completo (65 %), 500 problemas dan unas 1.300 trazas válidas, dentro
del rango de 1.000 a 2.000 que recomienda `docs/datasets.md`.

### 2026-10-12 · Fase 1 · Tirada E2 (interrumpida en 256 de 500): el profesor no sabe aplicar el cambio de clasificación

**Qué queríamos saber.** La tasa de aceptación sobre una muestra grande y cómo se reparte por
regla, criterio y trampa.

**Qué hicimos.** Los 500 primeros problemas de `train` con el enunciado E2, continuando desde el
piloto E2 con el mecanismo de reanudación de `distill.py`:

```
cp rlm/data/piloto_trazas.jsonl rlm/data/sft_traces.partial.jsonl
uv run python -m rlm.generate_problems --n 500 --split train --out rlm/data/sft_problemas.jsonl
nohup uv run python -m rlm.distill --data rlm/data/sft_problemas.jsonl --output rlm/data/sft_traces.jsonl > runs/distill.log 2>&1 &
```

Mismos parámetros que el piloto (2.048 tokens, lotes de 2). Lanzada el 12 de octubre hacia la
01:15; la sesión de 8 horas se cerró a las 07:35 con 256 problemas hechos. Ritmo: unos 1,6
minutos por problema. La paramos ahí para revisar los resultados antes de seguir. Trazas en
`E2_trazas_train000-255.jsonl`.

**Qué pasó.** 650 trazas verificadas de 1.024 (63,5 %). Rechazos: 279 `wrong_verdict`, 53 cortadas
sin cerrar el razonamiento, 26 `wrong_pct`, 9 `wrong_criterion`, 7 sin respuesta. 68 de los 256
problemas (27 %) no tienen ninguna traza válida. Longitud mediana de 968 tokens; un 6 % llega al
límite.

Por criterio que hace originario al producto:

| Productos originarios en los que vale… | Trazas válidas |
|---|---|
| Algún criterio de valor (MaxNOM o RVC) | 349 de 396 (88 %) |
| Solo un criterio de cambio de clasificación (CC, CTH o CTSH) | 14 de 224 (6 %) |

- En las trazas válidas de productos originarios, el profesor cita MaxNOM 268 veces, RVC 75, y
  CTH, CTSH o CC solo 20 veces en total.
- De los 210 `wrong_verdict` en productos originarios, 192 son casos que se resolvían por cambio
  de clasificación: el profesor comprueba el porcentaje, ve que no cumple y no prueba la otra
  alternativa.
- Casos rescatados por la tolerancia: 4 de 92. Solo operaciones insuficientes: 37 de 108.
- Por regla, la peor es `TCA-8414` (solo CTH): 35 %.

**Qué concluimos.** Un SFT con estas trazas enseñaría al alumno a resolverlo todo por MaxNOM: casi
no hay ejemplos de cambio de clasificación ni de tolerancia. Y en GRPO esos problemas darían
recompensa muy pocas veces, así que tampoco se corregiría fácilmente ahí. Dos lecturas posibles:
que el profesor de 4B no sabe aplicar el criterio, o que el enunciado no le explica lo bastante
qué significa. Lo segundo se puede probar barato, así que paramos la tirada y probamos a
explicar los criterios en el enunciado (siguiente entrada).

### 2026-10-12 · Fase 1 · Enunciados E3 y E4: definir los criterios ayuda, pero el profesor aplica criterios que la regla no tiene

**Qué queríamos saber.** Si explicando en el enunciado qué significa cada criterio el profesor
resuelve los casos de cambio de clasificación sin empeorar el resto.

**Qué hicimos.** Añadir a las reglas generales (E3):

> - Basta con que se cumpla una de las alternativas de la regla (separadas por «o»).
> - Cambio de clasificación (CC, CTH, CTSH): se cumple si ningún material no originario tiene el
>   mismo capítulo (2 primeras cifras del código SA), la misma partida (4 cifras) o la misma
>   subpartida (6 cifras) que el producto, respectivamente, ni pertenece a una partida excluida
>   por la regla. Los materiales originarios no cuentan para este criterio.
> - MaxNOM: se cumple si el valor total de los materiales no originarios no supera el porcentaje
>   indicado del precio que señala la regla.

Para medirlo sin esperar a una tirada entera, escribimos `rlm/pilot_subset.py` (hoy
`rlm/pilot_subset_deprecated.py`), que elige 20
problemas de los 256 de la tirada E2: los 10 primeros que solo son originarios por cambio de
clasificación y los 10 primeros del resto. Así cada resultado se compara con el de esos mismos
problemas en E2. Piloto con 2.048 tokens; lo cortamos con 18 problemas hechos. La tabla recoge los
16 que analizamos en ese momento; el fichero tiene 18 (33 trazas válidas de 72).

**Qué pasó (E3).**

| Grupo | E2 (mismos problemas) | E3 |
|---|---|---|
| Solo cambio de clasificación | 0 de 24 | 7 de 24 |
| Resto | 36 de 40 | 23 de 40 |

En clasificación el razonamiento ya sale bien: de los fallos, 14 son trazas cortadas y solo 3
errores reales. En el resto, la caída es casi toda por cortes: el profesor ahora comprueba cada
material contra el código del producto y da varias vueltas antes de cerrar. 30 de 64 trazas
llegaron al límite de 2.048 tokens; la mediana de las válidas pasa de 968 a 1.312. Además, en el
problema 0 (regla solo CTSH) dos muestras aplican MaxNOM porque lo ven definido.

**Qué hicimos después (E4).** Añadir «Solo cuentan los criterios que aparecen en la regla de la
partida» y subir el límite a 4.096 tokens. Con lotes de 2 problemas (8 secuencias) se quedó sin
memoria en el problema 12 (15,8 GB ocupados de 16).

**Qué pasó (E4).** En los 12 problemas hechos ya no hay cortes, pero aparece un error nuevo: el
profesor da por originario el producto con criterios que la regla no tiene (CTSH en una regla
que solo admite MaxNOM, CTSH en una que solo admite CTH). Su razonamiento lo dice:

> the rule says that the product is originario if any of the criteria (CTH, CTSH, CC, MAXNOM,
> RVC, WO) are met

Junta «basta con una alternativa» con la lista de los seis criterios que aparecía en la pregunta.
La línea «solo cuentan los criterios de la regla» no basta.

**Qué concluimos.** Explicar los criterios funciona, pero explicarlos todos en abstracto confunde
al profesor. Hay que explicar solo los de cada regla, y que la pregunta liste solo esos.

### 2026-10-12 · Fase 1 · Enunciado E5: instrucciones propias de cada regla

**Qué queríamos saber.** Si con instrucciones específicas de la regla desaparecen los criterios
inventados y se mantiene la mejora en cambio de clasificación.

**Qué hicimos.** Sustituir las definiciones genéricas por una sección debajo de la regla, generada
por `criteria_notes(rule)` a partir de los criterios que la regla admite. Ejemplo:

```
Regla de origen de la partida 8507: cambio de partida (CTH); o valor máximo de materiales no originarios (MaxNOM) del 50 % del precio franco fábrica (EXW).
Cómo se aplica esta regla:
- Esta regla solo admite los criterios CTH o MAXNOM. Basta con que se cumpla uno de ellos; ningún otro criterio cuenta.
- CTH (cambio de partida): se cumple si ningún material no originario está clasificado en la misma partida (4 primeras cifras del código SA) que el producto. Los materiales originarios no cuentan.
- MAXNOM: se cumple si el valor total de los materiales no originarios no supera el porcentaje del precio que fija la regla.
```

La pregunta pasa a decir «uno de los criterios de esta regla que se cumplan: CTH o MAXNOM» en vez
de los seis. Mismos 20 problemas dirigidos, 4.096 tokens, lotes de 1 problema
(`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`). Tardó alrededor de una hora (medido a ojo).

**Qué pasó.**

| Grupo | E2 (mismos problemas) | E5 |
|---|---|---|
| Solo cambio de clasificación | 1 de 40 | 16 de 40 |
| Resto | 36 de 40 | 35 de 40 |
| Total | 37 de 80 | 51 de 80 |

- Ningún criterio inventado.
- Longitud: mediana de 1.187 tokens en las válidas, máximo 2.175. Solo 2 de 80 pasan de 2.048.
- El problema 3 (batería, CTH se cumple y MaxNOM no), que fallaba siempre desde el piloto E1,
  sale 4 de 4.
- Los 4 problemas que siguen en 0 de 4 (12, 26, 31 y 37, todos `TCA-8414`) son originarios
  gracias a la tolerancia: 0 de 16 en esos casos. El profesor compara con el 10 % la suma de
  todos los materiales no originarios, cuando solo cuentan los que impiden el cambio de partida.
  En el problema 31 suma 106,88 € (56,25 %) cuando el único material que impide CTH es el rodete,
  10,28 € (5,4 %).

**Qué concluimos.** E5 corrige el sesgo hacia MaxNOM sin empeorar el resto. Queda la tolerancia,
que es un fallo de lectura del mismo tipo que los anteriores.

### 2026-10-12 · Fase 1 · Enunciado E6: la tolerancia dice qué suma cuenta

**Qué queríamos saber.** Si dejando explícito qué materiales entran en la tolerancia el profesor
resuelve esos casos.

**Qué hicimos.** Reescribir la línea de la tolerancia:

> Tolerancia: si los materiales no originarios que impiden cumplir un criterio de cambio de
> clasificación de esta regla suman como máximo el 10 % del precio EXW, ese criterio se cumple
> igualmente. En esa suma solo entran los materiales que impiden cumplirlo, no todos los no
> originarios. La tolerancia no amplía los umbrales de valor (MaxNOM, RVC).

`pilot_subset.py` admite ahora `--trap` para elegir los difíciles por trampa
(`--trap tolerancia_usada`). Decidimos no pilotar E6 y lanzar directamente la tirada: el cambio
es acotado, y el desglose por trampa de la tirada dirá si ha funcionado.

**Qué pasó.** En las cuatro versiones del enunciado (E3 a E6), los parámetros, la respuesta y las
ramas de los 4.000 problemas de `train` son idénticos; solo cambia el texto. Ninguna fuga de la
respuesta. Las huellas de referencia del principio son las de E6.

**Qué concluimos.** Sobre si es legítimo ampliar así el enunciado:

- Es información del dominio, no la solución: los anexos reales de reglas de origen llevan notas
  introductorias que definen los criterios. El modelo sigue teniendo que clasificar cada
  material, comparar códigos, aplicar exclusiones y tolerancia, sumar y calcular el porcentaje.
- Train, test y `ood` salen de la misma función `render`, así que todos los splits tienen el
  mismo formato.
- Coste: la tarea es más fácil que con E1, y las tasas de aceptación de versiones distintas no
  son comparables entre sí. El modelo aprende a resolver enunciados con este formato; si en la
  corrección llega un caso redactado de otra manera, puede rendir peor.
- En el uso real (agente), las instrucciones dependen solo de la regla, así que el agente puede
  montar el enunciado con las mismas funciones del generador.

Consecuencias para los pasos siguientes, por la longitud de las trazas (alrededor de 1.200 tokens
más un enunciado de unos 1.000):

- `train_sft.py` tiene `--max-length 2048` por defecto, contando enunciado y traza: habrá que
  subirlo para no cortar el final de las trazas, que es donde va la respuesta.
- `train_grpo.py` tiene `--max-completion-length 768`: un alumno que razone 1.200 tokens se
  cortaría y nunca cobraría la recompensa de formato.

### 2026-10-12/13 · Fase 1 · Tirada E6: 1.282 trazas válidas de 2.000 en 500 problemas

**Qué queríamos saber.** La tasa de aceptación con el enunciado E6 y si las trazas válidas cubren
todas las reglas, criterios y trampas, en particular la tolerancia.

**Qué hicimos.**

```
uv run python -m rlm.generate_problems --n 500 --split train --out rlm/data/sft_problemas.jsonl
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup uv run python -m rlm.distill --data rlm/data/sft_problemas.jsonl --output rlm/data/sft_traces.jsonl --max-new-tokens 3072 --batch-size 2 > runs/distill.log 2>&1 &
```

Profesor `Qwen/Qwen3-4B`, 4 muestras por problema, temperatura 0,6. Límite de 3.072 tokens: con
E5 la traza más larga fue de 2.175, y bajar de 4.096 permite volver a lotes de 2 problemas sin
quedarse sin memoria. Lanzada el 12 de octubre pasadas las 16:00.

**Qué pasó (revisión a los 56 problemas, 19:05).** Comparado con la tirada E2 sobre los mismos 56
problemas:

| | E2 | E6 |
|---|---|---|
| Total | 148 de 224 (66 %) | 156 de 224 (70 %) |
| Originarios solo por cambio de clasificación | 1 de 44 | 18 de 44 |
| Originarios por criterio de valor | 73 de 84 | 74 de 84 |
| No originarios | 74 de 96 | 64 de 96 |
| Casos rescatados por la tolerancia | 0 de 16 | 0 de 16 |
| `wrong_verdict` | 61 | 34 |
| `wrong_pct` | 5 | 25 |
| Problemas sin ninguna traza válida | 13 | 8 |

- El cambio de clasificación mejora mucho: en las trazas válidas de productos originarios, el
  profesor cita CTH, CC o CTSH 22 veces (2 en E2).
- La tolerancia sigue en 0. El profesor sigue comparando con el 10 % la suma de todos los
  materiales no originarios. Problema 12: «287.46 € … 47.91%. That's way over the 10% tolerance».
- La frase nueva de la tolerancia tiene un efecto secundario: los `wrong_pct` pasan de 5 a 25,
  17 de ellos en `TCA-8414`. El profesor responde con el porcentaje de los materiales de la misma
  partida en lugar del de todos los no originarios (13,23 % en vez de 85,08 % en el problema 21).
  El verificador las rechaza, así que las trazas aceptadas siguen siendo correctas; lo que se
  pierde es cantidad en esa regla (8 de 36 válidas).
- Ritmo: unos 3 minutos por problema (56 en unas 3 horas). Los 500 necesitan más de una sesión de
  24 horas; se continúa relanzando el mismo comando.

**Qué concluimos (provisional).** Seguimos con E6. El sesgo grave, que el SFT no viera casos de
cambio de clasificación, está corregido. La tolerancia no la arreglamos con texto: dos
redacciones distintas (E5 y E6) dan 0 de 16, así que lo tratamos como un límite del profesor de
4B, no del enunciado. Volver a la redacción de E5 reduciría los `wrong_pct` pero obligaría a
empezar la tirada de cero, y la tolerancia seguiría en 0.

Consecuencia para el resto de la práctica: el SFT no tendrá ejemplos de tolerancia. Es el caso
natural para comparar SFT y GRPO: el profesor no sabe aplicarla, pero en GRPO el verificador
premia el acierto, así que podemos medir si el alumno la aprende por refuerzo.

**Qué pasó (tirada completa, 13 de octubre).** Dos sesiones de DGX: la primera se cortó hacia
las 23:45 del 12 y se relanzó en otra a las 23:48; terminó el 13 por la tarde. En total, del orden
de 24 horas de GPU (tiempos a ojo).

| Resultado | Trazas |
|---|---|
| Verificadas | 1.282 de 2.000 (64,1 %) |
| `wrong_verdict` | 317 |
| `wrong_pct` | 297 |
| Cortadas sin cerrar el razonamiento | 86 |
| `wrong_criterion` | 13 |
| Sin respuesta | 5 |

- 407 de los 500 problemas tienen al menos una traza válida; 209 tienen las 4. 93 no tienen
  ninguna.
- Longitud de las válidas: mediana de 1.158 tokens. 91 trazas llegan al límite de 3.072.

Comparación con E2 sobre los mismos 256 primeros problemas, y sobre los 500 de E6:

| Grupo | E2 (0-255) | E6 (0-255) | E6 (500) |
|---|---|---|---|
| Total | 63,5 % | 64,6 % | 64,1 % |
| Originarios solo por cambio de clasificación | 6,2 % | 40,6 % | 39,0 % |
| Originarios por criterio de valor | 88,1 % | 89,6 % | 86,2 % |
| No originarios | 71,0 % | 53,5 % | 54,3 % |
| Casos rescatados por la tolerancia | 4,3 % | 10,9 % | 12,8 % |
| Solo operaciones insuficientes | 34,3 % | 21,3 % | 19,7 % |
| Umbral exacto | 87,5 % | 86,7 % | 84,0 % |
| La tolerancia no amplía el MaxNOM | 68,8 % | 74,3 % | 71,9 % |

| Regla | E2 (0-255) | E6 (0-255) | E6 (500) |
|---|---|---|---|
| `CETA-8413` (CTSH) | 56,8 % | 41,7 % | 40,8 % |
| `TCA-8414` (CTH) | 34,6 % | 36,7 % | 35,0 % |
| `TCA-8507` (CTH o MaxNOM) | 74,5 % | 83,5 % | 77,8 % |
| `UE-JP-8501` (CTH o RVC) | 67,4 % | 70,1 % | 64,7 % |
| `UE-KR-8479` (MaxNOM) | 81,0 % | 82,1 % | 86,5 % |
| `UE-KR-9403` (CC o MaxNOM) | 65,7 % | 67,1 % | 76,2 % |

**Qué concluimos.** No repetimos la tirada; entra al SFT tal como está.

- La tasa total apenas cambia respecto a E2, pero la composición sí, y es lo que importa para el
  SFT: los casos que solo se resuelven por cambio de clasificación pasan del 6 % al 40 %. Con E2
  el alumno habría aprendido a resolverlo todo por MaxNOM; con E6 hay 181 trazas válidas de ese
  tipo.
- 1.282 trazas válidas está dentro del rango de 1.000 a 2.000 que recomienda
  `docs/datasets.md`, con todas las reglas y todas las trampas representadas. La más escasa es
  la tolerancia, con 25 trazas.
- El precio está en los productos no originarios (del 71 % al 54 %) y en los `wrong_pct` (297,
  frente a 26 en los 256 problemas de E2). Comprobado sobre la tirada completa: en 162 de los
  297 `wrong_pct`, el porcentaje que da el profesor coincide con el de los materiales no
  originarios que impiden el cambio de clasificación (los de su capítulo, partida o subpartida),
  no con el de todos. Es la confusión que introdujo la frase de la tolerancia de E6. Los
  `wrong_pct` se concentran en `TCA-8414` (151), `CETA-8413` (65) y `UE-JP-8501` (52), las tres
  reglas con un criterio de clasificación, lo que explica que `CETA-8413` baje del 57 % al 41 %.
- En los no originarios de los problemas 0 a 255, las trazas válidas bajan de 287 a 216. Unos
  dos tercios de esa caída son `wrong_pct` (de 16 a 72) y el resto, `wrong_verdict` (de 69 a 99):
  el profesor da por originarios productos que no lo son, en 57 de los 99 casos con un material
  de la misma partida. Es el lado opuesto del sesgo de E2: ahora aplica el cambio de clasificación
  o la tolerancia también cuando no debe.
- Todas esas trazas las rechaza el verificador, así que no contaminan el conjunto; solo reducen
  su tamaño.
- Lección para el diseño: cada frase nueva del enunciado arregló el fallo para el que se escribió
  y pudo abrir otro. Medir sobre los mismos problemas antes y después (los 20 dirigidos) es lo
  que permitió verlo; E6 fue la única versión que no pilotamos, y es la que tiene el efecto
  secundario más claro.

Repetir con la redacción de la tolerancia de E5 recuperaría parte de los no originarios, pero
costaría otras 24 horas de GPU y la tolerancia seguiría prácticamente sin resolver. No compensa.

### 2026-10-16 · Fase 1 · Longitud de las secuencias del SFT: 4.096 tokens bastan

**Qué queríamos saber.** Qué `--max-length` necesita el SFT. El valor por defecto de
`train_sft.py` (2.048) cuenta enunciado y traza juntos, y la respuesta va al final: una secuencia
truncada enseñaría al alumno a razonar sin llegar nunca a responder.

**Qué hicimos.** Pasar las 1.282 trazas válidas por la plantilla de chat de `Qwen/Qwen3-1.7B` tal
como las ve el entrenamiento (prompt de sistema, enunciado y traza como respuesta del asistente) y
contar tokens. De paso, comprobar que la plantilla conserva el `<think>`: la de Qwen3 puede
manipular el razonamiento de los mensajes del asistente.

**Qué pasó.**

| Enunciado + traza | Secuencias |
|---|---|
| Más de 3.072 tokens | 102 de 1.282 |
| Más de 4.096 tokens | 0 |
| Mediana / máximo | 1.983 / 4.021 |

El `<think>` y el `</answer>` aparecen en el texto renderizado.

**Qué concluimos.** `--max-length 4096`: no se trunca ninguna traza y no hace falta descartar
ninguna. Para el GRPO, `--max-completion-length` tendrá que rondar los 3.000 tokens (el profesor
llega a escribir más de 3.000 en 102 casos), muy por encima de los 768 por defecto; con varias
muestras por problema, eso pesará en memoria.

### 2026-10-16 · Fase 1 · Configuración del SFT y prueba corta: cabe en 16 GB

**Qué queríamos saber.** Con qué datos y parámetros entrenar, cómo detectar el sobreajuste, y si
el entrenamiento cabe en la partición de 16 GB y cuánto tarda.

**Qué hicimos.** Cambios en `train_sft.py`:

- Datos: todas las trazas válidas, hasta 4 por problema (1.282). La alternativa, un máximo de 2
  por problema (unas 750), equilibraría problemas fáciles y difíciles con menos datos: 209
  problemas tienen sus 4 trazas válidas, así que los fáciles pesan hasta cuatro veces más. Queda
  como ablación.
- Validación por problema, no por traza. Las 4 trazas de un problema se parecen mucho; si se
  repartieran entre entrenamiento y validación, `eval_loss` saldría mejor de lo que es. El lado de
  cada problema lo decide una huella SHA-256 del enunciado (10 % a validación), así que el reparto
  es el mismo en cada ejecución y no depende del orden del fichero. Quedan 1.197 trazas de 377
  problemas para entrenar y 85 trazas de 30 problemas para validar. Son el 7 % de los 407
  problemas con alguna traza válida, no el 10 %: con una huella el reparto es aleatorio y no sale
  exacto.
- `eval_loss` cada 20 pasos y una vez antes de entrenar (`eval_on_start`), para tener el punto de
  partida de la curva.
- `log_history.json` en la carpeta de salida, con todas las pérdidas. Las curvas se dibujan con
  `python -m rlm.evaluate --history <ruta>`.

| Parámetro | Valor | Por qué |
|---|---|---|
| Modelo base | `Qwen/Qwen3-1.7B` | El de la propuesta; misma familia que el profesor |
| LoRA | rango 16, alpha 32, dropout 0,05, todas las capas lineales | Los valores de la plantilla |
| Épocas | 2 | Con unas 1.200 trazas, más épocas tienden a memorizar; lo dirá la curva de validación |
| Learning rate | 2·10⁻⁴, calentamiento del 5 % y bajada en coseno | Más estable con pocos pasos (150) |
| Lote | 1, con 16 de acumulación (16 trazas por paso) | Secuencias de hasta 4.021 tokens en 16 GB |
| `max_length` | 4.096 | Entrada anterior |
| `packing` | no | Con secuencias tan largas hay poco que empaquetar, y mezclaría trazas distintas en una muestra |
| `assistant_only_loss` | no | El formato prompt/completion ya calcula la pérdida solo sobre la respuesta |

Con transformers 5.17.0, `SFTConfig` ya no acepta `warmup_ratio`: el 5 % se pasa como
`warmup_steps=0.05`, que se interpreta como fracción del entrenamiento cuando es menor que 1.

Prueba de 0,1 épocas en la partición `MIG 1g.18gb`:

```
uv run --extra train python -m rlm.train_sft --data rlm/data/sft_traces.jsonl \
    --epochs 0.1 --output rlm/weights/sft_prueba
```

**Qué pasó.**

| Validación | Antes de entrenar | Tras 8 pasos (0,107 épocas) |
|---|---|---|
| `eval_loss` | 0,574 | 0,364 |
| `eval_mean_token_accuracy` | 0,861 | 0,876 |
| `eval_entropy` | 0,183 | 0,412 |

- Pérdida media de entrenamiento: 0,441. Sin errores de memoria.
- 352 s para los 8 pasos (44 s por paso contando la evaluación final). La evaluación de las 85
  trazas tarda unos 23 s, 57 s la primera vez.
- 75 pasos por época, 150 en total.

**Qué concluimos.**

- Antes de entrenar, el alumno ya predice el 86 % de los tokens de las trazas del profesor. Lo
  atribuimos a que alumno y profesor son de la misma familia (mismo tokenizador y estilo de
  razonamiento parecido), lo que apoya haber elegido un profesor Qwen3.
- `eval_loss` baja mucho en solo 8 pasos: el formato y el estilo se aprenden pronto.
- La entropía de validación sube de 0,18 a 0,41. No sabemos explicarlo con 8 pasos; lo miraremos
  en la curva completa.
- Cabe en 16 GB. Estimación para el SFT completo: alrededor de 1 h 30 min.

### 2026-10-16 · Fase 1 · `evaluate.py`: el mismo prompt que en el entrenamiento y dos métricas de acierto

**Qué queríamos saber.** Cómo medir el pass@1 del modelo base, del SFT y del GRPO de forma que la
comparación sea justa.

**Qué hicimos.** Implementar `evaluate_model` en `rlm/evaluate.py`:

- El prompt se construye en un solo sitio (`render_prompt`): `build_prompt` más la plantilla de
  chat con `add_generation_prompt=True` y `enable_thinking=True`, igual que en `distill.py` y en
  el SFT. Con `enable_thinking=False`, la plantilla de Qwen3 añade un `<think></think>` vacío y el
  modelo se salta el razonamiento. No da ningún error: solo saldrían peores resultados.
- Base y SFT son idénticos salvo el adaptador: el base es `Qwen3-1.7B` tal cual y el SFT, el mismo
  modelo con el LoRA fusionado (`merge_and_unload`).
- Generación sin azar por defecto, para que el pass@1 sea reproducible. `--temperature` queda
  como alternativa si el modelo entra en bucles.
- Dos métricas de acierto. `pass@1` exige la respuesta dentro de `<answer>`, como `/reasoning`.
  `pass@1_lenient` también acepta el último JSON escrito después de `</think>`. La diferencia
  entre las dos separa los fallos de formato de los de razonamiento. Además se miden el formato
  válido, las respuestas cortadas y los tokens medios.
- Desglose por veredicto esperado y por cada campo de `branches` (regla, trampas, redacción).
- Los resultados se acumulan en `reports/phase1_eval_<datos>.json`, que se guarda después de cada
  modelo: se puede evaluar el base un día y el SFT otro, y la tabla final muestra los dos.
- Generación por lotes (8 problemas a la vez) en lugar de reutilizar `ReasoningModel`, que va de
  uno en uno.

**Qué pasó.** Los ficheros de problemas no estaban en la DGX, porque no se suben al repositorio.
Regenerados `test.jsonl`, `test_ood.jsonl` y `train.jsonl` con la semilla 0: sus sha256 coinciden
con la tabla de referencia. En la primera prueba, el desglose contaba dos veces la etiqueta
`originario` (salía de la respuesta esperada y de `branches`); corregido para que cada etiqueta
cuente una vez por problema.

**Qué concluimos.** Todos los modelos se evalúan con los mismos problemas, el mismo prompt y el
mismo verificador. Queda comprobar que `rlm/inference.py` construye el prompt igual: si no,
`/reasoning` respondería distinto de lo que mide la evaluación.

### 2026-10-16 · Fase 1 · Modelo base en 4 problemas de `test`: falla sobre todo el formato, y la generación va a 430 ms por token

**Qué queríamos saber.** Si `evaluate.py` funciona de punta a punta y cómo se comporta el modelo
base antes de evaluar los 200 problemas.

**Qué hicimos.**

```
uv run --extra train python -m rlm.evaluate --data rlm/data/test.jsonl \
    --adapters base=none --n-examples 4 --out reports/prueba_eval.json
```

Partición `MIG 1g.18gb`, sin azar, 3.072 tokens como máximo, un lote de 4 problemas.

**Qué pasó.**

| pass@1 | lenient | Formato válido | Cortadas | Tokens medios |
|---|---|---|---|---|
| 0,25 | 0,25 | 0,25 | 0,25 | 1.580 |

| Problema | Tokens | Resultado | Qué hace |
|---|---|---|---|
| 0 | 801 | Falla | Razona y concluye (CTSH, 0,00 %), pero tras `</think>` escribe `<tool_call> reasoning process here` en lugar de `<answer>` |
| 1 | 1.514 | Acierta | Umbral exacto: MaxNOM al 50,00 % justo, con el formato correcto |
| 2 | 933 | Falla | Concluye originario por CTH y RVC con un 31,48 %, y vuelve a escribir `<tool_call>` |
| 3 | 3.072 | Falla (cortada) | Se cumplen CTH y RVC; cree que tiene que poner los dos en `criterio`, ve que el JSON no lo permite y da vueltas hasta el límite |

Velocidad: 430 ms por token, constante de 256 a 3.072 tokens. El lote tardó 1.323 s (22 minutos)
porque espera a la respuesta más larga. El modelo está entero en la GPU (11,8 de 16 GB libres tras
cargarlo) y no hay otros procesos. A este ritmo, los 200 problemas en lotes de 8 serían unas 9
horas por modelo. En la misma partición, el distill de E6 hacía unos 3 minutos por problema con un
modelo mayor (4B) y 8 secuencias, del orden de 0,1 a 0,2 s por token. No sabemos explicar la
diferencia.

**Qué concluimos.**

- `evaluate.py` funciona de punta a punta.
- Con 4 problemas el pass@1 no dice nada, pero aparecen dos fallos que no son del dominio: el
  `<tool_call>` y el bloqueo cuando se cumplen dos criterios. Nuestra hipótesis para el
  `<tool_call>`: el prompt de sistema de R1-Zero trae una plantilla con marcas y huecos (`<think>
  reasoning process here </think> <answer> answer here </answer>`), y Qwen3, entrenado para llamar
  a herramientas con `<tool_call>`, la rellena a su manera. No cambiamos el prompt: las trazas del
  SFT se generaron con él y el base tiene que evaluarse igual que el SFT. Probar otro prompt queda
  como experimento aparte.
- Hipótesis para los 200 problemas: buena parte de los fallos del base son de formato, no de
  razonamiento. Si se confirma, el SFT aportaría sobre todo formato. Lo dirá la comparación por
  motivo de fallo.
- La velocidad obliga a evaluar con lotes más grandes o en una instancia mayor (perfil
  Entrenamiento, 71 GB). Antes hay que medir cómo cambia el tiempo por token con el tamaño del
  lote.

### 2026-10-24 · Fase 1 · SFT completo: sin sobreajuste, `eval_loss` de 0,574 a 0,228

**Qué queríamos saber.** Cómo queda el alumno tras dos épocas de SFT y si la curva de validación
muestra sobreajuste.

**Qué hicimos.**

```
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup uv run --extra train \
    python -m rlm.train_sft --data rlm/data/sft_traces.jsonl \
    --output rlm/weights/sft_lora > runs/sft.log 2>&1 &
```

Configuración de la entrada del 16 de octubre. Lanzado el 24 a las 21:11 en la partición
`MIG 1g.18gb` (sesión de estudiante, 4 núcleos).

`source smoke/dgx_env.sh` avisa de que no existe `~/clusters/dgx`. El script solo decide dónde
van las cachés de uv y de Hugging Face y la configuración global de git; sin esa carpeta todo queda
en su sitio por defecto, así que no afecta al entrenamiento.

**Qué pasó.** Terminó sin errores en 3.551 s (59 minutos, unos 24 s por paso con las
evaluaciones incluidas, más rápido de lo estimado). Pérdida media de entrenamiento: 0,262. Curvas
en `rlm/weights/sft_lora/log_history.png`.

| Paso | Época | `eval_loss` | `loss` (entrenamiento) |
|---|---|---|---|
| 0 | 0 | 0,574 | — |
| 20 | 0,27 | 0,300 | 0,340 |
| 40 | 0,53 | 0,263 | 0,270 |
| 60 | 0,80 | 0,246 | 0,248 |
| 80 | 1,07 | 0,237 | 0,234 |
| 100 | 1,33 | 0,232 | 0,228 |
| 120 | 1,60 | 0,229 | 0,224 |
| 140 | 1,87 | 0,228 | 0,226 |
| 150 | 2 | 0,228 | 0,222 |

La `loss` de entrenamiento se registra cada 10 pasos; la del paso 10 fue 0,473.

| Validación | Antes de entrenar | Tras 8 pasos | Al final |
|---|---|---|---|
| `eval_mean_token_accuracy` | 0,861 | 0,876 | 0,915 |
| `eval_entropy` | 0,183 | 0,412 | 0,222 |

**Qué concluimos.**

- No hay sobreajuste. `eval_loss` baja todo el rato y acaba pegada a la de entrenamiento (0,228
  frente a 0,222). Si el modelo memorizara, la de validación subiría mientras la otra sigue
  bajando.
- Casi el 90 % de la bajada de `eval_loss` ocurre en los primeros 40 pasos (media época). Lo
  interpretamos como el aprendizaje del formato y del estilo del profesor; después solo afina.
- La segunda época aporta poco: `eval_loss` pasa de 0,237 al empezarla a 0,228 al acabarla, y está
  plana desde el paso 120. Más épocas no compensarían, y menos tampoco hacen falta, porque no hay
  sobreajuste que evitar.
- La subida de entropía de la prueba corta fue pasajera (0,18 → 0,41 → 0,22): al principio el
  modelo está menos seguro mientras cambia de estilo y luego se estabiliza.
- Elegimos el adaptador final (`rlm/weights/sft_lora`, igual que `checkpoint-150`), el de
  `eval_loss` más baja.
- Límite de esta medida: `eval_loss` dice si el modelo imita bien las trazas del profesor, no si
  acierta. Lo que dice si el SFT ha servido es el pass@1 en `test`, en la siguiente entrada.

### 2026-10-26 · Fase 1 · Evaluación del base y del SFT: el SFT duplica el pass@1 (28 % → 56 %)

**Qué queríamos saber.** Si el SFT ha servido: cuánto acierta frente al modelo base en problemas
que no ha visto, qué tipo de fallos arregla, dónde sigue fallando y si generaliza a reglas que no
estaban en el entrenamiento.

**Qué hicimos.** Evaluar los dos modelos en `test` (200 problemas) y `test_ood` (100: reglas de
Mercosur, la mitad con la redacción sin siglas), con el mismo prompt, sin azar y un máximo de
3.072 tokens. El SFT es el adaptador final de la entrada anterior. Partición `MIG 3g.71gb`
(perfil Entrenamiento); la velocidad y la memoria van en la entrada siguiente.

```
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup bash -c "
uv run --extra train python -m rlm.evaluate --data rlm/data/test.jsonl \
    --adapters base=none sft=rlm/weights/sft_lora --batch-size 50 &&
uv run --extra train python -m rlm.evaluate --data rlm/data/test_ood.jsonl \
    --adapters base=none sft=rlm/weights/sft_lora --batch-size 50
" > runs/eval.log 2>&1 &
uv run python -m rlm.analyze_eval reports/phase1_eval_test.json
uv run python -m rlm.analyze_eval reports/phase1_eval_test_ood.json --no-teacher
```

**Qué pasó.**

| | Base, `test` | SFT, `test` | Base, `test_ood` | SFT, `test_ood` |
|---|---|---|---|---|
| pass@1 | 28,0 % | 56,0 % | 33,0 % | 58,0 % |
| Intervalo al 95 % | 22,2–34,6 % | 49,1–62,7 % | 24,6–42,7 % | 48,2–67,2 % |
| pass@1 lenient | 29,0 % | 56,5 % | 38,0 % | 58,0 % |
| Formato válido | 19,0 % | 83,5 % | 21,0 % | 80,0 % |
| Cortadas a 3.072 tokens | 10,5 % | 16,5 % | 12,0 % | 20,0 % |

Comparación problema a problema: en `test`, el SFT arregla 69 problemas que el base fallaba y
estropea 13 que el base acertaba (43 los aciertan los dos y 75 los fallan los dos); test de
McNemar exacto, p = 2,3·10⁻¹⁰. En `test_ood`, 38 arreglados frente a 13 estropeados, p = 6,2·10⁻⁴.

Tipos de fallo:

| Tipo | Base, `test` | SFT, `test` | Base, `test_ood` | SFT, `test_ood` |
|---|---|---|---|---|
| `<tool_call>` en lugar de `<answer>` | 38 | 0 | 14 | 0 |
| Cierra el razonamiento sin `<answer>` | 7 | 0 | 11 | 0 |
| Cortada sin responder | 20 | 33 | 10 | 20 |
| Veredicto incorrecto | 39 | 32 | 14 | 15 |
| Porcentaje incorrecto | 29 | 22 | 13 | 7 |
| Criterio incorrecto | 11 | 1 | 5 | 0 |

pass@1 por grupo en `test`, con la tasa de aceptación del profesor en la destilación (orientativa:
otros problemas, otro modelo y generación con temperatura):

| Grupo | n | Base | SFT | Profesor |
|---|---|---|---|---|
| Todos | 200 | 28,0 % | 56,0 % | 64,1 % |
| Originarios por un criterio de valor | 74 | 28,4 % | 74,3 % | 86,2 % |
| Originarios solo por cambio de clasificación | 61 | 18,0 % | 26,2 % | 39,0 % |
| No originarios | 65 | 36,9 % | 63,1 % | 54,3 % |
| Criterio aplicado MaxNOM | 27 | 40,7 % | 100,0 % | 97,5 % |
| Criterio aplicado RVC | 13 | 0,0 % | 38,5 % | 72,6 % |
| Criterio aplicado CTSH (`CETA-8413`) | 28 | 25,0 % | 14,3 % | 27,8 % |
| Tolerancia usada | 30 | 10,0 % | 6,7 % | 12,8 % |
| Operación insuficiente | 17 | 23,5 % | 29,4 % | 19,7 % |
| Proveedor sin declaración | 50 | 16,0 % | 40,0 % | 56,7 % |
| Umbral exacto | 23 | 30,4 % | 73,9 % | 84,0 % |

En `test_ood`: `MERCOSUR-9401` (CC como único criterio) pasa de 8,3 % a 50,0 % (n = 24); con
siglas, de 32 % a 62 %, y con la redacción descriptiva, de 34 % a 54 % (n = 50 cada una); los no
originarios, de 33,3 % a 76,7 % (n = 30); la tolerancia queda en 8,3 % (n = 12) y el umbral exacto
baja de 36,4 % a 27,3 % (n = 11).

Longitud en `test`: el SFT acierta con razonamientos más cortos (mediana de 1.071 tokens, frente a
1.440 del base), y sus fallos son largos (mediana de 1.696). En casi todos los aciertos, el
razonamiento contiene el porcentaje y la suma esperados (entre el 96 y el 100 % según el modelo y
el conjunto; comprobación heurística de `analyze_eval.py`). Solo un acierto, del base en
`test_ood`, no muestra ninguno de los dos: no vemos aciertos por casualidad.

Fallos del SFT que se ven en la respuesta (las cifras son las del propio modelo):

- Problema 12 de `test` (`TCA-8414`, tolerancia usada): responde CTH, que es correcto, pero con
  un 9,75 % en lugar del 67,63 %. Lo mismo en los problemas 14, 22 y 25 de `test_ood`, los tres
  con tolerancia: da el porcentaje de los materiales que impiden el cambio de clasificación, no el
  de todos los no originarios.
- Problema 8 de `test` (`CETA-8413`, tolerancia usada): el porcentaje es correcto (24,83 %), pero
  responde "no originario". No aplica la tolerancia.
- Problema 17 de `test` (operación insuficiente): responde originario por RVC con el porcentaje
  correcto. Pasa por alto que la operación no confiere origen.
- Problema 5 de `test` y problema 2 de `test_ood`: se cumplen dos criterios y el modelo da vueltas
  a cuál poner («the rule allows either, the answer can be either. However…») hasta el límite de
  tokens. Es el mismo bloqueo que tenía el base en el problema 3.

**Qué concluimos.**

- **El SFT funciona y no es azar.** Duplica el pass@1 en `test` (28 % → 56 %) y la mejora es
  muy significativa. Fuera de distribución sube lo mismo (33 % → 58 %), así que lo aprendido se
  traslada a reglas que no estaban en el entrenamiento. Con una salvedad: desde el enunciado E5,
  cada problema explica los criterios de su regla, así que `test_ood` es menos difícil de lo que
  era al diseñarlo. La diferencia entre redacciones (62 % frente a 54 %) no es concluyente con 50
  problemas por grupo.
- **El formato queda resuelto.** El formato válido pasa del 19 % al 83,5 %, los `<tool_call>`
  desaparecen y los criterios mal etiquetados bajan de 11 a 1: el alumno ha aprendido a poner
  `NINGUNO`. La hipótesis de la entrada del 16 se confirma en parte: 45 de los 144 fallos del base
  son de entrega de la respuesta, y no de razonamiento. Los de razonamiento (veredicto, criterio y
  porcentaje) son 79 en el base y 55 en el SFT: mejoran, pero bastante menos. El pass@1 lenient no
  sirvió para medir esa parte, porque cuando el base escribe `<tool_call>` tampoco deja el JSON
  después del razonamiento.
- **El alumno aprende lo que sabe el profesor y hereda lo que no sabe.** Iguala o supera al
  profesor en los criterios de valor (MaxNOM 100 %) y en los no originarios (63 % frente a 54 %).
  Se queda por debajo en lo que el profesor hacía peor: solo cambio de clasificación (26 % frente
  a 39 %), CTSH (14 %, por debajo incluso del base, aunque con n = 28) y la tolerancia (7 %, frente
  al 13 % del profesor), como esperábamos con solo 25 trazas de tolerancia. RVC (38 % frente a
  73 %) es la mayor distancia con el profesor.
- **Un error del profesor ha pasado al alumno.** En la tirada E6, 162 de los `wrong_pct` del
  profesor daban el porcentaje de los materiales que impiden el cambio de clasificación en lugar
  del de todos los no originarios. Esas trazas se rechazaron, pero el alumno comete el mismo
  error en los casos de tolerancia. Nuestra hipótesis es que lo aprende de las trazas aceptadas,
  en las que el profesor calcula esa suma parcial para comprobar la tolerancia, sin que en ellas
  llegue a confundirla con el porcentaje final.
- **El SFT se corta más que el base** (16,5 % frente a 10,5 % en `test`). Una parte son bloqueos
  ante dos criterios válidos; otra, vueltas de comprobación («But wait…») que el alumno ha
  copiado del profesor y que, sin azar, entran en bucle.
- **Para el GRPO**, los casos con margen son la tolerancia, el cambio de clasificación (sobre todo
  CTSH), RVC y las respuestas que se cortan. La recompensa de formato penaliza no llegar a
  `<answer>`, así que el GRPO debería empujar también contra los bucles, siempre que
  `--max-completion-length` deje llegar a la respuesta (3.072 tokens como mínimo).

### 2026-10-26 · Fase 1 · Velocidad de generación: el límite es el coste fijo de cada token, no el tamaño del lote

**Qué queríamos saber.** Por qué la generación iba a 430 ms por token y cómo evaluar 600
respuestas (2 modelos × 300 problemas) en un tiempo razonable.

**Qué hicimos.** Medir el tiempo por token de `Qwen3-1.7B` con lotes de 1, 4 y 16 secuencias
(128 tokens nuevos, sin azar) en la partición `MIG 1g.18gb`. Después, evaluar en `MIG 3g.71gb`
(perfil Entrenamiento) con lotes grandes.

**Qué pasó.**

| Partición | Lote | ms por token |
|---|---|---|
| `MIG 1g.18gb` | 1 | 304 |
| `MIG 1g.18gb` | 4 | 300 |
| `MIG 1g.18gb` | 16 | 297 |
| `MIG 3g.71gb` | 100 | unos 480 |
| `MIG 3g.71gb` | 50 | 470 en el primer lote; unos 70 en los siguientes |

- Con lotes de 100, la evaluación se quedó sin memoria al acercarse a los 3.072 tokens:
  40,5 GB ocupados y 28,3 GB reservados pero sin usar. Es fragmentación, porque la caché de
  atención crece en cada token. Con `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` y lotes de
  50 no volvió a pasar.
- La evaluación completa (600 respuestas) tardó unas 1 h 40 min: `test` terminó a las 11:21 y
  `test_ood` a las 12:01, tras arrancar a las 10:22.

**Qué concluimos.** El tiempo por token casi no cambia con el lote: lo que cuesta es el trabajo
fijo de cada paso (lanzar desde Python muchas operaciones pequeñas a la GPU), no el cálculo. Por
eso conviene generar con lotes tan grandes como quepan en memoria. No sabemos explicar por qué el
primer lote de 50 fue siete veces más lento que los siguientes. Sospechamos de otro proceso
compitiendo en el nodo, pero no lo hemos comprobado. Para el GRPO, que genera varias respuestas
por problema en cada paso, este coste fijo pesa mucho; vLLM evitaría buena parte, pero no está
instalado.

### 2026-10-28 · Fase 1 · Recompensas del GRPO: exactitud, formato y esquema del JSON

**Qué queríamos saber.** Qué debe premiar el GRPO además de acertar, y con qué pesos, sabiendo que
cada recompensa cambia lo que aprende el modelo.

**Qué hicimos.** Añadir a `rlm/rewards.py` dos recompensas y usarlas junto a `format_reward`:

| Recompensa | Qué comprueba | Peso |
|---|---|---|
| `format_reward` | `<think>…</think><answer>…</answer>` estricto | 0,2 |
| `origin_accuracy_reward` | `OriginVerifier`, binaria | 1,0 |
| `domain_reward` | JSON con exactamente las 3 claves; `NINGUNO` si y solo si no es originario; criterio dentro de los de la regla; porcentaje entre 0 y 100 | 0,3 |

La exactitud pesa más que las otras dos juntas: si no, una respuesta bien formada pero incorrecta
cobraría casi lo mismo que una correcta. `domain_reward` acepta el porcentaje también como texto
(`"40.03"`) porque así lo escribe el 63 % de las trazas válidas del profesor. Descartamos penalizar
la longitud: TRL normaliza las recompensas dentro de cada grupo, de modo que en un grupo con la
misma exactitud un término pequeño de longitud decidiría toda la ventaja. Las respuestas cortadas ya
sacan 0 en las tres.

**Qué pasó.** Aplicadas a las 600 respuestas guardadas en `reports/phase1_eval_*.json`, la exactitud
coincide con el pass@1 de `evaluate.py` en las 600. En el SFT, formato y dominio dan los dos 0,835:
todas las respuestas con `<answer>` cumplen ya el esquema.

**Qué concluimos.** Lo último es una advertencia: partiendo del SFT, `domain_reward` podría no
añadir ninguna señal. Lo comprobamos en la entrada del GRPO completo.

### 2026-10-29 · Fase 1 · Memoria y velocidad del GRPO: cuDNN era la causa de los 300 ms por token

**Qué queríamos saber.** Por qué la primera prueba del GRPO se quedaba sin memoria y por qué la
generación tardaba unos 20 minutos por paso. Esta entrada corrige la del 2026-10-26 sobre la
velocidad de generación.

**Qué hicimos.**

- Memoria: leer el error (13,92 GiB pedidos en `convert_to_fp32`). Son los logits de 8 × 3.072 tokens
  con 152.000 palabras de vocabulario en un solo lote. Separamos generar de entrenar:
  `--batch-size 1` con acumulación de gradientes, de forma que se genera a la vez pero se entrena
  respuesta a respuesta (`--prompts-per-step`).
- Velocidad: descartar causas una a una en `MIG 1g.18gb` con scripts de diagnóstico sueltos y
  repetir la medida con `torch.backends.cuda.enable_cudnn_sdp(False)`.

**Qué pasó.**

| Comprobación | Resultado |
|---|---|
| Hilos de CPU | 4 hilos para 4 núcleos: bien |
| Coste de lanzar una operación a la GPU | 3,8 µs |
| Multiplicación de matrices en la GPU | 104 TFLOP/s |
| Bucle de Python | 0,47 s del total medido |
| `cProfile` | `scaled_dot_product_attention` se lleva 1,126 s de 1,204 s |
| Generación con cuDNN | 288 ms por token |
| Generación sin cuDNN | **15 ms por token** |

La explicación que manejamos es que cuDNN prepara en la CPU un plan para cada longitud de secuencia
nueva y, al generar, la longitud cambia en cada token. Tras el cambio, una prueba de 3 pasos del GRPO
en 16 GiB tardó 327 s, unos 1,8 minutos por paso, y no se quedó sin memoria.

En el portátil (RTX 5070 Laptop de 8 GB) se genera a 33 ms por token, pero el paso de entrenamiento
falla con `device not ready` con 7.668 de 8.151 MiB ocupados. El GRPO no cabe ahí.

**Qué concluimos.** La conclusión de la entrada del 2026-10-26 era errónea: no era el coste de lanzar
operaciones desde Python, sino cuDNN, y vLLM no hacía falta para resolverlo. Nadie exige vLLM en el
enunciado. Los tiempos apuntados antes de este cambio (destilación y evaluaciones) están inflados.
`disable_cudnn_attention()` está en `rlm/__init__.py` y lo usan `train_grpo`, `evaluate`, `distill` e
`inference`.

### 2026-10-30 · Fase 1 · Entrenar el GRPO en varias sesiones de 8 horas

**Qué queríamos saber.** Cómo entrenar sin la partición `3g.71gb`, en la que llevábamos dos días en
cola, cuando las sesiones de `1g.18gb` duran 8 horas como máximo.

**Qué hicimos.** Añadir a `train_grpo.py` `--resume-from-checkpoint last` (busca el último checkpoint
en `--output`, o empieza de cero si no hay) y `--max-hours` (guarda un checkpoint y para limpiamente
antes de que acabe la sesión). Se lanza el mismo comando en cada sesión. Probamos 3 sesiones simuladas
en local.

**Qué pasó.** El learning rate continúa sin saltos (5e-6, 3,33e-6, 1,67e-6) y el historial acumula
todos los pasos. Cada checkpoint ocupa unos 211 MB. `train_runtime` solo cuenta la última sesión, así
que para el tiempo total hay que sumar `step_time`.

**Qué concluimos.** La tirada se puede repartir sin afectar a la planificación del learning rate.

### 2026-11-02 · Fase 1 · GRPO completo: mejora el formato y corta menos, la exactitud no muestra tendencia

**Qué queríamos saber.** Si el GRPO mejora al SFT optimizando directamente la recompensa verificable,
y si en 400 pasos deja de aprender o se queda corto.

**Qué hicimos.** Partir de `rlm/weights/sft_lora`, en `MIG 1g.18gb`:

```
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup uv run --extra train \
    python -m rlm.train_grpo --steps 400 --save-steps 20 \
    --resume-from-checkpoint last --max-hours 7.5 \
    --output rlm/weights/final_rlm_lora >> runs/grpo_run01.log 2>&1 &
```

Un problema por paso, 8 respuestas por problema, 3.072 tokens como máximo, learning rate 5e-6 con
bajada lineal, temperatura 1,0, sin KL (`beta` 0), recorte 0,2, pesos 0,2 / 1,0 / 0,3. Son 400
problemas y 3.200 respuestas, el 10 % de `train`.

**Qué pasó.** Tardó unas 9,8 horas en dos sesiones, 88 s por paso de media. Medias por bloques de 50
pasos, a partir de `log_history.json`:

| Pasos | Acierto | Formato | Cortadas | Longitud | Pasos sin señal |
|---|---|---|---|---|---|
| 1-50 | 0,62 | 0,94 | 0,06 | 1.445 | 0,38 |
| 51-100 | 0,73 | 0,95 | 0,04 | 1.494 | 0,38 |
| 101-150 | 0,69 | 0,97 | 0,03 | 1.421 | 0,34 |
| 151-200 | 0,63 | 0,95 | 0,05 | 1.522 | 0,38 |
| 201-250 | 0,72 | 0,96 | 0,04 | 1.555 | 0,30 |
| 251-300 | 0,69 | 0,96 | 0,04 | 1.551 | 0,34 |
| 301-350 | 0,77 | 0,99 | 0,01 | 1.438 | 0,38 |
| 351-400 | 0,64 | 0,97 | 0,03 | 1.463 | 0,48 |

- **Acierto:** sube 1,05 puntos por cada 100 pasos, sin distinguirse del ruido (permutaciones,
  p = 0,47). A ojo, el cambio real en los 400 pasos podría ir de unos -7 a +16 puntos: la curva no
  demuestra que aprendiera ni que no.
- **Formato y cortadas:** entre los primeros y los últimos 100 pasos, el formato pasa de 0,95 a 0,98 y
  las cortadas de 0,05 a 0,02 (p ≈ 0,02 en ambos, sin corregir por hacer varias pruebas).
- **Pasos sin señal:** en 149 de 400 (37 %) las 8 respuestas sacan la misma recompensa, 110 con todas
  bien y 39 con todas mal. La ventaja es cero y ese paso no enseña nada.
- **`domain_reward`:** coincide con la de formato en 397 de 400 pasos, así que no aporta señal propia.
- **Pesos:** el adaptador cambia un 0,76 % respecto al SFT (norma de la diferencia entre norma).
- **Acierto en entrenamiento:** sale más alto (0,69, muestreando) que el pass@1 en `test` (0,625,
  greedy). Esperábamos lo contrario. Puede influir que al muestrear se corten menos respuestas (3,6 %
  frente a 10,5 %), o que los problemas de `train` sean más fáciles. No lo hemos comprobado.

**Qué concluimos.** Con un problema por paso, la curva es demasiado ruidosa para decidir si faltaban
pasos. No repetiríamos la tirada igual. Si hubiera otra, probaríamos primero `--prompts-per-step 4`,
que debería reducir los pasos sin señal y el ruido, comparando en validación y no en `test`.
`domain_reward` se queda porque el enunciado pide una tercera recompensa, pero no podemos atribuirle
ninguna mejora.

### 2026-11-03 · Fase 1 · Evaluación del GRPO: mejora un poco, y el mismo modelo cambia de veredicto en el 26 % de los problemas

**Qué queríamos saber.** Si el GRPO supera al SFT en `test` y `test_ood`, y cuánta de esa diferencia
se debe a las condiciones de evaluación: el SFT se había evaluado con cuDNN, lote 50 y `MIG 3g.71gb`.

**Qué hicimos.** Evaluar el GRPO sin cuDNN en `MIG 1g.18gb`, y repetir el SFT en esas condiciones para
medir el ruido, siempre sin azar y con un máximo de 3.072 tokens:

```
uv run --extra train python -m rlm.evaluate --data rlm/data/test.jsonl \
    --adapters grpo=rlm/weights/final_rlm_lora --batch-size 16
uv run --extra train python -m rlm.evaluate --data rlm/data/test.jsonl \
    --adapters sft=rlm/weights/sft_lora --batch-size 16 --out reports/sft_rerun_test.json
# y lo mismo con rlm/data/test_ood.jsonl (reports/sft_rerun_ood.json)
```

**Qué pasó.**

| | SFT original | SFT repetido | GRPO |
|---|---|---|---|
| pass@1 `test` (200) | 56,0 % | 58,5 % | 62,5 % |
| pass@1 `test_ood` (100) | 58,0 % | 59,0 % | 66,0 % |
| pass@1 conjunto (300) | 56,7 % | 58,7 % | 63,7 % |
| Formato válido (300) | 82,3 % | 84,0 % | 89,3 % |
| Cortadas (300) | 17,7 % | 16,0 % | 10,7 % |

- **Ruido:** las dos evaluaciones del mismo SFT discrepan en el veredicto de 78 de 300 problemas (26 %),
  42 a favor de la repetida y 36 en contra (p = 0,57). Solo 3 de 300 respuestas salen idénticas. Con
  decodificación greedy, un cambio de redondeo que altere un token cambia todo lo que viene detrás.
- **GRPO frente al SFT repetido:** +5 puntos. Arregla 42 problemas y estropea 27 (McNemar exacto,
  p = 0,091). Por separado, p = 0,27 en `test` y p = 0,26 en `test_ood`. Frente al SFT original salían
  +7 puntos y p = 0,022: ese margen extra no se sostiene al repetir la evaluación.
- **Cortadas:** 37 respuestas dejan de cortarse y 21 empiezan a cortarse (p = 0,048). El formato da las
  mismas cifras. De las 48 cortadas del SFT, 18 pasan a acertar, 19 terminan mal y 11 siguen cortadas.

| Tipo de fallo (300) | SFT repetido | GRPO |
|---|---|---|
| Veredicto incorrecto | 44 | 57 |
| Cortada sin respuesta | 48 | 32 |
| Porcentaje incorrecto | 30 | 19 |
| Criterio incorrecto | 2 | 1 |

Los veredictos incorrectos suben por decir «no originario» cuando sí lo es (29 → 41); al revés casi no
cambian (15 → 16). Un ejemplo es el problema 0 de `test`, una bomba 8413 con la regla «CTSH, salvo a
partir de la partida 8482». Los rodamientos (8482.10) son de la UE con declaración de proveedor, pero
el GRPO los trata como no originarios por caer en la excepción, concluye `NINGUNO` y calcula 16,51 %,
cuando la respuesta es originario, CTSH y 0 %.

Entre SFT original y repetido, los grupos también bailan: en `test_ood`, `NINGUNO` pasa de 23 a 18 de
30 sin cambiar el modelo. Por eso no sacamos conclusiones por regla o criterio de las tablas de
`reports/phase1_analysis_*.md`, que además comparan con el SFT original. Por ejemplo, la aparente
caída del GRPO en CTSH en `test` (de 14,3 % a 7,1 %) desaparece con el SFT repetido: 2 de 28 en ambos.

**Qué concluimos.** El GRPO mejora un poco: +5 puntos de pass@1 sin significación estadística, menos
cortes y mejor formato en el límite de la significación. A cambio hay más veredictos incorrectos: parte
de las respuestas que antes se cortaban ahora terminan mal. El ejemplo de la bomba apunta a un error
concreto, confundir la excepción de la regla con el origen del material, pero no hemos contado cuántas
veces ocurre. El pass@1 de una
sola evaluación greedy tiene más ruido del que suponíamos, y eso afecta también a comparaciones
anteriores con diferencias pequeñas; la del base frente al SFT (+28 y +25 puntos) queda muy por encima.

Detectar 5 puntos con un 80 % de probabilidad requeriría unos 700 problemas. Descartamos evaluar en
1.000 nuevos por falta de tiempo, así que la mejora del pass@1 queda como indicio, no como resultado
demostrado. Si se comparan más modelos, hay que hacerlo en las mismas condiciones.

### 2026-11-05 · Fase 1 · GRPO 2: tres problemas por paso, casi todos los pasos con señal y respuestas más cortas

**Qué queríamos saber.** Si dar más problemas por paso reduce los pasos sin señal y el ruido del GRPO 1,
y si así mejora al SFT de forma clara.

**Qué hicimos.** Partir otra vez de `rlm/weights/sft_lora`, en `MIG 1g.18gb`:

```
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup uv run --extra train \
    python -m rlm.train_grpo --steps 270 --prompts-per-step 3 --learning-rate 7.5e-6 --save-steps 20 \
    --resume-from-checkpoint last --max-hours 7.5 \
    --output rlm/weights/grpo_pps3 >> runs/grpo_pps3.log 2>&1 &
```

Tres problemas por paso con 8 respuestas cada uno (24 respuestas, acumulación de gradiente 24). El
resto, igual que en el GRPO 1. Son 810 problemas, el 20 % de `train`, frente a 400. Elegimos el
learning rate para que pasos × learning rate quede como en el GRPO 1 (270 × 7,5e-6 ≈ 400 × 5e-6 =
2e-3). Con la partición de 16 GB no cabían 4 problemas por paso, como habíamos propuesto.

**Qué pasó.** Sumando `step_time`, unas 44,6 horas: 11 minutos por paso de mediana (máximo 12),
siete veces lo del GRPO 1 con tres veces más respuestas. Habíamos estimado unos 3 minutos sin medirlo.
No hemos comprobado por qué. Una posibilidad es que generar 24 respuestas a la vez obligue a leer una
caché de atención tres veces mayor en cada token. Como cada sesión dura como mucho 7,5 horas, fueron al
menos seis. Medias por bloques de 45 pasos (135 problemas):

| Pasos | Acierto | Formato | Cortadas | Longitud | Grupos sin señal |
|---|---|---|---|---|---|
| 1-45 | 0,71 | 0,97 | 0,03 | 1.454 | 0,39 |
| 46-90 | 0,70 | 0,98 | 0,02 | 1.454 | 0,40 |
| 91-135 | 0,73 | 0,96 | 0,04 | 1.491 | 0,45 |
| 136-180 | 0,69 | 0,98 | 0,02 | 1.480 | 0,42 |
| 181-225 | 0,73 | 0,97 | 0,03 | 1.420 | 0,47 |
| 226-270 | 0,74 | 0,98 | 0,02 | 1.362 | 0,50 |

- **Señal:** el 44 % de los grupos saca la misma recompensa en sus 8 respuestas, como en el GRPO 1 (37 %).
  Pero ahora solo 19 de 270 pasos (7 %) se quedan sin ningún grupo útil, frente a 149 de 400 (37 %).
  En los dos GRPO, los grupos sin señal aumentan hacia el final: cuantas más veces acierta las 8, menos
  aprende de ese problema.
- **Acierto:** +1,5 puntos por cada 100 pasos, sin distinguirse del ruido (permutaciones, p = 0,34).
- **Formato y cortadas:** mejoran un poco (0,96 → 0,98 y 0,036 → 0,018 entre los primeros y los últimos
  34 pasos), pero sin significación (p = 0,13 y p = 0,18). Empezaron más altos que en el GRPO 1.
- **Longitud:** baja de 1.464 a 1.369 tokens, unos 36 por cada 100 pasos (p = 0,053). En el GRPO 1 no
  bajaba.
- **`domain_reward`:** coincide con la de formato en 260 de 270 pasos. Sigue sin aportar señal propia.
- **Pesos:** el adaptador cambia un 0,98 % respecto al SFT, frente al 0,76 % del GRPO 1. La similitud de
  coseno entre los dos cambios es 0,06. Los dos GRPO mueven los pesos en direcciones casi distintas,
  aunque más parecidas de lo que saldría por azar en 17 millones de parámetros (del orden de 0,0002).

`reports/phase1_grpo_curves.png` pone las dos tiradas en el mismo eje de problemas vistos. Mientras
coinciden (primeros 400 problemas), las medias móviles de acierto, formato y longitud se solapan. Lo
que añade el GRPO 2 es la segunda mitad, donde la longitud baja.

**Qué concluimos.** Tres problemas por paso eliminan casi todos los pasos que no enseñan nada, a cambio
de 4,5 veces más tiempo total. Las curvas de entrenamiento siguen sin mostrar que el acierto suba. La
señal más clara es que el modelo acorta sus respuestas. La comparación en validación que nos habíamos
propuesto no la hicimos: los dos GRPO se comparan en `test` y `test_ood` (entrada siguiente).

### 2026-11-05 · Fase 1 · Evaluación del GRPO 2: +8,7 puntos sobre el SFT, la primera mejora significativa

**Qué queríamos saber.** Si el GRPO 2 supera al SFT y al GRPO 1, en las mismas condiciones que la
evaluación del 03/11.

**Qué hicimos.** Lo mismo que con el GRPO 1: sin cuDNN, `MIG 1g.18gb`, lote 16, greedy, 3.072 tokens.

```
uv run --extra train python -m rlm.evaluate --data rlm/data/test.jsonl \
    --adapters grpo2=rlm/weights/grpo_pps3 --batch-size 16 --out reports/grpo2_test.json
# y lo mismo con rlm/data/test_ood.jsonl (reports/grpo2_ood.json)
```

**Qué pasó.**

| | SFT repetido | GRPO 1 | GRPO 2 |
|---|---|---|---|
| pass@1 `test` (200) | 58,5 % | 62,5 % | 66,0 % |
| pass@1 `test_ood` (100) | 59,0 % | 66,0 % | 70,0 % |
| pass@1 conjunto (300) | 58,7 % | 63,7 % | 67,3 % |
| Formato válido (300) | 84,0 % | 89,3 % | 91,3 % |
| Cortadas (300) | 16,0 % | 10,7 % | 8,7 % |
| Tokens de media (300) | 1.621 | 1.552 | 1.462 |

- **GRPO 2 frente al SFT repetido:** +8,7 puntos. Arregla 51 problemas y estropea 25 (McNemar exacto,
  p = 0,004; IC 95 % de la diferencia, de +3,0 a +14,3 puntos). Por separado, p = 0,040 en `test` y
  p = 0,061 en `test_ood`.
- **GRPO 2 frente al GRPO 1:** +3,7 puntos. Arregla 35 y estropea 24 (p = 0,19; IC de -1,3 a +8,7). La
  diferencia va en la misma dirección en `test` (+3,5) y en `test_ood` (+4), pero cambian 59 veredictos,
  menos que los 78 que cambiaron al repetir el mismo SFT.

| Tipo de fallo (300) | SFT repetido | GRPO 1 | GRPO 2 |
|---|---|---|---|
| Veredicto incorrecto | 44 | 57 | 57 |
| Cortada sin respuesta | 48 | 32 | 26 |
| Porcentaje incorrecto | 30 | 19 | 14 |
| Criterio incorrecto | 2 | 1 | 1 |

Las cortadas y los porcentajes mal calculados siguen bajando, pero los veredictos incorrectos no. Por
dirección: «no originario» cuando sí lo es, 29 → 41 → 36; «originario» cuando no lo es, 15 → 16 → 21.

El problema 0 de `test` (la bomba 8413 con la excepción de la partida 8482) sigue fallando, pero de otra
forma. El GRPO 2 llega a que todos los materiales son originarios y que se cumple CTSH, y luego vuelve a
preguntarse si los rodamientos de la 8482 invalidan la regla. Repite la duda hasta agotar los 3.072
tokens sin dar respuesta.

**Qué concluimos.** El GRPO 2 es el modelo final de la fase 1. Supera al SFT con significación
estadística y no es peor que el GRPO 1 en ninguna métrica. Lo elegimos mirando `test` y `test_ood`, que
son también los conjuntos con los que lo comparamos, así que su ventaja sobre el GRPO 1 puede estar algo
inflada. Esa ventaja, además, no se distingue del ruido de evaluación. La mejora viene de cortar menos y
calcular mejor el porcentaje. Decidir bien con una excepción de partida sigue sin resolverse, y el ejemplo
de la bomba sugiere que parte de las respuestas cortadas son dudas de este tipo que no se cierran.

### 2026-11-06 · Fase 1 · Cinco fallos del GRPO 2: la mitad de los errores viene de la tolerancia y de las operaciones insuficientes

**Qué queríamos saber.** Por qué se equivoca el modelo final (`grpo_pps3`), con casos concretos y
comprobando si cada uno es un patrón o una anécdota.

**Qué hicimos.** Cruzar los 98 fallos de `test` y `test_ood` con las ramas de cada problema (las
`branches` del generador), leer el `<think>` de los casos elegidos y contar cuántas veces se repite
cada error. Para ver si el GRPO tuvo ocasión de corregirlos, cruzamos también los grupos de
`completions/` con las ramas de `train`.

**Qué pasó.** Dos ramas concentran los fallos:

| Rama (300 problemas) | Problemas | Acierto GRPO 2 | Acierto en el resto | Acierto SFT repetido |
|---|---|---|---|---|
| Originario gracias a la tolerancia | 42 | 12 % | 76 % | 10 % |
| Solo operaciones insuficientes | 28 | 32 % | 71 % | 39 % |

Las dos ramas no se solapan: suman 56 de los 98 fallos en 70 de los 300 problemas. En el resto de
problemas, el GRPO 2 acierta el 82 % (188 de 230). Las demás ramas no se separan del ruido (Fisher,
p > 0,1), salvo «la tolerancia no amplía el MaxNOM», donde acierta más (92 %).

**Fallo 1 · `test` 8 · la tolerancia se compara con el total.** Bomba 8413.70, CETA, CTSH con
tolerancia del 10 %. Lo único que impide CTSH es el cuerpo de bomba (8413.70, tercer país): 15,53 €
de 210 €, un 7,40 %. Es originario. El modelo clasifica bien los seis materiales, suma los dos no
originarios y escribe: «The percentage of non-originario materials is (52.15 / 210.00) * 100 ≈ 24.83%.
This is over 10%, so the tolerance doesn't apply here». El enunciado dice lo contrario: «En esa suma
solo entran los materiales que impiden cumplirlo, no todos los no originarios». Después cuenta los
rodamientos (UE con declaración, que él mismo había marcado como originarios) entre los que impiden
CTSH por estar en la partida 8482 excluida. De los 37 fallos con tolerancia, en 30 no aparece el
porcentaje de los materiales que bloquean, y en unos 16 se ve el total comparado con el 10 %
(búsqueda de texto, aproximada).

**Fallo 2 · `test` 24 · la confusión al revés.** Otra bomba con tolerancia. Aquí el modelo sí aísla el
material que bloquea («the only non-originario material in the same subpartida is the body… 69.81 /
970.00 = 7.17%. That's under 10%, so the tolerance applies») y acierta veredicto y criterio. Pero
responde `pct_no_originario` = 7,17 cuando se pedía el total no originario (54,42 %). Los dos
porcentajes del problema se le mezclan en las dos direcciones. Además, el cálculo es 7,20 %, no 7,17 %.

**Fallo 3 · `test` 17 · ve las operaciones insuficientes y no lo aplica.** Motor 8501.40, UE-Japón;
solo se ha pintado y embalado. El modelo lo dice al principio («The operations are painting and
packaging, which are not sufficient for origin»), pasa a comprobar CTH y RVC, y responde originario
por RVC. En 18 de los 19 fallos de esta rama, el producto sería originario sin la norma de las
operaciones: el modelo la ignora y aplica la regla. Y 8 de sus 9 aciertos son casos en que la regla
tampoco se cumple, así que acierta por otra vía. De los 19 problemas en que esta norma es lo único que
lo hace no originario, acierta 1.

**Fallo 4 · `test` 0 · una duda que no se cierra.** Bomba 8413.70 con rodamientos 8482.10 de la UE con
declaración. El modelo llega a la respuesta («all materials are originario… CTSH is satisfied.
Therefore, the product is originario») y vuelve a preguntarse si la excepción de la partida 8482
afecta a un material originario. Repite esa duda hasta agotar los 3.072 tokens. Es la misma confusión
que en el fallo 1, sobre si la excepción afecta también a materiales originarios. No la hemos
contado en el resto de las cortadas.

**Fallo 5 · `test_ood` 26 · la tolerancia como requisito.** Silla 9401.39, UE-Mercosur, CC. Ningún
material no originario es del capítulo 94, así que CC se cumple sin tolerancia. El modelo lo ve,
material a material, pero luego escribe: «Total non-originary value is 42.91 + 39.42 = 82.33 €…
48.43%. This is way over the 10% to[lerance]», y concluye no originario. Trata la tolerancia como una
condición más. Es poco frecuente: de los 36 casos en que dice «no originario» y sí lo es, 29 son de
tolerancia (fallo 1) y solo 4 de este tipo.

**Por qué el GRPO no lo arregla.** Es el mismo error que tenía el profesor (entrada del 2026-10-12 sobre
la tirada E6: comparaba con el 10 % la suma de todos los no originarios). El profesor acertó el 13 % de
los problemas con tolerancia (25 de 196) y el 20 % de los de operaciones insuficientes (37 de 188),
frente a cerca del 70 % en el resto. El SFT vio así 25 trazas con tolerancia y 37 con operaciones
insuficientes de 1.282. En el GRPO, un problema solo enseña si alguna de sus 8 respuestas acierta:

| Problemas de entrenamiento (GRPO 2) | Vistos | Acierto medio | Las 8 mal | Con señal |
|---|---|---|---|---|
| Tolerancia | 96 | 0,16 | 69 | 25 |
| Operaciones insuficientes | 62 | 0,31 | 32 | 28 |
| Resto | 652 | 0,84 | 3 | 380 |

En el GRPO 1 pasa lo mismo (30 de 41 y 16 de 39 con las 8 mal). Los grupos en que todas fallan son
casi todos de estas dos ramas (101 de 104), justo donde el modelo más tendría que aprender.

**Qué concluimos.** El modelo no falla al azar. La mitad de sus errores viene de dos normas que el
profesor tampoco sabía aplicar. El SFT las copió de él y el GRPO no puede corregirlas, porque el modelo
casi nunca acierta en esos problemas y no hay respuesta buena de la que aprender. Para arreglarlo
habría que darle ejemplos correctos de estas dos ramas: trazas escritas desde el motor de origen
(`rlm/origin_engine.py` ya tiene la traza material a material) o un profesor mayor, filtrado con el
verificador. No lo hemos probado.

### 2026-11-07 · Fase 1 · El paso de GRPO a mano, y por qué el recorte no actuó en nuestros entrenamientos

**Qué queríamos saber.** Escribir el paso de GRPO sin `GRPOTrainer` (`rlm/grpo_step.py`) y comprobar
en qué se parece a lo que hizo TRL en nuestras dos tiradas.

**Qué hicimos.** Implementar las cinco funciones: ventajas normalizadas dentro del grupo, ratio de
políticas en espacio logarítmico, objetivo recortado con `min(ρ·A, clip(ρ, 1-ε, 1+ε)·A)`, estimador
de la KL de DeepSeekMath y la pérdida, con media por respuesta (`1/|o_i|`) y luego por grupo (`1/G`).
Pasan los 6 tests del enunciado, más dos nuestros: la KL con `beta > 0` y la fracción de recorte.

**Qué pasó.** Al comparar con `training_args.bin` y `log_history.json` de las dos tiradas:

- **El recorte nunca actuó:** `clip_ratio/region_mean` vale 0 en los 270 pasos del GRPO 2, y también en
  el GRPO 1. Con `num_iterations` = 1, TRL genera las respuestas y hace una sola actualización con
  ellas. La política que generó (`π_θ_old`) es la misma que se actualiza, así que ρ = 1 en todos los
  tokens y el recorte no tiene nada que limitar. El gradiente sí existe: con ρ = 1, el objetivo se
  reduce a un gradiente de política con la ventaja como peso. El recorte solo actuaría reutilizando
  cada lote para varias actualizaciones (`num_iterations` > 1).
- **La KL tampoco:** usamos `beta` = 0, así que nada ataba el modelo al SFT salvo el learning rate
  pequeño y el número de pasos. Por eso medimos cuánto cambiaban los pesos (0,76 % y 0,98 %).
- **La normalización es distinta:** nuestro `grpo_loss` promedia primero dentro de cada respuesta y
  luego entre respuestas, como la fórmula de las diapositivas. TRL usó `loss_type="dapo"`, que promedia
  todos los tokens del lote juntos. Así, cada token pesa lo mismo y las respuestas largas cuentan más.
  Con la media por respuesta, cada token de una respuesta larga pesa menos.

**Qué concluimos.** En nuestras tiradas, GRPO se comportó como un gradiente de política con ventajas
normalizadas por grupo, sin recorte ni KL. Los dos mecanismos están implementados y probados en
`grpo_step.py`, pero para que el recorte importe habría que reutilizar cada lote de respuestas, cosa
que no hicimos.
