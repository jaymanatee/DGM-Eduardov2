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
| `E3_trazas_dirigidos20_max2048.jsonl` | 16 de los 20 dirigidos | Cortado a mano |
| `E4_trazas_dirigidos20_max4096.jsonl` | 12 de los 20 dirigidos | Cortado por falta de memoria |
| `E5_trazas_dirigidos20.jsonl` | Los 20 dirigidos | Completo |
| `rlm/data/sft_traces.jsonl` | 0 a 499 | Tirada E6, la que va al SFT: 1.282 trazas válidas de 2.000 |

Los "20 dirigidos" son los problemas 0 a 10, 12, 15, 22, 26, 31, 37, 39, 47 y 49 de `train`:
los 10 primeros que solo son originarios por un criterio de cambio de clasificación y los 10
primeros del resto. Se obtienen con el script `pilot_subset.py`, que ya no está en el repositorio:
su código está en el anexo, al final de este cuaderno.

Referencias fijas con el enunciado E6 (se actualizan solo si cambia el generador a propósito):

| Fichero | Comando | sha256 |
|---|---|---|
| `train.jsonl` | `--n 4000 --split train --seed 0` | `b61dcfc8012aea2d7343a48eed8549f0a07b7a6c6c91bb4526605839cbf4e3d2` |
| `test.jsonl` | `--n 200 --split test --seed 0` | `9e66110dd87cd5bc9d3970ae38795c76d4b1aad649194f6e6c47f0f7d95840be` |
| `test_ood.jsonl` | `--n 100 --split ood --seed 0` | `58787a033551ba6f68a0c92d9375985752f5855ba4de37b7b38223200cee08c8` |
| `sft_problemas.jsonl` | `--n 500 --split train --seed 0` | `3327deb2e5d4a831c458cc7731bddeafc61b70f80b9c6821e9427f1e766ca17d` |

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

Para medirlo sin esperar a una tirada entera, escribimos `rlm/pilot_subset.py` (código en el
anexo), que elige 20
problemas de los 256 de la tirada E2: los 10 primeros que solo son originarios por cambio de
clasificación y los 10 primeros del resto. Así cada resultado se compara con el de esos mismos
problemas en E2. Piloto con 2.048 tokens; lo cortamos con 16 problemas hechos.

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
  frente a 26 en los 256 problemas de E2). Por lo visto en la revisión de los 56 primeros
  problemas, la causa probable es la frase de la tolerancia de E6: el profesor responde con el
  porcentaje de los materiales que impiden el cambio de clasificación en lugar del de todos los
  no originarios. No lo hemos comprobado problema a problema en la tirada completa. Esas trazas
  las rechaza el verificador, así que no contaminan el conjunto; solo reducen su tamaño. También
  baja `CETA-8413` (CTSH), del 57 % al 41 %, probablemente por la misma causa.
- Lección para el diseño: cada frase nueva del enunciado arregló el fallo para el que se escribió
  y pudo abrir otro. Medir sobre los mismos problemas antes y después (los 20 dirigidos) es lo
  que permitió verlo; E6 fue la única versión que no pilotamos, y es la que tiene el efecto
  secundario más claro.

Repetir con la redacción de la tolerancia de E5 recuperaría parte de los no originarios, pero
costaría otras 24 horas de GPU y la tolerancia seguiría prácticamente sin resolver. No compensa.

---

## Anexo: `pilot_subset.py`

Script que usamos para los pilotos dirigidos de los enunciados E3 a E5. No está en el
repositorio; para usarlo, guardar este código como `rlm/pilot_subset.py` y ejecutar:

```
uv run python -m rlm.pilot_subset --out rlm/data/piloto_def.jsonl
uv run python -m rlm.pilot_subset --trap tolerancia_usada --out rlm/data/piloto_tol.jsonl
```

Los problemas elegidos son siempre los mismos, pero la huella del fichero cambia con cada versión
del enunciado, porque cambia el texto. Con el primer comando: `4674dcf7…` con E5 y `c4635d3d…` con E6.

```python
"""Subconjunto de train para un piloto de destilación dirigido a los casos difíciles.

Entre los N primeros problemas de train elige k problemas difíciles y los k primeros del resto.
Por defecto, los difíciles son los originarios solo por un criterio de cambio de clasificación
(CC, CTH o CTSH); con ``--trap`` son los que tienen esa trampa en ``branches``. Son problemas
de train (misma semilla), así que el resultado se compara problema a problema con una
destilación anterior sobre esos N.

    uv run python -m rlm.pilot_subset --out rlm/data/piloto_def.jsonl
    uv run python -m rlm.pilot_subset --trap tolerancia_usada --out rlm/data/piloto_tol.jsonl

Imprime qué problemas de train ha elegido: la fila i del fichero es el problema ``train_idx[i]``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rlm.generate_problems import OriginGenerator, Problem, to_jsonl

CLASSIFICATION = {"CC", "CTH", "CTSH"}


def is_classification_only(problem: Problem) -> bool:
    """Originario, y ningún criterio de valor (MaxNOM, RVC) lo hace originario."""
    truth = json.loads(problem.answer)
    return truth["originario"] and set(truth["criterios_validos"]) <= CLASSIFICATION


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--from-first", type=int, default=256, help="N primeros de train")
    parser.add_argument("--k", type=int, default=10, help="problemas de cada grupo")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trap", default=None, help="rama de branches que define los difíciles")
    parser.add_argument("--out", default="rlm/data/piloto_def.jsonl")
    args = parser.parse_args()

    problems = OriginGenerator().generate(args.from_first, "train", args.seed)
    if args.trap:
        is_hard = lambda p: p.branches.get(args.trap) == "True"  # noqa: E731
    else:
        is_hard = is_classification_only
    hard = [i for i, p in enumerate(problems) if is_hard(p)][: args.k]
    rest = [i for i, p in enumerate(problems) if not is_hard(p)][: args.k]
    indices = sorted(hard + rest)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(to_jsonl([problems[i] for i in indices], "train"), encoding="utf-8")
    print(json.dumps({"train_idx": indices, "dificiles": hard, "resto": rest}))
    print(f"{len(indices)} problemas -> {out}")


if __name__ == "__main__":
    main()
```
