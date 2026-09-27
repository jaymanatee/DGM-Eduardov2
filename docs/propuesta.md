# Propuesta de tema: asistente de origen preferencial para exportadores

## Equipo

- Enrique Fernández-Baíllo Rodríguez de Tembleque (202115209@alu.comillas.edu)
- Jacobo Banús Bertram (202214133@alu.comillas.edu)
- Xabier Albizu Arias (202207372@alu.comillas.edu)

## El tema en una frase

Un agente que dice a una pyme exportadora española si su producto es **originario** según
las reglas de un acuerdo comercial de la UE, por qué criterio y con qué porcentaje. Si no lo
es, le dice qué cambiar para conseguirlo, y le prepara la declaración de origen.

## El usuario y su problema

El usuario es el responsable de comercio exterior de una pyme industrial, o el representante
aduanero que la asesora.

Para que su cliente en Reino Unido, Japón o Corea pague menos arancel, tiene que demostrar
que el producto cumple la regla de origen de su partida en ese acuerdo. Hoy lo hace con una
hoja de cálculo: lista de materiales, código y origen de cada componente, declaraciones de
proveedores y la regla leída en un anexo. Tarda horas por producto y mercado, así que
muchas veces no compensa y se paga el arancel general. Si se equivoca, la aduana puede
reclamar los derechos durante tres años.

Las cifras:

- los exportadores españoles dejaron sin aprovechar **648 M€** de ahorro arancelario (298 M€
  en 2022);
- solo en manufacturas, más de 216 M€ (cálculo propio con los mismos datos).

## Diez preguntas o tareas reales

1. ¿El acuerdo con Canadá rebaja el arancel de nuestras bombas centrífugas (8413.70) o ya
   entran al 0 % sin necesidad de acreditar el origen?
2. ¿Qué arancel paga en Japón un zapato de piel español, con preferencia y sin ella?
3. ¿Qué regla de origen tiene la partida 8507 en el acuerdo con Reino Unido?
4. Tengo un envío de 4.500 € a Corea, ¿necesito número REX?
5. Mi batería lleva celdas chinas por 95 € y la vendo a 250 € franco fábrica. ¿Es originaria
   para Reino Unido?
6. Con este Excel de materiales, ¿mi máquina es originaria para Japón, Corea y Canadá?
7. Un proveedor alemán no me manda su declaración de proveedor. ¿Qué hago y cómo afecta al
   cálculo?
8. ¿Qué componente tendría que comprar en la UE para que mi producto cumpla con Japón?
9. A partir de 2027, ¿mis coches eléctricos seguirán entrando sin arancel en Reino Unido con
   celdas coreanas?
10. La aduana japonesa pide justificar el origen de un envío de hace dos años. Prepárame el
    expediente.

## La tarea verificable (fase 1)

**El problema.** El enunciado da:

- la regla específica literal de la partida y las reglas generales (tolerancia, acumulación,
  operaciones insuficientes);
- el precio franco fábrica;
- la lista de materiales: de 3 a 10 filas con código SA, origen, valor y si hay declaración
  de proveedor.

**La respuesta** va en `<answer>` como JSON:
`{"originario": bool, "criterio": "CTH|CTSH|CC|MAXNOM|RVC|WO|NINGUNO", "pct_no_originario": float}`.

**Dos ejemplos:**

1. Batería de ion-litio (8507.60) hacia Reino Unido. Regla: CTH o MaxNOM 50 % (EXW). Precio
   franco fábrica: 250 €. No originarios: celdas chinas (8507.60, 95 €) y placa taiwanesa
   (8537.10, 20 €). → `{"originario": true, "criterio": "MAXNOM", "pct_no_originario": 46.0}`
2. Ventilador (8414.59). Regla del enunciado: CTH, con tolerancia del 10 % sobre el precio
   franco fábrica. Precio: 400 €. No originarios: motor chino (8501, 120 €) y rodete chino
   (8414.90, 30 €). El rodete es de la misma partida, pero pesa el 7,5 %, así que la
   tolerancia lo rescata.
   → `{"originario": true, "criterio": "CTH", "pct_no_originario": 37.5}`

**Cómo se verifica.** Un motor de reglas propio recalcula el caso. Comprueba el veredicto
exacto, que el criterio esté entre los que se cumplen (con reglas alternativas puede haber
varios) y el porcentaje a ±0,5 puntos.

**Cómo se construye el conjunto.**

- **Estrategia 1, generador.** El motor de reglas es a la vez `solve` y el verificador.
- **Estrategia 2, reglas minadas.** Las reglas específicas son reales y salen de la API del
  *UK Trade Tariff*, revisadas contra los anexos de EUR-Lex. Son unas 150-300 reglas de cinco
  acuerdos: TCA, UE-Japón, UE-Corea, CETA y PEM revisado.
- **Parámetros que se muestrean:**
  - acuerdo;
  - producto (unas 40 plantillas de los capítulos 84, 85, 87 y 94);
  - lista de materiales (código, origen UE / socio / tercero / desconocido, valor,
    declaración de proveedor sí o no);
  - precio franco fábrica y FOB;
  - operaciones realizadas;
  - formato del enunciado.
- **Casos trampa**, con al menos un 8 % cada uno:
  - material de la misma partida;
  - excepción "salvo a partir de la partida X";
  - tolerancia que no puede superar el MaxNOM;
  - operación insuficiente;
  - proveedor sin declaración;
  - subdivisión equivocada;
  - umbral exacto.
- **Tamaños:** 4.000 de entrenamiento, 200 de test (50 auditados a mano y contrastados con
  ROSA de la Comisión) y 100 fuera de distribución (reglas redactadas al estilo UE-Corea y el
  acuerdo UE-Mercosur, ausentes del entrenamiento). Clases equilibradas.
- **Control:** un 20-30 % de GSM8K.

**Tercera recompensa, trazabilidad.** Se premia que el razonamiento cite la regla literal y
liste los materiales no originarios con una suma que coincida con la del motor. Se penaliza
decir "originario" sin criterio.

## Las herramientas (fase 2)

- **Consulta:** la API pública del [UK Trade Tariff](https://api.trade-tariff.service.gov.uk/reference.html).
  El endpoint `rules_of_origin_schemes` da las reglas estructuradas y la variante `xi` las
  medidas de la tarifa de la UE. Se complementa con
  [WITS del Banco Mundial](https://wits.worldbank.org/witsapiintro.aspx) para el arancel del
  país de destino. Todas probadas y sin clave.

  La API es del Gobierno británico, y cada uso tiene su justificación:
  - las reglas del TCA son las mismas para el exportador de la UE, porque es un solo acuerdo;
  - la variante `xi` replica la tarifa de la UE, porque Irlanda del Norte la aplica por el
    Marco de Windsor;
  - para UE-Japón, UE-Corea, CETA y PEM usamos las reglas de los anexos de EUR-Lex, que la
    UE no publica mediante API.
- **Cálculo:** `evaluate_origin`, el mismo motor de la fase 1, que devuelve veredicto,
  porcentaje y traza material a material. El modelo no debe hacerlo de cabeza: cualquier
  error de suma o de regla acaba en una declaración falsa.
- **Acción** (con confirmación): `issue_origin_statement` genera un PDF con el texto exacto
  de la declaración de origen del anexo del acuerdo y registra el expediente en SQLite. Se
  comprueba viendo que el fichero existe, que el texto coincide con la plantilla y que la
  fila está en la base de datos.

## El corpus (fase 3)

- **Fuentes:** textos de origen de los acuerdos (EUR-Lex); guías de la Comisión (reglas de
  origen preferencial de 2025, *Guide for Traders* del TCA, guías de UE-Japón y de
  UE-Mercosur de 2026); guías de la AEAT (REX, exportador autorizado); notas explicativas de
  la Nomenclatura Combinada de los capítulos 84, 85, 87 y 94. Entre 40 y 70 documentos y
  varios miles de páginas.
- **Formato:** HTML y PDF.
- **Licencia:**
  - documentos de la UE, Decisión 2011/833/UE (reutilización citando la fuente);
  - AEAT, reutilización autorizada citando la fuente;
  - datos británicos, OGL v3.
- **Preguntas que solo responde el corpus:** "¿Qué tolerancia admite el PEM revisado y se
  permite el reintegro de derechos en textiles?" y "¿Qué necesita un exportador español para
  declarar el origen de un envío de 8.000 € a Brasil bajo el acuerdo con Mercosur?"

## Qué puede salir mal

- **Extraer las reglas cuesta más de lo previsto.** Usaremos la API británica como borrador
  estructurado y revisaremos contra EUR-Lex solo las reglas del alcance. Solo el TCA es
  idéntico por los dos lados.
- **El modelo de 1.7B se pierde con listas de materiales largas.** Máximo de 10 materiales y
  enunciados de menos de 1.500 tokens en la fase 1. Lo largo se resuelve en la fase 4 con
  herramientas.
- **El RAG confunde acuerdos**, porque los artículos son casi idénticos entre acuerdos.
  Trocearemos por artículo y por fila de regla, con el acuerdo como metadato y en la
  instrucción del embedding, y lo mediremos.

## Por qué este tema

Es un problema industrial con dinero medible y un calendario muy activo:

- Mercosur se aplica desde mayo de 2026;
- las reglas del coche eléctrico con Reino Unido se endurecen en 2027;
- desde 2028 las declaraciones de proveedor serán datos normalizados.

Además, la regla de origen es un algoritmo: el verificador sale solo. Y no existe ningún
benchmark abierto de razonamiento sobre origen preferencial que podamos usar o superar.
