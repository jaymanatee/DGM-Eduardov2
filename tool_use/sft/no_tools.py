"""Synthetic trajectories without tool calls: concept questions, requests outside the domain,
agreements we do not cover, and evaluations the user has not given enough data for.

A model that calls a tool for everything is not using tools well. These teach when not to
call: answer a concept directly, decline an agreement that is not covered, and ask for what
is missing instead of making it up.

The benchmark's own no-tool questions are kept out on purpose (what REX stands for, CTH
against CTSH, ex-works against FOB, the general tolerance, a greeting, a haiku, the EU-Mexico
agreement): the benchmark has to say whether the model learnt *when* not to call, not the
answers to those questions.
"""

from __future__ import annotations

import random
from datetime import date

from tool_use.sft.catalog import Product
from tool_use.sft.generate import (
    AGREEMENT_NAMES,
    EU_COUNTRIES,
    PARTNER,
    Conversation,
    Trajectory,
    describe_line,
    hs,
    money,
    pick,
    sample_scenario,
    spanish,
)

# (question variants ES, question variants EN, answer ES, answer EN)
CONCEPTS: dict[str, tuple[tuple[str, ...], tuple[str, ...], str, str]] = {
    "maxnom": (
        ("¿Qué significa MaxNOM en una regla de origen?", "¿Cómo funciona el criterio MaxNOM?"),
        ("What does MaxNOM mean in a rule of origin?", "How does the MaxNOM criterion work?"),
        "MaxNOM es el valor máximo de materiales no originarios. La regla fija un porcentaje del "
        "precio franco fábrica (por ejemplo, el 50 %) y el producto lo cumple si el valor de todos "
        "los materiales no originarios que usa no lo supera. No importa en qué partida estén "
        "clasificados: solo cuenta su valor.",
        "MaxNOM is the maximum value of non-originating materials. The rule sets a percentage of "
        "the ex-works price (50 %, for instance) and the product meets it if the value of all the "
        "non-originating materials it uses does not exceed it. Their tariff classification does "
        "not matter: only their value counts.",
    ),
    "rvc": (
        ("¿Qué es el contenido de valor regional (RVC)?", "¿Cómo se calcula el RVC?"),
        ("What is regional value content (RVC)?", "How is RVC calculated?"),
        "El contenido de valor regional (RVC) es la parte del valor del producto que es "
        "originaria, y la regla exige un mínimo. En el acuerdo UE-Japón se calcula sobre el precio "
        "FOB: (FOB − valor de los materiales no originarios) / FOB × 100. Es la otra cara de "
        "MaxNOM: en vez de limitar lo no originario, exige un mínimo de valor propio.",
        "Regional value content (RVC) is the share of the product's value that is originating, "
        "and the rule requires a minimum. In the EU-Japan agreement it is calculated on the FOB "
        "price: (FOB − value of non-originating materials) / FOB × 100. It is the other side of "
        "MaxNOM: instead of capping what is non-originating, it requires a minimum of own value.",
    ),
    "cc": (
        ("¿Qué quiere decir «cambio de capítulo» (CC)?", "¿Qué exige el criterio CC?"),
        ("What does 'change of chapter' (CC) mean?", "What does the CC criterion require?"),
        "El cambio de capítulo (CC) exige que todos los materiales no originarios estén "
        "clasificados en un capítulo del Sistema Armonizado (las dos primeras cifras del código) "
        "distinto del del producto. Es el criterio de cambio de clasificación más exigente.",
        "Change of chapter (CC) requires every non-originating material to be classified in a "
        "chapter of the Harmonized System (the first two digits of the code) other than the "
        "product's. It is the strictest of the change-of-classification criteria.",
    ),
    "wholly_obtained": (
        ("¿Cuándo es un producto «enteramente obtenido»?", "¿Qué significa WO en origen?"),
        ("When is a product 'wholly obtained'?", "What does WO mean for origin?"),
        "Un producto es enteramente obtenido (WO) cuando todo él procede de una de las partes, "
        "sin materiales de fuera: minerales extraídos allí, plantas cosechadas allí, animales "
        "nacidos y criados allí, pescado capturado por sus buques… Si lo es, es originario sin "
        "más; cualquier material no originario lo impide.",
        "A product is wholly obtained (WO) when all of it comes from one of the parties, with no "
        "outside materials: minerals extracted there, plants harvested there, animals born and "
        "raised there, fish caught by its vessels… If it is, it is originating, full stop; any "
        "non-originating material rules it out.",
    ),
    "supplier_declaration": (
        ("¿Para qué sirve la declaración del proveedor?", "¿Qué es una declaración del proveedor?"),
        ("What is a supplier's declaration for?", "What is a supplier's declaration?"),
        "Es el documento en el que el proveedor declara que los materiales que vende son "
        "originarios según las normas de un acuerdo concreto. Permite al fabricante contarlos como "
        "originarios al calcular el origen de su producto; sin ella, incluso un material comprado "
        "en la UE se trata como no originario.",
        "It is the document in which the supplier declares that the materials it sells are "
        "originating under the rules of a given agreement. It lets the manufacturer count them "
        "as originating when working out the origin of its product; without it, even a material "
        "bought in the EU is treated as non-originating.",
    ),
    "cumulation": (
        (
            "¿Qué es la acumulación bilateral de origen?",
            "¿Puedo contar como originarios materiales del país socio?",
        ),
        (
            "What is bilateral cumulation of origin?",
            "Can I count materials from the partner country as originating?",
        ),
        "Sí: con la acumulación bilateral, los materiales originarios del país socio (por "
        "ejemplo, de Japón en el acuerdo UE-Japón) cuentan como originarios cuando se incorporan "
        "a un producto fabricado en la UE, y al revés. Hay que poder demostrar su origen, "
        "normalmente con una declaración del proveedor.",
        "Yes: with bilateral cumulation, materials originating in the partner country (Japan, "
        "under the EU-Japan agreement) count as originating when they go into a product made in "
        "the EU, and the other way round. Their origin has to be proven, usually with a "
        "supplier's declaration.",
    ),
    "insufficient": (
        (
            "¿Qué son las operaciones insuficientes?",
            "Si solo embalamos y etiquetamos, ¿el producto es originario?",
        ),
        (
            "What are insufficient operations?",
            "If we only pack and label, is the product originating?",
        ),
        "Son operaciones demasiado simples para dar origen aunque se cumpla la regla de la "
        "partida: conservar la mercancía, embalar o reembalar, etiquetar, limpiar, pintar o "
        "montar de forma simple, mezclar… Si es lo único que se hace en la UE, el producto no "
        "es originario.",
        "They are operations too simple to confer origin even if the product-specific rule is "
        "met: preserving the goods, packing or repacking, labelling, cleaning, simple painting "
        "or assembly, mixing… If that is all that is done in the EU, the product is not "
        "originating.",
    ),
    "preferential": (
        (
            "¿Qué diferencia hay entre origen preferencial y no preferencial?",
            "¿El «made in EU» es lo mismo que el origen preferencial?",
        ),
        (
            "What is the difference between preferential and non-preferential origin?",
            "Is 'made in the EU' the same as preferential origin?",
        ),
        "No. El origen preferencial da derecho a los aranceles rebajados de un acuerdo comercial "
        "y se decide con las reglas de ese acuerdo. El no preferencial sirve para otras medidas "
        "(antidumping, contingentes, marcado de origen) y sigue el Código Aduanero de la Unión. "
        "Un producto puede ser «made in EU» y no ser originario a efectos de un acuerdo.",
        "No. Preferential origin gives access to the reduced duties of a trade agreement and is "
        "decided by that agreement's rules. Non-preferential origin serves other measures "
        "(anti-dumping, quotas, origin marking) and follows the Union Customs Code. A product "
        "can be 'made in the EU' and still not be originating under an agreement.",
    ),
    "statement": (
        (
            "¿Qué es la declaración de origen en factura?",
            "¿Dónde se pone la declaración de origen?",
        ),
        ("What is a statement on origin?", "Where does the statement on origin go?"),
        "Es el texto que el exportador escribe en la factura u otro documento comercial para "
        "declarar que los productos son originarios según el acuerdo; su redacción está fijada "
        "en un anexo del acuerdo. En el TCA, el acuerdo UE-Japón y el CETA, para envíos de más "
        "de 6.000 € el exportador de la UE tiene que indicar su número REX.",
        "It is the text the exporter writes on the invoice or another commercial document to "
        "declare that the products are originating under the agreement; its wording is set in "
        "an annex of the agreement. Under the TCA, the EU-Japan agreement and CETA, for "
        "consignments above EUR 6,000 the EU exporter has to give its REX number.",
    ),
    "importer_knowledge": (
        (
            "¿Qué es el «conocimiento del importador»?",
            "¿Se puede pedir la preferencia sin declaración del exportador?",
        ),
        (
            "What is 'importer's knowledge'?",
            "Can preference be claimed without the exporter's statement?",
        ),
        "En el TCA y en el acuerdo UE-Japón, sí: el importador puede pedir el trato preferencial "
        "basándose en su propio conocimiento de que el producto es originario, sin declaración del "
        "exportador. Tiene que poder justificarlo con información sobre los materiales y la "
        "fabricación si la aduana se la pide.",
        "Under the TCA and the EU-Japan agreement, yes: the importer can claim preference on the "
        "basis of its own knowledge that the product is originating, with no statement from the "
        "exporter. It must be able to back it up with information on the materials and the "
        "production if customs asks.",
    ),
    "hs_structure": (
        ("¿Cómo se lee un código del Sistema Armonizado?", "¿Qué es una subpartida?"),
        ("How do you read a Harmonized System code?", "What is a subheading?"),
        "El código del Sistema Armonizado tiene 6 cifras: las dos primeras son el capítulo, las "
        "cuatro primeras la partida y las seis la subpartida. La UE añade cifras para la "
        "Nomenclatura Combinada (8) y TARIC (10), pero las reglas de origen usan como mucho las 6 "
        "primeras.",
        "A Harmonized System code has 6 digits: the first two are the chapter, the first four "
        "the heading and all six the subheading. The EU adds digits for the Combined Nomenclature "
        "(8) and TARIC (10), but rules of origin use at most the first 6.",
    ),
    "approved_exporter": (
        (
            "¿Qué es un exportador autorizado en el acuerdo UE-Corea?",
            "¿Quién puede hacer la declaración de origen para Corea del Sur?",
        ),
        (
            "What is an approved exporter under the EU-Korea agreement?",
            "Who can make out the origin declaration for South Korea?",
        ),
        "En el acuerdo UE-Corea, la declaración de origen en factura la puede hacer cualquier "
        "exportador si el envío no pasa de 6.000 €. Por encima, solo un exportador autorizado, al "
        "que la aduana da un número de autorización que tiene que figurar en la declaración.",
        "Under the EU-Korea agreement, any exporter can make out the origin declaration on the "
        "invoice if the consignment does not exceed EUR 6,000. Above that, only an approved "
        "exporter can, and the authorisation number customs gives it must appear in the "
        "declaration.",
    ),
}

OFF_TOPIC = (
    (
        ("¿Cómo se dice «albarán» en inglés?",),
        ("How do you say 'invoice' in Spanish?",),
        "Se dice «delivery note».",
        "It is «factura».",
    ),
    (
        ("¿Cuál es la capital de Australia?",),
        ("What is the capital of Canada?",),
        "Canberra.",
        "Ottawa.",
    ),
    (
        ("¿Qué tiempo va a hacer mañana en Bilbao?",),
        ("What will the weather be like in Rotterdam tomorrow?",),
        "No tengo acceso a previsiones del tiempo. Puedo ayudarte con el origen preferencial, los "
        "tipos de cambio del BCE y la comprobación de NIF-IVA.",
        "I have no access to weather forecasts. I can help with preferential origin, ECB exchange "
        "rates and VAT number checks.",
    ),
    (
        ("Gracias, eso es todo.",),
        ("Thanks, that's all.",),
        "De nada. Aquí estoy si necesitas algo más.",
        "You're welcome. I'm here if you need anything else.",
    ),
)

OTHER_AGREEMENTS = (
    ("el acuerdo UE-Mercosur", "the EU-Mercosur agreement", "Brasil", "Brazil"),
    ("el acuerdo UE-Vietnam", "the EU-Vietnam agreement", "Vietnam", "Vietnam"),
    ("el acuerdo UE-Singapur", "the EU-Singapore agreement", "Singapur", "Singapore"),
    ("el acuerdo UE-Chile", "the EU-Chile agreement", "Chile", "Chile"),
    ("el acuerdo UE-Nueva Zelanda", "the EU-New Zealand agreement", "Nueva Zelanda", "New Zealand"),
    ("el acuerdo UE-Suiza", "the EU-Switzerland agreement", "Suiza", "Switzerland"),
    ("el acuerdo UE-Marruecos", "the EU-Morocco agreement", "Marruecos", "Morocco"),
    ("el acuerdo UE-Ucrania", "the EU-Ukraine agreement", "Ucrania", "Ukraine"),
)

COVERED = (
    "Solo trabajo con cuatro acuerdos: el TCA (UE-Reino Unido), el acuerdo UE-Japón, el acuerdo "
    "UE-Corea y el CETA (UE-Canadá).",
    "I only work with four agreements: the TCA (EU-UK), the EU-Japan agreement, the EU-Korea "
    "agreement and CETA (EU-Canada).",
)


def concept_trajectory(rng: random.Random, today: date, lang: str, topic: str) -> Trajectory:
    questions_es, questions_en, answer_es, answer_en = CONCEPTS[topic]
    t = Trajectory("concept", lang, today.isoformat(), outcome=topic)
    conv = Conversation(t)
    conv.user(pick(rng, questions_es if lang == "es" else questions_en))
    conv.answer(
        pick(
            rng,
            (
                "A general concept, not data about a product: I answer directly, no tool.",
                "This is about how origin works in general; no lookup needed.",
            ),
        ),
        answer_es if lang == "es" else answer_en,
    )
    return t


def off_topic_trajectory(rng: random.Random, today: date, lang: str) -> Trajectory:
    questions_es, questions_en, answer_es, answer_en = pick(rng, OFF_TOPIC)
    t = Trajectory("off_topic", lang, today.isoformat(), outcome="off_topic")
    conv = Conversation(t)
    conv.user(pick(rng, questions_es if lang == "es" else questions_en))
    conv.answer(
        "Nothing here for the tools: a short direct answer.",
        answer_es if lang == "es" else answer_en,
    )
    return t


def out_of_scope_trajectory(
    rng: random.Random, product: Product, today: date, lang: str
) -> Trajectory:
    agreement_es, agreement_en, country_es, country_en = pick(rng, OTHER_AGREEMENTS)
    code = hs(product.code, rng, lang)
    name = product.es if lang == "es" else product.en
    t = Trajectory("out_of_scope", lang, today.isoformat(), product=product.en, outcome="declined")
    conv = Conversation(t)
    if lang == "es":
        question = pick(
            rng,
            (
                f"¿Qué regla de origen tiene {code} en {agreement_es}?",
                f"¿Es originario «{name}» ({code}) para exportarlo a {country_es}?",
                f"Queremos vender «{name}» en {country_es}. ¿Qué exige la regla de origen?",
            ),
        )
        answer = (
            f"{COVERED[0]} {agreement_es[0].upper() + agreement_es[1:]} no está entre ellos, "
            "así que no puedo consultar su regla de origen ni evaluar el producto con él."
        )
    else:
        question = pick(
            rng,
            (
                f"What is the rule of origin for {code} under {agreement_en}?",
                f"Is the {name} ({code}) originating for export to {country_en}?",
                f"We want to sell the {name} in {country_en}. "
                "What does the rule of origin require?",
            ),
        )
        answer = (
            f"{COVERED[1]} {agreement_en[0].upper() + agreement_en[1:]} is not one of them, so "
            "I cannot look up its rule of origin or evaluate the product under it."
        )
    conv.user(question)
    conv.answer(
        f"{country_en}: {agreement_en} is not one of the four agreements the tools cover. I say "
        "so without calling a tool.",
        answer,
    )
    return t


def missing_info_trajectory(
    rng: random.Random, product: Product, agreement: str, today: date, lang: str
) -> Trajectory:
    """An evaluation without the ex-works price, the materials or the destination: ask."""
    missing = rng.choice(("price", "materials", "destination"))
    s = sample_scenario(rng, product, agreement, today)
    t = Trajectory(
        "missing_info",
        lang,
        today.isoformat(),
        product=product.en,
        agreement="" if missing == "destination" else agreement,
        outcome=missing,
    )
    conv = Conversation(t)
    es = lang == "es"
    name = product.es if es else product.en
    code = hs(product.code, rng, lang)
    made_in = EU_COUNTRIES[s.exporter_country][0 if es else 1]
    ag = pick(rng, AGREEMENT_NAMES[agreement][0 if es else 1])
    partner = PARTNER[agreement][0 if es else 1]
    lines = "; ".join(describe_line(ln, lang, rng, agreement) for ln in s.lines)
    price = money(s.exw, lang, rng)
    if missing == "price":
        question = (
            f"Fabricamos «{name}» ({code}) en {made_in}. Materiales: {lines}. ¿Es originario "
            f"según {ag}?"
            if es
            else f"We make the {name} ({code}) in {made_in}. Materials: {lines}. Is it originating "
            f"under {ag}?"
        )
        thinking = (
            "To evaluate I need the ex-works price and the user did not give it. I must not "
            "make one up, so I ask before calling anything."
        )
        answer = (
            "Para evaluarlo necesito el precio franco fábrica (EXW) del producto, en euros. "
            "¿Me lo das?"
            if es
            else "To evaluate it I need the product's ex-works (EXW) price, in euros. "
            "Could you give "
            "it to me?"
        )
    elif missing == "materials":
        question = (
            f"¿Es originario nuestro producto «{name}» ({code}) para exportarlo a {partner}? "
            f"Precio franco fábrica: {price}."
            if es
            else f"Is our {name} ({code}) originating for export to {partner}? Ex-works price: "
            f"{price}."
        )
        thinking = (
            "The verdict depends on the materials and the user listed none. I ask for the "
            "bill of materials rather than guess it."
        )
        answer = (
            "Para evaluarlo necesito la lista de materiales: para cada uno, su código SA, su "
            "valor en euros, de dónde viene y si el proveedor da declaración de origen."
            if es
            else "To evaluate it I need the bill of materials: for each one, its HS code, "
            "its value "
            "in euros, where it comes from and whether the supplier gives a declaration of "
            "origin."
        )
    else:
        question = (
            f"Fabricamos «{name}» ({code}) en {made_in}. Precio franco fábrica: {price}. "
            f"Materiales: {lines}. ¿Es originario?"
            if es
            else f"We make the {name} ({code}) in {made_in}. Ex-works price: {price}. Materials: "
            f"{lines}. Is it originating?"
        )
        thinking = (
            "Origin is always under an agreement, and the user did not say where the product "
            "goes. I ask instead of picking one."
        )
        answer = (
            "¿A qué país lo exportáis? El origen preferencial depende del acuerdo: trabajo con "
            "el TCA (Reino Unido), el acuerdo UE-Japón, el acuerdo UE-Corea y el CETA (Canadá)."
            if es
            else "Which country do you export it to? Preferential origin depends on the agreement: "
            "I work with the TCA (UK), the EU-Japan agreement, the EU-Korea agreement and CETA "
            "(Canada)."
        )
    conv.user(spanish(question, lang))
    conv.answer(thinking, answer)
    return t
