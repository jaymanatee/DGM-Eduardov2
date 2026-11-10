"""Synthetic trajectories of lookups: a rule, an exchange rate, a VAT number, several at once,
and the API errors the model has to report instead of filling the gap.

Same principle as ``generate.py``: the calls are written, the real tools run offline, and the
answer is written from what they returned. VIES answers are made up here, with the real
format and fictitious companies; real VAT numbers of real firms have no place in training
data.
"""

from __future__ import annotations

import random
import re
from datetime import date
from typing import Any

from tool_use import tools
from tool_use.sft.catalog import Product
from tool_use.sft.generate import (
    AGREEMENT_NAMES,
    CRITERION_NAMES,
    EU_COUNTRIES,
    LEGAL_FORM,
    MONTHS,
    NAME_PARTS,
    PARTNER,
    SHORT_NAME,
    Conversation,
    Offline,
    Trajectory,
    foreign_amount,
    hs,
    month_text,
    number,
    pick,
    published_months,
    spanish,
)

# ------------------------------------------------------------------------- rules of origin


def describe_criterion(c: dict[str, Any], lang: str) -> str:
    es = lang == "es"
    kind = c["kind"]
    if kind == "MAXNOM":
        basis = "franco fábrica" if c["basis"] == "EXW" and es else c["basis"]
        text = (
            f"materiales no originarios hasta el {c['threshold_pct']} % del precio {basis} (MaxNOM)"
            if es
            else f"non-originating materials up to {c['threshold_pct']} % of the {basis} price "
            "(MaxNOM)"
        )
    elif kind == "RVC":
        text = (
            f"contenido de valor regional de al menos el {c['threshold_pct']} % "
            f"({c['basis']}) (RVC)"
            if es
            else f"regional value content of at least {c['threshold_pct']} % ({c['basis']}) (RVC)"
        )
    else:
        text = CRITERION_NAMES[kind][0 if es else 1]
    if c.get("excluded_headings"):
        headings = ", ".join(c["excluded_headings"])
        text += f", salvo desde {headings}" if es else f", except from {headings}"
    return text


def _clip(text: str, size: int = 280) -> str:
    return text if len(text) <= size else text[: size - 1].rstrip() + "…"


def rule_text(rule: dict[str, Any], lang: str, product: Product | None) -> str:
    """What get_origin_rule returned, told to the user: alternatives, texts, tolerance."""
    es = lang == "es"
    sets = rule["rule_sets"]
    out = []
    for i, rs in enumerate(sets, 1):
        if len(sets) > 1:
            label = tools._subdivision_label(rs["subdivision"]).replace("\n", " ")
            label = re.sub(r"^[-\s]+", "", label)  # the API's '- - Embroidered'
            out.append(f"{i}) {_clip(label, 160)}:")
        if rs["criteria"]:
            alternatives = "; ".join(describe_criterion(c, lang) for c in rs["criteria"])
            out.append(
                ("Basta con cumplir una de estas: " if es else "Any one of these is enough: ")
                + alternatives
                + "."
            )
        if rs["not_machine_readable"]:
            quoted = " / ".join(_clip(t).rstrip(".") for t in rs["not_machine_readable"])
            out.append(
                f"Texto que hay que revisar a mano: «{quoted}»."
                if es
                else f'Text to be checked by hand: "{quoted}".'
            )
        if not rs["criteria"] and not rs["not_machine_readable"]:
            out.append(
                "La API no da ningún texto para esta parte."
                if es
                else "The API gives no text for this part."
            )
    tolerance = rule["general_tolerance_pct"]
    if tolerance:
        out.append(
            f"Tolerancia general: {tolerance} % del precio franco fábrica."
            if es
            else f"General tolerance: {tolerance} % of the ex-works price."
        )
    else:
        out.append(
            "La tolerancia general no se aplica a este producto."
            if es
            else "The general tolerance does not apply to this product."
        )
    if len(sets) > 1:
        choice = product.subdivisions.get(rule["agreement"]) if product else None
        if choice:
            name = product.es if es else product.en
            out.append(
                f"Para «{name}» aplica la subdivisión {choice[0]}."
                if es
                else f"For the {name}, subdivision {choice[0]} applies."
            )
        else:
            out.append(
                "Dime cuál describe tu producto si quieres que lo evalúe."
                if es
                else "Tell me which one describes your product if you want it evaluated."
            )
    return "\n".join(out)


def rule_question(product: Product, agreement: str, lang: str, rng: random.Random) -> str:
    code = hs(product.code, rng, lang)
    name = product.es if lang == "es" else product.en
    ag = pick(rng, AGREEMENT_NAMES[agreement][0 if lang == "es" else 1])
    partner = PARTNER[agreement][0 if lang == "es" else 1]
    if lang == "es":
        text = pick(
            rng,
            (
                f"¿Qué regla de origen tiene {code} en {ag}?",
                f"¿Qué regla de origen se aplica a «{name}» ({code}) si lo exportamos a {partner}?",
                f"Dime la regla de origen de la subpartida {code} según {ag}.",
                f"Vamos a vender «{name}» ({code}) en {partner}. "
                f"¿Qué exige la regla de origen de {ag}?",
            ),
        )
    else:
        text = pick(
            rng,
            (
                f"What is the rule of origin for {code} under {ag}?",
                f"Which rule of origin applies to the {name} ({code}) if we export it to "
                f"{partner}?",
                f"Give me the rule of origin of subheading {code} in {ag}.",
                f"We are going to sell the {name} ({code}) in {partner}. What does {ag} require?",
            ),
        )
    return spanish(text, lang)


def rule_trajectory(
    rng: random.Random, product: Product, agreement: str, today: date, lang: str
) -> Trajectory:
    t = Trajectory("rule", lang, today.isoformat(), product=product.en, agreement=agreement)
    t.seed = rng.getrandbits(32)
    conv = Conversation(t)
    question = rule_question(product, agreement, lang, rng)
    conv.user(question)
    named = (product.es if lang == "es" else product.en) in question
    with Offline(today, random.Random(t.seed)):
        [record] = conv.call(
            pick(
                rng,
                (
                    f"The user wants the rule of origin of HS {product.code} under {agreement}: "
                    "get_origin_rule, nothing to evaluate.",
                    f"A rule lookup: HS {product.code}, agreement {agreement}. The rule comes from "
                    "get_origin_rule, not from memory.",
                ),
            ),
            ("get_origin_rule", {"hs_code": product.code, "agreement": agreement}),
        )
    if not record.ok:
        raise ValueError(record.error)
    rule = record.result
    head = (
        f"Regla de origen de {hs(product.code, rng, lang)} en {SHORT_NAME[agreement][0]}"
        if lang == "es"
        else f"Rule of origin of {hs(product.code, rng, lang)} under {SHORT_NAME[agreement][1]}"
    )
    head += (
        f" ({len(rule['rule_sets'])} subdivisiones):"
        if lang == "es" and len(rule["rule_sets"]) > 1
        else f" ({len(rule['rule_sets'])} subdivisions):"
        if len(rule["rule_sets"]) > 1
        else ":"
    )
    conv.answer(
        "I summarise what get_origin_rule returned: alternatives, texts and tolerance.",
        head + "\n" + rule_text(rule, lang, product if named else None),
    )
    t.outcome = "subdivisions" if len(rule["rule_sets"]) > 1 else "rule"
    return t


# -------------------------------------------------------------------------- exchange rates

COMMON = (
    "USD",
    "GBP",
    "JPY",
    "CNY",
    "KRW",
    "CHF",
    "CAD",
    "TRY",
    "INR",
    "SEK",
    "PLN",
    "MXN",
    "AUD",
    "NOK",
    "CZK",
    "BRL",
    "HUF",
    "DKK",
    "ZAR",
    "SGD",
)
CURRENCY_NAMES = {
    "USD": ("el dólar estadounidense", "the US dollar"),
    "GBP": ("la libra esterlina", "the pound sterling"),
    "JPY": ("el yen", "the Japanese yen"),
    "CNY": ("el yuan", "the Chinese yuan"),
    "KRW": ("el won surcoreano", "the South Korean won"),
    "CHF": ("el franco suizo", "the Swiss franc"),
    "CAD": ("el dólar canadiense", "the Canadian dollar"),
    "TRY": ("la lira turca", "the Turkish lira"),
    "INR": ("la rupia india", "the Indian rupee"),
    "SEK": ("la corona sueca", "the Swedish krona"),
    "PLN": ("el esloti", "the Polish zloty"),
    "MXN": ("el peso mexicano", "the Mexican peso"),
    "AUD": ("el dólar australiano", "the Australian dollar"),
    "NOK": ("la corona noruega", "the Norwegian krone"),
    "CZK": ("la corona checa", "the Czech koruna"),
    "BRL": ("el real brasileño", "the Brazilian real"),
    "HUF": ("el forinto", "the Hungarian forint"),
    "DKK": ("la corona danesa", "the Danish krone"),
    "ZAR": ("el rand", "the South African rand"),
    "SGD": ("el dólar de Singapur", "the Singapore dollar"),
}


def _decimal(value: float, lang: str, digits: int) -> str:
    text = f"{value:.{digits}f}".rstrip("0").rstrip(".")
    return text.replace(".", ",") if lang == "es" else text


def _thousands(value: float, lang: str) -> str:
    text = f"{value:,.2f}"
    return text.replace(",", "X").replace(".", ",").replace("X", ".") if lang == "es" else text


def rate_sentence(result: dict[str, Any], lang: str) -> str:
    """'1 € = 0,856 GBP (1 GBP = 1,168 €)', and the conversion if there was an amount."""
    rate, cur = result["units_per_eur"], result["currency"]
    month = month_text(result["month"], lang)
    both = (
        f"1 € = {_decimal(rate, lang, 4)} {cur} (1 {cur} = {_decimal(1 / rate, lang, 4)} €)"
        if lang == "es"
        else f"EUR 1 = {_decimal(rate, lang, 4)} {cur} "
        f"({cur} 1 = EUR {_decimal(1 / rate, lang, 4)})"
    )
    if "amount_eur" in result:
        amount = foreign_amount(result["amount"], cur, lang)
        euros = _thousands(result["amount_eur"], lang)
        return (
            f"{amount} son {euros} € al tipo medio del BCE de {month}: {both}."
            if lang == "es"
            else f"{amount} is EUR {euros} at the ECB average rate for {month}: {both}."
        )
    return (
        f"Tipo medio del BCE de {month}: {both}."
        if lang == "es"
        else f"ECB average rate for {month}: {both}."
    )


def previous_month(today: date) -> str:
    year, month = (today.year, today.month - 1) if today.month > 1 else (today.year - 1, 12)
    return f"{year:04d}-{month:02d}"


def rate_request(rng: random.Random, today: date, lang: str, currency: str | None = None):
    """A rate question: (text, arguments, what the reasoning says about the month)."""
    currency = currency or pick(rng, COMMON)
    known = sorted(published_months(currency, today))
    form = rng.choice(("convert", "convert", "rate", "latest", "last_month"))
    name = CURRENCY_NAMES[currency][0 if lang == "es" else 1]
    args: dict[str, Any] = {"currency": currency}
    month_note = ""
    if form in ("convert", "rate"):
        month = pick(rng, known[-18:])
        args["month"] = month
        when = month_text(month, lang)
        month_note = f"The month is {month}."
    elif form == "last_month":
        month = previous_month(today)
        args["month"] = month
        when = "el mes pasado" if lang == "es" else "last month"
        month_note = f"Today is {today.isoformat()}, so last month is {month}."
    else:
        when = ""
        month_note = "No month given: I leave it out and get the latest one published."
    if form in ("convert", "last_month"):
        raw = rng.choice((rng.uniform(50, 5000), rng.uniform(5000, 300000)))
        amount = round(raw * published_months(currency, today)[known[-1]] / 100) * 100 or 100
        args["amount"] = number(amount)
        amt = foreign_amount(amount, currency, lang)
        text = (
            pick(
                rng,
                (
                    f"¿Cuántos euros son {amt} al tipo medio del BCE de {when}?",
                    f"Convierte {amt} a euros con el tipo medio del BCE de {when}.",
                    f"Pagamos {amt} a un proveedor {when}. ¿Cuánto es en euros según el BCE?",
                ),
            )
            if lang == "es"
            else pick(
                rng,
                (
                    f"How many euros is {amt} at the ECB monthly average rate for {when}?",
                    f"Convert {amt} to euros at the ECB average rate of {when}.",
                    f"We paid a supplier {amt} {'in ' if form == 'convert' else ''}{when}. "
                    "What is that in euros (ECB)?",
                ),
            )
        )
    elif form == "rate":
        text = (
            pick(
                rng,
                (
                    f"¿A cuánto estuvo de media {name} frente al euro en {when}, según el BCE?",
                    f"Tipo de cambio medio del BCE de {currency} en {when}.",
                ),
            )
            if lang == "es"
            else pick(
                rng,
                (
                    f"What was the ECB monthly average rate of {name} against the euro in {when}?",
                    f"ECB average {currency} rate for {when}, please.",
                ),
            )
        )
    else:
        text = (
            pick(
                rng,
                (
                    f"¿Cuál es el último tipo de cambio medio mensual del BCE para {name}?",
                    f"Dame el tipo medio más reciente del BCE de {currency} frente al euro.",
                ),
            )
            if lang == "es"
            else pick(
                rng,
                (
                    f"What is the latest ECB monthly average rate for {name}?",
                    f"Give me the most recent ECB average rate of {currency} against the euro.",
                ),
            )
        )
    return text, args, month_note


def rate_trajectory(rng: random.Random, today: date, lang: str) -> Trajectory:
    t = Trajectory("rate", lang, today.isoformat())
    t.seed = rng.getrandbits(32)
    conv = Conversation(t)
    text, args, month_note = rate_request(rng, today, lang)
    conv.user(text)
    with Offline(today, random.Random(t.seed)):
        [record] = conv.call(
            f"An ECB rate for {args['currency']}: get_exchange_rate. {month_note}"
            + (" I pass the amount so the tool converts it." if "amount" in args else ""),
            ("get_exchange_rate", args),
        )
    if not record.ok:
        raise ValueError(record.error)
    conv.answer(
        "I give the rate the tool returned, both ways, and the conversion if any.",
        rate_sentence(record.result, lang),
    )
    t.outcome = "converted" if "amount" in args else "rate"
    return t


# ------------------------------------------------------------------------------------- VIES

VAT_FORMATS = {
    "ES": lambda r: pick(r, "ABCDEFGHJ") + f"{r.randint(0, 99_999_999):08d}",
    "DE": lambda r: f"{r.randint(100_000_000, 999_999_999)}",
    "FR": lambda r: f"{r.randint(10, 99)}{r.randint(100_000_000, 999_999_999)}",
    "IT": lambda r: f"{r.randint(0, 99_999_999_999):011d}",
    "PT": lambda r: f"{r.randint(500_000_000, 599_999_999)}",
    "NL": lambda r: f"{r.randint(100_000_000, 999_999_999)}B{r.randint(1, 99):02d}",
    "PL": lambda r: f"{r.randint(1_000_000_000, 9_999_999_999)}",
    "CZ": lambda r: f"{r.randint(10_000_000, 99_999_999)}",
    "BE": lambda r: f"0{r.randint(100_000_000, 999_999_999)}",
    "AT": lambda r: f"U{r.randint(10_000_000, 99_999_999)}",
    "EL": lambda r: f"{r.randint(100_000_000, 999_999_999)}",
    "XI": lambda r: f"{r.randint(100_000_000, 999_999_999)}",
}
VAT_COUNTRY = {
    "ES": ("España", "Spain"),
    "DE": ("Alemania", "Germany"),
    "FR": ("Francia", "France"),
    "IT": ("Italia", "Italy"),
    "PT": ("Portugal", "Portugal"),
    "NL": ("los Países Bajos", "the Netherlands"),
    "PL": ("Polonia", "Poland"),
    "CZ": ("Chequia", "Czechia"),
    "BE": ("Bélgica", "Belgium"),
    "AT": ("Austria", "Austria"),
    "EL": ("Grecia", "Greece"),
    "XI": ("Irlanda del Norte", "Northern Ireland"),
}
STREETS = {
    "ES": ("Calle Mayor", "Avenida del Puerto"),
    "DE": ("Hauptstrasse", "Industriestrasse"),
    "FR": ("Rue de la Gare", "Avenue Jean Jaurès"),
    "IT": ("Via Roma", "Via dell'Industria"),
    "PT": ("Rua do Comércio", "Avenida da Liberdade"),
    "NL": ("Industrieweg", "Stationsstraat"),
    "PL": ("ul. Polna", "ul. Przemysłowa"),
    "CZ": ("Nádražní", "Průmyslová"),
}


def written_vat(vat: str, rng: random.Random) -> str:
    """As people write it: 'ESB12345678', 'es b12345678', 'FR 12 345 678 901'..."""
    form = rng.random()
    if form < 0.5:
        return vat
    if form < 0.75:
        return vat[:2] + " " + " ".join(vat[i : i + 3] for i in range(2, len(vat), 3))
    return vat[:2].lower() + " " + vat[2:]


def vies_case(rng: random.Random, today: date) -> tuple[str, dict | None, str]:
    """A VAT number, VIES's answer (None for British numbers) and the outcome."""
    outcome = rng.choices(
        ("valid", "withheld", "invalid", "busy", "british"), (45, 10, 25, 10, 10)
    )[0]
    if outcome == "british":
        return f"GB{rng.randint(100_000_000, 999_999_999)}", None, outcome
    country = pick(rng, list(VAT_FORMATS))
    vat = country + VAT_FORMATS[country](rng)
    stamp = f"{today.isoformat()}T{rng.randint(8, 18):02d}:{rng.randint(0, 59):02d}:00.000Z"
    body: dict[str, Any] = {
        "isValid": outcome in ("valid", "withheld"),
        "requestDate": stamp,
        "userError": "VALID" if outcome in ("valid", "withheld") else "INVALID",
        "name": "---",
        "address": "---",
        "vatNumber": vat[2:],
        "countryCode": country,
    }
    if outcome == "valid":
        code = country if country in STREETS else pick(rng, list(STREETS))
        body["name"] = (
            f"{pick(rng, NAME_PARTS[0])} {pick(rng, NAME_PARTS[1])} {LEGAL_FORM[code]}".upper()
        )
        city = pick(rng, EU_COUNTRIES[code][2]).upper()
        street = pick(rng, STREETS[code]).upper()
        body["address"] = f"{street} {rng.randint(1, 120)}, {rng.randint(10000, 99999)} {city}"
    if outcome == "busy":
        body = {"isValid": False, "userError": "MS_MAX_CONCURRENT_REQ", "requestDate": stamp}
    return vat, body, outcome


def vat_answer(vat: str, record: Any, lang: str) -> str:
    es = lang == "es"
    if not record.ok:
        if "British" in (record.error or ""):
            return (
                "No se puede comprobar en VIES: los números de IVA británicos dejaron de estar en "
                "VIES en 2021. Para un número del Reino Unido hay que usar el servicio del HMRC."
                if es
                else "It cannot be checked in VIES: British VAT numbers left VIES in 2021. "
                "A UK number "
                "has to be checked with HMRC's service."
            )
        return (
            f"No he podido comprobar {vat}: VIES no respondió ({record.error.split(': ')[-1]}, "
            "demasiadas consultas a la vez). Prueba de nuevo dentro de un rato."
            if es
            else f"I could not check {vat}: VIES did not answer "
            f"({record.error.split(': ')[-1]}, too "
            "many requests at once). Try again in a while."
        )
    r = record.result
    if not r["valid"]:
        return (
            f"No: según VIES, {r['vat_number']} no es un NIF-IVA válido."
            if es
            else f"No: according to VIES, {r['vat_number']} is not a valid VAT number."
        )
    if r["name"] is None:
        country = VAT_COUNTRY[r["vat_number"][:2]][0 if es else 1]
        return (
            f"Sí, {r['vat_number']} es un NIF-IVA válido según VIES, aunque {country} no comunica "
            "el nombre ni la dirección del titular."
            if es
            else f"Yes, {r['vat_number']} is a valid VAT number according to VIES, although "
            f"{country} does not share the holder's name or address."
        )
    return (
        f"Sí, {r['vat_number']} es un NIF-IVA válido según VIES. Titular: {r['name']}, "
        f"{r['address']}."
        if es
        else f"Yes, {r['vat_number']} is a valid VAT number according to VIES. "
        f"Holder: {r['name']}, "
        f"{r['address']}."
    )


def vat_request(rng: random.Random, vat: str, lang: str) -> str:
    shown = written_vat(vat, rng)
    if lang == "es":
        return pick(
            rng,
            (
                f"¿Es válido el NIF-IVA {shown}?",
                f"Antes de facturar, comprueba en VIES el número de IVA {shown}.",
                f"Comprueba el NIF-IVA {shown} y dime a nombre de quién está.",
                f"Un cliente nos da el número de IVA {shown}. ¿Está dado de alta?",
            ),
        )
    return pick(
        rng,
        (
            f"Is {shown} a valid EU VAT number?",
            f"Before invoicing, check the VAT number {shown} in VIES.",
            f"Check VAT number {shown} and tell me who it belongs to.",
            f"A customer gave us the VAT number {shown}. Is it registered?",
        ),
    )


def vat_trajectory(rng: random.Random, today: date, lang: str) -> Trajectory:
    t = Trajectory("vat", lang, today.isoformat())
    t.seed = rng.getrandbits(32)
    conv = Conversation(t)
    vat, body, outcome = vies_case(rng, today)
    conv.user(vat_request(rng, vat, lang))
    t.vies = {vat: body} if body else {}
    with Offline(today, random.Random(t.seed), vies=t.vies):
        [record] = conv.call(
            f"A VAT number the user wrote: {vat}. check_vat_number with it, as written.",
            ("check_vat_number", {"vat_number": vat}),
        )
    conv.answer(
        "I report what VIES answered, or why it could not be checked.",
        vat_answer(vat, record, lang),
    )
    t.outcome = outcome
    return t


# ----------------------------------------------------------------------- parallel lookups


def parallel_trajectory(
    rng: random.Random, products: list[Product], today: date, lang: str
) -> Trajectory:
    """Two independent lookups in one message: rates, a rate and a VAT number, two VAT
    numbers, or one product's rule under two agreements."""
    form = rng.choice(("two_rates", "rate_vat", "two_vats", "two_rules"))
    t = Trajectory("parallel", lang, today.isoformat(), outcome=form)
    t.seed = rng.getrandbits(32)
    conv = Conversation(t)
    vies: dict[str, dict] = {}
    thinking = pick(
        rng,
        (
            "Two independent lookups: both calls in one message.",
            "Neither lookup needs the other's result, so I make both calls at once.",
        ),
    )
    joiner = " Además, " if lang == "es" else " Also, "
    if form == "two_rates":
        first, second = rng.sample(COMMON, 2)
        month = pick(
            rng,
            sorted(set(published_months(first, today)) & set(published_months(second, today)))[
                -12:
            ],
        )
        when = month_text(month, lang)
        names = [CURRENCY_NAMES[c][0 if lang == "es" else 1] for c in (first, second)]
        text = (
            f"¿Cuáles fueron los tipos medios del BCE de {names[0]} y {names[1]} en {when}?"
            if lang == "es"
            else f"What were the ECB average rates of {names[0]} and {names[1]} in {when}?"
        )
        calls = [("get_exchange_rate", {"currency": c, "month": month}) for c in (first, second)]
    elif form == "rate_vat":
        rate_text, rate_args, _ = rate_request(rng, today, lang)
        vat, body, _ = vies_case(rng, today)
        if body:
            vies[vat] = body
        vat_text = vat_request(rng, vat, lang)
        text = rate_text + joiner + vat_text[0].lower() + vat_text[1:]
        calls = [("get_exchange_rate", rate_args), ("check_vat_number", {"vat_number": vat})]
    elif form == "two_vats":
        cases = [vies_case(rng, today) for _ in range(2)]
        for vat, body, _ in cases:
            if body:
                vies[vat] = body
        shown = [written_vat(v, rng) for v, _, _ in cases]
        text = (
            f"Comprueba en VIES estos dos NIF-IVA: {shown[0]} y {shown[1]}."
            if lang == "es"
            else f"Check these two VAT numbers in VIES: {shown[0]} and {shown[1]}."
        )
        calls = [("check_vat_number", {"vat_number": v}) for v, _, _ in cases]
    else:
        product = pick(rng, products)
        first, second = rng.sample(list(tools.AGREEMENTS), 2)
        code = hs(product.code, rng, lang)
        names = [pick(rng, AGREEMENT_NAMES[a][0 if lang == "es" else 1]) for a in (first, second)]
        text = (
            f"Dame la regla de origen de {code} en {names[0]} y en {names[1]}."
            if lang == "es"
            else f"Give me the rule of origin of {code} under {names[0]} and under {names[1]}."
        )
        calls = [
            ("get_origin_rule", {"hs_code": product.code, "agreement": a}) for a in (first, second)
        ]
        t.product = product.en
    conv.user(spanish(text, lang))
    t.vies = vies
    with Offline(today, random.Random(t.seed), vies=vies):
        records = conv.call(thinking, *calls)
    parts = []
    for (name, args), record in zip(calls, records, strict=True):
        if name == "get_exchange_rate":
            parts.append(rate_sentence(record.result, lang) if record.ok else record.error)
        elif name == "check_vat_number":
            parts.append(vat_answer(args["vat_number"], record, lang))
        else:
            head = SHORT_NAME[args["agreement"]][0 if lang == "es" else 1]
            parts.append(f"{head[0].upper() + head[1:]}:\n" + rule_text(record.result, lang, None))
    conv.answer("Both results, one after the other, as the tools gave them.", "\n\n".join(parts))
    return t


# ---------------------------------------------------------------------------- API errors

NOT_IN_ECB = (
    ("ARS", "el peso argentino", "the Argentine peso"),
    ("VND", "el dong vietnamita", "the Vietnamese dong"),
    ("AED", "el dírham de los Emiratos", "the UAE dirham"),
    ("CLP", "el peso chileno", "the Chilean peso"),
    ("EGP", "la libra egipcia", "the Egyptian pound"),
    ("TWD", "el dólar taiwanés", "the Taiwan dollar"),
    ("SAR", "el riyal saudí", "the Saudi riyal"),
)


def error_trajectory(rng: random.Random, today: date, lang: str) -> Trajectory:
    """The month is not published yet, or the ECB has no rate for the currency: the model
    says so, and does not make up a rate."""
    form = rng.choice(("unpublished", "unpublished", "no_ecb_rate"))
    t = Trajectory("api_error", lang, today.isoformat(), outcome=form)
    t.seed = rng.getrandbits(32)
    conv = Conversation(t)
    if form == "unpublished":
        currency = pick(rng, COMMON[:8])
        month = f"{today.year:04d}-{today.month:02d}"
        name = CURRENCY_NAMES[currency][0 if lang == "es" else 1]
        when = month_text(month, lang)
        conv.user(
            pick(
                rng,
                (
                    f"¿Cuál es el tipo medio del BCE de {name} en {when}?",
                    f"Convierte 2.500 {currency} a euros con la media del BCE de este mes.",
                ),
            )
            if lang == "es"
            else pick(
                rng,
                (
                    f"What is the ECB average rate of {name} for {when}?",
                    f"Convert {currency} 2,500 to euros at this month's ECB average.",
                ),
            )
        )
        args: dict[str, Any] = {"currency": currency, "month": month}
        if "2.500" in t.messages[0]["content"] or "2,500" in t.messages[0]["content"]:
            args["amount"] = 2500
        thinking = f"Today is {today.isoformat()}, so this month is {month}: get_exchange_rate."
    else:
        code, es_name, en_name = pick(rng, NOT_IN_ECB)
        conv.user(
            f"¿Cuántos euros son 10.000 {code} según el BCE?"
            if lang == "es"
            else f"How many euros are {code} 10,000 at the ECB rate?"
        )
        args = {"currency": code, "amount": 10000}
        thinking = f"A conversion of {code} to euros: get_exchange_rate with the amount."
    with Offline(today, random.Random(t.seed)):
        [record] = conv.call(thinking, ("get_exchange_rate", args))
    if record.ok:
        raise ValueError("expected an error")
    if form == "unpublished":
        latest = previous_month(today)
        conv.answer(
            "The tool says the month is not over, so there is no average yet. I say so and offer "
            "the latest month; I do not make up a rate.",
            f"El BCE todavía no ha publicado la media de {month_text(args['month'], lang)}: "
            "las medias "
            f"mensuales salen cuando termina el mes. La última disponible es la de "
            f"{month_text(latest, lang)}; si quieres, uso esa."
            if lang == "es"
            else f"The ECB has not published the average for {month_text(args['month'], lang)} "
            "yet: monthly "
            f"averages come out once the month is over. The latest one is "
            f"{month_text(latest, lang)}; I can use that if you like.",
        )
    else:
        conv.answer(
            "The ECB publishes no reference rate for this currency: I say so, with no rate of "
            "my own.",
            f"El BCE no publica tipo de referencia para {es_name} ({code}), así que no puedo "
            "hacer la conversión con su tipo oficial."
            if lang == "es"
            else f"The ECB publishes no reference rate for {en_name} ({code}), so I cannot convert "
            "it at an official ECB rate.",
        )
    return t


__all__ = [
    "MONTHS",
    "error_trajectory",
    "parallel_trajectory",
    "rate_trajectory",
    "rule_trajectory",
    "vat_trajectory",
]
