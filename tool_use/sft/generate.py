"""Synthetic tool-use trajectories for SFT: the conversations the tool loop should produce.

A trajectory is one whole conversation: the user's question, every assistant turn (reasoning
plus tool calls, or the final answer) and every tool result in between. Nothing here decides
whether a product is originating: the scenario is sampled, the right calls are written, and
the real tools run on them (rules and ECB rates from disk, see ``fetch.py``). Whatever they
return goes into the conversation as is, and the next turn is written from it. So the model
is never taught a verdict, a rate or an id that a tool did not give.

Each assistant turn keeps its reasoning apart from its text, so the same trajectories give
the two SFT variants: with a short ``<think>`` and with an empty one.

This module covers the trajectories where the mistakes of the benchmark were: evaluating a
product (with subdivisions, FOB, foreign currencies) and issuing a statement on origin.

Run (a few examples, to read)::

    uv run python -m tool_use.sft.generate --sample 6
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import os
import random
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any
from unittest import mock

from tool_use import tools
from tool_use.executor import ToolLoop
from tool_use.registry import ToolCallRecord
from tool_use.sft.catalog import PRODUCTS, Material, Product
from tool_use.sft.fetch import ecb_path, load_rule

# ------------------------------------------------------------------------------ vocabulary

EU_COUNTRIES = {  # ISO code: (es, en, cities)
    "ES": ("España", "Spain", ("Valladolid", "Sevilla", "Vigo", "Murcia", "Pamplona", "Girona")),
    "DE": ("Alemania", "Germany", ("Stuttgart", "Leipzig", "Dortmund", "Nuremberg", "Bremen")),
    "FR": ("Francia", "France", ("Lyon", "Toulouse", "Lille", "Nantes", "Grenoble")),
    "IT": ("Italia", "Italy", ("Bolonia", "Turín", "Brescia", "Verona", "Padua")),
    "PT": ("Portugal", "Portugal", ("Braga", "Aveiro", "Coímbra", "Leiria")),
    "NL": ("Países Bajos", "the Netherlands", ("Eindhoven", "Utrecht", "Tilburg")),
    "PL": ("Polonia", "Poland", ("Poznan", "Wroclaw", "Lodz")),
    "CZ": ("Chequia", "Czechia", ("Brno", "Ostrava", "Pilsen")),
}
LEGAL_FORM = {
    "ES": "SL",
    "DE": "GmbH",
    "FR": "SAS",
    "IT": "Srl",
    "PT": "Lda",
    "NL": "BV",
    "PL": "sp. z o.o.",
    "CZ": "s.r.o.",
}
NAME_PARTS = (
    ("Talleres", "Industrias", "Fabricados", "Componentes", "Manufacturas", "Grupo", "Técnicas"),
    (
        "Albarrán",
        "Montesur",
        "Riverola",
        "Castellet",
        "Norvalle",
        "Ubeira",
        "Larrinaga",
        "Brenner",
        "Kessler",
        "Moreau",
        "Ferrante",
        "Vasconcelos",
        "Kowalik",
        "Dvorak",
    ),
)
THIRD_COUNTRIES = (
    ("China", "China"),
    ("India", "India"),
    ("Turquía", "Turkey"),
    ("Estados Unidos", "the US"),
    ("Vietnam", "Vietnam"),
    ("México", "Mexico"),
    ("Marruecos", "Morocco"),
    ("Tailandia", "Thailand"),
    ("Suiza", "Switzerland"),
    ("Noruega", "Norway"),
)
PARTNER = {
    "TCA": ("el Reino Unido", "the UK"),
    "UE-JP": ("Japón", "Japan"),
    "UE-KR": ("Corea del Sur", "South Korea"),
    "CETA": ("Canadá", "Canada"),
}
# Partners of the other agreements are third countries here: a good trap.
OTHER_PARTNERS = {a: [p for b, p in PARTNER.items() if b != a] for a in PARTNER}
AGREEMENT_NAMES = {
    "TCA": (
        (
            "el Acuerdo de Comercio y Cooperación UE-Reino Unido",
            "el TCA",
            "el acuerdo con el Reino Unido",
        ),
        ("the EU-UK Trade and Cooperation Agreement", "the TCA", "the agreement with the UK"),
    ),
    "UE-JP": (
        ("el Acuerdo de Asociación Económica UE-Japón", "el acuerdo UE-Japón", "el EPA con Japón"),
        ("the EU-Japan Economic Partnership Agreement", "the EU-Japan EPA", "the EPA with Japan"),
    ),
    "UE-KR": (
        (
            "el Acuerdo de Libre Comercio UE-Corea",
            "el acuerdo UE-Corea",
            "el TLC con Corea del Sur",
        ),
        ("the EU-Korea Free Trade Agreement", "the EU-Korea FTA", "the FTA with South Korea"),
    ),
    "CETA": (
        ("el CETA", "el acuerdo UE-Canadá", "el Acuerdo Económico y Comercial Global UE-Canadá"),
        ("CETA", "the EU-Canada agreement", "the EU-Canada CETA"),
    ),
}
SHORT_NAME = {
    "TCA": ("el TCA", "the TCA"),
    "UE-JP": ("el acuerdo UE-Japón", "the EU-Japan EPA"),
    "UE-KR": ("el acuerdo UE-Corea", "the EU-Korea FTA"),
    "CETA": ("el CETA", "CETA"),
}
MONTHS = (
    ("enero", "January"),
    ("febrero", "February"),
    ("marzo", "March"),
    ("abril", "April"),
    ("mayo", "May"),
    ("junio", "June"),
    ("julio", "July"),
    ("agosto", "August"),
    ("septiembre", "September"),
    ("octubre", "October"),
    ("noviembre", "November"),
    ("diciembre", "December"),
)
CURRENCY_OF = {
    "the US": "USD", "the UK": "GBP", "Japan": "JPY", "China": "CNY", "South Korea": "KRW",
    "Switzerland": "CHF", "Turkey": "TRY", "India": "INR", "Canada": "CAD", "Mexico": "MXN",
    "Thailand": "THB", "Norway": "NOK",
}  # fmt: skip
CRITERION_NAMES = {
    "CTH": ("el cambio de partida (CTH)", "the change of tariff heading (CTH)"),
    "CTSH": ("el cambio de subpartida (CTSH)", "the change of tariff subheading (CTSH)"),
    "CC": ("el cambio de capítulo (CC)", "the change of chapter (CC)"),
    "MAXNOM": (
        "el límite de materiales no originarios (MaxNOM)",
        "the limit on non-originating materials (MaxNOM)",
    ),
    "RVC": ("el contenido de valor regional (RVC)", "the regional value content (RVC)"),
    "WO": (
        "el criterio de producto enteramente obtenido (WO)",
        "the wholly obtained criterion (WO)",
    ),
}
EXPORTER_THRESHOLD = tools.EXPORTER_REFERENCE_THRESHOLD_EUR


def pick(rng: random.Random, options):
    return options[rng.randrange(len(options))]


def spanish(text: str, lang: str) -> str:
    """'a el Reino Unido' -> 'al Reino Unido', 'de el TCA' -> 'del TCA'."""
    if lang != "es":
        return text
    return text.replace(" a el ", " al ").replace(" de el ", " del ").replace("De el ", "Del ")


# ------------------------------------------------------------------------------ formatting


def money(value: float, lang: str, rng: random.Random) -> str:
    """An amount in euros as people write it: '1.234,50 €', 'EUR 1,234.50'..."""
    whole = float(value).is_integer()
    if lang == "es":
        number = f"{value:,.0f}" if whole else f"{value:,.2f}"
        number = number.replace(",", "X").replace(".", ",").replace("X", ".")
        if rng.random() < 0.3:
            number = number.replace(".", "")
        return pick(rng, (f"{number} €", f"{number} €", f"{number} EUR", f"{number} euros"))
    number = f"{value:,.0f}" if whole else f"{value:,.2f}"
    return pick(rng, (f"EUR {number}", f"{number} EUR", f"€{number}", f"{number} euros"))


def foreign_amount(amount: float, currency: str, lang: str) -> str:
    number = f"{amount:,.0f}"
    if lang == "es":
        number = number.replace(",", ".")
    return f"{number} {currency}"


def hs(code: str, rng: random.Random, lang: str) -> str:
    """'8418.10', '841810', 'SA 8418.10', 'HS 8418 10'..."""
    dotted = f"{code[:4]}.{code[4:]}"
    prefix = "SA" if lang == "es" else "HS"
    return pick(
        rng, (dotted, dotted, code, f"{prefix} {dotted}", f"{prefix} {code[:4]} {code[4:]}")
    )


def month_text(month: str, lang: str) -> str:
    year, number = month.split("-")
    name = MONTHS[int(number) - 1][0 if lang == "es" else 1]
    return f"{name} de {year}" if lang == "es" else f"{name} {year}"


def number(value: float) -> float | int:
    """How the model writes a number in a call: 250, not 250.0."""
    return int(value) if float(value).is_integer() else round(value, 2)


# ------------------------------------------------------------------------- offline tools


def published_months(currency: str, today: date) -> dict[str, float]:
    """Monthly ECB averages known on ``today``: a month is published once it is over."""
    current = f"{today.year:04d}-{today.month:02d}"
    rows = csv.DictReader(io.StringIO(ecb_path(currency).read_text(encoding="utf-8")))
    return {r["TIME_PERIOD"]: float(r["OBS_VALUE"]) for r in rows if r["TIME_PERIOD"] < current}


class _Response:
    def __init__(self, text: str = "", payload: Any = None):
        self.text, self.payload, self.status_code = text, payload, 200

    def json(self) -> Any:
        return self.payload

    def raise_for_status(self) -> None:
        return None


class Offline:
    """The five tools, without network, on a given day.

    Rules come from ``data/roo``, ECB rates from ``data/ecb`` (only the months published on
    ``today``), VIES answers from ``vies`` (VAT number -> body), evaluation ids from ``rng``.
    Statements go to ``outputs/origin_statements`` inside a temporary directory, so the path
    the tool returns looks as it does in production.
    """

    def __init__(self, today: date, rng: random.Random, vies: dict[str, dict] | None = None):
        self.today, self.rng, self.vies = today, rng, vies or {}
        self._stack = contextlib.ExitStack()

    def _http_get(self, url: str, params: dict[str, Any] | None = None) -> _Response:
        if "data-api.ecb.europa.eu" in url:
            currency = url.split("/M.")[1].split(".")[0]
            text = ecb_path(currency).read_text(encoding="utf-8")
            header, *rows = text.splitlines()
            known = published_months(currency, self.today)
            rows = [r for r in rows if r.split(",")[6] in known]
            if params and "startPeriod" in params:
                rows = [r for r in rows if r.split(",")[6] == params["startPeriod"]]
            elif rows:
                rows = rows[-1:]
            return _Response("\n".join([header, *rows]) + "\n")
        if "vies" in url:
            country, number = url.rstrip("/").split("/ms/")[1].split("/vat/")
            return _Response(payload=self.vies[country + number])
        raise AssertionError(f"no offline answer for {url}")

    def __enter__(self) -> Offline:
        today = self.today

        class Today(date):
            @classmethod
            def today(cls) -> date:  # type: ignore[override]
                return today

        workdir = self._stack.enter_context(tempfile.TemporaryDirectory())
        self._stack.enter_context(contextlib.chdir(workdir))
        patches = (
            mock.patch.object(tools, "_rule_of_origin", lambda c, a: load_rule(c, a, today)),
            mock.patch.object(tools, "_http_get", self._http_get),
            mock.patch.object(tools, "_EVALUATIONS", {}),  # each conversation its own
            mock.patch.object(tools, "date", Today),
            mock.patch.object(tools.time, "sleep", lambda s: None),
            mock.patch.object(tools.secrets, "token_hex", lambda n: self.rng.randbytes(n).hex()),
            mock.patch.dict(os.environ, {"ARCA_STATEMENTS_DIR": "outputs/origin_statements"}),
        )
        for patch in patches:
            self._stack.enter_context(patch)
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stack.close()


# --------------------------------------------------------------------------- conversation


@dataclass
class Trajectory:
    kind: str
    lang: str
    today: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    product: str = ""
    agreement: str = ""
    outcome: str = ""
    seed: int = 0  # of the evaluation ids, so that replay() gets the same ones

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


class Conversation:
    """Writes the turns and runs the calls with the real registry, as ToolLoop does."""

    def __init__(self, trajectory: Trajectory):
        self.t = trajectory
        self.registry = tools.default_registry()

    def user(self, text: str) -> None:
        self.t.messages.append({"role": "user", "content": text})

    def call(self, thinking: str, *calls: tuple[str, dict[str, Any]]) -> list[ToolCallRecord]:
        content = "\n".join(
            "<tool_call>\n"
            + json.dumps({"name": name, "arguments": args}, ensure_ascii=False)
            + "\n</tool_call>"
            for name, args in calls
        )
        self.t.messages.append({"role": "assistant", "thinking": thinking, "content": content})
        records = [self.registry.call(name, args) for name, args in calls]
        self.t.messages += [ToolLoop.observation(r) for r in records]
        return records

    def answer(self, thinking: str, text: str) -> None:
        text = spanish(text, self.t.lang)
        self.t.messages.append({"role": "assistant", "thinking": thinking, "content": text})


# ------------------------------------------------------------------------------ scenarios


@dataclass
class Line:
    """One material of the bill as the user describes it, and what the call should say."""

    material: Material
    origin: str  # UE, SOCIO, TERCERO, DESCONOCIDO
    country: tuple[str, str] | None  # (es, en); None if unknown
    declared: bool
    value_eur: float
    currency: str | None = None  # paid in this currency instead of euros
    amount: float | None = None
    month: str | None = None


@dataclass
class Statement:
    invoice: str
    exporter: str
    place: str
    value_eur: float
    reference: str  # '' if the user gave none


@dataclass
class Scenario:
    product: Product
    agreement: str
    exw: float
    fob: float | None
    lines: list[Line]
    operations: list[str] | None
    exporter_country: str
    statement: Statement | None = None


def sample_lines(rng: random.Random, product: Product, agreement: str, exw: float) -> list[Line]:
    materials = rng.sample(product.materials, rng.randint(2, len(product.materials)))
    share = rng.uniform(0.15, 0.9)  # of the ex-works price, spread over the materials
    weights = [rng.uniform(0.2, 1.0) for _ in materials]
    total = sum(weights)
    lines = []
    for material, weight in zip(materials, weights, strict=True):
        value = exw * share * weight / total
        value = round(value) if value > 50 and rng.random() < 0.6 else round(value, 2)
        value = max(value, 0.5)
        origin = rng.choices(("UE", "SOCIO", "TERCERO", "DESCONOCIDO"), (36, 12, 42, 10))[0]
        if origin == "UE":
            code = pick(rng, list(EU_COUNTRIES))
            country = EU_COUNTRIES[code][:2] if rng.random() < 0.6 else ("la UE", "the EU")
        elif origin == "SOCIO":
            country = PARTNER[agreement]
        elif origin == "TERCERO":
            pool = list(THIRD_COUNTRIES) + OTHER_PARTNERS[agreement]
            country = pick(rng, pool)
        else:
            country = None
        declared = origin in ("UE", "SOCIO") and rng.random() < 0.8
        lines.append(Line(material, origin, country, declared, value))
    return lines


def sample_scenario(
    rng: random.Random, product: Product, agreement: str, today: date, *, currency: bool = False
) -> Scenario:
    low, high = product.exw_range
    exw = rng.uniform(low, high)
    exw = round(exw, -1) if exw > 100 and rng.random() < 0.7 else round(exw, 2)
    fob = round(exw * rng.uniform(1.02, 1.15), 2) if rng.random() < 0.35 else None
    lines = sample_lines(rng, product, agreement, exw)
    if currency:
        # Paid in the supplier's currency (US dollars, yen...), or in dollars otherwise.
        priced = [ln for ln in lines if ln.country and ln.country[1] in CURRENCY_OF]
        foreign = priced or [ln for ln in lines if ln.origin == "TERCERO"] or lines[:1]
        for line in foreign[: rng.choice((1, 1, 2))]:
            line.currency = CURRENCY_OF.get(line.country[1] if line.country else "", "USD")
            months = sorted(published_months(line.currency, today))[-14:]
            line.month = pick(rng, months)
            rate = published_months(line.currency, today)[line.month]
            raw = line.value_eur * rate
            step = 1000 if raw > 20000 else 10 if raw > 200 else 1
            line.amount = max(step, round(raw / step) * step)
    operations = None
    if rng.random() < 0.06:
        operations = rng.sample(["embalaje", "etiquetado", "limpieza"], rng.randint(1, 2))
    return Scenario(product, agreement, exw, fob, lines, operations, pick(rng, list(EU_COUNTRIES)))


def sample_statement(rng: random.Random, scenario: Scenario, lang: str) -> Statement:
    code = scenario.exporter_country
    year = rng.choice((2025, 2026))
    invoice = pick(
        rng,
        (
            f"F/{year}/{rng.randint(1, 999):03d}",
            f"FV-{year % 100}-{rng.randint(1, 9999):04d}",
            f"{year}-{rng.randint(1, 9999):04d}",
            f"INV{rng.randint(10000, 99999)}",
            f"A{rng.randint(100, 999)}/{year % 100}",
        ),
    )
    first, second = pick(rng, NAME_PARTS[0]), pick(rng, NAME_PARTS[1])
    exporter = f"{first} {second} {LEGAL_FORM[code]}"
    place = pick(rng, EU_COUNTRIES[code][2])
    value = round(rng.choice((rng.uniform(300, 5900), rng.uniform(6100, 60000))), -1)
    reference = ""
    if rng.random() < 0.55:
        if scenario.agreement == "UE-KR":
            reference = pick(
                rng,
                (f"{code}/{rng.randint(1, 9999):04d}/AE", f"{code}-AE-{rng.randint(10000, 99999)}"),
            )
        else:
            letters = "ABCDEFGHJKLMNPRSTUVWXYZ"
            reference = f"{code}REX{pick(rng, letters)}{rng.randint(10_000_000, 99_999_999)}"
    return Statement(invoice, exporter, place, value, reference)


# ----------------------------------------------------------------------------- user text


def describe_line(line: Line, lang: str, rng: random.Random, agreement: str) -> str:
    name = line.material.es if lang == "es" else line.material.en
    code = hs(line.material.code, rng, lang)
    if line.currency:
        price = foreign_amount(line.amount, line.currency, lang)
        when = (
            f"pagado en {month_text(line.month, lang)}"
            if lang == "es"
            else f"paid in {month_text(line.month, lang)}"
        )
        price = f"{price} ({when})"
    else:
        price = money(line.value_eur, lang, rng)
    if line.country is None:
        where = (
            pick(rng, ("de origen desconocido", "sin origen conocido"))
            if lang == "es"
            else (pick(rng, ("of unknown origin", "origin unknown")))
        )
    else:
        country = line.country[0] if lang == "es" else line.country[1]
        made = (
            pick(rng, ("de", "procedente de", "con origen en"))
            if lang == "es"
            else pick(rng, ("from", "made in", "sourced from"))
        )
        where = f"{made} {country}"
    declaration = ""
    if line.origin in ("UE", "SOCIO"):
        if line.declared:
            declaration = (
                pick(
                    rng,
                    (
                        ", con declaración del proveedor",
                        ", con declaración de origen del proveedor",
                    ),
                )
                if lang == "es"
                else pick(
                    rng, (", with a supplier's declaration", ", supplier's declaration available")
                )
            )
        else:
            declaration = (
                pick(
                    rng, (", sin declaración del proveedor", ", el proveedor no nos da declaración")
                )
                if lang == "es"
                else pick(
                    rng, (", no supplier's declaration", ", without a supplier's declaration")
                )
            )
    return pick(
        rng,
        (
            f"{name} {where} ({code}), {price}{declaration}",
            f"{name} ({code}) {where}{declaration}: {price}",
        ),
    )


def user_question(s: Scenario, lang: str, rng: random.Random) -> str:
    product = s.product.es if lang == "es" else s.product.en
    code = hs(s.product.code, rng, lang)
    agreement = pick(rng, AGREEMENT_NAMES[s.agreement][0 if lang == "es" else 1])
    partner = PARTNER[s.agreement][0 if lang == "es" else 1]
    made_in = EU_COUNTRIES[s.exporter_country][0 if lang == "es" else 1]
    lines = [describe_line(ln, lang, rng, s.agreement) for ln in s.lines]
    bullets = rng.random() < 0.6
    materials = (
        ("\n" + "\n".join(f"- {ln}" for ln in lines)) if bullets else " " + "; ".join(lines) + "."
    )
    exw = money(s.exw, lang, rng)
    if lang == "es":
        prices = (
            f"Precio franco fábrica (EXW): {exw}." if rng.random() < 0.5 else f"Precio EXW {exw}."
        )
        if s.fob is not None:
            prices += f" Precio FOB: {money(s.fob, lang, rng)}."
        ops = ""
        if s.operations:
            ops = (
                " Nosotros solo hacemos "
                + " y ".join(
                    {
                        "embalaje": "el embalaje",
                        "etiquetado": "el etiquetado",
                        "limpieza": "la limpieza",
                    }[o]
                    for o in s.operations
                )
                + "."
            )
        opening = pick(
            rng,
            (
                f"Fabricamos «{product}» ({code}) en {made_in} y lo exportamos a {partner}.",
                f"Producto: {product}, {code}, hecho en {made_in}, con destino {partner}.",
                f"Exportamos «{product}» ({code}) a {partner} desde {made_in}.",
            ),
        )
        ask = pick(
            rng,
            (
                f"¿Es originario según {agreement}?",
                f"¿Cumple las reglas de origen de {agreement}?",
                f"¿Tiene origen preferencial UE en {agreement}?",
            ),
        )
        head = pick(rng, ("Materiales:", "Lista de materiales:", "Materiales que usamos:"))
    else:
        prices = f"Ex-works price: {exw}." if rng.random() < 0.5 else f"EXW price {exw}."
        if s.fob is not None:
            prices += f" FOB price: {money(s.fob, lang, rng)}."
        ops = ""
        if s.operations:
            ops = (
                " All we do is "
                + " and ".join(
                    {"embalaje": "packing", "etiquetado": "labelling", "limpieza": "cleaning"}[o]
                    for o in s.operations
                )
                + "."
            )
        opening = pick(
            rng,
            (
                f"Our product is the {product} ({code}), made in {made_in} and exported to "
                f"{partner}.",
                f"Product: {product}, {code}, made in {made_in}, shipped to {partner}.",
                f"We export the {product} ({code}) from {made_in} to {partner}.",
            ),
        )
        ask = pick(
            rng,
            (
                f"Is it originating under {agreement}?",
                f"Does it meet the rules of origin of {agreement}?",
                f"Does it qualify for EU preferential origin under {agreement}?",
            ),
        )
        head = pick(rng, ("Materials:", "Bill of materials:", "Materials we use:"))
    text = f"{opening} {prices}{ops} {head}{materials}"
    if s.statement is None:
        return spanish(f"{text}\n{ask}" if bullets else f"{text} {ask}", lang)
    st = s.statement
    if lang == "es":
        ref = (
            f", número REX {st.reference}"
            if st.reference and s.agreement != "UE-KR"
            else (f", autorización de exportador autorizado {st.reference}" if st.reference else "")
        )
        request = (
            f"Si es originario según {agreement}, emite la declaración de origen para la factura "
            f"{st.invoice} (exportador {st.exporter}, {st.place}, valor del envío "
            f"{money(st.value_eur, lang, rng)}{ref})."
        )
    else:
        ref = (
            f", REX number {st.reference}"
            if st.reference and s.agreement != "UE-KR"
            else (f", approved exporter authorisation {st.reference}" if st.reference else "")
        )
        request = (
            f"If it is originating under {agreement}, issue the statement on origin for invoice "
            f"{st.invoice} (exporter {st.exporter}, {st.place}, consignment value "
            f"{money(st.value_eur, lang, rng)}{ref})."
        )
    return spanish(f"{text}\n{request}", lang)


# ----------------------------------------------------------------------------- reasoning


def origin_note(line: Line) -> str:
    where = "unknown origin" if line.country is None else f"from {line.country[1]}"
    if line.currency:
        where += f", paid in {line.currency}"
    status = line.origin + (" with a declaration" if line.declared else "")
    if line.origin in ("UE", "SOCIO") and not line.declared:
        status += " without a declaration"
    return f"{line.material.en} ({where}) is {status}"


def first_thinking(s: Scenario, rng: random.Random) -> str:
    product = f"The product is the {s.product.en}, HS {s.product.code}"
    parts = [
        pick(
            rng,
            (
                f"{product}; the rest are its materials.",
                f"{product}. Everything else listed is a material, not the product.",
            ),
        ),
        pick(rng, ("Origins: ", "Materials: "))
        + "; ".join(origin_note(ln) for ln in s.lines)
        + ".",
    ]
    if s.operations:
        parts.append(
            f"They only do {' and '.join(s.operations)}: operations {json.dumps(s.operations)}."
        )
    parts.append("A FOB price is given, so I pass it." if s.fob else "There is no FOB price.")
    return " ".join(parts)


VERDICT_ONLY_FROM_TOOL = (
    "Whether it is originating is for evaluate_origin to decide, not me.",
    "The verdict comes from evaluate_origin; I do not work it out.",
    "evaluate_origin decides the verdict.",
)


# ------------------------------------------------------------------------------- answers


def _not_checked_text(not_checked: list[str], lang: str) -> str:
    if not not_checked:
        return ""
    fob = [t.split(":")[0] for t in not_checked if "needs fob_price_eur" in t]
    hand = [t for t in not_checked if "needs fob_price_eur" not in t]
    out = []
    if fob:
        out.append(
            f" No he podido comprobar {', '.join(fob)} porque falta el precio FOB; si me lo das, "
            "lo compruebo."
            if lang == "es"
            else f" I could not check {', '.join(fob)} because the FOB price is missing; give it "
            "to me "
            "and I will check it."
        )
    if hand:
        quoted = "; ".join(t[:160] for t in hand)
        out.append(
            f" Esta parte de la regla no se puede comprobar automáticamente y hay que revisarla a "
            f"mano: «{quoted}»."
            if lang == "es"
            else f" This part of the rule cannot be checked automatically and has to be checked by "
            f'hand: "{quoted}".'
        )
    return "".join(out)


def verdict_text(s: Scenario, result: dict[str, Any], lang: str) -> str:
    product = s.product.es if lang == "es" else s.product.en
    agreement = SHORT_NAME[s.agreement][0 if lang == "es" else 1]
    nonorig = f"{result['non_originating_value_eur']:.2f}"
    pct = f"{result['pct_non_originating']}"
    tolerance = "tolerancia_usada" in result["warnings"]
    insufficient = "operacion_insuficiente" in result["warnings"]
    if lang == "es":
        nonorig = nonorig.replace(".", ",")
        pct = pct.replace(".", ",")
        if result["originating"]:
            text = (
                f"Sí: el producto «{product}» es originario según {agreement}. Cumple "
                f"{CRITERION_NAMES[result['criterion']][0]}; la regla aplicada es "
                f"{result['rule_applied']}. Los materiales no originarios suman {nonorig} € "
                f"({pct} % del precio franco fábrica)."
            )
            if tolerance:
                text += " Lo cumple gracias a la tolerancia general del 10 %."
        elif insufficient:
            text = (
                f"No: el producto «{product}» no es originario según {agreement}, porque el "
                "embalaje, el etiquetado o la limpieza por sí solos no confieren origen, "
                "sean cuales sean los materiales."
            )
        else:
            text = (
                f"No: con estos datos, el producto «{product}» no es originario según "
                f"{agreement}. No cumple ningún criterio de la regla ({result['rule_applied']}); "
                f"los materiales no originarios suman {nonorig} € ({pct} % del precio franco "
                "fábrica)."
            )
    else:
        if result["originating"]:
            text = (
                f"Yes: the {product} is originating under {agreement}. It meets "
                f"{CRITERION_NAMES[result['criterion']][1]}; the rule applied is "
                f"{result['rule_applied']}. Non-originating materials come to EUR {nonorig} "
                f"({pct} % of the ex-works price)."
            )
            if tolerance:
                text += " It does so thanks to the general tolerance of 10 %."
        elif insufficient:
            text = (
                f"No: the {product} is not originating under {agreement}, because packing, "
                "labelling or cleaning alone never confer origin, whatever the materials."
            )
        else:
            text = (
                f"No: with these data, the {product} is not originating under {agreement}. It "
                f"meets no criterion of the rule ({result['rule_applied']}); non-originating "
                f"materials come to EUR {nonorig} ({pct} % of the ex-works price)."
            )
    if insufficient and not result["originating"]:
        return text
    return text + _not_checked_text(result.get("not_checked", []), lang)


def conversion_text(lines: list[Line], rates: dict[str, dict], lang: str) -> str:
    parts = []
    for line in lines:
        r = rates[line.currency]
        name = line.material.es if lang == "es" else line.material.en
        if lang == "es":
            parts.append(
                f"{name}: {foreign_amount(line.amount, line.currency, lang)} son "
                f"{str(r['amount_eur']).replace('.', ',')} € al tipo medio del BCE de "
                f"{month_text(r['month'], lang)}"
            )
        else:
            parts.append(
                f"{name}: {foreign_amount(line.amount, line.currency, lang)} is "
                f"EUR {r['amount_eur']} at the ECB average rate for "
                f"{month_text(r['month'], lang)}"
            )
    return ("Conversión: " if lang == "es" else "Conversion: ") + "; ".join(parts) + ".\n\n"


# --------------------------------------------------------------------------- trajectories


def evaluation_arguments(s: Scenario, values: dict[int, float] | None = None) -> dict[str, Any]:
    materials = []
    for i, line in enumerate(s.lines):
        entry = {
            "name": line.material.es if line.material.es else line.material.en,
            "hs_code": line.material.code,
            "origin": line.origin,
            "value_eur": number((values or {}).get(i, line.value_eur)),
        }
        if line.declared:
            entry["supplier_declaration"] = True
        materials.append(entry)
    args: dict[str, Any] = {
        "agreement": s.agreement,
        "product_hs_code": s.product.code,
        "exw_price_eur": number(s.exw),
    }
    if s.fob is not None:
        args["fob_price_eur"] = number(s.fob)
    args["materials"] = materials
    if s.operations:
        args["operations"] = s.operations
    return args


def _material_names(s: Scenario, lang: str) -> None:
    """Materials are named in the call as the user named them."""
    for line in s.lines:
        line.material = Material(
            line.material.es if lang == "es" else line.material.en,
            line.material.en,
            line.material.code,
        )


def evaluation_trajectory(
    rng: random.Random,
    product: Product,
    agreement: str,
    today: date,
    lang: str,
    *,
    currency: bool = False,
    statement: bool = False,
) -> Trajectory:
    """Evaluate a product, maybe converting a foreign price first, maybe issuing a statement."""
    s = sample_scenario(rng, product, agreement, today, currency=currency)
    if statement:
        s.statement = sample_statement(rng, s, lang)
    kind = "statement" if statement else "currency" if currency else "evaluation"
    t = Trajectory(kind, lang, today.isoformat(), product=product.en, agreement=agreement)
    t.seed = rng.getrandbits(32)
    conv = Conversation(t)
    conv.user(user_question(s, lang, rng))
    _material_names(s, lang)
    choice = product.subdivisions.get(agreement)
    sets = load_rule(product.code, agreement, today)["rule_sets"]
    if len(sets) > 1 and choice is None:
        raise ValueError(f"{product.en} under {agreement}: no annotated subdivision")

    with Offline(today, random.Random(t.seed)):
        thinking = first_thinking(s, rng)
        values: dict[int, float] = {}
        rates: dict[str, dict] = {}
        foreign = [(i, ln) for i, ln in enumerate(s.lines) if ln.currency]
        if foreign:
            lookups = [
                (
                    "get_exchange_rate",
                    {"currency": ln.currency, "month": ln.month, "amount": number(ln.amount)},
                )
                for _, ln in foreign
            ]
            together = (
                " Both lookups are independent, so in one message." if len(foreign) > 1 else ""
            )
            records = conv.call(
                thinking
                + " Prices in a foreign currency must be in euros first: get_exchange_rate "
                "with the month and the amount." + together,
                *lookups,
            )
            for (i, line), record in zip(foreign, records, strict=True):
                if not record.ok:
                    raise ValueError(f"rate lookup failed: {record.error}")
                values[i] = record.result["amount_eur"]
                rates[line.currency] = record.result
            thinking = (
                pick(
                    rng,
                    (
                        "Now the evaluation, with the amounts in euros the tool returned.",
                        "The rates are in. I use the euro amounts as material values.",
                    ),
                )
                + " "
                + pick(rng, VERDICT_ONLY_FROM_TOOL)
            )
        else:
            thinking += " " + pick(rng, VERDICT_ONLY_FROM_TOOL)

        args = evaluation_arguments(s, values)
        [record] = conv.call(thinking, ("evaluate_origin", args))
        if not record.ok and "subdivisions" in (record.error or ""):
            number_, reason = choice
            label = tools._subdivision_label(sets[number_ - 1]["subdivision"])
            [record] = conv.call(
                f"The rule has {len(sets)} subdivisions. {reason[0].upper() + reason[1:]}, so "
                f'subdivision {number_}: "{label[:120]}".',
                ("evaluate_origin", {**args, "subdivision": number_}),
            )
        conversion = conversion_text([ln for _, ln in foreign], rates, lang) if foreign else ""

        if not record.ok:
            t.outcome = _answer_error(conv, s, record.error or "", lang, conversion)
            return t
        result = record.result
        verdict = verdict_text(s, result, lang)
        t.outcome = "originating" if result["originating"] else "not_originating"
        if s.statement is None:
            conv.answer(
                pick(
                    rng,
                    (
                        "evaluate_origin gave the verdict; I report it as it is, with what it "
                        "could not check.",
                        "I report the tool's result: verdict, criterion and "
                        "percentage, nothing of my own.",
                    ),
                ),
                conversion + verdict,
            )
            return t
        _statement_turns(conv, s, result, verdict, conversion, lang, rng, t)
    return t


def _answer_error(conv: Conversation, s: Scenario, error: str, lang: str, conversion: str) -> str:
    product = s.product.es if lang == "es" else s.product.en
    agreement = SHORT_NAME[s.agreement][0 if lang == "es" else 1]
    if "only on a FOB basis" in error:
        conv.answer(
            "The rule can only be checked with a FOB price and the user gave none: I ask for it.",
            f"La regla de origen del producto «{product}» en {agreement} solo se puede comprobar "
            "con el precio "
            "FOB. ¿Me lo das?"
            if lang == "es"
            else f"The rule of origin of the {product} under {agreement} can only be checked "
            "with the "
            "FOB price. Could you give it to me?",
        )
        return "needs_fob"
    if "checked by hand" in error:
        text = error.split("(", 1)[1].rsplit("). It has to", 1)[0] if "(" in error else ""
        quoted = text[:300]
        conv.answer(
            "The tool cannot check this rule: it must be checked by hand, so no verdict from me.",
            f"La regla de origen del producto «{product}» en {agreement} no se puede comprobar "
            "con la "
            f"herramienta: «{quoted}». Hay que revisarla a mano, así que no te doy un veredicto."
            + (" Por eso tampoco he emitido la declaración de origen." if s.statement else "")
            if lang == "es"
            else f"The rule of origin of the {product} under {agreement} cannot be checked by the "
            f'tool: "{quoted}". It has to be checked by hand, so I give no verdict.'
            + (" So I have not issued the statement on origin either." if s.statement else ""),
        )
        return "by_hand"
    raise ValueError(f"unexpected evaluate_origin error: {error}")


def _statement_turns(
    conv: Conversation,
    s: Scenario,
    result: dict[str, Any],
    verdict: str,
    conversion: str,
    lang: str,
    rng: random.Random,
    t: Trajectory,
) -> None:
    st = s.statement
    assert st is not None
    if not result["originating"]:
        conv.answer(
            "Not originating, so no statement: the user only wanted one if it was.",
            conversion
            + verdict
            + (
                " Por eso no he emitido la declaración de origen."
                if lang == "es"
                else " So I have not issued the statement on origin."
            ),
        )
        return
    reference_name = (
        (
            "número de autorización de exportador autorizado",
            "approved exporter authorisation number",
        )
        if s.agreement == "UE-KR"
        else ("número REX", "REX number")
    )
    if st.value_eur > EXPORTER_THRESHOLD and not st.reference:
        t.outcome = "originating_needs_reference"
        value = f"{st.value_eur:,.0f}".replace(",", ".") if lang == "es" else f"{st.value_eur:,.0f}"
        conv.answer(
            f"It is originating, but the consignment is {st.value_eur:,.0f} EUR, above 6,000 EUR, "
            f"and the user gave no {reference_name[1]}. I must not make one up: I ask for it "
            "instead of issuing.",
            conversion
            + verdict
            + (
                f"\n\nPara emitir la declaración necesito el {reference_name[0]} del exportador: "
                "el "
                f"envío vale {value} €, más de 6.000 €. ¿Me lo das?"
                if lang == "es"
                else f"\n\nTo issue the statement I need the exporter's {reference_name[1]}: the "
                f"consignment is worth EUR {value}, more than EUR 6,000. Could you give it to me?"
            ),
        )
        return
    arguments = {
        "evaluation_id": result["evaluation_id"],
        "invoice_number": st.invoice,
        "exporter_name": st.exporter,
        "place": st.place,
        "consignment_value_eur": number(st.value_eur),
    }
    if st.reference:
        arguments["exporter_reference"] = st.reference
    [record] = conv.call(
        pick(
            rng,
            (
                f"It is originating and the user asked for the statement. I issue it alone, in "
                f"this message, with the evaluation_id {result['evaluation_id']} that "
                "evaluate_origin returned.",
                f"Originating, and a statement was requested. Its own message, with evaluation_id "
                f"{result['evaluation_id']} from the evaluation and the invoice data the user "
                "gave.",
            ),
        ),
        ("issue_origin_statement", arguments),
    )
    if not record.ok:
        raise ValueError(f"statement refused: {record.error}")
    issued = record.result
    t.outcome = "statement_issued"
    conv.answer(
        "The tool confirms the statement and its file: I report the path and the text.",
        conversion
        + verdict
        + (
            f"\n\nHe emitido la declaración de origen para la factura {st.invoice}: "
            f"{issued['path']}.\n\n{issued['text']}"
            if lang == "es"
            else f"\n\nI have issued the statement on origin for invoice {st.invoice}: "
            f"{issued['path']}.\n\n{issued['text']}"
        ),
    )


# ------------------------------------------------------------------------------- replay


def assistant_text(message: dict[str, Any], thinking: bool) -> str:
    """What the model generates for this turn, in the variant with or without reasoning."""
    if not thinking:
        return message["content"]
    return f"<think>\n{message['thinking']}\n</think>\n\n{message['content']}"


def replay(t: Trajectory) -> list[dict[str, Any]]:
    """Run the real ToolLoop with the trajectory's turns as the model's output.

    The loop parses the calls, runs the tools (offline, same day, same ids) and appends the
    results; the messages it ends with must be the trajectory's. That is the check that a
    trajectory is something the loop would really produce.
    """
    turns = iter(assistant_text(m, thinking=True) for m in t.messages if m["role"] == "assistant")

    class Scripted(ToolLoop):
        def generate(self, messages: list[dict[str, Any]], use_tools: bool = True) -> str:
            return next(turns)

    loop = Scripted(registry=tools.default_registry(), thinking=True)
    [query] = [m["content"] for m in t.messages if m["role"] == "user"]
    with Offline(date.fromisoformat(t.today), random.Random(t.seed)):
        loop.run(query, max_turns=8)
    return loop.last_messages[1:]  # without the system prompt


# ----------------------------------------------------------------------------- sampling


def eligible_pairs(products: tuple[Product, ...] | list[Product]) -> list[tuple[Product, str]]:
    """Product and agreement pairs the generator can handle (annotated subdivisions)."""
    pairs = []
    for product in products:
        for agreement in tools.AGREEMENTS:
            if agreement in product.subdivisions and product.subdivisions[agreement] is None:
                continue
            pairs.append((product, agreement))
    return pairs


def random_today(rng: random.Random) -> date:
    """A day between March 2025 and November 2026: rates for the previous 14 months exist."""
    start, end = date(2025, 3, 1).toordinal(), date(2026, 11, 10).toordinal()
    return date.fromordinal(rng.randint(start, end))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=4, help="print this many trajectories")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--kind", choices=("evaluation", "currency", "statement", "subdivision"))
    args = parser.parse_args()
    rng = random.Random(args.seed)
    pairs = eligible_pairs(PRODUCTS)
    if args.kind == "subdivision":
        pairs = [(p, a) for p, a in pairs if a in p.subdivisions]
    for _ in range(args.sample):
        product, agreement = pick(rng, pairs)
        lang = rng.choice(("es", "en"))
        t = evaluation_trajectory(
            rng,
            product,
            agreement,
            random_today(rng),
            lang,
            currency=args.kind == "currency",
            statement=args.kind == "statement",
        )
        print(json.dumps(t.to_json(), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
