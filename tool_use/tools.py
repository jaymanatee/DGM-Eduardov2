"""The tools the model can call: five for preferential origin, plus the two worked examples.

| Type    | Tool                     | What it touches                                       |
|---------|--------------------------|-------------------------------------------------------|
| query   | ``get_origin_rule``      | UK Trade Tariff API, rules of origin (no key)         |
| query   | ``get_exchange_rate``    | ECB Data Portal, monthly reference rates (no key)     |
| query   | ``check_vat_number``     | VIES, the Commission's VAT number check (no key)      |
| compute | ``evaluate_origin``      | ``rlm.origin_engine``, the same engine as phase 1     |
| action  | ``issue_origin_statement`` | writes the statement on origin to a file (confirm) |

The docstring of each tool is the description the model reads and the type hints are the
schema, so both are written for the model. ``get_weather`` and ``calculator`` are the
examples from the slides: they stay here as reference but are not registered.

``evaluate_origin`` returns an ``evaluation_id`` and ``issue_origin_statement`` only accepts
an id whose verdict was "originating": the statement can never be issued for a product the
engine did not clear. Those verdicts live in memory, for the life of the process.

No ``from __future__ import annotations`` here: ``registry.py`` builds the Pydantic schema from
``inspect.signature``, which would see the annotations as strings and could not resolve the
custom types below (``HsCode``, ``BomLine``...).
"""

import copy
import csv
import functools
import io
import os
import re
import secrets
import time
from dataclasses import dataclass
from datetime import date, datetime
from fractions import Fraction
from pathlib import Path
from typing import Annotated, Any, Literal

import requests
from pydantic import BaseModel, BeforeValidator, Field

from rlm.origin_engine import (
    CRITERIA_PRIORITY,
    INSUFFICIENT_OPERATIONS,
    SUFFICIENT_OPERATIONS,
    VALUE_CRITERIA,
    Alt,
    Material,
    Rule,
    evaluate,
    pct,
)
from tool_use.registry import ToolRegistry, tool

OPEN_METEO_GEOCODING = "https://geocoding-api.open-meteo.com/v1/search"
OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"
UK_TARIFF_ROO = (
    "https://www.trade-tariff.service.gov.uk/api/v2/rules_of_origin_schemes/{code}/{country}"
)
ECB_EXR = "https://data-api.ecb.europa.eu/service/data/EXR/M.{currency}.EUR.SP00.A"
VIES_CHECK = "https://ec.europa.eu/taxation_customs/vies/rest-api/ms/{country}/vat/{number}"
HTTP_TIMEOUT = 10
# The ECB and VIES APIs fail now and then for a few seconds (time-outs, VIES's
# MS_MAX_CONCURRENT_REQ): 5 of ~110 calls in the first two benchmark runs. Up to three attempts
# with a short pause, so the benchmark measures the model and not the network.
HTTP_ATTEMPTS = 3
HTTP_RETRY_PAUSE_S = 1.0

# Above this value an EU exporter needs its REX number (or, for Korea, the approved-exporter
# authorisation) on the statement: Article 68 of the UCC Implementing Act.
EXPORTER_REFERENCE_THRESHOLD_EUR = 6000
# REX numbers start with the ISO code of the issuing country and 'REX' (e.g. ESREXB46123456).
# The model invented 'EX-123456' in several benchmark runs; this catches that kind.
REX_NUMBER = re.compile(r"[A-Z]{2}REX[A-Z0-9]{2,30}")

# Currencies with an ECB reference rate. Checked before calling: the ECB API does not answer
# an unknown code with an error, it hangs until the timeout.
ECB_CURRENCIES = frozenset(
    "USD JPY CZK DKK GBP HUF PLN RON SEK CHF ISK NOK TRY AUD BRL CAD CNY HKD IDR ILS INR KRW "
    "MXN MYR NZD PHP SGD THB ZAR".split()
)


# ----------------------------------------------------------------------------- agreements

Agreement = Literal["TCA", "UE-JP", "UE-KR", "CETA"]


@dataclass(frozen=True)
class AgreementInfo:
    title: str
    api_country: str  # partner code the UK API is queried with
    api_scheme: str  # scheme to keep: for Japan the API also returns CPTPP
    origin: str  # what fills "these products are of ... preferential origin"
    statement: str  # the annex text, with "..." where the exporter fills in
    reference_name: str  # what the number in brackets is called, for error messages
    annex: str
    signature: bool  # the annex asks for the exporter's signature


# Texts copied from the English version of each annex; footnote markers removed (the annexes
# say the footnotes need not be reproduced). Sources: TCA Annex ORIG-4; EU-Japan EPA Annex 3-D
# (DG TAXUD guidance, 16/12/2019); EU-Korea Protocol on origin, Annex III (OJ L 127,
# 14/5/2011, p. 1410); CETA Protocol on rules of origin, Annex 2.
_REX_SENTENCE = (
    "The exporter of the products covered by this document (Exporter Reference No ...) "
    "declares that, except where otherwise clearly indicated, these products are of ... "
    "preferential origin."
)
_AUTHORISATION_SENTENCE = (
    "The exporter of the products covered by this document (customs authorisation No ...) "
    "declares that, except where otherwise clearly indicated, these products are of ... "
    "preferential origin."
)

AGREEMENTS: dict[str, AgreementInfo] = {
    "TCA": AgreementInfo(
        "EU-UK Trade and Cooperation Agreement", "FR", "eu", "EU", _REX_SENTENCE,
        "REX number", "Annex ORIG-4", False,
    ),
    "UE-JP": AgreementInfo(
        "EU-Japan Economic Partnership Agreement", "JP", "japan", "EU", _REX_SENTENCE,
        "REX number", "Annex 3-D", False,
    ),
    "UE-KR": AgreementInfo(
        "EU-Korea Free Trade Agreement", "KR", "south-korea", "EU", _AUTHORISATION_SENTENCE,
        "approved exporter authorisation number", "Protocol on origin, Annex III", True,
    ),
    "CETA": AgreementInfo(
        "EU-Canada Comprehensive Economic and Trade Agreement", "CA", "canada", "Canada/EU",
        _AUTHORISATION_SENTENCE, "REX number", "Protocol on rules of origin, Annex 2", True,
    ),
}  # fmt: skip


def general_tolerance_pct(agreement: str, hs_code: str) -> int | None:
    """General value tolerance of the agreement for this product, or None if it does not apply.

    All four agreements allow 10 % of the price in non-originating materials that fail a
    tariff-shift rule. Textiles (chapters 50-63) have their own rules, and the TCA measures
    chapters 2 and 4-24 by weight, which the engine does not model.
    """
    chapter = int(hs_code[:2])
    if 50 <= chapter <= 63:
        return None
    if agreement == "TCA" and (chapter == 2 or 4 <= chapter <= 24):
        return None
    return 10


# --------------------------------------------------------------------------- argument types


_CODE_PREFIX = re.compile(r"^\s*(?:HS|SA|NC|CN|HTS|TARIC)(?=[\s.:\-]*\d)[\s.:\-]*", re.IGNORECASE)


def _hs6(value: Any) -> Any:
    """'8507.60', '8507 60 00', 'SA 8507.60' or 85076000 -> '850760'. Else left for the pattern.

    The prefix matters: in the thinking run the model copied 'SA 9403.40' as 'SA940340' and
    repeated the failing call four times.
    """
    if isinstance(value, int):
        value = str(value)
    if isinstance(value, str):
        digits = re.sub(r"[\s.\-]", "", _CODE_PREFIX.sub("", value))
        if digits.isdigit() and 6 <= len(digits) <= 10:
            return digits[:6]
        return digits
    return value


def _upper(value: Any) -> Any:
    return value.strip().upper() if isinstance(value, str) else value


def _compact_vat(value: Any) -> Any:
    """'fr 40 303.265-045' -> 'FR40303265045'."""
    return re.sub(r"[\s.\-]", "", value).upper() if isinstance(value, str) else value


def _file_safe(value: Any) -> Any:
    return re.sub(r"[\s/\\]+", "-", value.strip()) if isinstance(value, str) else value


HsCode = Annotated[
    str,
    BeforeValidator(_hs6),
    Field(pattern=r"^\d{6}$", description="HS subheading, 6 digits, e.g. '850760'"),
]
Operation = Literal[(*SUFFICIENT_OPERATIONS, *INSUFFICIENT_OPERATIONS)]


def _as_list(value: Any) -> Any:
    """A single value where a list is expected ('fabricacion' for ['fabricacion'])."""
    return [value] if isinstance(value, str) else value


Operations = Annotated[list[Operation], BeforeValidator(_as_list)]
Currency = Annotated[str, BeforeValidator(_upper), Field(pattern=r"^[A-Z]{3}$")]
Month = Annotated[str, Field(pattern=r"^\d{4}-(0[1-9]|1[0-2])$")]
VatNumber = Annotated[str, BeforeValidator(_compact_vat)]
InvoiceNumber = Annotated[
    str, BeforeValidator(_file_safe), Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,40}$")
]


class BomLine(BaseModel):
    """One line of the bill of materials."""

    name: str = Field(description="Material name, as on the bill of materials")
    hs_code: HsCode
    origin: Literal["UE", "SOCIO", "TERCERO", "DESCONOCIDO"] = Field(
        description="UE: made in the EU. SOCIO: made in the partner country of the agreement. "
        "TERCERO: any other country. DESCONOCIDO: origin unknown"
    )
    value_eur: float = Field(gt=0, description="Value of the material in euros")
    supplier_declaration: bool = Field(
        default=False,
        description="True if the supplier gave a supplier's declaration of preferential "
        "origin. Only UE or SOCIO materials with a declaration count as originating",
    )


class Criterion(BaseModel):
    """One alternative of the product-specific rule. get_origin_rule returns them ready."""

    kind: Literal["WO", "CC", "CTH", "CTSH", "MAXNOM", "RVC"] = Field(
        description="WO wholly obtained; CC change of chapter; CTH change of heading; "
        "CTSH change of subheading; MAXNOM maximum of non-originating materials; "
        "RVC minimum regional value content"
    )
    threshold_pct: int | None = Field(
        default=None,
        ge=1,
        le=100,
        description="MAXNOM: maximum % of non-originating materials. RVC: minimum % of "
        "regional value content. Leave empty for WO, CC, CTH and CTSH",
    )
    basis: Literal["EXW", "FOB"] = Field(default="EXW", description="Price the % refers to")
    excluded_headings: list[str] = Field(
        default_factory=list,
        description="4-digit headings excluded by the rule ('CTH except from heading 8503')",
    )


# ------------------------------------------------------------------------------ examples


@tool
def get_weather(location: str) -> float:
    """Return the current air temperature in degrees Celsius for a location.

    Args:
        location: City and country, for example 'Madrid, Spain'.
    """
    geo = requests.get(OPEN_METEO_GEOCODING, params={"name": location, "count": 1}, timeout=15)
    geo.raise_for_status()
    results = geo.json().get("results") or []
    if not results:
        raise ValueError(f"no coordinates found for {location!r}")
    lat, lon = results[0]["latitude"], results[0]["longitude"]
    forecast = requests.get(
        OPEN_METEO_FORECAST,
        params={
            "latitude": lat,
            "longitude": lon,
            "current_weather": True,
            "temperature_unit": "celsius",
        },
        timeout=15,
    )
    forecast.raise_for_status()
    return float(forecast.json()["current_weather"]["temperature"])


@tool
def calculator(expression: str) -> float:
    """Evaluate an arithmetic expression such as '(12.5 * 3) / 2 - 1'.

    Only numbers, parentheses and the operators + - * / ** % are allowed.

    Args:
        expression: The arithmetic expression to evaluate.
    """
    import ast
    import operator

    allowed = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
        ast.USub: operator.neg,
    }

    def evaluate(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in allowed:
            return allowed[type(node.op)](evaluate(node.left), evaluate(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in allowed:
            return allowed[type(node.op)](evaluate(node.operand))
        raise ValueError("unsupported expression")

    return float(evaluate(ast.parse(expression, mode="eval").body))


# ------------------------------------------------------------------------ get_origin_rule

_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_TAG = re.compile(r"<[^>]+>")
_UNTIL = re.compile(r"\(until (\d{1,2} \w+ \d{4})\)")
_PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_HEADING = re.compile(r"\b(\d{2})\.?(\d{2})\b")


def _clean(text: str) -> str:
    """API rule text (markdown with HTML) -> plain text."""
    text = _TAG.sub("", _LINK.sub(r"\1", text)).replace("&nbsp;", " ").replace("**", "")
    return re.sub(r"\s+", " ", text).strip()


def _expired(text: str, today: date) -> bool:
    """Transitional rules carry '(until 31 December 2023)'; drop them once past."""
    match = _UNTIL.search(text)
    return bool(match) and datetime.strptime(match.group(1), "%d %B %Y").date() < today


def _in_force(text: str, today: date) -> str:
    """The paragraphs of a rule that still apply. One API entry may hold the current rule and
    an expired transitional one ('... (until 31 December 2023)'): only the latter goes."""
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    return "\n\n".join(p for p in paragraphs if not _expired(p, today))


# Conditions the engine cannot check: a limit on some materials only ("provided that the
# value of non-originating materials of heading 8431 does not exceed 50 %"), weights, or parts
# that must be originating. Read as a plain criterion they would be wrong, sometimes in the
# exporter's favour, so the rule goes to not_machine_readable and is checked by hand.
_UNCHECKABLE = ("provided that", "must be originating", "weight")
# A limit on the materials of given headings or chapters only, or one item, "(ii) ...", of a
# list of conditions that all apply (the API sends each item as a rule of its own).
_PARTIAL = re.compile(
    r"materials of (?:sub)?headings? \d|materials of chapters? \d|^\(?[ivx]{1,4}\)|^- [a-z]\)"
)


def _criterion(text: str, rule_class: list[str]) -> dict[str, Any] | None:
    """One API rule -> a Criterion for evaluate_origin, or None if it does not fit the engine."""
    if len(rule_class) != 1 or rule_class[0] not in CRITERIA_PRIORITY:
        return None
    kind, lowered = rule_class[0], text.lower()
    if "same heading" in lowered:  # a cap on same-heading materials only, not a MaxNOM
        return None
    if any(condition in lowered for condition in _UNCHECKABLE) or _PARTIAL.search(lowered):
        return None
    criterion: dict[str, Any] = {"kind": kind}
    if kind in VALUE_CRITERIA:
        match = _PERCENT.search(text)
        if not match:
            return None
        criterion["threshold_pct"] = round(float(match.group(1)))
        criterion["basis"] = "FOB" if "FOB" in text or "free on board" in lowered else "EXW"
    exception = lowered.split("except", 1)[1] if "except" in lowered else ""
    if exception and "that of the product" not in exception:  # that one is just CTH itself
        headings = [a + b for a, b in _HEADING.findall(exception)]
        if not headings:  # e.g. "except from non-originating active cathode materials"
            return None
        criterion["excluded_headings"] = headings
    return criterion


def parse_rules_of_origin(
    payload: dict[str, Any], agreement: str, hs_code: str, today: date | None = None
) -> dict[str, Any]:
    """Turn the UK Trade Tariff answer into the compact result the model sees."""
    today = today or date.today()
    info = AGREEMENTS[agreement]
    schemes = [s for s in payload.get("data", []) if s["id"] == info.api_scheme]
    if not schemes:
        raise ValueError(f"the API has no {agreement} rules of origin for {hs_code}")
    included = {(i["type"], i["id"]): i for i in payload.get("included", [])}

    rule_sets = []
    for ref in schemes[0]["relationships"]["rule_sets"]["data"]:
        rule_set = included[(ref["type"], ref["id"])]
        rules = [
            included[(r["type"], r["id"])]["attributes"]
            for r in rule_set["relationships"]["rules"]["data"]
        ]
        rules = [{**r, "rule": _in_force(r["rule"], today)} for r in rules]
        rules = [r for r in rules if r["rule"]]
        texts = [_clean(r["rule"]) for r in rules]
        parsed = [_criterion(t, r["rule_class"]) for t, r in zip(texts, rules, strict=True)]
        entry: dict[str, Any] = {
            "heading": rule_set["attributes"]["heading"],
            "subdivision": rule_set["attributes"]["subdivision"],
            "rule_text": texts,
        }
        if any(r["operator"] == "and" for r in rules):
            # Alternatives joined by "and" must all hold: the engine only does "or".
            entry["criteria"] = []
            entry["not_machine_readable"] = texts
        else:
            entry["criteria"] = [c for c in parsed if c is not None]
            entry["not_machine_readable"] = [
                t for t, c in zip(texts, parsed, strict=True) if c is None
            ]
        rule_sets.append(entry)
    if not rule_sets:
        raise ValueError(f"no product-specific rule found for HS {hs_code}: is the code right?")

    note = (
        "Same rules on both sides of the TCA."
        if agreement == "TCA"
        else "Rules of the UK's agreement with the same partner, which mirrors the EU one; "
        "the EU annex on EUR-Lex is the legal text."
    )
    return {
        "agreement": agreement,
        "hs_code": hs_code,
        "general_tolerance_pct": general_tolerance_pct(agreement, hs_code),
        "rule_sets": rule_sets,
        "note": note
        + " 'criteria' are the alternatives evaluate_origin can check (any one is enough); "
        "those in 'not_machine_readable' must be checked by hand.",
    }


class ServiceUnavailable(RuntimeError):
    """An outside service did not answer after every attempt."""


_SERVICES = {
    "data-api.ecb.europa.eu": "The ECB API",
    "ec.europa.eu": "VIES",
    "www.trade-tariff.service.gov.uk": "The UK Trade Tariff API",
}


def _unavailable(url: str, reason: str) -> ServiceUnavailable:
    # The model never saw a network failure in training: with the bare 'ReadTimeout: ...' it
    # made up an ECB rate once in 7 simulated outages (entry of 11/11). Say what to do.
    host = url.split("/")[2]
    return ServiceUnavailable(
        f"{_SERVICES.get(host, host)} did not answer ({reason}) after {HTTP_ATTEMPTS} attempts: "
        "no data came back. Say so, without any figure of your own; it may answer later."
    )


def _http_get(url: str, params: dict[str, Any] | None = None) -> requests.Response:
    """GET with retries on time-outs, connection errors, 429 and 5xx; on the last one, raises
    ServiceUnavailable (other HTTP errors are raised at once)."""
    for attempt in range(1, HTTP_ATTEMPTS + 1):
        try:
            response = requests.get(url, params=params, timeout=HTTP_TIMEOUT)
        except (requests.Timeout, requests.ConnectionError) as error:
            if attempt == HTTP_ATTEMPTS:
                reason = "time-out" if isinstance(error, requests.Timeout) else "connection error"
                raise _unavailable(url, reason) from error
        else:
            transient = response.status_code == 429 or response.status_code >= 500
            if transient and attempt == HTTP_ATTEMPTS:
                raise _unavailable(url, f"HTTP {response.status_code}")
            if not transient:
                response.raise_for_status()
                return response
        time.sleep(HTTP_RETRY_PAUSE_S * attempt)
    raise AssertionError("unreachable")


@tool
def get_origin_rule(hs_code: HsCode, agreement: Agreement) -> dict:
    """Look up the product-specific rule of origin of an EU trade agreement for a product.

    Returns one or more numbered rule sets and the general tolerance. When there are several,
    each one has a 'subdivision' describing which products it covers. evaluate_origin applies
    the rule by itself: only tell it the 'number' of the rule set that matches the product.

    Args:
        hs_code: HS code of the final product, 6 digits, e.g. '850760' (longer codes are cut
            to 6 digits).
        agreement: TCA (United Kingdom), UE-JP (Japan), UE-KR (South Korea) or CETA (Canada).
    """
    rule = copy.deepcopy(_rule_of_origin(hs_code, agreement))
    rule["rule_sets"] = [{"number": i, **rs} for i, rs in enumerate(rule["rule_sets"], 1)]
    return rule


@functools.lru_cache(maxsize=512)
def _rule_of_origin(hs_code: str, agreement: str) -> dict:
    """The parsed rule, cached: evaluate_origin checks its criteria against the same answer."""
    info = AGREEMENTS[agreement]
    response = _http_get(UK_TARIFF_ROO.format(code=hs_code, country=info.api_country))
    return parse_rules_of_origin(response.json(), agreement, hs_code)


# ---------------------------------------------------------------------- get_exchange_rate


@tool
def get_exchange_rate(
    currency: Currency, month: Month | None = None, amount: float | None = None
) -> dict:
    """Monthly average ECB reference rate of a currency against the euro.

    Only for an amount the user gives in a currency other than the euro: never for prices
    already in EUR. Give 'amount' to get it converted (evaluate_origin needs euros).

    Args:
        currency: ISO code of the foreign currency, e.g. 'USD', 'JPY', 'GBP', 'KRW', 'CAD'.
            Never 'EUR'.
        month: A month as 'YYYY-MM', e.g. '2026-09', not a full date. The month the user
            names (purchase, invoice); leave it out for the latest month published.
        amount: Amount in that currency to convert into euros.
    """
    if currency not in ECB_CURRENCIES:
        raise ValueError(
            f"the ECB publishes no rate for {currency!r}. Codes: {sorted(ECB_CURRENCIES)}"
        )
    params = {"format": "csvdata"}
    params.update({"startPeriod": month, "endPeriod": month} if month else {"lastNObservations": 1})
    response = _http_get(ECB_EXR.format(currency=currency), params=params)
    rows = list(csv.DictReader(io.StringIO(response.text)))
    if not rows:
        raise ValueError(
            f"no ECB rate for {currency} in {month}: monthly averages appear once the month "
            "is over. Leave 'month' out for the latest one"
        )
    rate = float(rows[-1]["OBS_VALUE"])
    result = {
        "currency": currency,
        "month": rows[-1]["TIME_PERIOD"],
        "units_per_eur": round(rate, 6),
        "source": "ECB reference rate, monthly average",
    }
    if amount is not None:
        result |= {"amount": amount, "amount_eur": round(amount / rate, 2)}
    return result


# ----------------------------------------------------------------------- check_vat_number

_VIES_ANSWERS = {"VALID", "INVALID"}
# Busy or down for a moment: worth asking again. Anything else (INVALID_INPUT...) is final.
_VIES_TRANSIENT = {
    "MS_MAX_CONCURRENT_REQ",
    "GLOBAL_MAX_CONCURRENT_REQ",
    "MS_UNAVAILABLE",
    "SERVICE_UNAVAILABLE",
    "TIMEOUT",
}


@tool
def check_vat_number(vat_number: VatNumber) -> dict:
    """Check in VIES, the European Commission's system, whether an EU VAT number is valid.

    Use it to check a VAT number the user wrote (supplier, customer or exporter); never
    one you made up. A REX number ('ESREX' followed by digits) is not a VAT number.

    Args:
        vat_number: VAT number with its two-letter country prefix, e.g. 'DE123456789' or
            'it 123 456 789 01'. Greece uses 'EL' and Northern Ireland 'XI'. British (GB)
            numbers are not in VIES.
    """
    match = re.fullmatch(r"([A-Z]{2})([0-9A-Z+*]{2,12})", vat_number)
    if not match:
        raise ValueError(f"{vat_number!r} is not a VAT number with a country prefix")
    country, number = match.groups()
    if country == "GB":
        raise ValueError("British VAT numbers have not been in VIES since 2021")
    for attempt in range(1, HTTP_ATTEMPTS + 1):
        body = _http_get(VIES_CHECK.format(country=country, number=number)).json()
        answer = body.get("userError")
        if answer in _VIES_ANSWERS:
            break
        if answer not in _VIES_TRANSIENT or attempt == HTTP_ATTEMPTS:
            raise RuntimeError(f"VIES could not check the number: {answer}")
        time.sleep(HTTP_RETRY_PAUSE_S * attempt)

    def shared(field: str) -> str | None:  # VIES returns '---' when the country withholds it
        value = (body.get(field) or "").strip()
        return None if value in ("", "---") else value

    return {
        "vat_number": country + number,
        "valid": bool(body["isValid"]),
        "name": shared("name"),
        "address": shared("address"),
        "checked_at": body.get("requestDate"),
    }


# ------------------------------------------------------------------------ evaluate_origin

# Random ids: a sequential 'EV-0001' was guessed by the model (v0 benchmark) and matched an
# evaluation of another product, so a statement went out for goods that were not originating.
_EVALUATIONS: dict[str, dict[str, Any]] = {}


def _check_shape(criteria: list[dict[str, Any]], fob_price_eur: float | None) -> None:
    for c in criteria:
        if c["kind"] in VALUE_CRITERIA and c["threshold_pct"] is None:
            raise ValueError(f"criterion {c['kind']} needs threshold_pct")
    if fob_price_eur is None and any(
        c["kind"] in VALUE_CRITERIA and c["basis"] == "FOB" for c in criteria
    ):
        raise ValueError("a criterion is on a FOB basis: give fob_price_eur")


def assess_origin(
    agreement: str,
    product_hs_code: str,
    exw_price_eur: float,
    materials: list[dict[str, Any]],
    criteria: list[dict[str, Any]],
    operations: list[str] | None = None,
    fob_price_eur: float | None = None,
    tolerance_pct: int | None = None,
) -> dict:
    """The engine behind evaluate_origin, without the check against the official rule.

    The phase 1 cross-check uses it directly: its problems come with the synthetic rules of
    the problem generator, which are not the official ones, and with their own tolerance.
    """
    materials = [BomLine.model_validate(m).model_dump() for m in materials]
    criteria = [Criterion.model_validate(c).model_dump() for c in criteria]
    _check_shape(criteria, fob_price_eur)
    if tolerance_pct is None:
        tolerance_pct = general_tolerance_pct(agreement, product_hs_code) or 0

    rule = Rule(
        rule_id=f"{agreement}-{product_hs_code[:4]}",
        heading=product_hs_code[:4],
        agreement=agreement,
        alternatives=tuple(
            Alt(
                c["kind"],
                pct(c["threshold_pct"]) if c["kind"] in VALUE_CRITERIA else None,
                c["basis"],
                tuple(re.sub(r"\D", "", h)[:4] for h in c["excluded_headings"]),
            )
            for c in criteria
        ),
        source="api",
    )
    bom = [
        Material(
            m["name"],
            m["hs_code"],
            m["origin"],
            Fraction(str(m["value_eur"])),
            m["supplier_declaration"],
        )
        for m in materials
    ]
    result = evaluate(
        rule,
        product_hs_code,
        exw_price_eur,
        fob_price_eur if fob_price_eur is not None else exw_price_eur,
        bom,
        operations or ["fabricacion"],
        Fraction(tolerance_pct, 100),
    )

    evaluation_id = f"EV-{secrets.token_hex(3).upper()}"
    while evaluation_id in _EVALUATIONS:  # 16.7 million ids; a clash is very unlikely
        evaluation_id = f"EV-{secrets.token_hex(3).upper()}"
    _EVALUATIONS[evaluation_id] = {
        "agreement": agreement,
        "product_hs_code": product_hs_code,
        "originating": result.originating,
        "criterion": result.criterion,
        "used_tolerance": result.flags["tolerancia_usada"],
        "used_partner_materials": any(m.origin == "SOCIO" and m.originating for m in bom),
    }
    return {
        "evaluation_id": evaluation_id,
        "originating": result.originating,
        "criterion": result.criterion,
        "criteria_met": result.satisfied,
        "non_originating_value_eur": round(float(result.non_originating_value), 2),
        "pct_non_originating": result.pct_non_originating,
        "tolerance_pct": tolerance_pct,
        "warnings": [name for name, raised in result.flags.items() if raised],
        "trace": result.trace,
    }


def _describe(criterion: dict[str, Any]) -> str:
    if criterion["kind"] not in VALUE_CRITERIA:
        return criterion["kind"]
    return f"{criterion['kind']} {criterion['threshold_pct']} % {criterion.get('basis', 'EXW')}"


def _subdivision_label(text: str | None) -> str:
    """The whole text: what tells the subdivisions apart is often at the end ("... of
    vehicles of headings 8702, 8703 and 8704")."""
    if not text:
        return "all products"
    if text.strip().lower().rstrip(".") in ("other", "others"):
        return "others (any product not covered by the subdivisions above)"
    return text.strip()


def official_rule_set(agreement: str, hs_code: str, subdivision: int | None) -> dict[str, Any]:
    """The rule set of the official rule that evaluate_origin applies, or a ValueError that
    tells the model what to do (pick a subdivision, or check by hand)."""
    rule_sets = _rule_of_origin(hs_code, agreement)["rule_sets"]
    listing = "; ".join(
        f"{i}) {_subdivision_label(rs['subdivision'])}" for i, rs in enumerate(rule_sets, 1)
    )
    if subdivision is None and len(rule_sets) > 1:
        raise ValueError(
            f"the rule of origin of {hs_code} under {agreement} has {len(rule_sets)} "
            f"subdivisions: {listing}. Call evaluate_origin again with 'subdivision' set to "
            "the number of the one that matches the product."
        )
    if subdivision is not None and not 1 <= subdivision <= len(rule_sets):
        raise ValueError(f"no subdivision {subdivision}: the rule has {listing}")
    chosen = rule_sets[(subdivision or 1) - 1]
    if not chosen["criteria"]:
        texts = "; ".join(chosen["not_machine_readable"])
        raise ValueError(
            f"the rule of origin of {hs_code} under {agreement} cannot be checked by this tool "
            f"({texts[:300]}). It has to be checked by hand: tell the user, give no verdict."
        )
    return chosen


@tool
def evaluate_origin(
    agreement: Agreement,
    product_hs_code: HsCode,
    exw_price_eur: Annotated[float, Field(gt=0)],
    materials: Annotated[list[BomLine], Field(min_length=1)],
    operations: Operations | None = None,
    fob_price_eur: Annotated[float, Field(gt=0)] | None = None,
    subdivision: Annotated[int, Field(ge=1)] | None = None,
) -> dict:
    """Decide whether a product is originating, material by material, under the official
    rule of origin of the agreement, which the tool looks up by itself.

    This is the only source of an origin verdict: never work one out yourself. If the rule
    has several subdivisions, the tool lists them; call it again with 'subdivision'. The
    agreement's general tolerance is applied. Returns the verdict, the criteria met, the %
    of non-originating materials over the ex-works price, a trace per material, what could
    not be checked, and an evaluation_id for issue_origin_statement.

    Args:
        agreement: TCA (United Kingdom), UE-JP (Japan), UE-KR (South Korea) or CETA (Canada).
        product_hs_code: HS code of the final product, 6 digits.
        exw_price_eur: Ex-works price of the product, in euros.
        materials: Bill of materials, one entry per material.
        operations: What the producer does to the materials. Leave it out for normal
            manufacturing. Only embalaje, etiquetado, pintado or limpieza never confer origin.
        fob_price_eur: FOB price in euros, if the user gave it. Some rules have an
            alternative on a FOB basis that is only checked with it.
        subdivision: Number of the rule set that matches the product, as listed by
            get_origin_rule or by this tool. Only when the rule has several.
    """
    total = sum(m["value_eur"] for m in materials)
    if total > exw_price_eur:
        raise ValueError(
            f"the materials add up to {total:.2f} EUR, more than the ex-works price "
            f"({exw_price_eur:.2f} EUR): list each material once, with the values the user gave"
        )
    chosen = official_rule_set(agreement, product_hs_code, subdivision)
    criteria = chosen["criteria"]
    not_checked = list(chosen["not_machine_readable"])
    if fob_price_eur is None:  # alternatives are "or": leaving one out never makes a product pass
        fob = [c for c in criteria if c["kind"] in VALUE_CRITERIA and c.get("basis") == "FOB"]
        criteria = [c for c in criteria if c not in fob]
        not_checked += [f"{_describe(c)}: needs fob_price_eur" for c in fob]
        if not criteria:
            raise ValueError("this rule is only on a FOB basis: give fob_price_eur")
    result = assess_origin(
        agreement, product_hs_code, exw_price_eur, materials, criteria, operations, fob_price_eur
    )
    rule = " or ".join(_describe(c) for c in criteria)
    return {**result, "rule_applied": rule, "not_checked": not_checked}


# ----------------------------------------------------------------- issue_origin_statement


def statements_dir() -> Path:
    """Where statements are written; ``outputs/`` is git-ignored."""
    return Path(os.environ.get("ARCA_STATEMENTS_DIR", "outputs/origin_statements"))


def _japan_origin_criteria(evaluation: dict[str, Any]) -> str:
    """Codes of footnote 4 of Annex 3-D: A wholly obtained, C1 tariff shift, C2 value rule,
    D accumulation, E tolerance."""
    criterion = evaluation["criterion"]
    codes = ["A" if criterion == "WO" else "C2" if criterion in VALUE_CRITERIA else "C1"]
    if evaluation["used_partner_materials"]:
        codes.append("D")
    if evaluation["used_tolerance"]:
        codes.append("E")
    return " ".join(codes)


def render_statement(
    agreement: str,
    exporter_reference: str,
    exporter_name: str,
    place: str,
    issued: date,
    origin_criteria: str = "",
) -> str:
    """The statement as it goes on the invoice: annex sentence, then place, date and name."""
    info = AGREEMENTS[agreement]
    before_ref, rest = info.statement.split("...", 1)
    before_origin, after_origin = rest.split("...", 1)
    lines = [before_ref + exporter_reference + before_origin + info.origin + after_origin]
    if agreement == "UE-JP":
        lines.append(origin_criteria)
    lines += [f"{place}, {issued.isoformat()}", exporter_name]
    if info.signature:
        lines.append("Signature: ______________________")
    return "\n\n".join(lines) + "\n"


@tool(requires_confirmation=True)
def issue_origin_statement(
    evaluation_id: str,
    invoice_number: InvoiceNumber,
    exporter_name: str,
    place: str,
    consignment_value_eur: Annotated[float, Field(gt=0)],
    exporter_reference: str = "",
) -> dict:
    """Write the statement on origin for an invoice, with the exact text of the agreement.

    Only when the user asks for a statement and gave the invoice number, the exporter's name
    and the place; never fill those in yourself. Only after evaluate_origin has returned
    originating: true, in a later message: pass the evaluation_id it returned. If the
    product is not originating, do not call it; tell the user why. The statement is saved
    as a file and is not overwritten if the invoice already has one.

    Args:
        evaluation_id: The evaluation_id that evaluate_origin returned in this conversation.
            Never guess one.
        invoice_number: Number of the invoice the statement goes on, as the user wrote it.
        exporter_name: Legal name of the exporter, as the user wrote it.
        place: City where the statement is made out, as the user wrote it.
        consignment_value_eur: Total value of the originating products on the invoice, in
            euros.
        exporter_reference: REX number of the exporter: country code, 'REX', then the
            exporter's own number, as in 'ESREX...' (for UE-KR,
            the approved exporter authorisation number), only if the user gave it: never
            make one up. Leave it empty otherwise; above 6,000 euros it is required, so ask
            the user for it.
    """
    evaluation = _EVALUATIONS.get(evaluation_id)
    if evaluation is None:
        raise ValueError(f"unknown evaluation_id {evaluation_id!r}: call evaluate_origin first")
    if not evaluation["originating"]:
        raise ValueError(
            f"{evaluation_id} found the product NOT originating: no statement can be issued"
        )
    agreement = evaluation["agreement"]
    info = AGREEMENTS[agreement]
    exporter_reference = exporter_reference.strip()
    if exporter_reference and info.reference_name == "REX number":
        exporter_reference = re.sub(r"[\s.\-]", "", exporter_reference).upper()
    if (
        exporter_reference
        and info.reference_name == "REX number"
        and not REX_NUMBER.fullmatch(exporter_reference)
    ):
        raise ValueError(
            f"{exporter_reference!r} is not a REX number (country code, 'REX', then the "
            "exporter's own number). Use the one the user gave; if they gave none, "
            "leave it empty or ask them for it"
        )
    if consignment_value_eur > EXPORTER_REFERENCE_THRESHOLD_EUR and not exporter_reference:
        raise ValueError(
            f"consignment of {consignment_value_eur:.2f} EUR: above "
            f"{EXPORTER_REFERENCE_THRESHOLD_EUR} EUR the statement needs the exporter's "
            f"{info.reference_name}"
        )

    path = statements_dir() / f"{invoice_number}.txt"
    if path.exists():
        raise FileExistsError(f"invoice {invoice_number} already has a statement: {path}")
    issued = date.today()
    text = render_statement(
        agreement,
        exporter_reference,
        exporter_name.strip(),
        place.strip(),
        issued,
        _japan_origin_criteria(evaluation),
    )
    header = (
        f"Statement on origin - {info.title} ({info.annex})\n"
        f"Invoice {invoice_number} - product HS {evaluation['product_hs_code']} - "
        f"{evaluation_id}, criterion {evaluation['criterion']}\n"
        f"{'-' * 72}\n\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + text, encoding="utf-8")
    return {"path": str(path), "agreement": agreement, "issued": issued.isoformat(), "text": text}


# ----------------------------------------------------------------------------- registry


def default_registry() -> ToolRegistry:
    """The registry the executor and the agent use: the five origin tools, not the examples."""
    return ToolRegistry().register(
        get_origin_rule,
        get_exchange_rate,
        check_vat_number,
        evaluate_origin,
        issue_origin_statement,
    )
