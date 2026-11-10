"""Download, once, what the synthetic trajectories need from the APIs; afterwards, no network.

* Rules of origin: the raw UK Trade Tariff answer for every product of ``catalog.py`` under
  the four agreements, in ``tool_use/sft/data/roo/<code>_<country>.json`` (the same format as
  the test fixtures). The raw answer and not the parsed rule, because parsing depends on the
  date (transitional rules expire) and is done when the trajectories are generated.
* Exchange rates: the ECB monthly series of every currency since 2024, as the CSV the API
  returns, in ``tool_use/sft/data/ecb/<currency>.csv``.

VIES is not downloaded: its answers are made up with the real format when generating, since
real VAT numbers of real companies have no place in a training set.

Run (it skips what is already on disk, so it can be resumed)::

    uv run python -m tool_use.sft.fetch
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

from tool_use.sft.catalog import PRODUCTS
from tool_use.tools import (
    AGREEMENTS,
    ECB_CURRENCIES,
    ECB_EXR,
    UK_TARIFF_ROO,
    _http_get,
    parse_rules_of_origin,
)

DATA = Path(__file__).parent / "data"
ECB_START = "2024-01"
PAUSE_S = 0.3  # between requests: these are public services


def roo_path(code: str, agreement: str) -> Path:
    return DATA / "roo" / f"{code}_{AGREEMENTS[agreement].api_country}.json"


def ecb_path(currency: str) -> Path:
    return DATA / "ecb" / f"{currency}.csv"


def trim(payload: dict[str, Any], agreement: str) -> dict[str, Any]:
    """Keep the agreement's scheme and its rule sets and rules: the legal articles, proofs and
    links, and the scheme's introduction, are 99 % of the answer and are never parsed."""
    schemes = [s for s in payload.get("data", []) if s["id"] == AGREEMENTS[agreement].api_scheme]
    included = {(i["type"], i["id"]): i for i in payload.get("included", [])}
    kept, data = [], []
    for scheme in schemes:
        sets = scheme["relationships"]["rule_sets"]["data"]
        data.append(
            {
                "id": scheme["id"],
                "type": scheme["type"],
                "relationships": {"rule_sets": {"data": sets}},
            }
        )
        for ref in sets:
            rule_set = included[(ref["type"], ref["id"])]
            kept.append(rule_set)
            kept += [
                included[(r["type"], r["id"])] for r in rule_set["relationships"]["rules"]["data"]
            ]
    return {"data": data, "included": kept}


def load_rule(code: str, agreement: str, today: date | None = None) -> dict[str, Any]:
    """The rule as get_origin_rule's helper would parse it, read from disk."""
    payload = json.loads(roo_path(code, agreement).read_text(encoding="utf-8"))
    return parse_rules_of_origin(payload, agreement, code, today)


def fetch_rules() -> list[str]:
    failures = []
    for product in PRODUCTS:
        for agreement, info in AGREEMENTS.items():
            path = roo_path(product.code, agreement)
            if path.exists():
                continue
            url = UK_TARIFF_ROO.format(code=product.code, country=info.api_country)
            try:
                payload = _http_get(url).json()
            except Exception as error:  # noqa: BLE001 - reported, the rest goes on
                failures.append(f"{product.code} {agreement}: {error}")
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(trim(payload, agreement), ensure_ascii=False), encoding="utf-8"
            )
            time.sleep(PAUSE_S)
    return failures


def fetch_rates() -> list[str]:
    failures = []
    for currency in sorted(ECB_CURRENCIES):
        path = ecb_path(currency)
        if path.exists():
            continue
        params = {"format": "csvdata", "startPeriod": ECB_START}
        try:
            text = _http_get(ECB_EXR.format(currency=currency), params=params).text
        except Exception as error:  # noqa: BLE001
            failures.append(f"{currency}: {error}")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        time.sleep(PAUSE_S)
    return failures


def kind_of_rule(rule: dict[str, Any]) -> str:
    """What a trajectory with this rule teaches: the path evaluate_origin takes."""
    sets = rule["rule_sets"]
    if len(sets) > 1:
        return "subdivisions"
    only = sets[0]
    if not only["criteria"]:
        return "by_hand"
    if all(c.get("basis") == "FOB" for c in only["criteria"]):
        return "fob_only"
    if only["not_machine_readable"]:
        return "partly_readable"
    return "readable"


def summary(today: date | None = None) -> None:
    kinds: Counter[str] = Counter()
    for product in PRODUCTS:
        row = []
        for agreement in AGREEMENTS:
            if not roo_path(product.code, agreement).exists():
                row.append(f"{agreement}: missing")
                continue
            try:
                kind = kind_of_rule(load_rule(product.code, agreement, today))
            except ValueError as error:
                kind = f"error ({error})"
            kinds[kind.split(" ")[0]] += 1
            row.append(f"{agreement}: {kind}")
        print(f"{product.code} {product.en:32s} " + " | ".join(row))
    print(dict(kinds))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    if not args.summary_only:
        failures = fetch_rules() + fetch_rates()
        for failure in failures:
            print("FAILED", failure)
    summary()


if __name__ == "__main__":
    main()
