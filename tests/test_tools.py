"""The five origin tools, without network: HTTP answers come from recorded fixtures."""

import json
import re
from datetime import date
from pathlib import Path

import pytest

from rlm.generate_problems import ALL_RULES
from tool_use import tools
from tool_use.tools import (
    AGREEMENTS,
    check_vat_number,
    default_registry,
    evaluate_origin,
    get_exchange_rate,
    get_origin_rule,
    issue_origin_statement,
    parse_rules_of_origin,
    render_statement,
)

FIXTURES = Path(__file__).parent / "fixtures" / "tool_use"
ROOT = Path(__file__).resolve().parent.parent


class FakeResponse:
    def __init__(self, payload=None, text="", status=200):
        self.payload, self.text, self.status_code = payload, text, status

    def json(self):
        return self.payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


@pytest.fixture
def http(monkeypatch):
    """Replace requests.get; the test sets ``http.response`` and reads ``http.calls``."""

    class Http:
        response = FakeResponse()
        sequence: list = []  # if set, one item per call: a response or an exception to raise
        calls: list = []
        pauses: list = []

    def fake_get(url, params=None, timeout=None):
        Http.calls.append((url, params))
        item = Http.sequence.pop(0) if Http.sequence else Http.response
        if isinstance(item, Exception):
            raise item
        return item

    Http.calls, Http.sequence, Http.pauses = [], [], []
    monkeypatch.setattr(tools.requests, "get", fake_get)
    monkeypatch.setattr(tools.time, "sleep", Http.pauses.append)
    return Http


def roo(country):
    return json.loads((FIXTURES / f"roo_850760_{country}.json").read_text(encoding="utf-8"))


# ------------------------------------------------------------------------ get_origin_rule


def test_origin_rule_tca_drops_expired_rules_and_keeps_subdivisions(http):
    http.response = FakeResponse(roo("FR"))
    result = get_origin_rule(hs_code="8507.60.00", agreement="TCA")
    assert http.calls[0][0].endswith("/850760/FR")
    assert result["hs_code"] == "850760" and result["general_tolerance_pct"] == 10
    sets = {s["subdivision"]: s for s in result["rule_sets"]}
    assert len(sets) == 3
    others = sets["others"]
    assert others["criteria"] == [
        {"kind": "CTH"},
        {"kind": "MAXNOM", "threshold_pct": 50, "basis": "EXW"},
    ]
    assert not any("until" in t for s in result["rule_sets"] for t in s["rule_text"])
    packs = next(s for name, s in sets.items() if name.startswith("Accumulators"))
    # "CTH except from non-originating active cathode materials" is not a heading exception.
    assert packs["criteria"] == [{"kind": "MAXNOM", "threshold_pct": 30, "basis": "EXW"}]
    assert "cathode" in packs["not_machine_readable"][0]


def test_origin_rule_transitional_rules_apply_before_their_end_date():
    result = parse_rules_of_origin(roo("FR"), "TCA", "850760", today=date(2023, 6, 1))
    packs = next(s for s in result["rule_sets"] if s["subdivision"].startswith("Accumulators"))
    assert {"kind": "MAXNOM", "threshold_pct": 70, "basis": "EXW"} in packs["criteria"]


def test_origin_rule_japan_ignores_cptpp_and_reads_fob(http):
    http.response = FakeResponse(roo("JP"))
    result = get_origin_rule(hs_code="850760", agreement="UE-JP")
    assert http.calls[0][0].endswith("/850760/JP")
    [rule_set] = result["rule_sets"]
    assert rule_set["heading"] == "8502-8518"
    assert rule_set["criteria"] == [
        {"kind": "CTH"},
        {"kind": "MAXNOM", "threshold_pct": 50, "basis": "EXW"},
        {"kind": "RVC", "threshold_pct": 55, "basis": "FOB"},
    ]
    assert "EUR-Lex" in result["note"]


def test_origin_rule_korea_and_canada_wordings():
    [korea] = parse_rules_of_origin(roo("KR"), "UE-KR", "850760")["rule_sets"]
    assert korea["criteria"] == [
        {"kind": "CTH"},
        {"kind": "MAXNOM", "threshold_pct": 45, "basis": "EXW"},
    ]
    [canada] = parse_rules_of_origin(roo("CA"), "CETA", "850760")["rule_sets"]
    # A cap on materials of the same heading is not a MaxNOM: left for a human.
    assert canada["criteria"] == [{"kind": "CTH"}]
    assert "same heading" in canada["not_machine_readable"][0]


def test_origin_rule_errors_reach_the_model(http):
    registry = default_registry()
    bad_code = registry.call("get_origin_rule", {"hs_code": "8507", "agreement": "TCA"})
    assert not bad_code.ok and "invalid arguments" in bad_code.error
    bad_agreement = registry.call("get_origin_rule", {"hs_code": "850760", "agreement": "Japón"})
    assert not bad_agreement.ok and "invalid arguments" in bad_agreement.error
    http.response = FakeResponse({"data": [], "included": []})
    no_rule = registry.call("get_origin_rule", {"hs_code": "999999", "agreement": "UE-KR"})
    assert not no_rule.ok and "no UE-KR rules" in no_rule.error
    assert len(http.calls) == 1  # invalid arguments never reach the API


# ---------------------------------------------------------------------- get_exchange_rate

ECB_CSV = (
    "KEY,FREQ,CURRENCY,CURRENCY_DENOM,EXR_TYPE,EXR_SUFFIX,TIME_PERIOD,OBS_VALUE,OBS_STATUS\n"
    "EXR.M.USD.EUR.SP00.A,M,USD,EUR,SP00,A,2026-09,1.1513227272727276,A\n"
)


def test_exchange_rate_converts_to_euros(http):
    http.response = FakeResponse(text=ECB_CSV)
    result = get_exchange_rate(currency="usd", month="2026-09", amount=1151.32)
    assert http.calls[0][1] == {
        "format": "csvdata",
        "startPeriod": "2026-09",
        "endPeriod": "2026-09",
    }
    assert result["units_per_eur"] == pytest.approx(1.151323)
    assert result["amount_eur"] == 1000.0


def test_exchange_rate_errors(http):
    http.response = FakeResponse(text="")
    with pytest.raises(ValueError, match="once the month"):
        get_exchange_rate(currency="USD", month="2026-12")
    calls = len(http.calls)
    with pytest.raises(ValueError, match="no rate"):
        get_exchange_rate(currency="XYZ")
    assert len(http.calls) == calls  # unknown codes are rejected before the request


# ----------------------------------------------------------------------- check_vat_number

VIES = {
    "isValid": True,
    "requestDate": "2026-11-08T22:03:35.547Z",
    "userError": "VALID",
    "name": "---",
    "address": "---",
}


def test_vat_number_is_normalised_and_checked(http):
    http.response = FakeResponse(VIES)
    result = check_vat_number(vat_number="es a-28015865")
    assert http.calls[0][0].endswith("/ms/ES/vat/A28015865")
    assert result == {
        "vat_number": "ESA28015865",
        "valid": True,
        "name": None,
        "address": None,
        "checked_at": "2026-11-08T22:03:35.547Z",
    }


def test_vat_number_errors(http):
    http.response = FakeResponse({**VIES, "isValid": False, "userError": "MS_UNAVAILABLE"})
    with pytest.raises(RuntimeError, match="MS_UNAVAILABLE"):
        check_vat_number(vat_number="DE123456789")
    with pytest.raises(ValueError, match="since 2021"):
        check_vat_number(vat_number="GB123456789")
    with pytest.raises(ValueError, match="country prefix"):
        check_vat_number(vat_number="28015865")
    assert len(http.calls) == 3  # MS_UNAVAILABLE was asked three times, the rest never sent


def test_transient_failures_are_retried(http):
    busy = FakeResponse({**VIES, "userError": "MS_MAX_CONCURRENT_REQ"})
    http.sequence = [busy, FakeResponse(VIES)]
    assert check_vat_number(vat_number="ESA28015865")["valid"]
    assert len(http.calls) == 2 and http.pauses == [1.0]

    http.calls = []
    http.pauses.clear()  # sleep appends to this very list
    timeout = tools.requests.Timeout("read timed out")
    http.sequence = [timeout, FakeResponse(status=503), FakeResponse(text=ECB_CSV)]
    assert get_exchange_rate(currency="JPY", month="2026-09")["units_per_eur"] > 0
    assert len(http.calls) == 3 and http.pauses == [1.0, 2.0]


def test_permanent_failures_are_not_retried(http):
    http.sequence = [FakeResponse(status=404)]
    with pytest.raises(RuntimeError, match="404"):
        get_origin_rule(hs_code="850760", agreement="TCA")
    assert len(http.calls) == 1
    http.calls = []
    http.response = FakeResponse({**VIES, "userError": "INVALID_INPUT"})
    with pytest.raises(RuntimeError, match="INVALID_INPUT"):
        check_vat_number(vat_number="ESA28015865")
    assert len(http.calls) == 1
    http.calls = []
    http.sequence = [tools.requests.ConnectionError("down")] * 3
    with pytest.raises(tools.requests.ConnectionError):
        get_exchange_rate(currency="JPY")
    assert len(http.calls) == 3


# ------------------------------------------------------------------------ evaluate_origin


def tool_arguments(params):
    """A phase 1 problem as evaluate_origin arguments."""
    rule = ALL_RULES[params["rule_id"]]
    criteria = []
    for alt in rule.alternatives:
        criterion = {
            "kind": alt.kind,
            "basis": alt.basis,
            "excluded_headings": list(alt.excluded_headings),
        }
        if alt.threshold is not None:
            criterion["threshold_pct"] = int(alt.threshold * 100)
        criteria.append(criterion)
    return {
        "agreement": params["agreement"],
        "product_hs_code": params["product_code"],
        "exw_price_eur": params["exw"],
        "fob_price_eur": params["fob"],
        "tolerance_pct": params["tolerance_pct"],
        "operations": params["operations"],
        "criteria": criteria,
        "materials": [
            {
                "name": m["name"],
                "hs_code": m["code"],
                "origin": m["origin"],
                "value_eur": m["value"],
                "supplier_declaration": m["declared"],
            }
            for m in params["materials"]
        ],
    }


def test_evaluate_origin_matches_every_phase1_test_problem():
    rows = [json.loads(line) for line in (ROOT / "rlm/data/test.jsonl").read_text().splitlines()]
    registry = default_registry()
    for row in rows:
        record = registry.call("evaluate_origin", tool_arguments(row["params"]))
        assert record.ok, record.error
        expected = json.loads(row["answer"])
        assert record.result["originating"] == expected["originario"]
        assert record.result["criterion"] == expected["criterio"]
        assert record.result["pct_non_originating"] == expected["pct_no_originario"]
    assert len(rows) == 200


def battery(**overrides):
    arguments = {
        "agreement": "UE-JP",
        "product_hs_code": "850760",
        "exw_price_eur": 250,
        "criteria": [{"kind": "CTH"}, {"kind": "MAXNOM", "threshold_pct": 50}],
        "materials": [
            {"name": "celdas", "hs_code": "8507.60", "origin": "TERCERO", "value_eur": 95},
            {"name": "carcasa", "hs_code": "392690", "origin": "UE", "value_eur": 20,
             "supplier_declaration": True},
        ],
    }  # fmt: skip
    return arguments | overrides


def test_evaluate_origin_uses_the_agreement_tolerance_and_explains():
    result = evaluate_origin(**battery())
    # Cells are in the same heading: CTH fails (95 > 10 % of 250) but MaxNOM 38 % <= 50 % passes.
    assert result["originating"] and result["criterion"] == "MAXNOM"
    assert result["pct_non_originating"] == 38.0 and result["tolerance_pct"] == 10
    assert "material_misma_partida" in result["warnings"]
    assert re.fullmatch(r"EV-[0-9A-F]{6}", result["evaluation_id"])  # random: not guessable
    assert evaluate_origin(**battery())["evaluation_id"] != result["evaluation_id"]
    assert len(result["trace"]) == 4


def test_evaluate_origin_insufficient_operations_and_bad_criteria():
    assert not evaluate_origin(**battery(operations=["embalaje"]))["originating"]
    registry = default_registry()
    no_threshold = registry.call("evaluate_origin", battery(criteria=[{"kind": "MAXNOM"}]))
    assert not no_threshold.ok and "threshold_pct" in no_threshold.error
    fob = registry.call(
        "evaluate_origin", battery(criteria=[{"kind": "RVC", "threshold_pct": 55, "basis": "FOB"}])
    )
    assert not fob.ok and "fob_price_eur" in fob.error


# ----------------------------------------------------------------- issue_origin_statement


@pytest.fixture
def out_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("ARCA_STATEMENTS_DIR", str(tmp_path))
    return tmp_path


def test_statement_text_is_the_annex_sentence():
    for agreement, info in AGREEMENTS.items():
        text = render_statement(agreement, "ESREX123", "ACME S.L.", "Valencia", date(2026, 11, 9))
        sentence = text.split("\n\n")[0]
        assert (
            sentence.replace("ESREX123", "...", 1).replace(f" {info.origin} ", " ... ", 1)
            == info.statement
        )
    assert "Canada/EU preferential origin" in render_statement("CETA", "", "A", "B", date.today())


def test_issue_statement_writes_the_file_once(out_dir):
    evaluation = evaluate_origin(**battery())
    arguments = {
        "evaluation_id": evaluation["evaluation_id"],
        "invoice_number": "F/2026/118",
        "exporter_name": "Baterías del Turia S.L.",
        "place": "Valencia",
        "consignment_value_eur": 2500,
    }
    result = issue_origin_statement(**arguments)
    path = out_dir / "F-2026-118.txt"
    assert result["path"] == str(path) and path.exists()
    content = path.read_text(encoding="utf-8")
    assert "(Exporter Reference No ) declares that" in content
    assert "these products are of EU preferential origin." in content
    assert "\n\nC2\n\n" in content  # Annex 3-D origin criteria: MaxNOM is a value rule
    with pytest.raises(FileExistsError):
        issue_origin_statement(**arguments)


def test_issue_statement_refusals(out_dir):
    not_originating = evaluate_origin(**battery(operations=["etiquetado"]))["evaluation_id"]
    originating = evaluate_origin(**battery(agreement="UE-KR"))["evaluation_id"]
    common = {"invoice_number": "F-1", "exporter_name": "A", "place": "B"}
    with pytest.raises(ValueError, match="NOT originating"):
        issue_origin_statement(evaluation_id=not_originating, consignment_value_eur=100, **common)
    with pytest.raises(ValueError, match="approved exporter authorisation"):
        issue_origin_statement(evaluation_id=originating, consignment_value_eur=6500, **common)
    with pytest.raises(ValueError, match="call evaluate_origin first"):
        issue_origin_statement(evaluation_id="EV-9999", consignment_value_eur=100, **common)
    assert not list(out_dir.iterdir())


# ----------------------------------------------------------------------------- registry


def test_registry_has_the_five_tools_and_only_the_action_asks_for_confirmation():
    registry = default_registry()
    assert sorted(registry.tools) == [
        "check_vat_number",
        "evaluate_origin",
        "get_exchange_rate",
        "get_origin_rule",
        "issue_origin_statement",
    ]
    confirm = {name for name, t in registry.tools.items() if t.requires_confirmation}
    assert confirm == {"issue_origin_statement"}
    params = registry.tools["get_origin_rule"].definition["function"]["parameters"]
    assert params["properties"]["agreement"]["enum"] == ["TCA", "UE-JP", "UE-KR", "CETA"]
    assert "6 digits" in params["properties"]["hs_code"]["description"]
