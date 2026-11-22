"""The phase-4 task bank and its automatic checks: every check tells a right answer from the
typical wrong one, before any GPU time is spent on it."""

import json
import re
from pathlib import Path

import pytest

from agent.evaluate import CHECK_TYPES, check_answer, task_passed, tools_that_worked

BENCH = Path(__file__).resolve().parents[1] / "agent" / "bench" / "tasks.jsonl"
TASKS = {t["id"]: t for t in map(json.loads, BENCH.read_text(encoding="utf-8").splitlines())}
TOOLS = {
    "get_origin_rule", "get_exchange_rate", "check_vat_number", "evaluate_origin",
    "issue_origin_statement", "search_knowledge_base",
}  # fmt: skip

# id -> (a right answer, the typical wrong one), as the agent would write them.
ANSWERS = {
    "cambio-01": (
        "Sí, es originario según el TCA: los materiales no originarios suman 290,28 € (48,38 %), "
        "por debajo del 50 % del MAXNOM.",
        "No es originario: los materiales no originarios son el 51,67 %, más del 50 %.",
    ),
    "cambio-02": (
        "The laptops are not originating: non-originating materials are 58.94 % of the ex-works "
        "price, and heading 8473 is excluded from the CTH.",
        "Yes, the laptops are originating under the EU-Japan EPA (CTH).",
    ),
    "cambio-03": (
        "Sí, es originario: con la acumulación el panel coreano cuenta como originario y los "
        "materiales no originarios son el 44 %.",
        "No es originario: los materiales no originarios son el 68,05 %.",
    ),
    "veredicto-01": (
        "They are not originating: non-originating materials are 52.63 %, above the 45 % allowed.",
        "Yes, they are originating under the TCA.",
    ),
    "veredicto-02": (
        "Sí, son originarios según el TCA por cambio de partida (CTH): las partes de horno de la "
        "misma partida son el 7,5 % y entran en la tolerancia del 10 %.",
        "No son originarios: los materiales no originarios son el 70 %, más del 50 %.",
    ),
    "veredicto-03": (
        "Yes. Under subdivision 3 (others) the packs are originating: non-originating materials "
        "are 46.88 % of the ex-works price, within the 50 % MaxNOM.",
        "The packs are not originating: 46.88 % exceeds the 30 % allowed for battery packs.",
    ),
    "codigo-no-material-01": (
        "Sí, es originario según el acuerdo UE-Japón: cumple el CTH y los materiales no "
        "originarios son el 37,5 %.",
        "No he podido evaluarlo: falta el valor del material 8418.21.",
    ),
    "veredicto-04": (
        "No son originarias: reembalar y etiquetar son operaciones insuficientes.",
        "Sí, son originarias según el CETA.",
    ),
    "declaracion-01": (
        "Son originarias (CTH).\n\n[Comprobado por el sistema] Declaración de origen emitida: "
        "outputs/origin_statements/LV-2026-0312.txt",
        "Son originarias según el TCA. ¿Quiere que emita la declaración?",
    ),
    "declaracion-02": (
        "The seats are originating.\n\n[Comprobado por el sistema] Declaración de origen emitida: "
        "outputs/origin_statements/JP-4471.txt",
        "The seats are originating; the statement could not be issued.",
    ),
    "faltan-datos-01": (
        "Son originarios (22,35 %), pero el envío supera los 6.000 €: necesito el número REX del "
        "exportador para emitir la declaración.",
        "Son originarios.\n\n[Comprobado por el sistema] Declaración de origen emitida: "
        "outputs/origin_statements/CM-0915.txt",
    ),
    "declaracion-03": (
        "They are not originating: non-originating materials are 51.11 % (maximum 45 %), so no "
        "origin declaration can be made out.",
        "They are originating.\n\n[Comprobado por el sistema] Declaración de origen emitida: "
        "outputs/origin_statements/KR-2026-077.txt",
    ),
    "conocimiento-01": (
        "Hasta 1.000 dólares estadounidenses [1], que con la media del BCE de octubre de 2026 "
        "son 892,73 €.",
        "Hasta 1.000 USD, unos 1.120,16 €.",
    ),
    "conocimiento-02": (
        "Sí, es originario (CTH). Pero por encima de 6.000 € solo un exportador autorizado puede "
        "extender la declaración de origen, así que no podéis hacerlo vosotros.",
        "No es originario: los materiales no originarios son el 48,57 %, más del 45 %.",
    ),
    "conocimiento-03": (
        "Yes, they are originating under CETA (CTH, with the 10 % tolerance). You must keep the "
        "origin declaration and the supporting documents for three years.",
        "Yes, they are originating. Keep the documents for 4 years.",
    ),
    "vies-01": (
        "El NIF IT00743110157 es válido (MOTOROLA SOLUTIONS ITALIA). Los smartphones son "
        "originarios: los materiales no originarios son el 28,33 %.",
        "El NIF es válido, pero los smartphones no son originarios.",
    ),
    "imposible-01": (
        "No puedo darte un veredicto: la regla textil del TCA exige hilar y tejer, y la "
        "herramienta no la aplica. Hay que comprobarla a mano.",
        "No son originarios: el hilado es de Turquía.",
    ),
    "imposible-02": (
        "Solo puedo evaluar el origen con el TCA, el acuerdo con Japón, el de Corea y el CETA: "
        "el acuerdo UE-Australia no está entre ellos, y la base de conocimiento no lo cubre.",
        "Sí, son originarias según el acuerdo UE-Australia, y la declaración vale 12 meses.",
    ),
    "faltan-datos-02": (
        "El BCE no publica un tipo de cambio para el dong (VND), así que no puedo pasar los "
        "650.000 VND a euros. ¿Cuánto costaron los difusores en euros?",
        "Son originarias: los 650.000 VND son unos 21 € y el total no originario es el 52 %.",
    ),
    "larga-01": (
        "El cuadro es originario (MAXNOM, 48,11 %).\n\n[Comprobado por el sistema] Declaración "
        "de origen emitida: outputs/origin_statements/EXP-2026-1187.txt",
        "El cuadro no es originario: los materiales no originarios son el 50,4 %.",
    ),
}


def check_types(check: dict) -> set[str]:
    kinds = {check["type"]}
    if check["type"] == "not":
        kinds |= check_types(check["value"])
    elif check["type"] in ("all", "any"):
        for c in check["value"]:
            kinds |= check_types(c)
    return kinds


def test_twenty_tasks_with_unique_ids():
    assert len(TASKS) == 20


@pytest.mark.parametrize("task", TASKS.values(), ids=list(TASKS))
def test_task_shape(task):
    assert check_types(task["check"]) <= CHECK_TYPES
    assert set(task["required_tools"]) <= TOOLS
    assert set(task.get("forbidden_tools", [])) <= TOOLS
    if task["category"] != "imposible":
        assert len(task["required_tools"]) >= 2 or task["category"] == "faltan_datos"


@pytest.mark.parametrize("task", [t for t in TASKS.values() if "reference_call" in t], ids=str)
def test_reference_call_uses_the_codes_of_the_task(task):
    """Every material of the expected call is in the text: the reference was not made up."""
    text = re.sub(r"(\d{4})[.\s](\d{2})", r"\1\2", task["task"])
    args = task["reference_call"]["arguments"]
    assert args["product_hs_code"] in text
    for material in args["materials"]:
        assert material["hs_code"] in text, material


@pytest.mark.parametrize("task_id", list(ANSWERS))
def test_check_tells_right_from_wrong(task_id):
    right, wrong = ANSWERS[task_id]
    assert check_answer(TASKS[task_id], right)
    assert not check_answer(TASKS[task_id], wrong)


def test_every_task_has_answers():
    assert set(ANSWERS) == set(TASKS)


def test_numbers_in_spanish_and_english():
    task = {"check": {"type": "number", "value": 1042.28, "tolerance": 0.01}}
    for text in ("1.042,28 €", "1,042.28 EUR", "1042,28", "1042.28"):
        assert check_answer(task, text), text
    assert not check_answer(task, "104,228")


def _step(name, payload, note=""):
    return {"action": {"name": name}, "observation": note + json.dumps(payload)}


def test_tools_that_worked_reads_the_observation():
    steps = [
        _step("evaluate_origin", {"name": "evaluate_origin", "result": {"originating": True}}),
        _step("issue_origin_statement", {"name": "issue_origin_statement", "error": "needs REX"},
              "[confirmed automatically: ARCA_AGENT_CONFIRM=auto] "),
        {"thought": "no action", "action": None, "observation": "[no tool called] ..."},
    ]  # fmt: skip
    assert tools_that_worked(steps) == {"evaluate_origin"}


def test_forbidden_tool_that_ran_fails_the_task():
    task = TASKS["faltan-datos-02"]
    answer = ANSWERS["faltan-datos-02"][0]
    clean = {"succeeded": True, "final_answer": answer, "steps": []}
    assert task_passed(task, clean)
    invented = {
        **clean,
        "steps": [_step("evaluate_origin", {"name": "evaluate_origin", "result": {}})],
    }
    assert not task_passed(task, invented)


class _FakeAgent:
    """Answers every task with the right answer of ANSWERS, and crashes on one of them."""

    tokens_used = 0

    def __init__(self):
        self.calls = []

    def label(self, brain):
        return f"fake ({brain})"

    def run(self, task, max_steps, brain):
        from api.schemas import AgentResponse

        task_id = next(i for i, t in TASKS.items() if t["task"] == task)
        self.calls.append(task_id)
        if task_id == "vies-01":
            raise RuntimeError("CUDA out of memory")
        return AgentResponse(final_answer=ANSWERS[task_id][0], steps=[], n_steps=3)


def test_evaluate_saves_each_task_and_resumes(tmp_path, monkeypatch):
    import sys

    import agent.evaluate as evaluate
    import agent.react_agent as react_agent

    fake = _FakeAgent()
    monkeypatch.setattr(react_agent.ReActAgent, "from_env", classmethod(lambda cls: fake))
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "eval.json"
    argv = ["evaluate", "--bench", str(BENCH), "--brains", "base", "--out", str(out)]

    monkeypatch.setattr(sys, "argv", [*argv, "--ids", "cambio-01", "vies-01"])
    evaluate.main()
    first = json.loads(out.read_text())["base"]
    assert [r["id"] for r in first["rows"]] == ["cambio-01", "vies-01"]
    assert first["rows"][0]["correct"] and not first["rows"][1]["correct"]
    assert "CUDA out of memory" in first["rows"][1]["crash"]

    fake.calls.clear()
    monkeypatch.setattr(sys, "argv", [*argv, "--ids", "cambio-01", "larga-01", "--resume"])
    evaluate.main()
    assert fake.calls == ["larga-01"]  # cambio-01 was already there
    second = json.loads(out.read_text())["base"]
    assert second["n_tasks"] == 3 and second["by_category"]["larga"] == 1.0


def _eval_step(arguments, ok=True):
    payload = {"name": "evaluate_origin", **({"result": {}} if ok else {"error": "x"})}
    return {"thought": "", "action": {"name": "evaluate_origin", "arguments": arguments},
            "observation": json.dumps(payload)}  # fmt: skip


def test_argument_check_finds_the_wrong_price():
    """The smoke run: base passed declaracion-01 with the consignment value as ex-works price."""
    from agent.analyze import argument_check

    task = TASKS["declaracion-01"]
    reference = task["reference_call"]["arguments"]
    right = {**task, "steps": [_eval_step(reference)]}
    assert argument_check(right) == {"called": True, "ran": True, "exact": True, "errors": []}
    written = {**reference, "product_hs_code": "8450.11", "exw_price_eur": 4200}
    wrong = {**task, "steps": [_eval_step(written)]}
    assert argument_check(wrong)["errors"] == ["precio franco fábrica"]
    assert argument_check({**TASKS["conocimiento-01"], "steps": []}) is None


def test_analysis_report_runs(tmp_path, monkeypatch):
    import sys

    import agent.analyze as analyze
    import agent.evaluate as evaluate
    import agent.react_agent as react_agent

    fake = _FakeAgent()
    monkeypatch.setattr(react_agent.ReActAgent, "from_env", classmethod(lambda cls: fake))
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "phase4_eval.json"
    monkeypatch.setattr(sys, "argv", ["evaluate", "--bench", str(BENCH), "--brains", "base",
                                      "rlm", "--out", str(out)])  # fmt: skip
    evaluate.main()
    text = analyze.report(analyze.load_runs([out]))
    assert "`eval:base`" in text and "| larga |" in text


# Answers of the base brain in the first DGX run (2026-11-22), judged by hand. Two of them
# (veredicto-01, imposible-01) were right and the first version of the checks failed them.
DGX_ANSWERS = [
    ("cambio-01", False, "No, the refrigerators are not originating under the TCA agreement. "
     "The non-originating value is 310.00 euros, which constitutes 51.67% of the ex-works price."),
    ("veredicto-01", True, "The bicycles are not originating under the TCA agreement. The "
     "non-originating value is 200 EUR, which exceeds the 10% tolerance. The product does not "
     "meet the criteria for origin under the MAXNOM 45% EXW rule."),
    ("veredicto-03", False, "The battery packs are not originating under the TCA agreement."),
    ("declaracion-03", True, "The product is not originating under the EU-Korea FTA. The "
     "evaluation shows a percentage of non-originating materials of 51.11%."),
    ("conocimiento-02", False, "El producto es originario según el acuerdo UE-KR. La declaración "
     "de origen no puede ser extendida, ya que el porcentaje de materiales originarios es del "
     "48.57%, que está por debajo de la tolerancia del 10%."),
    ("conocimiento-03", False, "The gas cookers are originating under CETA. The origin "
     "declaration and supporting documents need to be kept for 5 years."),
    ("imposible-01", True, "The origin of the product cannot be determined with the available "
     "tools. Please consult a customs expert or provide additional information to determine if "
     "the product is originating for export to the UK under the TCA agreement."),
    ("imposible-02", False, "The electric coffee machines are originating under the TCA "
     "agreement. The validity period of the origin declaration is not specified."),
]  # fmt: skip


@pytest.mark.parametrize("task_id,right,answer", DGX_ANSWERS, ids=[a[0] for a in DGX_ANSWERS])
def test_real_answers_from_the_dgx(task_id, right, answer):
    assert check_answer(TASKS[task_id], answer) is right


def test_rescore_applies_the_current_checks(tmp_path):
    from agent.evaluate import rescore

    old = {**TASKS["veredicto-01"], "check": {"type": "regex", "value": "never matches xyz"}}
    answer = DGX_ANSWERS[1][2]
    row = {**old, "final_answer": answer, "steps": [], "n_steps": 2, "succeeded": True,
           "tokens_used": 10, "correct": False, "required_tools_used": False}  # fmt: skip
    out = tmp_path / "eval.json"
    out.write_text(json.dumps({"base": {"rows": [row]}}))
    results = rescore(out, list(TASKS.values()))
    assert results["base"]["rows"][0]["correct"] and results["base"]["success_rate"] == 1.0