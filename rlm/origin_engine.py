"""Motor de origen preferencial: ¿es originario este producto según esta regla?

Función pura, sin azar ni texto: ``evaluate`` recibe una regla, un producto y una lista de
materiales, y devuelve un ``Evaluation``. Lo usan el generador de problemas, el verificador y,
en la fase 2, la herramienta ``evaluate_origin``.

Semántica fijada (hay que mantenerla igual en motor, enunciado y verificador):

1. ``pct_no_originario`` es siempre el valor total de materiales no originarios entre el
   precio franco fábrica (EXW), gane el criterio que gane.
2. Cuentan como originarios los materiales de la UE o del país socio (acumulación bilateral)
   que tengan declaración de proveedor. Los de terceros países, no.
3. Origen desconocido, o proveedor UE/socio sin declaración: se trata como NO originario.
4. El motor devuelve el conjunto de criterios de la regla que se cumplen. ``answer`` lleva uno
   canónico (orden de prioridad fijo) y el conjunto completo va en ``branches["criterios_validos"]``
   para que el verificador acepte cualquiera.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any


ORIGINS = ("UE", "SOCIO", "TERCERO", "DESCONOCIDO")
ORIGINATING_ORIGINS = ("UE", "SOCIO")
CRITERIA_PRIORITY = ("WO", "CC", "CTH", "CTSH", "MAXNOM", "RVC")
CLASSIFICATION_DIGITS = {"CC": 2, "CTH": 4, "CTSH": 6}
VALUE_CRITERIA = ("MAXNOM", "RVC")
INSUFFICIENT_OPERATIONS = ("embalaje", "etiquetado", "pintado", "limpieza")
SUFFICIENT_OPERATIONS = ("fabricacion", "montaje", "mecanizado")


def pct(percent: int) -> Fraction:
    """Un porcentaje entero como fracción exacta (50 -> 1/2). Se evita el float en los umbrales."""
    return Fraction(percent, 100)


def _frac(x: Any) -> Fraction:
    return x if isinstance(x, Fraction) else Fraction(str(x))


@dataclass(frozen=True)
class Material:
    name: str
    code: str  # subpartida SA de 6 dígitos, sin punto
    origin: str  # UE | SOCIO | TERCERO | DESCONOCIDO
    value: Fraction
    declared: bool = False  # declaración de proveedor (solo tiene sentido para UE/SOCIO)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Material:
        return cls(data["name"], data["code"], data["origin"], _frac(data["value"]), data["declared"])

    @property
    def originating(self) -> bool:
        return self.origin in ORIGINATING_ORIGINS and self.declared


@dataclass(frozen=True)
class Alt:
    """Una alternativa de la regla. Una regla "CTH o MaxNOM 50 %" tiene dos."""

    kind: str  # WO | CC | CTH | CTSH | MAXNOM | RVC
    threshold: Fraction | None = None  # fracción 0-1, solo MAXNOM y RVC
    basis: str = "EXW"  # EXW | FOB, solo MAXNOM y RVC
    excluded_headings: tuple[str, ...] = ()  # "salvo a partir de la partida X"

    def __post_init__(self) -> None:
        if self.kind not in CRITERIA_PRIORITY:
            raise ValueError(f"criterio desconocido: {self.kind}")
        if self.kind in VALUE_CRITERIA and self.threshold is None:
            raise ValueError(f"{self.kind} necesita umbral")


@dataclass(frozen=True)
class Rule:
    rule_id: str
    heading: str  # partida de 4 dígitos
    agreement: str
    alternatives: tuple[Alt, ...]
    source: str = "placeholder"  # propuesta | api | placeholder


@dataclass
class Evaluation:
    originating: bool
    criterion: str  # canónico, o NINGUNO
    satisfied: list[str]  # todos los criterios de la regla que se cumplen
    non_originating_value: Fraction
    pct_non_originating: float  # sobre EXW, ver semántica 1
    flags: dict[str, bool]  # las ramas "trampa" que se han recorrido
    trace: list[str]  # material a material, para la recompensa de trazabilidad


def _check_alternative(
    alt: Alt,
    product_code: str,
    exw: Fraction,
    fob: Fraction,
    non_orig: list[Material],
    total: Fraction,
    tolerance: Fraction,
) -> tuple[bool, bool, bool, bool]:
    """Devuelve (cumple, usó_tolerancia, umbral_exacto, tolerancia_no_amplía_umbral)."""
    if alt.kind == "WO":
        return not non_orig, False, False, False

    if alt.kind in CLASSIFICATION_DIGITS:
        digits = CLASSIFICATION_DIGITS[alt.kind]
        failing = [
            m
            for m in non_orig
            if m.code[:digits] == product_code[:digits] or m.code[:4] in alt.excluded_headings
        ]
        if not failing:
            return True, False, False, False
        failing_value = sum((m.value for m in failing), Fraction(0))
        limit = tolerance * exw
        if failing_value <= limit:
            return True, True, failing_value == limit, False
        return False, False, False, False

    # MAXNOM / RVC: la tolerancia nunca ensancha un umbral de valor
    denominator = exw if alt.basis == "EXW" else fob
    assert alt.threshold is not None
    limit = alt.threshold * denominator if alt.kind == "MAXNOM" else (1 - alt.threshold) * denominator
    ok = total <= limit
    would_pass_with_tolerance = (not ok) and total <= limit + tolerance * denominator
    return ok, False, total == limit, would_pass_with_tolerance


def evaluate(
    rule: Rule,
    product_code: str,
    exw: Any,
    fob: Any,
    materials: list[Material],
    operations: list[str] | tuple[str, ...] = (),
    tolerance: Any = Fraction(0),
) -> Evaluation:
    """Aplica la regla a un producto. Es el ``solve`` real: todo lo demás es muestreo y texto."""
    exw, fob, tolerance = _frac(exw), _frac(fob), _frac(tolerance)
    non_orig = [m for m in materials if not m.originating]
    total = sum((m.value for m in non_orig), Fraction(0))
    heading = product_code[:4]
    excluded = {h for alt in rule.alternatives for h in alt.excluded_headings}

    flags = {
        "material_misma_partida": any(m.code[:4] == heading for m in non_orig),
        "subdivision_distinta": any(
            m.code[:4] == heading and m.code[:6] != product_code[:6] for m in non_orig
        ),
        "excepcion_partida": any(m.code[:4] in excluded and m.code[:4] != heading for m in non_orig),
        "proveedor_sin_declaracion": any(
            m.origin in ORIGINATING_ORIGINS and not m.declared for m in materials
        ),
        "origen_desconocido": any(m.origin == "DESCONOCIDO" for m in materials),
        "operacion_insuficiente": bool(operations)
        and all(op in INSUFFICIENT_OPERATIONS for op in operations),
        "tolerancia_usada": False,
        "umbral_exacto": False,
        "tolerancia_no_amplia_maxnom": False,
    }

    trace = []
    for m in materials:
        if m.originating:
            reason = "originario"
        elif m.origin in ORIGINATING_ORIGINS:
            reason = "no originario (sin declaración de proveedor)"
        elif m.origin == "DESCONOCIDO":
            reason = "no originario (origen desconocido)"
        else:
            reason = "no originario (tercer país)"
        trace.append(f"{m.name} [{m.code}] {float(m.value):.2f}: {reason}")

    pct_value = round(float(total * 100 / exw), 2)

    if flags["operacion_insuficiente"]:
        trace.append("Solo operaciones insuficientes: no confieren origen")
        return Evaluation(False, "NINGUNO", [], total, pct_value, flags, trace)

    satisfied: set[str] = set()
    used_tolerance: list[bool] = []
    for alt in rule.alternatives:
        ok, used_tol, exact, trap = _check_alternative(
            alt, product_code, exw, fob, non_orig, total, tolerance
        )
        flags["umbral_exacto"] |= exact
        flags["tolerancia_no_amplia_maxnom"] |= trap
        if ok:
            satisfied.add(alt.kind)
            used_tolerance.append(used_tol)
        trace.append(f"{alt.kind}: {'cumple' if ok else 'no cumple'}" + (" (con tolerancia)" if used_tol else ""))
    flags["tolerancia_usada"] = bool(used_tolerance) and all(used_tolerance)

    ordered = sorted(satisfied, key=CRITERIA_PRIORITY.index)
    criterion = ordered[0] if ordered else "NINGUNO"
    return Evaluation(bool(ordered), criterion, ordered, total, pct_value, flags, trace)
