#!/usr/bin/env python3
"""Demo de ARCA: las cuatro fases en pantalla, en castellano y sin JSON.

Habla con la misma API que sirve el proyecto (`uv run arca-api`), así que lo que se ve es el
sistema de verdad, no una maqueta. Una pantalla por fase, y entre pantalla y pantalla espera a
que pulses Intro, para que la demo la lleves tú y no el reloj.

    uv run python demo.py                      # las cuatro fases
    uv run python demo.py --fase 4             # solo una (se puede repetir: --fase 3 --fase 4)
    uv run python demo.py --problema 11        # qué problema de test resuelve la fase 1
    uv run python demo.py --tarea conocimiento # qué tarea hace el agente (fase 4)
    uv run python demo.py --auto               # sin pausas, para grabar de una tirada
    uv run python demo.py --base http://localhost:8811   # la API en otro puerto

La fase 4 tiene dos tareas, variantes en castellano de las del banco de la fase 4:

* ``declaracion``: evaluar unas lavadoras para el Reino Unido y emitir la declaración de origen
  (``declaracion-01``). La factura lleva la hora, porque ``issue_origin_statement`` no
  sobrescribe nunca una declaración ya emitida y la demo se puede repetir.
* ``conocimiento``: evaluar unos hornos para Japón y responder una pregunta de normativa
  (``kn-05``): el enrutador busca en la base de conocimiento antes del primer paso.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from textwrap import TextWrapper

ANCHO = 78
TEST = Path("rlm/data/test.jsonl")

PREGUNTA_RAG = "¿Cuánto tiempo es válida una declaración de origen en el acuerdo con Japón?"
PREGUNTA_TOOLS = "¿Cuál es la regla de origen de la partida 8507.60 en el acuerdo con Reino Unido?"

TAREA_DECLARACION = (
    "Fabricamos lavadoras (SA 8450.11) en La Rioja y las vendemos en el Reino Unido. Precio franco "
    "fábrica 390 €. Materiales: motor eléctrico de China (8501.40), 70 €; módulo electrónico de "
    "control de China (8537.10), 45 €; bomba de desagüe de China (8413.81), 15 €; tambor y partes "
    "de lavadora de Turquía (8450.90), 35 €; chapa de acero de la UE con declaración del proveedor "
    "(7209.17), 50 €. Si son originarias según el TCA, emite la declaración de origen para la "
    "factura {factura}: exportador Electrodomésticos Ribera SL, Logroño, valor del envío 4.200 €."
)
TAREA_CONOCIMIENTO = (
    "Fabricamos hornos eléctricos (SA 8516.60) en Navarra y los exportamos a Japón. Precio franco "
    "fábrica 480 €, precio FOB 505 €. Materiales: resistencias de China (8516.80), 60 €; placas "
    "de control de China (8537.10), 85 €; ventiladores de China (8414.59), 30 €; chapa de acero "
    "fabricada en la UE con declaración del proveedor (7209.17), 55 €. ¿Son originarios según el "
    "acuerdo UE-Japón? ¿Y hasta qué valor puede entrar en Japón un paquete pequeño como "
    "originario sin prueba de origen?"
)


class C:
    """Colores ANSI, vacíos si la salida no es una terminal."""

    AZUL = BOLD = DIM = VERDE = ROJO = AMBAR = OFF = ""

    @classmethod
    def encender(cls) -> None:
        cls.AZUL, cls.BOLD, cls.DIM = "\033[38;5;39m", "\033[1m", "\033[2m"
        cls.VERDE, cls.ROJO, cls.AMBAR = "\033[38;5;35m", "\033[38;5;203m", "\033[38;5;214m"
        cls.OFF = "\033[0m"


def portada(base: str) -> None:
    print(f"\n{C.BOLD}{C.AZUL}  ARCA{C.OFF}  ¿es originario mi producto?")
    print(f"{C.DIM}  Asistente de reglas de origen para pymes exportadoras{C.OFF}")
    print(f"{C.DIM}  {base}{C.OFF}\n")


def seccion(numero: str, titulo: str, subtitulo: str) -> None:
    print(f"\n{C.AZUL}{'─' * ANCHO}{C.OFF}")
    print(f"{C.BOLD}{C.AZUL} {numero}{C.OFF}  {C.BOLD}{titulo}{C.OFF}   {C.DIM}{subtitulo}{C.OFF}")
    print(f"{C.AZUL}{'─' * ANCHO}{C.OFF}\n")


def campo(etiqueta: str, texto: str, color: str = "", sangria: str = "    ") -> None:
    print(f"{C.DIM}  {etiqueta}{C.OFF}")
    envoltura = TextWrapper(
        width=ANCHO, initial_indent=sangria, subsequent_indent=sangria, break_long_words=False
    )
    for parrafo in str(texto).strip().split("\n"):
        print(f"{color}{envoltura.fill(parrafo) if parrafo.strip() else ''}{C.OFF}")
    print()


class Espera:
    """Un punto que gira mientras la API piensa, para que no parezca colgado."""

    def __init__(self, texto: str) -> None:
        self.texto, self._parar = texto, threading.Event()
        self._hilo = threading.Thread(target=self._girar, daemon=True)

    def _girar(self) -> None:
        marcos, inicio = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏", time.monotonic()
        for i in range(10**9):
            if self._parar.is_set():
                return
            segundos = time.monotonic() - inicio
            sys.stdout.write(
                f"\r  {C.AMBAR}{marcos[i % len(marcos)]}{C.OFF} {self.texto} "
                f"{C.DIM}{segundos:4.0f}s{C.OFF}"
            )
            sys.stdout.flush()
            time.sleep(0.1)

    def __enter__(self) -> Espera:
        if sys.stdout.isatty():
            self._hilo.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._parar.set()
        if sys.stdout.isatty():
            self._hilo.join(timeout=1)
            sys.stdout.write("\r" + " " * (ANCHO + 10) + "\r")
            sys.stdout.flush()


def llamar(base: str, ruta: str, cuerpo: dict, espera: str) -> dict | None:
    peticion = urllib.request.Request(
        f"{base}{ruta}",
        data=json.dumps(cuerpo).encode("utf-8"),
        headers={"content-type": "application/json"},
    )
    with Espera(espera):
        inicio = time.monotonic()
        try:
            with urllib.request.urlopen(peticion, timeout=600) as respuesta:
                datos = json.loads(respuesta.read())
        except urllib.error.HTTPError as error:
            print(f"  {C.ROJO}{ruta} devolvió {error.code}{C.OFF}: {error.read()[:200].decode()}\n")
            return None
        except (urllib.error.URLError, TimeoutError) as error:
            print(f"  {C.ROJO}No contesta {base}{ruta}{C.OFF}: {error}")
            print(f"  {C.DIM}¿Está levantada la API? uv run arca-api{C.OFF}\n")
            return None
    datos["_segundos"] = time.monotonic() - inicio
    return datos


def estado(base: str) -> bool:
    try:
        with urllib.request.urlopen(f"{base}/health", timeout=20) as respuesta:
            fases = json.loads(respuesta.read())["phases"]
    except Exception as error:  # noqa: BLE001 - sin API no hay demo que enseñar
        print(f"  {C.ROJO}No contesta {base}{C.OFF}: {error}")
        print(f"  {C.DIM}Levántala con: uv run arca-api{C.OFF}\n")
        return False
    nombres = {
        "reasoning": "Razonamiento",
        "tools": "Herramientas",
        "rag": "Conocimiento",
        "agent": "Agente",
    }
    for i, (clave, titulo) in enumerate(nombres.items(), 1):
        listo = fases.get(clave, {}).get("status") == "ready"
        marca = f"{C.VERDE}●{C.OFF}" if listo else f"{C.DIM}○{C.OFF}"
        print(
            f"   {marca} {C.DIM}Fase {i}{C.OFF}  {titulo:14}"
            f"{'' if listo else C.DIM + ' pendiente' + C.OFF}"
        )
    print()
    return True


def pausa(auto: bool) -> None:
    if auto:
        time.sleep(1.5)
        return
    try:
        input(f"{C.DIM}   ╌╌ Intro para seguir ╌╌{C.OFF}")
    except (EOFError, KeyboardInterrupt):
        sys.exit(0)


# ----------------------------------------------------------------------------------- las fases


def fase_rag(base: str) -> None:
    seccion("FASE 3", "Conocimiento normativo", "pregunta libre, respuesta con cita")
    campo("Pregunta", PREGUNTA_RAG)
    datos = llamar(base, "/rag", {"question": PREGUNTA_RAG}, "buscando en el corpus…")
    if datos is None:
        return
    campo("Respuesta", re.sub(r"\s*\[[0-9a-f]{10}-\d{4}\]", "", datos["answer"]), C.BOLD)
    citadas = {c["id"]: c for c in datos["chunks"]}
    for cita in datos["citations"]:
        trozo = citadas.get(cita)
        if trozo is None:
            continue
        frase = max(trozo["text"].split("\n"), key=len)[:300]
        campo(f"Fuente · {trozo['source']}", f"«{frase.strip()}»", C.DIM)
    print(
        f"{C.DIM}    {len(datos['chunks'])} pasajes recuperados ({datos['retriever']}) · "
        f"{datos['_segundos']:.1f} s{C.OFF}\n"
    )


def fase_tools(base: str) -> None:
    seccion("FASE 2", "Herramientas", "consulta el arancel oficial y aplica la regla")
    campo("Pregunta", PREGUNTA_TOOLS)
    datos = llamar(base, "/tools", {"query": PREGUNTA_TOOLS}, "llamando a las herramientas…")
    if datos is None:
        return
    for llamada in datos.get("tool_calls", []):
        argumentos = ", ".join(f"{k}={v}" for k, v in llamada["arguments"].items())
        marca = f"{C.VERDE}✓{C.OFF}" if llamada.get("ok") else f"{C.ROJO}✗{C.OFF}"
        print(f"  {C.DIM}Herramienta{C.OFF}")
        print(
            f"    {marca} {C.BOLD}{llamada['name']}{C.OFF}({argumentos})"
            f"   {C.DIM}{llamada.get('latency_ms', 0):.0f} ms{C.OFF}"
        )
        conjuntos = (llamada.get("result") or {}).get("rule_sets") or []
        for conjunto in conjuntos:
            legibles = len(conjunto.get("criteria") or [])
            manual = len(conjunto.get("not_machine_readable") or [])
            etiqueta = (conjunto.get("subdivision") or "—")[:52]
            print(
                f"      {C.DIM}{conjunto.get('number', '?')}){C.OFF} {etiqueta}"
                f"   {C.DIM}{legibles} aplicable(s), {manual} a mano{C.OFF}"
            )
        print()
    campo("Respuesta", datos["answer"], C.BOLD)
    print(f"{C.DIM}    {datos.get('turns', '?')} turnos · {datos['_segundos']:.1f} s{C.OFF}\n")


def _veredicto(texto: str) -> dict | None:
    bloque = re.search(r"\{.*\}", str(texto), re.S)
    if bloque is None:
        return None
    try:
        return json.loads(bloque.group())
    except json.JSONDecodeError:
        return None


def _resumen_problema(enunciado: str) -> str:
    producto = re.search(r"«([^»]+)»\s*\(subpartida ([\d.]+)\)", enunciado)
    destino = re.search(r"exporta a ([A-ZÁÉÍÓÚÑ][\w\s]+?) el producto", enunciado)
    exw = re.search(r"\(EXW\):\s*([\d.,]+)\s*€", enunciado)
    regla = re.search(r"Regla de origen de la partida \d+:\s*([^\n]+)", enunciado)
    partes = []
    if producto:
        partes.append(f"{producto.group(1)} ({producto.group(2)})")
    if destino:
        partes.append(f"a {destino.group(1).strip()}")
    if exw:
        partes.append(f"EXW {exw.group(1)} €")
    cabecera = " · ".join(partes) or enunciado.split("\n")[0][:ANCHO]
    return f"{cabecera}\nRegla: {regla.group(1).strip()}" if regla else cabecera


def fase_reasoning(base: str, indice: int) -> None:
    seccion("FASE 1", "Razonamiento", f"problema {indice} del conjunto de test, nunca visto")
    if not TEST.exists():
        print(f"  {C.ROJO}No encuentro {TEST}{C.OFF}\n")
        return
    lineas = TEST.read_text(encoding="utf-8").splitlines()
    if not 0 <= indice < len(lineas):
        print(f"  {C.ROJO}El test tiene {len(lineas)} problemas{C.OFF}\n")
        return
    problema = json.loads(lineas[indice])
    campo("Caso", _resumen_problema(problema["question"]))

    datos = llamar(base, "/reasoning", {"question": problema["question"]}, "razonando…")
    if datos is None:
        return

    pensado = (datos.get("thinking") or "").strip()
    if pensado:
        frases = [f.strip() for f in re.split(r"(?<=[.:])\s+", pensado) if 25 < len(f.strip())]
        extracto = " ".join(frases[-3:])[:420]
        campo("Cómo lo razona (final del razonamiento)", f"…{extracto}", C.DIM)

    suyo, oro = _veredicto(datos.get("answer")), _veredicto(problema["answer"])
    if suyo is None or oro is None:
        campo("Respuesta", datos.get("answer", ""), C.BOLD)
        return

    def linea(etiqueta: str, v: dict, color: str) -> None:
        pct = v.get("pct_no_originario")
        pct = f"{float(pct):.2f} %" if pct not in (None, "") else "—"
        texto = "originario" if v.get("originario") else "NO originario"
        print(
            f"  {C.DIM}{etiqueta:22}{C.OFF}{color}{texto:16}{C.OFF}"
            f"{C.DIM}criterio{C.OFF} {v.get('criterio', '—'):10}"
            f"{C.DIM}no originario{C.OFF} {pct}"
        )

    print()
    linea("Veredicto del modelo", suyo, C.BOLD)
    linea("Respuesta correcta", oro, C.DIM)
    print()

    iguales = [
        ("veredicto", bool(suyo.get("originario")) == bool(oro.get("originario"))),
        ("criterio", str(suyo.get("criterio")) == str(oro.get("criterio"))),
        (
            "porcentaje",
            abs(
                float(suyo.get("pct_no_originario") or -1)
                - float(oro.get("pct_no_originario") or 0)
            )
            < 0.005,
        ),
    ]
    fallos = [nombre for nombre, bien in iguales if not bien]

    def enumerar(nombres: list[str]) -> str:
        return " y ".join(filter(None, [", ".join(nombres[:-1]), nombres[-1]])) if nombres else ""

    if not fallos:
        print(f"  {C.VERDE}✓ Acierta los tres campos.{C.OFF}")
    else:
        aciertos = enumerar([n for n, bien in iguales if bien])
        print(f"  {C.AMBAR}▸ Acierta {aciertos}, pero falla {enumerar(fallos)}.{C.OFF}")
        if "porcentaje" in fallos:
            print(
                f"  {C.DIM}  Es el fallo típico: ordena bien el razonamiento y clasifica bien "
                f"los materiales,\n    pero la aritmética se le escapa. Por eso el veredicto lo "
                f"calcula una herramienta\n    determinista (fase 2) y no el modelo.{C.OFF}"
            )
    print(
        f"\n{C.DIM}    {datos.get('tokens_generated', '?')} tokens · "
        f"{datos['_segundos']:.1f} s · {datos.get('model', '')}{C.OFF}\n"
    )


def _carga(observacion: str | None) -> dict:
    """El JSON de una observación del agente, sin las notas que el bucle pone delante o detrás."""
    texto = observacion or ""
    inicio = texto.find("{")
    if inicio < 0:
        return {}
    try:
        datos, _ = json.JSONDecoder().raw_decode(texto, inicio)
    except json.JSONDecodeError:
        return {}
    return datos if isinstance(datos, dict) else {}


def _euros(valor: object) -> str:
    try:
        return f"{float(valor):,.2f} €".replace(",", "X").replace(".", ",").replace("X", ".")
    except (TypeError, ValueError):
        return str(valor)


def _paso(numero: int, paso: dict) -> None:
    """Una línea por llamada: qué herramienta, con qué y qué devolvió."""
    accion = paso.get("action") or {}
    nombre, argumentos = accion.get("name"), accion.get("arguments") or {}
    datos = _carga(paso.get("observation"))
    resultado, error = datos.get("result"), datos.get("error")
    marca = f"{C.ROJO}✗{C.OFF}" if error else f"{C.VERDE}✓{C.OFF}"
    cabecera = f"  {C.DIM}{numero:>2}{C.OFF} {marca} {C.BOLD}{nombre}{C.OFF}"
    sangria = "       "

    if nombre == "search_knowledge_base":
        enrutado = re.match(r"\[knowledge router, p=([\d.]+)\]", paso.get("thought") or "")
        quien = (
            f"{C.AMBAR}enrutador{C.OFF} {C.DIM}(p = {float(enrutado.group(1)):.2f}){C.OFF}"
            if enrutado
            else f"{C.DIM}el modelo{C.OFF}"
        )
        print(f"{cabecera}  ← {quien}")
        print(f"{sangria}{C.DIM}«{str(argumentos.get('query', ''))[:90]}»{C.OFF}")
        fuentes = list(dict.fromkeys(re.findall(r"\(from ([^)]+)\)", str(resultado or ""))))
        for fuente in fuentes:
            print(f"{sangria}{C.DIM}· {fuente}{C.OFF}")
    elif nombre == "evaluate_origin":
        materiales = argumentos.get("materials") or []
        print(
            f"{cabecera}  {C.DIM}{argumentos.get('agreement', '?')} · "
            f"{argumentos.get('product_hs_code', '?')} · EXW "
            f"{_euros(argumentos.get('exw_price_eur'))} · {len(materiales)} materiales{C.OFF}"
        )
        if isinstance(resultado, dict):
            si = resultado.get("originating")
            texto = f"{C.VERDE}originario{C.OFF}" if si else f"{C.ROJO}NO originario{C.OFF}"
            print(
                f"{sangria}{texto}  {C.DIM}criterio{C.OFF} {resultado.get('criterion', '—')}  "
                f"{C.DIM}no originario{C.OFF} {resultado.get('pct_non_originating', '—')} %  "
                f"{C.DIM}motor determinista · {resultado.get('evaluation_id', '')}{C.OFF}"
            )
    elif nombre == "get_exchange_rate":
        print(f"{cabecera}  {C.DIM}{argumentos}{C.OFF}")
        if isinstance(resultado, dict):
            print(
                f"{sangria}{C.DIM}{_euros(resultado.get('amount_eur'))} "
                f"(BCE, {resultado.get('month', '')}){C.OFF}"
            )
    elif nombre == "issue_origin_statement":
        print(
            f"{cabecera}  {C.DIM}factura {argumentos.get('invoice_number', '?')} · envío "
            f"{_euros(argumentos.get('consignment_value_eur'))}{C.OFF}"
        )
        if isinstance(resultado, dict):
            print(
                f"{sangria}{C.VERDE}declaración emitida{C.OFF}  {C.DIM}{resultado.get('path')}"
                f"{C.OFF}"
            )
    elif nombre == "final_answer":
        print(f"{cabecera}")
        return
    else:
        print(f"{cabecera}  {C.DIM}{json.dumps(argumentos, ensure_ascii=False)[:90]}{C.OFF}")
    if error:
        envoltura = TextWrapper(width=ANCHO, initial_indent=sangria, subsequent_indent=sangria)
        print(f"{C.AMBAR}{envoltura.fill(str(error)[:300])}{C.OFF}")


def fase_agente(base: str, tarea: str, cerebro: str) -> None:
    titulo = {
        "declaracion": "evalúa y emite la declaración",
        "conocimiento": "normativa y cálculo en la misma petición",
    }[tarea]
    seccion("FASE 4", "Agente", titulo)
    if tarea == "declaracion":
        texto = TAREA_DECLARACION.format(factura=time.strftime("LV-%m%d-%H%M%S"))
    else:
        texto = TAREA_CONOCIMIENTO
    campo("Tarea", texto)
    datos = llamar(base, "/agent", {"task": texto, "brain": cerebro}, "el agente trabaja…")
    if datos is None:
        return
    print(f"{C.DIM}  Pasos{C.OFF}")
    llamadas = [p for p in datos.get("steps", []) if p.get("action")]
    for numero, paso in enumerate(llamadas, 1):
        _paso(numero, paso)
    print()
    respuesta = datos.get("final_answer", "")
    # Con thinking, rara vez la respuesta trae delante el razonamiento cerrado por </think>
    # (1 de 20 tareas en la v3): se enseña lo que va detrás, que es la respuesta.
    campo("Respuesta", respuesta.rsplit("</think>", 1)[-1], C.BOLD)
    print(
        f"{C.DIM}    {datos.get('n_steps', '?')} pasos · {datos.get('tokens_used', 0)} tokens · "
        f"{datos['_segundos']:.1f} s · {datos.get('model', '')}{C.OFF}"
    )
    print(
        f"{C.DIM}    El veredicto sale del motor (fase 1), la regla y el cambio de las "
        f"herramientas (fase 2),\n    la normativa de la base de conocimiento (fase 3); el "
        f"agente decide el orden.{C.OFF}\n"
    )


def main() -> None:
    analizador = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    puerto = os.environ.get("ARCA_API_PORT") or "8811"
    analizador.add_argument(
        "--base", default=os.environ.get("ARCA_API_URL") or f"http://localhost:{puerto}"
    )
    analizador.add_argument("--fase", type=int, choices=(1, 2, 3, 4), action="append")
    analizador.add_argument("--problema", type=int, default=11)
    analizador.add_argument(
        "--tarea",
        choices=("declaracion", "conocimiento", "ambas"),
        default="ambas",
        help="qué hace el agente en la fase 4",
    )
    analizador.add_argument("--cerebro", choices=("rlm", "base", "thinking"), default="rlm")
    analizador.add_argument("--auto", action="store_true", help="sin pausas, para grabar")
    analizador.add_argument("--sin-color", action="store_true")
    argumentos = analizador.parse_args()

    if not argumentos.sin_color and sys.stdout.isatty():
        C.encender()
    base = argumentos.base.rstrip("/")

    print("\033[2J\033[H" if sys.stdout.isatty() else "")
    portada(base)
    if not estado(base):
        sys.exit(1)

    fases = argumentos.fase or [3, 2, 1, 4]
    tareas = ["declaracion", "conocimiento"] if argumentos.tarea == "ambas" else [argumentos.tarea]
    for numero in fases:
        if numero == 4:
            for tarea in tareas:
                pausa(argumentos.auto)
                fase_agente(base, tarea, argumentos.cerebro)
            continue
        pausa(argumentos.auto)
        if numero == 3:
            fase_rag(base)
        elif numero == 2:
            fase_tools(base)
        else:
            fase_reasoning(base, argumentos.problema)

    print(f"{C.AZUL}{'─' * ANCHO}{C.OFF}")
    print(
        f"{C.DIM}  Todo lo anterior sale de la misma API: uv run arca-api, o docker compose up "
        f"api.{C.OFF}\n"
    )


if __name__ == "__main__":
    main()