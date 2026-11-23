"""Calculadora de derivadas e integrales: función serverless para Vercel, con SymPy.

POST /api/calc
  {"op": "diff" | "int" | "defint", "expr": "x^2*sin(x)", "a": "0", "b": "pi", "order": 1}
"""
import json
import multiprocessing as mp
import re
from http.server import BaseHTTPRequestHandler

from sympy import (
    Abs, E, Expr, Integral, N, Symbol, acos, asin, atan, cos, cosh, cot, csc,
    diff, exp, integrate, log, nan, oo, pi, sec, simplify, sin, sinh,
    sqrt, tan, tanh, zoo,
)
from sympy import latex as _latex
from sympy.parsing.sympy_parser import (
    convert_xor, implicit_multiplication_application, parse_expr,
    standard_transformations,
)

TIMEOUT_S = 8            # tiempo máximo de cálculo antes de cortar
MEM_LIMIT = 1024 ** 3    # memoria virtual máxima del proceso que calcula
MAX_LEN = 200
MAX_ORDER = 5
OPS = ("diff", "int", "defint")

# Real: así abs(x) deriva a sign(x) y sqrt(x^2) da |x|, como se espera en un curso de análisis.
x = Symbol("x", real=True)

# Vocabulario permitido. parse_expr usa eval por dentro, así que todo identificador
# que no esté acá se rechaza ANTES de parsear.
NAMES = {
    "x": x, "pi": pi, "e": E, "oo": oo,
    "sin": sin, "cos": cos, "tan": tan, "cot": cot, "sec": sec, "csc": csc,
    "asin": asin, "acos": acos, "atan": atan,
    "arcsin": asin, "arccos": acos, "arctan": atan,
    "sinh": sinh, "cosh": cosh, "tanh": tanh,
    "exp": exp, "log": log, "ln": log, "sqrt": sqrt, "abs": Abs,
}
TRANSFORMS = standard_transformations + (convert_xor, implicit_multiplication_application)

_CHARS = re.compile(r"^[0-9A-Za-z_+\-*/^().,\s]+$")
_IDENT = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")
_BIGNUM = re.compile(r"\d{7,}")


def _parse(text, what="la expresión"):
    s = text.strip() if isinstance(text, str) else ""
    if not s:
        raise ValueError(f"Falta {what}.")
    if len(s) > MAX_LEN:
        raise ValueError(f"{what.capitalize()} es demasiado larga (máximo {MAX_LEN} caracteres).")
    if not _CHARS.match(s) or "__" in s:
        raise ValueError(f"Hay caracteres no permitidos en {what}.")
    if _BIGNUM.search(s):
        raise ValueError("Hay un número demasiado grande.")
    for name in _IDENT.findall(s):
        if name not in NAMES:
            raise ValueError(
                f"No reconozco «{name}». Usá x, pi, e y funciones como sin, cos, exp, ln, sqrt."
            )
    try:
        out = parse_expr(s, local_dict=dict(NAMES), transformations=TRANSFORMS)
    except Exception:
        raise ValueError(f"No pude interpretar {what}. Revisá paréntesis y operadores.") from None
    if not isinstance(out, Expr):
        raise ValueError(f"No pude interpretar {what}.")
    return out


def latex(expr):
    return _latex(expr, ln_notation=True)


def _pretty(expr):
    """Texto plano que se puede volver a pegar en la calculadora (^ y ln)."""
    return re.sub(r"\blog\(", "ln(", str(expr).replace("**", "^"))


def _compute(op, expr_s, a_s, b_s, order):
    f = _parse(expr_s)

    if op == "diff":
        res = simplify(diff(f, x, order))
        d = r"\frac{d}{dx}" if order == 1 else rf"\frac{{d^{{{order}}}}}{{dx^{{{order}}}}}"
        return {
            "ok": True,
            "input_latex": rf"{d}\left({latex(f)}\right)",
            "result_latex": latex(res),
            "result_text": _pretty(res),
        }

    if op == "int":
        res = integrate(f, x)
        if res.has(Integral):
            raise ValueError(
                "SymPy no encontró una primitiva en forma cerrada (puede que no exista con "
                "funciones elementales). Probá la integral definida: esa se puede aproximar numéricamente."
            )
        return {
            "ok": True,
            "input_latex": rf"\int {latex(f)}\,dx",
            "result_latex": latex(res) + " + C",
            "result_text": _pretty(res) + " + C",
        }

    # integral definida
    a = _parse(a_s, "el límite inferior")
    b = _parse(b_s, "el límite superior")
    if a.free_symbols or b.free_symbols:
        raise ValueError("Los límites tienen que ser números (por ejemplo 0, 1, pi, oo).")
    head = rf"\int_{{{latex(a)}}}^{{{latex(b)}}} {latex(f)}\,dx"

    res = integrate(f, (x, a, b))
    if res.has(zoo, nan) or res in (oo, -oo):
        raise ValueError("La integral diverge o no está definida en ese intervalo.")

    if res.has(Integral):
        # Sin primitiva simbólica: aproximación numérica.
        try:
            val = N(Integral(f, (x, a, b)), 12)
        except Exception:
            raise ValueError("No pude calcular esta integral ni exacta ni numéricamente.") from None
        if not (val.is_number and val.is_finite):
            raise ValueError("La integral no converge o no se pudo aproximar.")
        return {
            "ok": True,
            "input_latex": head,
            "result_latex": latex(val),
            "result_text": str(val),
            "approx": True,
        }

    out = {
        "ok": True,
        "input_latex": head,
        "result_latex": latex(res),
        "result_text": _pretty(res),
        "approx": False,
    }
    if res.is_number:
        out["numeric"] = str(N(res, 12))
    return out


def _worker(conn, args):
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (MEM_LIMIT, MEM_LIMIT))
    except Exception:
        pass
    try:
        out = _compute(*args)
    except ValueError as exc:
        out = {"ok": False, "error": str(exc)}
    except Exception:
        out = {"ok": False, "error": "No pude resolver esta expresión."}
    try:
        conn.send(out)
    finally:
        conn.close()


def run(op, expr, a, b, order):
    """Calcula en un proceso aparte para poder cortarlo si SymPy se cuelga."""
    ctx = mp.get_context("fork")
    recv, send = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_worker, args=(send, (op, expr, a, b, order)), daemon=True)
    proc.start()
    send.close()
    try:
        if recv.poll(TIMEOUT_S):
            try:
                return recv.recv()
            except EOFError:
                return {"ok": False, "error": "El cálculo se cortó (¿demasiada memoria?). Probá con algo más simple."}
        return {"ok": False, "error": f"Tardó más de {TIMEOUT_S} s y lo corté. Probá con una expresión más simple."}
    finally:
        if proc.is_alive():
            proc.kill()
        proc.join(1)


class handler(BaseHTTPRequestHandler):
    def _send(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send(405, {"ok": False, "error": "Usá POST con un JSON."})

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length") or 0)
            if not 0 < n <= 4096:
                raise ValueError("Pedido vacío o demasiado grande.")
            data = json.loads(self.rfile.read(n))
            if not isinstance(data, dict):
                raise ValueError("Formato inválido.")
            op = data.get("op")
            if op not in OPS:
                raise ValueError("Operación inválida.")
            try:
                order = int(data.get("order", 1))
            except (TypeError, ValueError):
                raise ValueError("El orden de derivación tiene que ser un entero.") from None
            if not 1 <= order <= MAX_ORDER:
                raise ValueError(f"El orden de derivación va de 1 a {MAX_ORDER}.")
        except (ValueError, json.JSONDecodeError) as exc:
            msg = str(exc) if isinstance(exc, ValueError) and not isinstance(exc, json.JSONDecodeError) else "JSON inválido."
            self._send(400, {"ok": False, "error": msg})
            return

        result = run(op, data.get("expr"), data.get("a"), data.get("b"), order)
        self._send(200 if result.get("ok") else 422, result)
