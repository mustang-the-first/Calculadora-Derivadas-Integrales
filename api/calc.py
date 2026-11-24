"""Calculadora de derivadas e integrales: función serverless para Vercel, con SymPy.

POST /api/calc
  {"op": "diff" | "int" | "defint", "expr": "x^2*sin(x)", "a": "0", "b": "pi", "order": 1}
"""
import json
import multiprocessing as mp
import os
import re
from http.server import BaseHTTPRequestHandler

from sympy import (
    Abs, Dummy, E, Expr, Integral, Mul, N, Pow, Symbol, acos, asin, atan, cos,
    cosh, cot, csc, diff, exp, integrate, limit, log, nan, oo, pi, sec,
    simplify, sin, sinh, sqrt, sympify, tan, tanh, zoo,
)
from sympy import latex as _latex
from sympy.integrals import manualintegrate as mi
from sympy.parsing.sympy_parser import (
    convert_xor, implicit_multiplication_application, parse_expr,
    standard_transformations,
)

TIMEOUT_S = 8            # tiempo máximo de cálculo antes de cortar
STEPS_TIMEOUT_S = 4      # tiempo extra para armar el paso a paso (el resultado ya se envió)
MAX_STEPS = 80           # más pasos que esto no se muestran
MEM_LIMIT = 1024 ** 3    # memoria virtual máxima del proceso que calcula
MAX_LEN = 200
MAX_ORDER = 5
OPS = ("diff", "int", "defint")
PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "public", "index.html")

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
    # log(x) y ln(x) son el logaritmo natural; log10(x) es en base 10 y log(x, b) en base b.
    "log10": lambda u: log(u, 10),
    # Nombres en castellano.
    "sen": sin, "tg": tan, "ctg": cot, "cosec": csc, "senh": sinh, "tgh": tanh,
    "arcsen": asin, "arctg": atan,
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


# ------------------------------------------------------------------ paso a paso
# Cada paso es {"depth": sangría, "text": explicación, "latex": fórmula o None}.

Uf = Symbol("u", real=True)  # variable genérica para las derivadas "de tabla"

_FN_ES = {
    sin: "seno", cos: "coseno", tan: "tangente", cot: "cotangente", sec: "secante",
    csc: "cosecante", asin: "arcoseno", acos: "arcocoseno", atan: "arcotangente",
    sinh: "seno hiperbólico", cosh: "coseno hiperbólico", tanh: "tangente hiperbólica",
    log: "logaritmo natural", Abs: "valor absoluto",
}


def _tex(expr):
    """LaTeX de expr; los símbolos auxiliares de SymPy (Dummy) se muestran con su nombre."""
    expr = sympify(expr)
    dummies = expr.atoms(Dummy)
    if dummies:
        expr = expr.xreplace({d: Symbol(d.name) for d in dummies})
    return latex(expr)


def _st(depth, text, tex=None):
    return {"depth": depth, "text": text, "latex": tex}


def _vname(sym):
    """Nombre de una variable para usar en el texto (sin el guion bajo de los Dummy)."""
    name = getattr(sym, "name", None) or str(sym)
    return "θ" if name == "theta" else name


def _dx(expr):
    return r"\frac{d}{dx}\left(" + _tex(expr) + r"\right)"


def _dsteps(f, d, out):
    """Agrega a out los pasos para derivar f respecto de x, con sangría d."""
    r = diff(f, x)
    lhs = _dx(f)

    if not f.has(x):
        out.append(_st(d, "La derivada de una constante es 0.", lhs + " = 0"))
        return
    if f == x:
        out.append(_st(d, "La derivada de x es 1.", lhs + " = 1"))
        return

    if f.is_Add:
        out.append(_st(d, "Regla de la suma: se deriva cada término por separado.",
                       lhs + " = " + " + ".join(_dx(t) for t in f.args)))
        for t in f.args:
            _dsteps(t, d + 1, out)
        return

    if f.is_Mul:
        c, g = f.as_independent(x, as_Add=False)
        if c != 1:
            out.append(_st(d, f"Regla del factor constante: la constante {_pretty(c)} sale de la derivada.",
                           lhs + " = " + _tex(c) + r" \cdot " + _dx(g)))
            _dsteps(g, d + 1, out)
            return
        neg = [a for a in f.args if a.is_Pow and a.exp.is_negative and a.has(x)]
        if neg:
            den = Mul(*[Pow(a.base, -a.exp) for a in neg])
            num = Mul(*[a for a in f.args if a not in neg])
            if num.has(x):
                out.append(_st(d, "Regla del cociente: (u/v)′ = (u′·v − u·v′) / v².",
                               "u = " + _tex(num) + r", \quad v = " + _tex(den)))
                _dsteps(num, d + 1, out)
                _dsteps(den, d + 1, out)
                out.append(_st(
                    d, "Se reemplaza en la fórmula:",
                    lhs + r" = \frac{\left(" + _tex(diff(num, x)) + r"\right)\left(" + _tex(den)
                    + r"\right) - \left(" + _tex(num) + r"\right)\left(" + _tex(diff(den, x))
                    + r"\right)}{\left(" + _tex(den) + r"\right)^{2}}"))
                return
        a, b = f.args[0], Mul(*f.args[1:])
        out.append(_st(d, "Regla del producto: (u·v)′ = u′·v + u·v′.",
                       "u = " + _tex(a) + r", \quad v = " + _tex(b)))
        _dsteps(a, d + 1, out)
        _dsteps(b, d + 1, out)
        out.append(_st(
            d, "Se reemplaza en la fórmula:",
            lhs + r" = \left(" + _tex(diff(a, x)) + r"\right)\left(" + _tex(b)
            + r"\right) + \left(" + _tex(a) + r"\right)\left(" + _tex(diff(b, x)) + r"\right)"))
        return

    if f.is_Pow:
        b, e = f.args
        if not b.has(x):
            if e == x:
                out.append(_st(d, "Derivada de una exponencial: (aˣ)′ = aˣ·ln(a).", lhs + " = " + _tex(r)))
            else:
                out.append(_st(d, f"Regla de la cadena con la exponencial: (aᵘ)′ = aᵘ·ln(a)·u′, con u = {_pretty(e)}.",
                               "u = " + _tex(e)))
                _dsteps(e, d + 1, out)
                out.append(_st(d, "Se reemplaza en la fórmula:",
                               lhs + " = " + _tex(f) + r" \cdot " + _tex(log(b))
                               + r" \cdot \left(" + _tex(diff(e, x)) + r"\right)"))
            return
        if not e.has(x):
            if b == x:
                out.append(_st(d, "Regla de la potencia: (xⁿ)′ = n·xⁿ⁻¹.", lhs + " = " + _tex(r)))
            else:
                out.append(_st(d, f"Regla de la cadena con la potencia: (uⁿ)′ = n·uⁿ⁻¹·u′, con u = {_pretty(b)} y n = {_pretty(e)}.",
                               "u = " + _tex(b)))
                _dsteps(b, d + 1, out)
                out.append(_st(d, "Se reemplaza en la fórmula:",
                               lhs + " = " + _tex(e * b ** (e - 1))
                               + r" \cdot \left(" + _tex(diff(b, x)) + r"\right)"))
            return
        g = exp(e * log(b))
        if not g.is_Pow:
            out.append(_st(d, "La base y el exponente dependen de x: se reescribe uᵛ = e^(v·ln u).",
                           _tex(f) + " = " + _tex(g)))
            _dsteps(g, d + 1, out)
            return

    if isinstance(f, exp):
        u = f.args[0]
        if u == x:
            out.append(_st(d, "La derivada de eˣ es eˣ.", lhs + " = " + _tex(f)))
        else:
            out.append(_st(d, f"Regla de la cadena: (eᵘ)′ = eᵘ·u′, con u = {_pretty(u)}.", "u = " + _tex(u)))
            _dsteps(u, d + 1, out)
            out.append(_st(d, "Se reemplaza en la fórmula:",
                           lhs + " = " + _tex(f) + r" \cdot \left(" + _tex(diff(u, x)) + r"\right)"))
        return

    if len(f.args) == 1 and f.func in _FN_ES:
        u = f.args[0]
        nombre = _FN_ES[f.func]
        outer = diff(f.func(Uf), Uf).subs(Uf, u)
        if u == x:
            out.append(_st(d, f"Derivada de tabla ({nombre}).", lhs + " = " + _tex(outer)))
        else:
            out.append(_st(d, f"Regla de la cadena ({nombre}): se deriva la función exterior y se multiplica por u′, con u = {_pretty(u)}.",
                           "u = " + _tex(u)))
            _dsteps(u, d + 1, out)
            out.append(_st(d, "Se reemplaza en la fórmula:",
                           lhs + " = " + _tex(outer) + r" \cdot \left(" + _tex(diff(u, x)) + r"\right)"))
        return

    out.append(_st(d, "Se deriva con las reglas del cálculo.", lhs + " = " + _tex(r)))


def _diff_steps(f, order):
    out = []
    cur = f
    base = 1 if order > 1 else 0
    for k in range(1, order + 1):
        if order > 1:
            out.append(_st(0, "Primera derivada." if k == 1 else f"Derivada de orden {k}: se deriva el resultado anterior."))
        _dsteps(cur, base, out)
        cur = simplify(diff(cur, x))
        if len(out) > MAX_STEPS:
            return None
    name = "f'(x)" if order == 1 else "f^{(" + str(order) + ")}(x)"
    out.append(_st(0, "Resultado", name + " = " + _tex(simplify(diff(f, x, order)))))
    return out


def _int_tex(expr, var):
    return r"\int " + _tex(expr) + r"\,d" + _tex(var)


def _signed_ints(terms, var):
    parts = []
    for i, t in enumerate(terms):
        t = sympify(t)
        neg = i > 0 and t.could_extract_minus_sign()
        parts.append((" - " if neg else (" + " if i else "")) + _int_tex(-t if neg else t, var))
    return "".join(parts)


def _minus_int(rest, var):
    rest = sympify(rest)
    if rest.could_extract_minus_sign():
        return " + " + _int_tex(-rest, var)
    return " - " + _int_tex(rest, var)


def _isteps(rule, d, out):
    """Agrega los pasos de una regla de manualintegrate. Devuelve False si algún tramo no se puede explicar."""
    t = type(rule).__name__
    v = rule.variable
    if t == "DontKnowRule":
        return False
    if t == "AlternativeRule":
        for alt in rule.alternatives:
            tmp = []
            if _isteps(alt, d, tmp):
                out.extend(tmp)
                return True
        return False

    big_i = _int_tex(rule.integrand, v) if rule.integrand is not None else ""

    if t == "ConstantRule":
        out.append(_st(d, "La integral de una constante es la constante por la variable.",
                       big_i + " = " + _tex(rule.eval())))
        return True
    if t == "ConstantTimesRule":
        out.append(_st(d, f"Se saca la constante {_pretty(rule.constant)} fuera de la integral.",
                       big_i + " = " + _tex(rule.constant) + " " + _int_tex(rule.other, v)))
        return _isteps(rule.substep, d + 1, out)
    if t == "PowerRule":
        out.append(_st(d, "Regla de la potencia: ∫ xⁿ dx = xⁿ⁺¹ / (n + 1).", big_i + " = " + _tex(rule.eval())))
        return True
    if t == "AddRule":
        out.append(_st(d, "Se integra cada término por separado.",
                       big_i + " = " + _signed_ints([s.integrand for s in rule.substeps], v)))
        return all(_isteps(s, d + 1, out) for s in rule.substeps)
    if t == "URule":
        uf, uv = rule.u_func, rule.u_var
        out.append(_st(d, "Sustitución: se elige u y se calcula du.",
                       "u = " + _tex(uf) + r", \quad du = " + _tex(diff(uf, v)) + r"\,d" + _tex(v)))
        out.append(_st(d, "La integral queda en términos de u:",
                       big_i + " = " + _int_tex(rule.substep.integrand, uv)))
        if not _isteps(rule.substep, d + 1, out):
            return False
        out.append(_st(d, f"Se vuelve a la variable {_vname(v)} reemplazando u.", _tex(rule.eval())))
        return True
    if t == "PartsRule":
        if rule.second_step is None:
            return False
        u, dv = rule.u, rule.dv
        vv = rule.v_step.eval()
        out.append(_st(d, "Integración por partes: ∫ u dv = u·v − ∫ v du.",
                       "u = " + _tex(u) + r", \quad dv = " + _tex(dv) + r"\,d" + _tex(v)))
        out.append(_st(d, "Se deriva u y se integra dv:",
                       "du = " + _tex(diff(u, v)) + r"\,d" + _tex(v) + r", \quad v = " + _tex(vv)))
        out.append(_st(d, "Se reemplaza en la fórmula:",
                       big_i + " = " + _tex(u * vv) + _minus_int(rule.second_step.integrand, v)))
        return _isteps(rule.second_step, d + 1, out)
    if t == "CyclicPartsRule":
        out.append(_st(d, "Integración por partes repetida: la integral original reaparece y se despeja.", big_i))
        for pr in rule.parts_rules:
            out.append(_st(d + 1, "Por partes:", "u = " + _tex(pr.u) + r", \quad dv = " + _tex(pr.dv) + r"\,d" + _tex(v)))
        out.append(_st(d, "Despejando la integral original:", big_i + " = " + _tex(rule.eval())))
        return True
    if t in ("RewriteRule", "CompleteSquareRule"):
        txt = "Se reescribe el integrando para poder integrarlo." if t == "RewriteRule" else "Se completa el cuadrado."
        out.append(_st(d, txt, big_i + " = " + _int_tex(rule.rewritten, v)))
        return _isteps(rule.substep, d + 1, out)
    if t == "TrigSubstitutionRule":
        out.append(_st(d, "Sustitución trigonométrica.", _tex(v) + " = " + _tex(rule.func)))
        out.append(_st(d, f"La integral queda en términos de {_vname(rule.theta)}:",
                       big_i + " = " + _int_tex(rule.rewritten, rule.theta)))
        if not _isteps(rule.substep, d + 1, out):
            return False
        out.append(_st(d, f"Se vuelve a la variable {_vname(v)} deshaciendo la sustitución.", _tex(rule.eval())))
        return True

    try:
        res = rule.eval()
    except Exception:
        return False
    out.append(_st(d, "Integral de tabla.", big_i + " = " + _tex(res)))
    return True


def _antiderivative_steps(f):
    """(pasos, F) con F una primitiva de f verificada por derivación; (None, None) si no se puede."""
    rule = mi.integral_steps(f, x)
    if rule.contains_dont_know():
        return None, None
    out = []
    if not _isteps(rule, 0, out) or len(out) > MAX_STEPS:
        return None, None
    F = rule.eval()
    chk = diff(F, x) - f
    if simplify(chk) != 0 and chk.equals(0) is not True:
        return None, None
    return out, F


def _int_steps(f):
    out, F = _antiderivative_steps(f)
    if out is None:
        return None
    out.append(_st(0, "Resultado: se agrega la constante de integración C.",
                   _int_tex(f, x) + " = " + _tex(F) + " + C"))
    return out


def _eval_at(F, p, side):
    """F evaluada en p (con límite si hace falta); None si no es un valor finito."""
    try:
        if p.is_infinite:
            val = limit(F, x, p)
        else:
            val = F.subs(x, p)
            if val.has(nan, zoo, oo, -oo):
                val = limit(F, x, p, side)
        if val.has(nan, zoo) or val in (oo, -oo):
            return None
        return val
    except Exception:
        return None


def _defint_steps(f, a, b, ref):
    out, F = _antiderivative_steps(f)
    if out is None:
        return None
    head = r"\int_{" + _tex(a) + r"}^{" + _tex(b) + "} " + _tex(f) + r"\,dx"
    out.append(_st(0, "Regla de Barrow: ∫ₐᵇ f(x) dx = F(b) − F(a), con F una primitiva de f.",
                   head + r" = \left[" + _tex(F) + r"\right]_{" + _tex(a) + "}^{" + _tex(b) + "}"))
    fb, fa = _eval_at(F, b, "-"), _eval_at(F, a, "+")
    if fb is None or fa is None:
        return None
    val = simplify(fb - fa)
    if ref is not None:  # el paso a paso tiene que coincidir con el resultado ya calculado
        try:
            want = complex(ref)
            if abs(complex(N(val, 15)) - want) > 1e-8 * max(1.0, abs(want)):
                return None
        except Exception:
            return None
    out.append(_st(0, "Se evalúa en los límites (con límites si hace falta).",
                   r"F\left(" + _tex(b) + r"\right) - F\left(" + _tex(a) + r"\right) = \left("
                   + _tex(fb) + r"\right) - \left(" + _tex(fa) + r"\right)"))
    out.append(_st(0, "Resultado", head + " = " + _tex(val)))
    return out


def _steps(op, expr_s, a_s, b_s, order, result):
    f = _parse(expr_s)
    if op == "diff":
        steps = _diff_steps(f, order)
    elif op == "int":
        steps = _int_steps(f)
    else:
        steps = _defint_steps(f, _parse(a_s, "el límite inferior"), _parse(b_s, "el límite superior"),
                              result.get("numeric"))
    if steps is not None and len(steps) > MAX_STEPS:
        return None
    return steps


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
        conn.send(out)  # primero el resultado, así no se pierde si el paso a paso se cuelga
        if out.get("ok") and not out.get("approx"):
            try:
                steps = _steps(*args, out)
            except Exception:
                steps = None
            conn.send({"steps": steps})
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
                result = recv.recv()
            except EOFError:
                return {"ok": False, "error": "El cálculo se cortó (¿demasiada memoria?). Probá con algo más simple."}
            if result.get("ok") and not result.get("approx"):
                try:
                    if recv.poll(STEPS_TIMEOUT_S):
                        result["steps"] = recv.recv().get("steps")
                except (EOFError, OSError):
                    pass
            return result
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

    def _send_page(self):
        # Respaldo: si Vercel no sirve public/ por su cuenta, la función entrega la página.
        try:
            with open(PAGE, "rb") as fh:
                body = fh.read()
        except OSError:
            self._send(404, {"ok": False, "error": "No se encontró la página."})
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.split("?")[0] in ("/", "/index.html"):
            self._send_page()
            return
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
