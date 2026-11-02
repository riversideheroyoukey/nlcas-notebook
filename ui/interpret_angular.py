"""F-9 (angular-momentum coupling): rule-based NL triggers.

Rule-based, never LLM-routed.  A Clebsch–Gordan coefficient carries six quantum
numbers and a 9j symbol nine; BP-6 records that the local supervisor mis-extracts
fields once an operation has four or more, and a *silently wrong quantum number*
produces a plausible-looking coefficient with no error anywhere — the worst
failure mode this notebook can have.  So the numbers are parsed here,
deterministically, and an unparseable instruction falls through rather than
guessing.

Layout mirrors ui/interpret_game.py: one ``_try_*`` per trigger, each returning
the interpreter response dict or None, registered as data in
ui/interpret_pipeline.py's interceptor list.
"""
from __future__ import annotations

import re
from functools import partial

from notebook.runtime import angular_momentum as am
from notebook.runtime.i18n import t
from notebook.runtime.nl_patterns import nlp

# key=value pairs: "j1=1/2", "m_1 = -1/2", "j'=3/2", full-width "Ｊ＝１".
_KV_RE = re.compile(r"([A-Za-zＡ-Ｚａ-ｚ]['′]?_?\d?)\s*[=＝]\s*([-+−]?[\d０-９]+(?:\s*/\s*[\d０-９]+)?)")
_FULLWIDTH = str.maketrans("０１２３４５６７８９／＋－＝Ａ", "0123456789/+-=A")


def _parse_kv(blob: str) -> "dict[str, str]":
    """``"j1=1/2, m1=-1/2"`` → ``{"j1": "1/2", "m1": "-1/2"}``.

    Keys are normalized by dropping ``_`` and mapping ``'``/``′`` to ``p``
    (``j'`` → ``jp``); CASE IS KEPT, because ``j1`` (a constituent) and ``J``
    (the total) are different quantum numbers.
    """
    out: "dict[str, str]" = {}
    for key, value in _KV_RE.findall(blob):
        key = key.translate(_FULLWIDTH).replace("_", "").replace("′", "'")
        key = key.replace("'", "p")
        out[key] = value.translate(_FULLWIDTH).replace("−", "-").replace(" ", "")
    return out


def _pick(args: "dict[str, str]", *names, default=None):
    """First present key among ``names`` (case-sensitive), else ``default``."""
    for n in names:
        if n in args:
            return args[n]
    return default


def _fail(lang, exc, registered) -> dict:
    """Render an :class:`AngularMomentumError` through i18n and stop the step."""
    if isinstance(exc, am.AngularMomentumError):
        message = t(f"angular.{exc.code}", lang, **exc.params)
    else:
        message = str(exc)
    return {"status": "failed", "error": message, "registered_expressions": registered}


async def _run(engine, loop, lang, registered, operation, method, *pargs, **kwargs):
    """Call an engine method off-thread and append whatever it registered."""
    try:
        result = await loop.run_in_executor(
            None, partial(getattr(engine, method), *pargs, **kwargs))
    except Exception as exc:   # bad quantum numbers (i18n'd) or engine failure
        return _fail(lang, exc, registered)
    for ex in (result if isinstance(result, list) else [result]):
        registered.append({"expr_id": ex.display_id,
                           "canonical": ex.canonical, "latex": ex.latex})
    return {"status": "completed", "operation": operation,
            "registered_expressions": registered}


# JP: "j1=1/2, m1=1/2, j2=1/2, m2=-1/2, J=1, M=0 のCG係数を計算して"
# EN: "compute the Clebsch-Gordan coefficient for j1=1/2, m1=1/2, ..., M=0"
async def _try_clebsch_gordan(engine, loop, lang, text, registered) -> "dict | None":
    if nlp("angular_clebsch_gordan", lang).search(text) is None:
        return None
    a = _parse_kv(text)
    needed = (_pick(a, "j1"), _pick(a, "m1"), _pick(a, "j2"), _pick(a, "m2"),
              _pick(a, "J", "j3", "j"), _pick(a, "M", "m3", "m"))
    if any(v is None for v in needed):
        return _fail(lang,
                     am.AngularMomentumError("missing_quantum_numbers",
                                             expected="j1, m1, j2, m2, J, M"),
                     registered)
    return await _run(engine, loop, lang, registered,
                      "clebsch_gordan", "clebsch_gordan_coefficient", *needed)


# JP: "3j記号 (1 2 3; 0 0 0) を計算して" / "6j記号 {1 2 3; 2 1 2}"
# EN: "compute the 9j symbol {1/2 1/2 1; 1/2 1/2 1; 1 1 0}"
async def _try_wigner_symbol(engine, loop, lang, text, registered) -> "dict | None":
    m = nlp("angular_wigner_symbol", lang).search(text)
    if m is None:
        return None
    kind = m.group("kind").translate(_FULLWIDTH) + "j"
    rows = [r.replace(",", " ").split()
            for r in re.split(r"[;；\\\\]+", m.group("body").translate(_FULLWIDTH))
            if r.strip()]
    return await _run(engine, loop, lang, registered,
                      f"wigner_{kind}", "wigner_symbol", kind, rows)


# JP: "I=3/2 と S=1/2 を合成して" / "3/2 と 1/2 の角運動量を合成して"
# EN: "couple angular momenta I=3/2 and S=1/2"
async def _try_couple(engine, loop, lang, text, registered) -> "dict | None":
    m = nlp("angular_couple", lang).search(text)
    if m is None:
        return None
    s1, s2 = _clean_symbols(m.group("s1"), m.group("s2"))
    total = _total_symbol(s1, s2)
    return await _run(engine, loop, lang, registered,
                      "couple_angular_momenta", "couple_angular_momenta",
                      m.group("j1"), m.group("j2"), total, (s1, s2))


_GENERIC_J_RE = re.compile(r"^j_?\d?$", re.IGNORECASE)


def _clean_symbols(s1, s2) -> "tuple[str, str]":
    """Names to label the coupled operators with.

    ``j1=3/2 と j2=1/2`` names nothing (the user just labelled the slots), so
    fall back to the generic ``j_1``, ``j_2``; ``I=3/2 と S=1/2`` does name
    them, and the identity should read ``I·S``.
    """
    out = []
    for raw, default in ((s1, "j_1"), (s2, "j_2")):
        out.append(default if not raw or _GENERIC_J_RE.match(raw) else raw)
    return out[0], out[1]


def _total_symbol(s1: str, s2: str) -> str:
    """Conventional NAME of the total angular momentum — display only.

    ``F`` when a nuclear spin ``I`` is one of the constituents (hyperfine),
    else ``J``.  This picks a letter, never a physical assumption: the coupling
    itself is symmetric in its inputs and the identity keeps the total
    symbolic either way.
    """
    return "F" if "I" in (s1, s2) else "J"


# JP: "j1=1/2, j2=1/2 の |J=1,M=0> を非結合基底で展開して"
# EN: "expand |J=1, M=0> in the uncoupled basis with j1=1/2, j2=1/2"
async def _try_coupled_state(engine, loop, lang, text, registered) -> "dict | None":
    m = nlp("angular_coupled_state", lang).search(text)
    if m is None:
        return None
    a = _parse_kv(text)
    needed = (_pick(a, "j1"), _pick(a, "j2"), _pick(a, "J", "j"), _pick(a, "M", "m"))
    if any(v is None for v in needed):
        return _fail(lang,
                     am.AngularMomentumError("missing_quantum_numbers",
                                             expected="j1, j2, J, M"),
                     registered)
    basis = ((m.group("b1"), m.group("b2"))
             if m.group("b1") and m.group("b2") else None)
    result = await _run(engine, loop, lang, registered,
                        "coupled_state_expansion", "coupled_state_expansion",
                        *needed, basis=basis)
    # The basis symbols are part of the answer, so say which ones were used —
    # especially when the system chose them, which the researcher never asked
    # for and would otherwise have to reverse-engineer from the output.
    if result.get("status") == "completed" and result["registered_expressions"]:
        expr_id = result["registered_expressions"][-1]["expr_id"]
        used = (engine.expressions[expr_id].angular_descriptor or {}).get("basis")
        key = ("angular.expansion_named_note" if basis
               else "angular.expansion_auto_basis_note")
        result["message"] = t(key, lang, expr_id=expr_id,
                              basis=", ".join(used or []))
    return result


# JP: "^2P_{3/2} のランデのg因子を求めて" / "L=1, S=1/2, J=3/2 のg因子"
# EN: "Lande g-factor of ^2P_{3/2}"
async def _try_lande_g(engine, loop, lang, text, registered) -> "dict | None":
    m = nlp("angular_lande_g", lang).search(text)
    if m is None:
        return None
    spec = m.group("spec").strip()
    a = _parse_kv(spec)
    if {"L", "S", "J"} <= set(a):
        return await _run(engine, loop, lang, registered, "lande_g_factor",
                          "lande_g_factor", a["L"], a["S"], a["J"])
    try:
        am.parse_term_symbol(spec)
    except am.AngularMomentumError:
        return None                       # not a level spec — let others try
    return await _run(engine, loop, lang, registered, "lande_g_factor",
                      "lande_g_factor", term_symbol=spec)


# JP: "j=1/2, m=1/2, k=1, q=0, j'=3/2, m'=1/2 のWigner-Eckartの定理を書き下して"
# EN: "Wigner-Eckart theorem for j=1/2, m=1/2, k=1, q=0, j'=3/2, m'=1/2"
async def _try_wigner_eckart(engine, loop, lang, text, registered) -> "dict | None":
    if nlp("angular_wigner_eckart", lang).search(text) is None:
        return None
    a = _parse_kv(text)
    needed = (_pick(a, "j"), _pick(a, "m"), _pick(a, "k"), _pick(a, "q"),
              _pick(a, "jp"), _pick(a, "mp"))
    if any(v is None for v in needed):
        return _fail(lang,
                     am.AngularMomentumError("missing_quantum_numbers",
                                             expected="j, m, k, q, j', m'"),
                     registered)
    return await _run(engine, loop, lang, registered,
                      "wigner_eckart", "wigner_eckart_matrix_element", *needed)
