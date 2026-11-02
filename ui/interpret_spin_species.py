"""F-13 (multi-species spin): rule-based NL triggers.

Two instructions: declare each site's spin magnitude, and take an
infinite-temperature expectation / moment of a registered spin expression with
those spins.

Rule-based for the same reason as F-9: the numbers ARE the physics, and a
mis-read spin produces a plausible wrong trace with no error anywhere.  The
site→spin pairs are read deterministically from the instruction, and an
unparseable one falls through rather than guessing.
"""
from __future__ import annotations

from functools import partial

from notebook.runtime.nl_patterns import nlp
from ui.interpret_common import number_word

_ZENKAKU = str.maketrans("０１２３４５６７８９／", "0123456789/")


def _pairs(text: str, lang: str) -> "dict[str, str]":
    """All site→spin pairs in the instruction.

    Both group spellings are accepted (``site``/``spin`` and
    ``site2``/``spin2``) so a language whose pattern has two alternative word
    orders — English's "site 1 has spin 1/2" and "spin 1/2 on site 1" — needs
    one regex, not two passes.
    """
    out: "dict[str, str]" = {}
    # **The NUCLIDE is read first, because it is the more specific reading.**
    # `サイト1のスピンは 13C` — the number pattern happily matches the `13` and
    # records spin 13 (measured 2026-08-22, the first version of this did
    # exactly that).  A mass number glued to an element symbol is not a
    # magnitude, and the pattern that can tell them apart must run first.
    from notebook.runtime.spin_value_set import find_nuclide, format_numerator
    for m in nlp("spin_site_nuclide_pair", lang).finditer(text):
        site = (m.groupdict().get("site") or "").translate(_ZENKAKU)
        nuc = find_nuclide(m.groupdict().get("nuclide") or "")
        if site and nuc:
            out[site] = format_numerator(int(nuc["two_s"]))
    for m in nlp("spin_site_spin_pair", lang).finditer(text):
        g = m.groupdict()
        site = g.get("site") or g.get("site2")
        spin = g.get("spin") or g.get("spin2")
        if site and spin and site.translate(_ZENKAKU) not in out:
            out[site.translate(_ZENKAKU)] = spin.translate(_ZENKAKU).replace(" ", "")
    return out


# JP: "サイト1のスピンを1/2、サイト2のスピンを1とする"
# EN: "site 1 has spin 1/2 and site 2 has spin 1" / "spin 1/2 on site 1, ..."
async def _try_declare_site_spins(engine, loop, lang, text, registered) -> "dict | None":
    if nlp("spin_site_spin_declare", lang).search(text) is None:
        return None
    spins = _pairs(text, lang)
    if not spins:
        return None
    try:
        step = await loop.run_in_executor(
            None, partial(engine.declare_site_spins, spins))
    except Exception as exc:                      # invalid site index or spin
        return {"status": "failed", "error": str(exc),
                "registered_expressions": registered}
    return {"status": "completed", "operation": "declare_site_spins",
            "message": step.visible_action,
            "registered_expressions": registered}


# JP: "E1 の2次モーメントを求めて" / "E1 の期待値を求めて"
# EN: "compute the 2nd moment of E1" / "expectation value of E1"
async def _try_spin_expectation(engine, loop, lang, text, registered) -> "dict | None":
    m = nlp("spin_expectation", lang).search(text)
    if m is None:
        return None
    eid = m.group("eid")
    if eid not in engine.expressions or eid not in engine._active_expr_ids():
        return None
    order = m.group("order")
    # 変換は `number_word` 1つ（全角も漢数字も語の序数も、そこが読む）。
    power = (number_word(order) if order else 1) or 1
    try:
        expr = await loop.run_in_executor(
            None, partial(engine.spin_expectation, eid, power))
    except Exception as exc:   # not a spin expression, or a bad moment order
        return {"status": "failed", "error": str(exc),
                "registered_expressions": registered}
    registered.append({"expr_id": expr.display_id,
                       "canonical": expr.canonical, "latex": expr.latex})
    return {"status": "completed", "operation": "spin_expectation",
            "expr_id": expr.display_id,
            "registered_expressions": registered}
