"""CAS コード → 利用者書式 の逆方向：添字の復元。

docs/developer_manuals/32_user_cas_boundary_repair_plan.md の Phase 4。
順方向を ``expr_lowering`` が持つのに対して、この モジュールが戻り側を持つ。

An undeclared subscripted symbol resolves to an ``IndexedSymbol`` and lowers to a
CAS function call — ``V_i`` becomes ``V(i)``, ``omega_0`` becomes ``omega(0)``.
That is the documented lowering policy (docs/07), but the CAS then returns its
results in that spelling, and the timeline was showing the researcher
``\\omega\\left(0\\right)`` for what they had typed as ``omega_0`` — CAS syntax in
the notebook, against Core Principle #1.

Restoring it needs a record of which bases were written with a subscript, because
a result canonical ``a*V(i)+b`` parses as an ordinary function call and no longer
says that ``V`` was ever an index base.  Hence the two halves here:

* :func:`record_indexed_bases` — walks the IR of what the researcher REGISTERED
  and notes each base with its index count, into ``SymbolContext.indexed_bases``;
* :func:`restore_indexed_bases` — rewrites ``V\\left(i\\right)`` back to ``V_{i}``
  in the display LaTeX, for those bases only.

Both live in one module because they are one mechanism: a change to what is
recorded is meaningless without the matching change to what is restored.

**Backend-neutral** (Core Principle #10): no CAS name, token or import appears
here.  The input is LaTeX and a SymbolContext.
"""
from __future__ import annotations

import re

#: Elementary function names that are never index bases.  Reuses the resolver's
#: CAS-NEUTRAL set so the two agree; a backend's own spellings are irrelevant
#: here (this module never sees CAS code, only display LaTeX and the IR).
def _math_function_names() -> frozenset:
    from notebook.runtime.semantic_resolver import COMMON_FUNCTION_NAMES
    return COMMON_FUNCTION_NAMES


_MATH_FUNCTION_NAMES = _math_function_names()

#: Index arguments a restored subscript may contain: one or more comma-separated
#: names or digits.  Anything else — arithmetic, a nested call — is a real
#: function argument, so ``sin(x+1)`` is never read as ``sin`` indexed by ``x+1``.
_INDEX_ARGS_RE = re.compile(
    r"^[A-Za-z\\][A-Za-z0-9\\]*(?:\s*,\s*[A-Za-z0-9\\]+)*$|^[0-9]+$")


def _flat_base_name(node) -> "str | None":
    r"""The canonical NAME of an indexed base: ``Symbol`` or ``Decorated``.

    A decorated base was skipped, so ``\hat\rho_{n\sigma,m\sigma}`` was never
    recorded as an index base and its CAS spelling `rho_hat(n, sigma, …)` stayed
    on screen — for the most common shape in this corpus (a hatted operator with
    indices), which is precisely what this module exists to restore.
    """
    from notebook.models.ir import Decorated, Symbol as _Sym
    if isinstance(node, _Sym):
        return node.name
    if isinstance(node, Decorated):
        inner = _flat_base_name(node.base)
        return f"{inner}_{node.decoration}" if inner else None
    return None


def record_symbol_usage(ir_ast, registry) -> None:
    """Note how *ir_ast* uses each base: subscripted, applied, or both.

    Call with the IR of a REGISTERED expression — the researcher's own writing.
    Results coming back from the CAS must not be used: their ``V(i)`` is
    indistinguishable from a genuine function call, which is the whole problem.

    Writes ``SymbolContext.indexed_bases`` (base → index count) and
    ``called_bases`` (names applied to arguments).  A base in both is ambiguous;
    :func:`restore_indexed_bases` then leaves it alone.
    """
    if ir_ast is None:
        return
    from notebook.models.ir import FunctionCall, IndexedSymbol, Symbol

    indexed_bases = registry.indexed_bases
    called = registry.called_bases
    seen: set[int] = set()

    def _walk(node) -> None:
        if node is None or id(node) in seen:
            return
        seen.add(id(node))
        _base_name = _flat_base_name(node.base) if isinstance(node, IndexedSymbol) else None
        if _base_name:
            n = len(node.indices or [])
            if n:
                indexed_bases[_base_name] = n
        elif isinstance(node, FunctionCall) and isinstance(node.func, Symbol):
            # Only UNDECLARED names matter: a declaration already answers the
            # question, and the builtin maths functions are never index bases.
            nm = node.func.name
            if (nm and nm not in called and registry.get(nm) is None
                    and nm not in _MATH_FUNCTION_NAMES):
                called.append(nm)
        # model_fields is read off the CLASS: reading it from the instance is
        # deprecated in pydantic 2.11 and removed in 3.
        for field in getattr(type(node), "model_fields", ()) or ():
            value = getattr(node, field, None)
            if isinstance(value, (list, tuple)):
                for item in value:
                    if hasattr(type(item), "model_fields"):
                        _walk(item)
            elif hasattr(type(value), "model_fields"):
                _walk(value)

    _walk(ir_ast)


def _base_spellings(base: str) -> str:
    r"""A regex alternation over every TeX command that spells *base*.

    Three Greek letters have two commands for one canonical name (``\epsilon``
    and ``\varepsilon``; likewise φ and θ), and the display side prints the one
    the RESEARCHER wrote (PHI-GLYPH-SPLIT).  Matching only the canonical
    spelling therefore missed exactly the cases where the researcher used the
    variant: ``\varepsilon_{n\sigma}`` came back from the CAS as
    ``\varepsilon\left(n , \sigma\right)`` and stayed that way — CAS syntax on
    screen, which is what this module exists to remove.
    """
    from notebook.runtime.latex_converter import TEX_GREEK_TO_NAME
    names = {base} | {cmd.lstrip("\\") for cmd, canon in TEX_GREEK_TO_NAME.items()
                      if canon == base}
    return "(?:" + "|".join(re.escape(n) for n in
                            sorted(names, key=len, reverse=True)) + ")"


def restore_indexed_bases(s: str, registry) -> str:
    """Turn ``V\\left(i\\right)`` back into ``V_{i}`` in display LaTeX.

    Only bases recorded by :func:`record_indexed_bases` are touched, and only
    when the argument list looks like indices.  A base already DECLARED is left
    to the caller's own registry pass, which knows its kind.

    A base that is also used as a CALLED name is skipped entirely: ``a_i`` and
    ``a_i(k)`` are different objects by design (CLAUDE.md), so restoring
    ``a(i)`` → ``a_i`` beside a genuine ``a_i(k)`` would show two quantities
    under one notation.  Leaving the CAS spelling visible is the honest failure.
    """
    bases = getattr(registry, "indexed_bases", None)
    if not bases:
        return s
    called = set(getattr(registry, "called_bases", ()) or ())
    for base, arity in bases.items():
        if not base or registry.get(base) is not None:
            continue
        if base in called:
            # Written BOTH ways in this session (``V_i`` and ``V(x)``), and the
            # CAS spelling of the two is identical — restoring would show the
            # function under the index notation.  Leave the CAS form visible;
            # the interpreter asks the researcher which V is meant, and the
            # answer is a declaration, which takes this branch out of play.
            continue
        esc = _base_spellings(base)
        # Every spelling a called name of this family can already have here:
        # `a_i(`, `a_{i}(`, with or without \left, with a Greek base's backslash.
        if re.search(rf"(?<![A-Za-z0-9_])\\?{esc}"
                     rf"(?:_[A-Za-z0-9]+|_\{{[^}}]*\}})\s*(?:\\left\s*)?\(", s):
            continue

        def _sub(m: "re.Match", _n=arity) -> str:
            args = m.group(3).strip().replace(r"\,", ",")
            if len(args.split(",")) != _n or not _INDEX_ARGS_RE.match(args):
                return m.group(0)
            # The name is put back AS MATCHED, not as the canonical: the two
            # spellings of ε/φ/θ are the researcher's choice and the display
            # keeps it (PHI-GLYPH-SPLIT).
            return f"{m.group(1)}{m.group(2)}_{{{args.replace(' ', '')}}}"

        # A Greek base arrives as its TeX command (`omega` is rendered
        # `\omega`), so the leading backslash is CAPTURED and put back rather
        # than excluded — without that, omega_0 stayed `\omega\left(0\right)`,
        # which is the very leak this exists to close.
        s = re.sub(rf"(?<![A-Za-z0-9_])(\\?)({esc})\s*"
                   rf"(?:\\left\s*\(|\()(.*?)(?:\\right\s*\)|\))", _sub, s)
    return s


#: The three Greek letters with two LaTeX spellings for two glyphs, and the
#: variant Maxima's ``tex1`` answers in.  Measured 2026-08-16 over all 34 Greek
#: names — exactly these three, so the table is the measurement, not a guess.
GREEK_GLYPH_VARIANTS: dict[str, tuple[str, str]] = {
    "phi":     (r"\phi", r"\varphi"),
    "epsilon": (r"\epsilon", r"\varepsilon"),
    "theta":   (r"\theta", r"\vartheta"),
}


def record_greek_glyphs(latex: str, registry: SymbolContext) -> None:
    r"""Note which spelling of φ/ε/θ the researcher wrote — FIRST USE ONLY.

    The same "first input is the declaration" rule as `spin_display_styles`.
    Call with the researcher's own LaTeX; a CAS RESULT must never be used,
    since its spelling is the backend's, which is the whole problem.
    """
    if not latex:
        return
    styles = registry.greek_glyph_styles
    for name, (plain, variant) in GREEK_GLYPH_VARIANTS.items():
        if name in styles:
            continue
        # Order is NOT load-bearing here, and saying it was would be wrong:
        # the BACKSLASH separates them, so `\varphi` does not contain `\phi`
        # (measured — a mutation that reversed the order changed nothing).
        # The variant is still tried first because that is the reading a human
        # checks first, not because the code needs it.
        if re.search(re.escape(variant) + r"(?![A-Za-z])", latex):
            styles[name] = variant
        elif re.search(re.escape(plain) + r"(?![A-Za-z])", latex):
            styles[name] = plain


def restore_greek_glyphs(s: str, registry: SymbolContext) -> str:
    r"""Write φ/ε/θ back in the spelling the researcher used.

    Nothing happens for a letter never written: the record is empty and the
    backend's spelling stands.  A researcher who wrote ``\varphi`` keeps
    ``\varphi`` — this restores what was WRITTEN, it does not pick a favourite.
    """
    styles = getattr(registry, "greek_glyph_styles", None)
    if not styles:
        return s
    for name, (plain, variant) in GREEK_GLYPH_VARIANTS.items():
        want = styles.get(name)
        if not want:
            continue
        other = variant if want == plain else plain
        s = re.sub(re.escape(other) + r"(?![A-Za-z])", want.replace("\\", "\\\\"), s)
        # …and the BARE name too (`epsilon`, no backslash).  A backend can答え
        # with either: Julia's latexify writes `- epsilon_{m} {c}_{m}`, and the
        # command form is only produced LATER in the same display pass — so a
        # restore that only knew `\epsilon` saw nothing to do and the researcher's
        # `\varepsilon` came back as `\epsilon` (measured 2026-08-22, 利用者報告).
        # Converting it here also takes it out of the later bare-name rule's
        # reach, which excludes anything already behind a backslash.
        s = re.sub(r"(?<![A-Za-z\\])" + re.escape(name) + r"(?![A-Za-z])",
                   want.replace("\\", "\\\\"), s)
    return s
