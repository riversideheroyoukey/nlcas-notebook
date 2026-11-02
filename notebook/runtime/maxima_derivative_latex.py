r"""Inverse of the derivative DISPLAY forms this system generates (BP-9 / H-10).

Two producers emit LaTeX for an unevaluated derivative:

* Maxima's own ``tex1`` (after ``clean_latex``)::

      'diff(f,z,2)        -> \frac{d^2}{d z^2} f
      'diff(A_z,t,1,z,1)  -> \frac{d^2}{d t d z} A_{z}

* ``lhs_formatting._format_provenance_lhs`` (the H-1 LHS label)::

      \frac{\partial phi}{\partial x}            \frac{dy}{dx}
      \frac{\partial^{2} phi}{\partial z\partial t}
      \frac{\partial}{\partial x} f(x, t)
      \frac{d^{3}}{dx^{2}dy}\left(a*b+c\right)

* ``maxima_backend_handler._apply_dot_notation_to_latex`` (the dot notation a
  time-dependent quantity is displayed with)::

      'diff(q,t,1) -> \dot{q}      'diff(q,t,3) -> \dddot{q}
      'diff(q,t,2) -> \ddot{q}     'diff(q,t,4) -> {q}^{(4)}

  ``\dot``/``\ddot`` are read by the shared accent table (``q_dot``/``q_ddot``,
  atoms whose meaning is mode-dependent — docs/19).  Orders 3 and up have no
  accent there, and :func:`restore_dot_display_latex` reads them here.

The first two write the denominator variables in DIFFERENTIATION ORDER (stated
in ``_format_provenance_lhs``'s docstring), so the inverse is well defined.

Without the inverse those strings fall through to the generic ``\frac{a}{b}``
→ ``(a)/(b)`` rule and become **silently wrong** Maxima: ``d`` and ``\partial``
turn into ordinary symbols, so ``\frac{\partial phi}{\partial x}`` reduces to
``phi/x`` — a valid expression with no error and the wrong value.  That is what
BP-9 records, and why everything here refuses to guess: a fraction is rewritten
only when it parses EXACTLY as a derivative and the denominator's orders add up
to the numerator's, and only when that reading is unique.  Anything else is
returned verbatim and stays a division.
"""
from __future__ import annotations

import re

from notebook.runtime.latex_brackets import PAREN_PAIRS, balanced, brace_group
from typing import Callable, Iterator

# A variable / operand name: base, then any number of subscript groups
# (``a_k_s`` is one name), then any PRIMES.  The base must not use ``\w`` — that
# swallows the '_' of ``A_{z}`` and strands the brace group.
#
# The trailing ``'*`` is the same rule the parser has (``f'(x)``; CLAUDE.md,
# "Math Input Conventions").  Without it the scan stopped ONE CHARACTER short of
# the end of ``\partial y'`` — so the denominator was never fully consumed and
# the researcher was told there is "no separator between d and a multi-letter
# name", a rule they had not broken, about a display THIS MODULE had generated
# (measured 2026-09-24, PRIMED-VAR-NOT-A-VARIABLE).
# 2026-09-24: the first alternative is new.  Once the derivative LABEL began
# spelling names the researcher's way (DERIV-DISPLAY-CANONICAL-NAME), the
# denominator could hold `d\bar{y}` — and without the braced argument the scan
# stopped at `\bar`, so `\partial f / \partial \bar{y}` degraded into the
# product `\partial*f/(\partial*y_bar)`.  **A widening opens a new exit.**
_NAME = (r"(?:\\[A-Za-z]+\{[^{}]+\}"          # \bar{y}, \hat{H} — a command WITH its argument
         r"|\\?[A-Za-z][A-Za-z0-9]*)"           # theta, \theta, f
         r"(?:_\{[^{}]+\}|_[A-Za-z0-9]+)*'*")

# ``d`` or ``\partial``.  \partial is matched first so the alternation does not
# stop at the 'd' of a following name.
_DEP = r"(?:\\partial|d)"

# ``^{2}`` or ``^2`` or nothing.
_ORD = r"(?:\^\{(\d+)\}|\^(\d+))?"

_FRAC = "\\frac"

# One denominator factor: dep, optional space, a name, optional order.
_FACTOR_RE = re.compile(r"\s*(?P<dep>" + _DEP + r")(?P<gap>\s*)(?P<var>" + _NAME
                        + r")" + _ORD)
# The head of a numerator that CARRIES its operand: dep, optional order, optional
# space.  What follows is read by _trailing_operand — the SAME scanner the
# operator form uses — rather than by a name pattern.  It used to be
# ``…(?P<operand>_NAME)\s*$``, and _NAME cannot describe an argument list, so a
# numerator whose operand was CALLED did not match this shape at all and the
# fraction stayed an ordinary division: measured 2026-08-11,
# ``\frac{d y(t)}{d t}`` simplified to ``y(t)/t`` and
# ``\frac{\partial \psi(x,t)}{\partial t}`` to ``psi(x,t)/t``, both reported
# SUCCESS — the exact cancellation BP-9 exists to prevent, still happening for
# every operand that is not a bare name (which is to say, for most wave
# functions and trajectories).  One scanner, because "what is being
# differentiated" is one question.
_NUM_HEAD_RE = re.compile(
    r"^\s*(?P<dep>" + _DEP + r")" + _ORD + r"(?P<gap>\s*)")
# A bare operator numerator: dep and an optional order, nothing else.
_NUM_OPERATOR_RE = re.compile(r"^\s*" + _DEP + _ORD + r"\s*$")

# A name safe to read as "differential d + name" when NOTHING separates the two:
# one letter plus digits/subscripts (dy, dV, dV_x, dx_1).  A longer bare name is
# refused — `\frac{delta}{dx}` and `\frac{data}{depth}` would otherwise become
# 'diff(elta,x,1) and 'diff(ata,epth,1), the same silent-wrong-answer failure
# BP-9 is about.  This mirrors the project's no-auto-split rule for juxtaposition
# (CLAUDE.md, "Math Input Conventions"): kx / ky / nk are ordinary two-letter
# physics names, so splitting them is never safe without an explicit separator.
# A PRIME does not make the base ambiguous: `dy'` is `d` + `y'` by exactly the
# reasoning that makes `dy` be `d` + `y` — one letter plus decorations — and the
# prime cannot start a name.  Measured 2026-09-24: the moment a primed variable
# started reaching `differentiate`, THIS MODULE began generating
# `\frac{d}{dy'}` and then refusing to read its own output, with the
# multi-letter-name reason.  **Widening a reading opens a new exit; count them
# again** (`\partial y'` is separated by the command and never came here).
_UNSEPARATED_BASE_RE = re.compile(
    r"^[A-Za-z][0-9]*(?:_\{[^{}]+\}|_[A-Za-z0-9]+)*'*$")

# Why a fraction that looks like a derivative was left alone (i18n message keys).
REASON_ORDER = "notation.deriv_reason_order"
REASON_AMBIGUOUS = "notation.deriv_reason_ambiguous"
REASON_NO_SEPARATOR = "notation.deriv_reason_no_separator"
REASON_NO_OPERAND = "notation.deriv_reason_no_operand"
#: Not a refusal — the fraction was never a derivative, so there is nothing to
#: tell the researcher.  Deliberately EMPTY: the one place that reports a
#: refusal already writes `if reason and on_refused …`, so silence needs no
#: second guard beside it (and a second one is how the two drift apart).
NOT_A_DERIVATIVE = ""


def _separated(dep: str, gap: str, order_seen: bool, name: str) -> bool:
    r"""Is the ``d``/``\partial`` unambiguously NOT part of *name*?

    True when a space sits between them, an order suffix (``d^{2}``) sits
    between them, the name is a LaTeX command (``d\psi``), or the dep is
    ``\partial`` (a command cannot merge into the following name).
    """
    return bool(gap) or order_seen or name.startswith("\\") or dep != "d"
# The operand that FOLLOWS an operator-style fraction, when it is a plain name
# (optionally applied to arguments).  Bracketed operands are scanned by
# _trailing_operand instead, which counts nesting — a regex cannot, and reading
# `\left` as if it were a symbol produced 'diff(,t,1) for
# `\frac{d}{dt}\left(f(t)+g(t)\right)`: an empty operand, which is worse than
# refusing.  \left / \right are therefore excluded here as well.
#: Balanced-delimiter scanning is CAS-neutral and shared (latex_brackets).
_PAREN_PAIRS = PAREN_PAIRS
_brace_group = brace_group
_balanced = balanced

#: Scripts that belong to the OPERAND, not to the derivative.  ``\frac{d}{dx}
#: x^2`` is the derivative of x², never the square of dx/dx — nobody writes the
#: latter without brackets — yet the exponent was left outside and bound to the
#: result instead.  Measured 2026-08-21, on the most elementary calculus there
#: is, all of these came back SUCCESSFUL and wrong:
#:
#:     \frac{d}{dx} x^2            → 'diff(x,x,1)^2      = 1     (should be 2x)
#:     \frac{d x^2}{dx}            → (d*x^2)/(d*x)       = x     (∂ cancelled)
#:     \frac{d}{dx} f(x)^2         → 'diff(f(x),x,1)^2
#:     \frac{d}{dt} G^0(t)         → 'diff(G,t,1)^0*(t)  = t
#:     \frac{d}{dt} G^{(HF)}_i(t)  → 'diff(G,t,1)^((HF))_{i}(t)
#:
#: The last two are the same defect reaching a NAME: a label superscript is part
#: of the name (`script_label`), so stopping the scan before it cuts the name in
#: half.
_OPERAND_SCRIPTS_RE = re.compile(
    r"^(?:\^\{[^{}]*\}|\^\\?[A-Za-z0-9]+|_\{[^{}]+\}|_\\?[A-Za-z0-9]+)*")

_TRAILING_NAME_RE = re.compile(r"^(?P<name>" + _NAME + r")")
_DELIMITER_COMMANDS = frozenset({"\\left", "\\right"})

def _prefix_commands() -> frozenset[str]:
    r"""LaTeX commands that CANNOT stand alone as the thing being differentiated.

    Two families, one property: each takes what follows it, so reading the
    command alone as the whole operand silently differentiates a symbol nobody
    wrote.  Measured 2026-08-21, on the most ordinary spellings there are —

        \frac{\partial}{\partial t} \hat c(t)  → 'diff(\hat,t,1)*c(t)
        \frac{d}{dt} \bar{q}(t)                 → 'diff(\bar,t,1)q(t)
        \frac{d}{dt} \mathcal{H}(t)             → 'diff(\mathcal,t,1)H(t)
        \frac{d}{dx} \sin x                     → 'diff(sin,x,1)*x   (= 0)

    — all four reported SUCCESS, and the first is the Heisenberg equation of
    motion.  The numerator-carrying spelling was worse: it did not match the
    derivative shape at all, so ``\frac{\partial \hat\psi(x,t)}{\partial t}``
    fell through to the generic fraction rule and CANCELLED to
    ``psi_hat(x,t)/t`` — the exact failure BP-9 exists to stop.

    Neither list is restated here: the decorations come from the converter's own
    accent table and the functions from the backend's own ``_TRIG_MAP``, the
    same way ``function_application`` reads ``_KNOWN_FUNCTIONS``.  Taking the
    argument is GRAMMAR, not a guess — ``\hat`` and ``\sin`` are not quantities.
    """
    global _PREFIX_COMMANDS_CACHE
    if _PREFIX_COMMANDS_CACHE is None:
        from notebook.runtime.latex_converter import _DECORATION_SUFFIXES
        from notebook.runtime.maxima_latex_converter import MaximaLatexConverter
        names = {cmd for _, cmd in _DECORATION_SUFFIXES}
        # Spelling variants the converter's `_decorations` also accepts, plus the
        # wrappers `_vector` / `_sqrt` read.  All are decorations by the same
        # test: `\widehat` alone names nothing.
        names |= {r"\widehat", r"\widetilde", r"\overline", r"\overbar",
                  r"\underline", r"\vec", r"\mathbf", r"\boldsymbol",
                  r"\sqrt"}
        names |= {c for c in MaximaLatexConverter._TRIG_MAP
                  if re.fullmatch(r"\\[A-Za-z]+", c)}
        _PREFIX_COMMANDS_CACHE = frozenset(names)
    return _PREFIX_COMMANDS_CACHE


_PREFIX_COMMANDS_CACHE: "frozenset[str] | None" = None



def _order_of(m: "re.Match", first: int, second: int) -> int:
    """Read an _ORD pair of groups (``^{n}`` / ``^n``); absent means 1."""
    return int(m.group(first) or m.group(second) or 1)


def _trailing_operand(text: str, at: int) -> "tuple[str, int] | None":
    """The operand FOLLOWING an operator-style fraction, or None.

    Accepts ``\\left( … \\right)``, a plain ``( … )``, or a name with optional
    arguments.  Whitespace between the fraction and its operand is optional —
    the generator writes one, Maxima's tex1 does too, but a user need not.
    """
    k = at
    while k < len(text) and text[k].isspace():
        k += 1
    for open_tok, close_tok in _PAREN_PAIRS:
        grouped = _balanced(text, k, open_tok, close_tok)
        if grouped:
            tail = _OPERAND_SCRIPTS_RE.match(text[grouped[1]:]).group(0)
            if tail:
                # `\left(f+g\right)^2` — the bracket is what the exponent applies
                # to, so it has to be kept when the contents are handed over.
                return "(" + grouped[0] + ")" + tail, grouped[1] + len(tail)
            return grouped
    m = _TRAILING_NAME_RE.match(text[k:])
    if not m or m.group("name") in _DELIMITER_COMMANDS:
        return None
    name = m.group("name")
    at_args = k + m.end("name")
    if name in _prefix_commands():
        # `\hat c_{n\sigma}(t)`, `\bar{q}(t)`, `\sin x` — the command takes what
        # follows it, so the operand does not end here (see _prefix_commands).
        # The RAW SPAN is returned, not a rebuilt name: rebuilding would have to
        # re-decide where the decoration goes, and the converter the caller hands
        # this to already knows.  Its own script (`\sin^2 x`) is left with it,
        # for `function_application` to read.
        end = _prefix_argument_end(text, at_args)
        return (text[k:end], end) if end is not None else None
    # Scripts written on the name itself (`x^2`, `G^{(HF)}_{n\sigma}`) sit before
    # the argument list and are part of the operand.
    _scripts = _OPERAND_SCRIPTS_RE.match(text[at_args:]).group(0)
    name, at_args = name + _scripts, at_args + len(_scripts)
    # A CALLED operand takes its arguments in either spelling.  Only the bare
    # `(` used to be read, so Maxima's own `f\left(x , y\right)` lost its
    # argument list: `\frac{d}{d x} f\left(x , y\right)` came back as
    # `'diff(f,x,1)*(x , y)` — the arguments detached into a PRODUCT, with no
    # error (AUDIT-DERIV-DISPLAY-RT).  No whitespace is skipped before the
    # bracket on purpose: `f (x)` may be a product, and deciding that needs the
    # SymbolContext, which is the SharedParser's job and not this module's.
    for open_tok, close_tok in _PAREN_PAIRS:
        if text.startswith(open_tok, at_args):
            args = _balanced(text, at_args, open_tok, close_tok)
            # An unbalanced bracket is refused rather than dropped: returning
            # the bare name would leave the arguments behind to become a
            # product, which is the failure above.
            if not args:
                return None
            # A script AFTER the argument list belongs to the operand too:
            # `\frac{d}{dx} f(x)^2` is the derivative of the square.
            tail = _OPERAND_SCRIPTS_RE.match(text[args[1]:]).group(0)
            return name + "(" + args[0] + ")" + tail, args[1] + len(tail)
    return name, at_args


def _prefix_argument_end(text: str, at: int) -> "int | None":
    r"""Where the argument of a prefix command ends, or None if it has none.

    Braced (``\hat{c}_{n\sigma}(t)``): the group, then the subscripts and the
    argument list that belong to the NAME inside it.  Bare (``\hat c_{n\sigma}(t)``,
    ``\sin x``): the same scan as any other operand, so a decorated decoration
    (``\hat{\vec r}``) needs no extra rule.
    """
    k = at
    # The command's OWN script comes before its argument (`\sin^2 x`), and it
    # goes over with it — `function_application` reads that pair.  Without this
    # the scan found no operand at all and the fraction stayed a division, so
    # `\frac{d}{dx}\sin^2 x` became `d/(d*x) * sin(x)^2`.
    k += _OPERAND_SCRIPTS_RE.match(text[k:]).end()
    while k < len(text) and text[k].isspace():
        k += 1
    group = _brace_group(text, k)
    if group is None:
        nxt = _trailing_operand(text, k)
        return nxt[1] if nxt else None
    end = group[1] + _OPERAND_SCRIPTS_RE.match(text[group[1]:]).end()
    for open_tok, close_tok in _PAREN_PAIRS:
        if text.startswith(open_tok, end):
            args = _balanced(text, end, open_tok, close_tok)
            # An unbalanced bracket is refused rather than dropped — the same
            # reason the called-name branch above refuses it.
            return args[1] if args else None
    return end


def _factorizations(den: str) -> "Iterator[list[tuple[str, int]]]":
    """Yield every way *den* fully splits into (variable, order) factors.

    ``\\partial z\\partial t`` has one reading, but ``dxdy`` has two — ``d x·d y``
    and ``d(xdy)`` — because nothing delimits a ``d``-form factor.  Both are
    produced here; the caller disambiguates with the numerator's total order.
    """
    if not den.strip():
        yield []
        return
    m = _FACTOR_RE.match(den)
    if not m:
        return
    var_full = m.group("var")
    order = _order_of(m, 4, 5)
    separated = _separated(m.group("dep"), m.group("gap"),
                           bool(m.group(4) or m.group(5)), var_full)
    # Try every prefix of the matched name, longest first: a shorter name means
    # the rest of the text starts the next factor.
    for cut in range(len(var_full), 0, -1):
        var = var_full[:cut]
        if not re.fullmatch(_NAME, var):
            continue
        if not separated and not _UNSEPARATED_BASE_RE.match(var):
            continue
        if cut == len(var_full):
            rest_at, this_order = m.end(), order
        else:
            rest_at, this_order = m.start("var") + cut, 1
        for tail in _factorizations(den[rest_at:]):
            yield [(var, this_order)] + tail


def _resolve_denominator(den: str,
                         total: int) -> "tuple[list[tuple[str, int]] | None, str]":
    """The unique factorization of *den* whose orders sum to *total*.

    Returns (factors, "") on success, or (None, reason-key) explaining why the
    reading was refused so the caller can warn instead of failing quietly.
    """
    all_readings = [f for f in _factorizations(den) if f]
    matches = {tuple(f) for f in all_readings if sum(k for _, k in f) == total}
    if len(matches) == 1:
        return list(next(iter(matches))), ""
    if len(matches) > 1:
        return None, REASON_AMBIGUOUS
    if not all_readings:
        if _FACTOR_RE.match(den):
            # The denominator starts like a derivative factor but nothing parses
            # — the unseparated-name guard is the only thing that rejects one.
            return None, REASON_NO_SEPARATOR
        # No differential in the DENOMINATOR at all, so this was never a
        # derivative: it is a MEASURE, `\frac{dk}{2\pi}` or `\frac{d\omega}{2\pi}`,
        # which is how every frequency and momentum integral is written.  The
        # numerator alone made it a candidate, and warning on numerator evidence
        # alone put "could not read this as a derivative" under an integral that
        # was read perfectly (measured 2026-08-08).  A derivative needs a
        # differential on BOTH sides; `\frac{delta}{dx}` still warns, because
        # there the denominator IS one.
        return None, NOT_A_DERIVATIVE
    return None, REASON_ORDER


# The two DISPLAY forms `_apply_dot_notation_to_latex` uses for the orders that
# have no accent in the shared table: `\dddot{q}` and `{q}^{(4)}`.
_DDDOT_RE = re.compile(
    r"\\dddot\s*(?:\{(?P<braced>[^{}]*)\}|(?P<bare>" + _NAME + r"))")
_ORDER_SUP_RE = re.compile(
    r"(?:\{(?P<braced>" + _NAME + r")\}|(?<![A-Za-z0-9_])(?P<bare>" + _NAME + r"))"
    r"\s*\^\s*\{\((?P<order>\d+)\)\}")


def restore_dot_display_latex(
        text: str, to_canonical: "Callable[[str], str]",
        dot_quantity: "Callable[[str], tuple[str, str] | None]") -> str:
    r"""Rewrite the third- and higher-order dot DISPLAY forms to the noun form.

    ``\dddot{q}`` → ``'diff(q,t,3)``; ``{q}^{(4)}`` → ``'diff(q,t,4)``.  Both
    are shapes this system PRINTS, and neither could be read back: ``\dddot``
    is not in the shared accent table so it degraded to the symbol ``\dddotq``,
    and ``{q}^{(4)}`` parsed as an ordinary power — ``q^((4))``, the fourth
    derivative silently displayed and then re-read as the fourth POWER
    (AUDIT-DERIV-DISPLAY-RT).

    The noun form is the exact canonical the display was generated from, so the
    round trip closes: ``'diff(q,t,3)`` prints as ``\dddot{q}`` again.  That is
    also why these do NOT become ``q_dddot`` atoms by analogy with
    ``\dot``/``\ddot``: no operation understands such an atom, and the display
    bridge would have to learn to print it back.

    *dot_quantity* maps a displayed base to ``(canonical_name, time_parameter)``
    when it names a quantity declared time-dependent with dot notation — the
    generator's own condition for emitting these forms.

    The two are gated differently, and deliberately:

    * ``\dddot`` is an accent, so it cannot be anything but a derivative; it is
      read for any base, with ``t`` when the quantity is not declared.
    * ``^{(n)}`` is a SUPERSCRIPT and collides with exponentiation, so it is
      read only for a declared dot-notation quantity.  An undeclared
      ``x^{(4)}`` stays the fourth power, which is what it says.
    """
    def _resolve(base: str, order: int, gated: bool) -> "str | None":
        found = dot_quantity(base) if base else None
        if found is None:
            if gated:
                return None
            name = to_canonical(base) or base
            return f"'diff({name},t,{order})" if name.strip() else None
        name, tvar = found
        return f"'diff({name},{tvar},{order})"

    def _sub(m: "re.Match", order: "int | None", gated: bool) -> str:
        base = (m.group("braced") if m.group("braced") is not None
                else m.group("bare")) or ""
        n = order if order is not None else int(m.group("order"))
        return _resolve(base.strip(), n, gated) or m.group(0)

    text = _DDDOT_RE.sub(lambda m: _sub(m, 3, False), text)
    return _ORDER_SUP_RE.sub(lambda m: _sub(m, None, True), text)


#: ``\partial_x``/``\partial_{x}`` with an optional order (``\partial_x^{2}``).
_PARTIAL_SUB_RE = re.compile(
    r"\\partial_(?:\{(?P<braced>[^{}]+)\}|(?P<bare>" + _NAME + r"))"
    r"(?:\^\{(?P<ord_b>\d+)\}|\^(?P<ord_p>\d+))?")


def partial_subscript_to_frac(text: str) -> str:
    r"""``\partial_x f`` → ``\frac{\partial}{\partial x} f``.

    The subscripted operator is an everyday spelling of the very derivative
    :func:`restore_derivative_display_latex` already reads, so it is rewritten
    INTO that spelling rather than given a reader of its own — the same move as
    ``_conjugate_mark`` and ``_subscript_separator``.  A second reader is how
    two spellings of one thing drift apart.

    Measured 2026-08-10: left unread, ``\partial_x f`` reached the CAS as the
    PRODUCT ``partial(x) * f`` and every operation on it reported success — a
    derivative silently turned into a multiplication.

    **Only the operator is rewritten; the operand is left exactly as written.**
    The first attempt moved the operand into the numerator
    (``\frac{\partial f}{\partial x}``) and had to re-emit it, which dropped the
    brackets of a compound one — ``\partial_x (f+g)`` became
    ``\frac{\partial f+g}{\partial x}``, i.e. ``∂f/∂x + g``.  Rewriting in place
    cannot lose anything, and it targets the spelling the reader handles best:
    the numerator form does NOT read a called or bracketed operand at all
    (``\frac{\partial f(x,t)}{\partial x}`` lowers to a product — a separate,
    pre-existing defect recorded in tests/corpus/notation_corpus.json).

    Leaves the text alone when no operand follows: ``\partial_x`` by itself is
    an operator the researcher is writing ABOUT, not one being applied.
    """
    out: list[str] = []
    i = 0
    while True:
        m = _PARTIAL_SUB_RE.search(text, i)
        if not m:
            out.append(text[i:])
            return "".join(out)
        var = m.group("braced") or m.group("bare")
        order = m.group("ord_b") or m.group("ord_p")
        if not _trailing_operand(text, m.end()):     # nothing to differentiate
            out.append(text[i:m.end()])
            i = m.end()
            continue
        power = f"^{{{order}}}" if order else ""
        out.append(text[i:m.start()])
        out.append(f"\\frac{{\\partial{power}}}{{\\partial {var}{power}}}")
        i = m.end()                                  # operand stays verbatim


def restore_derivative_display_latex(
        text: str, to_canonical: "Callable[[str], str]",
        on_refused: "Callable[[str, str], None] | None" = None) -> str:
    r"""Rewrite every derivative display form in *text* to Maxima's noun form.

    *to_canonical* converts an operand / variable fragment of LaTeX to canonical
    (the backend's own converter), so ``\varphi`` becomes ``phi`` and ``A_{z}``
    becomes ``A_z``.

    A fraction that is not a derivative at all (``\frac{a}{b}``) is copied
    through in silence.  One that LOOKS like a derivative but cannot be read
    unambiguously is also copied through — it stays an ordinary division, the
    behaviour that predates this module — but *on_refused* is called with
    ``(fraction_text, reason_key)`` so the caller can warn the researcher rather
    than let a half-recognised derivative pass unnoticed.

    ``\partial_x f`` is folded into the fraction spelling first, so there is one
    reader below rather than two (:func:`partial_subscript_to_frac`).
    """
    text = partial_subscript_to_frac(text)
    out: list[str] = []
    i = 0
    while True:
        j = text.find(_FRAC + "{", i)
        if j < 0:
            out.append(text[i:])
            return "".join(out)
        num_g = _brace_group(text, j + len(_FRAC))
        den_g = _brace_group(text, num_g[1]) if num_g else None
        if not den_g:
            out.append(text[i:j + len(_FRAC)])
            i = j + len(_FRAC)
            continue
        num, den = num_g[0], den_g[0]
        replacement, end, reason = _try_one(text, num, den, den_g[1], to_canonical)
        if replacement is None:
            if reason and on_refused is not None:
                on_refused(text[j:den_g[1]], reason)
            # **A fraction that is not a derivative may CONTAIN one.** Skipping
            # to the end of the denominator meant the scan never looked inside,
            # so `\frac{m \left(2 r^2 \left(\frac{d}{d t}\theta\right)…\right)}{2}`
            # — this system's own printed line, in the published central-force
            # example — came back with `d` and `t` as ordinary symbols and the
            # derivative CANCELLED into a plain fraction, which is the one
            # failure BP-9 exists to prevent (measured 2026-09-25).  Advancing
            # past `\frac{` alone copies the head verbatim and lets the next
            # turn of this loop find the inner one.
            out.append(text[i:j + len(_FRAC) + 1])
            i = j + len(_FRAC) + 1
        else:
            out.append(text[i:j])
            out.append(replacement)
            i = end


def _numerator_with_operand(num: str) -> "tuple[re.Match | None, str]":
    r"""Read a numerator that carries what is being differentiated.

    ``d y(t)`` → the head match plus ``y(t)``; ``\partial^{2} \psi(x,t)`` → the
    head plus ``\psi(x,t)``.  Returns ``(None, "")`` when the numerator is not
    this shape — including when something FOLLOWS the operand, which is not a
    derivative numerator but an expression that merely starts like one.

    The operand is read by :func:`_trailing_operand`, the same scanner the
    operator form (``\frac{d}{dt} f(x)``) uses, so the two spellings cannot
    disagree about what an operand is.
    """
    m = _NUM_HEAD_RE.match(num)
    if not m:
        return None, ""
    tail = _trailing_operand(num, m.end())
    if not tail:
        return None, ""
    operand, end = tail
    return (m, operand) if not num[end:].strip() else (None, "")


def _fraction_operand(text: str, at: int,
                      to_canonical: "Callable[[str], str]"
                      ) -> "tuple[str, int] | None":
    r"""One leading ``\frac{…}{…}`` at *at*, already converted — or None.

    An operator fraction's operand may itself be a fraction, which is how a
    mixed partial is ordinarily written: ``\frac{\partial}{\partial x}
    \frac{\partial}{\partial y} f``.  ``_trailing_operand`` cannot read that —
    its name pattern happily matches the literal ``\frac``, so measured
    2026-08-11 the ``d`` spelling came back as ``'diff(frac,x,1) * ddy * f``,
    a derivative of a SYMBOL CALLED "frac", reported as a success.

    Consumes exactly this one fraction (plus, when it is itself an operator
    form, its own operand) so that a following term stays outside:
    ``\frac{d}{dx}\frac{a}{b} + c`` differentiates ``a/b``, not ``a/b + c``.

    Nested ``'diff`` is what the CAS wants — verified against Maxima, which
    normalises ``'diff('diff(f,y,1),x,1)`` to ``'diff(f,x,1,y,1)`` and reports
    a zero difference from the single-fraction spelling.
    """
    k = at
    while k < len(text) and text[k].isspace():
        k += 1
    if not text.startswith(_FRAC + "{", k):
        return None
    num_g = _brace_group(text, k + len(_FRAC))
    den_g = _brace_group(text, num_g[1]) if num_g else None
    if not num_g or not den_g:
        return None
    canonical, end, _reason = _try_one(text, num_g[0], den_g[0], den_g[1],
                                       to_canonical)
    if canonical is not None:
        return canonical, end
    # An ordinary fraction is still a perfectly good operand.
    return (f"(({to_canonical(num_g[0]) or num_g[0]})"
            f"/({to_canonical(den_g[0]) or den_g[0]}))"), den_g[1]


def _try_one(text: str, num: str, den: str, after: int,
             to_canonical: "Callable[[str], str]"
             ) -> "tuple[str | None, int, str]":
    r"""Interpret one ``\frac{num}{den}`` (plus any trailing operand) as a
    derivative.

    Returns (canonical, index after the consumed text, "") on success, or
    (None, index, reason) — where an EMPTY reason means "this was never a
    derivative", so the caller stays quiet.
    """
    m_carry, carried = _numerator_with_operand(num)
    m_op = _NUM_OPERATOR_RE.match(num)
    if m_carry:
        operand_tex = carried
        total, end = _order_of(m_carry, 2, 3), after
        if not _separated(m_carry.group("dep"), m_carry.group("gap"),
                          bool(m_carry.group(2) or m_carry.group(3)), operand_tex) \
                and not _UNSEPARATED_BASE_RE.match(operand_tex):
            return None, after, REASON_NO_SEPARATOR
    elif m_op:
        chained = _fraction_operand(text, after, to_canonical)
        if chained:
            operand_tex, end = chained
            total = _order_of(m_op, 1, 2)
            factors, reason = _resolve_denominator(den, total)
            if not factors:
                return None, after, reason
            inner = ",".join(f"{to_canonical(v) or v},{k}" for v, k in factors)
            return f"'diff({operand_tex},{inner})", end, ""
        tail = _trailing_operand(text, after)
        if not tail:
            # "\frac{d}{dt}" with nothing to differentiate.  Only worth a warning
            # when the denominator really does read as a derivative factor;
            # otherwise this was an ordinary fraction all along.
            return (None, after,
                    REASON_NO_OPERAND if _FACTOR_RE.match(den) else "")
        operand_tex, end = tail
        total = _order_of(m_op, 1, 2)
    else:
        return None, after, ""
    factors, reason = _resolve_denominator(den, total)
    if not factors:
        return None, after, reason
    operand = to_canonical(operand_tex) or operand_tex
    if not operand.strip():
        return None, after, REASON_NO_OPERAND
    inner = ",".join(f"{to_canonical(v) or v},{k}" for v, k in factors)
    return f"'diff({operand},{inner})", end, ""
