r"""How far a written function reaches for its argument.

``\sin x`` is a call and ``\sin x \cos y`` is a product of two — the rule is
that a BARE function takes exactly ONE token, and it is the conventional
reading (``\sin x y`` is $\sin(x)\cdot y$, not $\sin(xy)$).  Everything here is
about the cases where "one token" is not the whole answer.

**The argument may itself be a function** (fixed 2026-08-16).  The reading used
to be a regex that stops at ``(``, so the NAME was taken as the argument and
the real argument became a factor — silently, in every combination::

    \log \sqrt{x}   → log(sqrt) * x
    \sin \cos x     → sin(cos) * x
    \tanh \sqrt{x}  → tanh(sqrt) * x
    \log \sin(x)    → log(sin) * x

Only ``\sqrt{\sin x}`` came out right, and only because ``\sqrt`` takes a
braced group.  ``\sin\cos x`` is the same defect: it reaches this rule as
``\sin \cos x`` because `_separate_adjacent_commands` puts the space in.

This module is a pure function of TEXT and a function-name alternation.  It has
no CAS-specific knowledge: which names are functions is the backend converter's
``_FN_ALT``, so a third backend adds its own without editing this file.

Extracted from ``latex_converter`` when the reading grew past a regex — the
same move as ``subscript_policy`` and ``sum_integral_prefix``.  Further
questions about a function's written form (a base subscript, ``\log_2``;
``\max_i`` as a binder) belong here rather than back in the shared converter.
"""
from __future__ import annotations

import re
from functools import lru_cache

from notebook.runtime.latex_brackets import balanced
from notebook.runtime.latex_typography import braced_group

#: One token of a bare function's argument — `\sin x`, `\sin 2x`, `\sin x_i`.
#: A function takes exactly ONE; that is what makes `\sin x \cos y` a product.
_ARG_TOKEN_RE = re.compile(r"[^\s+\-=,()\[\]{}]+")


@lru_cache(maxsize=8)
def _head_re(fn_alt: str) -> "re.Pattern[str]":
    """A known function NAME at a word boundary.

    Cached on the alternation itself, so each backend's table is compiled once
    however many converters are built.
    """
    return re.compile(rf"(?<![A-Za-z0-9_\\])({fn_alt})(?![A-Za-z0-9_])")


@lru_cache(maxsize=8)
def _script_head_re(fn_alt: str) -> "re.Pattern[str]":
    r"""A known function NAME immediately followed by ``^`` or ``_``.

    `_head_re`'s right boundary excludes `_` (that is what stops `sin` matching
    inside `sinh`), so the DECORATED forms need their own head — the same
    reason `_log_head_re` has one.
    """
    return re.compile(rf"(?<![A-Za-z0-9_\\])({fn_alt})(?=[\^_])")


#: `f^{-1}` is the INVERSE FUNCTION for these, and the reason is that their
#: RECIPROCAL ALREADY HAS ITS OWN NAME.  Nobody writes `\sin^{-1}` for `1/sin`
#: because that quantity is called `cosec`; the same holds for sec (1/cos), cot
#: (1/tan) and the hyperbolic trio.  So the ambiguity this file used to refuse —
#: "arcsine by convention, the reciprocal by the rule that reads `\sin^2 x`" —
#: is not a real ambiguity HERE: the competing reading has a different spelling
#: and no one uses this one for it.  (Pointed out by the researcher 2026-08-18;
#: the system's own `_KNOWN_FUNCTIONS` already carried `csc`, `sec`, `cot`
#: alongside the inverses, which is the argument in data form.)
#:
#: The names are Maxima's and were measured against a live session — `asin(1)`
#: is %pi/2 while `arcsin(1)` is an undefined function that stays put.
_NAMED_INVERSE = {
    "sin": "asin", "cos": "acos", "tan": "atan",
    "csc": "acsc", "sec": "asec", "cot": "acot",
    "sinh": "asinh", "cosh": "acosh", "tanh": "atanh",
    # already-inverse spellings, so `\arcsin^{-1}` does not silently invert twice
    "asin": None, "acos": None, "atan": None,
}


def inverse_function_name(fn: str) -> "str | None":
    """The inverse of *fn*, or None when `f^{-1}` has no settled reading.

    None means REFUSE, not "guess the reciprocal": for `\\log`, `\\exp` and
    friends the two readings genuinely compete and the spelling is rare, so the
    notation does not decide (`function_inverse_script`).
    """
    return _NAMED_INVERSE.get(fn)


#: The exponent of `\sin^2 x` as it reaches this module: `_superscript` has
#: already reduced `^{2}` to `^2` and `^{n+1}` to `^(n+1)`, so an exponent is
#: either a parenthesised group or one bare token.
_EXP_TOKEN_RE = re.compile(r"-?[A-Za-z0-9_.%]+")


def read_power_application(s: str, i: int, fn_alt: str, log_fn: str
                           ) -> "tuple[str, int] | None":
    r"""``\sin^2 x`` → ``sin(x)^2``.  The exponent applies to the VALUE.

    This is the conventional reading and there is nothing to guess about it:
    `\sin^2\theta` is $(\sin\theta)^2$ in every text that writes it.  Left
    unread it was **silently a different expression** — measured 2026-08-17,
    `\sin^2 x` reached Maxima as `sin^2 * x`, where `sin` is an ordinary SYMBOL
    (Maxima accepts that happily), so:

        \cos^2\theta + \sin^2\theta  → theta*cos^2 + theta*sin^2   (never 1)
        d/dx of \sin^2 x             → sin^2      (sin held constant)

    The Pythagorean identity is about as ordinary as physics notation gets, and
    both answers came back as successes.  `(\sin x)^2` and `\sin(x)^2` were
    always right; only the spelling everyone actually writes was wrong.

    A NEGATIVE exponent is deliberately NOT read here — `\sin^{-1}` is arcsine
    by strong convention, yet the very rule above (`\sin^2` = the value
    squared) makes it the reciprocal.  The notation genuinely does not say
    which, so it is left for `operand_guards` to refuse with both readings
    named, rather than given a plausible answer here.
    """
    m = _script_head_re(fn_alt).match(s, i)
    if m is None or s[m.end()] != "^":
        return None
    name, k = m.group(1), m.end() + 1
    if k < len(s) and s[k] == "(":
        group = balanced(s, k, "(", ")")
        if not group:
            return None
        inner, k = group[0], group[1]
        exp, bare = f"({inner})", inner
    else:
        em = _EXP_TOKEN_RE.match(s, k)
        if em is None:
            return None
        exp, bare, k = em.group(0), em.group(0), em.end()
    # `_superscript` has already PARENTHESISED a compound exponent, so
    # `\sin^{-1}` arrives as `^(-1)` and a test on the rendered `exp` never
    # sees the minus.  Ask the exponent ITSELF, not its rendering.
    _neg = bare.strip().startswith("-")
    if _neg and bare.strip().lstrip("(").rstrip(")").strip() != "-1":
        # `\sin^{-2}` has no reading at all: neither "the inverse, squared" nor
        # "the reciprocal squared" is written that way.
        return None
    if _neg:
        inverse = inverse_function_name(name)
        if inverse is None:
            return None              # no settled reading; see the docstring
        name, exp = inverse, None
    j = k
    while j < len(s) and s[j] in " \t":
        j += 1
    if j < len(s) and s[j] in "({":
        close = ")" if s[j] == "(" else "}"
        group = balanced(s, j, s[j], close)
        if not group:
            return None
        _inner = apply_all(group[0], fn_alt, log_fn)
        return (f"{name}({_inner})" if exp is None
                else f"{name}({_inner})^{exp}"), group[1]
    if j == k:                       # `\sin^2` with nothing after it
        return None
    arg = _read_argument(s, j, fn_alt, log_fn, allow_paren=False)
    if not arg:
        return None
    return ((f"{name}({arg[0]})" if exp is None
             else f"{name}({arg[0]})^{exp}"), arg[1])


#: A subscript written WITHOUT braces, at the stage this module runs: `_2`,
#: `_10`, `_b`, `_alpha`, `_%pi` — LaTeX commands have already become names.
_BARE_SUB_RE = re.compile(r"[A-Za-z0-9%]+")


@lru_cache(maxsize=8)
def _log_head_re(log_fn: str) -> "re.Pattern[str]":
    r"""The logarithm's name IMMEDIATELY followed by a subscript.

    `_head_re`'s right boundary excludes `_` — it is what stops `sin` matching
    inside `sinh` — so the based form needs its own head or it never matches at
    all.  (Measured: with the shared head the rule silently never fired, and
    every `\log_2` kept its old wrong reading.)
    """
    return re.compile(rf"(?<![A-Za-z0-9_\\]){re.escape(log_fn)}(?=_)")


def _read_argument(s: str, j: int, fn_alt: str, log_fn: str, *,
                   allow_paren: bool) -> "tuple[str, int] | None":
    r"""What a function applies to at *j*: one token, a braced or parenthesised
    group, or — when it is itself a function — that whole application.

    **Whatever is consumed here is SCANNED**, because the top-level sweep will
    never see it again.  Six spellings were silently wrong for exactly that
    reason (measured 2026-08-16), and one of them predates this module::

        \sin{\cos x}       → sin(cos * x)          a PRODUCT inside the sine
        \log{\sin x}       → log(sin * x)          (the older braced rule)
        \log \sin(\cos y)  → log(sin(cos * y))
        \log_2 (\sin x)    → log(sin * x)/log(2)

    ``\log(\sin x)`` came out right only because nothing consumed it — the
    sweep reached the inner `\sin x` on its own.  Consuming text and scanning
    it have to be the same act.

    *allow_paren* is off for a plain ``\log (x+1)``, where the parenthesised
    form is a spaced call the parser already reads correctly and rewriting it
    would churn canonicals that are right today.  It is ON for ``\log_2(x)``,
    which is NOT right today: left alone that is a call to an undefined
    function named ``log_2``.
    """
    if j >= len(s):
        return None
    nested = read_application(s, j, fn_alt, log_fn, as_argument=True)
    if nested is not None:
        return nested
    if s[j] == "{":
        group = braced_group(s, j)
    elif s[j] == "(":
        group = balanced(s, j, "(", ")") if allow_paren else None
    else:
        tok = _ARG_TOKEN_RE.match(s, j)
        return (tok.group(0), tok.end()) if tok else None
    if group is None:
        return None
    return apply_all(group[0], fn_alt, log_fn), group[1]


def read_log_base(s: str, i: int, fn_alt: str, log_fn: str) -> "tuple[str, int] | None":
    r"""``\log_b X`` → ``(log(X))/(log(b))``, or None when *i* is not that.

    The base of a logarithm is written as a subscript, and nothing read it: the
    two spellings went two different wrong ways (measured 2026-08-16), both
    silently, and both produce VALID CAS text — which is what makes them worse
    than an error::

        \log_2 x            → log(2) * x      the logarithm OF 2, times x
        \log_{10} x         → log(10) * x
        \log_2(x)           → log_2(x)        a call to an undefined function
        \log_2 \frac{1}{6}  → log_2(1*6^-1)

    `\log_2 6` therefore answered `log(2)*6 ≈ 4.16` where the value is ≈ 2.58.
    In information theory the bit is the default unit, so this is the spelling
    a researcher reaches for first.

    Change of base is a REWRITE, not a name: there is no `log_2` in the CAS to
    lower to, and ``log(X)/log(b)`` is the identity every backend can evaluate.
    """
    m = _log_head_re(log_fn).match(s, i)
    if m is None:                       # a subscript on any other function has
        return None                     # no reading here — see `apply_all`
    k = m.end()
    base = braced_group(s, k + 1)
    if base is None:
        tok = _BARE_SUB_RE.match(s, k + 1)
        base = (tok.group(0), tok.end()) if tok else None
    if base is None:
        return None
    j = base[1]
    while j < len(s) and s[j] in " \t":
        j += 1
    arg = _read_argument(s, j, fn_alt, log_fn, allow_paren=True)
    if arg is None:
        return None                     # `\log_2` with nothing applied — left
    return f"({log_fn}({arg[0]}))/({log_fn}({base[0]}))", arg[1]


def read_application(s: str, i: int, fn_alt: str, log_fn: str = "log", *,
                     as_argument: bool = False) -> "tuple[str, int] | None":
    r"""Read one ``fn(arg)`` starting at *i*, or None when *i* is not one.

    Returns the REWRITTEN text and the index just past what it consumed.

    *as_argument* is the difference between the two positions this is read
    from.  At TOP level a name already followed by ``(`` needs no rewriting and
    is left exactly as written (``\log(x)`` — and consuming it would rewrite
    canonicals that are correct today).  As somebody else's ARGUMENT the same
    shape has to be taken WHOLE, because otherwise the bare name is all the
    token scan can reach — which is exactly how ``\log \sqrt{x}`` became
    ``log(sqrt) * x``, ``\sqrt{x}`` having already become ``sqrt(x)`` by the
    time this runs.
    """
    m = _head_re(fn_alt).match(s, i)
    if m is None:
        return None
    name, k = m.group(1), m.end()
    if k < len(s) and s[k] == "{":
        group = braced_group(s, k)
        return ((f"{name}({apply_all(group[0], fn_alt, log_fn)})", group[1])
                if group else None)
    if k < len(s) and s[k] == "(":
        if not as_argument:
            return None
        group = balanced(s, k, "(", ")")
        return ((f"{name}({apply_all(group[0], fn_alt, log_fn)})", group[1])
                if group else None)
    j = k
    while j < len(s) and s[j] in " \t":
        j += 1
    if j == k:                       # nothing separates them: `sin_2`, `sinx`
        return None
    arg = _read_argument(s, j, fn_alt, log_fn, allow_paren=False)
    return (f"{name}({arg[0]})", arg[1]) if arg else None


def apply_all(s: str, fn_alt: str, log_fn: str = "log") -> str:
    r"""Rewrite every written function application in *s*.

    ``known-function{arg}`` and ``known-function<space>arg`` become ``fn(arg)``
    with nested applications read whole; ``\log_b X`` becomes the change-of-base
    quotient.  The based form is tried FIRST — ``log_2 x`` also matches the
    plain reading, which would take `_2 x` as the argument.

    A subscript on any OTHER function (``\sin_2 x``) is left exactly as it was.
    It has no reading here, and inventing one would be a guess; it is measured
    and tracked separately (open_items FN-SUBSCRIPT-NO-READING) rather than
    quietly given the log's.
    """
    if not fn_alt:
        return s
    out: list[str] = []
    i = 0
    while i < len(s):
        piece = (read_log_base(s, i, fn_alt, log_fn)
                 or read_power_application(s, i, fn_alt, log_fn)
                 or read_application(s, i, fn_alt, log_fn))
        if piece is None:
            out.append(s[i])
            i += 1
            continue
        text, i = piece
        out.append(text)
    return "".join(out)


#: Function names that must never become a bare VALUE.  A written function is
#: either applied (`sin(x)`) or it is not a quantity at all — Maxima will
#: happily treat `sin` as an ordinary symbol, which is how the two spellings
#: below stayed silently wrong.  Consulted by the resolver through the
#: backend's own `_KNOWN_FUNCTIONS`, so no list is restated anywhere.
def bare_function_reason(name: str, decoration: str,
                         exponent: "int | None" = None) -> str:
    r"""The i18n key for a known function NAME used as a value.

    Two spellings reach here, and they are refused for DIFFERENT reasons:

    * ``\sin^{-1} x`` — a negative exponent on a function is genuinely
      ambiguous.  `\sin^{-1}` is arcsine by strong convention, yet the rule
      that makes `\sin^2 x` the value squared makes it the reciprocal.  The
      notation does not say which, so this system does not choose (the same
      stance as `\max_{1<i<3}`, where integer and real readings differ).
    * ``\sin_2 x`` — a subscript on a function other than the logarithm has no
      standard reading at all (open_items FN-SUBSCRIPT-NO-READING).  Giving it
      the log's would be inventing one.

    Both used to pass through as a bare symbol and produce a VALID Maxima
    expression with a different meaning — `sin(2)*x`, `sin^(-1)*x` — which is
    the shape this project refuses on sight.
    """
    if decoration != "^":
        return "parser.unsupported_notation.function_subscript_no_reading"
    # `\sin^{-1}` no longer reaches here — a function whose reciprocal has its
    # own name reads as the inverse (see `_NAMED_INVERSE`).  What is left splits
    # in two, and they are refused for different reasons.
    if exponent is not None and exponent != -1:
        return "parser.unsupported_notation.function_negative_power"
    return "parser.unsupported_notation.function_inverse_script"
