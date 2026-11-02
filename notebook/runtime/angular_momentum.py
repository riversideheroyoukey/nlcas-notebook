"""F-9 — angular-momentum coupling: Clebsch–Gordan coefficients and the Wigner
3j / 6j / 9j symbols, computed **exactly** and **self-contained**.

Why not SymPy
-------------
``docs/open_items.md`` F-9 suggested ``sympy.physics.wigner``.  The application
venv has no SymPy (the second-quantization "SymPy route" is SymPy.jl *inside*
Julia, not a Python dependency), and this layer needs no CAS at all: every one
of these symbols is a closed-form Racah sum over factorials.  So this module
follows :mod:`notebook.runtime.spin_algebra` — a self-contained exact kernel
that the CAS never sees — rather than adding a Python dependency for arithmetic
we can do exactly ourselves.

Exact arithmetic
----------------
Every value here has the form ``Σ_d c_d √d`` with ``c_d`` rational and ``d`` a
square-free positive integer (:class:`Surd`).  Two facts keep this exact and
cheap:

1. The radicands come from ratios of **factorials**, so they are built in
   *prime-exponent* form (Legendre's formula) and the square root is taken by
   splitting each exponent into ``(even → outside, odd → inside)``.  No large
   integer is ever factored.
2. A product of two square-free radicands reduces with a gcd, not a
   factorization: ``√a·√b = g·√((a/g)(b/g))`` with ``g = gcd(a,b)``.

That closure under ``+`` and ``×`` is what makes the 9j symbol — a *sum* of
products of three 6j symbols, generally an irreducible sum of distinct surds —
representable without falling back to floats.

Public API
----------
``clebsch_gordan``, ``wigner_3j``, ``wigner_6j``, ``wigner_9j``,
``coupled_j_values``, ``dot_product_eigenvalue``, ``cg_expansion``,
``lande_g_factor``, ``wigner_eckart_coefficient``, ``parse_term_symbol``.

Conventions: Condon–Shortley phase; ``⟨j1 m1 j2 m2 | J M⟩`` for the CG
coefficient; the Racah/Wigner definitions of 3j/6j/9j (Edmonds).
"""
from __future__ import annotations

import math
import re
from fractions import Fraction

__all__ = [
    "AngularMomentumError", "Surd",
    "clebsch_gordan", "wigner_3j", "wigner_6j", "wigner_9j",
    "coupled_j_values", "dot_product_eigenvalue", "cg_expansion",
    "lande_g_factor", "wigner_eckart_coefficient", "parse_term_symbol",
    "parse_j", "j_latex", "j_canonical",
]


class AngularMomentumError(ValueError):
    """Invalid angular-momentum input.

    Carries a machine-readable ``code`` (+ format params) instead of a
    user-visible sentence: the caller renders it through ``t()`` so this
    module stays free of translatable text (CLAUDE.md i18n rule).
    """

    def __init__(self, code: str, **params):
        super().__init__(code)
        self.code = code
        self.params = params


# ---------------------------------------------------------------------------
# Exact numbers: Σ_d c_d √d
# ---------------------------------------------------------------------------

def _phase(n: int) -> int:
    """``(-1)**n`` for any integer ``n``.

    NOT ``(-1) ** n``: Python evaluates a negative integer power in floating
    point (``(-1) ** -2 == 1.0``), and a single float leaking into the
    coefficients turns an exact ``1/3`` into ``6004799503160661/2**54``.  The
    phase exponents here (``j1-j2-m3`` etc.) are routinely negative.
    """
    return -1 if n % 2 else 1


def _exact(value) -> Fraction:
    """Coerce to Fraction, refusing floats.

    A float here is always a bug upstream (see :func:`_phase`), and
    ``Fraction(0.333…)`` would absorb it into a plausible-looking but wrong
    exact number.  Fail instead of silently degrading.
    """
    if isinstance(value, float):
        raise TypeError(f"inexact float in exact angular-momentum arithmetic: {value!r}")
    return Fraction(value)


class Surd:
    """An exact real of the form ``Σ_d c_d √d`` (``d`` square-free ≥ 1).

    Closed under addition and multiplication (see the module docstring), which
    is what the 9j sum needs.  ``Surd()`` is zero.
    """
    __slots__ = ("terms",)

    def __init__(self, terms: "dict[int, Fraction] | None" = None):
        self.terms = {int(r): _exact(c) for r, c in (terms or {}).items() if c}

    # -- constructors ---------------------------------------------------
    @classmethod
    def rational(cls, value) -> "Surd":
        return cls({1: _exact(value)})

    @classmethod
    def term(cls, coeff, radicand: int) -> "Surd":
        return cls({radicand: _exact(coeff)})

    # -- algebra --------------------------------------------------------
    def is_zero(self) -> bool:
        return not self.terms

    def __bool__(self) -> bool:
        return bool(self.terms)

    def __eq__(self, other) -> bool:
        if isinstance(other, (int, Fraction)):
            other = Surd.rational(other)
        return isinstance(other, Surd) and self.terms == other.terms

    def __hash__(self):
        return hash(tuple(sorted(self.terms.items())))

    def __neg__(self) -> "Surd":
        return Surd({r: -c for r, c in self.terms.items()})

    def __add__(self, other) -> "Surd":
        if isinstance(other, (int, Fraction)):
            other = Surd.rational(other)
        out = dict(self.terms)
        for r, c in other.terms.items():
            out[r] = out.get(r, Fraction(0)) + c
        return Surd(out)

    __radd__ = __add__

    def __sub__(self, other) -> "Surd":
        return self + (-other if isinstance(other, Surd) else Surd.rational(-other))

    def __mul__(self, other) -> "Surd":
        if isinstance(other, (int, Fraction)):
            return Surd({r: c * other for r, c in self.terms.items()})
        out: "dict[int, Fraction]" = {}
        for r1, c1 in self.terms.items():
            for r2, c2 in other.terms.items():
                g = math.gcd(r1, r2)
                rad = (r1 // g) * (r2 // g)
                out[rad] = out.get(rad, Fraction(0)) + c1 * c2 * g
        return Surd(out)

    __rmul__ = __mul__

    def square(self) -> "Surd":
        return self * self

    def as_fraction(self) -> "Fraction | None":
        """The value as a Fraction if it is rational, else None."""
        if not self.terms:
            return Fraction(0)
        if set(self.terms) == {1}:
            return self.terms[1]
        return None

    def __float__(self) -> float:
        return float(sum(float(c) * math.sqrt(r) for r, c in self.terms.items()))

    def __repr__(self) -> str:
        return f"Surd({self.terms!r})"

    # -- rendering ------------------------------------------------------
    def latex(self) -> str:
        """LaTeX for display, e.g. ``\\frac{1}{\\sqrt{2}}``, ``-\\frac{\\sqrt{6}}{3}``."""
        if not self.terms:
            return "0"
        out = ""
        for rad, coeff in sorted(self.terms.items()):
            body = _term_latex(abs(coeff), rad)
            if out:
                out += " - " if coeff < 0 else " + "
            elif coeff < 0:
                out += "-"
            out += body
        return out

    def canonical(self) -> str:
        """CAS-ready form, e.g. ``sqrt(2)/2``, ``-3*sqrt(5)/7`` — the value can be
        fed straight back into an ordinary (Maxima) operation."""
        if not self.terms:
            return "0"
        out = ""
        for rad, coeff in sorted(self.terms.items()):
            body = _term_canonical(abs(coeff), rad)
            if out:
                out += " - " if coeff < 0 else " + "
            elif coeff < 0:
                out += "-"
            out += body
        return out


def _term_latex(coeff: Fraction, rad: int) -> str:
    """One ``|c|√d`` term as LaTeX (``coeff`` already non-negative)."""
    num, den = coeff.numerator, coeff.denominator
    if rad == 1:
        return str(num) if den == 1 else rf"\frac{{{num}}}{{{den}}}"
    # Prefer the un-rationalized 1/√2 shape when the denominator carries the
    # radicand — that is how coupling tables are written.
    if den % rad == 0:
        outer = den // rad
        low = rf"\sqrt{{{rad}}}" if outer == 1 else rf"{outer}\sqrt{{{rad}}}"
        return rf"\frac{{{num}}}{{{low}}}"
    top = rf"\sqrt{{{rad}}}" if num == 1 else rf"{num}\sqrt{{{rad}}}"
    return top if den == 1 else rf"\frac{{{top}}}{{{den}}}"


def _term_canonical(coeff: Fraction, rad: int) -> str:
    num, den = coeff.numerator, coeff.denominator
    if rad == 1:
        return str(num) if den == 1 else f"{num}/{den}"
    top = f"sqrt({rad})" if num == 1 else f"{num}*sqrt({rad})"
    return top if den == 1 else f"{top}/{den}"


# ---------------------------------------------------------------------------
# Prime-exponent factorials (no large integer is ever factored)
# ---------------------------------------------------------------------------

def _primes_up_to(n: int) -> "list[int]":
    if n < 2:
        return []
    sieve = bytearray([1]) * (n + 1)
    sieve[0:2] = b"\x00\x00"
    for p in range(2, int(n ** 0.5) + 1):
        if sieve[p]:
            sieve[p * p::p] = bytearray(len(sieve[p * p::p]))
    return [i for i in range(2, n + 1) if sieve[i]]


def _add_factorial(exps: "dict[int, int]", n: int, sign: int = 1) -> None:
    """Multiply (sign=+1) or divide (sign=-1) the prime-exponent map by ``n!``."""
    if n < 0:
        raise AngularMomentumError("negative_factorial", value=n)
    for p in _primes_up_to(n):
        e, q = 0, p
        while q <= n:
            e += n // q
            q *= p
        exps[p] = exps.get(p, 0) + sign * e


def _sqrt_of_exponents(exps: "dict[int, int]") -> "tuple[Fraction, int]":
    """√(∏ p^e) as ``(rational outside, square-free radicand)``."""
    outside = Fraction(1)
    rad = 1
    for p, e in exps.items():
        if not e:
            continue
        half, rem = divmod(e, 2)          # floors, so negative e works too
        outside *= Fraction(p) ** half
        if rem:
            rad *= p
    return outside, rad


def _sqrt_int(n: int) -> "tuple[int, int]":
    """√n as ``(integer outside, square-free radicand)`` for small n."""
    if n < 0:
        raise AngularMomentumError("negative_sqrt", value=n)
    outside, rad, k = 1, n, 2
    while k * k <= rad:
        while rad % (k * k) == 0:
            rad //= k * k
            outside *= k
        k += 1
    return outside, rad


# ---------------------------------------------------------------------------
# Quantum-number plumbing
# ---------------------------------------------------------------------------

_J_RE = re.compile(r"^\s*([-+]?\d+)\s*(?:/\s*(\d+))?\s*$")
_TRANSLATE_FULLWIDTH = str.maketrans("０１２３４５６７８９／＋－", "0123456789/+-")


def parse_j(value) -> Fraction:
    """Parse an angular-momentum quantum number: ``"3/2"``, ``"-1/2"``, ``1``.

    Half-integers only — anything else is rejected rather than rounded, so a
    typo can never silently become a different (valid) symbol.
    """
    if isinstance(value, Fraction):
        frac = value
    elif isinstance(value, int):
        frac = Fraction(value)
    elif isinstance(value, float):
        frac = Fraction(value).limit_denominator(2)
        if float(frac) != value:
            raise AngularMomentumError("not_half_integer", value=value)
    else:
        text = str(value).strip().translate(_TRANSLATE_FULLWIDTH)
        text = text.replace("−", "-")       # unicode minus
        m = _J_RE.match(text)
        if m is None:
            raise AngularMomentumError("bad_quantum_number", value=str(value))
        frac = Fraction(int(m.group(1)), int(m.group(2) or 1))
    if (2 * frac).denominator != 1:
        raise AngularMomentumError("not_half_integer", value=str(value))
    return frac


def _is_int(x: Fraction) -> bool:
    return x.denominator == 1


def _check_j(j: Fraction, name: str) -> None:
    if j < 0:
        raise AngularMomentumError("negative_j", name=name, value=j_latex(j))


def _check_jm(j: Fraction, m: Fraction, jname: str, mname: str) -> None:
    _check_j(j, jname)
    if not _is_int(j - m):
        raise AngularMomentumError("j_m_mismatch", jname=jname, mname=mname,
                                   j=j_latex(j), m=j_latex(m))
    if abs(m) > j:
        raise AngularMomentumError("m_out_of_range", jname=jname, mname=mname,
                                   j=j_latex(j), m=j_latex(m))


def _triangle(a: Fraction, b: Fraction, c: Fraction) -> bool:
    """Triangle rule |a-b| ≤ c ≤ a+b *and* integer perimeter."""
    return abs(a - b) <= c <= a + b and _is_int(a + b + c)


def j_latex(j: Fraction) -> str:
    """A (half-)integer quantum number as LaTeX: ``3/2`` → ``\\frac{3}{2}``."""
    if j.denominator == 1:
        return str(j.numerator)
    sign = "-" if j < 0 else ""
    return rf"{sign}\frac{{{abs(j.numerator)}}}{{{j.denominator}}}"


def j_canonical(j: Fraction) -> str:
    """A (half-)integer quantum number as CAS text: ``3/2`` → ``3/2``."""
    return str(j.numerator) if j.denominator == 1 else f"{j.numerator}/{j.denominator}"


# ---------------------------------------------------------------------------
# 3j / CG
# ---------------------------------------------------------------------------

def wigner_3j(j1, j2, j3, m1, m2, m3) -> Surd:
    """The Wigner 3j symbol ``(j1 j2 j3 / m1 m2 m3)`` (Racah's closed form)."""
    j1, j2, j3 = parse_j(j1), parse_j(j2), parse_j(j3)
    m1, m2, m3 = parse_j(m1), parse_j(m2), parse_j(m3)
    _check_jm(j1, m1, "j_1", "m_1")
    _check_jm(j2, m2, "j_2", "m_2")
    _check_jm(j3, m3, "j_3", "m_3")
    if m1 + m2 + m3 != 0 or not _triangle(j1, j2, j3):
        return Surd()

    exps: "dict[int, int]" = {}
    _add_factorial(exps, int(j1 + j2 - j3))
    _add_factorial(exps, int(j1 - j2 + j3))
    _add_factorial(exps, int(-j1 + j2 + j3))
    _add_factorial(exps, int(j1 + j2 + j3 + 1), -1)
    for j, m in ((j1, m1), (j2, m2), (j3, m3)):
        _add_factorial(exps, int(j + m))
        _add_factorial(exps, int(j - m))
    outside, rad = _sqrt_of_exponents(exps)

    t_lo = int(max(0, j2 - j3 - m1, j1 - j3 + m2))
    t_hi = int(min(j1 + j2 - j3, j1 - m1, j2 + m2))
    total = Fraction(0)
    for t in range(t_lo, t_hi + 1):
        denom = (math.factorial(t)
                 * math.factorial(int(j3 - j2 + t + m1))
                 * math.factorial(int(j3 - j1 + t - m2))
                 * math.factorial(int(j1 + j2 - j3 - t))
                 * math.factorial(int(j1 - t - m1))
                 * math.factorial(int(j2 - t + m2)))
        total += Fraction(_phase(t), denom)
    if not total:
        return Surd()
    return Surd.term(_phase(int(j1 - j2 - m3)) * outside * total, rad)


def clebsch_gordan(j1, m1, j2, m2, big_j, big_m) -> Surd:
    """``⟨j1 m1 j2 m2 | J M⟩`` in the Condon–Shortley convention."""
    j1, m1, j2, m2 = parse_j(j1), parse_j(m1), parse_j(j2), parse_j(m2)
    big_j, big_m = parse_j(big_j), parse_j(big_m)
    _check_jm(big_j, big_m, "J", "M")
    if m1 + m2 != big_m:
        return Surd()
    three_j = wigner_3j(j1, j2, big_j, m1, m2, -big_m)
    if three_j.is_zero():
        return Surd()
    out, rad = _sqrt_int(int(2 * big_j + 1))
    return three_j * Surd.term(_phase(int(j1 - j2 + big_m)) * out, rad)


# ---------------------------------------------------------------------------
# 6j / 9j
# ---------------------------------------------------------------------------

def _delta_exponents(exps: "dict[int, int]", a: Fraction, b: Fraction,
                     c: Fraction, sign: int = 1) -> None:
    """Accumulate the triangle coefficient Δ(a,b,c) into ``exps``."""
    _add_factorial(exps, int(a + b - c), sign)
    _add_factorial(exps, int(a - b + c), sign)
    _add_factorial(exps, int(-a + b + c), sign)
    _add_factorial(exps, int(a + b + c + 1), -sign)


def wigner_6j(j1, j2, j3, j4, j5, j6) -> Surd:
    """The Wigner 6j symbol ``{j1 j2 j3 / j4 j5 j6}`` (Racah's closed form).

    Its four triads are ``(j1 j2 j3)``, ``(j1 j5 j6)``, ``(j4 j2 j6)``,
    ``(j4 j5 j3)``; the symbol is zero unless every one of them is a triangle.
    """
    js = [parse_j(x) for x in (j1, j2, j3, j4, j5, j6)]
    for name, j in zip(("j_1", "j_2", "j_3", "j_4", "j_5", "j_6"), js):
        _check_j(j, name)
    j1, j2, j3, j4, j5, j6 = js
    triads = ((j1, j2, j3), (j1, j5, j6), (j4, j2, j6), (j4, j5, j3))
    if not all(_triangle(*tri) for tri in triads):
        return Surd()

    exps: "dict[int, int]" = {}
    for tri in triads:
        _delta_exponents(exps, *tri)
    outside, rad = _sqrt_of_exponents(exps)

    t_lo = int(max(sum(tri) for tri in triads))
    quads = (j1 + j2 + j4 + j5, j2 + j3 + j5 + j6, j1 + j3 + j4 + j6)
    t_hi = int(min(quads))
    total = Fraction(0)
    for t in range(t_lo, t_hi + 1):
        denom = math.factorial(int(quads[0] - t)) * math.factorial(int(quads[1] - t)) \
            * math.factorial(int(quads[2] - t))
        for tri in triads:
            denom *= math.factorial(int(t - sum(tri)))
        total += Fraction(_phase(t) * math.factorial(t + 1), denom)
    if not total:
        return Surd()
    return Surd.term(outside * total, rad)


def wigner_9j(j1, j2, j3, j4, j5, j6, j7, j8, j9) -> Surd:
    """The Wigner 9j symbol, as the standard single sum over three 6j symbols.

    ``{…} = Σ_x (-1)^{2x} (2x+1) {j1 j2 j3 / j6 j9 x} {j4 j5 j6 / j2 x j8}
    {j7 j8 j9 / x j1 j4}`` — the summand's own triangle conditions bound ``x``.
    Unlike 3j/6j this is generally a *sum* of distinct surds, which is why
    :class:`Surd` carries a whole term map.
    """
    js = [parse_j(x) for x in (j1, j2, j3, j4, j5, j6, j7, j8, j9)]
    for idx, j in enumerate(js, start=1):
        _check_j(j, f"j_{idx}")
    j1, j2, j3, j4, j5, j6, j7, j8, j9 = js
    rows = ((j1, j2, j3), (j4, j5, j6), (j7, j8, j9))
    cols = ((j1, j4, j7), (j2, j5, j8), (j3, j6, j9))
    if not all(_triangle(*tri) for tri in rows + cols):
        return Surd()

    x_lo = max(abs(j1 - j9), abs(j2 - j6), abs(j4 - j8))
    x_hi = min(j1 + j9, j2 + j6, j4 + j8)
    total = Surd()
    x = x_lo
    while x <= x_hi:
        a = wigner_6j(j1, j2, j3, j6, j9, x)
        b = wigner_6j(j4, j5, j6, j2, x, j8)
        c = wigner_6j(j7, j8, j9, x, j1, j4)
        if not (a.is_zero() or b.is_zero() or c.is_zero()):
            weight = _phase(int(2 * x)) * (2 * x + 1)
            total = total + a * b * c * Fraction(weight)
        x += 1
    return total


# ---------------------------------------------------------------------------
# Coupling: allowed J, the j1·j2 eigenvalue, and the coupled-state expansion
# ---------------------------------------------------------------------------

def coupled_j_values(j1, j2) -> "list[Fraction]":
    """The allowed total angular momenta ``|j1-j2| … j1+j2`` (step 1)."""
    j1, j2 = parse_j(j1), parse_j(j2)
    _check_j(j1, "j_1")
    _check_j(j2, "j_2")
    out, j = [], abs(j1 - j2)
    while j <= j1 + j2:
        out.append(j)
        j += 1
    return out


def dot_product_eigenvalue(j1, j2, big_j) -> Fraction:
    """The eigenvalue of ``j1·j2 = ½[J(J+1) − j1(j1+1) − j2(j2+1)]``.

    The exact operator identity behind the Landé interval rule (hyperfine
    structure, LS multiplets) — general in its inputs, so the physics choice of
    *which* Hamiltonian to put it in stays with the researcher.
    """
    j1, j2, big_j = parse_j(j1), parse_j(j2), parse_j(big_j)
    if big_j not in coupled_j_values(j1, j2):
        raise AngularMomentumError("j_not_in_coupling", j=j_latex(big_j),
                                   j1=j_latex(j1), j2=j_latex(j2))
    return (big_j * (big_j + 1) - j1 * (j1 + 1) - j2 * (j2 + 1)) / 2


def cg_expansion(j1, j2, big_j, big_m) -> "list[tuple[Fraction, Fraction, Surd]]":
    """``|J M⟩ = Σ ⟨j1 m1 j2 m2|J M⟩ |j1 m1⟩|j2 m2⟩`` as ``(m1, m2, coeff)``.

    Only the non-vanishing terms are returned, ordered by descending ``m1``.
    """
    j1, j2 = parse_j(j1), parse_j(j2)
    big_j, big_m = parse_j(big_j), parse_j(big_m)
    _check_jm(big_j, big_m, "J", "M")
    if big_j not in coupled_j_values(j1, j2):
        raise AngularMomentumError("j_not_in_coupling", j=j_latex(big_j),
                                   j1=j_latex(j1), j2=j_latex(j2))
    out = []
    m1 = j1
    while m1 >= -j1:
        m2 = big_m - m1
        if abs(m2) <= j2 and _is_int(j2 - m2):
            coeff = clebsch_gordan(j1, m1, j2, m2, big_j, big_m)
            if not coeff.is_zero():
                out.append((m1, m2, coeff))
        m1 -= 1
    return out


# ---------------------------------------------------------------------------
# Spectroscopy / magnetic resonance bridges
# ---------------------------------------------------------------------------

L_LETTERS = "SPDFGHIKLMNOQRTUV"


def parse_term_symbol(text: str) -> "tuple[Fraction, Fraction, Fraction]":
    """Parse an atomic term symbol ``^{2}P_{3/2}`` → ``(S, L, J)``.

    Accepts ``^2P_{3/2}``, ``^{2}P_{3/2}``, ``2P3/2`` and the ``{}``-less
    variants.  The multiplicity ``2S+1`` gives S; the letter gives L.
    """
    raw = str(text).strip().translate(_TRANSLATE_FULLWIDTH)
    m = re.match(r"^\^?\{?\s*(\d+)\s*\}?\s*([A-Za-z])\s*_?\{?\s*([\d/]+)\s*\}?$", raw)
    if m is None:
        raise AngularMomentumError("bad_term_symbol", value=raw)
    mult = int(m.group(1))
    letter = m.group(2).upper()
    if letter not in L_LETTERS:
        raise AngularMomentumError("bad_term_symbol", value=raw)
    spin = Fraction(mult - 1, 2)
    orbital = Fraction(L_LETTERS.index(letter))
    total = parse_j(m.group(3))
    if total not in coupled_j_values(orbital, spin):
        raise AngularMomentumError("j_not_in_coupling", j=j_latex(total),
                                   j1=j_latex(orbital), j2=j_latex(spin))
    return spin, orbital, total


def lande_g_factor(orbital, spin, total, g_l=1, g_s=2) -> Fraction:
    """The Landé g factor of an ``LS``-coupled level.

    ``g_J = g_L (J(J+1)+L(L+1)−S(S+1))/(2J(J+1))
            + g_S (J(J+1)+S(S+1)−L(L+1))/(2J(J+1))``

    — the projection theorem (a Wigner–Eckart corollary) applied to
    ``μ = −μ_B(g_L L + g_S S)``.  ``g_L``/``g_S`` are inputs, so the usual
    ``g_S = 2`` approximation is a *default*, not a baked-in physics choice:
    pass ``g_s = 2.002319…`` (as a Fraction) for the QED-corrected value.
    """
    orbital, spin, total = parse_j(orbital), parse_j(spin), parse_j(total)
    _check_j(orbital, "L")
    _check_j(spin, "S")
    if total not in coupled_j_values(orbital, spin):
        raise AngularMomentumError("j_not_in_coupling", j=j_latex(total),
                                   j1=j_latex(orbital), j2=j_latex(spin))
    if total == 0:
        raise AngularMomentumError("g_factor_undefined_j0")
    jj = total * (total + 1)
    ll = orbital * (orbital + 1)
    ss = spin * (spin + 1)
    return (Fraction(g_l) * (jj + ll - ss) + Fraction(g_s) * (jj + ss - ll)) / (2 * jj)


def wigner_eckart_coefficient(j, m, rank, q, jp, mp) -> Surd:
    """The geometric factor of the Wigner–Eckart theorem,

    ``⟨j' m'|T^k_q|j m⟩ = ⟨j m k q|j' m'⟩ / √(2j'+1) · ⟨j'‖T^k‖j⟩``.

    Everything m-dependent — hence every selection rule and every relative line
    strength — is this coefficient; the reduced matrix element is the one thing
    it cannot give you.
    """
    jp = parse_j(jp)
    coeff = clebsch_gordan(j, m, rank, q, jp, mp)
    if coeff.is_zero():
        return Surd()
    out, rad = _sqrt_int(int(2 * jp + 1))
    # 1/√(2j'+1) = √(2j'+1)/(2j'+1)
    return coeff * Surd.term(Fraction(out, int(2 * jp + 1)), rad)
