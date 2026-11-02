"""F-9 display forms: LaTeX for coupling brackets, Wigner symbols and coupled
states.

Kept apart from :mod:`notebook.runtime.angular_momentum` for the same reason
``maxima_derivative_latex`` is kept apart from the handler: the exact-arithmetic
kernel should stay readable as mathematics, and the notation it renders into is
a separate concern that grows on its own schedule.

These are LaTeX *notation* builders, not user-visible prose — the i18n rule does
not apply (there is nothing to translate in ``\\langle j_1 m_1 …``).
"""
from __future__ import annotations

from fractions import Fraction

from notebook.runtime.angular_momentum import L_LETTERS, Surd, j_canonical, j_latex
from notebook.runtime.integral_form import latex_var

__all__ = [
    "cg_braket_latex", "wigner_symbol_latex", "ket_latex",
    "expansion_latex", "expansion_canonical",
    "dot_product_latex", "dot_product_canonical",
    "lande_g_latex", "wigner_eckart_latex", "allowed_j_latex",
    "allowed_j_canonical", "term_symbol_latex",
]


def cg_braket_latex(j1, m1, j2, m2, big_j, big_m) -> str:
    """``⟨j1 m1 j2 m2 | J M⟩`` with the numbers substituted."""
    left = r"\,".join(j_latex(x) for x in (j1, m1, j2, m2))
    right = r"\,".join(j_latex(x) for x in (big_j, big_m))
    return rf"\langle {left} \,|\, {right} \rangle"


def wigner_symbol_latex(kind: str, rows: "list[list[Fraction]]") -> str:
    """A 3j / 6j / 9j symbol: 3j in round brackets, 6j and 9j in braces."""
    body = r" \\ ".join(" & ".join(j_latex(x) for x in row) for row in rows)
    env = "pmatrix" if kind == "3j" else "Bmatrix"
    return rf"\begin{{{env}}} {body} \end{{{env}}}"


def ket_latex(*quantum_numbers) -> str:
    """``|a, b⟩`` for the given quantum numbers."""
    return r"\left| " + ",\\, ".join(j_latex(x) for x in quantum_numbers) + r" \right\rangle"


def expansion_latex(j1, j2, big_j, big_m,
                    terms: "list[tuple[Fraction, Fraction, Surd]]",
                    basis: "tuple[str, str] | None" = None,
                    state_symbol: str | None = None) -> str:
    """``|J M⟩ = Σ ⟨…|J M⟩ |j1 m1⟩|j2 m2⟩`` — the coupled state in the
    uncoupled (product) basis.

    With ``basis=("chi", "phi")`` the product states are written as the NAMED
    functions ``\\chi(m_1)\\phi(m_2)`` instead of anonymous kets.  That display
    mirrors the canonical exactly (``chi(1/2)*phi(-1/2)``), which is the whole
    point: a named expansion is a real expression the CAS can work on, so what
    is shown and what can be computed with are the same thing.
    """
    rhs = ""
    for m1, m2, coeff in terms:
        negative = all(c < 0 for c in coeff.terms.values())
        body = (-coeff).latex() if negative else coeff.latex()
        if body == "1":
            body = ""
        if rhs:
            rhs += " - " if negative else " + "
        elif negative:
            rhs += "-"
        if basis:
            product = (_basis_latex(basis[0], m1) + r"\," + _basis_latex(basis[1], m2))
        else:
            product = ket_latex(j1, m1) + ket_latex(j2, m2)
        rhs += body + product
    lhs = (latex_var(state_symbol) if state_symbol
           else ket_latex(big_j, big_m))
    return f"{lhs} = {rhs or '0'}"


def _basis_latex(name: str, m) -> str:
    """``chi`` at ``m`` → ``\\chi\\!\\left(\\frac{1}{2}\\right)``.

    Basis names are plain identifiers (the engine rejects subscripted ones), so
    this is a straight Greek-or-literal lookup and the display maps back to the
    canonical without ambiguity.
    """
    return rf"{latex_var(name)}\!\left({j_latex(m)}\right)"


def expansion_canonical(terms: "list[tuple[Fraction, Fraction, Surd]]",
                        basis: "tuple[str, str]") -> str:
    """The named expansion as a CAS expression: ``sqrt(2)/2*chi(1/2)*phi(-1/2) + …``.

    The basis symbols are ordinary undeclared function names, so the notebook's
    existing undefined-symbol detection prompts the researcher to declare them —
    the same guidance path as any other new symbol.
    """
    out = ""
    for m1, m2, coeff in terms:
        negative = all(c < 0 for c in coeff.terms.values())
        body = (-coeff).canonical() if negative else coeff.canonical()
        if out:
            out += " - " if negative else " + "
        elif negative:
            out += "-"
        product = (f"{basis[0]}({j_canonical(m1)})*{basis[1]}({j_canonical(m2)})")
        out += product if body == "1" else f"{body}*{product}"
    return out or "0"


def _bold_j(symbol: str) -> str:
    return rf"\mathbf{{{symbol}}}"


def dot_product_latex(j1, j2, total_symbol: str = "J",
                      symbols: "tuple[str, str] | None" = None) -> str:
    """``j1·j2 = ½[J(J+1) − j1(j1+1) − j2(j2+1)]`` with j1, j2 substituted.

    The general operator identity behind the Landé interval rule; ``J`` stays
    SYMBOLIC so the researcher substitutes the level they want (or hands the
    whole thing to the CAS).
    """
    a, b = symbols or ("j_1", "j_2")
    lhs = rf"{_bold_j(a)}\cdot{_bold_j(b)}"
    c1 = _product_latex(j1)
    c2 = _product_latex(j2)
    return (rf"{lhs} = \frac{{1}}{{2}}\left[ {total_symbol}({total_symbol}+1)"
            rf" - {c1} - {c2} \right]")


def dot_product_canonical(j1, j2, total_symbol: str = "J") -> str:
    """CAS form of :func:`dot_product_latex` — composable with ordinary
    operations (substitute a value for J, multiply by a coupling constant, …)."""
    v1 = Fraction(j1) * (Fraction(j1) + 1)
    v2 = Fraction(j2) * (Fraction(j2) + 1)
    return (f"({total_symbol}*({total_symbol}+1) - {j_canonical(v1)}"
            f" - {j_canonical(v2)})/2")


def _product_latex(j) -> str:
    """``j(j+1)`` evaluated: ``3/2`` → ``\\frac{15}{4}``."""
    j = Fraction(j)
    return j_latex(j * (j + 1))


def allowed_j_latex(j1, j2, values, total_symbol: str = "J") -> str:
    """``J = 1, 2`` — the allowed total angular momenta of j1 ⊗ j2."""
    listed = ",\\; ".join(j_latex(v) for v in values)
    return (rf"{ket_latex(j1)} \otimes {ket_latex(j2)}:\quad "
            rf"{total_symbol} = {listed}")


def allowed_j_canonical(values) -> str:
    """The allowed J values as a CAS list, so they can be iterated over."""
    return "[" + ", ".join(j_canonical(v) for v in values) + "]"


def term_symbol_latex(spin, orbital, total) -> str:
    """``^{2}P_{3/2}`` from (S, L, J).

    The J subscript is written ``3/2``, not ``\\frac{3}{2}`` — a stacked
    fraction in a term-symbol subscript is not how spectroscopic notation is
    set, and it is unreadable at subscript size.
    """
    mult = int(2 * Fraction(spin) + 1)
    letter = L_LETTERS[int(Fraction(orbital))]
    return rf"^{{{mult}}}\!{letter}_{{{j_canonical(Fraction(total))}}}"


def lande_g_latex(level_latex: str, value: Fraction) -> str:
    """``g_J(^2P_{3/2}) = 4/3``."""
    return rf"g_J\left({level_latex}\right) = {j_latex(value)}"


def wigner_eckart_latex(j, m, rank, q, jp, mp, coeff: Surd,
                        operator: str = "T") -> str:
    """``⟨j' m'|T^k_q|j m⟩ = <coeff> ⟨j'‖T^k‖j⟩``.

    The reduced matrix element stays a symbol — it is exactly the part the
    theorem does *not* determine.
    """
    op = rf"{operator}^{{({j_latex(rank)})}}_{{{j_latex(q)}}}"
    lhs = (rf"\langle {j_latex(jp)}\,{j_latex(mp)} | {op} | "
           rf"{j_latex(j)}\,{j_latex(m)} \rangle")
    reduced = (rf"\langle {j_latex(jp)} \| {operator}^{{({j_latex(rank)})}} \| "
               rf"{j_latex(j)} \rangle")
    if coeff.is_zero():
        return f"{lhs} = 0"
    body = coeff.latex()
    return f"{lhs} = {'' if body == '1' else body} {reduced}"
