"""F-1 stage 3c — composable symbolic spin algebra (breaks "problem B").

The engine's commutator result is display LaTeX + an opaque canonical, so a second
commutator cannot be composed onto it.  This module holds spin-operator expressions
as **structured, composable data** instead:

    term       = (coeff, ops)        # coeff a Coeff, ops an ordered tuple
    operator   = ((comp, site), …)   # comp ∈ {x,y,z}, site an integer
    expression = [term, …]

Commutators use the su(2) relation [S^a_i, S^b_j] = i δ_ij ε_abc S^c_i and the
Leibniz rule [PQ,R] = P[Q,R] + [P,R]Q.  The result is the same data structure, so
it feeds straight back in — nested commutators [H,[H,…[H,O]…]] (equation-of-motion
/ moment hierarchies) compose cleanly.

Coefficients are exact: a polynomial over Gaussian integers in the coupling
symbols (commutators only introduce ±1 and i; the Hamiltonian introduces the
couplings), implemented self-contained (no SymPy dependency).
"""
from __future__ import annotations

import re

from notebook.runtime.shared_parser import strip_lhs
from fractions import Fraction


def _frac_latex(f: Fraction) -> str:
    """A positive Fraction as LaTeX: ``3``→``3``, ``1/16``→``\\frac{1}{16}``."""
    return str(f.numerator) if f.denominator == 1 else \
        rf"\frac{{{f.numerator}}}{{{f.denominator}}}"


def _mmul(m1, m2):
    """Multiply two parameter monomials (tuples of (name, exp))."""
    d: "dict[str, int]" = {}
    for name, e in m1 + m2:
        d[name] = d.get(name, 0) + e
    return tuple(sorted((n, e) for n, e in d.items() if e))


def _mdiv(m1, m2):
    """Divide two parameter monomials (subtract exponents)."""
    d: "dict[str, int]" = {}
    for name, e in m1:
        d[name] = d.get(name, 0) + e
    for name, e in m2:
        d[name] = d.get(name, 0) - e
    return tuple(sorted((n, e) for n, e in d.items() if e))


def _mono_latex(m) -> str:
    return " ".join(n if e == 1 else rf"{n}^{{{e}}}" for n, e in m)


_Z = Fraction(0)


class Coeff:
    """A polynomial over Gaussian *rationals* in the coupling symbols: a map from a
    parameter monomial to (real, imag) Fraction parts.  Rationals are needed for the
    spin-½ infinite-temperature traces (factors of 1/4); the commutator algebra
    itself stays integer."""
    __slots__ = ("d",)

    def __init__(self, d=None):
        self.d = {m: (Fraction(r), Fraction(i)) for m, (r, i) in (d or {}).items()
                  if (r, i) != (0, 0)}

    @staticmethod
    def int(n):
        return Coeff({(): (n, 0)})

    @staticmethod
    def imag(sign=1):
        return Coeff({(): (0, sign)})

    @staticmethod
    def param(name):
        return Coeff({((name, 1),): (1, 0)})

    def is_zero(self):
        return not self.d

    def __eq__(self, o):
        return isinstance(o, Coeff) and self.d == o.d

    def conj(self):
        return Coeff({m: (r, -i) for m, (r, i) in self.d.items()})

    def __truediv__(self, o):
        """Divide by a single-monomial Coeff (the homogeneous norms in Lanczos)."""
        if len(o.d) != 1:
            raise ValueError("Coeff division supports a monomial divisor only")
        (dm, (dr, di)), = o.d.items()
        den = dr * dr + di * di
        out = {}
        for m, (r, i) in self.d.items():
            out[_mdiv(m, dm)] = ((r * dr + i * di) / den, (i * dr - r * di) / den)
        return Coeff(out)

    def __add__(self, o):
        d = dict(self.d)
        for m, (r, i) in o.d.items():
            rr, ii = d.get(m, (_Z, _Z))
            d[m] = (rr + r, ii + i)
        return Coeff(d)

    def __mul__(self, o):
        d: "dict[tuple, tuple]" = {}
        for m1, (r1, i1) in self.d.items():
            for m2, (r2, i2) in o.d.items():
                m = _mmul(m1, m2)
                nr, ni = r1 * r2 - i1 * i2, r1 * i2 + r2 * i1
                rr, ii = d.get(m, (_Z, _Z))
                d[m] = (rr + nr, ii + ni)
        return Coeff(d)

    def to_canonical(self) -> str:
        """CAS-ready form of the polynomial: ``J^2*x1*x2/3``.

        The display counterpart of :meth:`to_latex`.  A spin trace is a number
        or a polynomial in the couplings — an ordinary CAS value — so it should
        be substitutable and combinable like any other result, not a picture.
        """
        if self.is_zero():
            return "0"
        parts = []
        for m in sorted(self.d):
            re, im = self.d[m]
            factors = [name if e == 1 else f"{name}^{e}" for name, e in m]
            for value, imaginary in ((re, False), (im, True)):
                if not value:
                    continue
                pieces = list(factors)
                if imaginary:
                    pieces.insert(0, "%i")
                if abs(value.numerator) != 1 or not pieces:
                    pieces.insert(0, str(abs(value.numerator)))
                body = "*".join(pieces)
                if value.denominator != 1:
                    body = f"({body})/{value.denominator}" if len(pieces) > 1 \
                        else f"{body}/{value.denominator}"
                parts.append(("-" if value < 0 else "+") + body)
        out = "".join(parts)
        return out[1:] if out.startswith("+") else out

    def to_latex(self) -> str:
        if self.is_zero():
            return "0"
        out = ""
        for idx, m in enumerate(sorted(self.d)):
            re, im = self.d[m]
            mstr = _mono_latex(m)
            if im == 0:
                sign = "-" if re < 0 else "+"
                num = "" if (abs(re) == 1 and mstr) else _frac_latex(abs(re))
            elif re == 0:
                sign = "-" if im < 0 else "+"
                num = "i" if abs(im) == 1 else f"{_frac_latex(abs(im))} i"
            else:                                   # genuinely complex (rare)
                sign = "+"
                num = f"({re} + {im} i)"
            body = (num + (" " if num and mstr else "") + mstr) or "1"
            out += (("-" + body) if sign == "-" else body) if idx == 0 \
                else f" {sign} {body}"
        return out


# Levi-Civita for [S^a, S^b] = i ε_abc S^c (same site).
_LEVI = {
    ("x", "y"): ("z", 1), ("y", "z"): ("x", 1), ("z", "x"): ("y", 1),
    ("y", "x"): ("z", -1), ("z", "y"): ("x", -1), ("x", "z"): ("y", -1),
}


def _comm_single(p, q):
    (a, si), (b, sj) = p, q
    if si != sj or a == b:
        return []
    c, sign = _LEVI[(a, b)]
    return [(Coeff.imag(sign), ((c, si),))]


def _comm_op_prod(p, q_ops):
    """[p, q1 q2 … qk] = Σ_i q1…[p,q_i]…qk."""
    out = []
    for idx, qi in enumerate(q_ops):
        for cf, ops in _comm_single(p, qi):
            out.append((cf, q_ops[:idx] + ops + q_ops[idx + 1:]))
    return out


def _comm_prod_prod(p_ops, q_ops):
    """[p1…pm, Q] = Σ_i p1…[p_i,Q]…pm."""
    out = []
    for idx, pi in enumerate(p_ops):
        for cf, ops in _comm_op_prod(pi, q_ops):
            out.append((cf, p_ops[:idx] + ops + p_ops[idx + 1:]))
    return out


class ComplexityLimitExceeded(Exception):
    """Raised by ``_collect`` when a term list grows past ``_TERM_LIMIT``.

    On a chain, adjacent Hamiltonian bonds share a site, so repeated commutators
    keep reintroducing same-site factors that never cancel in ``_collect``'s
    exact-match deduplication.  A 4-site chain's ``continued_fraction`` at
    (interpret_early.py's fixed) depth 8 grew 1 → 967 terms in 4 iterations,
    ~20s and still climbing, heading for a multi-minute computation with no
    useful result.  Raising here turned that silent hang into an immediate,
    honest failure.

    **A-5d (2026-08-06): the same-site products ARE reduced now, and this limit
    no longer fires for that 4-site chain — it completes in ~0.1s.**  The limit
    stays because the reduction MOVED the wall rather than abolishing it
    (measured: 4 sites 0.11s, 6 sites 4.9s, 8 sites hits this limit in ~1s), and
    because it is spin-½-specific: on the DISPLAY path a general-spin
    ``(S^z_2)^2`` is a deliberate result and is not collapsed, so an
    undeclared large system still reaches this.  See
    ``_reduce_same_site_half`` for which path reduces when, and
    ``interpreter.spin_algebra_too_large`` for what the researcher is told
    (the two paths have different levers).
    """


_TERM_LIMIT = 3000


#: 簡約を許すスピンの大きさ。**½ に限る**——一般スピン $S$ では
#: $(S^a)^2$ も $\{S^a,S^b\}$ も定数ではないので、同じ規則が使えない。
_REDUCIBLE_SPIN = Fraction(1, 2)


def _reduce_same_site_half(ops):
    """spin-½ の同一サイト積を畳む。``(残る ops, 係数)`` を返す。

    su(2) の単一サイト積は spin-½ でだけ閉じる:

        S^a S^a = 1/4            S^a S^b = (i/2) ε_abc S^c   (a≠b)

    どちらの恒等式も **½ 固有**である（一般スピンでは $(S^a)^2$ は演算子のまま）。
    したがってこの関数は、そのサイトのスピンが ½ だと**宣言されている**ときにだけ
    呼ばれる。呼ばないときの表現は従来どおり一般スピンのまま。

    **なぜ要るのか（実測 2026-08-06）。** 鎖では隣接ボンドがサイトを共有するので、
    交換子を重ねるたびに同一サイト因子が再導入され、`_collect` の完全一致
    デデュープでは決して打ち消し合わない。4サイト鎖で 2→12→76→474→2758 と
    約6倍/回で増え、6回目で打ち切りに当たっていた。畳むと
    2→6→14→21→24→28→30→32 と **頭打ちになる**——spin-½ の演算子空間は
    $4^N$ 次元の有限空間なので、項数はそこで止まる。**格子を固定したまま
    交換子を何段重ねても壁に当たらない**のはこのため。

    ただし飽和水準そのものは $4^N$ で増えるので、**格子を大きくすれば当たる**
    （実測 4サイト 0.11 秒／6サイト 4.9 秒／8サイトは打ち切り）。壁が消えたの
    ではなく、増大の法則が変わって壁が 4サイトから 8サイトへ動いた。
    `ComplexityLimitExceeded` はそのために残っている。
    """
    from itertools import groupby
    coeff = Coeff.int(1)
    out: "list[tuple[str, object]]" = []
    for site, grp in groupby(sorted(ops, key=lambda o: o[1]), key=lambda o: o[1]):
        run: "list[str]" = []
        for comp, _ in grp:
            if run and run[-1] == comp:                    # S^a S^a = 1/4
                run.pop()
                coeff = coeff * Coeff({(): (Fraction(1, 4), 0)})
            elif run:                                      # S^a S^b = (i/2) ε S^c
                c, sign = _LEVI[(run[-1], comp)]
                run[-1] = c
                coeff = coeff * Coeff({(): (0, Fraction(sign, 2))})
            else:
                run.append(comp)
        out += [(c, site) for c in run]
    return tuple(out), coeff


def _reducible_sites(spins):
    """``spins`` のうち ½ と宣言されたサイトの集合。"""
    if not spins:
        return frozenset()
    return frozenset(site for site, s in spins.items() if s == _REDUCIBLE_SPIN)


def _all_sites(*exprs):
    """*exprs* に現れる全サイト。

    **反跳法（`moments`／`continued_fraction`）専用。** そこでの内積 `inner` は
    `expectation`——すなわち **spin-½ の無限温度トレース**——であり、
    宣言の有無によらずそう定義されている（`expectation` の docstring／
    `expectation_general` が記号 $S$ を受け付けないこと）。つまりこの経路は
    最初から ½ に踏み込んでいるので、同一サイト積を畳んでよいかを研究者に
    訊く余地は無い。訊けば「計算が既に仮定していることを宣言してください」と
    言うだけになり、**壁に当たるのがまさにこの経路**である。

    表示に出る `commutator`／`nested_commutator` は別で、そちらは宣言された
    サイトしか畳まない——一般スピンの $(S^z_2)^2$ をそのまま見せる2サイト
    シナリオが現にあるため（open_items A-5d が「誤った修正方針」として
    警告していたもの）。
    """
    return frozenset(site for e in exprs for _, ops in e for _, site in ops)


def _canon_ops(ops):
    """Different sites commute → stable-sort by site (same-site order preserved)."""
    return tuple(sorted(ops, key=lambda o: o[1]))


def _collect(terms, half_sites=frozenset()):
    """項をまとめる。``half_sites`` に載ったサイトは同一サイト積を畳む。

    畳むのはここ——増大は**途中の項**で起きるので、最後にまとめて畳んでも
    間に合わない（`nested_commutator` の各段が既に膨れている）。
    """
    acc: "dict[tuple, Coeff]" = {}
    for c, ops in terms:
        if half_sites and any(site in half_sites for _, site in ops):
            _sites = {site for _, site in ops}
            if _sites <= half_sites:
                ops, extra = _reduce_same_site_half(ops)
                c = c * extra
        key = _canon_ops(ops)
        acc[key] = acc[key] + c if key in acc else c
    if len(acc) > _TERM_LIMIT:
        raise ComplexityLimitExceeded(
            f"term count exceeded {_TERM_LIMIT} — this system is larger than "
            "this module's validated range (2-3 sites); aborting instead of "
            "continuing an unbounded computation"
        )
    return [(c, key) for key, c in acc.items() if not c.is_zero()]


def commutator(e1, e2, spins=None):
    """[e1, e2] for two expressions (lists of (Coeff, ops)).

    *spins* は「サイト → スピンの大きさ」。渡されたサイトのうち **½** のものは
    同一サイト積を畳む（`_reduce_same_site_half`）。渡さなければ従来どおり
    一般スピンの表現を保つ——この区別が要るのは、$(S^a)^2$ が定数になるのが
    ½ に限られるからである。
    """
    out = []
    for c1, p in e1:
        for c2, q in e2:
            for cf, ops in _comm_prod_prod(p, q):
                out.append((c1 * c2 * cf, ops))
    return _collect(out, _reducible_sites(spins))


def nested_commutator(h, o, order: int, spins=None):
    """[H,[H,…[H,O]…]] with *order* nested commutators (order 1 = [H,O]).

    *spins* は各段へ伝える。**段ごとに畳まないと意味がない**——増大は途中の項で
    起きるので、最後にまとめて畳んでも既に膨れたあとになる。
    """
    cur = o
    for _ in range(order):
        cur = commutator(h, cur, spins)
    return cur


def spin_operator(comp: str, site: int):
    return [(Coeff.int(1), ((comp, site),))]


# ---------------------------------------------------------------------------
# Spin-expression parser: LaTeX/text → term list (the inverse of to_latex).
#
# Lets a custom spin Hamiltonian (e.g. "-\omega_0 S^z_1 - \omega_1 S^x_1") be
# entered as a general expression and then operated on with the general spin
# primitives (rotating_frame, commutator, …) instead of a canned per-scenario
# helper.  Round-trips with to_latex for sums of (param·int coefficient)·(product
# of spin operators) terms.
# ---------------------------------------------------------------------------

_SPIN_OP_RE = re.compile(
    r'(?:\\left\()?\s*S\^\{?([xyz])\}?_\{?(\d+)\}?\s*(?:\\right\))?'
    r'(?:\^\{?(\d+)\}?)?'
)


def _split_signed_terms(s: str) -> "list[tuple[str, str]]":
    """Split a sum at top-level +/- (respecting {…} and (…) nesting)."""
    out, depth, start, sign = [], 0, 0, "+"
    i = 0
    while i < len(s):
        ch = s[i]
        if ch in "{(":
            depth += 1
        elif ch in "})":
            depth -= 1
        elif ch in "+-" and depth == 0:
            chunk = s[start:i].strip()
            if chunk:
                out.append((sign, chunk))
            sign = ch
            start = i + 1
        i += 1
    chunk = s[start:].strip()
    if chunk:
        out.append((sign, chunk))
    return out


def _parse_coeff_product(s: str) -> "Coeff":
    """Parse a single product of factors (params / integers), no sign."""
    c = Coeff.int(1)
    for factor in s.split():
        factor = factor.strip()
        if not factor:
            continue
        if re.fullmatch(r"\d+", factor):
            c = c * Coeff.int(int(factor))
        else:
            c = c * Coeff.param(factor)
    return c


def _parse_coeff(s: str, sign: str) -> "Coeff":
    """Parse a coefficient with a leading sign.  Handles a product of params /
    integers and a single parenthesized polynomial factor, e.g. the (ω−ω₀) that
    rotating_frame produces — so to_latex output round-trips."""
    c = Coeff.int(-1) if sign == "-" else Coeff.int(1)
    s = (s.replace(r"\left(", "(").replace(r"\right)", ")")
         .replace("*", " ").replace("\\,", " ").strip())
    if s.startswith("(") and s.endswith(")"):
        poly = Coeff.int(0)
        for isign, term in _split_signed_terms(s[1:-1].strip()):
            t = _parse_coeff_product(term)
            poly = poly + (Coeff.int(-1) * t if isign == "-" else t)
        return c * poly
    return c * _parse_coeff_product(s)


def parse_spin_expression(text: str):
    """Parse "−\\omega_0 S^z_1 − \\omega_1 S^x_1" → term list (inverse of to_latex).

    The optional "LHS =" prefix is stripped.  A term may carry a product of spin
    operators and a coefficient that is a product of parameters / integers.  Raises
    ValueError if no spin operator is found (not a spin expression)."""
    # Bracket-aware: this parser has NO CAS behind it (CLAUDE.md F-13,
    # CAS-bypass Python), so a mis-split decides the value with nothing
    # downstream able to object.
    s = strip_lhs((text or "").strip())
    if not _SPIN_OP_RE.search(s):
        raise ValueError("not a spin expression (no S^a_i operator found)")
    terms = []
    for sign, term in _split_signed_terms(s):
        ops = []
        for m in _SPIN_OP_RE.finditer(term):
            comp, site, pow_ = m.group(1), int(m.group(2)), m.group(3)
            ops.extend([(comp, site)] * (int(pow_) if pow_ else 1))
        coeff_str = _SPIN_OP_RE.sub(" ", term)
        terms.append((_parse_coeff(coeff_str, sign), tuple(ops)))
    return _collect(terms)


# (No `rotating_frame` helper: "moving to a frame rotating at ω about the z-axis"
# under the RWA / co-rotating-drive convention is just ADDING the frame term +ω S^z
# to the lab Hamiltonian — which is plain expression addition.  The scenario does it
# explicitly via "H_rot = E1 + ω S^z_1", so the physical choice is in the user's
# input, not hidden in a dressed-up "rotating frame" operation.  The general U H U†
# similarity transform is NOT implemented; it would be a separate, honestly-named op.)


def coeff_vector_norm(expr) -> str:
    """Euclidean norm √(Σ_a c_a²) of the single-operator coefficient vector of a spin
    expression H = Σ_a c_a S^a (read straight off the components — H is already the
    component-form inner product, so no vector is materialised).

    General — not a physics-named helper: it just norms whatever linear coefficients
    are present, for ANY single-spin expression.  Physically, for the Zeeman form
    H = −B·S this is |B|, the magnitude of the (effective) field acting on the spin.
    Coefficients may be polynomial (e.g. ω−ω₀ before the Ω substitution)."""
    comps: "dict[str, Coeff]" = {}
    for c, ops in expr:
        if len(ops) == 1:
            (comp, _site), = ops
            comps[comp] = comps.get(comp, Coeff({})) + c
    sq = []
    for comp in ("z", "y", "x"):                 # z first → √(Ω²+ω₁²) convention
        c = comps.get(comp)
        if not c or c.is_zero():
            continue
        cl = c.to_latex()
        if cl.startswith("-"):                   # the square is sign-independent
            cl = cl[1:].strip()
        token = cl if re.fullmatch(r"\\?[A-Za-z][A-Za-z0-9_{}\\^]*", cl) \
            else rf"\left({cl}\right)"
        sq.append(rf"{token}^2")
    return rf"\sqrt{{ {' + '.join(sq)} }}"


# (The canonical rotating-frame Hamiltonian and effective-field helpers were
# removed: the scenario is now built from the GENERAL primitives parse_spin_
# expression / rotating_frame / effective_field / commutator, so no canned
# per-scenario Hamiltonian is baked in.)


# Per-component bond couplings for the named spin models.
_MODEL_COUPLINGS = {
    "heisenberg": {"x": "J", "y": "J", "z": "J"},
    "ising":      {"z": "J"},
    "xy":         {"x": "J", "y": "J"},
    "xxz":        {"x": "J", "y": "J", "z": "J_z"},   # anisotropy via J_z (Δ=J_z/J)
}


def build_hamiltonian(bonds, kind: str = "heisenberg", n_sites=None, field=None):
    """Spin Hamiltonian on a finite lattice from a bond list (each (i,j)).

    heisenberg: J Σ S_i·S_j; ising: J Σ S^z_i S^z_j; xy: J Σ (S^x S^x+S^y S^y);
    xxz: J Σ (S^x S^x+S^y S^y) + J_z Σ S^z S^z.
    *field* = (symbol, dir) adds −h Σ_i S^dir_i over sites 1…n_sites (e.g. a
    transverse field ("h","x")).
    """
    couplings = _MODEL_COUPLINGS[kind]
    terms = [(Coeff.param(js), ((c, i), (c, j_)))
             for i, j_ in bonds for c, js in couplings.items()]
    if field:
        hsym, fdir = field
        coeff = Coeff.int(-1) * Coeff.param(hsym)
        terms += [(coeff, ((fdir, s),)) for s in range(1, (n_sites or 0) + 1)]
    return terms


def _ops_to_latex(ops) -> str:
    """Operator product → LaTeX, collapsing identical adjacent factors to a power."""
    out, i = [], 0
    while i < len(ops):
        j = i
        while j < len(ops) and ops[j] == ops[i]:
            j += 1
        comp, site = ops[i]
        base = rf"S^{{{comp}}}_{{{site}}}"
        out.append(base if j - i == 1 else rf"\left({base}\right)^{{{j - i}}}")
        i = j
    return " ".join(out)


def to_latex(expr) -> str:
    """Render an expression (list of (Coeff, ops)) as a signed LaTeX sum."""
    if not expr:
        return "0"
    s = ""
    for idx, (c, ops) in enumerate(expr):
        op_l = _ops_to_latex(ops)
        cl = c.to_latex()
        if len(c.d) > 1:                          # polynomial coeff → parenthesise
            sign, body = "+", (rf"\left( {cl} \right) {op_l}".strip() if op_l else cl)
        elif cl == "1":
            sign, body = "+", op_l
        elif cl == "-1":
            sign, body = "-", op_l
        elif cl.startswith("-"):
            sign, body = "-", f"{cl[1:].strip()} {op_l}".strip()
        else:
            sign, body = "+", f"{cl} {op_l}".strip()
        s = (("-" + body) if sign == "-" else body) if idx == 0 else f"{s} {sign} {body}"
    return s


# ---------------------------------------------------------------------------
# Infinite-temperature spin-½ inner product (A,B) = ⟨A† B⟩_{T=∞} = Tr(A†B)/Tr 1.
#
# The normalised trace factorises over sites; at one site the ordered product of
# spin-½ operators reduces, via S^a S^b = ¼δ_ab I + (i/2)ε_abc S^c, to a multiple
# of I plus a traceless part, and only the I-coefficient survives.  This is exactly
# the high-temperature moment expansion (recursion method) input.
# ---------------------------------------------------------------------------

def _gmul(x, y):
    (a, b), (c, d) = x, y
    return (a * c - b * d, a * d + b * c)


def _site_identity_coeff(comps):
    """The identity coefficient of an ordered spin-½ product at one site."""
    state = {"I": (Fraction(1), _Z)}                 # element = Σ_p coeff·S^p (p∈I,x,y,z)
    for a in comps:
        new: "dict[str, tuple]" = {}

        def _add(p, v):
            r, i = new.get(p, (_Z, _Z))
            new[p] = (r + v[0], i + v[1])
        for p, v in state.items():
            if p == "I":
                _add(a, v)                            # I·S^a = S^a
            elif p == a:
                _add("I", (v[0] / 4, v[1] / 4))       # (S^a)² = ¼ I
            else:
                c, sign = _LEVI[(p, a)]               # S^p S^a = (i/2)ε_pac S^c
                _add(c, _gmul(v, (_Z, Fraction(sign, 2))))
        state = new
    return state.get("I", (_Z, _Z))


def expectation(expr):
    """⟨expr⟩ at infinite temperature for spin-½ (a Coeff: couplings × rational)."""
    total = Coeff()
    for coeff, ops in expr:
        bysite: "dict[int, list]" = {}
        for comp, site in ops:
            bysite.setdefault(site, []).append(comp)
        g = (Fraction(1), _Z)
        for comps in bysite.values():
            g = _gmul(g, _site_identity_coeff(comps))
            if g == (_Z, _Z):
                break
        if g != (_Z, _Z):
            total = total + coeff * Coeff({(): g})
    return total


# ---------------------------------------------------------------------------
# F-1s: general-spin-S infinite-temperature single-site trace.
#
# Tr(S^{a1}…S^{ak})/Tr 1 for a general spin is a polynomial in x = S(S+1).  It is
# computed exactly: expand S^x,S^y into S^± and, for each basis state |m⟩, follow
# the (deterministic) raising/lowering path; only closed paths (return to m)
# survive the diagonal, and on a closed path the √ matrix elements pair up over
# each edge, so the amplitude is rational.  Evaluating exactly at a few spins and
# Lagrange-interpolating in x gives the polynomial.  At x=3/4 (spin-½) it
# reproduces the spin-½ trace.
# ---------------------------------------------------------------------------
from math import isqrt as _isqrt


def _site_trace_at(comps, spin: Fraction):
    """Exact Tr(Π S^comp)/(2S+1) at a definite spin, as (re, im) Fractions."""
    ms = [spin - i for i in range(int(2 * spin) + 1)]      # m = S, S−1, …, −S
    re, im = Fraction(0), Fraction(0)
    for m0 in ms:
        # DFS over the ± expansion: (index, m, rational, radicand product, i power)
        stack = [(0, m0, Fraction(1), 1, 0)]
        while stack:
            idx, m, rat, rad, ip = stack.pop()
            if idx == len(comps):
                if m == m0:
                    s = _isqrt(rad)                        # closed ⇒ rad is a square
                    mag = rat * s
                    ph = ip % 4
                    if ph == 0:   re += mag
                    elif ph == 1: im += mag
                    elif ph == 2: re -= mag
                    else:         im -= mag
                continue
            a = comps[-1 - idx]                            # apply right-to-left
            rp = (spin - m) * (spin + m + 1)               # |c_+(m)|²
            rm = (spin + m) * (spin - m + 1)               # |c_−(m)|²
            if a == "z":
                stack.append((idx + 1, m, rat * m, rad, ip))
            elif a == "x":                                 # S^x = (S^+ + S^-)/2
                if rp: stack.append((idx + 1, m + 1, rat / 2, rad * int(rp), ip))
                if rm: stack.append((idx + 1, m - 1, rat / 2, rad * int(rm), ip))
            else:                                          # S^y = (S^+ − S^-)/(2i)
                if rp: stack.append((idx + 1, m + 1, rat / 2, rad * int(rp), ip + 3))
                if rm: stack.append((idx + 1, m - 1, rat / 2, rad * int(rm), ip + 1))
    d = 2 * spin + 1
    return re / d, im / d


def _lagrange(xs, ys):
    """Exact Lagrange interpolation: polynomial coefficients [c0, c1, …] (Fractions)
    with P(xs[j]) = ys[j]."""
    n = len(xs)
    coeffs = [Fraction(0)] * n
    for j in range(n):
        # basis poly Π_{k≠j} (x − xs[k]) / (xs[j] − xs[k])
        num = [Fraction(1)]                                # polynomial coeffs, low→high
        denom = Fraction(1)
        for k in range(n):
            if k == j:
                continue
            num = [Fraction(0)] + num                      # ×x
            for i in range(len(num) - 1):
                num[i] -= xs[k] * num[i + 1]
            denom *= (xs[j] - xs[k])
        for i in range(len(num)):
            coeffs[i] += ys[j] * num[i] / denom
    return coeffs


def _site_trace_poly(comps, var: str = "x"):
    """Tr(Π S^comp)/Tr 1 as a Coeff polynomial in ``var`` = S(S+1).

    ``var`` names the polynomial variable so that DIFFERENT SITES CAN CARRY
    DIFFERENT SPINS (F-13): an electron and a nucleus contribute
    ``x_1 = S(S+1)`` and ``x_2 = I(I+1)`` independently.  The default ``"x"``
    is the uniform-spin case every site shares.
    """
    if not comps:
        return Coeff.int(1)
    deg = len(comps) // 2
    spins = [Fraction(j + 1, 2) for j in range(deg + 1)]   # 1/2, 1, 3/2, …
    xs = [s * (s + 1) for s in spins]
    res = [_site_trace_at(comps, s) for s in spins]
    cre = _lagrange(xs, [r for r, _ in res])
    cim = _lagrange(xs, [i for _, i in res])
    d = {}
    for n in range(len(cre)):
        r, i = cre[n], cim[n]
        if r or i:
            d[() if n == 0 else ((var, n),)] = (r, i)
    return Coeff(d)


def subst_coeff(coeff: Coeff, var: str, value) -> Coeff:
    """Substitute a rational ``value`` for the parameter ``var`` in a Coeff.

    Used to put a DEFINITE spin into a site's trace polynomial
    (``x_2 → I(I+1) = 2`` for a spin-1 nucleus) while other sites stay
    symbolic.  Exact throughout — ``value`` must be rational.
    """
    value = Fraction(value)
    out: "dict[tuple, tuple]" = {}
    for m, (r, i) in coeff.d.items():
        factor = Fraction(1)
        rest = []
        for name, e in m:
            if name == var:
                factor *= value ** e
            else:
                rest.append((name, e))
        key = tuple(sorted(rest))
        rr, ii = out.get(key, (_Z, _Z))
        out[key] = (rr + r * factor, ii + i * factor)
    return Coeff(out)


def site_spin_variable(site) -> str:
    """The polynomial variable standing for site ``i``'s ``S_i(S_i+1)``.

    ``x1``, ``x2``, … — deliberately WITHOUT a subscript.  An undeclared
    ``x_1`` is lowered to the indexed form ``x(1)`` on the way to the CAS
    (the project's subscript rule), and a later ``substitute x_1 = 3/4`` then
    matches nothing and reports success anyway.  Since these results are meant
    to be composed with ordinary operations, the name has to survive that trip
    intact.  One place decides it so nothing can drift.
    """
    return f"x{site}"


def _normalize_site_spins(spins) -> "dict | None":
    """Validate a site→spin map into ``{site: Fraction | None}``.

    ``None`` for a site means "keep this site's spin symbolic".  A definite
    spin must be a non-negative half-integer — anything else is rejected rather
    than rounded, for the same reason the angular-momentum layer rejects it: a
    silently altered spin gives a plausible wrong trace.
    """
    if spins is None:
        return None
    out: "dict[object, object]" = {}
    for site, spec in spins.items():
        if spec is None:
            out[site] = None
            continue
        value = spec if isinstance(spec, Fraction) else Fraction(str(spec).strip())
        if value < 0 or (2 * value).denominator != 1:
            raise ValueError(f"spin must be a non-negative half-integer: {spec!r}")
        out[site] = value
    return out


def expectation_general(expr, spins=None):
    """⟨expr⟩ at infinite temperature for general spin, as a Coeff polynomial in
    x = S(S+1) (and the couplings).  At x=3/4 it equals the spin-½ expectation.

    ``spins`` (F-13) makes the sites INDEPENDENT species:

    * ``None`` — every site shares the one variable ``x`` (uniform spin).
    * a ``{site: spin}`` map — each site gets its own ``x_i``; a site mapped to
      a definite value is evaluated there (``x_2 → 2`` for a spin-1 nucleus),
      a site mapped to ``None`` — or absent from the map — stays symbolic as
      ``x_i``.  Once the caller says the sites differ, no site silently falls
      back to the shared ``x``.

    So ``⟨(I·S)²⟩`` is ``x_1 x_2 / 3``, and with S=½, I=1 it is ``1/2``.
    """
    spins = _normalize_site_spins(spins)
    total = Coeff()
    for coeff, ops in expr:
        bysite: "dict[int, list]" = {}
        for comp, site in ops:
            bysite.setdefault(site, []).append(comp)
        prod = Coeff.int(1)
        for site, comps in bysite.items():
            if spins is None:
                prod = prod * _site_trace_poly(comps)
            else:
                var = site_spin_variable(site)
                poly = _site_trace_poly(comps, var)
                value = spins.get(site)
                prod = prod * (poly if value is None
                               else subst_coeff(poly, var, value * (value + 1)))
            if prod.is_zero():
                break
        if not prod.is_zero():
            total = total + coeff * prod
    return total


def multiply(a, b):
    """Operator product A·B of two expressions (concatenate the ordered factors)."""
    return [(ca * cb, oa + ob) for ca, oa in a for cb, ob in b]


def dagger(expr):
    """A† : conjugate each coefficient and reverse each (Hermitian-factor) product."""
    return [(c.conj(), tuple(reversed(ops))) for c, ops in expr]


def inner(a, b):
    """(A,B) = ⟨A† B⟩_{T=∞} (spin-½)."""
    return expectation(multiply(dagger(a), b))


def moments(h, o, m: int):
    """Even Hamiltonian moments M_{2k} = (L^k O, L^k O), k=0…m, with L=[H,·]
    (the recursion-method / high-temperature moments).  Returns a list of Coeff.

    同一サイト積は **宣言によらず畳む**（`_all_sites` 参照）——`inner` が
    spin-½ のトレースである以上、この関数は既に ½ を仮定している。
    畳むのは各段でなければ効かない：増大はこのループの中で起きる。"""
    half = _all_sites(h, o)
    out, cur = [], o
    out.append(inner(cur, cur))
    for _ in range(m):
        cur = _collect(commutator(h, cur), half)
        out.append(inner(cur, cur))
    return out


def general_low_moments(kind: str, comp: str = "x"):
    """The general-lattice (un-expanded) low moments in terms of the coordination
    number z, for a vertex-transitive lattice (spin-½, ∞ T).  Returns [M_0, M_2].

    M_2 = z · (single-bond contribution): different bonds j≠j′ share only the
    central site, where the resulting single operator on j (or j′) is traceless, so
    cross-bond terms vanish and the second moment is exactly z times one bond's.
    (Higher moments M_4… bring in two-bond overlaps / triangles and are deferred.)
    """
    o0 = spin_operator(comp, 0)
    m0 = inner(o0, o0)
    rep = commutator(build_hamiltonian([(0, 1)], kind=kind), o0)
    m2 = Coeff.param("z") * inner(rep, rep)
    return [m0, m2]


def general_b1_squared(kind: str, comp: str = "x"):
    """The first recursion coefficient b_1² = M_2/M_0 as a general formula in z."""
    m0, m2 = general_low_moments(kind, comp)
    return m2 / m0


def energy_moment(kind: str, edges, n: int):
    """Infinite-temperature energy moment ⟨H^n⟩ = Tr H^n / Tr 1 on a finite graph
    (the high-temperature expansion coefficient of the energy)."""
    h = build_hamiltonian(edges, kind=kind)
    p = h
    for _ in range(n - 1):
        p = multiply(p, h)
    return expectation(p)


def general_energy_moment(kind: str, edges, n: int):
    """⟨H^n⟩ on a finite graph for general spin S (a Coeff polynomial in x=S(S+1))."""
    h = build_hamiltonian(edges, kind=kind)
    p = h
    for _ in range(n - 1):
        p = multiply(p, h)
    return expectation_general(p)


def general_spin_energy_moment_latex(kind: str, upto: int = 3) -> str:
    """⟨H^n⟩ as general-spin-S formulas in x=S(S+1) and the lattice invariants |E|, N△.
    Reduces to the spin-½ result at x=3/4 (e.g. Heisenberg ⟨H²⟩ = J² x²/3 |E|)."""
    e, ntri = Coeff.param("|E|"), Coeff.param(r"N_\triangle")
    c2 = general_energy_moment(kind, [(0, 1)], 2)
    parts = [rf"\langle H^2 \rangle = {(c2 * e).to_latex()}"]
    if upto >= 3:
        c3 = general_energy_moment(kind, [(0, 1)], 3)
        d3 = general_energy_moment(kind, [(0, 1), (1, 2), (2, 0)], 3) + Coeff.int(-3) * c3
        parts.append(rf"\langle H^3 \rangle = {((c3 * e) + (d3 * ntri)).to_latex()}")
    return r",\ ".join(parts) + r"\quad (x = S(S+1))"


def energy_moment_invariants(kind: str):
    """The cluster coefficients of the low energy moments: ⟨H²⟩=c₂|E|,
    ⟨H³⟩=c₃|E|+d₃N△.  c₂,c₃ are the single-bond values; d₃ is isolated from the
    triangle cluster (|E|=3, N△=1).  Returns (c₂, c₃, d₃) as Coeff."""
    c2 = energy_moment(kind, [(0, 1)], 2)
    c3 = energy_moment(kind, [(0, 1)], 3)
    d3 = energy_moment(kind, [(0, 1), (1, 2), (2, 0)], 3) + Coeff.int(-3) * c3
    return c2, c3, d3


# Representative clusters for the ⟨H⁴⟩ cluster fit, with their invariant rows
# [ |E|, |E|², wedges (Σ C(deg,2)), triangles, 4-cycles ].  Five graphs span the
# five invariants; the fitted formula is verified against held-out graphs in tests.
_H4_FIT = [
    ([(1, 2)],                          [1, 1, 0, 0, 0]),
    ([(1, 2), (2, 3)],                  [2, 4, 1, 0, 0]),
    ([(1, 2), (3, 4)],                  [2, 4, 0, 0, 0]),
    ([(1, 2), (2, 3), (3, 1)],          [3, 9, 3, 1, 0]),
    ([(1, 2), (2, 3), (3, 4), (4, 1)],  [4, 16, 4, 0, 1]),
]


def _rational_inverse(mat):
    """Exact inverse of a small integer/Fraction matrix (Gauss–Jordan)."""
    n = len(mat)
    a = [[Fraction(mat[i][j]) for j in range(n)] +
         [Fraction(1 if i == j else 0) for j in range(n)] for i in range(n)]
    for col in range(n):
        piv = next(r for r in range(col, n) if a[r][col] != 0)
        a[col], a[piv] = a[piv], a[col]
        inv = a[col][col]
        a[col] = [x / inv for x in a[col]]
        for r in range(n):
            if r != col and a[r][col] != 0:
                f = a[r][col]
                a[r] = [x - f * y for x, y in zip(a[r], a[col])]
    return [row[n:] for row in a]


def energy_moment4_coeffs(kind: str):
    """Fit ⟨H⁴⟩ = a|E| + b|E|² + c·wedges + d·N△ + e·N□ from the representative
    clusters.  Returns (a, b, c, d, e) as Coeff (exact)."""
    rhs = [energy_moment(kind, edges, 4) for edges, _ in _H4_FIT]
    minv = _rational_inverse([row for _, row in _H4_FIT])
    out = []
    for i in range(5):
        c = Coeff()
        for j in range(5):
            c = c + rhs[j] * Coeff({(): (minv[i][j], 0)})
        out.append(c)
    return out


def energy_moment_general_latex(kind: str, upto: int = 3) -> str:
    """⟨H^n⟩ as general-lattice formulas in the graph invariants (the same loops
    3c-graph counts): ⟨H²⟩∝|E|=½tr A², ⟨H³⟩ carries N△=⅙tr A³, ⟨H⁴⟩ carries the
    4-cycle count N□."""
    c2, c3, d3 = energy_moment_invariants(kind)
    e, ntri = Coeff.param("|E|"), Coeff.param(r"N_\triangle")
    parts = [rf"\langle H^2 \rangle = {(c2 * e).to_latex()}",
             rf"\langle H^3 \rangle = {((c3 * e) + (d3 * ntri)).to_latex()}"]
    if upto >= 4:
        a, b, c, d, f = energy_moment4_coeffs(kind)
        e2, wdg, sq = e * e, Coeff.param(r"N_\wedge"), Coeff.param(r"N_\square")
        h4 = (a * e + b * e2 + c * wdg + d * ntri + f * sq).to_latex()
        parts.append(rf"\langle H^4 \rangle = {h4}")
    return r",\ ".join(parts)


def energy_cumulants_general(kind: str):
    """The connected energy cumulants κ₂,κ₃,κ₄ as general-lattice formulas in the
    graph invariants (⟨H⟩=0 → κ₂=⟨H²⟩, κ₃=⟨H³⟩, κ₄=⟨H⁴⟩−3⟨H²⟩²; the |E|²
    disconnected piece cancels in κ₄, leaving only the connected loops)."""
    c2, c3, d3 = energy_moment_invariants(kind)
    a, b, c, d, e = energy_moment4_coeffs(kind)
    big_e, ntri, nsq = Coeff.param("|E|"), Coeff.param(r"N_\triangle"), \
        Coeff.param(r"N_\square")
    wdg = Coeff.param(r"N_\wedge")
    e2 = big_e * big_e
    k2 = c2 * big_e
    k3 = (c3 * big_e) + (d3 * ntri)
    k4 = ((a * big_e) + ((b + Coeff.int(-3) * c2 * c2) * e2)
          + (c * wdg) + (d * ntri) + (e * nsq))
    return k2, k3, k4


def _half():
    return Coeff({(): (Fraction(1, 2), 0)})


def specific_heat_general_latex(kind: str) -> str:
    """High-temperature series of the specific heat C(β)=κ₂β²−κ₃β³+½κ₄β⁴+O(β⁵),
    with the cumulants in the graph invariants (a finite-temperature observable
    built from the infinite-temperature energy moments)."""
    k2, k3, k4 = energy_cumulants_general(kind)
    beta = Coeff.param(r"\beta")
    b2 = beta * beta
    c = (k2 * b2) + (Coeff.int(-1) * k3 * b2 * beta) + (k4 * _half() * b2 * b2)
    return rf"C(\beta) = {c.to_latex()} + O(\beta^5)"


def specific_heat_concrete_latex(kind: str, edges) -> str:
    """High-temperature C(β) series with the cumulants evaluated on a concrete graph."""
    m2 = energy_moment(kind, edges, 2)
    m3 = energy_moment(kind, edges, 3)
    m4 = energy_moment(kind, edges, 4)
    k2, k3, k4 = m2, m3, m4 + Coeff.int(-3) * m2 * m2
    beta = Coeff.param(r"\beta")
    b2 = beta * beta
    c = (k2 * b2) + (Coeff.int(-1) * k3 * b2 * beta) + (k4 * _half() * b2 * b2)
    return rf"C(\beta) = {c.to_latex()} + O(\beta^5)"


def greens_function(b2_list):
    """The (terminating) recursion-method Green's function G(z)=1/(z−b₁²/(z−b₂²/…))
    as a rational function, returned as (numerator, denominator) Coeff in z (and the
    couplings).  For a finite cluster the chain terminates, so G is exactly rational."""
    z = Coeff.param("z")
    num, den = Coeff.int(1), z
    for b2 in reversed(b2_list):
        num, den = den, (z * den + Coeff.int(-1) * (b2 * num))
    return num, den


def _cf_latex(b2_list) -> str:
    inner = "z"
    for b2 in reversed(b2_list):
        inner = rf"z - \cfrac{{{b2.to_latex()}}}{{{inner}}}"
    return rf"\cfrac{{1}}{{{inner}}}"


def _coeff_sqrt(c):
    """Exact square root of a single positive square-rational monomial, else None."""
    from math import isqrt
    if len(c.d) != 1:
        return None
    (mono, (re, im)), = c.d.items()
    if im != 0 or re <= 0 or any(e % 2 for _, e in mono):
        return None
    rn, rd = isqrt(re.numerator), isqrt(re.denominator)
    if rn * rn != re.numerator or rd * rd != re.denominator:
        return None
    return Coeff({tuple((n, e // 2) for n, e in mono): (Fraction(rn, rd), 0)})


def spectral_peaks(b2_list):
    """Exact poles+weights of the spectral function S(ω)=Σ w_p δ(ω−ω_p) for a
    terminating chain, in closed form.  Returns [(weight, ω) …] for the cases whose
    pole polynomial factors with a clean (monomial) √ — one coefficient (±√b₁²,
    weight ½ each) or two (a central peak at 0 plus ±√(b₁²+b₂²)) — else None."""
    half = Coeff({(): (Fraction(1, 2), 0)})
    neg = Coeff.int(-1)
    if len(b2_list) == 1:
        w0 = _coeff_sqrt(b2_list[0])
        return None if w0 is None else [(half, w0), (half, neg * w0)]
    if len(b2_list) == 2:
        s = b2_list[0] + b2_list[1]
        w1 = _coeff_sqrt(s)
        if w1 is None:
            return None
        w_center = b2_list[1] / s            # b₂²/(b₁²+b₂²) at ω=0
        w_side = (b2_list[0] / s) * half     # b₁²/(2(b₁²+b₂²)) at ±√(b₁²+b₂²)
        return [(w_center, Coeff()), (w_side, w1), (w_side, neg * w1)]
    return None


def _delta(omega) -> str:
    if omega.is_zero():
        return r"\delta(\omega)"
    ol = omega.to_latex()
    return (rf"\delta(\omega + {ol[1:].strip()})" if ol.startswith("-")
            else rf"\delta(\omega - {ol})")


def spectral_latex(b2_list) -> str:
    """Display the recursion-method Green's function (continued fraction) and, when
    the poles are obtainable in closed form, the spectral function S(ω) as exact
    delta peaks with weights; otherwise the rational form G(z)=P(z)/Q(z) (the poles
    are the roots of the denominator Q)."""
    if not b2_list:
        return r"G(z) = \cfrac{1}{z},\quad S(\omega) = \delta(\omega)"
    cf = rf"G(z) = {_cf_latex(b2_list)}"
    peaks = spectral_peaks(b2_list)
    if peaks:
        s_omega = " + ".join(rf"{w.to_latex()}\,{_delta(om)}" for w, om in peaks)
        return rf"{cf},\quad S(\omega) = {s_omega}"
    num, den = greens_function(b2_list)
    return rf"{cf},\quad G(z) = \frac{{{num.to_latex()}}}{{{den.to_latex()}}}"


def terminator_spectral_latex(binf: str = r"b_\infty") -> str:
    """The infinite-system continuous-band result (analytic, no numerics): when the
    recursion coefficients approach a constant b_∞, the tail continued fraction sums
    to the square-root terminator t(z)=(z−√(z²−4b_∞²))/2, giving a continuous band.
    For a uniform chain (all b_n²=b_∞²) the spectral density is the Wigner semicircle
    S(ω)=√(4b_∞²−ω²)/(2πb_∞²) on the band |ω|≤2b_∞."""
    t = rf"t(z) = \frac{{z - \sqrt{{z^2 - 4 {binf}^2}}}}{{2}}"
    s = (rf"S(\omega) = \frac{{\sqrt{{4 {binf}^2 - \omega^2}}}}"
         rf"{{2\pi {binf}^2}}\quad (|\omega| \le 2 {binf})")
    return rf"{t},\quad {s}"


def _scale(c, expr):
    return [(c * cc, ops) for cc, ops in expr]


def continued_fraction(h, o, depth: int):
    """Recursion-method continued-fraction coefficients b_n² via operator Lanczos:
    f_0=O, f_{n+1}=[H,f_n] − b_n² f_{n-1}, b_n²=(f_n,f_n)/(f_{n-1},f_{n-1}).  Returns
    [b_1², b_2², …] (stops when an f_n is null under the ∞-T inner product).

    `moments` と同じく、同一サイト積は **宣言によらず畳む**（`_all_sites` 参照）。
    4サイト鎖のスペクトル関数が 185.9 秒で打ち切り → 0.1 秒で完了に変わるのは
    ここである。"""
    half = _all_sites(h, o)
    fs = [o]
    norms = [inner(o, o)]
    if norms[0].is_zero():
        return []
    fs.append(_collect(commutator(h, o), half))
    norms.append(inner(fs[1], fs[1]))
    b2s = []
    neg = Coeff.int(-1)
    for n in range(1, depth + 1):
        if norms[n].is_zero():
            break
        b2 = norms[n] / norms[n - 1]
        b2s.append(b2)
        f_next = _collect(commutator(h, fs[n]) + _scale(neg * b2, fs[n - 1]),
                          half)
        fs.append(f_next)
        norms.append(inner(f_next, f_next))
    return b2s
