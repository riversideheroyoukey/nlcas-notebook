"""Lattice / graph pure functions (F-1, B): bond generators, dispersion
relations, Bloch matrices, adjacency-matrix identities, neighbor-sum expansion.

Extracted from backend_handler.py (R-10a, 2026-07-04) — these are pure,
backend-independent string builders (CAS-bypassing closed forms), not part of
the BackendHandler abstraction.  backend_handler re-exports every name for
backward compatibility, so both import paths work.
"""
from __future__ import annotations

import re


# ---------------------------------------------------------------------------
# Pair-sum bond-list generators (F-1 / B).
#
# A pairwise lattice/molecular sum Σ_{(i,j)} body(i,j) differs by domain only
# in WHICH index pairs (i,j) it runs over.  That pair structure is captured by
# a named generator: (n_sites, boundary) → [(a, b), …].  Keeping these in a
# registry means new domains (2D / triangular lattices, molecular geometries)
# are added by registering another generator, without touching the expansion,
# the backends, or the schema.
#
#   nearest_neighbor — 1D chain bonds ⟨ij⟩ (condensed matter, classical Ising)
#   all_pairs        — every unordered pair i<j (molecular / quantum-chemistry
#                      pairwise sums Σ_{i<j} 1/r_ij, Madelung-type lattice sums)
#   ordered_pairs    — every ORDERED pair i≠j.  Σ_{i≠j} is NOT Σ_{i<j}: it has
#                      N(N-1) terms, twice as many, because (2,1) is written as
#                      well as (1,2).  Both spellings used to map to all_pairs,
#                      so `\sum_{i \ne j} V_{i,j}` over 3 sites expanded to
#                      THREE terms and the display was rewritten to `\sum_{i<j}`
#                      — half the sum, silently, with the researcher's own
#                      notation replaced (measured 2026-08-10).  Whether the
#                      double counting matters is physics: Σ_{i≠j} is the
#                      convention for a Coulomb energy written with an explicit
#                      1/2, and for a hopping term where (i,j) and (j,i) carry
#                      different amplitudes.
# ---------------------------------------------------------------------------

_BOND_GENERATORS: "dict[str, object]" = {}


def register_bond_generator(name: str):
    """Register a bond-list generator ``(n_sites, boundary, shape) -> list[(a, b)]``.

    ``shape`` is an optional ``(Lx, Ly)`` tuple for 2D lattices (None for 1D /
    geometry-free generators).  A generator may raise ``ValueError`` to report
    an invalid geometry (e.g. a non-square site count without an explicit shape).
    """
    def _deco(fn):
        _BOND_GENERATORS[name] = fn
        return fn
    return _deco


def _shape_or_square(shape, n_sites: int) -> "tuple[int, int]":
    """Resolve an (Lx, Ly) grid: use *shape* if given, else assume an L×L
    square with L = √n_sites (raising if n_sites is not a perfect square)."""
    if shape is not None:
        Lx, Ly = shape
        if Lx < 1 or Ly < 1:
            raise ValueError("lattice_shape dimensions must be ≥ 1")
        return Lx, Ly
    L = int(round(n_sites ** 0.5))
    if L * L != n_sites:
        raise ValueError(
            f"n_sites={n_sites} is not a perfect square; give "
            f"lattice_shape='LxxLy' for a non-square 2D lattice")
    return L, L


@register_bond_generator("nearest_neighbor")
def _bonds_nearest_neighbor(n, boundary, shape=None):
    bonds = [(a, a + 1) for a in range(1, n)]
    if boundary == "periodic" and n >= 3:
        bonds.append((n, 1))
    return bonds


@register_bond_generator("all_pairs")
def _bonds_all_pairs(n, boundary, shape=None):
    # Every unordered pair i<j; boundary is irrelevant for a complete pair set.
    return [(a, b) for a in range(1, n + 1) for b in range(a + 1, n + 1)]


@register_bond_generator("ordered_pairs")
def _bonds_ordered_pairs(n, boundary, shape=None):
    """Every ordered pair i≠j — N(N-1) of them, both (a,b) and (b,a)."""
    return [(a, b) for a in range(1, n + 1) for b in range(1, n + 1) if a != b]


def _periodic_xy(boundary) -> "tuple[bool, bool]":
    """Per-direction periodicity: open=neither, periodic=both (torus),
    cylinder=x only (a tube wrapped along x)."""
    return (boundary in ("periodic", "cylinder"), boundary == "periodic")


@register_bond_generator("square_2d")
def _bonds_square_2d(n, boundary, shape=None):
    # 2D square lattice on an Lx×Ly grid, row-major linear index
    # idx(x,y) = (y-1)*Lx + x.  Bonds: right (+x) and up (+y) neighbors.
    # boundary: open (no wrap), periodic (torus), cylinder (x wrap only).
    Lx, Ly = _shape_or_square(shape, n)
    px, py = _periodic_xy(boundary)
    def idx(x, y):
        return (y - 1) * Lx + x
    bonds = []
    for y in range(1, Ly + 1):
        for x in range(1, Lx):
            bonds.append((idx(x, y), idx(x + 1, y)))
        if px and Lx >= 3:
            bonds.append((idx(Lx, y), idx(1, y)))
    for x in range(1, Lx + 1):
        for y in range(1, Ly):
            bonds.append((idx(x, y), idx(x, y + 1)))
        if py and Ly >= 3:
            bonds.append((idx(x, Ly), idx(x, 1)))
    return bonds


@register_bond_generator("triangular_2d")
def _bonds_triangular_2d(n, boundary, shape=None):
    # Triangular lattice = square lattice + one diagonal per cell
    # (x,y)–(x+1,y+1), giving 6-fold coordination in the bulk.
    Lx, Ly = _shape_or_square(shape, n)
    px, py = _periodic_xy(boundary)
    def idx(x, y):
        return (y - 1) * Lx + x
    bonds = list(_bonds_square_2d(n, boundary, (Lx, Ly)))
    for y in range(1, Ly):
        for x in range(1, Lx):
            bonds.append((idx(x, y), idx(x + 1, y + 1)))
    if px:
        for y in range(1, Ly):                      # +x edge diagonal wrap
            bonds.append((idx(Lx, y), idx(1, y + 1)))
    if py:
        for x in range(1, Lx):                      # +y edge diagonal wrap
            bonds.append((idx(x, Ly), idx(x + 1, 1)))
    if px and py and Lx >= 2 and Ly >= 2:           # corner diagonal wrap
        bonds.append((idx(Lx, Ly), idx(1, 1)))
    return bonds


@register_bond_generator("honeycomb_2d")
def _bonds_honeycomb_2d(n, boundary, shape=None):
    # Honeycomb (brick-wall representation): horizontal bonds everywhere, but
    # vertical bonds only on alternating cells (x+y even, 0-indexed), giving
    # 3-fold coordination in the bulk.
    Lx, Ly = _shape_or_square(shape, n)
    px, py = _periodic_xy(boundary)
    def idx(x, y):
        return (y - 1) * Lx + x
    bonds = []
    for y in range(1, Ly + 1):
        for x in range(1, Lx):
            bonds.append((idx(x, y), idx(x + 1, y)))
        if px and Lx >= 3:
            bonds.append((idx(Lx, y), idx(1, y)))
    for x in range(1, Lx + 1):
        for y in range(1, Ly):
            if ((x - 1) + (y - 1)) % 2 == 0:        # brick-wall vertical rungs
                bonds.append((idx(x, y), idx(x, y + 1)))
        if py and Ly >= 3 and ((x - 1) + (Ly - 1)) % 2 == 0:
            bonds.append((idx(x, Ly), idx(x, 1)))
    return bonds


def _parse_edges(edges) -> "list[tuple[int, int]]":
    """Parse a custom (arbitrary-graph) edge list into [(a, b), …].

    Accepts a list/tuple of integer pairs, or a string of pairs such as
    ``"(1,2),(2,3),(1,3)"`` or ``"1-2, 2-3, 3-1"``.
    """
    pairs: "list[tuple[int, int]]" = []
    if isinstance(edges, (list, tuple)):
        for e in edges:
            if isinstance(e, (list, tuple)) and len(e) == 2:
                try:
                    pairs.append((int(e[0]), int(e[1])))
                except (TypeError, ValueError):
                    pass
    elif isinstance(edges, str):
        found = re.findall(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)", edges)
        if not found:
            found = re.findall(r"(\d+)\s*-\s*(\d+)", edges)
        pairs = [(int(a), int(b)) for a, b in found]
    return [(a, b) for a, b in pairs if a != b]


# ---------------------------------------------------------------------------
# F-1 stage 3c-fourier: single-band tight-binding dispersion.
#
# A translation-invariant lattice with ONE site per unit cell is described not
# by a finite bond list but by its unit-cell hopping vectors ΔR (in primitive-
# vector units).  The Bloch sum of a bilinear H = Σ_{ij} A_{ij} O_i O_j is
#   ε(k) = Σ_{ΔR} t · e^{i k·ΔR} = Σ_{ΔR>0} 2t cos(k·ΔR)
# (the +ΔR and −ΔR partners combine into a cosine).  Each generator name maps
# to its set of positive nearest-neighbour ΔR vectors; multi-sublattice lattices
# (honeycomb) are a separate multi-band increment and are rejected here.
# ---------------------------------------------------------------------------

_CRYSTAL_1BAND: "dict[str, dict]" = {
    # name: (positive nearest-neighbour ΔR vectors, k-symbols, optional `shell2`
    # next-nearest ΔR for a t–t′ model)
    "nearest_neighbor": {"deltas": [(1,)], "ksyms": ["k"], "shell2": [(2,)]},
    "square_2d":        {"deltas": [(1, 0), (0, 1)], "ksyms": ["k_x", "k_y"],
                         "shell2": [(1, 1), (1, -1)]},
    # 120° convention (a1+a2 is also a nearest neighbour); the 6 second neighbours
    # at distance √3 are ±(a1−a2), ±(2a1+a2), ±(a1+2a2).
    "triangular_2d":    {"deltas": [(1, 0), (0, 1), (1, 1)],
                         "ksyms": ["k_x", "k_y"],
                         "shell2": [(1, -1), (2, 1), (1, 2)]},
}

# Multi-band (multi-sublattice) crystals: a bipartite 2-sublattice lattice has a
# 2×2 Bloch matrix Ã(k) = [[0, f(k)], [f*(k), 0]] with f(k) = t Σ_n e^{i k·ΔR_n}
# over the inter-sublattice (A→B) bonds.  With zero diagonal (nearest-neighbour
# only) the bands are ε_±(k) = ±|f(k)|, and
#   |f|²/|t|² = N + 2 Σ_{m<n} cos(k·(ΔR_m − ΔR_n))
# is a clean combinatorial form (no CAS needed).  ΔR are in reduced primitive-
# vector coordinates; ksyms are the reduced momenta k_1=k·a_1, k_2=k·a_2.
_CRYSTAL_MB: "dict[str, dict]" = {
    # honeycomb / graphene: 3 A→B nearest-neighbour bonds (intra-cell + two
    # neighbouring cells) → ε_±(k) = ±t√(3 + 2cos k_1 + 2cos k_2 + 2cos(k_1−k_2)).
    # `intra_shell` is the same-sublattice (next-nearest) triangular shell shared
    # by both sublattices (A/B symmetric, so d_A=d_B=d); supplying a t' amplitude
    # adds the diagonal term d(k) = 2t'(cos k_1 + cos k_2 + cos(k_1−k_2)) and the
    # bands become ε_±(k) = d(k) ± |t|√(…).
    "honeycomb_2d": {"n_basis": 2, "ksyms": ["k_1", "k_2"],
                     "inter_bonds": [(0, 0), (-1, 0), (0, -1)],
                     "intra_shell": [(1, 0), (0, 1), (1, -1)]},
}

# ≥3-sublattice STAR-structured crystals: one central sublattice couples to m
# others that do not interconnect (a bordered / arrowhead Bloch matrix).  The
# spectrum is exactly a flat band ε₀=0 (multiplicity m−1) plus ε_±=±|t|√(Σ_j|h_j|²),
# where each `star_links[j]` is the bond list (ΔR in reduced coords) of link j.
# (Non-bipartite ≥3-sublattice lattices — kagome with B–C coupling — need a cubic
# and are deferred.)
# Two solvable ≥3-sublattice templates:
#   "star" — bordered/arrowhead matrix (one central sublattice) → flat band 0 +
#            ε_±=±|t|√(Σ_j|h_j|²)  (Lieb).
#   "ring" — fully-coupled triangle whose three bonds close (δ_AB+δ_BC+δ_CA=0);
#            the cubic factors into a constant flat band + ε_±=t(1±√|f|²)  (kagome).
_CRYSTAL_NBAND: "dict[str, dict]" = {
    # Lieb lattice: corner A coupled to edge-centres B (x) and C (y); B,C uncoupled.
    # |h_AB|²=2+2cos k_x, |h_AC|²=2+2cos k_y → ε=0, ±t√(4+2cos k_x+2cos k_y).
    "lieb_2d": {"type": "star", "n_basis": 3, "ksyms": ["k_x", "k_y"],
                "star_links": [[(0, 0), (-1, 0)], [(0, 0), (0, -1)]]},
    # Kagome: A,B,C all mutually coupled (non-bipartite). The closed-triangle
    # identity factors the cubic → flat band 2t and ε_±=−t(1±√|f|²) with
    # |f|²=3+2(cos k_x+cos k_y+cos(k_x+k_y)) (the triangular structure factor).
    "kagome_2d": {"type": "ring", "n_basis": 3, "ksyms": ["k_x", "k_y"],
                  "ring_bonds": [(0, 0), (-1, 0), (0, 1)]},
}

_SINGLE_TOKEN_COEFF = re.compile(r"^[A-Za-z\\][A-Za-z0-9_^{}\\']*$")


def _k_dot_delta(delta, ksyms) -> str:
    """LaTeX for k·ΔR given integer ΔR components and k-symbol names, with proper
    signs: ``(1,-1)`` → ``k_1 - k_2``, ``(2,0)`` → ``2 k_x``."""
    out = ""
    for d, ks in zip(delta, ksyms):
        if d == 0:
            continue
        token = ks if abs(d) == 1 else f"{abs(d)} {ks}"
        if not out:
            out = ("-" + token) if d < 0 else token
        else:
            out += (f" - {token}") if d < 0 else (f" + {token}")
    return out or "0"


def _fold_two_coeff(coeff: str) -> str:
    """Fold the cosine-pair factor 2 into a hopping coefficient for display.

    ``-t`` → ``-2t``, ``t`` → ``2t``, ``J`` → ``2J``, ``1``/`""` → ``2``.
    A non-atomic coefficient (spaces, fractions, sums) falls back to ``2 (coeff)``.
    """
    c = (coeff or "").strip()
    if c in ("", "1"):
        return "2"
    if c == "-1":
        return "-2"
    neg = c.startswith("-")
    core = c[1:].strip() if neg else c
    if _SINGLE_TOKEN_COEFF.match(core):
        return ("-2" if neg else "2") + core
    return rf"2 \left( {c} \right)"


def build_dispersion_single_band(coeff: str, pair_constraint: str, nnn_coeff=None):
    """Build the single-band tight-binding dispersion ε(k) for a lattice (3c).

    *coeff* is the hopping amplitude (the bond body's scalar prefactor, e.g.
    ``-t``); *pair_constraint* is a registered single-band crystal name.
    *nnn_coeff* (optional) adds a second (next-nearest) hopping shell t', e.g. the
    square-lattice t–t′ model ε(k) = −2t(cos k_x+cos k_y) − 2t′(cos(k_x+k_y)+cos(k_x−k_y)).
    Returns ``(display_latex, None)`` or ``(None, error_message)``.
    """
    pc = (pair_constraint or "nearest_neighbor").strip() or "nearest_neighbor"
    if pc in _CRYSTAL_MB:
        return None, (f"dispersion_relation: lattice {pc!r} is multi-band; "
                      f"use build_dispersion_multiband")
    crystal = _CRYSTAL_1BAND.get(pc)
    if crystal is None:
        return None, (f"dispersion_relation: {pc!r} is not a translation-invariant "
                      f"single-band lattice; known: {sorted(_CRYSTAL_1BAND)}")
    ksyms = crystal["ksyms"]
    rhs = _shell_term(coeff, crystal["deltas"], ksyms)
    if nnn_coeff:
        shell2 = crystal.get("shell2")
        if not shell2:
            return None, (f"dispersion_relation: lattice {pc!r} has no defined "
                          f"next-nearest shell")
        nnn_term = _shell_term(nnn_coeff, shell2, ksyms)
        rhs += (f" {nnn_term}" if nnn_term.startswith("-") else f" + {nnn_term}")
    kvar = "k" if len(ksyms) == 1 else r"\mathbf{k}"
    return rf"\varepsilon({kvar}) = {rhs}", None


def _abs_coeff(coeff: str) -> str:
    """Magnitude of a hopping amplitude for a ±√ band: ``-t``→``t``, ``""``→``""``."""
    c = (coeff or "").strip()
    if c.startswith("-"):
        c = c[1:].strip()
    return "" if c in ("", "1") else c


def _negate_disp(s: str) -> str:
    """Flip the leading sign of a display token: ``-2t``→``2t``, ``2t``→``-2t``."""
    s = s.strip()
    return s[1:].strip() if s.startswith("-") else f"-{s}"


def _sq(amp: str) -> str:
    """Square of an amplitude token for a |·|² band: ``v``→``v^2``, ``-t``→``t^2``."""
    c = (amp or "").strip()
    if c.startswith("-"):
        c = c[1:].strip()
    return f"{c}^2" if _SINGLE_TOKEN_COEFF.match(c) else rf"\left( {c} \right)^2"


def _cos_of_diff(vec, ksyms) -> str:
    """``\\cos`` of k·vec, with the sign normalised (cos is even) so the leading
    nonzero component is positive.  A bare single momentum symbol is unparenthesised
    (``\\cos k_x``); anything else is parenthesised (``\\cos(k_x + k_y)``)."""
    v = list(vec)
    lead = next((c for c in v if c != 0), 0)
    if lead < 0:
        v = [-c for c in v]
    arg = _k_dot_delta(tuple(v), ksyms)
    if arg == "0":
        return "1"
    return rf"\cos {arg}" if arg in ksyms else rf"\cos({arg})"


def _shell_term(coeff: str, deltas, ksyms) -> str:
    """A hopping-shell contribution ``2t Σ_ΔR cos(k·ΔR)`` as display LaTeX:
    ``-2t \\cos k`` for a single ΔR, ``-2t \\left( … \\right)`` for several."""
    cos_terms = [_cos_of_diff(d, ksyms) for d in deltas]
    inner = " + ".join(cos_terms)
    pref = _fold_two_coeff(coeff)
    return f"{pref} {inner}" if len(cos_terms) == 1 else rf"{pref} \left( {inner} \right)"


def _link_inner(bonds, ksyms) -> "tuple[int, list[str]]":
    """For a set of equal-amplitude bonds, the real part of |Σ_n e^{i k·ΔR_n}|² is
    ``N + 2 Σ_{m<n} cos(k·(ΔR_m − ΔR_n))``.  Returns ``(N, [cos-term, …])``."""
    n = len(bonds)
    cos_terms = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = tuple(a - b for a, b in zip(bonds[i], bonds[j]))
            cos_terms.append(_cos_of_diff(diff, ksyms))
    return n, cos_terms


def _format_abs_sq(const: int, cos_terms) -> str:
    """``N + 2cos… + 2cos…`` display for an accumulated |Σ e|² inner."""
    out = str(const)
    for ct in cos_terms:
        out += f" + 2{'' if ct.startswith(chr(92)) else ' '}{ct}"
    return out


def _combine_coeffs(a: str, b: str, op: str) -> str:
    """Display ``a ± b`` for two signed amplitude tokens, normalising double signs:
    ``("-t_A'", "-t_B'", "+")`` → ``-t_A' - t_B'``; with ``"-"`` → ``-t_A' + t_B'``."""
    a, b = a.strip(), b.strip()
    bneg = b.startswith("-")
    core = b[1:].strip() if bneg else b
    plus = (op == "+") != bneg          # XOR: a '-' op cancels b's leading '-'
    return f"{a} + {core}" if plus else f"{a} - {core}"


def build_dispersion_multiband(coeff: str, pair_constraint: str, nnn_coeff=None,
                               onsite=None, nnn_b=None):
    """Build the multi-band (bipartite 2-sublattice) dispersion ε_±(k) (3c).

    The general 2×2 Hermitian band structure is
        ε_±(k) = (d_A+d_B)/2 ± √( ((d_A−d_B)/2)² + |f(k)|² ),
    with f(k) the inter-sublattice Bloch sum (|f|²/|t|² = N + 2 Σ_{m<n} cos(k·ΔR)).

    - nearest-neighbour only: d_A=d_B=0 → ε_±=±|t|√(…) (graphene).
    - *nnn_coeff* t' (A/B symmetric intra-shell): a uniform diagonal d(k) into the
      average (d_A+d_B)/2 = 2t'Σ cos(k·ΔR_shell).
    - *onsite* Δ (staggered sublattice potential, +Δ on A / −Δ on B): the
      antisymmetric (d_A−d_B)/2 = Δ → gapped Dirac ε_±=±√(Δ² + |f|²).
    - *nnn_b* t_B′ (with nnn_coeff = t_A′): per-sublattice next-nearest amplitudes,
      d_α = 2t_α′ S(k).  The symmetric part (t_A′+t_B′)S(k) is the band average; the
      antisymmetric (t_A′−t_B′)S(k) (+Δ) goes under the root.
    Returns ``(display_latex, None)`` or ``(None, error_message)``.
    """
    pc = (pair_constraint or "").strip()
    crystal = _CRYSTAL_MB.get(pc)
    if crystal is None:
        return None, (f"dispersion_relation: {pc!r} is not a known multi-band "
                      f"lattice; known: {sorted(_CRYSTAL_MB)}")
    bonds, ksyms = crystal["inter_bonds"], crystal["ksyms"]
    inner = _format_abs_sq(*_link_inner(bonds, ksyms))
    amp = _abs_coeff(coeff)
    amp_sq = f"{amp}^2 " if amp else ""

    def _shell_sum_inner():
        shell = crystal.get("intra_shell")
        if not shell:
            return None
        return " + ".join(_cos_of_diff(dr, ksyms) for dr in shell)

    # Per-sublattice next-nearest amplitudes (d_A ≠ d_B): the general 2×2 form.
    if nnn_b:
        s_inner = _shell_sum_inner()
        if s_inner is None:
            return None, (f"dispersion_relation: lattice {pc!r} has no defined "
                          f"next-nearest shell")
        shell_grp = rf"\left( {s_inner} \right)"
        avg = rf"\left( {_combine_coeffs(nnn_coeff, nnn_b, '+')} \right) {shell_grp}"
        half = rf"\left( {_combine_coeffs(nnn_coeff, nnn_b, '-')} \right) {shell_grp}"
        if onsite:
            half += f" + {onsite}"
        root = rf"\left( {half} \right)^2 + {amp_sq}\left( {inner} \right)"
        return rf"\varepsilon_\pm(\mathbf{{k}}) = {avg} \pm \sqrt{{ {root} }}", None

    # antisymmetric diagonal (d_A−d_B)/2 = Δ goes under the root with |f|²;
    # symmetric diagonal (d_A+d_B)/2 = d(k) sits outside as the band average.
    if onsite:
        root = rf"{onsite}^2 + {amp_sq}\left( {inner} \right)"
        band = rf"\pm \sqrt{{ {root} }}"
    else:
        band = rf"\pm {amp} \sqrt{{ {inner} }}" if amp else rf"\pm \sqrt{{ {inner} }}"

    diag = ""
    if nnn_coeff:
        s_inner = _shell_sum_inner()
        if s_inner is None:
            return None, (f"dispersion_relation: lattice {pc!r} has no defined "
                          f"next-nearest shell")
        diag = rf"{_fold_two_coeff(nnn_coeff)} \left( {s_inner} \right) "

    return rf"\varepsilon_\pm(\mathbf{{k}}) = {diag}{band}", None


def build_dispersion_nband(coeff: str, pair_constraint: str):
    """Build the ≥3-sublattice dispersion for a solvable lattice template (3c).

    "star" (Lieb): a central sublattice coupled to m others that do not
    interconnect → bordered/arrowhead matrix, flat band ε₀=0 (multiplicity m−1)
    and ε_±(k)=±|t|√(Σ_j |h_j|²).  Lieb → ε=0, ±t√(4+2cos k_x+2cos k_y).

    "ring" (kagome): a fully-coupled triangle whose three bonds close, so the cubic
    factors into a constant flat band and ε_±(k)=coeff·(1±√|f|²).  Kagome →
    ε=2t, −t(1±√(3+2cos k_x+2cos k_y+2cos(k_x+k_y))).
    Returns ``(display_latex, None)`` or ``(None, error_message)``.
    """
    pc = (pair_constraint or "").strip()
    crystal = _CRYSTAL_NBAND.get(pc)
    if crystal is None:
        return None, (f"dispersion_relation: {pc!r} is not a known solvable "
                      f"≥3-sublattice lattice; known: {sorted(_CRYSTAL_NBAND)}")
    ksyms = crystal["ksyms"]

    if crystal["type"] == "ring":
        inner = _format_abs_sq(*_link_inner(crystal["ring_bonds"], ksyms))
        flat = _negate_disp(_fold_two_coeff(coeff))         # ε_flat = −2·coeff
        pref = (coeff or "").strip()
        body = rf"{pref} " if pref else ""
        band = rf"{body}\left( 1 \pm \sqrt{{ {inner} }} \right)"
        return (rf"\varepsilon_{{\mathrm{{flat}}}} = {flat}, \quad "
                rf"\varepsilon_\pm(\mathbf{{k}}) = {band}", None)

    const = 0
    cos_terms: "list[str]" = []
    for link in crystal["star_links"]:
        c, terms = _link_inner(link, ksyms)
        const += c
        cos_terms += terms
    inner = _format_abs_sq(const, cos_terms)
    amp = _abs_coeff(coeff)
    band = rf"\pm {amp} \sqrt{{ {inner} }}" if amp else rf"\pm \sqrt{{ {inner} }}"
    return (rf"\varepsilon_0 = 0, \quad \varepsilon_\pm(\mathbf{{k}}) = {band}", None)


def build_dispersion(coeff: str, pair_constraint: str, nnn_coeff=None, onsite=None,
                     nnn_b=None):
    """Dispatch to the single-, multi-, or N-band dispersion builder (F-1 3c)."""
    pc = (pair_constraint or "nearest_neighbor").strip() or "nearest_neighbor"
    if pc in _CRYSTAL_NBAND:
        return build_dispersion_nband(coeff, pc)
    if pc in _CRYSTAL_MB:
        return build_dispersion_multiband(coeff, pc, nnn_coeff, onsite, nnn_b)
    return build_dispersion_single_band(coeff, pc, nnn_coeff)


# ---------------------------------------------------------------------------
# F-1 stage 3c-fourier: the abstract crystal descriptor.
#
# A crystal is given size-independently by its bonds (α, β, ΔR, amplitude) in
# reduced primitive-vector coordinates.  The Bloch Hamiltonian is
#   H̃_{αβ}(k) = Σ_bonds t · e^{i k·ΔR}   (with the Hermitian conjugate added),
# of which every named lattice above is a special case.  Closed-form bands follow
# for N≤2 (scalar, or the general 2×2 Hermitian formula); for N≥3 only the matrix
# is built (generic eigenvalues have no clean closed form — see kagome).
# ---------------------------------------------------------------------------

def _exp_term(amp: str, dr, ksyms) -> str:
    """A Bloch term ``amp·e^{i k·ΔR}`` for display; ΔR=0 → just ``amp``."""
    phase = _k_dot_delta(dr, ksyms)
    if phase == "0":
        return amp
    sign, body = ("-", phase[1:].strip()) if phase.startswith("-") else ("", phase)
    expo = f"{sign}i ({body})" if (" + " in body or " - " in body) else f"{sign}i {body}"
    return rf"{amp} e^{{{expo}}}"


def _join_signed(terms) -> str:
    """Join display terms, using ``-`` for a term that already carries a sign."""
    if not terms:
        return "0"
    out = terms[0]
    for tm in terms[1:]:
        out += f" {tm}" if tm.lstrip().startswith("-") else f" + {tm}"
    return out


def _diag_entry(bonds, ksyms) -> str:
    """Diagonal (same-sublattice) Bloch entry: onsite amplitudes plus 2t cos(k·ΔR)
    for each ΔR≠0 (its Hermitian partner makes the cosine)."""
    terms = []
    for dr, amp in bonds:
        if all(x == 0 for x in dr):
            terms.append(amp)
        else:
            terms.append(rf"{_fold_two_coeff(amp)} {_cos_of_diff(dr, ksyms)}")
    return _join_signed(terms)


def _offdiag_entry(bonds, ksyms, conj=False) -> str:
    """Off-diagonal Bloch entry Σ amp·e^{±i k·ΔR} (conjugate for the lower triangle)."""
    terms = [_exp_term(amp, tuple(-x for x in dr) if conj else dr, ksyms)
             for dr, amp in bonds]
    return _join_signed(terms)


def _weighted_abs_sq(bonds, ksyms) -> str:
    """|Σ_n a_n e^{i k·ΔR_n}|² = Σ a_n² + 2 Σ_{m<n} a_m a_n cos(k·(ΔR_m−ΔR_n))."""
    if not bonds:
        return "0"
    out = " + ".join(_sq(amp) for _, amp in bonds)
    for m in range(len(bonds)):
        for n in range(m + 1, len(bonds)):
            (drm, am), (drn, an) = bonds[m], bonds[n]
            diff = tuple(x - y for x, y in zip(drm, drn))
            cm = am[1:].strip() if am.startswith("-") else am
            cn = an[1:].strip() if an.startswith("-") else an
            neg = am.startswith("-") ^ an.startswith("-")
            out += f" {'-' if neg else '+'} 2 {cm} {cn} {_cos_of_diff(diff, ksyms)}"
    return out


_KVARS = {1: ["k"], 2: ["k_x", "k_y"], 3: ["k_x", "k_y", "k_z"]}


def build_crystal_bloch(bonds_raw):
    """Build the Bloch Hamiltonian H̃(k) and bands from an abstract crystal (3c).

    *bonds_raw* is a list of ``(α, β, ΔR, amplitude)`` — sublattice indices, the
    cell displacement ΔR (reduced coords, dimension 1–3), and a c-number amplitude.
    Each bond's Hermitian conjugate is added automatically.
    Returns ``(matrix_latex_or_None, band_latex_or_None, error_or_None)``: the N×N
    matrix (None for N=1), and closed-form bands for N≤2 (None for N≥3).
    """
    if not bonds_raw:
        return None, None, "crystal: no bonds given"
    dims = {len(dr) for _, _, dr, _ in bonds_raw}
    if len(dims) != 1:
        return None, None, "crystal: every ΔR must have the same dimension"
    dim = dims.pop()
    if dim not in _KVARS:
        return None, None, "crystal: ΔR dimension must be 1, 2, or 3"
    ksyms = _KVARS[dim]
    n = 1 + max(max(a, b) for a, b, _, _ in bonds_raw)

    diag = {i: [] for i in range(n)}
    off: "dict[tuple, list]" = {}
    for a, b, dr, amp in bonds_raw:
        amp = amp.strip()
        if a == b:
            diag[a].append((dr, amp))
        else:
            if a > b:
                a, b, dr = b, a, tuple(-x for x in dr)
            off.setdefault((a, b), []).append((dr, amp))

    kvar = "k" if dim == 1 else r"\mathbf{k}"

    def cell(i, j):
        if i == j:
            return _diag_entry(diag[i], ksyms)
        if i < j:
            return _offdiag_entry(off.get((i, j), []), ksyms)
        return _offdiag_entry(off.get((j, i), []), ksyms, conj=True)

    matrix = None
    if n >= 2:
        rows = r" \\ ".join(" & ".join(cell(i, j) for j in range(n))
                            for i in range(n))
        matrix = rf"\tilde H({kvar}) = \begin{{pmatrix}} {rows} \end{{pmatrix}}"

    band = None
    if n == 1:
        band = rf"\varepsilon({kvar}) = {_diag_entry(diag[0], ksyms)}"
    elif n == 2:
        da, db = _diag_entry(diag[0], ksyms), _diag_entry(diag[1], ksyms)
        inner = _weighted_abs_sq(off.get((0, 1), []), ksyms)
        if da == db:
            pre = "" if da in ("0", "") else f"{da} "
            band = rf"\varepsilon_\pm({kvar}) = {pre}\pm \sqrt{{ {inner} }}"
        else:
            dbn = db.startswith("-")
            dbc = db[1:].strip() if dbn else db
            num_sum = f"{da} - {dbc}" if dbn else f"{da} + {db}"
            num_diff = f"{da} + {dbc}" if dbn else f"{da} - {db}"
            avg = rf"\frac{{ {num_sum} }}{{2}}"
            half = rf"\left( \frac{{ {num_diff} }}{{2}} \right)^2"
            band = rf"\varepsilon_\pm({kvar}) = {avg} \pm \sqrt{{ {half} + {inner} }}"
    else:
        # N≥3: a general closed form is blocked (cubic/quartic with cos entries →
        # Abel–Ruffini for N≥5).  The analytic representation is the implicit band
        # (secular) equation det(H̃(k) − εI) = 0; the bands are its roots.
        band = rf"\det\!\left(\tilde H({kvar}) - \varepsilon I\right) = 0"
    return matrix, band, None


# ---------------------------------------------------------------------------
# F-1 stage 3c-graph: the adjacency matrix A of a finite graph and its algebra.
#
# Unlike the infinite-lattice dispersion (symbolic cos entries → cubic walls), a
# finite graph's adjacency matrix is a concrete INTEGER matrix, so its powers,
# walk counts, degree sequence and spectrum are exact.  A is built from the same
# bond generators (or a custom edge list) used for F-1 expansion; registering it
# as an ordinary matrix expression lets the existing eigenvalues / determinant /
# trace operations chain for free.
# ---------------------------------------------------------------------------

def build_adjacency_matrix(pair_constraint: str, n_sites=None, boundary: str = "open",
                           lattice_shape=None, edges=None):
    """Build the adjacency matrix A (list of int rows) of a finite graph (3c-graph).

    The graph comes from a bond generator (``pair_constraint`` + site count /
    ``lattice_shape`` + ``boundary``) or an explicit ``custom`` edge list.
    Returns ``(rows, None)`` or ``(None, error_message)``.
    """
    pc = (pair_constraint or "nearest_neighbor").strip() or "nearest_neighbor"
    if pc == "custom":
        bonds = _parse_edges(edges)
        if not bonds:
            return None, ("adjacency: pair_constraint 'custom' needs an 'edges' "
                          "list, e.g. '(1,2),(2,3),(1,3)'")
        n = max(max(a, b) for a, b in bonds)
    else:
        shape = None
        if lattice_shape not in (None, ""):
            m = re.match(r"\s*(\d+)\s*[x×*]\s*(\d+)\s*$", str(lattice_shape))
            if not m:
                return None, f"adjacency: invalid lattice_shape {lattice_shape!r}"
            shape = (int(m.group(1)), int(m.group(2)))
        try:
            n = shape[0] * shape[1] if shape else int(n_sites)
        except (TypeError, ValueError):
            return None, "adjacency: n_sites must be an integer"
        if n < 1:
            return None, "adjacency: site count must be ≥ 1"
        if boundary not in ("open", "periodic", "cylinder"):
            return None, "adjacency: boundary must be 'open', 'periodic', or 'cylinder'"
        gen = _BOND_GENERATORS.get(pc)
        if gen is None:
            return None, (f"adjacency: unknown pair_constraint {pc!r}; "
                          f"known: {sorted(_BOND_GENERATORS)}")
        try:
            bonds = gen(n, boundary, shape)
        except ValueError as exc:
            return None, f"adjacency: {exc}"
    a = [[0] * n for _ in range(n)]
    for i, j in bonds:
        a[i - 1][j - 1] = 1
        a[j - 1][i - 1] = 1
    return a, None


def adjacency_to_canonical(rows) -> str:
    """A as a Maxima matrix literal ``matrix([0,1,…],[1,0,…],…)`` (CAS-ready)."""
    return "matrix(" + ",".join("[" + ",".join(map(str, r)) + "]" for r in rows) + ")"


def adjacency_to_latex(rows) -> str:
    """A as a pmatrix for display."""
    body = r" \\ ".join(" & ".join(map(str, r)) for r in rows)
    return rf"\begin{{pmatrix}} {body} \end{{pmatrix}}"


def _matmul(a, b):
    n = len(a)
    return [[sum(a[i][k] * b[k][j] for k in range(n)) for j in range(n)]
            for i in range(n)]


def graph_degree_sequence(rows) -> "list[int]":
    """Coordination numbers z_n = Σ_j A_{nj} (row sums)."""
    return [sum(r) for r in rows]


def graph_triangle_count(rows) -> int:
    """Number of triangles = tr(A³)/6."""
    a3 = _matmul(_matmul(rows, rows), rows)
    return sum(a3[i][i] for i in range(len(rows))) // 6


def graph_walk_trace(rows, m: int) -> int:
    """Number of closed walks of length m = tr(A^m) (the m-th spectral moment)."""
    p = [[1 if i == j else 0 for j in range(len(rows))] for i in range(len(rows))]
    for _ in range(m):
        p = _matmul(p, rows)
    return sum(p[i][i] for i in range(len(rows)))


def graph_edge_count(rows) -> int:
    """Number of edges |E| = ½ Σ degrees."""
    return sum(graph_degree_sequence(rows)) // 2


def graph_four_cycle_count(rows) -> int:
    """Number of 4-cycles (squares) N□ in the graph."""
    n = len(rows)
    nbr = [{j for j in range(n) if rows[i][j]} for i in range(n)]
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            common = nbr[i] & nbr[j]
            c = len(common - {i, j})
            total += c * (c - 1) // 2          # pairs of common neighbours
    return total // 2                          # each 4-cycle counted on both diagonals


# General spectral-graph identities (hold for any simple undirected graph): a
# symbolic left/right statement plus the two integer functions that evaluate them
# on a concrete A.  Used to state the theorems and to verify them on a graph.
_GRAPH_IDENTITIES = [
    (r"\operatorname{tr} A", "0",
     lambda a: graph_walk_trace(a, 1), lambda a: 0),
    (r"\operatorname{tr} A^2", "2|E|",
     lambda a: graph_walk_trace(a, 2), lambda a: 2 * graph_edge_count(a)),
    (r"\operatorname{tr} A^3", r"6 N_\triangle",
     lambda a: graph_walk_trace(a, 3), lambda a: 6 * graph_triangle_count(a)),
]


def graph_identity_statements() -> str:
    """The general spectral-graph identities as a symbolic display (no graph)."""
    return r",\ ".join(f"{lhs} = {rhs}" for lhs, rhs, _, _ in _GRAPH_IDENTITIES)


def graph_identity_check(rows):
    """Verify the identities on a concrete A: both sides evaluated to the same
    integer.  Returns ``(display_latex, all_hold)``."""
    lines, ok = [], True
    for lhs, rhs, lf, rf in _GRAPH_IDENTITIES:
        lv, rv = lf(rows), rf(rows)
        if lv == rv:
            lines.append(f"{lhs} = {rhs} = {lv}")
        else:
            ok = False
            lines.append(rf"{lhs} = {lv} \neq {rhs} = {rv}")
    return r",\ ".join(lines), ok


def build_neighbor_sum_terms(
    body: str, index1: str, index2: str, n_sites, boundary: str = "open",
    pair_constraint: str = "nearest_neighbor", lattice_shape=None, edges=None,
    pin_site=None,
) -> "tuple[str | None, str | None]":
    """Backend-agnostic expansion of a pairwise lattice/molecular sum (F-1 / B).

    Generates the index pairs for *pair_constraint* (default the 1D-chain
    nearest-neighbor bonds ⟨ij⟩; ``all_pairs`` gives every i<j for molecular /
    quantum-chemistry pairwise sums), substitutes the two site indices in
    *body* at the user-notation level (e.g. ``Sz_i`` → ``Sz_1``), and returns
    the explicit sum as a user-notation string ``"(term1) + (term2) + …"``.
    The caller resolves that string through its own backend
    parser/resolver/generator, so this helper carries no backend dependency and
    is shared by the Julia and Maxima handlers.

    Returns ``(sum_str, None)`` on success or ``(None, error_message)``.
    """
    body = (body or "").strip()
    idx1 = (index1 or "").strip()
    idx2 = (index2 or "").strip()
    boundary = (boundary or "open").strip().lower() or "open"
    pair_constraint = (pair_constraint or "nearest_neighbor").strip() or "nearest_neighbor"
    if not body:
        return None, "expand_neighbor_sum: missing body"
    if not idx1 or not idx2 or idx1 == idx2:
        return None, ("expand_neighbor_sum: index1 and index2 must be two "
                      "distinct site-index names")
    # custom: an arbitrary graph given as an explicit edge list (molecules,
    # defects, any lattice).  The bonds come straight from `edges`; geometry
    # and boundary do not apply.
    if pair_constraint == "custom":
        bonds = _parse_edges(edges)
        if not bonds:
            return None, ("expand_neighbor_sum: pair_constraint 'custom' needs an "
                          "'edges' list, e.g. '(1,2),(2,3),(1,3)'")
        gen = None
        shape = None
    else:
        # 2D lattice shape "LxxLy" (e.g. "3x4"); separators x / × / *.  When given
        # it overrides n_sites with Lx*Ly (the true site count).
        shape = None
        if lattice_shape not in (None, ""):
            m = re.match(r"\s*(\d+)\s*[x×*]\s*(\d+)\s*$", str(lattice_shape))
            if not m:
                return None, (f"expand_neighbor_sum: invalid lattice_shape "
                              f"{lattice_shape!r}; use 'LxxLy' e.g. '3x4'")
            shape = (int(m.group(1)), int(m.group(2)))
        try:
            n = shape[0] * shape[1] if shape else int(n_sites)
        except (TypeError, ValueError):
            return None, "expand_neighbor_sum: n_sites must be an integer"
        if n < 2:
            return None, "expand_neighbor_sum: site count must be ≥ 2"
        if boundary not in ("open", "periodic", "cylinder"):
            return None, ("expand_neighbor_sum: boundary must be 'open', "
                          "'periodic', or 'cylinder'")
        gen = _BOND_GENERATORS.get(pair_constraint)
        if gen is None:
            return None, (f"expand_neighbor_sum: unrecognized lattice "
                          f"{pair_constraint!r}; known named lattices: "
                          f"{sorted(_BOND_GENERATORS)}. For other lattices, "
                          f"specify the bonds directly, e.g. \"the "
                          f"nearest-neighbor sum of ... over the graph "
                          f"(1,2),(2,3),(1,3)\"")

    # Normalize multi-index braced subscripts so a two-index pair symbol like a
    # molecular distance r_{ij} (parsed as two indices i,j) substitutes cleanly:
    # split a run of 2+ adjacent lowercase index letters with commas
    # (r_{ij} → r_{i,j}), so substituting i→1, j→2 yields r_{1,2} = r(1,2)
    # rather than the ambiguous r_{12} = r(12).  Uppercase multi-char labels
    # (H_{BCS}) and already-comma'd forms are left untouched.
    def _split_brace_indices(s: str) -> str:
        return re.sub(
            r"_\{([a-z]{2,})\}",
            lambda m: "_{" + ",".join(m.group(1)) + "}",
            s,
        )

    def _sub_site(text: str, idx: str, site: int) -> str:
        # (1) Inside a braced subscript _{...}: replace the index when it stands
        #     alone (bounded by {, comma, or }), e.g. r_{i,j} → r_{1,j}.
        def _brace(m: "re.Match") -> str:
            inner = re.sub(
                r"(?<![A-Za-z0-9])" + re.escape(idx) + r"(?![A-Za-z0-9])",
                str(site), m.group(1))
            return "_{" + inner + "}"
        text = re.sub(r"_\{([^{}]*)\}", _brace, text)
        # (2) Bare / chained subscript _idx where idx is a complete index — it
        #     ends at a non-alphanumeric, an underscore (next chained index), or
        #     end of string.  `_i_j` → `_1_j`; `Sz_i` → `Sz_1`; `_init` is left
        #     alone (idx is followed by a letter).
        text = re.sub(r"_" + re.escape(idx) + r"(?![A-Za-z0-9])", f"_{site}", text)
        return text

    if gen is not None:  # custom already has its bonds from the edge list
        try:
            bonds = gen(n, boundary, shape)
        except ValueError as exc:
            return None, f"expand_neighbor_sum: {exc}"
    if not bonds:
        return None, "expand_neighbor_sum: no index pairs generated"
    # pin_site (B stage 3a): concretize a neighbour sum Σ_{j∈nn(n)} body(n,j) at a
    # concrete site n — keep only bonds containing pin_site and orient each so the
    # pinned index → pin_site and the other index → the neighbour.
    if pin_site is not None:
        try:
            ps = int(pin_site)
        except (TypeError, ValueError):
            return None, "expand_neighbor_sum: pin_site must be an integer"
        pinned = []
        for a, b in bonds:
            if a == ps:
                pinned.append((ps, b))
            elif b == ps:
                pinned.append((ps, a))
        if not pinned:
            return None, (f"expand_neighbor_sum: site {ps} has no neighbours "
                          f"on this lattice")
        bonds = pinned
    body = _split_brace_indices(body)
    terms = [f"({_sub_site(_sub_site(body, idx1, a), idx2, b)})" for a, b in bonds]
    return " + ".join(terms), None
