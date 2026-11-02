from __future__ import annotations
from typing import List, Literal, Optional, Dict
from pydantic import BaseModel, Field

from notebook.models.quantity import Quantity, MathType


class SymbolStatistics(BaseModel):
    """Statistics or Commutativity of a symbol."""
    kind: Literal["bose", "fermi", "noncommutative"] = "bose"


class SymbolProperty(BaseModel):
    """Extended properties for a mathematical symbol in the context."""
    quantity: Quantity
    statistics: SymbolStatistics = Field(default_factory=SymbolStatistics)
    einstein_summation: bool = False


class SymbolContext(BaseModel):
    """Semantic Layer: Manages the roles, dimensions, and statistics of all symbols."""
    registry: Dict[str, SymbolProperty] = Field(default_factory=dict)
    domains: list[str] = Field(default_factory=list)
    session_mode: str = "standard_symbolic"
    einstein_summation: bool = False
    notation_usage: Dict[str, str] = Field(default_factory=dict)
    """Maps raw LaTeX token to resolved role (e.g. "\\Delta_k" -> "gap_function").
    Populated during notation resolution; survives save/load so the same symbol
    is not re-disambiguated on session reload."""

    bracket_functionals: Dict[str, str] = Field(default_factory=dict)
    """Maps a symbol name → its CAS function name for display-only bracket
    functionals (R-9).  ``E[X]`` (expectation), ``Var[X]``, a user-declared
    ``F[X]`` etc. render as ``Sym[args]`` (display) but pass ``cas_name(args)``
    to the CAS with NO operator algebra — the symbol is NOT registered as an
    operator Quantity.  E is added automatically in statistics-family domains
    (notation profile ``E_bracket``); others are declared explicitly.  Survives
    save/load.
    """

    canonical_pairs: list[tuple[str, str]] = Field(default_factory=list)
    """Canonical conjugate pairs (q_i, p_i) for Hamiltonian mechanics.

    Declared via "p_theta を theta の共役運動量とする" / "theta と p_theta を
    正準対とする" and consumed by the ``poisson_bracket`` operation to build
    {f,g} = Σ_i (∂f/∂q_i ∂g/∂p_i − ∂f/∂p_i ∂g/∂q_i) without writing the pairs
    on every call.  Each entry is (coordinate, conjugate_momentum).  Survives
    save/load so a session's canonical structure persists.
    """

    greek_glyph_styles: Dict[str, str] = Field(default_factory=dict)
    r"""Maps a Greek letter's canonical name → the command the researcher WROTE.

    Three letters have two LaTeX spellings for two different glyphs, and the
    CAS answers in the one the researcher may not have used: Maxima's ``tex1``
    returns ``\varphi`` for ``phi``, ``\varepsilon`` for ``epsilon`` and
    ``\vartheta`` for ``theta`` (measured 2026-08-16 over all 34 Greek names —
    exactly these three).  The converter maps BOTH spellings to one canonical,
    so the canonical cannot say which was written and the paper ends up with
    the same quantity in two glyphs on adjacent lines::

        \phi_{0} = \phi                                   ← what was written
        \frac{\partial \phi_{0}}{\partial x} = \frac{d}{d x} \varphi   ← what came back

    Populated on FIRST USE, like `spin_display_styles` above: the first input
    is the notation declaration for the session.  Nothing is recorded until a
    letter is actually written, so a session that never writes one is
    unaffected, and a researcher who writes ``\varphi`` keeps ``\varphi``.
    """

    spin_display_styles: Dict[str, str] = Field(default_factory=dict)
    """Maps spin operator base name → user's preferred display style.

    Populated on first use of a spin component in an expression.  The first
    input is treated as the notation declaration for the entire session.

    Values:
      "suffix"      — ``Sy_1``   (component as identifier suffix, default)
      "superscript" — ``S^y_1``  (component as LaTeX superscript)

    Example: if the user's first input is ``H = -gamma * B * S^z_1``,
    ``spin_display_styles["S"] = "superscript"`` is recorded and all
    subsequent results show ``S^x_1``, ``S^y_1``, ``S^z_1``.
    """

    spin_projections: Dict[str, List[int]] = Field(default_factory=dict)
    r"""演算子の名前 → そのスピン射影 $m$ の値集合（**分子**、÷2）。

    電子なら ``{"c": [-1, 1]}``（$m=\pm1/2$）。分子で持つのは半整数を float に
    しないため（F-9 の `Surd` と同じ理由）。宣言に書かれた**粒子**から決まり
    （`models/particles.json` の `two_s` / `projection_numerators`）、決まらない
    粒子では**空のまま**——推測しない。

    これがあると `\sum_\sigma` を値集合に展開できる。展開が要るのは、QA が形式和
    どうしの交換子でダミー入れ替えの相殺を畳まないためである（実測 2026-08-22、
    `notebook/runtime/spin_value_set.py` にその測定がある）。

    追加キーなので、これより前に保存されたセッションはそのまま読める。
    """

    site_spins: Dict[str, str] = Field(default_factory=dict)
    """F-13: the spin magnitude of each lattice SITE, e.g. ``{"1": "1/2", "2": "1"}``.

    Declared by the researcher ("サイト1のスピンを1/2、サイト2を1とする") when the
    sites are different species — an electron and a nucleus (hyperfine), two
    nuclear species (heteronuclear NMR), a mixed-spin chain.  The spin-trace
    evaluation then uses a separate ``x_i = S_i(S_i+1)`` per site instead of one
    shared ``x`` (``spin_algebra.expectation_general``).

    A site may be mapped to ``""`` to say "this site's spin stays symbolic".
    Sites absent from the map are symbolic too, but the map being NON-EMPTY is
    what switches the evaluation to per-site mode: an empty map means the
    uniform-spin behaviour that predates F-13.

    Values are strings so the session file stays plain JSON; they are parsed as
    exact half-integers on use.  Survives save/load.
    """

    indexed_bases: Dict[str, int] = Field(default_factory=dict)
    """docs/32 Phase 4: bases the researcher has written with a subscript, and how
    many indices they carried, e.g. ``{"V": 1, "A": 2}``.

    An UNDECLARED subscripted symbol resolves to an ``IndexedSymbol`` and lowers
    to a CAS function call — ``V_i`` becomes ``V(i)``, ``omega_0`` becomes
    ``omega(0)``.  That is the documented lowering policy, but the CAS then
    returns results in that spelling, and without this record the timeline showed
    the researcher ``\\omega\\left(0\\right)`` — CAS syntax, in violation of Core
    Principle #1.  The display bridge uses this map to restore the subscript.

    Recorded from the IR of what the researcher REGISTERED, not from CAS output:
    a result canonical ``a*V(i)+b`` parses as an ordinary FunctionCall, so the
    base has to be learned from the source expression.

    Not a declaration and not user-visible: it records what was written.  A base
    that is later declared as a name (``V_i`` as "名前の一部") no longer produces
    an IndexedSymbol at all, so the entry simply stops being consulted.
    Additive persisted key — an older session file loads with an empty map and
    behaves exactly as before.
    """

    called_bases: List[str] = Field(default_factory=list)
    """docs/32 Phase 4: undeclared names the researcher has APPLIED to arguments,
    e.g. ``["V"]`` after writing ``V(x)``.

    Kept beside :attr:`indexed_bases` because the CAS spelling of the two is the
    same: an undeclared ``V_i`` lowers to ``V(i)``, so a result containing
    ``V(...)`` cannot be told apart from a genuine function call.  A base
    appearing in BOTH lists is the real ambiguity — the display bridge then
    restores nothing for it (showing CAS syntax beats showing two different
    quantities under one notation), and the interpreter asks the researcher which
    it is.

    Declared names are not recorded: a declaration already answers the question.
    Additive persisted key.
    """

    def register(self, name: str, quantity: Quantity, statistics_kind: Literal["bose", "fermi", "noncommutative"] = "bose") -> SymbolProperty:
        """Register or update a symbol in the context."""
        prop = SymbolProperty(quantity=quantity, statistics=SymbolStatistics(kind=statistics_kind))
        self.registry[name] = prop
        return prop

    def get(self, name: str) -> Optional[SymbolProperty]:
        """Retrieve a symbol property by name."""
        return self.registry.get(name)

    def get_quantity(self, name: str) -> Optional[Quantity]:
        """Backward compatible method to get the Quantity."""
        prop = self.get(name)
        return prop.quantity if prop else None

    def register_quantity(self, name: str, **kwargs) -> Quantity:
        """Backward compatible method to merge and register Quantity kwargs."""
        existing = self.get(name)
        if existing is not None:
            qty = existing.quantity.model_copy(update=kwargs)
            # preserve statistics
            self.registry[name] = SymbolProperty(quantity=qty, statistics=existing.statistics)
        else:
            kind = kwargs.pop("kind", "anonymous")
            qty = Quantity(name=name, kind=kind, **kwargs)
            self.registry[name] = SymbolProperty(quantity=qty, statistics=SymbolStatistics(kind="bose"))
        return qty

    def add_canonical_pair(self, coordinate: str, momentum: str) -> None:
        """Register a canonical conjugate pair (q, p).  Idempotent: an existing
        pair with the same coordinate is replaced (its momentum updated)."""
        coordinate, momentum = coordinate.strip(), momentum.strip()
        self.canonical_pairs = [
            (q, p) for (q, p) in self.canonical_pairs if q != coordinate
        ]
        self.canonical_pairs.append((coordinate, momentum))

    def is_commutative(self, name_a: str, name_b: str) -> bool:
        """Check if two symbols commute.
        
        Default rule:
        - Bose and Bose commute.
        - Fermi and Fermi anti-commute (thus they do NOT commute, AB = -BA).
        - Noncommutative with anything does not commute.
        """
        prop_a = self.get(name_a)
        prop_b = self.get(name_b)

        if not prop_a or not prop_b:
            # If unknown, assume commutative (Bose) by default, or safe to assume non-commutative?
            # Usually, standard variables are commutative.
            return True

        stat_a = prop_a.statistics.kind
        stat_b = prop_b.statistics.kind

        if stat_a == "bose" and stat_b == "bose":
            return True
        
        return False
