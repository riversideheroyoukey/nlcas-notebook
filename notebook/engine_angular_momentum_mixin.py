"""NotebookEngine mixin: F-9 angular-momentum coupling (Clebsch–Gordan,
Wigner 3j/6j/9j, coupling decomposition, Landé g, Wigner–Eckart).

CAS-bypass, like the spin algebra: every value is computed exactly in
:mod:`notebook.runtime.angular_momentum` and registered directly, so this works
identically in a Maxima and a Julia session — angular-momentum coupling is not
a backend's business (Core Principle #6 stays intact; nothing is "routed"
anywhere).

The results are not dead ends.  A coefficient's canonical is an exact CAS value
(``sqrt(6)/3``) and the coupling identity's canonical is a symbolic formula in
``J`` (``(J*(J+1)-15/4-3/4)/2``), so both feed straight into ordinary
operations — substitute, multiply, solve.  That is deliberately where the
physics stops: this layer supplies the exact algebra of coupling, and the
researcher supplies the Hamiltonian.
"""
from __future__ import annotations

import re

from notebook.models.expression import Expression  # noqa: F401
from notebook.runtime import angular_momentum as am
from notebook.runtime import angular_momentum_latex as amx

# A basis name must be a plain identifier: no subscript, no decoration.
_PLAIN_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9]*$")


class EngineAngularMomentumMixin:
    """F-9: angular-momentum coupling (see module docstring)."""

    # ------------------------------------------------------------------
    # Coefficients and symbols
    # ------------------------------------------------------------------

    def clebsch_gordan_coefficient(self, j1, m1, j2, m2, big_j, big_m) -> Expression:
        """``⟨j1 m1 j2 m2 | J M⟩`` as an exact registered value."""
        nums = [am.parse_j(x) for x in (j1, m1, j2, m2, big_j, big_m)]
        value = am.clebsch_gordan(*nums)
        return self._register_angular(
            latex=f"{amx.cg_braket_latex(*nums)} = {value.latex()}",
            canonical=value.canonical(),
            descriptor={"kind": "clebsch_gordan",
                        "quantum_numbers": [am.j_canonical(x) for x in nums],
                        "value": value.canonical()},
        )

    def wigner_symbol(self, kind: str, rows: "list[list]") -> Expression:
        """A Wigner ``3j`` / ``6j`` / ``9j`` symbol as an exact registered value.

        ``rows`` is the symbol as written: two rows for 3j and 6j, three for 9j.
        """
        funcs = {"3j": am.wigner_3j, "6j": am.wigner_6j, "9j": am.wigner_9j}
        widths = {"3j": (2, 3), "6j": (2, 3), "9j": (3, 3)}
        if kind not in funcs:
            raise am.AngularMomentumError("unknown_wigner_symbol", kind=kind)
        want_rows, want_cols = widths[kind]
        if len(rows) != want_rows or any(len(r) != want_cols for r in rows):
            raise am.AngularMomentumError("wigner_symbol_shape", kind=kind,
                                          rows=want_rows, cols=want_cols)
        nums = [[am.parse_j(x) for x in row] for row in rows]
        flat = [x for row in nums for x in row]
        # 3j is written (j j j / m m m) but called (j1 j2 j3, m1 m2 m3) — the
        # written layout IS the call order for all three symbols.
        value = funcs[kind](*flat)
        return self._register_angular(
            latex=f"{amx.wigner_symbol_latex(kind, nums)} = {value.latex()}",
            canonical=value.canonical(),
            descriptor={"kind": f"wigner_{kind}",
                        "rows": [[am.j_canonical(x) for x in row] for row in nums],
                        "value": value.canonical()},
        )

    # ------------------------------------------------------------------
    # Coupling
    # ------------------------------------------------------------------

    def couple_angular_momenta(self, j1, j2, total_symbol: str = "J",
                               symbols: "tuple[str, str] | None" = None
                               ) -> "list[Expression]":
        """Couple ``j1 ⊗ j2``: register the allowed total angular momenta AND the
        exact ``j1·j2`` identity.

        Two expressions, because the coupling answers two different questions:
        *which levels exist* (the multiplet structure — spectroscopy) and *how
        far apart they are* (the ``j1·j2`` eigenvalue — the Landé interval rule
        for hyperfine and fine structure).  The second keeps ``J`` symbolic, so
        it composes with the researcher's own Hamiltonian rather than assuming
        one.
        """
        a, b = am.parse_j(j1), am.parse_j(j2)
        values = am.coupled_j_values(a, b)
        listed = [am.j_canonical(v) for v in values]
        allowed = self._register_angular(
            latex=amx.allowed_j_latex(a, b, values, total_symbol),
            canonical=amx.allowed_j_canonical(values),
            descriptor={"kind": "coupling", "j1": am.j_canonical(a),
                        "j2": am.j_canonical(b), "allowed": listed,
                        "total_symbol": total_symbol},
        )
        identity = self._register_angular(
            latex=amx.dot_product_latex(a, b, total_symbol, symbols),
            canonical=amx.dot_product_canonical(a, b, total_symbol),
            descriptor={"kind": "coupling_dot_product", "j1": am.j_canonical(a),
                        "j2": am.j_canonical(b), "allowed": listed,
                        "total_symbol": total_symbol},
        )
        return [allowed, identity]

    def coupled_state_expansion(self, j1, j2, big_j, big_m,
                                basis: "tuple[str, str] | None" = None,
                                state_symbol: str | None = None) -> Expression:
        """``|J M⟩`` written in the uncoupled basis ``|j1 m1⟩|j2 m2⟩``.

        The product states are written as NAMED functions, so the canonical is
        a genuine expression (``sqrt(2)/2*chi_1(1/2)*chi_2(-1/2) + …``) and the
        display mirrors it exactly.  An anonymous ket could not do that — it has
        no canonical the CAS can act on — and a result that merely *looks*
        computable is the failure this avoids.

        ``basis`` names the two bases; **the researcher decides, and if they do
        not, the system picks a pair that is guaranteed free** in this session
        (``chi_1``/``chi_2``, bumped on collision).  Supplied names are checked
        against everything already in use — see :meth:`_check_basis_names_free`.
        """
        a, b = am.parse_j(j1), am.parse_j(j2)
        big_j, big_m = am.parse_j(big_j), am.parse_j(big_m)
        terms = am.cg_expansion(a, b, big_j, big_m)
        auto = basis is None
        if auto:
            basis = self._fresh_basis_names()
        else:
            basis = (str(basis[0]), str(basis[1]))
            self._check_basis_names_free(basis)
        descriptor = {"kind": "coupled_state", "j1": am.j_canonical(a),
                      "j2": am.j_canonical(b), "J": am.j_canonical(big_j),
                      "M": am.j_canonical(big_m),
                      "basis": list(basis), "basis_auto": auto,
                      "cas_usable": True,
                      "terms": [[am.j_canonical(m1), am.j_canonical(m2),
                                 c.canonical()] for m1, m2, c in terms]}
        # The ordinary registration path, so the undefined-symbol scan runs and
        # the basis symbols are surfaced like any other new symbol.
        expr = self.register_expression(
            amx.expansion_canonical(terms, basis),
            latex=amx.expansion_latex(a, b, big_j, big_m, terms, basis,
                                      state_symbol),
            skip_cas_bind=True)
        expr.angular_descriptor = descriptor
        return expr

    # ------------------------------------------------------------------
    # Spectroscopy / magnetic resonance
    # ------------------------------------------------------------------

    def lande_g_factor(self, orbital=None, spin=None, total=None,
                       term_symbol: str | None = None) -> Expression:
        """The Landé g factor of an LS-coupled level, from (L, S, J) or a term
        symbol such as ``^2P_{3/2}``."""
        if term_symbol:
            spin, orbital, total = am.parse_term_symbol(term_symbol)
        elif orbital is None or spin is None or total is None:
            raise am.AngularMomentumError("lande_missing_inputs")
        else:
            orbital, spin, total = (am.parse_j(orbital), am.parse_j(spin),
                                    am.parse_j(total))
        value = am.lande_g_factor(orbital, spin, total)
        level = amx.term_symbol_latex(spin, orbital, total)
        return self._register_angular(
            latex=amx.lande_g_latex(level, value),
            canonical=am.j_canonical(value),
            descriptor={"kind": "lande_g", "L": am.j_canonical(orbital),
                        "S": am.j_canonical(spin), "J": am.j_canonical(total),
                        "term_symbol": level, "value": am.j_canonical(value)},
        )

    def wigner_eckart_matrix_element(self, j, m, rank, q, jp, mp,
                                     operator: str = "T") -> Expression:
        """``⟨j' m'|T^k_q|j m⟩`` reduced by the Wigner–Eckart theorem.

        The registered LaTeX keeps the reduced matrix element ``⟨j'‖T^k‖j⟩``
        symbolic — that factor is precisely what the theorem does not give —
        while the canonical is the exact geometric coefficient, which is what
        selection rules and relative line strengths are made of.
        """
        nums = [am.parse_j(x) for x in (j, m, rank, q, jp, mp)]
        coeff = am.wigner_eckart_coefficient(*nums)
        return self._register_angular(
            latex=amx.wigner_eckart_latex(*nums, coeff, operator=operator),
            canonical=coeff.canonical(),
            descriptor={"kind": "wigner_eckart",
                        "quantum_numbers": [am.j_canonical(x) for x in nums],
                        "operator": operator, "value": coeff.canonical()},
        )

    # ------------------------------------------------------------------
    # Naming
    # ------------------------------------------------------------------

    #: Candidate auto-generated basis names, in preference order.  Greek letters
    #: conventionally used for state factors, and — crucially — SUBSCRIPT-FREE:
    #: ``chi_1(1/2)`` parses as ``chi`` subscripted by ``1(1/2)`` (the project's
    #: subscript rule), so a generated name must never carry one.
    _BASIS_NAME_POOL = ("chi", "psi", "xi", "eta", "zeta", "phi", "theta",
                        "sigma", "tau", "upsilon", "u", "v", "w")

    def _fresh_basis_names(self) -> "tuple[str, str]":
        """A pair of basis names guaranteed unused in this session.

        Used when the researcher does not name the bases themselves.  Picking
        two DIFFERENT letters (rather than one letter with indices) keeps each
        name a plain identifier, which is what makes the canonical unambiguous.
        """
        free = [n for n in self._BASIS_NAME_POOL if not self._symbol_in_use(n)]
        if len(free) < 2:
            raise am.AngularMomentumError("no_free_basis_name")
        return free[0], free[1]

    def _check_basis_names_free(self, basis: "tuple[str, str]") -> None:
        """Refuse basis names that are already in use in this session.

        A basis name becomes a FUNCTION symbol in the canonical
        (``phi(-1/2)``).  If the session already bound that name — H-1 sends
        ``phi : x^2`` to the CAS, so the binding is live — the expansion
        silently degrades into nonsense that still reports success:

            phi = x^2  →  simplify  →  … chi(-(1/2))*(x^2)(1/2) …

        No error anywhere, which is the failure mode this notebook must never
        have.  Names are therefore checked BEFORE anything is registered, and a
        clash stops the operation with a message asking for a free symbol.
        """
        first, second = basis
        if first == second:
            raise am.AngularMomentumError("basis_names_identical", name=first)
        for name in basis:
            # A subscripted name would be read as a subscript, not a function:
            # "chi_1(1/2)" parses as chi subscripted by "1(1/2)".
            if not _PLAIN_NAME_RE.match(name):
                raise am.AngularMomentumError("basis_name_invalid", name=name)
            if self._symbol_in_use(name):
                raise am.AngularMomentumError("basis_name_taken", name=name)

    def _symbol_in_use(self, name: str) -> bool:
        """True if ``name`` already denotes something in this session."""
        if name in getattr(self, "lhs_alias_map", {}):
            return True
        if name in (getattr(self.symbol_context, "registry", {}) or {}):
            return True
        if name in (getattr(self.symbol_context, "bracket_functionals", {}) or {}):
            return True
        # An earlier expansion's basis names too: reusing one for a DIFFERENT
        # pair of angular momenta would make two unrelated states share a
        # symbol, and neither expression would say so.
        for expr in self.expressions.values():
            used = (expr.angular_descriptor or {}).get("basis") or []
            if name in used:
                return True
        return False

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def _register_angular(self, latex: str, canonical: "str | None",
                          descriptor: dict) -> Expression:
        """Register an F-9 result with its coupling provenance.

        ``register_direct`` (not ``register_expression``): the value is already
        exact and the display carries bra-ket / matrix notation that is display
        form, not a canonical the parser should try to read back.  ``canonical``
        is None for objects that are genuinely not CAS values (a coupled state).
        """
        return self.register_direct(latex, canonical=canonical,
                                    angular_descriptor=descriptor)
