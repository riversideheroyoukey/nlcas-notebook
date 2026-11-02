"""Maxima semantic resolver: backend-specific hooks only.

All shared Surface-AST → Semantic-IR traversal lives in BaseSemanticResolver
(semantic_resolver.py).  This subclass supplies exactly the Maxima-specific
behavior: builtin function spellings, the %i imaginary unit, the EE (electrical
engineering) j/i convention, the rigorous prime-derivative resolution (with
PrimeDerivativeError confirmation requests), and the -k_B negation recovery.
"""
from notebook.models.ir import (
    Expr, Symbol, Number, Mul, FunctionCall, Derivative, IndexSymbol,
    BracketExpr,
)
from notebook.models.symbol_context import SymbolContext
from notebook.runtime.maxima_latex_converter import MaximaLatexConverter
from notebook.runtime.semantic_resolver import BaseSemanticResolver


class PrimeDerivativeError(ValueError):
    """Raised when prime notation cannot unambiguously determine the differentiation variable.

    Carries an i18n key so MaximaBackendHandler.parse_ir() can translate the
    message before surfacing it to the user.
    """
    def __init__(self, i18n_key: str, **fmt_kwargs: object) -> None:
        super().__init__(i18n_key)
        self.i18n_key = i18n_key
        self.fmt_kwargs = fmt_kwargs


def _collect_symbols(expr: Expr, out: set) -> None:
    """Recursively collect all Symbol names from an expression tree."""
    if isinstance(expr, Symbol):
        out.add(expr.name)
    elif isinstance(expr, Number):
        return
    # Named fields that may hold sub-expressions
    for field in ("args", "indices", "dummy_indices"):
        val = getattr(expr, field, None)
        if isinstance(val, (list, tuple)):
            for item in val:
                if isinstance(item, Expr):
                    _collect_symbols(item, out)
    for field in ("left", "right", "base", "exponent", "expr", "lhs", "rhs",
                  "func", "lower", "upper", "var", "a", "b", "body",
                  "sub", "script", "x", "y", "label"):
        val = getattr(expr, field, None)
        if isinstance(val, Expr):
            _collect_symbols(val, out)


def _free_vars_in(expr: Expr, context: SymbolContext) -> list[str]:
    """Return symbol names that are free variables (not declared scalars/functions/operators).

    A symbol is a free variable when it is either:
    - not registered in SymbolContext, or
    - registered with kind == "anonymous"

    Registered scalars, functions, and operators are treated as constants.
    """
    names: set = set()
    _collect_symbols(expr, names)
    free: list[str] = []
    for name in sorted(names):
        prop = context.get(name)
        if prop is None or prop.quantity.kind == "anonymous":
            free.append(name)
    return free


def _load_function_families() -> frozenset:
    """Maxima's own ``bessel_j``-shaped function names, from the backend spec.

    Data, not code (``backends/maxima.function_families.json``): the list is a
    mechanical extraction from the official manual, and keeping 124 names inline
    would put a table in a source file that nobody can check against its source.
    Missing or unreadable is not fatal — the names then behave as they did
    before the list existed (folded by the single-index default), so a broken
    file degrades to the old behaviour rather than to a wrong answer.
    """
    import json
    from pathlib import Path
    path = (Path(__file__).resolve().parents[2]
            / "backends" / "maxima.function_families.json")
    try:
        return frozenset(json.loads(path.read_text())["function_families"])
    except (OSError, ValueError, KeyError, TypeError):  # degraded, not wrong
        return frozenset()


class MaximaSemanticResolver(BaseSemanticResolver):
    """Semantic Resolver for Maxima backend.

    Transforms pure syntax Surface AST nodes into Maxima-compatible semantic
    representations.  See BaseSemanticResolver for the shared traversal; only
    the hooks below differ from Julia.
    """

    #: A Maxima session has no operator algebra (open_items F-2), so
    #: ``\psi^\dagger`` there is a NAME — the same thing ``\hat\psi`` is —
    #: unless the base is DECLARED a matrix / vector / tensor / operator.
    #: Measured 2026-08-20: five spellings of one creation operator became five
    #: different things, among them ``psi^dagger(x)`` (psi raised to the power
    #: ``dagger(x)``) and ``'transpose(psi) * x``.
    _DAGGER_DEFAULT_IS_ADJOINT = False

    #: This backend's function table, so the shared bare-name guard
    #: (`_reject_bare_function_names`) needs no list of its own — which names
    #: are functions is the backend's knowledge, exactly as `_FN_ALT` is.
    _FUNCTION_NAMES = MaximaLatexConverter._KNOWN_FUNCTIONS

    # Maxima builtin function names (added to the common elementary set) for the
    # "f (x)" space-disambiguation.  Maxima spellings: ceiling/signum/conjugate/
    # realpart/imagpart, plus atan2 and the integer/number-theory functions.
    known_function_names = BaseSemanticResolver.known_function_names | frozenset({
        "atan2", "ceiling", "signum", "conjugate", "realpart", "imagpart",
        "factorial", "binomial", "log10", "carg",
    }) | _load_function_families()

    imaginary_unit_name = "%i"
    resolve_sum_integral_prefix = True

    # ------------------------------------------------------------------
    # Square brackets
    # ------------------------------------------------------------------

    def _square_bracket_role(self, expr, context) -> str:
        """In Maxima ``[...]`` is a list, and a matrix is a list of rows.

        A bracket whose ELEMENTS are themselves brackets can therefore only be
        a nested list — a commutator of two lists is meaningless in any CAS.
        That is the case the CAS itself produces and stores:
        ``[[epsilon, Delta],[Delta, -epsilon]]`` (the S-y two-level Hamiltonian)
        used to resolve to a commutator of commutators and ``expand`` returned
        ``0`` as a success (open_items AUDIT-BRACKET-BACKEND).

        A FLAT two-element bracket stays a commutator: ``[A, B]`` on declared
        matrices is a real Maxima workflow (the code generator even picks ``.``
        over ``*`` for it — R-4/SR-4, tests/test_matrix_commutator.py), and with
        nothing declared the commutator reading is the established behaviour.
        Write a genuine two-element list as ``[a, b; ]``-style matrix input or
        build it from a CAS result; the ambiguity of a bare flat ``[a, b]`` is
        not resolvable here without guessing.
        """
        if any(isinstance(_a, BracketExpr) and _a.kind == "square"
               for _a in expr.args):
            return "list"
        return "commutator"

    # ------------------------------------------------------------------
    # Prime handling
    # ------------------------------------------------------------------

    def _prime_call_is_quantum(self, curr: Expr, inner_func: Expr,
                               context: SymbolContext) -> bool:
        return (
            "second_quantization" in context.domains or
            "quantum_mechanics" in context.domains or
            context.session_mode == "second_quantization"
        )

    def _resolve_prime_call_derivative(self, inner_func: Expr, expr: FunctionCall,
                                       order: int, context: SymbolContext) -> Expr:
        # In calculus/standard domain, prime represents total derivative.
        resolved_args = [self._resolve_node(arg, context) for arg in expr.args]
        if len(resolved_args) == 0:
            # f' with no args: keep as symbolic identifier (e.g. in equations)
            op_name = (inner_func.name if isinstance(inner_func, Symbol) else "") + ("'" * order)
            return Symbol(name=op_name) if isinstance(inner_func, Symbol) else inner_func
        if len(resolved_args) > 1:
            # g'(x, y): partial derivative variable is ambiguous
            raise PrimeDerivativeError(
                "engine.prime_derivative.specify_partial_variable"
            )
        arg0 = resolved_args[0]
        if isinstance(arg0, Symbol):
            # Simple case: f'(x) → diff(f(x), x)
            return Derivative(
                expr=FunctionCall(func=inner_func, args=resolved_args),
                vars=[(arg0, order)],
                is_partial=False,
            )
        # Complex argument: infer the differentiation variable from SymbolContext.
        # Collect all symbols in the argument; exclude declared scalars/functions/operators.
        free = _free_vars_in(arg0, context)
        if len(free) == 1:
            return Derivative(
                expr=FunctionCall(func=inner_func, args=resolved_args),
                vars=[(Symbol(name=free[0]), order)],
                is_partial=False,
            )
        # Cannot determine unique variable
        example = (inner_func.name if isinstance(inner_func, Symbol) else "f") + "'" * order
        hint_var = free[0] if free else "x"
        raise PrimeDerivativeError(
            "engine.prime_derivative.specify_variable",
            example=f"{example}({arg0})",
            var=hint_var,
        )

    # ------------------------------------------------------------------
    # Symbol handling (electrical-engineering convention)
    # ------------------------------------------------------------------

    def _resolve_backend_builtin_call(self, expr, context):
        r"""Maxima's own ``sum(body, i, lo, hi)`` / ``kron_delta(a, b)``, read back.

        These are the spellings THIS system writes for ``IndexSum`` and
        ``KroneckerDelta``, so a result that comes back carrying them is the
        same kind of object as the expression it was computed from — and until
        2026-08-21 it was not: the contraction that turns
        ``\sum_{n m}\delta_{nm}A_{n,m}`` into ``\sum_m A_{m,m}`` fired on the
        researcher's spelling and not on the CAS's, so substituting a
        diagonalising assumption into a Hamiltonian left the δ standing
        (the eigenvalue never appeared).  The noun form ``'sum`` reaches here
        with its quote still attached, so both spellings are listed.
        """
        from notebook.models.ir import (
            FunctionCall, IndexSum, IndexSymbol, KroneckerDelta, Symbol as _Sym)
        if not isinstance(expr, FunctionCall) or not isinstance(expr.func, _Sym):
            return None
        name = expr.func.name.lstrip("'")
        if name == "kron_delta" and len(expr.args) == 2:
            sides = []
            for arg in expr.args:
                node = self._resolve_node(arg, context)
                sides.append(IndexSymbol(name=node.name)
                             if isinstance(node, _Sym) else node)
            return KroneckerDelta(left=sides[0], right=sides[1])
        if name == "sum" and len(expr.args) in (2, 4):
            dummy = self._resolve_node(expr.args[1], context)
            if not isinstance(dummy, (_Sym, IndexSymbol)):
                return None
            index = IndexSymbol(name=dummy.name)
            # The dummy is read BEFORE the body, so a subscript naming it
            # resolves as an INDEX (the same order the `IndexSum` branch keeps).
            with self._binding([index.name]):
                body = self._resolve_node(expr.args[0], context)
            lower = (self._resolve_node(expr.args[2], context)
                     if len(expr.args) == 4 else None)
            upper = (self._resolve_node(expr.args[3], context)
                     if len(expr.args) == 4 else None)
            return IndexSum(body=body, dummy_indices=[index],
                            lower=lower, upper=upper,
                            noun=expr.func.name.startswith("'"))
        return None

    def _resolve_symbol_backend_special(self, expr: Symbol,
                                        context: SymbolContext) -> "Expr | None":
        # Electrical-engineering convention: `j` is unambiguously the
        # imaginary unit in this domain, so convert it everywhere (not only
        # in exponents).  Respect an explicit user declaration of `j`.
        if (expr.name == "j"
                and "electrical_engineering" in getattr(context, "domains", [])
                and context.get("j") is None):
            return Symbol(name="%i")
        return None

    def _imaginary_i_blocked(self, context: SymbolContext) -> bool:
        # Electrical engineering: `i` is electric current, NOT the imaginary
        # unit (that role is `j`).  So `i` is never imaginary here — not even
        # in an exponent — and `x^i` stays the i-th power, not `x^{√-1}`.
        return "electrical_engineering" in getattr(context, "domains", [])

    # ------------------------------------------------------------------
    # Subscript negation recovery (-k_B)
    # ------------------------------------------------------------------

    def _unwrap_subscript_negation(self, e: Expr) -> "tuple[bool, Expr]":
        """Return (is_negative, inner_expr) where inner_expr has any
        leading Mul(-1, ...) stripped — recovers the registered name from
        -k_B, which SharedParser parses as Subscript(base=Mul(-1, k), sub=B)."""
        if isinstance(e, Mul) and len(e.args) == 2:
            if isinstance(e.args[0], Number) and e.args[0].value == -1:
                return True, e.args[1]
            if isinstance(e.args[1], Number) and e.args[1].value == -1:
                return True, e.args[0]
        return False, e

    # ------------------------------------------------------------------
    # Index normalization
    # ------------------------------------------------------------------

    def _normalize_extracted_indices(self, idxs: list, is_up: bool) -> list:
        for i, idx in enumerate(idxs):
            if isinstance(idx, Symbol):
                idxs[i] = IndexSymbol(name=idx.name, is_up=is_up)
            elif isinstance(idx, Number):
                # Numeric subscripts like H_0, a_1 → IndexSymbol(name="0")
                idxs[i] = IndexSymbol(name=str(int(idx.value)), is_up=is_up)
            elif isinstance(idx, IndexSymbol):
                idx.is_up = is_up
            else:
                idxs[i] = self.index_from_compound(idx, is_up)
        return idxs
