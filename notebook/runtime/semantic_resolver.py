"""Base semantic resolver: shared Surface-AST → Semantic-IR resolution.

The resolution logic is structurally identical for Maxima and Julia — per the
Backend Encapsulation Rule (CLAUDE.md), the shared traversal lives HERE and
the subclasses override only the hook methods / class attributes that encode
a real backend difference:

  imaginary_unit_name           "%i" (Maxima) vs "im" (Julia)
  resolve_sum_integral_prefix   Maxima resolves Σ/∫ prefixes for Juxtaposition
                                AND Mul; Julia only checks Σ inside Mul
  known_function_names          per-backend builtin spellings (f (x) rule)
  _prime_call_is_quantum        which domains make f'(x) an adjoint operator
  _resolve_prime_call_derivative  the calculus interpretation of f'(x)
  _resolve_backend_builtin_call   the CAS's own sum/delta spelling, read back
  _resolve_declared_operator_call Julia converts operator-call args to :site
  _resolve_symbol_backend_special Maxima's EE convention: j → %i
  _imaginary_i_blocked          Maxima's EE convention: i is current, never √-1
  _unwrap_subscript_negation    Maxima recovers -k_B (Mul(-1,k)_B) names
  _resolve_prime_standalone_other Julia primes IndexedSymbol bases (a_k' → a'_k)
  _normalize_extracted_indices  per-backend index-node normalization
  _square_bracket_role          [a,b] is a commutator (Julia/QA) vs a list or
                                matrix literal (Maxima)

Anything backend-shaped that is NOT in this list is a regression, not a special
case: ``_resolve_bracketexpr`` hardcoded Julia's commutator reading here for
every backend, and a Maxima 2×2 Hamiltonian expanded to 0 with no error
(open_items AUDIT-BRACKET-BACKEND).  Add the hook, don't patch the call site.

Before 2026-07-04 the two resolvers were ~90% verbatim copies, and fixes
landed on one side only (the declared-Delta Laplacian guard existed in Julia
for weeks while Maxima kept the bug — audit finding SR-3/AR-1).
"""
from abc import ABC, abstractmethod
from contextlib import contextmanager
from notebook.models.ir import (
    Expr, Symbol, Number, Add, Mul, NonCommutativeMul, PrimeExpr, Pow,
    FunctionCall, Derivative, Integral, Commutator, Anticommutator,
    Equation, CasOperation, Juxtaposition, BracketExpr, Assignment,
    KroneckerDelta, DiracDelta, IndexSum, IndexSymbol, IndexedSymbol,
    Subscript, Superscript, DotProduct, CrossProduct,
    Gradient, Divergence, Curl, Laplacian,
)
from notebook.models.symbol_context import SymbolContext
from notebook.runtime import name_marks
from notebook.runtime.shared_parser import UnknownNotationError
from notebook.runtime.subscript_policy import (
    classify_called_subscript, declared_subscript_role,
    extremum_binder, read_subscript_name,
)
from notebook.runtime.sum_integral_prefix import (
    may_be_prefix, try_resolve_integral, try_resolve_limit, try_resolve_sum,
)

# Elementary functions whose names are IDENTICAL in Maxima and Julia user input.
# Used by the "f (x)" (space-before-paren) disambiguation: a callee in this set
# (or a backend-specific extension, or a declared callable) stays a function call;
# anything else with a space before the paren is a product ("a (x+1)" = a·(x+1)).
# Backend-specific names that DIFFER (Maxima ceiling/conjugate/realpart vs Julia
# ceil/conj/real) live in each resolver's ``known_function_names`` override, per
# the Backend Encapsulation Rule — NOT here.
COMMON_FUNCTION_NAMES: frozenset = frozenset({
    "sin", "cos", "tan", "cot", "sec", "csc",
    "asin", "acos", "atan", "arcsin", "arccos", "arctan",
    "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
    "exp", "log", "sqrt", "abs", "floor", "round", "gamma",
})
# SymbolContext kinds whose quantities are applied to arguments (a call), not
# multiplied: declared functions, operators, and field quantities.
CALLABLE_KINDS: frozenset = frozenset({
    "function", "operator", "vector_field", "scalar_field",
})


class BaseSemanticResolver(ABC):
    """Shared Surface AST → Semantic IR resolver; subclasses supply hooks only."""

    # Builtin function names recognized by THIS backend for the "f (x)"
    # space-disambiguation.  Subclasses override to add backend-specific names
    # (Maxima: ceiling/conjugate/…; Julia: ceil/conj/…).
    known_function_names: frozenset = COMMON_FUNCTION_NAMES

    # The backend's imaginary-unit token, produced for \mathrm{i}, x^i, EE j.
    # ANNOTATION ONLY — deliberately unassigned, so a subclass that forgets to
    # declare it fails loudly instead of silently emitting Maxima's "%i"
    # (open_items AUDIT-SHARED-CAS-LEAK: a shared default for a CAS token is a
    # leak).  MaximaSemanticResolver: "%i"; JuliaSemanticResolver: "im".
    imaginary_unit_name: str

    # Maxima resolves sum/integral prefixes for BOTH Juxtaposition and Mul up
    # front; Julia historically checks only sums, and only inside the Mul
    # branch.  Preserved exactly via this switch.
    resolve_sum_integral_prefix: bool = True

    # ------------------------------------------------------------------
    # Hooks — the ONLY places backend behavior may differ
    # ------------------------------------------------------------------

    @abstractmethod
    def _prime_call_is_quantum(self, curr: Expr, inner_func: Expr,
                               context: SymbolContext) -> bool:
        """True when f'(x) means an adjoint/creation operator, not a derivative."""

    @abstractmethod
    def _resolve_prime_call_derivative(self, inner_func: Expr, expr: FunctionCall,
                                       order: int, context: SymbolContext) -> Expr:
        """Resolve f'(x) in the calculus interpretation (backend-specific rigor)."""

    @abstractmethod
    def _normalize_extracted_indices(self, idxs: list, is_up: bool) -> list:
        """Convert raw extracted index nodes into IndexSymbols (backend rules)."""

    #: 代数的な添字の読み。印の族と同じ module に置いてある（`name_marks`）。
    index_from_compound = staticmethod(name_marks.index_from_compound)

    @abstractmethod
    def _square_bracket_role(self, expr: BracketExpr,
                             context: SymbolContext) -> str:
        """``"commutator"`` or ``"list"`` — what ``[a, b]`` means to this backend.

        There is no shared answer and therefore no default: ``[A,B]`` is a
        commutator in Julia/QuantumAlgebra and a list/matrix literal in Maxima.
        A new backend must state its convention (open_items
        AUDIT-BRACKET-BACKEND: this was hardcoded to ``"commutator"`` here, so
        Maxima matrices silently expanded to zero).
        """

    def _resolve_backend_builtin_call(self, expr: FunctionCall,
                                      context: SymbolContext) -> "Expr | None":
        """The CAS's own spelling of a structure the IR has a node for.

        Default: none.  Maxima reads back its `sum(body, i, lo, hi)` and
        `kron_delta(a, b)` — the two it writes itself for `IndexSum` and
        `KroneckerDelta`.
        """
        return None

    def _resolve_declared_operator_call(self, expr: FunctionCall,
                                        context: SymbolContext) -> "Expr | None":
        """Backend-specific handling of calls on declared quantum operators.
        Default: none (Maxima).  Julia converts symbol args to :site form."""
        return None

    def _resolve_symbol_backend_special(self, expr: Symbol,
                                        context: SymbolContext) -> "Expr | None":
        """First-chance backend-specific symbol resolution (Maxima: EE j → %i)."""
        return None

    def _imaginary_i_blocked(self, context: SymbolContext) -> bool:
        """True when a bare ``i`` must NEVER become the imaginary unit
        (Maxima EE convention: i is electric current)."""
        return False

    def _unwrap_subscript_negation(self, e: Expr) -> "tuple[bool, Expr]":
        """Strip a leading Mul(-1, …) so -k_B recovers the name k_B.
        Default: no-op (Julia).  Maxima overrides."""
        return False, e

    def _resolve_prime_standalone_other(self, inner_resolved: Expr, order: int,
                                        context: SymbolContext) -> Expr:
        """素の記号でないものに付いたプライム。読みは `name_marks`；Julia は上書き
        する（a_k' → a'_k）。"""
        return name_marks.prime_on_non_symbol(inner_resolved, order)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def callee_is_function(self, func: Expr, context: SymbolContext) -> bool:
        """True when *func* names something applied to arguments (a call), not a
        scalar to be multiplied.  A builtin CAS function name (this backend's
        ``known_function_names``), or a SymbolContext quantity declared with a
        callable kind, counts as a function.  Keeps the ``f (x)`` product-vs-call
        decision uniform in structure while honouring per-backend function tables.
        """
        if not isinstance(func, Symbol):
            return False  # e.g. a Number callee ("2 (x+1)") is never a function
        if func.name in self.known_function_names:
            return True
        prop = context.get(func.name)
        return prop is not None and prop.quantity.kind in CALLABLE_KINDS

    #: Greek index NAMES that must survive as one index whatever the splitting
    #: rule is (``\mu`` is not ``m``·``u``).
    _GREEK_INDEX_NAMES: frozenset = frozenset({
        "mu", "nu", "alpha", "beta", "gamma", "delta", "epsilon", "zeta",
        "eta", "theta", "iota", "kappa", "lambda", "rho", "sigma", "tau",
        "upsilon", "phi", "chi", "psi", "omega",
    })

    @classmethod
    def _flatten_juxtaposition(cls, node: Expr) -> "list[Expr]":
        """Space-separated pieces of *node*, or ``[node]`` if it is not one.

        Inside a subscript a space separates INDICES (``A_{i j}``); the parser
        reports that as a nested Juxtaposition, which has to be unwrapped before
        the pieces can be read as indices.
        """
        if isinstance(node, Juxtaposition):
            return (cls._flatten_juxtaposition(node.left)
                    + cls._flatten_juxtaposition(node.right))
        return [node]

    def _parse_indices_from_symbol(self, sym: Symbol,
                                   context: "SymbolContext | None" = None
                                   ) -> "list[IndexSymbol]":
        """Split a subscript token into indices.

        **Indices must be separated by a comma or a space** (open_items
        AUDIT-SUBSCRIPT-SPLIT).  ``A_{i,j}`` and ``A_{i j}`` are two indices;
        ``E_{rot}`` is ONE.  This used to split any lowercase alphabetic run per
        character, which is the auto-split that CLAUDE.md's Math Input
        Conventions already forbid for expression bodies, and for the same
        reason — measured silently wrong: ``E_{rot}`` → ``E(r, o, t)``,
        ``T_{kin}`` → ``T(k, i, n)``, ``c_{k,up}`` → ``c(k, u, p)`` (spin up
        became u·p), ``bessel_jn`` → ``bessel(j, n)``.  The old discriminator was
        letter CASE — ``E_{HF}`` survived, ``E_{rot}`` did not — and was
        documented nowhere.

        The one exception is an einstein-summation session, whose entire premise
        is that a repeated index is implicit: ``A_{ij}B_{jk}`` is how that
        notation is written, and the mode is opted into per session.  Outside it,
        write the separator.
        """
        name = sym.name
        if name.lower() in self._GREEK_INDEX_NAMES:
            return [IndexSymbol(name=name)]
        if (len(name) > 1 and name.isalpha() and name.islower()
                and context is not None and context.einstein_summation):
            return [IndexSymbol(name=char) for char in name]
        return [IndexSymbol(name=name)]

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def resolve(self, expr: Expr, context: SymbolContext,
                skip_alphatize: bool = False,
                before_alphatize=None) -> Expr:
        r"""Surface AST → semantic AST.

        *before_alphatize* is handed the resolved IR while the dummy indices are
        still spelled the way the RESEARCHER wrote them, and may edit it in
        place.  Why anything needs that: notebook/runtime/placeholder_sum_bounds.
        """
        resolved = self._resolve_node(expr, context)
        if context.einstein_summation:
            from notebook.runtime.auto_einstein import auto_einstein_sum
            resolved = auto_einstein_sum(resolved)

        from notebook.runtime.index_contractor import contract_indices, alphatize_indices
        resolved = contract_indices(resolved, context)
        if before_alphatize is not None:
            before_alphatize(resolved)
        if not skip_alphatize:
            resolved = alphatize_indices(resolved)

        self._reject_bare_function_names(resolved)
        return resolved

    #: Whether a DECLARED base makes its subscript an index.  Backend-owned,
    #: because what counts as "an indexed quantity" differs: Maxima reads a
    #: matrix/vector/tensor that way, while a QuantumAlgebra operator takes
    #: indices by construction (CLAUDE.md, julia_index_policy rule 3).  The
    #: shared default stays `declared_subscript_role`.
    @staticmethod
    def _subscript_base_role(name: str, context) -> "str | None":
        return declared_subscript_role(name, context)

    #: Names this backend writes as FUNCTIONS.  Set by each subclass from its
    #: own converter table (Backend Encapsulation Rule: shared logic here,
    #: parameterised by a class attribute).  Empty means "do not check".
    _FUNCTION_NAMES: "frozenset[str]" = frozenset()

    def _reject_bare_function_names(self, ir) -> None:
        r"""A written function must be APPLIED, never left as a value.

        Measured 2026-08-17, two spellings left the function's name standing as
        an ordinary symbol, and Maxima accepts that without complaint:

            \sin^{-1} x  → sin^(-1)*x       \sin_2 x → sin(2)*x

        Both are well-formed CAS expressions meaning something else, which is
        the failure mode this project refuses on sight.  `\sin^2 x` used to be
        a third; it has a conventional reading and is now READ
        (`read_power_application`), so what is left here is only what the
        notation genuinely does not decide.

        Checked once over the finished IR rather than in each resolver branch:
        the two shapes are `Pow(base=Symbol)` and `IndexedSymbol(base=Symbol)`,
        and a walk that names them both in one place cannot drift apart.  An
        APPLIED function is a `FunctionCall`, so `sin(x)` and `sin(x)^2` are
        untouched, and a researcher's own `f_2` never matches because `f` is
        not one of these names.
        """
        if not self._FUNCTION_NAMES:
            return
        from notebook.models.ir import IndexedSymbol, Pow, Symbol
        from notebook.runtime.function_application import bare_function_reason
        from notebook.runtime.shared_parser import UnknownNotationError

        def visit(node) -> None:
            base = getattr(node, "base", None)
            if isinstance(node, (Pow, IndexedSymbol)) and isinstance(base, Symbol) \
                    and base.name in self._FUNCTION_NAMES:
                deco = "^" if isinstance(node, Pow) else "_"
                _exp = getattr(getattr(node, "exponent", None), "value", None)
                raise UnknownNotationError(
                    bare_function_reason(base.name, deco, _exp),
                    name=base.name)
            for child in getattr(node, "__dict__", {}).values():
                if isinstance(child, list):
                    for c in child:
                        if hasattr(c, "__dict__"):
                            visit(c)
                elif hasattr(child, "__dict__"):
                    visit(child)

        visit(ir)

    # ------------------------------------------------------------------
    # Shared node resolution
    # ------------------------------------------------------------------

    def _resolve_node(self, expr: Expr, context: SymbolContext) -> Expr:

        if isinstance(expr, Mul) and expr.args:
            first = expr.args[0]
            if isinstance(first, Symbol):
                if first.name in ("nabla", "grad", "gradient"):
                    if len(expr.args) == 2:
                        rest = self._resolve_node(expr.args[1], context)
                    else:
                        rest = Mul(args=[self._resolve_node(a, context) for a in expr.args[1:]])
                    return Gradient(expr=rest)
                elif first.name in ("div", "divergence"):
                    if len(expr.args) == 2:
                        rest = self._resolve_node(expr.args[1], context)
                    else:
                        rest = Mul(args=[self._resolve_node(a, context) for a in expr.args[1:]])
                    return Divergence(expr=rest)
                elif first.name in ("rot", "curl"):
                    if len(expr.args) == 2:
                        rest = self._resolve_node(expr.args[1], context)
                    else:
                        rest = Mul(args=[self._resolve_node(a, context) for a in expr.args[1:]])
                    return Curl(expr=rest)
                elif first.name in ("Delta", "laplacian", "Laplacian"):
                    # "laplacian"/"Laplacian" (explicit function names) always → ∇².
                    # "Delta" → ∇² only when NOT registered in SymbolContext;
                    # if registered (e.g. BCS order parameter Δ, finite
                    # difference), the user's declaration wins and this is
                    # symbol multiplication.
                    _is_laplacian = (
                        first.name in ("laplacian", "Laplacian")
                        or (first.name == "Delta"
                            and (context is None or context.get("Delta") is None))
                    )
                    if _is_laplacian:
                        if len(expr.args) == 2:
                            rest = self._resolve_node(expr.args[1], context)
                        else:
                            rest = Mul(args=[self._resolve_node(a, context) for a in expr.args[1:]])
                        return Laplacian(expr=rest)

            if isinstance(first, Pow) and isinstance(first.base, Symbol) and first.base.name == "nabla" and isinstance(first.exponent, Number) and first.exponent.value == 2:
                if len(expr.args) == 2:
                    rest = self._resolve_node(expr.args[1], context)
                else:
                    rest = Mul(args=[self._resolve_node(a, context) for a in expr.args[1:]])
                return Laplacian(expr=rest)

        if self.resolve_sum_integral_prefix and may_be_prefix(expr):
            sum_res = try_resolve_sum(expr, self._resolve_node, context, self._binding)
            if sum_res:
                return sum_res

            int_res = try_resolve_integral(expr, self._resolve_node, context)
            if int_res:
                return int_res

            lim_res = try_resolve_limit(expr, self._resolve_node, context)
            if lim_res:
                return lim_res

        handler = self._RESOLVE_DISPATCH.get(type(expr))
        if handler is not None:
            return handler(self, expr, context)
        # Leaf nodes (Number, Bra, Ket, OpaqueExpr) are returned as is
        return expr

    # ------------------------------------------------------------------
    # R4/所見C (docs/27): per-node-type resolvers, extracted verbatim from the
    # former 622-line if/elif chain.  _resolve_node keeps the two pre-dispatch
    # guards (vector-op Mul, sum/integral prefix) inline, then dispatches by
    # type through _RESOLVE_DISPATCH.  Each handler keeps its isinstance guard so
    # its body is byte-for-byte the original branch (no re-indentation); the
    # guard is redundant under the table but makes each resolver self-contained.
    # The IR node types have no subclass relationships, so type() dispatch is
    # equivalent to the original isinstance chain.  (subscript→IndexedSymbol
    # constraint lives in _resolve_subscript, unchanged.)
    # ------------------------------------------------------------------
    def _resolve_juxtaposition(self, expr, context):
        if isinstance(expr, Juxtaposition):
            # Check if right is a curly BracketExpr (e.g. subscript braces like _{i} or _{i,j} or _{ij})
            if isinstance(expr.right, BracketExpr) and expr.right.kind == "curly":
                base_res = self._resolve_node(expr.left, context)
                indices = []
                for arg in expr.right.args:
                    arg_res = self._resolve_node(arg, context)
                    if isinstance(arg_res, Symbol):
                        indices.extend(self._parse_indices_from_symbol(arg_res, context))
                    elif isinstance(arg_res, IndexSymbol):
                        indices.append(arg_res)
                    else:
                        raise ValueError(f"IndexedSymbol index must resolve to a Symbol or IndexSymbol, got {type(arg_res)}")
                if isinstance(base_res, Symbol) and base_res.name in ("mathrm", "text"):
                    if len(indices) == 1 and indices[0].name == "i":
                        return Symbol(name=self.imaginary_unit_name)
                return IndexedSymbol(base=base_res, indices=indices)

            # R-9: display-only bracket functional.  E[X] → expectation(X),
            # F[X] → F(X), Var[X] → variance(X) …  A square bracket applied to a
            # symbol declared a bracket functional (or E in a statistics domain via
            # the notation profile) is a functional application — NOT a commutator.
            # Display keeps `Sym[args]`; the CAS gets `cas_name(args)` with no
            # operator algebra (the symbol is never registered as an operator).
            if isinstance(expr.right, BracketExpr) and expr.right.kind == "square":
                fn = self._bracket_functional_name(expr.left, context)
                if fn is not None:
                    f_args = [self._resolve_node(a, context) for a in expr.right.args]
                    return FunctionCall(func=Symbol(name=fn), args=f_args)
                # R-9: matrix element access A[i,j] when the base is a declared
                # matrix / tensor / vector → IndexedSymbol (the code generator
                # renders it as A[i,j] via array_access).  Undeclared A[i,j] is
                # left to the generic path (R-3 explicit error / future candidate
                # dialog), per the declaration-driven (A) policy.
                base_el = self._resolve_node(expr.left, context)
                if (isinstance(base_el, Symbol)
                        and get_algebra_type(base_el, context) in ("matrix", "tensor", "vector")):
                    el_indices: list = []
                    for a in expr.right.args:
                        a_res = self._resolve_node(a, context)
                        if isinstance(a_res, Symbol):
                            el_indices.extend(self._parse_indices_from_symbol(a_res, context))
                        elif isinstance(a_res, IndexSymbol):
                            el_indices.append(a_res)
                        elif isinstance(a_res, Number):
                            el_indices.append(IndexSymbol(name=str(a_res.value)))
                        else:
                            el_indices = []
                            break
                    if el_indices:
                        return IndexedSymbol(base=base_el, indices=el_indices)

            left_res = self._resolve_node(expr.left, context)
            if isinstance(left_res, Symbol) and left_res.name == "nabla":
                right_res = self._resolve_node(expr.right, context)
                return Gradient(expr=right_res)
            if isinstance(left_res, Symbol) and left_res.name == "Delta":
                # "Delta f" → Laplacian only when Delta is not a registered
                # physics symbol (see the Mul branch above).
                if context is None or context.get("Delta") is None:
                    right_res = self._resolve_node(expr.right, context)
                    return Laplacian(expr=right_res)
            if isinstance(left_res, Pow):
                if isinstance(left_res.base, Symbol) and left_res.base.name == "nabla":
                    if isinstance(left_res.exponent, Number) and left_res.exponent.value == 2:
                        right_res = self._resolve_node(expr.right, context)
                        return Laplacian(expr=right_res)

            right_res = self._resolve_node(expr.right, context)

            # Non-commutative check
            if is_noncommutative(left_res, context) or is_noncommutative(right_res, context):
                args = []
                for item in [left_res, right_res]:
                    if isinstance(item, (Mul, NonCommutativeMul)):
                        args.extend(item.args)
                    else:
                        args.append(item)
                return NonCommutativeMul(args=args)
            else:
                # Flatten Mul if either side is already a Mul
                args = []
                for item in [left_res, right_res]:
                    if isinstance(item, Mul):
                        args.extend(item.args)
                    else:
                        args.append(item)
                return Mul(args=args)
        return None

    def _resolve_bracket_as_list(self, expr, context):
        """Resolve a square bracket, and every square bracket NESTED in it, as a
        list — never as a commutator.

        Used once a backend has decided this bracket is a list: the decision has
        to propagate downward, because the elements of a list are list elements
        whatever they look like.  Without the recursion the rows of
        ``[[epsilon, Delta],[Delta, -epsilon]]`` would each still resolve to a
        commutator and the 2×2 matrix would collapse to ``[0, 0]``.
        """
        args = []
        for arg in expr.args:
            if isinstance(arg, BracketExpr) and arg.kind == "square":
                args.append(self._resolve_bracket_as_list(arg, context))
            else:
                args.append(self._resolve_node(arg, context))
        return BracketExpr(args=args, kind="square")

    def _resolve_bracketexpr(self, expr, context):
        if isinstance(expr, BracketExpr):
            # What ``[a, b]`` MEANS is a per-backend convention, not a shared
            # rule: a commutator in Julia/QuantumAlgebra, a list/matrix literal
            # in Maxima.  Deciding it here would be a Backend Encapsulation
            # violation — and it was one: this method returned Commutator
            # unconditionally, so a Maxima session's 2×2 Hamiltonian expanded to
            # 0 with no error (open_items AUDIT-BRACKET-BACKEND).
            if expr.kind == "square":
                if self._square_bracket_role(expr, context) == "list":
                    return self._resolve_bracket_as_list(expr, context)
                if len(expr.args) == 2:
                    return Commutator(
                        a=self._resolve_node(expr.args[0], context),
                        b=self._resolve_node(expr.args[1], context))
            elif expr.kind == "curly" and len(expr.args) == 2:
                return Anticommutator(
                    a=self._resolve_node(expr.args[0], context),
                    b=self._resolve_node(expr.args[1], context))

            resolved_args = [self._resolve_node(arg, context) for arg in expr.args]
            return BracketExpr(args=resolved_args, kind=expr.kind)
        return None

    def _resolve_assignment(self, expr, context):
        if isinstance(expr, Assignment):
            lhs_res = self._resolve_node(expr.lhs, context)
            rhs_res = self._resolve_node(expr.rhs, context)
            return Assignment(
                lhs=lhs_res,
                rhs=rhs_res,
                is_operator_def=expr.is_operator_def,
                operator_kind=expr.operator_kind
            )
        return None

    def _resolve_add(self, expr, context):
        if isinstance(expr, Add):
            return Add(args=[self._resolve_node(arg, context) for arg in expr.args])
        return None

    def _resolve_mul(self, expr, context):
        if isinstance(expr, Mul):
            if not self.resolve_sum_integral_prefix:
                resolved_sum = try_resolve_sum(expr, self._resolve_node, context, self._binding)
                if resolved_sum is not None:
                    return resolved_sum

            resolved_args = []
            has_nc = False
            for arg in expr.args:
                res_arg = self._resolve_node(arg, context)
                if is_noncommutative(res_arg, context):
                    has_nc = True

                if isinstance(res_arg, (Mul, NonCommutativeMul)):
                    resolved_args.extend(res_arg.args)
                else:
                    resolved_args.append(res_arg)

            if has_nc:
                return NonCommutativeMul(args=resolved_args)
            return Mul(args=resolved_args)
        return None

    def _resolve_dotproduct(self, expr, context):
        if isinstance(expr, DotProduct):
            left_res = self._resolve_node(expr.left, context)
            right_res = self._resolve_node(expr.right, context)
            if isinstance(left_res, Symbol) and left_res.name == "nabla":
                return Divergence(expr=right_res)
            return DotProduct(left=left_res, right=right_res)
        return None

    def _resolve_crossproduct(self, expr, context):
        if isinstance(expr, CrossProduct):
            left_res = self._resolve_node(expr.left, context)
            right_res = self._resolve_node(expr.right, context)
            if isinstance(left_res, Symbol) and left_res.name == "nabla":
                return Curl(expr=right_res)
            return CrossProduct(left=left_res, right=right_res)
        return None

    def _resolve_noncommutativemul(self, expr, context):
        if isinstance(expr, NonCommutativeMul):
            resolved_args = []
            for arg in expr.args:
                res_arg = self._resolve_node(arg, context)
                if isinstance(res_arg, (Mul, NonCommutativeMul)):
                    resolved_args.extend(res_arg.args)
                else:
                    resolved_args.append(res_arg)
            return NonCommutativeMul(args=resolved_args)
        return None

    #: 印の読みは `name_marks` に一本化してある（共役・随伴・プライム・代数添字は
    #: 「名前に付いた印が何を意味するか」という一つの問いで、resolver の中に散って
    #: いた）。ここに残るのは**ノードの構築**と、バックエンドごとの既定だけ。
    _script_mark = staticmethod(name_marks.script_mark)

    #: 未宣言の基底の ``X^\dagger`` を随伴と読むか。True が共有の既定（Julia:
    #: QuantumAlgebra / Symbolics は任意の式に随伴を持つ）。Maxima は False に
    #: 上書きする —— 演算子代数が無いので、未宣言の記号の随伴は名前である。
    #: Backend Encapsulation Rule: その差はバックエンドの側に置く。
    _DAGGER_DEFAULT_IS_ADJOINT = True

    def _dagger_reading(self, base_surface, context):
        r"""``X^\dagger`` は随伴か、装飾された名前か。読みは `name_marks`。

        解決は resolver にしかできないので callback で渡し、**高々1回**に留める
        （プライム付きの名前は解決時に登録簿へ登録されるので、二度解決すると
        二重登録になる）。
        """
        _resolved: list = []

        def _base():
            if not _resolved:
                _resolved.append(self._resolve_node(base_surface, context))
            return _resolved[0]

        kind, node = name_marks.dagger_reading(
            base_surface, lambda: get_algebra_type(_base(), context),
            default_is_adjoint=self._DAGGER_DEFAULT_IS_ADJOINT)
        return (kind, _base()) if kind == "adjoint" else (kind, node)

    def _resolve_applied_script(self, expr, context):
        r"""``\psi^*(x)`` → ``Conjugate(psi(x))``.

        The parser reads a superscript followed by an argument list as an
        exponent that SWALLOWED the list — `\psi^*(x)` arrives as
        ``Superscript(psi, script=FunctionCall(func={ast}, args=[x]))``.  Left
        alone it lowered to ``psi^[ast](x)``: a symbol raised to a LIST,
        displayed as ``\psi^{\left[ ast \right](x)}`` — the conjugate of a wave
        function, which is how it is written whenever a magnetic field is
        present, reaching the CAS as nonsense (measured 2026-08-09).

        Only the conjugate marks are read this way.  ``f^{2}(x)`` is a genuine
        ambiguity (the square of the value, or the second iterate) and is left
        exactly as it was — this is not the place to settle it.
        """
        script = getattr(expr, "script", None)
        if not isinstance(script, FunctionCall) or getattr(script, "spaced", False):
            return None
        _mark = name_marks.script_mark(script.func)
        if not name_marks.is_mark(_mark):
            return None
        from notebook.models.ir import Conjugate, Adjoint
        # Same rule as `_resolve_conjugate_then_subscript`: hand over the node
        # this would have been WITHOUT the star.  For a subscripted base that
        # means re-swallowing the argument list into the subscript — the shape
        # `\psi_n(x)` itself arrives in — so the two spellings cannot disagree
        # about whether `n` is an index or part of the name.
        base = expr.base
        _wrap = Conjugate
        if name_marks.is_dagger(_mark):
            # `\psi^\dagger(x)`.  For the NAME reading the mark belongs to the
            # base, not around the call: `psi_dagger(x)`, not `dagger(psi(x))`
            # — there is no such CAS function to wrap it in.
            _kind, _node = self._dagger_reading(base, context)
            if _kind == "adjoint":
                _wrap = Adjoint
            else:
                _wrap, base = (lambda expr: expr), _node
        if isinstance(base, Subscript):
            inner = Subscript(base=base.base,
                              sub=FunctionCall(func=base.sub,
                                               args=list(script.args), spaced=False),
                              bracketed=base.bracketed)
        else:
            inner = FunctionCall(func=base, args=list(script.args), spaced=False)
        return _wrap(expr=self._resolve_node(inner, context))

    def _resolve_conjugate_then_subscript(self, expr, context):
        r"""``\varphi^*_{m,s}(x)`` → ``Conjugate(varphi[m,s](x))``.

        The mirror of `_resolve_applied_script` on the subscript side: here it
        is the SUBSCRIPT that swallowed the argument list, so
        ``\psi^*_n(x)`` arrives as
        ``Subscript(base=Superscript(psi, ^*), sub=FunctionCall(n, [x]))``.

        This is the star-in-front spelling every physics text uses for a matrix
        element, and it was refused — with a message about MULTIPLE indices,
        which `\psi^*_n(x)` does not even have: the star was being counted as
        one of them (measured 2026-08-09).  The rebuilt call goes back through
        `_resolve_subscript`, so whether the indices fold into a name or become
        `varphi[m,s]` stays that method's decision, not a second copy of it.
        """
        sub, base = expr.sub, expr.base
        if not isinstance(sub, FunctionCall) or getattr(sub, "spaced", False):
            return None
        if not isinstance(base, Superscript):
            return None
        _mark = name_marks.script_mark(base.script)
        if not name_marks.is_mark(_mark):
            return None
        from notebook.models.ir import Conjugate, Adjoint
        _wrap, _inner_base = Conjugate, base.base
        if name_marks.is_dagger(_mark):
            _kind, _node = self._dagger_reading(base.base, context)
            if _kind == "adjoint":
                _wrap = Adjoint
            else:
                _wrap, _inner_base = (lambda expr: expr), _node
        # Hand over the node this WOULD have been without the star, untouched.
        # Rebuilding the call here instead made the conjugate disagree with the
        # plain form about a SINGLE index — `\psi_n(x)` folded to the name
        # `psi_n(x)` while `\psi^*_n(x)` became `psi[n](x)`, so the two halves
        # of `\varphi^*_n(x)\varphi_n(x)` were different CAS objects and the
        # product would never pair up (measured 2026-08-09).  Which of the two
        # a single subscript is stays `subscript_policy`'s question.
        return _wrap(
            expr=self._resolve_node(
                Subscript(base=_inner_base, sub=sub, bracketed=expr.bracketed),
                context))

    def _resolve_pow(self, expr, context):
        if isinstance(expr, Pow):
            self._in_exponent = True
            try:
                exponent_res = self._resolve_node(expr.exponent, context)
            finally:
                self._in_exponent = False

            base_res = self._resolve_node(expr.base, context)

            # Adjoint, Conjugate, Transpose disambiguation based on MathType
            from notebook.models.ir import Transpose, Conjugate, Adjoint

            exp_sym = name_marks.script_mark(exponent_res)

            if exp_sym is not None:
                if exp_sym in ("T", "mathrm", "text"): # Note: \mathrm{T} may resolve differently, but normally T is parsed
                    # If it's explicitly T
                    alg = get_algebra_type(base_res, context)
                    if alg in ("matrix", "vector", "tensor"):
                        return Transpose(expr=base_res)
                elif exp_sym in name_marks.CONJUGATE_MARKS:
                    return Conjugate(expr=base_res)
                elif name_marks.is_dagger(exp_sym):
                    _kind, _node = self._dagger_reading(expr.base, context)
                    return (Adjoint(expr=_node) if _kind == "adjoint"
                            else self._resolve_node(_node, context))

            return Pow(base=base_res, exponent=exponent_res)
        return None

    @staticmethod
    def _negated_operand(node):
        """``-b`` → ``b``; anything else → None.  A subtraction arrives as
        ``Add(a, Mul(-1, b))``."""
        if (isinstance(node, Mul) and len(node.args) == 2
                and isinstance(node.args[0], Number) and node.args[0].value == -1):
            return node.args[1]
        return None

    def _resolve_functioncall(self, expr, context):
        if isinstance(expr, FunctionCall):
            # Check if func is a PrimeExpr (e.g. f'(x) or a'(i))
            if isinstance(expr.func, PrimeExpr):
                # Count the order of primes
                order = 0
                curr = expr.func
                while isinstance(curr, PrimeExpr):
                    order += 1
                    curr = curr.expr

                # Resolve the inner function expression
                inner_func = self._resolve_node(curr, context)

                if self._prime_call_is_quantum(curr, inner_func, context):
                    # In quantum domain, prime represents Dagger / adjoint.
                    # We map this to a symbol name with prime(s) e.g. a'
                    if isinstance(inner_func, Symbol):
                        op_name = inner_func.name + ("'" * order)
                        # We also check if it is registered as an operator, to keep statistics
                        prop = context.get(inner_func.name)
                        if prop is not None:
                            context.register(op_name, prop.quantity, prop.statistics.kind)

                        resolved_args = [self._resolve_node(arg, context) for arg in expr.args]
                        return FunctionCall(func=Symbol(name=op_name), args=resolved_args)
                else:
                    # In calculus/standard domain, prime represents a derivative.
                    return self._resolve_prime_call_derivative(inner_func, expr, order, context)

            if isinstance(expr.func, Symbol):
                func_name = expr.func.name

                # matrix(...) rows: each bracketed arg is a ROW (list), never a
                # commutator.  Without this a 2-element row [a, b] would resolve
                # to Commutator(a, b) (a*b - b*a), collapsing 2×2 matrices to
                # matrix(0, 0).  Rows of other lengths already resolve as lists.
                if func_name == "matrix":
                    rows = [
                        self._resolve_bracket_as_list(_row, context)
                        if isinstance(_row, BracketExpr)
                        else self._resolve_node(_row, context)
                        for _row in expr.args
                    ]
                    return FunctionCall(func=Symbol(name="matrix"), args=rows)

                # Vector calculus function aliases
                if func_name in ("div", "divergence"):
                    if len(expr.args) != 1:
                        raise ValueError(f"{func_name} requires 1 argument")
                    return Divergence(expr=self._resolve_node(expr.args[0], context))
                elif func_name in ("rot", "curl"):
                    if len(expr.args) != 1:
                        raise ValueError(f"{func_name} requires 1 argument")
                    return Curl(expr=self._resolve_node(expr.args[0], context))
                elif func_name in ("grad", "gradient"):
                    if len(expr.args) != 1:
                        raise ValueError(f"{func_name} requires 1 argument")
                    return Gradient(expr=self._resolve_node(expr.args[0], context))
                elif func_name in ("laplacian", "Laplacian"):
                    if len(expr.args) != 1:
                        raise ValueError(f"{func_name} requires 1 argument")
                    return Laplacian(expr=self._resolve_node(expr.args[0], context))

                if func_name in ("delta", "KroneckerDelta"):
                    if func_name == "delta" and len(expr.args) == 1:
                        # `\delta(\mathbf r - \mathbf r')` — the DIRAC delta.
                        # Physics writes the two apart: a SUBSCRIPT is the
                        # Kronecker delta (`\delta_{ij}`), an ARGUMENT LIST is
                        # Dirac.  No guess is involved, and the reading was
                        # missing entirely: the argument form fell into the
                        # Kronecker branch and answered the researcher with
                        # `KroneckerDelta requires exactly 2 arguments, got 1`
                        # — internal wording for a completely ordinary line
                        # (the equal-time anticommutator, the completeness
                        # relation, the equation of motion; measured
                        # 2026-08-20).  Maxima has `delta` and its `laplace`
                        # applies the sifting property (`laplace(delta(t-3),t,s)`
                        # = `%e^-(3*s)`, measured), so this is not a dead end.
                        _arg = self._resolve_node(expr.args[0], context)
                        _x, _y = _arg, Number(value=0)
                        if isinstance(_arg, Add) and len(_arg.args) == 2:
                            # δ(a − b): keep the two points apart in the IR so
                            # the pair is visible to anything that looks.
                            _a, _b = _arg.args
                            _neg = self._negated_operand(_b)
                            if _neg is not None:
                                _x, _y = _a, _neg
                        return DiracDelta(x=_x, y=_y)
                    if len(expr.args) != 2:
                        raise ValueError(f"KroneckerDelta requires exactly 2 arguments, got {len(expr.args)}")
                    left_res = self._resolve_node(expr.args[0], context)
                    right_res = self._resolve_node(expr.args[1], context)
                    if isinstance(left_res, Symbol):
                        left_res = IndexSymbol(name=left_res.name)
                    if isinstance(right_res, Symbol):
                        right_res = IndexSymbol(name=right_res.name)
                    return KroneckerDelta(
                        left=left_res,
                        right=right_res
                    )
                elif func_name == "DiracDelta":
                    if len(expr.args) != 2:
                        raise ValueError(f"DiracDelta requires exactly 2 arguments, got {len(expr.args)}")
                    return DiracDelta(
                        x=self._resolve_node(expr.args[0], context),
                        y=self._resolve_node(expr.args[1], context)
                    )
                elif func_name == "Integral":
                    if len(expr.args) < 2 or len(expr.args) > 4:
                        raise ValueError(f"Integral requires 2 to 4 arguments, got {len(expr.args)}")
                    body = self._resolve_node(expr.args[0], context)
                    var_node = self._resolve_node(expr.args[1], context)
                    if not isinstance(var_node, Symbol):
                        raise ValueError("Integral variable must be a symbol")
                    lower_node = self._resolve_node(expr.args[2], context) if len(expr.args) > 2 else None
                    upper_node = self._resolve_node(expr.args[3], context) if len(expr.args) > 3 else None
                    return Integral(expr=body, var=var_node, lower=lower_node, upper=upper_node)

                elif func_name == "IndexSum":
                    if len(expr.args) != 2:
                        raise ValueError(f"IndexSum requires exactly 2 arguments, got {len(expr.args)}")
                    # The dummy is read BEFORE the body, so that a subscript in
                    # the body naming it resolves as an INDEX (docs/33 §9.1).
                    dummy_node = expr.args[1]
                    dummy_indices = []
                    if isinstance(dummy_node, Symbol):
                        dummy_indices.extend(self._parse_indices_from_symbol(dummy_node, context))
                    elif isinstance(dummy_node, BracketExpr) and dummy_node.kind == "square":
                        for arg in dummy_node.args:
                            resolved_arg = self._resolve_node(arg, context)
                            if isinstance(resolved_arg, Symbol):
                                dummy_indices.extend(self._parse_indices_from_symbol(resolved_arg, context))
                            elif isinstance(resolved_arg, IndexSymbol):
                                dummy_indices.append(resolved_arg)
                    else:
                        resolved_dummy = self._resolve_node(dummy_node, context)
                        if isinstance(resolved_dummy, Symbol):
                            dummy_indices.extend(self._parse_indices_from_symbol(resolved_dummy, context))
                        elif isinstance(resolved_dummy, IndexSymbol):
                            dummy_indices.append(resolved_dummy)
                        else:
                            raise ValueError("IndexSum summation variables must be symbols")
                    with self._binding(self._index_names(dummy_indices)):
                        body = self._resolve_node(expr.args[0], context)
                    return IndexSum(body=body, dummy_indices=dummy_indices)
                elif func_name in ("integrate", "Integral"):
                    if len(expr.args) not in (2, 4):
                        raise ValueError(f"Integral requires 2 or 4 arguments, got {len(expr.args)}")
                    body = self._resolve_node(expr.args[0], context)
                    var_node = self._resolve_node(expr.args[1], context)
                    if not isinstance(var_node, Symbol):
                        raise ValueError("Integral variable must resolve to a Symbol")
                    lower = self._resolve_node(expr.args[2], context) if len(expr.args) == 4 else None
                    upper = self._resolve_node(expr.args[3], context) if len(expr.args) == 4 else None
                    return Integral(expr=body, var=var_node, lower=lower, upper=upper)

            # Backend-specific: the CAS's OWN spelling of a structure this IR
            # has a node for (`\'sum(...)`, `kron_delta(...)`).  Reading it back
            # is what makes a RESULT the same kind of object as the input it
            # came from — without it, the index machinery (contraction, ranges,
            # alpha-conversion) sees a result as an ordinary function call and
            # stops applying (measured 2026-08-21: `\sum_{n m}\delta_{nm}A_{n,m}`
            # contracts, and the CAS's own `\'sum(\'sum(kron_delta(m,n)*A(n,m),
            # m,…),n,…)` — this system's output for the same quantity — does not).
            builtin = self._resolve_backend_builtin_call(expr, context)
            if builtin is not None:
                return builtin

            # Backend-specific: calls on declared quantum operators (Julia
            # converts symbol args to :site form).  Default no-op for Maxima.
            op_call = self._resolve_declared_operator_call(expr, context)
            if op_call is not None:
                return op_call

            if self._spaced_paren_is_a_product(expr.func, expr.args,
                                               getattr(expr, "spaced", False),
                                               context):
                return Mul(args=[
                    self._resolve_node(expr.func, context),
                    self._resolve_node(expr.args[0], context),
                ])

            return FunctionCall(
                func=self._resolve_node(expr.func, context),
                args=[self._resolve_node(arg, context) for arg in expr.args]
            )
        return None

    def _resolve_kroneckerdelta(self, expr, context):
        if isinstance(expr, KroneckerDelta):
            left_res = self._resolve_node(expr.left, context)
            right_res = self._resolve_node(expr.right, context)
            if isinstance(left_res, Symbol):
                left_res = IndexSymbol(name=left_res.name)
            if isinstance(right_res, Symbol):
                right_res = IndexSymbol(name=right_res.name)
            return KroneckerDelta(
                left=left_res,
                right=right_res
            )
        return None

    def _resolve_diracdelta(self, expr, context):
        if isinstance(expr, DiracDelta):
            return DiracDelta(
                x=self._resolve_node(expr.x, context),
                y=self._resolve_node(expr.y, context)
            )
        return None

    # Index names bound by an enclosing sum, innermost last.  A subscript naming
    # one of these is an INDEX even when it is single — the sum says so, no
    # guessing (subscript_policy, docs/33 §9.1).  A LIST because sums nest.
    _bound_index_stack: "list[str]" = []

    @contextmanager
    def _binding(self, names):
        """Resolve a sum's BODY with its dummy indices in scope.

        The dummy has to be read BEFORE the body, which is the opposite of how
        two of the three construction sites were written; the third (the
        ``\\sum_i`` prefix form) already had it that way.
        """
        stack = self.__dict__.setdefault("_bound_index_stack", [])
        added = [n for n in names if n]
        stack.extend(added)
        try:
            yield
        finally:
            del stack[len(stack) - len(added):]

    @staticmethod
    def _index_names(nodes):
        return [getattr(n, "name", None) for n in nodes or ()]


    def _resolve_indexsum(self, expr, context):
        if isinstance(expr, IndexSum):
            dummies = [self._resolve_node(idx, context) for idx in expr.dummy_indices]
            with self._binding(self._index_names(dummies)):
                body = self._resolve_node(expr.body, context)
            return IndexSum(body=body, dummy_indices=dummies)
        return None

    def _resolve_derivative(self, expr, context):
        if isinstance(expr, Derivative):
            res_vars = []
            for sym, order in expr.vars:
                res_sym = self._resolve_node(sym, context)
                if isinstance(res_sym, Symbol):
                    res_vars.append((res_sym, order))
                else:
                    raise ValueError("Derivative variable must resolve to a Symbol")
            return Derivative(
                expr=self._resolve_node(expr.expr, context),
                vars=res_vars,
                is_partial=expr.is_partial
            )
        return None

    def _resolve_integral(self, expr, context):
        if isinstance(expr, Integral):
            res_var = self._resolve_node(expr.var, context)
            if not isinstance(res_var, Symbol):
                raise ValueError("Integral variable must resolve to a Symbol")
            return expr.model_copy(update={
                "expr": self._resolve_node(expr.expr, context),
                "var": res_var,
                "lower": self._resolve_node(expr.lower, context) if expr.lower else None,
                "upper": self._resolve_node(expr.upper, context) if expr.upper else None,
            })
        return None

    def _resolve_commutator(self, expr, context):
        if isinstance(expr, Commutator):
            return Commutator(
                a=self._resolve_node(expr.a, context),
                b=self._resolve_node(expr.b, context)
            )
        return None

    def _resolve_anticommutator(self, expr, context):
        if isinstance(expr, Anticommutator):
            return Anticommutator(
                a=self._resolve_node(expr.a, context),
                b=self._resolve_node(expr.b, context)
            )
        return None

    def _resolve_equation(self, expr, context):
        if isinstance(expr, Equation):
            return Equation(
                lhs=self._resolve_node(expr.lhs, context),
                rhs=self._resolve_node(expr.rhs, context),
                op=expr.op,
            )
        return None

    def _resolve_casoperation(self, expr, context):
        if isinstance(expr, CasOperation):
            resolved_args = {}
            for k, v in expr.args.items():
                if isinstance(v, Expr):
                    resolved_args[k] = self._resolve_node(v, context)
                elif isinstance(v, list):
                    resolved_args[k] = [
                        self._resolve_node(item, context) if isinstance(item, Expr) else item
                        for item in v
                    ]
                else:
                    resolved_args[k] = v
            return CasOperation(operation=expr.operation, args=resolved_args)
        return None

    def _bracket_functional_name(self, left_node, context) -> "str | None":
        """R-9: the CAS function name if *left_node* (the head before ``[...]``)
        is a display-only bracket functional (``E[X]``, declared ``F[X]`` …),
        else None.  Resolution order (declaration > domain profile), consistent
        with the (A) declaration-driven policy."""
        from notebook.models.ir import Subscript
        name = None
        if isinstance(left_node, Symbol):
            name = left_node.name
        elif isinstance(left_node, Subscript) and isinstance(left_node.base, Symbol):
            name = left_node.base.name          # conditional expectation E_θ[X]
        if not name or context is None:
            return None
        cas = getattr(context, "bracket_functionals", {}).get(name)
        if cas:
            return cas
        # Statistics-family domains: E[...] is expectation (notation profile).
        if name == "E":
            rule = notation_rule_for("E_bracket", context)
            if rule and rule.get("meaning") == "expectation_value":
                return "expectation"
        return None

    def _resolve_bracket_functional_head(self, expr, context):
        r"""``E[X]`` is the EXPECTATION, not ``E`` indexed by ``X``.

        A bracket glued to a name reads as a subscript (`_parse_attached_
        subscript`), which collides with R-9's bracket functionals.  Which it
        is depends on the declaration and the domain profile, so it is decided
        here — through the SAME helper the juxtaposition path uses, since
        asking twice is how two readings drift apart.  An applied form
        (``E[X](t)``) is never a functional, and ``E_\theta[X]`` never reaches
        here (its base is a Subscript, so the bracket is still juxtaposed).
        """
        if not getattr(expr, "bracketed", False):
            return None
        if isinstance(expr.sub, FunctionCall):
            return None
        fn = self._bracket_functional_name(expr.base, context)
        if fn is None:
            return None
        sub = expr.sub
        args = list(sub.args) if isinstance(sub, BracketExpr) else [sub]
        return FunctionCall(func=Symbol(name=fn),
                            args=[self._resolve_node(a, context) for a in args])

    def _prime_means_transpose(self, sym, context) -> bool:
        """AUDIT-XTX / R-11: whether a standalone prime on *sym* denotes matrix
        transpose.  True when the symbol is a declared matrix/vector/tensor, or
        the active domains' notation profile maps prime→transpose."""
        prop = context.get(sym.name)
        if prop is not None and prop.quantity.math_type.algebra in ("matrix", "vector", "tensor"):
            return True
        rule = notation_rule_for("prime", context)
        return bool(rule and rule.get("meaning") == "transpose")

    def _resolve_primeexpr(self, expr, context):
        if isinstance(expr, PrimeExpr):
            # Count the order of primes
            order = 0
            curr = expr
            while isinstance(curr, PrimeExpr):
                order += 1
                curr = curr.expr

            inner_resolved = self._resolve_node(curr, context)
            if isinstance(inner_resolved, Symbol):
                # AUDIT-XTX / R-11: a single standalone prime means TRANSPOSE when
                # the base is a matrix/vector/tensor (declaration wins), or the
                # active domains' notation profile maps prime→transpose
                # (statistics / econometrics / mathematical_economics).  So
                # X' → transpose(X), and X'X → Mul(transpose(X), X) which the
                # Maxima generator emits as `transpose(X) . X` (matrix product).
                if order == 1 and self._prime_means_transpose(inner_resolved, context):
                    from notebook.models.ir import Transpose
                    return Transpose(expr=inner_resolved)
                op_name = inner_resolved.name + ("'" * order)
                # Keep statistics if registered
                prop = context.get(inner_resolved.name)
                if prop is not None:
                    context.register(op_name, prop.quantity, prop.statistics.kind)
                return Symbol(name=op_name)
            return self._resolve_prime_standalone_other(inner_resolved, order, context)
        return None

    def _spaced_paren_is_a_product(self, callee, args, spaced, context) -> bool:
        r"""`f (x)` — a SPACE before the parenthesis means multiplication.

        The callee must not be a known CAS function nor declared callable
        (a function / operator / field), and there must be exactly ONE argument
        (a comma list `(x, y)` cannot be a product).  This is the context-aware
        half the parser deferred: it records `spaced`, the resolver decides.

        **One question, two readers.** The plain-name branch has asked it since
        the rule was written; the APPLIED-SUBSCRIPT fold did not, so the same
        space was a product after `y` and a CALL after `a_dot` — measured
        2026-09-25 on the published Lagrangian, where this system prints
        `… + \dot{a} \left(m+M\right) p^2` and reads its own line back as
        `a_dot(m + M) * p^2`.
        """
        return bool(spaced) and len(args) == 1 and not self.callee_is_function(
            callee, context)

    def _spaced_product_split(self, node, context):
        r"""``a (b)`` → ``(a, b)`` where the space is MULTIPLICATION, else None.

        One question, two shapes: the parenthesis may hang off a plain name
        (`a (b)`) or off a subscripted one (`c_o (Q-a)`, where the parser
        attaches it to the SUBSCRIPT's operand).  The head is returned as a
        SURFACE node — for the subscripted shape it is rebuilt with
        `subscript_name_node`, so "decoration or index?" stays
        `subscript_policy`'s decision rather than becoming a second copy of it
        (`a_i (k)` is `a(i)·k`, matching the bare `a_i`).
        """
        if isinstance(node, FunctionCall):
            head, call = node.func, node
        elif isinstance(node, Subscript) and isinstance(node.sub, FunctionCall):
            from notebook.runtime.shared_parser import subscript_name_node
            head, call = subscript_name_node(node.base, node.sub.func), node.sub
        else:
            return None
        if not self._spaced_paren_is_a_product(
                call.func, call.args, getattr(call, "spaced", False), context):
            return None
        return head, call.args[0]

    def _script_over_a_spaced_product(self, expr, context):
        r"""``a (b)^2`` is ``a · b^2`` — the exponent belongs to the FACTOR.

        The parser cannot tell `sin (x)` from `a (b)`, so it records `spaced`
        and hands a FunctionCall to `^`.  Where that call turns out to be a
        PRODUCT, the exponent was never on the product: `a*(b)^2` and `a b^2`
        both give `a·b^2`, and only **this system's own multiplication** — the
        juxtaposition — read it as `(a·b)^2`.  Measured 2026-09-25 on the
        published central-force energy and the newsvendor cost function, where
        the CAS differential is not zero: a DIFFERENT QUANTITY, `completed`.

        The head is handed back whole, so a chain re-associates by recursion;
        a genuine call (`sin (x)^2`) is left exactly as it was.
        """
        split = self._spaced_product_split(expr.base, context)
        if split is None:
            return None
        head, arg = split
        return self._resolve_node(
            Juxtaposition(left=head,
                          right=Superscript(base=arg, script=expr.script)),
            context)

    def _resolve_subscript(self, expr, context):
        if isinstance(expr, Subscript):
            binder = extremum_binder(expr)   # `\max_i` — a binder, not an index
            if binder is not None:           # (subscript_policy has the why)
                raise UnknownNotationError(
                    "parser.unsupported_notation.extremum_binder", name=binder)
            functional = self._resolve_bracket_functional_head(expr, context)
            if functional is not None:
                return functional
            applied = self._resolve_conjugate_then_subscript(expr, context)
            if applied is not None:
                return applied
            if isinstance(expr.base, Symbol) and expr.base.name == "delta":
                sub = expr.sub
                if isinstance(sub, BracketExpr) and sub.kind == "curly":
                    if len(sub.args) == 2:
                        return KroneckerDelta(left=self._resolve_node(sub.args[0], context), right=self._resolve_node(sub.args[1], context))
                    elif len(sub.args) == 1 and isinstance(sub.args[0], Symbol):
                        name = sub.args[0].name
                        if len(name) == 2:
                            return KroneckerDelta(left=Symbol(name=name[0]), right=Symbol(name=name[1]))
                elif isinstance(sub, Symbol) and len(sub.name) == 2:
                    return KroneckerDelta(left=Symbol(name=sub.name[0]), right=Symbol(name=sub.name[1]))

            # Check if this subscript reconstructs a registered symbol name.
            # The negation hook lets Maxima recover the name from -k_B, where
            # SharedParser produces Subscript(base=Mul(-1, k), sub=B).
            # How a subscript spelling is read as ONE name — including the digit
            # form, which tokenizes as Number(value=1) rather than Symbol("1")
            # (SUBST-DIGIT-SUB) — belongs to subscript_policy, so that this and
            # the applied form below cannot drift apart.
            def build_sub_name(e):
                return read_subscript_name(e, self._unwrap_subscript_negation)

            is_neg, base_unwrapped = self._unwrap_subscript_negation(expr.base)
            expr_for_name = Subscript(
                simplification_log=[],
                base=base_unwrapped,
                sub=expr.sub,
                bracketed=expr.bracketed,   # carried: a rebuild that drops the
                                            # spelling makes `f[i](x)` fold again
            ) if is_neg else expr

            combined_name = build_sub_name(expr_for_name)
            if combined_name and context.get(combined_name) is not None:
                sym = Symbol(name=combined_name)
                if is_neg:
                    return Mul(simplification_log=[], args=[Number(simplification_log=[], value=-1), sym])
                return sym

            # ``X_sub(args)`` — is the subscript part of the NAME or an INDEX?
            # The reading is subscript_policy's; building the node is ours,
            # because only the resolver has _resolve_node.  See that module for
            # why the rule lives in one place (it has drifted twice) and for the
            # measurement behind the two-index line.
            if isinstance(expr.sub, FunctionCall):
                # **A SPACE before the parenthesis is multiplication here too.**
                # The plain-name branch has asked that since the rule was
                # written; this one did not, so the same space was a product
                # after `y` and a CALL after `a_dot` — measured 2026-09-25 on
                # the published Lagrangian, where this system prints
                # `… + \dot{a} \left(m+M\right) p^2` and read its own line
                # back as `a_dot(m + M) * p^2`.
                #
                # The head is **the node this would have been without the
                # call** (`_resolve_conjugate_then_subscript`'s move), so
                # whether the subscript folds into a name or stays an index
                # remains `subscript_policy`'s decision rather than a second
                # copy of it: `a_i (k)` is `a(i)·k`, matching the bare `a_i`.
                _split = self._spaced_product_split(expr, context)
                if _split is not None:
                    return Mul(args=[self._resolve_node(n, context)
                                     for n in _split])
                reading = classify_called_subscript(
                    expr, context,
                    read_name=build_sub_name,
                    count_indices=lambda nm, ctx: len(
                        self._parse_indices_from_symbol(Symbol(name=nm), ctx)),
                    extract_indices=lambda node, ctx: self._extract_indices(
                        node, ctx, is_up=False),
                    bound_indices=frozenset(
                        self.__dict__.get("_bound_index_stack") or ()),
                    base_role=self._subscript_base_role,
                    is_builtin_family=lambda nm: nm in self.known_function_names,
                )
                args = [self._resolve_node(a, context) for a in expr.sub.args]
                if reading.kind == "indexed":
                    return FunctionCall(
                        func=IndexedSymbol(
                            base=Symbol(name=reading.base_name),
                            indices=reading.indices),
                        args=args)
                if reading.kind == "name":
                    folded = FunctionCall(func=Symbol(name=reading.fold_name),
                                          args=args)
                    # The fold is also how a CAS builtin whose name carries an
                    # underscore arrives — `kron_delta(m, n)` is parsed as `kron`
                    # subscripted by `delta(m,n)` and folded back here, so it
                    # never reached `_resolve_functioncall`'s hook.  Ask it once
                    # the name exists.
                    builtin = self._resolve_backend_builtin_call(folded, context)
                    return builtin if builtin is not None else folded
                # Never a raw pydantic ValidationError in the researcher's face;
                # the engine boundary translates this key.
                raise UnknownNotationError(
                    "parser.unsupported_notation.called_subscript")

            base_res = self._resolve_node(expr.base, context)

            indices = self._extract_indices(expr.sub, context, is_up=False)

            if isinstance(base_res, IndexedSymbol):
                return IndexedSymbol(base=base_res.base, indices=base_res.indices + indices)
            return IndexedSymbol(base=base_res, indices=indices)
        return None

    def _resolve_superscript(self, expr, context):
        if isinstance(expr, Superscript):
            applied = self._resolve_applied_script(expr, context)
            if applied is not None:
                return applied
            reassociated = self._script_over_a_spaced_product(expr, context)
            if reassociated is not None:
                return reassociated
            base_res = self._resolve_node(expr.base, context)
            if context.einstein_summation:
                # Einstein summation treats symbol exponents as upper indices
                # Check if the script is a pure index expression
                def is_pure_index(script_expr):
                    if isinstance(script_expr, Symbol): return True
                    if hasattr(script_expr, "kind") and getattr(script_expr, "kind") == "curly":
                        return all(isinstance(a, Symbol) for a in script_expr.args)
                    return False

                if is_pure_index(expr.script):
                    indices = self._extract_indices(expr.script, context, is_up=True)
                    if isinstance(base_res, IndexedSymbol):
                        return IndexedSymbol(base=base_res.base, indices=base_res.indices + indices)
                    return IndexedSymbol(base=base_res, indices=indices)

            # Normal exponent
            return self._resolve_node(Pow(base=expr.base, exponent=expr.script), context)
        return None

    def _resolve_indexedsymbol(self, expr, context):
        if isinstance(expr, IndexedSymbol):
            # Check if this represents \mathrm{i} or \text{i} (since backslash is skipped in tokenizer)
            if isinstance(expr.base, Symbol) and expr.base.name in ("mathrm", "text"):
                if len(expr.indices) == 1 and isinstance(expr.indices[0], (Symbol, IndexSymbol)) and expr.indices[0].name == "i":
                    return Symbol(name=self.imaginary_unit_name)

            return IndexedSymbol(
                base=self._resolve_node(expr.base, context),
                indices=[self._resolve_node(idx, context) for idx in expr.indices]
            )
        return None

    def _resolve_indexsymbol(self, expr, context):
        if isinstance(expr, IndexSymbol):
            return expr
        return None

    def _resolve_symbol(self, expr, context):
        if isinstance(expr, Symbol):
            # Protect explicit imaginary units
            if expr.name in (r"\mathrm{i}", r"\text{i}"):
                return Symbol(name=self.imaginary_unit_name)

            # Backend-specific first-chance resolution (Maxima EE: j → %i)
            special = self._resolve_symbol_backend_special(expr, context)
            if special is not None:
                return special

            if expr.name == "i":
                if context.get("i") is not None:
                    return expr
                if self._imaginary_i_blocked(context):
                    return expr
                if getattr(self, "_in_exponent", False):
                    return Symbol(name=self.imaginary_unit_name)
                return expr

            # If the full symbol name is explicitly registered in context, do not split it by underscore
            if context.get(expr.name) is not None:
                return expr
            return expr
        return None

    _RESOLVE_DISPATCH = {
        Juxtaposition: _resolve_juxtaposition,
        BracketExpr: _resolve_bracketexpr,
        Assignment: _resolve_assignment,
        Add: _resolve_add,
        Mul: _resolve_mul,
        DotProduct: _resolve_dotproduct,
        CrossProduct: _resolve_crossproduct,
        NonCommutativeMul: _resolve_noncommutativemul,
        Pow: _resolve_pow,
        FunctionCall: _resolve_functioncall,
        KroneckerDelta: _resolve_kroneckerdelta,
        DiracDelta: _resolve_diracdelta,
        IndexSum: _resolve_indexsum,
        Derivative: _resolve_derivative,
        Integral: _resolve_integral,
        Commutator: _resolve_commutator,
        Anticommutator: _resolve_anticommutator,
        Equation: _resolve_equation,
        CasOperation: _resolve_casoperation,
        PrimeExpr: _resolve_primeexpr,
        Subscript: _resolve_subscript,
        Superscript: _resolve_superscript,
        IndexedSymbol: _resolve_indexedsymbol,
        IndexSymbol: _resolve_indexsymbol,
        Symbol: _resolve_symbol,
    }


    def _extract_indices(self, script_expr: Expr, context: SymbolContext, is_up: bool) -> "list[Expr]":
        idxs = []
        if isinstance(script_expr, Symbol):
            idxs = self._parse_indices_from_symbol(script_expr, context)
        elif hasattr(script_expr, "kind") and getattr(script_expr, "kind") == "curly":
            # It's a BracketExpr e.g. {i, j} or {i j} or {ij}
            for arg in getattr(script_expr, "args"):
                # A SPACE is one of the two index separators (open_items
                # AUDIT-SUBSCRIPT-SPLIT), and the parser reports a space-
                # separated group as one Juxtaposition — so ``A_{i j}`` must give
                # the same two indices as ``A_{i,j}``.  Without this the space
                # form was left unresolved entirely, which made "separate them
                # with a comma OR a space" only half true.
                for piece in self._flatten_juxtaposition(arg):
                    if isinstance(piece, Symbol):
                        idxs.extend(self._parse_indices_from_symbol(piece, context))
                    else:
                        idxs.append(self._resolve_node(piece, context))
        else:
            idxs = [self._resolve_node(script_expr, context)]

        return self._normalize_extracted_indices(idxs, is_up)


_NOTATION_PROFILE_LOADER = None


def notation_rule_for(token: str, context: SymbolContext):
    """R-11: return the notation_profiles.json rule dict for *token* under the
    context's active domains, or None.  Lets the resolver consult the same
    domain-cluster notation profiles the LLM Supervisor uses as hints, so
    e.g. prime→transpose (statistics) or E_bracket→expectation resolve without
    a candidate dialog when the active domains make the meaning unambiguous."""
    domains = list(getattr(context, "domains", None) or [])
    if not domains:
        return None
    global _NOTATION_PROFILE_LOADER
    if _NOTATION_PROFILE_LOADER is None:
        from notebook.runtime.notation_profile_loader import NotationProfileLoader
        _NOTATION_PROFILE_LOADER = NotationProfileLoader()
    return _NOTATION_PROFILE_LOADER.get_rule(token, domains)


def get_algebra_type(expr: Expr, context: SymbolContext) -> str:
    """Helper to get the math algebra type of an expression (scalar, vector, matrix, operator)."""
    from notebook.models.ir import (
        Symbol, IndexedSymbol, PrimeExpr, Pow, FunctionCall, Decorated)
    if isinstance(expr, Symbol):
        prop = context.get(expr.name)
        if prop is not None:
            return prop.quantity.math_type.algebra
        return "scalar"
    elif isinstance(expr, Decorated):
        # A DECORATED name is its own quantity — `\hat H` is not `H`, and a
        # superscript LABEL (`G^0`, which is a Decorated too) is not `G`.  Look
        # up the flattened name, the same spelling the generator writes.  Without
        # this the declaration never reached the reading: `\hat H` declared as a
        # matrix still gave `H_hat^T` instead of `transpose(H_hat)`, and
        # `(G^0)^{-1}` stayed the scalar reciprocal (measured 2026-08-21,
        # DECL-NAME-READING).
        from notebook.runtime.expr_generator import ExprGenerator
        prop = context.get(ExprGenerator().generate(expr))
        return prop.quantity.math_type.algebra if prop is not None else "scalar"
    elif isinstance(expr, PrimeExpr):
        return get_algebra_type(expr.expr, context)
    elif isinstance(expr, IndexedSymbol):
        return get_algebra_type(expr.base, context)
    elif (isinstance(expr, FunctionCall) and isinstance(expr.func, Symbol)
            and expr.func.name == "matrix"):
        # A matrix LITERAL.  Without this the generic `args` walk below sees rows
        # of NUMBERS and answers "scalar", which made two products elementwise
        # and silent (MATMUL-TEMPLATE-STAR; the values are in
        # `tests/test_matmul_template_star.py`).  The name is the IR's, not
        # Maxima's, so reading it here is not a backend leak.
        return "matrix"
    elif hasattr(expr, "args") and getattr(expr, "args"):
        # For Add, Mul, Commutator etc., if any arg is non-scalar, return that (simple heuristic)
        for arg in getattr(expr, "args"):
            t = get_algebra_type(arg, context)
            if t != "scalar":
                return t
        return "scalar"
    elif isinstance(expr, Pow):
        return get_algebra_type(expr.base, context)
    elif isinstance(expr, FunctionCall):
        return get_algebra_type(expr.func, context)
    return "scalar"

def is_noncommutative(expr: Expr, context: SymbolContext) -> bool:
    """Helper to check if an expression is non-commutative in the given context."""
    from notebook.models.ir import Symbol, Bra, Ket, Commutator, IndexedSymbol, PrimeExpr, NonCommutativeMul, Mul, Add, Pow, FunctionCall

    if isinstance(expr, Symbol):
        prop = context.get(expr.name)
        if prop is not None:
            return prop.statistics.kind in ("noncommutative", "fermi")
        return False
    elif isinstance(expr, (Bra, Ket, Commutator, NonCommutativeMul)):
        return True
    elif isinstance(expr, IndexedSymbol):
        return is_noncommutative(expr.base, context)
    elif isinstance(expr, PrimeExpr):
        return is_noncommutative(expr.expr, context)
    elif isinstance(expr, Mul):
        return any(is_noncommutative(arg, context) for arg in expr.args)
    elif isinstance(expr, Add):
        return any(is_noncommutative(arg, context) for arg in expr.args)
    elif isinstance(expr, Pow):
        return is_noncommutative(expr.base, context)
    elif isinstance(expr, FunctionCall):
        if is_noncommutative(expr.func, context):
            return True
        if isinstance(expr.func, Symbol):
            prop = context.get(expr.func.name)
            if prop is not None and prop.statistics.kind in ("noncommutative", "fermi"):
                return True
        return False
    return False

