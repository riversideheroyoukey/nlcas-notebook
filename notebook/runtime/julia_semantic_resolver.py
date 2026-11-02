"""Julia (QuantumAlgebra.jl) semantic resolver: backend-specific hooks only.

All shared Surface-AST → Semantic-IR traversal lives in BaseSemanticResolver
(semantic_resolver.py).  This subclass supplies exactly the Julia-specific
behavior: builtin function spellings, the ``im`` imaginary unit, the broader
quantum-domain set for prime-as-adjoint, :site conversion of declared-operator
call arguments, priming of IndexedSymbol bases, and index normalization that
stringifies complex subscripts (e.g. -k) for the QA index machinery.
"""
from notebook.models.ir import (
    Expr, Symbol, Number, FunctionCall, Derivative, PrimeExpr,
    IndexSymbol, IndexedSymbol,
)
from notebook.models.symbol_context import SymbolContext
from notebook.runtime.julia_latex_converter import JuliaLatexConverter
from notebook.runtime.semantic_resolver import BaseSemanticResolver
from notebook.runtime.subscript_policy import declared_subscript_role


class JuliaSemanticResolver(BaseSemanticResolver):
    """Semantic Resolver for Julia (QuantumAlgebra.jl) backend.

    Transforms pure syntax Surface AST nodes into Julia-specific semantic
    representations.  See BaseSemanticResolver for the shared traversal
    (bracket → Commutator/Anticommutator, juxtaposition → order-preserving
    Mul/NonCommutativeMul, etc.); only the hooks below differ from Maxima.
    """

    #: This backend's function table, so the shared bare-name guard
    #: (`_reject_bare_function_names`) needs no list of its own — which names
    #: are functions is the backend's knowledge, exactly as `_FN_ALT` is.
    _FUNCTION_NAMES = JuliaLatexConverter._KNOWN_FUNCTIONS

    @staticmethod
    def _subscript_base_role(name: str, context) -> "str | None":
        r"""A declared QA OPERATOR is an indexed quantity here.

        QuantumAlgebra's operators "take an arbitrary number of indices as
        arguments" (its README), so `a_i(k)` is the researcher writing operator
        `a` with indices `i` and `k` — not a function named `a_i`.

        Without this the single-index applied form was FOLDED to a name and
        emitted as `a_i(k)`, which Julia answers with `Sym a_i is not callable`
        — raw CAS text (Core Principle #1).  The generator already refuses the
        multi-index form with the right advice ("write the argument as one more
        index"); measured 2026-08-18, `a(QuIndex("i"), QuIndex("k"))` is `a(ik)`
        and works, so the advice is sound and both counts should reach it.
        The code comment claiming "the index count is irrelevant" described the
        REFUSAL BRANCH, not the behaviour: a single index never got there
        (open_items JULIA-APPLIED-INDEXED-COUNT).
        """
        prop = context.get(name) if name else None
        if prop is not None and getattr(prop.quantity, "kind", None) == "operator":
            return "indexed"
        return declared_subscript_role(name, context)

    # Julia/Base builtin function names (added to the common elementary set) for
    # the "f (x)" space-disambiguation.  Julia spellings differ from Maxima:
    # ceil/sign/conj/real/imag (vs ceiling/signum/conjugate/realpart/imagpart),
    # plus cbrt and the two-argument atan.
    known_function_names = BaseSemanticResolver.known_function_names | frozenset({
        "ceil", "sign", "conj", "real", "imag", "cbrt", "log10", "log2", "angle",
    })

    def _square_bracket_role(self, expr, context) -> str:
        """In QuantumAlgebra ``[A, B]`` is the commutator — always, at every
        nesting depth.  A commutator of commutators is ordinary operator algebra
        (the Jacobi identity), so nesting must NOT be read as a list here; that
        is precisely why this is a hook and not a shared rule."""
        return "commutator"

    imaginary_unit_name = "im"
    # Historically the Julia resolver only recognised Σ-sums, and only inside
    # the Mul branch (explicit ``*`` products); no ∫-prefix handling.  The
    # asymmetry was preserved behind this flag in 2026-07-04 (R-7) with the
    # note "no measured harm — latex_to_canonical inserts the `*`, so the
    # existing input path does not hit it; if a space-separated sum is ever
    # mis-resolved in a Julia session, set this True and check the full suite".
    #
    # 2026-08-07: that mis-resolution was measured.  `\sum_k c'_{k,u} c_{k,u}`
    # (no `*`) read `\sum_k` as an indexed SCALAR named `sum`, and the refusal
    # told the researcher to declare `sum_k` as a name — advice about a quantity
    # they never wrote.  Turned on, per the ticket's own instruction.
    resolve_sum_integral_prefix = True

    # ------------------------------------------------------------------
    # Prime handling
    # ------------------------------------------------------------------

    # Domains whose sessions use creation/annihilation operator notation:
    # prime means adjoint, not a calculus derivative.
    _QUANTUM_DOMAINS = frozenset({
        "second_quantization", "quantum_mechanics",
        "condensed_matter", "nuclear_physics",
    })

    def _prime_call_is_quantum(self, curr: Expr, inner_func: Expr,
                               context: SymbolContext) -> bool:
        # Also treat as quantum if the inner function is a registered operator
        # in the context (catches any physics domain).
        _inner_prop = context.get(curr.name) if isinstance(curr, Symbol) else None
        return (
            bool(self._QUANTUM_DOMAINS & set(context.domains)) or
            context.session_mode == "second_quantization" or
            (_inner_prop is not None and _inner_prop.quantity.kind == "operator")
        )

    def _resolve_prime_call_derivative(self, inner_func: Expr, expr: FunctionCall,
                                       order: int, context: SymbolContext) -> Expr:
        # In calculus/standard domain, prime represents derivative
        resolved_args = [self._resolve_node(arg, context) for arg in expr.args]
        if len(resolved_args) >= 1 and isinstance(resolved_args[0], Symbol):
            return Derivative(
                expr=FunctionCall(func=inner_func, args=resolved_args),
                vars=[(resolved_args[0], order)],
                is_partial=False
            )
        # If first arg is not a Symbol, fallback to standard function call
        return FunctionCall(
            func=self._resolve_node(expr.func, context),
            args=resolved_args
        )

    def _resolve_prime_standalone_other(self, inner_resolved: Expr, order: int,
                                        context: SymbolContext) -> Expr:
        if isinstance(inner_resolved, IndexedSymbol):
            if isinstance(inner_resolved.base, Symbol):
                op_name = inner_resolved.base.name + ("'" * order)
                prop = context.get(inner_resolved.base.name)
                if prop is not None:
                    context.register(op_name, prop.quantity, prop.statistics.kind)
                return IndexedSymbol(base=Symbol(name=op_name), indices=inner_resolved.indices)
            else:
                # fallback if base is not a simple symbol
                return PrimeExpr(expr=inner_resolved)
        return inner_resolved

    # ------------------------------------------------------------------
    # Declared-operator calls: convert symbol args to :site form
    # ------------------------------------------------------------------

    def _resolve_declared_operator_call(self, expr: FunctionCall,
                                        context: SymbolContext) -> "Expr | None":
        if not isinstance(expr.func, Symbol):
            return None
        # Check if it is a Quantum Operator Call (strip adjoint prime/dagger)
        base_op_name = expr.func.name.rstrip("'†")
        prop = context.get(base_op_name)
        if prop and prop.quantity.kind == "operator":
            resolved_args = []
            for arg in expr.args:
                res_arg = self._resolve_node(arg, context)
                if isinstance(res_arg, Symbol) and not res_arg.name.startswith(":"):
                    resolved_args.append(Symbol(name=f":{res_arg.name}"))
                else:
                    resolved_args.append(res_arg)
            return FunctionCall(
                func=self._resolve_node(expr.func, context),
                args=resolved_args
            )
        return None

    # ------------------------------------------------------------------
    # Index normalization
    # ------------------------------------------------------------------

    def _normalize_extracted_indices(self, idxs: list, is_up: bool) -> list:
        for i, idx in enumerate(idxs):
            if isinstance(idx, Symbol):
                idxs[i] = IndexSymbol(name=idx.name, is_up=is_up)
            elif isinstance(idx, IndexSymbol):
                idx.is_up = is_up
            elif not isinstance(idx, IndexSymbol):
                # 代数的な添字（`-k`・`k+q`）。読みは共有側の一箇所に置いた。
                idxs[i] = self.index_from_compound(idx, is_up)
        return idxs
