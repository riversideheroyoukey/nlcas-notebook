from __future__ import annotations
import re as _re_idx
from typing import Set, List, Optional

from notebook.models.ir import (
    Expr, Symbol, Number, Add, Mul, NonCommutativeMul, Pow, FunctionCall, Derivative,
    Integral, Commutator, Bra, Ket, Equation, OpaqueExpr,
    IndexSymbol, IndexedSymbol, KroneckerDelta, DiracDelta, IndexSum
)
from notebook.models.symbol_context import SymbolContext

def substitute_symbol_or_index(expr: Expr, target_name: str, replacement: Expr) -> Expr:
    """Recursively replace occurrences of symbol/index named `target_name` with `replacement`."""
    if isinstance(expr, Symbol):
        # Strip leading colon added by JuliaSemanticResolver for operator args (":k" → "k").
        bare_name = expr.name[1:] if expr.name.startswith(":") and len(expr.name) > 1 else expr.name
        if bare_name == target_name:
            if isinstance(replacement, (Symbol, IndexSymbol)):
                return Symbol(name=replacement.name)
            return replacement
        return expr

    elif isinstance(expr, IndexSymbol):
        if expr.name == target_name:
            if isinstance(replacement, (Symbol, IndexSymbol)):
                return IndexSymbol(name=replacement.name, is_up=expr.is_up,
                                   domain=expr.domain, range=expr.range)
            return replacement
        # Handle composite index names like "-k", "k+q" where target_name is a component.
        # Only applies when replacement is a simple symbol (has a plain name).
        if isinstance(replacement, (Symbol, IndexSymbol)) and target_name in expr.name:
            new_name = _re_idx.sub(
                r'(?<![A-Za-z_0-9])' + _re_idx.escape(target_name) + r'(?![A-Za-z_0-9])',
                replacement.name,
                expr.name,
            )
            if new_name != expr.name:
                return IndexSymbol(name=new_name, is_up=expr.is_up,
                                   domain=expr.domain, range=expr.range)
        return expr

    elif isinstance(expr, IndexedSymbol):
        new_base = substitute_symbol_or_index(expr.base, target_name, replacement)
        new_indices = [
            substitute_symbol_or_index(idx, target_name, replacement)
            for idx in expr.indices
        ]
        return IndexedSymbol(base=new_base, indices=new_indices)

    elif isinstance(expr, KroneckerDelta):
        return KroneckerDelta(
            left=substitute_symbol_or_index(expr.left, target_name, replacement),
            right=substitute_symbol_or_index(expr.right, target_name, replacement)
        )

    elif isinstance(expr, DiracDelta):
        return DiracDelta(
            x=substitute_symbol_or_index(expr.x, target_name, replacement),
            y=substitute_symbol_or_index(expr.y, target_name, replacement)
        )

    elif isinstance(expr, IndexSum):
        # Shadowing check: if target_name is a dummy index in this nested sum, do not substitute inside
        if any(idx.name == target_name for idx in expr.dummy_indices):
            return expr
        new_dummy = [
            substitute_symbol_or_index(idx, target_name, replacement)
            for idx in expr.dummy_indices
        ]
        new_body = substitute_symbol_or_index(expr.body, target_name, replacement)
        # model_copy so fields this pass knows nothing about survive.  `lower`/
        # `upper` were added later and were silently dropped here, which put a
        # `\sum_{i=1}^{N}` back to the two-argument `sum(x(i), i)` Maxima
        # rejects — after the resolver had read the range correctly.
        return expr.model_copy(update={"body": new_body, "dummy_indices": new_dummy})

    elif isinstance(expr, Add):
        return Add(args=[substitute_symbol_or_index(arg, target_name, replacement) for arg in expr.args])

    elif isinstance(expr, Mul):
        new_args = [substitute_symbol_or_index(arg, target_name, replacement) for arg in expr.args]
        # Simplify double-negation: Mul([-1, Mul([-1, x])]) → x
        # This arises when substituting k → -k into g(-k), yielding g(-(-k)).
        if (len(new_args) == 2
                and isinstance(new_args[0], Number) and new_args[0].value == -1
                and isinstance(new_args[1], Mul) and len(new_args[1].args) == 2
                and isinstance(new_args[1].args[0], Number) and new_args[1].args[0].value == -1):
            return new_args[1].args[1]
        return Mul(args=new_args)

    elif isinstance(expr, NonCommutativeMul):
        return NonCommutativeMul(args=[substitute_symbol_or_index(arg, target_name, replacement) for arg in expr.args])

    elif isinstance(expr, Pow):
        return Pow(
            base=substitute_symbol_or_index(expr.base, target_name, replacement),
            exponent=substitute_symbol_or_index(expr.exponent, target_name, replacement)
        )

    elif isinstance(expr, FunctionCall):
        return FunctionCall(
            func=substitute_symbol_or_index(expr.func, target_name, replacement),
            args=[substitute_symbol_or_index(arg, target_name, replacement) for arg in expr.args]
        )

    elif isinstance(expr, Commutator):
        return Commutator(
            a=substitute_symbol_or_index(expr.a, target_name, replacement),
            b=substitute_symbol_or_index(expr.b, target_name, replacement)
        )

    elif isinstance(expr, Derivative):
        new_expr = substitute_symbol_or_index(expr.expr, target_name, replacement)
        new_vars = [
            (substitute_symbol_or_index(v, target_name, replacement), order)
            for v, order in expr.vars
        ]
        return Derivative(expr=new_expr, vars=new_vars, is_partial=expr.is_partial)

    elif isinstance(expr, Integral):
        new_expr = substitute_symbol_or_index(expr.expr, target_name, replacement)
        new_var = substitute_symbol_or_index(expr.var, target_name, replacement)
        new_lower = substitute_symbol_or_index(expr.lower, target_name, replacement) if expr.lower else None
        new_upper = substitute_symbol_or_index(expr.upper, target_name, replacement) if expr.upper else None
        # model_copy, not a fresh Integral: a field added later (measure_dim)
        # is carried through instead of being silently dropped by every walker.
        return expr.model_copy(update={"expr": new_expr, "var": new_var,
                                       "lower": new_lower, "upper": new_upper})

    elif isinstance(expr, Equation):
        return Equation(
            lhs=substitute_symbol_or_index(expr.lhs, target_name, replacement),
            rhs=substitute_symbol_or_index(expr.rhs, target_name, replacement),
            op=expr.op,
        )

    return expr


def check_domain_compatibility(idx1: Expr, idx2: Expr, context: SymbolContext) -> None:
    """Validate that the domain and range of two indices are compatible before contracting."""
    if isinstance(idx1, IndexSymbol) and isinstance(idx2, IndexSymbol):
        dom1 = idx1.domain
        dom2 = idx2.domain
        if dom1 and dom2 and dom1 != dom2:
            raise ValueError(
                f"Domain mismatch: cannot contract index '{idx1.name}' (domain: {dom1}) "
                f"with index '{idx2.name}' (domain: {dom2})"
            )
        rng1 = idx1.range
        rng2 = idx2.range
        if rng1 and rng2 and rng1 != rng2:
            raise ValueError(
                f"Range mismatch: cannot contract index '{idx1.name}' (range: {rng1}) "
                f"with index '{idx2.name}' (range: {rng2})"
            )


def propagate_domain_info(idx1: Expr, idx2: Expr, context: SymbolContext) -> None:
    """Propagate domain/range metadata between two contracted index symbols."""
    if isinstance(idx1, IndexSymbol) and isinstance(idx2, IndexSymbol):
        if idx1.domain and not idx2.domain:
            idx2.domain = idx1.domain
        elif idx2.domain and not idx1.domain:
            idx1.domain = idx2.domain

        if idx1.range and not idx2.range:
            idx2.range = idx1.range
        elif idx2.range and not idx1.range:
            idx1.range = idx2.range


def contract_indices(expr: Expr, context: SymbolContext, log: list[str] | None = None) -> Expr:
    if log is None:
        log = []
    """Recursively simplify KroneckerDelta and perform discrete index contraction on discrete sums (IndexSum)."""
    if isinstance(expr, KroneckerDelta):
        left = contract_indices(expr.left, context, log=log)
        right = contract_indices(expr.right, context, log=log)
        if isinstance(left, (Symbol, IndexSymbol)) and isinstance(right, (Symbol, IndexSymbol)):
            if left.name == right.name:
                return Number(value=1)
        return KroneckerDelta(left=left, right=right)

    elif isinstance(expr, IndexSum):
        body = contract_indices(expr.body, context, log=log)
        dummy_indices = list(expr.dummy_indices)

        if isinstance(body, Mul):
            factors = list(body.args)
        else:
            factors = [body]

        changed = True
        while changed:
            changed = False
            for dummy_idx in list(dummy_indices):
                d_name = dummy_idx.name
                for i, factor in enumerate(factors):
                    if isinstance(factor, KroneckerDelta):
                        left_name = factor.left.name if isinstance(factor.left, (Symbol, IndexSymbol)) else None
                        right_name = factor.right.name if isinstance(factor.right, (Symbol, IndexSymbol)) else None

                        other_side = None
                        if left_name == d_name:
                            other_side = factor.right
                        elif right_name == d_name:
                            other_side = factor.left

                        if other_side is not None:
                            # Verify and propagate domain details
                            check_domain_compatibility(dummy_idx, other_side, context)
                            propagate_domain_info(dummy_idx, other_side, context)

                            # Consume this delta and this dummy index
                            factors.pop(i)
                            dummy_indices.remove(dummy_idx)
                            log.append(f"Contracted index '{d_name}' with '{other_side.name}' via KroneckerDelta.")

                            # Substitute in all other factors
                            factors = [substitute_symbol_or_index(f, d_name, other_side) for f in factors]
                            changed = True
                            break
                if changed:
                    break

        # Rebuild or eliminate the discrete sum
        if not dummy_indices:
            if not factors:
                return Number(value=1)
            elif len(factors) == 1:
                return factors[0]
            else:
                return Mul(args=factors)
        else:
            if not factors:
                new_body = Number(value=1)
            elif len(factors) == 1:
                new_body = factors[0]
            else:
                new_body = Mul(args=factors)
            return expr.model_copy(update={"body": new_body, "dummy_indices": dummy_indices})

    # Recursive traversal for all other node types
    elif isinstance(expr, Add):
        return Add(args=[contract_indices(arg, context, log=log) for arg in expr.args])

    elif isinstance(expr, Mul):
        return Mul(args=[contract_indices(arg, context, log=log) for arg in expr.args])

    elif isinstance(expr, Pow):
        return Pow(
            base=contract_indices(expr.base, context, log=log),
            exponent=contract_indices(expr.exponent, context, log=log)
        )

    elif isinstance(expr, FunctionCall):
        return FunctionCall(
            func=contract_indices(expr.func, context, log=log),
            args=[contract_indices(arg, context, log=log) for arg in expr.args]
        )

    elif isinstance(expr, Commutator):
        return Commutator(
            a=contract_indices(expr.a, context, log=log),
            b=contract_indices(expr.b, context, log=log)
        )

    elif isinstance(expr, Derivative):
        new_expr = contract_indices(expr.expr, context, log=log)
        new_vars = [(contract_indices(v, context, log=log), order) for v, order in expr.vars]
        return Derivative(expr=new_expr, vars=new_vars, is_partial=expr.is_partial)

    elif isinstance(expr, Integral):
        new_expr = contract_indices(expr.expr, context, log=log)
        new_var = contract_indices(expr.var, context, log=log)
        new_lower = contract_indices(expr.lower, context, log=log) if expr.lower else None
        new_upper = contract_indices(expr.upper, context, log=log) if expr.upper else None
        # model_copy, not a fresh Integral: a field added later (measure_dim)
        # is carried through instead of being silently dropped by every walker.
        return expr.model_copy(update={"expr": new_expr, "var": new_var,
                                       "lower": new_lower, "upper": new_upper})

    elif isinstance(expr, Equation):
        return Equation(
            lhs=contract_indices(expr.lhs, context, log=log),
            rhs=contract_indices(expr.rhs, context, log=log),
            op=expr.op,
        )

    return expr


def get_free_variables(expr: Expr) -> Set[str]:
    """Identify all unbound symbols/index symbols in the expression tree."""
    if isinstance(expr, (Symbol, IndexSymbol)):
        return {expr.name}
    elif isinstance(expr, IndexedSymbol):
        s = get_free_variables(expr.base)
        for idx in expr.indices:
            s.update(get_free_variables(idx))
        return s
    elif isinstance(expr, IndexSum):
        dummy_names = {idx.name for idx in expr.dummy_indices}
        return get_free_variables(expr.body) - dummy_names
    elif isinstance(expr, Add):
        s = set()
        for arg in expr.args:
            s.update(get_free_variables(arg))
        return s
    elif isinstance(expr, Mul):
        s = set()
        for arg in expr.args:
            s.update(get_free_variables(arg))
        return s
    elif isinstance(expr, Pow):
        return get_free_variables(expr.base) | get_free_variables(expr.exponent)
    elif isinstance(expr, FunctionCall):
        s = get_free_variables(expr.func)
        for arg in expr.args:
            s.update(get_free_variables(arg))
        return s
    elif isinstance(expr, KroneckerDelta):
        return get_free_variables(expr.left) | get_free_variables(expr.right)
    elif isinstance(expr, DiracDelta):
        return get_free_variables(expr.x) | get_free_variables(expr.y)
    elif isinstance(expr, Commutator):
        return get_free_variables(expr.a) | get_free_variables(expr.b)
    elif isinstance(expr, Derivative):
        s = get_free_variables(expr.expr)
        for v, _ in expr.vars:
            s.update(get_free_variables(v))
        return s
    elif isinstance(expr, Integral):
        s = get_free_variables(expr.expr) | get_free_variables(expr.var)
        if expr.lower:
            s.update(get_free_variables(expr.lower))
        if expr.upper:
            s.update(get_free_variables(expr.upper))
        return s
    elif isinstance(expr, Equation):
        return get_free_variables(expr.lhs) | get_free_variables(expr.rhs)
    return set()


def generate_unique_name(base: str, occupied: Set[str]) -> str:
    """Generate a readable, unique variable name using index convention cyclic letters or increments."""
    # List of common subscript index symbols
    candidates = ["j", "k", "l", "m", "n", "p", "q", "r", "s", "t"]
    for c in candidates:
        if c not in occupied:
            return c
    counter = 1
    while f"{base}_{counter}" in occupied:
        counter += 1
    return f"{base}_{counter}"


def _alphatize(expr: Expr, occupied: Set[str], log: list[str]) -> Expr:
    """Recursively rename bound variables to resolve variable name collisions."""
    if isinstance(expr, IndexSum):
        new_dummy_indices = []
        body = expr.body
        # Names in scope INSIDE this binder: what the caller had, plus this
        # sum's own dummies.  The caller's set is NOT touched, so a SIBLING sum
        # does not see this one's dummies and keeps the name as written.
        #
        # Renaming a sibling is gratuitous — their scopes are disjoint, so no
        # capture is possible — and it is not free: Maxima's noun forms do not
        # alpha-convert (measured 2026-08-17, `'sum(x(i),i,a,b) -
        # 'sum(x(j),j,a,b)` does not reduce), so `\sum_i x_i - \sum_i x_i` came
        # back unsimplified.  Renaming a NESTED dummy is still required and
        # still happens: the inner one really is inside the outer's scope.
        inner = set(occupied)
        for dummy_idx in expr.dummy_indices:
            d_name = dummy_idx.name
            if d_name in inner:
                new_name = generate_unique_name(d_name, inner)
                log.append(f"Renamed dummy index '{d_name}' -> '{new_name}' to avoid collision.")
                # Substitute the old name in the body and in subsequent dummy indices
                body = substitute_symbol_or_index(body, d_name, Symbol(name=new_name))
                # Rewrite a COMPOSITE sibling that mentions the old name
                # (`-i` beside `i`), never a sibling that IS it: that one is an
                # earlier binder which legitimately kept the name.  Rewriting
                # it collapsed two indices into one — `\sum_{i,i}` came out as
                # `sum(sum(…, j), j)` and `\sum_{i,i,i} x_i` as
                # `sum(sum(sum(x(j), k), j), j)`, three binders and two names,
                # silently (found by mutation 2026-08-17).
                new_dummy_indices = [
                    idx if idx.name == d_name else
                    substitute_symbol_or_index(idx, d_name, Symbol(name=new_name))
                    for idx in new_dummy_indices
                ]
                dummy_idx = dummy_idx.model_copy(update={"name": new_name})
                inner.add(new_name)
            else:
                inner.add(d_name)
            new_dummy_indices.append(dummy_idx)

        new_body = _alphatize(body, inner, log)
        return expr.model_copy(update={"body": new_body,
                                       "dummy_indices": new_dummy_indices})

    # Recursive cases
    elif isinstance(expr, IndexedSymbol):
        return IndexedSymbol(
            base=_alphatize(expr.base, occupied, log),
            indices=[_alphatize(idx, occupied, log) for idx in expr.indices]
        )
    elif isinstance(expr, KroneckerDelta):
        return KroneckerDelta(
            left=_alphatize(expr.left, occupied, log),
            right=_alphatize(expr.right, occupied, log)
        )
    elif isinstance(expr, DiracDelta):
        return DiracDelta(
            x=_alphatize(expr.x, occupied, log),
            y=_alphatize(expr.y, occupied, log)
        )
    elif isinstance(expr, Add):
        return Add(args=[_alphatize(arg, occupied, log) for arg in expr.args])
    elif isinstance(expr, Mul):
        return Mul(args=[_alphatize(arg, occupied, log) for arg in expr.args])
    elif isinstance(expr, NonCommutativeMul):
        return NonCommutativeMul(args=[_alphatize(arg, occupied, log) for arg in expr.args])
    elif isinstance(expr, Pow):
        return Pow(
            base=_alphatize(expr.base, occupied, log),
            exponent=_alphatize(expr.exponent, occupied, log)
        )
    elif isinstance(expr, FunctionCall):
        return FunctionCall(
            func=_alphatize(expr.func, occupied, log),
            args=[_alphatize(arg, occupied, log) for arg in expr.args]
        )
    elif isinstance(expr, Commutator):
        return Commutator(
            a=_alphatize(expr.a, occupied, log),
            b=_alphatize(expr.b, occupied, log)
        )
    elif isinstance(expr, Derivative):
        new_expr = _alphatize(expr.expr, occupied, log)
        new_vars = [(_alphatize(v, occupied, log), order) for v, order in expr.vars]
        return Derivative(expr=new_expr, vars=new_vars, is_partial=expr.is_partial)
    elif isinstance(expr, Integral):
        new_expr = _alphatize(expr.expr, occupied, log)
        new_var = _alphatize(expr.var, occupied, log)
        new_lower = _alphatize(expr.lower, occupied, log) if expr.lower else None
        new_upper = _alphatize(expr.upper, occupied, log) if expr.upper else None
        # model_copy, not a fresh Integral: a field added later (measure_dim)
        # is carried through instead of being silently dropped by every walker.
        return expr.model_copy(update={"expr": new_expr, "var": new_var,
                                       "lower": new_lower, "upper": new_upper})
    elif isinstance(expr, Equation):
        return Equation(
            lhs=_alphatize(expr.lhs, occupied, log),
            rhs=_alphatize(expr.rhs, occupied, log),
            op=expr.op,
        )

    return expr


def alphatize_indices(expr: Expr, occupied_names: Optional[Set[str]] = None, log: list[str] | None = None) -> Expr:
    if log is None:
        log = []
    """Entry point for renaming bound variables in IndexSum to prevent collisions."""
    if occupied_names is None:
        occupied_names = get_free_variables(expr)
    else:
        occupied_names = set(occupied_names) | get_free_variables(expr)
    return _alphatize(expr, occupied_names, log)
