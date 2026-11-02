"""How an operator expression is written FOR SymPy — one place.

Extracted from `julia_code_generator.py` 2026-08-07, when the hygiene ratchet
fired (1049 > 1000).  Same move as `julia_index_policy` and `subscript_policy`
before it, and the same reason: the concern had grown into a coherent unit of
its own.  "How do we write this for SymPy?" is now asked and answered here.

That the answer had been split is what made the SymPy route wrong.  The
builder in `julia_sympy_mixin` was doing its own regex conversion because the
serializer here stripped scalar coefficients, and the two disagreed about the
one thing that mattered — with the result that the braced spelling
`c'_{k+q,u}` reached SymPy unconverted and came back as the literal 1, and a
coefficient `V(q)` was dropped outright.  See
`tests/test_sympy_route_operator_spelling.py`.

The generator mixes this in, so `self.context` / `self.generate` resolve
through the MRO exactly as before; nothing here holds state of its own.
"""
from __future__ import annotations

import re as _re
from typing import Optional

from notebook.models.ir import (
    Expr, Symbol, Number, IndexSymbol, IndexedSymbol, Adjoint, Conjugate,
    Transpose, IndexSum, Mul, NonCommutativeMul, Add, FunctionCall, Pow,
)


class JuliaSympySerializeMixin:
    """Serialize IR operator expressions to SymPy source text."""

    def _to_sympy_op_name(self, symbol_name: str, force_creation: bool = False) -> str:
        """Map an IR operator symbol name to the SymPy engine convention.

        Fermion family (c, d, f): creation → "cd", annihilation → "c".
        Boson  family (a, b):     creation → "ad", annihilation → "a".
        Prime or dagger suffix marks creation.
        """
        base = symbol_name.rstrip("'†")
        is_creation = force_creation or "'" in symbol_name or "†" in symbol_name
        if base.lower() in ("c", "d", "f"):
            return "cd" if is_creation else "c"
        if base.lower() in ("a", "b"):
            return "ad" if is_creation else "a"
        # Generic fallback: append "d" for creation (may not be handled by Julia regex)
        return (base + "d") if is_creation else base

    def _serialize_single_op_to_sympy(self, node: Expr) -> str:
        """Serialize one operator IR node to SymPy cd/c/ad/a string form."""
        if isinstance(node, IndexedSymbol):
            if isinstance(node.base, Symbol):
                op_name = self._to_sympy_op_name(node.base.name)
            else:
                op_name = "c"
            idx_str = ",".join(idx.name for idx in node.indices)
            return f"{op_name}({idx_str})"

        if isinstance(node, FunctionCall):
            if isinstance(node.func, Symbol):
                op_name = self._to_sympy_op_name(node.func.name)
            else:
                op_name = "c"
            arg_strs = []
            for a in node.args:
                if isinstance(a, IndexSymbol):
                    arg_strs.append(a.name.lstrip(":"))
                elif isinstance(a, Symbol):
                    # Strip leading colon added by JuliaSemanticResolver for operator args
                    arg_strs.append(a.name.lstrip(":"))
                else:
                    arg_strs.append(self.generate(a))
            return f"{op_name}({','.join(arg_strs)})"

        if isinstance(node, Adjoint):
            inner = node.expr
            if isinstance(inner, FunctionCall) and isinstance(inner.func, Symbol):
                op_name = self._to_sympy_op_name(inner.func.name, force_creation=True)
                arg_strs = [
                    a.name if isinstance(a, IndexSymbol) else self.generate(a)
                    for a in inner.args
                ]
                return f"{op_name}({','.join(arg_strs)})"
            if isinstance(inner, IndexedSymbol) and isinstance(inner.base, Symbol):
                op_name = self._to_sympy_op_name(inner.base.name, force_creation=True)
                idx_str = ",".join(idx.name for idx in inner.indices)
                return f"{op_name}({idx_str})"

        # Fallback: try generic generate and wrap
        return f"c({self.generate(node)})"

    def _is_scalar_factor(self, node: Expr) -> bool:
        """Return True if *node* is a pure scalar factor (not a quantum operator).

        Scalar factors — bare Symbol names, Numbers, Pow of a Symbol, and
        subscripted scalar parameters like ξ_k — should be stripped from the
        operator product string passed to the SymPy helper.  Including them
        would mis-represent them as operators (e.g. Symbol("Delta") → "c(Delta)",
        IndexedSymbol("xi", [k]) → treated as an operator product).

        The SymPy helper computes purely algebraic commutator structure; scalar
        coefficients factor out and are handled by the surrounding expression.
        """
        if isinstance(node, Number):
            return True
        if isinstance(node, Symbol):
            # A bare Symbol is a scalar/parameter unless registered as an operator.
            if self.context:
                prop = self.context.get(node.name)
                if prop and hasattr(prop, "quantity"):
                    kind = getattr(prop.quantity, "kind", None)
                    if kind in ("operator", "fermion", "boson"):
                        return False  # it is an operator — keep it
            return True  # bare symbol treated as scalar parameter
        if isinstance(node, IndexedSymbol) and isinstance(node.base, Symbol):
            # Subscripted symbol: distinguish operators (c_k, a_k) from
            # scalar parameters (xi_k, Delta_k, omega_k).
            base = node.base.name
            if base.endswith("'") or base.endswith("†"):
                return False  # creation/annihilation operator (c'_k, a†_k)
            if self.context:
                prop = self.context.get(base)
                if prop and hasattr(prop, "quantity"):
                    kind = getattr(prop.quantity, "kind", None)
                    if kind in ("operator", "fermion", "boson"):
                        return False  # registered operator
                    if kind in ("scalar", "vector", "matrix", "tensor", "anonymous"):
                        return True   # registered scalar parameter
            # Heuristic without context (mirrors the CLAUDE.md design rule):
            # Single-letter bases are almost universally annihilation operators
            # (c_k, a_k, d_k); multi-char bases are physics parameters (xi_k,
            # Delta_k, omega_k).
            if len(base) == 1 and base.isalpha():
                return False  # single-letter → treat as operator
            return True  # multi-char → treat as scalar parameter
        if isinstance(node, Pow):
            return self._is_scalar_factor(node.base)
        if isinstance(node, (Adjoint, Conjugate)):
            # Adjoint/conjugate of a Symbol may be a scalar (e.g. Δ*)
            return self._is_scalar_factor(node.expr)
        if isinstance(node, FunctionCall) and isinstance(node.func, Symbol):
            # conj(Δ), conjugate(Δ) etc. are scalar when their argument is scalar.
            if node.func.name in ("conj", "conjugate", "re", "im", "abs"):
                return all(self._is_scalar_factor(a) for a in node.args)
        return False

    def _serialize_lhs_to_sympy(self, node: Expr, extra_vars: set) -> str:
        """Serialize the LHS operator product to a SymPy prod_str string.

        extra_vars is updated in-place with dummy index names from any
        IndexSum wrappers found while unwrapping.

        Scalar factors (bare Symbols, Numbers) are stripped so they are not
        mistaken for operators.  E.g. Mul([Symbol("Delta"), FunctionCall("c'",
        [idx("k")]…)]) → "cd(k)" rather than "c(Delta)*cd(k)".
        """
        if isinstance(node, (Mul, NonCommutativeMul)):
            op_parts = [
                self._serialize_single_op_to_sympy(a)
                for a in node.args
                if not self._is_scalar_factor(a)
            ]
            if not op_parts:
                return ""
            return "*".join(op_parts)
        if isinstance(node, IndexSum):
            # Unwrap: collect dummy vars and recurse into body
            for idx in node.dummy_indices:
                for token in _re.findall(r'[A-Za-z_]\w*', idx.name):
                    extra_vars.add(token)
            return self._serialize_lhs_to_sympy(node.body, extra_vars)
        # Single operator
        return self._serialize_single_op_to_sympy(node)

    def serialize_to_sympy(self, node: Expr, extra_vars: set) -> str:
        r"""Serialize an operator expression to SymPy syntax, KEEPING scalars.

        The sibling `_serialize_lhs_to_sympy` strips scalar factors on purpose
        (a commutator's structure does not need them).  `normal_order` does:
        the scalar IS the physics — `V(q)` in the electron-electron interaction
        `V(q) c'_{k+q,u} c'_{p-q,d} c_{p,d} c_{k,u}`.

        Why this exists at all: the caller used to hand the driver the
        researcher's own canonical, which is SPACE-separated (this project's
        "space = multiplication" convention) and may spell operators with
        braced subscripts.  SymPy needs `*` and `cd(...)`.  Measured 2026-08-07,
        the driver then failed to sympify and — because it caught that failure
        and substituted 1 — returned a plausible wrong answer with no error:
        `V(q) cd(k+q,u) c(k,u)` came back as `cd(k+q,u)*c(k,u)`, the
        interaction strength silently gone, and the braced spelling came back
        as the bare literal `1`.

        Working from the IR instead of a regex is what makes the two spellings
        one case: `c'(k+q,u)` and `c'_{k+q,u}` resolve to the same nodes.

        `extra_vars` is updated in place with dummy index names from IndexSum
        wrappers, exactly as in `_serialize_lhs_to_sympy`.

        Raises `UnknownNotationError` for anything it cannot write faithfully —
        never a silent pass-through, which is the failure this replaces.
        """
        if isinstance(node, IndexSum):
            for idx in node.dummy_indices:
                for token in _re.findall(r'[A-Za-z_]\w*', idx.name):
                    extra_vars.add(token)
            return self.serialize_to_sympy(node.body, extra_vars)

        if isinstance(node, Add):
            return " + ".join(
                self.serialize_to_sympy(a, extra_vars) for a in node.args)

        if isinstance(node, (Mul, NonCommutativeMul)):
            return "*".join(
                self.serialize_to_sympy(a, extra_vars) for a in node.args)

        if self._is_scalar_factor(node):
            return self._scalar_to_sympy(node, extra_vars)
        return self._serialize_single_op_to_sympy(node)

    def _scalar_to_sympy(self, node: Expr, extra_vars: set) -> str:
        """Serialize a SCALAR IR node to SymPy source text.

        Deliberately small: the shapes a coefficient in front of an operator
        product actually takes.  Anything else raises rather than being guessed
        at — a coefficient written wrong is the same silent-wrong this whole
        path is being repaired for.
        """
        from notebook.runtime.shared_parser import UnknownNotationError

        if isinstance(node, Number):
            return str(node.value)
        if isinstance(node, (Symbol, IndexSymbol)):
            return node.name.lstrip(":")
        if isinstance(node, IndexedSymbol) and isinstance(node.base, Symbol):
            # A subscripted scalar (V_q, xi_k) becomes a SymPy function call,
            # the same shape the operators use, so indices stay reachable.
            idx = ",".join(i.name.lstrip(":") for i in node.indices)
            return f"{node.base.name}({idx})"
        if isinstance(node, FunctionCall) and isinstance(node.func, Symbol):
            args = ",".join(
                self._scalar_to_sympy(a, extra_vars) for a in node.args)
            return f"{node.func.name}({args})"
        if isinstance(node, Pow):
            return (f"({self._scalar_to_sympy(node.base, extra_vars)})"
                    f"**({self._scalar_to_sympy(node.exp, extra_vars)})")
        if isinstance(node, Add):
            return "(" + " + ".join(
                self._scalar_to_sympy(a, extra_vars) for a in node.args) + ")"
        if isinstance(node, (Mul, NonCommutativeMul)):
            return "*".join(
                self._scalar_to_sympy(a, extra_vars) for a in node.args)
        raise UnknownNotationError(
            "parser.unsupported_notation.sympy_scalar_coefficient",
            detail=type(node).__name__)

    def _find_first_op_base_name(self, node: Expr) -> Optional[str]:
        """Return the base symbol name of the first operator found (DFS)."""
        if isinstance(node, IndexedSymbol) and isinstance(node.base, Symbol):
            return node.base.name
        if isinstance(node, FunctionCall) and isinstance(node.func, Symbol):
            return node.func.name
        if isinstance(node, (Mul, NonCommutativeMul, Add)):
            for a in node.args:
                name = self._find_first_op_base_name(a)
                if name is not None:
                    return name
        if isinstance(node, IndexSum):
            return self._find_first_op_base_name(node.body)
        if isinstance(node, (Adjoint, Conjugate, Transpose)):
            return self._find_first_op_base_name(node.expr)
        return None

    def _detect_statistics(self, lhs: Expr) -> str:
        """Detect 'fermion' or 'boson' from the LHS of a commutator.

        Checks context first; falls back to heuristic (c/d/f → fermion, a/b → boson).
        """
        name = self._find_first_op_base_name(lhs)
        if name is not None:
            base = name.rstrip("'†")
            if self.context:
                prop = self.context.get(base)
                if prop and hasattr(prop, 'quantity'):
                    kind = getattr(prop.quantity, 'kind', None)
                    if kind in ("fermion", "boson"):
                        return kind
            if base.lower() in ("c", "d", "f"):
                return "fermion"
            if base.lower() in ("a", "b"):
                return "boson"
        return "fermion"  # conservative default for condensed matter
