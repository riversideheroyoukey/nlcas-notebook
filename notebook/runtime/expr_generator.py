"""Backend-agnostic IR AST → CAS string renderer."""
from __future__ import annotations

from notebook.models.ir import Expr, Symbol, Number, Add, Mul, NonCommutativeMul, Pow, FunctionCall, Adjoint, Conjugate, Transpose, Subscript, Superscript, Decorated

class ExprGenerator:
    """Converts IR AST nodes to generic CAS-compatible string form.

    The output syntax (``f(x)``, ``a + b``, ``a * b``, ``a^b``) is shared by
    both Maxima and Julia, so this class is used by the common CommandBuilder
    base without creating a dependency on any specific backend.
    """

    def generate(self, ir_ast: Expr) -> str:
        if isinstance(ir_ast, Symbol):
            return ir_ast.name
        if isinstance(ir_ast, Number):
            v = ir_ast.value
            return str(int(v)) if isinstance(v, float) and v == int(v) else str(v)
        if isinstance(ir_ast, Add):
            return self._generate_add(ir_ast)
        if isinstance(ir_ast, (Mul, NonCommutativeMul)):
            return self._generate_mul(ir_ast)
        if isinstance(ir_ast, Pow):
            return self._generate_pow(ir_ast)
        if isinstance(ir_ast, Decorated):
            base_str = self.generate(ir_ast.base)
            return f"{base_str}_{ir_ast.decoration}"
        if isinstance(ir_ast, Adjoint):
            return f"ctranspose({self.generate(ir_ast.expr)})"
        if isinstance(ir_ast, Conjugate):
            return f"conjugate({self.generate(ir_ast.expr)})"
        if isinstance(ir_ast, Transpose):
            return f"transpose({self.generate(ir_ast.expr)})"
        if isinstance(ir_ast, FunctionCall):
            func_name = self.generate(ir_ast.func)
            args_str = ", ".join(self.generate(arg) for arg in ir_ast.args)
            return f"{func_name}({args_str})"

        if isinstance(ir_ast, Subscript):
            return f"{self.generate(ir_ast.base)}_{self.generate(ir_ast.sub)}"
        if isinstance(ir_ast, Superscript):
            base_str = self.generate(ir_ast.base)
            script_str = self.generate(ir_ast.script)
            if isinstance(ir_ast.script, (Add, Mul, Pow, Subscript, Superscript)):
                return f"{base_str}^({script_str})"
            return f"{base_str}^{script_str}"

        from notebook.models.ir import Equation, Derivative, OpaqueExpr
        if isinstance(ir_ast, Equation):
            if ir_ast.op == "≈":
                # An approximation has no CAS form.  It is READ (both sides are
                # ordinary expressions, and the notebook must be able to write
                # `G^{(2)} ≈ …` — that line is the whole content of a decoupling)
                # but it cannot be COMPUTED WITH: handed to Maxima the relation
                # became an ordinary factor and `simplify` answered
                # `A*B*approx+C` as a success.  Refusing here is the point where
                # the researcher is told, with the two sides still on screen.
                from notebook.runtime.shared_parser import UnknownNotationError
                raise UnknownNotationError("generator.approx_no_cas_form")
            # Maxima's own ASCII relational syntax (<=, >=, <, >, =) is
            # identical to the IR's, so no per-op translation is needed here.
            return f"{self.generate(ir_ast.lhs)}{ir_ast.op}{self.generate(ir_ast.rhs)}"
        if isinstance(ir_ast, Derivative):
            return self._generate_derivative(ir_ast)
        if isinstance(ir_ast, OpaqueExpr):
            return ir_ast.text

        # No silent fallback: str(node) here used to leak the pydantic repr
        # ("simplification_log=[] a=Symbol(...)") into canonicals and CAS
        # commands (e.g. for A[i,j] / E[X] inputs).  Unsupported IR nodes must
        # fail loudly so the caller reports a clear error instead of storing
        # a corrupted expression.
        from notebook.runtime.shared_parser import UnknownNotationError
        raise UnknownNotationError(
            "generator.unsupported_ir_node", node=type(ir_ast).__name__)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_neg1(self, node: Expr) -> bool:
        return isinstance(node, Number) and node.value == -1

    def _generate_add(self, node: Add) -> str:
        if not node.args:
            return "0"
        result = self.generate(node.args[0])
        for arg in node.args[1:]:
            s = self.generate(arg)
            if s.startswith("-"):
                result += f" - {s[1:]}"
            else:
                result += f" + {s}"
        return result

    def _generate_mul(self, node: Mul, operator: str = " * ") -> str:
        if not node.args:
            return "1"

        def _part(arg) -> str:
            s = self.generate(arg)
            # Parenthesise Add sub-expressions inside Mul
            return f"({s})" if isinstance(arg, Add) else s

        # A leading -1 coefficient is emitted as the SIGN, at any arity.
        # Restricting this to exactly two args made the boundary
        # non-idempotent: Mul(-1, 4, g) rendered as "-1 * 4 * g", which
        # _generate_add then turned into "a - 1 * 4 * g" by dropping only the
        # sign character — so re-parsing gave Mul(-1, 1, 4, g) and every further
        # round trip grew another "1 * " (measured over the saved-session corpus,
        # docs/32 Phase 5).  Harmless arithmetically, unbounded in size.
        if len(node.args) >= 2 and self._is_neg1(node.args[0]):
            rest = [_part(a) for a in node.args[1:]]
            return f"-{operator.join(rest)}"

        return operator.join(_part(arg) for arg in node.args)

    def _generate_pow(self, node: Pow) -> str:
        base = self.generate(node.base)
        exp = self.generate(node.exponent)
        # Parenthesise compound bases/exponents to avoid ambiguity.
        #
        # A Pow BASE must be parenthesised too, because ``^`` is RIGHT-
        # associative in both target CAS languages: division is emitted as
        # ``base^-1``, so ``k/r^2`` became ``k * r^2^-1`` — read as
        # ``r^(2^-1)`` = ``sqrt(r)``.  No error anywhere; ``expand(k/r^2)``
        # simply returned ``k*sqrt(r)``.  Found 2026-07-30 via the central-force
        # scenario, whose energy conservation stopped being 0 (docs/32 §8).
        # A Pow EXPONENT needs no parens — right associativity already gives
        # ``a^b^c`` = ``a^(b^c)``, which is what the nested node means.
        if isinstance(node.base, (Add, Mul, Pow)):
            base = f"({base})"
        if isinstance(node.exponent, (Add, Mul)):
            exp = f"({exp})"
        return f"{base}^{exp}"

    def _generate_derivative(self, node: "Derivative") -> str:  # type: ignore[name-defined]
        expr_s = self.generate(node.expr)
        var_parts = []
        for sym, order in node.vars:
            var_parts.append(f"{sym.name},{order}")
        vars_s = ",".join(var_parts)
        return f"'diff({expr_s},{vars_s})"
