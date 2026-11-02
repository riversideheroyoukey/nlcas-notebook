from typing import Optional
from notebook.runtime.code_generator import BaseCodeGenerator
from notebook.runtime.expr_generator import ExprGenerator
from notebook.models.symbol_context import SymbolContext

def _escape_prime(name: str) -> str:
    r"""Spell a PRIMED name so Maxima reads it as one atom: ``x'`` → ``x\'``.

    A prime standing on a symbol means "a different quantity of the same name"
    — ``G(\mathbf r t, \mathbf r' t')`` — and the resolver already reads it
    that way, producing ``Symbol("x'")``.  Only the CAS SPELLING was wrong:
    a bare quote is Maxima's noun-form operator, not an identifier character,
    so the command was a syntax error.  Measured 2026-08-20, `simplify` on
    ``x' + x`` came back as the adapter's own scratch name ``_res_`` **with
    status success**.

    Escaped, it is an ordinary atom — measured on live Maxima: ``subst``,
    ``diff``, ``integrate`` and ``ratsimp(x'^2 - x'*x')`` = 0 all behave — and
    ``tex1`` renders it ``{\it x'}``, the researcher's own notation, so the
    display round trip closes.  The noun-form quote is a PREFIX (``'diff``)
    and is never touched here, because this only ever sees a NAME.
    """
    # A LEADING quote is Maxima's noun-form operator and part of the name the
    # resolver produced (`'sum`, `'integrate`, `'limit`); escaping it turns the
    # held form into a symbol nobody can evaluate.  Only a quote that FOLLOWS
    # something is a prime.
    return name[:1] + name[1:].replace("'", "\\'")

from notebook.models.ir import (
    Expr, BracketExpr, Assignment, Commutator, Anticommutator,
    IndexSymbol, IndexedSymbol, KroneckerDelta, IndexSum, Integral, Limit,
    DotProduct, CrossProduct, Gradient, Divergence, Curl, Laplacian,
    Mul, NonCommutativeMul, FunctionCall, Symbol, Number, DiracDelta, Pow,
    Add
)

def product_operator(nodes, context: "SymbolContext | None") -> str:
    """`" . "` if ANY operand is matrix/vector/tensor typed, else `" * "`.

    **Maxima's `*` is ELEMENTWISE on matrices/vectors/tensors** — `A*B` for two
    matrices silently gives the elementwise product, not the matrix product
    (AUDIT-MATMUL-STAR).  `.` degrades to ordinary multiplication when either
    factor is scalar, so it is safe whenever any operand is non-scalar.

    **Two callers, one question.**  `MaximaCodeGenerator._mul_operator` uses it
    for a general `Mul` / `NonCommutativeMul` and (via
    `_noncommutative_product_op`) for an expanded Commutator/Anticommutator;
    `MaximaBackendHandler.build_command` uses it for the TEMPLATE path, where
    the join was a literal `*` in the backend spec.  Measured 2026-09-07:
    `[2,1;1,2]` and `[1,0;0,3]` joined by `multiply_expressions` gave
    `matrix([2,0],[0,6])` — the elementwise product — reported as `completed`,
    while `matrix_multiply` on the same pair gave the correct
    `matrix([2,3],[1,6])`.
    """
    if context is None:
        return " * "
    from notebook.runtime.semantic_resolver import get_algebra_type
    for node in nodes:
        if get_algebra_type(node, context) in ("matrix", "vector", "tensor"):
            return " . "
    return " * "


#: どの欄がその操作の2つのオペランドか。テンプレートは `backends/maxima.full.json`。
_TEMPLATE_PRODUCT_OPERANDS = {
    "multiply_expressions": ("a", "b"),
    "combine_expressions": ("expr_a", "expr_b"),
}


def template_product_operator(operation: str, ir_args: dict, raw_op: str,
                              context: "SymbolContext | None") -> "str | None":
    """テンプレート経路の結合子、または `None`（触らない）。

    `multiply_expressions` / `combine_expressions` は2つのオペランドを**テンプレートの
    リテラル `*`** で繋いでおり、そこは IR の `Mul` を通らないので生成器の判断が
    及ばなかった——`product_operator` の説明どおりの罠が、もう一段上で起きていた。
    `combine_expressions` は研究者の書いた演算子を運ぶので、`*` のときだけ判断する
    （`+`/`-` は行列でも要素ごとで正しく、`/` には選ぶべき行列演算子が無い）。
    """
    from notebook.models.ir import Expr
    keys = _TEMPLATE_PRODUCT_OPERANDS.get(operation)
    if keys is None:
        return None
    if operation == "combine_expressions" and str(raw_op or "").strip() != "*":
        return None
    nodes = [n for n in (ir_args.get(k) for k in keys) if isinstance(n, Expr)]
    return product_operator(nodes, context) if nodes else None


class MaximaCodeGenerator(ExprGenerator, BaseCodeGenerator):
    """Code Generator for Maxima backend.
    
    Translates semantically resolved AST nodes into Maxima compatible code strings:
    - BracketExpr is rendered as Maxima lists "[a, b]".
    - Assignment is rendered as LHS=RHS.
    - Commutator is expanded to its algebraic form "((A)*(B) - (B)*(A))" as a fallback.
    - Reuses the shared ExprGenerator logic for all other standard math expressions.
    """
    
    def __init__(self, context: Optional[SymbolContext] = None,
                 hold_integrals: bool = False):
        self.context = context
        # Emit Maxima's NOUN form `'integrate(...)` instead of `integrate(...)`.
        # Opening a vector into components must not EVALUATE as a side effect:
        # an expansion coefficient over an arbitrary basis
        # (`h_{mn} = \int d^3x \phi_m^* \hat h \phi_n`) is a DEFINITION and
        # cannot be evaluated at all, and asking Maxima to try it over all space
        # does not merely fail to simplify — measured 2026-08-08, it comes back
        # `ok=False` with an empty answer (Maxima asks a convergence question
        # the persistent adapter cannot answer).  `'(...)` around the whole
        # expression holds it too, but its `tex1` prints the WORD "integrate"
        # instead of an ∫, so the noun has to be per call.
        self.hold_integrals = hold_integrals

    def _mul_operator(self, nodes) -> str:
        """Product operator (spaced, for joining an n-ary product).

        The decision itself is :func:`product_operator` — the TEMPLATE path asks
        it too (`multiply_expressions` / `combine_expressions` join their two
        operands with a literal `*` in `backends/maxima.full.json`, which is the
        same elementwise trap one level up: MATMUL-TEMPLATE-STAR).
        """
        return product_operator(nodes, self.context)

    def _noncommutative_product_op(self, a_node: Expr, b_node: Expr) -> str:
        """Product operator for expanded commutators/anticommutators (R-4).

        Same non-scalar-operand test as ``_mul_operator``, but returns the
        historical unspaced/left-spaced forms ("*" / " . ") this call site's
        f-string ``f"({a}){op}({b})"`` already expects.
        """
        return " . " if self._mul_operator((a_node, b_node)) == " . " else "*"

    def _constrained_range(self, ir_ast) -> "tuple[str, str]":
        r"""The ONE range a constrained pair sum's two indices share.

        The indices of `\sum_{i<j}^{N}` are not two independent dummies with a
        bound written over one of them — the constraint is what ties them into a
        single range, and that is the discriminator the general multi-index case
        lacks (a superscript over `\sum_{m,n}` genuinely cannot be attributed
        without guessing, which is why `IndexSum.lower`/`upper` are not
        per-index).  A range the researcher WROTE wins; whichever end was left
        unwritten falls back to this sum's own placeholder pair, exactly as a
        rangeless ordinary sum does — a placeholder says "not stated", which is
        the honest answer, where a literal 1 would be inventing a physics
        choice.
        """
        _ph = (ir_ast.placeholder_bounds[0]
               if ir_ast.placeholder_bounds
               and len(ir_ast.placeholder_bounds[0]) == 2 else None)
        lo = (self.generate(ir_ast.lower) if ir_ast.lower is not None
              else (_ph[0] if _ph else None))
        hi = (self.generate(ir_ast.upper) if ir_ast.upper is not None
              else (_ph[1] if _ph else None))
        return lo, hi

    def _generate_constrained_sum(self, ir_ast) -> str:
        r"""Lower `\sum_{i REL j}` — see notebook/runtime/constrained_sum.py.

        Rendering only.  Which rewrite applies, and which constraints have none,
        is decided there and measured against a live Maxima.
        """
        from notebook.runtime.constrained_sum import classify_constrained
        from notebook.runtime.shared_parser import UnknownNotationError

        _fn = "product" if ir_ast.op == "product" else "sum"

        def _refuse(key: str):
            return UnknownNotationError(
                key,
                binder="Π" if ir_ast.op == "product" else "Σ",
                constraint=ir_ast.constraint,
                fn=_fn,
            )

        names = [self.generate(d) for d in ir_ast.dummy_indices]
        lo, hi = self._constrained_range(ir_ast)
        # Classify BEFORE looking at the range, so the REASON a notation is
        # refused never depends on whether the placeholder pass happened to
        # have run over this IR.  Written the other way round, adjacency and a
        # chain both came back as the generic message when called from a layer
        # that does not fill ranges — the reason is what the researcher reads,
        # and it must be a property of the notation alone.
        plan = classify_constrained(ir_ast.constraint, names,
                                    lo or "", hi or "", ir_ast.op)
        if plan.kind == "refuse":
            raise _refuse(plan.i18n_key)
        if lo is None or hi is None:
            # Maxima's `sum` takes FOUR arguments, so there is no shape to emit
            # without a range.  Reached only when the placeholder pass has not
            # run over this IR; the registration path always fills one.
            raise _refuse("parser.unsupported_notation.constrained_sum")
        plan = classify_constrained(ir_ast.constraint, names, lo, hi, ir_ast.op)

        outer, inner = names[0], names[1]
        body = self.generate(ir_ast.body)
        if plan.kind == "nested":
            return (f"{_fn}({_fn}({body}, {inner}, {plan.inner_lo}, "
                    f"{plan.inner_hi}), {outer}, {plan.outer_lo}, "
                    f"{plan.outer_hi})")

        # minus_diagonal: Σ_{i≠j} B(i,j) = Σ_i Σ_j B(i,j) − Σ_i B(i,i).  The
        # diagonal is the body with the INNER index renamed to the outer one,
        # done on the IR through the shared substitution rather than on the
        # generated text — a textual swap would also hit an `j` inside a name.
        from notebook.models.ir import Symbol
        from notebook.runtime.index_contractor import substitute_symbol_or_index
        diag = self.generate(
            substitute_symbol_or_index(ir_ast.body, inner, Symbol(name=outer)))
        return (f"{_fn}({_fn}({body}, {inner}, {lo}, {hi}), {outer}, {lo}, {hi})"
                f" - {_fn}({diag}, {outer}, {lo}, {hi})")


    def _is_matrix_power(self, node) -> bool:
        """``A^{n}`` (n an integer) where ``A`` is declared a matrix / vector /
        tensor — the NON-COMMUTATIVE power, not the element-wise one."""
        from notebook.models.ir import Number
        from notebook.runtime.semantic_resolver import get_algebra_type
        exp = node.exponent
        # ANY integer power, not only -1: `matrix([1,2],[3,4])^2` is
        # `matrix([1,4],[9,16])` — each entry squared — while `^^2` is the
        # matrix square `matrix([7,10],[15,22])` (measured).
        if not (isinstance(exp, Number) and float(exp.value).is_integer()):
            return False
        if self.context is None:
            return False
        return get_algebra_type(node.base, self.context) in (
            "matrix", "vector", "tensor")

    def generate(self, ir_ast: Expr) -> str:
        if isinstance(ir_ast, BracketExpr):
            args_str = ", ".join(self.generate(arg) for arg in ir_ast.args)
            return f"[{args_str}]"

        elif isinstance(ir_ast, Assignment):
            return f"{self.generate(ir_ast.lhs)}={self.generate(ir_ast.rhs)}"

        elif isinstance(ir_ast, (Mul, NonCommutativeMul)):
            return self._generate_mul(ir_ast, operator=self._mul_operator(ir_ast.args))

        elif isinstance(ir_ast, Commutator):
            a = self.generate(ir_ast.a)
            b = self.generate(ir_ast.b)
            op = self._noncommutative_product_op(ir_ast.a, ir_ast.b)
            return f"(({a}){op}({b}) - ({b}){op}({a}))"

        elif isinstance(ir_ast, Anticommutator):
            a = self.generate(ir_ast.a)
            b = self.generate(ir_ast.b)
            op = self._noncommutative_product_op(ir_ast.a, ir_ast.b)
            return f"(({a}){op}({b}) + ({b}){op}({a}))"

        elif isinstance(ir_ast, DotProduct):
            return f"(({self.generate(ir_ast.left)}) . ({self.generate(ir_ast.right)}))"
            
        elif isinstance(ir_ast, CrossProduct):
            return f"(({self.generate(ir_ast.left)}) ~ ({self.generate(ir_ast.right)}))"
            
        elif isinstance(ir_ast, Gradient):
            return f"grad({self.generate(ir_ast.expr)})"
            
        elif isinstance(ir_ast, Divergence):
            return f"div({self.generate(ir_ast.expr)})"
            
        elif isinstance(ir_ast, Curl):
            return f"curl({self.generate(ir_ast.expr)})"
            
        elif isinstance(ir_ast, Laplacian):
            return f"laplacian({self.generate(ir_ast.expr)})"

        elif isinstance(ir_ast, DiracDelta):
            # Maxima's own `delta` — measured: `laplace(delta(t-3), t, s)` is
            # `%e^-(3*s)`, so the sifting property is there for the transform.
            # (`integrate` does NOT sift; that is the CAS's boundary, not this
            # notation's.)  `y = 0` is the plain one-argument spelling.
            _x = self.generate(ir_ast.x)
            if isinstance(ir_ast.y, Number) and ir_ast.y.value == 0:
                return f"delta({_x})"
            return f"delta({_x} - {self.generate(ir_ast.y)})"

        elif isinstance(ir_ast, Pow) and self._is_matrix_power(ir_ast):
            # A DECLARED matrix's `A^{-1}` is the inverse.  Maxima's `^` is
            # ELEMENT-WISE: measured, `matrix([1,2],[3,4])^-1` is
            # `matrix([1,1/2],[1/3,1/4])` — the reciprocal of each entry, with
            # no error.  The non-commutative power is the inverse and stays
            # symbolic (`A^^-1 . A` simplifies to 1), which is what a Dyson
            # equation needs.  Same discriminator as the `^T` and `^\dagger`
            # branches in the resolver: the declaration decides.  (2026-08-21)
            # A compound base MUST be parenthesised, for the same reason
            # `_generate_pow` does it: `^` is right-associative, so
            # `(1 - G^0\Sigma)^{-1}` emitted bare became
            # `1 - G_sup0 . Sigma^^-1` — the inverse bound to Sigma and the
            # grouping gone (measured 2026-08-21, while writing the Dyson
            # equation's general solution).  This branch had skipped the rule
            # its scalar sibling already carried.
            _b = self.generate(ir_ast.base)
            if isinstance(ir_ast.base, (Add, Mul, NonCommutativeMul, Pow)):
                _b = f"({_b})"
            return f"{_b}^^{self.generate(ir_ast.exponent)}"

        elif isinstance(ir_ast, Symbol):
            return _escape_prime(ir_ast.name)

        elif isinstance(ir_ast, IndexSymbol):
            # An INDEX can be primed too (`c_{k',\sigma}` — a second momentum),
            # and it reaches the CAS by the same route as a name.
            return _escape_prime(ir_ast.name)

        elif isinstance(ir_ast, FunctionCall) and isinstance(ir_ast.func, IndexedSymbol):
            # An APPLIED indexed quantity is Maxima's own "subscripted function"
            # (manual: "one of a family of subscripted functions selected by a
            # particular subscript value") — `D[i,j](x)`.  Brackets regardless of
            # declaration, and measured on real Maxima 2026-07-31: no side effect
            # when undefined (`arrays` stays empty), `subst` reaches the index
            # (`D[1,j](x)`), a sum over the index expands term by term, and
            # `tex1` gives back `D_{i,j}(x)` — the researcher's own notation, so
            # the display needs no bridge and the round trip closes.
            #
            # The alternative `D(i,j)(x)` also evaluates in Maxima, but its
            # `tex1` is `D\left(i , j\right)(x)` and re-reading THAT yields a
            # PRODUCT — the same silent break as AUDIT-DERIV-DISPLAY-RT.
            idx = ", ".join(self.generate(i) for i in ir_ast.func.indices)
            args = ", ".join(self.generate(a) for a in ir_ast.args)
            return f"{self.generate(ir_ast.func.base)}[{idx}]({args})"

        elif isinstance(ir_ast, IndexedSymbol):
            base_str = self.generate(ir_ast.base)

            # Default style in Maxima is "function_call" (e.g. x(i))
            style = "function_call"
            
            if self.context:
                # Look the base up by the name in the IR, not by the emitted
                # code: the emitted form escapes a primed name for Maxima
                # (`a\'`), and rstrip("'") on that leaves a trailing backslash.
                clean_name = (ir_ast.base.name if isinstance(ir_ast.base, Symbol)
                              else base_str).rstrip("'†")
                prop = self.context.get(clean_name)
                if prop:
                    explicit_style = getattr(prop.quantity, "index_lowering_style", None)
                    if explicit_style and explicit_style != "backend_default":
                        style = explicit_style
                    else:
                        algebra = prop.quantity.math_type.algebra
                        if algebra in ("vector", "matrix", "tensor"):
                            style = "array_access"
            
            if style == "array_access":
                indices_str = ", ".join(self.generate(idx) for idx in ir_ast.indices)
                return f"{base_str}[{indices_str}]"
            elif style == "index_alias":
                # docs/07 §8.2: lower an indexed quantity under a DISTINCT name so
                # it cannot collide with a genuine function call of the same base
                # (V_i and V(x) are both V(...) otherwise).  Measurement probe.
                indices_str = ", ".join(self.generate(idx) for idx in ir_ast.indices)
                return f"{base_str}__idx({indices_str})"
            elif style == "atom":
                joined_indices = "_".join(self.generate(idx) for idx in ir_ast.indices)
                return f"{base_str}_{joined_indices}"
            else:
                # Default function_call
                indices_str = ", ".join(self.generate(idx) for idx in ir_ast.indices)
                return f"{base_str}({indices_str})"

        elif isinstance(ir_ast, KroneckerDelta):
            left = self.generate(ir_ast.left)
            right = self.generate(ir_ast.right)
            return f"kron_delta({left}, {right})"

        elif isinstance(ir_ast, IndexSum):
            # One nested sum(...) per dummy, outermost = the first index the
            # researcher wrote (reading order, the OUTER→INNER convention
            # integral_descriptor uses).  Measured in Maxima: the nesting order
            # does not change the result, and the CAS lifts index-independent
            # factors out of the inner sum on its own.
            if ir_ast.constraint:
                # `\sum_{i \ne j}`, `\sum_{i<j}`, `\sum_{\langle i,j \rangle}`.
                # Read here so the researcher can write it and see it; what it
                # LOWERS to (or why it cannot) is `constrained_sum`'s decision,
                # not this branch's — the same split `subscript_policy` and
                # `julia_index_policy` take.
                #
                # This used to refuse the whole family, on the ground that
                # Maxima's `sum` has no place for a condition.  True, and the
                # wrong conclusion: a constraint between two INTEGER indices is
                # a pair of RANGES, which `sum` has every place for.  Unread,
                # the condition still became one of the indices and the CAS
                # summed over a dummy named `ne` (measured 2026-08-09), so the
                # reading stays; only the refusal narrowed.
                return self._generate_constrained_sum(ir_ast)
            body = self.generate(ir_ast.body)
            res = body
            # The CAS's noun mark stays ON the name when the sum came back
            # quoted: unquoted, `sum(...)` asks the CAS to EVALUATE, which is a
            # different request from the one it answered with.
            _quote = "'" if getattr(ir_ast, "noun", False) else ""
            # Maxima spells Π `product`, with the same four-argument shape as
            # `sum` — so the range machinery below (written range, placeholder
            # pair) needs no branch of its own.
            _fn = _quote + ("product" if ir_ast.op == "product" else "sum")
            _lo = ir_ast.lower is not None and ir_ast.upper is not None
            for idx in reversed(ir_ast.dummy_indices):
                dummy_str = self.generate(idx)
                if _lo and len(ir_ast.dummy_indices) == 1:
                    # Maxima's sum takes FOUR arguments; the two-argument form
                    # is rejected outright.  A range the researcher wrote is
                    # what makes the four-argument form available.
                    res = (f"{_fn}({res}, {dummy_str}, "
                           f"{self.generate(ir_ast.lower)}, "
                           f"{self.generate(ir_ast.upper)})")
                else:
                    # A sum written WITHOUT a range still has to become a CAS
                    # object, and Maxima's `sum` needs four arguments.  The
                    # placeholder pair was allocated once, at registration, and
                    # travels in the IR — so this sum keeps its own range and is
                    # never identified with another that merely also lacks one.
                    _pos = ir_ast.dummy_indices.index(idx)
                    _b = (ir_ast.placeholder_bounds[_pos]
                          if _pos < len(ir_ast.placeholder_bounds) else None)
                    res = (f"{_fn}({res}, {dummy_str}, {_b[0]}, {_b[1]})"
                           if _b and len(_b) == 2 else f"{_fn}({res}, {dummy_str})")
            return res
        elif isinstance(ir_ast, Limit):
            # Maxima's own spelling, including the SIDE.  `limit(f, x, a, plus)`
            # is a different question from `limit(f, x, a)` — measured,
            # `limit(abs(x)/x, x, 0)` is `ind` while the two sides are 1 and -1.
            body = self.generate(ir_ast.expr)
            var = self.generate(ir_ast.var)
            point = self.generate(ir_ast.point)
            if ir_ast.direction:
                return f"limit({body}, {var}, {point}, {ir_ast.direction})"
            return f"limit({body}, {var}, {point})"
        elif isinstance(ir_ast, Integral):
            if ir_ast.measure_dim:
                # `\int d^3k …` — read, displayable, and composable, but there is
                # no d-dimensional integral to hand Maxima.  Refused HERE rather
                # than at the parser so the researcher can still write it, see
                # it, and expand the vector into components; left unread it came
                # back as the product `(d³·int·k·f(k))/(8π³)`, a plausible answer
                # to a question nobody asked (measured 2026-08-08).
                from notebook.runtime.shared_parser import UnknownNotationError
                raise UnknownNotationError(
                    "parser.unsupported_notation.multidim_measure",
                    measure=f"d^{ir_ast.measure_dim}{ir_ast.var.name}",
                    var=ir_ast.var.name,
                )
            body = self.generate(ir_ast.expr)
            var = self.generate(ir_ast.var)
            name = "'integrate" if self.hold_integrals else "integrate"
            if ir_ast.lower is not None and ir_ast.upper is not None:
                lower = self.generate(ir_ast.lower)
                upper = self.generate(ir_ast.upper)
                return f"{name}({body}, {var}, {lower}, {upper})"
            return f"{name}({body}, {var})"
            
        return super().generate(ir_ast)
