import re as _re
from typing import Optional
from notebook.runtime.backend_handler import MATH_FUNC_NAMES
from notebook.runtime.julia_index_policy import classify_indexed
from notebook.runtime.julia_undefined_mixin import JuliaUndefinedSymbolMixin as _JUM
from notebook.runtime.julia_sympy_serialize import JuliaSympySerializeMixin
from notebook.runtime.code_generator import BaseCodeGenerator
from notebook.runtime.spin_label_canon import (canon_spin_label, qa_index_literal,
                                                qa_join_index_symbols)
from notebook.runtime.expr_generator import ExprGenerator
from notebook.runtime.finite_sum_expansion import expand_finite_index_sum
from notebook.runtime.spin_value_set import expand_spin_index_sum
from notebook.models.symbol_context import SymbolContext
from notebook.models.ir import (
    Expr, BracketExpr, Assignment, Commutator, Anticommutator, Equation,
    KroneckerDelta, DiracDelta, IndexSum, Integral, Limit, Symbol, Number,
    IndexSymbol, IndexedSymbol, Adjoint, Conjugate, Transpose,
    DotProduct, CrossProduct, Gradient, Divergence, Curl, Laplacian,
    Mul, NonCommutativeMul, Add, FunctionCall, Pow,
)

#: Single letters this backend treats as quantum operators without a
#: declaration: the second-quantization convention (a/b bosons, c/d/f
#: fermions), plus the two QuantumAlgebra exports into Main.  ONE definition
#: because two code paths need it — the subscript form (`c_k`) and the applied
#: form (`c(1)`) — and stating it twice is how the applied form ended up
#: refusing `c(1)` while the subscript form accepted `c_1`.
_QA_OPERATOR_LETTERS: frozenset[str] = (
    frozenset({"a", "b", "c", "d"}) | _JUM._QA_BUILTIN_OP_NAMES)


#: How a bound index is written inside the body, and how the enclosing sum
#: binds it.  Kept together because the scan below must tell the two apart.
_QA_SUM_BINDER = "_QA.∑("
_QA_BOUND_INDEX_RE = _re.compile(r":([A-Za-z_]\w*)")


def _first_occurrence_order(code: str, dummies: "list[str]") -> "list[str]":
    r"""The dummy names in the order QuantumAlgebra will number them.

    QA replaces a summed index with `#ᵢ` numbered by FIRST OCCURRENCE IN THE
    BODY — measured 2026-08-07 with marker parameters, and unchanged by
    `normal_form`.  The binder positions (`_QA.∑(:m, …)`) are skipped: they are
    where the name is BOUND, not where it occurs.

    Any dummy that does not occur in the body at all is appended in binding
    order, so the returned list always names every one.
    """
    order: "list[str]" = []
    for m in _QA_BOUND_INDEX_RE.finditer(code):
        if code[max(0, m.start() - len(_QA_SUM_BINDER)):m.start()] == _QA_SUM_BINDER:
            continue                      # the binder, not an occurrence
        name = m.group(1)
        if name in dummies and name not in order:
            order.append(name)
    order.extend(d for d in dummies if d not in order)
    return order


#: 添字が**整数サイト**か——`str.isdigit()` ではなく ASCII の数字だけを見る。
#:
#: `isdigit()` は上付き数字にも True を返す（`'⁰'.isdigit()` は True、
#: `isdecimal()` は False）。整数スピンの射影 $m=0$ を `⁰` と綴った瞬間、
#: この枝が「整数サイトだから Julia の整数リテラルで出す」と判断し、
#: `m'(1, ⁰)` という**未定義の識別子**が CAS に届いて生の `ParseError:` が
#: 返っていた（実測 2026-09-12。`⁺`/`⁻` は isdigit が False なので正しく
#: `QuIndex("⁺")` に包まれ、**3値の集合だけが落ちる**という形だった）。
def _IS_INTEGER_SITE(text: str) -> bool:
    return bool(_ASCII_INT_RE.fullmatch(text))


_ASCII_INT_RE = __import__("re").compile(r"[0-9]+")


class JuliaCodeGenerator(JuliaSympySerializeMixin, ExprGenerator, BaseCodeGenerator):
    """Code Generator for Julia (QuantumAlgebra.jl / Symbolics.jl) backend.

    Translates semantically resolved AST nodes into Julia compatible code strings:
    - Commutator is rendered using QuantumAlgebra's "comm(a, b)" function.
    - BracketExpr is rendered as comm(a, b) or anticommutator products.
    - Assignment is rendered as normal assignment or operator declarations.
    - Equation is rendered using Symbolics.jl equation operator "LHS ~ RHS".
    - Reuses the shared ExprGenerator logic for all other standard math expressions.

    ## SymPy routing for algebraic momentum indices (IMPLEMENTED)
    #
    # When a Commutator contains algebraic indices (e.g. -k, k+q, a*k+b),
    # QuantumAlgebra.jl would crash because QuIndex requires single-char symbols.
    # The SymPy engine (_nb_sympy_fermion_comm / _nb_sympy_boson_comm) handles
    # these correctly; this class automatically routes to it via:
    #   generate(Commutator) → _has_algebraic_indices() → _generate_sympy_commutator()
    #
    # The result is a Julia Tuple{String,String} (value_str, latex_str).
    # The command loop in _julia_driver.jl detects this type and prints
    # element [1] as the value and element [2] as the LaTeX — no changes
    # needed in JuliaBackendHandler or engine.py.
    #
    # IndexSum-wrapped Hamiltonians are handled via _generate_sympy_sum_commutator,
    # which calls _nb_sympy_sum_fermion_comm / _nb_sympy_sum_boson_comm on the
    # Julia side. These are implemented and auto-routed via _has_algebraic_indices.
    #
    # Reference: docs/developer_manuals/04_second_quantization_engine_design.md
    """

    def __init__(self, context: Optional[SymbolContext] = None):
        self.context = context
        self._bound_indices: list[str] = []
        # Maps expression-ID → user-input canonical string (set by BackendHandler
        # from engine's __input__<eid> entries in expr_lookup).
        self.input_lookup: dict[str, str] = {}
        # When True: IndexSum bodies are inlined (no _nb_eval_index_sum) and
        # bound indices use _QA.QuIndex("name")) instead of sumindex(n).
        # Used by operator_substitute to avoid any sumindex dependency.
        self.static_index_mode: bool = False
        # While rendering the BODY of an IndexSum that will be wrapped in
        # QuantumAlgebra's own `∑`: bound indices are written as the plain
        # symbol `:name`, because `∑(:name, body)` is what binds them.
        self._qa_sum_mode: bool = False
        # Filled by the outermost IndexSum with the dummy names in the order
        # QuantumAlgebra will number them (`#₁`, `#₂`, …).  The builder reads it
        # so the display can name them back; empty when nothing was summed.
        self.sum_dummy_order: "list[str]" = []
        # Set while in _qa_sum_mode if a bound index appeared NEGATED (`-k`).
        # `∑` cannot express that — `-:k` is a MethodError, and handing it
        # `sumindex(-1)` yields `#₂`, i.e. a DIFFERENT summation index
        # (measured 2026-08-07).  The old string-eval helper stays in charge of
        # that case rather than being silently mistranslated.
        self._qa_sum_saw_negative: bool = False

    def generate(self, ir_ast: Expr) -> str:
        if isinstance(ir_ast, FunctionCall) and isinstance(ir_ast.func, IndexedSymbol):
            # An indexed quantity APPLIED to arguments (`a_{i,\sigma}(k)`).  Maxima
            # has a shape for this (a subscripted function, `a[i,sigma](k)`);
            # QuantumAlgebra does not — an operator takes indices and the RESULT
            # is a QuExpr, which is not callable.  Measured 2026-07-31:
            # `f(QuIndex("i"), QuIndex("s"))(k)` → "MethodError: objects of type
            # QuExpr are not callable".  Refuse with the fix rather than let that
            # reach the researcher (Core Principle #1).  The index count is
            # irrelevant — a single index fails the same way.
            from notebook.runtime.shared_parser import UnknownNotationError
            _base = getattr(ir_ast.func.base, "name", "")
            _idx = ", ".join(getattr(i, "name", str(i)) for i in ir_ast.func.indices)
            raise UnknownNotationError(
                "parser.unsupported_notation.julia_applied_indexed",
                name=f"{_base}_{{{_idx}}}", base=_base, index=_idx)

        if isinstance(ir_ast, Commutator):
            # Try to expand expression references before routing.
            expanded = self._maybe_expand_commutator(ir_ast)
            if expanded is not None:
                return expanded
            if self._has_algebraic_indices(ir_ast):
                return self._generate_sympy_commutator(ir_ast)
            a = self.generate(ir_ast.a)
            b = self.generate(ir_ast.b)
            return f"comm({a}, {b})"
            
        elif isinstance(ir_ast, Anticommutator):
            a = self.generate(ir_ast.a)
            b = self.generate(ir_ast.b)
            return f"(({a})*({b}) + ({b})*({a}))"
            
        elif isinstance(ir_ast, BracketExpr):
            if len(ir_ast.args) != 2:
                raise ValueError("Bracket expressions in Julia must have exactly 2 arguments")
            a = self.generate(ir_ast.args[0])
            b = self.generate(ir_ast.args[1])
            if ir_ast.kind == "square":
                return f"comm({a}, {b})"
            elif ir_ast.kind == "curly":
                return f"(({a})*({b}) + ({b})*({a}))"
            else:
                raise ValueError(f"Unknown BracketExpr kind: {ir_ast.kind}")
                
        elif isinstance(ir_ast, Assignment):
            if ir_ast.is_operator_def:
                ops_macro = "@fermion_ops" if ir_ast.operator_kind == "fermion" else "@boson_ops"
                return f"{ops_macro} {self.generate(ir_ast.lhs)}; nothing"
            return f"{self.generate(ir_ast.lhs)} = {self.generate(ir_ast.rhs)}"
            
        elif isinstance(ir_ast, Equation):
            # Symbolics.jl uses ~ for equations. Deliberately deferred gap:
            # this hardcodes "~" regardless of ir_ast.op, so an inequality
            # (op="<=" etc.) would render as "~" too — Julia/2nd-quantization
            # sessions are out of scope for the Maxima-only inequality feature
            # (see notebook/runtime/expr_generator.py's Equation branch).
            return f"{self.generate(ir_ast.lhs)} ~ {self.generate(ir_ast.rhs)}"

        elif isinstance(ir_ast, Adjoint):
            inner = ir_ast.expr
            if isinstance(inner, Symbol) and self.context:
                prop = self.context.get(inner.name)
                if prop and prop.quantity.kind == "operator":
                    return f"{inner.name}'"
            return f"adjoint({self.generate(inner)})"
            
        elif isinstance(ir_ast, DotProduct):
            return f"dot({self.generate(ir_ast.left)}, {self.generate(ir_ast.right)})"
            
        elif isinstance(ir_ast, CrossProduct):
            return f"cross({self.generate(ir_ast.left)}, {self.generate(ir_ast.right)})"
            
        elif isinstance(ir_ast, Gradient):
            return f"grad({self.generate(ir_ast.expr)})"
            
        elif isinstance(ir_ast, Divergence):
            return f"div({self.generate(ir_ast.expr)})"
            
        elif isinstance(ir_ast, Curl):
            return f"curl({self.generate(ir_ast.expr)})"
            
        elif isinstance(ir_ast, Laplacian):
            return f"laplacian({self.generate(ir_ast.expr)})"
            
        elif isinstance(ir_ast, Conjugate):
            # Julia autodeclares unknowns as real Symbolics variables, making
            # conj(X) = X silently.  Use a "XStar" symbol instead so the
            # conjugate is preserved as a distinct symbol in the expression.
            # clean_latex() converts DeltaStar → \Delta^{\ast} in display.
            #
            # Conjugate(Conjugate(x)) = x: strip double conjugate at IR level.
            inner = ir_ast.expr
            if isinstance(inner, Conjugate):
                return self.generate(inner.expr)
            if isinstance(inner, Symbol):
                return f"{inner.name}Star"
            if isinstance(inner, IndexedSymbol) and isinstance(inner.base, Symbol):
                indices_code = ", ".join(self.generate(idx) for idx in inner.indices)
                return f"{inner.base.name}Star({indices_code})"
            return f"conj({self.generate(inner)})"
            
        elif isinstance(ir_ast, Transpose):
            return f"transpose({self.generate(ir_ast.expr)})"

        elif isinstance(ir_ast, KroneckerDelta):
            left = self.generate(ir_ast.left)
            right = self.generate(ir_ast.right)
            return f"_QA.δ({left}, {right})"

        elif isinstance(ir_ast, DiracDelta):
            x = self.generate(ir_ast.x)
            y = self.generate(ir_ast.y)
            return f"SymPy.sympy.DiracDelta({x} - {y})"

        elif isinstance(ir_ast, IndexSymbol):
            # Support negated indices like "-k" produced from c_{-k}.
            # Strip the leading minus so we can look up the inner name.
            # Also strip leading colon added by JuliaSemanticResolver for operator args.
            name = ir_ast.name
            if name.startswith(":") and len(name) > 1:
                name = name[1:]
            neg = name.startswith("-")
            inner = name[1:] if neg else name
            if inner in self._bound_indices:
                if self.static_index_mode:
                    # Static mode: use QuIndex directly, no sumindex needed.
                    return qa_index_literal(inner, neg=neg)
                if self._qa_sum_mode:
                    # The enclosing `∑(:name, …)` binds it; write the symbol.
                    if neg:
                        self._qa_sum_saw_negative = True
                    else:
                        return f":{inner}"
                # Normal mode: use _QA.sumindex (module alias, avoids
                # any shadowing of the QuantumAlgebra name in the Main scope).
                pos = self._bound_indices.index(inner) + 1
                if neg:
                    return f"_QA.sumindex(-{pos})"
                return f"_QA.sumindex({pos})"
            else:
                if neg:
                    if _IS_INTEGER_SITE(inner):
                        return f"(-{inner})"
                    return qa_index_literal(inner, neg=True)
                # Pure integer index → integer literal for full QA simplification
                if _IS_INTEGER_SITE(inner):
                    return inner
                return qa_index_literal(inner)

        elif isinstance(ir_ast, IndexedSymbol):
            # The DECISION lives in julia_index_policy (see its docstring for the
            # rule and the measurements behind it); rendering stays here, because
            # only the generator owns `generate`.
            base_str = self.generate(ir_ast.base)
            _names = [getattr(i, "name", "") for i in ir_ast.indices]
            plan = classify_indexed(base_str, _names, self.context,
                                    self._bound_indices, _QA_OPERATOR_LETTERS)

            if plan.kind == "atom_full":
                return plan.value
            if plan.kind == "refuse":
                from notebook.runtime.shared_parser import UnknownNotationError
                raise UnknownNotationError(plan.i18n_key, **plan.fmt)
            if plan.kind == "quantum_index":
                return f"{base_str}({', '.join(self.generate(i) for i in ir_ast.indices)})"
            if plan.kind == "array":
                return f"{base_str}[{', '.join(_names)}]"
            if plan.kind == "atom":
                return f"{base_str}_{'_'.join(_names)}"

            # param: a c-number carrying an index.  Bound indices keep the
            # `:name` spelling so `_nb_eval_index_sum` can replace them with
            # sumindex(n) by name; everything else is rendered as an ordinary
            # index literal (which enforces QA's 1-char-plus-subindex rule).
            parts: list[str] = []
            for idx in ir_ast.indices:
                iname = getattr(idx, "name", "")
                if isinstance(idx, IndexSymbol) and iname.startswith("-"):
                    inner = iname[1:]
                    if inner in self._bound_indices and not self.static_index_mode:
                        parts.append(
                            f"_QA.sumindex(-{self._bound_indices.index(inner) + 1})")
                    else:
                        parts.append(qa_index_literal(inner, neg=True))
                elif iname in self._bound_indices and not self.static_index_mode:
                    parts.append(f":{iname}")
                else:
                    parts.append(self.generate(idx))
            return f'param(:{base_str}, {", ".join(parts)})'

        elif isinstance(ir_ast, Limit):
            # **Write it down first, compute later.**  A limit REGISTERS and
            # DISPLAYS in a Julia session — the equal-time Green's function
            # `\lim_{t' \to t^+} G(t,t')` is a line a derivation turns on — and
            # is refused at the point of USE, because Symbolics.jl has no limit
            # and QuantumAlgebra is an operator algebra.  Same stance as
            # `lattice_descriptor` and `extremum_descriptor`.
            from notebook.runtime.shared_parser import UnknownNotationError
            _var = getattr(ir_ast.var, "name", "")
            raise UnknownNotationError(
                "parser.unsupported_notation.julia_limit", var=_var)

        elif isinstance(ir_ast, Integral):
            body = self.generate(ir_ast.expr)
            var = self.generate(ir_ast.var)
            lower = self.generate(ir_ast.lower) if ir_ast.lower else ""
            upper = self.generate(ir_ast.upper) if ir_ast.upper else ""
            body_esc = body.replace('"', '\\"')
            var_esc = var.replace('"', '\\"')
            lower_esc = lower.replace('"', '\\"')
            upper_esc = upper.replace('"', '\\"')
            return f'_nb_eval_continuous_integral("{body_esc}", "{var_esc}", "{lower_esc}", "{upper_esc}")'

        elif isinstance(ir_ast, IndexSum):
            # **A range that was WRITTEN must not be dropped.**  QuantumAlgebra's
            # `∑` is a FORMAL sum — it binds an index but has nowhere to put a
            # range — and this branch used to hand it a bounded sum anyway, so
            # `\sum_{i=1}^{2}` silently became `\sum_i` (measured 2026-08-22:
            # `[U \sum_{i=1}^{2} n_{i↑}n_{i↓}, c_{1↑}]` answered
            # `\sum_i [U c†_{1↓}c_{1↑}c_{1↓}]` as a SUCCESS — a Σ over an index
            # the summand does not contain).  "The backend does not take it" is
            # not a reason: a concrete integer range is finitely many terms, and
            # expanding is mechanical and exact.
            _expanded = expand_finite_index_sum(ir_ast)
            if _expanded is not None:
                return self._fenced(_expanded)

            # A bound index still has to satisfy QA's own spelling rule.  It was
            # exempted here on the ground that it never reaches `QuIndex` — true
            # of the BODY, false of the BINDER: measured, `∑(:sigma, …)` raises
            # `ArgumentError("index names must be single character, got sigma.")`
            # and that raw CAS text reached the researcher (Core Principle #1).
            # `\sum_\sigma` is the most ordinary spin sum there is.
            # A rangeless SPIN sum expands over the declared value set — the
            # researcher named the particle, and the particle fixes the
            # projections (`spin_value_set`).  Needed because QA does not fold
            # the dummy-swap cancellations between two formal `∑`s (measured).
            # Runs BEFORE the canonical rename so the reading is made on the
            # name the researcher wrote (`\sigma`, not the collapsed `s`).
            _spin = expand_spin_index_sum(
                ir_ast, dict(getattr(self.context, "spin_projections", {}) or {}))
            if _spin is not None:
                return self._fenced(_spin)

            ir_ast = self._canon_bound_index_names(ir_ast)

            # Save current bound indices
            old_bound = list(self._bound_indices)
            for idx in ir_ast.dummy_indices:
                self._bound_indices.append(idx.name)

            # Render the body once in `∑` mode.  If a bound index turned up
            # negated, that mode cannot express it, so render again the old way
            # — deciding by what the body actually contains rather than by a
            # guess about what it might.
            _outer_sum_mode = self._qa_sum_mode
            _outer_saw_neg = self._qa_sum_saw_negative
            self._qa_sum_mode = True
            self._qa_sum_saw_negative = False
            body = self.generate(ir_ast.body)
            _saw_neg = self._qa_sum_saw_negative
            self._qa_sum_mode = _outer_sum_mode
            self._qa_sum_saw_negative = _outer_saw_neg or _saw_neg
            if _saw_neg:
                body = self.generate(ir_ast.body)

            # Dummy indices are represented by their names for the helper
            dummy_names = ", ".join(idx.name for idx in ir_ast.dummy_indices)

            # Restore bound indices
            self._bound_indices = old_bound

            if self.static_index_mode:
                # Static mode: inline the body directly — no string-eval, no sumindex.
                # The bound indices were already rendered as _QA.QuIndex("name")).
                return body

            # QuantumAlgebra has its own sum — `∑(ind, A::QuExpr)` — and using it
            # is what makes the RESULT a sum.  The string-eval helper below only
            # substitutes `sumindex(n)` for the dummy, which produces an
            # expression that merely *carries* a summation index; measured
            # 2026-08-07 the two are not equal, `normal_form` gives
            # `∑₁ + ∑₁ a†(#₁) a(#₁)` versus `1 + a†(#₁) a(#₁)`, and the latex
            # has no Σ at all — which is why the Σ then had to be pasted back
            # on as a string in engine.py, using a DIFFERENT index name than the
            # body ended up with (`\sum_{k}[ξ_j c†_j c_j]`).
            _dummies = [i.name for i in ir_ast.dummy_indices]
            if _dummies and not self._qa_sum_saw_negative:
                res = body
                for name in reversed(_dummies):
                    res = f"_QA.∑(:{name}, {res})"
                if not _outer_sum_mode:
                    # Outermost sum: record which researcher index each `#ᵢ`
                    # will be.  QuantumAlgebra numbers its dummies by FIRST
                    # OCCURRENCE IN THE BODY, not by binding order — measured
                    # 2026-08-07 with marker parameters, and preserved across
                    # `normal_form`.  Binding order is what `_collect_dummy_names`
                    # returns, and using it printed a sum that was not written
                    # (`\sum_{m,n} c'_{n,u} c_{m,u}` came back as `c†_m c_n`).
                    # The body is OUR OWN output, so the order is computable
                    # rather than guessed.
                    self.sum_dummy_order = _first_occurrence_order(res, _dummies)
                return res

            body_esc = body.replace('"', '\\"')
            dummy_esc = dummy_names.replace('"', '\\"')
            return f'_nb_eval_index_sum("{body_esc}", "{dummy_esc}")'

        elif isinstance(ir_ast, FunctionCall) and isinstance(ir_ast.func, Symbol):
            fn = ir_ast.func.name
            if fn == "matrix":
                # Matrix literal → Julia native "[a b; c d]" (rows ';'-separated,
                # elements space-separated).  Rows arrive as square BracketExpr;
                # render their elements directly, never as comm().
                def _row(_node: Expr) -> str:
                    if isinstance(_node, BracketExpr):
                        return " ".join(self.generate(_e) for _e in _node.args)
                    return self.generate(_node)
                return "[" + "; ".join(_row(_r) for _r in ir_ast.args) + "]"
            base = fn.rstrip("'†")
            prop = self.context.get(base) if self.context else None
            if ir_ast.args:
                # Convert plain Symbol / negated Symbol args to proper index strings.
                # Handles FunctionCall nodes produced by operator_substitute where
                # the rhs is written with function notation: u(k), alpha(k), beta'(-k).
                idx_strs = [self._plain_sym_as_index(a) for a in ir_ast.args]
                if all(s is not None for s in idx_strs):
                    joined = ", ".join(idx_strs)  # type: ignore[arg-type]
                    if prop and prop.quantity.kind in ("scalar", "anonymous"):
                        # c-number function of quantum index → param
                        return f"param(:{base}, {qa_join_index_symbols(idx_strs, self._bound_indices)})"
                    elif prop and prop.statistics and prop.statistics.kind in ("fermi", "bose", "noncommutative"):
                        # QA operator with function-notation index
                        return f"{fn}({qa_join_index_symbols(idx_strs, self._bound_indices)})"
                    elif fn != base:
                        # fn has a prime/dagger suffix (e.g. "beta'") but the base is
                        # not registered in context (operator defined after substitution).
                        # In Julia/QuantumAlgebra, f'(index) is always a creation operator.
                        return f"{fn}({joined})"

        # Imaginary unit: bare Symbol("i") that is neither a bound summation index
        # nor a declared symbol in SymbolContext → emit Julia's "im".
        #
        # Safety chain (any True → keep as "i"):
        #   1. name starts with ":" → operator arg already marked by resolver
        #   2. "i" in _bound_indices → active IndexSum dummy variable
        #   3. context.get("i") is not None → explicitly declared by user
        # If none apply, the symbol is undeclared and free-standing — in a quantum
        # many-body / quantum optics session this is almost certainly √-1.
        if isinstance(ir_ast, Symbol):
            name = ir_ast.name
            # `i` is always the imaginary unit; `j` is too, but only in the
            # electrical-engineering domain (in physics `j` is a summation index).
            _ee = (self.context is not None
                   and "electrical_engineering" in getattr(self.context, "domains", []))
            if name == "i" or (name == "j" and _ee):
                if (not name.startswith(":")
                        and name not in self._bound_indices
                        and (self.context is None or self.context.get(name) is None)):
                    return "im"
            return name

        return super().generate(ir_ast)

    def _fenced(self, expanded: "Expr") -> str:
        r"""Render an EXPANDED sum, fenced, because it stands where a Σ stood.

        A `Σ` is one factor; its expansion is a SUM of factors, and the caller
        that decided not to parenthesise the `Σ` decided that on the NODE, not on
        the string.  Measured 2026-08-22 — and it was my own defect from the
        commit that added the expansion, invisible in the case I checked first:

            U \sum_{i=1}^{2} f_i   →   U * f(1) + f(2)

        i.e. the coefficient reached only the FIRST term.  The published
        scenario's answers happened to be unaffected (its commutator picks out
        `i = 1` anyway), which is exactly why one green case is not a check.
        """
        from notebook.models.ir import Add, Mul, NonCommutativeMul
        code = self.generate(expanded)
        if isinstance(expanded, (Add, Mul, NonCommutativeMul)) and len(expanded.args) > 1:
            return f"({code})"
        return code

    def _canon_bound_index_names(self, node: "IndexSum") -> "IndexSum":
        """Rename each dummy to its QA-legal spelling, binder and body together.

        `\\sigma` → `s` through the one shared table (`spin_label_canon`), so the
        binder and every occurrence in the body still match — `_nb_eval_index_sum`
        pairs them BY NAME.  A name with no legal spelling raises this system's
        own reason rather than letting QuantumAlgebra answer in its own words.
        """
        from notebook.runtime.index_contractor import substitute_symbol_or_index
        from notebook.runtime.spin_label_canon import qa_index_name

        body, dummies, changed = node.body, [], False
        for idx in node.dummy_indices:
            canon = qa_index_name(idx.name)      # raises with the reason
            if canon != idx.name:
                body = substitute_symbol_or_index(
                    body, idx.name, IndexSymbol(name=canon))
                idx = idx.model_copy(update={"name": canon})
                changed = True
            dummies.append(idx)
        if not changed:
            return node
        return node.model_copy(update={"body": body, "dummy_indices": dummies})

    def _generate_pow(self, node: "Pow") -> str:
        """Override: emit integer reciprocals as Julia rationals (1//n).

        In Julia, ``4^-1`` evaluates to ``0.25`` (Float64), which causes
        QuantumAlgebra's ``normal_form`` to propagate float coefficients
        (e.g. ``0.25*J`` instead of ``J/4``).  Using ``(1//4)`` keeps the
        coefficient as an exact rational throughout the computation.

        Only applies when the exponent is the integer literal ``-1`` and the
        base is a positive integer literal.  All other cases fall through to
        the generic ``base^exp`` form.
        """
        from notebook.models.ir import Pow as _Pow
        if (isinstance(node.exponent, Number)
                and node.exponent.value == -1
                and isinstance(node.base, Number)
                and node.base.value == int(node.base.value)
                and int(node.base.value) > 0):
            return f"(1//{int(node.base.value)})"
        return super()._generate_pow(node)

    def _plain_sym_as_index(self, node: Expr) -> "str | None":
        """Convert Symbol, IndexSymbol, or Mul([-1, Symbol]) to an index string
        for use inside QuantumAlgebra function calls (operators and param).

        Positive indices use ":name" Julia symbol literals so that
        _nb_eval_index_sum's ":k" -> sumindex(n) replacement fires correctly.
        Negative indices (Symbol("-k"), IndexSymbol("-k"), Mul([-1, k])) emit
        sumindex(-n) for bound indices or QuIndex for unbound, since Julia has
        no symbol-literal syntax for negated identifiers.

        Returns None for complex nodes so the caller falls back to generate().
        """
        inner: "str | None" = None
        is_neg = False

        if isinstance(node, (Symbol, IndexSymbol)):
            name = node.name
            # Strip leading colon — JuliaSemanticResolver adds ":k" to operator args
            # for _nb_eval_index_sum replacement; we need the bare name here.
            if name.startswith(":") and len(name) > 1:
                name = name[1:]
            if name.startswith("-"):
                inner, is_neg = name[1:], True
            else:
                inner = name
        elif isinstance(node, Number) and node.value == int(node.value):
            # Integer literal used as a site/mode index: Mz(2) → param(:Mz, QuIndex("2"))
            inner = str(int(node.value))
        elif (isinstance(node, Mul) and len(node.args) == 2
              and isinstance(node.args[0], Number) and node.args[0].value == -1
              and isinstance(node.args[1], Symbol)):
            sym_name = node.args[1].name
            # Strip leading colon (JuliaSemanticResolver adds ":k" to operator args)
            if sym_name.startswith(":") and len(sym_name) > 1:
                sym_name = sym_name[1:]
            if sym_name.startswith("-"):
                # Double negation: -(-k) = k → treat as positive index
                inner, is_neg = sym_name[1:], False
            else:
                inner, is_neg = sym_name, True

        if inner is None:
            return None

        if is_neg:
            if inner in self._bound_indices and not self.static_index_mode:
                pos = self._bound_indices.index(inner) + 1
                return f"_QA.sumindex(-{pos})"
            # Pure integer index: use integer literal for QuantumAlgebra so that
            # commutators involving operators at different concrete sites simplify
            # correctly (QuIndex treats labels as symbolic; integer literals are
            # concrete and enable full normal_form simplification).
            if _IS_INTEGER_SITE(inner.lstrip("-")):
                return f"(-{inner})"
            return qa_index_literal(inner, neg=True)
        else:
            # Pure integer index → use integer literal in both static and normal mode.
            # Integer literals enable full QuantumAlgebra normal_form simplification
            # (QuIndex("1") is treated as symbolic; literal 1 is concrete).
            # This applies to both static_index_mode and normal mode.
            if _IS_INTEGER_SITE(inner):
                return inner
            if self.static_index_mode:
                return qa_index_literal(inner)
            # Normal mode: ":name" so _nb_eval_index_sum replaces bound ones.
            # An UNBOUND ":name" reaches QA as an index literal, so QA's
            # single-character rule applies to it exactly as it does to a
            # QuIndex — spin labels collapse, and anything else multi-character
            # is refused before the CAS.  Measured 2026-08-01: a declared
            # `G_0(k, omega)` lowers to `param(:G_0, :k, :omega)` and Julia
            # answers `ArgumentError: index names must be single character`,
            # which is the raw-CAS-text leak this guard exists to stop.
            # BOUND names must stay verbatim and unchecked: they never reach
            # QuIndex — `_nb_eval_index_sum` replaces `:name` with sumindex(n)
            # by matching the name, so a multi-character dummy is fine there.
            if inner not in self._bound_indices:
                _canon = canon_spin_label(inner)
                if _canon is not None:
                    inner = _canon
            return f":{inner}"

    # ------------------------------------------------------------------
    # Expression-reference expansion and commutator distribution
    # ------------------------------------------------------------------

    def _maybe_expand_commutator(self, comm: Commutator) -> "str | None":
        """Distribute [H, target] term-by-term through the SymPy helpers when H
        is a multi-term Hamiltonian.

        Two cases are handled:

        Case 1 — LHS is an *opaque* expression reference (Symbol whose name
          starts with '_nb_'): look up the original user-facing canonical from
          self.input_lookup, re-parse to IR, and collect terms.

        Case 2 — LHS is a *direct* complex IR node (IndexSum or Add): the
          expression has already been resolved from the canonical string in
          build_command (expr_lookup["E1"] → full Hamiltonian IR).  Collect
          terms directly without any re-parsing step.

        Returns generated Julia code, or None if expansion is not possible
        (fall through to normal QuantumAlgebra routing).
        """
        lhs = comm.a
        ir_to_expand: "Expr | None" = None

        # Case 1: opaque expression variable like Symbol("_nb_E1")
        if isinstance(lhs, Symbol) and lhs.name.startswith("_nb_"):
            eid = lhs.name[len("_nb_"):]
            input_canon = self.input_lookup.get(eid)
            if input_canon:
                from notebook.runtime.shared_parser import SharedParser
                from notebook.runtime.julia_semantic_resolver import JuliaSemanticResolver
                try:
                    surface = SharedParser().parse(input_canon)
                    ir_to_expand = JuliaSemanticResolver().resolve(surface, self.context)
                    # Unwrap equation/assignment: "H = rhs" or "H ~ rhs" → use rhs.
                    # The user often stores Hamiltonians as "H = Σ_k ..." which
                    # resolves to an Equation IR; _collect_hamiltonian_terms only
                    # understands the expression part.
                    from notebook.models.ir import Equation as _EqIR, Assignment as _AssignIR
                    if isinstance(ir_to_expand, (_EqIR, _AssignIR)):
                        ir_to_expand = ir_to_expand.rhs
                except Exception:
                    ir_to_expand = None
        # Case 2: LHS is the full IR of a multi-term expression (IndexSum or Add)
        elif isinstance(lhs, (IndexSum, Add)):
            ir_to_expand = lhs

        if ir_to_expand is None:
            return None

        # Collect all (coeff, op) pairs from the IR.
        terms = self._collect_hamiltonian_terms(ir_to_expand)
        if not terms:
            return None

        # Only route to SymPy when at least one term has algebraic indices.
        # Pure single-letter-index Hamiltonians can be handled by QuantumAlgebra.
        if not any(self._has_algebraic_indices(t) for _, t in terms):
            return None

        return self._generate_sympy_sum_commutator(terms, comm.b)

    def _collect_hamiltonian_terms(
        self, ir: Expr
    ) -> "list[tuple[str, Expr]]":
        """Walk the IR and collect (coeff_str, op_expr) pairs.

        Handles:
          IndexSum(body, indices)  → recurse into body with bound indices
          Add(args)               → recurse into each arg
          Mul/NonCommutativeMul   → split scalars from operators
          Single operator         → coeff="1", op=node

        Returns list of (coeff_str, operator_product_node) pairs.
        Each coeff_str is a SymPy-evaluable expression string (e.g. "xi_k").
        """
        terms: list[tuple[str, Expr]] = []
        self._collect_terms_recursive(ir, "1", terms)
        return terms

    def _collect_terms_recursive(
        self,
        node: Expr,
        coeff_prefix: str,
        out: "list[tuple[str, Expr]]",
    ) -> None:
        """Recursive helper for _collect_hamiltonian_terms."""
        if isinstance(node, IndexSum):
            # Unwrap: bound index variables don't affect the SymPy coeff representation
            self._collect_terms_recursive(node.body, coeff_prefix, out)
        elif isinstance(node, Add):
            for arg in node.args:
                self._collect_terms_recursive(arg, coeff_prefix, out)
        elif isinstance(node, (Mul, NonCommutativeMul)):
            scalars = [a for a in node.args if self._is_scalar_factor(a)]
            ops = [a for a in node.args if not self._is_scalar_factor(a)]
            if not ops:
                return  # no operator part — pure scalar, commutator is 0
            coeff_parts: list[str] = [] if coeff_prefix == "1" else [coeff_prefix]
            for s in scalars:
                coeff_parts.append(self._scalar_to_sympy_str(s))
            coeff_str = "*".join(coeff_parts) if coeff_parts else "1"
            # Rebuild operator product.
            # When the sole operator part is itself a sum-like node (Add or
            # IndexSum), recurse so the scalar coefficient is distributed into
            # each addend:
            #   ξ_k * (c†_k c_k + c†_{-k} c_{-k})  →  two terms
            #   -1 * IndexSum(pairing1 + pairing2)   →  two terms with coeff -1
            if len(ops) == 1 and isinstance(ops[0], (Add, IndexSum)):
                self._collect_terms_recursive(ops[0], coeff_str, out)
                return
            if len(ops) == 1:
                op_node = ops[0]
            else:
                op_node = (NonCommutativeMul if isinstance(node, NonCommutativeMul)
                           else Mul)(args=ops)
            out.append((coeff_str, op_node))
        else:
            # Single node — check if it's an operator
            if not self._is_scalar_factor(node):
                coeff_str = coeff_prefix
                out.append((coeff_str, node))

    @staticmethod
    def _sanitize_index_for_sympy(idx_name: str) -> str:
        """Turn an index name into a valid Python/SymPy identifier.

        Examples:
          "k"   → "k"
          "-k"  → "mk"   (minus-k)
          "-q"  → "mq"
          "k+q" → "kpq"
        """
        s = idx_name.strip()
        if s.startswith("-"):
            return "m" + s[1:].strip()
        return s.replace("+", "p").replace("-", "m").replace("*", "t")

    def _scalar_to_sympy_str(self, node: Expr) -> str:
        """Serialize a scalar IR node to a SymPy-evaluable string.

        Uses underscore notation for indexed scalars (xi_k, not xi(k)) so that
        the Julia eval context needs only sp.Symbol for everything — no SymPy
        Function objects required, which avoids issues with bare-name scalars
        like Delta being mistaken for UndefinedFunction classes.

        Examples:
          Symbol("Delta")                         → "Delta"
          IndexedSymbol(Symbol("xi"), [idx("k")]) → "xi_k"
          IndexedSymbol(Symbol("xi"), [idx("-k")]) → "xi_mk"
          Number(value=2)                         → "2"
          Adjoint(Symbol("Delta"))                → "conjugate_Delta"
          FunctionCall("conj", [Symbol("Delta")]) → "conjugate_Delta"
        """
        if isinstance(node, Number):
            return str(node.value)
        if isinstance(node, Symbol):
            return node.name
        if isinstance(node, IndexedSymbol) and isinstance(node.base, Symbol):
            # Use underscore notation: xi_k, xi_mk (for xi_{-k}), etc.
            idx_parts = [self._sanitize_index_for_sympy(i.name) for i in node.indices]
            return f"{node.base.name}_{'_'.join(idx_parts)}"
        if isinstance(node, Adjoint) and isinstance(node.expr, Symbol):
            return f"conjugate_{node.expr.name}"
        if isinstance(node, Adjoint) and isinstance(node.expr, IndexedSymbol):
            if isinstance(node.expr.base, Symbol):
                idx_parts = [self._sanitize_index_for_sympy(i.name)
                             for i in node.expr.indices]
                return f"conjugate_{node.expr.base.name}_{'_'.join(idx_parts)}"
        if isinstance(node, (Adjoint, Conjugate)):
            return f"conjugate_{self._scalar_to_sympy_str(node.expr)}"
        if isinstance(node, FunctionCall) and isinstance(node.func, Symbol):
            # conj(Delta) → "conjugate_Delta" (plain SymPy Symbol, no eval needed)
            if node.func.name in ("conj", "conjugate"):
                return f"conjugate_{self._scalar_to_sympy_str(node.args[0])}"
        if isinstance(node, (Mul, NonCommutativeMul)):
            return "*".join(self._scalar_to_sympy_str(a) for a in node.args)
        # Fallback: use the code generator itself
        return self.generate(node)

    def _generate_sympy_sum_commutator(
        self,
        terms: "list[tuple[str, Expr]]",
        target: Expr,
    ) -> str:
        """Generate a _nb_sympy_sum_*_comm call for a distributed commutator.

        terms: list of (coeff_str, op_expr) extracted from the Hamiltonian.
        target: the target operator (b side of the commutator).
        """
        import json
        term_dicts: list[dict] = []
        all_vars: set[str] = set()

        for coeff_str, op_node in terms:
            try:
                ops_str = self._serialize_lhs_to_sympy(op_node, all_vars)
            except Exception:
                ops_str = ""
            if not ops_str:
                continue
            # Collect index variables from coeff (e.g. xi(k) → k)
            for tok in _re.findall(r'\(([^)]+)\)', coeff_str):
                for v in _re.findall(r'[A-Za-z_]\w*', tok):
                    all_vars.add(v)
            # Collect index variables from ops (e.g. cd(j) → j; operators
            # may carry dummy indices like "j" that the coeff doesn't mention)
            for tok in _re.findall(r'\(([^)]+)\)', ops_str):
                for v in _re.findall(r'[A-Za-z_]\w*', tok):
                    all_vars.add(v)
            term_dicts.append({"coeff": coeff_str, "ops": ops_str})

        if not term_dicts:
            return "nothing"

        # Collect index variables from the target side too
        all_vars |= self._collect_index_vars(target)

        target_str = self._serialize_single_op_to_sympy(target)
        vars_str = ",".join(sorted(all_vars))
        stats = self._detect_statistics_from_terms(terms)
        fn = ("_nb_sympy_sum_fermion_comm"
              if stats == "fermion"
              else "_nb_sympy_sum_boson_comm")

        terms_json = json.dumps(term_dicts).replace('"', '\\"')
        return f'{fn}("{terms_json}", "{target_str}", "{vars_str}")'

    def _detect_statistics_from_terms(
        self, terms: "list[tuple[str, Expr]]"
    ) -> str:
        """Detect fermion/boson from the first operator found in terms."""
        for _, op_node in terms:
            name = self._find_first_op_base_name(op_node)
            if name:
                base = name.rstrip("'†")
                if self.context:
                    prop = self.context.get(base)
                    if prop and hasattr(prop, "quantity"):
                        kind = getattr(prop.quantity, "kind", None)
                        if kind in ("fermion", "boson"):
                            return kind
                if base.lower() in ("c", "d", "f"):
                    return "fermion"
                if base.lower() in ("a", "b"):
                    return "boson"
        return "fermion"

    # ------------------------------------------------------------------
    # SymPy routing helpers (Steps 1-4 of the algebraic-index plan)
    # ------------------------------------------------------------------

    def _has_algebraic_indices(self, node: Expr) -> bool:
        """Return True if any IndexSymbol in the IR has an algebraic name.

        Algebraic: starts with '-', contains '+', or contains '*'.
        Simple single-letter labels (k, q, i, …) are categorical.
        """
        if isinstance(node, IndexSymbol):
            name = node.name
            return name.startswith("-") or "+" in name or "*" in name
        if isinstance(node, (Mul, NonCommutativeMul, Add)):
            return any(self._has_algebraic_indices(a) for a in node.args)
        if isinstance(node, IndexedSymbol):
            return (self._has_algebraic_indices(node.base) or
                    any(self._has_algebraic_indices(idx) for idx in node.indices))
        if isinstance(node, FunctionCall):
            # Check function name (IndexSymbol path) and arguments.
            # Also detect algebraic index in function args: c(-k) has
            # args=[Mul(-1, k)] which is an Add/Mul of symbols — algebraic.
            for a in node.args:
                if isinstance(a, (Mul, Add)):
                    return True  # algebraic expression as index: c(-k), c(k+q)
                if self._has_algebraic_indices(a):
                    return True
            return self._has_algebraic_indices(node.func)
        if isinstance(node, (Adjoint, Conjugate, Transpose,
                              Gradient, Divergence, Curl, Laplacian)):
            return self._has_algebraic_indices(node.expr)
        if isinstance(node, IndexSum):
            return (self._has_algebraic_indices(node.body) or
                    any(self._has_algebraic_indices(idx) for idx in node.dummy_indices))
        if isinstance(node, Commutator):
            return (self._has_algebraic_indices(node.a) or
                    self._has_algebraic_indices(node.b))
        return False

    def _collect_index_vars(self, node: Expr) -> set:
        """Collect all alphabetic tokens appearing in IndexSymbol names."""
        if isinstance(node, IndexSymbol):
            return set(_re.findall(r'[A-Za-z_]\w*', node.name))
        result: set = set()
        if isinstance(node, (Mul, NonCommutativeMul, Add)):
            for a in node.args:
                result |= self._collect_index_vars(a)
        elif isinstance(node, IndexedSymbol):
            result |= self._collect_index_vars(node.base)
            for idx in node.indices:
                result |= self._collect_index_vars(idx)
        elif isinstance(node, FunctionCall):
            result |= self._collect_index_vars(node.func)
            for a in node.args:
                result |= self._collect_index_vars(a)
        elif isinstance(node, (Adjoint, Conjugate, Transpose,
                                Gradient, Divergence, Curl, Laplacian)):
            result |= self._collect_index_vars(node.expr)
        elif isinstance(node, IndexSum):
            result |= self._collect_index_vars(node.body)
            for idx in node.dummy_indices:
                result |= self._collect_index_vars(idx)
        elif isinstance(node, Commutator):
            result |= self._collect_index_vars(node.a)
            result |= self._collect_index_vars(node.b)
        return result


    def _generate_sympy_commutator(self, commutator: Commutator) -> str:
        """Generate a _nb_sympy_*comm call for algebraic-index commutators.

        Called when _has_algebraic_indices detects arithmetic in the indices,
        i.e. cases that QuantumAlgebra.jl cannot handle (QuIndex requires a
        single-char Symbol name; arithmetic like -k, k+q, a*k+b is rejected).

        Returns a Julia expression string that evaluates to a
        Tuple{String,String} (value_str, latex_str) — the command loop in
        _julia_driver.jl unpacks this and prints the two parts separately.
        """
        extra_vars: set = set()
        prod_str = self._serialize_lhs_to_sympy(commutator.a, extra_vars)
        target_str = self._serialize_single_op_to_sympy(commutator.b)

        # Collect every index variable name that appears in the commutator
        all_vars = self._collect_index_vars(commutator) | extra_vars
        vars_str = ",".join(sorted(all_vars))

        stats = self._detect_statistics(commutator.a)
        fn = "_nb_sympy_fermion_comm" if stats == "fermion" else "_nb_sympy_boson_comm"

        prod_esc   = prod_str.replace('"', '\\"')
        target_esc = target_str.replace('"', '\\"')
        vars_esc   = vars_str.replace('"', '\\"')
        return f'{fn}("{prod_esc}", "{target_esc}", "{vars_esc}")'
