"""Julia backend mixin: CAS command construction (build_command + builders).

Extracted from JuliaBackendHandler (AUD-4 stage 2b) as a mixin to shrink the
handler class.  Holds the build_command dispatcher and the per-operation
_build_* helpers that turn common operations into Julia/QuantumAlgebra command
strings.  Mixed into JuliaBackendHandler; sibling methods (_make_build_ctx,
_maybe_*, etc.) resolve via self / MRO.  Depends only on BuildResult and a few
stateless helpers from julia_handler_common.
"""
from __future__ import annotations

import json
import re

from notebook.runtime.backend_handler import BuildResult
from notebook.runtime.expr_ref_fields import is_expr_ref
from notebook.runtime.julia_handler_common import (
    _DOMAIN_TO_JULIA_TYPE,
    _adjoint_ir,
    _collect_dummy_names,
    quantum_operator_names,
    _relabel_index_in_canonical,
    _subst_quantum_op,
)


def _sum_order(gen, dummies: "list[str]") -> "list[str]":
    """Dummy names in the order the backend will number them, if it says so.

    The generator records the FIRST-OCCURRENCE order when it emits
    QuantumAlgebra's own `∑` — that is the order QA numbers `#ᵢ` by (measured
    2026-08-07 with marker parameters, and preserved across `normal_form`), NOT
    the binding order that `_collect_dummy_names` returns.  Using the binding
    order printed a sum that was not written: `\\sum_{m,n} c'_{n,u} c_{m,u}`
    came back as `c†_m c_n`.

    Paths that emit no `∑` leave it empty and the binding order is all there
    is; those paths produce no `#ᵢ` either, so only the Σ prefix reads the list
    and the order does not matter there.
    """
    return list(getattr(gen, "sum_dummy_order", None) or dummies)


def _nothing_replaced(before, after) -> bool:
    """True when the IR came back from `_subst_quantum_op` untouched.

    The rule named an operator the expression does not contain, so nothing could
    happen — and nothing did, silently: measured 2026-08-15, `operator_substitute`
    with an absent `op_name` returned the expression unchanged as a SUCCESS,
    giving the researcher exactly what a rule that DID apply gives
    (open_items NOOP-REWRITE-FAMILY).

    Comparing the IR before and after is deliberately NOT a second matcher: it
    reads the OUTPUT of the one that ran.  A guard that re-decided "does this
    operator occur?" on its own would be a second answer to one question, and
    the two would drift — the substituter knows about primes, sublattice labels,
    index signs and the exact/template distinction, none of which a scan for the
    name would.
    """
    return before == after


class JuliaBuildMixin:
    """CAS command-construction helpers mixed into JuliaBackendHandler."""

    def build_command(self, operation: str, args: dict, expr_lookup: dict) -> BuildResult:
        from notebook.models.ir import CasOperation
        bctx = self._make_build_ctx(args, expr_lookup)

        if operation in ("solve_equation", "solve_system"):
            return self._build_solve(operation, args, expr_lookup)
        if operation == "build_inverse_transform":
            return self._build_inverse_transform(args, expr_lookup)
        if operation == "apply_canonical_constraints":
            return self._build_apply_constraints(args, expr_lookup)
        if operation == "derive_metric_constraints":
            return self._build_metric_constraints(args, expr_lookup)
        if operation == "operator_substitute":
            return self._build_operator_substitute(args, bctx)
        if operation == "operator_substitute_pair":
            return self._build_operator_substitute_pair(args, bctx)
        if operation == "relabel_index":
            return self._build_relabel_index(args, bctx)
        if operation == "combine_expressions":
            r = self._maybe_sympy_combine(args, expr_lookup, bctx.ctx)
            if r is not None:
                return r
        if operation == "collect_by_operators":
            return self._build_collect_by_operators(args, expr_lookup, bctx)
        if operation == "collect_spin_pm":
            r = self._maybe_collect_spin_pm(args, expr_lookup, bctx.ctx)
            if r is not None:
                return r
        if operation == "substitute":
            return self._build_substitute(args, bctx)
        if operation == "declare_symmetry":
            return self._build_declare_symmetry(args)
        if operation == "pair_substitute":
            return self._build_pair_substitute(args, bctx)
        if operation == "operator_taylor_expand":
            return self._build_operator_taylor_expand(args, bctx)
        if operation == "extract_order":
            return self._build_extract_order(args, bctx)
        if operation == "truncate_series":
            return self._build_truncate_series(args, bctx)
        if operation == "wick_contract":
            return self._build_wick_contract(args, bctx)
        if operation == "heisenberg_operator":
            return self._build_heisenberg_operator(args, bctx)
        if operation == "commutator":
            r = self._build_commutator(args, expr_lookup, bctx)
            if r is not None:
                return r
        if operation == "normal_order":
            r = self._maybe_sympy_normal_order(args, expr_lookup, bctx.ctx)
            if r is not None:
                return r
        if operation == "vacuum_expectation":
            r = self._maybe_sympy_vacuum_expectation(args, expr_lookup, bctx.ctx)
            if r is not None:
                return r
        if operation == "define_scalar_parameter":
            return self._build_define_scalar_parameter(args)
        if operation == "change_integration_variables":
            return self._build_change_integration_variables(args, bctx)
        if operation == "expand_neighbor_sum":
            return self._build_expand_neighbor_sum(args, bctx)

        _compiler = getattr(self._builder, "_compiler", None)
        if _compiler is not None:
            # The compiler instance is reused across calls, so a stale order
            # from a previous expression would be read as this one's.
            _compiler.sum_dummy_order = []
        build = self._builder.build(CasOperation(operation=operation, args=bctx.ir_args))
        if not build.ok:
            return BuildResult(ok=False, command="", error=build.error)
        # The default template path summed something only if it emitted `∑`.
        # Its dummy ORDER is the generator's to report — the engine's fallback
        # reads `\sum_{m,n}` off the source's LaTeX, which is the order the
        # researcher WROTE, and QuantumAlgebra numbers `#ᵢ` by first occurrence
        # in the body instead (measured 2026-08-07).
        _order = list(getattr(_compiler, "sum_dummy_order", None) or ())
        if _order and "_QA.∑(" in build.command:
            return BuildResult(ok=True, command=build.command,
                               sum_index_hint=",".join(_order))
        return BuildResult(ok=True, command=build.command)

    # ------------------------------------------------------------------
    # Per-operation build helpers
    # ------------------------------------------------------------------

    def _build_operator_substitute(self, args: dict, bctx: "_BuildCtx") -> "BuildResult":
        """operator_substitute: IR-level substitution of a quantum operator for a
        linear combination of new operators, followed by normal_form().
        Symbolics.jl's substitute() does not apply to QuantumAlgebra objects,
        so we do the substitution at the IR level before code generation.

        Supports two calling modes:
          Single-rule: args has op_name, formal_index, rhs  (one operator replaced)
          Multi-rule:  args has op_rules list [{op_name, formal_index, rhs}, ...]
                       All rules are applied sequentially to the SAME source IR.
                       Required when substituting two different operators simultaneously
                       (e.g. Bogoliubov transformation with explicit spin: a and b).
        """
        import re
        parser = bctx.parser
        resolver = bctx.resolver
        ctx = bctx.ctx
        ir_args = bctx.ir_args
        expr_lookup = bctx.expr_lookup
        expr_ref = args.get("expr", "")

        # Helper: get parseable source canonical or detect CAS-derived source.
        def _get_source_canonical(eref: str) -> "tuple[str | None, bool]":
            """Return (canonical, is_cas_derived).

            is_cas_derived=True means the stored canonical is a Julia opaque
            variable (_nb_En) and IR-level substitution cannot be chained on it.
            """
            stored = expr_lookup.get(eref, "")
            if stored.startswith("_nb_"):
                return stored, True
            ic = expr_lookup.get(f"__input__{eref}")
            if ic:
                return ic, False
            return stored or None, False

        from notebook.models.ir import Equation as _EqIR, Symbol as _Sym

        def _parse_and_resolve_source(ic: str):
            surf = parser.parse(ic)
            if surf is None:
                return None
            ir = resolver.resolve(surf, ctx, skip_alphatize=True)
            if isinstance(ir, _EqIR):
                ir = ir.rhs
            return ir

        def _auto_exact_index(formal_idx: str, rhs_raw: str) -> bool:
            """Auto-detect whether formal_idx is a concrete site label (exact filter).

            Integer labels ("2") and uppercase sublattice labels ("A","B") are
            ALWAYS concrete — even if they appear inside the RHS (e.g. Mz(2)).
            The RHS-appearance check is only used for symbolic letter indices.

            Returns True (exact filter):
              - formal_idx is an integer string: Sz(2)→Mz  or  Sz(2)→Mz(2)
              - formal_idx is a single uppercase char: Sz(A)→mz

            Returns False (template / uniform substitution):
              - formal_idx is a lowercase letter appearing in RHS:
                  c(k) → u g(k) + v g†(-k)   (Bogoliubov template)
              - formal_idx is a lowercase letter NOT in RHS:
                  Sz(i) → mz                  (uniform mean-field, all sites)
            """
            # Integer label → always concrete (highest priority)
            try:
                int(formal_idx)
                return True
            except ValueError:
                pass
            # Uppercase single char → sublattice label, always concrete
            if len(formal_idx) == 1 and formal_idx.isupper():
                return True
            # Symbolic letter: template if it appears in RHS
            if re.search(r'\b' + re.escape(formal_idx) + r'\b', rhs_raw):
                return False  # appears in RHS → template parameter
            return False  # lowercase/multi-char not in RHS → generic (uniform)

        def _build_op_names_for_rule(op_str: str, rhs_ir) -> "set[str]":
            return quantum_operator_names(ctx, rhs_ir) | {op_str, op_str + "'"}

        # --- Multi-rule path ---
        # Use bctx.ir_args (not the raw args dict): _make_build_ctx expands any
        # bare E-id string field inside each rule dict (e.g. "rhs": "E6") to its
        # canonical text, so a multi-rule substitution can reference a CAS
        # result (e.g. a build_inverse_transform output) by id.
        op_rules_list = bctx.ir_args.get("op_rules")
        if op_rules_list:
            _ic, _is_cas = _get_source_canonical(expr_ref)
            if _is_cas:
                return BuildResult(
                    ok=False, command="",
                    error=(
                        f"operator_substitute: {expr_ref} is CAS-derived. "
                        "Apply all rules simultaneously to the original expression."
                    ),
                )
            if not _ic:
                return BuildResult(ok=False, command="",
                                   error=f"operator_substitute: no canonical for {expr_ref}")
            main_ir = _parse_and_resolve_source(_ic)
            if main_ir is None:
                return BuildResult(ok=False, command="",
                                   error=f"operator_substitute: cannot parse {_ic!r}")

            for _rule in op_rules_list:
                _r_op_raw = _rule.get("op_name", "")
                _r_idx_raw = _rule.get("formal_index", "k")
                _r_rhs_raw = _rule.get("rhs", "")
                _r_spin_raw = _rule.get("spin_label", "")
                _r_op_str = _r_op_raw.name if isinstance(_r_op_raw, _Sym) else str(_r_op_raw)
                _r_idx_str = _r_idx_raw.name if isinstance(_r_idx_raw, _Sym) else str(_r_idx_raw)
                _r_static_idx = str(_r_spin_raw).strip() or None

                _r_rhs_surf = parser.parse(_r_rhs_raw)
                if _r_rhs_surf is None:
                    return BuildResult(ok=False, command="",
                                       error=f"operator_substitute: cannot parse rhs {_r_rhs_raw!r}")
                _r_rhs_ir = resolver.resolve(_r_rhs_surf, ctx)
                # An E-id-expanded rhs (e.g. "rhs": "E6") may resolve to the
                # source expression's full "LHS = RHS" text (engine.run_operation
                # reconstructs equation form in expr_lookup for _resolve_equation_
                # source) rather than a bare RHS — unwrap to RHS only, same as
                # main_ir above, so the substitution inserts a plain formula
                # instead of embedding a Julia ``~`` equation node.
                if isinstance(_r_rhs_ir, _EqIR):
                    _r_rhs_ir = _r_rhs_ir.rhs

                # Negative formal_idx (e.g. "-k"): user stated the rule for c(-k,σ).
                # Use index_sign="negative" with the stripped positive formal_idx.
                _r_idx_sign = None
                if _r_idx_str.startswith("-"):
                    _r_idx_str = _r_idx_str[1:]  # strip leading '-'
                    _r_idx_sign = "negative"

                _r_explicit_exact = _rule.get("exact_index")
                _r_exact = (
                    False if _r_idx_sign else  # negated index → always template
                    bool(_r_explicit_exact) if _r_explicit_exact is not None else
                    _auto_exact_index(_r_idx_str, _r_rhs_raw)
                )
                main_ir = _subst_quantum_op(
                    main_ir, _r_op_str, _r_rhs_ir, _r_idx_str,
                    _build_op_names_for_rule(_r_op_str, _r_rhs_ir),
                    _r_idx_sign,
                    _r_static_idx,
                    exact_index=_r_exact,
                )

            from notebook.runtime.julia_code_generator import JuliaCodeGenerator as _JCG
            _jcg = _JCG(context=ctx)
            _jcg.static_index_mode = True
            inner_code = _jcg.generate(main_ir)
            dummies = _collect_dummy_names(main_ir)
            _star_syms = sorted(set(re.findall(r'\b([A-Z][A-Za-z]*)Star\b', inner_code)))
            _preamble = "".join(f"@variables {s}Star; " for s in _star_syms)
            return BuildResult(ok=True, command=f"{_preamble}_nb_eval_site_deltas(normal_form({inner_code}))",
                               sum_index_hint=",".join(_sum_order(_jcg, dummies)),
                               input_canonical="")

        # --- Single-rule path ---
        # Guard: if source is a CAS-derived expression (_nb_En), its __input__
        # has been propagated from the ORIGINAL source's canonical — not E2's
        # own content. Silently applying the substitution to the wrong expression
        # would produce incorrect results (e.g. a-operators reappearing after
        # a chained Bogoliubov substitution). Block and advise instead.
        _src_stored = expr_lookup.get(expr_ref, "")
        if _src_stored.startswith("_nb_"):
            return BuildResult(
                ok=False, command="",
                error=(
                    f"operator_substitute: {expr_ref} is a CAS-derived expression. "
                    "Chained single-operator substitution is not supported. "
                    "Use multi-rule substitution on the original expression: "
                    "\"E1のa(k)にXを、b(k)にYを代入する\"."
                ),
            )

        op_name_raw = args.get("op_name", "")
        formal_idx_raw = args.get("formal_index", "k")
        spin_label_raw = args.get("spin_label", "")

        # op_name and formal_index may have been parsed to IR Symbol nodes
        op_name_str = (
            op_name_raw.name if isinstance(op_name_raw, _Sym) else str(op_name_raw)
        )
        formal_idx_str = (
            formal_idx_raw.name if isinstance(formal_idx_raw, _Sym) else str(formal_idx_raw)
        )
        _single_static_idx = str(spin_label_raw).strip() or None

        # rhs was already parsed to IR by the loop above
        rhs_ir_node = ir_args.get("rhs")
        if rhs_ir_node is None:
            return BuildResult(ok=False, command="", error="operator_substitute: missing rhs")

        # Get original user-input canonical for the target expression
        input_canonical = expr_lookup.get(f"__input__{expr_ref}")
        if not input_canonical and is_expr_ref(expr_ref):
            input_canonical = expr_lookup.get(expr_ref, "")
        if not input_canonical:
            return BuildResult(ok=False, command="", error=f"operator_substitute: no input canonical for {expr_ref}")

        surface_main = parser.parse(input_canonical)
        if surface_main is None:
            return BuildResult(ok=False, command="", error=f"operator_substitute: cannot parse {input_canonical!r}")
        # skip_alphatize=True: parallel IndexSum nodes (Σ_k H_kin + Σ_k H_pair)
        # should keep the same dummy index k, not be renamed k→j, so that
        # static_index_mode generates the same QuIndex("k") for all sums and
        # normal_form() treats them as the same physical momentum.
        main_ir = resolver.resolve(surface_main, ctx, skip_alphatize=True)

        # Unwrap Equation (e.g. "H = Σ_k ...") → use RHS only.
        # normal_form() requires a QuExpr, not a Julia/Symbolics Equation.
        if isinstance(main_ir, _EqIR):
            main_ir = main_ir.rhs

        # Which names this session treats as quantum operators — asked in one
        # place (`quantum_operator_names`), because the DISPLAY asks the same
        # question and used to answer "all of them" (PRIME-DISPLAY-DAGGER).
        # The operand being substituted is one by construction, and the RHS's
        # own primed bases join it so an operator absent from the context
        # (a Bogoliubov `beta'(-k)`) is still adjointed correctly.
        _op_names = quantum_operator_names(ctx, rhs_ir_node)
        _op_names |= {op_name_str, op_name_str + "'"}

        _rhs_raw_str = args.get("rhs", "")
        if not isinstance(_rhs_raw_str, str):
            _rhs_raw_str = str(_rhs_raw_str)
        _explicit_exact = args.get("exact_index")
        _single_exact = (
            bool(_explicit_exact)
            if _explicit_exact is not None
            else _auto_exact_index(formal_idx_str, _rhs_raw_str)
        )

        substituted_ir = _subst_quantum_op(
            main_ir, op_name_str, rhs_ir_node, formal_idx_str, _op_names,
            None, _single_static_idx,
            exact_index=_single_exact,
        )

        if _nothing_replaced(main_ir, substituted_ir):
            from notebook.runtime.i18n import t as _t_os
            return BuildResult(ok=False, command="", reason_code="lhs_not_found",
                               error=_t_os("engine.operator_not_in_expression",
                                           self.lang, name=op_name_str))

        from notebook.runtime.julia_code_generator import JuliaCodeGenerator as _JCG
        from notebook.models.ir import IndexSum as _IndexSumIR, Add as _AddIR
        _jcg = _JCG(context=ctx)
        _jcg.static_index_mode = True   # inline bodies, use QuIndex not sumindex
        inner_code = _jcg.generate(substituted_ir)

        dummies = _collect_dummy_names(substituted_ir)
        # Pre-declare any XStar symbols (e.g. DeltaStar) so _nb_eval does not
        # need to auto-declare them mid-expression, which can fail inside
        # normal_form's evaluation context.
        _star_syms = sorted(set(re.findall(r'\b([A-Z][A-Za-z]*)Star\b', inner_code)))
        _preamble = "".join(f"@variables {s}Star; " for s in _star_syms)
        return BuildResult(ok=True, command=f"{_preamble}_nb_eval_site_deltas(normal_form({inner_code}))",
                           sum_index_hint=",".join(_sum_order(_jcg, dummies)),
                           input_canonical="")

    def _build_operator_substitute_pair(self, args: dict, bctx: "_BuildCtx") -> "BuildResult":
        """operator_substitute_pair: Nambu-style paired substitution.
        Applies TWO independent rules explicitly provided by the user:
          c(k)  → rhs     (positive index only)
          c(-k) → rhs_neg (negative index only)
        c†(k) and c†(-k) get the adjoints of these rules respectively.
        Unlike operator_substitute, NO rule is auto-derived from the other.
        """
        import re
        parser = bctx.parser
        resolver = bctx.resolver
        ctx = bctx.ctx
        ir_args = bctx.ir_args
        expr_lookup = bctx.expr_lookup

        expr_ref       = args.get("expr", "")
        op_name_raw    = args.get("op_name", "")
        formal_idx_raw = args.get("formal_index", "k")

        from notebook.models.ir import Symbol as _Sym
        op_name_str    = op_name_raw.name if isinstance(op_name_raw, _Sym) else str(op_name_raw)
        formal_idx_str = formal_idx_raw.name if isinstance(formal_idx_raw, _Sym) else str(formal_idx_raw)

        rhs_ir_node     = ir_args.get("rhs")
        rhs_neg_ir_node = ir_args.get("rhs_neg")
        if rhs_ir_node is None:
            return BuildResult(ok=False, command="", error="operator_substitute_pair: missing rhs")
        if rhs_neg_ir_node is None:
            return BuildResult(ok=False, command="", error="operator_substitute_pair: missing rhs_neg")

        input_canonical = expr_lookup.get(f"__input__{expr_ref}")
        if not input_canonical and is_expr_ref(expr_ref):
            input_canonical = expr_lookup.get(expr_ref, "")
        if not input_canonical:
            return BuildResult(ok=False, command="",
                               error=f"operator_substitute_pair: no input canonical for {expr_ref}")

        surface_main = parser.parse(input_canonical)
        if surface_main is None:
            return BuildResult(ok=False, command="",
                               error=f"operator_substitute_pair: cannot parse {input_canonical!r}")
        main_ir = resolver.resolve(surface_main, ctx, skip_alphatize=True)

        from notebook.models.ir import Equation as _EqIR2
        if isinstance(main_ir, _EqIR2):
            main_ir = main_ir.rhs

        _op_names2 = quantum_operator_names(
            ctx, rhs_ir_node, rhs_neg_ir_node) | {op_name_str, op_name_str + "'"}

        # Step 1: substitute c(k) only (positive index)
        intermediate_ir = _subst_quantum_op(
            main_ir, op_name_str, rhs_ir_node, formal_idx_str, _op_names2,
            index_sign="positive"
        )
        # Step 2: substitute c(-k) only (negative index) using rhs_neg
        substituted_ir = _subst_quantum_op(
            intermediate_ir, op_name_str, rhs_neg_ir_node, formal_idx_str, _op_names2,
            index_sign="negative"
        )

        if _nothing_replaced(main_ir, substituted_ir):
            from notebook.runtime.i18n import t as _t_op
            return BuildResult(ok=False, command="", reason_code="lhs_not_found",
                               error=_t_op("engine.operator_not_in_expression",
                                           self.lang, name=op_name_str))

        from notebook.runtime.julia_code_generator import JuliaCodeGenerator as _JCG2
        _jcg2 = _JCG2(context=ctx)
        _jcg2.static_index_mode = True
        inner_code2 = _jcg2.generate(substituted_ir)

        dummies2 = _collect_dummy_names(substituted_ir)
        _star_syms2 = sorted(set(re.findall(r'\b([A-Z][A-Za-z]*)Star\b', inner_code2)))
        _preamble2 = "".join(f"@variables {s}Star; " for s in _star_syms2)
        return BuildResult(ok=True, command=f"{_preamble2}_nb_eval_site_deltas(normal_form({inner_code2}))",
                           sum_index_hint=",".join(_sum_order(_jcg2, dummies2)),
                           input_canonical="")

    def _build_relabel_index(self, args: dict, bctx: "_BuildCtx") -> "BuildResult":
        """relabel_index: swap dummy sum index (e.g. k → -k) at canonical level.
        QuantumAlgebra.jl requires single-char QuIndex and cannot represent
        "-k" as an index, so we perform the swap in the user-facing canonical
        string and re-evaluate in Julia, just like operator_substitute does.
        """
        import re
        parser = bctx.parser
        resolver = bctx.resolver
        ctx = bctx.ctx
        expr_lookup = bctx.expr_lookup

        old_idx  = str(args.get("old_index", "k"))
        new_idx  = str(args.get("new_index", "-k"))
        expr_ref = str(args.get("expr", ""))

        # SymPy chain: if the expression has sympy_canonical, re-route to
        # _nb_sympy_substitute.  Directly re-parsing via QA would apply the
        # K-convention (-k → K) and produce incorrect results.
        _sympy_canon = self._get_sympy_canonical(expr_ref, expr_lookup)
        if _sympy_canon is not None:
            _sub_args = {**args, "lhs": old_idx, "rhs": new_idx}
            _sub = self._maybe_sympy_substitute(_sub_args, expr_lookup, ctx)
            if _sub is not None:
                return _sub

        input_canonical = expr_lookup.get(f"__input__{expr_ref}")
        if not input_canonical and is_expr_ref(expr_ref):
            input_canonical = expr_lookup.get(expr_ref, "")
        if not input_canonical:
            return BuildResult(ok=False, command="",
                               error=f"relabel_index: no input canonical for {expr_ref}")

        relabeled = _relabel_index_in_canonical(input_canonical, old_idx, new_idx)

        surface = parser.parse(relabeled)
        if surface is None:
            return BuildResult(ok=False, command="",
                               error="relabel_index: cannot parse relabeled canonical")

        ir = resolver.resolve(surface, ctx, skip_alphatize=True)

        from notebook.models.ir import Equation as _EqIR2
        if isinstance(ir, _EqIR2):
            ir = ir.rhs

        from notebook.runtime.julia_code_generator import JuliaCodeGenerator as _JCG2
        _jcg2 = _JCG2(context=ctx)
        _jcg2.static_index_mode = True
        code = _jcg2.generate(ir)

        dummies = _collect_dummy_names(ir)
        _star_syms2 = sorted(set(re.findall(r'\b([A-Z][A-Za-z]*)Star\b', code)))
        _preamble2 = "".join(f"@variables {s}Star; " for s in _star_syms2)
        return BuildResult(ok=True, command=f"{_preamble2}normal_form({code})",
                           sum_index_hint=",".join(_sum_order(_jcg2, dummies)),
                           input_canonical=relabeled)

    def _build_collect_by_operators(self, args: dict, expr_lookup: dict, bctx: "_BuildCtx") -> "BuildResult":
        """collect_by_operators: SymPy routing first, then QA inline fallback.

        Group QuTerms by their bare operator content, combining scalar Params
        into Symbolics coefficients so that normal_form can add them.
        Example: u_k²ξ_k α†α and -v_k²ξ_k α†α → (u_k²-v_k²)ξ_k α†α
        NOTE: each line becomes "; "-separated by JuliaAdapter, so ternary
        "? x : y" and multi-arg calls must stay on ONE line each.
        """
        # SymPy chain first
        _cbo = self._maybe_sympy_collect_by_operators(args, expr_lookup, bctx.ctx)
        if _cbo is not None:
            return _cbo

        # QA inline fallback
        expr_ref = str(args.get("expr", ""))
        if not expr_ref:
            return BuildResult(ok=False, command="", error="collect_by_operators: missing expr")
        nb_var = bctx.nb_var_for_expr(expr_ref) or f"_nb_{expr_ref}"
        cmd = "\n".join([
            f"let _nb_src = {nb_var}",
            "  _nb_psym = Dict()",
            "  for (_nb_term, _) in _nb_src.terms",
            "    for _nb_p in _nb_term.params",
            "      if !haskey(_nb_psym, _nb_p)",
            "        _nb_pname = string(_nb_p.name)",
            "        _nb_sym_str = isempty(_nb_p.inds) ? _nb_pname : (_nb_pname * \"_\" * string(_nb_p.inds[1]))",
            "        _nb_sym = Symbol(_nb_sym_str)",
            "        _nb_autodeclare(_nb_sym)",
            "        _nb_psym[_nb_p] = eval(_nb_sym)",
            "      end",
            "    end",
            "  end",
            "  _nb_eps = isempty(_nb_src.terms) ? [] : filter(_ -> false, first(keys(_nb_src.terms)).params)",
            "  _nb_groups = Dict()",
            "  for (_nb_term, _nb_coeff) in _nb_src.terms",
            "    _nb_sc = _nb_coeff",
            "    for _nb_p in _nb_term.params",
            "      _nb_sc = _nb_sc * _nb_psym[_nb_p]",
            "    end",
            "    _nb_bk = _QA.QuTerm(_nb_term.nsuminds, _nb_term.δs, _nb_eps, _nb_term.expvals, _nb_term.corrs, _nb_term.bares)",
            "    _nb_groups[_nb_bk] = get(_nb_groups, _nb_bk, 0) + _nb_sc",
            "  end",
            "  _nb_nt = Dict{_QA.QuTerm, Number}()",
            "  for (_nb_key, _nb_sc) in _nb_groups",
            "    _nb_sc_s = if _nb_sc isa Symbolics.Num",
            "      try",
            "        _nb_str = replace(string(Symbolics.simplify(_nb_sc)), \"^\" => \"**\")",
            "        _nb_sp_e = SymPy.sympify(_nb_str)",
            "        _nb_sp_vs = [SymPy.sympify(string(v)) for (_, v) in _nb_psym]",
            "        _nb_n = length(_nb_sp_vs)",
            "        _nb_prods = [_nb_sp_vs[i]*_nb_sp_vs[j] for i in 1:_nb_n for j in (i+1):_nb_n]",
            "        _nb_tgts = vcat(_nb_sp_vs, _nb_prods)",
            "        _nb_f = SymPy.factor(SymPy.collect(_nb_sp_e, _nb_tgts))",
            "        eval(Meta.parse(replace(string(_nb_f), \"**\" => \"^\")))",
            "      catch; Symbolics.simplify(_nb_sc); end",
            "    else; _nb_sc; end",
            "    iszero(_nb_sc_s) || (_nb_nt[_nb_key] = _nb_sc_s)",
            "  end",
            "  normal_form(_QA.QuExpr(_nb_nt))",
            "end",
        ])
        return BuildResult(ok=True, command=cmd)

    def _build_substitute(self, args: dict, bctx: "_BuildCtx") -> "BuildResult":
        """substitute: scalar parameter substitution in a QuExpr.
        Generates inline Julia code that iterates over the QuExpr's terms
        dict and applies Symbolics.substitute() on each coefficient.
        No helper function required — avoids Julia-restart dependency.
        """
        import re
        ctx = bctx.ctx
        expr_lookup = bctx.expr_lookup

        # SymPy chain: if the expression has sympy_canonical, use subs()
        # instead of the QA Symbolics.substitute() template.
        _sub_sympy = self._maybe_sympy_substitute(args, expr_lookup, ctx)
        if _sub_sympy is not None:
            return _sub_sympy

        # QA operator identity: if lhs is a registered operator (non-commutative,
        # fermion, or boson), use _nb_operator_identity which expands the
        # operator product via Cartesian product + normal_form.
        # This handles e.g. Sx → (Sp+Sm)/2 inside a QuExpr.
        _op_id = self._maybe_operator_identity(args, expr_lookup, ctx)
        if _op_id is not None:
            return _op_id

        expr_ref = str(args.get("expr", ""))
        lhs_str  = str(args.get("lhs", ""))
        rhs_str  = str(args.get("rhs", ""))
        if not (expr_ref and lhs_str and rhs_str):
            return BuildResult(ok=False, command="",
                               error="substitute: missing expr, lhs, or rhs")

        nb_var = bctx.nb_var_for_expr(expr_ref) or f"_nb_{expr_ref}"

        def _canonical_to_param(s: str) -> str:
            """Convert 'xi(-k)' or 'xi_-k' → param(:xi, _QA.QuIndex("K"))."""
            m = re.match(r'^([A-Za-z]\w*)[_\(](-?)([A-Za-z0-9]+)\)?$', s.strip())
            if m:
                base = m.group(1)
                neg  = m.group(2) == "-"
                idx  = m.group(3)
                qi   = (idx.upper() if len(idx) == 1 and idx.isalpha() else idx) if neg else idx
                return f'param(:{base}, _QA.QuIndex("{qi}"))'
            return f'param(:{s.strip()})'

        # Two substitution strategies depending on how the scalar is stored:
        #
        # A) Indexed scalar like xi(-k): stored as QuantumAlgebra Param in
        #    QuTerm.params.  Use Param-based iteration.
        #    Detected by: lhs_str matches "name(index)" pattern.
        #
        # B) Non-indexed scalar like Delta or DeltaStar: declared with
        #    @variables in the Julia session → stored as Symbolics.Num in
        #    the numeric coefficient.  Use Symbolics.substitute on coefficients.
        #    Detected by: lhs_str has NO index (no parenthesis/underscore+char).
        #
        _has_index = bool(re.match(r'^[A-Za-z]\w*[\(_]', lhs_str.strip()))

        if _has_index:
            # Strategy A: Param-based substitution (QuTerm.params vector).
            lhs_julia = _canonical_to_param(lhs_str)
            rhs_julia = _canonical_to_param(rhs_str)
            cmd = "\n".join([
                f"let _nb_src={nb_var}, _nb_lhs_qu={lhs_julia}, _nb_rhs_qu={rhs_julia}",
                "  _nb_old_p = only(keys(_nb_lhs_qu.terms)).params[1]",
                "  _nb_new_p = only(keys(_nb_rhs_qu.terms)).params[1]",
                "  _nb_nt = Dict{_QA.QuTerm, Number}()",
                "  for (_nb_op, _nb_c) in _nb_src.terms",
                "    _nb_new_ps = [p == _nb_old_p ? _nb_new_p : p for p in _nb_op.params]",
                "    _nb_new_op = _QA.QuTerm(_nb_op.nsuminds, _nb_op.δs, _nb_new_ps, _nb_op.expvals, _nb_op.corrs, _nb_op.bares)",
                "    _nb_nt[_nb_new_op] = get(_nb_nt, _nb_new_op, 0) + _nb_c",
                "  end",
                "  normal_form(_QA.QuExpr(_nb_nt))",
                "end",
            ])
        else:
            # Strategy B: Symbolics.substitute on expression.
            # lhs must be a simple identifier; rhs may be an arbitrary expression.
            # Both must already be declared in the Julia session (or will be
            # re-declared here if they are plain identifiers).
            lhs_sym = lhs_str.strip()
            rhs_sym = rhs_str.strip()

            def _merge_sqrt_products(s: str) -> str:
                """Merge sqrt(a)*sqrt(b) → sqrt(a*b) so that sqrt(integer) is
                folded into a symbolic argument before Julia evaluates it.
                Only handles non-nested arguments (no inner parentheses)."""
                for _ in range(5):
                    new = re.sub(
                        r'sqrt\(([^()]*)\)\*sqrt\(([^()]*)\)',
                        lambda m: f'sqrt({m.group(1)}*{m.group(2)})',
                        s,
                    )
                    if new == s:
                        break
                    s = new
                return s

            rhs_sym = _merge_sqrt_products(rhs_sym)

            def _is_numeric_literal(s: str) -> bool:
                try:
                    float(s)
                    return True
                except ValueError:
                    return False

            def _is_simple_identifier(s: str) -> bool:
                """True iff s is a bare Julia identifier (no operators/parens)."""
                import re as _re2
                return bool(_re2.fullmatch(r'[A-Za-z_]\w*', s))

            # Use eval(Meta.parse(...)) to retrieve the exact session variables
            # rather than @variables, which could create a differently-typed
            # symbol that wouldn't match the one already inside _nb_src.
            lhs_esc = lhs_sym.replace('"', '\\"')
            rhs_esc = rhs_sym.replace('"', '\\"')
            lines = [f"let _nb_src={nb_var}"]
            lines += [
                f'  _nb_sub = Dict(eval(Meta.parse("{lhs_esc}")) => eval(Meta.parse("{rhs_esc}")))',
                # Try Symbolics.substitute first (works for Num, BasicSymbolic, etc.).
                # Fall back to QuExpr term iteration if the expression is a quantum operator expr.
                "  if _nb_src isa _QA.QuExpr",
                "    _nb_nt = Dict{_QA.QuTerm, Number}()",
                "    for (_nb_op, _nb_c) in _nb_src.terms",
                "      _nb_new_c = _nb_c isa Symbolics.Num ? Symbolics.substitute(_nb_c, _nb_sub) : _nb_c",
                "      _nb_nt[_nb_op] = get(_nb_nt, _nb_op, 0) + _nb_new_c",
                "    end",
                "    normal_form(_QA.QuExpr(_nb_nt))",
                "  else",
                "    Symbolics.substitute(_nb_src, _nb_sub)",
                "  end",
                "end",
            ]
            cmd = "\n".join(lines)
        return BuildResult(ok=True, command=cmd)

    # declare_symmetry: in Julia/QuantumAlgebra there is no built-in symmetry
    # declaration for Symbolics functions — the symmetry is stored in
    # SymbolContext (Python-level) and used at Wick contraction time to
    # normalize κ(j,i) → -κ(i,j) etc.  BackendHandler._build_declare_symmetry
    # (base class) validates and, since JuliaBackendHandler leaves
    # ``_SYMMETRY_COMMANDS`` at its ``None`` default, emits the no-op "true"
    # command so the engine sees a successful side-effect-only operation.

    def _build_pair_substitute(self, args: dict, bctx: "_BuildCtx") -> "BuildResult":
        """pair_substitute: replace operator-pair products with c-numbers.
        Used for mean-field / Hartree-Fock decoupling:
          c†(k)c(k) → n_k,  c†(k)c†(-k) → Delta,  etc.

        args:
          expr       — expression ID or canonical string
          pair_rules — list of {"pattern": "<operator product>", "rhs": "<scalar canonical>"}

        H-3: the `pattern` is written in the SAME user notation as expression
        input — e.g. `c'(k)*c(k)`, `c'(-k)*c(-k)`, `Sz_1*Sz_2` — and is resolved
        here through the standard parser → resolver → JuliaCodeGenerator pipeline
        into a Julia operator expression.  The Julia side (`_nb_bares_key`) then
        evaluates it and derives the `string(term.bares)` match key, so users
        never have to write the internal bares form (`c†(K) c(K)`, Unicode
        superscripts, the -k→K convention).

        Backward compatibility: legacy raw bares strings (containing `†` or a
        Unicode component superscript) are passed through unchanged; the Julia
        side's try/catch falls back to matching them verbatim.
        """
        expr_ref   = str(args.get("expr", ""))
        pair_rules = args.get("pair_rules", [])
        if not expr_ref:
            return BuildResult(ok=False, command="",
                               error="pair_substitute: missing expr")
        if not pair_rules:
            return BuildResult(ok=False, command="",
                               error="pair_substitute: pair_rules must be non-empty")
        if not isinstance(pair_rules, list):
            return BuildResult(ok=False, command="",
                               error="pair_substitute: pair_rules must be a list")

        nb_var = bctx.nb_var_for_expr(expr_ref) or f"_nb_{expr_ref}"

        from notebook.runtime.julia_code_generator import JuliaCodeGenerator
        from notebook.models.ir import Equation as _EqIR, FunctionCall as _FC
        parser, resolver, ctx = bctx.parser, bctx.resolver, bctx.ctx
        # Unicode component superscripts used by QA bares display (Sᶻ, Sˣ, Sʸ).
        _BARES_MARKERS = ("†", "ᶻ", "ˣ", "ʸ")

        def _count_ops_ir(node) -> int:
            """Count operator factors (FunctionCall nodes) in a resolved IR.

            Each quantum operator — `c(k)`, `c'(i,s)`, … — is one FunctionCall
            regardless of how many indices it carries, so index Symbols are not
            recursed into.  Used to distinguish a 2-operator pair from a
            higher-order product.
            """
            if isinstance(node, _FC):
                return 1
            total = 0
            for a in getattr(node, "args", None) or []:
                total += _count_ops_ir(a)
            return total

        def _resolve_pattern(pat: str) -> "tuple[str, int]":
            """User-notation operator product → (Julia operator expression, op count).

            Legacy raw bares strings pass through untouched (the Julia side
            matches them verbatim); their operator count is the whitespace token
            count.  Anything else is parsed and code-generated so the Julia side
            can evaluate it and read its bares key.
            """
            if any(mk in pat for mk in _BARES_MARKERS):
                return pat, len(pat.split())
            surf = parser.parse(pat)
            if surf is None:
                return pat, len(pat.split())
            ir = resolver.resolve(surf, ctx, skip_alphatize=True)
            if isinstance(ir, _EqIR):
                ir = ir.rhs
            return JuliaCodeGenerator(context=ctx).generate(ir), _count_ops_ir(ir)

        # Build JSON for _nb_pair_substitute:
        # [[pattern1, rhs1], [pattern2, rhs2], ...]
        rules_list = []
        for rule in pair_rules:
            if isinstance(rule, dict):
                pattern = str(rule.get("pattern", ""))
                rhs     = str(rule.get("rhs", ""))
            elif isinstance(rule, (list, tuple)) and len(rule) == 2:
                pattern, rhs = str(rule[0]), str(rule[1])
            else:
                return BuildResult(ok=False, command="",
                                   error=f"pair_substitute: invalid rule format: {rule!r}")
            if not pattern or not rhs:
                return BuildResult(ok=False, command="",
                                   error="pair_substitute: each rule needs non-empty pattern and rhs")
            jpat, n_ops = _resolve_pattern(pattern)
            # pair_substitute is, by definition, for a 2-operator product.  A
            # higher-order pattern (e.g. a 4-fermion interaction) would be
            # matched with the order-independent canonical key, which discards
            # the fermionic sign that operator ordering carries — silently
            # wrong.  Reject it and route the user to wick_contract, which
            # decouples a pair out of an n-operator product and keeps the
            # remaining operators.
            if n_ops > 2:
                return BuildResult(
                    ok=False, command="",
                    error=(f"pair_substitute: pattern {pattern!r} has {n_ops} "
                           "operators; this operation replaces a 2-operator pair "
                           "only. For mean-field decoupling of a higher-order "
                           "(e.g. 4-fermion) interaction, use wick_contract."))
            rules_list.append([jpat, rhs])

        import json as _json
        pairs_json = _json.dumps(rules_list)
        # Escape for Julia string literal: backslash and double-quote
        pairs_julia = pairs_json.replace("\\", "\\\\").replace('"', '\\"')
        cmd = f'_nb_pair_substitute({nb_var}, "{pairs_julia}")'
        return BuildResult(ok=True, command=cmd)

    def _build_operator_taylor_expand(self, args: dict, bctx: "_BuildCtx") -> "BuildResult":
        """operator_taylor_expand: f(H) ≈ Σ a_k H^k"""
        func  = str(args.get("func", ""))
        op    = str(args.get("op", ""))
        var   = str(args.get("var", ""))
        order = int(args.get("order", 0))
        if not func or not op or not var:
            return BuildResult(ok=False, command="",
                               error="operator_taylor_expand: missing func, op, or var")
        nb_op = bctx.nb_var_for_expr(op) or op
        func_esc = func.replace("\\", "\\\\").replace('"', '\\"')
        return BuildResult(ok=True,
                           command=f'_nb_operator_taylor_expand("{func_esc}", {nb_op}, "{var}", {order})')

    def _build_heisenberg_operator(self, args: dict,
                                   bctx: "_BuildCtx") -> "BuildResult":
        r"""PAIR-REP: 何が ad_H の連鎖を閉じるか、そしてその行列は何か。

        `A(t)=e^{iHt}Ae^{-iHt}` は `Σ (it)^n/n! ad_H^n(A)`（docs/33 §5）。連鎖が
        **有限個の演算子で閉じる**なら——二次形式なら閉じ、原子極限の相互作用も
        閉じることが実測されている——係数は `exp(itM)` の1行になる。だから QA に
        訊くのは級数ではなく、**閉包とその行列**だけである。

        各項から**素の演算子**を作り直して基底の同一性を見る（QuTerm の params を
        空にして QuExpr を組み直す）。残った数値とパラメータがその成分になる。
        係数の組み立てはここでしない——QA は c 数を分解するので（docs/33 §6.1）、
        指数関数はスカラー側（SymPy）の仕事である。

        返すのは ``"基底;…|行;…"``、閉じなければ ``NOTCLOSED``。
        """
        from notebook.runtime.pair_representation import CLOSURE_CAP
        op_ref = str(args.get("operator", ""))
        ham_ref = str(args.get("hamiltonian", ""))
        if not op_ref or not ham_ref:
            return BuildResult(ok=False, command="",
                               error="heisenberg_operator: missing operator or hamiltonian")

        # An operand is EITHER an expression id (the Hamiltonian, normally) or a
        # written operator (`c(m)`).  `ir_args` already holds the resolved IR of
        # both — the id expanded, the written one parsed — so the Julia source
        # comes from the generator rather than from the raw text, which would
        # spell a symbolic index `m` as a Julia variable instead of `:m`.
        def _src(ref: str, key: str) -> str:
            node = bctx.ir_args.get(key)
            if node is not None:
                from notebook.runtime.julia_code_generator import JuliaCodeGenerator
                # NOT `static_index_mode`: that spelling drops the ∑ of a bound
                # sum (measured — the Hamiltonian arrived as a single term and
                # its commutator kept an unconsumed δ).
                return JuliaCodeGenerator(context=bctx.ctx).generate(node)
            # `ir_args` is preferred over `nb_var_for_expr` because the latter
            # re-parses `input_canonical` and DROPS the summation.
            return bctx.nb_var_for_expr(ref) or ref

        op_src, ham_src = _src(op_ref, "operator"), _src(ham_ref, "hamiltonian")
        cap = CLOSURE_CAP
        command = "\n".join([
            f"_nb_h = {ham_src}",
            f"_nb_basis = Any[{op_src}]",
            f"_nb_keys = String[string({op_src})]",
            "_nb_rows = String[]",
            "_nb_i = 1",
            f"while _nb_i <= length(_nb_basis) && length(_nb_basis) <= {cap}",
            # Julia's soft scope does not reach into a loop run this way, so the
            # names the walk advances have to be declared (measured: without it
            # the whole walk died with `UndefVarError: _nb_i`).
            "  global _nb_i, _nb_basis, _nb_keys, _nb_rows",
            # Same δ evaluation every other operator operation in this file
            # applies.  The walk was the one site that did not ask, and it is
            # the site where the answer decides whether the chain is judged
            # CONCRETE: the Hubbard atom's chain closes carrying `δ(↑↓)`, so
            # without this the plainest interacting case there is was refused
            # as "a family with free indices" (measured 2026-08-22).
            "  _nb_ad = _nb_eval_site_deltas(normal_form(comm(_nb_h, _nb_basis[_nb_i])))",
            "  _nb_cells = String[]",
            "  for (_nb_k, _nb_c) in _nb_ad.terms",
            "    _nb_bare = typeof(_nb_k)(_nb_k.nsuminds, _nb_k.δs, "
            "QuantumAlgebra.Param[], _nb_k.expvals, _nb_k.corrs, _nb_k.bares)",
            "    _nb_be = typeof(_nb_ad)(Dict(_nb_bare => 1))",
            "    _nb_s = string(_nb_be)",
            "    _nb_j = findfirst(==(_nb_s), _nb_keys)",
            "    if _nb_j === nothing",
            "      push!(_nb_basis, _nb_be)",
            "      push!(_nb_keys, _nb_s)",
            "      _nb_j = length(_nb_keys)",
            "    end",
            '    _nb_p = join(string.(_nb_k.params), "*")',
            '    push!(_nb_cells, string(_nb_j) * ":" * string(_nb_c) * '
            '(isempty(_nb_p) ? "" : "*" * _nb_p))',
            "  end",
            '  push!(_nb_rows, join(_nb_cells, ","))',
            "  _nb_i += 1",
            "end",
            f'length(_nb_basis) > {cap} ? "NOTCLOSED" : '
            'join(_nb_keys, ";") * "|" * join(_nb_rows, ";")',
        ])
        return BuildResult(ok=True, command=command)

    def scalar_simplify_command(self, expr: str) -> str:
        r"""PAIR-REP の係数を SymPy で簡約し、``値|latex`` で返す。

        ``cos(wt)cos(wu) + sin(wt)sin(wu)`` が ``cos(w(t-u))`` になるのはここ——
        異時刻の反交換関係が**時間差だけの関数**になるという、この表現でいちばん
        見たい性質が出る場所である。
        """
        escaped = expr.replace("\\", "\\\\").replace('"', '\\"')
        return "\n".join([
            f'_nb_ps = SymPy.simplify(SymPy.sympy.sympify("{escaped}"))',
            'string(_nb_ps) * "|" * SymPy.latex(_nb_ps)',
        ])

    def pair_exponential_command(self, entries: "list[str]", size: int,
                                 time_var: str) -> str:
        r"""PAIR-REP: `exp(i t M)` の**第1行**を、値と LaTeX の対で返す。

        係数の住処はスカラー側である（QA は c 数を分解する）。このセッションには
        SymPy が既に読み込まれているので、行列指数はそこへ出す——`ad` の連鎖が
        線形常微分方程式であることの、そのままの言い換えである。

        表示も SymPy 自身の latex を使う: 同じ量の綴りを2箇所で決めないため。
        """
        if not entries or size <= 0 or len(entries) != size * size:
            raise ValueError("pair_exponential: bad matrix")
        cells = ", ".join(
            'SymPy.sympy.sympify("' + e.replace("\\", "\\\\").replace('"', '\\"')
            + '")' for e in entries)
        command = "\n".join([
            f'_nb_pt = SymPy.symbols("{time_var}")',
            f"_nb_pM = SymPy.sympy.Matrix({size}, {size}, [{cells}])",
            "_nb_pE = (SymPy.sympy.I * _nb_pt * _nb_pM).exp()",
            'join([string(SymPy.simplify(_nb_pE[1,_nb_j])) * "|" * '
            'SymPy.latex(SymPy.simplify(_nb_pE[1,_nb_j])) '
            f'for _nb_j in 1:{size}], "@@")',
        ])
        return command

    def _build_extract_order(self, args: dict, bctx: "_BuildCtx") -> "BuildResult":
        """extract_order: extract a specific power-of-var term."""
        expr_ref = str(args.get("expr", ""))
        var      = str(args.get("var", ""))
        order    = int(args.get("order", 0))
        if not expr_ref or not var:
            return BuildResult(ok=False, command="",
                               error="extract_order: missing expr or var")
        return BuildResult(ok=True,
                           command=f'_nb_extract_order({bctx.julia_expr_arg(expr_ref)}, "{var}", {order})')

    def _build_truncate_series(self, args: dict, bctx: "_BuildCtx") -> "BuildResult":
        """truncate_series: drop terms above given order in var."""
        expr_ref = str(args.get("expr", ""))
        var      = str(args.get("var", ""))
        order    = int(args.get("order", 0))
        if not expr_ref or not var:
            return BuildResult(ok=False, command="",
                               error="truncate_series: missing expr or var")
        return BuildResult(ok=True,
                           command=f'_nb_truncate_series({bctx.julia_expr_arg(expr_ref)}, "{var}", {order})')

    def _build_wick_contract(self, args: dict, bctx: "_BuildCtx") -> "BuildResult":
        """wick_contract: Wick contraction of n-th order operator products.
        Extracts a specified operator pair, replaces it with a c-number
        (contraction value), and keeps the remaining (n-2)-th order product.

        Returns a JSON array of QuExpr strings — one per contraction term.
        The engine registers each as a separate Expression.

        args:
          expr              — expression ID
          contraction_rules — list of {op1, op2, value} dicts
            op1:   operator spec for first operator  (e.g. "a'" or "a")
            op2:   operator spec for second operator
            value: contraction value template string; use INDEX1, INDEX2 as
                   placeholders for the indices of op1 and op2 respectively.
                   If the value is a plain function name (no INDEX1/INDEX2),
                   it is applied as value(INDEX1, INDEX2).
          include_constant  — bool, include doubly-contracted scalar terms
        """
        expr_ref          = str(args.get("expr", ""))
        contraction_rules = args.get("contraction_rules", [])
        include_constant  = bool(args.get("include_constant", False))

        if not expr_ref:
            return BuildResult(ok=False, command="",
                               error="wick_contract: missing expr")
        if not contraction_rules:
            return BuildResult(ok=False, command="",
                               error="wick_contract: contraction_rules must be non-empty")

        nb_var = bctx.nb_var_for_expr(expr_ref) or f"_nb_{expr_ref}"

        # Build the rules list for _nb_wick_contract.
        # op spec "a'" means creation (prime = dagger), "a" means annihilation.
        julia_rules = []
        for rule in contraction_rules:
            if not isinstance(rule, dict):
                return BuildResult(ok=False, command="",
                                   error=f"wick_contract: invalid rule: {rule!r}")
            op1_raw = str(rule.get("op1", ""))
            op2_raw = str(rule.get("op2", ""))
            value   = str(rule.get("value", ""))
            if not op1_raw or not op2_raw or not value:
                return BuildResult(ok=False, command="",
                                   error="wick_contract: each rule needs op1, op2, value")

            # Parse operator spec: strip primes to get base name; prime = create
            op1_name   = op1_raw.rstrip("'†")
            op1_create = op1_raw.endswith("'") or op1_raw.endswith("†")
            op2_name   = op2_raw.rstrip("'†")
            op2_create = op2_raw.endswith("'") or op2_raw.endswith("†")

            # If value has no INDEX1/INDEX2 placeholders, use as plain scalar.
            # (Phase 2 will support index-dependent values.)
            # Users can write "V_HF(INDEX1,INDEX2)" explicitly for index-dependent values.
            value_tmpl = value

            julia_rules.append({
                "op1_name":   op1_name,
                "op1_create": op1_create,
                "op2_name":   op2_name,
                "op2_create": op2_create,
                "value":      value_tmpl,
            })

        import json as _json

        # Build symmetry_map from SymbolContext for value function names.
        # This allows κ(j,i) → -κ(i,j) normalization in _nb_wc_build_value.
        _sym_map: dict[str, str] = {}
        if self._context:
            for _rule in julia_rules:
                _vt = _rule.get("value", "")
                # Extract function name from template e.g. "kappa(INDEX1,...)"
                _fn = _vt.split("(")[0].strip() if "(" in _vt else _vt.strip()
                if _fn:
                    _prop = self._context.get(_fn)
                    if _prop and _prop.quantity.symmetry:
                        _sym_map[_fn] = _prop.quantity.symmetry

        rules_json      = _json.dumps(julia_rules)
        symmetry_json   = _json.dumps(_sym_map)
        rules_escaped   = rules_json.replace("\\", "\\\\").replace('"', '\\"')
        symmetry_escaped = symmetry_json.replace("\\", "\\\\").replace('"', '\\"')
        incl_const      = "true" if include_constant else "false"
        cmd = (f'_nb_wick_contract({nb_var}, "{rules_escaped}", '
               f'{incl_const}, "{symmetry_escaped}")')
        return BuildResult(ok=True, command=cmd)

    def _build_commutator(self, args: dict, expr_lookup: dict, bctx: "_BuildCtx") -> "BuildResult | None":
        """commutator: try SymPy sum commutator expansion before falling through
        to the QuantumAlgebra template (normal_form(comm(a, b))).

        Returns BuildResult if handled, None to fall through to template dispatch.

        The template-based system calls generate(Add_IR) and generate(IndexedSymbol_IR)
        separately to produce the {a} and {b} template slots — it never creates a
        Commutator IR node, so _maybe_expand_commutator in JuliaCodeGenerator is
        never triggered via the template path.

        Here we construct a temporary Commutator IR node and call
        _maybe_expand_commutator directly.  If it succeeds (multi-term Hamiltonian
        with algebraic indices), the SymPy helper is used.  Otherwise we fall
        through to the QuantumAlgebra template.
        """
        ir_args = bctx.ir_args
        ctx = bctx.ctx

        _comp = getattr(self._builder, "_compiler", None)
        if (
            _comp is not None
            and hasattr(_comp, "_maybe_expand_commutator")
            and "a" in ir_args
            and "b" in ir_args
        ):
            from notebook.models.ir import Commutator as _CommIR
            _comm_node = _CommIR(a=ir_args["a"], b=ir_args["b"])
            _expanded = _comp._maybe_expand_commutator(_comm_node)
            if _expanded is not None:
                return BuildResult(ok=True, command=_expanded)

        # Mixed QA/SymPy commutator: if either operand has sympy_canonical,
        # QA cannot compute comm(QuExpr, Tuple) — route entirely to SymPy.
        _a_ref = str(args.get("a", ""))
        _b_ref = str(args.get("b", ""))
        _a_sympy = self._get_sympy_canonical(_a_ref, expr_lookup)
        _b_sympy = self._get_sympy_canonical(_b_ref, expr_lookup)
        if _a_sympy or _b_sympy:
            _mixed = self._build_mixed_commutator(
                _a_ref, _b_ref, _a_sympy, _b_sympy,
                args, expr_lookup, ctx
            )
            if _mixed is not None:
                return _mixed

        return None

    def _build_define_scalar_parameter(self, args: dict) -> "BuildResult":
        """define_scalar_parameter: emit @variables Name::Type based on domain.
        The julia.full.json template is "nothing" (no-op), so we intercept here
        to emit a typed declaration when the user specifies a domain.
        """
        domain = str(args.get("domain", "") or "").lower().strip()
        all_names: list[str] = []
        if args.get("_all_names"):
            all_names = [str(n) for n in args["_all_names"] if n]
        elif args.get("name"):
            all_names = [str(args["name"])]
        if not all_names:
            return BuildResult(ok=False, command="", error="define_scalar_parameter: missing name")
        julia_type = _DOMAIN_TO_JULIA_TYPE.get(domain, "")
        parts: list[str] = []
        for nm in all_names:
            decl = f"@variables {nm}::{julia_type}" if julia_type else f"@variables {nm}"
            parts.append(decl)
            if julia_type:
                parts.append(f'_nb_register_domain("{nm}", "{julia_type}")')
        return BuildResult(ok=True, command="; ".join(parts))

    def _build_change_integration_variables(self, args: dict, bctx: "_BuildCtx") -> "BuildResult":
        """change_integration_variables: multivariable change of variables with the
        Jacobian determinant auto-inserted.  Produces the new integrand

            substitute(f, Dict(old => transform)) * det(Symbolics.jacobian(transform, new_vars))

        which the user then integrates over the new variables.  Mirrors the
        Maxima implementation but emits Symbolics.jl syntax.
        """
        from notebook.runtime.julia_code_generator import JuliaCodeGenerator
        from notebook.models.ir import Equation as _EqIR
        parser, resolver, ctx = bctx.parser, bctx.resolver, bctx.ctx
        expr_lookup = bctx.expr_lookup

        def _split_top(s: str) -> "list[str]":
            parts, depth, buf = [], 0, []
            for ch in str(s):
                if ch in "([{":
                    depth += 1
                elif ch in ")]}":
                    depth -= 1
                if ch == "," and depth == 0:
                    parts.append("".join(buf).strip()); buf = []
                else:
                    buf.append(ch)
            if "".join(buf).strip():
                parts.append("".join(buf).strip())
            return [p for p in parts if p]

        def _to_julia(s: str) -> str:
            surf = parser.parse(str(s))
            if surf is None:
                return str(s)
            ir = resolver.resolve(surf, ctx)
            if isinstance(ir, _EqIR):
                ir = ir.rhs
            return JuliaCodeGenerator(context=ctx).generate(ir)

        expr_ref = args.get("expr", "")
        integrand_src = (expr_lookup.get(f"__input__{expr_ref}")
                         or expr_lookup.get(expr_ref, expr_ref))
        old_vars = _split_top(args.get("old_vars", ""))
        new_vars = _split_top(args.get("new_vars", ""))
        transform_raw = _split_top(args.get("transform", ""))
        if not (old_vars and new_vars and transform_raw
                and len(old_vars) == len(transform_raw)):
            return BuildResult(
                ok=False, command="",
                error="change_integration_variables: old_vars and transform must "
                      "have equal length, and new_vars must be given")

        integrand = _to_julia(integrand_src)
        transform = [_to_julia(t) for t in transform_raw]
        subst_pairs = ", ".join(f"{o} => {t}" for o, t in zip(old_vars, transform))
        jac_list = ", ".join(transform)
        nv_list = ", ".join(new_vars)
        cmd = (f"substitute({integrand}, Dict({subst_pairs})) "
               f"* det(Symbolics.jacobian([{jac_list}], [{nv_list}]))")
        return BuildResult(ok=True, command=cmd)

    def _build_expand_neighbor_sum(self, args: dict, bctx: "_BuildCtx") -> "BuildResult":
        """expand_neighbor_sum (F-1): expand a nearest-neighbor bond sum on a 1D
        chain into the explicit sum of per-site operator terms.

        `Σ_{⟨ij⟩} body(i,j)` on an N-site chain becomes
            body(1,2) + body(2,3) + … + body(N-1,N)        [open]
            … + body(N,1)                                  [periodic, N≥3]

        The site indices in `body` are substituted at the user-notation level
        (e.g. `Sz_i*Sz_j` → `Sz_1*Sz_2`) and the whole sum is then resolved
        through the standard parser → resolver → JuliaCodeGenerator pipeline,
        reusing the existing per-site spin/operator handling.

        args:
          body     — operator product over two site indices, e.g. "Sz_i*Sz_j"
          index1   — first site index letter (e.g. "i")
          index2   — second site index letter (e.g. "j")
          n_sites  — number of chain sites N (int ≥ 2)
          boundary — "open" (default) or "periodic"
        """
        from notebook.runtime.julia_code_generator import JuliaCodeGenerator
        from notebook.runtime.backend_handler import build_neighbor_sum_terms
        from notebook.models.ir import Equation as _EqIR

        parser, resolver, ctx = bctx.parser, bctx.resolver, bctx.ctx
        sum_str, err = build_neighbor_sum_terms(
            args.get("body", ""), args.get("index1", ""), args.get("index2", ""),
            args.get("n_sites"), args.get("boundary", "open"),
            args.get("pair_constraint", "nearest_neighbor"),
            args.get("lattice_shape"),
            args.get("edges"),
            args.get("pin_site"),
        )
        if err:
            return BuildResult(ok=False, command="", error=err)

        # Apply the same spin-component normalization the normal input path uses
        # (S^z_1 → Sz_1, spin arrows, …); the direct parse below would otherwise
        # mis-read S^z as an operator S applied to a symbol z.
        if hasattr(self, "pre_process_canonical"):
            sum_str = self.pre_process_canonical(sum_str)

        surf = parser.parse(sum_str)
        if surf is None:
            return BuildResult(ok=False, command="",
                               error=f"expand_neighbor_sum: could not parse expanded sum {sum_str!r}")
        ir = resolver.resolve(surf, ctx, skip_alphatize=True)
        if isinstance(ir, _EqIR):
            ir = ir.rhs
        cmd = JuliaCodeGenerator(context=ctx).generate(ir)
        return BuildResult(ok=True, command=cmd)

