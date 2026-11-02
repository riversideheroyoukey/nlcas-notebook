"""NotebookEngine mixin: FREE-PARAM: arbitrary constants of general solutions (ODE general
solutions, specialization by conditions), and solve-result registration.

Extracted verbatim from engine.py (refactor Stage 3, 2026-07-04),
following the EngineHistoryMixin / EnginePersistenceMixin pattern:
state is reached via self; NotebookEngine appears only in annotations
(PEP 563), so there is no import from engine.py and no cycle.
"""
from __future__ import annotations

import re
from datetime import (  # noqa: F401
    datetime,
    timezone,
)
from notebook.runtime.shared_parser import split_relation, strip_lhs
from notebook.models.expression import (  # noqa: F401
    Expression,
)
from notebook.models.step import (  # noqa: F401
    Step,
    StepKind,
    StepStatus,
)
from notebook.runtime.i18n import (  # noqa: F401
    t,
)


class EngineFreeParamsMixin:
    """FREE-PARAM: arbitrary constants of general solutions (ODE general  (see module docstring)."""

    def _fresh_free_param(self, base: str = "C", avoid_in: str = "") -> str:
        """A free-parameter name (FREE-PARAM) that does NOT collide with anything in
        scope: a registered symbol, an existing expression's free parameter, or a
        symbol appearing in *avoid_in* (the result/integrand — so integrating an
        expression that already contains ``C`` does not conflate it with the new
        constant).  Returns base, else base_1, base_2, ….”"""
        used = set(self.symbol_context.registry.keys())
        for e in self.expressions.values():
            used.update(getattr(e, "free_parameters", []) or [])
        if avoid_in:
            try:
                used.update(self._handler.canonical_variables(avoid_in))
            except Exception:
                pass
        if base not in used:
            return base
        i = 1
        while f"{base}_{i}" in used:
            i += 1
        return f"{base}_{i}"
    def _declare_free_parameters(self, names) -> None:
        """Declare minted free-parameter names in the SymbolContext (docs/32 §2.7).

        A free parameter is a name the CAS ALREADY HOLDS — ``ode2``'s ``%k1``
        was renamed to ``k_1`` inside Maxima — but nothing declared it on the
        Python side, so ``k_1`` read as an undeclared subscripted symbol and
        lowered to ``k(1)``.  Substituting an initial condition
        (``k_1 = 0`` — the whole point of FREE-PARAM) then addressed a
        different object and silently did nothing.  Declaring the name at the
        moment it is attached to a result keeps both directions consistent.

        Only names that actually reach an expression are declared: solve_ode_general
        mints ``k_1``/``k_2``/``C`` up front but a given ODE uses only some of them,
        and declaring an unused one would reserve the name for nothing.
        """
        for _nm in names or []:
            if _nm and _nm not in self.symbol_context.registry:
                self.register_quantity(_nm, kind="scalar")

    def specialize_by_condition(self, expr_id: str, var: str, point: str,
                                value: str) -> Expression:
        """Determine a free parameter (FREE-PARAM) from a condition F(var=point)=value
        and return the specialized (particular) solution.  E.g. the general
        antiderivative F(x)+C with the condition F(0)=0 → C solved → particular F.
        Single-parameter case (one condition fixes one constant)."""
        e = self.expressions[expr_id]
        params = list(e.free_parameters)
        if not params:
            raise ValueError("specialize_by_condition: expression has no free parameters")
        F = e.canonical
        c = params[0]
        # solve F(var=point) = value for the constant, then substitute it back.
        cmd = (f"subst({c}=rhs(solve(subst({var}=({point}), {F})=({value}), {c})[1]), {F})")
        val, latex = self._cas_value_and_latex(cmd)
        ex = self._register_plain(latex, val)
        ex.free_parameters = params[1:]   # any remaining undetermined constants
        return ex
    def specialize_by_conditions(self, expr_id: str, conditions: "list[dict]",
                                 depvar: "str | None" = None,
                                 indepvar: "str | None" = None) -> Expression:
        """Specialize an equation-form general solution (FREE-PARAM) by SEVERAL
        conditions on the dependent variable and its derivatives.

        Each condition is ``{order, point, value}`` meaning
        ``d^order q / dt^order |_{t=point} = value`` (order 0 = the value itself).
        The conditions form a linear system in the free parameters; solving it and
        substituting back yields the particular solution.  Generalizes
        :meth:`specialize_by_condition` to N parameters and equation form — the same
        mechanism the math↔physics workflow needs for ODE (``k_1,k_2``) and, later,
        Laplace (``q(0),\\dot q(0)``) results.

        Equivalent in intent to Maxima's ``ic1``/``ic2``/``bc2``, but expressed as a
        backend-neutral condition system so the CAS mechanism stays hidden."""
        e = self.expressions[expr_id]
        params = list(e.free_parameters)
        if not params:
            raise ValueError(t("engine.specialize_no_params", self.lang))
        if not conditions:
            raise ValueError(t("engine.specialize_no_conditions", self.lang))
        if len(conditions) > len(params):
            raise ValueError(t("engine.specialize_overdetermined", self.lang,
                               n_cond=len(conditions), n_param=len(params)))
        if depvar is None or indepvar is None:
            _dv, _iv = self._infer_ode_variables(e)
            depvar = depvar or _dv
            indepvar = indepvar or _iv
        indepvar = indepvar or "t"
        F = e.canonical or ""
        # This goes into `subst(t=(p), rhs)` and `diff(rhs, …)`, so a mis-split
        # is CAS code.  `strip_lhs` returns the text unchanged when it is not an
        # equation — the old `else F` branch — and is bracket-aware.
        rhs = strip_lhs(F)
        norm = self._handler.normalize_bound
        eqs: list[str] = []
        for cond in conditions:
            order = int(cond.get("order", 0))
            point = norm(str(cond["point"]))
            value = norm(str(cond["value"]))
            if order == 0:
                lhs = f"subst({indepvar}=({point}), {rhs})"
            else:
                lhs = f"subst({indepvar}=({point}), diff({rhs}, {indepvar}, {order}))"
            eqs.append(f"{lhs}=({value})")
        # Solve for ALL free parameters at once (order-independent and correct for
        # both the well-determined case n_cond == n_param and the partial case
        # n_cond < n_param, where unconstrained constants come back as backend
        # placeholders that are mapped back to their original names below).
        sol_raw = self._adapter.execute(
            f"block([_s], _s: solve([{', '.join(eqs)}], [{', '.join(params)}]),"
            f" if _s = [] then \"[]\" else string(_s[1]))")
        if not (sol_raw.ok and sol_raw.value) or sol_raw.value.strip() in ("[]", ""):
            raise ValueError(t("engine.specialize_unsolvable", self.lang))
        pairs = self._handler.parse_solve_result(sol_raw.value)
        subst_pairs: list[str] = []
        rename_pairs: list[str] = []
        remaining: set[str] = set()
        for var, vval in pairs:
            subst_pairs.append(f"{var}=({vval})")
            if self._handler.is_free_solve_const(vval):
                rename_pairs.append(f"{vval.strip()}={var}")  # %rN → original name
                remaining.add(var)
        inner = f"subst([{', '.join(subst_pairs)}], {rhs})"
        cmd = (f"subst([{', '.join(rename_pairs)}], {inner})"
               if rename_pairs else inner)
        val, latex = self._cas_value_and_latex(cmd)
        # Re-attach the dependent-variable LHS so the particular solution keeps the
        # equation form "q = …" of the general solution it came from.
        if depvar:
            val = f"{depvar} = {val}"
            latex = f"{depvar} = {latex}"
        ex = self._register_plain(latex, val)
        ex.free_parameters = [p for p in params if p in remaining]  # still undetermined
        return ex
    def _infer_ode_variables(self, e: Expression) -> "tuple[str | None, str]":
        """Infer the dependent and independent variables of an ODE expression.

        The dependent variable is the time-dependent quantity that appears under a
        derivative (``q_dot``/``q_ddot``/``diff(q,…)``) in the equation; the
        independent variable is its declared time parameter (default ``t``).  A
        forcing term that is also time-dependent (e.g. ``v`` in an RLC equation)
        does not appear differentiated, so it is not mistaken for the unknown."""
        cano = e.canonical or ""
        fallback: tuple[str | None, str] = (None, "t")
        for name, prop in self.symbol_context.registry.items():
            q = prop.quantity
            if not q.time_dependent:
                continue
            t = q.time_parameter or "t"
            if (f"{name}_dot" in cano or f"{name}_ddot" in cano
                    or f"diff({name}" in cano):
                return name, t
            if fallback[0] is None:
                fallback = (name, t)
        return fallback
    def solve_ode_general(self, expr_id: str, depvar: "str | None" = None,
                          indepvar: "str | None" = None) -> Expression:
        """General solution of a linear ODE (FREE-PARAM): the result carries its
        arbitrary constants as named free parameters (``k_1``, ``k_2`` for a
        2nd-order equation; ``C`` for 1st order), kept symbolic until a later
        condition specializes them — the math / general-solution workflow.

        ``depvar``/``indepvar`` are inferred from the session's time-dependent
        quantities when omitted."""
        e = self.expressions[expr_id]
        if depvar is None or indepvar is None:
            _dv, _iv = self._infer_ode_variables(e)
            depvar = depvar or _dv
            indepvar = indepvar or _iv
        if not depvar:
            raise ValueError(t("engine.ode_no_depvar", self.lang))
        avoid = e.canonical or ""
        names = {"k1": self._fresh_free_param("k_1", avoid_in=avoid),
                 "k2": self._fresh_free_param("k_2", avoid_in=avoid),
                 "c":  self._fresh_free_param("C", avoid_in=avoid)}
        step, _val = self.run_operation("solve_ode_general", {
            "equation": expr_id, "depvar": depvar,
            "indepvar": indepvar or "t", "const_names": names,
        })
        oid = step.output_refs[0] if step.output_refs else None
        return self.expressions[oid] if oid else e
    def substitute_free_parameters(self, expr_id: str,
                                   pairs: "list[tuple[str, str]]",
                                   s: str = "s") -> Expression:
        """Fix named free parameters (FREE-PARAM) by VALUE: substitute each
        ``param = value`` and return the specialized expression.  Used to state
        initial conditions explicitly on an s-domain expression — e.g. ``q(0)=0,
        q̇(0)=0`` substitutes the IC free parameters ``q_ic0, q_ic1`` with 0 — but is
        general (any free parameter, any value).  The general value-substitution side
        of FREE-PARAM (cf. :meth:`specialize_by_condition` for the solve side)."""
        e = self.expressions[expr_id]
        names = {p for p, _ in pairs}
        unknown = names - set(e.free_parameters or [])
        if unknown:
            raise ValueError(t("engine.not_free_parameters", self.lang,
                               params=", ".join(sorted(unknown))))
        subst_str = ", ".join(f"{p}=({v})" for p, v in pairs)

        def _img(side: str) -> str:
            # The side is the researcher's writing, so it is lowered before it
            # becomes CAS code — measured, `\sum_{i=1}^{3} c_i` went to Maxima
            # verbatim, could not be read, and the ADAPTER'S SCRATCH NAME came
            # back: the registered expression read `_res_ = v*y+2*x`
            # (open_items BUILDER-OPERAND-NOT-LOWERED).  The substitution PAIRS
            # (`subst_str`) are names and values the caller already validated,
            # and stay as they are — lowering one side and not the other is what
            # CLAUDE.md records as the way this goes wrong.
            _low = self._handler.to_backend_code(side)
            _code = side if _low.status == "refused" else _low.code
            _code = self._handler.hold_aggregates(_code)
            r = self._adapter.execute(f"subst([{subst_str}], {_code})")
            return r.value.strip() if (r.ok and r.value) else side
        cano = e.canonical or ""
        # BOTH halves are sent to the CAS and the results are REGISTERED.
        # Measured 2026-08-13 on `\sum_{i=1}^{3} c_i = u*x + v*y`, the naive
        # split sent `subst([u=(2)], \sum_{i)` and
        # `subst([u=(2)], 1}^{3} c_i = u*x + v*y)`; both failed and the adapter's
        # own scratch name came back, registering the expression `_res_ = _res_`
        # — the same signature the solve builder was repaired for in 第3期.
        _rel = split_relation(cano)
        if _rel is not None and "=" in _rel[1]:
            lhs, rhs = _rel[0], _rel[2]
            li, ri = _img(lhs), _img(rhs)
            image = f"{li} = {ri}"
            latex = f"{self._raw_tex(li)} = {self._raw_tex(ri)}"
        else:
            image = _img(cano)
            latex = self._raw_tex(image)
        ex = self._register_plain(latex, image)
        ex.free_parameters = [p for p in e.free_parameters if p not in names]
        return ex
    def _register_plain(self, latex: str, canonical: str) -> Expression:
        """Register a display-only result expression (no CAS, no alias)."""
        display_id = self._alloc_expr_id()
        step_id = self._alloc_step_id()
        expr = Expression(
            display_id=display_id, branch_id=self.active_branch_id,
            latex=latex, canonical=canonical, created_by_step=step_id,
        )
        step = Step(
            step_id=step_id, kind=StepKind.expression_registration,
            branch_id=self.active_branch_id,
            parent_step_ids=self._current_parent_ids(),
            visible_action=t("engine.register", self.lang, display_id=display_id),
            output_refs=[display_id], state_affecting=True,
            status=StepStatus.success, timestamp=datetime.now(timezone.utc),
        )
        self.expressions[display_id] = expr
        self.steps[step_id] = step
        self._advance_head(step)
        return expr
    def _rename_free_solve_vars(self, cas_value: str) -> "tuple[str, list[str]]":
        """Under-determined / parametric solve (FREE-PARAM instance 4): a solve with
        fewer equations than unknowns leaves free variables, which Maxima returns as
        ``%r1, %r2, …``.  Rename each to the original variable that equals it (so the
        solution reads ``x = 1 - y`` with ``y`` free), falling back to a fresh name,
        and return (renamed_value, free_parameter_names).  A determined solve (no
        ``%rN``) is returned unchanged with an empty list."""
        parse_fn = getattr(self._handler, "parse_solve_result", None)
        if not parse_fn:
            return cas_value, []
        try:
            pairs = parse_fn(cas_value)
        except Exception:
            return cas_value, []
        rN = set()
        for _var, val in pairs:
            rN.update(re.findall(r"%r\d+", val))
        if not rN:
            return cas_value, []
        rename: dict[str, str] = {}
        free_names: list[str] = []
        # Pure mapping: a pair "var = %rN" means that var is itself the free variable.
        for var, val in pairs:
            v = val.strip()
            if v in rN and v not in rename and var not in free_names:
                rename[v] = var
                free_names.append(var)
        # Any remaining %rN (not a pure single-variable image) → a fresh parameter.
        for r in sorted(rN):
            if r not in rename:
                nm = self._fresh_free_param("t", avoid_in=cas_value)
                while nm in free_names:
                    nm = self._fresh_free_param("t", avoid_in=cas_value + " " + " ".join(free_names))
                rename[r] = nm
                free_names.append(nm)
        out = cas_value
        for r, nm in rename.items():
            out = re.sub(rf"{re.escape(r)}(?!\d)", nm, out)
        return out, free_names
    def _register_solve_results(
        self,
        cas_value: str,
        step_id: str,
        cas_latex: str = "",
        output_id: str = "",
    ) -> "list[Expression]":
        """Parse CAS solve() output and register each solution as an expression.

        Maxima returns e.g. ``[x = 2]`` or ``[[x = 2, y = 1]]``.
        Julia returns e.g. ``"m = F/a"`` or ``"c_k = ..., c_dag = ..."``.
        Each ``var = val`` pair is registered as a separate expression with
        the variable name as the LHS label.

        *output_id* is the slot ``run_operation`` allocated before the CAS ran
        (the command wrapping needs a name for the result).  The FIRST solution
        row is registered straight into it.  It used to take a number of its
        own and then be copied into the slot, with the number popped afterwards
        — which left a hole in the researcher's numbering AND a registration
        step whose output_refs named a deleted expression
        (INTERNAL-INTERMEDIATE-EID-GAP; measured: a 2-equation inverse
        transform produced E3 and E5, plus a step claiming to output E4).

        Dispatches to ``self._handler.parse_solve_result()`` which is
        implemented on both ``MaximaBackendHandler`` and ``JuliaBackendHandler``.

        Returns the list of registered expressions (may be empty if parsing fails).
        """
        parse_fn = getattr(self._handler, "parse_solve_result", None)
        if parse_fn is None:
            return []
        pairs = parse_fn(cas_value)
        # Build a var→latex_rhs map from cas_latex (Maxima LaTeX output).
        # cas_latex is e.g. "\left[ g=l\,\omega^{2}\right] " which contains
        # proper LaTeX for symbols like \omega, while cas_value has "omega".
        _latex_rhs_map: dict[int, str] = {}   # index → LaTeX RHS
        _latex_lhs_map: dict[int, str] = {}   # index → LaTeX LHS
        if cas_latex:
            _lat_parse_fn = getattr(self._handler, "parse_solve_result", None)
            if _lat_parse_fn:
                # Maxima tex1() wraps list results in \left[ ... \right].
                # Convert to plain brackets so parse_solve_result can strip them.
                import re as _re_sr
                _cas_lat_norm = _re_sr.sub(r'\\left\s*\[', '[', cas_latex)
                _cas_lat_norm = _re_sr.sub(r'\\right\s*\]', ']', _cas_lat_norm)
                # NOTE: do NOT strip \, here — parse_solve_result now tracks
                # brace depth and skips \, tokens, so they are not confused
                # with list separators.  Preserving \, is required so that
                # _apply_dot_notation_to_latex can match the Leibniz patterns
                # (e.g. {{d^2}\over{d\,t^2}}\,q → \ddot{q}).
                for _idx, (_lv, _lr) in enumerate(_lat_parse_fn(_cas_lat_norm)):
                    _latex_lhs_map[_idx] = _lv.strip()
                    _latex_rhs_map[_idx] = _lr.strip()
        exprs: list[Expression] = []
        for _pair_idx, (var, val) in enumerate(pairs):
            # Skip a trivial "var = var" row — it appears when a free variable of an
            # under-determined solve was renamed to itself (FREE-PARAM instance 4);
            # the variable is recorded as a free parameter, so the row is redundant.
            if var.strip() == val.strip():
                continue
            # val is Maxima canonical (e.g. "l*omega^2"); use postprocess to get
            # display LaTeX.  Prefer the latex_rhs extracted from cas_latex (which
            # already has \omega etc.) if available; fall back to postprocess.
            canonical = val
            _raw_lat = _latex_rhs_map.get(_pair_idx, "")
            if _raw_lat:
                latex_display = self._handler.postprocess_result_latex(
                    val, _raw_lat, self.symbol_context
                ) or _raw_lat
            else:
                latex_display = (
                    self._handler.postprocess_result_latex(val, val, self.symbol_context)
                    or val
                )
            # lhs_label: always use the canonical var form so that lhs_alias_map
            # stores the canonical key (used by substitution lookups).
            # Separately compute a LaTeX display form for the LHS and patch
            # expr.latex after registration.
            _lhs_latex_raw = _latex_lhs_map.get(_pair_idx, "")
            if _lhs_latex_raw:
                _lhs_display = self._handler.postprocess_result_latex(
                    var, _lhs_latex_raw, self.symbol_context
                ) or _lhs_latex_raw
            else:
                _lhs_display = None
            expr = self.register_expression(
                canonical,
                latex=latex_display,
                lhs_label=var,   # canonical form — keeps lhs_alias_map correct
                skip_cas_bind=True,   # solve results are not session-wide bindings
                _advance_head=False,  # don't insert into lineage; solve step's
                                      # output_refs already lists this expression
                # First row claims the pre-allocated slot; the rest take new
                # numbers as before.
                reuse_display_id=(output_id if not exprs else ""),
            )
            # Patch expr.latex to use proper LaTeX LHS (e.g. \ddot{q} instead of
            # 'diff(q(t),t,2)) for display, without affecting lhs_alias_map keys
            # (the alias is keyed on `var`, the canonical form, above).
            #
            # Through the SAME door as every other registration: written here
            # raw, a solved row published `c_{k} = \frac{alpha_{mk} v + …}` —
            # the α of the two rows above it printed as the italic word "alpha",
            # in the same session (measured 2026-08-09).
            if _lhs_display and _lhs_display != var:
                expr.latex = self._handler.normalize_display_latex(
                    f"{_lhs_display} = {latex_display}"
                )
                self.expressions[expr.display_id] = expr
            exprs.append(expr)
        return exprs
