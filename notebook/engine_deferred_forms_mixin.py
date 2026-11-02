"""NotebookEngine mixin: Deferred-evaluation display objects: definite-integral forms (S-o),
symbolic bond sums (B), and dispersion descriptors (F-1) — registered
without invoking the CAS until explicitly evaluated.

Extracted verbatim from engine.py (refactor Stage 3, 2026-07-04),
following the EngineHistoryMixin / EnginePersistenceMixin pattern:
state is reached via self; NotebookEngine appears only in annotations
(PEP 563), so there is no import from engine.py and no cycle.
"""
from __future__ import annotations

from datetime import (  # noqa: F401
    datetime,
    timezone,
)
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


class EngineDeferredFormsMixin:
    """Deferred-evaluation display objects: definite-integral forms (S-o),  (see module docstring)."""

    def register_direct(self, latex: str, canonical: str | None = None,
                        **descriptor_fields) -> Expression:
        """Register an expression built DIRECTLY — no CAS call, no IR parse.

        The common core of every deferred-evaluation / CAS-bypass registration
        in the notebook: allocate ids, build the Expression (+ whichever
        ``*_descriptor`` the caller passes), record an
        ``expression_registration`` step, advance the head.

        ``canonical`` defaults to ``latex``: a display-only object (an abstract
        lattice sum, an unevaluated ∫…∫, a coupled state) has no
        backend-computable canonical.  Callers that DO have a computed value
        (``evaluate_integral_form``, the F-9 exact symbols) pass it explicitly.
        """
        display_id = self._alloc_expr_id()
        step_id = self._alloc_step_id()
        # Bypassing the CAS is not a reason to bypass the DISPLAY: this is
        # written down like anything else.  Left raw, an expanded matrix element
        # published `\varphi_{m,s}(x_x,x_y,x_z)^{\ast}` while the line above it —
        # its own source — read `\varphi^{\ast}_{m,s}(x)` (measured 2026-08-09).
        # The canonical is untouched; only what is written down is normalised.
        latex = self._handler.normalize_display_latex(latex)
        expr = Expression(
            display_id=display_id,
            branch_id=self.active_branch_id,
            latex=latex,
            canonical=latex if canonical is None else canonical,
            created_by_step=step_id,
            **{k: dict(v) for k, v in descriptor_fields.items() if v is not None},
        )
        step = Step(
            step_id=step_id,
            kind=StepKind.expression_registration,
            branch_id=self.active_branch_id,
            parent_step_ids=self._current_parent_ids(),
            visible_action=t("engine.register", self.lang, display_id=display_id),
            output_refs=[display_id],
            state_affecting=True,
            status=StepStatus.success,
            timestamp=datetime.now(timezone.utc),
        )
        self.expressions[display_id] = expr
        self.steps[step_id] = step
        self._advance_head(step)
        return expr

    def register_bond_sum(self, latex: str, descriptor: dict) -> Expression:
        """Register an UNEXPANDED symbolic bond sum Σ_C body(i,j) (B, stage 1).

        The expression displays the ``\\sum_{…}`` form and carries the
        descriptor needed to concretize it later; the CAS is not invoked
        (an abstract lattice sum has no backend-computable canonical).
        """
        expr = self.register_direct(latex, lattice_descriptor=descriptor)
        # Equation form "H = Σ…": register the LHS name so the symbolic sum can
        # be concretized by name ("H を N サイトで展開").  skip_cas_bind: the RHS
        # is an abstract sum, not a backend-bindable definition.
        if "=" in latex:
            self._update_lhs_alias(expr.display_id, latex, skip_cas_bind=True)
        return expr
    def register_extremum_form(self, latex: str, descriptor: dict) -> Expression:
        r"""Register a display-only extremum binder (``\max_i x_i``).

        Unlike the two deferred forms above, nothing is deferred: the backend
        has no command for an extremum over a symbolic index at all (measured
        2026-08-16).  What this buys is that the notation can be WRITTEN DOWN
        and shown — refusing it at registration, which is what this system did
        for part of that day, stopped a researcher putting an ordinary
        objective function in their own notebook.  A generic operation on it is
        refused at the point of use (`operand_guards`).
        """
        expr = self.register_direct(latex, extremum_descriptor=descriptor)
        # Equation form "z = \max_i x_i": register the LHS name so the object
        # can be referred to.  skip_cas_bind for the same reason as a bond sum
        # — the RHS is not something the backend can hold.
        if "=" in latex:
            self._update_lhs_alias(expr.display_id, latex, skip_cas_bind=True)
        return expr

    def evaluate_extremum_form(self, expr_id: str) -> Expression:
        r"""Evaluate a CONCRETE-range extremum (``\max_{i=1}^{3} x_i``).

        The boundary is measured, not assumed: ``lmax(makelist(x(i),i,1,3))``
        answers ``max(x(1),x(2),x(3))`` and ``lmax(makelist(x(i),i,1,n))``
        FAILS, so only integer bounds have a command.  Anything else raises
        with a reason rather than sending something the CAS will reject.

        The body goes through ``to_backend_code`` — it is the researcher's own
        writing, and handing that to a raw CAS command is exactly how
        ``\int_0^1 \sum_i f_i(x) dx`` reached Maxima as ``sum_i`` earlier the
        same day.
        """
        from notebook.runtime.extremum_form import (
            build_eval_command, build_extremum_latex, build_set_eval_command,
            is_concrete_range, is_strict_range, read_range, read_value_set,
        )
        src = self.expressions[expr_id]
        desc = src.extremum_descriptor or {}
        rng = read_range(desc)
        vset = None if is_concrete_range(rng) else read_value_set(desc)
        if not is_concrete_range(rng) and vset is None:
            # A STRICT inequality is the one case that genuinely cannot be read
            # (integers give `i = 2`, reals give something else, and the
            # notation does not say which), so it gets its own reason.  Saying
            # "the range is not concrete integers" about `1 < i < 3` — where
            # both bounds plainly ARE integers — is the kind of refusal whose
            # reason does not fit its input (open_items EXTREMUM-EXPLICIT-SET).
            key = ("extremum.range_is_strict" if is_strict_range(desc)
                   else "extremum.range_not_concrete")
            raise ValueError(t(key, self.lang, expr_id=expr_id))
        body_canonical = self._handler.latex_to_canonical(desc["body_latex"])
        lowered = self._handler.to_backend_code(body_canonical)
        if not lowered.ok:
            raise ValueError(t("engine.cannot_lower_integrand", self.lang,
                               expr_id=expr_id, reason=lowered.reason))
        # The body's IR has to be READ for the display bridge, because this
        # object was registered through `register_direct` — no IR parse — so
        # nothing has yet told `indexed_bases` that the researcher wrote `x_i`.
        # Without this the answer displayed `max(x\left(1\right), …)` while the
        # SUM path, whose registration does parse, showed `x_{1}` for the same
        # writing (measured 2026-08-16).  The reading is the source's, never the
        # result's: a result canonical says nothing about how it was written.
        from notebook.runtime.indexed_display import record_symbol_usage
        try:
            record_symbol_usage(self._handler.parse_ir(body_canonical),
                                self.symbol_context)
        except Exception:                    # a body the parser cannot read is
            pass                             # still evaluable; display degrades
        command = (build_set_eval_command(desc["op"], lowered.code, vset)
                   if vset is not None
                   else build_eval_command(desc["op"], lowered.code, rng))
        value, value_latex = self._cas_value_and_latex(command)
        latex = f"{build_extremum_latex(desc)} = {value_latex}"
        return self.register_direct(latex, canonical=value)

    def _cas_value_and_latex(self, command: str) -> "tuple[str, str]":
        """Execute a raw CAS command and return (value, display_latex).

        Shared by the integral-form change-of-variables and evaluate steps.
        Raises ValueError on CAS failure.
        """
        result = self._adapter.execute(command)
        if not (result.ok and result.value and not result.value.startswith("ERROR")):
            raise ValueError(result.value or "CAS error")
        value = result.value.strip()
        # execute() already renders LaTeX as a side effect of the same command
        # (Maxima: tex1() wrapper; Julia: driver-side tex output) — reuse it
        # instead of re-issuing a Maxima-only tex1() call, which errors under
        # the Julia backend.
        raw_latex = result.latex.strip() if result.latex else ""
        if not raw_latex:
            try:
                tex_r = self._adapter.execute(f"tex1({value})")
                if tex_r.ok and tex_r.value:
                    raw_latex = tex_r.value.strip()
            except Exception:
                pass
        if not raw_latex:
            raw_latex = value
        display_latex = self._handler.postprocess_result_latex(
            value, raw_latex, self.symbol_context)
        return value, display_latex
    def register_integral_form(self, latex: str, descriptor: dict) -> Expression:
        """Register a display-only unevaluated multiple definite integral (S-o).

        The CAS is not invoked — the expression shows the ∫…∫ form and carries the
        descriptor; change-of-variables rewrites it and an evaluate step computes
        the value.  Mirrors :meth:`register_bond_sum`.
        """
        return self.register_direct(latex, integral_descriptor=descriptor)
    def change_variables_integral_form(
        self, expr_id: str, old_vars: list[str], new_vars: list[str],
        transform: list[str], new_bounds: list[list[str]],
    ) -> Expression:
        """Change of variables on an unevaluated definite integral (S-o step 3).

        Computes the new integrand f∘T·det J (via the change_integration_variables
        builder), simplifies it, and registers a new display-only integral with the
        supplied *new_bounds* (OUTER → INNER).  Still unevaluated.
        """
        from notebook.runtime.integral_form import build_integral_latex
        src = self.expressions[expr_id]
        desc = src.integral_descriptor or {}
        integrand = desc.get("integrand_canonical") or desc.get("integrand_latex")
        build = self._handler.build_command(
            "change_integration_variables",
            {"expr": integrand, "old_vars": ", ".join(old_vars),
             "new_vars": ", ".join(new_vars), "transform": ", ".join(transform)},
            {},
        )
        if not build.ok:
            raise ValueError(build.error)
        raw, _ = self._cas_value_and_latex(build.command)
        # Simplify the raw Jacobian product into textbook form.  Use the unified
        # `simplify` op (trigsimp∘ratsimp) so sin²+cos²=1 folds (ratsimp alone
        # leaves (sin²+cos²)² in polar/spherical measures).
        sbuild = self._handler.build_command("simplify", {"expr": "E0"}, {"E0": raw})
        new_canonical, new_latex = self._cas_value_and_latex(
            sbuild.command if sbuild.ok else raw)
        new_desc = {
            "integrand_canonical": new_canonical,
            "integrand_latex": new_latex,
            "bounds": [list(b) for b in new_bounds],
            "region": None,
        }
        latex = build_integral_latex(new_desc, integrand_latex=new_latex,
                                     format_const=self._handler.format_constant_for_latex)
        return self.register_integral_form(latex, new_desc)
    def evaluate_integral_form(self, expr_id: str) -> Expression:
        """Evaluate an unevaluated definite integral (S-o step 4): assemble the
        iterated CAS integrate, run it, and register ``∫…∫ = value``."""
        from notebook.runtime.integral_form import build_eval_command, build_integral_latex
        src = self.expressions[expr_id]
        desc = src.integral_descriptor or {}
        integrand = desc.get("integrand_canonical") or desc.get("integrand_latex")
        # The integrand is USER FORM when the researcher typed the integral (the
        # S-o input route stores `latex_to_canonical` output) and CAS text when a
        # change of variables produced it.  `build_eval_command` writes a raw CAS
        # command, so it needs the second — and taking the first verbatim is how
        # `\int_0^1 \sum_i f_i(x) \, dx` reached Maxima as
        # `integrate((\sum_i*f_i(x)), x, 0, 1)` and came back as
        # `sum_i*'integrate(f_i(x),x,0,1)` — a bare symbol named `sum_i` times an
        # integral — registered as a success (measured 2026-08-16).  The general
        # route reads that same Σ correctly; this route simply never walked
        # through the door.  Refusing beats handing raw text to the CAS
        # (CLAUDE.md, `to_backend_code`): the caller turns it into a failed step.
        lowered = self._handler.to_backend_code(integrand)
        if not lowered.ok:
            raise ValueError(t("engine.cannot_lower_integrand", self.lang,
                               expr_id=expr_id, reason=lowered.reason))
        cmd = build_eval_command(lowered.code, desc["bounds"])
        value, value_latex = self._cas_value_and_latex(cmd)
        latex = build_integral_latex(desc, result_latex=value_latex,
                                     format_const=self._handler.format_constant_for_latex)
        return self.register_direct(latex, canonical=value)
    def register_dispersion(self, latex: str, descriptor: dict) -> Expression:
        """Register a tight-binding dispersion ε(k) (F-1 stage 3c-fourier).

        Like :meth:`register_bond_sum`, the closed-form ε(k) is built
        deterministically from the lattice's unit-cell hopping structure, so
        the CAS is not invoked.  Unlike a named bond sum it is a derived
        quantity, so no LHS alias is bound.
        """
        return self.register_direct(latex, lattice_descriptor=descriptor)

    def expand_vector_components(self, expr_id: str) -> "tuple[Expression, list[str]]":
        r"""Open every vector in an expression into its components.

        The answer to `\int \frac{d^3k}{(2\pi)^3} e^{i k\cdot x} \tilde f(k)`:
        there is no d-dimensional integral to hand a CAS, so the dimension is
        opened explicitly and what comes out is an ordinary iterated integral
        the CAS can hold.  The rewrite itself — and why the measure, the dot
        product and the function OF the vector all have to be opened together —
        lives in notebook/runtime/vector_components.py.

        The component symbols are REGISTERED as scalars, so a later substitution
        or differentiation sees them as quantities rather than unknowns, and
        their names are returned for the caller to tell the researcher (F-9's
        rule: the system may pick names, and then it says which).
        """
        from notebook.models.quantity import Quantity, MathType
        from notebook.runtime.vector_components import expand
        src = self.expressions[expr_id]
        ir = src.ir_ast or self._sum_bounds_ir(expr_id, src.canonical)
        if ir is None:
            raise ValueError(t("engine.cannot_read_expression", self.lang,
                               expr_id=expr_id))
        suffixes = self._handler.coord_component_suffixes(
            getattr(self.session, "coordinate_system", None) or "cartesian")
        opened, names = expand(ir, self.symbol_context, suffixes,
                               infinity=self._handler.infinity_name)
        for name in names:
            if self.symbol_context.get_quantity(name) is None:
                self.symbol_context.register(name, quantity=Quantity(
                    name=name, kind="anonymous",
                    math_type=MathType(algebra="scalar")))
        value, value_latex = self._cas_value_and_latex(
            self._handler.generate_backend_code(opened, hold_integrals=True))
        return self.register_direct(value_latex or value, canonical=value), names

    def evaluate_held_integrals(self, expr_id: str) -> Expression:
        r"""Carry out an integral that was being HELD — the separate request.

        Opening a vector into components deliberately does not evaluate: over an
        arbitrary basis the coefficient `h_{mn} = \int d^3x \phi_m^* \hat h
        \phi_n` is a DEFINITION and there is nothing to carry out.  So the
        expansion holds (`'integrate`), and this is where the researcher says to
        try.  A failure here is honest — Maxima asking a convergence question it
        cannot be answered is a real answer to "evaluate this", and a silent one
        only if it happens without being asked.
        """
        src = self.expressions[expr_id]
        value, value_latex = self._cas_value_and_latex(f"ev({src.canonical}, nouns)")
        return self.register_direct(value_latex or value, canonical=value)
