"""R1a (docs/27): the interceptor registry runner + the FREE-PARAM batch.

Pins the registry's contract (ordered firing, first-non-None short-circuit,
mutable context) and the migrated batch's exact order — so a future edit that
reorders or drops an interceptor trips a test.
"""
import asyncio

import pytest

from ui.interpret_registry import (
    Interceptor, InterceptorContext, run_interceptors,
)


def _ctx(text="x"):
    return InterceptorContext(engine=None, loop=None, lang="ja",
                              text=text, registered=[])


def _run(ctx, interceptors):
    return asyncio.run(run_interceptors(ctx, interceptors))


class TestRunner:
    def test_fires_in_order_until_first_non_none(self):
        fired = []

        def mk(name, result):
            async def _run_one(ctx):
                fired.append(name)
                return result
            return Interceptor(name, _run_one)

        got = _run(_ctx(), [mk("a", None), mk("b", {"hit": "b"}), mk("c", {"hit": "c"})])
        # The winner's own payload is returned unchanged except for `route`,
        # which the runner stamps so the claiming interceptor is answerable
        # afterwards (第2期; see TestRouteAttribution below).
        assert got == {"hit": "b", "route": "b"}
        assert fired == ["a", "b"]          # c never fires (short-circuit)

    def test_all_none_returns_none(self):
        async def _none(ctx):
            return None
        got = _run(_ctx(), [Interceptor("a", _none), Interceptor("b", _none)])
        assert got is None

    def test_empty_list_returns_none(self):
        assert _run(_ctx(), []) is None

    def test_interceptor_may_rewrite_context_text_for_followers(self):
        seen = []

        async def rewriter(ctx):
            ctx.text = "narrowed"
            return None

        async def observer(ctx):
            seen.append(ctx.text)
            return None

        _run(_ctx("original"),
             [Interceptor("rw", rewriter), Interceptor("obs", observer)])
        assert seen == ["narrowed"]         # follower sees the rewrite


class TestRouteAttribution:
    r"""第2期: WHICH interceptor claimed the input is answerable afterwards.

    The firing ORDER was already data (TestFreeParamBatch below pins it); the
    attribution was not.  That asymmetry has a measured cost — on 2026-08-09 a
    defect was filed against `register_expression`, a path no researcher takes,
    because there was no way to ask which interceptor really handles the input
    (docs/handoffs/one_question_many_answers_handoff.md §2.2).
    """

    class _FakeEngine:
        """Just enough of an engine: an insertion-ordered step table."""
        def __init__(self):
            self.steps = {}

        def record(self, step_id):
            from notebook.models.step import Step, StepKind
            self.steps[step_id] = Step(step_id=step_id, kind=StepKind.manual_note,
                                       branch_id="main", visible_action="x")

    def _ctx_with_engine(self, engine, text="x"):
        return InterceptorContext(engine=engine, loop=None, lang="ja",
                                  text=text, registered=[])

    def test_the_winner_is_recorded_on_the_context_and_the_response(self):
        async def _none(ctx):
            return None

        async def _win(ctx):
            return {"status": "completed"}

        ctx = self._ctx_with_engine(self._FakeEngine())
        got = _run(ctx, [Interceptor("first", _none), Interceptor("second", _win)])
        assert ctx.route == "second"
        assert got["route"] == "second"

    def test_no_winner_leaves_the_route_unset(self):
        async def _none(ctx):
            return None
        ctx = self._ctx_with_engine(self._FakeEngine())
        assert _run(ctx, [Interceptor("a", _none)]) is None
        assert ctx.route is None            # "no interceptor claimed it"

    def test_steps_are_attributed_to_the_interceptor_that_made_them(self):
        engine = self._FakeEngine()
        engine.record("S0")                 # pre-existing (e.g. the YOU card)

        async def _side_effect_then_fall_through(ctx):
            ctx.engine.record("S1")
            return None                     # falls through, but S1 is still its doing

        async def _win(ctx):
            ctx.engine.record("S2")
            return {"status": "completed"}

        _run(self._ctx_with_engine(engine),
             [Interceptor("faller", _side_effect_then_fall_through),
              Interceptor("winner", _win)])
        assert engine.steps["S0"].route is None      # recorded before any ran
        assert engine.steps["S1"].route == "faller"
        assert engine.steps["S2"].route == "winner"

    def test_an_existing_attribution_is_never_overwritten(self):
        """A nested run (an interceptor that itself runs a registry) must keep
        the INNER attribution — it is the more specific answer."""
        engine = self._FakeEngine()

        async def _inner_win(ctx):
            ctx.engine.record("S1")
            return {"ok": 1}

        async def _outer(ctx):
            await run_interceptors(self._ctx_with_engine(ctx.engine),
                                   [Interceptor("inner", _inner_win)])
            return {"ok": 1}

        _run(self._ctx_with_engine(engine), [Interceptor("outer", _outer)])
        assert engine.steps["S1"].route == "inner"

    def test_an_engine_without_steps_is_harmless(self):
        """Unit tests and scripts pass engine=None; attribution must not care."""
        async def _win(ctx):
            return {"ok": 1}
        got = _run(_ctx(), [Interceptor("only", _win)])
        assert got["route"] == "only"


class TestFreeParamBatch:
    def test_registered_order_is_pinned(self):
        # Import here so a collection error in the pipeline surfaces clearly.
        from ui.interpret_pipeline import _FREEPARAM_INTERCEPTORS
        names = [it.name for it in _FREEPARAM_INTERCEPTORS]
        assert names == [
            # 2026-08-05: taylor_expand sits beside definite_integral — both are
            # rule-based only because their four required fields cannot be
            # filled positionally, and neither can swallow anything else (each
            # needs its own verb plus a stated variable).
            # 2026-09-06: `series_expand_no_var` は taylor_expand の**後ろ**。
            # 変数が書かれていればそちらが答え、書かれていないときだけこちらが
            # 拾って「変数を入力してください」と訊く（黙って代数展開に落ちて
            # `exp(x)` を `%e^x` のまま completed で返していた）。
            # 2026-09-18: `integral_in_coordinates` は definite_integral の**前**。
            # `in <名前> coordinates` / `<名前>座標で` は副詞句であって「前置詞＋変数」
            # ではなく、後ろに置くと共有の前置詞表が `polar` を変数の欄に入れて
            # `polar*x^2+C` を completed で返す（実測、HEAD でも同じ）。断るほうが
            # 先に claim する必要がある。
            "integral_in_coordinates",
            "definite_integral", "taylor_expand", "series_expand_no_var",
            "add_assumption", "solve_ode_general",
            # 2026-08-15: `laplace_definition` deleted with the macro it drove
            # (SCENARIO-MACRO case 7).  `define_laplace` stays — it declares the
            # transform PAIR, which is a naming the CAS does not hold under the
            # researcher's own symbol (open_items LAPLACE-PAIR-ANCHOR).
            "define_laplace", "substitute_free_params",
            "specialize_conditions", "specialize_condition",
            # F-9 angular-momentum group: specific triggers first, the loose
            # "<anything> の g 因子" pattern last (see the notes in the registry).
            "angular_wigner_symbol", "angular_clebsch_gordan",
            "angular_coupled_state", "angular_couple",
            "angular_wigner_eckart", "angular_lande_g",
            "z_transform_by_definition",
            # 2026-09-24: `en_genitive_noun` は**最後**。`<操作> the <名詞> of <式>`
            # の名詞が操作を名指していないときだけ断るので、上のどの specific な
            # 読みにも先に機会を与える必要がある（F-1 の `nearest-neighbor sum of …`
            # は割り込みの走行より前に claim される）。それでも**式の自動登録より
            # 前**である——そこが `of` と `widget` を被和関数の因子にしていた場所。
            "en_genitive_noun",
        ]

    def test_every_entry_records_its_ordering_rationale(self):
        from ui.interpret_pipeline import _FREEPARAM_INTERCEPTORS
        assert all(it.note for it in _FREEPARAM_INTERCEPTORS), \
            "each interceptor must document why it sits where it does"


class TestPreludeBatch:
    """R1b (docs/27): the heterogeneous top-level interceptor run.  The three
    non-uniform entries (notation tuple-rewrite, early try/except, change_vars
    supervisor arg) are normalized through InterceptorContext; the order below is
    the exact pre-LLM firing order and must not drift."""

    def test_registered_order_is_pinned(self):
        from ui.interpret_pipeline import _PRELUDE_INTERCEPTORS
        names = [it.name for it in _PRELUDE_INTERCEPTORS]
        assert names == [
            # `norm_notation` は **`notation_candidates` より前**（2026-09-10、
            # NORM-OPERATION の記法配線）。あちらは二重縦棒を「絶対値か行列式か」の
            # ダイアログに落とさないために断っており、先に走ると配線が届かない。
            "vector_op", "declare_bracket_functional",
            "norm_notation", "notation_candidates",
            "plan_auto_select", "plan_candidates", "plan_freeform", "plan_workflow",
            # 2026-08-01: scalar_handoff sits beside numeric_fork_nl — both are
            # "this expression leaves the theory timeline and continues
            # elsewhere", and both name their target explicitly, so neither can
            # swallow an ordinary compute-ask (SCALAR-HANDOFF, docs/34 §5.1).
            "scalar_handoff", "numeric_fork_nl", "early_intercepts", "poisson",
            # EXTREMUM-BINDER (2026-08-16): a display-only `\max_i x_i` sits
            # beside the display-only integral for the same reason — both claim
            # RAW MATH before inline-formula handling, so the binder is not
            # parsed as a plain expression (read as a subscript it computed
            # `i*x(i)` silently).  Its claim needs a SUBSCRIPT, so `\max(a, b)`
            # falls through and computes as an ordinary call.
            "expr_id_equation", "extremum_input", "integral_input",
            "evaluate_extremum", "evaluate_integral",
            # B-10 ②-g: opening a vector into components sits beside evaluating
            # an integral because it is the OTHER half of the same object — a
            # volume measure is written, held, and only then either expanded or
            # evaluated.  Its pattern names no quantities, so it cannot swallow
            # an ordinary compute-ask.
            "expand_components",
            "change_vars",
        ]

    def test_every_entry_records_its_ordering_rationale(self):
        from ui.interpret_pipeline import _PRELUDE_INTERCEPTORS
        assert all(it.note for it in _PRELUDE_INTERCEPTORS), \
            "each interceptor must document why it sits where it does"

    def test_early_tail_order_is_pinned(self):
        """R1c (docs/27): the full early-intercept sequence, now a registry —
        _try_early_intercepts is a thin dispatcher.  This pins the ENTIRE firing
        order (head → spin workflow → spectral/spin/graph → fourier/bond-sum),
        which is load-bearing at several points: spin_commutator_general before
        the lattice spin_commutator, graph_identity before graph (恒等式 must not
        be grabbed as a plain graph), bond_sum_substitute before the bare
        symbolic_bond_sum register (a substitution must not be stored as a fresh
        unexpanded sum)."""
        from ui.interpret_early import _EARLY_INTERCEPTORS
        names = [it.name for it in _EARLY_INTERCEPTORS]
        assert names == [
            # PAIR-REP sits among the head interceptors: its trigger names an
            # operator and an E-id and nothing else claims that shape, but it
            # must precede the spin/lattice workflow so `c(m) を E1 のもとで…`
            # is not read as an ordinary expression registration first.
            "inner_product", "matrix_transform", "heisenberg_operator",
            "pair_bracket", "save_lattice_figure",
            "neighbor_sum", "neighbor_sum_concretize", "bond_sum_commutator",
            "declare_site_spins", "spin_register", "spin_field_magnitude", "spin_commutator_general",
            "spin_commutator", "spin_expectation", "specific_heat", "energy_moment",
            "spectral_terminator", "spectral_function", "spin_spectral",
            "graph_identity", "graph",
            "crystal", "dispersion", "bond_sum_substitute",
            "symbolic_bond_sum",
        ]
        assert all(it.note for it in _EARLY_INTERCEPTORS)

    def test_equation_order_is_pinned(self):
        """R1d (docs/27): _try_equation_intercepts is now a thin dispatcher over
        _EQUATION_INTERCEPTORS.  Order is load-bearing — pending_followup must be
        LAST so every other H-5..H-9/L-1 intercept gets first crack at the input."""
        from ui.interpret_equation_intercepts import _EQUATION_INTERCEPTORS
        names = [it.name for it in _EQUATION_INTERCEPTORS]
        assert names == [
            "rhs_set", "evaluate_rhs", "metric_constraints", "apply_constraints",
            "canonical_constraints", "inverse_transform", "solve", "both_sides",
            "pending_followup",
        ]
        assert all(it.note for it in _EQUATION_INTERCEPTORS)

    def test_rule_order_is_pinned(self):
        """R1d (docs/27): _try_rule_short_circuits is now a thin dispatcher over
        _RULE_INTERCEPTORS.  Order is load-bearing — wrt_qdot before the plain
        differentiate (a q̇-derivative must not be taken as an ordinary variable
        derivative), and transform_subst before the single-source substitute
        paths."""
        from ui.interpret_rules import _RULE_INTERCEPTORS
        names = [it.name for it in _RULE_INTERCEPTORS]
        assert names == [
            # 2026-09-06: `expr_with_assignment` は substitute_into の**後ろ**。
            # `E1 with x = 2` / `evaluate E1 at x = 2` を**形**で読む枝で、
            # 通常の代入の綴りが先に答える（直す前は前者が式として登録され、
            # canonical `(a*x + b)*with*x = 2` を返していた）。
            "wrt_qdot", "differentiate", "transform_subst", "substitute_into",
            "expr_with_assignment",
            "dainyu", "substitute", "set_notation", "set_latex_name", "combine",
        ]
        assert all(it.note for it in _RULE_INTERCEPTORS)

    def test_notation_wrapper_rewrites_context_text(self):
        """The notation entry must thread its (response, rewritten_text) tuple
        back into ctx.text so followers see the narrowed form."""
        import asyncio
        from ui.interpret_pipeline import _run_notation_interceptor
        from ui.interpret_registry import InterceptorContext
        import ui.interpret_pipeline as pipe

        calls = {}

        async def _fake_notation(engine, loop, lang, text, tokens, registered):
            calls["text"], calls["tokens"] = text, tokens
            return None, "narrowed"

        orig = pipe._try_notation_candidates
        pipe._try_notation_candidates = _fake_notation
        try:
            ctx = InterceptorContext(engine=None, loop=None, lang="ja",
                                     text="raw", registered=[], tokens=["tok"])
            resp = asyncio.run(_run_notation_interceptor(ctx))
        finally:
            pipe._try_notation_candidates = orig
        assert resp is None
        assert ctx.text == "narrowed"        # follower sees the rewrite
        assert calls == {"text": "raw", "tokens": ["tok"]}

    def test_early_wrapper_converts_complexity_error_to_failed(self):
        """The early entry must catch SpinAlgebraComplexityLimitExceeded and turn
        it into a short-circuiting failed response (not let it propagate)."""
        import asyncio
        from ui.interpret_pipeline import _run_early_interceptor
        from ui.interpret_registry import InterceptorContext
        from notebook.runtime.spin_algebra import ComplexityLimitExceeded
        import ui.interpret_pipeline as pipe

        async def _boom(engine, loop, lang, text, registered):
            raise ComplexityLimitExceeded("too big")

        orig = pipe._try_early_intercepts
        pipe._try_early_intercepts = _boom
        try:
            ctx = InterceptorContext(engine=None, loop=None, lang="en",
                                     text="x", registered=[])
            resp = asyncio.run(_run_early_interceptor(ctx))
        finally:
            pipe._try_early_intercepts = orig
        assert resp is not None and resp["status"] == "failed"
