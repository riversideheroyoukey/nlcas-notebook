"""Tests for L-1: general linear basis change and its inverse.

The inverse of a forward transform ``new_i = Σ_j M_ij old_j`` is the solution of
that linear system for the OLD basis (``old = M^{-1} new``).  L-1 contributes the
NL trigger and the old-basis auto-identification, then delegates to solve_system.

Covers:
- _parse_inverse_transform: explicit-basis and auto-basis forms, no-match
- _identifiers_in_order / _infer_old_basis: basis inference and coefficient exclusion
- pipeline: dispatch to solve_system, explicit vars, inference, prompt fallback
"""
import asyncio
from unittest.mock import MagicMock

from notebook.models.quantity import Quantity
from notebook.models.symbol_context import SymbolContext
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operation_validator import OperationValidator


def _make_expression(eid: str, canonical: str, latex: str = "") -> MagicMock:
    expr = MagicMock()
    expr.display_id = eid
    expr.canonical = canonical
    expr.latex = latex or canonical
    expr.created_by_step = "step_1"
    expr.undefined_symbols = []
    return expr


def _make_step(out_ids):
    step = MagicMock()
    step.output_refs = out_ids
    step.step_id = "step_out"
    step.visible_action = "solve"
    return step


def _make_engine(expressions: dict) -> MagicMock:
    engine = MagicMock()
    engine.active_branch_id = "branch_1"
    engine.steps = {}
    engine.branches = {}
    engine.expressions = expressions
    engine.symbol_context = SymbolContext()
    engine.quantity_registry = {}
    engine.lhs_alias_map = {}
    engine.pending_op = None
    engine.lang = "ja"
    engine._active_expr_ids.return_value = set(expressions.keys())
    engine._handler = MagicMock()
    engine._handler.operands_provided.return_value = None
    engine._handler.latex_to_canonical.side_effect = lambda x: x
    engine._handler.normalize_display_latex.side_effect = lambda x: x
    return engine


def _make_supervisor() -> MagicMock:
    sup = MagicMock(spec=LLMSupervisor)
    sup._v = OperationValidator()
    sup.needs_expression_injection.return_value = False
    return sup


def _run(engine, supervisor, text: str) -> dict:
    from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline
    req = InterpretRequest(tokens=[InputToken(type="text", value=text)])
    return asyncio.run(run_interpret_pipeline(req, engine, supervisor))


# ---------------------------------------------------------------------------
# Unit: parsing
# ---------------------------------------------------------------------------

class TestParseInverseTransform:
    def test_auto_basis_form(self):
        from ui.interpret_pipeline import _parse_inverse_transform
        r = _parse_inverse_transform("E1, E2 の逆変換を作って")
        # 消費者が読む欄だけを見る（`==` で dict 全体を固定すると、欄が
        # 1つ増えただけで落ちる——2026-09-07 に `named` を足して実際に落ちた）。
        assert r["eids"] == ["E1", "E2"] and r["vars"] == []
        assert r["named"] is True          # 日本語は `逆変換` を必ず名指す

    def test_explicit_basis_form(self):
        from ui.interpret_pipeline import _parse_inverse_transform
        r = _parse_inverse_transform("E1とE2とE3 を q_1, q_2, q_3 について逆変換して")
        assert r["eids"] == ["E1", "E2", "E3"]
        assert r["vars"] == ["q_1", "q_2", "q_3"]

    def test_no_match(self):
        from ui.interpret_pipeline import _parse_inverse_transform
        assert _parse_inverse_transform("E1 の右辺を計算して") is None
        assert _parse_inverse_transform("E1 を m について解いて") is None

    # inv_transform_vars/inv_transform externalized to i18n/nl_patterns.json
    # (所見G/(b) migration); EN needs lang='en' explicitly now.
    def test_auto_basis_form_en(self):
        from ui.interpret_pipeline import _parse_inverse_transform
        r = _parse_inverse_transform("Invert E1, E2", lang="en")
        assert r["eids"] == ["E1", "E2"] and r["vars"] == []
        # 英語の裸の `invert` は逆変換を**名指していない**（`inverse_matrix` の
        # 別名でもある）。E-id が2つ在るのでここは逆変換で正しいが、1つなら
        # 呼び出し側が降りる——tests/test_matrix_word_routing.py 参照。
        assert r["named"] is False

    def test_explicit_basis_form_en(self):
        from ui.interpret_pipeline import _parse_inverse_transform
        r = _parse_inverse_transform(
            "Build the inverse transform of E1, E2, E3 for q_1, q_2, q_3", lang="en")
        assert r["eids"] == ["E1", "E2", "E3"]
        assert r["vars"] == ["q_1", "q_2", "q_3"]

    def test_no_match_en(self):
        from ui.interpret_pipeline import _parse_inverse_transform
        assert _parse_inverse_transform("solve E1 for m", lang="en") is None


class TestMatrixTransform:
    def test_parse_bracket_matrix(self):
        from ui.interpret_pipeline import _parse_bracket_matrix
        assert _parse_bracket_matrix("[[u, -v], [v, u]]") == [["u", "-v"], ["v", "u"]]

    def test_parse_matrix_transform(self):
        from ui.interpret_pipeline import _parse_matrix_transform
        r = _parse_matrix_transform(
            "M = [[u,-v],[v,u]]、旧基底 c_k, c_mk、新基底 alpha_k, alpha_mk の逆変換を求めて")
        assert r == {"M": [["u", "-v"], ["v", "u"]],
                     "old": ["c_k", "c_mk"], "new": ["alpha_k", "alpha_mk"]}

    def test_parse_matrix_transform_dimension_mismatch(self):
        from ui.interpret_pipeline import _parse_matrix_transform
        # 2x2 M but 3 new-basis names → rejected
        assert _parse_matrix_transform(
            "M = [[u,-v],[v,u]]、旧基底 a, b、新基底 X, Y, Z の逆変換") is None

    def test_no_match_without_matrix(self):
        from ui.interpret_pipeline import _parse_matrix_transform
        assert _parse_matrix_transform("E1, E2 の逆変換を作って") is None

    # Regression: the "逆"/"変換" gate was a plain Japanese-only substring
    # check, so English text never reached _OLD_BASIS_RE/_NEW_BASIS_RE below
    # even though those already had "old basis"/"new basis" alternatives.
    def test_parse_matrix_transform_en(self):
        from ui.interpret_pipeline import _parse_matrix_transform
        r = _parse_matrix_transform(
            "Build the inverse transform: M = [[u,-v],[v,u]], "
            "old basis: c_k, c_mk, new basis: alpha_k, alpha_mk")
        assert r == {"M": [["u", "-v"], ["v", "u"]],
                     "old": ["c_k", "c_mk"], "new": ["alpha_k", "alpha_mk"]}

    def test_no_match_without_matrix_en(self):
        from ui.interpret_pipeline import _parse_matrix_transform
        assert _parse_matrix_transform("Build the inverse transform of E1, E2") is None


class TestOperatorBasisInference:
    def test_basis_atoms_keep_operators_whole(self):
        from ui.interpret_pipeline import _basis_atoms_in_order
        # operator forms must not collapse to the base name "c"
        assert _basis_atoms_in_order("u c(1) - v c'(2)") == ["u", "c(1)", "v", "c'(2)"]

    def test_atom_base(self):
        from ui.interpret_pipeline import _atom_base
        assert _atom_base("c'(2)") == "c"
        assert _atom_base("c(1)") == "c"
        assert _atom_base("q_1") == "q_1"

    def test_infer_returns_none_for_underdetermined_operators(self):
        # 2 equations but 4 distinct operator atoms → cannot infer (caller prompts)
        from ui.interpret_pipeline import _infer_old_basis, _forward_has_operators
        from notebook.models.quantity import Quantity
        e1 = _make_expression("E1", "u*c(1) - v*c'(2)", "alpha_k = u c(1) - v c'(2)")
        e2 = _make_expression("E2", "u*c(2) + v*c'(1)", "alpha_mk = u c(2) + v c'(1)")
        engine = _make_engine({"E1": e1, "E2": e2})
        for nm in ("u", "v"):
            engine.symbol_context.register(nm, Quantity(name=nm, kind="scalar"))
        engine.symbol_context.register(
            "c", Quantity(name="c", kind="operator"), statistics_kind="fermi")
        assert _infer_old_basis(engine, ["E1", "E2"]) is None
        assert _forward_has_operators(engine, ["E1", "E2"]) is True

    def test_operator_surrogates(self):
        from notebook.runtime.julia_backend_handler import JuliaBackendHandler
        m = JuliaBackendHandler._operator_surrogates("u*c(1) - v*c'(2)", "c(1), c'(2)")
        assert m == {"c(1)": "nbA_c_1", "c'(2)": "nbA_cD_2"}


class TestParseCanonicalConstraints:
    def test_basic_form(self):
        from ui.interpret_pipeline import _parse_canonical_constraints
        assert _parse_canonical_constraints("E1 の正準拘束を求めて") == ["E1"]

    def test_bogoliubov_phrasing(self):
        from ui.interpret_pipeline import _parse_canonical_constraints
        assert _parse_canonical_constraints(
            "E1 と E2 の Bogoliubov 拘束を導出して") == ["E1", "E2"]

    def test_no_match(self):
        from ui.interpret_pipeline import _parse_canonical_constraints
        assert _parse_canonical_constraints("E1 の逆変換を作って") is None
        assert _parse_canonical_constraints("E1 を展開して") is None

    def test_verify_phrasing(self):
        from ui.interpret_pipeline import (_parse_canonical_constraints,
                                           _is_verify_request)
        assert _parse_canonical_constraints("E1 は正準か？") == ["E1"]
        assert _parse_canonical_constraints("E1 と E2 は正準変換ですか") == ["E1", "E2"]
        assert _is_verify_request("E1 は正準か？") is True
        assert _is_verify_request("E1 の正準拘束を求めて") is False

    # These constraint interceptors are rule-based-only (no schema operation
    # names them, they compute the (anti)commutators directly), so a miss has
    # no LLM fallback in any language — English needs its own patterns.  The
    # grammar is externalized to nl_patterns.json and looked up by session
    # language (docs/28 §3.1), so the EN forms are passed lang="en".
    def test_basic_form_en(self):
        from ui.interpret_pipeline import _parse_canonical_constraints
        assert _parse_canonical_constraints(
            "derive the canonical constraints of E1", "en") == ["E1"]

    def test_bogoliubov_phrasing_en(self):
        from ui.interpret_pipeline import _parse_canonical_constraints
        assert _parse_canonical_constraints(
            "derive the Bogoliubov constraints of E1, E2", "en") == ["E1", "E2"]

    def test_verify_phrasing_en(self):
        from ui.interpret_pipeline import (_parse_canonical_constraints,
                                           _is_verify_request)
        assert _parse_canonical_constraints("is E1 canonical?", "en") == ["E1"]
        assert _parse_canonical_constraints(
            "are E1, E2 canonical transformations?", "en") == ["E1", "E2"]
        assert _parse_canonical_constraints(
            "verify that E1 is canonical", "en") == ["E1"]
        assert _is_verify_request("is E1 canonical?", "en") is True
        assert _is_verify_request(
            "derive the canonical constraints of E1", "en") is False

    def test_operator_statistics_reads_context(self):
        from ui.interpret_pipeline import _operator_statistics
        from notebook.models.quantity import Quantity
        e = _make_expression("E1", "u*c(1) - v*c'(1)")
        engine = _make_engine({"E1": e})
        engine.symbol_context.register(
            "c", Quantity(name="c", kind="operator"), statistics_kind="fermi")
        assert _operator_statistics(engine, "u*c(1) - v*c'(1)") == "fermi"
        # no fermion operator in context → defaults to bose (commutator)
        assert _operator_statistics(engine, "u*x + v*y") == "bose"


class TestMetricConstraints:
    def test_parse_inner_product_unicode_and_latex(self):
        from ui.interpret_pipeline import _parse_inner_product
        assert _parse_inner_product("⟨e_1, e_2⟩ = s") == ("e_1", "e_2", "s")
        assert _parse_inner_product(r"\langle a, b \rangle = 1/2") == ("a", "b", "1/2")
        assert _parse_inner_product("f = a x") is None

    def test_parse_metric_constraints(self):
        from ui.interpret_pipeline import _parse_metric_constraints
        assert _parse_metric_constraints("E1, E2 の計量保存条件を求めて") == ["E1", "E2"]
        assert _parse_metric_constraints("E1 と E2 の直交条件を出して") == ["E1", "E2"]
        assert _parse_metric_constraints("E1 の逆変換を作って") is None

    def test_parse_metric_constraints_en(self):
        from ui.interpret_pipeline import _parse_metric_constraints
        assert _parse_metric_constraints(
            "derive the metric-preserving constraints of E1, E2", "en") == ["E1", "E2"]
        assert _parse_metric_constraints(
            "find the orthogonality conditions for E1", "en") == ["E1"]
        assert _parse_metric_constraints("build the inverse transform of E1", "en") is None

    def test_gram_rows_defaults(self):
        from ui.interpret_pipeline import _gram_rows
        gram = {("e_1", "e_2"): "s"}                 # off-diagonal only
        # diagonal defaults to 1; backend-neutral rows joined by ';'
        assert _gram_rows(gram, ["e_1", "e_2"]) == "1, s; s, 1"

    def test_parse_maxima_matrix(self):
        from ui.interpret_pipeline import _parse_maxima_matrix
        m = _parse_maxima_matrix("matrix([a^2-1,(a-1)*s],[(a-1)*s,0])")
        assert m == [["a^2-1", "(a-1)*s"], ["(a-1)*s", "0"]]

    def test_maxima_build_command(self):
        from notebook.runtime.maxima_backend_handler import MaximaBackendHandler
        h = MaximaBackendHandler.__new__(MaximaBackendHandler)
        h._context = None
        res = h.build_command(
            "derive_metric_constraints",
            {"forward": "E1, E2", "basis": "e_1, e_2", "gram": "1, s; s, 1"},
            {"E1": "f1 = a*e_1", "E2": "f2 = e_2"},
        )
        assert res.ok
        assert res.command == (
            "block([_M, _G], _M: matrix([coeff(a*e_1, e_1), coeff(a*e_1, e_2)], "
            "[coeff(e_2, e_1), coeff(e_2, e_2)]), _G: matrix([1, s], [s, 1]), "
            "trigsimp(ratsimp(_M . _G . transpose(_M) - _G)))")

    def test_julia_build_command(self):
        from notebook.runtime.julia_backend_handler import JuliaBackendHandler
        from unittest.mock import MagicMock
        h = JuliaBackendHandler.__new__(JuliaBackendHandler)
        h._context = None
        h._builder = MagicMock()
        res = h._build_metric_constraints(
            {"forward": "E1, E2", "basis": "e_1, e_2", "gram": "1, s; s, 1"},
            {"E1": "f1 = a*e_1", "E2": "f2 = e_2"},
        )
        assert res.ok
        assert "Symbolics.coeff(a*e_1, e_1)" in res.command
        assert "_M*_G*permutedims(_M) - _G" in res.command
        assert "_G = [1 s; s 1]" in res.command
        assert 'matrix(' in res.command          # emits a Maxima-style matrix string


class TestTransformSubstitution:
    def test_parse_both_orders(self):
        from ui.interpret_pipeline import _parse_transform_subst
        assert _parse_transform_subst("E2, E3 を E1 に代入して") == (["E2", "E3"], "E1")
        assert _parse_transform_subst("E1 に E2, E3 を代入して") == (["E2", "E3"], "E1")
        assert _parse_transform_subst("E2 を E1 に適用して") == (["E2"], "E1")
        assert _parse_transform_subst("E1 を展開して") is None

    def test_parse_both_orders_en(self):
        from ui.interpret_pipeline import _parse_transform_subst
        assert _parse_transform_subst("substitute E2, E3 into E1", "en") == (["E2", "E3"], "E1")
        assert _parse_transform_subst("apply E2 to E1", "en") == (["E2"], "E1")
        assert _parse_transform_subst("expand E1", "en") is None

    def test_op_name_index(self):
        from ui.interpret_pipeline import _op_name_index
        assert _op_name_index("c_k") == ("c", "k")
        assert _op_name_index("c(1)") == ("c", "1")
        assert _op_name_index("c_mk") == ("c", "mk")
        assert _op_name_index("L") == ("L", None)

    def test_is_operator_symbol(self):
        from ui.interpret_pipeline import _is_operator_symbol
        from notebook.models.quantity import Quantity
        engine = _make_engine({})
        engine.symbol_context.register(
            "c", Quantity(name="c", kind="operator"), statistics_kind="fermi")
        assert _is_operator_symbol(engine, "c_1") is True
        assert _is_operator_symbol(engine, "q_1") is False

    def test_subst_lhs_rhs_from_canonical(self):
        from ui.interpret_pipeline import _subst_lhs_rhs
        e = _make_expression("E2", "q_1 = Q_1 + Q_2")
        engine = _make_engine({"E2": e})
        engine.lhs_alias_map = {}
        assert _subst_lhs_rhs(engine, "E2") == ("q_1", "Q_1 + Q_2")


class TestParseApplyConstraints:
    def test_explicit_relation(self):
        from ui.interpret_pipeline import _parse_apply_constraints
        r = _parse_apply_constraints("E2 を E1 の拘束で簡約して")
        assert r == {"target": "E2", "relations": ["E1"]}

    def test_auto_uses_session_constraints(self):
        from ui.interpret_pipeline import _parse_apply_constraints
        r = _parse_apply_constraints("正準拘束のもとで E3 を簡約して")
        assert r == {"target": "E3", "relations": None}

    def test_no_match_without_keyword(self):
        from ui.interpret_pipeline import _parse_apply_constraints
        assert _parse_apply_constraints("E3 を簡約して") is None
        assert _parse_apply_constraints("E1 の正準拘束を求めて") is None

    def test_explicit_relation_en(self):
        from ui.interpret_pipeline import _parse_apply_constraints
        assert _parse_apply_constraints(
            "reduce E2 using the constraints of E1", "en") == {"target": "E2", "relations": ["E1"]}

    def test_auto_uses_session_constraints_en(self):
        from ui.interpret_pipeline import _parse_apply_constraints
        assert _parse_apply_constraints(
            "simplify E3 under the canonical constraints", "en") == {"target": "E3", "relations": None}

    def test_no_match_without_keyword_en(self):
        from ui.interpret_pipeline import _parse_apply_constraints
        assert _parse_apply_constraints("simplify E3", "en") is None


class TestApplyConstraintsBuild:
    def _maxima(self):
        from notebook.runtime.maxima_backend_handler import MaximaBackendHandler
        h = MaximaBackendHandler.__new__(MaximaBackendHandler)
        h._context = None
        return h

    def _julia(self):
        from notebook.runtime.julia_backend_handler import JuliaBackendHandler
        from unittest.mock import MagicMock
        h = JuliaBackendHandler.__new__(JuliaBackendHandler)
        h._context = None
        h._builder = MagicMock()
        return h

    def test_maxima_reduces_rhs_in_block(self):
        h = self._maxima()
        lookup = {"E2": "F = (u*a - v*b)/(u^2+v^2)", "E1": "u^2+v^2 = 1"}
        res = h.build_command("apply_canonical_constraints",
                              {"expr": "E2", "relations": "E1"}, lookup)
        assert res.ok
        # The SHAPE is what this pins — the RHS is reduced inside a block that
        # shadows the P2-bound `F` — not the exact spelling of the operands.
        # They are lowered now (BUILDER-OPERAND-NOT-LOWERED, 2026-08-15), so a
        # division arrives as a negative power; measured, Maxima answers
        # `F = a*u-b*v` for both spellings, so asserting the old text would pin
        # the lowering rather than the behaviour.
        cmd = res.command.replace(" ", "")
        assert cmd.startswith("block([F],F=ratsubst(1,u^2+v^2,"), cmd
        assert cmd.endswith("))"), cmd

    def test_julia_substitutes_relation(self):
        h = self._julia()
        lookup = {"E2": "F = (u*a - v*b)/(u^2+v^2)", "E1": "u^2+v^2 = 1"}
        res = h._build_apply_constraints({"expr": "E2", "relations": "E1"}, lookup)
        assert res.ok
        assert "Symbolics.substitute" in res.command
        assert "u^2+v^2 => 1" in res.command
        assert '"F = "' in res.command          # the LHS is reattached

    def test_missing_relations_errors(self):
        h = self._maxima()
        res = h.build_command("apply_canonical_constraints",
                              {"expr": "E2", "relations": ""},
                              {"E2": "F = x"})
        assert not res.ok


class TestIdentifiersAndBasis:
    def test_identifiers_in_order_dedup(self):
        from ui.interpret_pipeline import _identifiers_in_order
        assert _identifiers_in_order("a11*q1 + a12*q2 + a11*q2") == \
            ["a11", "q1", "a12", "q2"]

    def test_infer_basis_with_scalar_coefficients(self):
        # u, v declared scalar → coefficients; the basis is c_k, c_mk
        from ui.interpret_pipeline import _infer_old_basis
        e1 = _make_expression("E1", "u*c_k + v*c_mk", "alpha_k = u c_k + v c_mk")
        e2 = _make_expression("E2", "- v*c_k + u*c_mk", "alpha_mk = - v c_k + u c_mk")
        engine = _make_engine({"E1": e1, "E2": e2})
        for nm in ("u", "v"):
            engine.symbol_context.register(nm, Quantity(name=nm, kind="scalar"))
        assert _infer_old_basis(engine, ["E1", "E2"]) == ["c_k", "c_mk"]

    def test_infer_basis_excludes_new_vars(self):
        # new basis (LHS) names must never appear in the inferred old basis
        from ui.interpret_pipeline import _infer_old_basis
        e1 = _make_expression("E1", "x + y", "X = x + y")
        e2 = _make_expression("E2", "x - y", "Y = x - y")
        engine = _make_engine({"E1": e1, "E2": e2})
        assert _infer_old_basis(engine, ["E1", "E2"]) == ["x", "y"]

    def test_infer_returns_none_when_ambiguous(self):
        # undeclared coefficients → too many atoms → cannot infer (caller prompts)
        from ui.interpret_pipeline import _infer_old_basis
        e1 = _make_expression("E1", "a11*q1 + a12*q2", "Q1 = a11 q1 + a12 q2")
        e2 = _make_expression("E2", "a21*q1 + a22*q2", "Q2 = a21 q1 + a22 q2")
        engine = _make_engine({"E1": e1, "E2": e2})
        assert _infer_old_basis(engine, ["E1", "E2"]) is None


# ---------------------------------------------------------------------------
# Pipeline dispatch
# ---------------------------------------------------------------------------

class TestInverseTransformPipeline:
    def test_explicit_vars_dispatch(self):
        e1 = _make_expression("E1", "u*c_k + v*c_mk", "alpha_k = u c_k + v c_mk")
        e2 = _make_expression("E2", "- v*c_k + u*c_mk", "alpha_mk = - v c_k + u c_mk")
        e_out = _make_expression("E3", "[[c_k = ..., c_mk = ...]]")
        engine = _make_engine({"E1": e1, "E2": e2, "E3": e_out})
        out_step = _make_step(["E3"])
        engine.run_operation.return_value = (out_step, "[[c_k=..., c_mk=...]]")

        sup = _make_supervisor()
        resp = _run(engine, sup, "E1とE2 を c_k, c_mk について逆変換して")

        assert resp["status"] == "completed"
        assert resp["operation"] == "build_inverse_transform"
        call = engine.run_operation.call_args
        assert call[0][0] == "build_inverse_transform"
        assert "c_k" in call[0][1]["vars"] and "c_mk" in call[0][1]["vars"]
        assert call[0][1]["equations"] == "E1, E2"
        # the new-basis (LHS) symbols are passed so the backend can unbind them
        assert "alpha_k" in call[0][1]["new_basis"]
        assert "alpha_mk" in call[0][1]["new_basis"]
        sup.interpret.assert_not_called()

    def test_auto_basis_infers_and_dispatches(self):
        e1 = _make_expression("E1", "u*c_k + v*c_mk", "alpha_k = u c_k + v c_mk")
        e2 = _make_expression("E2", "- v*c_k + u*c_mk", "alpha_mk = - v c_k + u c_mk")
        e_out = _make_expression("E3", "[[c_k = ..., c_mk = ...]]")
        engine = _make_engine({"E1": e1, "E2": e2, "E3": e_out})
        for nm in ("u", "v"):
            engine.symbol_context.register(nm, Quantity(name=nm, kind="scalar"))
        out_step = _make_step(["E3"])
        engine.run_operation.return_value = (out_step, "[[c_k=..., c_mk=...]]")

        sup = _make_supervisor()
        resp = _run(engine, sup, "E1, E2 の逆変換を作って")

        assert resp["status"] == "completed"
        call = engine.run_operation.call_args
        assert call[0][0] == "build_inverse_transform"
        assert call[0][1]["vars"] == "c_k, c_mk"

    def test_ambiguous_basis_prompts(self):
        e1 = _make_expression("E1", "a11*q1 + a12*q2", "Q1 = a11 q1 + a12 q2")
        e2 = _make_expression("E2", "a21*q1 + a22*q2", "Q2 = a21 q1 + a22 q2")
        engine = _make_engine({"E1": e1, "E2": e2})

        sup = _make_supervisor()
        resp = _run(engine, sup, "E1, E2 の逆変換を作って")

        assert resp["status"] == "needs_confirmation"
        assert engine.pending_op is not None
        assert engine.pending_op["operation"] == "build_inverse_transform"
        engine.run_operation.assert_not_called()

    def test_single_equation_is_rejected(self):
        e1 = _make_expression("E1", "u*c_k", "alpha = u c_k")
        engine = _make_engine({"E1": e1})

        sup = _make_supervisor()
        resp = _run(engine, sup, "E1 の逆変換を作って")

        assert resp["status"] == "failed"
        engine.run_operation.assert_not_called()


class TestMaximaBuildCommand:
    """build_inverse_transform command generation (Maxima)."""

    def _handler(self):
        from notebook.runtime.maxima_backend_handler import MaximaBackendHandler
        h = MaximaBackendHandler.__new__(MaximaBackendHandler)
        h._context = None
        return h

    def test_localizes_new_basis_in_block(self):
        # the new-basis symbols are bound to their RHS by H-1; the command must wrap
        # solve() in block([...]) so they are unbound and the system is non-degenerate
        h = self._handler()
        expr_lookup = {
            "E1": "alpha_k = u*c_k + v*c_mk",
            "E2": "alpha_mk = - v*c_k + u*c_mk",
        }
        res = h.build_command(
            "build_inverse_transform",
            {"equations": "E1, E2", "vars": "c_k, c_mk",
             "new_basis": "alpha_k, alpha_mk"},
            expr_lookup,
        )
        assert res.ok
        assert res.command == (
            "block([alpha_k, alpha_mk], "
            "solve([alpha_k = u*c_k + v*c_mk, alpha_mk = - v*c_k + u*c_mk], "
            "[c_k, c_mk]))"
        )

    def test_without_new_basis_is_plain_solve(self):
        h = self._handler()
        res = h.build_command(
            "build_inverse_transform",
            {"equations": "E1", "vars": "x", "new_basis": ""},
            {"E1": "y = a*x"},
        )
        assert res.ok
        assert res.command == "solve([y = a*x], [x])"

    def test_missing_vars_errors(self):
        h = self._handler()
        res = h.build_command(
            "build_inverse_transform",
            {"equations": "E1", "vars": ""},
            {"E1": "y = a*x"},
        )
        assert not res.ok


class TestJuliaBuildCommand:
    """build_inverse_transform command generation (Julia / Symbolics.jl)."""

    def _handler(self):
        from notebook.runtime.julia_backend_handler import JuliaBackendHandler
        from unittest.mock import MagicMock
        h = JuliaBackendHandler.__new__(JuliaBackendHandler)
        h._context = None
        h._builder = MagicMock()
        return h

    def test_declares_basis_and_solves(self):
        h = self._handler()
        expr_lookup = {
            "E1": "alpha_k = u*c_k + v*c_mk",
            "E2": "alpha_mk = - v*c_k + u*c_mk",
        }
        res = h._build_inverse_transform(
            {"equations": "E1, E2", "vars": "c_k, c_mk",
             "new_basis": "alpha_k, alpha_mk"},
            expr_lookup,
        )
        assert res.ok
        # both the new basis and the old basis are declared as free Symbolics vars
        assert "@variables alpha_k alpha_mk c_k c_mk" in res.command
        assert "Symbolics.solve_for([alpha_k ~ u*c_k + v*c_mk, " \
               "alpha_mk ~ - v*c_k + u*c_mk], [c_k, c_mk])" in res.command
        assert "Symbolics.simplify." in res.command
        # the old basis labels are joined back onto the solution values
        assert '["c_k", "c_mk"]' in res.command

    def test_missing_vars_errors(self):
        h = self._handler()
        res = h._build_inverse_transform({"equations": "E1", "vars": ""},
                                         {"E1": "y = a*x"})
        assert not res.ok
