"""Tests for C-Ext Phase 2: extract_order, truncate_series.

Note: series expansion (Taylor) was already implemented as taylor_expand
and is tested via the existing test suite. extract_order and truncate_series
are genuinely new operations added in C-Ext Phase 2.
"""
import pytest
from notebook.engine import NotebookEngine
from notebook.models.session import SessionMode


@pytest.fixture
def julia_engine():
    with NotebookEngine(title="test", mode=SessionMode.second_quantization) as e:
        e.start()
        for name in ("H0", "V", "W", "X", "ep"):
            e.run_operation("define_scalar_parameter", {"name": name})
        yield e


@pytest.fixture
def maxima_engine():
    with NotebookEngine(title="test", mode=SessionMode.standard_symbolic,
                        coordinate_system="cartesian") as e:
        e.start()
        yield e


class TestExtractOrderJulia:
    def test_extract_constant(self, julia_engine):
        e = julia_engine
        expr = e.register_expression("H0 + ep*V + ep^2*W")
        _, v = e.run_operation("extract_order", {"expr": expr.display_id, "var": "ep", "order": 0})
        assert "H0" in str(v)

    def test_extract_first_order(self, julia_engine):
        e = julia_engine
        expr = e.register_expression("H0 + ep*V + ep^2*W")
        _, v = e.run_operation("extract_order", {"expr": expr.display_id, "var": "ep", "order": 1})
        assert "V" in str(v)

    def test_extract_second_order(self, julia_engine):
        e = julia_engine
        expr = e.register_expression("H0 + ep*V + ep^2*W")
        _, v = e.run_operation("extract_order", {"expr": expr.display_id, "var": "ep", "order": 2})
        assert "W" in str(v)

    def test_extract_absent_order_zero(self, julia_engine):
        e = julia_engine
        expr = e.register_expression("ep*V + ep^2*W")
        _, v = e.run_operation("extract_order", {"expr": expr.display_id, "var": "ep", "order": 0})
        assert str(v).strip() in ("0", "0//1")

    def test_extract_numeric_polynomial(self, julia_engine):
        e = julia_engine
        expr = e.register_expression("1 + 2*ep + 3*ep^2 + 4*ep^3")
        _, v = e.run_operation("extract_order", {"expr": expr.display_id, "var": "ep", "order": 2})
        assert "3" in str(v)


class TestTruncateSeriesJulia:
    def test_truncate_drops_higher_order(self, julia_engine):
        e = julia_engine
        expr = e.register_expression("H0 + ep*V + ep^2*W + ep^3*X")
        _, v = e.run_operation("truncate_series", {"expr": expr.display_id, "var": "ep", "order": 1})
        assert "H0" in str(v) and "W" not in str(v) and "X" not in str(v)

    def test_truncate_keeps_all_in_range(self, julia_engine):
        e = julia_engine
        expr = e.register_expression("H0 + ep*V + ep^2*W")
        _, v = e.run_operation("truncate_series", {"expr": expr.display_id, "var": "ep", "order": 2})
        assert "H0" in str(v) and "V" in str(v) and "W" in str(v)


class TestExtractOrderMaxima:
    def test_extract_numeric_coeff(self, maxima_engine):
        e = maxima_engine
        expr = e.register_expression("1 + 2*x + 3*x^2 + 4*x^3")
        _, v = e.run_operation("extract_order", {"expr": expr.display_id, "var": "x", "order": 2})
        assert str(v).strip() == "3"

    def test_extract_zero_order(self, maxima_engine):
        e = maxima_engine
        expr = e.register_expression("1 + x + x^2")
        _, v = e.run_operation("extract_order", {"expr": expr.display_id, "var": "x", "order": 0})
        assert str(v).strip() == "1"


class TestTruncateSeriesMaxima:
    def test_truncate_drops_higher(self, maxima_engine):
        e = maxima_engine
        expr = e.register_expression("1 + x + x^2 + x^3 + x^4")
        _, v = e.run_operation("truncate_series", {"expr": expr.display_id, "var": "x", "order": 2})
        assert "x^3" not in str(v) and "x^4" not in str(v)
        assert "x^2" in str(v)

    def test_truncate_symbolic(self, maxima_engine):
        e = maxima_engine
        expr = e.register_expression("H_0 + lambda*V + lambda^2*W + lambda^3*X")
        _, v = e.run_operation("truncate_series", {"expr": expr.display_id, "var": "lambda", "order": 2})
        assert "lambda^3" not in str(v)
        assert "H_0" in str(v) and "V" in str(v) and "W" in str(v)


class TestExtractOrderOperator:
    """extract_order on QuExpr (operator perturbation series)."""

    @pytest.fixture(autouse=True)
    def _engine(self):
        with NotebookEngine(title="test", mode=SessionMode.second_quantization) as e:
            e.start()
            e.run_operation("define_fermion_operator", {"name": "c"})
            for name in ("xi", "ep", "g"):
                e.run_operation("define_scalar_parameter", {"name": name})
            self.engine = e
            yield

    def _H(self):
        """H = xi*c†(1)c(1) + ep*(c†(1)c(2) + c†(2)c(1))"""
        step, _ = self.engine.run_operation("expand", {
            "expr": "xi*c'(1)*c(1) + ep*(c'(1)*c(2) + c'(2)*c(1))"
        })
        return step.output_refs[0]

    def test_extract_zeroth_order_is_H0(self):
        eid = self._H()
        _, v = self.engine.run_operation("extract_order",
                                         {"expr": eid, "var": "ep", "order": 0})
        assert "xi" in str(v) and "ep" not in str(v)

    def test_extract_first_order_is_V(self):
        eid = self._H()
        _, v = self.engine.run_operation("extract_order",
                                         {"expr": eid, "var": "ep", "order": 1})
        s = str(v)
        assert "ep" not in s and "c" in s and "xi" not in s

    def test_extract_absent_order_returns_zero(self):
        eid = self._H()
        _, v = self.engine.run_operation("extract_order",
                                         {"expr": eid, "var": "ep", "order": 2})
        assert str(v).strip() == "0"

    def test_extract_second_order_4body(self):
        """H with 4-body term: ep²*g*c†c†cc → extract n=2 gives operator."""
        step, _ = self.engine.run_operation("expand", {
            "expr": "xi*c'(1)*c(1) + ep*c'(1)*c(1) + ep^2*g*c'(1)*c'(2)*c(2)*c(1)"
        })
        eid = step.output_refs[0]
        _, v = self.engine.run_operation("extract_order",
                                         {"expr": eid, "var": "ep", "order": 2})
        assert "g" in str(v) and "c" in str(v)


class TestTruncateSeriesOperator:
    """truncate_series on QuExpr (operator perturbation series)."""

    @pytest.fixture(autouse=True)
    def _engine(self):
        with NotebookEngine(title="test", mode=SessionMode.second_quantization) as e:
            e.start()
            e.run_operation("define_fermion_operator", {"name": "c"})
            for name in ("xi", "ep", "g"):
                e.run_operation("define_scalar_parameter", {"name": name})
            self.engine = e
            yield

    def test_truncate_drops_higher_order_operators(self):
        """H0 + ep*V + ep²*g*W truncated at n=1 gives H0+ep*V, no W."""
        step, _ = self.engine.run_operation("expand", {
            "expr": "xi*c'(1)*c(1) + ep*(c'(1)*c(2)+c'(2)*c(1)) + ep^2*g*c'(1)*c'(2)*c(2)*c(1)"
        })
        eid = step.output_refs[0]
        _, v = self.engine.run_operation("truncate_series",
                                          {"expr": eid, "var": "ep", "order": 1})
        assert "g" not in str(v)
        assert "xi" in str(v) and "ep" in str(v)

    def test_truncate_keeps_full_range(self):
        step, _ = self.engine.run_operation("expand", {
            "expr": "xi*c'(1)*c(1) + ep*(c'(1)*c(2)+c'(2)*c(1))"
        })
        eid = step.output_refs[0]
        _, v = self.engine.run_operation("truncate_series",
                                          {"expr": eid, "var": "ep", "order": 2})
        assert "xi" in str(v) and "ep" in str(v)


class TestOperatorTaylorExpand:
    """operator_taylor_expand: f(H) ≈ Σ a_k H^k using QuantumAlgebra."""

    @pytest.fixture(autouse=True)
    def _engine(self):
        with NotebookEngine(title="test", mode=SessionMode.second_quantization) as e:
            e.start()
            e.run_operation("define_fermion_operator", {"name": "c"})
            for name in ("xi", "beta", "t"):
                e.run_operation("define_scalar_parameter", {"name": name})
            self.engine = e
            yield

    def _H_diag(self):
        """H = xi * c†(1)c(1)  (projector: n² = n)"""
        step, _ = self.engine.run_operation("expand", {"expr": "xi*c'(1)*c(1)"})
        return step.output_refs[0]

    def _H_offdiag(self):
        """H2 = t*(c†(1)c(2) + c†(2)c(1))"""
        step, _ = self.engine.run_operation("expand",
                                             {"expr": "t*(c'(1)*c(2)+c'(2)*c(1))"})
        return step.output_refs[0]

    def test_exp_projector_structure(self):
        """exp(-beta*H) with projector H has structure 1 + coeff*n."""
        eid = self._H_diag()
        _, v = self.engine.run_operation("operator_taylor_expand",
                                          {"func": "exp(-beta*x)", "op": eid,
                                           "var": "x", "order": 3})
        s = str(v)
        assert "beta" in s and "xi" in s
        # Identity term present
        assert s.startswith("1")

    def test_exp_projector_exploits_n_squared(self):
        """H = xi*n with n²=n: all powers collapse to same operator."""
        eid = self._H_diag()
        _, v = self.engine.run_operation("operator_taylor_expand",
                                          {"func": "exp(-beta*x)", "op": eid,
                                           "var": "x", "order": 4})
        # Should be: 1 + (Σ a_k xi^k) c†(1)c(1)
        # Only one operator structure c†(1)c(1), not c†c†cc etc.
        assert "c†(1) c†" not in str(v)

    def test_off_diagonal_hop_generates_density(self):
        """exp(-beta*H2): H2² generates diagonal density terms."""
        eid = self._H_offdiag()
        _, v = self.engine.run_operation("operator_taylor_expand",
                                          {"func": "exp(-beta*x)", "op": eid,
                                           "var": "x", "order": 2})
        s = str(v)
        # 0th: 1, 1st: -beta*t*(hop), 2nd: (beta*t)²/2 * (diagonal)
        assert "beta" in s and "t" in s

    def test_resolvent_expansion(self):
        """1/(1-x) = 1 + x + x² + x³ applied to H."""
        eid = self._H_diag()
        _, v = self.engine.run_operation("operator_taylor_expand",
                                          {"func": "1/(1-x)", "op": eid,
                                           "var": "x", "order": 3})
        s = str(v)
        assert "xi" in s and "xi^2" in s and "xi^3" in s

    def test_cos_operator(self):
        """cos(H) = 1 - H²/2 + H⁴/24 - ..."""
        eid = self._H_diag()
        _, v = self.engine.run_operation("operator_taylor_expand",
                                          {"func": "cos(x)", "op": eid,
                                           "var": "x", "order": 4})
        s = str(v)
        assert "xi^2" in s or "xi^4" in s


class TestTaylorExpandReadsItsFourFieldsFromLanguage:
    r"""Preparing S-j (two-level perturbation theory), 2026-08-05.

    `taylor_expand` has FOUR required fields — expr / var / point / order — and
    the positional field mapper could not fill them from a sentence:

        E1 を lam について 0 のまわりで 2 次までテイラー展開して
            → point=2, order=0            (swapped)
        taylor expand E1 in lam around 0 to order 2
            → var="expand", point="in"    (words taken by position)

    Both were returned as `needs_confirmation` with those values already filled
    in, so clicking through gives an expansion about the wrong point at the
    wrong order — the BP-6 shape, answered the way F-9 answered it: read each
    field from its own marker.

    Bilingual by rule (the defect was in both languages, and the English one was
    worse).
    """
    import asyncio as _asyncio

    def _session(self, lang):
        import ui.notebook_bridge as nb
        from notebook.runtime.llm_supervisor import LLMSupervisor
        from notebook.runtime.operation_validator import OperationValidator
        nb.start_engine(title="sj", mode="standard", language=lang,
                        domains=["quantum_mechanics"])
        eng = nb.get_engine()
        sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                            domains=["quantum_mechanics"], primary_backend="maxima")
        return nb, eng, sup

    def _run(self, eng, sup, text):
        import asyncio
        from ui.interpret_pipeline import (
            InterpretRequest, InputToken, run_interpret_pipeline)
        tok = InputToken(type="text", value=text)
        return asyncio.get_event_loop().run_until_complete(
            run_interpret_pipeline(InterpretRequest(tokens=[tok]), eng, sup))

    @pytest.mark.parametrize("lang,decl,expand", [
        ("ja", "Ea, Eb, V, lam をスカラーパラメータとして定義して",
         "E1 を lam について 0 のまわりで 2 次までテイラー展開して"),
        ("en", "Define Ea, Eb, V, lam as scalars",
         "taylor expand E1 in lam around 0 to order 2"),
    ])
    def test_the_second_order_correction_comes_out(self, lang, decl, expand):
        """The S-j physics: the exact eigenvalue of a 2x2, expanded in the
        coupling, IS the Rayleigh-Schrodinger series.  Second order is
        V^2/(E_n - E_m) — no `expand_perturbation_series` operation needed."""
        nb, eng, sup = self._session(lang)
        try:
            r = self._run(eng, sup, decl)
            if r["status"] == "needs_confirmation" and r.get("candidates"):
                c = r["candidates"][-1]
                eng.run_operation(c["operation"], c["args"])
            self._run(eng, sup, "(Ea + Eb + sqrt(4*V^2*lam^2 + (Ea-Eb)^2))/2")
            r = self._run(eng, sup, expand)
            assert r["status"] == "completed", r
            assert r["operation"] == "taylor_expand"
            got = eng.expressions[r["expr_id"]].canonical.replace(" ", "")
            assert got in ("(V^2*lam^2)/(Eb-Ea)+Eb", "Eb+(V^2*lam^2)/(Eb-Ea)"), got
        finally:
            nb.stop_engine()

    @pytest.mark.parametrize("lang,text,expected", [
        ("ja", "E1 を lam について 0 のまわりで 2 次までテイラー展開して",
         {"expr": "E1", "var": "lam", "point": "0", "order": "2"}),
        ("ja", "E1 を x で 3 次までテイラー展開して",
         {"expr": "E1", "var": "x", "point": None, "order": "3"}),
        ("ja", "E1 を x について 4 次までマクローリン展開する",
         {"expr": "E1", "var": "x", "point": None, "order": "4"}),
        ("en", "taylor expand E1 in lam around 0 to order 2",
         {"expr": "E1", "var": "lam", "point": "0", "order": "2"}),
        ("en", "Taylor E1 in x to 3rd order",
         {"expr": "E1", "var": "x", "point": None, "order": "3"}),
        ("en", "maclaurin expansion of E1 in x to order 4",
         {"expr": "E1", "var": "x", "point": None, "order": "4"}),
        ("en", "taylor series of E2 with respect to eps at 0 to order 3",
         {"expr": "E2", "var": "eps", "point": "0", "order": "3"}),
    ])
    def test_each_field_is_read_from_its_own_marker(self, lang, text, expected):
        """Pins the fields directly: word order and an omitted centre must not
        move a value into another field.

        Read through `pick_group`, not `m.groupdict()`: `re` forbids reusing a
        group name across an alternation, so a new word order has to spell its
        groups `expr2`/`var2`/… and a whole-dict comparison then fails on the
        BRANCH NAMES rather than on any value.  Measured 2026-09-05 when the
        value-first word order (`expand E1 to third order in x`) was added.
        """
        from notebook.runtime.nl_patterns import nlp
        from ui.interpret_common import number_word, pick_group
        m = nlp("taylor_expand_nl", lang).match(text)
        assert m is not None, text

        def _read(k):
            """**消費者と同じ読み方で比べる。** `order` は序数トークンのまま
            捕まる（`3rd`）ので、`number_word` を通して数にするのが読みである
            ——`_try_taylor_expand` がそうしている。生のトークンを固定すると、
            綴りを1つ増やすたびにテストが落ちる（実測 2026-09-06、序数の綴りを
            共有 fragment に寄せたとき `3` が `3rd` になって落ちた）。"""
            v = pick_group(m, k)
            return str(number_word(v)) if k == "order" and v is not None else v

        assert {k: _read(k) for k in expected} == expected

    def test_a_plain_expand_is_still_a_plain_expand(self):
        """`E1 を展開して` must keep meaning `expand` — the new pattern needs a
        stated variable and order, so it cannot swallow it."""
        from notebook.runtime.nl_patterns import nlp
        assert nlp("taylor_expand_nl", "ja").match("E1 を展開して") is None
        assert nlp("taylor_expand_nl", "en").match("expand E1") is None
