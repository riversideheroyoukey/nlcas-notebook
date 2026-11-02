"""FREE-PARAM: free parameters (arbitrary constants) of a general solution.

Instance 1 — indefinite integration carries the constant of integration C; the
result is the general antiderivative F(x)+C, specialized by value substitution or
by a condition F(x0)=y0.
"""
import asyncio

import ui.notebook_bridge as nb
from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operation_validator import OperationValidator


def _sup():
    return LLMSupervisor(validator=OperationValidator(), llm_client=None,
                         domains=[], primary_backend="maxima")


def _run(eng, sup, value, ttype="text"):
    tok = (InputToken(type="math", value=value, latex=value, canonical="")
           if ttype == "math" else InputToken(type="text", value=value))
    return asyncio.get_event_loop().run_until_complete(
        run_interpret_pipeline(InterpretRequest(tokens=[tok]), eng, sup))


def _eid(r):
    return r.get("expr_id") or r["registered_expressions"][-1]["expr_id"]


class TestIndefiniteIntegralConstant:
    def test_integration_carries_C(self):
        nb.start_engine(title="fp1", mode="standard", language="ja", domains=[])
        eng = nb.get_engine()
        sup = _sup()
        try:
            _run(eng, sup, "x^2", "math")
            r = _run(eng, sup, "E1 を x で不定積分して")
            e = eng.expressions[_eid(r)]
            assert e.canonical.replace(" ", "") == "x^3/3+C"
            assert e.free_parameters == ["C"]
            assert e.latex == r"\int x^2 \, dx = \frac{x^3}{3} + C"
        finally:
            nb.stop_engine()

    def test_constant_name_avoids_collision(self):
        """If C already appears (here, in the integrand) the constant is named C_1."""
        nb.start_engine(title="fpc", mode="standard", language="ja", domains=[])
        eng = nb.get_engine()
        sup = _sup()
        try:
            _run(eng, sup, "C*x^2", "math")
            r = _run(eng, sup, "E1 を x で不定積分して")
            e = eng.expressions[_eid(r)]
            assert e.free_parameters == ["C_1"]
            assert e.canonical.replace(" ", "") == "(C*x^3)/3+C_1"
        finally:
            nb.stop_engine()

    def test_value_substitution_fixes_C(self):
        nb.start_engine(title="fp2", mode="standard", language="ja", domains=[])
        eng = nb.get_engine()
        sup = _sup()
        try:
            _run(eng, sup, "x^2", "math")
            i = _eid(_run(eng, sup, "E1 を x で不定積分して"))
            r = _run(eng, sup, f"{i} の C に 0 を代入して")
            assert eng.expressions[_eid(r)].canonical.replace(" ", "") == "x^3/3"
        finally:
            nb.stop_engine()

    def test_ode_general_solution_carries_two_constants(self):
        """A 2nd-order linear ODE → general solution with arbitrary constants k₁,k₂."""
        nb.start_engine(title="fpode", mode="standard", language="ja", domains=[])
        eng = nb.get_engine()
        sup = _sup()
        try:
            _run(eng, sup, "q を時間依存関数として定義する")
            e1 = _eid(_run(eng, sup, r"\ddot{q} + w^2 q = 0", "math"))
            r = _run(eng, sup, f"{e1} を一般に解いて")
            assert r["operation"] == "solve_ode_general"
            e = eng.expressions[_eid(r)]
            assert e.free_parameters == ["k_1", "k_2"]
            c = e.canonical.replace(" ", "")
            assert "k_1" in c and "k_2" in c and "sin" in c and "cos" in c
        finally:
            nb.stop_engine()

    def test_ode_general_solution_en(self):
        """English phrasing gets the same solve_ode_general routing as Japanese."""
        for phrase in ("solve {e} in general", "find the general solution of {e}",
                       "solve {e} for q as a differential equation",
                       "solve the differential equation {e}"):
            nb.start_engine(title="fpode_en", mode="standard", language="en", domains=[])
            eng = nb.get_engine()
            sup = _sup()
            try:
                _run(eng, sup, "define q as a function of t")
                e1 = _eid(_run(eng, sup, r"\ddot{q} + w^2 q = 0", "math"))
                r = _run(eng, sup, phrase.format(e=e1))
                assert r["operation"] == "solve_ode_general"
                e = eng.expressions[_eid(r)]
                assert e.free_parameters == ["k_1", "k_2"]
                c = e.canonical.replace(" ", "")
                assert "k_1" in c and "k_2" in c and "sin" in c and "cos" in c
            finally:
                nb.stop_engine()

    def test_substitute_free_params_en(self):
        """"Set k_1=0, k_2=1" → `substitute_free_parameters`, English.

        Moved here 2026-08-15 from tests/test_laplace_definition.py, where it
        reached the operation through `laplace_by_definition`'s IC parameters —
        an operation since deleted as a macro (SCENARIO-MACRO case 7).  The
        English coverage of THIS operation had no other home, so it was rebuilt
        on a free-parameter source that owes nothing to Laplace: a general ODE
        solution's arbitrary constants.
        """
        nb.start_engine(title="fpsubst_en", mode="standard", language="en",
                        domains=[])
        eng = nb.get_engine()
        sup = _sup()
        try:
            _run(eng, sup, "define q as a function of t")
            e1 = _eid(_run(eng, sup, r"\ddot{q} + w^2 q = 0", "math"))
            r1 = _run(eng, sup, f"solve {e1} in general")
            assert eng.expressions[_eid(r1)].free_parameters == ["k_1", "k_2"]
            r2 = _run(eng, sup, "Set k_1=0, k_2=1")
            assert r2["operation"] == "substitute_free_parameters"
            e = eng.expressions[_eid(r2)]
            assert e.free_parameters == []
            assert e.canonical.replace(" ", "") == "q=cos(t*w)"
        finally:
            nb.stop_engine()

    def test_ode_two_condition_specialization(self):
        """q(0)=q0, q̇(0)=v0 fixes k₁,k₂ → particular solution (no free params left)."""
        nb.start_engine(title="fpode2b", mode="standard", language="ja", domains=[])
        eng = nb.get_engine()
        sup = _sup()
        try:
            _run(eng, sup, "q を時間依存関数として定義する")
            e1 = _eid(_run(eng, sup, r"\ddot{q} + w^2 q = 0", "math"))
            g = _eid(_run(eng, sup, f"{e1} を一般に解いて"))
            r = _run(eng, sup, f"{g} を q(0)=q0, q'(0)=v0 で特殊化して")
            assert r["operation"] == "specialize_by_conditions"
            e = eng.expressions[_eid(r)]
            assert e.free_parameters == []
            c = e.canonical.replace(" ", "")
            # k₁ = v0/w (sin term), k₂ = q0 (cos term); no arbitrary constants remain
            assert "k_1" not in c and "k_2" not in c
            assert "q0" in c and "v0" in c
        finally:
            nb.stop_engine()

    def test_ode_solve_trigger_phrase_variants(self):
        """"の微分方程式を解いて" routes to solve_ode_general just like "一般に解いて"
        (added after a user flagged that "E1 を解いて" / "E1 の微分方程式を解いて"
        were NOT recognized, and that the algebraic-solve fallback silently produces
        a meaningless rearrangement instead of an ODE solution)."""
        for phrase in ("の微分方程式を解いて", "の微分方程式の解を求めて"):
            nb.start_engine(title="fpode_phrase", mode="standard", language="ja", domains=[])
            eng = nb.get_engine()
            sup = _sup()
            try:
                _run(eng, sup, "q を時間依存関数として定義する")
                e1 = _eid(_run(eng, sup, r"\ddot{q} + w^2 q = 0", "math"))
                r = _run(eng, sup, f"{e1} {phrase}")
                assert r["operation"] == "solve_ode_general", (phrase, r)
                assert eng.expressions[_eid(r)].free_parameters == ["k_1", "k_2"]
            finally:
                nb.stop_engine()

    def test_specialize_by_initial_or_boundary_condition_statement_jp(self):
        """"E5 の初期条件を x(0)=1, x'(0)=0 とする" / "…の境界条件を…とする" — states
        the conditions as a declaration rather than an imperative verb, and covers a
        genuine two-point BOUNDARY value problem (not just t=0 initial conditions)."""
        nb.start_engine(title="fpspec_stmt", mode="standard", language="ja", domains=[])
        eng = nb.get_engine()
        sup = _sup()
        try:
            _run(eng, sup, "q を時間依存関数として定義する")
            e1 = _eid(_run(eng, sup, r"\ddot{q} + w^2 q = 0", "math"))
            g = _eid(_run(eng, sup, f"{e1} を一般に解いて"))
            r = _run(eng, sup, f"{g} の初期条件を q(0)=q0, q'(0)=v0 とする")
            assert r["operation"] == "specialize_by_conditions", r
            assert eng.expressions[_eid(r)].free_parameters == []

            g2 = _eid(_run(eng, sup, f"{e1} を一般に解いて"))
            r2 = _run(eng, sup, f"{g2} の境界条件を q(0)=q0, q(10)=q1 とする")
            assert r2["operation"] == "specialize_by_conditions", r2
            e2 = eng.expressions[_eid(r2)]
            assert e2.free_parameters == []
            assert "q0" in e2.canonical and "q1" in e2.canonical
        finally:
            nb.stop_engine()

    def test_specialize_trigger_phrase_variants_jp(self):
        """"決めて"/"定めて"/"決定して" specialize just like "特殊化して" — added after
        a user flagged "特殊化" as unfamiliar CS jargon rather than physics wording."""
        for tail in ("決めて", "定めて", "決定して"):
            nb.start_engine(title="fpspec_jp", mode="standard", language="ja", domains=[])
            eng = nb.get_engine()
            sup = _sup()
            try:
                _run(eng, sup, "q を時間依存関数として定義する")
                e1 = _eid(_run(eng, sup, r"\ddot{q} + w^2 q = 0", "math"))
                g = _eid(_run(eng, sup, f"{e1} を一般に解いて"))
                r = _run(eng, sup, f"{g} を q(0)=q0, q'(0)=v0 で{tail}")
                assert r["operation"] == "specialize_by_conditions", (tail, r)
                assert eng.expressions[_eid(r)].free_parameters == []
            finally:
                nb.stop_engine()

    def test_specialize_trigger_phrase_variants_en(self):
        for phrase in ("specialize {g} with q(0)=q0, q'(0)=v0",
                       "determine the constants in {g} from q(0)=q0, q'(0)=v0",
                       "set the initial conditions of {g} to q(0)=q0, q'(0)=v0"):
            nb.start_engine(title="fpspec_en", mode="standard", language="en", domains=[])
            eng = nb.get_engine()
            sup = _sup()
            try:
                _run(eng, sup, "define q as a function of t")
                e1 = _eid(_run(eng, sup, r"\ddot{q} + w^2 q = 0", "math"))
                g = _eid(_run(eng, sup, f"solve {e1} in general"))
                r = _run(eng, sup, phrase.format(g=g))
                assert r["operation"] == "specialize_by_conditions", (phrase, r)
                assert eng.expressions[_eid(r)].free_parameters == []
            finally:
                nb.stop_engine()

    def test_ode_partial_specialization_keeps_one_constant(self):
        """One condition q(0)=q0 fixes k₂ (cos term) but leaves k₁ free."""
        nb.start_engine(title="fpodepp", mode="standard", language="ja", domains=[])
        eng = nb.get_engine()
        sup = _sup()
        try:
            _run(eng, sup, "q を時間依存関数として定義する")
            e1 = _eid(_run(eng, sup, r"\ddot{q} + w^2 q = 0", "math"))
            g = _eid(_run(eng, sup, f"{e1} を一般に解いて"))
            r = _run(eng, sup, f"{g} を q(0)=q0 で特殊化して")
            e = eng.expressions[_eid(r)]
            # k₁ stays free (mapped back from the backend placeholder); k₂ → q0
            assert e.free_parameters == ["k_1"]
            c = e.canonical.replace(" ", "")
            assert "k_1" in c and "k_2" not in c and "q0" in c
        finally:
            nb.stop_engine()

    def test_ode_first_order_carries_one_constant(self):
        """A 1st-order linear ODE → general solution with a single constant C."""
        nb.start_engine(title="fpode1", mode="standard", language="ja", domains=[])
        eng = nb.get_engine()
        sup = _sup()
        try:
            _run(eng, sup, "q を時間依存関数として定義する")
            e1 = _eid(_run(eng, sup, r"\dot{q} + k q = 0", "math"))
            r = _run(eng, sup, f"{e1} の一般解を求めて")
            e = eng.expressions[_eid(r)]
            assert e.free_parameters == ["C"]
            assert "C" in e.canonical and "%e" in e.canonical
        finally:
            nb.stop_engine()

    def test_condition_fixes_C(self):
        nb.start_engine(title="fp3", mode="standard", language="ja", domains=[])
        eng = nb.get_engine()
        sup = _sup()
        try:
            _run(eng, sup, "x^2", "math")
            i = _eid(_run(eng, sup, "E1 を x で不定積分して"))
            # F(0)=0 → C=0
            r0 = _run(eng, sup, f"{i} で x=0 のとき 0 という条件で C を決めて")
            assert r0["operation"] == "specialize_by_condition"
            assert eng.expressions[_eid(r0)].canonical.replace(" ", "") == "x^3/3"
            assert eng.expressions[_eid(r0)].free_parameters == []
            # F(1)=1 → 1/3+C=1 → C=2/3
            r1 = _run(eng, sup, f"{i} で x=1 のとき 1 という条件で C を決めて")
            assert eng.expressions[_eid(r1)].canonical.replace(" ", "") == "x^3/3+2/3"
        finally:
            nb.stop_engine()

    def test_condition_fixes_C_en(self):
        # Regression: _SPECIALIZE_RE was Japanese-only; unlike its sibling
        # patterns (_SPECIALIZE_MULTI_RE_EN etc.) it had no English counterpart,
        # and specialize_by_condition isn't schema-registered (rule-based-only
        # by design), so English input had no fallback path at all.
        nb.start_engine(title="fp3_en", mode="standard", language="en", domains=[])
        eng = nb.get_engine()
        sup = _sup()
        try:
            _run(eng, sup, "x^2", "math")
            i = _eid(_run(eng, sup, "E1 を x で不定積分して"))
            # F(0)=0 → C=0
            r0 = _run(eng, sup, f"In {i}, when x=0, determine C such that this equals 0")
            assert r0["operation"] == "specialize_by_condition"
            assert eng.expressions[_eid(r0)].canonical.replace(" ", "") == "x^3/3"
            assert eng.expressions[_eid(r0)].free_parameters == []
            # F(1)=1 → 1/3+C=1 → C=2/3
            r1 = _run(eng, sup, f"In {i}, when x=1, determine C such that this equals 1")
            assert eng.expressions[_eid(r1)].canonical.replace(" ", "") == "x^3/3+2/3"
        finally:
            nb.stop_engine()


class TestUnderdeterminedSolve:
    """Instance 4 — an under-determined system (fewer equations than unknowns) has
    free variables.  Maxima returns them as %rN; they become named free parameters
    (the original free variable), and the solution is expressed in terms of them.
    Specializing a free variable by value gives a particular solution."""

    def _solve(self):
        nb.start_engine(title="us", mode="standard", language="ja", domains=[])
        eng = nb.get_engine()
        sup = _sup()
        e1 = _eid(_run(eng, sup, "x + y + z = 1", "math"))
        e2 = _eid(_run(eng, sup, "x - y = 0", "math"))
        r = _run(eng, sup, f"{e1} と {e2} の連立方程式を x,y,z について解いて")
        return eng, sup, r

    def test_free_variable_is_named_parameter(self):
        eng, sup, r = self._solve()
        try:
            assert r["operation"] == "solve_system"
            rows = r["registered_expressions"]
            # x and y are expressed in terms of the free variable z (z=z dropped)
            assert len(rows) == 2
            for row in rows:
                ex = eng.expressions[row["expr_id"]]
                assert ex.free_parameters == ["z"]
                assert "%r" not in ex.canonical          # no Maxima artifact
                assert "z" in ex.canonical
        finally:
            nb.stop_engine()

    def test_specialize_free_variable(self):
        eng, sup, r = self._solve()
        try:
            xid = r["registered_expressions"][0]["expr_id"]
            # x = (1 - z)/2 ; set z = 0 → x = 1/2
            r2 = _run(eng, sup, f"{xid} で z=0 とする")
            assert r2["operation"] == "substitute_free_parameters"
            ex = eng.expressions[_eid(r2)]
            assert ex.canonical.replace(" ", "") == "1/2"
            assert ex.free_parameters == []
        finally:
            nb.stop_engine()

    def test_determined_system_has_no_free_parameters(self):
        """Regression: a fully determined system still solves with no free params."""
        nb.start_engine(title="det", mode="standard", language="ja", domains=[])
        eng = nb.get_engine()
        sup = _sup()
        try:
            e1 = _eid(_run(eng, sup, "x + y = 3", "math"))
            e2 = _eid(_run(eng, sup, "x - y = 1", "math"))
            r = _run(eng, sup, f"{e1} と {e2} の連立方程式を x,y について解いて")
            assert r["operation"] == "solve_system"
            for row in r["registered_expressions"]:
                ex = eng.expressions[row["expr_id"]]
                assert ex.free_parameters == []
                assert "%r" not in ex.canonical
        finally:
            nb.stop_engine()
