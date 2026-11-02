"""Rule-based symbol→value substitution: "En の X に Y を代入して".

Regression: the supervisor's positional token mapping failed to fill the
substitute `rhs` when the value Y contained a "/" (e.g. λ/μ), so the natural
command was routed to the LLM.  A dedicated intercept now parses it.
"""
import asyncio

import ui.notebook_bridge as nb
from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operation_validator import OperationValidator


def _run(eng, sup, value, ttype="text"):
    tok = (InputToken(type="math", value=value, latex=value, canonical="")
           if ttype == "math" else InputToken(type="text", value=value))
    return asyncio.get_event_loop().run_until_complete(
        run_interpret_pipeline(InterpretRequest(tokens=[tok]), eng, sup))


class TestSubstituteIntoEnglish:
    """The same M/M/1 operations driven by ENGLISH commands (input-NL i18n).

    The NL patterns live in i18n/nl_patterns.json; the formula-registration paths
    use the language-aware _is_nl_instruction so an English command like
    "In E2, substitute …" is routed to the operation, not registered as a formula.
    """
    def _engine(self):
        nb.start_engine(title="subst-en", mode="standard", language="en", domains=[])
        return nb.get_engine(), LLMSupervisor(
            validator=OperationValidator(), llm_client=None, domains=[],
            primary_backend="maxima")

    def test_substitute_into_english(self):
        eng, sup = self._engine()
        try:
            _run(eng, sup, r"\rho = \lambda/\mu", "math")          # E1
            _run(eng, sup, r"L = \rho/(1-\rho)", "math")           # E2
            r = _run(eng, sup, r"In E2, substitute \lambda/\mu for \rho")
            assert r["operation"] == "substitute"
            c = eng.expressions[
                r.get("expr_id") or r["registered_expressions"][-1]["expr_id"]
            ].canonical.replace(" ", "")
            assert "rho" not in c and "lambda" in c and "mu" in c
        finally:
            nb.stop_engine()

    def test_both_sides_divide_english(self):
        eng, sup = self._engine()
        try:
            _run(eng, sup, r"\rho = \lambda/\mu", "math")
            _run(eng, sup, r"L = \rho/(1-\rho)", "math")
            _run(eng, sup, r"In E2, substitute \lambda/\mu for \rho")   # E3: L=λ/(μ-λ)
            r = _run(eng, sup, r"Divide both sides of E3 by \lambda")    # E4: L/λ=1/(μ-λ)
            assert r["operation"] == "both_sides_op"
            latex = r["latex"]
            assert r"\frac{L}{\lambda}" in latex
            assert "{lambda}" not in latex   # Greek backslash preserved in label
        finally:
            nb.stop_engine()


def _eid(r):
    return r.get("expr_id") or r["registered_expressions"][-1]["expr_id"]


class TestSubstituteInto:
    def _engine(self):
        nb.start_engine(title="subst", mode="standard", language="ja", domains=[])
        return nb.get_engine(), LLMSupervisor(
            validator=OperationValidator(), llm_client=None, domains=[],
            primary_backend="maxima")

    def test_substitute_symbol_with_slash_value(self):
        """"E2 の ρ に λ/μ を代入して" — the value λ/μ (with "/") is extracted."""
        eng, sup = self._engine()
        try:
            _run(eng, sup, r"\rho = \lambda/\mu", "math")          # E1
            _run(eng, sup, r"L = \rho/(1-\rho)", "math")           # E2 = L=ρ/(1-ρ)
            r = _run(eng, sup, r"E2 の \rho に \lambda/\mu を代入して")
            assert r["operation"] == "substitute"
            c = eng.expressions[_eid(r)].canonical.replace(" ", "")
            assert "rho" not in c                       # ρ was replaced
            # L = (λ/μ)/(1-λ/μ) = λ/(μ-λ)  (Maxima form: -λ/(λ-μ))
            assert "lambda" in c and "mu" in c
        finally:
            nb.stop_engine()

    def test_substitute_then_simplify_gives_mm1_L(self):
        """End-to-end M/M/1: L = ρ/(1-ρ), ρ=λ/μ → L = λ/(μ-λ)."""
        eng, sup = self._engine()
        try:
            _run(eng, sup, r"\rho = \lambda/\mu", "math")
            _run(eng, sup, r"L = \rho/(1-\rho)", "math")
            _run(eng, sup, r"E2 の \rho に \lambda/\mu を代入して")
            r = _run(eng, sup, "E3 を簡約して")
            c = eng.expressions[_eid(r)].canonical.replace(" ", "")
            # RHS only (the "L =" lives in latex).  λ/(μ-λ) up to Maxima's sign form.
            assert c in ("-(lambda/(lambda-mu))", "-lambda/(lambda-mu)",
                         "lambda/(mu-lambda)")
        finally:
            nb.stop_engine()

    def test_eid_substitution_still_works(self):
        """Regression: the E-id→E-id form "E1 を E2 に代入して" is unaffected."""
        eng, sup = self._engine()
        try:
            _run(eng, sup, r"\rho = \lambda/\mu", "math")          # E1
            _run(eng, sup, r"L = \rho/(1-\rho)", "math")           # E2
            r = _run(eng, sup, "E1 を E2 に代入して")
            assert r["operation"] == "substitute"
            assert "rho" not in eng.expressions[_eid(r)].canonical
        finally:
            nb.stop_engine()

    def test_both_sides_divide_by_greek_operand(self):
        """M/M/1 Little's law: divide both sides of L = λ/(μ-λ) by λ.

        Root-cause regression: a Greek LaTeX operand (\\lambda) must be (1) converted
        to the CAS canonical (lambda) for the division, and (2) kept as \\lambda for
        the LHS label so it renders \\frac{L}{\\lambda} — not \\frac{L}{lambda}.
        """
        eng, sup = self._engine()
        try:
            _run(eng, sup, r"\rho = \lambda/\mu", "math")          # E1
            _run(eng, sup, r"L = \rho/(1-\rho)", "math")           # E2
            _run(eng, sup, r"E2 の \rho に \lambda/\mu を代入して")  # E3: L = λ/(μ-λ)
            r = _run(eng, sup, r"E3 の両辺を \lambda で割って")       # E4: L/λ = 1/(μ-λ)
            assert r["operation"] == "both_sides_op"
            # Result RHS W = 1/(μ-λ)  (Maxima sign form -1/(λ-μ)).
            c = eng.expressions[_eid(r)].canonical.replace(" ", "")
            assert c in ("-(1/(lambda-mu))", "-1/(lambda-mu)", "1/(mu-lambda)")
            # LHS label keeps the Greek backslash, not the CAS token "lambda".
            latex = r["latex"]
            assert r"\frac{L}{\lambda}" in latex or r"L/\lambda" in latex
            assert "{lambda}" not in latex and "/lambda" not in latex
        finally:
            nb.stop_engine()
