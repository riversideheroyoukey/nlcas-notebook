"""Rule-based NL trigger for add_assumption ("X と仮定する" / "assume X"), and
the domain(kind)/sign split for define_scalar_parameter ("正の整数" / "positive
integer" — a compound declaration that keeps BOTH qualifiers instead of one
silently overwriting the other, and sends a real assume() to the CAS for the
sign part).
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


class TestAddAssumptionNLTrigger:
    def test_bare_condition_jp(self):
        nb.start_engine(title="asm1", mode="standard", language="ja", domains=[])
        eng = nb.get_engine()
        sup = _sup()
        try:
            r = _run(eng, sup, "x > 0 と仮定する")
            assert r["operation"] == "add_assumption"
            assert len(eng.assumptions) == 1
            asm = next(iter(eng.assumptions.values()))
            assert asm.canonical.replace(" ", "") == "x>0"
        finally:
            nb.stop_engine()

    def test_bare_condition_en(self):
        nb.start_engine(title="asm2", mode="standard", language="en", domains=[])
        eng = nb.get_engine()
        sup = _sup()
        try:
            r = _run(eng, sup, "assume x > 0")
            assert r["operation"] == "add_assumption"
            asm = next(iter(eng.assumptions.values()))
            assert asm.canonical.replace(" ", "") == "x>0"
        finally:
            nb.stop_engine()

    def test_bare_eid_jp(self):
        """"E1 と仮定する": the condition IS E1's own value (no expansion needed)."""
        nb.start_engine(title="asm3", mode="standard", language="ja", domains=[])
        eng = nb.get_engine()
        sup = _sup()
        try:
            e1 = _eid(_run(eng, sup, "x^2 - 1 > 0", "math"))
            r = _run(eng, sup, f"{e1} と仮定する")
            assert r["operation"] == "add_assumption"
            asm = next(iter(eng.assumptions.values()))
            assert asm.canonical.replace(" ", "") == "x^2-1>0"
        finally:
            nb.stop_engine()

    def test_embedded_eid_is_expanded_jp(self):
        """"E1 > 0 と仮定する": E1 is EMBEDDED in a larger condition and must be
        expanded to its real value before reaching the CAS (not sent as the
        literal undefined symbol "E1"), while the display keeps "E1 > 0"."""
        nb.start_engine(title="asm4", mode="standard", language="ja", domains=[])
        eng = nb.get_engine()
        sup = _sup()
        try:
            e1 = _eid(_run(eng, sup, "x^2 - 1", "math"))
            r = _run(eng, sup, f"{e1} > 0 と仮定する")
            assert r["operation"] == "add_assumption"
            asm = next(iter(eng.assumptions.values()))
            assert "E1" not in asm.canonical
            assert asm.canonical.replace(" ", "") == "(x^2-1)>0"
            assert asm.display == f"{e1} > 0"
        finally:
            nb.stop_engine()

    def test_assume_before_ode_prevents_hang(self):
        """The documented use case: an explicit assume() step, stated by the
        researcher (never defaulted by the tool), lets a damped 2nd-order ODE's
        general solution resolve instead of Maxima asking an interactive
        question that would hang the persistent adapter."""
        nb.start_engine(title="asm5", mode="standard", language="ja", domains=[])
        eng = nb.get_engine()
        sup = _sup()
        try:
            _run(eng, sup, "q を時間依存関数として定義する")
            _run(eng, sup, "gamma を正の実数とする")
            _run(eng, sup, "omega_0 を正の実数とする")
            e1 = _eid(_run(eng, sup, r"\ddot{q} + 2 \gamma \dot{q} + \omega_0^2 q = 0", "math"))
            r = _run(eng, sup, "(2*gamma)^2 < 4*omega_0^2 と仮定する")
            assert r["operation"] == "add_assumption"
            r2 = _run(eng, sup, f"{e1} を一般に解いて")
            assert r2["operation"] == "solve_ode_general"
            e = eng.expressions[_eid(r2)]
            c = e.canonical.replace(" ", "")
            assert "sin" in c and "cos" in c
        finally:
            nb.stop_engine()


class TestDomainSignSplit:
    def test_positive_integer_keeps_both_qualifiers_jp(self):
        nb.start_engine(title="ds1", mode="standard", language="ja", domains=[])
        eng = nb.get_engine()
        sup = _sup()
        try:
            r = _run(eng, sup, "n を正の整数とする")
            assert r["status"] == "confirmed"
            q = eng.symbol_context.get("n").quantity
            assert q.domain == "integer"
            assert q.sign == "positive"
            asm = next(iter(eng.assumptions.values()))
            assert asm.canonical.replace(" ", "") == "n>0"
            assert eng._adapter.execute("is(n > 0)").value == "true"
        finally:
            nb.stop_engine()

    def test_positive_integer_keeps_both_qualifiers_en(self):
        nb.start_engine(title="ds2", mode="standard", language="en", domains=[])
        eng = nb.get_engine()
        sup = _sup()
        try:
            r = _run(eng, sup, "n is a positive integer")
            assert r["status"] == "confirmed"
            q = eng.symbol_context.get("n").quantity
            assert q.domain == "integer"
            assert q.sign == "positive"
            assert eng._adapter.execute("is(n > 0)").value == "true"
        finally:
            nb.stop_engine()

    def test_negative_real_sends_negative_assumption(self):
        nb.start_engine(title="ds3", mode="standard", language="ja", domains=[])
        eng = nb.get_engine()
        sup = _sup()
        try:
            _run(eng, sup, "p を負の実数とする")
            q = eng.symbol_context.get("p").quantity
            assert q.domain == "real"
            assert q.sign == "negative"
            assert eng._adapter.execute("is(p > 0)").value == "false"
            assert eng._adapter.execute("is(p < 0)").value == "true"
        finally:
            nb.stop_engine()

    def test_sign_assumption_display_uses_tex_form(self):
        """assume_sign's auto-generated assumption must keep the symbol's TeX
        display form (e.g. "\\gamma > 0"), not fall back to the internal ascii
        canonical ("gamma > 0"), for the UI tile / TeX export — the Maxima-facing
        canonical stays plain ascii regardless."""
        nb.start_engine(title="ds6", mode="standard", language="ja", domains=[])
        eng = nb.get_engine()
        sup = _sup()
        try:
            _run(eng, sup, r"\gamma を正の実数とする")
            asm = next(iter(eng.assumptions.values()))
            assert asm.canonical.replace(" ", "") == "gamma>0"
            assert asm.display.replace(" ", "") == r"\gamma>0"
        finally:
            nb.stop_engine()

    def test_plain_integer_has_no_sign(self):
        """A bare "整数" (integer) declaration with no sign qualifier must not
        fabricate a sign — only compound phrasings like "正の整数" carry one."""
        nb.start_engine(title="ds4", mode="standard", language="ja", domains=[])
        eng = nb.get_engine()
        sup = _sup()
        try:
            _run(eng, sup, "n を整数とする")
            q = eng.symbol_context.get("n").quantity
            assert q.domain == "integer"
            assert q.sign is None
            assert eng.assumptions == {}
        finally:
            nb.stop_engine()

    def test_julia_backend_sign_is_metadata_only_no_crash(self):
        """Julia/Symbolics has no sign predicate: the domain KIND still maps to
        a type annotation, but the sign qualifier must no-op rather than send
        an unsupported assume() to the Julia session."""
        nb.start_engine(title="ds5", mode="second_quantization", language="ja",
                         domains=["condensed_matter"])
        eng = nb.get_engine()
        sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                            domains=["condensed_matter"], primary_backend=eng.session.primary_backend)
        try:
            r = _run(eng, sup, "xi を正の実数とする")
            assert r["status"] == "confirmed"
            q = eng.symbol_context.get("xi").quantity
            assert q.domain == "real"
            assert q.sign == "positive"
            assert eng.assumptions == {}
        finally:
            nb.stop_engine()


class TestSubscriptDialogKeepsQualifiers:
    r"""A declaration that opens the subscript-role dialog must not lose its
    qualifiers (open_items SIGN-DIALOG-ASSUME, measured 2026-07-31).

    ``_enrich_domain`` supplements ``domain``/``sign`` onto the result's args,
    but picking a candidate calls ``run_operation`` with THAT candidate's args
    — a different dict.  Enriching only the default meant every dialog path
    registered the symbol with NEITHER qualifier: 「omega_D を正の実数とする」
    kept neither the "正の" nor the "実数", with no error anywhere.

    The sign is not decoration.  It is forwarded to Maxima's ``assume()``, and
    without it ``integrate(1/sqrt(xi^2+Delta^2), xi, 0, omega_D)`` stops on an
    unanswered sign question — which times out and takes the rest of the
    session with it (the S-c handoff measurement, docs/34 §6.2, hit exactly
    this).  These tests assert the CAS-visible consequence (``is(omega_D > 0)``),
    not just the stored metadata.
    """

    @staticmethod
    def _pick(r, which):
        """Run the chosen candidate exactly as the UI's confirm click does."""
        cands = r["candidates"]
        assert len(cands) == 2, cands
        return cands[0] if which == "index" else cands[-1]

    def test_dialog_candidates_carry_qualifiers_jp(self):
        nb.start_engine(title="sd1", mode="standard", language="ja", domains=[])
        eng = nb.get_engine()
        sup = _sup()
        try:
            r = _run(eng, sup, "omega_D を正の実数とする")
            assert r["status"] == "needs_confirmation"
            for cand in r["candidates"]:
                assert cand["args"].get("sign") == "positive", cand
                assert cand["args"].get("domain") == "real", cand
        finally:
            nb.stop_engine()

    def test_dialog_candidates_carry_qualifiers_en(self):
        nb.start_engine(title="sd2", mode="standard", language="en", domains=[])
        eng = nb.get_engine()
        sup = _sup()
        try:
            r = _run(eng, sup, "define omega_D as a positive real number")
            assert r["status"] == "needs_confirmation"
            for cand in r["candidates"]:
                assert cand["args"].get("sign") == "positive", cand
                assert cand["args"].get("domain") == "real", cand
        finally:
            nb.stop_engine()

    def test_name_part_choice_reaches_the_cas_jp(self):
        """"名前の一部" → the full name omega_D is what Maxima must know is > 0."""
        nb.start_engine(title="sd3", mode="standard", language="ja", domains=[])
        eng = nb.get_engine()
        sup = _sup()
        try:
            r = _run(eng, sup, "omega_D を正の実数とする")
            chosen = self._pick(r, "name")
            eng.run_operation(chosen["operation"], chosen["args"])
            q = eng.symbol_context.get("omega_D").quantity
            assert q.domain == "real"
            assert q.sign == "positive"
            assert eng._adapter.execute("is(omega_D > 0)").value == "true"
        finally:
            nb.stop_engine()

    def test_index_choice_reaches_the_cas_en(self):
        """"component index" → the BASE name carries the qualifiers instead."""
        nb.start_engine(title="sd4", mode="standard", language="en", domains=[])
        eng = nb.get_engine()
        sup = _sup()
        try:
            r = _run(eng, sup, "define x_k as a negative real number")
            chosen = self._pick(r, "index")
            eng.run_operation(chosen["operation"], chosen["args"])
            q = eng.symbol_context.get("x").quantity
            assert q.domain == "real"
            assert q.sign == "negative"
            assert eng._adapter.execute("is(x < 0)").value == "true"
        finally:
            nb.stop_engine()

    def test_unqualified_declaration_is_untouched(self):
        """No qualifier in the text → no qualifier invented on either candidate."""
        nb.start_engine(title="sd5", mode="standard", language="ja", domains=[])
        eng = nb.get_engine()
        sup = _sup()
        try:
            r = _run(eng, sup, "y_k をスカラーとする")
            assert r["status"] == "needs_confirmation"
            for cand in r["candidates"]:
                assert "sign" not in cand["args"], cand
                assert "domain" not in cand["args"], cand
        finally:
            nb.stop_engine()
