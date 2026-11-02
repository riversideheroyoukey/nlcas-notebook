r"""Naming a display-only expression must not destroy the registration.

`\int d^3x f(x)` is display-only by design: the code generator refuses it
(`parser.unsupported_notation.multidim_measure`) so the researcher can write
it, see it, and open it into components — the "write it down first, compute
later" rule in CLAUDE.md.  Every spelling of it registered EXCEPT the one that
gives it a name:

    \int d^3x f(x)              → registered
    h_{m,n} = \int d^3x f(x)    → registered   (compound LHS: no P2 bind)
    A       = \int d^3x f(x)    → RuntimeError: parser.unsupported_notation.multidim_measure

The plain-symbol LHS is the only one that reaches P2's CAS bind, and that bind
already means to skip an unlowerable RHS ("A refusal SKIPS the bind") — but it
tested only `status == "refused"`, while a refusal raised by the CODE GENERATOR
arrives as UnknownNotationError, which `to_backend_code` re-raises on purpose so
an OPERATION can translate it.  On the bind path there is no operation to fail,
so it killed the registration and answered the researcher with a raw i18n key.

Measured 2026-08-20 on `H = \int d^3x \psi^*_s(x) T(x) \psi_s(x)` — the first
line of a many-electron Green's-function notebook.
"""
import asyncio

import pytest

import ui.notebook_bridge as nb
from notebook.engine import NotebookEngine
from notebook.runtime.shared_parser import UnknownNotationError
from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operation_validator import OperationValidator


def _math(eng, sup, latex):
    tok = InputToken(type="math", value=latex, latex=latex, canonical="")
    return asyncio.get_event_loop().run_until_complete(
        run_interpret_pipeline(InterpretRequest(tokens=[tok]), eng, sup))


@pytest.mark.parametrize("lang", ["ja", "en"])
def test_named_multidim_integral_registers(lang):
    nb.start_engine(title="h1_display_only", mode="standard", language=lang,
                    domains=["quantum_mechanics"])
    try:
        eng = nb.get_engine()
        sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                            domains=["quantum_mechanics"], primary_backend="maxima")
        for latex in (r"\int d^3x f(x)",
                      r"h_{m,n} = \int d^3x f(x)",
                      r"A = \int d^3x f(x)",
                      r"H = \int d^3x \psi^*_s(x) T(x) \psi_s(x)"):
            assert _math(eng, sup, latex)["status"] == "registered", latex
        # The name is a usable alias afterwards, i.e. the equation was really
        # written down and not half-registered.
        assert "A" in eng.lhs_alias_map and "H" in eng.lhs_alias_map
    finally:
        nb.stop_engine()


def test_alias_keys_survives_an_unreadable_lhs():
    """`_alias_keys` catches UnknownNotationError so an unreadable LHS still
    yields the as-written key — but the name was never imported at module
    level, so evaluating the except clause raised NameError instead and lost
    the definition entirely (the very outcome its comment forbids)."""
    class _Raises:
        def latex_to_canonical(self, _s):
            raise UnknownNotationError("parser.unsupported_notation.multidim_measure")

    eng = object.__new__(NotebookEngine)
    eng._handler = _Raises()
    assert NotebookEngine._alias_keys(eng, r"\int d^3x f(x)") == [r"\int d^3x f(x)"]
