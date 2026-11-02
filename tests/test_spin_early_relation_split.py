r"""The spin interceptor splits RAW TEXT, and nothing downstream can object.

ONE-QUESTION-MANY-ANSWERS ratchet, seventh lowering (`ui/interpret_early.py`
3→0).  Counting where the three values went:

* `_try_early_spin_register` splits the request text itself — right side to the
  spin parser, left side into the REGISTERED label;
* `_lhs_label` reads an expression's own left side for a `[A, B] = …` display,
  which is registered too;
* `_expand_spin_eids` substitutes a referenced expression's right side into the
  text handed to `spin_algebra.parse_spin_expression`.

The third is what makes this file different from the six before it.  Everywhere
else the mis-split text ended up in a CAS command, so Maxima or Julia had a
chance to refuse it.  `spin_algebra` is CAS-bypass Python (CLAUDE.md, F-9/F-13):
there is no parser between that string and the answer.

Measured 2026-08-13.  Ordinary math-chip input, no exotic route:

    H_{n=1} = \omega S^z_1
        → status completed, stored as `H_{n = \omega S^{z}_{1}`
        the researcher's own subscript read as the equation, `1}` simply gone

and at the helper level, on an expression stored as `\sum_{i=1}^{2} S^z_i = H`:

    _lhs_label        → `\sum_{i`              (a label that gets registered)
    _expand_spin_eids → `1}^{2} S^z_i = H`     (the parser's input, i.e. the value)
"""
import asyncio

import pytest

import ui.notebook_bridge as nb
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operation_validator import OperationValidator
from ui.interpret_early import _expand_spin_eids, _lhs_label
from ui.interpret_pipeline import InputToken, InterpretRequest, run_interpret_pipeline


def _run(engine, supervisor, value, ttype="math"):
    token = (InputToken(type="math", value=value, latex=value, canonical="")
             if ttype == "math" else InputToken(type="text", value=value))
    return asyncio.run(run_interpret_pipeline(
        InterpretRequest(tokens=[token]), engine, supervisor))


@pytest.fixture
def spin_session():
    nb.start_engine(title="test_spin_split", mode="standard", language="ja",
                    domains=["spin_systems"])
    engine = nb.get_engine()
    supervisor = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                               domains=["spin_systems"], primary_backend="maxima")
    try:
        yield engine, supervisor
    finally:
        nb.stop_engine()


def test_a_name_carrying_its_own_equals_survives_registration(spin_session):
    r"""`H_{n=1}` is a NAME, not an equation — and the result is persisted."""
    engine, supervisor = spin_session
    assert _run(engine, supervisor, r"H_{n=1} = \omega S^z_1")["status"] == "completed"
    assert engine.expressions["E1"].latex == r"H_{n=1} = \omega S^{z}_{1}"


def test_the_ordinary_spin_hamiltonian_is_untouched(spin_session):
    engine, supervisor = spin_session
    assert _run(engine, supervisor, r"H = \omega S^z_1")["status"] == "completed"
    assert engine.expressions["E1"].latex == r"H = \omega S^{z}_{1}"


class TestTheHelpers:
    r"""Both read an expression whose latex is `\sum_{i=1}^{2} S^z_i = H`."""

    @pytest.fixture
    def with_sum_expression(self, spin_session):
        engine, supervisor = spin_session
        _run(engine, supervisor, r"\sum_{i=1}^{2} S^z_i = H")
        assert engine.expressions["E1"].latex == r"\sum_{i=1}^{2} S^z_i = H"
        return engine

    def test_the_registered_label_is_the_whole_left_side(self, with_sum_expression):
        assert _lhs_label(with_sum_expression, "E1") == r"\sum_{i=1}^{2} S^z_i"

    def test_the_parser_input_is_the_right_side(self, with_sum_expression):
        """No CAS stands between this string and the computed value."""
        assert _expand_spin_eids(with_sum_expression, "E1") == "H"

    def test_an_unknown_reference_is_left_alone(self, with_sum_expression):
        assert _expand_spin_eids(with_sum_expression, "E9") == "E9"
