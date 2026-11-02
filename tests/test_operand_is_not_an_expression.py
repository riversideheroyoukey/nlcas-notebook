"""NL-GRAMMAR ①: the leading formula is the operation's `expr`, or it is an operand.

One question — "does the formula at the head of the sentence become a numbered
expression?" — used to have two answers:

  * Japanese (`_FORMULA_OP_RE`) split at the script transition and registered
    the formula **whatever operation was named**.
  * English (`_en_verb_formula_re`) fires only for operations whose required
    field is `expr`.

So the two languages numbered their expressions differently.  Measured
2026-08-23 over ``tests/corpus/instruction_corpus.json``: 3 of 48 scenarios
diverged, and in ``bcs_gap_equation`` 12 of 14 E-ids meant different things in
the two languages — which is why the English scenario doc was written with
``Invert E3, E4, E5, E6`` where the Japanese said ``E4, E5, E6, E7``.  A
researcher reading the worked example in the other language refers to the wrong
expression.

Bilingual by rule (the ``feedback_bilingual_unit_tests`` memory): the defect was
found by comparing the languages, so a test that checks one of them cannot see
it.
"""
import pytest

from notebook.runtime.operand_boundary import formula_is_an_operation_operand
from notebook.runtime.operation_validator import OperationValidator


@pytest.fixture(scope="module")
def ask():
    """The discriminator, wired the way the pipeline wires it."""
    v = OperationValidator()

    def _ask(rest: str, backend: str) -> bool:
        from notebook.runtime.llm_supervisor import _tokenize
        return formula_is_an_operation_operand(
            rest, v.resolve_alias, v.max_alias_words, v.get_spec,
            lambda _op, spec: backend in spec.get("supported_backends", []),
            _tokenize)
    return _ask


class TestAnOperandIsNotRegisteredAsAnExpression:
    """`heisenberg_equation` takes `operator`/`hamiltonian`, never an `expr`."""

    @pytest.mark.parametrize("rest,backend", [
        ("の Heisenberg 方程式 E1", "julia_full"),
        ("のハイゼンベルク方程式 E1", "julia_full"),
    ])
    def test_japanese(self, ask, rest, backend):
        assert ask(rest, backend) is True

    def test_the_english_spelling_reaches_the_same_answer(self, ask):
        # English puts the verb first, so this side is not reached through the
        # same door — but asked the same question it must answer the same way.
        assert ask("Heisenberg equation of E1 for a(1)", "julia_full") is True


class TestTheOperationsExprIsStillRegistered:
    """The common case must not change: a formula that IS the `expr` registers.

    `differentiate` is `['expr', 'var']` — MORE than one required field — so
    "the operation takes exactly one expr" would have broken the single most
    common instruction in the corpus (100 occurrences).  The discriminator is
    whether a field literally named `expr` is required, and this pins that.
    """

    @pytest.mark.parametrize("rest,backend", [
        ("を微分して", "maxima"),
        ("を p で微分して", "maxima"),
        ("を簡約して", "maxima"),
        ("を正規順序化して", "julia_full"),
    ])
    def test_japanese(self, ask, rest, backend):
        assert ask(rest, backend) is False

    @pytest.mark.parametrize("rest,backend", [
        ("differentiate with respect to p", "maxima"),
        ("simplify", "maxima"),
        ("normal order", "julia_full"),
    ])
    def test_english(self, ask, rest, backend):
        assert ask(rest, backend) is False


class TestNothingNamedMeansNoChange:
    """When the rest of the sentence names no operation, the answer is False.

    That keeps a bare formula, and a formula followed by an instruction this
    system does not recognise, registering exactly as before — refusing there
    would make plain formula input unwritable, which is the opposite of what
    "write it down first, compute later" requires.
    """

    @pytest.mark.parametrize("rest,backend", [
        ("をスカラーとする", "maxima"),   # tokenizes to ['スカラー'] — no alias
        ("の右辺を計算して", "maxima"),    # an H-5 phrase, no schema alias
        ("", "maxima"),
    ])
    def test_no_alias_no_opinion(self, ask, rest, backend):
        assert ask(rest, backend) is False

    def test_a_declaration_subject_is_not_an_expression_either(self, ask):
        """`as a scalar` DOES name an operation — and the answer is "operand".

        Measured while writing this file, against my own wrong expectation:
        `define_scalar_parameter` requires `name`, not `expr`, so the rule says
        the subject is not an expression.  That is the right answer and the one
        CLAUDE.md already states ("A Declaration's Subject Is Not an
        Expression") — the Japanese side reaches it by a different route, the
        backend veto `should_bypass_auto_register_formula`, because
        `をスカラーとする` tokenizes to a word the alias table does not list.
        Same outcome, two routes; recorded here rather than unified, because
        nothing measured says the second route is reachable where the first is
        not.
        """
        assert ask("as a scalar", "maxima") is True


class TestTheBackendIsAsked:
    """An operation the session's backend cannot run is not "named" here.

    Otherwise a Maxima session would refuse to register a formula because some
    Julia-only operation happens to share a word with the sentence.
    """

    def test_a_julia_only_operation_is_ignored_in_a_maxima_session(self, ask):
        assert ask("の Heisenberg 方程式 E1", "julia_full") is True
        assert ask("の Heisenberg 方程式 E1", "maxima") is False
