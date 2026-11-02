r"""The SymPy route must not depend on how the researcher spells an operator.

Found 2026-08-07 while measuring what a Green's-function perturbation expansion
would need.  Three silent-wrongs sat on top of each other, all on the path an
electron-electron interaction inevitably takes (momentum transfer q makes the
indices algebraic, which routes to SymPy):

1. **The braced spelling came back as the literal 1.**
   `c'_{k+q,u} c_{k,u}` → `latex: '1'`, `status: completed`, no error.
   The builder converted operator notation with a regex that only matched the
   PAREN form `c'(`, so the braced form reached SymPy unconverted.  This is the
   spelling CLAUDE.md documents for multi-index operators — and the spelling
   the QA index error itself tells the researcher to use.

2. **A scalar coefficient was silently dropped.**
   `V(q) cd(k+q,u) c(k,u)` → `cd(k + q,u)*c(k,u)`.  The interaction strength,
   gone.  Cause: this project writes products with SPACES, SymPy needs `*`, so
   sympify failed on the coefficient.

3. **The driver turned that parse failure into 1.**
   `try sp.sympify(...) catch; sp.Integer(1)` — which is why (1) and (2) were
   silent rather than loud.  (3) is the one that mattered: without it the other
   two would have been an error message on first use.

The fix serializes from the IR (the builder had already built it, to detect the
algebraic indices in the first place), so both spellings become one string.

`vacuum_expectation` had the identical pair of lines and is fixed with it —
that is the operation the BCS scenario uses.
"""
import pytest

import ui.notebook_bridge as nb


@pytest.fixture
def eng():
    nb.start_engine(title="sympy_spelling", mode="second_quantization",
                    language="ja", domains=["condensed_matter"])
    e = nb.get_engine()
    e.run_operation("define_fermion_operator", {"name": "c"})
    try:
        yield e
    finally:
        nb.stop_engine()


def _normal_order(e, canonical):
    ex = e.register_expression(canonical)
    step, _ = e.run_operation("normal_order", {"expr": ex.display_id})
    return step, e.expressions[step.output_refs[0]]


def _sent_to_sympy(step):
    """The expression string the rendered command hands the SymPy helper."""
    cmd = str(getattr(step, "rendered_command", ""))
    assert "_nb_sympy" in cmd, f"expected the SymPy route, got: {cmd[:120]}"
    return cmd.split('("', 1)[1].split('",', 1)[0]


# The same physics written the two ways this system accepts.
_SAME_EXPRESSION = [
    (r"c'_{k+q,u} c_{k,u}", r"c'(k+q,u) c(k,u)"),
    (r"c_{-k,d} c_{k,u}", r"c(-k,d) c(k,u)"),
    (r"c'_{k+q,u} c'_{p-q,d} c_{p,d} c_{k,u}",
     r"c'(k+q,u) c'(p-q,d) c(p,d) c(k,u)"),
]


@pytest.mark.parametrize("braced,paren", _SAME_EXPRESSION)
def test_both_spellings_send_identical_sympy(eng, braced, paren):
    """The invariant that matters: one physics, one string.

    Not "the braced form now works" — that would let the two drift apart again.
    """
    s_braced, _ = _normal_order(eng, braced)
    s_paren, _ = _normal_order(eng, paren)
    assert _sent_to_sympy(s_braced) == _sent_to_sympy(s_paren)


@pytest.mark.parametrize("braced,paren", _SAME_EXPRESSION)
def test_neither_spelling_collapses_to_one(eng, braced, paren):
    """The observed symptom, pinned directly: `latex == '1'` for an expression
    that plainly still has operators in it."""
    for canonical in (braced, paren):
        _, out = _normal_order(eng, canonical)
        assert out.latex.strip() not in ("1", "-1"), (
            f"{canonical} normal-ordered to {out.latex!r}")
        assert "c" in out.latex


class TestTheScalarCoefficientSurvives:
    r"""`V(q)` is the electron-electron interaction; losing it loses the physics."""

    @pytest.mark.parametrize("canonical", [
        r"V(q) c'_{k+q,u} c_{k,u}",
        r"V(q) c'(k+q,u) c(k,u)",
        r"V(q) c'_{k+q,u} c'_{p-q,d} c_{p,d} c_{k,u}",
    ])
    def test_coefficient_is_still_there(self, eng, canonical):
        eng.run_operation("define_scalar_parameter", {"name": "V"})
        step, out = _normal_order(eng, canonical)
        assert "V" in _sent_to_sympy(step), _sent_to_sympy(step)
        assert "V" in out.latex, out.latex

    def test_the_product_separator_reaches_sympy_as_a_star(self, eng):
        """A space is this project's multiplication sign; SymPy cannot read it."""
        step, _ = _normal_order(eng, r"c'_{k+q,u} c_{k,u}")
        sent = _sent_to_sympy(step)
        assert "*" in sent, sent
        assert " " not in sent.replace("k + q", "").replace("p - q", ""), sent


class TestEveryMomentumIsDeclared:
    """A momentum missing from vars_str is one SymPy treats as an ordinary free
    symbol, so δ(k, -k) stops evaluating to 0.  The old scan read indices only
    from inside parentheses, so for the braced spelling it found none and fell
    back to the single default "k"."""

    def test_all_three_momenta_are_passed(self, eng):
        step, _ = _normal_order(eng, r"c'_{k+q,u} c'_{p-q,d} c_{p,d} c_{k,u}")
        cmd = str(step.rendered_command)
        vars_str = cmd.rsplit('"', 2)[-2]
        assert set(vars_str.split(",")) >= {"k", "p", "q"}, vars_str


class TestTheAlgebraIsActuallyRun:
    """Guard against a fix that makes the string pretty but stops computing."""

    def test_anticommutation_still_fires(self, eng):
        _, out = _normal_order(eng, r"c_{k,u} c'_{k,u}")
        assert "1" in out.latex and "c" in out.latex, out.latex

    def test_the_qa_path_is_untouched(self, eng):
        """Simple indices must still go to QuantumAlgebra, not SymPy."""
        step, out = _normal_order(eng, r"c'_{k,u} c'_{p,d} c_{p,d} c_{k,u}")
        assert "_nb_sympy" not in str(step.rendered_command)
        assert out.latex.strip() not in ("1", "-1")


class TestTheDriverRefusesWhatItCannotRead:
    """The backstop.  With the builder fixed this should never fire in normal
    use — which is exactly why it must be tested directly: a swallowed parse
    failure is invisible until something else breaks."""

    def test_unreadable_input_is_an_error_not_a_one(self, eng):
        bad = r"c\'_{k+q,u} c_{k,u}"          # the pre-fix, unconverted form
        cmd = (f'_nb_sympy_normal_order("{bad}", '
               f'"{{\\"c\\": \\"fermion\\"}}", "k")')
        res = eng._adapter.execute(cmd)
        assert not getattr(res, "ok", True), (
            f"expected a refusal, got value={getattr(res, 'value', None)!r}")
        assert getattr(res, "value", "") != "1"
