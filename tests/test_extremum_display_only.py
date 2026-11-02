r"""EXTREMUM-BINDER: `\max_i x_i` is written down, shown, and refused at USE.

The notation has no backend form — measured 2026-08-16,
``lmax(makelist(x[i],i,1,3))`` works but ``lmax(makelist(x[i],i,1,n))`` fails,
so an extremum over a symbolic index has no command at all.  Read as an
ordinary subscript it lowered to ``max(i) * x(i)``, and Maxima's ``max`` of one
argument IS that argument, so ``\max_i x_i`` computed as ``i·x(i)``: valid CAS
text, a different quantity, and silent.

The first fix refused it in the resolver, which stopped the silent-wrong **and
stopped the notation being writable** — a researcher could not put their own
objective function in their own notebook.  `\max_i` and `\sup_x` are everyday
notation in optimisation, analysis, statistics and machine learning, so the
answer is this system's own display-only pattern (`lattice_descriptor`,
`integral_descriptor`): write it down now, refuse at the point of use.

These tests pin all three halves of that, because losing any one of them
recreates a defect this file exists to prevent:

* it REGISTERS and DISPLAYS (lose this → back to unwritable),
* a generic operation is REFUSED with a reason (lose this → back to `i·x(i)`),
* `\max(a, b)` still COMPUTES (lose this → the refusal message points nowhere).
"""
import asyncio

import pytest

import ui.notebook_bridge as nb
from notebook.runtime.extremum_form import build_extremum_latex, parse_latex_extremum
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operation_validator import OperationValidator
from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline

DOMAIN = "operations_research"


def _sup():
    return LLMSupervisor(validator=OperationValidator(), llm_client=None,
                         domains=[DOMAIN], primary_backend="maxima")


def _run(eng, sup, cmd):
    return asyncio.get_event_loop().run_until_complete(
        run_interpret_pipeline(
            InterpretRequest(tokens=[InputToken(type="text", value=cmd)]), eng, sup))


class TestTheDescriptorReadsEverySpelling:
    r"""The pure reader — no engine, no CAS."""

    @pytest.mark.parametrize("latex,op,index,body", [
        (r"\max_i x_i",                "max",    "i",            "x_i"),
        (r"\max_{i} x_i",              "max",    "i",            "x_i"),
        (r"\min_i x_i",                "min",    "i",            "x_i"),
        (r"\sup_{x \in X} f(x)",       "sup",    r"x \in X",     "f(x)"),
        (r"\arg\max_i x_i",            "argmax", "i",            "x_i"),
        (r"\min_{1\le i\le n} d_i",    "min",    r"1\le i\le n", "d_i"),
        (r"\max_{i,j} A_{i,j}",        "max",    "i,j",          "A_{i,j}"),
        (r"\limsup_n a_n",             "limsup", "n",            "a_n"),
    ])
    def test_read(self, latex, op, index, body):
        d = parse_latex_extremum(latex)
        assert d is not None, latex
        assert (d["op"], d["index_latex"], d["body_latex"]) == (op, index, body)

    @pytest.mark.parametrize("latex", [
        r"\max(a, b)",                     # an ordinary call — must not be claimed
        r"\max\{a, b\}",
        r"\max z = c_1 x + c_2 y",         # an LP written in prose (a real memo)
        r"\max_i",                         # a binder with nothing bound
        r"x_i + y_i",
        r"\sum_i x_i",
    ])
    def test_not_claimed(self, latex):
        assert parse_latex_extremum(latex) is None, latex

    @pytest.mark.parametrize("latex", [
        r"\max_i x_i", r"\sup_{x \in X} f(x)", r"\arg\max_i x_i",
        r"z = \max_i x_i", r"2 \max_i x_i",
    ])
    def test_rebuild_keeps_what_was_written(self, latex):
        """Including anything written BEFORE the binder — an equation's left
        side, or a coefficient."""
        rebuilt = build_extremum_latex(parse_latex_extremum(latex))
        for token in (r"\max", r"\sup", r"\arg", "z =", "2"):
            if token in latex:
                assert token in rebuilt, f"{latex} → {rebuilt} lost {token!r}"


class TestItIsWritableAndShown:
    @pytest.mark.parametrize("latex", [
        r"\max_i x_i",
        r"\sup_{x \in X} f(x)",
        r"z = \max_i x_i",
    ])
    def test_registers_with_a_descriptor(self, latex):
        nb.start_engine(title="extremum", mode="standard", language="ja",
                        domains=[DOMAIN])
        try:
            eng, sup = nb.get_engine(), _sup()
            r = _run(eng, sup, latex)
            assert r["status"] == "registered", r
            expr = eng.expressions[r["expr_id"]]
            assert expr.extremum_descriptor is not None
            assert r"\max" in expr.latex or r"\sup" in expr.latex
        finally:
            nb.stop_engine()


class TestItIsRefusedAtTheEndOfUse:
    def test_a_generic_operation_fails_with_a_reason(self):
        nb.start_engine(title="extremum_op", mode="standard", language="ja",
                        domains=[DOMAIN])
        try:
            eng, sup = nb.get_engine(), _sup()
            eid = _run(eng, sup, r"\max_i x_i")["expr_id"]
            out = _run(eng, sup, f"{eid} を簡約して")
            assert out["status"] == "failed", out
            # The reason must name the object and point somewhere: the whole
            # difference from the old `i*x(i)` is that the researcher is told.
            assert eid in str(out["error"])
            assert "max" in str(out["error"])
        finally:
            nb.stop_engine()

    def test_a_symbolic_range_says_so(self):
        """`\\max_{i=1}^{n}` names a range the CAS cannot take — measured,
        `lmax(makelist(x(i),i,1,n))` fails — so it must be told, not sent."""
        nb.start_engine(title="extremum_sym", mode="standard", language="ja",
                        domains=[DOMAIN])
        try:
            eng, sup = nb.get_engine(), _sup()
            eid = _run(eng, sup, r"\max_{i=1}^{n} x_i")["expr_id"]
            out = _run(eng, sup, f"{eid} を評価して")
            assert out["status"] == "failed", out
            assert eid in str(out["error"])
        finally:
            nb.stop_engine()

    def test_the_ordinary_call_the_message_points_at_still_computes(self):
        """A refusal that points at something broken is not a refusal."""
        from notebook.models.symbol_context import SymbolContext
        from notebook.runtime.maxima_backend_handler import MaximaBackendHandler
        h = MaximaBackendHandler(None)
        h._context = SymbolContext()
        assert h.to_backend_code(
            h.latex_to_canonical(r"\max(a, b, c)")).code == "max(a, b, c)"


class TestAConcreteRangeComputes:
    r"""The boundary is measured, not assumed: `lmax(makelist(x(i),i,1,3))`
    answers and `lmax(makelist(x(i),i,1,n))` fails, so a concrete integer range
    — and only that — has a value."""

    @pytest.mark.parametrize("latex,expected", [
        (r"\max_{i=1}^{3} x_i",        "max(x(1),x(2),x(3))"),
        (r"\min_{i=2}^{4} f(i)",       "min(f(2),f(3),f(4))"),
        (r"\max_{1\le i\le 5} i^2-3i", "10"),
        (r"\min_{1\le i\le 4} i^2",    "1"),
    ])
    def test_value(self, latex, expected):
        nb.start_engine(title="extremum_val", mode="standard", language="ja",
                        domains=[DOMAIN])
        try:
            eng, sup = nb.get_engine(), _sup()
            eid = _run(eng, sup, latex)["expr_id"]
            out = _run(eng, sup, f"{eid} を評価して")
            assert out["status"] == "completed", out
            assert out["canonical"].replace(" ", "") == expected
        finally:
            nb.stop_engine()

    def test_the_written_spelling_survives_into_the_result(self):
        r"""`\max_{1\le i\le 5}` is written back as written — the researcher's
        own range, not the `i=1`-plus-superscript one it could be rewritten to."""
        nb.start_engine(title="extremum_disp", mode="standard", language="ja",
                        domains=[DOMAIN])
        try:
            eng, sup = nb.get_engine(), _sup()
            eid = _run(eng, sup, r"\max_{1\le i\le 5} i^2-3i")["expr_id"]
            out = _run(eng, sup, f"{eid} を評価して")
            assert r"\max_{1\le i\le 5}" in eng.expressions[out["expr_id"]].latex
        finally:
            nb.stop_engine()

    def test_an_indexed_body_displays_as_the_researcher_wrote_it(self):
        r"""`x_i` must come back as `x_{1}`, not `x\left(1\right)`.

        This object is registered through `register_direct` — no IR parse — so
        nothing had told the display bridge that `x` was written subscripted,
        and the answer showed the CAS spelling while the SUM path, whose
        registration does parse, showed `x_{1}` for the same writing.
        """
        nb.start_engine(title="extremum_idx", mode="standard", language="ja",
                        domains=[DOMAIN])
        try:
            eng, sup = nb.get_engine(), _sup()
            eid = _run(eng, sup, r"\max_{i=1}^{3} x_i")["expr_id"]
            out = _run(eng, sup, f"{eid} を評価して")
            latex = eng.expressions[out["expr_id"]].latex
            assert "x_{1}" in latex, latex
            assert r"x\left(1\right)" not in latex, latex
        finally:
            nb.stop_engine()


class TestTheRangeReaderRefusesWhatItCannotDecide:
    @pytest.mark.parametrize("latex", [
        r"\max_{1 < i < 3} x_i",          # strict: `i=2` over integers, else not
        r"\max_{i \in \{1,2,3\}} x_i",    # a SET, read by read_value_set instead
        r"\max_i x_i",                    # no range at all
    ])
    def test_no_contiguous_range_is_read(self, latex):
        from notebook.runtime.extremum_form import parse_latex_extremum, read_range
        assert read_range(parse_latex_extremum(latex)) is None, latex

    @pytest.mark.parametrize("latex,rng", [
        (r"\max_{i=1}^{3} x_i",        ("i", "1", "3")),
        (r"\max_{1\le i\le 3} x_i",    ("i", "1", "3")),
        (r"\max_{i=1}^{n} x_i",        ("i", "1", "n")),
        (r"\min_{2\le k\le m} y_k",    ("k", "2", "m")),
    ])
    def test_range_is_read(self, latex, rng):
        from notebook.runtime.extremum_form import parse_latex_extremum, read_range
        assert read_range(parse_latex_extremum(latex)) == rng


class TestAnExplicitSetComputes:
    r"""`\max_{i \in \{1,2,3\}}` was refused with a reason that was FALSE.

    The message said "the range is not concrete integers" about `\{1,2,3\}`,
    which is nothing but concrete integers (open_items EXTREMUM-EXPLICIT-SET).
    It had simply never been read — and it was bundled in `docs/refusals.md`
    with `\max_{1<i<3}`, which IS undecidable, so one row's ambiguity made the
    unambiguous one look like a judgement call.

    Measured 2026-08-18: `lmax(makelist(x[i], i, [1,3,7]))` answers
    `max(x[1],x[3],x[7])`, so a NON-CONTIGUOUS set works — which is the whole
    reason this spelling exists, since a run would be written as a range.
    """

    @pytest.mark.parametrize("latex,expected", [
        (r"\max_{i \in \{1,2,3\}} x_i", "max(x(1),x(2),x(3))"),
        (r"\max_{i \in \{1,3,7\}} x_i", "max(x(1),x(3),x(7))"),
        (r"\min_{i \in \{1,2\}} x_i",   "min(x(1),x(2))"),
        (r"\max_{i \in \{1,2,3\}} i^2", "9"),
    ])
    def test_value(self, latex, expected):
        nb.start_engine(title="extremum_set", mode="standard", language="ja",
                        domains=[DOMAIN])
        try:
            eng, sup = nb.get_engine(), _sup()
            eid = _run(eng, sup, latex)["expr_id"]
            out = _run(eng, sup, f"{eid} を評価して")
            assert out["status"] == "completed", out
            assert out["canonical"].replace(" ", "") == expected
        finally:
            nb.stop_engine()

    @pytest.mark.parametrize("latex,vset", [
        (r"\max_{i \in \{1,2,3\}} x_i", ("i", ["1", "2", "3"])),
        (r"\max_{i \in \{1,3,7\}} x_i", ("i", ["1", "3", "7"])),
        (r"\min_{k \in \{2\}} y_k",     ("k", ["2"])),
    ])
    def test_the_set_is_read(self, latex, vset):
        from notebook.runtime.extremum_form import parse_latex_extremum, read_value_set
        assert read_value_set(parse_latex_extremum(latex)) == vset

    @pytest.mark.parametrize("latex", [
        r"\max_{i \in \{a,b\}} x_i",     # symbolic members are not values
        r"\max_{i=1}^{3} x_i",           # a range, not a set
        r"\max_i x_i",                   # no domain at all
    ])
    def test_what_is_not_a_value_set(self, latex):
        from notebook.runtime.extremum_form import parse_latex_extremum, read_value_set
        assert read_value_set(parse_latex_extremum(latex)) is None

    def test_the_written_spelling_survives(self):
        nb.start_engine(title="extremum_set2", mode="standard", language="ja",
                        domains=[DOMAIN])
        try:
            eng, sup = nb.get_engine(), _sup()
            eid = _run(eng, sup, r"\max_{i \in \{1,3,7\}} x_i")["expr_id"]
            out = _run(eng, sup, f"{eid} を評価して")
            assert r"\max_{i \in \{1,3,7\}}" in out["latex"], out["latex"]
        finally:
            nb.stop_engine()


class TestTheStrictInequalityGetsItsOwnReason:
    r"""`\max_{1<i<3}` is the one that genuinely cannot be read, and saying
    "the range is not concrete integers" about it was wrong twice over: both
    bounds plainly ARE integers, and the real obstacle is the strictness."""

    def test_strict_is_told_apart(self):
        from notebook.runtime.extremum_form import parse_latex_extremum, is_strict_range
        assert is_strict_range(parse_latex_extremum(r"\max_{1<i<3} x_i"))
        assert not is_strict_range(parse_latex_extremum(r"\max_{1\le i\le 3} x_i"))
        assert not is_strict_range(parse_latex_extremum(r"\max_{i=1}^{n} x_i"))

    @pytest.mark.parametrize("lang", ["ja", "en"])
    def test_the_reason_names_the_integer_and_real_readings(self, lang):
        from notebook.runtime.i18n import t
        msg = t("extremum.range_is_strict", lang, expr_id="E1")
        assert msg != "extremum.range_is_strict"
        assert "E1" in msg

    def test_the_two_reasons_are_different(self):
        from notebook.runtime.i18n import t
        assert (t("extremum.range_is_strict", "ja", expr_id="E1")
                != t("extremum.range_not_concrete", "ja", expr_id="E1"))
