"""A bound index reusing a name the expression uses elsewhere: WARN, never refuse.

`\\sum_{i=n}^{I} \\sum_{n>m}^{N} V_{i,n}` binds `n` in the inner sum while `n` is
also the outer sum's lower limit.  Both readings are legitimate writing, and
which one was meant is in the researcher's head, not in the text — so the system
cannot decide it, and must not pretend to.  What it must not do is stay silent:
the bound `n` and the outer `n` compute as different things and the expression
succeeds either way.

The delivery is tested end to end on purpose.  A warning that is produced but
not wired to the screen is the failure mode this project has already hit twice
(`resp["warning"]` is set by the coordinate fallback and rendered nowhere; the
undefined-symbol banner is rendered on the ERROR path only).
"""

import asyncio

import pytest

from notebook.engine import NotebookEngine
from notebook.models.symbol_context import SymbolContext
from notebook.runtime.bound_name_collision import find_collisions
from notebook.runtime.maxima_backend_handler import MaximaBackendHandler
from notebook.runtime.maxima_semantic_resolver import MaximaSemanticResolver
from notebook.runtime.shared_parser import SharedParser


def _ir(latex: str, context=None):
    canonical = MaximaBackendHandler(None).latex_to_canonical(latex)
    return MaximaSemanticResolver().resolve(SharedParser().parse(canonical),
                                            context or SymbolContext())


class TestDetection:

    @pytest.mark.parametrize("latex,name", [
        # the researcher's own example: `n` is a limit AND a dummy
        (r"\sum_{i=n}^{I} \sum_{n>m}^{N} V_{i,n}", "n"),
        # a sum whose own upper limit names its dummy
        (r"\sum_{i=1}^{i} x_i",                    "i"),
    ])
    def test_a_collision_is_reported(self, latex, name):
        assert find_collisions(_ir(latex)) == [(name, "bound_name.also_free")]

    @pytest.mark.parametrize("latex", [
        r"\sum_{i=1}^{N} x_i",
        r"\sum_{i=1}^{N} \sum_{j>i}^{N} V_{i,j}",
        r"\sum_{i=1}^{I} \sum_{j>i}^{J} \sum_{k>j}^{K} V_{i,j,k}",
        r"\sum_{m,n} h_{m,n}",
    ])
    def test_ordinary_writing_is_quiet(self, latex):
        # A chained nested sum names each dummy once — reporting those would
        # make the warning noise and it would stop being read.
        assert find_collisions(_ir(latex)) == []

    @pytest.mark.parametrize("latex", [
        r"n \sum_{n=1}^{N} x_n",              # a free factor of the same name
        r"\sum_{i=1}^{3} \sum_{i>1}^{3} x_i",  # a NESTED reuse of one name
    ])
    def test_the_renamer_gets_there_first(self, latex):
        # The dummy-renaming step already moves a dummy off a name in view, so
        # there is nothing left to report.  This is the division of labour:
        # what survives to be WARNED about is what renaming cannot see — a name
        # used in a LIMIT rather than as another dummy.  Pinned so a change
        # there shows up here rather than as sudden noise in the UI.
        assert find_collisions(_ir(latex)) == []

    def test_nothing_to_walk(self):
        assert find_collisions(None) == []


class TestItIsAWarningNotARefusal:

    def test_the_expression_registers_and_computes(self):
        with NotebookEngine(title="warn") as e:
            e.start()
            h = e._handler
            # The renamer handles a NESTED reuse on its own (the inner dummy
            # becomes `j`), so the surviving cases are the ones it cannot see:
            # a name that is a LIMIT rather than another dummy.
            latex = r"\sum_{i=1}^{i} x_i"
            expr = e.register_expression(h.latex_to_canonical(latex), latex=latex)
            assert expr.warnings, "the collision must be reported"
            # …and the operation is NOT blocked.
            step, _ = e.run_operation("simplify", {"expr": expr.display_id})
            assert step.output_refs


class TestTheWarningReachesTheScreen:

    @staticmethod
    def _response(latex: str) -> dict:
        from tests.test_debug_repl import _sup
        from ui.interpret_pipeline import InterpretRequest, run_interpret_pipeline
        with NotebookEngine(title="deliver") as e:
            e.start()
            req = InterpretRequest(tokens=[{"type": "math", "latex": latex}])
            return asyncio.run(run_interpret_pipeline(req, e, _sup()))

    def test_it_is_in_the_registration_response(self):
        res = self._response(r"\sum_{i=n}^{I} \sum_{n>m}^{N} V_{i,n}")
        assert res.get("status") == "registered"
        rows = res.get("registered_expressions") or []
        assert rows and rows[0].get("warnings"), res

    def test_the_message_is_translated_and_names_the_index(self):
        rows = self._response(r"\sum_{i=n}^{I} \sum_{n>m}^{N} V_{i,n}")[
            "registered_expressions"]
        message = rows[0]["warnings"][0]
        assert "bound_name" not in message          # not a raw key
        assert "n" in message

    def test_an_ordinary_expression_carries_no_warning(self):
        rows = self._response(r"\sum_{i=1}^{N} \sum_{j>i}^{N} V_{i,j}")[
            "registered_expressions"]
        assert rows[0].get("warnings") == []

    @pytest.mark.parametrize("lang", ["ja", "en"])
    def test_both_languages_have_the_message(self, lang):
        from notebook.runtime.i18n import t
        for key in ("bound_name.also_free", "bound_name.also_declared"):
            msg = t(key, lang, name="n")
            assert msg != key and "n" in msg

    def test_the_renderer_reads_the_field(self):
        # The wiring check: the timeline card must actually look at `warnings`.
        # Produced-but-never-rendered is how `resp["warning"]` ended up dead.
        import pathlib
        js = pathlib.Path("ui/notebook_timeline.js").read_text()
        assert "data.warnings" in js
        assert "expr-warning" in js
        css = pathlib.Path("ui/notebook_main.css").read_text()
        assert ".expr-warning" in css
