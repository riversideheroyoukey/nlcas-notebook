r"""The bar `|` fences four different notations, and only one dialog fits.

open_items ABS-PIPE-RESIDUAL said the R-8 confirmation UI was wired for "part
of `|X|`" and the rest was refused without a chance to confirm.  Measured
2026-08-17 that is true, and the part that WAS wired is the worse half:

    \|v\|   →  claimed by the abs/determinant dialog  →  rewrote to `\abs(v\)`
            →  the converter reduced that to `\abs(v`  — unbalanced, silent

`\|…\|` is LaTeX's ‖…‖, the NORM.  Its backslashes fell into the single-pair
regex's `pre` and `inner` groups, so the dialog offered exactly the two readings
that are wrong for it, and either choice produced nonsense.

The shape is now decided first (`_pipe_shape`), and each gets its own answer.
"""
import asyncio

import pytest

import ui.notebook_bridge as nb
from ui.interpret_notation import _pipe_shape, _try_notation_candidates


class TestTheShapeIsDecidedFirst:
    @pytest.mark.parametrize("text,shape", [
        (r"|x|", "other"), (r"|A|", "other"), (r"|x|^2", "other"), (r"2|x|", "other"),
        (r"\|v\|", "norm"), (r"||x||", "norm"),
        (r"\langle a | b \rangle", "braket"), (r"| \psi \rangle", "braket"),
        (r"|x| + |y|", "pairs"), (r"|A| |B|", "pairs"),
    ])
    def test_shape(self, text, shape):
        assert _pipe_shape(text) == shape


class _Pipes:
    def setup_method(self):
        nb.start_engine(title="pipe", mode="standard", language="ja", domains=[])
        self.eng = nb.get_engine()
        self.loop = asyncio.new_event_loop()

    def teardown_method(self):
        nb.stop_engine()

    def _run(self, text, lang="ja"):
        return self.loop.run_until_complete(
            _try_notation_candidates(self.eng, self.loop, lang, text, [], []))


class TestNeitherReadingFits(_Pipes):
    @pytest.mark.parametrize("text,key", [
        (r"\|v\|", "norm"), (r"||x||", "norm"),
        (r"\langle a | b \rangle", "braket"), (r"| \psi \rangle", "braket"),
    ])
    def test_refused_with_its_own_reason(self, text, key):
        resp, _ = self._run(text)
        assert resp is not None and resp["status"] == "failed"
        from notebook.runtime.i18n import t
        assert resp["error"] == t(f"notation.pipe_{key}", "ja")

    @pytest.mark.parametrize("text", [r"\|v\|", r"||x||"])
    def test_the_norm_is_never_offered_abs_or_determinant(self, text):
        """The defect this file exists for: both choices are wrong, and the
        rewrite was malformed whichever was picked."""
        resp, _ = self._run(text)
        assert "candidates" not in resp

    @pytest.mark.parametrize("lang", ["ja", "en"])
    @pytest.mark.parametrize("key", ["norm", "braket"])
    def test_both_reasons_are_translated(self, lang, key):
        from notebook.runtime.i18n import t
        msg = t(f"notation.pipe_{key}", lang)
        assert msg != f"notation.pipe_{key}" and len(msg) > 30


class TestSeveralPairs(_Pipes):
    r"""`|x| + |y|` is ordinary mathematics that the single-pair regex could not
    see — the substance of ABS-PIPE-RESIDUAL."""

    @pytest.mark.parametrize("text,expect", [
        (r"|x| + |y|", "abs(x) + abs(y)"),
        (r"|A| |B|", "abs(A) abs(B)"),
    ])
    def test_one_decision_covers_every_pair(self, text, expect):
        """Asking per pair would ask the same question twice: the bars in one
        expression are the same notation."""
        resp, _ = self._run(text)
        assert resp["status"] == "needs_confirmation"
        assert resp["candidates"][0]["args"]["rewritten"] == expect
        assert [c["args"]["role"] for c in resp["candidates"]] == [
            "abs", "determinant"]

    def test_a_remembered_answer_applies_to_all_of_them(self):
        self.eng.symbol_context.notation_usage["|·|"] = "abs"
        resp, new = self._run(r"|x| + |y|")
        assert resp is None and new == "abs(x) + abs(y)"

    def test_a_declaration_narrows_it_without_asking(self):
        for n in ("M", "N"):
            self.eng.run_operation("define_matrix", {"name": n})
        resp, new = self._run(r"|M| |N|")
        assert resp is None and new == "determinant(M) determinant(N)"


class TestTheSinglePairIsUnchanged(_Pipes):
    @pytest.mark.parametrize("text,expect", [
        (r"|x|", "abs(x)"), (r"|x|^2", "abs(x)^2"), (r"2|x|", "2abs(x)"),
    ])
    def test_still_offers_the_dialog(self, text, expect):
        resp, _ = self._run(text)
        assert resp["candidates"][0]["args"]["rewritten"] == expect
