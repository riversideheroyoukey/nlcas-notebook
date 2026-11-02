r"""英語の指示が、**書いた式**を計算すること（live CAS）。

実測 2026-08-22、利用者が通る経路で:

    simplify a x^2 + b x + c   →  x^2                    completed
    factor x^2 - 1             →  x^2                    completed

空白を含む式が先頭の1項に切り詰められ、**別の式が計算されて成功として返って
いた**。文字列の検査でも捕まるが、ここは値で見る——「動詞＋式」の族で本当に
困るのは「切れた」ことではなく「**違う答えが返った**」ことだからである。
"""
import pytest

from notebook.engine import NotebookEngine


@pytest.fixture(scope="module")
def maxima_session():
    """ONE Maxima process for the module."""
    with NotebookEngine(title="en_formula") as engine:
        engine.start()
        yield engine


def _value(engine, operation: str, formula: str) -> str:
    expr = engine.register_expression(formula)
    step, _ = engine.run_operation(operation, {"expr": expr.display_id})
    return engine.expressions[(step.output_refs or [None])[0]].canonical


class TestTheWholeFormulaReachesTheCas:
    def test_factoring_uses_every_term(self, maxima_session):
        """`x^2 - 1` の因数分解は `(x-1)(x+1)`——先頭項だけなら `x^2` のまま。"""
        out = _value(maxima_session, "factor", "x^2 - 1")
        assert "x" in out and out.strip() not in ("x^2", "x**2")
        assert "-1" in out.replace(" ", "") or ")" in out

    def test_simplify_keeps_every_term(self, maxima_session):
        out = _value(maxima_session, "simplify", "a*x^2 + b*x + c")
        for name in ("a", "b", "c"):
            assert name in out, f"{name} が消えた: {out}"


class TestTheBoundaryIsTheAliasNotTheFirstToken:
    r"""切り出しの側も値で確かめる——同じ問いを2層で測る。"""

    @pytest.mark.parametrize("text, want", [
        ("simplify a x^2 + b x + c", "a x^2 + b x + c"),
        ("factor x^2 - 1", "x^2 - 1"),
        ("expand (x + 1)^2", "(x + 1)^2"),
        ("trigsimp sin(x)^2 + cos(x)^2", "sin(x)^2 + cos(x)^2"),
    ])
    def test_the_formula_is_whole(self, text, want):
        from ui.interpret_pipeline import _en_verb_formula_re
        m = _en_verb_formula_re().match(text)
        assert m is not None and m.group(2) == want
