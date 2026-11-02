r"""展開した和は、書き下した形と**同じ値**でなければならない（live CAS の差分検査）。

文字列の比較では捕まらない族がある。実測 2026-08-22——展開を入れた当のコミットで、
`Σ` の展開を囲い忘れたために

    U \sum_{i=1}^{2} f_i   →   U * f(1) + f(2)

となり、**係数が先頭の項にしか届いていなかった**。そのとき見た事例
（`[U\sum_i \hat n_{i↑}\hat n_{i↓}, \hat c_{1↑}]`）は $i=1$ の項しか拾わないので
答えは正しく、**1件の緑が検査の役をしなかった**。

だからこの検査は $i=2$ 側に効く量を見る: 係数が2項目に届いていなければ
交換子が 0 になり、差が 0 にならない。
"""
import pytest

from notebook.engine import NotebookEngine
from notebook.models.session import SessionMode


@pytest.fixture(scope="module")
def qa_session():
    """ONE Julia process for the module (起動が ~1s かかるため)。"""
    with NotebookEngine(title="expand_diff", mode=SessionMode.second_quantization) as engine:
        engine.start()
        engine.run_operation("define_fermion_operator", {"name": "c"})
        engine.run_operation("define_scalar_parameter", {"name": "U"})
        yield engine


def _register(engine, latex: str) -> str:
    return engine.register_expression(
        engine._handler.latex_to_canonical(latex), latex=latex).display_id


def _value(engine, operation: str, args: dict) -> str:
    step, _ = engine.run_operation(operation, args)
    return engine.expressions[(step.output_refs or [None])[0]].latex


_SUMMED = (r"U \sum_{i=1}^{2} c^\dagger_{i\uparrow} c_{i\uparrow} "
           r"c^\dagger_{i\downarrow} c_{i\downarrow}")
_WRITTEN = (r"U c^\dagger_{1\uparrow} c_{1\uparrow} c^\dagger_{1\downarrow} c_{1\downarrow} "
            r"+ U c^\dagger_{2\uparrow} c_{2\uparrow} c^\dagger_{2\downarrow} c_{2\downarrow}")


class TestTheCoefficientReachesEveryTerm:
    def test_the_difference_is_zero(self, qa_session):
        a = _register(qa_session, _SUMMED)
        b = _register(qa_session, _WRITTEN)
        out = _value(qa_session, "combine_expressions",
                     {"expr_a": a, "expr_b": b, "op": "-"})
        assert out.strip().endswith("0"), out

    def test_the_second_term_carries_the_coefficient(self, qa_session):
        r"""$i=2$ 側に効く交換子。囲い忘れていると U が届かず 0 になる。"""
        a = _register(qa_session, _SUMMED)
        out = _value(qa_session, "commutator", {"a": a, "b": r"c_{2\uparrow}"})
        assert "U" in out, out
        assert out.strip() not in ("0", "0.0"), out
