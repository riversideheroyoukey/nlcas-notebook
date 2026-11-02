"""solve 一族 —— 日本語側が英語より狭かった（① 九問目、2026-09-05）。

機械的な worklist の次は `解い`（コーパス15行・4キー）。実パイプラインで綴りの空間を
測ると、**落ちていたのは全部日本語**だった:

    E1 を x について解け / 解く      辞書形・命令形。て形の `解い` しか読まない
    E1 を x に関して解いて           `diff_rule` は `に関する` を読むのに solve は読まない
    E2 と E3 を x と y について解いて  `連立` の語を必須にしていた（英語は要求しない）
    E1 を解いて / E1 の解を求めて     変数を書かないとどの規則も取らない

最後のひとつは**同じ問いに3つの答え**だった: 連立は変数なしで確認ダイアログに行き
（正しい）、英語の単一は別名走査で操作に届いて「引数が足りません」と言い、日本語の
単一だけが「解釈できません」で落ちていた。確認の分岐が `_h9_op == "solve_system"` と
決め打ちだったのが理由で、違うのはフィールド名だけである。

**そして確認ダイアログの「答え」の側も狭かった。** `^[A-Za-z]$` の1文字で、
それ以外は失敗ではなく**確認そのものが捨てられ、答えが新しい式として登録**されていた
（`\\theta` → status=registered）。単一方程式が確認に行くようになった以上ここを通る人が
増えるので、答えの読み方も `{{symbol}}` に揃えた（`tests/test_nl_symbol_fragment.py`
の表に入っている）。
"""
import asyncio

import pytest

import ui.notebook_bridge as nb
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operation_validator import OperationValidator
from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline


def _run(eng, sup, text):
    return asyncio.get_event_loop().run_until_complete(
        run_interpret_pipeline(
            InterpretRequest(tokens=[InputToken(type="text", value=text)]), eng, sup))


def _session(lang):
    nb.start_engine(title="solve_spell", mode="standard", language=lang,
                    domains=["classical_mechanics"])
    eng = nb.get_engine()
    sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                        domains=["classical_mechanics"], primary_backend="maxima")
    _run(eng, sup, "a, b をスカラーとする" if lang == "ja" else "Define a, b as scalars")
    _run(eng, sup, "a*x + b = 0")      # E1
    _run(eng, sup, "x + y = 3")        # E2
    _run(eng, sup, "x - y = 1")        # E3
    return eng, sup


#: 変数を名指す綴り。日本語は活用と助詞、英語は語順。
NAMED = [
    ("ja", "E1 を x について解いて", "solve_equation", "x"),
    ("ja", "E1 を x について解け",   "solve_equation", "x"),
    ("ja", "E1 を x について解く",   "solve_equation", "x"),
    ("ja", "E1 を x に関して解いて", "solve_equation", "x"),
    ("en", "solve E1 for x",        "solve_equation", "x"),
    ("en", "Solve E1 for x",        "solve_equation", "x"),
    ("ja", "E2 と E3 の連立方程式を x と y について解いて", "solve_system", "x, y"),
    ("ja", "E2 と E3 を x と y について解いて",             "solve_system", "x, y"),
    ("ja", "E2, E3 の連立方程式を x, y について解いて",       "solve_system", "x, y"),
    ("en", "solve E2 and E3 for x and y",                 "solve_system", "x, y"),
    ("en", "solve E2, E3 for x, y",                       "solve_system", "x, y"),
]

#: 変数を名指さない綴り。**訊く**のが正しい答えで、三者が揃っている必要がある。
UNNAMED = [
    ("ja", "E1 を解いて"),
    ("ja", "E1 の解を求めて"),
    ("ja", "E2 と E3 の連立方程式を解いて"),
    ("en", "solve E1"),
    ("en", "find the solution of E1"),
    ("en", "solve E2 and E3"),
]


class TestEverySpellingReachesTheOperation:
    @pytest.mark.parametrize("lang,text,operation,var", NAMED)
    def test_named_variable(self, lang, text, operation, var):
        eng, sup = _session(lang)
        try:
            r = _run(eng, sup, text)
            assert r["status"] == "completed", r
        finally:
            nb.stop_engine()


class TestAnUnnamedVariableIsAskedFor:
    """The same question, and it must have ONE answer: ask.  Before this it had
    three — the system asked, English's single said which argument was missing,
    and Japanese's single said it could not interpret the sentence."""

    @pytest.mark.parametrize("lang,text", UNNAMED)
    def test_the_researcher_is_asked(self, lang, text):
        eng, sup = _session(lang)
        try:
            r = _run(eng, sup, text)
            assert r["status"] == "needs_confirmation", r
            assert r.get("confirmation_prompt"), r
        finally:
            nb.stop_engine()


class TestTheAnswerToThePromptIsAVariableName:
    """…and a variable name is not one Latin letter."""

    @pytest.mark.parametrize("lang", ["ja", "en"])
    @pytest.mark.parametrize("var", ["x", r"\theta", "x_1", "omega_D"])
    def test_answering_solves_it(self, lang, var):
        nb.start_engine(title="answer", mode="standard", language=lang,
                        domains=["classical_mechanics"])
        eng = nb.get_engine()
        sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                            domains=["classical_mechanics"], primary_backend="maxima")
        try:
            _run(eng, sup, "a, b をスカラーとする" if lang == "ja"
                 else "Define a, b as scalars")
            _run(eng, sup, f"a*{var} + b = 0")
            asked = _run(eng, sup, "E1 を解いて" if lang == "ja" else "solve E1")
            assert asked["status"] == "needs_confirmation", asked
            done = _run(eng, sup, var)
            assert done["status"] == "completed", done
            assert done["canonical"].replace(" ", "") == "-(b/a)", done
        finally:
            nb.stop_engine()

    @pytest.mark.parametrize("lang", ["ja", "en"])
    def test_the_answer_is_never_registered_as_an_expression(self, lang):
        """The failure mode was not a refusal: the pending question was dropped
        and the answer became a new expression."""
        nb.start_engine(title="answer2", mode="standard", language=lang,
                        domains=["classical_mechanics"])
        eng = nb.get_engine()
        sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                            domains=["classical_mechanics"], primary_backend="maxima")
        try:
            _run(eng, sup, "a, b をスカラーとする" if lang == "ja"
                 else "Define a, b as scalars")
            _run(eng, sup, r"a*\theta + b = 0")
            _run(eng, sup, "E1 を解いて" if lang == "ja" else "solve E1")
            r = _run(eng, sup, r"\theta")
            assert r["status"] != "registered", r
        finally:
            nb.stop_engine()
