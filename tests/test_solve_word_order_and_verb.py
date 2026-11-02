r"""SOLVE 一族——**枠は語順で決まり、動詞は1つの表**（NL-GRAMMAR ①、2026-09-12）。

測定器②（片側にしか無い断片）が `ja_solve_verb`（7鍵が読む）に対する英語側の欠落を
指したので測った。**英語の語彙は割れていなかった**——`solve` は5鍵に散っていたが、
どの枠でも `solve` が唯一の自然な動詞で、`find E1 for x` は英語ではない。実測すると
**枠ごとの語彙は日英で対称**だった:

    解く形（`solve E1 for x` / `E1 を x について解いて`）      → solve / 解[いけくき] のみ
    求める形（`find x from E1` / `E1 から x を求めて`）        → compute 動詞

割れていたのは**語順と、枠ごとの取りこぼし**である。

**silent-wrong が1件。** 変数が先に来る語順に**英語だけ鍵が無く**、別名走査が答えて
いた——`solve for x and y in E1 and E2` は**1本目だけを x について解いて** `3-y` を
`completed` で返す（正解は x=2, y=1）。日本語の同じ語順は声を出して落ちるので、
**日英の割れでもある**（`x と y について E1 と E2 を解いて` → 「解釈できません」）。
単一の逆順は日本語だけが枝を持っていた（`x について E1 を解いて`）。

ほかに4件、どれも「兄弟の鍵は知っているのにこの鍵は知らない」形:

- `E1 と E2 を解いて` が落ちる——`連立` を必須にしていた。**兄弟の
  `solve_system_vars[ja]` は既に任意で、その註が「英語は最初からその語を要求して
  いない」と書いている**（2026-09-05）。同じ直しがもう片方に届いていなかった。
- `solve the equations E1 and E2` が落ちる——名詞句（`the equations`）は変数ありの
  枝だけが許していた。
- `E1 から x を解いて`／`解け`／`解き` が落ちる——`solve_from_expr[ja]` は
  `compute_verb_ja` だけを読み、**解く動詞を読んでいなかった**（英語側の
  `solve x from E1` は別名走査で通っていたので、ここも日英の割れ）。
- `find the solution to E1` が落ちる——`of` だけを読んでいた。

**英語の動詞も `en_solve_verb` に畳んだ**。語彙は割れていなかったので**予防**で
あって欠陥の修理ではない——ただし新しい枠を足すときに綴りを1つ増やせるようになる。

コーパス差分 0/1130。ODE の枠（`find the general solution of E1` など）は測ったが
**この変更の範囲外**——落ちるのは読みではなく操作で、「従属変数を特定できません」
と言っている（下拵えの式が微分方程式として解けないため）。
"""
import asyncio
import json
import pathlib

import pytest

import ui.notebook_bridge as nb
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.nl_patterns import fragment
from notebook.runtime.operation_validator import OperationValidator
from ui.interpret_pipeline import (InputToken, InterpretRequest,
                                   run_interpret_pipeline)

_PATTERNS = json.loads(
    (pathlib.Path(__file__).resolve().parents[1] / "i18n" / "nl_patterns.json")
    .read_text(encoding="utf-8"))

_SYS = ["x + y = 3", "x - y = 1"]
_ONE = ["a x + b = 0"]


def _ask(lang, setup, text):
    async def go():
        nb.start_engine(title="solve-order", mode="standard", language=lang,
                        domains=["classical_mechanics"])
        try:
            engine = nb.get_engine()
            sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                                domains=["classical_mechanics"],
                                primary_backend=engine.session.primary_backend)
            for line in setup:
                await run_interpret_pipeline(
                    InterpretRequest(tokens=[InputToken(type="text", value=line)]),
                    engine, sup)
            before = set(engine.expressions)
            resp = await run_interpret_pipeline(
                InterpretRequest(tokens=[InputToken(type="text", value=text)]),
                engine, sup)
            fresh = {k: (engine.expressions[k].latex or engine.expressions[k].canonical)
                     for k in engine.expressions if k not in before}
            return resp, fresh
        finally:
            nb.stop_engine()
    return asyncio.run(go())


class TestTheVariablesMayComeFirst:
    r"""**畳む前、英語のこの語順には鍵が無く、別名走査が1本目だけ解いていた。**"""

    @pytest.mark.parametrize("lang,text", [
        ("en", "solve for x and y in E1 and E2"),
        ("en", "solve E1 and E2 for x and y"),
        ("ja", "x と y について E1 と E2 を解いて"),
        ("ja", "E1 と E2 を x と y について解いて"),
    ])
    def test_a_system_is_solved_as_a_system(self, lang, text):
        resp, fresh = _ask(lang, _SYS, text)
        assert resp.get("operation") == "solve_system", (text, resp)
        # **操作名だけでは足りない**——1本目だけ解いても `solve_equation` ではなく
        # `completed` が返っていたので、値まで見る。
        assert sorted(fresh.values()) == ["x = 2", "y = 1"], (text, fresh)

    @pytest.mark.parametrize("lang,text", [
        ("en", "solve for x in E1"), ("en", "solve E1 for x"),
        ("ja", "x について E1 を解いて"), ("ja", "E1 を x について解いて"),
    ])
    def test_a_single_equation_still_takes_both_orders(self, lang, text):
        resp, _ = _ask(lang, _ONE, text)
        assert resp.get("operation") == "solve_equation", (text, resp)
        assert resp.get("status") == "completed", (text, resp)


class TestALatexVariableIsNotAnsweredAsZero:
    r"""**変数先行の語順の2つ目の silent-wrong。** 鍵が無く別名走査が答えていたので、
    LaTeX の変数が取りこぼされて**もっともらしい誤答**が `completed` で返っていた
    （実測 2026-09-12、E1 = `\theta + a = 0`）::

        solve for \theta in E1   → `\theta = 0` を登録（正解は `\theta = -a`）
        （θ を含まない式に対して）  → `\vartheta = 0` を登録——**研究者が書いていない記号**

    生き残った変異が名指したのはこれで、**テストが区別していなかった**だけだった
    （`solve for x in E1` は素の識別子なので、枝が無くても別名走査が正しく答える）。
    """

    def test_a_latex_variable_is_read(self):
        resp, fresh = _ask("en", ["\\theta + a = 0"], "solve for \\theta in E1")
        assert resp.get("operation") == "solve_equation", resp
        assert list(fresh.values()) == ["\\theta = -a"], fresh

    def test_a_variable_absent_from_the_equation_is_not_answered(self):
        resp, fresh = _ask("en", ["a x + b = 0"], "solve for \\theta in E1")
        assert resp.get("status") != "completed", (resp, fresh)
        assert fresh == {}, fresh


class TestTheSiblingKeyKnewItAlready:
    r"""どれも「兄弟の鍵は知っているのにこの鍵は知らない」形。"""

    @pytest.mark.parametrize("lang,setup,text,want", [
        # `連立` は兄弟鍵では既に任意だった
        ("ja", _SYS, "E1 と E2 を解いて", "solve_system"),
        ("ja", _SYS, "E1 と E2 の連立方程式を解いて", "solve_system"),
        # 名詞句は変数ありの枝だけが許していた
        ("en", _SYS, "solve the equations E1 and E2", "solve_system"),
        ("en", _SYS, "solve E1 and E2", "solve_system"),
        # `solve_from_expr[ja]` が解く動詞を読んでいなかった
        ("ja", _ONE, "E1 から x を解いて", "solve_equation"),
        ("ja", _ONE, "E1 から x を解け", "solve_equation"),
        ("ja", _ONE, "E1 から x を求めて", "solve_equation"),
        ("en", _ONE, "find x from E1", "solve_equation"),
        # `the solution of` しか読んでいなかった
        ("en", _ONE, "find the solution to E1", "solve_equation"),
        ("en", _ONE, "find the solution of E1", "solve_equation"),
        ("en", _ONE, "find the solution to E1 for x", "solve_equation"),
        ("en", _ONE, "find the solution of E1 for x", "solve_equation"),
    ])
    def test_the_spelling_reaches_the_operation(self, lang, setup, text, want):
        resp, _ = _ask(lang, setup, text)
        assert resp.get("operation") == want, (text, resp)


class TestTheVerbIsOneTablePerLanguage:

    def test_both_languages_have_a_solve_verb_fragment(self):
        r"""測定器②が指したのはこれ——日本語にだけ断片が在った。"""
        assert fragment("ja_solve_verb") == "解[いけくき]"
        assert fragment("en_solve_verb") == "solve"

    @pytest.mark.parametrize("key", ["solve_system_vars", "solve_system", "solve_single"])
    def test_the_english_keys_read_it(self, key):
        assert "{{en_solve_verb}}" in _PATTERNS[key]["en"], key

    @pytest.mark.parametrize("key", ["solve_system_vars", "solve_system",
                                     "solve_single", "solve_single_novar",
                                     "solve_from_expr"])
    def test_the_japanese_keys_read_it(self, key):
        assert "{{ja_solve_verb}}" in _PATTERNS[key]["ja"], key


class TestACompoundSentenceIsNotSwallowed:
    r"""`solve_from_expr[ja]` の語尾は活用だけを受ける——最初 `\w*` と書いたら
    複文の後半まで飲んだ（行末に錨があるので、飲めば1つの要求として通ってしまう）。"""

    def test_the_tail_takes_inflections_only(self):
        from notebook.runtime.nl_patterns import nlp
        assert nlp("solve_from_expr", "ja").match("E1 から x を解いて")
        assert nlp("solve_from_expr", "ja").match("E1 から x を解きます")
        assert nlp("solve_from_expr", "ja").match("E1 から x を解いてから展開して") is None
