r"""「何について解くか」—— 読みと guard で綴りが割れていた（二十二問目と同じ形）。

実測 2026-09-06（`E1: x^2 - 4 = 0`。x について解くと ±2）。

**silent-wrong 1件、ただし性質が悪い:**

    solve the equation E1 for x
      → 別名走査が `equation` を変数に取り、Maxima が解けず
        **「解が見つかりません。与えられた変数について、この方程式（または
        連立方程式）に解はありません」** と返す。
      —— **研究者の式について偽を述べる。**

**読めなかった綴り（安全な失敗だが実在する言い方）:**
`E1 を x について解きなさい` / `E1 の解を x について求めて` / `E1 から x を求めて` /
`x について E1 を解いて` / `find x from E1` / `determine x from E1` /
`find the solution of E1 for x`。

最後のものは特に、**変数なしの同じ言い方（`find the solution of E1`）は通って
いた**——1つの鍵が片方の形だけ知っていた。

**同じ問いに答えている場所**（畳んだ）:
- `解[いけく]` が `solve_single` / `solve_single_novar` / `solve_system` /
  `solve_system_vars` / `ode_general` / **guard `ja_op_var_tail`** に6回。
  1つの fragment `ja_solve_verb` にして `き`（`解きなさい`）を足した。
- 「何について」の助詞が読みと guard に二重。`ja_solve_var_particle` に。
- `solve_single[en]` は `for` だけを自前で綴っていた → `{{en_op_var_head}}`。

**読みが受ける綴りは guard も知る。** 新しく読めるようにした4綴りは、最初
guard が知らず `x` が E-id に書き換わっていた（二十二問目で測った穴が solve 一族
にもあった）。**尾ではなく前置きの形で書く**——`を` ＋ 求める動詞だけで絞ると
`Heisenberg 方程式 H を計算して` の `H` まで守り、コーパス3行が変わる（実測）。

**`で` は入れない。** 「x で解く」の `で` は**手段**を指す（「代入法で解く」
「ニュートン法で解く」）。「について」「に関して」が曖昧さのない標準の言い方な
ので、`で` を変数の助詞にすると手段の文を変数の指定と読むことになる
（`に対して` を入れなかったのと同じ判断）。コーパスにも0件。
"""
import asyncio
import types

import pytest

import ui.notebook_bridge as nb
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.nl_patterns import fragment
from notebook.runtime.operation_validator import OperationValidator
from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline
from ui.supervisor_alias_rewrite import rewrite_symbols_to_eids

SOLVES = [
    ("ja", "E1 を x について解いて"),
    ("ja", "E1 を x に関して解いて"),
    ("ja", "E1 を x について解け"),
    ("ja", "E1 を x について解きなさい"),
    ("ja", "E1 を x について解いてください"),
    ("ja", "E1 の解を x について求めて"),
    ("ja", "E1 から x を求めて"),
    ("ja", "x について E1 を解いて"),
    ("en", "solve E1 for x"),
    ("en", "solve E1 with respect to x"),
    ("en", "solve E1 w.r.t. x"),
    ("en", "solve E1 in x"),
    ("en", "solve for x in E1"),
    ("en", "solve the equation E1 for x"),
    ("en", "find x from E1"),
    ("en", "determine x from E1"),
    ("en", "find the solution of E1 for x"),
]
ASK_FOR_VAR = [("ja", "E1 を解いて"), ("ja", "E1 を解け"), ("ja", "E1 の解を求めて"),
               ("en", "solve E1"), ("en", "find the solution of E1")]


def _ask(lang, text):
    nb.start_engine(title="solve", mode="standard", language=lang, domains=[])
    try:
        eng = nb.get_engine()
        sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                            domains=[], primary_backend="maxima")

        def run(x):
            return asyncio.get_event_loop().run_until_complete(
                run_interpret_pipeline(
                    InterpretRequest(tokens=[InputToken(type="text", value=x)]),
                    eng, sup))
        run("x^2 - 4 = 0")
        resp = run(text)
        vals = [e.canonical for e in eng.expressions.values()]
        return resp, vals
    finally:
        nb.stop_engine()


class TestEverySpellingSolvesForTheWrittenVariable:
    @pytest.mark.parametrize("lang,text", SOLVES)
    def test_solves(self, lang, text):
        resp, vals = _ask(lang, text)
        assert resp.get("status") == "completed", (text, resp.get("status"),
                                                   resp.get("error"))
        assert any("2" in str(v) for v in vals[1:]), (text, vals)

    @pytest.mark.parametrize("lang,text", ASK_FOR_VAR)
    def test_a_missing_variable_is_asked_for(self, lang, text):
        """**変えてはいけない挙動。** 変数を書かなければ訊く。"""
        resp, _ = _ask(lang, text)
        assert resp.get("status") == "needs_confirmation", (text, resp)
        prompt = str(resp.get("confirmation_prompt") or "")
        assert ("変数" in prompt) if lang == "ja" else ("variable" in prompt), prompt


class TestTheRefusalNeverLiesAboutTheEquation:
    def test_a_filler_noun_is_not_taken_as_the_variable(self):
        """直す前は `equation` を変数に取り、**「この方程式に解はありません」**と
        研究者の式について偽を述べていた。"""
        resp, vals = _ask("en", "solve the equation E1 for x")
        assert resp.get("status") == "completed", resp.get("error")
        assert "No solution" not in str(resp.get("error") or "")


class TestTheGuardKnowsEverySpellingTheReadingAccepts:
    """二十二問目で測った穴が solve 一族にもあった。"""

    @staticmethod
    def _rw(text):
        eng = types.SimpleNamespace(lhs_alias_map={"x": "E9"},
                                    expressions={"E9": None})
        return rewrite_symbols_to_eids(text, eng)

    @pytest.mark.parametrize("text", [t for _, t in SOLVES])
    def test_a_solve_variable_is_never_rewritten(self, text):
        assert "E9" not in self._rw(text), text

    @pytest.mark.parametrize("text", [
        "E1 の両辺を x で割る",
        "E1 を x で置き換える",
        "multiply both sides of E1 by x",
        "Heisenberg 方程式 x を計算して",
    ])
    def test_another_operation_s_operand_is_still_rewritten(self, text):
        """**尾だけで絞ってはいけない。** `を` ＋ 求める動詞だけで守ると
        `Heisenberg 方程式 H を計算して` の `H` まで守り、コーパス3行が変わる
        （実測 2026-09-06）。前置きの形（`E<n> から`）で書く。"""
        assert "E9" in self._rw(text), text


class TestTheVocabularyLivesInOnePlace:
    def test_the_solve_verb_is_one_fragment(self):
        import json
        import pathlib
        d = json.loads(pathlib.Path("i18n/nl_patterns.json").read_text(
            encoding="utf-8"))
        for key in ("solve_single", "solve_single_novar", "solve_system",
                    "solve_system_vars", "ode_general"):
            assert "{{ja_solve_verb}}" in d[key]["ja"], key
            assert "解[いけく" not in d[key]["ja"], (key, "動詞をまた直書きしている")
        assert "{{ja_solve_verb}}" in d["_fragments"]["ja_op_var_tail"]

    @pytest.mark.parametrize("form", ["解い", "解け", "解く", "解き"])
    def test_every_verb_form_is_in_it(self, form):
        import re
        assert re.fullmatch(fragment("ja_solve_verb"), form), form

    def test_no_solve_key_spells_its_own_head(self):
        """**表の名前が 2026-09-19 に変わった**（五十四問目）。共有だったのは
        `en_op_var_head` だが、`solve_single` だけが `in` を**直書きで足して**おり、
        兄弟2鍵では `compute the solution of E1 in x` が落ち、
        `solve E1 and E2 in x and y` は別名走査が答えて `x = 3-y` を `completed` で
        返していた。solve 一族だけの表 `en_solve_var_head`（`en_op_var_head` ＋ `in`）を
        作り、3鍵がそれを読む——`in` を共有表へ入れると 2a-sym が両辺の因子まで守る。

        だから断言も「1鍵が共有表を読む」から**「どの鍵も自前で綴らない」**へ広げる。
        """
        import json
        import pathlib
        d = json.loads(pathlib.Path("i18n/nl_patterns.json").read_text(
            encoding="utf-8"))
        for key in ("solve_single", "solve_solution_of_var", "solve_system_vars"):
            assert "{{en_solve_var_head}}" in d[key]["en"], key
            assert "{{en_op_var_head}}" not in d[key]["en"], (key, "自前で綴っている")
        assert "{{en_op_var_head}}" in d["_fragments"]["en_solve_var_head"]


class TestWhatWasDeliberatelyLeftOut:
    @pytest.mark.parametrize("text", ["E1 を 代入法 で解く", "E1 を数値的に解く",
                                      "E1 を ニュートン法 で解く"])
    def test_a_method_is_never_read_as_the_variable(self, text):
        """「x で解く」の `で` は**手段**を指す（「代入法で解く」）。曖昧さのない
        「について」「に関して」があるので `で` は入れない——入れると手段の文が
        変数の指定として読まれる。いまは安全に失敗する（実測 2026-09-06）。

        **`E1 を x で解く` が通るのは別名走査の副作用**で、`E1 を x で解いて` は
        失敗する（同じ `で` で答えが割れている）。それを揃えるために `で` を
        助詞にするのは**手段の文を巻き込む**ので採らない——コーパスにも0件。
        """
        resp, _ = _ask("ja", text)
        assert resp.get("status") != "completed", (text, resp)

    def test_de_is_not_in_the_particle_fragment(self):
        assert fragment("ja_solve_var_particle") == "(?:について|に関して)"
