r"""代入の指示が式として登録される —— 十問目と同じ形が、動詞1つ違いで再発していた。

実測 2026-09-06（`E1 = a*x + b` に `x = 2`。正しくは `2a + b`）:

| 打った文 | 直す前 |
|---|---|
| `plug x = 2 into E1` | **registered** / canonical **`2*into(a*x+b)`** |
| `E1 with x = 2` | **registered** / canonical **`(a*x + b)*with*x = 2`** |
| `evaluate E1 at x = 2` | failed（安全だが読めない） |
| `E1 を x = 2 で評価して` | 同上 |
| `E1 の x に 2 を入れて` | 同上 |

**「この文は指示か」に2つの表が答えている。** 操作スキーマの en 別名と
`instruction_marker[en]`。実測: 別名の先頭語 **173種のうち151種**を marker が
知らない。`plug` は `transform_subst[en]` が既に知っていた（入力元が E-id の
ときだけ）のに、`substitute` の別名にも marker にも無かった。

**全面的に畳むと壊れる。** その151語のうち **14語**（`dot`・`to`・`integral`・
`assume`・`evolve`・`time`・`anticommutator`・`partial`・`spin`・`log`・
`lattice`・`order`・`rot`・`nearest`）は**式として登録される行に出る**ので、
marker に足すと式の登録が止まる。足せるのは「別名の先頭語であって、コーパスに
一度も出ない動詞」——`plug`/`insert`/`apply` は0件。

**`with` は語では足せない。** 式の行に6件出る（`the Ising model with 2 sites`
など）。ただし**それらは `=` を含まない**ので、`E<n> with <lhs> = <rhs>` という
**形**なら安全。語ではなく形で読む。

**測って「直さない」と決めたものが2つある**（どちらも既に正しい挙動だった）:
`substitute x = 2` は `expr` が要ると訊き、`substitute x = 2 into E1 and simplify`
は**2手順の計画**を出して確認を求める。
"""
import asyncio

import pytest

import ui.notebook_bridge as nb
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.nl_patterns import nlp
from notebook.runtime.operation_validator import OperationValidator
from ui.interpret_common import _is_nl_instruction
from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline

#: 代入に届くべき綴り（日英）。値は `2a + b`。
WORKS = [
    ("en", "substitute x = 2 into E1"),
    ("en", "substitute x = 2 in E1"),
    ("en", "substitute x with 2 in E1"),
    ("en", "substitute 2 for x in E1"),
    ("en", "replace x with 2 in E1"),
    ("en", "replace x by 2 in E1"),
    ("en", "set x to 2 in E1"),
    ("en", "let x = 2 in E1"),
    ("en", "plug x = 2 into E1"),
    ("en", "E1 with x = 2"),
    ("en", "evaluate E1 at x = 2"),
    ("ja", "E1 の x に 2 を代入して"),
    ("ja", "E1 に x = 2 を代入して"),
    ("ja", "E1 の x を 2 に置き換えて"),
    ("ja", "E1 の x を 2 で置き換えて"),
    ("ja", "x = 2 を E1 に代入して"),
    ("ja", "E1 で x = 2 とおいて"),
    ("ja", "E1 の x = 2 を代入して"),
    ("ja", "E1 の x に 2 を入れて"),
    ("ja", "E1 の x に 2 を代入"),
    ("ja", "E1 を x = 2 で評価して"),
]


def _ask(lang, text):
    nb.start_engine(title="subst", mode="standard", language=lang, domains=[])
    try:
        eng = nb.get_engine()
        sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                            domains=[], primary_backend="maxima")

        def run(x):
            return asyncio.get_event_loop().run_until_complete(
                run_interpret_pipeline(
                    InterpretRequest(tokens=[InputToken(type="text", value=x)]),
                    eng, sup))
        run("f = a*x + b")
        resp = run(text)
        val = None
        for e in eng.expressions.values():
            val = e.canonical
        return resp, val
    finally:
        nb.stop_engine()


class TestEverySpellingReachesSubstitution:
    @pytest.mark.parametrize("lang,text", WORKS)
    def test_substitutes(self, lang, text):
        resp, val = _ask(lang, text)
        assert resp.get("status") == "completed", (text, resp.get("status"))
        assert resp.get("operation") == "substitute", (text, resp.get("operation"))
        assert val == "b+2*a", (text, val)


class TestAnInstructionIsNotRegisteredAsAFormula:
    """十問目と同じ形。`plug`/`with` の2綴りで再発していた。"""

    @pytest.mark.parametrize("text", [
        "plug x = 2 into E1",
        "E1 with x = 2",
        "evaluate E1 at x = 2",
    ])
    def test_the_door_calls_it_an_instruction(self, text):
        assert _is_nl_instruction(text, "en"), text

    @pytest.mark.parametrize("text", ["plug x = 2 into E1", "E1 with x = 2"])
    def test_no_expression_carrying_the_verb_is_registered(self, text):
        _, val = _ask("en", text)
        assert "into" not in val and "with" not in val, (text, val)


class TestTheMarkerIsWidenedOnlyWhereItIsSafe:
    """**全面的に畳むと壊れる。** その境界を測定として固定する。"""

    @pytest.mark.parametrize("word", ["plug", "insert", "apply"])
    def test_a_word_absent_from_the_corpus_is_added(self, word):
        assert nlp("instruction_marker", "en").search(word), word

    @pytest.mark.parametrize("word", [
        "with", "dot", "to", "integral", "assume", "evolve", "time",
        "anticommutator", "partial", "spin", "log", "lattice", "order",
        "rot", "nearest",
    ])
    def test_a_word_that_appears_in_formula_lines_is_not_added(self, word):
        """これらを marker に足すと**式の登録が止まる**。実測でコーパスの
        式の行に出る語（`with` は6件、`lattice` は14件…）。"""
        assert not nlp("instruction_marker", "en").fullmatch(word), word

    def test_with_is_read_by_shape_not_by_word(self):
        """`with` は形でだけ読む。`=` を含まない `with` の文は指示にしない。"""
        assert nlp("expr_with_assignment", "en").match("E1 with x = 2")
        assert nlp("expr_with_assignment", "en").match(
            "Find the commutator of S^x_1 for the Ising model with 2 sites") is None

    @pytest.mark.parametrize("text", [
        "E1 with periodic boundary conditions",
        "E2 with 2 sites",
        "E1 with the same bonds",
    ])
    def test_the_equals_sign_is_what_makes_it_a_substitution(self, text):
        """**`=` を外す変異は最初どのテストも落とさなかった。** 測ると振る舞いは
        変わる——式番号で始まって `with` が続くだけの文（`E1 with periodic
        boundary conditions`）を lhs=`periodic`, rhs=`boundary conditions` の
        代入として claim する。コーパスの式の行は `Find`/`Expand` で始まるので
        当たらず、**空振りしていたのは負例が足りなかったから**だった。
        """
        assert nlp("expr_with_assignment", "en").match(text) is None, text

    def test_iresvverb_is_local_to_the_substitution_pattern(self):
        r"""`入れ` は `substitute_into[ja]` だけで受ける（共有表は `ja_subst_verb`、
        2026-09-12 に `op_subst_verb_ja` から改名）。

        **ここに書いてあった理由は実測で覆った。** 「一般的な動詞なので共有表に
        足すと横取りする」と書いていたが、入れて8綴り打っても**横取りは0件**で、
        変わるのは `E1 に E2 を入れて` が通るようになる点だけだった。局所に留めて
        いるのは**用心**——`入れ` はコーパス0行・シナリオの指示行0件で、6鍵が読む
        表へ移す判断には証拠が足りない。
        """
        from notebook.runtime.nl_patterns import fragment
        assert "入れ" not in fragment("ja_subst_verb")
        assert nlp("substitute_into", "ja").match("E1 の x に 2 を入れて")


class TestWhatWasMeasuredAndLeftAlone:
    """**直さないと決めたもの**——どちらも既に正しい挙動だった。"""

    def test_a_missing_target_is_asked_for(self):
        resp, _ = _ask("en", "substitute x = 2")
        assert resp.get("status") == "failed"
        assert "expr" in str(resp.get("error")), resp.get("error")

    def test_a_chained_request_is_shown_as_a_plan(self):
        resp, _ = _ask("en", "substitute x = 2 into E1 and simplify")
        assert resp.get("status") == "needs_confirmation"
        prompt = str(resp.get("confirmation_prompt") or "")
        assert "substitute" in prompt and "simplify" in prompt, prompt
