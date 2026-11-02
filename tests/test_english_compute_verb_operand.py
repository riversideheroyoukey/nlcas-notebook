r"""MATRIX-INLINE-OPERAND-EN（2026-09-11）。

**その場に書いた行列は、計算動詞で始めても対象である。**

英語の「動詞＋式」を切る規則（`_en_verb_formula_re`）は**文頭に操作の別名**を
要求する。`compute the determinant of [4, 0; 0, 1]` は `compute` で始まるので
当たらず、書いた行列は**登録されないまま**「式を名指していない文」に見え、
直前の式に落ちていた。実測 2026-09-11、`A = [2,1;1,2]` が在るセッションで
（**綴りごとに新しいセッション**で測る——落下先は直前の行に汚染される）:

| 打った文 | 正解 | 前 |
|---|---|---|
| `compute the determinant of [4, 0; 0, 1]` | 4 | **3**（A の行列式） |
| `find the trace of [4, 0; 0, 1]` | 5 | **4**（A のトレース） |
| `compute the inverse matrix of [4, 0; 0, 1]` | `[1/4,0;0,1]` | **A の逆行列** |
| `compute the transpose of [4, 0; 0, 1]` | `[4,0;0,1]` | **A の転置** |
| `compute the eigenvalues of [4, 0; 0, 1]` | 1, 4 | **A の固有値 3, 1** |

**どれも `completed`。** 日本語は全部正しく（`<式> の行列式を求めて` は**式で
始まる**ので先頭式登録が拾う）、英語の「動詞＋末尾の式」自体も
`expand`/`simplify`/`factor`/`transpose` では動いていた——落ちていたのは
**文頭に計算動詞が立つ綴りだけ**である。

直し方は、その語彙が既に1箇所に在ることを使う（`compute_verb_en`、
`operand_boundary.split_leading_compute_verb`）: 飲んでから判定し、**組み直す
ときに戻す**——落とすと、計算動詞を要求する `matrix_trigger` が今度は当たらない。
"""
import asyncio

import pytest

import ui.notebook_bridge as nb
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operand_boundary import (skip_leading_compute_verb,
                                               split_leading_compute_verb)
from notebook.runtime.operation_validator import OperationValidator
from ui.interpret_pipeline import (InputToken, InterpretRequest,
                                   run_interpret_pipeline)


class TestTheVerbIsSplitOffNotDropped:

    @pytest.mark.parametrize("text,prefix,rest", [
        ("compute the determinant of [4, 0; 0, 1]", "compute the ",
         "determinant of [4, 0; 0, 1]"),
        ("Find the trace of E1", "Find the ", "trace of E1"),
        ("please calculate the norm of v", "please calculate the ", "norm of v"),
        ("take the total derivative of E1", "take the ", "total derivative of E1"),
        ("determinant of [4, 0; 0, 1]", "", "determinant of [4, 0; 0, 1]"),
        ("simplify E1", "", "simplify E1"),
    ])
    def test_split(self, text, prefix, rest):
        assert split_leading_compute_verb(text) == (prefix, rest)

    @pytest.mark.parametrize("text,expected", [
        ("compute [A, B]", "[A, B]"),
        ("[A, B]", "[A, B]"),
        ("compute the determinant of [4, 0; 0, 1]", None),
    ])
    def test_the_bracket_reader_is_unchanged(self, text, expected):
        """括弧の読みは**括弧で始まる残り**だけを返す——切り出しを分けても同じ."""
        assert skip_leading_compute_verb(text) == expected


# ---------------------------------------------------------------------------
# 生きた経路。**綴りごとに新しいセッション**で測る。
# ---------------------------------------------------------------------------

#: おとりの行列（det 3 / trace 4）。これが在るからこそ、落下は失敗ではなく
#: **別の答え**として現れる——おとり無しで測ると欠陥が見えない。
_DECOY = {
    "ja": r"A = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}",
    "en": r"A = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}",
}
#: 書く行列（det 4 / trace 5 / inverse [1/4,0;0,1]）。
_M = "[4, 0; 0, 1]"


def _ask(lang, text):
    async def go():
        nb.start_engine(title="compute-verb", mode="standard", language=lang,
                        domains=["physics"])
        try:
            engine = nb.get_engine()
            sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                                domains=["physics"],
                                primary_backend=engine.session.primary_backend)
            await run_interpret_pipeline(
                InterpretRequest(tokens=[InputToken(type="text",
                                                    value=_DECOY[lang])]),
                engine, sup)
            return await run_interpret_pipeline(
                InterpretRequest(tokens=[InputToken(type="text", value=text)]),
                engine, sup)
        finally:
            nb.stop_engine()
    return asyncio.run(go())


class TestTheMatrixWrittenInTheSentenceIsTheOperand:

    @pytest.mark.parametrize("lang,text,want,decoy", [
        # 英語——計算動詞つき（これが壊れていた綴り）
        ("en", f"compute the determinant of {_M}", "4", "3"),
        ("en", f"find the determinant of {_M}", "4", "3"),
        ("en", f"calculate the determinant of {_M}", "4", "3"),
        ("en", f"compute the trace of {_M}", "5", "4"),
        ("en", f"find the trace of {_M}", "5", "4"),
        # 英語——計算動詞なし（元から通っていた綴り。壊していないことを見る）
        ("en", f"determinant of {_M}", "4", "3"),
        ("en", f"trace of {_M}", "5", "4"),
        # 日本語——同じ問いに同じ答え
        ("ja", f"{_M} の行列式を求めて", "4", "3"),
        ("ja", f"{_M} のトレースを求めて", "5", "4"),
    ])
    def test_a_scalar_answer(self, lang, text, want, decoy):
        r = _ask(lang, text)
        assert r.get("status") == "completed", r
        got = (r.get("canonical") or "").strip()
        assert got == want, f"{got!r} (おとりの答えは {decoy!r}): {r}"

    @pytest.mark.parametrize("lang,text,want,decoy", [
        ("en", f"compute the inverse matrix of {_M}", "1/4", "2/3"),
        ("en", f"compute the transpose of {_M}", "[4,0]", "[2,1]"),
        ("en", f"compute the eigenvalues of {_M}", "[[1,4]", "[[3,1]"),
        ("ja", f"{_M} の逆行列を求めて", "1/4", "2/3"),
        ("ja", f"{_M} を転置して", "[4,0]", "[2,1]"),
    ])
    def test_a_matrix_answer(self, lang, text, want, decoy):
        r = _ask(lang, text)
        assert r.get("status") == "completed", r
        got = (r.get("canonical") or "").replace(" ", "")
        assert want in got, f"{got!r} (おとりの答えは {decoy!r}): {r}"


class TestWhatMustNotChange:
    r"""飲んだ動詞を**戻す**ので、規則が要求する語はそのまま残る."""

    @pytest.mark.parametrize("lang,text,want", [
        ("en", "expand (x+1)^2", "x^2+2*x+1"),
        ("en", "simplify (x^2-1)/(x-1)", "x+1"),
        ("en", "factor x^2 - 1", "(x-1)*(x+1)"),
        ("ja", "(x+1)^2 を展開して", "x^2+2*x+1"),
    ])
    def test_the_spellings_that_already_worked(self, lang, text, want):
        r = _ask(lang, text)
        assert r.get("status") == "completed", r
        assert (r.get("canonical") or "").strip() == want, r

    @pytest.mark.parametrize("lang,text", [
        ("en", "compute the determinant of E1"),
        ("en", "find the trace of E1"),
    ])
    def test_a_sentence_naming_an_expression_is_not_re_registered(self, lang, text):
        r"""式参照が混じっていれば、それは参照の指示——式として登録しない
        （`_formula_runs_to_the_end` の3つ目の条件）."""
        r = _ask(lang, text)
        assert r.get("status") == "completed", r
        assert (r.get("canonical") or "").strip() in ("3", "4"), r   # おとり A の答え


class TestTheSentenceStaysOnItsOwnRoute:
    r"""**飲んだ動詞を戻すのは、答えではなく規則のため**（実測 2026-09-11）。

    落としても値は同じだった（19綴り）。変わるのは**どの規則が答えるか**:
    `matrix_trigger` は計算動詞を要求するので、`determinant E2` に縮めると
    行列のインターセプタが降り、別名走査が答える。そのインターセプタは
    「LLM を通さずに行列の族を回す」ために在り、固有値と特性多項式の区別の
    ような**そこにしかない知識**を持っている——値が今たまたま一致することは、
    族全体で一致することの保証にならない。
    """

    @pytest.mark.parametrize("text", [
        f"compute the determinant of {_M}",
        f"find the trace of {_M}",
        f"compute the eigenvalues of {_M}",
    ])
    def test_the_matrix_interceptor_is_the_one_that_answers(self, text):
        import ui.interpret_pipeline as P
        seen: list = []
        original = P._try_matrix_ops

        async def spy(engine, loop, supervisor_text, registered, lang="ja"):
            r = await original(engine, loop, supervisor_text, registered, lang)
            seen.append(None if r is None else r.get("status"))
            return r

        P._try_matrix_ops = spy
        try:
            _ask("en", text)
        finally:
            P._try_matrix_ops = original
        assert "completed" in seen, seen
