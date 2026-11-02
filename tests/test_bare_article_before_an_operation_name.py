r"""REFUSAL-NOT-REACHED ④（2026-09-17）。

**裸の冠詞は操作名の一部ではない。**

英語の「動詞＋式」を切る規則（`_en_verb_formula_re`）は**文頭に操作の別名**を
要求する。`the curl of \vec{A}` は `the` で始まるので当たらず、書かれた式が
**登録されないまま**オペランドの欄へ**生の LaTeX** として入り、成分組み立てが
`\vec{A}(t)_x` を CAS へ渡して**生の `Maxima evaluation error`** が返っていた
（Core Principle #1）。日本語は同じ文が `<式> の回転を求めて` と**式で始まる**ので
先頭式登録が拾い、全部正しかった——日英の割れである。

実測 2026-09-17、`A` をベクトル場として宣言した同じセッションで13綴り:

| 綴り | ja | en（前） |
|---|---|---|
| `\vec{A} の回転` / `the curl of \vec{A}` | **OK** | **生CASエラー** |
| `\vec{A}(t) の回転` / `the curl of \vec{A}(t)` | 理由つき断り | **生CASエラー** |
| `\mathbf{A} の回転` / `the curl of \mathbf{A}` | **OK** | **生CASエラー** |
| `A の回転` / `the curl of A` | OK | OK（素の名前は 2a-sym が解決する） |

**冠詞そのものは既に2箇所が知っていた。** `split_leading_compute_verb` は計算動詞の
**後ろ**の `the` を飲み、この規則は別名の**後ろ**の `the` を飲む。欠けていたのは
**別名の手前**だけで、だから `compute the curl of …` は通り `the curl of …` だけが
落ちる、という差になっていた——一方の経路が既に正しい答えを持っている型
（docs/developer_manuals/35 §2 徴候②）。

**`the` だけで `a`/`an` は入れない。これは実測で決めた。** コーパスの英語で冠詞から
始まる13行のうち `the` は10行（`the commutator of E4 and E5` ほか、どれも操作を
名指す文）。`a` の3行は**冠詞ではなく記号名**である——`a as the creation operator
for photon` の `a` は研究者が宣言しているボソン演算子で、冠詞として読めば別名 `as`
を持つ操作の要求に化ける。

**冠詞は leadin に入れて組み直しで戻す**（MATRIX-INLINE-OPERAND-EN と同じ規約）。
外に出すと `the curl E1` が `curl E1` になり、値は同じでも**答える規則が変わる**。
"""
import asyncio

import pytest

import ui.notebook_bridge as nb
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operation_validator import OperationValidator
from ui.interpret_pipeline import (InputToken, InterpretRequest,
                                   _en_verb_formula_re, _formula_runs_to_the_end,
                                   run_interpret_pipeline)


# ---------------------------------------------------------------------------
# 規則そのもの
# ---------------------------------------------------------------------------
class TestTheArticleIsPartOfTheLeadIn:

    @pytest.mark.parametrize("text,leadin,formula", [
        (r"the curl of \vec{A}",          "the curl",        r"\vec{A}"),
        (r"curl of \vec{A}",              "curl",            r"\vec{A}"),
        (r"the divergence of \vec{A}(t)", "the divergence",  r"\vec{A}(t)"),
        (r"the determinant of [4, 0; 0, 1]", "the determinant", "[4, 0; 0, 1]"),
        (r"simplify \frac{x^2}{y}",       "simplify",        r"\frac{x^2}{y}"),
    ])
    def test_it_matches_and_keeps_the_article(self, text, leadin, formula):
        m = _en_verb_formula_re().match(text)
        assert m, f"当たらない: {text!r}"
        assert m.group(1).strip() == leadin, m.group(1)
        assert m.group(2).strip() == formula, m.group(2)

    @pytest.mark.parametrize("text", [
        # `a` は冠詞として読まない——コーパスでは記号名である
        r"a curl of \vec{A}",
        r"an expansion of x^2",
        # 冠詞だけで別名が無ければ、当たってはいけない
        r"the vector \vec{A}",
    ])
    def test_a_and_an_are_not_articles_here(self, text):
        m = _en_verb_formula_re().match(text)
        assert m is None or not _formula_runs_to_the_end(m.group(2)), \
            f"読んではいけない綴りに当たった: {text!r} → {m and m.group(2)!r}"

    def test_a_and_an_are_left_out_on_a_measured_ground(self):
        r"""`a`/`an` を入れない理由は**綴りが実在しない**こと（実測 2026-09-17）。

        **最初に書いた理由は測って偽だった。** 「`a as the creation operator for
        photon` の `a` が冠詞として読まれる」と書いたが、別名は3文字未満を除くので
        `as` はそもそも候補に入らない。`a|an` を冠詞に足してコーパス全行に当てると
        **新たに claim される行は 0**——つまり `a` を入れても実害は測れない。

        では**なぜ入れないのか。** 入れる理由が測れないからである。
        「通らない綴りを文法に入れない」（2026-09-11、`上から`/`下から` と同じ判断）
        の裏返しで、**実在も実害も測れないものは足さない**。コーパスで `a`/`an` から
        始まる5行はどれも**記号名**（`A = [2, 0; 0, 1]` の H-1 等式が2行、
        `A as a vector field` などの宣言が3行）で、冠詞として書かれた行は1行も無い。

        この検査は、その両方を数え直せる形で固定する。
        """
        import json
        import pathlib
        import re
        rows = json.loads(
            (pathlib.Path(__file__).resolve().parent / "corpus"
             / "instruction_corpus.json").read_text())["rows"]
        a_rows = [r["en"] for r in rows
                  if (r.get("en") or "").lower().startswith(("a ", "an "))]
        assert a_rows, "コーパスに `a` で始まる英語行が無い——理由を測り直すこと"
        # ① どれも**その `a` 自身が記号名**である: 直後に来るのは `=`（H-1 の等式）
        #    か `as`（宣言）で、冠詞として何かを修飾している行は1行も無い。
        assert all(re.match(r"(?i)^an?\s+(=|as\b)", s) for s in a_rows), a_rows
        # ② `a|an` を冠詞に足しても、実在行は1行も新たに claim されない
        cur = _en_verb_formula_re()
        alt = re.compile(cur.pattern.replace("(?:the\\s+)?", "(?:the|a|an)\\s+", 1))
        newly = [r["en"] for r in rows
                 if (r.get("en") and not cur.match(r["en"])
                     and (m := alt.match(r["en"]))
                     and _formula_runs_to_the_end(m.group(2)))]
        assert newly == [], f"実害が測れるなら理由を書き直すこと: {newly}"


# ---------------------------------------------------------------------------
# 生きた経路。**綴りごとに新しいセッション**で測る。
# ---------------------------------------------------------------------------
_DECL = {"ja": "A をベクトル場とする", "en": "Define A as a vector field"}


def _ask(lang, text):
    async def go():
        nb.start_engine(title="article", mode="standard", language=lang,
                        domains=["electromagnetism"])
        try:
            engine = nb.get_engine()
            sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                                domains=["electromagnetism"],
                                primary_backend=engine.session.primary_backend)
            await run_interpret_pipeline(
                InterpretRequest(tokens=[InputToken(type="text",
                                                    value=_DECL[lang])]),
                engine, sup)
            return await run_interpret_pipeline(
                InterpretRequest(tokens=[InputToken(type="text", value=text)]),
                engine, sup)
        finally:
            nb.stop_engine()
    return asyncio.run(go())


class TestTheSecondGuardIsWhatStopsABareName:
    r"""`the norm of A` は規則に**当たる**が、登録されない——止めるのは次の関門.

    コーパスの `the …` 10行のうち9行は `_formula_runs_to_the_end` で降りる
    （E番号か節の語を持つ）。10行目の `the norm of A` だけは広げた規則に当たり、
    **その次の `_HAS_MATH_OP` で降りる**——素の名前は「登録する数式」ではなく、
    2a-sym が式番号へ書き換えるべき**名前**だからである。

    **最初の関門だけを測っていたら「ここには何も届かない」という偽の結論**に
    なっていた。コーパス差分が0だった理由は、関門が1つではなく2つあること。
    """

    def test_it_matches_the_rule(self):
        from ui.interpret_pipeline import _HAS_MATH_OP
        m = _en_verb_formula_re().match("the norm of A")
        assert m and _formula_runs_to_the_end(m.group(2)), "規則には当たるはず"
        assert not _HAS_MATH_OP.search(m.group(2)), \
            "素の名前は数式ではない——ここで降りるのが登録されない理由"

    @pytest.mark.parametrize("formula,is_math", [
        (r"\vec{A}", True),      # LaTeX コマンドを持つ
        ("[4, 0; 0, 1]", True),  # 括弧を持つ
        ("A", False),            # 素の名前
        ("q", False),
    ])
    def test_the_second_guard_separates_them(self, formula, is_math):
        from ui.interpret_pipeline import _HAS_MATH_OP
        assert bool(_HAS_MATH_OP.search(formula)) is is_math


class TestAWrittenVectorFieldIsTheOperand:
    """引数の無い `\\vec{A}` は**計算できる**——英語だけが落ちていた."""

    @pytest.mark.parametrize("lang,text", [
        ("en", r"the curl of \vec{A}"),
        ("en", r"the divergence of \vec{A}"),
        ("en", r"the curl of \mathbf{A}"),
        ("en", r"curl of \vec{A}"),            # 元から通っていた綴り
        ("en", r"compute the curl of \vec{A}"),
        ("ja", r"\vec{A} の回転を求めて"),
        ("ja", r"\vec{A} の発散を求めて"),
        ("ja", r"\mathbf{A} の回転を求めて"),
    ])
    def test_it_computes(self, lang, text):
        r = _ask(lang, text)
        assert r.get("status") == "completed", r
        assert "Maxima evaluation error" not in str(r.get("error") or ""), r


class TestAnAppliedFieldKeepsItsReason:
    """引数つきは**理由つきで断る**——生の CAS 文言ではなく."""

    @pytest.mark.parametrize("lang,text", [
        ("en", r"the curl of \vec{A}(t)"),
        ("en", r"the divergence of \vec{A}(t)"),
        ("en", r"curl of \vec{A}(t)"),
        ("ja", r"\vec{A}(t) の回転を求めて"),
        ("ja", r"\vec{A}(t) の発散を求めて"),
    ])
    def test_the_refusal_reaches_the_researcher(self, lang, text):
        r = _ask(lang, text)
        assert r.get("status") != "completed", r
        err = str(r.get("error") or "")
        assert "Maxima evaluation error" not in err, \
            f"生の CAS 文言が研究者に返っている: {err!r}"
        assert ("ベクトル場" in err or "vector field" in err), err


class TestTheBareNameStillWorks:
    """素の名前は 2a-sym が式番号へ書き換える——冠詞を足しても変わらない."""

    @pytest.mark.parametrize("lang,text", [
        ("en", "the curl of A"),
        ("en", "curl of A"),
        ("ja", "A の回転を求めて"),
    ])
    def test_it_computes(self, lang, text):
        r = _ask(lang, text)
        assert r.get("status") == "completed", r
