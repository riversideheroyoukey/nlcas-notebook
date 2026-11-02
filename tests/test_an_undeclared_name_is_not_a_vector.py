r"""REFUSAL-NOT-REACHED ②（2026-09-17）。

**「推測しない」と「測っていない」は違う。**

`vector_op_types` は「型が確信を持って言えるときだけ断る」と書き、未宣言の名前が
混ざると `None`（判定しない）を返していた。ところが**このモジュール自身の docstring
が挙げている断る例**は `x^2+y^2` の発散であり、それはまさに未宣言の記号でできている。
実測 2026-09-17、その綴りは断られず**生の `Maxima evaluation error`** を返していた
（Core Principle #1）。

同じ測定で**もう1件**出た。未宣言の `\vec{B}` は3操作とも黙って誤答する:

| 打った文 | 前 | 後 |
|---|---|---|
| `f = x^2 + y^2` の発散／回転 | **生CASエラー** | 理由つきの断り |
| `\vec{B}` の**勾配** | **`[0,0,0]`** が `completed` | 理由つきの断り |
| `\vec{B}` の発散／回転 | 計算される | 同じ |
| 宣言済みのベクトル場／スカラー場 | 正しい | 同じ |
| 成分リスト `[1,2,3]` | 正しい | 同じ |

`[0,0,0]` は docstring が「2026-08-06 に直した」と書いた `[0=0, 0=0, 0=0]` と同じ形で、
**宣言されていない側で生き残っていた**。

**規則は1つ。** この系でベクトルだと言えるのは3つだけ——**成分リスト**、**宣言**、
そして**この系自身の包み `vec(...)`**（`\vec{B}` / `\mathbf{B}` はどちらもここへ降りる）。
どれも無ければスカラーである。未宣言の記号はこの系では自由変数＝CAS の記号原子で
（CLAUDE.md「Free variables」）、ベクトルになるには上の3つのどれかが要る。

**包みを数えるのが要だった。** 素朴に「未宣言はスカラー」とだけ書くと `vec(B)` まで
スカラーになり、`\nabla \times \vec{B}` が**断られる**——実測して分かった退行で、
広げる前に測っていなければ入れていた。

**断り文も直した。** この断りは未宣言を判定しなかったせいで目に触れにくく、日本語が
「E1 は scalar なので、divergence の対象にできません」と内部名のままだった。語は
どちらも既に在る（型は `declare.algebra_*`、操作名は `_operation_label`）——隣の
guard には渡っていて、ここには渡っていなかっただけである。
"""
import asyncio

import pytest

import ui.notebook_bridge as nb
from notebook.models.symbol_context import SymbolContext
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operation_validator import OperationValidator
from notebook.runtime.vector_op_types import infer_algebra, operand_mismatch
from ui.interpret_pipeline import (InputToken, InterpretRequest,
                                   run_interpret_pipeline)


class TestWhatCountsAsAVector:
    """成分リスト・宣言・包み——この3つだけ."""

    @pytest.mark.parametrize("text,want", [
        # 包み（宣言されていなくてもベクトル）
        ("vec(A)",            "vector"),
        ("vec(A(t))",         "vector"),
        # 成分リスト
        ("[a, b, c]",         "vector"),
        (r"\left[ 1, 2, 3 \right]", "vector"),
        # 未宣言の記号だけ → スカラー（ここが 2026-09-17 の変更点）
        ("x^2 + y^2",         "scalar"),
        ("a*x + b",           "scalar"),
        ("sin(x)",            "scalar"),
        # **和**はベクトル。ここは 2026-09-17 まで None で、理由は「内積とも外積とも
        # 取れる」だった——それは**積**の話で、和には当たらない。名前を数える方式の
        # 出力を写していた（書いたのは自分）。積は下の `vec(A) vec(B)` が None のまま。
        ("vec(A) + vec(B)",   "vector"),
        ("vec(A) vec(B)",     None),
        # 名前が無い
        ("3",                 None),
    ])
    def test_infer(self, text, want):
        assert infer_algebra(text, SymbolContext()) == want

    def test_the_wrapper_is_not_counted_as_a_quantity_name(self):
        """`vec` は包みの綴りであって量の名前ではない."""
        from notebook.runtime.vector_op_types import _NOT_A_QUANTITY
        assert "vec" in _NOT_A_QUANTITY

    @pytest.mark.parametrize("op,text,mismatched", [
        ("divergence", "x^2 + y^2", True),    # スカラーに発散 → 断る
        ("curl",       "x^2 + y^2", True),
        ("gradient",   "x^2 + y^2", False),   # スカラーに勾配 → 正しい
        ("gradient",   "vec(B)",    True),    # ベクトルに勾配 → 断る
        ("divergence", "vec(B)",    False),   # ベクトルに発散 → 正しい
        ("curl",       "vec(A) + vec(B)", False),   # 不明なら断らない
    ])
    def test_mismatch(self, op, text, mismatched):
        got = operand_mismatch(op, text, SymbolContext())
        assert (got is not None) is mismatched, got


# ---------------------------------------------------------------------------
# 生きた経路（日英）
# ---------------------------------------------------------------------------
def _ask(lang, setup, text):
    async def go():
        nb.start_engine(title="vec-type", mode="standard", language=lang,
                        domains=["electromagnetism"])
        try:
            engine = nb.get_engine()
            sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                                domains=["electromagnetism"],
                                primary_backend=engine.session.primary_backend)
            for s in setup:
                await run_interpret_pipeline(
                    InterpretRequest(tokens=[InputToken(type="text", value=s)]),
                    engine, sup)
            return await run_interpret_pipeline(
                InterpretRequest(tokens=[InputToken(type="text", value=text)]),
                engine, sup)
        finally:
            nb.stop_engine()
    return asyncio.run(go())


class TestNoRawCasErrorForATypeMismatch:

    @pytest.mark.parametrize("lang,text", [
        ("ja", "E1 の発散を求めて"), ("ja", "E1 の回転を求めて"),
        ("en", "the divergence of E1"), ("en", "the curl of E1"),
    ])
    def test_a_scalar_expression_is_refused_with_a_reason(self, lang, text):
        r = _ask(lang, ["f = x^2 + y^2"], text)
        assert r.get("status") != "completed", r
        err = str(r.get("error") or "")
        assert "Maxima evaluation error" not in err, f"生の CAS 文言: {err!r}"
        assert ("スカラー" in err or "scalar" in err), err

    @pytest.mark.parametrize("lang,text", [
        ("ja", "E1 の勾配を求めて"), ("en", "the gradient of E1"),
    ])
    def test_an_undeclared_vec_is_refused_for_gradient(self, lang, text):
        """`[0,0,0]` を成功として返していた形."""
        r = _ask(lang, [r"u = \vec{B}"], text)
        assert r.get("status") != "completed", r
        assert ("ベクトル" in str(r.get("error") or "")
                or "vector" in str(r.get("error") or "")), r


class TestTheRefusalIsInTheResearchersLanguage:

    def test_japanese_carries_no_internal_names(self):
        r = _ask("ja", ["f = x^2 + y^2"], "E1 の発散を求めて")
        err = str(r.get("error") or "")
        for internal in ("scalar", "vector", "divergence"):
            assert internal not in err, f"内部名 {internal!r} が日本語文に: {err!r}"

    def test_english_names_the_operation_and_the_types(self):
        r = _ask("en", ["f = x^2 + y^2"], "the divergence of E1")
        err = str(r.get("error") or "")
        assert "scalar" in err and "vector" in err and "divergence" in err, err


class TestWhatStillWorks:
    """広げすぎていないこと——宣言済み・包み・成分リストは通るまま."""

    @pytest.mark.parametrize("lang,setup,text", [
        ("ja", ["A をベクトル場とする"], "A の発散を求めて"),
        ("ja", ["A をベクトル場とする"], "A の回転を求めて"),
        ("ja", ["f をスカラー場とする"], "f の勾配を求めて"),
        ("ja", ["u = [1, 2, 3]"], "E1 の発散を求めて"),
        ("ja", [r"u = \vec{B}"], "E1 の発散を求めて"),
        ("en", ["Define A as a vector field"], "the curl of A"),
    ])
    def test_it_computes(self, lang, setup, text):
        r = _ask(lang, setup, text)
        assert r.get("status") == "completed", r
