r"""NESTED-VEC-OP-LATEX (2026-09-17) — ∇ を LaTeX で書くと、計算に届かなかった。

入れ子のベクトル演算を7組 × 綴り6つ × 日英で並べ、**正しく答えていた Unicode の綴り
（`∇·(∇×A)`）を基準**にすると、HEAD で一致したのは 84 セル中 13:

| 綴り | HEAD |
|---|---|
| LaTeX 4形（`\nabla \cdot (\nabla \times \vec{A})`・括弧なし・`\left(`…） | **0 / 56** —— 全部 `registered`、計算も断りもされない |
| 名前 `curl(grad(f))` | 6 / 14 —— 中の演算が型を変える組を「E1 はスカラーです」で断る |
| 連鎖 `A の回転の発散` / `the divergence of the curl of A` | 7 / 14 —— 英語は0。`the curl of the curl of A` は**回転を1回だけ**適用して `completed` |

根は「∇ の演算子はどう綴られるか」に**2つの表**が答えていたこと——単項は LaTeX を知り、
入れ子は知らなかった。入れ子を単項の表から組み立てる形にして 84 / 84。

族を広げて測ると、同じ割り込み口にもう2つあった:

* **H-1 で値を与えた名前に LaTeX の単項が届かない**——`u = x^2 + y^2` のあとの `\nabla u` は
  `registered`、`∇u` ・ `u の勾配を求めて` は計算される。割り込み口は宣言の種類だけを見ていて、
  H-1 の名前は `kind='anonymous'` で記録される。
* **H-1 のスカラーの勾配が等式の形だった**——`u の勾配を求めて` が `[2*x = 2*x, 2*y = 2*y, 0 = 0]`、
  ラプラシアンが `4 = 4`（日英とも `completed`）。スカラー側の組み立てが値を右辺から読んで
  いなかった（ベクトル側は NORM-OPERATION で読むようになっていた）。型の推論を直すと
  `h = div(A)` の勾配で `[0 = 0, 0 = 0, 0 = 0]` が出て、それで見つかった。
"""
import asyncio

import pytest

import ui.notebook_bridge as nb
from notebook.models.symbol_context import SymbolContext
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operation_validator import OperationValidator
from ui.interpret_pipeline import (InputToken, InterpretRequest,
                                   run_interpret_pipeline)


def _ask(lang, setup, text, domain="electromagnetism"):
    async def go():
        nb.start_engine(title="nested-vec", mode="standard", language=lang, domains=[domain])
        try:
            engine = nb.get_engine()
            sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                                domains=[domain],
                                primary_backend=engine.session.primary_backend)
            for line in setup:
                await run_interpret_pipeline(
                    InterpretRequest(tokens=[InputToken(type="text", value=line)]), engine, sup)
            return await run_interpret_pipeline(
                InterpretRequest(tokens=[InputToken(type="text", value=text)]), engine, sup)
        finally:
            nb.stop_engine()
    return asyncio.run(go())


def _answer(r):
    return (r.get("status"), " ".join(str(r.get("canonical") or "").split()),
            r.get("error") or "")


_FIELDS = {"ja": ["A をベクトル場とする", "f をスカラー場とする"],
           "en": ["define A as a vector field", "define f as a scalar field"]}

#: operator → (LaTeX, Unicode, name, Japanese noun, English noun)
_OP = {"div": (r"\nabla \cdot", "∇·", "div", "発散", "divergence"),
       "curl": (r"\nabla \times", "∇×", "curl", "回転", "curl"),
       "grad": (r"\nabla", "∇", "grad", "勾配", "gradient")}

#: the seven pairs measured — five compute, two are refused (a scalar where a vector
#: is needed, and the converse); a refusal must be the SAME refusal in every spelling
_PAIRS = [("div", "curl", "A"), ("curl", "grad", "f"), ("div", "grad", "f"),
          ("curl", "curl", "A"), ("grad", "div", "A"), ("div", "div", "A"),
          ("grad", "curl", "A")]


def _spellings(outer, inner, field, lang):
    lo, _, no, jo, eo = _OP[outer]
    li, _, ni, ji, ei = _OP[inner]
    vf = rf"\vec{{{field}}}" if field == "A" else field
    return {
        "named": f"{no}({ni}({field}))",
        "latex": rf"{lo} ({li} {vf})",
        "latex-bare": rf"{lo} ({li} {field})",
        "latex-left": rf"{lo} \left({li} {vf}\right)",
        "latex-no-parens": rf"{lo} {li} {vf}",
        "chain": (f"{field} の{ji}の{jo}を求めて" if lang == "ja"
                  else f"the {eo} of the {ei} of {field}"),
    }


class TestEverySpellingLandsWhereTheUnicodeOneDoes:
    @pytest.mark.parametrize("lang", ["ja", "en"])
    @pytest.mark.parametrize("outer,inner,field", _PAIRS)
    def test_the_family(self, lang, outer, inner, field):
        unicode = f"{_OP[outer][1]}({_OP[inner][1]}{field})"
        want = _answer(_ask(lang, _FIELDS[lang], unicode))
        assert want[0] in ("completed", "failed"), want
        for name, text in _spellings(outer, inner, field, lang).items():
            got = _answer(_ask(lang, _FIELDS[lang], text))
            assert got == want, (name, text, got, want)


class TestTheEnglishChainAppliesBothOperations:
    def test_the_curl_of_the_curl_is_not_one_curl(self):
        """直す前は別名走査が同じ語を1つの操作と読み、回転を1回だけ適用していた。"""
        twice = _answer(_ask("en", _FIELDS["en"], "the curl of the curl of A"))
        once = _answer(_ask("en", _FIELDS["en"], r"\nabla \times \vec{A}"))
        assert twice != once, twice
        assert twice == _answer(_ask("ja", _FIELDS["ja"], "A の回転の回転を求めて")), twice


_VALUES = ["u = x^2 + y^2", "v = [y, -x, 0]"]


class TestANameWithAValueIsReadByItsValue:
    @pytest.mark.parametrize("lang", ["ja", "en"])
    @pytest.mark.parametrize("text,want", [
        (r"\nabla u", "[2*x,2*y,0]"),
        (r"\nabla \times v", "[0,0,-2]"),
        (r"\nabla \cdot v", "0"),
        ("curl(v)", "[0,0,-2]"),
    ])
    def test_the_latex_spelling_computes(self, lang, text, want):
        r = _ask(lang, _VALUES, text)
        assert r.get("status") == "completed", r
        assert "".join(str(r.get("canonical") or "").split()) == want, r

    @pytest.mark.parametrize("lang,text", [("ja", "g の回転を求めて"), ("en", "the curl of g")])
    def test_a_name_for_a_gradient_is_a_vector(self, lang, text):
        """`g = grad(f)` を「未宣言の名前 grad とスカラー場 f」＝スカラーと読んでいた。"""
        r = _ask(lang, _FIELDS[lang] + ["g = grad(f)"], text)
        assert r.get("status") == "completed", r
        assert "".join(str(r.get("canonical") or "").split()) == "[0,0,0]", r

    @pytest.mark.parametrize("lang,text", [("ja", "h の勾配を求めて"), ("en", "the gradient of h")])
    def test_a_name_for_a_divergence_is_a_scalar(self, lang, text):
        r = _ask(lang, _FIELDS[lang] + ["h = div(A)"], text)
        assert _answer(r) == _answer(_ask(lang, _FIELDS[lang], "∇(∇·A)")), r


class TestTheGradientOfADefinedScalarIsNotAnEquation:
    @pytest.mark.parametrize("lang,text,want", [
        ("ja", "u の勾配を求めて", "[2*x,2*y,0]"),
        ("en", "the gradient of u", "[2*x,2*y,0]"),
        ("ja", "u のラプラシアンを求めて", "4"),
        ("en", "the Laplacian of u", "4"),
    ])
    def test_the_value_is_read_from_the_right_hand_side(self, lang, text, want):
        """HEAD は `[2*x = 2*x, 2*y = 2*y, 0 = 0]` / `4 = 4` を `completed` で返していた。"""
        r = _ask(lang, ["u = x^2 + y^2"], text)
        assert r.get("status") == "completed", r
        assert "".join(str(r.get("canonical") or "").split()) == want, r


class TestABackendWithoutTheOperationStillWrites:
    def test_a_julia_session_registers_the_notation(self):
        """ベクトル解析は Maxima だけ。拾って断ると `\\nabla u` が書けなくなる（実測した退行）。"""
        r = _ask("ja", ["u = x^2 + y^2"], r"\nabla u", domain="condensed_matter")
        assert r.get("status") == "registered", r


class TestOneTableSpellsTheOperators:
    def test_every_prefix_reads_as_outer_and_as_inner(self):
        """入れ子の綴りは単項の表から**組み立てる**——表に足した綴りは両方の位置で読める。"""
        from notebook.runtime.vector_op_spelling import VEC_OP_PREFIXES, read_nested, read_single
        samples = {"divergence": [r"\nabla \cdot", "∇·", "div", "divergence"],
                   "curl": [r"\nabla \times", "∇×", "curl", "rot"],
                   "gradient": [r"\nabla", "∇", "grad", "gradient"],
                   "scalar_laplacian": [r"\nabla^2", "∇²", r"\Delta", "laplacian"]}
        assert {op for _, op in VEC_OP_PREFIXES} == set(samples)
        for op, spellings in samples.items():
            for s in spellings:
                assert read_single(f"{s} A") == (op, "A"), s
                assert read_nested(rf"{s} (\nabla \times A)") == (op, "curl(A)"), s
                assert read_nested(rf"\nabla \cdot ({s} A)")[0] == "divergence", s

    @pytest.mark.parametrize("text", [r"\nabla \times \vec{A}", "the curl of A", "div t",
                                      r"(\nabla f) + g"])
    def test_what_is_not_nested_is_not_read_as_nested(self, text):
        from notebook.runtime.vector_op_spelling import read_nested
        assert read_nested(text) is None


class TestTheTypeOfACall:
    @pytest.mark.parametrize("text,want", [
        ("(grad(f))", "vector"), ("curl(A)", "vector"),
        ("(div(A))", "scalar"), ("laplacian(f)", "scalar"),
        # vector × scalar.  None until the type was read from the STRUCTURE — counting
        # names could not tell this product from `grad(f) . grad(g)`.
        ("grad(f)*div(A)", "vector"),
        ("x^2+y^2", "scalar"),
    ])
    def test_infer_algebra(self, text, want):
        from notebook.runtime.vector_op_types import infer_algebra
        assert infer_algebra(text, SymbolContext()) == want
