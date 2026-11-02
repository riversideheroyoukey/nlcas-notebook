r"""行列の語が、別の操作に取られていた。

実測 2026-09-07（worklist の `matrix`、6行・6鍵）。読みの層そのものは健在で
（単項7操作 × 動詞あり／なし × 日英 = 28綴りが全通）、**割れていたのは
「どの語がどの操作のものか」という日英のデータ**だった。

| 打った文 | 直す前 |
|---|---|
| `invert E1` | 「逆変換には**最低2つの変換式が必要です**」 |
| `Invert E1` | 同上 |
| `the matrix product of E1 and E2` | **`define matrix: name=product`** を提案 |
| `matrix product of E1 and E2` | 同上 |

**① `invert` は `inverse_matrix` の別名でもある。** `inv_transform[en]` は裸の
動詞 `invert` を受けるが、日本語側は名詞 `逆変換` を必ず要求する——つまり
`invert` は**日本語に対応物の無い英語だけの綴り**。そして「E-id が1つなら
逆基底変換ではありえない」という guard は **claim した1行下に在り**、claim
してから断っていたので、研究者が書いていない変換について語る断り文が返って
いた。名指したかを `named` 群で持ち、名指しておらず1つなら**降りる**。
日本語は必ず名指すので従来どおり理由つきで断る（そちらが親切）。

**② 別名表の日英が非対称だった。**

| 操作 | ja | en（直す前） |
|---|---|---|
| `matrix_multiply` | 行列積・行列の積・行列積を求める | matrix_multiply・matmul・matrix multiply（**`matrix product` が無い**） |
| `define_matrix` | 行列を定義・行列とする・マトリックスとする（**全部に宣言の動詞**） | define_matrix・**`matrix`**（裸の名詞）・define matrix |

英語で最も普通の綴り `matrix product` が無く、代わりに裸の名詞 `matrix` が
**宣言**の別名だったので、行列に言及した英文が「`product` という名前の行列を
宣言しますか」になっていた。`laplace` を削除したときと同じ判断——
**綴りが実在の用法かを先に訊く**。裸の `matrix` は要求ではなく名詞である。
宣言は `type_kw_matrix` 経由なので4綴りとも通ったまま（実測）。

綴り空間 **37/42 → 42/42**。

**値の欠陥は別に起票した（MATMUL-TEMPLATE-STAR）。**
`multiply the matrices E1 and E2` と `E1 と E2 を掛け合わせて` は
`[2,1;1,2]·[1,0;0,3]` に対し **`matrix([2,0],[0,6])`（要素積）を completed で
返す**。読みの層で振り替えるのは誤り——日本語側は「行列」と書いていないし、
一般の `combine_expressions` も誤ったままになる。**値の欠陥は値の層で直す。**
"""
import asyncio
import json
import pathlib

import pytest

import ui.notebook_bridge as nb
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.nl_patterns import nlp, pick_group
from notebook.runtime.operation_validator import OperationValidator
from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline

_SCHEMA = json.loads(
    (pathlib.Path(__file__).resolve().parent.parent / "schemas"
     / "common_operations.schema.json").read_text(encoding="utf-8"))


def _aliases(op):
    for o in _SCHEMA["operations"]:
        if o.get("operation") == op:
            return o.get("aliases") or {}
    raise AssertionError(op)


#: 数表 2つ。`[2,1;1,2] . [1,0;0,3] = [2,3;1,6]`、要素積なら `[2,0;0,6]`。
MATS = ["[2, 1; 1, 2]", "[1, 0; 0, 3]"]


def _ask(lang, text, prep=MATS):
    """**どの操作が走ったか**を返す。status だけでは足りない——
    `needs_confirmation` は「別の操作を提案した」でもありうる。"""
    nb.start_engine(title="mw", mode="standard", language=lang, domains=[])
    calls: list = []
    try:
        eng = nb.get_engine()
        sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                            domains=[], primary_backend="maxima")
        orig = eng.run_operation

        def spy(op, args):
            calls.append(op)
            return orig(op, args)
        eng.run_operation = spy

        def run(x):
            return asyncio.get_event_loop().run_until_complete(
                run_interpret_pipeline(InterpretRequest(
                    tokens=[InputToken(type="text", value=x)]), eng, sup))
        for p in prep:
            run(p)
        calls.clear()
        r = run(text)
        return calls, r.get("status"), r
    finally:
        nb.stop_engine()


UNARY_JA = [("逆行列", "inverse_matrix"), ("行列式", "determinant"),
            ("固有値", "eigenvalues"), ("固有ベクトル", "eigenvectors"),
            ("転置", "transpose"), ("トレース", "trace"),
            ("対角行列", "diagonal_matrix")]
UNARY_EN = [("inverse", "inverse_matrix"), ("determinant", "determinant"),
            ("eigenvalues", "eigenvalues"), ("eigenvectors", "eigenvectors"),
            ("transpose", "transpose"), ("trace", "trace"),
            ("diagonal matrix", "diagonal_matrix")]


class TestTheUnaryFamilyReadsWithAndWithoutAVerb:
    """ここは直す前から全通——**読みの層は健在だった**、を固定しておく
    （回帰したときに、割れがデータ側か読み側かをすぐ切り分けられる）。"""

    @pytest.mark.parametrize("word,want", UNARY_JA)
    @pytest.mark.parametrize("frame", ["E1 の{w}を求めて", "E1 の{w}"])
    def test_japanese(self, frame, word, want):
        calls, st, _ = _ask("ja", frame.format(w=word))
        assert st == "completed" and want in calls, (frame, word, st, calls)

    @pytest.mark.parametrize("word,want", UNARY_EN)
    @pytest.mark.parametrize("frame", ["find the {w} of E1", "the {w} of E1"])
    def test_english(self, frame, word, want):
        calls, st, _ = _ask("en", frame.format(w=word))
        assert st == "completed" and want in calls, (frame, word, st, calls)


class TestInvertIsTheMatrixInverseWhenNoTransformIsNamed:
    @pytest.mark.parametrize("text", ["invert E1", "Invert E1",
                                      "invert the matrix E1"])
    def test_single_expression(self, text):
        """直す前は「逆変換には最低2つの変換式が必要です」——**研究者が書いて
        いない変換**について語る断り文。"""
        calls, st, _ = _ask("en", text)
        assert st == "completed" and "inverse_matrix" in calls, (text, st, calls)

    @pytest.mark.parametrize("text", ["Invert E1, E2", "Invert E1 and E2"])
    def test_two_expressions_are_still_the_basis_transform(self, text):
        """L-1 の逆基底変換は残す（三十四問目で固定した綴り）。"""
        calls, st, _ = _ask("en", text)
        assert "build_inverse_transform" in calls, (text, st, calls)

    def test_japanese_names_the_transform_so_it_still_refuses_with_a_reason(self):
        """日本語は `逆変換` という名詞を必ず書くので、1つしか無ければ
        **理由つきで断る**ほうが親切——降りると別名表で別の操作になる。"""
        from notebook.runtime.i18n import t
        calls, st, resp = _ask("ja", "E1 の逆変換を求めて")
        assert st == "failed", (st, calls)
        assert t("interpreter.inverse_transform_need_eqs", "ja") in str(
            resp.get("error") or "")

    @pytest.mark.parametrize("lang,text,named", [
        ("en", "invert E1", False),
        ("en", "build the inverse transform of E1", True),
        ("ja", "E1 の逆変換を求めて", True),
    ])
    def test_the_pattern_says_whether_the_transform_was_named(self, lang, text, named):
        m = nlp("inv_transform", lang).search(text)
        assert m is not None, text
        assert bool(pick_group(m, "named")) is named, text


class TestTheMatrixProductIsNotADeclaration:
    @pytest.mark.parametrize("text", [
        "find the matrix product of E1 and E2",
        "the matrix product of E1 and E2",
        "matrix product of E1 and E2",
    ])
    def test_english(self, text):
        """直す前は `define matrix: name=product` を**唯一の候補**として提案
        していた（裸の名詞 `matrix` が宣言の別名だったため）。"""
        calls, st, resp = _ask("en", text)
        assert st == "completed" and "matrix_multiply" in calls, (text, st, calls)
        assert resp.get("operation") != "define_matrix", text

    @pytest.mark.parametrize("text", ["E1 と E2 の行列積を求めて", "E1 と E2 の行列積"])
    def test_japanese(self, text):
        calls, st, _ = _ask("ja", text)
        assert st == "completed" and "matrix_multiply" in calls, (text, st, calls)


class TestTheDeclarationStillWorks:
    """裸の `matrix` を別名から外しても宣言は通る——**宣言は
    `type_kw_matrix` 経由**で、別名表を通っていない（実測）。"""

    @pytest.mark.parametrize("lang,text", [
        ("ja", "A を行列とする"), ("ja", "A をマトリックスとする"),
        ("en", "Define A as a matrix"), ("en", "declare A as a matrix"),
        ("en", "declare A as matrices"),
    ])
    def test_define_matrix(self, lang, text):
        calls, st, _ = _ask(lang, text, prep=[])
        assert "define_matrix" in calls, (text, st, calls)


class TestTheAliasDataIsSymmetric:
    """別名は**データ**なので、日英の非対称はデータとして数える。"""

    def test_the_ordinary_english_spelling_of_a_matrix_product_is_an_alias(self):
        assert "matrix product" in _aliases("matrix_multiply")["en"]

    def test_a_bare_noun_is_not_a_declaration_request(self):
        """`laplace` を削除したときと同じ判断——**実在の用法かを先に訊く**。
        「matrix」は名詞であって「行列を定義せよ」という要求ではない。"""
        assert "matrix" not in _aliases("define_matrix")["en"], (
            "裸の名詞が宣言の別名に戻っている")

    def test_every_japanese_declaration_alias_carries_the_verb(self):
        """日本語側にはもともと裸の `行列` が無い——その非対称が症状だった。"""
        for a in _aliases("define_matrix")["ja"]:
            assert a not in ("行列", "マトリックス", "マトリクス"), a
