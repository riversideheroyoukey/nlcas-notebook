r"""「X を Y とする」の**もう一方の行き先**と、語で書かれた数。

`X を Y とする` / `set X to Y` / `let X be Y` は、Y が**型の語**なら宣言
（`decl_stmt` → `define_scalar_parameter` ほか）、**値や式**なら H-1 の
`X = Y` である。判別子は最初から在った——`_looks_like_type_label` の docstring
が「`x を 0 とする`（代入）を弾くための guard」と自分で書いている——のに、
型でない側の枝が**日英とも作られていなかった**（実測 2026-09-10）:

| 打った文 | 直す前 |
|---|---|
| `x を 3 とする` / `x を a + b とする` | 解釈できません |
| `let x be 3` / `set x to a + b` | 同上 |
| `set x to 3` | `define_function` → **生の `Maxima evaluation error`** |
| `set E1 to zero` | 同上 |

そして値の欄には、もっと基本的な穴があった: **語で書かれた数を誰も読まない。**

| 打った文 | 直す前 |
|---|---|
| `set the right-hand side of E1 to zero` | `a*x + b = zero`（**記号**）を completed |
| `x をゼロとする` | `define_scalar_parameter` を **confirmed** |
| `E1 の右辺をゼロとする` | 解釈できません |
| `substitute x = zero into E1` | 記号 `zero` を代入して completed |

数詞の表（`ui/interpret_common`）に 0 が無かったのは偶然ではない: その表は
「何階微分か」「何回か」を**数える**ために育っており、そこに 0 は書かれない。
値の欄が初めて要ったときに、いちばん普通の値が表に無いと分かった。
表は `notebook/runtime/number_words.py` へ降りている（`notebook.runtime` の
2箇所が数詞を訊くので、`ui` に置いたままだと表が2枚になる）。
"""
import json
import pathlib

import pytest

from notebook.runtime.backend_handler import _looks_like_type_label
from notebook.runtime.maxima_latex_converter import latex_to_canonical
from notebook.runtime.nl_patterns import nlp, nlp_search
from notebook.runtime.number_words import number_literal, number_word

_SCHEMA = json.loads(
    (pathlib.Path(__file__).resolve().parents[1]
     / "schemas" / "common_operations.schema.json").read_text(encoding="utf-8"))


class TestTheNumberWordTables:
    """**数え方と値は別の語彙である。** 序数と倍数副詞は値にならない。"""

    @pytest.mark.parametrize("token,want", [
        ("zero", "0"), ("ゼロ", "0"), ("零", "0"), ("〇", "0"),
        ("one", "1"), ("三", "3"), ("ten", "10"), ("十", "10"),
        ("0", "0"), ("０", "0"), ("3", "3"), ("-3", "-3"),
    ])
    def test_a_written_number_is_that_number(self, token, want):
        assert number_literal(token) == want, token

    @pytest.mark.parametrize("token", ["third", "3rd", "twice", "thrice", "first",
                                       "x", "", "一定", "a + b", "zeroth"])
    def test_what_is_not_a_value_is_refused(self, token):
        """序数（`third`）も倍数副詞（`twice`）も**数え方**であって値ではない。
        `set the right-hand side to twice` は文ですらない。"""
        assert number_literal(token) is None, token

    @pytest.mark.parametrize("token,want", [
        ("third", 3), ("3rd", 3), ("twice", 2), ("thrice", 3),
        ("three", 3), ("三", 3), ("ten", 10), ("3", 3), ("３", 3),
    ])
    def test_counting_still_reads_everything_it_did(self, token, want):
        assert number_word(token) == want, token

    def test_zero_joined_the_counting_vocabulary_too(self):
        """表は合成で書かれているので、基数に足せば数える側にも届く。
        既に `number_word("0")` は 0 を返していたので、0 という返り値は
        消費者にとって新しい種類ではない。"""
        assert number_word("zero") == 0
        assert number_word("ゼロ") == 0

    def test_the_cardinals_are_a_subset_of_what_counting_reads(self):
        """**部分集合の関係はデータで書く。** 片方だけに綴りが足されると
        「当たるのに読めない」綴りができる（`three times` がそれだった）。"""
        from notebook.runtime.number_words import _CARDINAL_WORDS, _NUMBER_WORDS
        for word, value in _CARDINAL_WORDS.items():
            assert _NUMBER_WORDS.get(word) == value, word


class TestANumberIsNotATypeLabel:
    """`_looks_like_type_label` は「数は型ではない」と自分で書いていたのに、
    数字の綴りしか知らなかった。"""

    @pytest.mark.parametrize("label", ["ゼロ", "零", "zero", "one", "0", "3", "ten"])
    def test_a_written_number_is_not_a_type(self, label):
        assert _looks_like_type_label(label) is False, label

    @pytest.mark.parametrize("label", ["スカラー", "a scalar", "scalar", "行列",
                                       "一定", "１電子エネルギー", "vector"])
    def test_a_type_noun_is_still_a_type(self, label):
        """総当たりの2つ（日本語なら型の説明／ラテン文字4字以上なら物理の名詞）は
        消せない——表に無い名詞をスカラーとして受けるのが正しいので。"""
        assert _looks_like_type_label(label) is True, label


class TestTheValueDoorReadsAWrittenNumber:
    """値の欄はどれも LaTeX→正準形の扉を通るので、そこ1箇所で読む。"""

    @pytest.mark.parametrize("written,want", [
        ("zero", "0"), ("ゼロ", "0"), ("零", "0"), ("one", "1"), ("ten", "10"),
    ])
    def test_a_whole_expression_that_is_a_number_is_that_number(self, written, want):
        assert latex_to_canonical(written) == want, written

    @pytest.mark.parametrize("written,name", [
        ("x + one", "one"), ("a*zero", "zero"), ("one_1", "one"),
        ("\\alpha one", "one"), ("ten x", "ten"),
    ])
    def test_a_word_inside_an_expression_stays_a_name(self, written, name):
        """**文の一部は決して書き換えない。** 式の中に立っている `one` は
        名前であって値ではない——部分置換にすると研究者の記号が黙って数になる。"""
        assert name in latex_to_canonical(written), written

    def test_an_ordinary_expression_is_untouched(self):
        assert latex_to_canonical("a x + b") == "a*x + b"


class TestTheNameDoorMustReturnAName:
    """式の扉を名前に使っている場所がある（`_declared_names_source`）。
    そこで名前が数に化けると、宣言そのものが落ちて別名走査が動詞を宣言した。"""

    @pytest.mark.parametrize("text,want", [
        ("declare zero as a scalar", "zero"),
        ("declare one as a scalar", "one"),
        ("declare q as a scalar", "q"),
        ("declare G^0 as a matrix", "G_sup0"),
    ])
    def test_a_name_spelled_like_a_number_survives(self, text, want):
        """最後の行は DECL-NAME-READING が入れた性質——名前は**式の側と同じ扉**を
        通る。ここで守っているのは「その扉が名前を返さなかったら書いた綴りを
        返す」ことであって、扉を通すのをやめることではない。"""
        from notebook.runtime.maxima_backend_handler import MaximaBackendHandler
        h = MaximaBackendHandler(None)
        res = h.extra_pre_phase1(text, "en")
        assert res is not None, text
        assert (res.args or {}).get("name") == want, res


class TestTheValueBranchOfADeclarationShapedSentence:
    """**もう一方の行き先。** 型でなければ H-1 の `X = Y`。"""

    @pytest.mark.parametrize("lang,text,name,value", [
        ("ja", "x を 3 とする", "x", "3"),
        ("ja", "x を 3 とします", "x", "3"),
        ("ja", "x を 3 とおく", "x", "3"),
        ("ja", "x を 3 と置く", "x", "3"),
        ("ja", "x を 3 とみなす", "x", "3"),
        ("ja", "x を a + b とする", "x", "a + b"),
        ("ja", "x を 0 とする", "x", "0"),
        ("ja", "x をゼロとする", "x", "0"),
        ("ja", "x を零とする", "x", "0"),
        ("ja", "x を１０とする", "x", "10"),
        ("en", "set x to 3", "x", "3"),
        ("en", "set x to a + b", "x", "a + b"),
        ("en", "let x be 3", "x", "3"),
        ("en", "let x be zero", "x", "0"),
        ("en", "set x to zero", "x", "0"),
        ("en", "let x be a + b", "x", "a + b"),
    ])
    def test_it_becomes_an_equation(self, lang, text, name, value):
        m = nlp_search("value_definition", text) if lang == "ja" else \
            nlp("value_definition", "en").search(text)
        assert m is not None, text
        assert m.group("name") == name, text
        got = m.group("value").strip()
        assert (number_literal(got) or got) == value, (text, got)
        assert _looks_like_type_label(got) is False, text

    @pytest.mark.parametrize("lang,text", [
        ("ja", "x をスカラーとする"), ("ja", "x を実スカラーとする"),
        ("ja", "x を一定とする"),
        ("en", "let x be a scalar"), ("en", "set x to a scalar"),
    ])
    def test_a_type_complement_is_left_to_the_declaration(self, lang, text):
        """型の語なら**この枝は掴まない**（掴んだら `x = スカラー` を登録する）。"""
        m = nlp("value_definition", lang).search(text)
        if m is not None:
            assert _looks_like_type_label(m.group("value").strip()) is True, text


class TestItDoesNotStealTheSentencesThatNameAnExpression:
    """式番号を含む文には当たらない。理由は2つあり、どちらも実測である。"""

    @pytest.mark.parametrize("lang,text", [
        # (1) 素の式番号——「全体を0に」か「右辺を0に」かは書かれていない
        ("ja", "E1 を 0 とする"), ("en", "set E1 to zero"), ("en", "set E1 to 0"),
        # (2) 代入の一族——動詞を剥がすと `let x = 2 in E1` が
        #     canonical `2*in(a*x+b)` として登録されていた
        ("en", "set x = 0 in E1"), ("en", "let x = 2 in E1"),
        ("en", "set x = zero in E1"),
        # H-5z（右辺を値にする）
        ("en", "set the right-hand side of E1 to 0"),
        ("en", "set the right-hand side of E1 to zero"),
        ("ja", "E1 の右辺を 0 とする"), ("ja", "E1 の右辺をゼロとする"),
        # 初期条件・表示名
        ("en", "Set the initial conditions of E4 to x(0)=1"),
        ("en", "set latex name of E1 to Q"),
    ])
    def test_the_more_specific_reading_keeps_priority(self, lang, text):
        assert nlp("value_definition", lang).search(text) is None, text

    @pytest.mark.parametrize("lang,text", [
        ("ja", "結果を展開して x を 3 とする"),
        ("ja", "両辺を整理して y をゼロとする"),
        ("en", "expand the result and let x be 3"),
        ("en", "differentiate the result and set x to 3"),
    ])
    def test_a_compound_sentence_is_not_a_bare_definition(self, lang, text):
        """**錨（`^`）が要るのはここである。** 式番号の guard は式番号を含む文
        しか見ないので、これは別の軸——文の頭に別の指示が立っている形。錨を
        外すと `x を 3 とする` だけを掴み、**前半の指示を黙って捨てて**
        成功を返す（実測 2026-09-10、変異試験で発覚）。"""
        assert nlp("value_definition", lang).search(text) is None, text


class TestTheRightHandSideValue:
    """H-5z は新しい式の LaTeX を**書かれた綴りから**組むので、値の側だけ
    直すと canonical は `0`、画面と .tex は `= zero` になる。"""

    @pytest.mark.parametrize("lang,text,want", [
        ("en", "set the right-hand side of E1 to zero", "0"),
        ("en", "set the right-hand side of E1 to one", "1"),
        ("en", "set the right-hand side of E1 to 0", "0"),
        ("ja", "E1 の右辺をゼロとする", "0"),
        ("ja", "E1 の右辺を零とする", "0"),
        ("ja", "E1 の右辺を 0 とする", "0"),
        ("ja", "E1 の右辺を０とする", "0"),
    ])
    def test_the_value_is_read_as_a_number(self, lang, text, want):
        from ui.interpret_common import pick_group
        m = nlp("rhs_set", lang).search(text)
        assert m is not None, text
        val = (pick_group(m, "value") or "").strip()
        assert (number_literal(val) or val) == want, (text, val)


class TestSetIsNoLongerABareAliasOfDefineFunction:
    """`laplace` を削除したときと同じ判断——**綴りが実在の用法かを先に訊く**。
    指示コーパスとシナリオで `set` を含む行は7つ——**6つが H-5z**（「右辺を…に
    する」）、**1つが初期条件**——で、`define_function` に届くものは1つも無い。
    別名を持っていたせいで `set E1 to zero` が生の `Maxima evaluation error`
    を返していた（Core Principle #1）。"""

    def test_define_function_no_longer_lists_bare_set(self):
        for spec in _SCHEMA["operations"]:
            if spec["operation"] == "define_function":
                assert "set" not in [a.lower() for a in spec["aliases"]["en"]]
                assert "define" in spec["aliases"]["en"]
                return
        pytest.fail("define_function が見つからない")

    def test_no_other_operation_claimed_the_word(self):
        """削除で別の操作が `set` を取り戻す、ということが起きていない
        （別名索引は先着1件なので、衝突していれば行き先が変わる）。"""
        owners = [spec["operation"] for spec in _SCHEMA["operations"]
                  if "set" in [a.lower() for a in spec.get("aliases", {}).get("en", [])]]
        assert owners == [], owners

    def test_the_multiword_set_aliases_are_untouched(self):
        words = {a.lower() for spec in _SCHEMA["operations"]
                 for a in spec.get("aliases", {}).get("en", [])}
        assert {"set latex name", "set notation", "set symmetry"} <= words


class TestSetToATypeStaysARefusal:
    """`set X to <型>` は断りのままである。**実測に基づく決定**——宣言の繋辞
    （`decl_stmt[en]`）に素の `to` を足せば `set x to a scalar` は通るが、
    同時に英語でいちばん多い前置詞を宣言の読み手の前に置くことになり、

        convert E1 to polar coordinates   → `E1` をスカラー宣言
        expand E1 to third order in x     → `exp` という名前を宣言
        transform E1 to the frequency domain / change E1 to spherical coordinates

    が宣言として読まれる（実測 2026-09-10）。研究者には動く綴りが2つある
    （`let x be a scalar` / `declare x as a scalar`）ので、足す側が損である。"""

    def test_the_declaration_copula_lives_in_the_shared_fragment(self):
        r"""繋辞は 2026-09-12 に断片 `en_decl_copula` へ畳んだので、ここが見るのは
        「`decl_stmt` がその断片を読んでいること」だけである——**語彙そのもの**
        （素の `to` が無いこと）は `tests/test_english_declaration_copula.py` が
        1箇所で数えており、そちらは全ての宣言の鍵に効く。"""
        d = json.loads((pathlib.Path(__file__).resolve().parents[1]
                        / "i18n" / "nl_patterns.json").read_text(encoding="utf-8"))
        assert "{{en_decl_copula}}" in d["decl_stmt"]["en"], d["decl_stmt"]["en"]


# ---------------------------------------------------------------------------
# **研究者が通る経路で測る。** 上の各クラスは綴りと判別子を直接見ており、
# 「当たる」ことしか言えない。どの規則が先に掴むかはパイプラインが決めるので、
# 横取りの有無は打ってみないと分からない（`\sum_{\langle i,j\rangle}` が
# 一般経路では silent-wrong に見えて実際は格子の枝が取る、という前例がある）。
# ---------------------------------------------------------------------------
import asyncio  # noqa: E402

import ui.notebook_bridge as nb  # noqa: E402
from notebook.runtime.llm_supervisor import LLMSupervisor  # noqa: E402
from notebook.runtime.operation_validator import OperationValidator  # noqa: E402
from ui.interpret_pipeline import (  # noqa: E402
    InterpretRequest, InputToken, run_interpret_pipeline)


def _typed(lang, lines):
    """指示を順に打って、最後の応答と、そのとき在る式の一覧を返す。"""
    nb.start_engine(title="vd", mode="standard", language=lang,
                    domains=["differential_equations"])
    try:
        eng = nb.get_engine()
        sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                            domains=["differential_equations"],
                            primary_backend="maxima")
        resp = None
        for line in lines:
            resp = asyncio.get_event_loop().run_until_complete(
                run_interpret_pipeline(
                    InterpretRequest(tokens=[InputToken(type="text", value=line)]),
                    eng, sup))
        exprs = [(e.display_id, e.canonical or "", e.latex or "")
                 for e in eng.expressions.values()]
        return resp, exprs
    finally:
        nb.stop_engine()


@pytest.mark.parametrize("lang,text,canonical,latex", [
    ("ja", "x を 3 とする", "x = 3", "x = 3"),
    ("ja", "x をゼロとする", "x = 0", "x = 0"),
    ("ja", "x を零とする", "x = 0", "x = 0"),
    ("en", "set x to 3", "x = 3", "x = 3"),
    ("en", "set x to zero", "x = 0", "x = 0"),
    ("en", "let x be zero", "x = 0", "x = 0"),
])
def test_a_written_value_is_registered_as_an_equation(lang, text, canonical, latex):
    """**画面と計算が同じことを言う。** 綴りごと畳んでいるので、`latex` にも
    `ゼロ`/`zero` は残らない（残ると数式内に日本語が入り組版が壊れる)。"""
    resp, exprs = _typed(lang, ["y = a x + b", text])
    assert resp.get("status") in ("registered", "completed"), resp
    assert (exprs[-1][1], exprs[-1][2]) == (canonical, latex), exprs


@pytest.mark.parametrize("lang,text", [
    ("ja", "x を a + b とする"), ("en", "set x to a + b"), ("en", "let x be a + b"),
])
def test_an_expression_value_reaches_h1(lang, text):
    resp, exprs = _typed(lang, ["y = a x + b", text])
    assert resp.get("status") == "registered", resp
    assert exprs[-1][2] == "x = a + b", exprs


@pytest.mark.parametrize("lang,text", [
    ("ja", "x をスカラーとする"), ("en", "let x be a scalar"),
    ("en", "declare x as a scalar"),
])
def test_the_type_complement_still_declares(lang, text):
    resp, _ = _typed(lang, ["y = a x + b", text])
    assert resp.get("status") == "confirmed", resp
    assert resp.get("operation") == "define_scalar_parameter", resp


@pytest.mark.parametrize("lang,text", [
    ("en", "set the right-hand side of E1 to zero"),
    ("ja", "E1 の右辺をゼロとする"),
    ("ja", "E1 の右辺を零とする"),
    ("en", "set the right-hand side of E1 to 0"),
])
def test_the_right_hand_side_is_set_to_the_number(lang, text):
    """錨が要る理由でもある——錨が無いと `value_definition` が `E1` を名前として
    拾い、H-5z を横取りする。"""
    resp, exprs = _typed(lang, ["y = a x + b", text])
    assert resp.get("status") == "completed", resp
    assert exprs[-1][1] == "a*x + b = 0", exprs
    assert exprs[-1][2] == "a x + b = 0", exprs


@pytest.mark.parametrize("lang,text,rhs", [
    ("en", "set x = zero in E1", "0"),
    ("en", "let x = 2 in E1", "2"),
    ("en", "put x = 2 in E1", "2"),
    ("en", "evaluate E1 at x = zero", "0"),
    ("ja", "E1 の x をゼロに置換して", "0"),
])
def test_the_substitution_family_is_not_stolen(lang, text, rhs):
    """`value_definition` は動詞を剥がすので、掴んでしまうと代入が式登録に化ける
    ——実測で `let x = 2 in E1` が canonical `2*in(a*x+b)` として**登録**された。"""
    resp, _ = _typed(lang, ["y = a x + b", text])
    assert resp.get("status") == "completed", resp
    assert resp.get("operation") == "substitute", resp
    assert (resp.get("operation_args") or resp.get("args") or {}).get("rhs", rhs) == rhs, resp


@pytest.mark.parametrize("lang,text", [
    ("en", "set E1 to zero"), ("en", "set E1 to 0"), ("ja", "E1 を 0 とする"),
])
def test_a_bare_expression_id_is_refused_not_answered_in_cas_words(lang, text):
    """「方程式全体を0にする」のか「右辺を0にする」のかは書かれていない——
    決めていないことは断る。直す前は `define_function` の裸の別名 `set` が
    掴んで、生の `Maxima evaluation error` を返していた（Core Principle #1）。"""
    resp, _ = _typed(lang, ["y = a x + b", text])
    assert resp.get("status") == "failed", resp
    assert "Maxima" not in (resp.get("error") or ""), resp
