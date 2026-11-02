"""「値を**置く**」族 —— 日英で同じ言い方が、どちらも読まれていなかった。

十問目（`tests/test_an_instruction_is_not_a_formula.py`）で代入の綴り空間を測った
残りがこれである。実測 2026-09-05、`E1: f = a*x + b` に x=2 を置く要求:

    E1 の x を 2 とおいて / とおく / と置いて / と置く   → 4綴りとも failed
    set x = 2 in E1 / let … / put … / set x to 2 in E1 → 4綴りとも failed
    In E1, set x = 2 / In E1, let x = 2                → 2綴りとも failed

**日本語で置換が読めるのは動詞が「置き換え」「置換」「代入」のときだけ**で、
`とおく`（＝値を置く）は末尾の動詞列に無かった。英語側は `set`/`let`/`put` の
どれも `subst_replace` の4枝（replace…with / substitute…for、各2語順）に無い。
`fp_subst` は**自由パラメータ**専用で、`x` がそれでなければ正しく降りる。

**入れなかったものにも理由がある:**

- `E1 の x を 2 とする` —— `とする` は宣言の語尾でもあり、コーパスの実在行
  `E4 の初期条件を x(0)=1, x'(0)=0 とする` が lhs=`初期条件` として掴まれる。
  下の負例で固定してある。**2026-09-10（起票 SUBST-TOSHITE）に `=` の形だけ
  入れた**——判別子は区切りの記号で、`を` は助詞（後ろに型の名詞が来うる）、
  `=`／`＝`／`→` は書かれた数式である。`を` の形は依然として入れない。
- `with x = 2 in E1` —— `with` は前置詞であって命令の動詞ではない。族から外した。

`put` は `instruction_marker[en]` にも足してある。無いと H-1 が
`put x = 2 in E1` を**式として登録**する（十問目とまったく同じ形）。
"""
import asyncio

import pytest

import ui.notebook_bridge as nb
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.nl_patterns import nlp, nlp_any, pick_group
from notebook.runtime.operation_validator import OperationValidator
from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline
from ui.interpret_rules import _try_rule_short_circuits  # noqa: F401  (module import)


#: 「x を 2 に」と言っている綴り。日英そろえてある。
SPELLINGS = [
    ("ja", "E1 の x を 2 とおいて"),
    ("ja", "E1 の x を 2 とおく"),
    ("ja", "E1 の x を 2 と置いて"),
    ("ja", "E1 の x を 2 と置く"),
    ("ja", "E1 で x を 2 とおいて"),
    ("ja", "E1 の x = 2 とおいて"),
    # `=` ＋ `として`/`とする`（2026-09-10、起票 SUBST-TOSHITE）。**助詞の形は
    # 入れない**——`を` の後ろには型の名詞が来うるので宣言と衝突する。
    ("ja", "E1 で x = 2 として"),
    ("ja", "E1 で x = 2 とする"),
    ("ja", "E1 の x = 2 として"),
    ("en", "set x = 2 in E1"),
    ("en", "let x = 2 in E1"),
    ("en", "put x = 2 in E1"),
    ("en", "set x to 2 in E1"),
    ("en", "In E1, set x = 2"),
    ("en", "In E1, let x = 2"),
]


#: **本体の読みをそのまま呼ぶ。** ここには「同じ読み方」と称する写しが在ったが、
#: 接尾辞の判定が `isalpha()` で、本体（`pick_group`）の `isalnum()` とずれていた
#: ——数字の接尾辞（`eid2`）を足した 2026-09-10 に、**本体は読めるのにテストだけが
#: 読めない**という形で出た。写しは答えが2つある状態なので、消して本体を呼ぶ。
_pick = pick_group


class TestThePatternReadsTheSameThreeThings:
    @pytest.mark.parametrize("lang,text", SPELLINGS)
    def test_lhs_rhs_and_target(self, lang, text):
        m = (nlp("subst_replace", lang).search(text)
             or nlp("subst_replace_alt", "ja").search(text))
        assert m is not None, f"[{lang}] どの綴りにも当たらない: {text!r}"
        assert (_pick(m, "lhs"), _pick(m, "rhs"), _pick(m, "eid")) == ("x", "2", "E1"), \
            (text, m.groupdict())


class TestWhatWasDeliberatelyLeftOut:
    """入れなかったものは、入れられない理由があって入れていない。"""

    def test_a_declaration_tail_is_not_a_substitution(self):
        """`とする` を足すと、この実在のコーパス行が lhs=`初期条件` になる。"""
        line = "E4 の初期条件を x(0)=1, x'(0)=0 とする"
        assert nlp("subst_replace_alt", "ja").search(line) is None, \
            "宣言・条件指定の文が置換として掴まれている"

    def test_the_plain_to_suru_form_is_not_claimed(self):
        assert nlp("subst_replace_alt", "ja").search("E1 の x を 2 とする") is None

    def test_a_preposition_is_not_an_instruction_verb(self):
        assert not nlp_any("instruction_marker", "with x = 2 in E1")


class TestTheEqualsFormOfToSuru:
    r"""**`として` は `=` の形だけ入れた**（起票 SUBST-TOSHITE、2026-09-10）。

    `E1 で x = 2 として` は英語の鏡 `set x = 2 in E1` が通るのに落ちていた——
    日英の割れ。`として`/`とする` は宣言・H-5z・初期条件・`fp_subst` と共有された
    語尾なので既存の枝に足すと横取りになるが、**区切りの記号が判別する**:
    `を` は助詞で後ろに型の名詞が来うる（`スカラー`・`正の実数`・`r の共役運動量`）、
    `=` は書かれた数式で、型の名詞がその形を取ることはない。
    """

    @pytest.mark.parametrize("text", [
        "E1 で x = 2 として", "E1 で x = 2 とする",
        "E1 で x = 2 とします", "E1 の x = 2 として",
        "E1 に対して x = 2 として", "E1 で x ＝ 2 として",
    ])
    def test_the_equals_form_is_claimed(self, text):
        m = nlp("subst_replace_alt", "ja").search(text)
        assert m is not None, text
        assert pick_group(m, "eid") == "E1", text
        assert pick_group(m, "lhs").strip() == "x", text
        assert pick_group(m, "rhs").strip() == "2", text

    @pytest.mark.parametrize("text", [
        # 助詞の形は入れない——ここが宣言・H-5z・初期条件と衝突する
        "E1 の x を 2 とする", "E1 の x を 2 として",
        "E1 の右辺を 0 とする", "E20 の右辺を 0 とする",
        "E4 の初期条件を x(0)=1, x'(0)=0 とする",
    ])
    def test_the_particle_form_is_still_left_out(self, text):
        assert nlp("subst_replace_alt", "ja").search(text) is None, text

    @pytest.mark.parametrize("text", [
        # **式番号が要る。** 無いと文の途中から当たり、定義や初期条件を横取りする
        # ——`H = J S_1 S_2 とする` は H-1 の定義であって代入ではないし、実在の
        # コーパス行 `E4 の初期条件を x(0)=1, x'(0)=0 とする` は `x(0)=1` の `=` で
        # 当たってしまう（実測 2026-09-10、式番号を任意にする変異で出た）。
        "H = J S_1 S_2 とする", "x = 2 として", "x = 2 とする",
        "\\omega = 2 \\pi f とする",
    ])
    def test_it_needs_an_expression_id(self, text):
        assert nlp("subst_replace_alt", "ja").search(text) is None, text

    @pytest.mark.parametrize("text", [
        "x をスカラーとする", "\\gamma を正の実数とする",
        "p_r を r の共役運動量とする", "a を光子の生成演算子とする",
        "G を行列とする", "x を時間依存とする",
    ])
    def test_a_declaration_is_untouched_even_with_the_injected_scaffold(self, text):
        r"""**素朴に足したときに壊れるのはここで、コーパスの生の文を走査しても
        出ない。** 式番号を書いていない文には足場 `E1 に対して ` が前置される
        （`interpreter.expr_ref_prefix`）ので、この鍵から見ると E-id つきの文に
        見える。実測 2026-09-10: 素朴版は足場つきで**実在63行**を取り、その22件が
        `define_scalar_parameter`、14件が `declare_time_dependent` だった。
        `=` の枝は 0 行である。
        """
        for probe in (text, "E1 に対して " + text):
            assert nlp("subst_replace_alt", "ja").search(probe) is None, probe


class TestPutIsAnInstructionVerb:
    """無いと H-1 が式として登録する —— 十問目と同じ形。"""

    def test_put_is_recognised(self):
        assert nlp_any("instruction_marker", "put x = 2 in E1")

    def test_h1_declines_it(self):
        from ui.interpret_pipeline import _is_valid_h1_input
        assert _is_valid_h1_input("put x = 2 in E1") is None
        assert _is_valid_h1_input("set x = 2 in E1") is None
        assert _is_valid_h1_input("let x = 2 in E1") is None


class TestTheBranchSuffixIsNotEnumerated:
    """`_sr_pick` が枝を列挙していたので、枝を足すとパターンだけ直る形だった。"""

    def test_a_new_branch_is_read_without_being_listed(self):
        import re
        m = re.compile(r"(?P<lhsZ>x)=(?P<rhsZ>2) in (?P<eidZ>E\d+)").search("x=2 in E7")
        assert (_pick(m, "lhs"), _pick(m, "rhs"), _pick(m, "eid")) == ("x", "2", "E7")

    def test_an_absent_stem_is_not_invented(self):
        import re
        m = re.compile(r"(?P<lhsA>x)").search("x")
        assert _pick(m, "eid") is None


def _run(eng, sup, text):
    return asyncio.get_event_loop().run_until_complete(
        run_interpret_pipeline(
            InterpretRequest(tokens=[InputToken(type="text", value=text)]), eng, sup))


@pytest.fixture(scope="module", params=["ja", "en"])
def session(request):
    """`f = a*x + b` を登録した E1 を1つ。以降の代入は毎回 E1 を対象にする。"""
    lang = request.param
    nb.start_engine(title="setting_a_value", mode="standard", language=lang,
                    domains=["classical_mechanics"])
    eng = nb.get_engine()
    sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                        domains=["classical_mechanics"], primary_backend="maxima")
    _run(eng, sup, "a, b をスカラーとする" if lang == "ja" else "Define a, b as scalars")
    _run(eng, sup, "f = a*x + b")
    yield lang, eng, sup
    nb.stop_engine()


def test_the_value_reaches_the_cas(session):
    """画面まで: どの綴りでも `b+2*a` が返り、余分な式を作らないこと。

    単体でパターンだけ見ても足りない —— 落ちていた綴りの一部は「失敗」ではなく
    **式として登録**されていたので、値と作られた式の数まで見ないと再発が分からない。
    """
    lang, eng, sup = session
    for spell_lang, text in SPELLINGS:
        if spell_lang != lang:
            continue
        n_before = len(eng.expressions)
        resp = _run(eng, sup, text)
        assert resp.get("status") == "completed", (text, resp.get("status"),
                                                   resp.get("error"))
        assert (resp.get("canonical") or "").replace(" ", "") == "b+2*a", (text, resp)
        assert len(eng.expressions) - n_before == 1, (text, "式が余分に登録された")
