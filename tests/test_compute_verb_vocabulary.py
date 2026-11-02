"""「これは計算せよという要求か」—— 動詞の一覧が ja 9箇所・en 8箇所にあった。

worklist の `求め`（10キー・42行）から。数えると、「計算せよ」を表す動詞の一覧が
これだけの場所にあり、**それぞれ違う部分集合**しか知らなかった（実測 2026-09-05）:

| | 知っていた語 |
|---|---|
| `verb_tail_strip[ja]` | 取り出し・算出・計算・求め・出し・取り（**評価・求めよ は無い**） |
| `matrix_trigger[ja]` / `pb_bracket_verb_guard[ja]` | `求め` と `計算` だけ |
| `binary_bracket_phrase[ja]` | 計算・求め・出・とっ・取っ |
| `evaluate_extremum[ja]` | 評価・計算・求め |
| `inv_transform[ja]` / `canon_constraint[ja]` / `metric_constraint[ja]` | 求め・出し・導出・作っ・作る（＋生成） |
| `solve_single_novar[ja]` | 求め・出し |
| `matrix_trigger[en]` | find・compute・calculate・get |
| `binary_bracket_phrase[en]` | take・compute・calculate・find・evaluate・give me |
| `evaluate_extremum[en]` | evaluate・compute |
| `canon_constraint[en]` / `metric_constraint[en]` | derive・find（＋build） |
| `solve_single_novar[en]` | `find the solution of` と直書き |
| `operand_stopwords[en]` | take/apply/compute/calculate/find/get/obtain |

`determine` はどこにも無く、`求めよ`・`評価して` は `verb_tail_strip` の外に無い。
4つの族 × 動詞を実パイプラインで測ると、基準の動詞と同じ操作に届かない綴りが
**ja 15/28・en 19/32** あった。

**2つのクラスに分けてある。** 「計算せよ」（値を出す）と「導出・構築せよ」
（新しい対象を作る）は別で、`E1 の逆行列を作って` / `build the inverse of E1` は
誰も書かない。族が受けるクラスを名指す形にした——入れなかったものにも理由がある。

**同じ測定で silent-wrong が1件。** `derive` は `differentiate` の別名だが、
系自身は `canon_constraint[en]` / `metric_constraint[en]` で「導出する」の意味で
使っている。`derive the inverse of E1` は

    differentiate {expr: E1, var: 'inverse'}   ← **completed**

——逆行列を頼んで「`inverse` という変数で微分」が成功として返っていた。`inverse`
（裸の名詞）が `inverse_matrix` の別名に**無かった**ためで、`inv`・`invert`・
`inverse matrix` は在った。足すと両方の操作が名指されるので、系は**訊く**
（ALIAS-COLLISION-UNREACHABLE の形）。
"""
import asyncio

import pytest

import ui.notebook_bridge as nb
from notebook.runtime.llm_supervisor import LLMSupervisor, _tokenize
from notebook.runtime.nl_patterns import nlp, fragment
from notebook.runtime.operation_validator import OperationValidator
from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline


COMPUTE_JA = ["求めて", "求める", "求めよ", "計算して", "算出して", "出して",
              "評価して", "取り出して"]
COMPUTE_EN = ["find", "compute", "calculate", "get", "obtain", "determine",
              "evaluate", "work out"]
DERIVE_JA = ["導出して", "作って", "生成して", "構成して"]
DERIVE_EN = ["derive", "build", "construct", "form"]


class TestTheTwoClassesAreOneListEach:
    @pytest.mark.parametrize("verb", COMPUTE_JA)
    def test_every_compute_verb_ja(self, verb):
        import re
        assert re.fullmatch(fragment("compute_verb_ja"), verb), verb

    @pytest.mark.parametrize("verb", COMPUTE_EN)
    def test_every_compute_verb_en(self, verb):
        import re
        assert re.fullmatch(fragment("compute_verb_en"), verb, re.I), verb

    @pytest.mark.parametrize("verb", DERIVE_JA)
    def test_every_derive_verb_ja(self, verb):
        import re
        assert re.fullmatch(fragment("derive_verb_ja"), verb), verb

    @pytest.mark.parametrize("verb", DERIVE_EN)
    def test_every_derive_verb_en(self, verb):
        import re
        assert re.fullmatch(fragment("derive_verb_en"), verb, re.I), verb

    def test_the_two_classes_are_kept_apart(self):
        """`E1 の逆行列を作って` は誰も書かない —— 分けてあることを固定する。"""
        import re
        for v in DERIVE_JA:
            assert not re.fullmatch(fragment("compute_verb_ja"), v), v
        for v in DERIVE_EN:
            assert not re.fullmatch(fragment("compute_verb_en"), v, re.I), v

    def test_the_tail_strip_exposes_the_noun(self):
        """動詞の尾を剥がして名詞の別名を露出させること。

        **並び順は効かない。** `verb_tail_strip` のコメントは「`取り出して` を
        自身の接尾辞 `出して` より先に置くため長い順」と書いていたが、実測すると
        偽——`$` アンカーと最左一致・バックトラックが同じ結果を出す（順序を
        逆にする変異は落ちず、それが正しい）。長い順は可読性のため。
        """
        assert _tokenize("E1 の実部を取り出して") == ["E1", "実部"]
        assert _tokenize("E1 の固有値を求めて") == ["E1", "固有値"]
        assert _tokenize("E1 の解を計算して") == ["E1", "解"]


class TestEverySiteAsksTheOneList:
    """ガードとパターンが同じ一覧を見る —— 見ていなければ、その族だけ語彙が狭い。"""

    @pytest.mark.parametrize("verb", COMPUTE_JA)
    def test_the_matrix_guard(self, verb):
        assert nlp("matrix_trigger", "ja").search(f"E1 の固有値を{verb}"), verb

    @pytest.mark.parametrize("verb", COMPUTE_EN)
    def test_the_matrix_guard_en(self, verb):
        assert nlp("matrix_trigger", "en").search(f"{verb} the eigenvalues of E1"), verb

    @pytest.mark.parametrize("verb", COMPUTE_JA)
    def test_the_solve_pattern(self, verb):
        assert nlp("solve_single_novar", "ja").search(f"E1 の解を{verb}"), verb

    @pytest.mark.parametrize("verb", COMPUTE_EN)
    def test_the_solve_pattern_en(self, verb):
        assert nlp("solve_single_novar", "en").match(f"{verb} the solution of E1"), verb

    @pytest.mark.parametrize("verb", DERIVE_JA + ["求めて", "算出して"])
    def test_the_constraint_families_take_both_classes(self, verb):
        """制約と変換は**導出クラスも**受ける（そちらが自然な言い方だから）。"""
        assert nlp("metric_constraint", "ja").search(f"E1 の計量条件を{verb}"), verb
        assert nlp("canon_constraint", "ja").search(f"E1 の正準拘束を{verb}"), verb
        assert nlp("inv_transform", "ja").search(f"E1 の逆変換を{verb}"), verb

    @pytest.mark.parametrize("verb", DERIVE_EN + ["find", "obtain", "determine"])
    def test_the_constraint_families_en(self, verb):
        assert nlp("metric_constraint", "en").search(
            f"{verb} the metric constraints of E1"), verb
        assert nlp("canon_constraint", "en").search(
            f"{verb} the canonical constraints of E1"), verb


class TestTheBareNounIsAnAlias:
    """`inv` / `invert` / `inverse matrix` は在って、裸の `inverse` だけ無かった。"""

    def test_inverse_names_the_operation(self):
        v = OperationValidator()
        assert v.operations_for_alias("inverse") == ["inverse_matrix"]

    def test_it_collides_with_nothing(self):
        v = OperationValidator()
        assert len(v.operations_for_alias("inverse")) == 1


def _run(eng, sup, text):
    return asyncio.get_event_loop().run_until_complete(
        run_interpret_pipeline(
            InterpretRequest(tokens=[InputToken(type="text", value=text)]), eng, sup))


@pytest.fixture(scope="module", params=["ja", "en"])
def session(request):
    lang = request.param
    nb.start_engine(title="compute_verb", mode="standard", language=lang,
                    domains=["classical_mechanics"])
    eng = nb.get_engine()
    sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                        domains=["classical_mechanics"], primary_backend="maxima")
    _run(eng, sup, "A = [2, 1; 1, 2]")
    yield lang, eng, sup
    nb.stop_engine()


def test_every_compute_verb_reaches_the_same_operation(session):
    """画面まで —— 動詞を替えても同じ操作・同じ値。"""
    lang, eng, sup = session
    verbs = COMPUTE_JA if lang == "ja" else COMPUTE_EN
    for noun, want in (("固有値", "eigenvalues") if lang == "ja" else ("eigenvalues", "eigenvalues"),
                       ("逆行列", "inverse_matrix") if lang == "ja" else ("inverse", "inverse_matrix")):
        for verb in verbs:
            text = f"E1 の {noun}を{verb}" if lang == "ja" else f"{verb} the {noun} of E1"
            resp = _run(eng, sup, text)
            assert resp.get("status") == "completed", (text, resp.get("status"),
                                                       resp.get("error"))
            assert resp.get("operation") == want, (text, resp.get("operation"))


def test_a_sentence_naming_two_operations_asks(session):
    """`derive the inverse of E1` は微分と逆行列の両方を名指している —— 訊く。"""
    lang, eng, sup = session
    if lang != "en":
        pytest.skip("英語の別名衝突")
    resp = _run(eng, sup, "derive the inverse of E1")
    assert resp.get("status") == "needs_confirmation", resp.get("status")
    cands = {c["operation"] for c in (resp.get("candidates") or [])}
    assert {"differentiate", "inverse_matrix"} <= cands, cands
    assert resp.get("confirmation_prompt"), "何を候補にしたか言っていない"
