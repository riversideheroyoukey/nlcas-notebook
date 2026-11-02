r"""代入の動詞は1つの表（NL-GRAMMAR ①、2026-09-12）。

**この族は、繋辞の件とは構造が違う。** 形（frame）ごとに語彙が違うのは**理由の
ある違い**で、統一すると文法的でない綴りを受けることになる——`apply`／`適用` は
式→式には合うが値の代入には合わず、`plug 3 for x` は英語として言わない。実測して
理由の無いものだけを畳んだ。

実測（形4種 × 動詞 × 活用 × 日英 = 85通り、**コーパスの文の動詞だけ差し替え**）::

    畳む前 63/85 が正しい操作に届く
    畳んだ後 80/85
    残る5件は `plug 3 for x` / `apply x with 3 in E1` の類（**理由のある拒否**）

畳んだのは3つ:

1. **活用が鍵ごとに違った。** `substitute_into[ja]` は
   `(?:代入|置換|置き換え|入れ[てるた]?)(?:して|する)?$` と綴っており、**`入れ` にだけ
   活用が書いてあって `置き換え` には無い**。行末に錨があるので `置き換えて`——
   いちばん普通のテ形、しかも**コーパスが形Bで実際に書いている形**——が落ちていた。
   同じ語を演算子の族（`op_subst_verb_ja`）は `置き換え[てるた]?|置き換える` として
   受ける。**同じ問いに2つの答え。** `ja_subst_verb` に畳み、活用は既存の
   `ja_suru_infl` を読む。
2. **英語の2つの枠が動詞を1つずつしか持っていなかった。** 演算子の族は
   `substitute … for …` と `replace … with …` だけで、`replace … for …`（形Aは受ける）
   と `substitute … with …`（形Bは受ける）が落ちていた。`en_subst_verb` を両方の枠が
   読む。
3. **式→式の断り文が入力に当たっていなかった。** `replace E2 in E1` /
   `E1 に E2 を置換して` は「操作 '代入する' に解釈しましたが、必要な引数が不足して
   います: ['lhs', 'rhs']」——**書いた文に lhs も rhs も無い**ので、別の問いについての
   断りである（`substitute E2 in E1` は通る。違いは動詞だけ）。動詞表を共有させて
   受けるようにした。**向きは助詞・前置詞が決める**（`を`=出どころ／`に`=行き先、
   `into`/`in`/`to`）ので、動詞を広げても入れ替わらない。

**`入れ` は共有の表に入れない——ただし、以前そこに書いてあった理由は実測で
再現しなかった。** 註は「一般的な動詞なので共有一覧に足すと横取りする」と言って
いたが、共有表へ入れて8綴りを打つと**横取りは0件**で、変わるのは
`E1 に E2 を入れて` が通るようになる点だけだった。局所に留めているのは**用心で
あって実測ではない**——`入れ` はコーパス0行・シナリオの指示行にも0件なので、
6鍵が読む表へ一般動詞を移す判断には証拠が足りない（通らない綴りを文法に入れない）。
移すなら、`入れ` を含む指示をシナリオに1行足してから測ること。

**綴り空間の測定を先に取った。** `replace` は3鍵で**コーパス0行**、`apply`/`plug`/
`適用`/`置き換える` も0行——つまりこの部分空間にコーパス差分は構造的に無力で、
打ってみるまで何も言えない（CORPUS-WORKING-ONLY）。
"""
import asyncio
import json
import pathlib
import re

import pytest

import ui.notebook_bridge as nb
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.nl_patterns import fragment, nlp
from notebook.runtime.operation_validator import OperationValidator
from ui.interpret_pipeline import (InputToken, InterpretRequest,
                                   run_interpret_pipeline)

_PATTERNS = json.loads(
    (pathlib.Path(__file__).resolve().parents[1] / "i18n" / "nl_patterns.json")
    .read_text(encoding="utf-8"))


class TestTheVerbTableIsShared:

    @pytest.mark.parametrize("word", [
        "代入", "代入する", "代入します", "代入して", "代入せよ", "代入しろ",
        "置換", "置換する", "置換して", "置換します",
        "置き換え", "置き換える", "置き換えて", "置き換えた", "置き換えます",
    ])
    def test_every_inflection_is_in_one_table(self, word):
        assert re.fullmatch(fragment("ja_subst_verb"), word), word

    def test_the_general_verb_is_not_in_the_shared_table(self):
        r"""`入れ` は `substitute_into` の枠でだけ受ける。**この検査が固定して
        いるのは決定であって、その理由ではない**——「共有表に足すと横取りする」
        という以前の理由は測ると再現しなかった（横取り0）。留めている理由は
        「コーパスにもシナリオにも1行も無い綴りを、6鍵が読む表へ移さない」。"""
        assert not re.fullmatch(fragment("ja_subst_verb"), "入れて")
        assert nlp("substitute_into", "ja").search("E1 の x に 3 を入れて")

    def test_the_inflections_come_from_the_shared_list(self):
        assert "{{ja_suru_infl}}" in _PATTERNS["_fragments"]["ja_subst_verb"]

    @pytest.mark.parametrize("key", [
        "substitute_into", "scalar_substitute_op",
        "operator_substitute_single", "operator_substitute_multi",
    ])
    def test_every_japanese_key_reads_the_table(self, key):
        assert "{{ja_subst_verb}}" in _PATTERNS[key]["ja"], key

    @pytest.mark.parametrize("key", [
        "operator_substitute_in_form", "operator_substitute_multi",
    ])
    def test_both_english_frames_read_the_table(self, key):
        en = _PATTERNS[key]["en"]
        assert en.count("{{en_subst_verb}}") == 2, (key, en)

    def test_the_apply_verbs_are_the_frame_specific_extra(self):
        r"""`apply`/`plug`/`適用` は**式→式の枠だけ**の語で、共有の表には無い
        （`apply x with 3 in E1` は英語ではない）。"""
        assert fragment("en_subst_verb") == "(?:substitute|replace)"
        assert "apply" in fragment("en_subst_apply_verb")
        assert "適用" in fragment("ja_subst_apply_verb")
        assert "適用" not in fragment("ja_subst_verb")


# ---------------------------------------------------------------------------
# **研究者が通る経路で測る。** 上は「当たる」ことしか言えない。
# ---------------------------------------------------------------------------
def _ask(lang, domain, setup, text):
    async def go():
        nb.start_engine(title="subst-verb", mode="standard", language=lang,
                        domains=[domain])
        try:
            engine = nb.get_engine()
            sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                                domains=[domain],
                                primary_backend=engine.session.primary_backend)
            for line in setup:
                await run_interpret_pipeline(
                    InterpretRequest(tokens=[InputToken(type="text", value=line)]),
                    engine, sup)
            return await run_interpret_pipeline(
                InterpretRequest(tokens=[InputToken(type="text", value=text)]),
                engine, sup)
        finally:
            nb.stop_engine()
    return asyncio.run(go())


_VALUE = ["f = a x + b"]
_CHAIN = ["f = a q + b", "q = 2 t"]


class TestTheSpellingReachesTheOperation:
    """畳む前は 63/85、いまは 80/85。ここに並ぶのは**変わった17件**の代表。"""

    @pytest.mark.parametrize("text", [
        # 形A——活用が鍵ごとに違っていた分
        "E1 の x に 3 を置き換えて", "E1 の x に 3 を置き換える",
        "E1 の x に 3 を置き換えます", "E1 の x に 3 を代入します",
        "E1 の x に 3 を置換します", "E1 の x に 3 を入れます",
    ])
    def test_the_value_substitution_takes_every_inflection(self, text):
        r = _ask("ja", "classical_mechanics", _VALUE, text)
        assert r.get("operation") == "substitute", (text, r)
        assert (r.get("canonical") or "").replace(" ", "") == "b+3*a", (text, r)

    @pytest.mark.parametrize("lang,text", [
        # 形C——「必要な引数が不足しています: ['lhs','rhs']」で断られていた
        ("ja", "E1 に E2 を置換して"), ("ja", "E1 に E2 を置き換えて"),
        ("ja", "E2 を E1 に置換して"),
        ("en", "replace E2 in E1"), ("en", "replace E2 into E1"),
        ("en", "in E1, replace E2"),
    ])
    def test_the_expression_into_expression_frame_takes_replace(self, lang, text):
        r = _ask(lang, "classical_mechanics", _CHAIN, text)
        assert r.get("operation") == "substitute", (text, r)
        assert (r.get("canonical") or "").replace(" ", "") == "2*a*t+b", (text, r)

    def test_the_direction_comes_from_the_particle_not_the_verb(self):
        r"""動詞を広げても入れ替わらないこと——`を`／`に` が出どころと行き先を決める。"""
        a = _ask("ja", "classical_mechanics", _CHAIN, "E1 に E2 を置換して")
        b = _ask("ja", "classical_mechanics", _CHAIN, "E2 を E1 に置換して")
        assert (a.get("canonical") or "").replace(" ", "") == "2*a*t+b", a
        assert (b.get("canonical") or "").replace(" ", "") == "2*a*t+b", b


class TestWhatStaysRefusedIsGrammar:
    r"""**形ごとの語彙の違いは、多くが理由のある違いである。** 統一すると英語として
    言わない綴りを受けることになるので、ここは畳まない。"""

    @pytest.mark.parametrize("text", [
        "in E1, plug 3 for x", "in E1, apply 3 for x", "apply x with 3 in E1",
    ])
    def test_the_apply_verbs_do_not_fit_a_value_substitution(self, text):
        r = _ask("en", "classical_mechanics", _VALUE, text)
        assert r.get("operation") != "substitute", (text, r)

    @pytest.mark.parametrize("text", ["apply E2 to E1", "plug E2 into E1"])
    def test_but_they_do_fit_the_expression_frame(self, text):
        r = _ask("en", "classical_mechanics", _CHAIN, text)
        assert r.get("operation") == "substitute", (text, r)


class TestTheOperatorFrameAgreesWithTheValueFrame:
    r"""演算子の族は英語の枠を2つ持ちながら、**それぞれ動詞を1つずつ**しか
    受けていなかった——`replace … for …` は形Aが受け、`substitute … with …` は
    形Bが受けるのに、ここでは両方落ちていた。Julia セッションなので代表2件。"""

    _SETUP = ["define c as an electron annihilation operator", "H = c'(1) c(1)"]

    @pytest.mark.parametrize("text", [
        "in E1, replace c(2) for c(1)",
        "in E1, substitute c(1) with c(2)",
        "in E1, substitute c(2) for c(1)",
        "in E1, replace c(1) with c(2)",
    ])
    def test_both_verbs_in_both_frames(self, text):
        r = _ask("en", "condensed_matter", self._SETUP, text)
        assert r.get("operation") == "operator_substitute", (text, r)
