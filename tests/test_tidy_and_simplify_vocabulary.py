r"""「簡約せよ」「整理せよ」——同じ指示に、いくつの表が別々に答えていたか。

実測 2026-09-07（worklist の `整理`、6行・7鍵）。`整理` は7鍵**すべてが自前で
綴っており**、fragment を経由するものが1つも無かった。

三十七問目で「`整理` は `collect` とも競合する**曖昧語**なので
`simplify_verb_ja` には入れず、曖昧でない場所（両辺）とダイアログにだけ置く」と
決めてある。ここでの問いはその決定の是非ではなく、**その決定が7箇所で一貫して
いるか**——答えは「していない」だった。

| 打った文 | 直す前 |
|---|---|
| `E2 の右辺を**簡約**して` | **failed**（`整理` なら通る＝**曖昧語のほうが強い**） |
| `E2 の右辺を単純化／簡略化／簡単に…` | **failed** |
| `**reduce** the right-hand side of E2` | **式として登録される**（status=registered） |
| `E1 を E3 の拘束のもとで**単純化**して` | **failed**（gate は通り、本体で落ちる） |
| `collect operators in E3` ほか英語6綴り | **1つも当たらない** |

4箇所が同じ語彙を別々に持っていた:

| どこ | 簡約の語彙 |
|---|---|
| `simplify_verb_ja` / `_en`（fragment） | 簡約・単純化・簡単に・簡略化・簡略 / simplify・reduce |
| `rhs_op` | **簡約系が1語も無い**（整理・計算・評価…）／ en は `simplify` のみ |
| `_RHS_OP_MAP`（keyword→操作） | `整理`・`simplify` のみ |
| `apply_constraints` | 簡約・整理・簡単・簡略（**単純化が無い**） |
| `instruction_marker[en]`（「この文は指示か」） | `simplify` のみ |

**正規表現を広げるだけでは足りなかった。** `_RHS_OP_MAP` は**値**（研究者が
書いた語そのもの）で引くので、`rhs_op` の alternation に簡約の語彙を差し込んだ
だけの中間状態で測ると、`C = [A, B]` の右辺に対し `E1 の右辺を簡約して` が
**`commutator` を実行した**——`.get()` が `None` を返し、`_parse_rhs_to_operation`
が**式の形から操作を引き直す**ため。これは私がこの修正の途中で自分で作った欠陥で、
中間状態を測らなければ入っていた。

**`collect_operator_terms` は英語の綴りが `ja` の欄に埋め込まれていた。**
`ja` は E-id が先という日本語語順を要求するので、動詞が先に来る自然な英語は
**1つも当たらない**——キー自身のコメントが挙げていた `collect operators in E3`
すら。当たるのは `E3 collect by operators` だけで、誰もそうは書かない。
読み手（`JuliaBackendHandler.extra_pre_phase1`）も `nlp(..., "ja")` と言語を
直書きしていた——**同じメソッドが `lang` を引数で受け取っているのに**。

綴り空間 **20/33 → 33/33**。
"""
import asyncio
import json
import pathlib
import re

import pytest

import ui.notebook_bridge as nb
from notebook.runtime.julia_backend_handler import JuliaBackendHandler
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.nl_patterns import fragment, nlp, pick_group
from notebook.runtime.operation_validator import OperationValidator
from ui.interpret_equation_ops import _RHS_OP_MAP, rhs_operation_for
from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline

_JSON = json.loads(
    (pathlib.Path(__file__).resolve().parent.parent / "i18n" / "nl_patterns.json")
    .read_text(encoding="utf-8"))


def _words(name):
    body = fragment(name)
    assert body.startswith("(?:") and body.endswith(")"), (name, body)
    return body[3:-1].split("|")


SIMP_JA = _words("simplify_verb_ja")
SIMP_EN = _words("simplify_verb_en")
PREP = ["f = (x+1)^2 - x^2 - 1", "g = a*x + b", "h = x + y"]


def _ask(lang, text, prep=PREP):
    """**どの操作が走ったか**を返す。status だけでは足りない——`registered` は
    「指示が式として登録された」で、これは成功に見えて指示が消えている。"""
    nb.start_engine(title="tidy", mode="standard", language=lang, domains=[])
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
        resp = run(text)
        return calls, resp.get("status")
    finally:
        nb.stop_engine()


class TestEverySimplifySpellingReachesSimplify:
    """語彙は fragment から取る——書き写すと書き写した分しか守れない。"""

    @pytest.mark.parametrize("v", SIMP_JA)
    def test_both_sides_ja(self, v):
        calls, st = _ask("ja", f"E1 の両辺を{v}して")
        assert st == "completed" and "simplify" in calls, (v, st, calls)

    @pytest.mark.parametrize("v", SIMP_EN)
    def test_both_sides_en(self, v):
        calls, st = _ask("en", f"{v} both sides of E1")
        assert st == "completed" and "simplify" in calls, (v, st, calls)

    @pytest.mark.parametrize("v", SIMP_JA)
    def test_right_hand_side_ja(self, v):
        """直す前は `整理` 以外の**すべて**が failed だった。"""
        calls, st = _ask("ja", f"E2 の右辺を{v}して")
        assert st == "completed" and "simplify" in calls, (v, st, calls)

    @pytest.mark.parametrize("v", SIMP_EN)
    def test_right_hand_side_en(self, v):
        """`reduce …` は**式として登録**されていた（status=registered）。"""
        calls, st = _ask("en", f"{v} the right-hand side of E2")
        assert st == "completed" and "simplify" in calls, (v, st, calls)

    @pytest.mark.parametrize("v", SIMP_JA)
    def test_under_constraints_ja(self, v):
        """gate（`constr_simplify_verb`）は `単純化` を通し、本体が落としていた。"""
        calls, st = _ask("ja", f"E1 を E3 の拘束のもとで{v}して")
        assert st == "completed" and "apply_canonical_constraints" in calls, (v, st, calls)

    @pytest.mark.parametrize("v", SIMP_EN)
    def test_under_constraints_en(self, v):
        calls, st = _ask("en", f"{v} E1 using the constraints of E3")
        assert st == "completed" and "apply_canonical_constraints" in calls, (v, st, calls)


class TestTheAmbiguousWordStaysAmbiguous:
    """`整理` を fragment に入れない決定は**変えていない**。曖昧なままにする。"""

    def test_bare_tidy_asks(self):
        calls, st = _ask("ja", "E1 を整理して")
        assert st == "needs_confirmation", (st, calls)

    def test_tidy_is_not_in_the_shared_vocabulary(self):
        assert "整理" not in fragment("simplify_verb_ja")
        assert "整理" not in fragment("simplify_verb_en")

    @pytest.mark.parametrize("text", ["E1 の両辺を整理して", "E2 の右辺を整理して",
                                      "E1 を E3 の拘束のもとで整理して"])
    def test_tidy_is_unambiguous_where_the_sentence_says_where(self, text):
        """辺／拘束が書かれていれば曖昧でないので、そこは `整理` を受ける。"""
        calls, st = _ask("ja", text)
        assert st == "completed", (text, st, calls)


class TestTheVocabularyLivesInOnePlace:
    @pytest.mark.parametrize("word", sorted(set(SIMP_JA) | set(SIMP_EN)))
    def test_no_key_spells_a_simplify_word_itself(self, word):
        ref = re.compile(r"\{\{\w+\}\}")
        # `有理式簡約` は**別の操作**（`rational_simplify`）の名前で、たまたま
        # `簡約` を含む。語彙の重複ではないので走査から外す。
        other = ("有理式簡約",)
        offenders = []
        for key, langs in _JSON.items():
            if key == "_fragments" or not isinstance(langs, dict):
                continue
            for lang, pat in langs.items():
                if lang.startswith("_"):
                    continue
                body = ref.sub("", str(pat))
                for o in other:
                    body = body.replace(o, "")
                if word in body:
                    offenders.append(f"{key}[{lang}]")
        assert not offenders, (
            f"{word!r} を自前で綴っている: {offenders}"
            "——簡約の語彙は simplify_verb_ja / _en にだけ置く")

    @pytest.mark.parametrize("word", sorted(set(SIMP_JA) | set(SIMP_EN)))
    def test_the_keyword_map_does_not_spell_one(self, word):
        """**値で引く表も綴りの置き場所**。ここに書き足すと4箇所目になる。"""
        assert word not in _RHS_OP_MAP, (
            f"{word!r} が _RHS_OP_MAP に書かれている——fragment に訊く")

    @pytest.mark.parametrize("word", sorted(set(SIMP_JA) | set(SIMP_EN)))
    def test_the_keyword_map_still_answers_simplify(self, word):
        assert rhs_operation_for(word) == "simplify", word

    @pytest.mark.parametrize("word,want", [
        ("整理", "simplify"), ("展開", "expand"), ("因数分解", "factor"),
        ("有理化", "rational_simplify"), ("計算", None), ("評価", None),
        ("expand", "expand"), ("compute", None), ("evaluate", None),
    ])
    def test_the_other_families_are_untouched(self, word, want):
        assert rhs_operation_for(word) == want, word

    def test_the_readers_splice_instead_of_spelling(self):
        assert "{{simplify_verb_ja}}" in _JSON["rhs_op"]["ja"]
        assert "{{simplify_verb_en}}" in _JSON["rhs_op"]["en"]
        assert "{{simplify_verb_ja}}" in _JSON["apply_constraints"]["ja"]
        assert "{{simplify_verb_en}}" in _JSON["apply_constraints"]["en"]
        assert "{{simplify_verb_en}}" in _JSON["instruction_marker"]["en"]


class TestAnInstructionIsNotRegisteredAsAnExpression:
    """`instruction_marker` は「この文は指示か」に答える表で、簡約の語彙を
    自前で持っていた——`simplify` は知っていて `reduce` を知らないので、
    `reduce the right-hand side of E2` が**式として登録**されていた。"""

    @pytest.mark.parametrize("v", SIMP_EN)
    def test_every_simplify_verb_marks_an_instruction(self, v):
        assert nlp("instruction_marker", "en").search(f"{v} the right-hand side of E2"), v

    @pytest.mark.parametrize("v", SIMP_EN)
    def test_it_is_not_registered(self, v):
        _, st = _ask("en", f"{v} the right-hand side of E2")
        assert st != "registered", v


class TestCollectReadsEnglishInEnglishWordOrder:
    """英語の綴りが `ja` の欄に埋め込まれていた。`ja` は E-id が先という日本語
    語順を要求するので、**自然な英語は1つも当たらなかった**。"""

    HANDLER = JuliaBackendHandler(None)

    @pytest.mark.parametrize("text", [
        "collect operators in E3",            # ← キー自身のコメントが挙げていた例
        "collect by operators in E3",
        "group by operators in E3",
        "collect like terms in E3",
        "collect the like terms of E3",
        "collect E3 by operators",
        "group E3 by operators",
    ])
    def test_english(self, text):
        r = self.HANDLER.extra_pre_phase1(text, "en")
        assert getattr(r, "operation", None) == "collect_by_operators", (text, r)
        assert r.args["expr"] == "E3", (text, r.args)

    @pytest.mark.parametrize("text", [
        "E3の演算子でまとめて", "E3を演算子係数でまとめる", "E3の同類項をまとめて",
        "E3 の同類項を整理して", "E3 を演算子係数で整理して",
    ])
    def test_japanese(self, text):
        r = self.HANDLER.extra_pre_phase1(text, "ja")
        assert getattr(r, "operation", None) == "collect_by_operators", (text, r)
        assert r.args["expr"] == "E3", (text, r.args)

    def test_english_is_no_longer_inside_the_japanese_pattern(self):
        assert "collect" not in _JSON["collect_operator_terms"]["ja"]
        assert "group" not in _JSON["collect_operator_terms"]["ja"]

    def test_both_languages_use_the_same_group_name(self):
        for lang in ("ja", "en"):
            m = nlp("collect_operator_terms", lang).search(
                "E3の同類項をまとめて" if lang == "ja" else "collect operators in E3")
            assert pick_group(m, "eid") == "E3", lang

    def test_the_reader_does_not_hardcode_a_language(self):
        src = (pathlib.Path(__file__).resolve().parent.parent / "notebook" / "runtime"
               / "julia_backend_handler.py").read_text(encoding="utf-8")
        assert 'nlp("collect_operator_terms", "ja")' not in src
        assert 'nlp("collect_operator_terms", _l)' in src
