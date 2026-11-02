r"""二項括弧は1つの形 —— 交換子とポアソン括弧で語彙が5通りに割れていた。

実測 2026-09-06（`classical_mechanics`。`q,p` を正準対、`A = q^2`、`B = p^2`）。
`commutator`（julia_full）と `poisson_bracket`（maxima）は**同じ `[X, Y]` 記法・
同じ `a`/`b` 欄**を共有し、ドメインで相互排他である（Phase 0 の
`_try_bracket_parse` が既にそう書いている）。にもかかわらず NL の読みが割れていた。

| # | 打った文 | 直す前 | 原因 |
|---|---|---|---|
| 1 | `q と p を正準共役対とする` | `define_scalar_parameter` | `正準(?:共役｜座標対｜対｜ペア)` が `共役` で止まり `対` が余る → 正準対が登録されず**ポアソン括弧が `__no_canonical_pairs__` で失敗** |
| 2 | `A と B のポアソン括弧をとって` | failed | `とっ`（ひらがな）が `compute_verb_ja` に無い（`取っ` だけ）。`binary_bracket_phrase` は自前で `｜とっ` を持っていた |
| 3 | `A と B のポアソン括弧は？` | failed | `binary_bracket_phrase` がポアソン括弧を知らない。さらに `BRACKET_KIND_OPS` にも無く、**語彙を足しても表が知らなければ読みは黙って降りる**（2026-08-27 の∇族の註と同じ形） |
| 4 | `[A, B] を計算して` | **式として登録**（canonical `[A, B]`） | `operands_provided` は True（括弧操作）・`treats_bracket_as_commutator` は False（式）——**同じ問いに2つの答え** |
| 5 | `compute [A, B]` | failed | 括弧の読みが「先頭が括弧か」だけを見ていた（英語は動詞が前） |

**行列リテラルは括弧の操作ではない。** `[0, -b*c/d; a*d/b, 0]` はコーパスに実在
する（`lotka_volterra_stability`）。「本物の二項括弧は `;` を含まない」という判別
は Phase 0 が既に持っていたので、同じものを使う。Julia の
`treats_bracket_as_commutator` は無条件 `True` を返していたので、**その除外が
効いていなかった**——既定が同じ答えを出すようになったのでオーバーライドを削除した。
"""
import asyncio

import pytest

import ui.notebook_bridge as nb
from notebook.runtime.julia_backend_handler import JuliaBackendHandler
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.maxima_backend_handler import MaximaBackendHandler
from notebook.runtime.nl_patterns import fragment, nlp
from notebook.runtime.operand_boundary import (BINARY_KIND_OPS,
                                               skip_leading_compute_verb)
from notebook.runtime.operation_validator import OperationValidator
from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline

PREP = {"ja": ["q と p を正準共役対とする", "A = q^2", "B = p^2"],
        "en": ["define q and p as a canonical pair", "A = q^2", "B = p^2"]}
WORKS = [
    ("ja", "A と B のポアソン括弧"),
    ("ja", "A と B のポアソン括弧を計算して"),
    ("ja", "A と B のポアソン括弧を求めて"),
    ("ja", "A と B のポアソン括弧をとって"),
    ("ja", "A と B のポアソン括弧は？"),
    ("ja", "A, B のポアソン括弧"),
    ("ja", "[A, B] を計算して"),
    ("en", "the Poisson bracket of A and B"),
    ("en", "compute the Poisson bracket of A and B"),
    ("en", "Poisson bracket of A and B"),
    ("en", "the Poisson bracket of A with B"),
    ("en", "the Poisson bracket between A and B"),
    ("en", "take the Poisson bracket of A and B"),
    ("en", "compute [A, B]"),
]
#: 正準対の宣言。**6綴りすべて**が同じ対を登録する。
PAIR_DECLS = ["q と p を正準共役対とする", "q と p を正準共役ペアとする",
              "q と p を正準共役とする", "q と p を正準対とする",
              "q と p を正準ペアとする", "q と p を正準座標対とする"]


def _ask(lang, text, prep=None):
    nb.start_engine(title="bracket", mode="standard", language=lang,
                    domains=["classical_mechanics"])
    try:
        eng = nb.get_engine()
        sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                            domains=["classical_mechanics"], primary_backend="maxima")

        def run(x):
            return asyncio.get_event_loop().run_until_complete(
                run_interpret_pipeline(
                    InterpretRequest(tokens=[InputToken(type="text", value=x)]),
                    eng, sup))
        for p in (PREP[lang] if prep is None else prep):
            run(p)
        resp = run(text)
        vals = [e.canonical for e in eng.expressions.values()]
        return resp, vals
    finally:
        nb.stop_engine()


class TestEverySpellingReachesTheBracket:
    @pytest.mark.parametrize("lang,text", WORKS)
    def test_computes(self, lang, text):
        resp, vals = _ask(lang, text)
        assert resp.get("status") == "completed", (text, resp.get("status"),
                                                   resp.get("error"))
        assert resp.get("operation") == "poisson_bracket", (text, resp)
        # {q^2, p^2} = 4 q p
        assert any("4" in str(v) and "p" in str(v) and "q" in str(v)
                   for v in vals[2:]), (text, vals)


class TestTheCanonicalPairDeclaration:
    @pytest.mark.parametrize("decl", PAIR_DECLS)
    def test_every_spelling_declares_the_pair(self, decl):
        """**語を並べた選択肢では合成語が落ちる。** `正準(?:共役|座標対|対|ペア)`
        は `共役` で止まり続く `対` が余るので、`正準共役対`・`正準共役ペア` が
        落ちていた——そして落ちるとポアソン括弧が `__no_canonical_pairs__` で
        失敗する（実測 2026-09-06）。"""
        resp, _ = _ask("ja", "A と B のポアソン括弧",
                       prep=[decl, "A = q^2", "B = p^2"])
        assert resp.get("status") == "completed", (decl, resp.get("error"))


class TestABracketIsOneShape:
    @pytest.mark.parametrize("kind", ["交換子", "交換関係", "反交換子", "反交換関係",
                                      "ポアソン括弧"])
    def test_the_kind_word_is_in_the_shared_vocabulary(self, kind):
        assert nlp("binary_operation_phrase", "ja").search(f"A と B の{kind}"), kind

    @pytest.mark.parametrize("kind", ["ポアソン括弧", "poisson bracket",
                                      "交換子", "commutator", "反交換子"])
    def test_the_table_knows_every_kind_the_vocabulary_reads(self, kind):
        """**語彙を足しても表が知らなければ読みは黙って降りる。**
        実際に踏んだ（2026-09-06）——`binary_bracket_phrase` に「ポアソン括弧」を
        入れても `BRACKET_KIND_OPS` に無く、`nlp` に当たりながら `None` を返した。
        """
        assert kind in BINARY_KIND_OPS, kind

    @pytest.mark.parametrize("spelling", ["ポアソン括弧", "ポワソン括弧",
                                          "ポアソンの括弧"])
    def test_the_spelling_variants_reach_the_table(self, spelling):
        from notebook.runtime.operand_boundary import _normalise_binary_kind
        assert _normalise_binary_kind(spelling) in BINARY_KIND_OPS, spelling

    @pytest.mark.parametrize("spelling", ["ポワソン括弧", "ポアソンの括弧"])
    def test_a_spelling_variant_computes_end_to_end(self, spelling):
        """綴りの揺れも計算に届く。

        **正規化を消す変異は1件も落とさない——そしてそれが正しい。**
        測ると答えは変わらなかった（実測 2026-09-06）: `pb_bracket` が
        `ポ[アワ]ソン(?:の)?括弧` を持っており、**別の経路が同じ答えを出す**。
        正規化は冗長性であって死骸ではない——`binary_bracket_phrase` 側だけで
        読める綴り（`をとって` / `は？` / `A, B の…`）では、そちらが唯一の
        経路になる。2a-sym の lookbehind と同じ判断で残し、測定を書いた。
        """
        resp, vals = _ask("ja", f"A と B の{spelling}を計算して")
        assert resp.get("operation") == "poisson_bracket", (spelling, resp)
        assert any("4" in str(v) for v in vals[2:]), (spelling, vals)

    @pytest.mark.parametrize("spelling", ["ポワソン括弧", "ポアソンの括弧"])
    def test_the_variant_reaches_the_shared_reading_too(self, spelling):
        """`binary_bracket_phrase` 側**だけ**が読む綴りで、正規化が唯一の経路
        になることを固定する（動詞の省略・`は？`・`A, B の…`）。"""
        from notebook.runtime.operand_boundary import read_binary_operation_phrase
        got = read_binary_operation_phrase(
            f"A と B の{spelling}は？", "ja", lambda op: op == "poisson_bracket")
        assert got is not None and got[0] == "poisson_bracket", (spelling, got)


class TestTheVerbVocabularyIsShared:
    @pytest.mark.parametrize("verb", ["取って", "取る", "とって", "とる", "とっ"])
    def test_hiragana_and_kanji_are_both_in_the_shared_list(self, verb):
        """`binary_bracket_phrase` が自前で `|とっ` を持っていたのは、共有一覧に
        無かったからで、その場しのぎだった。"""
        assert verb in fragment("compute_verb_ja"), verb

    def test_the_pattern_no_longer_carries_its_own_copy(self):
        import json
        import pathlib
        d = json.loads(pathlib.Path("i18n/nl_patterns.json").read_text(
            encoding="utf-8"))
        assert "|とっ)" not in d["binary_operation_phrase"]["ja"]


class TestOneAnswerToIsThisABracketOperation:
    """`operands_provided` と `treats_bracket_as_commutator` が食い違っていた。"""

    @pytest.mark.parametrize("backend,handler", [("maxima", MaximaBackendHandler),
                                                 ("julia_full", JuliaBackendHandler)])
    @pytest.mark.parametrize("text", ["[A, B]", "{A, B}", "[A, B] を計算して"])
    def test_both_say_yes(self, backend, handler, text):
        h = handler(None)
        assert h.operands_provided(text) is True, (backend, text)
        assert h.treats_bracket_as_commutator(text) is True, (backend, text)

    @pytest.mark.parametrize("backend,handler", [("maxima", MaximaBackendHandler),
                                                 ("julia_full", JuliaBackendHandler)])
    @pytest.mark.parametrize("text", ["[0, -b*c/d; a*d/b, 0]", "[2, 1; 1, 2]"])
    def test_a_matrix_literal_is_not_a_bracket_operation(self, backend, handler, text):
        """**行列リテラルは括弧の操作ではない**（コーパスに実在）。Julia は
        無条件 `True` を返していたので、この除外が効いていなかった。"""
        assert handler(None).treats_bracket_as_commutator(text) is False, (backend, text)


class TestEnglishPutsTheVerbFirst:
    @pytest.mark.parametrize("text,rest", [
        ("compute [A, B]", "[A, B]"),
        ("evaluate {A, B}", "{A, B}"),
        ("take [A, B]", "[A, B]"),
        ("please compute [A, B]", "[A, B]"),
        ("[A, B]", "[A, B]"),
    ])
    def test_the_leading_verb_is_skipped(self, text, rest):
        assert skip_leading_compute_verb(text) == rest, text

    @pytest.mark.parametrize("text", ["A と B の交換子",
                                      "the commutator of A and B",
                                      "define A as a matrix"])
    def test_a_non_bracket_text_is_declined(self, text):
        assert skip_leading_compute_verb(text) is None, text
