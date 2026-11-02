r"""簡約の語彙は1つ —— 同じ問いに3箇所が別々に答えていた。

実測 2026-09-07。「どの語が『簡約せよ』か」に、**3つの表が違う答え**を持っていた:

| 表 | 日本語 | 英語 |
|---|---|---|
| 操作スキーマの別名 | `簡約｜単純化` | `simplify` |
| `constr_simplify_verb` | `簡約｜整理｜簡単に｜簡略` | `reduce｜simplify` |
| `both_sides_simplify` | `整理｜簡約｜ratsimp` | `simplify` |

結果、経路ごとに通る語が違った（すべて `failed`、silent-wrong ではない）:

| 語 | 式そのもの | 両辺 |
|---|---|---|
| 簡約 | ✔ | ✔ |
| 単純化 | ✔ | **✘** |
| 簡単に／簡略化 | **✘** | **✘** |
| simplify | ✔ | ✔ |
| reduce | **✘** | **✘** |

`簡単に`・`簡略`・`reduce` は `constr_simplify_verb` に**入っていた**のに、拘束の
簡約という1経路でしか使われていなかった。

**`整理` は fragment に入れない。** `collect`（同類項をまとめる）とも競合する
**曖昧な語**で、素の式に対しては候補ダイアログを出すのが正しい（Phase 1.5 の
`ambiguous_tidy`）。両辺に対しては曖昧でないので、そちらの鍵にだけ置く。

**別名は「トークン」、fragment は「生のテキスト」を見る。** トークナイザは
`簡単にして` を **`簡単`** にするので、スキーマの別名は `簡単`、正規表現側は
`簡単に` が正しい。両者が字面で一致しないのは**設計どおり**なので、テストは
「同じ語を読めること」で突き合わせる（字面の一致ではなく）。
"""
import asyncio
import json
import pathlib

import pytest

import ui.notebook_bridge as nb
from notebook.runtime.llm_supervisor import LLMSupervisor, _tokenize
from notebook.runtime.nl_patterns import fragment, nlp
from notebook.runtime.operation_validator import OperationValidator
from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline

CM = ["classical_mechanics"]
_P = json.loads(pathlib.Path("i18n/nl_patterns.json").read_text())

#: 曖昧でない「簡約せよ」の語。**どの経路でも通ること。**
WORDS = [("ja", "簡約して"), ("ja", "単純化して"), ("ja", "簡単にして"),
         ("ja", "簡略化して"), ("en", "simplify"), ("en", "reduce")]


def _ask(lang, prep, text):
    nb.start_engine(title="sv", mode="standard", language=lang, domains=CM)
    try:
        eng = nb.get_engine()
        sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                            domains=CM, primary_backend="maxima")

        def run(x):
            return asyncio.get_event_loop().run_until_complete(
                run_interpret_pipeline(
                    InterpretRequest(tokens=[InputToken(type="text", value=x)]),
                    eng, sup))
        for p in prep:
            run(p)
        r = run(text)
        regs = r.get("registered_expressions") or []
        return r, (r.get("latex") or (regs[-1].get("latex") if regs else "") or "")
    finally:
        nb.stop_engine()


class TestEverySpellingSimplifies:
    @pytest.mark.parametrize("lang,word", WORDS)
    def test_the_expression_itself(self, lang, word):
        text = f"E1 を{word}" if lang == "ja" else f"{word} E1"
        r, body = _ask(lang, ["f = x + x"], text)
        assert r.get("status") == "completed", (text, r.get("error"))
        assert "2" in body and "x" in body, (text, body)

    @pytest.mark.parametrize("lang,word", WORDS)
    def test_both_sides(self, lang, word):
        text = f"E1 の両辺を{word}" if lang == "ja" else f"{word} both sides of E1"
        r, body = _ask(lang, ["y = x + x"], text)
        assert r.get("status") == "completed", (text, r.get("error"))
        assert "2" in body and "x" in body, (text, body)


class TestTheAmbiguousWordStaysAmbiguous:
    r"""`整理` は `collect` とも競合する——素の式では訊くのが正しい。"""

    def test_a_bare_expression_asks(self):
        r, _ = _ask("ja", ["f = x + x"], "E1 を整理して")
        assert r.get("status") == "needs_confirmation", r.get("status")

    def test_both_sides_does_not_ask(self):
        """両辺に対しては曖昧でない。"""
        r, body = _ask("ja", ["y = x + x"], "E1 の両辺を整理して")
        assert r.get("status") == "completed", r.get("error")
        assert "2" in body, body

    def test_the_shared_fragment_excludes_it(self):
        assert "整理" not in fragment("simplify_verb_ja")


class TestOneVocabularyInBothPlaces:
    @pytest.mark.parametrize("key,lang,frag", [
        ("both_sides_simplify", "ja", "simplify_verb_ja"),
        ("both_sides_simplify", "en", "simplify_verb_en"),
        ("constr_simplify_verb", "ja", "simplify_verb_ja"),
        ("constr_simplify_verb", "en", "simplify_verb_en"),
    ])
    def test_no_key_spells_the_verb_itself(self, key, lang, frag):
        assert "{{" + frag + "}}" in _P[key][lang], (key, lang)

    @pytest.mark.parametrize("lang,word", WORDS)
    def test_the_schema_alias_reads_the_same_word(self, lang, word):
        """**別名はトークン、fragment は生のテキストを見る。** だから字面では
        比べず、「同じ語を読めるか」で突き合わせる——`簡単にして` の
        トークンは `簡単`、正規表現に要るのは `簡単に` である。
        """
        spec = OperationValidator().get_spec("simplify")
        aliases = {a.lower() for a in spec["aliases"][lang]}
        text = f"E1 を{word}" if lang == "ja" else f"{word} E1"
        toks = {t.lower() for t in _tokenize(text)}
        assert toks & aliases, (word, sorted(toks), sorted(aliases))

    @pytest.mark.parametrize("lang,word", WORDS)
    def test_the_regex_fragment_reads_the_same_word(self, lang, word):
        import re
        frag = fragment("simplify_verb_ja" if lang == "ja" else "simplify_verb_en")
        assert re.search(frag, word, re.I), (word, frag)
