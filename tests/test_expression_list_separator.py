r"""式番号の並びは1つの読み —— 3鍵が自前で区切りを綴っていた。

実測 2026-09-06。共有 fragment `list_sep_ja` / `list_sep_en`（`、`・`,`・`と`・
`および`・`ならびに`・`や` / `,`・`and`・`&`）は既にあり、兄弟の
`inv_transform_vars` はそれを使っている。にもかかわらず `inv_transform` /
`verify_canon` / `canon_constraint` は `\s*[と,、]` / `\s*,\s*` と自前で綴っており、
**同じ問いに2つの答え**があった。

| 打った文 | 直す前 |
|---|---|
| `E1 および E2 の逆変換を求めて` | 「逆変換には変換式が**2本以上必要です**」——研究者は2本書いている |
| `E1, E2 の逆変換`（動詞なし） | 引数不足。兄弟 `inv_transform_vars` は動詞を要求しない |
| `Invert E1 and E2` | **`Maxima evaluation error`**（生の CAS 文言、Core Principle #1） |
| `Are E1 and E2 canonical?` | 一致せず |
| `Derive the canonical constraints of E1 and E2` | **`E1` だけ捕まえる**——E2 を黙って落とす |

**断り文が入力に当たっていなかった。** 「変換式が2本以上必要です」は一般論として
真だが、`E1 および E2` と書いた研究者に対しては偽である——区切りを読めずに1本しか
捕まえていないだけだった。

**コーパスは `について` 付きの1形しか通していない。** `inv_transform`（変数なし）と
`verify_canon` は 0行で、だから今まで誰も踏まなかった。
"""
import asyncio
import re

import pytest

import ui.notebook_bridge as nb
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.nl_patterns import fragment, nlp
from notebook.runtime.operation_validator import OperationValidator
from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline

DOM = ["classical_mechanics"]
PREP = ["y_1 = x_1 + x_2", "y_2 = x_1 - x_2"]

#: 逆変換の綴り。**どれも同じ2式を読む。**
INVERT = [
    ("ja", "E1, E2 を x_1, x_2 について逆変換して"),
    ("ja", "E1、E2 を x_1, x_2 について逆変換して"),
    ("ja", "E1 と E2 を x_1, x_2 について逆変換して"),
    ("ja", "E1 および E2 を x_1, x_2 について逆変換して"),
    ("ja", "E1, E2 の逆変換を求めて"),
    ("ja", "E1 と E2 の逆変換を求めて"),
    ("ja", "E1 および E2 の逆変換を求めて"),
    ("ja", "E1, E2 の逆変換"),
    ("en", "Invert E1, E2 for x_1, x_2"),
    ("en", "Invert E1 and E2 for x_1, x_2"),
    ("en", "Build the inverse transform of E1, E2 for x_1, x_2"),
    ("en", "Invert E1, E2"),
    ("en", "Invert E1 and E2"),
    ("en", "Build the inverse transform of E1, E2"),
]
#: 読みの層。E-id を**2つとも**捕まえること。
READS = [
    ("inv_transform", "ja", "E1, E2 の逆変換を求めて"),
    ("inv_transform", "ja", "E1 と E2 の逆変換を求めて"),
    ("inv_transform", "ja", "E1 および E2 の逆変換を求めて"),
    ("inv_transform", "ja", "E1, E2 の逆変換"),
    ("inv_transform", "en", "Invert E1, E2"),
    ("inv_transform", "en", "Invert E1 and E2"),
    ("verify_canon", "ja", "E1, E2 は正準変換ですか"),
    ("verify_canon", "ja", "E1 および E2 は正準変換ですか"),
    ("verify_canon", "en", "Is E1, E2 canonical?"),
    ("verify_canon", "en", "Are E1 and E2 canonical?"),
    ("canon_constraint", "ja", "E1, E2 の正準拘束を求めて"),
    ("canon_constraint", "ja", "E1 および E2 の正準拘束を求めて"),
    ("canon_constraint", "en", "Derive the canonical constraints of E1, E2"),
    ("canon_constraint", "en", "Derive the canonical constraints of E1 and E2"),
    ("inv_transform_vars", "ja", "E1 および E2 を x について逆変換して"),
    ("inv_transform_vars", "en", "Invert E1 and E2 for x"),
]


def _ask(lang, text):
    nb.start_engine(title="ls", mode="standard", language=lang, domains=DOM)
    try:
        eng = nb.get_engine()
        sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                            domains=DOM, primary_backend="maxima")

        def run(x):
            return asyncio.get_event_loop().run_until_complete(
                run_interpret_pipeline(
                    InterpretRequest(tokens=[InputToken(type="text", value=x)]),
                    eng, sup))
        for p in PREP:
            run(p)
        r = run(text)
        regs = r.get("registered_expressions") or []
        return r, " | ".join((e.get("latex") or "") for e in regs)
    finally:
        nb.stop_engine()


class TestEverySpellingInvertsBothEquations:
    @pytest.mark.parametrize("lang,text", INVERT)
    def test_it_computes(self, lang, text):
        r, body = _ask(lang, text)
        assert r.get("status") == "completed", (text, r.get("error"))
        # 2式を読めていれば x_1 と x_2 の両方が出る。
        assert "x_{1}" in body and "x_{2}" in body, (text, body)

    @pytest.mark.parametrize("lang,text", INVERT)
    def test_no_raw_cas_words_reach_the_researcher(self, lang, text):
        """`Invert E1 and E2` は生の `Maxima evaluation error` を返していた。"""
        r, _ = _ask(lang, text)
        assert "Maxima" not in (r.get("error") or ""), (text, r.get("error"))


class TestTheListSeparatorIsOneFragment:
    @pytest.mark.parametrize("key,lang,text", READS)
    def test_both_expression_ids_are_captured(self, key, lang, text):
        m = nlp(key, lang).search(text)
        assert m is not None, (key, text)
        gd = m.groupdict()
        got = gd.get("eids") or next((g for g in m.groups() if g), None)
        assert set(re.findall(r"E\d+", got or "")) == {"E1", "E2"}, (key, text, got)

    @pytest.mark.parametrize("key,lang,frag", [
        ("inv_transform", "ja", "list_sep_ja"),
        ("inv_transform", "en", "list_sep_en"),
        ("inv_transform_vars", "ja", "list_sep_ja"),
        ("inv_transform_vars", "en", "list_sep_en"),
        ("verify_canon", "ja", "list_sep_ja"),
        ("verify_canon", "en", "list_sep_en"),
        ("canon_constraint", "ja", "list_sep_ja"),
        ("canon_constraint", "en", "list_sep_en"),
    ])
    def test_no_key_spells_the_separator_itself(self, key, lang, frag):
        """**同じ問いに2つの答えを作らない。** 自前綴りに戻すとここで落ちる。"""
        import json
        import pathlib
        src = json.loads(pathlib.Path("i18n/nl_patterns.json").read_text())[key][lang]
        assert "{{" + frag + "}}" in src, (key, lang, src)

    @pytest.mark.parametrize("word", ["、", ",", "と", "および", "ならびに", "や"])
    def test_the_japanese_fragment_holds_every_word(self, word):
        rx = re.compile("^" + fragment("list_sep_ja") + "$")
        assert rx.match(word) or rx.match(f" {word} "), word

    @pytest.mark.parametrize("word", [",", "and", "&", ", and"])
    def test_the_english_fragment_holds_every_word(self, word):
        rx = re.compile("^" + fragment("list_sep_en") + "$", re.I)
        assert rx.match(word) or rx.match(f" {word} "), word
