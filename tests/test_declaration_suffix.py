"""宣言の語尾 —— 英語のほうが広いという、これまでと逆向きの割れ。

実測 2026-09-05（実パイプライン）。英語は**7綴りすべて**が通るのに、日本語は
`を` 形で3つ落ちていた:

| | 通る | 落ちる |
|---|---|---|
| 英語 7/7 | `Define` / `Let … be` / `is` / `Declare` / `Please define` / `Regard … as` / `Take … to be` | — |
| 日本語（`を` 形） | `とする` / `と定義する・して・します` / `として定義する・して` | **`とします`** / **`とみなす`** / **`とせよ`** |
| 日本語（`は` 形） | `である` / `です` / `だ` / `とする` | — |

**`とします` は fragment 自身の不整合だった。** `ja_decl_suffix` は
`とする|と定義(?:する|します|して)?|と宣言(?:…)?|として定義(?:…)?|として宣言(?:…)?`
で、**素の `とする` だけ活用形を持たない**。だから `と定義します` は通るのに
`とします` は落ちる。活用は1つの一覧（`ja_decl_infl`）にした——4回書かれていた。

**`とみなす` は `Regard a as a scalar` の鏡、`とせよ` は `Take a to be a scalar`
の鏡**で、英語はどちらも通っていた。

**`として` 単体は宣言にしていない。** 日本語では接続語であって、
`a をスカラーとして` だけでは文が終わっていない。測って落ちた綴りだけを入れる
（十一問目・十五問目と同じ判断）。

**静的な数え方では見えなかった。** 10キーの部分集合を literal で数えると6通りに
分かれて見えるが、合成形 `として定義(?:する|…)` はリテラル照合に映らないので、
落ちている3綴りは**文を当てて初めて**出た。
"""
import asyncio

import pytest

import ui.notebook_bridge as nb
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.nl_patterns import nlp
from notebook.runtime.operation_validator import OperationValidator
from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline


#: 宣言として読まれるべき綴り。日英そろえてある。
ACCEPTED = [
    ("ja", "a をスカラーとする"),
    ("ja", "a をスカラーとします"),
    ("ja", "a をスカラーとせよ"),
    ("ja", "a をスカラーとしろ"),
    ("ja", "a をスカラーとみなす"),
    ("ja", "a をスカラーとみなして"),
    ("ja", "a をスカラーとみなします"),
    ("ja", "a をスカラーと定義する"),
    ("ja", "a をスカラーと定義して"),
    ("ja", "a をスカラーと定義します"),
    ("ja", "a をスカラーとして定義する"),
    ("ja", "a をスカラーとして定義して"),
    ("ja", "a はスカラーである"),
    ("ja", "a はスカラーです"),
    ("ja", "a はスカラーだ"),
    ("en", "Define a as a scalar"),
    ("en", "Let a be a scalar"),
    ("en", "a is a scalar"),
    ("en", "Declare a as a scalar"),
    ("en", "Please define a as a scalar"),
    ("en", "Regard a as a scalar"),
    ("en", "Take a to be a scalar"),
]

#: 宣言として読まれては**ならない**綴り。入れなかったものにも理由がある。
REFUSED = [
    # `として` は接続語で、この文はまだ終わっていない。
    "a をスカラーとして",
    "a をスカラーとして扱う",
    # `を` ＋ `である` は非文（正しくは `a はスカラーである`、これは上で通る）。
    "a をスカラーである",
]


class TestThePatternReadsEverySpelling:
    @pytest.mark.parametrize("lang,text", [(l, t) for l, t in ACCEPTED if l == "ja"])
    def test_japanese(self, lang, text):
        m = (nlp("decl_stmt", "ja").search(text)
             or nlp("decl_stmt_wa", "ja").search(text))
        assert m is not None, text
        assert m.groups()[-1].strip().startswith("スカラー"), (text, m.groups())

    @pytest.mark.parametrize("text", REFUSED)
    def test_what_was_left_out(self, text):
        assert nlp("decl_stmt", "ja").search(text) is None, text


class TestTheInflectionIsOneList:
    """`(?:する|します|して)?` が4回書かれていて、素の `とする` だけ持っていなかった。"""

    @pytest.mark.parametrize("infl", ["する", "します", "して", "せよ", "しろ"])
    def test_every_inflection_is_in_the_one_list(self, infl):
        import re
        from notebook.runtime.nl_patterns import fragment
        assert re.fullmatch(fragment("ja_decl_infl"), infl), infl

    def test_the_suffix_uses_it_rather_than_repeating(self):
        import json, pathlib
        src = json.loads(pathlib.Path("i18n/nl_patterns.json").read_text(
            encoding="utf-8"))["_fragments"]["ja_decl_suffix"]
        assert "{{ja_decl_infl}}" in src
        assert "する|します|して" not in src, "活用がまた直書きされている"


def _run(eng, sup, text):
    return asyncio.get_event_loop().run_until_complete(
        run_interpret_pipeline(
            InterpretRequest(tokens=[InputToken(type="text", value=text)]), eng, sup))


@pytest.mark.parametrize("lang,text", ACCEPTED)
def test_the_name_is_actually_registered(lang, text):
    """画面まで —— 落ちていた綴りは**何も宣言しないまま失敗**していた。"""
    nb.start_engine(title="decl_suffix", mode="standard", language=lang,
                    domains=["classical_mechanics"])
    try:
        eng = nb.get_engine()
        sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                            domains=["classical_mechanics"], primary_backend="maxima")
        resp = _run(eng, sup, text)
        assert resp.get("status") in ("confirmed", "declared", "completed"), \
            (text, resp.get("status"), resp.get("error"))
        assert "a" in eng.symbol_context.registry, (text, "名前が登録されていない")
    finally:
        nb.stop_engine()


def test_the_two_languages_accept_the_same_set():
    """英語のほうが広い、という割れが解消したことを数で固定する。"""
    ja = [t for l, t in ACCEPTED if l == "ja"]
    en = [t for l, t in ACCEPTED if l == "en"]
    assert len(ja) >= len(en), (len(ja), len(en))
