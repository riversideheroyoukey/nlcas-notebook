r"""展開の語彙と変数の綴り —— 同じ3語を2鍵が違う部分集合で持っていた。

実測 2026-09-07。`expand` は6つの意味に分かれる（代数展開・級数展開・成分展開・
非結合基底での展開・格子和の展開・右辺の展開）。役割の分かれ方は正しかったが、
**その中の語彙**が割れていた。

**(1) 展開の動詞。** 同じ3語（展開・分解・書き下 / expand・decompose・write）を
2鍵が**違う部分集合**で持ち、しかも日英が食い違っていた:

| 鍵 | ja | en |
|---|---|---|
| `expand_components` | `展開｜分解`（`書き下` なし） | **`expand` のみ** |
| `angular_coupled_state` | `展開｜書き下`（`分解` なし） | `expand｜write｜decompose` |

`expand_components` は ja に `分解` があって en に `decompose` が無く、
`angular_coupled_state` は**その逆**。片方を見て直すともう片方が残る形である。

**(2) 級数展開の変数。** 三十五問目で共有 fragment
（`en_op_var_head_wide` / `ja_op_var_particle`）に畳んだのに、`taylor_expand_nl` は
自前で綴っていた（`in|with respect to|w.r.t.|by` / `について|に関して|で|…`）。
結果 **3綴りが silent-wrong**——展開が起きないまま `completed` が返っていた:

| 打った文 | 直す前 |
|---|---|
| `E1 を x **による** 0 のまわりで 3次まで展開して` | `h = e^x`（展開されず） |
| `expand E1 **for** x around 0 to order 3` | 同上 |
| `expand E1 **over** x around 0 to order 3` | 同上 |

**同じ穴を2問続けて別の鍵で踏んだ**（三十五問目は `def_integral`、ここは
`taylor_expand_nl`）。fragment を作っても、**読む側を全部数えないと残る**。
"""
import asyncio
import json
import pathlib

import pytest

import ui.notebook_bridge as nb
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.nl_patterns import fragment, nlp
from notebook.runtime.operation_validator import OperationValidator
from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline

CM = ["classical_mechanics"]
_P = json.loads(pathlib.Path("i18n/nl_patterns.json").read_text())

#: 級数展開の変数の綴り。**どれも 3次まで展開されること。**
SERIES_VAR = [
    ("ja", "E1 を x について 0 のまわりで 3次まで展開して"),
    ("ja", "E1 を x で 0 のまわりで 3次まで展開して"),
    ("ja", "E1 を x に関して 0 のまわりで 3次まで展開して"),
    ("ja", "E1 を x による 0 のまわりで 3次まで展開して"),
    ("en", "expand E1 in x around 0 to order 3"),
    ("en", "expand E1 with respect to x around 0 to order 3"),
    ("en", "expand E1 by x around 0 to order 3"),
    ("en", "expand E1 for x around 0 to order 3"),
    ("en", "expand E1 over x around 0 to order 3"),
]
#: 展開の動詞。読みの層で固定する（成分展開はベクトル積分の準備が要るため）。
EXPAND_VERB = [
    ("expand_components", "ja", "E1 を成分に展開して"),
    ("expand_components", "ja", "E1 を成分に分解して"),
    ("expand_components", "ja", "E1 を成分に書き下して"),
    ("expand_components", "ja", "成分に展開して"),
    ("expand_components", "en", "expand E1 into components"),
    ("expand_components", "en", "decompose E1 into components"),
    ("expand_components", "en", "write E1 into components"),
    ("expand_components", "en", "write out E1 into components"),
    ("angular_coupled_state", "ja", "u と v の非結合基底で展開して"),
    ("angular_coupled_state", "ja", "u と v の非結合基底で分解して"),
    ("angular_coupled_state", "ja", "u と v の非結合基底で書き下して"),
    ("angular_coupled_state", "en", "expand in the uncoupled basis u and v"),
    ("angular_coupled_state", "en", "decompose in the uncoupled basis u and v"),
    ("angular_coupled_state", "en", "write in the uncoupled basis u and v"),
]


def _ask(lang, prep, text):
    nb.start_engine(title="ex", mode="standard", language=lang, domains=CM)
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


class TestTheSeriesVariableIsOneReading:
    @pytest.mark.parametrize("lang,text", SERIES_VAR)
    def test_the_expansion_actually_happens(self, lang, text):
        r, body = _ask(lang, ["h = exp(x)"], text)
        assert r.get("status") == "completed", (text, r.get("error"))
        # 読めないと `h = e^x` のまま `completed` が返る。
        assert "x^{3}" in body or "x^3" in body, (text, body)

    def test_it_reads_the_shared_fragment(self):
        """三十五問目で括り出したものを、この鍵も読むこと。"""
        assert "{{en_op_var_head_wide}}" in _P["taylor_expand_nl"]["en"]
        assert "{{ja_op_var_particle}}" in _P["taylor_expand_nl"]["ja"]

    def test_every_key_that_names_an_operation_variable_reads_it(self):
        """**fragment を作っても、読む側を全部数えないと残る。** 三十五問目で
        `def_integral` を直し、同じ穴が `taylor_expand_nl` に残っていた。"""
        users_en = {k for k, v in _P.items()
                    if not k.startswith("_") and isinstance(v, dict)
                    and "{{en_op_var_head_wide}}" in str(v.get("en", ""))}
        assert {"diff_rule", "def_integral", "taylor_expand_nl"} <= users_en, users_en


class TestTheExpandVerbIsOneVocabulary:
    @pytest.mark.parametrize("key,lang,text", EXPAND_VERB)
    def test_every_spelling_is_read(self, key, lang, text):
        assert nlp(key, lang).search(text) is not None, (key, lang, text)

    @pytest.mark.parametrize("key,lang,frag", [
        ("expand_components", "ja", "expand_verb_ja"),
        ("expand_components", "en", "expand_verb_en"),
        ("angular_coupled_state", "ja", "expand_verb_ja"),
        ("angular_coupled_state", "en", "expand_verb_en"),
    ])
    def test_no_key_spells_the_verb_itself(self, key, lang, frag):
        assert "{{" + frag + "}}" in _P[key][lang], (key, lang)

    @pytest.mark.parametrize("word", ["展開", "分解", "書き下"])
    def test_the_japanese_vocabulary(self, word):
        import re
        assert re.search(fragment("expand_verb_ja"), word), word

    @pytest.mark.parametrize("word", ["expand", "decompose", "write", "write out"])
    def test_the_english_vocabulary(self, word):
        import re
        assert re.search(fragment("expand_verb_en"), word, re.I), word
