r"""範囲の終点は1つの読み —— `まで`/`へ`/`に`/`up to` が鍵ごとに割れていた。

実測 2026-09-06。三十五問目で始点（`から`/`from`）を畳んだので、その対を測った。
終点は15鍵に現れるが役割は2つで、**そのどちらの中でも鍵ごとに自前綴り**だった:

* **変数の写像**（`から … へ／に` / `from … to`）—— `change_vars`・`z_transform`
* **次数の上限**（`… まで／で` / `(up) to (order) N`）—— `f1_spec_depth`・
  `taylor_expand_nl`・`series_expand_no_var`・`numeric_fork_sweep_nl`

| 打った文 | 直す前 |
|---|---|
| `E1 を x,y から r,t **に** … 変数変換して` | **`y^2+x^2=y^2+x^2`**（恒等式。変換が起きずに `completed`） |
| `… in general form **to 4th order**` | 既定の3次（⟨H⁴⟩ が無い） |
| `expand E1 in x around 0 **up to** 3rd order` | **`h = e^x`**（展開されず `completed`） |
| `sweep a from 0 **up to** 2 with 3 points` | failed |

**3つとも silent-wrong である。** とくに変数変換の恒等式は、研究者が極座標へ移した
つもりで `x^2+y^2` のまま先へ進む形になる。

**序数の綴りが2つ残っていた。** 三十三問目で `ordinal_token` に畳んだとき、
`taylor_expand_nl` と `series_expand_no_var` は**既に語序数を持っていたので触らな
かった**——結果、同じ表が2箇所に残った。ここで共有 fragment に寄せた。

**触らなかった綴り**: `0 から 1 までで定積分`・`0 から 2 へ掃引` は日本語として
不自然なので広げていない（実在しない綴りは足さない）。
"""
import asyncio
import json
import pathlib

import pytest

import ui.notebook_bridge as nb
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operation_validator import OperationValidator
from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline

CM = ["classical_mechanics"]
CD = ["condensed_matter"]
_P = json.loads(pathlib.Path("i18n/nl_patterns.json").read_text())

CHANGE_VARS = [
    ("ja", ["f = x^2 + y^2"],
     "E1 を x,y から r,t へ x=r*cos(t), y=r*sin(t) で変数変換して"),
    ("ja", ["f = x^2 + y^2"],
     "E1 を x,y から r,t に x=r*cos(t), y=r*sin(t) で変数変換して"),
    ("en", ["f = x^2 + y^2"],
     "change of variables in E1 from x,y to r,t via x=r*cos(t), y=r*sin(t)"),
]
HIGH_T = [
    ("ja", "ハイゼンベルク模型の高温展開を一般式で4次まで求めて"),
    ("ja", "ハイゼンベルク模型の高温展開を一般式で4次で求めて"),
    ("en", "Find the high-temperature expansion of the Heisenberg model in general form up to 4th order"),
    ("en", "Find the high-temperature expansion of the Heisenberg model in general form to 4th order"),
    ("en", "Find the high-temperature expansion of the Heisenberg model in general form to order 4"),
]
SERIES = [
    ("ja", "E1 を x について 0 のまわりで 3次まで展開して"),
    ("ja", "E1 を x について 0 のまわりで 3次で展開して"),
    ("en", "expand E1 in x around 0 to order 3"),
    ("en", "expand E1 in x around 0 to 3rd order"),
    ("en", "expand E1 in x around 0 up to 3rd order"),
]
SWEEP = [
    ("ja", "E1 で a を 0 から 2 まで 3 点で掃引して"),
    ("en", "sweep a from 0 to 2 with 3 points for E1"),
    ("en", "sweep a from 0 up to 2 with 3 points for E1"),
]


def _ask(dom, lang, prep, text):
    nb.start_engine(title="rg", mode="standard", language=lang, domains=dom)
    try:
        eng = nb.get_engine()
        sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                            domains=dom, primary_backend="maxima")

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


class TestTheVariableMappingEndpoint:
    @pytest.mark.parametrize("lang,prep,text", CHANGE_VARS)
    def test_the_change_actually_happens(self, lang, prep, text):
        r, body = _ask(CM, lang, prep, text)
        assert r.get("status") == "completed", (text, r.get("error"))
        # 変換されないと `y^2+x^2=y^2+x^2` という**恒等式**が返る。移った後の
        # 式に旧変数は残らないので、`x` が消えて `r` が出ていることで判る。
        assert "r" in body, (text, body)
        assert "x" not in body and "y" not in body, (text, body)

    @pytest.mark.parametrize("key", ["change_vars", "z_transform_by_definition"])
    def test_the_mapping_marker_is_one_fragment(self, key):
        assert "{{ja_map_to}}" in _P[key]["ja"], key


class TestTheOrderEndpoint:
    @pytest.mark.parametrize("lang,text", HIGH_T)
    def test_the_high_temperature_order_is_honoured(self, lang, text):
        r, body = _ask(CD, lang, [], text)
        assert r.get("status") == "completed", (text, r.get("error"))
        assert "H^4" in body, (text, body)

    @pytest.mark.parametrize("lang,text", SERIES)
    def test_the_series_order_is_honoured(self, lang, text):
        r, body = _ask(CM, lang, ["h = exp(x)"], text)
        assert r.get("status") == "completed", (text, r.get("error"))
        assert "x^{3}" in body or "x^3" in body, (text, body)

    @pytest.mark.parametrize("lang,text", SWEEP)
    def test_the_sweep_range_is_read(self, lang, text):
        """数値層は承認を訊く（`needs_confirmation`）のが正しい挙動。"""
        r, _ = _ask(CM, lang, ["g = a^2"], text)
        assert r.get("status") == "needs_confirmation", (text, r.get("status"))

    @pytest.mark.parametrize("key,lang", [
        ("f1_spec_depth", "en"), ("taylor_expand_nl", "en"),
        ("series_expand_no_var", "en"), ("numeric_fork_sweep_nl", "en"),
    ])
    def test_the_english_endpoint_is_one_fragment(self, key, lang):
        assert "{{en_upto}}" in _P[key][lang], (key, lang)


class TestTheOrdinalTableIsNotDuplicated:
    r"""三十三問目で畳んだ表が、2鍵に残っていた。"""

    @pytest.mark.parametrize("key", ["taylor_expand_nl", "series_expand_no_var",
                                     "f1_spec_depth"])
    def test_no_key_spells_the_ordinal_words_itself(self, key):
        en = _P[key]["en"]
        assert "first|second|third" not in en, (key, en)
        assert "{{ordinal_token}}" in en, key
