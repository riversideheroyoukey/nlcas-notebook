r"""定積分の語彙 —— 評価の動詞が自前綴りで、`求めて` も `find` も落ちていた。

実測 2026-09-07。`定積分` を持つ3鍵のうち、`def_integral` は三十五・三十六問目で
共有 fragment へ寄せてあったが、残り2つは自前綴りのままだった。

| 打った文 | 直す前 |
|---|---|
| `この定積分を**求めて**` | failed |
| `**この積分**を評価して` | failed（対象語に無い） |
| `E1 を**求めて**` | failed |
| `**find** this integral` | **「`integrate indefinitely` と解釈しました」**（不定積分へ） |
| `**calculate** this integral` | 同上 |
| `**integrate** 1 over the region x^2+y^2 <= 1` | **指示が式として登録される** |

**不定積分に流れるのがいちばん重い。** 研究者は範囲つきの定積分を評価してほしくて
打っているのに、別の操作の名前が返る。この経路の註は 2026-08-03 に同型
（`evaluate this integral` が `\int x^2+1 \, devaluate` を返した）を記録しており、
**そのとき動詞を1つ足しただけで一覧を共有しなかった**ので、`find`/`calculate` に
同じ穴が残っていた。

**`region_integral[en]` は `definite integrate` 限定だった。** 素の `integrate` で
書くと領域積分の読みに当たらず、文がそのまま**式として登録**される（二十四問目の形）。

**E-id のときは譲る設計は保たれている。** `E5 を計算して` は「積分に固有の要求」
ではないので、対象が積分フォームでなければ供給側（supervisor）に渡す——動詞を
広げてもその判断は消費者側にあり、変わらない。
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
_P = json.loads(pathlib.Path("i18n/nl_patterns.json").read_text())

#: 領域積分の綴り。**表示専用フォームとして登録される**のが正しい。
REGION = [
    ("ja", "x^2+y^2 <= 1 の領域で 1 を定積分して"),
    ("en", "definite integrate 1 over the region x^2+y^2 <= 1"),
    ("en", "definite-integrate 1 over the region x^2+y^2 <= 1"),
    ("en", "integrate 1 over the region x^2+y^2 <= 1"),
]
#: 範囲つき定積分フォームの評価。**どれも 1/3。**
EVALUATE = [
    ("ja", "この定積分を評価して"), ("ja", "この定積分を計算して"),
    ("ja", "この定積分を求めて"), ("ja", "この積分を評価して"),
    ("ja", "この積分を求めて"), ("ja", "E1 を評価して"), ("ja", "E1 を求めて"),
    ("en", "evaluate this integral"), ("en", "compute this integral"),
    ("en", "find this integral"), ("en", "calculate this integral"),
    ("en", "determine this integral"),
    ("en", "evaluate E1"), ("en", "find E1"),
]


def _ask(lang, prep, text):
    nb.start_engine(title="di", mode="standard", language=lang, domains=CM)
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


class TestTheRegionIntegralIsRegistered:
    @pytest.mark.parametrize("lang,text", REGION)
    def test_it_becomes_a_display_only_form(self, lang, text):
        r, body = _ask(lang, [], text)
        assert r.get("status") == "registered", (text, r.get("status"))
        # 指示が式として登録されると、文がそのまま canonical になる。
        assert r"\iint" in body, (text, body)
        assert "integrate" not in body and "領域" not in body, (text, body)


class TestEveryComputeVerbEvaluatesTheIntegral:
    @pytest.mark.parametrize("lang,text", EVALUATE)
    def test_it_gives_one_third(self, lang, text):
        r, body = _ask(lang, [r"\int_0^1 x^2 dx"], text)
        assert r.get("status") == "completed", (text, r.get("error"))
        assert "1" in body and "3" in body, (text, body)

    @pytest.mark.parametrize("lang,text", EVALUATE)
    def test_it_never_falls_to_the_indefinite_integral(self, lang, text):
        """**別の操作の名前が返ってはいけない。** 2026-08-03 に同型が記録され、
        そのとき動詞を1つ足しただけで一覧を共有しなかった。"""
        r, _ = _ask(lang, [r"\int_0^1 x^2 dx"], text)
        msg = str(r.get("error") or "")
        assert "indefinit" not in msg and "不定積分" not in msg, (text, msg)


class TestTheVerbListIsShared:
    @pytest.mark.parametrize("key,lang,frag", [
        ("evaluate_integral", "ja", "compute_verb_ja"),
        ("evaluate_integral", "en", "compute_verb_en"),
    ])
    def test_no_key_spells_the_compute_verb_itself(self, key, lang, frag):
        assert "{{" + frag + "}}" in _P[key][lang], (key, lang)

    def test_the_region_reading_accepts_a_bare_integrate(self):
        assert "(?:definite[-\\s]?)?integrate" in _P["region_integral"]["en"]


class TestWhatMustNotChange:
    def test_a_plain_expression_is_left_to_the_supervisor(self):
        """**E-id のときは譲る。** 対象が積分フォームでなければ、
        `E1 を求めて` は積分に固有の要求ではない。"""
        r, _ = _ask("ja", ["f = x + 1"], "E1 を求めて")
        assert r.get("status") != "completed" or "int" not in str(r.get("latex")), r
