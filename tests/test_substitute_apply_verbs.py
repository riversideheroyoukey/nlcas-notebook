r"""「式を式に代入せよ」は1つの問い —— 動詞と前置詞が2箇所に別々に書かれていた。

実測 2026-09-06（`classical_mechanics`。`y = 2x`、`z = y + 1`）。この要求を読む鍵は
2つあり、**消費者が役割を分けている**:

* `transform_subst` —— **複数ソース／演算子のときだけ**受ける（註にそう書いてある）
* `dainyu` —— 単一ソースの可換な場合が落ちてくる先

問いは1つなのに、英語の動詞一覧が**二重に、しかも違う内容で**書かれていた:

| 鍵 | en の動詞 | ja の動詞 |
|---|---|---|
| `transform_subst` | `substitute｜apply｜plug` | `代入｜適用` |
| `dainyu` | **`substitute` のみ** | `代入｜適用` |

結果、単一ソースの英語だけが落ちていた（**日本語の鏡は4綴りとも通る**）:

| 打った文 | 直す前 |
|---|---|
| `apply E1 to E2` | failed（`E2 に E1 を適用して` は通る） |
| `plug E1 into E2` | failed |
| `substitute E1 in E2` | failed |
| `In E2, substitute E1` | failed（`E2 に E1 を代入して` は通る） |

意味を取り違える silent-wrong ではなく**全部 failed** だが、日英で書けるものが
違うという割れである。動詞・前置詞を共有 fragment に出し、目的語が先に来る語順を
両方の枝に足した。

**呼ぶ側は `pick_group`。** `dainyu` の消費者は `src1 or src2 or src3` と接尾辞を
並べており、枝を足すのに2箇所を触る必要があった（二十八問目で切り出した道具）。

**`E1 で x = 2 として` は触れていない。** `として` はこの系でいちばん多義な語尾で、
`fp_subst`（自由パラメータ）が既に claim している。広げるなら先に測る必要がある
（open_items SUBST-TOSHITE）。
"""
import asyncio

import pytest

import ui.notebook_bridge as nb
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.nl_patterns import fragment, nlp, pick_group
from notebook.runtime.operation_validator import OperationValidator
from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline

DOM = ["classical_mechanics"]

#: 式を式に代入する。`y = 2x` を `z = y + 1` へ → `z = 2x + 1`。
APPLY = [
    ("en", "Substitute E1 into E2"),
    ("en", "substitute E1 in E2"),
    ("en", "apply E1 to E2"),
    ("en", "plug E1 into E2"),
    ("en", "In E2, substitute E1"),
    ("en", "in E2, apply E1"),
    ("ja", "E2 に E1 を代入して"),
    ("ja", "E1 を E2 に代入して"),
    ("ja", "E2 に E1 を適用して"),
    ("ja", "E1 を E2 に適用して"),
]
#: 記号を値で置き換える側（対照）。`f = a x + b` の x に 2 → `2a + b`。
REPLACE = [
    ("en", "In E1, substitute 2 for x"),
    ("en", "substitute 2 for x in E1"),
    ("en", "In E1, replace x with 2"),
    ("en", "replace x with 2 in E1"),
    ("en", "set x = 2 in E1"),
    ("en", "let x = 2 in E1"),
    ("en", "put x = 2 in E1"),
    ("en", "substitute x = 2 into E1"),
    ("en", "plug x = 2 into E1"),
    ("ja", "E1 の x に 2 を代入して"),
    ("ja", "E1 の x を 2 で置き換えて"),
    ("ja", "E1 の x を 2 に置換して"),
    ("ja", "E1 の x に 2 を入れて"),
    ("ja", "E1 に x = 2 を代入して"),
]
#: 代入の読みが**取ってはいけない**もの（別の操作・別の欄）。
NOT_AN_APPLY = [
    "substitute 2 for x in E1",
    "In E1, substitute 2 for x",
    "In E1, substitute E7 for c(k,u) and E8 for c(-k,d)",
    "In E1, substitute mz for S^z_2",
]


def _run(eng, sup, text):
    return asyncio.get_event_loop().run_until_complete(
        run_interpret_pipeline(
            InterpretRequest(tokens=[InputToken(type="text", value=text)]), eng, sup))


def _ask(lang, prep, text):
    nb.start_engine(title="sb", mode="standard", language=lang, domains=DOM)
    try:
        eng = nb.get_engine()
        sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                            domains=DOM, primary_backend="maxima")
        for p in prep:
            _run(eng, sup, p)
        return _run(eng, sup, text)
    finally:
        nb.stop_engine()


class TestApplyingAnEquationToAnother:
    @pytest.mark.parametrize("lang,text", APPLY)
    def test_every_spelling_reaches_the_substitution(self, lang, text):
        r = _ask(lang, ["y = 2*x", "z = y + 1"], text)
        assert r.get("status") == "completed", (text, r.get("error"))
        got = (r.get("latex") or "").split("=")[-1].replace(" ", "")
        assert got in ("2x+1", "1+2x"), (text, r.get("latex"))


class TestReplacingASymbolStillWorks:
    @pytest.mark.parametrize("lang,text", REPLACE)
    def test_the_other_meaning_is_untouched(self, lang, text):
        r = _ask(lang, ["f = a*x + b"], text)
        assert r.get("status") == "completed", (text, r.get("error"))
        got = (r.get("latex") or "").split("=")[-1].replace(" ", "")
        assert got in ("2a+b", "b+2a"), (text, r.get("latex"))


class TestOneQuestionOnePlace:
    @pytest.mark.parametrize("key", ["transform_subst", "dainyu"])
    def test_both_keys_read_the_shared_verb_list(self, key):
        """**同じ問いなので、片方だけに動詞を足せる形にしない。**"""
        import json
        import pathlib
        src = json.loads(pathlib.Path("i18n/nl_patterns.json").read_text())[key]
        assert "{{en_subst_apply_verb}}" in src["en"], key
        assert "{{en_subst_into}}" in src["en"] or "tgt2" in src["en"], key
        assert "{{ja_subst_apply_verb}}" in src["ja"], key

    @pytest.mark.parametrize("verb", ["substitute", "apply", "plug"])
    def test_the_verb_list_holds_every_measured_verb(self, verb):
        assert verb in fragment("en_subst_apply_verb")

    @pytest.mark.parametrize("key,text,src,tgt", [
        ("dainyu", "apply E1 to E2", "E1", "E2"),
        ("dainyu", "In E2, substitute E1", "E1", "E2"),
        ("transform_subst", "plug E1 into E2", "E1", "E2"),
        ("transform_subst", "in E2, apply E1, E3", "E1, E3", "E2"),
    ])
    def test_the_branches_are_read_by_pick_group(self, key, text, src, tgt):
        """語順の枝は接尾辞を並べずに読む。"""
        m = nlp(key, "en").search(text)
        assert m is not None, (key, text)
        stem = "src" if key == "dainyu" else "srcs"
        assert pick_group(m, stem) == src, m.groupdict()
        assert pick_group(m, "tgt") == tgt, m.groupdict()

    @pytest.mark.parametrize("text", NOT_AN_APPLY)
    def test_the_apply_reading_claims_nothing_else(self, text):
        for key in ("transform_subst", "dainyu"):
            assert nlp(key, "en").search(text) is None, (key, text)
