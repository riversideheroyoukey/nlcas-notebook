r"""`から` / `from` は5つの役割 —— 変数の前置詞と文末が鍵ごとに割れていた。

実測 2026-09-06。`from` は11鍵に現れ、役割が5つある（範囲の始点／両辺から引く／
左右から掛ける／式から解く／式を引く）。役割そのものは正しく分かれていたが、
**その中で「変数をどの語で導くか」と「文はどこで終わるか」が鍵ごとに自前綴り**
だった。

| 打った文 | 直す前 |
|---|---|
| `integrate x^2 in x from 0 to 1` | **`x^3/3 + C`**（不定積分。範囲が黙って落ちる） |
| `integrate x^2 for x from 0 to 1` | 同上 |
| `x^2 を x について 0 から 1 まで定積分して` | failed（`で` しか受けない） |
| `subtract a from both sides of E1.` | failed（末尾のピリオド） |

**範囲を書いたのに不定積分が返るのは silent-wrong である。** `+ C` が付いていても、
研究者は `from 0 to 1` と書いている。

**fragment の名前が嘘をついていた。** 変数の前置詞は二十二問目で
`en_deriv_var_head` / `ja_deriv_var_particle` に括り出されていたが、それを読むのは
`diff_rule` だけで、定積分は自前で `over|with respect to|w.r.t.|by` と綴っていた
（`for` と `in` を知らない）。**微分だけのものではなくなった**ので
`en_op_var_head_wide` / `ja_op_var_particle` に改名した——名前が嘘をつくと次の人が
3つ目のコピーを書く。

**文末のピリオドは 14鍵・51行に及んでいた。** コーパスの英文に `.` を足すと、
当たっていた鍵が外れる——`Solve E4 for w.` `Evaluate this definite integral.`
`evolve c(m) under E3.` など。共有 fragment `en_sentence_end` を1つ置いて 14→0。
`solve_from_expr` だけが `\.?\s*$` を持っており、**答えは1鍵にあって他になかった**。
"""
import asyncio
import json
import pathlib

import pytest

import ui.notebook_bridge as nb
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.nl_patterns import nlp
from notebook.runtime.operation_validator import OperationValidator
from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline

DOM = ["classical_mechanics"]
_PATTERNS = json.loads(pathlib.Path("i18n/nl_patterns.json").read_text())
_CORPUS = json.loads(pathlib.Path("tests/corpus/instruction_corpus.json").read_text())
_ROWS = _CORPUS["rows"] if isinstance(_CORPUS, dict) else _CORPUS

#: 定積分。**どの前置詞でも 1/3**。
INTEGRAL = [
    ("ja", "x^2 を x で 0 から 1 まで定積分して"),
    ("ja", "x^2 を x について 0 から 1 まで定積分して"),
    ("ja", "x^2 を x に関して 0 から 1 まで定積分して"),
    ("en", "integrate x^2 over x from 0 to 1"),
    ("en", "integrate x^2 with respect to x from 0 to 1"),
    ("en", "integrate x^2 w.r.t. x from 0 to 1"),
    ("en", "integrate x^2 by x from 0 to 1"),
    ("en", "integrate x^2 in x from 0 to 1"),
    ("en", "integrate x^2 for x from 0 to 1"),
]
#: `から` の他の役割（対照——壊していないこと）。
OTHER_ROLES = [
    (["y = x + a"], "ja", "E1 の両辺から a を引いて", "y-a"),
    (["y = x + a"], "en", "subtract a from both sides of E1", "y-a"),
    (["y = x + a"], "en", "subtract a from both sides of E1.", "y-a"),
    (["y = 2*x + 1"], "ja", "E1 から x を求めて", "x="),
    (["y = 2*x + 1"], "en", "find x from E1", "x="),
    (["y_1 = x + 1", "y_2 = x"], "ja", "E1 から E2 を引く", "y_"),
    (["y_1 = x + 1", "y_2 = x"], "en", "subtract E2 from E1", "y_"),
]


def _ask(lang, prep, text):
    nb.start_engine(title="fr", mode="standard", language=lang, domains=DOM)
    try:
        eng = nb.get_engine()
        sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                            domains=DOM, primary_backend="maxima")

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


class TestTheDefiniteIntegralKeepsItsRange:
    @pytest.mark.parametrize("lang,text", INTEGRAL)
    def test_every_variable_head_gives_one_third(self, lang, text):
        r, body = _ask(lang, [], text)
        assert r.get("status") == "completed", (text, r.get("error"))
        # 範囲が落ちると `x^3/3 + C`（不定積分）になる。
        assert "C" not in body and "x" not in body, (text, body)
        assert "1" in body and "3" in body, (text, body)

    @pytest.mark.parametrize("key,frag", [
        ("def_integral", "en_op_var_head_wide"),
        ("diff_rule", "en_op_var_head_wide"),
    ])
    def test_the_english_head_is_one_fragment(self, key, frag):
        assert "{{" + frag + "}}" in _PATTERNS[key]["en"], key

    @pytest.mark.parametrize("key", ["def_integral", "diff_rule"])
    def test_the_japanese_particle_is_one_fragment(self, key):
        assert "{{ja_op_var_particle}}" in _PATTERNS[key]["ja"], key

    def test_the_fragment_name_is_not_about_derivatives_only(self):
        """**名前が嘘をつくと次の人が3つ目のコピーを書く。**"""
        frags = _PATTERNS["_fragments"]
        assert "en_deriv_var_head" not in frags
        assert "ja_deriv_var_particle" not in frags
        assert "en_op_var_head_wide" in frags and "ja_op_var_particle" in frags


class TestTheOtherRolesOfFromStillWork:
    @pytest.mark.parametrize("prep,lang,text,want", OTHER_ROLES)
    def test_it_computes(self, prep, lang, text, want):
        r, body = _ask(lang, prep, text)
        assert r.get("status") == "completed", (text, r.get("error"))
        assert want.replace(" ", "") in body.replace(" ", ""), (text, body)


class TestATrailingPeriodIsNotPartOfTheInstruction:
    r"""コーパスの英文に `.` を足しても、当たっていた鍵は当たり続けること。

    実測 2026-09-06、直す前は **14鍵・51行**が外れていた。答えは
    `solve_from_expr` の `\.?\s*$` に1つだけあり、他の鍵は知らなかった。
    """

    #: コーパスに行が無い鍵は**コーパス駆動の列挙に映らない**——実際
    #: `both_sides_subtract` がそれで、14鍵の測定から漏れていた（三十問目の
    #: トレースと同じ形）。だから手持ちの綴りも足す。
    EXTRA = [
        "subtract a from both sides of E1", "add a to both sides of E1",
        "divide both sides of E1 by 2", "multiply both sides of E1 by 2",
        "Invert E1 and E2", "Invert E1, E2 for x_1, x_2",
        "integrate x^2 over x from 0 to 1", "solve E1 for x",
        "change of variables in E1 from x to r via x=r",
        "sweep x from 0 to 1 with 5 points",
    ]

    def test_no_key_loses_a_line_to_a_period(self):
        keys = [k for k, v in _PATTERNS.items()
                if not k.startswith("_") and isinstance(v, dict) and "en" in v]
        texts = [(r.get("en") or "").strip() for r in _ROWS] + self.EXTRA
        broken = []
        for k in keys:
            try:
                rx = nlp(k, "en")
            except Exception:
                continue
            for t in texts:
                if not t or t.endswith("."):
                    continue
                if rx.search(t) and not rx.search(t + "."):
                    broken.append(f"{k}: {t[:56]}")
        assert not broken, ("末尾のピリオドで当たらなくなる鍵:\n  "
                            + "\n  ".join(broken[:12]))

    def test_no_english_pattern_ends_with_a_bare_anchor(self):
        """**素の `\\s*$` を残さない**——残すとその鍵だけがピリオドを拒む。"""
        left = [k for k, v in _PATTERNS.items()
                if not k.startswith("_") and isinstance(v, dict)
                and isinstance(v.get("en"), str) and "\\s*$" in v["en"]]
        assert not left, left

    @pytest.mark.parametrize("key", [
        "solve_single", "evaluate_integral", "heisenberg_operator_nl",
        "operator_substitute_in_form", "expand_components", "laplace_define",
        "taylor_expand_nl",
        "both_sides_multiply_side", "evaluate_extremum",
        "z_transform_by_definition", "scalar_handoff_nl", "subst_replace",
        "operator_substitute_multi",
    ])
    def test_the_measured_keys_read_the_shared_fragment(self, key):
        assert "{{en_sentence_end}}" in _PATTERNS[key]["en"], key
