r"""微分の「どの変数で・どの種別か」——いくつの表が別々に答えていたか。

実測 2026-09-06（worklist の `derivative`、47行・4鍵）。

| 問い | 答えていた場所 |
|---|---|
| どの変数で | `diff_rule[en]` の自前綴り（`with respect to｜w.r.t.｜by`）／fragment `en_op_var_head`（`with respect to｜w.r.t.｜for`）／2a-sym の guard |
| 〃（日本語） | `diff_rule[ja]` の `(?:で｜について)` と `に関する`／`ja_op_var_tail` の `(?:で｜について)` |
| どの種別か | `diff_rule[en]` 内の `(?:total｜partial)?`／`total_derivative_marker`／`partial_derivative_marker`／`ja_deriv_word` |

**silent-wrong は4種あった:**

| 打った文 | 直す前 |
|---|---|
| `take the total derivative of E1 for t` | **`var='total'`** ← 種別の語が変数に |
| `differentiate E1 totally with respect to t` | **`var='totally'`** |
| `differentiate E1 by t`（`t` が登録済み） | **`t` が E-id に書き換わる** |
| `E1 の t に関する全微分を求めて`（同上） | **同上** |

後2つは 2a-sym の guard の穴で、**どちらも `diff_rule` が受ける綴り**である
——読みが受けるのに guard が知らない、という食い違いそのもの。

**`by｜in｜over` を共有 `en_op_var_head` に足すのは誤り**（実測）: あれは共有
なので `multiply both sides of E2 by A` の `A` まで守り、日本語側は守らないので
**日英が割れる**（コーパス2005対で10件変化、うち2件がこれ）。英語の guard は
日本語と同じ**構造**——「操作の語 … 前置詞 記号」——にする。語順が逆で固定長
lookbehind では跨げないので、全文検査の枝として書く。

**`に対して` は入れない。** 数学の日本語として `t で微分`／`t について微分`／
`t に関する微分`／`t による微分` は実在するが、`t に対して微分` はそうではない
（`laplace` の別名を削除したときと同じ判断——実在する綴りだけを入れる）。
"""
import asyncio
import types

import pytest

import ui.notebook_bridge as nb
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operation_validator import OperationValidator
from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline
from ui.supervisor_alias_rewrite import rewrite_symbols_to_eids

EN_HEADS = ["with respect to", "w.r.t.", "wrt", "for", "by", "in", "over"]
EN_KINDS = [("total ", "total"), ("partial ", "partial"), ("ordinary ", "total")]
JA_FRAMES = [("で", "E1 を t で{kw}して"),
             ("について", "E1 を t について{kw}して"),
             ("に関する", "E1 の t に関する{kw}を求めて"),
             ("による", "E1 の t による{kw}を求めて")]
JA_KINDS = [("全微分", "total"), ("偏微分", "partial"), ("常微分", "total"),
            ("全導関数", "total"), ("偏導関数", "partial")]


def _ask(lang, text):
    """(種別, var) を返す。値ではなく**何を頼んだと読まれたか**を見る。"""
    nb.start_engine(title="dvk", mode="standard", language=lang, domains=[])
    seen: list = []
    try:
        eng = nb.get_engine()
        sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                            domains=[], primary_backend="maxima")
        orig = eng.run_operation

        def spy(op, args):
            seen.append((op, dict(args)))
            return orig(op, args)
        eng.run_operation = spy

        def run(x):
            return asyncio.get_event_loop().run_until_complete(
                run_interpret_pipeline(
                    InterpretRequest(tokens=[InputToken(type="text", value=x)]),
                    eng, sup))
        run("q を時間依存関数として定義する" if lang == "ja"
            else "Define q as a time-dependent function")
        run("f = q^2 + t")
        seen.clear()
        resp = run(text)
        got = [(o, a) for o, a in seen if o == "differentiate"]
        if not got:
            return None, None, resp
        a = got[-1][1]
        kind = ("total" if a.get("_is_total_derivative") else
                "partial" if a.get("_is_partial_derivative") else None)
        return kind, a.get("var"), resp
    finally:
        nb.stop_engine()


class TestEveryVariableHeadReachesTheVariable:
    """`for`/`in`/`over` では `diff_rule` が当たらず、**種別の語が変数**になっていた。"""

    @pytest.mark.parametrize("head", EN_HEADS)
    @pytest.mark.parametrize("kw,want", EN_KINDS)
    def test_english(self, head, kw, want):
        kind, var, _ = _ask("en", f"take the {kw}derivative of E1 {head} t")
        assert var == "t", (head, kw, var)
        assert kind == want, (head, kw, kind)

    @pytest.mark.parametrize("particle,frame", JA_FRAMES)
    @pytest.mark.parametrize("kw,want", JA_KINDS)
    def test_japanese(self, particle, frame, kw, want):
        kind, var, _ = _ask("ja", frame.format(kw=kw))
        assert var == "t", (particle, kw, var)
        assert kind == want, (particle, kw, kind)

    @pytest.mark.parametrize("adv,want", [("totally", "total"),
                                          ("partially", "partial")])
    def test_the_adverb_is_not_the_variable(self, adv, want):
        kind, var, _ = _ask("en", f"differentiate E1 {adv} with respect to t")
        assert var == "t", (adv, var)
        assert kind == want, (adv, kind)


class TestTheGuardKnowsEverySpellingTheReadingAccepts:
    """読みが受ける綴りは guard も守る。片方だけ知っているのが穴だった。"""

    @staticmethod
    def _rewrite(text):
        eng = types.SimpleNamespace(lhs_alias_map={"t": "E9"},
                                    expressions={"E9": None})
        return rewrite_symbols_to_eids(text, eng)

    @pytest.mark.parametrize("text", [
        "differentiate E1 by t",
        "differentiate E1 in t",
        "differentiate E1 over t",
        "differentiate E1 with respect to t",
        "differentiate E1 for t",
        "take the total derivative of E1 by t",
        "take the partial derivative of E1 in t",
        "E1 を t で全微分して",
        "E1 を t について偏微分して",
        "E1 の t に関する全微分を求めて",
        "E1 の t による偏導関数を求めて",
    ])
    def test_a_derivative_variable_is_never_rewritten(self, text):
        assert "E9" not in self._rewrite(text), text

    @pytest.mark.parametrize("text", [
        "multiply both sides of E2 by t",
        "divide both sides of E5 by t",
        "E1 の両辺を t で割る",
        "E1 を t で置き換える",
    ])
    def test_a_non_derivative_operand_is_still_rewritten(self, text):
        """**共有 fragment を広げてはいけない理由。** `by` を `en_op_var_head` に
        入れると両辺の演算子まで守られ、日本語側は守らないので日英が割れる
        （実測: コーパス2005対で10件変化、うち2件がこれ）。"""
        assert "E9" in self._rewrite(text), text


class TestTheVocabularyLivesInOnePlace:
    def test_diff_rule_en_uses_the_shared_head(self):
        import json
        import pathlib
        d = json.loads(pathlib.Path("i18n/nl_patterns.json").read_text(
            encoding="utf-8"))
        assert "{{en_op_var_head_wide}}" in d["diff_rule"]["en"]
        assert "with\\s+respect\\s+to" not in d["diff_rule"]["en"], (
            "変数の前置詞をまた自前で綴っている")
        assert "{{en_op_var_head}}" in d["_fragments"]["en_op_var_head_wide"]

    def test_diff_rule_ja_uses_the_shared_particle(self):
        import json
        import pathlib
        d = json.loads(pathlib.Path("i18n/nl_patterns.json").read_text(
            encoding="utf-8"))
        assert d["diff_rule"]["ja"].count("{{ja_op_var_particle}}") == 2
        assert "{{ja_op_var_particle}}" in d["_fragments"]["ja_op_var_tail"]

    @pytest.mark.parametrize("word", ["total", "partial", "totally",
                                      "partially", "ordinary"])
    def test_a_kind_word_is_never_an_operand(self, word):
        from notebook.runtime.nl_patterns import nlp
        assert nlp("operand_stopwords", "en").match(word), word


class TestWhatWasDeliberatelyLeftOut:
    @pytest.mark.parametrize("text", ["E1 を t に対して全微分して",
                                      "E1 を t に対して偏微分して"])
    def test_ni_taishite_is_not_added(self, text):
        """`t に対して微分` は数学の日本語として実在しない。実在する綴りだけ入れる。"""
        _, var, resp = _ask("ja", text)
        assert var != "t" or resp.get("status") != "completed", text
