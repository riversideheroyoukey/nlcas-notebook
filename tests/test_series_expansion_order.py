r"""「N次まで展開」は代数展開ではない —— 次数を書いた文は級数展開である。

実測 2026-09-06（`E1 = exp(x)`。代数展開 → `%e^x`（何も起きない）／
3次テイラー → `x^3/6+x^2/2+x+1`。**多項式で測ると両方の読みが一致してしまい
空振りする**——最初に `(x+1)^2` で測って見落としかけた）。

| 打った文 | 直す前 |
|---|---|
| `E1 を x について 3次まで展開して` | **completed / `expand` / `%e^x`** |
| `E1 を x で 3次まで展開して` | 同上 |
| `E1 を x のまわりで 3次まで展開して` | 同上 |
| `E1 を x について 3次の展開をして` | 同上 |
| `E1 を x について ３次まで展開して`（全角） | 同上 |
| `E1 を x について 3次まで展開せよ` | 同上 |
| `E1 を 3次まで展開して`（変数なし） | 同上 |
| `expand E1 to 3rd order`（変数なし） | 同上 |
| `expand E1 to order 3`（変数なし） | 同上 |
| `E1 を x について 三次まで展開して`（漢数字） | failed |
| `E1 を x について 3次まで級数展開して` | failed |

**研究者が次数を書いているのに代数展開が走り、何も起きないまま成功が返る。**

原因は2つ。**(a) `taylor_expand_nl[ja]` が「テイラー／テーラー／マクローリン」の
語を必須にしていた**——英語側は `expand … to order N in x` だけで通っており、
十三問目が英語で直したときに日本語が残った（十八問目と同じ日英の非対称）。
**(b) 数字の語彙が2つあった**——当時の `ja_digit`（2026-09-10 に `ordinal_token` へ統合）は全角を知らず、`taylor_expand_nl` は
自前で `[0-9０-９]+` と書いていたので、**漢数字が落ちていた**。

**変数が書かれていない `N次まで展開` は推測しない。** `taylor_expand` へ回して
`pending_op` に載せ、この系に1つある訊き方
（`interpreter.missing_field.var`）で変数を訊く——自前で `str(_e)` を返すと、
監督層が日本語で返している同じ問いに英語の生の文言で答えることになる。

**主語は E-id に限る。** コーパスの
`イジング模型 正方格子 2x2 の高温展開を4次まで求めて` は**式として登録される行**
で、`展開` と `4次` を含む——自由な主語を許すと横取りする。
"""
import asyncio

import pytest

import ui.notebook_bridge as nb
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.nl_patterns import nlp
from notebook.runtime.operation_validator import OperationValidator
from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline

#: 次数と変数を書いた文 —— 級数展開に届くこと。
SERIES = [
    ("ja", "E1 を x について 3次まで展開して"),
    ("ja", "E1 を x で 3次まで展開して"),
    ("ja", "E1 を x のまわりで 3次まで展開して"),
    ("ja", "E1 を x について 3次の展開をして"),
    ("ja", "E1 を x について 三次まで展開して"),
    ("ja", "E1 を x について ３次まで展開して"),
    ("ja", "E1 を x について 3次まで級数展開して"),
    ("ja", "E1 を x について 3次まで展開せよ"),
    ("ja", "E1 を x について 3次までテイラー展開して"),
    ("ja", "E1 を x について 3次までマクローリン展開して"),
    ("en", "expand E1 in x to order 3"),
    ("en", "expand E1 to third order in x"),
    ("en", "Taylor expand E1 in x to order 3"),
]
#: 次数を書かない文 —— 代数展開のまま。
ALGEBRA = [("ja", "E1 を展開して"), ("ja", "E1 を展開する"), ("ja", "E1 を展開せよ"),
           ("en", "expand E1"), ("en", "expand E1 algebraically")]
#: 次数はあるが変数が無い文 —— 訊く（推測しない）。
ASK = [("ja", "E1 を 3次まで展開して"), ("ja", "E1 を 三次まで展開せよ"),
       ("en", "expand E1 to 3rd order"), ("en", "expand E1 to order 3")]


def _ask(lang, text):
    nb.start_engine(title="series", mode="standard", language=lang, domains=[])
    try:
        eng = nb.get_engine()
        sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                            domains=[], primary_backend="maxima")

        def run(x):
            return asyncio.get_event_loop().run_until_complete(
                run_interpret_pipeline(
                    InterpretRequest(tokens=[InputToken(type="text", value=x)]),
                    eng, sup))
        run("f = exp(x)")
        resp = run(text)
        val = None
        for e in eng.expressions.values():
            val = e.canonical
        return resp, val
    finally:
        nb.stop_engine()


class TestAnOrderMeansASeries:
    @pytest.mark.parametrize("lang,text", SERIES)
    def test_reaches_taylor(self, lang, text):
        resp, _ = _ask(lang, text)
        assert resp.get("operation") == "taylor_expand", (text, resp.get("operation"))
        args = resp.get("args") or {}
        assert args.get("var") == "x", (text, args)
        assert str(args.get("order")) == "3", (text, args)

    @pytest.mark.parametrize("lang,text", ALGEBRA)
    def test_no_order_stays_algebraic(self, lang, text):
        """**変えてはいけないもの。** 次数を書かない展開は代数展開である。"""
        resp, val = _ask(lang, text)
        assert resp.get("status") == "completed", (text, resp.get("status"))
        assert val == "%e^x", (text, val)

    @pytest.mark.parametrize("lang,text", ASK)
    def test_a_missing_variable_is_asked_for_not_guessed(self, lang, text):
        resp, val = _ask(lang, text)
        assert resp.get("operation") == "taylor_expand", (text, resp.get("operation"))
        assert resp.get("status") == "needs_confirmation", (text, resp.get("status"))
        assert not (resp.get("args") or {}).get("var"), resp.get("args")
        assert val == "exp(x)", (text, val)          # 代数展開が走っていない
        prompt = resp.get("confirmation_prompt") or ""
        # 訊き方はこの系に1つ（`interpreter.missing_field.var`）。生の英語の
        # `Missing required fields:` を返さない。
        assert "Missing required fields" not in prompt, prompt
        assert ("変数" in prompt) if lang == "ja" else ("variable" in prompt), prompt


class TestWhatMustNotBeClaimed:
    """コーパスに実在する、**横取りしてはいけない**綴り。"""

    @pytest.mark.parametrize("text", [
        "イジング模型 正方格子 2x2 の高温展開を4次まで求めて",
        "ハイゼンベルク模型の高温展開を一般式で4次まで求めて",
    ])
    def test_a_high_temperature_expansion_line_is_not_claimed(self, text):
        assert nlp("series_expand_no_var", "ja").match(text) is None, text
        assert nlp("taylor_expand_nl", "ja").match(text) is None, text

    @pytest.mark.parametrize("lang,text", [
        ("ja", "E1 を展開して"), ("en", "expand E1"),
        ("ja", "E1 の 3次の係数を取り出して"), ("en", "extract the order 3 term of E1"),
    ])
    def test_the_no_var_branch_does_not_claim_it(self, lang, text):
        assert nlp("series_expand_no_var", lang).match(text) is None, text

    @pytest.mark.parametrize("lang,text", [("ja", "E9 を 3次まで展開して"),
                                           ("en", "expand E9 to 3rd order")])
    def test_an_unregistered_expression_id_is_not_claimed(self, lang, text):
        """守りは**2枚**——パターンの E-id 限定と、ハンドラの「登録済みか」検査。

        測ると両者は同じ答えを出す（`高温展開` はパターンが止め、`E9` は
        ハンドラが止める）。**パターン側だけ壊す変異は1件も落とさない**ので、
        ここでハンドラ側を固定する。冗長性であって死骸ではない
        （2a-sym の lookbehind と同じ判断）。
        """
        resp, val = _ask(lang, text)
        assert resp.get("operation") != "taylor_expand", (text, resp)
        assert val == "exp(x)", (text, val)


class TestTheDigitVocabularyIsOneList:
    """数の綴りは `ordinal_token` **1つ**。`taylor_expand_nl` の自前の分を畳んだ。

    2026-09-10 に `ja_digit`（`一`〜`九`）を廃してここへ寄せた。同じ問いに
    2つの表があり、**片方にしか `十` が無かった**ので、`十階微分` は通るのに
    `十次まで展開` だけが「解釈できません」で落ちていた（四十八問目）。
    """

    @pytest.mark.parametrize("digit", ["3", "３", "三", "12", "１２", "十", "tenth"])
    def test_the_number_vocabulary_reads_every_spelling(self, digit):
        import re
        from notebook.runtime.nl_patterns import fragment
        assert re.fullmatch(fragment("ordinal_token"), digit), digit

    def test_taylor_pattern_uses_it_rather_than_repeating(self):
        import json
        import pathlib
        d = json.loads(pathlib.Path("i18n/nl_patterns.json").read_text(
            encoding="utf-8"))
        assert "{{ordinal_token}}" in d["taylor_expand_nl"]["ja"]
        assert "[0-9０-９]" not in d["taylor_expand_nl"]["ja"], "数字をまた自前で綴っている"
