"""「どの変数について」を言う語 —— 同じ問いが言語ごとに別の場所で答えられていた。

「どのトークンが**文法**であってオペランドではないか」は、この系では2箇所で
答えられている:

- 日本語 …… `llm_supervisor._tokenize`（`_JP_PARTICLES_STRIP`）
- 英語 …… `i18n/nl_patterns.json` の `operand_stopwords[en]`

`operand_stopwords` の `ja` が `(?!)`（決して一致しない）なのは、コメントいわく
「`_tokenize` が既に日本語の助詞を剥がすから」。**その前提が偽だった**——剥がして
いたのは**1文字の助詞だけ**で、`について` は `に` ＋ `ついて` ではないから
`について解いて` は `ついて解いて` になり、別名 `解いて` はどのトークンにも
現れない。逆に英語側は複合語 `respect` / `w.r.t.` を知っていながら `over` を
知らなかった。**互いが相手の知っていることを知らない。**

実測 2026-09-05（`E1 = f = x^2 + a*x`、実パイプライン）:

    E1 を x に関して微分して     → failed（`で` と `について` は通る）
    E1 を x について積分して     → failed  ← 微分では通る同じ助詞
    E1 を x に関して積分して     → failed
    integrate E1 over x        → **completed**、正準形 `over*(x^2+a*x)+C`
                                  ← `over` が変数として CAS に届く（silent-wrong）

コーパスでは22行のトークン列が変わり、**全部が改善**（`ついて解いて` →
`解いて`、`ついて逆変換` → `逆変換`）。それらの行がそれでも動いていたのは
**インターセプタが生テキストで先に取っていたから**で、supervisor に落ちる経路は
壊れたままだった。
"""
import asyncio
import re

import pytest

import ui.notebook_bridge as nb
from notebook.runtime.llm_supervisor import LLMSupervisor, _tokenize
from notebook.runtime.nl_patterns import nlp
from notebook.runtime.operation_validator import OperationValidator
from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline


class TestACompoundParticleIsStrippedWhole:
    """`について` は `に` ＋ `ついて` ではない。"""

    @pytest.mark.parametrize("text,alias", [
        ("E4 を w について解いて", "解いて"),
        ("E4 を w に関して解いて", "解いて"),
        ("E1 を x について積分して", "積分"),
        ("E1 を x に関して微分して", "微分"),
        ("E1, E2 を a(1) について逆変換して", "逆変換"),
    ])
    def test_the_alias_is_exposed(self, text, alias):
        toks = _tokenize(text)
        assert alias in toks, (text, toks)
        assert not any(t.startswith("ついて") or t.startswith("関して") for t in toks), \
            (text, toks, "複合助詞の残骸がトークンになっている")

    def test_a_bare_compound_particle_becomes_nothing(self):
        assert "について" not in _tokenize("E3 を \\lambda について 0 のまわりで")
        assert "ついて" not in _tokenize("E3 を \\lambda について 0 のまわりで")

    def test_the_single_character_particles_still_go(self):
        """複合を先に置いても、元からの1文字の剥がしは効いたまま。"""
        assert "積分" in _tokenize("E1 を x で積分して")
        assert "微分" in _tokenize("E1 を x で微分して")

    def test_the_variable_itself_survives(self):
        toks = _tokenize("E4 を w について解いて")
        assert "w" in toks and "E4" in toks, toks


class TestAnEnglishPrepositionIsNotAVariable:
    """`over` は変数ではない —— `respect` / `w.r.t.` と同じ答えを返すべき語。"""

    @pytest.mark.parametrize("word", ["over", "around", "about", "respect", "with"])
    def test_it_is_grammar(self, word):
        assert nlp("operand_stopwords", "en").match(word), word

    @pytest.mark.parametrize("word", ["x", "theta", "omega"])
    def test_a_real_variable_is_not(self, word):
        assert not nlp("operand_stopwords", "en").match(word), word


def _run(eng, sup, text):
    return asyncio.get_event_loop().run_until_complete(
        run_interpret_pipeline(
            InterpretRequest(tokens=[InputToken(type="text", value=text)]), eng, sup))


#: (言語, 綴り, 期待する操作)。日英そろえてある —— この欠陥は片方の言語でしか
#: 答えていなかったことによるものなので。
SPELLINGS = [
    ("ja", "E1 を x について微分して", "differentiate"),
    ("ja", "E1 を x に関して微分して", "differentiate"),
    ("ja", "E1 を x で微分して", "differentiate"),
    ("ja", "E1 を x について積分して", "integrate_indefinite"),
    ("ja", "E1 を x に関して積分して", "integrate_indefinite"),
    ("ja", "E1 を x で積分して", "integrate_indefinite"),
    ("en", "differentiate E1 with respect to x", "differentiate"),
    ("en", "differentiate E1 w.r.t. x", "differentiate"),
    ("en", "differentiate E1 by x", "differentiate"),
    ("en", "integrate E1 with respect to x", "integrate_indefinite"),
    ("en", "integrate E1 w.r.t. x", "integrate_indefinite"),
    ("en", "integrate E1 over x", "integrate_indefinite"),
]

#: 操作ごとの正解。`over` が変数として届くと `over*(x^2+a*x)+C` になるので、
#: **値まで見ないと** silent-wrong は捕まらない。積分定数はセッション内で採番
#: される（`C`, `C_1`, …）ので、そこだけ正規表現で受ける。
EXPECTED = {
    "differentiate": r"2\*x\+a$",
    "integrate_indefinite": r"x\^3/3\+\(a\*x\^2\)/2\+C(_\d+)?$",
}


@pytest.fixture(scope="module", params=["ja", "en"])
def session(request):
    lang = request.param
    nb.start_engine(title="which_variable", mode="standard", language=lang,
                    domains=["classical_mechanics"])
    eng = nb.get_engine()
    sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                        domains=["classical_mechanics"], primary_backend="maxima")
    _run(eng, sup, "a をスカラーとする" if lang == "ja" else "Define a as a scalar")
    _run(eng, sup, "f = x^2 + a*x")
    yield lang, eng, sup
    nb.stop_engine()


def test_the_variable_reaches_the_cas(session):
    lang, eng, sup = session
    for spell_lang, text, want_op in SPELLINGS:
        if spell_lang != lang:
            continue
        resp = _run(eng, sup, text)
        assert resp.get("status") == "completed", (text, resp.get("status"),
                                                   resp.get("error"))
        got = (resp.get("canonical") or "").replace(" ", "")
        assert re.fullmatch(EXPECTED[want_op], got), (text, got)
