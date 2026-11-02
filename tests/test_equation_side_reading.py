"""等式の「辺」の言い方 —— 右辺しか読めず、外れた綴りは**式全体**に効いていた。

2026-09-05 の調査（`docs/open_items.md`）。`E1 = x^2 + a*x = 0`（左辺が複合の等式）
で測ると、**英語側に silent-wrong が4件**あった:

| 打った文 | 直す前 |
|---|---|
| `simplify the right side of E1` | **completed**、canonical `x^2+a*x=0` ← 式全体 |
| `simplify the RHS of E1` | 同上 |
| `simplify the left-hand side of E1` | **completed**、canonical **`left-hand`** ← 語が式に |
| `set the left-hand side of E1 to 0` | **confirmed / define_function** ← 宣言になった |

原因は十三問目と同じ形——`rhs_op[en]` が `right[- ]hand side` を要求するので
`right side` / `RHS` は当たらず、別名走査が `simplify` を見て **`E1` 全体**に
適用し、`right side of` という語は捨てられる。`left-hand` が式になるのは
`side` が文法語で `left-hand` がそうでないため。

**②の仕組みはこれを捕まえなかった**——`left-hand` は研究者の語であって
システムの語ではないから。辺の語を文法語に入れたので、いまは捕まる。

**左辺は「左辺が名前でないとき」だけ意味がある。** H-1 の等式は canonical に
右辺しか持たず（CLAUDE.md）、左辺は `E` のような名前なので簡約する対象が無い。
そこは**理由つきで断る**（黙って右辺を出すのが、直す前の姿だった）。

**左辺に値を置く操作は無いので、断る。** 「左辺を 0 とする」は等式そのものを
書き直す別の要求で、黙って右辺に置けば研究者が書いたのと違う式が登録される。
"""
import asyncio

import pytest

import ui.notebook_bridge as nb
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.nl_patterns import nlp, pick_group
from notebook.runtime.operation_validator import OperationValidator
from ui.interpret_equation_ops import side_of
from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline


#: (言語, 綴り, どちらの辺か)。日英そろえてある。
SIDE_SPELLINGS = [
    ("ja", "E1 の右辺を整理して", "right"),
    ("ja", "E1 の左辺を整理して", "left"),
    ("en", "simplify the right-hand side of E1", "right"),
    ("en", "simplify the right side of E1", "right"),
    ("en", "simplify the RHS of E1", "right"),
    ("en", "simplify the left-hand side of E1", "left"),
    ("en", "simplify the left side of E1", "left"),
    ("en", "simplify the LHS of E1", "left"),
]


class TestTheSideIsRead:
    @pytest.mark.parametrize("lang,text,side", SIDE_SPELLINGS)
    def test_which_side(self, lang, text, side):
        m = nlp("rhs_op", lang).search(text)
        assert m is not None, text
        assert pick_group(m, "eid") == "E1", text
        assert side_of(m) == side, (text, m.groupdict())

    @pytest.mark.parametrize("lang,text,side", [
        ("ja", "E1 の右辺を 0 とする", "right"),
        ("ja", "E1 の左辺を 0 とする", "left"),
        ("en", "set the right-hand side of E1 to 0", "right"),
        ("en", "set the RHS of E1 to 0", "right"),
        ("en", "let the left side of E1 be 0", "left"),
        ("en", "regard the LHS of E1 as 0", "left"),
    ])
    def test_which_side_when_setting(self, lang, text, side):
        m = nlp("rhs_set", lang).search(text)
        assert m is not None, text
        assert side_of(m) == side, (text, m.groupdict())


class TestTheSideWordIsNotAnOperand:
    """`left-hand` が式として読まれ、canonical が `left-hand` になっていた。"""

    @pytest.mark.parametrize("word", ["side", "left-hand", "right-hand", "lhs", "rhs"])
    def test_it_is_grammar(self, word):
        assert nlp("operand_stopwords", "en").match(word), word

    @pytest.mark.parametrize("word", ["left", "right"])
    def test_the_bare_direction_is_not_grammar(self, word):
        """裸の `left`/`right` は `side` フィールドの**正当な列挙値**である。

        文法語に入れたら②が `multiply both sides … from the left` に誤って鳴った
        ——**フルスイートと実例再生成の両方が捕まえた**（実測 2026-09-05）。
        silent-wrong だったのは `left-hand` のほうで、裸の向きではない。
        """
        assert nlp("operand_stopwords", "en").match(word) is None, word


def _run(eng, sup, text):
    return asyncio.get_event_loop().run_until_complete(
        run_interpret_pipeline(
            InterpretRequest(tokens=[InputToken(type="text", value=text)]), eng, sup))


def _session(lang, equation):
    nb.start_engine(title="side", mode="standard", language=lang,
                    domains=["classical_mechanics"])
    eng = nb.get_engine()
    sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                        domains=["classical_mechanics"], primary_backend="maxima")
    _run(eng, sup, "a をスカラーとする" if lang == "ja" else "Define a as a scalar")
    _run(eng, sup, equation)
    return eng, sup


@pytest.mark.parametrize("lang,text,side", SIDE_SPELLINGS)
def test_the_right_side_of_the_equation_is_operated_on(lang, text, side):
    """画面まで —— `x^2 + a*x = 0` の左辺は `x^2+a*x`、右辺は `0`。

    直す前は `right side` / `RHS` / `left-hand side` のどれも**式全体**
    （`x^2+a*x=0`）か語そのもの（`left-hand`）を返していた。
    """
    eng, sup = _session(lang, "x^2 + a*x = 0")
    try:
        resp = _run(eng, sup, text)
        assert resp.get("status") == "completed", (text, resp.get("error"))
        got = (resp.get("canonical") or "").replace(" ", "")
        want = "x^2+a*x" if side == "left" else "0"
        assert got == want, (text, got, want)
    finally:
        nb.stop_engine()


@pytest.mark.parametrize("lang,text", [
    ("ja", "E1 の左辺を整理して"),
    ("en", "simplify the left-hand side of E1"),
])
def test_a_named_equation_refuses_the_left_side_with_a_reason(lang, text):
    """H-1 の等式では左辺は**名前** —— 黙って右辺を出さず、理由を言う。"""
    eng, sup = _session(lang, "E = a*x + 3")
    try:
        n_before = len(eng.expressions)
        resp = _run(eng, sup, text)
        assert resp.get("status") == "failed", (text, resp.get("status"))
        assert len(eng.expressions) == n_before, (text, "式が作られた")
        err = str(resp.get("error") or "")
        assert "E" in err and ("名前" in err or "name" in err), (text, err)
    finally:
        nb.stop_engine()


@pytest.mark.parametrize("lang,text", [
    ("ja", "E1 の左辺を 0 とする"),
    ("en", "set the left-hand side of E1 to 0"),
])
def test_setting_the_left_side_is_refused_with_a_reason(lang, text):
    """左辺に値を置く操作は無い —— 黙って右辺に置くと違う式が登録される。"""
    eng, sup = _session(lang, "x^2 + a*x = 0")
    try:
        n_before = len(eng.expressions)
        resp = _run(eng, sup, text)
        assert resp.get("status") == "failed", (text, resp.get("status"))
        assert len(eng.expressions) == n_before, (text, "式が作られた")
        assert resp.get("error"), (text, "理由が無い")
    finally:
        nb.stop_engine()
