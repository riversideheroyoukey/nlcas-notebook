"""操作が答えたら、そのカードにはラベルがある —— 8箇所で別々に答えていた問い。

タイムラインは ``registered_expressions`` の各項をカードにし、そのどれも
``expr_id`` を出していなければ ``data`` のカードを描く（``notebook_main.js``）。
つまり**ラベルは、描かれたほうのオブジェクトに載っていなければ画面に出ない**。
インターセプタが自分の結果を append しておきながらその項にラベルを載せなければ、
応答をどれだけ丁寧に埋めても**カードは無ラベル**になる。

実測 2026-08-27: 操作に届いたコーパス189行のうち**28行**がどのカードにも
ラベルを持たず、しかも**8つの別々の route**にまたがっていた——`simplify` は
`evaluate_rhs` と `both_sides` の両方、ほかに `build_inverse_transform`・
`taylor_expand`・`expand_neighbor_sum`・`fork_scalar_handoff`・
`solve_ode_general`・`pair_bracket` 2種。それぞれが「操作は何を返すか」を
自分で答えており、8つが不完全だった。

**答えは `run_interceptors` に1つ置いた**——全インターセプタの応答が通る場所で、
route の帰属を既に付けているのと同じ形である。ステップは**応答からではなく
インターセプタが実際に作ったもの**から取る: `resp["step_id"]` を読む最初の実装は
28件中3件しか直せなかった（これらのハンドラの多くは `step_id` を返さない）。
"""
import asyncio

import pytest

import ui.notebook_bridge as nb
from notebook.engine_history_mixin import (is_internal_expr_id,
                                           label_names_internal_scaffold)
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operation_validator import OperationValidator
from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline


def _run(eng, sup, text):
    return asyncio.get_event_loop().run_until_complete(
        run_interpret_pipeline(
            InterpretRequest(tokens=[InputToken(type="text", value=text)]), eng, sup))


def _cards(data):
    """``notebook_main.js`` 230-247 の分岐そのまま。"""
    out = [("registered", e.get("expr_id"), e.get("operation_label"))
           for e in (data.get("registered_expressions") or [])]
    if not any(e.get("expr_id") == data.get("expr_id")
               for e in (data.get("registered_expressions") or [])):
        out.append(("data", data.get("expr_id"), data.get("operation_label")))
    return out


#: 28件が出ていた route を代表する綴り。うち `both_sides` は SCAFFOLD-LABEL の
#: ため意図的にラベル無しのままで、下の別クラスがその**理由**を押さえる。
LABELLED = [
    ("E1 の右辺を整理して",            "evaluate_rhs"),
    ("E1 を x について解いて",         "solve"),
    ("E1 の x を 2 に置換して",        "substitute"),
    ("E1 を x で偏微分して",           "differentiate"),
    ("E1 を展開して",                 "expand"),
]


@pytest.fixture
def session():
    nb.start_engine(title="card_label", mode="standard", language="ja",
                    domains=["classical_mechanics"])
    eng = nb.get_engine()
    sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                        domains=["classical_mechanics"], primary_backend="maxima")
    _run(eng, sup, "a, b, A をスカラーとする")
    _run(eng, sup, "f = a*(x+1) + b*(x+1)")
    yield eng, sup
    nb.stop_engine()


class TestACompletedOperationDrawsALabelledCard:
    @pytest.mark.parametrize("text,route", LABELLED)
    def test_the_rendered_card_carries_a_label(self, session, text, route):
        eng, sup = session
        r = _run(eng, sup, text)
        assert r["status"] == "completed", r
        cards = _cards(r)
        assert cards, r
        assert any(lbl for _s, _e, lbl in cards), (
            f"{route}: every card the researcher sees is unlabelled — the "
            f"response was {r.get('operation_label')!r}")

    def test_a_registration_is_not_given_an_operation_label(self, session):
        """The fill must not invent one: a registration draws the expression
        itself, and "register" is not an operation."""
        eng, sup = session
        r = _run(eng, sup, "g = a + b")
        assert not r.get("operation_label"), r


class TestAScaffoldNameIsNeverShown:
    """`_T1` is a key ``_alloc_internal_expr_id`` hands out for scaffolding the
    researcher never sees.  A few interceptors really do run their operation on
    one — H-8's both-sides path and H-5's RHS evaluation rewrite the equation,
    register the result as scaffolding and operate on THAT — so a label built
    from the operation's own args named a key nobody has seen.  Measured
    2026-08-27: 40 published labels read `_T1 を登録する` / `simplify [_T1]`.

    Closed 2026-09-05 by letting the handler say what the scaffold STANDS FOR
    (`label_rhs_of` / `label_expr_ref` / `label_operation`) while the wording
    stays in ``_visible_action``.  The predicate below is kept as the safety
    net: a future scaffold operation with no hint leaves the card blank rather
    than naming the key, because a label naming an unseen key is worse than
    none — it asks the researcher what `_T1` is.
    """

    def test_the_predicate(self):
        assert is_internal_expr_id("_T1") and is_internal_expr_id("_T12")
        assert not is_internal_expr_id("E1") and not is_internal_expr_id("_Tx")
        assert label_names_internal_scaffold("_T1 を簡約する")
        assert label_names_internal_scaffold("simplify [_T1]")
        assert not label_names_internal_scaffold("E1 を簡約する")

    @pytest.mark.parametrize("text,expected", [
        ("E1 の両辺に A をかける", "E1 の両辺に乗じる"),
        ("E1 の両辺を 2 で割る",   "E1 の両辺を割る"),
        ("E1 の右辺を整理して",     "E1 の右辺を簡約する"),
    ])
    def test_the_card_names_the_researchers_expression(self, session, text, expected):
        eng, sup = session
        r = _run(eng, sup, text)
        assert r["status"] == "completed", r
        labels = [lbl for _s, _e, lbl in _cards(r)]
        assert expected in labels, labels
        for lbl in labels:
            assert not lbl or not label_names_internal_scaffold(lbl), lbl
