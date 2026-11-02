r"""総時間微分は ∂f/∂t を含む —— 連鎖律だけでは明示的な時間依存が落ちる。

実測 2026-09-06（`analytical_mechanics` セッション）。`d/dt` は連鎖律だけを
組んでいて、**式にもコード内のコメントにも `∂f/∂t` が無かった**:

    ratsimp(Σ_k diff(f, q^(k)) * q^(k+1))          ← これだけ

正しくは `df/dt = Σ_i Σ_k ∂f/∂q_i^(k)·q_i^(k+1) **+ ∂f/∂t**`。

| 入力 | 直す前 | 正しい |
|---|---|---|
| `f = q^2 + t` | `2 q q̇` | `2 q q̇ + 1` |
| `L = q̇²/2 − q²/2 + F q cos(ωt)` | `F q̇ cos(ωt) + (q̈ − q) q̇` | **`− F q ω sin(ωt)`** が加わる |

**なぜ効くか。** 解析力学で `dL/dt` を使う中心的な場面はエネルギー定理
`dH/dt = −∂L/∂t` で、それは明示的時間依存**そのもの**についての主張である。
駆動系・時間に依る拘束はどれもこの項を持つ。

**素の `diff(f,t)` では二重計上になる。** `declare_time_dependent` は
`depends(q,t)` を送っており（実測: `dependencies` が `[q(t)]`）、Maxima は
そこで連鎖項をもう一度出す。座標とドット鎖を一時的に t 非依存の記号へ置換して
から微分し、戻す。

**公開実例には届いていなかった** —— 総時間微分のコマンド8件を走査すると、
被微分式に明示的な `t` を含むものは0件。だから実例の再生成でも 7000 件の
テストでも出ず、**測って初めて出た**。
"""
import asyncio

import pytest

import ui.notebook_bridge as nb
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operation_validator import OperationValidator
from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline


def _run(eng, sup, text):
    return asyncio.get_event_loop().run_until_complete(
        run_interpret_pipeline(
            InterpretRequest(tokens=[InputToken(type="text", value=text)]), eng, sup))


@pytest.fixture(params=["ja", "en"])
def lagrangian(request):
    lang = request.param
    nb.start_engine(title="dtdt", mode="standard", language=lang,
                    domains=["analytical_mechanics"])
    eng = nb.get_engine()
    sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                        domains=["analytical_mechanics"], primary_backend="maxima")
    try:
        yield eng, sup, lang
    finally:
        nb.stop_engine()


def _declare_q(eng, sup, lang):
    _run(eng, sup, "q を時間依存関数として定義する" if lang == "ja"
         else "Define q as a time-dependent function")


class TestTheExplicitTimeTermIsIncluded:
    def test_a_bare_t_contributes_one(self, lagrangian):
        eng, sup, lang = lagrangian
        _declare_q(eng, sup, lang)
        _run(eng, sup, "f = q^2 + t")
        _, val = eng.run_operation("differentiate",
                                   {"expr": "E1", "var": "t",
                                    "_is_total_derivative": True})
        assert eng._adapter.execute(
            f"ratsimp(({val}) - (2*q*q_dot + 1))").value.strip() == "0", val

    def test_a_driven_lagrangian_keeps_the_forcing_term(self, lagrangian):
        """`− F q ω sin(ωt)` —— 直す前は丸ごと落ちていた項。"""
        eng, sup, lang = lagrangian
        _declare_q(eng, sup, lang)
        _run(eng, sup, "F, w をスカラーとする" if lang == "ja"
             else "Define F, w as scalars")
        _run(eng, sup, r"L = \dot{q}^2/2 - q^2/2 + F q \cos(w t)")
        _, val = eng.run_operation("differentiate",
                                   {"expr": "E1", "var": "t",
                                    "_is_total_derivative": True})
        want = "-F*q*w*sin(w*t) + F*q_dot*cos(w*t) + q_dot*q_ddot - q*q_dot"
        assert eng._adapter.execute(
            f"ratsimp(({val}) - ({want}))").value.strip() == "0", val

    def test_an_expression_without_explicit_t_is_unchanged(self, lagrangian):
        """明示的な `t` を含まない式は**一文字も変わらない**（既存の実例がこれ）。"""
        eng, sup, lang = lagrangian
        _declare_q(eng, sup, lang)
        _run(eng, sup, "f = q^2")
        _, val = eng.run_operation("differentiate",
                                   {"expr": "E1", "var": "t",
                                    "_is_total_derivative": True})
        assert eng._adapter.execute(
            f"ratsimp(({val}) - (2*q*q_dot))").value.strip() == "0", val


class TestTheAnswerAgreesWithAnIndependentRoute:
    """**判定は実 CAS の差分**で書く（文字列比較では見えない）。

    独立経路: `q` を最初から `Q(t)` と書いて `d/dt` し、あとで原子表記へ戻す。
    系の連鎖律実装とは別経路なので、一致すれば差分検証になる。
    """

    ATOMS = ["q", "q_dot", "q_ddot", "q_dddot", "q_ddddot"]

    @pytest.mark.parametrize("body", [
        "q_dot^2/2 - q^2/2 + F*q*cos(w*t)",
        "q^2 + t",
        "q*t^2 + q_dot*t",
        "q^2",                      # 明示 t 無し（対照）
    ])
    def test_matches(self, lagrangian, body):
        eng, sup, lang = lagrangian
        _declare_q(eng, sup, lang)
        ad = eng._adapter
        indep = (
            "ratsimp(subst([_d3=q_dddot,_d2=q_ddot,_d1=q_dot,_d0=q], "
            "subst(['diff(Q(t),t,3)=_d3,'diff(Q(t),t,2)=_d2,"
            "'diff(Q(t),t,1)=_d1,Q(t)=_d0], "
            "diff(subst([q_ddot='diff(Q(t),t,2),q_dot='diff(Q(t),t,1),q=Q(t)], "
            f"{body}), t))))")
        want = ad.execute(indep)
        assert want.ok, indep
        _run(eng, sup, f"g = {body}")
        _, val = eng.run_operation("differentiate",
                                   {"expr": "E1", "var": "t",
                                    "_is_total_derivative": True})
        diff = ad.execute(f"ratsimp(({val}) - ({want.value}))")
        assert diff.value.strip() == "0", (body, val, want.value)


def test_a_session_without_the_domain_is_untouched():
    """ドメインなしでは Maxima 自身の `diff(f,t,1)` を送る。この枝は通らない。"""
    nb.start_engine(title="plain", mode="standard", language="ja", domains=[])
    try:
        eng = nb.get_engine()
        sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                            domains=[], primary_backend="maxima")
        _run(eng, sup, "q を時間依存関数として定義する")
        _run(eng, sup, "f = q^2 + t")
        step, val = eng.run_operation("differentiate",
                                      {"expr": "E1", "var": "t",
                                       "_is_total_derivative": True})
        assert step.rendered_command == "diff(q^2 + t,t,1)"
        assert "_nb_free_" not in step.rendered_command
    finally:
        nb.stop_engine()


class TestEveryGeneralisedCoordinateIsHeldFixed:
    r"""明示項は **q, q̇, q̈ … を全部止めて**取る。座標だけでは足りない。

    変異試験で「ドット鎖を置換対象から外す」が**1件も落とさなかった**ので、
    理由を測った——現状 `depends` を持つのは `q` だけ（実測
    `dependencies` = `[q(t)]`）なので、鎖を止めても止めなくても同じ答えになる。
    つまり**当て方の誤りではなく、守りが現状では無効**だった。

    それでも鎖を残すのは、無効だからではなく**不変条件がそれだから**である:
    「∂f/∂t は一般化座標とその微分を全部固定して取る」。実測すると、
    `depends(q_dot,t)` が宣言された途端に答えが割れる——

        鎖あり: -F q w sin(wt)
        鎖なし: q̇·diff(q̇,t) - F q w sin(wt)      ← 連鎖項の二重計上

    ので、これは冗長ではなく将来の経路に対する守りである。その事実を測定として
    ここに固定する（2a-sym の lookbehind と同じ判断）。
    """

    def test_the_explicit_term_survives_a_dependency_on_qdot(self, lagrangian):
        eng, sup, lang = lagrangian
        _declare_q(eng, sup, lang)
        _run(eng, sup, "F, w をスカラーとする" if lang == "ja"
             else "Define F, w as scalars")
        _run(eng, sup, r"L = \dot{q}^2/2 - q^2/2 + F q \cos(w t)")
        step, _ = eng.run_operation("differentiate",
                                    {"expr": "E1", "var": "t",
                                     "_is_total_derivative": True})
        ad = eng._adapter
        # 現状どの経路も宣言しないが、宣言された場合を測る。
        assert ad.execute("depends(q_dot,t)").ok
        assert ad.execute("dependencies").value.replace(" ", "") == "[q(t),q_dot(t)]"
        val = ad.execute(step.rendered_command)
        assert val.ok, step.rendered_command
        want = "-F*q*w*sin(w*t) + F*q_dot*cos(w*t) + q_dot*q_ddot - q*q_dot"
        assert ad.execute(f"ratsimp(({val.value}) - ({want}))").value.strip() == "0", (
            "q_dot の t 依存が宣言されると連鎖項が二重計上される: " + val.value)
