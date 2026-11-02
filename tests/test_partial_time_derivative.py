r"""∂/∂t は座標を止める —— 偏微分を頼んで全微分が返っていた。

実測 2026-09-06。**読みの層は既に揃っていた**（21綴り 21/21 が
`_is_partial_derivative` を立てる）。揃っていなかったのは計算の層で、旗は
**表示ラベルにしか届いていなかった**——`_preprocess_args` が `_is_*` を剥がすのは
検証器に見せないためなのに、ビルダにも見えなくなっていた。

| 打った文 | 直す前 | 正しい |
|---|---|---|
| `f = q^2 + t` の ∂f/∂t | `2 q q̇ + 1` | **`1`** |
| 駆動振動子 `L = q̇²/2 − q²/2 + F q cos(ωt)` の ∂L/∂t | `−F q ω sin(ωt) + F q̇ cos(ωt) + q̇q̈ − q q̇` | **`−F q ω sin(ωt)`** |
| `2階偏微分` | 全微分の**1階**（ラベルは `∂²f/∂t²`） | 2階 |

`dH/dt = −∂L/∂t`（エネルギー定理）は**まさにこの量についての主張**なので、
解析力学で ∂L/∂t が要る場面の中心がこれである。

**なぜ素の `diff(f,t)` では駄目か。** `declare_time_dependent` は `depends(q,t)` を
送るので、Maxima は `diff(f,t)` で連鎖律を適用する——返るのは d/dt である。
座標を一時的に t 非依存の記号へ置換してから微分し、戻す。これは**全微分の最後の項
と同じ式**なので、1箇所（`_hold_atoms_and_diff`）に置いてある。

**解析力学の外は直さない——直すところが無い。** 一度「`depends` はどのドメインでも
送られるのだから同じ欠陥だ」と広げて、コーパスの `scenario_coulomb_gauge` に
捕まった（2026-09-06）。そこは `A_x` を時間依存関数として宣言して `∂A_x/∂t` を
計算する場面で、広げた実装は `A_x` 自身を凍らせて **0** を返した。読み違えていたのは
`depends(x,t)` の意味である——それは「x は t の関数である」という宣言なので、そう
宣言された式は t だけの関数であり、**∂ と d は本当に同じ**。CAS は正しかった。
解析力学だけが違うのは、そこでは `q` と `q̇` が**独立な原子**だと宣言されており、
`L(q, q̇, t)` に独立な枠が3つあるからである。
"""
import asyncio

import pytest

import ui.notebook_bridge as nb
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.maxima_backend_handler import _derivative_kind
from notebook.runtime.operation_validator import OperationValidator
from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline


def _run(eng, sup, text):
    return asyncio.get_event_loop().run_until_complete(
        run_interpret_pipeline(
            InterpretRequest(tokens=[InputToken(type="text", value=text)]), eng, sup))


@pytest.fixture(params=["ja", "en"])
def lang(request):
    return request.param


def _session(lang, domain):
    nb.start_engine(title="pdt", mode="standard", language=lang, domains=[domain])
    eng = nb.get_engine()
    sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                        domains=[domain], primary_backend="maxima")
    return eng, sup


def _declare(eng, sup, lang, name):
    _run(eng, sup, f"{name} を時間依存関数として定義する" if lang == "ja"
         else f"Define {name} as a time-dependent function")


#: 偏微分を求める綴り。**どれも同じ2つ（式・変数）と同じ種別を言う。**
PARTIAL_SPELLINGS = [
    ("ja", "E1 を t で偏微分して"),
    ("ja", "E1 を t で偏微分する"),
    ("ja", "E1 を t について偏微分して"),
    ("ja", "E1 の t による偏微分"),
    ("ja", "E1 の t に関する偏導関数"),
    ("ja", "E1 を t で偏微分せよ"),
    ("ja", "E1 を t で偏微分してください"),
    ("en", "Take the partial derivative of E1 with respect to t"),
    ("en", "partial derivative of E1 with respect to t"),
    ("en", "differentiate E1 partially with respect to t"),
    ("en", "partially differentiate E1 with respect to t"),
    ("en", "the partial derivative of E1 in t"),
    ("en", "compute the partial derivative of E1 with respect to t"),
]
TOTAL_SPELLINGS = [
    ("ja", "E1 を t で全微分して"),
    ("ja", "E1 の t による全微分"),
    ("en", "take the total derivative of E1 with respect to t"),
    ("en", "differentiate E1 totally with respect to t"),
]


class TestTheKindReachesTheCommand:
    r"""`f = q^2 + t` で ∂f/∂t = 1、df/dt = 2 q q̇ + 1 —— **別の式**である。"""

    @pytest.mark.parametrize("lg,text", PARTIAL_SPELLINGS)
    def test_every_partial_spelling_holds_the_coordinates(self, lg, text):
        eng, sup = _session(lg, "analytical_mechanics")
        try:
            _declare(eng, sup, lg, "q")
            _run(eng, sup, "f = q^2 + t")
            r = _run(eng, sup, text)
            assert r.get("status") == "completed", (text, r.get("error"))
            assert eng._adapter.execute(
                f"ratsimp(({r['latex'].split('=')[-1]}) - 1)").value.strip() == "0", r
        finally:
            nb.stop_engine()

    @pytest.mark.parametrize("lg,text", TOTAL_SPELLINGS)
    def test_the_total_spellings_still_carry_the_chain(self, lg, text):
        eng, sup = _session(lg, "analytical_mechanics")
        try:
            _declare(eng, sup, lg, "q")
            _run(eng, sup, "f = q^2 + t")
            r = _run(eng, sup, text)
            assert r.get("status") == "completed", (text, r.get("error"))
            _, val = eng.run_operation("differentiate",
                                       {"expr": "E1", "var": "t",
                                        "_is_total_derivative": True})
            assert eng._adapter.execute(
                f"ratsimp(({val}) - (2*q*q_dot + 1))").value.strip() == "0", val
        finally:
            nb.stop_engine()

    def test_the_builder_reads_the_kind_the_engine_passes(self):
        """旗はビルダまで来る——ここが切れていたのが原因である。"""
        assert _derivative_kind({"_derivative_kind": "partial"}) == "partial"
        assert _derivative_kind({"_derivative_kind": "total"}) == "total"
        assert _derivative_kind({}) == ""


class TestTheEnergyTheoremCase:
    r"""`dH/dt = −∂L/∂t` —— 解析力学で ∂L/∂t が要る場面の中心。"""

    def test_a_driven_lagrangian_keeps_only_the_explicit_term(self, lang):
        eng, sup = _session(lang, "analytical_mechanics")
        try:
            _declare(eng, sup, lang, "q")
            _run(eng, sup, "F, w をスカラーとする" if lang == "ja"
                 else "Define F, w as scalars")
            _run(eng, sup, r"L = \dot{q}^2/2 - q^2/2 + F q \cos(w t)")
            _, val = eng.run_operation("differentiate",
                                       {"expr": "E1", "var": "t",
                                        "_is_partial_derivative": True})
            assert eng._adapter.execute(
                f"ratsimp(({val}) - (-F*q*w*sin(w*t)))").value.strip() == "0", val
        finally:
            nb.stop_engine()

    def test_partial_and_total_differ_by_exactly_the_chain_terms(self, lang):
        """∂ と d の差が連鎖項そのものであること——定義の差分検証。"""
        eng, sup = _session(lang, "analytical_mechanics")
        try:
            _declare(eng, sup, lang, "q")
            _run(eng, sup, "F, w をスカラーとする" if lang == "ja"
                 else "Define F, w as scalars")
            _run(eng, sup, r"L = \dot{q}^2/2 - q^2/2 + F q \cos(w t)")
            _, part = eng.run_operation("differentiate",
                                        {"expr": "E1", "var": "t",
                                         "_is_partial_derivative": True})
            _, tot = eng.run_operation("differentiate",
                                       {"expr": "E1", "var": "t",
                                        "_is_total_derivative": True})
            chain = ("diff(q_dot^2/2 - q^2/2 + F*q*cos(w*t),q)*q_dot"
                     "+diff(q_dot^2/2 - q^2/2 + F*q*cos(w*t),q_dot)*q_ddot")
            assert eng._adapter.execute(
                f"ratsimp(({tot}) - ({part}) - ({chain}))").value.strip() == "0", (tot, part)
        finally:
            nb.stop_engine()


class TestTheOrderIsHonoured:
    @pytest.mark.parametrize("lg,text,want", [
        ("ja", "E1 を t で2階偏微分して", "0"),
        ("en", "take the 2nd order partial derivative of E1 with respect to t", "0"),
    ])
    def test_second_order(self, lg, text, want):
        """直す前は**全微分の1階**を返し、ラベルだけ `∂²f/∂t²` だった。"""
        eng, sup = _session(lg, "analytical_mechanics")
        try:
            _declare(eng, sup, lg, "q")
            _run(eng, sup, "f = q^2 + t")
            r = _run(eng, sup, text)
            assert r.get("status") == "completed", (text, r.get("error"))
            assert eng._adapter.execute(
                f"ratsimp(({r['latex'].split('=')[-1]}) - ({want}))").value.strip() == "0", r
        finally:
            nb.stop_engine()


class TestWhatMustNotChange:
    def test_a_coordinate_variable_is_a_plain_diff(self, lang):
        """∂/∂q は座標そのものについての微分——止めるものは無い。"""
        eng, sup = _session(lang, "analytical_mechanics")
        try:
            _declare(eng, sup, lang, "q")
            _run(eng, sup, "f = q^2 + t")
            _, val = eng.run_operation("differentiate",
                                       {"expr": "E1", "var": "q",
                                        "_is_partial_derivative": True})
            assert eng._adapter.execute(
                f"ratsimp(({val}) - 2*q)").value.strip() == "0", val
        finally:
            nb.stop_engine()

    def test_no_time_dependent_quantity_means_a_plain_diff(self, lang):
        """時間依存の量が1つも無ければ、∂/∂t は素の `diff` のままでよい。"""
        eng, sup = _session(lang, "classical_mechanics")
        try:
            _run(eng, sup, "u = a t^2")
            _, val = eng.run_operation("differentiate",
                                       {"expr": "E1", "var": "t",
                                        "_is_partial_derivative": True})
            assert eng._adapter.execute(
                f"ratsimp(({val}) - 2*a*t)").value.strip() == "0", val
        finally:
            nb.stop_engine()
