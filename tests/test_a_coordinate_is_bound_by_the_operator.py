r"""COORD-NAME-BOUND (2026-09-17) — 座標名を H-1 で定義すると、ベクトル解析が
まるごと生の CAS エラーになっていた。

``nb_curl_x(Ax,Ay,Az) := diff(Az,y) - diff(Ay,z)`` は座標変数を**関数の中に**
綴っているので、H-1 の P2 が送る ``y : x^3`` がそこに届く。呼び出しは
「``x^3`` で微分せよ」になり Maxima が断り、研究者に返るのは
**生の ``Maxima evaluation error``**（Core Principle #1）。実測 2026-09-17、
日英6通り × ∇の4操作 × ``y = x^3`` / ``x = 2`` のすべてが落ちた。

**研究者は座標名を書いていない。** ``∇×`` が x・y・z を束縛するのは ``∑`` が
添字を、``∫`` が測度を束縛するのと同じで、名前はセッションの座標系から来る。
だからこれは**名前の衝突**であって、演算子が使うべき値ではない——
``block([x,y,z], …)`` はそれを言うための Maxima 自身の機構である。

**「座標名の定義を断る」という読み方は実測で却下した。** シナリオの入力
1994行のうち4行が座標名を定義しており、どちらも普通の書き方である——LP の目的関数
``z = c_1 x + c_2 y`` と ``y = x^3``。

**微分変数には同じことをしてはならない。** ``r = \sqrt{x^2+y^2+z^2}`` のあとの
``diff(f, r)`` は同じように落ちるが、守ると **0** を返す（正解は ``-1/r^2``）。
P2 は代入時に評価するので ``f = 1/r`` は ``1/sqrt(x^2+y^2+z^2)`` として格納され、
記号 ``r`` はもう入っていない。**大きな声のエラーを静かな誤答に替える**ことになる
ので、そちらは翻訳した断りが要る（open_items DEFINED-NAME-AS-VARIABLE）。
"""
import asyncio

import pytest

import ui.notebook_bridge as nb
from notebook.models.symbol_context import SymbolContext
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.maxima_backend_handler import MaximaBackendHandler
from notebook.runtime.maxima_command_builder import MaximaCommandBuilder
from notebook.runtime.operation_validator import OperationValidator
from ui.interpret_pipeline import (InputToken, InterpretRequest,
                                   run_interpret_pipeline)

_DOM = ["electromagnetism"]


def _ask(lang, setup, text):
    async def go():
        nb.start_engine(title="coord-bound", mode="standard", language=lang,
                        domains=_DOM)
        try:
            engine = nb.get_engine()
            sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                                domains=_DOM,
                                primary_backend=engine.session.primary_backend)
            for line in setup:
                await run_interpret_pipeline(
                    InterpretRequest(tokens=[InputToken(type="text", value=line)]),
                    engine, sup)
            return await run_interpret_pipeline(
                InterpretRequest(tokens=[InputToken(type="text", value=text)]),
                engine, sup)
        finally:
            nb.stop_engine()
    return asyncio.run(go())


_DECL = {
    "ja": ["A をベクトル場とする", "f をスカラー場とする"],
    "en": ["define A as a vector field", "define f as a scalar field"],
}

#: 座標名を束縛する書き方。どちらも実在する（シナリオ 1994 行中の 4 行）。
_BINDINGS = [[], ["y = x^3"], ["x = 2"]]

_SPELLINGS = {
    "ja": [r"\nabla \times \vec{A}", r"\nabla \cdot \vec{A}", r"\nabla f",
           r"\nabla^2 f", r"\vec{A} の回転を求めて", "A の発散を求めて"],
    "en": [r"\nabla \times \vec{A}", r"\nabla \cdot \vec{A}", r"\nabla f",
           r"\nabla^2 f", r"the curl of \vec{A}", "the divergence of A"],
}


class TestTheCoordinateIsBoundByTheOperator:
    """束縛が在っても、∇ の答えは束縛が無いときと**同じ**でなければならない。"""

    @pytest.mark.parametrize("lang", ["ja", "en"])
    @pytest.mark.parametrize("idx", range(6))
    def test_the_answer_does_not_depend_on_a_coordinate_name_binding(self, lang, idx):
        text = _SPELLINGS[lang][idx]
        answers = []
        for binding in _BINDINGS:
            r = _ask(lang, _DECL[lang] + binding, text)
            assert r.get("status") == "completed", (binding, text, r)
            # 生の CAS 文言が研究者に届かないこと（これが起票の症状）
            assert "Maxima" not in str(r.get("error") or ""), (binding, text, r)
            answers.append((r.get("canonical") or "").replace(" ", ""))
        assert len(set(answers)) == 1, (text, answers)

    @pytest.mark.parametrize("lang", ["ja", "en"])
    def test_a_concrete_field_lands_on_the_same_value(self, lang):
        """noun 形だけでなく**値**でも同じこと（A = (0,0,x y) の回転は [x,-y,0]）。"""
        setup = _DECL[lang] + ["A_x = 0", "A_y = 0", "A_z = x*y"]
        for binding in _BINDINGS:
            r = _ask(lang, setup + binding, r"\nabla \times \vec{A}")
            assert r.get("status") == "completed", (binding, r)
            assert (r.get("canonical") or "").replace(" ", "") == "[x,-y,0]", (binding, r)


class TestTheDefinitionOfACoordinateNameIsStillWritable:
    """「座標名の定義を断る」を選ばなかったことを固定する。

    LP の目的関数 ``z = c_1 x + c_2 y`` は実在の綴りで（`scenario_lp_vertex_optimization`）、
    断ればそれが書けなくなる——「まず書けること、断るのは使用地点」。
    """

    @pytest.mark.parametrize("lang", ["ja", "en"])
    @pytest.mark.parametrize("line", ["y = x^3", "z = c_1 x + c_2 y", "x = 2"])
    def test_it_registers(self, lang, line):
        r = _ask(lang, _DECL[lang], line)
        assert r.get("status") in ("registered", "completed"), (line, r)


class _RecordingAdapter:
    def __init__(self):
        self.commands = []

    def execute(self, command):
        self.commands.append(command)
        return None


class _Session:
    def __init__(self, coord_system, custom_vars=None, custom_lame=None):
        self.coordinate_system = coord_system
        self.custom_coord_vars = custom_vars
        self.custom_coord_lame = custom_lame


def _handler_for(session):
    h = MaximaBackendHandler(MaximaCommandBuilder())
    h.set_symbol_context(SymbolContext())
    h.init_coordinate_system(_RecordingAdapter(), session)
    return h


class TestTheShadowedNamesAreTheInstalledOnes:
    """守る名前は**表から引く**のではなく、設置した ``nb_*`` が束縛している名前。

    カスタム座標系の変数は研究者が名づけたもので、`reserved_coord_vars` は
    それを知らずデカルトへ落ちる。表から引いていたら、そのセッションだけ
    守り損ねる（しかも静かに）。
    """

    @pytest.mark.parametrize("coord_system,want", [
        ("cartesian",   ("x", "y", "z")),
        ("spherical",   ("r", "theta", "phi")),
        ("cylindrical", ("r", "phi", "z")),
    ])
    def test_a_named_system_uses_its_own_variables(self, coord_system, want):
        assert tuple(_handler_for(_Session(coord_system))._nb_coord_vars) == want

    def test_a_custom_system_uses_the_researchers_names(self):
        h = _handler_for(_Session("custom", ("xi", "eta", "zeta"),
                                  ("1", "1", "1")))
        assert tuple(h._nb_coord_vars) == ("xi", "eta", "zeta")

    def test_an_incomplete_custom_system_falls_back_with_the_fallbacks_variables(self):
        """ラメ係数が欠けていればデカルトへ落ちる——守る名前もデカルトのもの。"""
        h = _handler_for(_Session("custom", ("xi", "eta", "zeta"), None))
        assert tuple(h._nb_coord_vars) == ("x", "y", "z")

    def test_switching_the_system_mid_session_moves_the_shadow(self):
        h = _handler_for(_Session("cartesian"))
        h.build_command("define_coordinate_system",
                        {"vars": "u,v,w", "lame": "1,1,1"}, {})
        assert tuple(h._nb_coord_vars) == ("u", "v", "w")


class TestTheCommandSaysWhatItBinds:
    @pytest.mark.parametrize("coord_system,prefix", [
        ("cartesian", "block([x, y, z], "),
        ("spherical", "block([r, theta, phi], "),
    ])
    def test_the_vector_operators_shadow_their_coordinates(self, coord_system, prefix):
        h = _handler_for(_Session(coord_system))
        build = h.build_command("divergence",
                                {"expr": "A", "coordinate_system": coord_system}, {})
        assert build.ok, build.error
        assert build.command.startswith(prefix), build.command

    @pytest.mark.parametrize("operation", ["divergence", "curl", "gradient",
                                           "scalar_laplacian"])
    def test_all_four_are_shadowed(self, operation):
        h = _handler_for(_Session("cartesian"))
        build = h.build_command(operation, {"expr": "A"}, {})
        assert build.ok, build.error
        assert build.command.startswith("block([x, y, z], "), build.command

    def test_it_is_unconditional(self):
        """束縛が在るかを訊かない。

        訊こうとすると帳簿側の述語（`defining_expr_id`）を使うことになるが、
        それは「CAS が値を持っている」と同義ではない——解の結果は id を持つのに
        束縛しない（実測、`shadow_own_definition` にも記録がある）。束縛が無い
        名前を守るのは no-op なので、判別子は要らない（`_build_solve_system`
        が同じ理由で同じことをしている）。
        """
        h = _handler_for(_Session("cartesian"))          # 何も束縛していない
        build = h.build_command("curl", {"expr": "A"}, {})
        assert build.command.startswith("block([x, y, z], "), build.command


class TestADifferentiationVariableIsNotShadowed:
    """境界を固定する——ここを広げると、大きな声のエラーが静かな誤答になる。

    ``r = \\sqrt{x^2+y^2+z^2}`` のあとの ``diff(f, r)`` を守ると **0** が返る
    （正解 ``-1/r^2``）。P2 は代入時に評価するので ``f`` にはもう ``r`` が無い。
    """

    @pytest.mark.parametrize("operation,args", [
        ("differentiate",       {"expr": "x^2", "var": "y", "order": 1}),
        ("integrate_indefinite", {"expr": "x^2", "var": "y"}),
        ("integrate_definite",  {"expr": "x^2", "var": "y",
                                 "lower": "0", "upper": "1"}),
    ])
    def test_the_variable_the_researcher_wrote_keeps_its_meaning(self, operation, args):
        h = _handler_for(_Session("cartesian"))
        build = h.build_command(operation, args, {})
        assert build.ok, build.error
        assert "block([" not in build.command, build.command


class TestADeclarationBindsItsVariablesToo:
    r"""3つ目の出口——**宣言の依存変数**（`depends([B_x,…],[x,y,z])`）。

    回帰シナリオの replay が見つけた（起票ではない）。COORD-NAME-BOUND を直したときは
    「先に宣言、後で束縛」の順しか測っておらず、**束縛してから宣言**すると
    `B をベクトル場とする` が日英とも生の `Maxima evaluation error` だった。時間依存は
    もっと悪く、`t = 2` のあとの `q を時間依存とする` は CAS 側が失敗しているのに
    「✓ 宣言しました」を `confirmed` で返していた——以後 `q` の時間微分は依存を見ない。
    """

    _DECL_A = {"ja": "A をベクトル場とする", "en": "define A as a vector field"}
    _DECL_G = {"ja": "g をスカラー場とする", "en": "define g as a scalar field"}

    @pytest.mark.parametrize("lang", ["ja", "en"])
    def test_a_vector_field_declared_after_the_binding_computes(self, lang):
        r = _ask(lang, ["y = x^3", self._DECL_A[lang]], r"\nabla \times \vec{A}")
        assert r.get("status") == "completed", r
        assert "'diff(A_z,y,1)" in (r.get("canonical") or ""), r

    @pytest.mark.parametrize("lang", ["ja", "en"])
    def test_a_scalar_field_declared_after_the_binding_computes(self, lang):
        r = _ask(lang, ["y = x^3", self._DECL_G[lang]], r"\nabla g")
        assert r.get("status") == "completed", r
        assert "'diff(g,y,1)" in (r.get("canonical") or ""), r

    def test_a_time_dependence_declared_after_the_binding_reaches_the_cas(self):
        """「宣言しました」と CAS の `dependencies` が食い違わないこと。"""
        async def go():
            nb.start_engine(title="coord-bound-t", mode="standard", language="ja",
                            domains=["classical_mechanics"])
            try:
                engine = nb.get_engine()
                sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                                    domains=["classical_mechanics"],
                                    primary_backend=engine.session.primary_backend)
                for line in ["t = 2", "q を時間依存とする"]:
                    await run_interpret_pipeline(
                        InterpretRequest(tokens=[InputToken(type="text", value=line)]),
                        engine, sup)
                return engine._adapter.execute("dependencies").value
            finally:
                nb.stop_engine()
        assert "q(t)" in asyncio.run(go())
