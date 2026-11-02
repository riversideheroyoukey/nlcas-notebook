r"""CHANGE-VARIABLE-NOOP (2026-09-19) — 置換積分は、積分が無ければ何もできない。

`E1 の y を u に変数変換して` は `z^2+y` を**そのまま**返して `completed` だった
（実測 2026-09-19、日英とも）。命令は `changevar(z^2 + y, (u)-(y), u, y)` で、
**Maxima の `changevar` は積分（やシグマ）の形にしか効かない**——ただの式に当てると
入力をそのまま返すので、研究者には「変換した」と読める成功が返る。

**着手条件どおり、まず CAS を測った**（実測 2026-09-19）:

====================================================  =========================
`changevar(z^2 + y, u-y, u, y)`                       `z^2+y`（**何も起きない**）
`changevar('integrate(y^2,y), u-y, u, y)`             `'integrate(u^2,u)`
`changevar(integrate(y^2,y), u-y, u, y)`              `y^3/3`（先に評価される）
`changevar('integrate(y*%e^(y^2),y,0,1), u-y^2, u, y)`  `('integrate(%e^u,u,0,1))/2`
`subst(u, y, z^2 + y)`                                `z^2+u`
====================================================  =========================

だから直し方は2つ組で決まった:

* **積分は名詞のまま下ろす**（`hold_integrals`、既に在る仕組みの2人目の利用者）。
  下ろしていなかったので `\int y^2 \, dy` は `\int*y^2*dy` のまま CAS へ行き、
  `dy*int*y^2` が**成功として**返っていた（BUILDER-OPERAND-NOT-LOWERED、
  `_build_relabel_index` が 2026-08-15 に直したのと同じ形）。
* **積分が無ければ断る。** `changevar` にできることは何も無く、名前を置き換え
  たいだけなら `substitute` が在る——断り文はその道を名指し、**両方の道を実測した**。
"""
import asyncio

import pytest

import ui.notebook_bridge as nb
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operation_validator import OperationValidator
from ui.interpret_pipeline import (InputToken, InterpretRequest,
                                   run_interpret_pipeline)


def _run(lang, lines, domain="classical_mechanics"):
    async def go():
        nb.start_engine(title="cvt", mode="standard", language=lang, domains=[domain])
        try:
            engine = nb.get_engine()
            sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                                domains=[domain],
                                primary_backend=engine.session.primary_backend)
            out = []
            for line in lines:
                out.append(await run_interpret_pipeline(
                    InterpretRequest(tokens=[InputToken(type="text", value=line)]),
                    engine, sup))
            cmds = [s.rendered_command for s in engine.steps.values()
                    if s.kind == "cas_operation" and s.rendered_command]
            return out[-1], (cmds[-1] if cmds else "")
        finally:
            nb.stop_engine()
    return asyncio.run(go())


def _value(r):
    return "".join(str(r.get("canonical") or "").split())


def _msg(r):
    return str(r.get("error") or r.get("message") or "")


_PLAIN = {"ja": "E1 の y を u に変数変換して",
          "en": "change variable y to u in E1"}
_INTEGRAL = r"\int y^2 \, dy"


class TestAPlainExpressionIsRefusedWithAReason:
    @pytest.mark.parametrize("lang", ["ja", "en"])
    def test_it_no_longer_answers_with_the_input(self, lang):
        r, _ = _run(lang, ["z^2 + y", _PLAIN[lang]])
        assert r.get("status") == "failed", r
        assert _value(r) != "z^2+y", r
        assert "Maxima" not in _msg(r), r

    @pytest.mark.parametrize("lang,fragment", [
        ("ja", "置換"), ("en", "substitut"),
    ])
    def test_the_refusal_names_the_road(self, lang, fragment):
        r, _ = _run(lang, ["z^2 + y", _PLAIN[lang]])
        assert fragment in _msg(r), r
        assert "y" in _msg(r) and "u" in _msg(r), r


class TestAnIntegralActuallyChangesItsVariable:
    r"""`\int y^2 \, dy` は `\int*y^2*dy` のまま CAS へ行き `dy*int*y^2` が
    返っていた（`completed`）。下ろす、しかも**名詞のまま**下ろすのが要る。"""

    @pytest.mark.parametrize("lang", ["ja", "en"])
    def test_the_variable_changes(self, lang):
        r, cmd = _run(lang, [_INTEGRAL, _PLAIN[lang]])
        assert r.get("status") == "completed", r
        assert _value(r) == "'integrate(u^2,u)", (r, cmd)

    def test_the_command_holds_the_integral_as_a_noun(self):
        """名詞にしないと Maxima が先に評価し、`changevar` は必ず空振りする。"""
        _, cmd = _run("ja", [_INTEGRAL, _PLAIN["ja"]])
        assert "'integrate(y^2" in cmd, cmd
        assert "int*y^2" not in cmd, cmd


class TestTheRoadsTheRefusalNamesWork:
    """**案内した回避策も測る**——どちらも日英で通ることを確かめてある。"""

    @pytest.mark.parametrize("lang,lines,want", [
        ("ja", ["z^2 + y", "E1 の y を u に置換して"], "z^2+u"),
        ("en", ["z^2 + y", "substitute u for y in E1"], "z^2+u"),
        ("ja", [_INTEGRAL, "E1 の y を u に変数変換して"], "'integrate(u^2,u)"),
        ("en", [_INTEGRAL, "change variable y to u in E1"], "'integrate(u^2,u)"),
    ])
    def test_the_road(self, lang, lines, want):
        r, _ = _run(lang, lines)
        assert r.get("status") == "completed", r
        assert _value(r) == want, r


class TestTheEnglishFrameWasMissingFromTheGuard:
    r"""**回帰シナリオを組み立てているときに出た ja/en 割れ。**

    2a-sym の枠の表は日本語だけが変数変換を持っており（`ja_change_var_old` /
    `_new`）、英語は**どちらの名前も守っていなかった**。実測 2026-09-19、`u` を
    解いたあと `change variable r to u in E1` は「必要な引数が不足しています」、
    日本語の同義文は通る。**状態のあるシナリオは、単体のプローブが作らない
    組み合わせを作る**——`u` を解いておくのがそれだった。
    """

    _SETUP = {"ja": [r"\int r^2 \, dr", "2 u = 8", "E2 を u について解いて"],
              "en": [r"\int r^2 \, dr", "2 u = 8", "solve E2 for u"]}
    _LINE = {"ja": "E1 の r を u に変数変換して",
             "en": "change variable r to u in E1"}

    @pytest.mark.parametrize("lang", ["ja", "en"])
    def test_a_solved_new_variable_does_not_break_it(self, lang):
        r, _ = _run(lang, self._SETUP[lang] + [self._LINE[lang]])
        assert r.get("status") == "completed", r
        assert _value(r) == "'integrate(u^2,u)", r

    @pytest.mark.parametrize("text", [
        "change variable r to u in E1",
        "change of variable r to u in E1",
        "u-substitution r to u in E1",
    ])
    def test_the_guard_keeps_both_names(self, text):
        import types
        from ui.supervisor_alias_rewrite import rewrite_symbols_to_eids
        engine = types.SimpleNamespace(lhs_alias_map={"u": "E9", "r": "E8"})
        assert rewrite_symbols_to_eids(text, engine) == text, text


class TestWhatIsNotTouched:
    def test_the_range_bearing_form_keeps_its_own_refusal(self):
        r"""範囲つきの定積分は表示専用で、自分の操作
        （`change_variables_integral_form`）を持つ——その断りは別で、変わらない。"""
        r, _ = _run("ja", [r"\int_0^1 y^2 \, dy", _PLAIN["ja"]])
        assert r.get("status") == "failed", r
        assert "表示専用" in _msg(r), r

    def test_an_integral_with_a_nontrivial_relation(self):
        """ヤコビアンは CAS が入れる——`u = y^2` で `\\int y e^{y^2} dy`。"""
        r, _ = _run("ja", [r"\int y \exp(y^2) \, dy",
                           "E1 の y を u に変数変換して"])
        # 関係は `u - y` として組まれる（`new_expr` を書かない綴り）ので、
        # ここで固定するのは「積分の形のまま返る」こと。
        assert r.get("status") == "completed", r
        assert "'integrate" in _value(r), r
