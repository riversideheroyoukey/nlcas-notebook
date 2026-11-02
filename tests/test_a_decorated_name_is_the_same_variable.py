r"""DECORATED-NAME-AS-DIFF-VAR (2026-09-19) — 微分の変数は研究者が書いた名前である。

式は `latex_to_canonical` を通り、**変数は通らなかった**。だから同じ名前が2つの
CAS 対象になり、微分は 0 を返した——`completed` で。実測 2026-09-19、
`z^2 + \bar{y}` を登録して:

    diff(z^2 + y_bar, bar(y), 1)  →  0

式の側は飾りのある名前 `y_bar`、変数の側は**関数呼び出し** `bar(y)`。
`\hat{H}` / `\tilde{x}` も同じで、**日英ともこうなる**。

**`solve` は 2026-08-21 に同じ扉を通すよう直されている**（`solve(z^2 + y_bar,
y_bar)` は正しい）。そして `math_operand` の docstring は「研究者は notebook が
**表示した**ものを書き戻す。だから捕まえた被演算子——被積分関数、**変数**、変換
規則——は渡す前に変換する。各所で言うから、当たる所と当たらない所ができる」と
自分で書いている。**同じ問いの4件目**で、微分だけが呼んでいなかった。

**プライムは同じ欄のもう1つの原因だった。** `symbol` の文字クラスに `'` が無く、
英語では `y'` が **`y` に切り詰められて** `diff(z^2 + y', y, 1)` → `0` を
`completed` で返し（日本語は綴りが一致せず大きな声で落ちた）。プライムは
「別の変数の印」だと変換器自身が書いており、`t'` / `x'` / `\sigma'` は
グリーン関数の普通の綴りである。
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
        nb.start_engine(title="dec", mode="standard", language=lang, domains=[domain])
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


#: `d(z^2 + X)/dX = 1` —— 式と変数が**同じ名前**を指していれば必ず 1 になる。
_DECORATED = [r"\bar{y}", r"\hat{H}", r"\tilde{x}", r"\vec{v}"]


class TestADecoratedVariableIsTheNameTheExpressionUses:
    @pytest.mark.parametrize("name", _DECORATED)
    @pytest.mark.parametrize("lang,tmpl", [
        ("ja", "E1 を {v} で偏微分して"),
        ("en", "differentiate E1 with respect to {v} partially"),
    ])
    def test_the_derivative_is_one(self, lang, tmpl, name):
        r, cmd = _run(lang, [f"z^2 + {name}", tmpl.format(v=name)])
        assert r.get("status") == "completed", r
        assert _value(r) == "1", (r, cmd)

    @pytest.mark.parametrize("name,call", [(r"\bar{y}", "bar("), (r"\hat{H}", "hat("),
                                           (r"\tilde{x}", "tilde(")])
    def test_the_variable_is_not_a_function_call(self, name, call):
        """飾りは**名前の一部**であって関数ではない（これが 0 の原因だった）。"""
        _, cmd = _run("ja", [f"z^2 + {name}", f"E1 を {name} で偏微分して"])
        assert call not in cmd, cmd


class TestAPrimeIsPartOfTheName:
    r"""`y'` は別の変数である。切り詰めると式の `y'` と変数の `y` が別物になる。"""

    @pytest.mark.parametrize("lang,text", [
        ("ja", "E1 を y' で偏微分して"),
        ("en", "differentiate E1 with respect to y' partially"),
    ])
    def test_the_derivative_is_one(self, lang, text):
        r, cmd = _run(lang, ["z^2 + y'", text])
        assert r.get("status") == "completed", r
        assert _value(r) == "1", (r, cmd)

    @pytest.mark.parametrize("lang,text,want", [
        ("ja", "E1 を y' で 0 から 1 まで定積分して", "(2*z^2+1)/2"),
        ("ja", "E1 を y' について解いて", None),
        ("en", "solve E1 for y'", None),
    ])
    def test_the_other_families_read_it_too(self, lang, text, want):
        """`symbol` は共有なので、広げた効果は1つの族に留まらない。"""
        r, _ = _run(lang, ["z^2 + y'", text])
        assert r.get("status") == "completed", r
        if want is not None:
            assert _value(r) == want, r


class TestTheOrdinaryVariablesAreUnchanged:
    @pytest.mark.parametrize("name,expr,want", [
        (r"\omega", r"z^2 + \omega", "1"),
        (r"\omega_0", r"z^2 + \omega_0", "1"),
        ("x", "z^2 + x", "1"),
        ("t", "z^2 + t", "1"),
    ])
    def test_it_still_differentiates(self, name, expr, want):
        r, _ = _run("ja", [expr, f"E1 を {name} で偏微分して"])
        assert r.get("status") == "completed", r
        assert _value(r) == want, r


class TestTheLagrangianAtomsStillWork:
    r"""`q` と `q̇` は独立な原子——扉を通すと `\dot{q}` は `q_dot` になる。
    そこが壊れると解析力学が丸ごと壊れるので、族ごと固定する。"""

    _SETUP = {"ja": ["q を時間依存とする", r"L = \dot{q}^2/2 - q^2/2"],
              "en": ["declare q as time dependent", r"L = \dot{q}^2/2 - q^2/2"]}

    @pytest.mark.parametrize("lang,text,want", [
        ("ja", r"E1 を \dot{q} で偏微分して", "q_dot"),
        ("ja", "E1 を q で偏微分して", "-q"),
        ("ja", "E1 を t で全微分して", "(q_ddot-q)*q_dot"),
        ("ja", r"E1 を \ddot{q} で偏微分して", "0"),
        ("en", r"differentiate E1 with respect to \dot{q} partially", "q_dot"),
        ("en", "differentiate E1 with respect to q partially", "-q"),
    ])
    def test_the_value(self, lang, text, want):
        r, _ = _run(lang, self._SETUP[lang] + [text], domain="analytical_mechanics")
        assert r.get("status") == "completed", r
        assert _value(r) == want, r


class TestSolveAlreadyWentThroughTheDoor:
    """先例——2026-08-21 に solve だけが直されていた。退行させない。"""

    def test_the_variable_is_the_decorated_name(self):
        _, cmd = _run("ja", [r"z^2 + \bar{y}", r"E1 を \bar{y} について解いて"])
        assert "y_bar" in cmd and "bar(y)" not in cmd, cmd


class TestTheCardShowsTheResearchersSpelling:
    r"""**扉を通したら、カードが内部名を印字しはじめた。**

    ラベルは `args['var']` をそのまま入れており、変数が正準形になった日に
    `E2 をlambda で偏微分する` / `E66 をy_bar で偏微分する` になった（実測
    2026-09-19）。すぐ隣の `substitute` の枝は同じ理由で**表示の扉**
    （`_display_arg` → `canonical_name_to_latex`）を通している。

    **コーパス差分が捕まえた——ただし比較器を直したあとで。** 欄の名前を
    `operation_args` と書いていたので、`args` の変化が見えず「0」と出ていた。
    """

    @pytest.mark.parametrize("lang,lines,want", [
        ("ja", [r"\bar{y}^2 + t'", r"E1 を \bar{y} で偏微分して"], r"\bar{y}"),
        ("ja", [r"\bar{y}^2 + t'", r"E1 を t' で偏微分して"], "t'"),
        ("ja", [r"x^2 + \lambda", r"E1 を \lambda で偏微分して"], r"\lambda"),
        ("en", [r"\bar{y}^2 + t'",
                r"differentiate E1 with respect to \bar{y} partially"], r"\bar{y}"),
    ])
    def test_the_label_carries_the_written_name(self, lang, lines, want):
        async def go():
            nb.start_engine(title="card", mode="standard", language=lang,
                            domains=["classical_mechanics"])
            try:
                engine = nb.get_engine()
                sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                                    domains=["classical_mechanics"],
                                    primary_backend=engine.session.primary_backend)
                for line in lines:
                    await run_interpret_pipeline(
                        InterpretRequest(tokens=[InputToken(type="text", value=line)]),
                        engine, sup)
                return [s.visible_action for s in engine.steps.values()
                        if s.kind == "cas_operation" and s.operation]
            finally:
                nb.stop_engine()
        labels = asyncio.run(go())
        assert labels, "カードが1枚も無い"
        # `want` そのものが判別子である——壊れていた版は `y_bar` / `lambda`
        # （バックスラッシュ無し）を印字するので、この一致では通らない。
        assert want in labels[-1], labels


#: 変数を取る族のうち、**別名走査**が欄を埋める綴り（インターセプタを持たない側）。
#: **値は変数の綴りで変わる**（`y'` は Maxima 側で `y\\'`、`x_1` は `x(1)`）ので、
#: ここで固定するのは「操作に届いたか」——値は下の1件で見る。
_ALIAS_SCAN = [
    ("ja", "E1 を {v} で積分して"),
    ("ja", "E1 を {v} について部分分数分解して"),
    ("ja", "E1 の {v} = 1 における留数を求めて"),
    ("en", "integrate E1 over {v}"),
    ("en", "the residue of E1 at {v} = 1"),
]


class TestANameIsNotOnlyAPlainIdentifier:
    r"""**別名走査が欄を埋める経路は、素の識別子しか変数にできなかった。**

    実測 2026-09-19、`z^2 + X` を登録して `X` を変数に書いた指示を族ごと:
    `\bar{y}` / `y'` / `x_1` のどれでも「必要な引数が不足しています」——
    不定積分・部分分数・留数・極限・collect、日英とも（24セル）。
    `\alpha` だけが通っていたのは**手書きの規則が1つ在った**からで、
    `\bar{y}` が通らない理由は誰も決めていなかった。

    **`x_1` は普通の変数である**——通ると不定積分も微分も正しく計算する。
    """

    @pytest.mark.parametrize("name,canon", [(r"\bar{y}", "y_bar"), ("y'", "y'"),
                                            ("x_1", "x(1)")])
    @pytest.mark.parametrize("lang,tmpl", _ALIAS_SCAN)
    def test_the_variable_reaches_the_operation(self, lang, tmpl, name, canon):
        text = tmpl.format(v=name)
        r, _ = _run(lang, [f"z^2 + {name}", text])
        assert r.get("status") == "completed", (text, r)
        assert "Maxima" not in _msg(r), (text, r)

    @pytest.mark.parametrize("lang,text", [
        ("ja", r"E1 を \bar{y} で積分して"),
        ("en", r"integrate E1 over \bar{y}"),
    ])
    def test_and_the_value_is_the_decorated_name(self, lang, text):
        r, _ = _run(lang, [r"z^2 + \bar{y}", text])
        assert _value(r) == "y_bar*z^2+y_bar^2/2+C", r

    def test_the_order_is_as_written(self):
        r"""救った名前を末尾へ足すと、`E1 の \bar{y} = 1 における留数を求めて` が
        `var=1, point=y_bar` になり生の CAS 文言が返る（実測、自分で一度踏んだ）。"""
        r, _ = _run("ja", [r"z^2 + \bar{y}", r"E1 の \bar{y} = 1 における留数を求めて"])
        assert r.get("status") == "completed", r
        assert "Maxima" not in _msg(r), r


class TestTheSharedNameConverterAgreesWithTheBackends:
    r"""**校正**——ヘルパが答えるときは、実物の変換器と一致しなければならない。"""

    _NAMES = [r"\bar{y}", r"\hat{H}", r"\tilde{x}", r"\alpha", r"\alpha_k", "y'",
              "x_1", "H", r"\bar{\phi}", r"\mathcal{H}", r"\dot{q}", r"\ddot{q}",
              r"\omega_0", "E1", "k_B", r"\Delta", r"\bar{y}_1"]

    @pytest.mark.parametrize("name", _NAMES)
    def test_it_agrees_with_both_backends(self, name):
        from notebook.runtime.julia_backend_handler import JuliaBackendHandler
        from notebook.runtime.latex_converter import latex_name_to_canonical
        from notebook.runtime.maxima_backend_handler import MaximaBackendHandler
        mine = latex_name_to_canonical(name)
        assert mine is not None, name
        for handler in (MaximaBackendHandler(None), JuliaBackendHandler(None)):
            assert mine == (handler.latex_to_canonical(name) or "").strip(), (name, mine)

    @pytest.mark.parametrize("text", [r"\vec{A}", r"\mathbf{r}", r"\sqrt{x}",
                                      "a+b", "f(x)", "[1, 2]", "2", ""])
    def test_what_is_not_a_name_gets_no_answer(self, text):
        r"""**包みは飾りではない**——`\vec{A}` の変換器の答えは `vec(A)` で、
        名前ではない。表に無いコマンドには答えない（実測、`vec{A}` を返していた）。"""
        from notebook.runtime.latex_converter import latex_name_to_canonical
        assert latex_name_to_canonical(text) is None, text


class TestAWiderReadingOpensANewExit:
    r"""`x_1` が変数の欄に入るようになった日に、`limit` が CAS まで届いた。

    実測 2026-09-19、Maxima は添字付き（`x(1)`）を**操作によって**受けたり断ったり
    する——定積分・`limit`・`taylor`・`sum` は断り、`diff`・不定積分・`residue`・
    `partfrac`・`coeff`・`solve`・`charpoly` は受ける。だから事前ではなく
    **断られてから読み替える**（DEFINED-NAME-AS-VARIABLE と同じ場所）。
    """

    def test_the_limit_refusal_is_translated(self):
        r, _ = _run("ja", ["z^2 + x_1", "E1 の x_1 に関する極限を 0 で求めて"])
        assert r.get("status") == "failed", r
        assert "Maxima" not in _msg(r), r
        assert "x_1" in _msg(r) and "x(1)" in _msg(r), r

    @pytest.mark.parametrize("text,want", [
        ("E1 を x_1 で積分して", "x(1)*z^2+x(1)^2/2+C"),
        ("E1 を x_1 で偏微分して", "1"),
    ])
    def test_an_operation_the_cas_accepts_is_untouched(self, text, want):
        """断らない操作に断り文は要らない——不定積分と微分は計算する。"""
        r, _ = _run("ja", ["z^2 + x_1", text])
        assert r.get("status") == "completed", (text, r)
        assert _value(r) == want, (text, r)


class TestTheDisplayRestoreReadsBracesInPairs:
    r"""**飾りのある名前を変数に書けるようにしたら、公開物の `.tex` が組版に失敗した。**

    表示の復元（`phi_bar` → `\bar{\phi}`、`G_sup0` → `G^{0}`）は波括弧を
    `\{?` / `\}?` と**独立に**任意にしていたので、閉じ波括弧が隣に在るだけで
    それを食っていた。実測 2026-09-19 —— この族に限らない、前から在った欠陥:

        \sqrt{H_hat}                        → \sqrt{\hat{H}          （組版不能）
        \frac{\partial H_hat}{\partial x}   → \frac{\partial \hat{H}{\partial x}
        {G_sup0}                            → {G^{0}

    捕まえたのは `test_tex_compiles` —— **実際に組む**検査が在る理由そのもので、
    静的な走査も 5000 以上のテストも緑だった。
    """

    @pytest.mark.parametrize("text,want", [
        (r"\frac{\partial}{\partial y_bar}\left(x\right)",
         r"\frac{\partial}{\partial \bar{y}}\left(x\right)"),
        (r"\frac{\partial H_hat}{\partial x}", r"\frac{\partial \hat{H}}{\partial x}"),
        (r"\frac{d y_bar}{d t}", r"\frac{d \bar{y}}{d t}"),
        (r"\sqrt{H_hat}", r"\sqrt{\hat{H}}"),
        (r"{G_sup0}", r"{G^{0}}"),
        (r"\frac{\partial G_sup0}{\partial t}", r"\frac{\partial G^{0}}{\partial t}"),
        # 波括弧が**対で書かれている**綴りは従来どおり読む
        (r"phi_{bar}", r"\bar{\phi}"),
        (r"x_{i} + H_hat", r"x_{i} + \hat{H}"),
        (r"G_{sup0}", r"G^{0}"),
    ])
    def test_the_braces_stay_balanced(self, text, want):
        from notebook.runtime.backend_handler import restore_decorated_name
        got = restore_decorated_name(text)
        assert got == want, (text, got)
        assert got.count("{") == got.count("}"), got

    def test_the_published_examples_have_balanced_braces(self):
        """公開物そのものを数える——この欠陥が出た場所である。"""
        import pathlib as _p
        bad = []
        for tex in sorted(_p.Path("domains").glob("*/examples/*.tex")):
            body = tex.read_text(encoding="utf-8")
            if body.count("{") != body.count("}"):
                bad.append(f"{tex.name}: {body.count('{')} vs {body.count('}')}")
        assert not bad, bad

class TestTheAliasScanReadsADecoratedVariableToo:
    r"""**2026-09-19 に解決**（DECORATED-VAR-NOT-TOKENIZED）——ここは strict xfail
    だった。別名走査が欄を埋める経路には「素の識別子と `\alpha` だけ」という
    手書きの規則が在り、`\bar{y}` / `y'` / `x_1` を落としていた。"""

    @pytest.mark.parametrize("lang,text", [
        ("ja", r"E1 を \bar{y} で積分して"),
        ("en", r"integrate E1 over \bar{y}"),
    ])
    def test_the_indefinite_integral_reads_it(self, lang, text):
        r, _ = _run(lang, [r"z^2 + \bar{y}", text])
        assert r.get("status") == "completed", r
        assert _value(r) == "y_bar*z^2+y_bar^2/2+C", r
