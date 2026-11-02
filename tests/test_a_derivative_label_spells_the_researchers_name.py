r"""DERIV-DISPLAY-CANONICAL-NAME（2026-09-24）——導関数のラベルは研究者の綴りで書く。

`_format_provenance_lhs` が微分変数を**内部の正準名**のまま埋めていた。組版すると
`\frac{d}{dtheta}` は斜体の単語「dtheta」であって `d\theta` ではない
（CLAUDE.md が 2026-08-22 に記録した `\[ H_hat_0 \]` と同じ型）。

**そして表示だけの話ではなかった。** BP-9 の逆変換がその綴りを読み戻せないので、
`d` と変数が**約分されて**ただの分数・ただの積になる——BP-9 が防ぐために在る当の
失敗である:

| 生成された表示 | 読み戻すと |
|---|---|
| `\frac{df}{dtheta}` | `((df)/(dtheta))` ← ∂ が消える |
| `\frac{df}{domega_0}` | `((df)/(domega_0))` |
| `\frac{d}{dtheta}\left(a*b+c\right)` | `((d)/(dtheta))*(a*b+c)` ← **積** |
| `\frac{\partial phi}{\partial x}` | 往復はするが「phi」と組版される |
| `\frac{d}{dt}\left(H_hat_0\right)` | 内部名が公開 TeX に出る |

**「素の名前は LaTeX でどう綴るか」に3つの答えが在った。**
`backend_handler.canonical_name_to_latex`（完全——研究者が宣言した `latex_name`、
装飾、ギリシャ文字＋添字、スピン成分）、`lhs_formatting._latex_var`（ギリシャ文字
だけ。**同じファイルの積分**が使う）、`integral_form.latex_var`（ギリシャ文字だけ）。
導関数はどれも訊いていなかった。素の名前（`x`・`r`・`t`・`k_B`・`E_total`）では
3つが一致するので、畳んでも他は動かない。

**広げたら出口が2つ増えた**（数え直した）: ①`_NAME` はコマンド＋ブレース引数
（`\bar{y}`）を知らず、`\partial \bar{y}` が `\partial*f/(\partial*y_bar)` に
なった。②`_DERIV_BUMP_RE` の変数クラスがブレースを弾き、かつ**表示の綴りと
正準名を直接比べて**いたので、次数の繰り上げ（d/dk(d/dk(X)) → d²/dk²）が黙って
止まるところだった。比較も同じ扉を通す。
"""
import pytest

from notebook.engine import NotebookEngine
from notebook.runtime.integral_form import latex_var
from notebook.runtime.lhs_formatting import (_bump_same_var_derivative,
                                             _format_provenance_lhs, _latex_var)


def _handler():
    return NotebookEngine()._handler


class TestOneDoorAnswersHowANameIsSpelled:
    """3つの答えを1つに畳んだ。素の名前では元から一致していた。"""

    @pytest.mark.parametrize("name,want", [
        ("theta",   r"\theta"),
        ("Theta",   r"\Theta"),
        ("omega_0", r"\omega_{0}"),
        ("y_bar",   r"\bar{y}"),
        ("H_hat_0", r"\hat{H}_{0}"),
        ("x", "x"), ("r", "r"), ("t", "t"), ("k_B", "k_B"), ("E_total", "E_total"),
        ("y'", "y'"), ("x_1", "x_1"),
    ])
    def test_both_helpers_agree(self, name, want):
        assert _latex_var(name) == want, name
        assert latex_var(name) == want, name


class TestTheLabelSpellsTheName:
    @pytest.mark.parametrize("origin,dvars,is_partial,latex", [
        ("f", ["theta"],   False, r"\frac{df}{d\theta}"),
        ("f", ["theta"],   True,  r"\frac{\partial f}{\partial \theta}"),
        ("f", ["omega_0"], False, r"\frac{df}{d\omega_{0}}"),
        ("f", ["y_bar"],   False, r"\frac{df}{d\bar{y}}"),
        ("f", ["theta", "theta"], False, r"\frac{d^{2}f}{d\theta^{2}}"),
        ("phi", ["x"],     True,  r"\frac{\partial \phi}{\partial x}"),
        ("H_hat_0", ["t"], False, r"\frac{d}{dt}\left(\hat{H}_{0}\right)"),
        ("y_bar", ["x"],   False, r"\frac{d}{dx}\left(\bar{y}\right)"),
        # 式は名前ではない——そのまま包む
        ("a*b+c", ["theta"], False, r"\frac{d}{d\theta}\left(a*b+c\right)"),
        # 素の名前は今までどおり
        ("f", ["x"], False, r"\frac{df}{dx}"),
    ])
    def test_display(self, origin, dvars, is_partial, latex):
        got = _format_provenance_lhs(origin, dvars, is_partial, qty=None,
                                     parse_ir=_handler().parse_ir)
        assert got == latex, (origin, dvars, got)


class TestTheLabelReadsBack:
    r"""**表示が直ったことは往復が閉じたことでもある。** 直す前、総微分の分母が
    2文字以上の名前だと `d` が約分されてただの分数になっていた。"""

    @pytest.mark.parametrize("origin,dvars,is_partial,canonical", [
        ("f", ["theta"],   False, "'diff(f,theta,1)"),
        ("f", ["theta"],   True,  "'diff(f,theta,1)"),
        ("f", ["omega_0"], False, "'diff(f,omega_0,1)"),
        ("f", ["y_bar"],   False, "'diff(f,y_bar,1)"),
        ("f", ["y_bar"],   True,  "'diff(f,y_bar,1)"),
        ("f", ["theta", "theta"], False, "'diff(f,theta,2)"),
        ("phi", ["x"],     True,  "'diff(phi,x,1)"),
        ("H_hat_0", ["t"], False, "'diff(H_hat_0,t,1)"),
        ("a*b+c", ["theta"], False, "'diff(a*b+c,theta,1)"),
    ])
    def test_roundtrip(self, origin, dvars, is_partial, canonical):
        h = _handler()
        latex = _format_provenance_lhs(origin, dvars, is_partial, qty=None,
                                       parse_ir=h.parse_ir)
        assert h.latex_to_canonical(latex) == canonical, latex


class TestTheOrderBumpStillCollapses:
    r"""d/dk(d/dk(X)) → d²/dk²。**綴りを変えると黙って止まる**ので、比較も同じ扉。"""

    @pytest.mark.parametrize("var,bumped", [
        ("theta",   r"\frac{d^{2}}{d\theta^{2}}\left(a*b+c\right)"),
        ("x",       r"\frac{d^{2}}{dx^{2}}\left(a*b+c\right)"),
        ("y_bar",   r"\frac{d^{2}}{d\bar{y}^{2}}\left(a*b+c\right)"),
        ("omega_0", r"\frac{d^{2}}{d\omega_{0}^{2}}\left(a*b+c\right)"),
        ("y'",      r"\frac{d^{2}}{dy'^{2}}\left(a*b+c\right)"),
    ])
    def test_bump(self, var, bumped):
        h = _handler()
        first = _format_provenance_lhs("a*b+c", [var], False, qty=None,
                                       parse_ir=h.parse_ir)
        assert _bump_same_var_derivative(first, var, 1, parse_ir=h.parse_ir) == bumped


class TestWhatTheResearcherSees:
    r"""カードに出る綴り——日英とも。`\frac{d}{dtheta}` は「dtheta」と組版される。"""

    @staticmethod
    def _run(lang, setup, line):
        import asyncio

        import ui.notebook_bridge as nb
        from notebook.runtime.llm_supervisor import LLMSupervisor
        from notebook.runtime.operation_validator import OperationValidator
        from ui.interpret_pipeline import (InputToken, InterpretRequest,
                                           run_interpret_pipeline)
        domain = "classical_mechanics"

        async def go():
            nb.start_engine(title="lbl", mode="standard", language=lang,
                            domains=[domain])
            try:
                engine = nb.get_engine()
                sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                                    domains=[domain],
                                    primary_backend=engine.session.primary_backend)
                for text in (setup, line):
                    resp = await run_interpret_pipeline(
                        InterpretRequest(tokens=[InputToken(type="text", value=text)]),
                        engine, sup)
                last = list(engine.expressions.values())[-1]
                return resp, last.latex
            finally:
                nb.stop_engine()
        return asyncio.run(go())

    @pytest.mark.parametrize("lang", ["ja", "en"])
    def test_a_greek_variable(self, lang):
        line = (r"E1 を \theta で微分して" if lang == "ja"
                else r"differentiate E1 with respect to \theta")
        resp, latex = self._run(lang, r"\theta^3", line)
        assert resp.get("status") == "completed", resp.get("error")
        assert r"d\theta" in latex, latex
        assert "dtheta" not in latex, latex
        assert latex.endswith(r"3 \theta^2"), latex
