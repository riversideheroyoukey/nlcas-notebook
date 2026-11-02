r"""運動方程式の記法（GF-EOM、2026-08-21）。

シナリオ S-ak の続き——グリーン関数の運動方程式を作り、高次のグリーン関数を
Hartree–Fock 分解で低次に近似する——を書き下したときに実測で出た6件。前半（Dyson の
道）に無かった綴りだけが壊れていて、**エラーになったものは一つも無い**:

    \frac{\partial}{\partial t}\hat c_{n\sigma}(t) → 'diff(\hat,t,1)*c_{n,sigma}(t)
    \frac{\partial \hat\psi(x,t)}{\partial t}      → psi_hat(x,t)/t   （∂ が約分）
    \frac{d}{dx}\sin x                             → 'diff(sin,x,1)*x ＝ 0
    \hat T(\mathbf r)                              → time_order(…, hat)
    A \approx B + C                                → A*B*approx+C（simplify が成功）
    E33 を G_{n\sigma}(\omega) について解いて       → 解釈されない

最初の3件は**一つの原因**である: 微分される対象を読む走査が、`\hat` や `\sin` を
「それだけで完結した名前」と見ていた。コマンドは単独では量ではないので、その先まで
取るのは推測ではなく文法である。
"""
import asyncio

import pytest

import ui.notebook_bridge as nb
from notebook.engine import NotebookEngine
from notebook.runtime.i18n import t
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.nl_patterns import nlp
from notebook.runtime.operation_validator import OperationValidator
from ui.interpret_equation_ops import _parse_solve
from ui.supervisor_alias_rewrite import rewrite_symbols_to_eids
from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline


def _handler():
    return NotebookEngine()._handler


def _run(lang, lines, domain="quantum_mechanics"):
    """1セッションに順に入力して、(最後の応答, 全式) を返す。"""
    nb.start_engine(title="gf_eom", mode="standard", language=lang,
                    domains=[domain])
    try:
        eng = nb.get_engine()
        sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                            domains=[domain], primary_backend="maxima")
        resp = None
        for line in lines:
            resp = asyncio.get_event_loop().run_until_complete(
                run_interpret_pipeline(
                    InterpretRequest(tokens=[InputToken(type="text", value=line)]),
                    eng, sup))
        return resp, {i: e for i, e in eng.expressions.items()}
    finally:
        nb.stop_engine()


class TestDecoratedOperandIsDifferentiated:
    r"""装飾は名前の一部なので、微分の対象はコマンドの先まで伸びる。"""

    @pytest.mark.parametrize("latex,canonical", [
        # 演算子形（分数の後ろに対象が立つ）
        (r"\frac{d}{dt} \hat c(t)",                    "'diff(c_hat(t),t,1)"),
        (r"\frac{d}{dt} \hat{c}(t)",                   "'diff(c_hat(t),t,1)"),
        (r"\frac{\partial}{\partial t} \hat\psi(x,t)", "'diff(psi_hat(x,t),t,1)"),
        (r"\frac{\partial}{\partial t} \hat c_{n\sigma}(t)",
         "'diff(c_hat_{n,sigma}(t),t,1)"),
        (r"\frac{d}{dt} \bar{q}(t)",                   "'diff(q_bar(t),t,1)"),
        (r"\frac{d}{dt} \mathcal{H}(t)",               "'diff(H_cal(t),t,1)"),
        (r"\frac{d}{dt} \dot q",                       "'diff(q_dot,t,1)"),
        # 分子が対象を抱える形。こちらは以前**導関数として一致すらせず**、
        # 一般の分数規則に落ちて ∂ が約分されていた。
        (r"\frac{\partial \hat\psi(x,t)}{\partial t}", "'diff(psi_hat(x,t),t,1)"),
        (r"\frac{d \bar{q}(t)}{dt}",                   "'diff(q_bar(t),t,1)"),
    ])
    def test_decorated(self, latex, canonical):
        assert _handler().latex_to_canonical(latex) == canonical

    @pytest.mark.parametrize("latex,canonical", [
        (r"\frac{d}{dx} \sin x",      "'diff(sin(x),x,1)"),
        (r"\frac{d}{dx} \sin(x)",     "'diff(sin(x),x,1)"),
        (r"\frac{d}{dx} \log x",      "'diff(log(x),x,1)"),
        # 関数自身の上付きは引数と一緒に渡る（読むのは `function_application`）。
        (r"\frac{d}{dx} \sin^2 x",    "'diff(sin(x)^2,x,1)"),
        (r"\frac{d}{dx} \sin^{-1} x", "'diff(asin(x),x,1)"),
    ])
    def test_function_command(self, latex, canonical):
        """関数名も単独では量ではない（`\\sin` は `sin` という記号ではない）。"""
        assert _handler().latex_to_canonical(latex) == canonical

    @pytest.mark.parametrize("latex,canonical", [
        (r"\frac{d f(t)}{dt}",     "'diff(f(t),t,1)"),
        (r"\frac{d}{dt} f(t)",     "'diff(f(t),t,1)"),
        (r"\frac{d}{d z} \varphi", "'diff(phi,z,1)"),
    ])
    def test_undecorated_unchanged(self, latex, canonical):
        """装飾の無い形は以前のまま——この修正は取りこぼしを埋めるだけ。"""
        assert _handler().latex_to_canonical(latex) == canonical


class TestKineticOperatorIsNotTimeOrdering:
    r"""``\hat T`` は1体の運動エネルギーであって、時間順序演算子ではない。

    装飾を畳んだ段階で ``T_hat(...)`` になっており、松原の τ 添字を読む規則が
    **任意の識別子**を許していたため ``_hat`` を τ として読んでいた。読むのは τ だけ
    ——そう書いてあるときだけ「順序である」と言えるからで、素の ``T`` を読まない
    のと同じ理由である。
    """

    def test_one_body_kinetic_term(self):
        latex = (r"\hat H_0 = \int d^3r\, \hat\psi^\dagger_\sigma(\mathbf{r}) "
                 r"\hat T(\mathbf{r}) \hat\psi_\sigma(\mathbf{r})")
        canonical = _handler().latex_to_canonical(latex)
        assert "T_hat(vec(r))" in canonical
        assert "time_order" not in canonical

    @pytest.mark.parametrize("latex,expect", [
        (r"\langle T[\psi(x) \psi^\dagger(y)]\rangle",
         "expectation(time_order(psi(x)*psi^\\dagger(y)))"),
        (r"\langle \mathcal{T}[\psi(x) \psi^\dagger(y)]\rangle",
         "expectation(time_order(psi(x)*psi^\\dagger(y)))"),
        (r"\langle T_\tau[\psi(x) \psi^\dagger(y)]\rangle",
         "expectation(time_order(psi(x)*psi^\\dagger(y), tau))"),
        (r"\langle T_{\tau}[\psi(x) \psi^\dagger(y)]\rangle",
         "expectation(time_order(psi(x)*psi^\\dagger(y), tau))"),
    ])
    def test_ordering_spellings_still_read(self, latex, expect):
        assert _handler().latex_to_canonical(latex) == expect

    def test_matrix_element_still_a_subscript(self):
        assert _handler().latex_to_canonical("T[i,j]") == "T[i,j]"


class TestApproximationIsWritableAndRefusedAtUse:
    r"""近似の関係は**書けて**、計算では断られる。

    `\neq` で直したのと同じ形が残っていた: `\approx` は生のコマンドのまま残り、
    暗黙の積が入って `A * approx * B + C` になっていた。`\neq` と違って
    パーサの拒否文字には入れない——近似は書けなければ意味がない（それが導出の
    turning point である）ので、断るのは**計算する地点**に置く。
    """

    @pytest.mark.parametrize("latex", [
        r"A \approx B + C", r"A \simeq B + C", r"A \cong B + C",
    ])
    def test_every_spelling_folds_to_one(self, latex):
        assert _handler().latex_to_canonical(latex) == "A ≈ B + C"

    def test_registers_and_keeps_the_written_display(self):
        latex = (r"G^{(2)}_{p\sigma'',s\sigma,m\sigma'} \approx "
                 r"\langle \hat c^\dagger_{p\sigma''} \hat c_{p\sigma''}\rangle "
                 r"G_{s\sigma,m\sigma'}")
        resp, exprs = _run("ja", [latex])
        assert resp["status"] == "registered", resp
        expr = list(exprs.values())[-1]
        assert expr.latex == latex          # 研究者の綴りのまま表示される
        assert "≈" in expr.canonical

    def test_computing_with_it_is_refused_with_a_reason(self):
        nb.start_engine(title="approx", mode="standard", language="ja",
                        domains=["quantum_mechanics"])
        try:
            eng = nb.get_engine()
            expr = eng.register_expression("A ≈ B + C", latex=r"A \approx B + C")
            with pytest.raises(Exception) as exc:
                eng.run_operation("simplify", {"expr": expr.display_id})
            assert t("generator.approx_no_cas_form", "ja") in str(exc.value)
        finally:
            nb.stop_engine()


class TestSolveForTheWrittenVariable:
    r"""解く変数は**研究者の綴りのまま**受け取り、降ろすのはバックエンドの扉に任せる。

    綴りを読む側では波括弧つきの添字が許されておらず（コマンドで始まる名前だけが
    許されていた）、許したとしても NL 側の二度目の降ろしが `G_{n\sigma}` を
    `G_{nsigma}` という**一つの名前**に潰していた——式のどこにも現れない変数なので、
    Maxima は「解なし」と答える。
    """

    @pytest.mark.parametrize("text,want", [
        (r"E33 を G_{n\sigma}(\omega) について解いて", r"G_{n\sigma}(\omega)"),
        (r"E33 を G_{n\sigma} について解いて",          r"G_{n\sigma}"),
        (r"E4 を \ddot{q} について解いて",              "q_ddot"),
        (r"E4 を q について解いて",                      "q"),
        (r"E4 を Az_z について解いて",                   "Az_z"),
        (r"E4 を \Delta について解いて",                 r"\Delta"),
    ])
    def test_japanese(self, text, want):
        assert _parse_solve(text, "ja") == ("solve_equation",
                                            {"equation": text.split()[0], "var": want})

    @pytest.mark.parametrize("text,want", [
        (r"solve E33 for G_{n\sigma}(\omega)", r"G_{n\sigma}(\omega)"),
        (r"solve E33 for G_{n\sigma}",         r"G_{n\sigma}"),
        (r"solve E4 for \ddot{q}",             "q_ddot"),
        ("solve E4 for q",                     "q"),
        (r"solve E4 for \Delta",               r"\Delta"),
    ])
    def test_english(self, text, want):
        assert _parse_solve(text, "en") == ("solve_equation",
                                            {"equation": text.split()[1], "var": want})

    @pytest.mark.parametrize("lang,lines", [
        ("ja", [r"(\omega - \varepsilon_{n\sigma}/\hbar) G_{n\sigma}(\omega) = "
                r"1 + \Sigma^{(HF)}_{n\sigma} G_{n\sigma}(\omega)/\hbar",
                r"E1 を G_{n\sigma}(\omega) について解いて"]),
        ("en", [r"(\omega - \varepsilon_{n\sigma}/\hbar) G_{n\sigma}(\omega) = "
                r"1 + \Sigma^{(HF)}_{n\sigma} G_{n\sigma}(\omega)/\hbar",
                r"solve E1 for G_{n\sigma}(\omega)"]),
    ])
    def test_hartree_fock_green_function_end_to_end(self, lang, lines):
        """運動方程式を解くと Hartree–Fock のグリーン関数が出る。"""
        resp, exprs = _run(lang, lines)
        assert resp["status"] == "completed", resp
        got = exprs[resp["expr_id"]].canonical.replace(" ", "")
        assert got == "hbar/(hbar*omega-epsilon(n,sigma)-Sigma_psupHF(n,sigma))", got


class TestOperandCarriesItsScripts:
    r"""``\frac{d}{dx} x^2`` は x² の微分。指数が微分の外に残ると別物になる。"""

    @pytest.mark.parametrize("latex,canonical", [
        (r"\frac{d}{dx} x^2",                    "'diff(x^2,x,1)"),
        (r"\frac{d x^2}{dx}",                    "'diff(x^2,x,1)"),
        (r"\frac{d}{dx} f(x)^2",                 "'diff(f(x)^2,x,1)"),
        (r"\frac{d}{dt} G^0(t)",                 "'diff(G_sup0(t),t,1)"),
        (r"\frac{d}{dt} G^{(HF)}_{n\sigma}(t)",  "'diff(G_psupHF_{n,sigma}(t),t,1)"),
        (r"\frac{d}{dt}\left(f(t)+g(t)\right)^2", "'diff((f(t)+g(t))^2,t,1)"),
        # 括弧だけ（指数なし）は以前のまま——中身がそのまま渡る。
        (r"\frac{d}{dt}\left(f(t)+g(t)\right)",  "'diff(f(t)+g(t),t,1)"),
    ])
    def test_scripts_belong_to_the_operand(self, latex, canonical):
        assert _handler().latex_to_canonical(latex) == canonical


class TestLabelIsNeverAnIndex:
    r"""上付きラベルは**名前**であって添字ではない——宣言があっても。

    `\Sigma` を行列と宣言したセッションで `\Sigma^{(HF)}(n,\sigma)` が
    `Sigma[psupHF](n,sigma)` に割れ、続く代入がどこにも一致せず**成功として
    何もしなかった**（AUDIT-SUBST-INDEXED と同じ形）。
    """

    def test_declared_matrix_base_does_not_split_the_label(self):
        nb.start_engine(title="label", mode="standard", language="ja",
                        domains=["quantum_mechanics"])
        try:
            eng = nb.get_engine()
            eng.run_operation("define_matrix", {"name": "Sigma"})
            got = eng._handler.to_backend_code("Sigma_psupHF(n,sigma)")
            assert got.code == "Sigma_psupHF(n, sigma)", got
        finally:
            nb.stop_engine()

    def test_display_restores_the_label_beside_a_subscript(self):
        """Maxima の `{\\it G\\_psupHF}_{n,\\sigma}` は `G^{(HF)}_{n,\\sigma}`。"""
        eng = NotebookEngine()
        got = eng._handler.postprocess_result_latex(
            "G_psupHF[n,sigma](omega)",
            r"{\it G\_psupHF}_{n,\sigma}(\omega)", eng.symbol_context)
        assert got == r"G^{(HF)}_{n,\sigma}(\omega)", got

    def test_the_digit_split_case_still_merges(self):
        """`x_ic1` は Maxima が `{\\it x\\_ic}_{1}` に割る——1つの名前に戻す。"""
        eng = NotebookEngine()
        got = eng._handler.postprocess_result_latex(
            "x_ic1", r"{\it x\_ic}_{1}", eng.symbol_context)
        assert got == "x_{ic1}", got


class TestDefinedNameIsNotRewrittenInsideALabel:
    r"""2a-sym は**同じ名前**だけを E-ID に置き換える。

    `G` が定義済みのセッションで `G^{(HF)}_{n\sigma}(t,t')` が
    `E22^{(HF)}_{n\sigma}(t,t')` に化け、式が登録できなくなっていた。指数
    （`G^2`）はこれまでどおり置き換わる——そちらは定義された量そのものだから。
    """

    class _Eng:
        def __init__(self, alias):
            self.lhs_alias_map = alias

    @pytest.mark.parametrize("text,want", [
        (r"G^{(HF)}_{n\sigma}(t,t')", r"G^{(HF)}_{n\sigma}(t,t')"),   # ラベル
        (r"G^0 + 1",                  r"G^0 + 1"),                    # 退化ラベル
        (r"G^2 + 1",                  r"E22^2 + 1"),                  # 指数
        (r"G + 1",                    r"E22 + 1"),                    # そのもの
        (r"G_{n\sigma}",              r"G_{n\sigma}"),                # 添字（既存）
    ])
    def test_only_the_same_name_is_rewritten(self, text, want):
        assert rewrite_symbols_to_eids(text, self._Eng({"G": "E22"})) == want


class TestSolveVariableComesFromTheResearchersText:
    r"""解く変数が**定義済みの名前**でも E-ID に化けない。

    `_parse_solve` の連立分岐だけが alias_free_text から読み直しており、単一
    方程式の分岐は E-ID 置換後の文を読んでいた——`solve(…, E1_psupHF[…])`。
    """

    @pytest.mark.parametrize("text,alias_free,want", [
        (r"E33 を E1^{(HF)}_{n\sigma}(\omega) について解いて",
         r"E33 を G^{(HF)}_{n\sigma}(\omega) について解いて",
         r"G^{(HF)}_{n\sigma}(\omega)"),
        (r"solve E33 for E1^{(HF)}_{n\sigma}(\omega)",
         r"solve E33 for G^{(HF)}_{n\sigma}(\omega)",
         r"G^{(HF)}_{n\sigma}(\omega)"),
    ])
    def test_variable_is_reread(self, text, alias_free, want):
        lang = "ja" if "解い" in text else "en"
        op, args = _parse_solve(text, lang, alias_free)
        assert (op, args["var"]) == ("solve_equation", want)


# ---------------------------------------------------------------------------
# 2026-08-21 (2nd ring): what it took to make the derivation COMPUTABLE.
# The first ring made the notation WRITABLE; measuring "can the CAS do it"
# reached a different layer, and every defect below was found by asking a step
# to run rather than by reading what it displayed.
# ---------------------------------------------------------------------------

class TestTheCasOwnSpellingIsReadBack:
    r"""結果に含まれる ``'sum(...)``・``kron_delta(...)`` は、入力と同じ物として読む。

    読み直せないと、添字の縮約が**研究者の綴りにだけ**効く。対角化の仮定を代入
    しても δ が残り、固有値が出ない。
    """

    @pytest.mark.parametrize("canonical,expect", [
        # 縮約: δ が m の和を潰す
        ("'sum(kron_delta(m,n)*A(n,m),m,1,3)", "A(n, n)"),
        # 名詞形の印は残る（`'sum` は「評価するな」という別の要求）
        ("'sum(x(i),i,a,b)", "'sum(x(i), i, a, b)"),
        ("sum(x(i),i,a,b)",  "sum(x(i), i, a, b)"),
    ])
    def test_round_trip(self, canonical, expect):
        assert _handler().to_backend_code(canonical).code == expect

    def test_nested_cas_sums_contract(self):
        """入れ子の `'sum` でも縮約する——代入が返すのはこの形である。"""
        got = _handler().to_backend_code(
            "'sum('sum(kron_delta(m,n)*A(n,m),m,nbSumLo2,nbSumHi2),n,nbSumLo1,nbSumHi1)"
        ).code
        assert got == "'sum(A(n, n), n, nbSumLo1, nbSumHi1)", got


class TestBracketedSubscriptOnAnUnderscoredName:
    r"""``c_hat[n, sigma](t)`` —— **この系自身が書く綴り**——が読み直せる。

    括弧が名前の最後の下線片に付いていたため、`d_hat[i](t)` は「d の添字
    (hat の添字 [i](t))」として読まれ、生成器が落ちていた。往復は tex1 側でしか
    測られていなかった。
    """

    @pytest.mark.parametrize("code", [
        "c_hat[n, sigma](t)", "d_hat[n](t)", "f[i](x)", "f[i, j](x)",
        "phi[n, sigma](vec(r))",
        "time_order(psi_hat_dagger[I, sigma](x))",
    ])
    def test_round_trips(self, code):
        assert _handler().to_backend_code(code).code == code

    @pytest.mark.parametrize("code,expect", [
        # 交換子・行列リテラルは触らない
        ("[A, B]", "((A)*(B) - (B)*(A))"),
        ("A [B, C]", "A * ((B)*(C) - (C)*(B))"),
    ])
    def test_bracket_meanings_kept(self, code, expect):
        assert _handler().to_backend_code(code).code == expect

    def test_a_binders_subscript_is_not_a_name(self):
        r"""``\sum_{p \sigma}\left[…\right]`` の括弧は**被和項**である。"""
        latex = (r"\Sigma^{(HF)}_{n\sigma} = \sum_{p \sigma''}"
                 r"\left[V_{n\sigma,p\sigma''} - \delta_{\sigma\sigma''}\right]"
                 r"\langle \hat n_{p\sigma''} \rangle")
        resp, _ = _run("ja", [latex])
        assert resp["status"] == "registered", resp


class TestDecoratedNameWithIndicesDisplays:
    r"""装飾つきの名前に添字が付いた形の表示（二重下付きは組版できない）。"""

    @pytest.mark.parametrize("name,expect", [
        ("psi_hat_sigma",        r"\hat{\psi}_{\sigma}"),
        ("psi_hat_dagger_sigma", r"\hat{\psi}^{\dagger}_{\sigma}"),
        ("c_hat_n_sigma",        r"\hat{c}_{n, \sigma}"),
        ("psi_hat",              r"\hat{\psi}"),
        ("H_0_hat",              r"\hat{H_0}"),      # 既存の読みは変えない
        ("k_B",                  "k_B"),
        ("x_ic1",                "x_ic1"),
    ])
    def test_restore(self, name, expect):
        from notebook.runtime.latex_converter import restore_decorations
        assert restore_decorations(name) == expect


class TestDeclarationCountsIndicesNotDecorations:
    r"""``\hat H_0`` の宣言は ``H_hat_0``——下線の数で「多重添字」と読まない。"""

    @pytest.mark.parametrize("lang,text,want", [
        ("ja", r"\hat H_0 をスカラーとする",        "H_hat_0"),
        ("en", r"declare \hat H_0 as a scalar",     "H_hat_0"),
        ("ja", r"\hat H を行列とする",              "H_hat"),
        ("ja", r"a_k_s をスカラーとする",           "a"),   # 本物の多重添字は剥がす
    ])
    def test_declared_name(self, lang, text, want):
        nb.start_engine(title="decl2", mode="standard", language=lang,
                        domains=["quantum_mechanics"])
        try:
            eng = nb.get_engine()
            sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                                domains=["quantum_mechanics"], primary_backend="maxima")
            asyncio.get_event_loop().run_until_complete(
                run_interpret_pipeline(
                    InterpretRequest(tokens=[InputToken(type="text", value=text)]),
                    eng, sup))
            assert want in eng.symbol_context.registry, list(eng.symbol_context.registry)
        finally:
            nb.stop_engine()


class TestSidedMultiplicationSolvesDyson:
    r"""非可換の積では**左右が別の式**で、それは研究者が決める。

    側を言えないまま両辺演算をすると、ラベルは左・値は右に掛かって式が壊れる。
    Maxima 自身は `(1-AB)^^-1 . (1-AB)` を 1 にできるので、足りなかったのは側だけ。
    """

    @pytest.mark.parametrize("lang,lines", [
        ("ja", [r"G を行列とする", r"\Sigma を行列とする", r"G^0 を行列とする",
                r"(1 - G^0 \Sigma) G = G^0",
                r"E1 の両辺に (1 - G^0 \Sigma)^{-1} を左から掛けて"]),
        ("en", ["declare G as a matrix", r"declare \Sigma as a matrix",
                "declare G^0 as a matrix",
                r"(1 - G^0 \Sigma) G = G^0",
                r"multiply both sides of E1 by (1 - G^0 \Sigma)^{-1} from the left"]),
    ])
    def test_formal_solution(self, lang, lines):
        resp, exprs = _run(lang, lines)
        assert resp["status"] == "completed", resp
        got = exprs[resp["expr_id"]].canonical.replace(" ", "")
        assert got == "G=(1-G_sup0.Sigma)^^(-1).G_sup0", got


class TestDiagonalisationIntroducesTheEigenvalue:
    r"""固有値は**計算の結果**として出る——書き写したのではない。"""

    @pytest.mark.parametrize("lang,lines", [
        ("ja", [r"\hat H_0 をスカラーとする",
                r"\hat H_0 = \sum_{n m} T_{n\sigma,m\sigma} \hat\rho_{n\sigma,m\sigma}",
                r"T_{n\sigma,m\sigma} = \varepsilon_{n\sigma} \delta_{nm}",
                r"E1 に E2 を代入して", r"E3 を簡約して"]),
        ("en", [r"declare \hat H_0 as a scalar",
                r"\hat H_0 = \sum_{n m} T_{n\sigma,m\sigma} \hat\rho_{n\sigma,m\sigma}",
                r"T_{n\sigma,m\sigma} = \varepsilon_{n\sigma} \delta_{nm}",
                "substitute E2 into E1", "simplify E3"]),
    ])
    def test_kronecker_collapses_the_sum(self, lang, lines):
        resp, exprs = _run(lang, lines)
        assert resp["status"] == "completed", resp
        got = exprs[resp["expr_id"]].canonical.replace(" ", "").replace("\n", "")
        assert got.startswith("'sum(epsilon(n,sigma)*rho_hat(n,sigma,n,sigma)"), got
