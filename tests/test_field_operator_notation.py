r"""多電子系の教科書記法が、書いたとおりの意味で通ること（2026-08-20 実測）。

T=0 のグリーン関数を教科書どおりの綴りで18行打ち込んだところ、**意味が保たれた
ものは 0 件**だった。大半はエラーではなく「別の意味の式」として成功していた。
ここで固定するのは、そのうち根治した族:

* ``\hat\psi``（括弧なし装飾）が ``p_hatsi`` になっていた——装飾の引数が
  「次の1文字」で、ギリシャ文字は変換後に複数文字の識別子だったため。
* ``\hat\psi_\sigma(\mathbf r)``（第二量子化でいちばん普通に書かれる形）が、
  研究者の書いていない ``M_{i,j}(x)`` の話を理由に断られていた——パーサは
  ``Decorated`` を構造で持っているのに、名前の読み取り口がそれを知らず
  「連鎖添字が基底の形」に落ちていた。
* ``\varphi_{m\sigma'}`` が積 ``phi(m) * sigma'`` になっていた——添字トークンに
  プライムを許していなかったので区切りが働かず、``\varphi_{m\sigma}``（無印）
  だけが正しく2添字に読まれていた。
* プライム付きの名前が Maxima の**構文エラー**になり、しかもアダプタがそれを
  成功として報告していた（``simplify`` が ``_res_`` を返した）。
* ``v(\mathbf r - \mathbf r')`` が ``v(0)`` になっていた——ベクトル包みの外に
  付いたプライムが黙って捨てられていた。
"""
import ui.notebook_bridge as nb
import pytest


@pytest.fixture()
def eng():
    nb.start_engine(title="field_op_notation", mode="standard", language="ja",
                    domains=["quantum_mechanics"])
    try:
        yield nb.get_engine()
    finally:
        nb.stop_engine()


def _low(eng, latex: str) -> str:
    return eng._handler.to_backend_code(eng._handler.latex_to_canonical(latex)).code


class TestBareDecorationTakesTheWholeName:
    def test_hat_over_a_greek_letter(self, eng):
        h = eng._handler
        assert h.latex_to_canonical(r"\hat\psi") == "psi_hat"
        assert h.latex_to_canonical(r"\hat{\psi}") == "psi_hat"
        assert h.latex_to_canonical(r"\tilde\omega") == "omega_tilde"
        assert h.latex_to_canonical(r"\dot\theta") == "theta_dot"

    def test_the_argument_stops_at_the_token(self, eng):
        """``\\hat A B`` decorates A only — a space is still a boundary."""
        assert eng._handler.latex_to_canonical(r"\hat A B") == "A_hat*B"


class TestDecorationIsNotAnIndexSeparator:
    def test_the_field_operator_writes_and_displays(self, eng):
        h = eng._handler
        tex = r"\hat\psi_\sigma(\mathbf{r})"
        can = h.latex_to_canonical(tex)
        assert can == "psi_hat_sigma(vec(r))"
        ex = eng.register_expression(can, latex=tex)
        assert ex.latex == tex          # 書いたとおりに残る

    def test_a_chained_index_base_is_still_refused(self, eng):
        """装飾は名前、連鎖添字は読みが無い——線は動かしていない。"""
        from notebook.runtime.shared_parser import UnknownNotationError
        with pytest.raises(UnknownNotationError):
            eng._handler.parse_ir("A_i_j(x)")


class TestPrimedIndex:
    def test_a_primed_index_is_an_index_not_a_product(self, eng):
        assert _low(eng, r"\varphi_{m\sigma'}") == "phi(m, sigma\\')"
        assert _low(eng, r"\varphi_{m\sigma}") == "phi(m, sigma)"

    def test_the_prime_may_sit_on_the_first_index(self, eng):
        assert _low(eng, r"c_{k'\sigma}") == "c(k\\', sigma)"

    def test_a_primed_subscript_on_an_applied_form(self, eng):
        r"""``\psi^\dagger_{\sigma'}(x)`` —— 第二のスピン。無印は読めるのに
        プライム付きだけが「連鎖添字が基底の形」として断られていた。"""
        assert _low(eng, r"\psi_{\sigma'}(x)") == "psi_sigma\\'(x)"
        assert _low(eng, r"\psi^\dagger_{\sigma'}(x)") == "psi_dagger_sigma\\'(x)"
        assert _low(eng, r"\psi_\sigma(x)") == "psi_sigma(x)"

    def test_a_call_delimiter_is_still_not_a_factor(self, eng):
        """``\\left`` の除外は動かしていない（空白を入れると積に化ける）。"""
        assert _low(eng, r"G_{0}\left(k,\omega\right)") == "G_0(k, omega)"


class TestPrimedNameReachesTheCAS:
    def test_a_primed_name_is_one_atom(self, eng):
        assert _low(eng, r"x' + x") == "x\\' + x"
        r = eng._adapter.execute("string(ratsimp(x\\'^2 - x\\'*x\\'));")
        assert r.ok and r.value.strip() == "0"

    def test_simplify_returns_the_answer_not_the_scratch_name(self, eng):
        h = eng._handler
        ex = eng.register_expression(h.latex_to_canonical(r"x' + 2 x'"),
                                     latex=r"x' + 2 x'")
        step, val = eng.run_operation("simplify", {"expr": ex.display_id})
        assert step.status.value == "success"
        assert "_res_" not in val
        assert eng.expressions[step.output_refs[0]].latex == "3 x'"

    def test_two_points_do_not_collapse_into_one(self, eng):
        """``v(\\mathbf r - \\mathbf r')`` was reaching Maxima as ``v(0)``."""
        code = _low(eng, r"v(\mathbf{r}-\mathbf{r}')")
        assert code == "v(vec(r) - vec(r\\'))"
        assert eng._adapter.execute(f"string({code});").value != "v(0)"

    def test_a_prime_on_a_compound_is_refused_with_a_reason(self, eng):
        from notebook.runtime.shared_parser import UnknownNotationError
        with pytest.raises(UnknownNotationError) as ei:
            eng._handler.parse_ir("(a+b)'")
        assert ei.value.i18n_key == "parser.unsupported_notation.prime_on_compound"


class TestASyntaxErrorIsAnError:
    def test_the_adapter_does_not_report_a_syntax_error_as_success(self, eng):
        a = eng._adapter
        assert not a.execute("string(1 +);").ok
        assert not a.execute("string(x' + x);").ok      # 未エスケープのプライム
        assert a.execute("string(1+2);").value.strip() == "3"
        assert a.execute("declare(zz9, scalar)").ok


class TestChangeOfVariablesLowersItsBody:
    def test_the_integral_survives_the_substitution(self, eng):
        h = eng._handler
        tex = r"\int_{-\infty}^{\infty} e^{-x^2}\,dx"
        ex = eng.register_expression(h.latex_to_canonical(tex), latex=tex)
        step, val = eng.run_operation("change_integration_variables", {
            "expr": ex.display_id, "old_vars": "[x, y]", "new_vars": "[r, theta]",
            "transform": "x = r*cos(theta), y = r*sin(theta)"})
        assert "integrate(%e^(-x^2), x, -inf, inf)" in step.rendered_command
        assert "int_" not in val and "inf^inf" not in val


class TestTheAdjointIsOneQuantity:
    r"""同じ量の5つの綴りが5つの別物になっていた（2026-08-20 実測）:

        \psi^\dagger        → 'transpose(psi)          スカラー記号の転置
        \psi^\dagger(x)     → psi^dagger(x)            psi の dagger(x) 乗
        \psi^\dagger_\sigma → 'transpose(psi)(sigma)   転置を σ に適用
        \psi^{\dagger}(x)   → 'transpose(psi) * x      転置と x の積
                                                        （この綴りはシステム自身の表示が出す形）

    Maxima セッションには演算子代数が無い（F-2）ので、`\psi^\dagger` はそこでは
    **名前**——`\hat\psi` とまったく同じもの——であり、同じ `Decorated` 機構を通る。
    随伴が本当に意味を持つ場合、すなわち**行列・ベクトル・テンソルと宣言された基底**
    のときだけ随伴のままにする。判別材料を宣言に置くのは、隣の `^T` の分岐が既に
    そう読んでいたのと同じ形である。
    """

    def test_every_spelling_is_the_same_quantity(self, eng):
        for tex in (r"\psi^\dagger(x)", r"\psi^{\dagger}(x)",
                    r"\psi^{\dag}(x)", r"\psi^†(x)"):
            assert _low(eng, tex) == "psi_dagger(x)", tex

    def test_the_family_composes_with_index_and_argument(self, eng):
        assert _low(eng, r"\psi^\dagger") == "psi_dagger"
        assert _low(eng, r"\psi^\dagger_\sigma") == "psi_dagger(sigma)"
        assert _low(eng, r"\psi^\dagger_\sigma(x)") == "psi_dagger_sigma(x)"
        assert _low(eng, r"c^\dagger_k") == "c_dagger(k)"

    def test_a_declared_matrix_keeps_the_adjoint(self, eng):
        from notebook.models.quantity import Quantity, MathType
        eng.symbol_context.register(
            "A", quantity=Quantity(name="A", kind="scalar",
                                   math_type=MathType(algebra="matrix")))
        assert _low(eng, r"A^\dagger") == "ctranspose(A)"

    def test_the_display_returns_the_researchers_notation(self, eng):
        h = eng._handler
        tex = r"\psi^{\dagger}_\sigma(x) \psi_\sigma(x)"
        ex = eng.register_expression(h.latex_to_canonical(tex), latex=tex)
        step, _ = eng.run_operation("simplify", {"expr": ex.display_id})
        out = eng.expressions[step.output_refs[0]].latex
        assert r"\psi^{\dagger}_{\sigma}" in out, out
        assert "dagger}" not in out.replace(r"^{\dagger}", ""), out   # 添字に化けない


def test_the_three_decoration_tables_agree():
    """装飾の名前は3箇所に書かれている——パーサ・変換器・表示。ずれると、
    CAS が返した `psi_dagger` が「dagger という添字」に読み戻される（実測でそうなった）。
    片方だけ足せないように数え合わせる。"""
    from notebook.runtime.shared_parser import _DECORATION_NAMES
    from notebook.runtime.latex_converter import (
        _DECORATION_SUFFIXES, _SCRIPT_DECORATIONS)
    from notebook.runtime.backend_handler import _DECORATED_NAME_RE

    converter = {s[1:] for s, _ in _DECORATION_SUFFIXES} | set(_SCRIPT_DECORATIONS)
    display = set(_DECORATED_NAME_RE.pattern.split("(?P<dec>")[1]
                  .split(")")[0].split("|"))
    assert _DECORATION_NAMES == converter, (_DECORATION_NAMES ^ converter)
    assert _DECORATION_NAMES == display, (_DECORATION_NAMES ^ display)


class TestTheTwoDeltas:
    r"""添字なら Kronecker、引数リストなら Dirac —— 物理はそう書き分けている。

    引数形の読みが無かったので ``\delta(\mathbf r-\mathbf r')`` は Kronecker の
    分岐に落ち、``KroneckerDelta requires exactly 2 arguments, got 1`` という
    内部語を返していた。等時刻の反交換関係・完全性・運動方程式という、この分野の
    ごく普通の3行がそれで止まっていた（2026-08-20 実測）。

    Maxima は ``delta`` を持ち、``laplace(delta(t-3),t,s)`` は ``%e^-(3*s)`` を
    返す（実測）ので、降ろし先は行き止まりではない。``integrate`` はふるい分けを
    しないが、それは CAS の境界であってこの記法の問題ではない。
    """

    def test_a_subscript_is_the_kronecker_delta(self, eng):
        assert _low(eng, r"\delta_{ij}") == "kron_delta(i, j)"
        assert _low(eng, r"\delta_{\sigma\sigma'}") == "kron_delta(sigma, sigma\\')"

    def test_an_argument_list_is_the_dirac_delta(self, eng):
        assert _low(eng, r"\delta(t-t')") == "delta(t - t\\')"
        assert _low(eng, r"\delta(x)") == "delta(x)"
        assert _low(eng, r"\delta(\mathbf{r}-\mathbf{r}')") == "delta(vec(r) - vec(r\\'))"

    def test_two_arguments_stay_kronecker(self, eng):
        """``\\delta(i,j)`` は既存の綴り。引数が2つなら従来どおり Kronecker。"""
        assert _low(eng, r"\delta(i,j)") == "kron_delta(i, j)"

    def test_the_cas_can_use_it(self, eng):
        r"""行き止まりにしない —— Maxima の laplace はふるい分けを知っている。"""
        r = eng._adapter.execute("string(laplace(delta(t-3), t, s));")
        assert r.ok and r.value.replace(" ", "") == "%e^-(3*s)"


class TestASuperscriptLabelIsPartOfTheName:
    r"""摂動次数の上付きは指数ではない（GF-NOTATION、2026-08-20）。

    実測した壊れ方——いずれもエラーなし:

        G = G^0 + G^0 \Sigma G   →  Maxima: G*Sigma+1   （G^0 が 1）
        \Sigma^{(1)}             →  Sigma
        \Sigma^{(2)}(k,\omega)   →  Sigma^2 と (k,omega) の積
        f^{(3)}(x)               →  f^3 * x

    規則は3段（`notebook/runtime/script_label.py` に理由）。ここでは
    **指数として読む日常的な式を壊していないこと**も同時に固定する。
    """

    def test_the_dyson_equation_does_not_collapse(self, eng):
        assert _low(eng, r"G^0") == "G_sup0"
        code = _low(eng, r"G = G^0 + G^0 \Sigma G")
        assert "G_sup0" in code
        assert eng._adapter.execute(f"string({code.split('=')[1]});").value != "G*Sigma+1"

    def test_the_parenthesised_order_label(self, eng):
        assert _low(eng, r"\Sigma^{(1)}") == "Sigma_psup1"
        assert _low(eng, r"\Sigma^{(2)}") == "Sigma_psup2"
        assert _low(eng, r"\Sigma^{(2)}(k,\omega)") == "Sigma_psup2(k, omega)"

    def test_an_ordinary_exponent_is_untouched(self, eng):
        """`x^2` は本物の2乗。ここを推測で倒すと日常的な式が壊れる。"""
        assert _low(eng, r"x^2") == "x^2"
        assert _low(eng, r"G^2") == "G^2"
        assert _low(eng, r"e^{-x^2}") == "%e^(-x^2)"

    def test_a_label_is_order_independent(self, eng):
        r"""`c^\dagger_k` と `c_k^\dagger` は同じ演算子——ラベルはスロットではない。
        テンソルの上下添字は水平位置がスロットを意味する（`T^\mu{}_\nu`）ので、
        この規則には含めない（open_items MIXED-SCRIPT-NESTING）。"""
        h = eng._handler
        tex = r"G_\sigma^0 - G^0_\sigma"
        ex = eng.register_expression(h.latex_to_canonical(tex), latex=tex)
        _, val = eng.run_operation("simplify", {"expr": ex.display_id})
        assert val.strip() == "0"

    def test_the_display_returns_the_spelling_that_was_written(self, eng):
        """`G^{2}` と `G^{(2)}` は再入力すると別の読みになるので、括弧の有無は
        往復に必要な情報である。"""
        h = eng._handler
        for tex, want in ((r"\Sigma^{(1)} + \Sigma^{(2)}", (r"\Sigma^{(1)}", r"\Sigma^{(2)}")),
                          (r"G^0_\sigma(k) + G^0_\sigma(k)", (r"G^{0}_{\sigma}",))):
            ex = eng.register_expression(h.latex_to_canonical(tex), latex=tex)
            step, _ = eng.run_operation("simplify", {"expr": ex.display_id})
            out = eng.expressions[step.output_refs[0]].latex
            for w in want:
                assert w in out, (tex, out)
            assert "sup" not in out, out

    def test_what_is_displayed_types_back_in(self, eng):
        """表示された `G^{0}` を打ち直すと同じ CAS 対象に戻る。"""
        assert _low(eng, r"G^{0}") == _low(eng, r"G^0") == "G_sup0"
        assert _low(eng, r"\Sigma^{(2)}") == "Sigma_psup2"

    def test_a_binder_keeps_its_upper_LIMIT(self, eng):
        r"""`\int_0^1` の `^1` は範囲の上限であってラベルではない。除外を忘れると
        退化ラベルと読まれ、積分そのものが消える（実測: フルスイート25件が落ちた）。"""
        assert _low(eng, r"\int_0^1 x dx") == "integrate(x, x, 0, 1)"
        assert "sum(" in _low(eng, r"\sum_{i=1}^{3} i")

    def test_the_derivative_order_form_is_no_longer_a_power(self, eng):
        r"""`f^{(3)}(x)` は3階微分の標準記法。ラベルか微分階数かの区別は未決だが、
        **どちらにせよ指数ではない**——今までは `f^3 * x` だった。"""
        assert _low(eng, r"f^{(3)}(x)") == "f_psup3(x)"


class TestTheExpectationBracket:
    r"""``\langle X \rangle`` は期待値。読まれていなかったので ``langle``・``rangle``
    という**未定義の記号2つの積**として CAS に渡っていた（2026-08-20 実測、
    ``simplify`` は成功を返す）。添字付き ``\langle A \rangle_0`` では ``rangle``
    が添字付き関数 ``rangle(0)`` になっていた。

    落とし先は R-9 の bracket functional と同じ ``expectation(...)``。実測で
    行き止まりにならない（``expectation(X)+expectation(Y)`` も
    ``diff(expectation(X)*t, t)`` も通る）。状態は**第2引数**——どの状態で平均を
    取ったかを CAS 側で置換・比較できる。
    """

    def test_the_bracket_becomes_the_functional(self, eng):
        assert _low(eng, r"\langle A B \rangle") == "expectation(A * B)"
        assert _low(eng, r"\left\langle H \right\rangle") == "expectation(H)"

    def test_the_state_is_the_second_argument(self, eng):
        assert _low(eng, r"\langle H \rangle_0") == "expectation(H, 0)"

    def test_a_pair_is_not_an_average(self, eng):
        r"""``\langle a, b \rangle`` は内積・対の記法。トップレベルのカンマが
        判別材料で、最近接対 ``\sum_{\langle i,j \rangle}`` はこの手前の
        ``_subscript_constraint`` が既に食べている。"""
        assert "expectation" not in eng._handler.latex_to_canonical(r"\langle a, b \rangle")
        from notebook.runtime.shared_parser import UnknownNotationError
        # 隣接和は生成器で理由付きに断られる（読みは通り、書き換え先が無い）。
        with pytest.raises(UnknownNotationError) as ei:
            _low(eng, r"\sum_{\langle i,j \rangle} V_{i,j}")
        assert "adjacency" in ei.value.i18n_key

    def test_the_display_returns_the_brackets(self, eng):
        h = eng._handler
        for tex, want in ((r"\langle A B \rangle + \langle A B \rangle", r"\langle A B \rangle"),
                          (r"\langle H \rangle_0 + 1", r"\langle H \rangle_{0}")):
            ex = eng.register_expression(h.latex_to_canonical(tex), latex=tex)
            step, _ = eng.run_operation("simplify", {"expr": ex.display_id})
            out = eng.expressions[step.output_refs[0]].latex
            assert want in out, (tex, out)
            assert "expectation" not in out, out

    def test_the_greens_function_line_reads(self, eng):
        h = eng._handler
        tex = r"-i \langle T \psi(x,t) \psi^\dagger(y,u) \rangle"
        code = h.to_backend_code(h.latex_to_canonical(tex)).code
        assert code == "-i * expectation(T * psi(x, t) * psi_dagger(y, u))"

    def test_the_functional_is_not_reported_as_an_undefined_symbol(self, eng):
        """⚠ バナーが、研究者の書いていない `expectation` を「未定義」と告げていた
        （既存の `E[X]` も同じだった）。"""
        h = eng._handler
        ex = eng.register_expression(h.latex_to_canonical(r"\langle A B \rangle"),
                                     latex=r"\langle A B \rangle")
        assert "expectation" not in (ex.undefined_symbols or [])

    def test_defining_a_bracket_does_not_define_a_function_called_expectation(self, eng):
        h = eng._handler
        d = r"\langle H \rangle = E_0"
        eng.register_expression(h.latex_to_canonical(d), latex=d)
        assert "expectation" not in eng.symbol_context.registry
        assert "H_exp" in eng.symbol_context.registry

    def test_both_spellings_are_registered_so_the_definition_is_usable(self, eng):
        """(B) 両綴り登録。左辺は `H_exp` という名前を作り、右辺は
        `expectation(H)` を出すので、鍵が片方だけだと定義が使用地点で効かない。"""
        h = eng._handler
        d = r"\langle H \rangle = E_0"
        eng.register_expression(h.latex_to_canonical(d), latex=d)
        assert set(eng.lhs_alias_map) >= {r"\langle H \rangle", "expectation(H)"}
        u = r"\langle H \rangle + 1"
        ex = eng.register_expression(h.latex_to_canonical(u), latex=u)
        step, val = eng.run_operation("substitute",
                                      {"expr": ex.display_id, "lhs": "expectation(H)",
                                       "rhs": "E_0"})
        assert step.status.value == "success"
        out = eng.expressions[step.output_refs[0]].latex
        # ラベルは研究者の記法で出る（CAS の綴りが漏れない）
        assert r"\langle H \rangle" in out and "expectation" not in out, out


class TestScriptLetters:
    r"""``\mathcal{H}``（ハミルトニアン密度）・``\mathcal{L}``（ラグランジアン密度）は
    日常記法なのに ``mathcalH`` / ``mathcalL`` という名前に化けていた（2026-08-20）。
    括弧が後段で落ち、コマンド名が文字に貼り付く——素の ``\hat\psi`` → ``p_hatsi``
    と同じ形。装飾として読む: ``\mathcal{H}`` は ``H`` とは別の量である。"""

    def test_script_letters_are_decorations(self, eng):
        assert _low(eng, r"\mathcal{H}") == "H_cal"
        assert _low(eng, r"\mathcal{L}") == "L_cal"
        assert _low(eng, r"\mathscr{T}") == "T_scr"

    def test_they_display_back(self, eng):
        from notebook.runtime.latex_converter import restore_decorations
        assert restore_decorations("H_cal") == r"\mathcal{H}"
        assert restore_decorations("T_scr") == r"\mathscr{T}"


class TestTimeOrdering:
    r"""``\langle T[\psi(x,t)\psi^\dagger(y,u)]\rangle`` の ``T`` は演算子。

    実測（2026-08-20）: 素の ``T`` はただの因子、``T[\ldots]`` は生の pydantic
    ValidationError、``\mathcal{T}`` は ``mathcalT``、松原の ``T_\tau`` は
    ``T(tau)``＝τ で添字付けした T。

    **素の ``T`` は読まない。** 同じコーパスの ``\int d^3x\,\psi^* T(x) \psi`` と
    ``H = T + V`` では ``T`` が運動エネルギーで、推測すればそちらを壊す。
    """

    def test_the_unambiguous_spellings_are_read(self, eng):
        want = "expectation(time_order(psi(x, t) * psi_dagger(y, u)))"
        for tex in (r"\langle T[\psi(x,t) \psi^\dagger(y,u)] \rangle",
                    r"\langle T\{\psi(x,t) \psi^\dagger(y,u)\} \rangle",
                    r"\langle \mathcal{T} \psi(x,t) \psi^\dagger(y,u) \rangle"):
            assert _low(eng, tex) == want, tex

    def test_the_matsubara_ordering_carries_its_variable(self, eng):
        code = _low(eng, r"\langle T_\tau \psi(\tau) \psi^\dagger(\tau') \rangle")
        assert code == "expectation(time_order(psi(tau) * psi_dagger(tau\\'), tau))"

    def test_a_bare_T_keeps_its_other_meanings(self, eng):
        """運動エネルギーの ``T`` を壊さない——推測しない側の証拠。"""
        assert _low(eng, "H = T + V") == "H=T + V"
        assert _low(eng, r"T[i,j]") == "T(i, j)"       # 行列要素の読みも不変

    def test_a_bare_T_inside_the_bracket_is_warned_not_refused(self, eng):
        h = eng._handler
        tex = r"\langle T \psi(x,t) \psi^\dagger(y,u) \rangle"
        ex = eng.register_expression(h.latex_to_canonical(tex), latex=tex)
        assert ex.warnings and "T" in ex.warnings[0]
        # 断ってはいない: 式は登録され、因子として通っている
        assert "expectation(T *" in _low(eng, tex)

    def test_no_warning_when_the_spelling_decides(self, eng):
        h = eng._handler
        for tex in (r"\langle T[\psi(x,t) \psi^\dagger(y,u)] \rangle",
                    r"\langle A B \rangle", "H = T + V"):
            ex = eng.register_expression(h.latex_to_canonical(tex), latex=tex)
            assert not ex.warnings, (tex, ex.warnings)

    def test_the_display_returns_the_operator(self, eng):
        h = eng._handler
        tex = (r"\langle T[\psi(x,t) \psi^\dagger(y,u)] \rangle"
               r" + \langle T[\psi(x,t) \psi^\dagger(y,u)] \rangle")
        ex = eng.register_expression(h.latex_to_canonical(tex), latex=tex)
        step, _ = eng.run_operation("simplify", {"expr": ex.display_id})
        out = eng.expressions[step.output_refs[0]].latex
        assert r"\langle T\left[" in out, out
        assert "time" not in out, out

    def test_the_greens_function_definition_reads(self, eng):
        tex = r"G(x,t,y,u) = -i \langle T[\psi(x,t) \psi^\dagger(y,u)] \rangle"
        assert _low(eng, tex) == (
            "G(x, t, y, u)=-i * expectation(time_order(psi(x, t) * psi_dagger(y, u)))")


class TestAnAlgebraicIndex:
    r"""``c_{-k}``・``c_{k+q}``・``A_{i+1}`` —— 多体と格子の標準記法で、両 CAS が
    そのまま受ける（`c(-k)`・`c(k+q)`）。実測（2026-08-20）では2通りに壊れていた:

        c_{-k}     →  生の pydantic ValidationError（BCS の対の相手）
        c_{k+q}    →  **`c(k) + q`**（波括弧が落ち、残りが別の項に）
        x_{n+1}=2x_n → `x(n) + 1 = 2 x(n)` という**偽の等式**

    読みは Julia のリゾルバが最初から持っていた（「complex subscript は文字列
    IndexSymbol にする」）。共有側が持っていなかっただけなので、一箇所に移した。
    """

    def test_a_negated_or_shifted_index(self, eng):
        assert _low(eng, r"c_{-k}") == "c(-k)"
        assert _low(eng, r"c_{k+q}") == "c(k + q)"
        assert _low(eng, r"c_{k-q}") == "c(k - q)"
        assert _low(eng, r"A_{i+1}") == "A(i + 1)"
        assert _low(eng, r"A_{2i}") == "A(2 * i)"

    def test_the_recurrence_is_a_true_equation(self, eng):
        assert _low(eng, r"x_{n+1} = 2 x_n") == "x(n + 1)=2 * x(n)"

    def test_a_bracketed_compound(self, eng):
        assert _low(eng, r"A[i+1]") == "A(i + 1)"
        assert _low(eng, r"A[k^2]") == "A(k^2)"

    def test_the_ordinary_readings_are_unchanged(self, eng):
        assert _low(eng, r"A_{i,j}") == "A(i, j)"
        assert _low(eng, r"E_{rot}") == "E(rot)"
        assert _low(eng, r"A_{f(x)}") == "A_f(x)"     # 添字が適用形＝名前の一部
        assert _low(eng, r"a_i(k)") == "a_i(k)"


class TestAVectorValuedIndex:
    r"""``\sum_{\mathbf k'}`` —— 走らせる名前は包みの中の記号。

    実測: 括弧なしの ``\mathbf k`` は ``mathbf * k`` という**積**になり、和の中では
    ``mathbf`` が第二の束縛添字にまでなっていた（`sum(sum(v(j*k - j*k'), k'), j)`）。
    括弧ありは ``IndexSum.dummy_indices`` の型検査で生の pydantic エラー。
    """

    def test_the_bare_vector_command_takes_the_whole_name(self, eng):
        h = eng._handler
        assert h.latex_to_canonical(r"\mathbf k") == "vec(k)"
        assert h.latex_to_canonical(r"\vec v") == "vec(v)"

    def test_a_vector_runs_as_its_name(self, eng):
        assert _low(eng, r"\sum_{\mathbf{k}} f(\mathbf{k})") == "sum(f(vec(k)), k)"
        assert _low(eng, r"\sum_{\mathbf{k}'} f(\mathbf{k}')") == "sum(f(vec(k\\')), k\\')"

    def test_the_first_order_self_energy_reads(self, eng):
        h = eng._handler
        tex = (r"\Sigma^{(1)}(\mathbf{k}) = \sum_{\mathbf{k}'} "
               r"v(\mathbf{k}-\mathbf{k}') n_{\mathbf{k}'}")
        ex = eng.register_expression(h.latex_to_canonical(tex), latex=tex)
        assert ex.canonical  # 登録できる（生の pydantic で落ちない）
        code = _low(eng, tex)
        assert "sum(" in code and "Sigma_psup1" in code, code


class TestANounFormGluedToAName:
    r"""`i\hbar\frac{\partial}{\partial t}G` —— 運動方程式。

    CAS の名詞形 `'diff(...)` が前の名前に貼り付くと、クォートが**プライム**として
    読まれ、保留が消える: `i * hbar' * diff(G,t,1)`（ħ にプライムが付き、微分が
    評価形になる）。適用形のプライム `a'(1)` はクォートの後に名前が無いので無関係。
    """

    def test_the_equation_of_motion_keeps_hbar_and_the_hold(self, eng):
        code = _low(eng, r"i\hbar\frac{\partial}{\partial t} G(x,t)")
        assert code == "i * hbar * 'diff(G(x, t),t,1)", code

    def test_an_applied_prime_is_untouched(self, eng):
        assert _low(eng, r"a'(1) a(1)") == "a\\'(1) * a(1)"

    def test_a_space_before_the_noun_form_counts_too(self, eng):
        r"""`S^2 \frac{\partial^2 V}{\partial S^2}` —— 空白があっても同じこと。

        これは**実例の再生成が捕まえた回帰**（EXAMPLES-DRIFT の検査が働いた形）:
        プライムを1つも含まない Black–Scholes 方程式が、`S^2` にプライムが付いた
        形として読まれ、拒否に落ちた。それまでは「プライムを黙って落とす」既定が
        この綴りを通していた —— 落とす挙動は有害なだけでなく、ここでは保留を失う
        代わりに動いてもいた。
        """
        h = eng._handler
        can = h.latex_to_canonical(
            r"\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 "
            r"\frac{\partial^2 V}{\partial S^2} - r V = 0")
        assert "S^2*'diff(V,S,2)" in can, can
        assert h.to_backend_code(can).status == "lowered"


class TestTheSameStateBraKet:
    r"""``\langle \Phi_0 | A | \Phi_0 \rangle`` —— 教科書がグリーン関数の定義を
    書くときの綴り。

    断り文の前提「縦棒は区切りであって囲いではない」は真だが、**結論が今日ずれた**:
    ⟨…⟩ が読めるようになり、状態を第2引数に持つ ``expectation(A, state)`` が動く
    ようになったので、両側が同じ状態の形はその対象そのものである。外側2つが文字
    どおり同一であることが判断材料なので推測は入らない（CONSTRAINED-SUM と同じ形の
    訂正）。

    状態が2つある形（行列要素・内積）と片側だけの形は、名前が決まっていないので
    読まない —— open_items BRAKET-TWO-STATES。
    """

    def test_the_two_spellings_are_one_object(self, eng):
        h = eng._handler
        assert (h.latex_to_canonical(r"\langle \Phi_0 | H | \Phi_0 \rangle")
                == h.latex_to_canonical(r"\langle H \rangle_{\Phi_0}")
                == "expectation(H, Phi_0)")

    def test_two_states_and_one_sided_forms_are_still_refused(self, eng):
        import asyncio
        from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline
        from notebook.runtime.llm_supervisor import LLMSupervisor
        from notebook.runtime.operation_validator import OperationValidator
        sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                            domains=["quantum_mechanics"], primary_backend="maxima")
        for tex in (r"\langle \phi | H | \psi \rangle",
                    r"\langle \phi | \psi \rangle",
                    r"| \psi \rangle"):
            tok = InputToken(type="math", value=tex, latex=tex, canonical="")
            r = asyncio.get_event_loop().run_until_complete(
                run_interpret_pipeline(InterpretRequest(tokens=[tok]), eng, sup))
            assert r["status"] == "failed", (tex, r["status"])
            # ブラケット専用の文面であること（絶対値・行列式の話ではない）
            assert "ブラ" in r["error"], (tex, r["error"])
            # そして、通る書き方を名指ししていること
            assert "両側が同じ状態" in r["error"], tex

    def test_the_readable_form_is_not_claimed_by_the_abs_dialog(self, eng):
        """入口で外さないと、縦棒の対として絶対値／行列式ダイアログに落ちる
        （2026-08-17 にノルムで塞いだのと同じ穴）。"""
        import asyncio
        from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline
        from notebook.runtime.llm_supervisor import LLMSupervisor
        from notebook.runtime.operation_validator import OperationValidator
        sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                            domains=["quantum_mechanics"], primary_backend="maxima")
        eng.symbol_context.notation_usage["applied_subscript:G"] = "index"
        tex = (r"G_{\sigma\sigma'}(\mathbf{r}t,\mathbf{r}'t') = "
               r"-i\langle\Phi_0| T[\hat\psi_\sigma(\mathbf{r}t)"
               r"\hat\psi^\dagger_{\sigma'}(\mathbf{r}'t')]|\Phi_0\rangle")
        tok = InputToken(type="math", value=tex, latex=tex, canonical="")
        r = asyncio.get_event_loop().run_until_complete(
            run_interpret_pipeline(InterpretRequest(tokens=[tok]), eng, sup))
        assert r["status"] == "registered", r
        can = eng.expressions["E1"].canonical
        assert "expectation(time_order(" in can and "Phi_0)" in can, can
        assert "abs(" not in can and "det(" not in can, can


class TestTheVectorWrapperDoesNotReachTheDocument:
    r"""``vec`` はこの系がベクトルを包む綴りで、研究者の記法ではない。

    公開 TeX をレンダリングした利用者が見つけた（2026-08-20）:
    ``\varphi_{m,\sigma}(vec(r_x, r_y, r_z))`` —— **成分に開いたあとは包みが要らない**。
    3成分の関数と書けば足りるし、包んだままだと `vec` という語が本文の数式に居座って
    成分も読みにくい（原則#1: 通常の UI に CAS 構文を出さない）。
    """

    def _disp(self, eng, canonical):
        tex = eng._adapter.execute(f"tex1({canonical});").value
        return eng._handler.postprocess_result_latex(
            canonical, tex, eng.symbol_context)

    def test_a_single_argument_returns_the_bold_name(self, eng):
        assert self._disp(eng, "f(vec(r))") == r"f\left(\mathbf{r}\right)"
        assert self._disp(eng, "vec(r)") == r"\mathbf{r}"

    def test_expanded_components_drop_the_wrapper(self, eng):
        assert (self._disp(eng, "f(vec(r_x,r_y,r_z))")
                == r"f\left(r_{x}, r_{y}, r_{z}\right)")

    def test_two_points_stay_two_points(self, eng):
        out = self._disp(eng, "v(vec(r) - vec(r\\'))")
        assert out == r"v\left(\mathbf{r}-\mathbf{r'}\right)", out


class TestCommaAndSpaceMix:
    r"""「添字はコンマ**または**空白で区切る」——両方が混ざる形が読まれていなかった。

    実測 2026-08-20（シナリオを書いていて発見）: 対になったラベルを並べる標準的な
    書き方 ``V_{n\sigma,m\sigma'}`` で、区画の中の空白が読まれず**積**になっていた。

        V_{n\sigma}            → V(n, sigma)            正しい
        V_{n\sigma,m\sigma'}   → V(n*sigma, m*sigma')   積
        \sum_{n\sigma,m\sigma'} → 和の**添字**まで積

    区画ごとに同じ規則を当てる。規則が一つなら、綴りが混ざっても答えは一つになる。
    """

    def test_paired_labels_in_a_comma_list(self, eng):
        assert _low(eng, r"V_{n\sigma}") == "V(n, sigma)"
        assert _low(eng, r"V_{n\sigma,m\sigma}") == "V(n, sigma, m, sigma)"
        assert _low(eng, r"A_{i j,k l}") == "A(i, j, k, l)"

    def test_the_sum_binds_each_label(self, eng):
        code = _low(eng, r"\sum_{n\sigma,m\sigma} V_{n\sigma,m\sigma}")
        assert code.count("sum(") == 4, code

    def test_the_existing_readings_are_unchanged(self, eng):
        assert _low(eng, r"A_{i,j}") == "A(i, j)"
        assert _low(eng, r"E_{rot}") == "E(rot)"
        assert _low(eng, r"c_{k,up}") == "c(k, up)"
        assert _low(eng, r"\delta_{ij}") == "kron_delta(i, j)"
        assert _low(eng, r"c_{k+q}") == "c(k + q)"


class TestAMatrixPowerIsNotElementWise:
    r"""宣言された行列の ``A^{-1}`` は**逆行列**。

    Maxima の ``^`` は成分ごとの冪である（実測: ``matrix([1,2],[3,4])^-1`` は
    ``matrix([1,1/2],[1/3,1/4])``——各成分の逆数、エラー無し）。非可換冪 ``^^`` が
    逆行列で、記号のまま保たれ ``A^^-1 . A`` は 1 に簡約される。``invert(A)`` は
    記号では失敗するので使えない。

    判別材料は**宣言**で、隣の ``^T`` / ``^\dagger`` の分岐と同じ —— 同じ問いの
    3つ目だけが未回答だった（2026-08-21、Dyson 方程式の一般解を書いていて発覚）。
    """

    def _matrix(self, eng, name="A"):
        from notebook.models.quantity import Quantity, MathType
        eng.symbol_context.register(
            name, quantity=Quantity(name=name, kind="scalar",
                                    math_type=MathType(algebra="matrix")))

    def test_a_declared_matrix_uses_the_non_commutative_power(self, eng):
        self._matrix(eng)
        assert _low(eng, r"A^{-1}") == "A^^-1"
        assert _low(eng, r"A^2") == "A^^2"

    def test_an_undeclared_base_is_untouched(self, eng):
        assert _low(eng, r"B^{-1}") == "B^-1"
        assert _low(eng, r"x^2") == "x^2"

    def test_a_compound_base_keeps_its_grouping(self, eng):
        r"""``^`` は右結合なので、複合基底は括弧が要る —— 素で出すと
        ``(1-G^0\Sigma)^{-1}`` が ``1 - G_sup0 . Sigma^^-1`` になり、逆行列が
        \Sigma に付いて括りも消えた（実測。スカラー側の `_generate_pow` が
        既に持っていた規則を、この分岐が写していなかった）。"""
        self._matrix(eng, "Sigma")
        code = _low(eng, r"(1 - G^0 \Sigma)^{-1} G^0")
        assert code == "(1 - G_sup0 . Sigma)^^-1 . G_sup0", code

    def test_the_display_returns_the_researchers_exponent(self, eng):
        r"""Maxima の tex1 は非可換冪を ``A^{\langle 2\rangle}`` と綴る。"""
        a = eng._adapter
        for canonical, want in (("A^^-1", "A^{-1}"), ("A^^2", "A^{2}")):
            tex = a.execute(f"tex1({canonical});").value
            out = eng._handler.postprocess_result_latex(
                canonical, tex, eng.symbol_context)
            assert out == want, (canonical, out)

    def test_the_dyson_general_solution(self, eng):
        """一般解に Hartree を代入する筋（シナリオ S-ak の末尾）。"""
        h = eng._handler
        self._matrix(eng, "G"); self._matrix(eng, "Sigma")
        gen = eng.register_expression(
            h.latex_to_canonical(r"G = (1 - G^0 \Sigma)^{-1} G^0"),
            latex=r"G = (1 - G^0 \Sigma)^{-1} G^0")
        step, _ = eng.run_operation("substitute", {
            "expr": gen.display_id, "lhs": "Sigma", "rhs": "Sigma_psup1"})
        out = eng.expressions[step.output_refs[0]].latex
        assert r"\Sigma^{(1)}" in out and "^{-1}" in out, out
