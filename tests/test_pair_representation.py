r"""PAIR-REP: ハイゼンベルク描像の演算子を「係数⊗演算子の対」として持つ（第1スライス）。

`A(t) = e^{iHt}Ae^{-iHt}` は新しい種類のオブジェクトではなく、QA が既に表せる
演算子の入れ子交換子の和である（docs/33 §5）。`[H,A] = λA` のとき——対角な二次形式に
対する消滅・生成演算子がそれ——その和は `e^{iλt}A` に**厳密に**閉じる。

ここで固定するのは3つ:

* **閉じる場合は厳密に出る。** 打ち切りではない。`c_m` の係数は `e^{-i ε_m t}`、
  `c†_m` は `e^{+i ε_m t}` で、符号が逆であることまで含めて構造から出る。
* **閉じない場合は理由付きで断る。** 非対角な二次形式は複数の演算子に広がり、
  係数は行列指数になる（第2スライス）。級数を打ち切って返すと、それは**短時間展開**
  であって摂動論ではない（docs/33 §6.2）——黙って返せば周波数空間で誤答になる。
* **係数は QA の外にある。** QA は c 数を分解するので（`exp(-im*eps*t)` が cos/sin に
  なり `exp(0)` の残骸が付く、実測）、係数は記述子側に、演算子は QA 側に置く。
"""
import pytest

import ui.notebook_bridge as nb
from notebook.runtime.i18n import t
from notebook.runtime.pair_representation import (
    REASON_NOT_CLOSED, REASON_NOT_CONCRETE, coefficient_product, coefficient_sum, is_unity,
    is_vanishing, matrix_entries, name_to_subscript_latex, pair_display_latex,
    pair_terms_of, parse_closure, parse_exponential_row, qa_to_input,
)


class TestReadingTheClosure:
    """CAS の返した閉包を読む部分——CAS を起動せずに固定できる。"""

    @pytest.mark.parametrize("raw,keys,rows", [
        ("c(m)|1:-1*eps(m)", ["c(m)"], [{1: "-1*eps(m)"}]),
        ("c(1);c(2)|2:-1*w;1:-1*w", ["c(1)", "c(2)"],
         [{2: "-1*w"}, {1: "-1*w"}]),
    ])
    def test_parses(self, raw, keys, rows):
        assert parse_closure(raw) == (keys, rows)

    @pytest.mark.parametrize("raw", ["NOTCLOSED", "", "   ", "no-bar"])
    def test_not_closed_reads_as_none(self, raw):
        assert parse_closure(raw) is None

    @pytest.mark.parametrize("raw,want", [
        # 対角: 1×1。QA の Float 係数は整数で書き戻す。
        ("c(m)|1:-1.0*eps(m)", ["-1*eps_m"]),
        # 非対角: 2×2、空きは 0。
        ("c(1);c(2)|2:-1*w;1:-1*w", ["0", "-1*w", "-1*w", "0"]),
    ])
    def test_matrix_entries(self, raw, want):
        assert matrix_entries(*parse_closure(raw)) == want

    @pytest.mark.parametrize("text,want", [
        ("eps(m)", "eps_m"), ("V(p, q)", "V_p_q"), ("w", "w"),
    ])
    def test_names_reach_sympy(self, text, want):
        from notebook.runtime.pair_representation import sympy_name
        assert sympy_name(text) == want

    def test_parses_the_exponential_row(self):
        assert parse_exponential_row(r"cos(t*w)|\cos{x}@@-I*sin(t*w)|-i\sin{x}") \
            == [("cos(t*w)", r"\cos{x}"), ("-I*sin(t*w)", r"-i\sin{x}")]

    @pytest.mark.parametrize("text,want", [
        ("c†(2) c(1) c(2)", "c'(2) c(1) c(2)"),
        ("c(2)", "c(2)"),
    ])
    def test_qa_printing_becomes_input_spelling(self, text, want):
        """対は次の括弧の被演算子——保存する綴りは読み直せなければならない。"""
        assert qa_to_input(text) == want

    @pytest.mark.parametrize("text,want", [
        ("eps(m)", "eps_{m}"),
        ("c†(m)", r"c^{\dagger}_{m}"),
        ("c'(m)", r"c^{\dagger}_{m}"),
    ])
    def test_qa_printing_to_user_subscripts(self, text, want):
        assert name_to_subscript_latex(text) == want

    def test_display_fences_a_summed_operator(self):
        """フェルミオンの交換子は `1 - 2c^†c` を返す——囲まないと係数が第1項に
        しか掛からないように読める。"""
        got = pair_display_latex([{"coefficient_latex": "e^{i x}",
                                   "operator_latex": r"1 - 2 c^{\dagger} c"}])
        assert got == r"e^{i x} \left(1 - 2 c^{\dagger} c\right)"

    def test_display_fences_a_summed_coefficient(self):
        """原子極限では係数が ``1 - e^{-iUt}`` になる——囲まないと項が増えて読める。"""
        got = pair_display_latex([
            {"coefficient_latex": "", "operator_latex": "c_{1}"},
            {"coefficient_latex": "1 - e^{- i U t}",
             "operator_latex": r"c^{\dagger}_{2} c_{1} c_{2}"},
        ])
        assert got == (r"c_{1} + \left(1 - e^{- i U t}\right) "
                       r"c^{\dagger}_{2} c_{1} c_{2}")


class TestAssemblingCoefficients:
    @pytest.mark.parametrize("a,b,want", [
        ("cos(t*w)", "-I*sin(u*w)", "(cos(t*w))*(-I*sin(u*w))"),
        ("1", "cos(t*w)", "cos(t*w)"),
        ("cos(t*w)", "1", "cos(t*w)"),
    ])
    def test_product(self, a, b, want):
        assert coefficient_product(a, b) == want

    def test_sum(self):
        assert coefficient_sum(["a", "b"]) == "(a) + (b)"
        assert coefficient_sum(["a"]) == "a"
        assert coefficient_sum([]) == "0"

    @pytest.mark.parametrize("latex,unity,zero", [
        ("1", True, False), ("0", False, True), ("c_{m}", False, False),
    ])
    def test_recognising_the_operator_part(self, latex, unity, zero):
        assert is_unity(latex) is unity
        assert is_vanishing(latex) is zero

    def test_a_non_pair_operand_is_a_coefficient_of_one(self):
        """`{c(t), c^†}` は物理で普通に書く——対でない側は係数 1。"""
        term, = pair_terms_of(None, latex=r"c^{\dagger}_{m}", canonical="c'(m)")
        assert (term["coefficient"], term["operator"]) == ("1", "c'(m)")


def _julia_session(lines=()):
    """演算子と対角ハミルトニアン（E1）／非対角（E2）を用意したセッション。"""
    nb.start_engine(title="pair_rep", mode="standard", language="ja",
                    domains=["condensed_matter"])
    eng = nb.get_engine()
    eng.run_operation("define_fermion_operator", {"name": "c"})
    eng.run_operation("define_scalar_parameter", {"name": "eps"})
    eng.run_operation("define_scalar_parameter", {"name": "w"})
    eng.register_expression(r"\sum_n eps_n c'(n) c(n)")
    eng.register_expression(r"w (c'(1) c(2) + c'(2) c(1))")
    eng.run_operation("define_scalar_parameter", {"name": "U"})
    eng.register_expression(r"U c'(1) c(1) c'(2) c(2)")
    return eng


class TestTheChainThatCloses:
    """閉じるなら**厳密に**出る——打ち切りではない。"""

    def test_diagonal_gives_a_single_phase(self):
        eng = _julia_session()
        try:
            got = {op: eng.heisenberg_operator(op, "E1", "t").latex
                   for op in ("c(m)", "c'(m)")}
        finally:
            nb.stop_engine()
        assert got["c(m)"] == "e^{- i eps_{m} t} c_{m}"
        assert got["c'(m)"] == r"e^{i eps_{m} t} c^{\dagger}_{m}"

    def test_non_diagonal_quadratic_mixes_two_operators(self):
        """2サイトの跳び移り: `c_1(t) = cos(wt)c_1 - i sin(wt)c_2`（教科書どおり）。"""
        eng = _julia_session()
        try:
            got = eng.heisenberg_operator("c(1)", "E2", "t")
        finally:
            nb.stop_engine()
        assert got.latex == (r"\cos{\left(t w \right)} c_{1} "
                             r"-  i \sin{\left(t w \right)} c_{2}")
        assert len(got.pair_representation["terms"]) == 2

    def test_the_atomic_limit_interaction_closes_too(self):
        r"""`U n_1 n_2` のもとで `c_1(t) = c_1 + (1-e^{-iUt}) c^†_2c_1c_2`。

        `c^†_2c_1c_2 = -c_1n_2` なので、これは
        `c_1[(1-n_2) + n_2 e^{-iUt}]` ——Hubbard 原子の厳密解である。
        """
        eng = _julia_session()
        try:
            got = eng.heisenberg_operator("c(1)", "E3", "t")
        finally:
            nb.stop_engine()
        assert got.latex == (r"c_{1} + \left(1 - e^{- i U t}\right) "
                             r"c^{\dagger}_{2} c_{1} c_{2}")

    def test_the_pair_keeps_its_provenance(self):
        eng = _julia_session()
        try:
            expr = eng.heisenberg_operator("c(m)", "E1", "t")
        finally:
            nb.stop_engine()
        desc = expr.pair_representation
        assert (desc["hamiltonian_ref"], desc["time_var"]) == ("E1", "t")
        assert desc["source_latex"] == "c_{m}"
        term, = desc["terms"]
        # 係数と演算子は分かれて入っている——それが対表現の要点。
        assert term["operator"] == "c(m)"
        assert term["coefficient"] == "exp(-I*eps_m*t)"

    def test_the_stored_operator_re_enters_as_input(self):
        """対が持つのは**研究者の綴り**。QA の印字（`c†(m)`）を戻すと 0 になる。"""
        eng = _julia_session()
        try:
            b = eng.heisenberg_operator("c'(m)", "E1", "u")
        finally:
            nb.stop_engine()
        term, = b.pair_representation["terms"]
        assert term["operator"] == "c'(m)"


class TestWhatIsRefused:
    def test_a_chain_over_index_families_is_refused_with_a_reason(self):
        r"""一般の2体項は「閉じた」ように見えるが、基底が族である。

        文字列一致で1つの演算子と数えていただけで、そのまま係数を組むと
        別々の行列要素が同じ記号に潰れる（実測 2026-08-22）。
        """
        eng = _julia_session()
        try:
            ham = eng.register_expression(
                r"\sum_{p q s n} V_{p,q,s,n} c'(p) c'(q) c(s) c(n)")
            with pytest.raises(ValueError) as exc:
                eng.heisenberg_operator("c(1)", ham.display_id, "t")
        finally:
            nb.stop_engine()
        assert t(REASON_NOT_CONCRETE, "ja", operator="c(1)",
                 hamiltonian=ham.display_id)[:24] in str(exc.value)

    def test_a_pair_is_refused_by_generic_operations(self):
        eng = _julia_session()
        try:
            a = eng.heisenberg_operator("c(m)", "E1", "t")
            with pytest.raises(ValueError) as exc:
                eng.run_operation("simplify", {"expr": a.display_id})
        finally:
            nb.stop_engine()
        # 生の CAS 内部文字列ではなく、次にできることを言う。
        assert "反交換関係" in str(exc.value)
        assert "Bracket expressions" not in str(exc.value)


class TestDifferentTimeAlgebra:
    r"""この表現が構造的に正しいことの見どころ（docs/34 §2）。

    `{c(t), c†(t')}` は **係数の積 × 等時刻代数**として出る。時間を添字に畳む案が
    踏んでいた `δ(t,t')` の罠は、この形では踏みようがない。
    """

    def test_anticommutator_is_a_scalar_of_the_time_difference(self):
        eng = _julia_session()
        try:
            a = eng.heisenberg_operator("c(m)", "E1", "t")
            b = eng.heisenberg_operator("c'(m)", "E1", "u")
            got = eng.pair_bracket("anticommutator", a.display_id, b.display_id)
            crossing = eng.fork_scalar_handoff(got.display_id)
        finally:
            nb.stop_engine()
        # 位相は**時間差**だけの関数になる（係数はスカラー側で簡約される）。
        assert got.latex == r"e^{i eps_{m} \left(- t + u\right)}"
        # 演算子が消えたので対ではなくスカラー——**行き止まりではない**:
        # SCALAR-HANDOFF でスカラー CAS へ渡せる。
        assert got.pair_representation is None
        assert crossing.target_backend == "maxima"
        assert "eps" in [d.name for d in crossing.declarations]

    def test_two_site_anticommutator_is_the_free_propagator(self):
        r"""`{c_1(t), c_1^†(u)} = cos(w(t-u))` —— 2サイトの自由伝播関数。

        4つの組み合わせのうち2つが生き残り、`cos·cos + sin·sin` がスカラー側で
        まとまる。多項どうしの括弧が意味を持つのはここである。
        """
        eng = _julia_session()
        try:
            a = eng.heisenberg_operator("c(1)", "E2", "t")
            b = eng.heisenberg_operator("c'(1)", "E2", "u")
            got = eng.pair_bracket("anticommutator", a.display_id, b.display_id)
        finally:
            nb.stop_engine()
        assert got.latex == r"\cos{\left(w \left(t - u\right) \right)}"
        assert got.pair_representation is None

    def test_commutator_keeps_the_operator_and_stays_a_pair(self):
        eng = _julia_session()
        try:
            a = eng.heisenberg_operator("c(m)", "E1", "t")
            b = eng.heisenberg_operator("c'(m)", "E1", "u")
            got = eng.pair_bracket("commutator", a.display_id, b.display_id)
        finally:
            nb.stop_engine()
        # 演算子部が和なので**括弧で囲われる**——係数が第1項にしか掛からない
        # ように読めるのを防ぐ。
        assert got.latex.endswith(r"\left(1 - 2 {c}_{m}^{\dagger} {c}_{m}\right)")
        assert got.pair_representation is not None


class TestNaturalLanguage:
    @pytest.mark.parametrize("lang,text", [
        ("ja", "c(m) を E1 のもとで時間発展させて"),
        ("ja", "c(m) を E1 で時刻 t まで時間発展させて"),
        ("en", "evolve c(m) under E1"),
        ("en", "evolve c(m) in time t under E1"),
    ])
    def test_both_languages_reach_the_same_operator(self, lang, text):
        import asyncio
        from notebook.runtime.llm_supervisor import LLMSupervisor
        from notebook.runtime.operation_validator import OperationValidator
        from ui.interpret_pipeline import (
            InterpretRequest, InputToken, run_interpret_pipeline)
        nb.start_engine(title="pair_rep_nl", mode="standard", language=lang,
                        domains=["condensed_matter"])
        eng = nb.get_engine()
        try:
            eng.run_operation("define_fermion_operator", {"name": "c"})
            eng.run_operation("define_scalar_parameter", {"name": "eps"})
            eng.register_expression(r"\sum_n eps_n c'(n) c(n)")
            sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                                domains=["condensed_matter"],
                                primary_backend="julia_full")
            r = asyncio.get_event_loop().run_until_complete(
                run_interpret_pipeline(
                    InterpretRequest(tokens=[InputToken(type="text", value=text)]),
                    eng, sup))
            assert r["status"] == "completed", r
            assert r["latex"] == "e^{- i eps_{m} t} c_{m}"
        finally:
            nb.stop_engine()
