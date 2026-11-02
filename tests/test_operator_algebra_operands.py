r"""非可換演算子代数を、自然言語から**実際に計算させる**ための境界。

実測 2026-08-22。QuantumAlgebra は `{c(1,↑), c†(1,↑)} = 1` を最初から正しく
返していた——届いていなかったのは指示の側で、詰まりは4箇所とも
「どこがオペランドの終わりか」を空白とカンマで近似していたことだった:

    c(1,u) と c'(1,u) の反交換子      → b="c'(1"            （生の ParseError）
    [c(1,u), c'(1,u)]                → a="c(1"             （黙って別の場所で切る）
    n_{1↑} と n_{1↓} の交換子         → [c†_{1↑}, c_{1↑}]   （別の演算子を成功として返す）
    normal order c†c c†c （英語）     → expr="c†_{1↑}"      （積が1トークンに切れる）

境界は括弧の深さ、助詞（と／の）、キーワード（of／and）——**研究者が書いたもの**
であって、空白ではない。
"""
import pytest

from notebook.runtime.llm_supervisor import RuleBasedIntentParser
from notebook.runtime.operation_validator import OperationValidator


def _parser(backend="julia_full"):
    return RuleBasedIntentParser(OperationValidator(), backend)


class TestProductOperandSurvives:
    """積で書かれたオペランドが、切られずに届くこと。"""

    _N1U = r"c^\dagger_{1\uparrow} c_{1\uparrow}"
    _N1D = r"c^\dagger_{1\downarrow} c_{1\downarrow}"

    @pytest.mark.parametrize("lang, text", [
        ("ja", f"{_N1U} と {_N1D} の交換子"),
        ("en", f"commutator of {_N1U} and {_N1D}"),
    ])
    def test_both_operands_are_whole(self, lang, text):
        f, p1 = _parser().parse(text, set(), lang, None)
        r = f or p1
        assert r is not None and r.operation == "commutator"
        assert r.args == {"a": self._N1U, "b": self._N1D}

    @pytest.mark.parametrize("lang, text, op", [
        ("ja", r"c_{1\uparrow} と c^\dagger_{1\uparrow} の反交換子", "anticommutator"),
        ("en", r"anticommutator of c_{1\uparrow} and c^\dagger_{1\uparrow}", "anticommutator"),
        ("ja", r"c_{1\uparrow} と c^\dagger_{1\uparrow} の交換関係", "commutator"),
    ])
    def test_kind_selects_the_operation(self, lang, text, op):
        f, p1 = _parser().parse(text, set(), lang, None)
        r = f or p1
        assert r is not None and r.operation == op

    @pytest.mark.parametrize("lang, text", [
        ("ja", r"E1 と c^\dagger_{1\uparrow} c_{1\uparrow} の交換子"),
        ("en", r"commutator of E1 and c^\dagger_{1\uparrow} c_{1\uparrow}"),
    ])
    def test_an_expression_reference_may_be_either_operand(self, lang, text):
        f, p1 = _parser().parse(text, {"E1"}, lang, None)
        r = f or p1
        assert r is not None and r.operation == "commutator"
        assert "E1" in (r.args["a"], r.args["b"])
        assert self._N1U in (r.args["a"], r.args["b"])

    def test_maxima_reads_the_same_phrase_as_a_poisson_bracket(self):
        """交換子の古典版は同じ書き方で、バックエンドで排他である。"""
        f, p1 = _parser("maxima").parse("q と p の交換子", set(), "ja", None)
        r = f or p1
        assert r is not None and r.operation == "poisson_bracket"
        assert r.args == {"a": "q", "b": "p"}


class TestUnaryOperandRunsToTheEnd:
    r"""英語の「動詞 式」でも、式は最後まで届くこと。

    日本語の「式 を…して」は `_FORMULA_OP_RE` の字種境界で切り出されるが、
    その境界は全 ASCII の英語では原理的に発火しない——同じ問いに片側だけ
    答えがある状態だった。
    """

    @pytest.mark.parametrize("text", [
        r"normal order c^\dagger_{1\uparrow} c_{1\uparrow} c^\dagger_{1\uparrow} c_{1\uparrow}",
        r"vacuum expectation of c^\dagger_{1\uparrow} c_{1\uparrow}",
        r"wick contraction of c^\dagger_{1\uparrow} c_{1\uparrow} c^\dagger_{1\downarrow} c_{1\downarrow}",
    ])
    def test_the_english_formula_boundary_exists(self, text):
        """境界は**操作の別名**——専用の正規表現は一般規則に畳んだ（2026-08-22）。"""
        from ui.interpret_pipeline import _en_verb_formula_re
        m = _en_verb_formula_re().match(text)
        assert m is not None, f"{text!r} の式が切り出せない"
        formula = m.group(2).strip()
        assert formula.startswith("c^\\dagger") or formula.startswith("c_")
        # The whole product, not its first factor.
        assert formula == text[m.start(2):].strip()

    def test_it_does_not_claim_an_expression_reference(self):
        """E-id を書いたときは既存の経路のままでよい（式の登録は不要）。"""
        from ui.interpret_pipeline import (_en_verb_formula_re,
                                           _formula_runs_to_the_end)
        m = _en_verb_formula_re().match("normal order E1")
        assert m is not None and m.group(2).strip() == "E1"
        assert not _formula_runs_to_the_end(m.group(2))


class TestArrowIsAValueAndSigmaIsAVariable:
    """スピン射影は値、スピン変数は変数——書かれたものから落ちてくること。"""

    @pytest.mark.parametrize("written, canon", [
        ("up", "↑"), ("uparrow", "↑"), ("dn", "↓"), ("down", "↓"), ("downarrow", "↓"),
    ])
    def test_every_projection_spelling_becomes_the_arrow(self, written, canon):
        from notebook.runtime.spin_label_canon import canon_spin_label
        assert canon_spin_label(written) == canon

    def test_the_spin_variable_stays_a_letter(self):
        from notebook.runtime.spin_label_canon import canon_spin_label
        assert canon_spin_label("sigma") == "s"

    @pytest.mark.parametrize("spelling", ["↑", r"\uparrow", r"\Uparrow"])
    def test_arrow_input_spellings_agree(self, spelling):
        from notebook.runtime.spin_label_canon import canon_spin_arrows
        assert canon_spin_arrows(spelling) == "uparrow"

    def test_the_canonical_name_is_ascii_so_the_parser_can_read_it(self):
        r"""正準形は再パースされる。矢印そのものを入れると、研究者自身の
        ハミルトニアンが「読めない文字」で登録に失敗した（実測）。"""
        from notebook.runtime.shared_parser import SharedParser
        from notebook.runtime.spin_label_canon import canon_spin_arrows
        name = canon_spin_arrows("↑")
        SharedParser().parse(f"c(1, {name})")      # raises if unreadable

    def test_a_projection_is_not_a_symbol_to_declare(self):
        r"""`\uparrow` は添字の VALUE であって物理パラメータではない。
        宣言を要求されると、そのセッションでは何も計算できない。"""
        from notebook.runtime.julia_undefined_mixin import JuliaUndefinedSymbolMixin as M
        assert "uparrow" in M._KNOWN_SYMBOL_NAMES
        assert "downarrow" in M._KNOWN_SYMBOL_NAMES


class TestLatexSubscriptCarriesSiteAndSpin:
    r"""`c_{1\uparrow}` は2つの添字である——バックスラッシュが区切りだから。"""

    @pytest.mark.parametrize("src, want", [
        (r"c_{1\uparrow}", r"c_{1,\uparrow}"),
        (r"c_{k\uparrow}", r"c_{k,\uparrow}"),
        (r"c_{i\sigma}",   r"c_{i,\sigma}"),
        (r"c_{k,\uparrow}", r"c_{k,\uparrow}"),
    ])
    def test_the_separator_is_inserted(self, src, want):
        from notebook.runtime.latex_converter import LatexConverter
        assert LatexConverter().to_canonical(src) == want


class TestPublishedInstructionMathFragments:
    r"""指示文の中の数式断片が、**釣り合った波括弧で**切り出されること。

    公開 TeX の指示カード（`_you_card_latex`）は、地の文と数式を混ぜた1行を
    断片ごとに `$…$` で囲む。その断片規則の波括弧が任意（`\{?…\}?`）だったため
    `c_{1\uparrow}` が `c_{1` で切れ、`$c_{1$$\uparrow$\}` という出力になって
    uplatex が `Missing } inserted` で止まっていた（実測 2026-08-22）。

    狭めすぎも同じくらい悪い: 素の実行を1文字に限ると `normal_order` が
    `normal_o` + `rder` に割れる（同日、再生成の差分で気づいた）。
    """

    @pytest.mark.parametrize("text, want", [
        (r"c_{1\uparrow} と c^\dagger_{1\uparrow} の反交換子",
         [r"c_{1\uparrow}", r"c^\dagger_{1\uparrow}"]),
        (r"p_{\theta} を theta の共役運動量とする", [r"p_{\theta}"]),
        (r"p_\theta", [r"p_\theta"]),
        ("normal_order E1", ["normal_order"]),
        ("radical_simplify E3", ["radical_simplify"]),
        ("S^z_1", ["S^z_1"]),
        ("x_i", ["x_i"]),
    ])
    def test_fragments(self, text, want):
        from ui.routes.export import _MATH_FRAG_RE
        assert [m.group(0) for m in _MATH_FRAG_RE.finditer(text)] == want

    @pytest.mark.parametrize("text", [
        r"c_{1\uparrow} と c^\dagger_{1\uparrow} の反交換子",
        r"c^\dagger_{1\uparrow} c_{1\uparrow} と c^\dagger_{1\downarrow} c_{1\downarrow} の交換子",
    ])
    def test_every_fragment_has_balanced_braces(self, text):
        from ui.routes.export import _MATH_FRAG_RE
        for m in _MATH_FRAG_RE.finditer(text):
            frag = m.group(0)
            assert frag.count("{") == frag.count("}"), frag


class TestAWrittenRangeIsNotDropped:
    r"""``\sum_{i=1}^{2}`` は**展開して**バックエンドへ渡す。

    QuantumAlgebra の ``∑`` は形式和で、範囲を置く場所が無い。生成器はそこへ
    範囲つきの和を素通しさせていたので、**書いた範囲が黙って消えた**（実測
    2026-08-22: ``[U \sum_{i=1}^{2} n_{i↑}n_{i↓}, c_{1↑}]`` が
    ``\sum_i[U c†_{1↓}c_{1↑}c_{1↓}]`` を成功として返す——i を含まない項に Σ_i が
    掛かっている）。「バックエンドが受け取らない」は理由にならない: 範囲が具体的な
    整数なら和は有限個の項で、展開は機械的かつ厳密である。
    """

    @staticmethod
    def _gen(src: str) -> str:
        from notebook.models.symbol_context import SymbolContext
        from notebook.runtime.julia_code_generator import JuliaCodeGenerator
        from notebook.runtime.julia_semantic_resolver import JuliaSemanticResolver
        from notebook.runtime.shared_parser import SharedParser
        ctx = SymbolContext()
        ir = JuliaSemanticResolver().resolve(SharedParser().parse(src), ctx)
        return JuliaCodeGenerator(context=ctx).generate(ir)

    def test_a_concrete_range_expands(self):
        # 囲われるのは `Σ` が1つの因子だったから（TestAnExpandedSumStaysOneFactor）。
        assert self._gen(r"\sum_{i=1}^2*f_i") == "(f(1) + f(2))"

    def test_a_product_expands_too(self):
        """Σ と Π の違いは項に applied する演算子だけ。"""
        assert self._gen(r"\prod_{i=1}^{3}*f_i") == "(f(1) * f(2) * f(3))"

    def test_no_range_stays_the_formal_sum(self):
        r"""範囲が書かれていないものは展開しない——QA の形式和が書かれたものと一致する。"""
        assert "_QA.∑(" in self._gen(r"\sum_n*f_n")

    def test_a_symbolic_range_is_not_invented(self):
        from notebook.models.symbol_context import SymbolContext
        from notebook.runtime.finite_sum_expansion import expand_finite_index_sum
        from notebook.runtime.julia_semantic_resolver import JuliaSemanticResolver
        from notebook.runtime.shared_parser import SharedParser
        ir = JuliaSemanticResolver().resolve(
            SharedParser().parse(r"\sum_{i=1}^{N}*f_i"), SymbolContext())
        assert expand_finite_index_sum(ir) is None

    def test_a_huge_range_is_left_to_the_formal_sum(self):
        from notebook.models.symbol_context import SymbolContext
        from notebook.runtime.finite_sum_expansion import (EXPANSION_CAP,
                                                           expand_finite_index_sum)
        from notebook.runtime.julia_semantic_resolver import JuliaSemanticResolver
        from notebook.runtime.shared_parser import SharedParser
        ir = JuliaSemanticResolver().resolve(
            SharedParser().parse(rf"\sum_{{i=1}}^{{{EXPANSION_CAP + 1}}}*f_i"),
            SymbolContext())
        assert expand_finite_index_sum(ir) is None


class TestASpinSumIsALegalIndex:
    r"""``\sum_\sigma`` は QA の1文字規則を満たす綴りに落ちること。

    束縛添字は「QuIndex に届かないから」と1文字規則を免除されていた——本体には
    真だが**束縛子には偽**で、実測 ``∑(:sigma, …)`` は
    ``ArgumentError("index names must be single character, got sigma.")`` を投げ、
    その生の CAS 文が研究者に返っていた（原則#1）。
    """

    def test_sigma_becomes_the_one_character_spelling(self):
        from notebook.models.symbol_context import SymbolContext
        from notebook.runtime.julia_code_generator import JuliaCodeGenerator
        from notebook.runtime.julia_semantic_resolver import JuliaSemanticResolver
        from notebook.runtime.shared_parser import SharedParser
        ctx = SymbolContext()
        ir = JuliaSemanticResolver().resolve(
            SharedParser().parse(r"\sum_sigma*f_{1,sigma}"), ctx)
        code = JuliaCodeGenerator(context=ctx).generate(ir)
        assert "_QA.∑(:s," in code and "sigma" not in code


class TestTheSigmaPrefixFollowsTheResult:
    r"""結果に残っていない添字の Σ を貼らないこと。"""

    def test_a_contracted_sum_gets_no_prefix(self):
        from notebook.runtime.backend_handler import BackendHandler
        latex = r"U {c}_{1{\downarrow}}^{\dagger} {c}_{1{\uparrow}} {c}_{1{\downarrow}}"
        assert not BackendHandler.result_carries_index(latex, ["i"])

    def test_a_surviving_sum_keeps_its_index(self):
        from notebook.runtime.backend_handler import BackendHandler
        latex = r"- U {c}_{i{\uparrow}}^{\dagger} {c}_{i{\downarrow}}^{\dagger}"
        assert BackendHandler.result_carries_index(latex, ["i"])

    def test_the_imaginary_unit_is_not_an_index(self):
        r"""``e^{- i U t}`` の `i` は添字の位置に無い。"""
        from notebook.runtime.backend_handler import BackendHandler
        latex = r"c_{1\uparrow} + \left(1 - e^{- i U t}\right) c_{1\downarrow}"
        assert not BackendHandler.result_carries_index(latex, ["i"])


class TestTheLatexBraceIsTheSameBracket:
    r"""``\{X, Y\}`` は ``{X, Y}`` と同じ反交換子。数式チップと手書きが出す綴り。"""

    @pytest.mark.parametrize("text", [
        r"\{c_{1\uparrow}, c^\dagger_{1\uparrow}\}",
        r"{c_{1\uparrow}, c^\dagger_{1\uparrow}}",
    ])
    def test_both_spellings_read(self, text):
        from notebook.runtime.operand_boundary import _bracket_operands
        got = _bracket_operands(text, "{")
        assert got is not None
        assert got[0].strip() == r"c_{1\uparrow}"

    @pytest.mark.parametrize("text", [r"\{A, B\}", "{A, B}", "[A, B]"])
    def test_the_start_check_agrees(self, text):
        from notebook.runtime.operand_boundary import starts_a_bracket_operand_form
        assert starts_a_bracket_operand_form(text)

    def test_an_ordinary_formula_does_not(self):
        from notebook.runtime.operand_boundary import starts_a_bracket_operand_form
        assert not starts_a_bracket_operand_form("x + y")


class TestAnExpandedSumStaysOneFactor:
    r"""展開した和は**1つの因子**として囲われること。

    `Σ` は1つの因子で、その展開は因子の**和**である。`Σ` を括弧で囲わないと
    決めた呼び手は、その判断を**ノード**に対してしていて文字列に対してではない。
    実測 2026-08-22（展開を入れた当のコミットの欠陥で、最初に見た事例では
    見えなかった）:

        U \sum_{i=1}^{2} f_i   →   U * f(1) + f(2)

    ——係数が**先頭の項にしか届かない**。公開シナリオの答えがたまたま無事だったのは
    そこの交換子が i=1 の項しか拾わないからで、**1件の緑は検査ではない**。
    """

    @staticmethod
    def _gen(src: str) -> str:
        from notebook.models.symbol_context import SymbolContext
        from notebook.runtime.julia_code_generator import JuliaCodeGenerator
        from notebook.runtime.julia_semantic_resolver import JuliaSemanticResolver
        from notebook.runtime.shared_parser import SharedParser
        ctx = SymbolContext()
        ir = JuliaSemanticResolver().resolve(SharedParser().parse(src), ctx)
        return JuliaCodeGenerator(context=ctx).generate(ir)

    @pytest.mark.parametrize("src, want", [
        (r"U*\sum_{i=1}^2*f_i",  "U * (f(1) + f(2))"),
        (r"-w*\sum_{i=1}^2*f_i", "-w * (f(1) + f(2))"),
        (r"\sum_{i=1}^2*f_i",    "(f(1) + f(2))"),
    ])
    def test_the_coefficient_reaches_every_term(self, src, want):
        assert self._gen(src) == want

    def test_a_single_term_needs_no_fence(self):
        assert self._gen(r"U*\sum_{i=1}^1*f_i") == "U * f(1)"


class TestASpinSumExpandsOverTheDeclaredValues:
    r"""``\sum_\sigma`` は**宣言された**射影の値集合に展開されること。

    展開が要るのは QA の限界である: 形式和どうしの交換子で、ダミー入れ替えで
    相殺する項を QA は畳まない（実測 2026-08-22、`[A,A]=0` は正しく出るので
    欠陥ではなく限界）。値集合は**研究者が書いた粒子**から来る——統計は分子の
    偶奇を決めるが大きさは決めないので、$2S$ は粒子ごとのデータである。
    """

    def test_the_electron_gives_the_two_arrows(self):
        from notebook.runtime.spin_value_set import index_labels, ladder_numerators
        assert ladder_numerators(1) == [-1, 1]          # m = ∓1/2、分子で持つ
        assert index_labels([-1, 1]) == ["↓", "↑"]

    def test_a_massless_boson_is_not_the_full_ladder(self):
        """光子はスピン1だが m=0 が無い——梯子を既定にしつつ、例外は明示する。"""
        import json
        import pathlib
        from notebook.runtime.spin_value_set import projection_numerators
        table = {p["key"]: p for p in json.loads(
            pathlib.Path("models/particles.json").read_text())["particles"]}
        assert projection_numerators(table["photon"]) == [-2, 2]      # m = ∓1
        assert projection_numerators(table["magnon"]) == [-2, 0, 2]   # m = -1,0,1
        assert projection_numerators(table["electron"]) == [-1, 1]

    def test_an_unknown_particle_invents_nothing(self):
        from notebook.runtime.spin_value_set import projection_numerators
        assert projection_numerators({"key": "x", "statistics": "fermion"}) is None

    def test_numerators_read_back_as_the_physics(self):
        from notebook.runtime.spin_value_set import format_numerator
        assert [format_numerator(n) for n in (-3, -1, 0, 1, 2)] == \
               ["-3/2", "-1/2", "0", "1/2", "1"]

    def test_a_bare_s_is_not_expanded(self):
        r"""1文字にスピンの意味を読み込むのは、この系が他所でしない推測。"""
        from notebook.models.ir import IndexSum, IndexSymbol, IndexedSymbol, Symbol
        from notebook.runtime.spin_value_set import expand_spin_index_sum
        node = IndexSum(
            body=IndexedSymbol(base=Symbol(name="c"), indices=[IndexSymbol(name="s")]),
            dummy_indices=[IndexSymbol(name="s")])
        assert expand_spin_index_sum(node, {"c": [-1, 1]}) is None

    def test_sigma_is_expanded_when_the_set_is_declared(self):
        from notebook.models.ir import Add, IndexSum, IndexSymbol, IndexedSymbol, Symbol
        from notebook.runtime.spin_value_set import expand_spin_index_sum
        node = IndexSum(
            body=IndexedSymbol(base=Symbol(name="c"), indices=[IndexSymbol(name="sigma")]),
            dummy_indices=[IndexSymbol(name="sigma")])
        out = expand_spin_index_sum(node, {"c": [-1, 1]})
        assert isinstance(out, Add) and len(out.args) == 2

    def test_nothing_declared_means_the_formal_sum_stands(self):
        from notebook.models.ir import IndexSum, IndexSymbol, IndexedSymbol, Symbol
        from notebook.runtime.spin_value_set import expand_spin_index_sum
        node = IndexSum(
            body=IndexedSymbol(base=Symbol(name="c"), indices=[IndexSymbol(name="sigma")]),
            dummy_indices=[IndexSymbol(name="sigma")])
        assert expand_spin_index_sum(node, {}) is None

    def test_a_written_range_is_still_the_other_rule(self):
        """範囲つきは `finite_sum_expansion` の領分——ここは触らない。"""
        from notebook.models.ir import (IndexSum, IndexSymbol, IndexedSymbol,
                                        Number, Symbol)
        from notebook.runtime.spin_value_set import expand_spin_index_sum
        node = IndexSum(
            body=IndexedSymbol(base=Symbol(name="c"), indices=[IndexSymbol(name="sigma")]),
            dummy_indices=[IndexSymbol(name="sigma")],
            lower=Number(value=1), upper=Number(value=2))
        assert expand_spin_index_sum(node, {"c": [-1, 1]}) is None


class TestTheValueSetSurvivesSaveAndLoad:
    r"""宣言から取った値集合は保存され、古いファイルは空で読めること。

    最初の実装は `SymbolContext` にフィールドを足しただけで**保存されていなかった**
    ——docstring に「追加キーなので古いセッションもそのまま読める」と書いたが、
    実測すると保存側に書き出しが無く、読み直したセッションは `\sum_\sigma` を
    展開しなくなっていた。書いた主張も測ってから言う。
    """

    def test_round_trip(self, tmp_path):
        from notebook.engine import NotebookEngine
        eng = NotebookEngine(title="rt")
        eng.symbol_context.spin_projections = {"c": [-1, 1]}
        path = tmp_path / "s.json"
        eng.save(str(path))
        assert NotebookEngine.load(str(path)).symbol_context.spin_projections == {"c": [-1, 1]}

    def test_a_file_without_the_key_loads_empty(self, tmp_path):
        """空は「まだ言われていない」——形式和のままで、誤りはしない。"""
        import json
        from notebook.engine import NotebookEngine
        eng = NotebookEngine(title="rt")
        eng.symbol_context.spin_projections = {"c": [-1, 1]}
        path = tmp_path / "s.json"
        eng.save(str(path))
        raw = json.loads(path.read_text())
        raw.pop("spin_projections", None)
        legacy = tmp_path / "old.json"
        legacy.write_text(json.dumps(raw, ensure_ascii=False))
        assert NotebookEngine.load(str(legacy)).symbol_context.spin_projections == {}


class TestNoteBoldReachesThePublishedTeX:
    r"""シナリオのメモの `**強調**` が、公開 TeX で太字になること。

    実測 2026-08-22: 公開 `.tex` 11本にアスタリスクが生のまま出ていた。組版は
    通る（エラーではない）が、PDF ではただの誤字に見える。シナリオ文書は
    GitHub 上では markdown として読まれるので `**` で書かれており、
    `【…】`（このシステム自身の規約）と**両方**を受ける。
    """

    @pytest.mark.parametrize("text, want", [
        ("これは**強調**である", r"これは\textbf{強調}である"),
        ("**先頭**から", r"\textbf{先頭}から"),
        ("**空**と**二つ**", r"\textbf{空}と\textbf{二つ}"),
        ("【従来】の書き方", r"\textbf{従来}の書き方"),
    ])
    def test_bold(self, text, want):
        from ui.routes.export import _note_inline
        assert _note_inline(text) == want

    @pytest.mark.parametrize("text", [
        "a * b は積",          # 単独の `*` は掛け算
        "* 箇条書き",           # 単独の `*` は箇条書きの印
        "**未閉じ の場合",       # 閉じていない
    ])
    def test_a_lone_asterisk_is_untouched(self, text):
        from ui.routes.export import _note_inline
        assert r"\textbf" not in _note_inline(text)


class TestEveryEnglishVerbFormulaKeepsItsFormula:
    r"""英語の「動詞＋式」で、式が**最後まで**届くこと（一般規則）。

    日本語は字種の変わり目で式を切り出すが、その境界は全 ASCII の英語では
    原理的に発火しない。境界を持っていたのは2つの操作だけで、**操作ごとに
    足していた**。実測 2026-08-22、利用者が通る経路で:

        simplify a x^2 + b x + c   →  x^2   completed
        factor x^2 - 1             →  x^2   completed

    式が空白を含むだけで別の式が計算され、成功として返っていた（同型の英語指示
    72 のうち 63）。境界は**操作の別名そのもの**——別名表は既にデータである。
    """

    def test_every_alias_carries_the_whole_formula(self):
        """データ駆動の網羅検査: 別名を1つ足しても、この検査が面倒を見る。"""
        from notebook.runtime.operation_validator import OperationValidator
        from ui.interpret_pipeline import _en_verb_formula_re, _nl_word_count
        rx, v = _en_verb_formula_re(), OperationValidator()
        formula, checked, bad = "a x^2 + b x + c", 0, []
        for op in v.known_operations:
            spec = v.get_spec(op) or {}
            if list(spec.get("required_fields") or []) != ["expr"]:
                continue
            for alias in (spec.get("aliases", {}) or {}).get("en") or []:
                alias = alias.strip()
                if len(alias) < 3 or "_" in alias:
                    continue
                checked += 1
                m = rx.match(f"{alias} {formula}")
                if not (m and m.group(2) == formula and _nl_word_count(m.group(2)) < 2):
                    bad.append((op, alias, m.group(2) if m else None))
        assert checked > 50, f"別名が少なすぎる（{checked}）——表を読めていない"
        assert not bad, bad

    @pytest.mark.parametrize("text", [
        "substitute x=2 into E1",          # 式の終わりが文末ではない
        "integrate E1 over x from 0 to 1",
        "E1 を展開して",                     # 日本語は既存の経路
        "solve E1 for x",
    ])
    def test_it_does_not_claim_what_it_cannot_bound(self, text):
        from ui.interpret_pipeline import _en_verb_formula_re
        assert _en_verb_formula_re().match(text) is None

    def test_an_english_sentence_is_not_a_formula(self):
        """語数のガード——宣言文を式として登録しない。"""
        from ui.interpret_pipeline import _en_verb_formula_re, _nl_word_count
        m = _en_verb_formula_re().match(
            "define c as an electron annihilation operator")
        assert m is not None                       # 別名 `define` には当たる
        assert _nl_word_count(m.group(2)) >= 2     # …が、式ではないので通さない

    def test_the_word_count_ignores_notation(self):
        from ui.interpret_pipeline import _nl_word_count
        assert _nl_word_count(r"c^\dagger_{1} c_{1}") == 0     # LaTeX コマンド
        assert _nl_word_count("diff(V,t) + diff(V,S,2)") == 0  # 関数呼び出し
        assert _nl_word_count("sin(x)^2 + cos(x)^2") == 0
        assert _nl_word_count("as an electron operator") >= 2


class TestTheRuleClaimsOnlyWhatItsPremiseCovers:
    r"""「式は文末まで」という前提が成り立つ文だけを取ること。

    最後の条件（式参照）は**実例の再生成が捕まえた**（2026-08-22）:
    `Invert E3, E4, E5, E6 for c(k,u), …` の `invert` は行列の逆の別名なので
    この規則が当たり、逆変換の指示ぜんぶを1つの「式」として登録して壊していた。
    利用者経路のテストは全部緑のままだった——**値でも文字列でもなく、実物の
    シナリオが落ちて初めて見えた**。
    """

    @staticmethod
    def _claims(text: str) -> bool:
        from ui.interpret_pipeline import (_en_verb_formula_re,
                                           _formula_runs_to_the_end)
        m = _en_verb_formula_re().match(text)
        return bool(m) and _formula_runs_to_the_end(m.group(2))

    @pytest.mark.parametrize("text", [
        "simplify a x^2 + b x + c",
        "factor x^2 - 1",
        "expand (x + 1)^2",
        "vacuum expectation of a b",
        r"normal order c^\dagger_{1} c_{1}",
    ])
    def test_claims_a_bare_formula(self, text):
        assert self._claims(text)

    @pytest.mark.parametrize("text", [
        # 式参照を並べた指示 —— 逆変換のインターセプタの領分
        "Invert E3, E4, E5, E6 for c(k,u), c(-k,d), c'(k,u), c'(-k,d)",
        "normal order E1",
        # 後続の節がある —— 式はそこで終わっている
        "integrate f over x from 0 to 1",
        "differentiate a x^2 with respect to x",
        # 英文
        "define c as an electron annihilation operator",
        # 宣言 —— `as` は述語を導くので、式はそこで終わっている
        r"Define k, m, \mu, \hbar as scalars",
        "Define A as a scalar",
    ])
    def test_declines_what_its_premise_does_not_cover(self, text):
        assert not self._claims(text)

    def test_the_declaration_veto_is_asked_not_re_answered(self):
        r"""「これは宣言か」は**バックエンドが既に持っている答え**である。

        日本語のインライン式経路は前から訊いていた。訊かずに済ませたとき、
        `Define k, m, \mu, \hbar as scalars` が式として登録され、**14本の
        シナリオ**が壊れた（実測 2026-08-22、実例の再生成が捕まえた）。
        """
        from notebook.runtime.maxima_backend_handler import MaximaBackendHandler
        h = MaximaBackendHandler.__new__(MaximaBackendHandler)
        decl = r"Define k, m as scalars"
        assert h.should_bypass_auto_register_formula("k, m as scalars", decl)
        assert not h.should_bypass_auto_register_formula(
            "a x^2 + b x + c", "simplify a x^2 + b x + c")
