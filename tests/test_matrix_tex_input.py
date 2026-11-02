r"""行列の TeX 書法 —— **自分が印刷した綴りを読み戻せること**（TEX-MATRIX-INPUT）。

実測 2026-09-10。研究者が最初に書く行列の綴りは `\begin{pmatrix}` で、それは
**このシステム自身が表示に使っている綴り**でもある
（`BackendHandler.matrix_literal_to_pmatrix`、公開実例に28箇所）。9綴り中
**通るのは `[2, 1; 1, 2]` の1つだけ**だった:

    入力   [2, 1; 1, 2]
    表示   \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}
    再入力 \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}   → 「解釈できません」

しかも「読めない」のではなく**黙って崩れて**いた（`\\` が消えて2行が1行に融合、
`\pmatrix` は `\pm` の**前方一致**に食われて `+atrix`）。崩れたあと `&` が
パーサに届き、「ワープロや PDF から貼り付けた記号が混ざっていないか」という
**研究者が書いたものと無関係な断り**が返っていた。

直したのは3層:

1. `notebook/runtime/matrix_environment.py` —— 環境を変換器の入口で読む。
   落とす先は `[a, b; c, d]` と**同じ節点**（`matrix(...)`）。
2. `latex_converter._operators` —— コマンドの置換を `replace_tex_command`
   （**その名前が終わるところでだけ置き換える**、同じファイルに既にあった）へ。
3. `interpret_pipeline._nl_word_count` —— **環境の名前は英単語ではない**。
   `\begin`/`\end` はコマンドとして除かれていたのに `pmatrix` が2回数えられ、
   素の行列が「英文だ」と判定されて式の経路を失っていた。
"""
import asyncio

import pytest

import ui.notebook_bridge as nb
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.maxima_backend_handler import MaximaBackendHandler
from notebook.runtime.operation_validator import OperationValidator
from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline

#: 2×2 の同じ行列を、研究者が書きうるすべての綴りで。
#: `[2, 1; 1, 2]` が**既に通っていた対照**で、族の意味はそれが固定する。
MATRIX_SPELLINGS = [
    "[2, 1; 1, 2]",
    r"\begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}",
    r"\begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}",
    r"\begin{matrix} 2 & 1 \\ 1 & 2 \end{matrix}",
    r"\begin{Bmatrix} 2 & 1 \\ 1 & 2 \end{Bmatrix}",
    r"\begin{smallmatrix} 2 & 1 \\ 1 & 2 \end{smallmatrix}",
    r"\begin{array}{cc} 2 & 1 \\ 1 & 2 \end{array}",
    r"\pmatrix{2 & 1 \\ 1 & 2}",
    # 空白のない書き方（TeX エディタと、このシステム自身の公開実例の綴り）
    r"\begin{pmatrix}2&1\\1&2\end{pmatrix}",
    # 行送り指定つき（`\[` `\]` を消す前に読めていないと崩れる）
    r"\begin{pmatrix} 2 & 1 \\[2pt] 1 & 2 \end{pmatrix}",
    # 末尾の空行
    r"\begin{pmatrix} 2 & 1 \\ 1 & 2 \\ \end{pmatrix}",
    # 表示数式の区切りに包まれた形
    r"\[ \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix} \]",
]

WANT = "matrix([2, 1], [1, 2])"


def _handler():
    return MaximaBackendHandler(None)


class TestEverySpellingOfOneMatrixReachesTheSameNode:
    """1つの意味 × その綴り全部 —— 記法コーパスと同じ書き方。"""

    @pytest.mark.parametrize("spelling", MATRIX_SPELLINGS)
    def test_the_canonical_is_the_same_matrix(self, spelling):
        h = _handler()
        got = h.pre_process_canonical(h.latex_to_canonical(spelling))
        assert got == WANT, (spelling, got)

    @pytest.mark.parametrize("spelling", MATRIX_SPELLINGS)
    def test_it_reaches_the_same_node_in_julia_too(self, spelling):
        """落とす先はバックエンド非依存の `matrix(...)`。"""
        from notebook.runtime.julia_backend_handler import JuliaBackendHandler
        h = JuliaBackendHandler(None)
        got = h.normalize_matrix_literal(h.latex_to_canonical(spelling))
        assert got == WANT, (spelling, got)


class TestTheDelimitersSayWhichQuantityItIs:
    r"""縦棒は行列ではない —— `vmatrix` は行列式、`Vmatrix` はノルム。

    一律に行列と読むと、**スカラーの量が 2×2 の数表として登録される**。
    """

    def test_vertical_bars_are_a_determinant(self):
        h = _handler()
        got = h.pre_process_canonical(
            h.latex_to_canonical(r"\begin{vmatrix} 2 & 1 \\ 1 & 2 \end{vmatrix}"))
        assert got == f"determinant({WANT})", got

    def test_double_bars_are_not_read_as_a_matrix(self):
        """ノルムの操作がまだ無いので読まない（`docs/refusals.md`）。

        黙って行列として読むほうが高くつく——ノルムはスカラーである。
        """
        h = _handler()
        got = h.latex_to_canonical(r"\begin{Vmatrix} 2 & 1 \\ 1 & 2 \end{Vmatrix}")
        assert "matrix([2, 1]" not in got, got

    def test_the_norm_environment_is_refused_with_the_norm_reason(self):
        r"""**断りの理由が入力に当たっていること。**

        「ノルムか」は `ui/interpret_notation` が既に答えている問いだが、そちらは
        **縦棒の文字**を見ているので `\begin{Vmatrix}` は届かず、一般の
        「解釈できません」——理由も代わりの書き方も出ない断り——に落ちていた。
        それは TEX-MATRIX-INPUT が直そうとしている当のものである。
        """
        from notebook.runtime.matrix_environment import is_norm_environment_text
        assert is_norm_environment_text(
            r"N = \begin{Vmatrix} 2 & 1 \\ 1 & 2 \end{Vmatrix}")
        assert not is_norm_environment_text(
            r"N = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}")


class TestItDoesNotGuess:
    @pytest.mark.parametrize("spelling", [
        r"\begin{pmatrix} 1 & 2 \\ 3 \end{pmatrix}",          # 列数が揃わない
        r"\begin{pmatrix} 1 & \\ 3 & 4 \end{pmatrix}",        # 空のセル
        r"\begin{pmatrix} \end{pmatrix}",                     # 空
    ])
    def test_a_body_it_cannot_read_is_left_alone(self, spelling):
        """揃え方を決める根拠がどこにも無いものは、書き換えない。"""
        from notebook.runtime.matrix_environment import rewrite
        assert rewrite(spelling) == spelling, spelling

    def test_an_escaped_ampersand_is_not_a_column_separator(self):
        from notebook.runtime.matrix_environment import rewrite
        got = rewrite(r"\begin{pmatrix} a \& b \\ c \& d \end{pmatrix}")
        assert got == r"matrix([a \& b], [c \& d])", got

    @pytest.mark.parametrize("spelling,want", [
        (r"\begin{pmatrix} 1 & 2 & 3 \end{pmatrix}", "matrix([1, 2, 3])"),
        (r"\begin{pmatrix} 1 \\ 2 \\ 3 \end{pmatrix}",
         "matrix([1], [2], [3])"),
        (r"\begin{pmatrix} a \end{pmatrix}", "matrix([a])"),
    ])
    def test_a_single_row_is_still_a_matrix(self, spelling, want):
        """**1行の行列を `;` 区切りのリテラルで書くと、行列ではなくなる。**

        `[1, 2, 3]` には `;` が無いので `normalize_matrix_literal` が素通りし、
        リスト（Julia では交換子）として登録される。だから書き換え先は
        リテラルではなく `matrix(...)` を**直接**にしてある。
        """
        h = _handler()
        assert h.pre_process_canonical(h.latex_to_canonical(spelling)) == want

    def test_a_nested_matrix_is_folded_from_the_inside(self):
        from notebook.runtime.matrix_environment import rewrite
        got = rewrite(r"\begin{pmatrix} \begin{pmatrix} 1 & 0 \\ 0 & 1 "
                      r"\end{pmatrix} & 0 \\ 0 & 1 \end{pmatrix}")
        assert got == "matrix([matrix([1, 0], [0, 1]), 0], [0, 1])", got


class TestACommandIsReplacedWhereItsNameEnds:
    r"""`\pm` の置換が前方一致で、`\pmatrix` / `\pmb` / `\pmod` / `\cdots` を食っていた。

    同じファイルの `replace_tex_command` が既にこの問いに答えており（docstring に
    「``\i`` → ``%i`` が ``\int`` を書き換えた」と記録がある）、`_operators` だけが
    訊いていなかった。**一つの問いに二つの答え**である。
    """

    @pytest.mark.parametrize("latex,want", [
        (r"a \pm b", "a + b"),
        (r"a \mp b", "a - b"),
        (r"a \cdot b", "a . b"),
        (r"a \times b", "a ~ b"),
        (r"\nabla f", "nabla*f"),
    ])
    def test_the_operators_still_read(self, latex, want):
        assert _handler().latex_to_canonical(latex) == want, latex

    @pytest.mark.parametrize("latex,eaten", [
        (r"\pmb{x}", "+"),
        (r"\pmod{3}", "+"),
        (r"x_1 + \cdots + x_n", " . s"),
    ])
    def test_a_longer_command_is_not_eaten(self, latex, eaten):
        got = _handler().latex_to_canonical(latex)
        assert eaten not in got, (latex, got)

    def test_the_ellipsis_reads_like_its_siblings(self):
        r"""`\cdots` だけが ` . s` に化けていた——`\ldots` / `\dots` は素通り。

        ここが押さえるのは「省略記号が読める」ことではなく、**3つが同じ扱いに
        なる**ことである（省略記号そのものは計算できない。読まないコマンドが
        正準形に残るのはこの変換器の一般の振る舞いで、それはこの問いの外）。
        """
        h = _handler()
        forms = {h.latex_to_canonical(f"x_1 + \\{c} + x_n").replace(c, "…")
                 for c in ("cdots", "ldots", "dots")}
        assert len(forms) == 1, forms


class TestTheRoundTripCloses:
    r"""**自分が印刷した綴りを読み戻せること。**

    表示の綴りは手で書かず、`[2, 1; 1, 2]` を登録して**システムが返した LaTeX**
    をそのまま打ち直す——思いついた綴りではなく、研究者が画面から写す綴りで
    測るため（[[feedback-the-corpus-input-not-the-imagined-one]]）。
    """

    @staticmethod
    def _register(text, domain="linear_algebra", lang="ja"):
        async def go():
            nb.start_engine(title="mtx", mode="standard", language=lang,
                            domains=[domain])
            try:
                engine = nb.get_engine()
                sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                                    domains=[domain],
                                    primary_backend=engine.session.primary_backend)
                resp = await run_interpret_pipeline(
                    InterpretRequest(tokens=[InputToken(type="text", value=text)]),
                    engine, sup)
                regs = resp.get("registered_expressions") or []
                return resp.get("status"), (regs[-1] if regs else None)
            finally:
                nb.stop_engine()
        return asyncio.run(go())

    def test_what_the_system_prints_types_back_in(self):
        status, first = self._register("[2, 1; 1, 2]")
        assert status == "registered" and first, (status, first)
        printed = first["latex"]
        assert r"\begin{pmatrix}" in printed, printed

        status2, again = self._register(printed)
        assert status2 == "registered", (printed, status2)
        assert again["canonical"] == first["canonical"] == WANT, (first, again)

    @pytest.mark.parametrize("lang,text", [
        ("ja", r"A = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}"),
        ("en", r"A = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}"),
    ])
    def test_an_equation_whose_right_side_is_a_tex_matrix(self, lang, text):
        status, reg = self._register(text, lang=lang)
        assert status == "registered" and reg["canonical"] == WANT, (text, status, reg)

    @pytest.mark.parametrize("lang,text", [
        ("ja", r"\begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}"),
        ("en", r"\begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}"),
    ])
    def test_a_bare_matrix_is_a_formula_not_an_english_sentence(self, lang, text):
        """環境の名前（`pmatrix`）が英単語として2回数えられ、素の行列が
        「英文だ」と判定されて式の経路を失っていた。"""
        status, reg = self._register(text, lang=lang)
        assert status == "registered" and reg["canonical"] == WANT, (text, status, reg)


class TestTheRefusalFitsWhatWasWritten:
    r"""`\begin{Vmatrix}` は**ノルムの理由**で断られる（同じ文言を二重縦棒と共有）。"""

    @pytest.mark.parametrize("lang,needle", [
        ("ja", "ノルム"), ("en", "norm"),
    ])
    def test_the_error_names_the_norm_in_both_languages(self, lang, needle):
        status, err = TestTheRefusalFitsWhatWasWritten._ask(
            r"N = \begin{Vmatrix} 2 & 1 \\ 1 & 2 \end{Vmatrix}", lang)
        assert status == "failed", (status, err)
        assert needle in (err or ""), err
        assert "Could not interpret" not in (err or ""), err

    @pytest.mark.parametrize("lang", ["ja", "en"])
    def test_the_reason_names_the_spelling_that_was_written(self, lang):
        r"""**研究者が書いた綴りを名指すこと。**

        文言は二重縦棒と共有だが、共有した結果として
        `\begin{Vmatrix}` と書いた人に「二重縦棒は…」とだけ言うのでは、
        書いていない綴りの話になる（四十三問目の型）。だから文言は両方を挙げる。
        変異試験で最初これが抜けており、綴りを消しても落ちなかった。
        """
        _, err = self._ask(
            r"N = \begin{Vmatrix} 2 & 1 \\ 1 & 2 \end{Vmatrix}", lang)
        assert "Vmatrix" in (err or ""), err

    def test_the_double_bar_spelling_gets_the_same_reason(self):
        """綴りは2つ、断りは1つ。"""
        _, a = self._ask(r"N = \begin{Vmatrix} 2 & 1 \\ 1 & 2 \end{Vmatrix}", "ja")
        _, b = self._ask(r"N = \|v\|", "ja")
        assert a == b, (a, b)

    @staticmethod
    def _ask(text, lang):
        async def go():
            nb.start_engine(title="mtx", mode="standard", language=lang,
                            domains=["linear_algebra"])
            try:
                engine = nb.get_engine()
                sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                                    domains=["linear_algebra"],
                                    primary_backend=engine.session.primary_backend)
                resp = await run_interpret_pipeline(
                    InterpretRequest(tokens=[InputToken(type="text", value=text)]),
                    engine, sup)
                return resp.get("status"), resp.get("error")
            finally:
                nb.stop_engine()
        return asyncio.run(go())


class TestTheMatrixComputes:
    """登録できることと計算できることは別 —— 操作まで通す。"""

    @staticmethod
    def _run(lines, domain="linear_algebra", lang="ja"):
        async def go():
            nb.start_engine(title="mtx", mode="standard", language=lang,
                            domains=[domain])
            try:
                engine = nb.get_engine()
                sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                                    domains=[domain],
                                    primary_backend=engine.session.primary_backend)
                for line in lines:
                    resp = await run_interpret_pipeline(
                        InterpretRequest(tokens=[InputToken(type="text", value=line)]),
                        engine, sup)
                return resp, [e.canonical for e in engine.expressions.values()]
            finally:
                nb.stop_engine()
        return asyncio.run(go())

    def test_eigenvalues_of_a_tex_entered_matrix(self):
        resp, canons = self._run(
            [r"A = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}", "E1 の固有値を求めて"])
        assert resp.get("status") == "completed", resp.get("error")
        assert any("3" in c and "1" in c for c in canons[1:]), canons

    def test_the_determinant_spelling_computes_to_its_value(self):
        r"""`\begin{vmatrix} 2 & 1 \\ 1 & 2 \end{vmatrix}` は **3**。"""
        resp, canons = self._run(
            [r"D = \begin{vmatrix} 2 & 1 \\ 1 & 2 \end{vmatrix}", "E1 を簡約して"])
        assert resp.get("status") == "completed", resp.get("error")
        assert canons[-1] == "3", canons
