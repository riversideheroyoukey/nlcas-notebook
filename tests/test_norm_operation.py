r"""ノルムの操作 —— **種類は研究者が書く。既定は置かない。**

起票 NORM-OPERATION は「記法は読めるのに落とす先の操作が無い」「ベクトルの
ユークリッドノルムは `sqrt(v . v)` と書けば**いま計算できる**ので配線だけ」と
書いていた。**測ると前提が2つとも崩れた**（2026-09-10）:

| 打った文 | 直す前 |
|---|---|
| `E1 と E2 の内積を求めて`（具体ベクトル） | **生の `Maxima evaluation error`** |
| `E1 のノルムを求めて` / `norm of E1` / `大きさ` / `magnitude` | 解釈できません |
| `\|v\|` / `\|\|v\|\|` | 解釈できません（道を名指さない汎用の断り） |
| `\begin{Vmatrix}…\end{Vmatrix}` | ノルムの断り（設計どおり、ただし文言が「操作が無い」） |

内積が壊れていた根因は **`expr_lookup` の H-1 等式復元**である。`solve_equation`
がその形を要る（外すと `y = x^2-1` を `x^2-1 = 0` として解く silent-wrong になる、
実測）ので復元自体は正しく、**受け取る側**が等式の左辺を名前として扱っていた:

    nb_dot([a = [1, 2, 3]_x, a = [1, 2, 3]_y, …], …)

**CAS の境界は起票より広い。** Maxima は `mat_norm(A, frobenius|1|inf)` を持ち、
具体行列でも記号行列でも値を返す。スペクトルノルムだけ `mat_norm` に無いので
定義どおり √(AᵀA の最大固有値) を組む——記号行列でも通る（実測）。

**ベクトルと行列の非対称は本物である。** ベクトルの `\|v\|` は2ノルム（大きさ）が
事実上ただ一つの読みだが、行列の `\|A\|` には競合する慣用が複数あり、既定を1つ
置くのは問題依存の数学の選択を焼き込むこと（CLAUDE.md「No hardcoded modeling
choices」）。だから行列は**種類を名指して断る**。
"""
import asyncio
import json
import pathlib

import pytest

import ui.notebook_bridge as nb
from notebook.runtime.llm_supervisor import LLMSupervisor, _norm_kind_written
from notebook.runtime.operation_validator import OperationValidator
from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline

DOM = ["classical_mechanics"]
_SCHEMA = json.loads((pathlib.Path(__file__).resolve().parents[1]
                      / "schemas" / "common_operations.schema.json"
                      ).read_text(encoding="utf-8"))


def _run(setup, lang, text):
    nb.start_engine(title="norm", mode="standard", language=lang, domains=DOM)
    try:
        eng = nb.get_engine()
        sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                            domains=DOM, primary_backend="maxima")
        loop = asyncio.get_event_loop()
        for line in setup:
            loop.run_until_complete(run_interpret_pipeline(
                InterpretRequest(tokens=[InputToken(type="text", value=line)]), eng, sup))
        r = loop.run_until_complete(run_interpret_pipeline(
            InterpretRequest(tokens=[InputToken(type="text", value=text)]), eng, sup))
        last = list(eng.expressions.values())[-1]
        return r, (last.canonical or "")
    finally:
        nb.stop_engine()


class TestTheVectorNormComputes:
    r"""`\|v\|` は2ノルム——**種類は要らない**。これは記法の慣用であって
    モデルの選択ではない（`|v|` も同じ量を指す）。"""

    @pytest.mark.parametrize("lang,text", [
        ("ja", "E1 のノルムを求めて"), ("ja", "E1 の大きさを求めて"),
        ("en", "compute the norm of E1"), ("en", "find the magnitude of E1"),
    ])
    def test_a_concrete_vector(self, lang, text):
        r, canon = _run(["v = [1, 2, 3]"], lang, text)
        assert r.get("status") == "completed", (text, r.get("error"))
        assert canon.replace(" ", "") == "sqrt(14)", (text, canon)

    @pytest.mark.parametrize("lang,text", [
        ("ja", "E1 のノルムを求めて"), ("en", "compute the norm of E1"),
    ])
    def test_a_symbolic_vector(self, lang, text):
        r, canon = _run(["v = [a, b, c]"], lang, text)
        assert r.get("status") == "completed", (text, r.get("error"))
        assert "sqrt(" in canon and "a^2" in canon, canon

    def test_a_declared_vector_field(self):
        """**操作の水準で測る。** 宣言済みのベクトル場は自然言語からは名前で
        渡せない（別名走査が `expr` に入れられるのは式番号だけ——起票
        OPERAND-BOUNDARY の残件で、ノルムに固有の話ではない）。ここで押さえるのは
        「場を渡せば成分から2ノルムが出る」ことである。"""
        nb.start_engine(title="norm", mode="standard", language="ja", domains=DOM)
        try:
            eng = nb.get_engine()
            sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                                domains=DOM, primary_backend="maxima")
            asyncio.get_event_loop().run_until_complete(run_interpret_pipeline(
                InterpretRequest(tokens=[InputToken(
                    type="text", value="A をベクトル場として定義する")]), eng, sup))
            _step, val = eng.run_operation("norm", {"expr": "A"})
            canon = str(getattr(val, "value", val))
        finally:
            nb.stop_engine()
        assert "A_x^2" in canon.replace(" ", ""), canon


class TestAVectorIsUsuallyWrittenInTeX:
    r"""**研究者が普通に書くベクトルは TeX である**（利用者の指摘、2026-09-10）。

    `\begin{pmatrix} 3 \\ 4 \end{pmatrix}` は `matrix([3],[4])` として届く
    ——`[3, 4]` という CAS 綴りは平文入力箱の形にすぎない。最初の実装はこれを
    「行列」として扱い、**列ベクトルに種類を要求していた**。

    **1行/1列なら選択はしていない**: Frobenius もスペクトルも2ノルムと一致する。
    種類を書けばそれに従う（列和は別の値なので、書けば違う答えが返る）。
    """

    @pytest.mark.parametrize("lang,setup,text,want", [
        ("ja", r"v = \begin{pmatrix} 3 \\ 4 \end{pmatrix}", "E1 のノルムを求めて", "5"),
        ("ja", r"v = \begin{pmatrix} 1 \\ 2 \\ 3 \end{pmatrix}", "E1 のノルムを求めて", "sqrt(14)"),
        ("ja", r"v = \begin{bmatrix} 3 \\ 4 \end{bmatrix}", "E1 の大きさを求めて", "5"),
        ("ja", r"v = \begin{pmatrix} 3 & 4 \end{pmatrix}", "E1 のノルムを求めて", "5"),
        ("en", r"v = \begin{pmatrix} 3 \\ 4 \end{pmatrix}", "compute the norm of E1", "5"),
    ])
    def test_a_tex_vector_needs_no_kind(self, lang, setup, text, want):
        r, canon = _run([setup], lang, text)
        assert r.get("status") == "completed", (text, r.get("error"))
        assert canon.replace(" ", "") == want, (text, canon)

    def test_a_written_kind_is_still_honoured(self):
        """1列でも、種類を書けばその値が返る（列和は2ノルムと違う）。"""
        r, canon = _run([r"v = \begin{pmatrix} 3 \\ 4 \end{pmatrix}"], "ja",
                        "E1 の最大絶対列和を求めて")
        assert r.get("status") == "completed", r.get("error")
        assert canon.strip() == "7", canon


class TestTheDeferredHalfIsNotForgotten:
    r"""**塞がっている side は、塞がりが取れた日に落ちる形で書く。**

    最初この族は `A のノルムを求めて` と自然言語で書き、落ちたので「別名走査は
    式番号しか `expr` に入れられない（起票 OPERAND-BOUNDARY の残件）」と註を付けて
    **操作の直接呼び出しに書き換えた**。それは「未解決の問題にテストを合わせた」
    ことであり、塞がりが取れても誰も確かめない状態を作る（利用者の指摘、
    2026-09-10）。**しかもその先は実際に壊れていた**——素の名前が定義式へ解決されて
    おらず、名前で渡した瞬間に「ベクトルにも行列にも見えません」と断っていた。

    そこで `xfail(strict=True)` にしてある。**通るようになった日にスイートが落ちる**
    ので、そのとき註と実装を見直すことになる——「あとで確かめる」を人の記憶ではなく
    機構に預ける唯一の書き方である。

    **その日が来た（2026-09-11、OPERAND-BOUNDARY 残①）。** 宣言済みの名前が式
    クラスの欄に入るようになり、この2件は XPASS してスイートを落とした——機構が
    設計どおりに働いたので、xfail を外して素のアサーションにしてある。族としての
    回帰は `tests/test_declared_name_as_operand.py`。
    """

    @pytest.mark.parametrize("lang,setup,text", [
        ("ja", "A をベクトル場として定義する", "A のノルムを求めて"),
        ("en", "Define A as a vector field", "compute the norm of A"),
    ])
    def test_a_name_reaches_the_norm_through_natural_language(self, lang, setup, text):
        r, _ = _run([setup], lang, text)
        assert r.get("status") == "completed", (text, r.get("error"))

    def test_the_operation_itself_already_takes_a_name(self):
        """**操作の側は先に用意してある**ので、塞がりが取れたら配線だけで通る
        （上の xfail はそれを確かめる番人である）。"""
        nb.start_engine(title="norm", mode="standard", language="ja", domains=DOM)
        try:
            eng = nb.get_engine()
            sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                                domains=DOM, primary_backend="maxima")
            asyncio.get_event_loop().run_until_complete(run_interpret_pipeline(
                InterpretRequest(tokens=[InputToken(
                    type="text", value="A をベクトル場として定義する")]), eng, sup))
            _step, val = eng.run_operation("norm", {"expr": "A"})
            canon = str(getattr(val, "value", val))
        finally:
            nb.stop_engine()
        assert "A_x^2" in canon.replace(" ", ""), canon


class TestABareNameResolvesToItsExpression:
    r"""**素の名前でも渡ってくる**（利用者の指摘、2026-09-10）。

    宣言済みのベクトル場は今でも名前で渡せるし、OPERAND-BOUNDARY の残件が解ければ
    H-1 の名前（`v`・`M`）も来る。名前は `Quantity.defining_expr_id` で自分の式を
    指しているので、そこから値を取る——無いと、名前で渡された瞬間に
    「ベクトルにも行列にも見えません」と断ることになる。
    """

    def _by_name(self, setup, args):
        nb.start_engine(title="norm", mode="standard", language="ja", domains=DOM)
        try:
            eng = nb.get_engine()
            sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                                domains=DOM, primary_backend="maxima")
            for line in setup:
                asyncio.get_event_loop().run_until_complete(run_interpret_pipeline(
                    InterpretRequest(tokens=[InputToken(type="text", value=line)]), eng, sup))
            _step, val = eng.run_operation("norm", args)
            return str(getattr(val, "value", val))
        finally:
            nb.stop_engine()

    @pytest.mark.parametrize("setup,args,want", [
        ([r"v = \begin{pmatrix} 3 \\ 4 \end{pmatrix}"], {"expr": "v"}, "5"),
        (["v = [1, 2, 3]"], {"expr": "v"}, "sqrt(14)"),
        ([r"M = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}"],
         {"expr": "M", "kind": "frobenius"}, "sqrt(30)"),
    ])
    def test_an_h1_name_resolves(self, setup, args, want):
        assert self._by_name(setup, args).replace(" ", "") == want

    def test_a_name_that_is_a_matrix_still_needs_its_kind(self):
        """名前で渡しても、種類を書かない行列は断る（**的外れな断りでなく**）。
        直す前は「ベクトルに `frobenius` は無い」という別の理由が返っていた。"""
        with pytest.raises(Exception) as exc:
            self._by_name([r"M = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}"],
                          {"expr": "M"})
        why = str(exc.value)
        assert "Frobenius" in why and "ベクトルに" not in why, why


class TestAMatrixNormNeedsItsKind:
    r"""**既定を置かない。** Frobenius・スペクトル・最大絶対列和・行和はどれも
    別の値で、どれかを既定にするのは研究者の選択を勝手にすることである。"""

    @pytest.mark.parametrize("lang,text", [
        ("ja", "E1 のノルムを求めて"), ("en", "compute the norm of E1"),
    ])
    def test_without_a_kind_it_refuses_and_names_the_kinds(self, lang, text):
        r, _ = _run(["M = [1, 2; 3, 4]"], lang, text)
        assert r.get("status") == "failed", r
        why = r.get("error") or ""
        for word in (("Frobenius", "スペクトル") if lang == "ja"
                     else ("Frobenius", "spectral")):
            assert word in why, (lang, why)
        assert "Maxima" not in why, why

    @pytest.mark.parametrize("lang,text,want", [
        ("ja", "E1 の Frobenius ノルムを求めて", "sqrt(30)"),
        ("ja", "E1 のフロベニウスノルムを求めて", "sqrt(30)"),
        ("en", "compute the Frobenius norm of E1", "sqrt(30)"),
    ])
    def test_frobenius(self, lang, text, want):
        r, canon = _run(["M = [1, 2; 3, 4]"], lang, text)
        assert r.get("status") == "completed", (text, r.get("error"))
        assert canon.replace(" ", "") == want, (text, canon)

    @pytest.mark.parametrize("lang,text", [
        ("ja", "E1 のスペクトルノルムを求めて"),
        ("en", "compute the spectral norm of E1"),
    ])
    def test_spectral(self, lang, text):
        r, canon = _run(["M = [1, 2; 3, 4]"], lang, text)
        assert r.get("status") == "completed", (text, r.get("error"))
        assert canon.replace(" ", "") == "sqrt(sqrt(221)+15)", (text, canon)

    @pytest.mark.parametrize("lang,text,want", [
        ("ja", "E1 の最大絶対列和を求めて", "6"),
        ("ja", "E1 の最大絶対行和を求めて", "7"),
        ("en", "compute the 1-norm of E1", "6"),
        ("en", "compute the infinity norm of E1", "7"),
    ])
    def test_the_column_and_row_sums(self, lang, text, want):
        r, canon = _run(["M = [1, -2; 3, 4]"], lang, text)
        assert r.get("status") == "completed", (text, r.get("error"))
        assert canon.strip() == want, (text, canon)

    def test_an_unknown_kind_is_refused_never_quietly_substituted(self):
        """**知らない種類を黙って既定に置き換えない。** 自然言語からは閉じた語彙
        しか来ないが、操作を直接呼べば任意の語が届く——そこで黙って Frobenius を
        返すと、研究者は**別の問いへの正しい答え**を受け取る（変異試験で発覚）。"""
        nb.start_engine(title="norm", mode="standard", language="ja", domains=DOM)
        try:
            eng = nb.get_engine()
            sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                                domains=DOM, primary_backend="maxima")
            asyncio.get_event_loop().run_until_complete(run_interpret_pipeline(
                InterpretRequest(tokens=[InputToken(type="text", value="M = [1, 2; 3, 4]")]),
                eng, sup))
            with pytest.raises(Exception) as exc:
                eng.run_operation("norm", {"expr": "E1", "kind": "p"})
            why = str(exc.value)
        finally:
            nb.stop_engine()
        assert "p" in why and "Frobenius" in why, why

    def test_a_symbolic_matrix_computes_too(self):
        r, canon = _run(["M = [a, b; c, d]"], "ja", "E1 の Frobenius ノルムを求めて")
        assert r.get("status") == "completed", r.get("error")
        assert "sqrt(" in canon and "d^2" in canon, canon


class TestTheKindIsReadFromWhatWasWritten:
    r"""**群の名前が正準の種類名そのもの**なので、綴り→種類の表は1枚しかない。"""

    @pytest.mark.parametrize("text,want", [
        ("E1 のフロベニウスノルムを求めて", "frobenius"),
        # **自分の断り文が名指す綴り**——日本語の文中でもローマ字で書く
        ("E1 の Frobenius ノルムを求めて", "frobenius"),
        ("E1 のスペクトルノルムを求めて", "spectral"),
        ("E1 の作用素ノルムを求めて", "spectral"),
        ("E1 の最大絶対列和を求めて", "max_col_sum"),
        ("E1 の∞ノルムを求めて", "max_row_sum"),
        # `無限大ノルム` —— 綴り空間の ratchet が「表に在るのに誰も打っていない」と
        # 捕まえた（2026-09-10）。表に足したら、打って何になるかも書く。
        ("E1 の無限大ノルムを求めて", "max_row_sum"),
        ("E1 の1ノルムを求めて", "max_col_sum"),
        ("E1 の2ノルムを求めて", "euclidean"),
        ("E1 の列和を求めて", "max_col_sum"),
        ("E1 の行和を求めて", "max_row_sum"),
        ("compute the maximum absolute column sum of E1", "max_col_sum"),
        ("compute the maximum absolute row sum of E1", "max_row_sum"),
        ("compute the euclidean norm of E1", "euclidean"),
        ("compute the vector norm of E1", None),
        ("compute the inf-norm of E1", "max_row_sum"),
        ("E1 のユークリッドノルムを求めて", "euclidean"),
        ("compute the Frobenius norm of E1", "frobenius"),
        ("compute the operator norm of E1", "spectral"),
        ("compute the 1-norm of E1", "max_col_sum"),
        ("compute the infinity norm of E1", "max_row_sum"),
    ])
    def test_it_reads_the_kind(self, text, want):
        assert _norm_kind_written(text) == want, text

    def test_a_roman_spelling_in_a_japanese_sentence_reads(self):
        """`E1 の Frobenius ノルムを求めて` —— 日本語側にローマ字の写しは**置かない**。
        読み手は `nlp_search` で全言語を試すので英語側の枝が読む（「guard は言語で
        絞らない」）。写しを置いた版は変異試験で生き残った＝何も足していなかった。"""
        from notebook.runtime.nl_patterns import nlp
        text = "E1 の Frobenius ノルムを求めて"
        # 日本語の枝だけでは読めない（写しを置いていないので）……
        assert nlp("norm_kind", "ja").search(text) is None, text
        # ……それでも読める。読み手が全言語を試すからである。
        assert _norm_kind_written(text) == "frobenius"

    @pytest.mark.parametrize("text", [
        "E1 のノルムを求めて", "compute the norm of E1", "E1 の大きさを求めて",
    ])
    def test_an_unqualified_norm_has_no_kind(self, text):
        """**書かれていないものを補わない。** 行列ならここで断りになる。"""
        assert _norm_kind_written(text) is None, text


class TestTheDotProductOfAConcreteVector:
    r"""起票が「いま計算できる」と書いていた道。**実測では生の CAS 文言だった。**

    根因は `expr_lookup` の H-1 等式復元で、`solve_equation` はその形を要るので
    復元は正しい——受け取る側が等式の左辺を**名前**として扱っていた
    （`nb_dot([a = [1, 2, 3]_x, …])`）。値は `strip_lhs` で取る。
    """

    @pytest.mark.parametrize("lang,text,want", [
        ("ja", "E1 と E2 の内積を求めて", "32"),
        ("en", "compute the dot product of E1 and E2", "32"),
    ])
    def test_the_dot_product_computes(self, lang, text, want):
        r, canon = _run(["a = [1, 2, 3]", "b = [4, 5, 6]"], lang, text)
        assert r.get("status") == "completed", (text, r.get("error"))
        assert canon.strip() == want, (text, canon)

    def test_the_cross_product_computes(self):
        r, canon = _run(["a = [1, 2, 3]", "b = [4, 5, 6]"], "ja", "E1 と E2 の外積を求めて")
        assert r.get("status") == "completed", r.get("error")
        assert canon.replace(" ", "") == "[-3,6,-3]", canon

    def test_solving_an_equation_still_sees_its_left_side(self):
        """**復元を切る側の直しは測って却下した。** 切ると `y = x^2-1` を
        `x^2-1 = 0` として解き、`x = ±1` を返す（silent-wrong）。"""
        r, _ = _run(["y = x^2 - 1"], "ja", "E1 を x について解いて")
        assert r.get("status") == "completed", r.get("error")
        made = [e.get("latex") or "" for e in (r.get("registered_expressions") or [])]
        assert any("sqrt" in m for m in made), made


class TestWhatIsRefusedNamesARoadThatWorks:
    def test_an_operand_that_is_neither_vector_nor_matrix(self):
        r, _ = _run(["x = a + b"], "ja", "E1 のノルムを求めて")
        assert r.get("status") == "failed", r
        assert "Maxima" not in (r.get("error") or ""), r.get("error")

    def test_an_unknown_operand_with_a_kind_says_so_not_something_else(self):
        """**断りの理由は入力に当たっていなければならない。** 相手が何かを決める
        前に種類の可否を訊くと、解決できない相手に種類を付けたとき
        「ベクトルに `frobenius` ノルムは無い」という**別の話**が返る
        （変異試験で発覚、2026-09-10）。"""
        r, _ = _run(["x = a + b"], "ja", "E1 の Frobenius ノルムを求めて")
        assert r.get("status") == "failed", r
        why = r.get("error") or ""
        assert "ベクトルにも行列にも見えません" in why, why
        assert "ベクトルに対して" not in why, why

    def test_a_flat_two_element_bracket_is_a_commutator(self):
        r"""`v = [3, 4]` は2成分ベクトルではなく**交換子**に降りる
        （CLAUDE.md の記法規則）。断り文はその事実と逃げ道を名指す。"""
        r, _ = _run(["v = [3, 4]"], "ja", "E1 のノルムを求めて")
        assert r.get("status") == "failed", r
        assert "交換子" in (r.get("error") or ""), r.get("error")

    def test_the_column_vector_escape_hatch_works(self):
        """断り文が名指す `[3; 4]`（列ベクトル＝行列）が本当に通ること。"""
        r, canon = _run(["v = [3; 4]"], "ja", "E1 の Frobenius ノルムを求めて")
        assert r.get("status") == "completed", r.get("error")
        assert canon.strip() == "5", canon

    # `test_the_notation_refusal_no_longer_claims_there_is_no_operation` は
    # ここに在ったが、**記法を配線した時点で前提が変わった**——`\begin{Vmatrix}`
    # はもう断らず計算する。同じ性質（断り文が事実であること）は
    # `TestANormInsideARelationIsAStatement` が、いま実際に断っている場合
    # （関係式の中にある）について引き継いでいる。


class TestTheNotationIsWiredToTheOperation:
    r"""**二重縦棒は「求めるもの」として読む**（2026-09-10、起票の残りを配線）。

    `\|v\|` / `||v||` / `\begin{Vmatrix}…\end{Vmatrix}` はどれも「中身のノルムを
    求めよ」である。操作は先に作ってあったので、足りなかったのは配線だけだった。
    """

    @pytest.mark.parametrize("lang,text", [
        ("ja", "\\|E1\\| を計算して"),
        ("ja", "||E1|| を計算して"),
        ("ja", "\\|v\\| を計算して"),
        ("ja", "\\|v\\|"),
        ("ja", "||v||"),
        ("en", "compute \\|E1\\|"),
        ("en", "\\|v\\|"),
    ])
    def test_the_double_bar_computes_a_vector_norm(self, lang, text):
        r, canon = _run([r"v = \begin{pmatrix} 1 \\ 2 \\ 3 \end{pmatrix}"], lang, text)
        assert r.get("status") == "completed", (text, r.get("error"))
        assert canon.replace(" ", "") == "sqrt(14)", (text, canon)

    @pytest.mark.parametrize("text", [
        "\\|\\begin{pmatrix} 3 \\\\ 4 \\end{pmatrix}\\|",
        "||\\begin{pmatrix} 3 \\\\ 4 \\end{pmatrix}||",
    ])
    def test_a_written_expression_inside_the_bars(self, text):
        """**中身は式番号・名前・書かれた式のどれでも来る。** 書かれた式は
        研究者の LaTeX なので、式の扉を通してから操作に渡す（通さないと
        `\begin{pmatrix}…` がそのまま降りて「ベクトルにも行列にも見えません」）。"""
        r, canon = _run([], "ja", text)
        assert r.get("status") == "completed", (text, r.get("error"))
        assert canon.strip() == "5", (text, canon)

    def test_the_norm_environment_computes(self):
        r"""`\begin{Vmatrix}` は縦棒の**文字**を持たないので、中身は環境の綴りを
        知っている側（`matrix_environment.norm_environment_operand`）から取る。"""
        r, canon = _run([], "ja", "\\begin{Vmatrix} 3 \\\\ 4 \\end{Vmatrix}")
        assert r.get("status") == "completed", r.get("error")
        assert canon.strip() == "5", canon

    def test_a_matrix_in_the_notation_still_needs_its_kind(self):
        r, _ = _run([r"M = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}"],
                    "ja", "\\|E1\\| を計算して")
        assert r.get("status") == "failed", r
        assert "Frobenius" in (r.get("error") or ""), r.get("error")

    def test_a_kind_written_beside_the_notation_is_read(self):
        r, canon = _run([r"M = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}"],
                        "ja", "\\|E1\\| の Frobenius ノルムを計算して")
        assert r.get("status") == "completed", r.get("error")
        assert canon.replace(" ", "") == "sqrt(30)", canon


class TestTheWiringRunsBeforeTheRefusal:
    r"""**順序が要件である。** `notation_candidates` は同じ綴りを「絶対値か行列式か」の
    ダイアログに落とさないために断っており、そちらが先に走ると配線は永遠に届かない。
    テキストで並べ替える変異は脆いので、**性質として固定する**。
    """

    def test_norm_notation_precedes_notation_candidates(self):
        from ui.interpret_pipeline import _PRELUDE_INTERCEPTORS
        names = [i.name for i in _PRELUDE_INTERCEPTORS]
        assert "norm_notation" in names, names
        assert "notation_candidates" in names, names
        assert names.index("norm_notation") < names.index("notation_candidates"), names


class TestANormInsideARelationIsAStatement:
    r"""**`=` が入っていたら要求ではない**——括弧の族で先に立てた規則と同じ
    （CLAUDE.md「A BRACKET FOLLOWED BY `=` IS A RELATION, NOT A REQUEST」）。
    `\|v\| = 5` は「大きさは5である」という主張、`E2 = \|v\|` は定義である。
    """

    @pytest.mark.parametrize("text", [
        "\\|v\\| = 5", "E2 = \\|v\\|", "||v|| = 5",
    ])
    def test_it_is_not_claimed_as_a_request(self, text):
        r, _ = _run([r"v = \begin{pmatrix} 1 \\ 2 \\ 3 \end{pmatrix}"], "ja", text)
        assert r.get("status") == "failed", (text, r)
        why = r.get("error") or ""
        assert "述べた式" in why or "関係式" in why, (text, why)

    def test_the_refusal_no_longer_claims_the_notation_is_unwired(self):
        """**断り文は事実でなければならない。** 配線したので「まだ配線されて
        いません」は偽になった——いま断っているのは*関係式の中にある*ことである。"""
        r, _ = _run([r"v = \begin{pmatrix} 1 \\ 2 \\ 3 \end{pmatrix}"], "ja",
                    "\\|v\\| = 5")
        why = r.get("error") or ""
        assert "配線されていません" not in why, why


class TestTheSingleBarIsUntouched:
    r"""`|x|` が絶対値か行列式かは**まだ決まっていない問い**で、答える相手は
    研究者である。ノルムの読みは `\|` と `||` だけを綴るので、ここには当たらない。"""

    @pytest.mark.parametrize("text", ["|x|", "|x| + |y|", "|E1|"])
    def test_a_single_bar_is_not_read_as_a_norm(self, text):
        from notebook.runtime.nl_patterns import nlp_search
        assert nlp_search("norm_notation", text) is None, text


class TestTheKindIsNotMathematics:
    def test_kind_is_an_identifier_field(self):
        """`max_col_sum` を数式として構文解析すると `\\max` の添字付き
        （最大／最小を取る束縛子）に読まれ、**ノルムと無関係な断り**が返っていた。"""
        from notebook.runtime.expr_ref_fields import IDENTIFIER_KEYS
        assert "kind" in IDENTIFIER_KEYS

    def test_the_schema_lists_no_default_kind(self):
        """**既定を置かない**ことは、スキーマの側でも見えていなければならない。"""
        spec = next(o for o in _SCHEMA["operations"] if o["operation"] == "norm")
        assert spec["required_fields"] == ["expr"], spec
        assert "kind" in spec["optional_fields"], spec
        assert "defaults" not in spec, spec
