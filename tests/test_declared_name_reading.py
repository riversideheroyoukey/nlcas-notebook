r"""宣言が読む名前は、式が読む名前と同じでなければならない（DECL-NAME-READING）。

宣言はこの系のほとんどの読みの**判別材料**である —— `^T`・`^\dagger`・`^{-1}` が
転置／随伴／逆行列になるか、添字が index か名前か、記号が行列か。その名前を宣言だけが
別の規則で読んでいた。実測 2026-08-21（シナリオ S-ak で Dyson 方程式の一般解を書くのに
`G^0 を行列とする` が要って発覚）:

    G を行列とする              → `G`     ✓
    G_0 を行列とする            → `G_0`   ✓
    \Sigma を行列とする         → `Sigma` ✓
    \hat H を行列とする         → **`H`**（`H_hat` ではない）
    G^0 を行列とする            → **失敗**
    declare G^0 as a matrix     → **成功を返して `declare` を宣言**

根因は、名前抽出が専用の正規表現（`_SYM_PAT`）で、`^` がクラスに無く、装飾は綴りごと
落ち、アンカーが無いので文中の動詞が名前として拾われたこと。
"""
import asyncio

import pytest

import ui.notebook_bridge as nb
from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operation_validator import OperationValidator


def _declare(lang, text):
    """1文だけ宣言して、登録簿に入った名前を返す。"""
    nb.start_engine(title="decl", mode="standard", language=lang,
                    domains=["quantum_mechanics"])
    try:
        eng = nb.get_engine()
        sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                            domains=["quantum_mechanics"], primary_backend="maxima")
        r = asyncio.get_event_loop().run_until_complete(
            run_interpret_pipeline(
                InterpretRequest(tokens=[InputToken(type="text", value=text)]),
                eng, sup))
        return r["status"], list(eng.symbol_context.registry)
    finally:
        nb.stop_engine()


class TestTheDeclaredNameIsTheWrittenName:
    @pytest.mark.parametrize("text,want", [
        ("G を行列とする", "G"),
        ("G_0 を行列とする", "G_0"),
        (r"\Sigma を行列とする", "Sigma"),
        (r"\hat H を行列とする", "H_hat"),        # ← `H` ではない
        ("G^0 を行列とする", "G_sup0"),           # ← 以前は失敗
        ("G^{(1)} を行列とする", "G_psup1"),
    ])
    def test_japanese(self, text, want):
        status, names = _declare("ja", text)
        assert status == "confirmed", (text, status)
        assert want in names, (text, names)

    @pytest.mark.parametrize("text,want", [
        ("declare G as a matrix", "G"),
        ("declare G^0 as a matrix", "G_sup0"),    # ← 以前は `declare` を宣言していた
        ("define G^0 as a matrix", "G_sup0"),
        ("declare G_0 as a matrix", "G_0"),
    ])
    def test_english(self, text, want):
        status, names = _declare("en", text)
        assert status == "confirmed", (text, status)
        assert want in names, (text, names)
        assert "declare" not in names and "define" not in names, (text, names)


class TestTheDeclarationReachesTheExpression:
    r"""宣言が着地する先が違えば、式の側の読みが黙って変わる。"""

    def test_a_labelled_name_becomes_a_matrix(self):
        nb.start_engine(title="decl2", mode="standard", language="ja",
                        domains=["quantum_mechanics"])
        try:
            eng = nb.get_engine(); h = eng._handler
            sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                                domains=["quantum_mechanics"], primary_backend="maxima")
            for text in ("G^0 を行列とする", r"\Sigma を行列とする"):
                asyncio.get_event_loop().run_until_complete(
                    run_interpret_pipeline(
                        InterpretRequest(tokens=[InputToken(type="text", value=text)]),
                        eng, sup))
            # 行列積になる（宣言が無ければ可換積）
            assert h.to_backend_code(
                h.latex_to_canonical(r"G^0 \Sigma")).code == "G_sup0 . Sigma"
            # そして `^{-1}` が逆行列になる
            assert h.to_backend_code(
                h.latex_to_canonical(r"(G^0)^{-1}")).code == "G_sup0^^-1"
        finally:
            nb.stop_engine()

    def test_a_decorated_name_lands_on_the_decorated_quantity(self):
        r"""`\hat H` を宣言したら、効くのは `\hat H^T` であって `H^T` ではない。"""
        nb.start_engine(title="decl3", mode="standard", language="ja",
                        domains=["quantum_mechanics"])
        try:
            eng = nb.get_engine(); h = eng._handler
            sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                                domains=["quantum_mechanics"], primary_backend="maxima")
            asyncio.get_event_loop().run_until_complete(
                run_interpret_pipeline(
                    InterpretRequest(tokens=[InputToken(
                        type="text", value=r"\hat H を行列とする")]), eng, sup))
            assert h.to_backend_code(
                h.latex_to_canonical(r"\hat H^T")).code == "transpose(H_hat)"
            assert h.to_backend_code(
                h.latex_to_canonical(r"H^T")).code == "H^T"
        finally:
            nb.stop_engine()


def _declare_tokens(lang, tokens):
    """トークン列（数式チップ混じり）で宣言し、(status, 登録簿, 式) を返す。"""
    nb.start_engine(title="decl_chip", mode="standard", language=lang,
                    domains=["quantum_mechanics"])
    try:
        eng = nb.get_engine()
        sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                            domains=["quantum_mechanics"], primary_backend="maxima")
        r = asyncio.get_event_loop().run_until_complete(
            run_interpret_pipeline(InterpretRequest(tokens=tokens), eng, sup))
        return (r["status"], list(eng.symbol_context.registry),
                [e.latex for e in eng.expressions.values()])
    finally:
        nb.stop_engine()


def _m(v):
    return InputToken(type="math", value=v, latex=v, canonical="")


def _t(v):
    return InputToken(type="text", value=v)


class TestTheSubjectOfADeclarationIsNotAnExpression:
    r"""宣言の主語を**数式チップ**で書いても宣言になる。

    実測 2026-08-22（利用者報告）: `\hat H_0` を数式チップで入れて
    「をスカラーとする」と書くと、チップが先に**式として登録され**、指示は
    `E1 をスカラーとする` になって何にも認識されない——宣言は起きず、要らない式が
    1つ残る。テキストで打つと通るので、**同じ文が入力経路で別物になっていた**。

    「この文は宣言か」を訊く口はバックエンド側に既にあり
    （`should_bypass_auto_register_formula`、インライン数式の経路がそれを使って
    いる）、数式チップの経路だけが訊いていなかった。
    """

    @pytest.mark.parametrize("lang,tokens,want", [
        # 利用者が報告した形（チップ1つ）
        ("ja", [_m(r"\hat H_0"), _t(" をスカラーとする")], "H_hat_0"),
        # 装飾と名前が別チップに割れた形（手書き入力が作る）
        ("ja", [_m(r"\hat"), _m("H_0"), _t(" をスカラーとする")], "H_hat_0"),
        ("ja", [_m(r"\Sigma"), _t(" を行列とする")], "Sigma"),
        ("en", [_t("declare "), _m(r"\hat H_0"), _t(" as a scalar")], "H_hat_0"),
    ])
    def test_a_math_chip_subject_declares(self, lang, tokens, want):
        status, registry, exprs = _declare_tokens(lang, tokens)
        assert status == "confirmed", (status, registry)
        assert want in registry, registry
        # 主語は式ではない——余計な式を残さない。
        assert exprs == [], exprs

    @pytest.mark.parametrize("lang,tokens", [
        # 宣言でない文の数式チップは、これまでどおり式として登録される。
        ("ja", [_m(r"\hat H_0 = a + b")]),
        ("ja", [_m("x^2 + 2x"), _t(" を因数分解して")]),
    ])
    def test_a_chip_that_is_not_a_subject_still_registers(self, lang, tokens):
        _status, registry, exprs = _declare_tokens(lang, tokens)
        assert exprs, "式として登録されるはず"
        assert registry == [] or "H_hat_0" not in registry


def _declare_sequence(lang, texts):
    """同じセッションで順に宣言し、(status 列, 登録簿の代数, 最後のエラー)。"""
    nb.start_engine(title="redecl", mode="standard", language=lang,
                    domains=["quantum_mechanics"])
    try:
        eng = nb.get_engine()
        sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                            domains=["quantum_mechanics"], primary_backend="maxima")
        statuses, err = [], ""
        for text in texts:
            r = asyncio.get_event_loop().run_until_complete(
                run_interpret_pipeline(
                    InterpretRequest(tokens=[InputToken(type="text", value=text)]),
                    eng, sup))
            statuses.append(r["status"])
            err = str(r.get("error") or err)
        algebras = {n: p_.quantity.math_type.algebra
                    for n, p_ in eng.symbol_context.registry.items()}
        return statuses, algebras, err
    finally:
        nb.stop_engine()


class TestADeclarationIsNotSilentlyReplaced:
    r"""型を変える宣言は**既定で断る**。上書きは「宣言し直して」と明示したときだけ。

    宣言はこの系のほとんどの読みの判別材料である（添字が index か名前か、`^{-1}`
    が逆行列か逆数か、積が可換か）。型を変えるとは、**その名前で既に書いた式の
    意味が変わる**ということなので、黙って起きてよいことではない。

    直前まで返っていたのは `Maxima evaluation error` ——CAS 内部の文字列で、何が
    起きたかも次に何をすればよいかも言っていなかった（実測 2026-08-22）。
    """

    def test_the_same_type_again_is_fine(self):
        statuses, algebras, _ = _declare_sequence(
            "ja", [r"\hat H_0 をスカラーとする", r"\hat H_0 をスカラーとする"])
        assert statuses == ["confirmed", "confirmed"], statuses
        assert algebras["H_hat_0"] == "scalar"

    def test_a_different_type_is_refused_with_a_reason(self):
        statuses, algebras, err = _declare_sequence(
            "ja", [r"\hat H_0 をスカラーとする", r"\hat H_0 を行列とする"])
        assert statuses == ["confirmed", "failed"], statuses
        # 断り文は**型の名前を利用者の言葉で**、名前を**利用者の綴りで**述べる。
        assert "スカラー" in err and "行列" in err
        assert r"\hat{H}_{0}" in err
        assert "Maxima" not in err          # 内部語を返さない
        # 断ったのだから、登録も変わっていない。
        assert algebras["H_hat_0"] == "scalar"

    def test_saying_re_declare_allows_it(self):
        statuses, algebras, _ = _declare_sequence(
            "ja", [r"\hat H_0 をスカラーとする",
                   r"\hat H_0 を行列として宣言し直して"])
        # CAS も通っている——`remove` を撃たなければ Maxima は declare を断る。
        assert statuses == ["confirmed", "confirmed"], statuses
        assert algebras["H_hat_0"] == "matrix"

    def test_english_needs_the_same_word(self):
        refused, _algebras, err = _declare_sequence(
            "en", ["declare H as a scalar", "declare H as a matrix"])
        assert refused == ["confirmed", "failed"], refused
        assert "scalar" in err and "matrix" in err
        allowed, algebras, _ = _declare_sequence(
            "en", ["declare H as a scalar", "re-declare H as a matrix"])
        assert allowed == ["confirmed", "confirmed"], allowed
        assert algebras["H"] == "matrix"
