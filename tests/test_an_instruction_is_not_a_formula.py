"""指示文は式ではない —— H-1 が「これは指示か」を日本語側だけで聞いていた。

``_is_valid_h1_input`` の条件7は長らく ``_HAS_CJK_RE``、つまり **日本語側の答え**
だけだった。同じ問いの英語側の答え（``instruction_marker[en]`` の動詞）は
``_is_nl_instruction`` が既に持っていて、パイプラインの他の2箇所はそれを聞いている
のに、H-1 だけが聞いていなかった。結果、操作を名指した英文が **式として登録** され
ていた（実測 2026-09-05、セッション言語 en・``E1: f = a*x + b``）:

    substitute x = 2 into E1  →  E2 を登録  canonical=2*into(a*x+b)
    set x = 2 in E1           →  E3 を登録  canonical=2*in(a*x+b)
    let x = 2 in E1           →  E4 を登録

英単語 ``into`` / ``in`` が **CAS の関数呼び出し**になり、``registered`` が返る——
研究者には成功に見える。``In E1, substitute x = 2`` だけが助かっていたのは条件8
（右辺が裸の数）にたまたま当たったからで、「指示かどうか」を聞いた結果ではない。

**guard は言語で絞らない。** 読みは絞ってよいが guard は駄目、というのがこの
コードベースの規則で（``nlp_any`` の docstring）、ja セッションに英文を打つ
researcher を取りこぼさないため条件7は ``nlp_any("instruction_marker", raw)`` を
聞く。コーパス 523行×2言語で受理の差は 0（下の ``TestTheCorpusAgrees``）。

**そして「どの式についての操作か」も、既に答えが渡っていた。** H-1 を通り抜けた
文は ``map_tokens_to_fields`` に届くが、この関数は ``known_expr_ids`` を引数で
受け取りながら本体で一度も読まず、位置だけで expr フィールドを埋めていた:

    substitute x = 2 into E1  →  {'expr': '=', 'lhs': 'x', 'rhs': '2'}

``E1`` が同じ文に立っているのに ``=`` が式として選ばれている。セッションが実際に
持っている E-id は推測ではないので、expr フィールドはそれを先に取る。
"""
import asyncio
import json
import pathlib

import pytest

import ui.interpret_pipeline as P
import ui.notebook_bridge as nb
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.nl_patterns import nlp_any
from notebook.runtime.operation_validator import OperationValidator
from notebook.runtime.token_field_mapping import map_tokens_to_fields
from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline

REPO = pathlib.Path(__file__).resolve().parent.parent
_ROWS = json.loads(
    (REPO / "tests" / "corpus" / "instruction_corpus.json").read_text(encoding="utf-8")
)["rows"]


# ---------------------------------------------------------------- 表: 指示文
#: ``=`` を含み、かつ **操作を名指している** 文。式として登録されてはならない。
#: 日英そろえてあるのは、この欠陥がまさに「片方の言語でしか聞いていなかった」
#: ことによるものだから。
INSTRUCTIONS = [
    ("ja", "E1 の x = 2 を代入して"),
    ("ja", "E1 に x = 2 を代入して"),
    ("ja", "E1 の x を 2 に置換して"),
    ("ja", "E4 の初期条件を x(0)=1 とする"),
    ("en", "substitute x = 2 into E1"),
    ("en", "set x = 2 in E1"),
    ("en", "let x = 2 in E1"),
    ("en", "In E1, substitute x = 2"),
    ("en", "replace x with 2 in E1"),
    ("en", "Set the initial conditions of E4 to x(0)=1"),
    ("en", "Change variables in E1 from x,y to r,theta"),
    ("en", "solve E1 for x = 2"),
]

#: 本物の式。どちらの言語のセッションで打たれても H-1 で登録され続けること。
#: 全部コーパスの実在の綴り（``scripts/build_instruction_corpus.py`` 由来）。
EQUATIONS = [
    "H = J * x * y",
    "f = a*x + b",
    "E_tot = (1/2) m (\\dot{r}^2 + r^2 \\dot{theta}^2) - k / r",
    "Z = R + j*w*L + 1/(j*w*C)",
    "C = K D / Q + h Q / 2",
    "L_z = m r^2 \\dot{theta}",
    "D = 4 \\omega_0^2 - (2 \\gamma)^2",
    "EOM = E3 - E4",
    "A = [2, 0; 0, 1]",
]


class TestAnInstructionIsNotReadAsAFormula:
    @pytest.mark.parametrize("lang,text", INSTRUCTIONS)
    def test_h1_declines_it(self, lang, text):
        assert P._is_valid_h1_input(text) is None, (
            f"[{lang}] 操作を名指した文が H-1 の式として受理された: {text!r}")

    @pytest.mark.parametrize("text", EQUATIONS)
    def test_a_real_equation_still_registers(self, text):
        assert P._is_valid_h1_input(text) is not None, (
            f"本物の式が H-1 に受理されなくなった: {text!r}")


class TestTheCorpusAgrees:
    """実在の綴りで、両方向を測る。ここが「移す前に新旧を比べる」の凍結版。"""

    def test_no_named_operation_is_read_as_a_formula(self):
        """操作を名指す 530 綴りのうち、H-1 が式として受けるものは 0。"""
        viol, n = [], 0
        for row in _ROWS:
            op = row.get("operation")
            if not op or op == "(expression)":
                continue
            for key in ("jp", "en"):
                text = (row.get(key) or "").strip()
                if not text:
                    continue
                n += 1
                if P._is_valid_h1_input(text) is not None:
                    viol.append((row["id"], key, op, text))
        assert n >= 500, f"綴りが {n} 件しか読めていない（コーパスの読み込み失敗？）"
        assert viol == [], f"{len(viol)} 件が式として読まれた: {viol[:5]}"

    def test_condition_7_never_rejects_a_real_formula(self):
        """条件7 **だけ** で落ちる式登録行は 0。

        他の条件（左辺が裸の E-id、左辺に括弧、右辺に ``=``…）で落ちる行は
        この変更と無関係なので、条件7を外した判定と比べて差だけを見る。
        """
        def _accept_without_condition_7(raw):
            m = P._split_at_top_level_eq(raw)
            if m is None:
                return False
            lhs, rhs = m.group("lhs").strip(), m.group("rhs").strip()
            return not (not P._LHS_VALID_RE.search(lhs)
                        or P._EXPR_ID_RE.match(lhs)
                        or "(" in lhs
                        or "=" in rhs
                        or P._RHS_BARE_NUMBER_RE.match(rhs)
                        or P._RHS_TRAILING_WORD_RE.search(rhs))

        diff, n = [], 0
        for row in _ROWS:
            if row.get("operation") != "(expression)":
                continue
            for key in ("jp", "en"):
                text = (row.get(key) or "").strip()
                if not text:
                    continue
                n += 1
                if (P._is_valid_h1_input(text) is not None) != \
                        _accept_without_condition_7(text):
                    diff.append((row["id"], key, text))
        assert n >= 300, f"綴りが {n} 件しか読めていない"
        assert diff == [], f"条件7 だけで落ちる式登録行: {diff[:5]}"

    def test_the_guard_is_not_narrowed_by_language(self):
        """条件7 が聞くのは ``nlp_any``——日本語セッションに打たれた英文も捕まる。"""
        assert nlp_any("instruction_marker", "substitute x = 2 into E1")
        assert nlp_any("instruction_marker", "E1 の x に 2 を代入して")
        assert not nlp_any("instruction_marker", "H = J * x * y")


class TestAPunctuationTokenIsNotAnExpression:
    """式には名前か数がある。``=`` はどうやっても式ではない。

    **最初に書いた規則「実在する E-id を先に取る」は実測で覆った** ——
    `heisenberg_equation` のフィールドは `operator, hamiltonian` の順で、
    `ハイゼンベルク方程式 a(1) E5` の E-id は**ハミルトニアン**の側である。
    順序は書かれたとおりが正しく、落とすべきなのは式になりえないものだけ。
    """

    def test_a_bare_relational_token_is_not_the_expression(self):
        args = map_tokens_to_fields(
            ["x", "=", "2", "into", "E1"], ["expr", "lhs", "rhs"], {"E1"}, "en")
        assert args["expr"] == "E1", args

    def test_it_holds_in_japanese_too(self):
        args = map_tokens_to_fields(
            ["x", "=", "2", "E1"], ["expr", "lhs", "rhs"], {"E1"}, "ja")
        assert args["expr"] == "E1", args

    def test_a_leading_expression_id_is_unchanged(self):
        args = map_tokens_to_fields(
            ["E1", "x", "=", "2"], ["expr", "lhs", "rhs"], {"E1"}, "en")
        assert args["expr"] == "E1", args

    def test_the_written_order_of_expression_operands_is_kept(self):
        """順序は書かれたとおり —— E-id を前に寄せると別の操作が壊れる。

        `heisenberg_equation(operator, hamiltonian)` の実際の綴り。
        2026-09-05 に「E-id を先に」と書いたらこれが落ちた（フルスイートが捕まえた）。
        """
        args = map_tokens_to_fields(
            ["a(1)", "E5"], ["operator", "hamiltonian"], {"E5"}, "ja")
        assert args["operator"] == "a(1)", args
        assert args["hamiltonian"] == "E5", args

    def test_an_operand_written_before_an_expression_id_keeps_its_place(self):
        args = map_tokens_to_fields(
            ["a'(1)", "E1"], ["expr", "operator"], {"E1"}, "ja")
        assert args["expr"] == "a'(1)", args
        assert args["operator"] == "E1", args

    def test_a_token_carrying_a_name_is_still_an_expression(self):
        """落とすのは句読点だけ —— 行列リテラルもギリシャ文字も式のまま。"""
        for tok in ("[2,1;1,2]", "\\alpha", "a'(1)", "x^2"):
            args = map_tokens_to_fields([tok], ["expr"], set(), "ja")
            assert args.get("expr") == tok, (tok, args)


def _run(eng, sup, text):
    return asyncio.get_event_loop().run_until_complete(
        run_interpret_pipeline(
            InterpretRequest(tokens=[InputToken(type="text", value=text)]), eng, sup))


@pytest.mark.parametrize("lang,text", [
    ("ja", "E1 の x = 2 を代入して"),
    ("ja", "E1 に x = 2 を代入して"),
    ("en", "In E1, substitute x = 2"),
    ("en", "substitute x = 2 into E1"),
])
def test_the_value_actually_reaches_the_cas(lang, text):
    """画面まで: ``f = a*x + b`` に x=2 を入れて ``b+2*a`` が返ること。

    単体では ``expr`` が正しく埋まったことしか言えない。silent-wrong だったのは
    **登録が成功として返る**ことなので、値まで見ないと再発を捕まえられない。
    """
    nb.start_engine(title="instruction_not_formula", mode="standard",
                    language=lang, domains=["classical_mechanics"])
    try:
        eng = nb.get_engine()
        sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                            domains=["classical_mechanics"], primary_backend="maxima")
        _run(eng, sup, "a, b をスカラーとする" if lang == "ja"
             else "Define a, b as scalars")
        _run(eng, sup, "f = a*x + b")
        n_before = len(eng.expressions)
        resp = _run(eng, sup, text)
        assert resp.get("status") == "completed", (text, resp.get("status"),
                                                   resp.get("error"))
        assert (resp.get("canonical") or "").replace(" ", "") == "b+2*a", resp
        assert len(eng.expressions) - n_before == 1, "式が余分に登録された"
    finally:
        nb.stop_engine()
