r"""「この文は指示か」 —— 手書きの動詞一覧**と**スキーマの別名、両方に訊く。

実測 2026-09-10（INSTRUCTION-MARKER-HANDLIST）。英語側の答えは
`instruction_marker[en]` の**手書きの動詞35語**だけだった。だから操作を名指した
英文が、動詞がその一覧に無いというだけで**式として登録**されていた:

    relabel k to -k in E1        → `relabel*k*to -k*in*(\sum_k …)` を登録（registered）
    the residue of E1 at z = 1   → `the*residue*of (1/(z-1))*at*z = 1` を登録

どちらも `registered` が返る——研究者には成功に見える。日本語の対
（`E1 の k を -k に指標変換して`）は前から `relabel_index` に届いていたので、
**日英の割れ**でもある。

**答えはスキーマのデータに在った。** `formula_is_an_operation_operand` が
「残りの文が名指した操作は何か」を別名走査で既に答えており、CLAUDE.md も
「境界は**操作の別名そのもの**、スキーマのデータから作る」と書いている。

**一覧は消さない —— 二つは違う問いに答えている**（実測）:

| 証拠 | 何を言うか | 一方だけだと |
|---|---|---|
| スキーマの別名 | この文は操作を名指しているか | 英語別名を持つ 103 操作のうち **65 は1語で名指せる**（走査が拾う） |
| 動詞の一覧 | この文は命令形か | 非 CJK の実在入力480行のうち **37行**は一覧だけが「指示」と言う（`Find …` / `Save the lattice figure` / `subtract E12 from E11`——動詞が別名ではない） |

**`relabel` は別名データのほうが欠けていた。** スキーマの別名は
`relabel index` という**句**で、`relabel k to -k` には現れない。裸の動詞は実在の
用法（日本語の `指標変換` に対応）で、`resolve_alias("relabel")` は他のどの操作
にも当たらない——衝突なしで足せた。`rename` は足していない: `set_latex_name`
とぶつかる。

**コーパスはこの変更について何も言わない。** `relabel` / `residue` /
`指標変換` / `リラベル` を含む行は **0行**（実測）——差分0は「安全」ではなく
「測っていない」である（CORPUS-WORKING-ONLY）。だからここで打つ。
"""
import asyncio

import pytest

import ui.interpret_pipeline as P
import ui.notebook_bridge as nb
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operand_boundary import text_names_an_operation
from notebook.runtime.operation_validator import OperationValidator
from ui.interpret_common import _is_nl_instruction
from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline


def _names(text):
    v = OperationValidator()
    return text_names_an_operation(
        text, v.resolve_alias, v.max_alias_words, v.get_spec,
        P._tokenize_for_alias_scan)


def _run(lang, domain, setup, text):
    async def go():
        nb.start_engine(title="marker", mode="standard", language=lang,
                        domains=[domain])
        try:
            engine = nb.get_engine()
            sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                                domains=[domain],
                                primary_backend=engine.session.primary_backend)
            for line in setup:
                await run_interpret_pipeline(
                    InterpretRequest(tokens=[InputToken(type="text", value=line)]),
                    engine, sup)
            before = set(engine.steps)
            resp = await run_interpret_pipeline(
                InterpretRequest(tokens=[InputToken(type="text", value=text)]),
                engine, sup)
            new = [s for sid, s in engine.steps.items() if sid not in before]
            op = next((s.operation for s in new if s.operation), None)
            args = next((dict(s.operation_args or {}) for s in new if s.operation), {})
            return resp, op, {k: v for k, v in args.items() if not k.startswith("_")}
        finally:
            nb.stop_engine()
    return asyncio.run(go())


#: 「操作を名指しているのに、動詞が一覧に無い」英文。族は**式として登録されない**
#: こと——登録されると、研究者の指示が自分の言葉でできた式になる。
NAMED_BUT_NOT_LISTED = [
    ("condensed_matter", [r"H = \sum_k xi_k c'_k c_k"],
     "relabel k to -k in E1", "relabel_index"),
    ("complex_analysis", ["f = 1/(z-1)"],
     "the residue of E1 at z = 1", "residue"),
]


@pytest.mark.parametrize("domain,setup,text,want_op", NAMED_BUT_NOT_LISTED,
                         ids=[c[2][:22] for c in NAMED_BUT_NOT_LISTED])
class TestAnInstructionNamingAnOperationIsNotAFormula:
    def test_it_is_not_registered_as_an_expression(self, domain, setup, text, want_op):
        resp, op, _ = _run("en", domain, setup, text)
        assert resp.get("route") != "expr_id_formula", resp
        assert op == want_op, (op, resp.get("error"))

    def test_the_guard_says_instruction(self, domain, setup, text, want_op):
        assert not _is_nl_instruction(text, "en"), (
            "この文の動詞は一覧に無い——それがこの検査の前提")
        assert _is_nl_instruction(text, "en", _names), text


class TestAnOperationThisBackendLacksIsStillAnInstruction:
    r"""**「指示か」と「このバックエンドが実行できるか」は別の問い。**

    最初この走査は `formula_is_an_operation_operand` に倣って
    「バックエンドが対応しているか」も訊いていた。**変異試験がそれを覆した**
    ——その条件を外しても検査が1本も落ちず、外した状態を実測すると
    **落とすほうが正しかった**（[[feedback-verify-the-guard-catches-its-own-defect]] ⑨）:

        Maxima セッションで `wick contract E1 + E2`
            訊いていたとき → `wick*contract (matrix(…)) + (matrix(…))` を**登録**
            訊かないとき   → 断り

    持っていない操作を名指した文の正しい答えは、誤った登録ではなく断りである。
    """

    @pytest.mark.parametrize("text", [
        "wick contract E1 + E2",
        "normal order E1 + E2",
    ])
    def test_it_is_refused_not_registered(self, text):
        resp, op, _ = _run("en", "linear_algebra",
                           ["A = [2, 1; 1, 2]", "B = [1, 0; 0, 3]"], text)
        assert resp.get("route") != "expr_id_formula", resp
        assert resp.get("status") == "failed", resp
        assert op is None, op


class TestBothLanguagesReachTheSameOperation:
    r"""`relabel` は日本語側だけが届いていた —— 日英の割れ。"""

    @pytest.mark.parametrize("lang,text", [
        ("ja", "E1 の k を -k に指標変換して"),
        ("ja", "E1 の k を -k にリラベルして"),
        ("en", "relabel k to -k in E1"),
    ])
    def test_relabel_reaches_relabel_index(self, lang, text):
        _, op, args = _run(lang, "condensed_matter",
                           [r"H = \sum_k xi_k c'_k c_k"], text)
        assert op == "relabel_index", (lang, text, op)
        assert args.get("old_index") == "k" and args.get("new_index") == "-k", args


class TestAnEquationWhoseRightSideNamesAnOperation:
    r"""`y = trace of E1` —— H-1 の条件7（これは指示文か）にも同じ証拠が要る。

    直す前は H-1 が拾って**式として登録**しており、しかも登録された式は
    研究者が書いたものと無関係だった（実測 2026-09-10）:

        y = trace of E1        → canonical `of(matrix([2,1],[1,2]))*[]`
        T = determinant of E1  → `matrix([2*%determinant(of), …])`

    表示にいたっては生の TeX マクロ（`\ifx\endpmatrix\undefined\pmatrix{…`）
    が `latex` 欄に入っていた。**`of` が関数呼び出しとして読まれた**結果である
    ——`substitute x = 2 into E1` が `2*into(a*x+b)` になっていたのと同じ形で、
    あちらは動詞が一覧に在ったので直り、こちらは在らず残っていた。
    """

    @pytest.mark.parametrize("text,want_op", [
        ("y = trace of E1", "trace"),
        ("T = determinant of E1", "determinant"),
    ])
    def test_it_reaches_the_operation_not_h1(self, text, want_op):
        resp, op, args = _run("en", "linear_algebra", ["A = [2, 1; 1, 2]"], text)
        assert resp.get("route") != "h1_equation", resp
        assert op == want_op, (op, resp)
        assert args.get("expr") == "E1", args


class TestTheNotationDialogIsNotTakenAway:
    r"""**すべての gate に足すのが正しいとは限らない**（実測して決めた）。

    縦棒の意味（絶対値か行列式か）は、その文が操作を名指していても**未決**で、
    訊く相手は研究者である。ここに証拠を足すと、答えられる問いが断りに変わる:

        `the determinant of |E1|`
            足さない → 縦棒の意味を訊くダイアログ
            足す     → ノルム／絶対値の断り
    """

    def test_the_bars_still_ask(self):
        resp, _, _ = _run("en", "linear_algebra", ["A = [2, 1; 1, 2]"],
                          "the determinant of |E1|")
        assert resp.get("status") == "needs_confirmation", resp
        assert resp.get("candidates"), resp


class TestTheTwoSourcesAnswerDifferentQuestions:
    """片方だけでは足りない —— どちらを外しても実在の入力が落ちる。"""

    @pytest.mark.parametrize("text", [
        "Find the general solution of E1",
        "compute the right-hand side of E1",
        "Save the lattice figure",
        "subtract E12 from E11",
        "Change variables in E1 from x,y to r,theta via x=r*cos(theta), y=r*sin(theta)",
    ])
    def test_the_verb_list_names_what_the_alias_table_cannot(self, text):
        """これらの動詞は**操作の別名ではない**（命令の語である）。"""
        assert _is_nl_instruction(text, "en"), text
        assert not _names(text), (
            f"{text!r} が別名走査で取れるなら、この行は例として古い")

    @pytest.mark.parametrize("text", [
        "the residue of E1 at z = 1",
        "relabel k to -k in E1",
    ])
    def test_the_alias_table_names_what_the_verb_list_cannot(self, text):
        assert not _is_nl_instruction(text, "en"), text
        assert _names(text), text


class TestTheAliasIsRealUsageAndCollidesWithNothing:
    r"""裸の `relabel` を別名に足した判断 —— 実在の用法か、衝突しないか。"""

    def test_the_bare_verb_resolves_to_the_index_operation(self):
        v = OperationValidator()
        assert v.resolve_alias("relabel") == "relabel_index"

    def test_a_word_that_would_collide_was_not_added(self):
        """`rename` は `set_latex_name` とぶつかるので足していない。

        別名の索引は first-come で衝突を解くので、足すと**どちらかが
        その語で到達不能**になる（BP-6）。
        """
        v = OperationValidator()
        assert v.resolve_alias("rename") is None


class TestPlainFormulaInputIsStillAFormula:
    """**広げすぎていないこと。** 名詞でもある別名を含む素の式が指示に倒れない。"""

    @pytest.mark.parametrize("text", [
        "x^2 + 1",
        "[2, 1; 1, 2]",
        "E5 + 2 * E7",
        r"H = -t \sum_{\langle i j \rangle} c'_i c_j",
        r"\frac{1}{2}",
    ])
    def test_it_is_not_an_instruction(self, text):
        assert not _is_nl_instruction(text, "en", _names), text
