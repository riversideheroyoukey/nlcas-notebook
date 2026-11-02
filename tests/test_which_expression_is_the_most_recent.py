r"""OPERAND-BOUNDARY 残②（2026-09-11）。

**「どれが直前の式か」にも、答えは1つしかない。**

「落ちてよいか」は先に1箇所へ畳んだ
（`operand_boundary.may_fall_back_to_the_recent_expression`）が、**どれに落ちるか**
には3通りの実装が残っていた:

| 場所 | 答え方 |
|---|---|
| 行列単項 `_recent()` / q̇ 規則 `_recent_ids[-1]` | 登録順の最後 ∩ 系譜 ∩ 非 hidden |
| `interpret_plan._find_target` | E 番号の最大 ∩ 系譜 ∩ 非 hidden |
| 足場 step 2d / 候補畳み込み | E 番号の最大 ∩ 系譜 ∩ **全式**（hidden を含む） |

**起票が疑っていた「分岐で割れる」は実測で偽だった**——fork → 枝で登録 →
main に戻る、のどこでも3つは一致し、コーパス1102行でも保存済みセッション18本
でも差分0。それでも畳むのは、**割れる状態が2つ具体的に在る**からである
（下の2クラスがその状態を作って測る）。

採った答えは**いちばん保守的なもの**——`_active_expr_ids()`（系譜 ∩ 非 hidden）
の中で E 番号が最大。隠した式も、研究者が見たことのない内部鍵も返さない。
"""
import pytest

from notebook.engine import NotebookEngine


@pytest.fixture
def engine():
    with NotebookEngine(title="most-recent") as e:
        yield e


class TestTheAnswerItself:

    def test_the_latest_expression(self, engine):
        engine.register_expression("x^2")
        b = engine.register_expression("x^3")
        assert engine.most_recent_expr_id() == b.display_id

    def test_an_empty_session_has_none(self, engine):
        assert engine.most_recent_expr_id() is None

    def test_a_hidden_expression_is_never_the_answer(self, engine):
        r"""**割れ①**: 隠れた式が最大番号を持つとき。足場と候補畳み込みだけが
        `hidden` を除いておらず、研究者から**意図的に隠した式**を選んでいた."""
        a = engine.register_expression("x^2")
        b = engine.register_expression("x^3")
        engine.expressions[b.display_id] = engine.expressions[
            b.display_id].model_copy(update={"hidden": True})
        assert engine.most_recent_expr_id() == a.display_id

    def test_an_internal_scaffold_is_never_the_answer(self, engine):
        r"""**割れ②**: 可視の内部式（`_Tn`）が登録順の最後に来たとき。登録順で
        答える2箇所だけが `_T1` を返す——研究者が一度も見たことのない鍵である."""
        a = engine.register_expression("x^2")
        t = engine.register_expression("x^5", internal=True)
        assert not t.display_id.startswith("E"), t.display_id
        assert engine.most_recent_expr_id() == a.display_id


class TestBranchingDoesNotChangeTheAnswer:
    r"""起票の仮説「枝を切ると食い違いうる」は**実測で偽**だった——
    それでも、枝が答えを正しく絞ることは押さえておく."""

    def test_a_branch_sees_its_own_lineage(self, engine):
        engine.register_expression("x^2")
        b = engine.register_expression("x^3")
        c = engine.register_expression("x^4")
        assert engine.most_recent_expr_id() == c.display_id

        fork_step = next(s for s in engine.steps.values()
                         if b.display_id in (s.output_refs or []))
        engine.fork_branch(fork_step.step_id, label="alt")
        # 枝は fork 時点までしか見えない
        assert engine.most_recent_expr_id() == b.display_id
        d = engine.register_expression("y^5")
        assert engine.most_recent_expr_id() == d.display_id

        engine.switch_branch("main")
        assert engine.most_recent_expr_id() == c.display_id


# ---------------------------------------------------------------------------
# 経路。**同じ答えを返すか**を、割れる状態（最大番号が hidden）で測る。
# ---------------------------------------------------------------------------
import asyncio  # noqa: E402

import ui.notebook_bridge as nb  # noqa: E402
from notebook.runtime.llm_supervisor import LLMSupervisor  # noqa: E402
from notebook.runtime.operation_validator import OperationValidator  # noqa: E402
from ui.interpret_pipeline import (InputToken, InterpretRequest,  # noqa: E402
                                   run_interpret_pipeline)


def _run_with_hidden_latest(lang, lines, probe):
    r"""行列を3つ登録し、**最大番号を hidden にしてから** *probe* を打つ。

    畳む前は、足場と候補畳み込みだけが隠した E3 を選び、他は E2 を選んでいた。
    """
    async def go():
        nb.start_engine(title="most-recent-routes", mode="standard",
                        language=lang, domains=["physics"])
        try:
            engine = nb.get_engine()
            sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                                domains=["physics"],
                                primary_backend=engine.session.primary_backend)
            for line in lines:
                await run_interpret_pipeline(
                    InterpretRequest(tokens=[InputToken(type="text", value=line)]),
                    engine, sup)
            latest = max(engine.expressions)
            engine.expressions[latest] = engine.expressions[latest].model_copy(
                update={"hidden": True})
            return await run_interpret_pipeline(
                InterpretRequest(tokens=[InputToken(type="text", value=probe)]),
                engine, sup)
        finally:
            nb.stop_engine()
    return asyncio.run(go())


#: 行列式が 3 / 4 / 5 と分かれるので、答えがどれを見たか値で分かる。
_MATRICES = [
    r"\begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}",
    r"\begin{pmatrix} 4 & 0 \\ 0 & 1 \end{pmatrix}",
    r"\begin{pmatrix} 5 & 0 \\ 0 & 1 \end{pmatrix}",
]


class TestEveryRouteGivesTheSameAnswer:

    @pytest.mark.parametrize("lang,probe", [
        ("ja", "行列式を求めて"),
        ("en", "find the determinant"),
    ])
    def test_the_matrix_unary_route(self, lang, probe):
        r = _run_with_hidden_latest(lang, _MATRICES, probe)
        assert r.get("status") == "completed", r
        assert (r.get("canonical") or "").strip() == "4", r   # 隠した E3 なら 5

    @pytest.mark.parametrize("lang,probe", [
        ("ja", "固有値と固有ベクトルを求めて"),
        ("en", "eigenvalues and eigenvectors"),
    ])
    def test_the_plan_target_route(self, lang, probe):
        r = _run_with_hidden_latest(lang, _MATRICES, probe)
        assert r.get("status") == "completed", r
        assert "5" not in (r.get("canonical") or ""), r       # 隠した E3 の固有値
        assert "4" in (r.get("canonical") or ""), r

    @pytest.mark.parametrize("lang,probe", [
        ("ja", "簡約して"),
        ("en", "simplify"),
    ])
    def test_the_scaffold_route(self, lang, probe):
        r = _run_with_hidden_latest(lang, _MATRICES, probe)
        assert r.get("status") == "completed", r
        assert "5" not in (r.get("canonical") or ""), r

    @pytest.mark.parametrize("lang,setup,probe", [
        ("en", [r"A = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}",
                r"B = \begin{pmatrix} 5 & 0 \\ 0 & 1 \end{pmatrix}"],
         "compute the determinant of [4, 0; 0, 1]"),
    ])
    def test_the_matrix_unary_route_when_the_operand_is_written_inline(
            self, lang, setup, probe):
        r"""行列単項の落下に**実際に届く綴り**（実測 2026-09-11）。

        足場が式番号を入れてしまう綴りではこの枝に届かないので、届く1つで測る
        ——英語の「動詞＋末尾の式」。隠した `B`（行列式 5）を選ばないこと
        **だけ**を主張する: この文が返すべき 4 を返さないのは別の欠陥で
        （open_items MATRIX-INLINE-OPERAND-EN）、そこを直したときにこの
        アサーションが邪魔にならないようにしてある。
        """
        r = _run_with_hidden_latest(lang, setup, probe)
        assert r.get("status") == "completed", r
        assert (r.get("canonical") or "").strip() != "5", r

    def test_the_matrix_written_in_the_sentence_is_the_operand(self):
        r"""`[4, 0; 0, 1]` の行列式は 4。

        **2026-09-11 に根治**（MATRIX-INLINE-OPERAND-EN）。直す前は 3——
        別の行列 A の行列式——が `completed` で返っていた。文頭の計算動詞が
        `_en_verb_formula_re` の文頭アンカーを潰し、書いた行列が登録されない
        まま直前の式へ落ちていた。族としての回帰は
        `tests/test_english_compute_verb_operand.py`。
        """
        r = _run_with_hidden_latest(
            "en", [r"A = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}",
                   r"B = \begin{pmatrix} 5 & 0 \\ 0 & 1 \end{pmatrix}"],
            "compute the determinant of [4, 0; 0, 1]")
        assert (r.get("canonical") or "").strip() == "4", r
