r"""**記録に残るのは、研究者が打った文。** 解釈器に渡す足場ではない。

実測 2026-09-10（DECL-BARE-NOUN-EN（残）②）。式が名指されていない指示には、
直前の式番号を補って解釈器へ渡す仕組みがある（`ui/interpret_pipeline` 2d）:

    supervisor_text = f"{inject_id} に対して {supervisor_text}"

この文字列がそのまま `record_llm_supervision` に渡され、`user_text` として
**保存され、公開実例に出ていた**——コミット済みの `.json` に6件:

    "user_text": "E19 に対して G を行列とする"          ← 研究者は `G を行列とする` と打った
    "user_text": "E14 に対して B をベクトル場として定義する"

**一つの欄に2つの意味を入れていたのが根である**: 「研究者が書いたもの」と
「解釈器に渡したもの」。欄の名前は `user_text` で、下流（タイムラインの YOU
カード・TeX 出力・advisory・保存ファイル）は全部**前者**として読む。

**足場の言語も直した。** 接続詞は日本語直書きで、英語のセッションにも
`E1 に対して simplify` が渡っていた（実測）。読む相手はその言語の自然言語
パーサと（設定されていれば）LLM なので、セッションの言語で書く
（`interpreter.expr_ref_prefix`）。英語側に `for` / `over` / `by` を使わないのは、
それらが `en_op_var_head` の綴りで、**注入した語が微分の変数や解く対象として
読まれうる**ため。

**表示側の整形は消した。** `ui/timeline_serializer` に「先頭の式参照を削る」
整形（10操作を手で並べた表つき）が在ったが、足場が乗っていたのは
`llm_supervision` ステップのほうで、その整形が読む `user_input` ステップには
**1件も無かった**。発火しないうえ、発火すれば**研究者自身の言葉を削る**
——`E5 の x をスカラーとする` の `E5 の` は研究者が打った文の一部で、
保存済みセッションに実在する。
"""
import asyncio
import json
import pathlib
import re

import pytest

import ui.notebook_bridge as nb
from notebook.models.step import StepKind as SK
from notebook.runtime.i18n import t
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operation_validator import OperationValidator
from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline

REPO = pathlib.Path(__file__).resolve().parent.parent

#: 注入が実際に起きる指示 —— **式を名指していない**もの。族は
#: 「操作は名指されているが、どの式に対してかは書かれていない」で、
#: 直前の式が補われる。
INJECTING = [
    ("ja", "linear_algebra", ["A = [2, 1; 1, 2]"], "固有値を求めて"),
    ("ja", "quantum_mechanics", ["H = a + b"], "G を行列とする"),
    ("ja", "electromagnetism", ["E = a + b"], "B をベクトル場として定義する"),
    ("en", "linear_algebra", ["A = [2, 1; 1, 2]"], "simplify"),
    ("en", "linear_algebra", ["A = [2, 1; 1, 2]"], "expand"),
    ("en", "linear_algebra", ["A = [2, 1; 1, 2]"], "the determinant"),
]


def _run(lang, domain, setup, text):
    """走らせて、この1文が作った全ステップの `user_text` を返す。"""
    async def go():
        nb.start_engine(title="scaffold", mode="standard", language=lang,
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
            return resp, [(s.kind, s.user_text) for s in new if s.user_text]
        finally:
            nb.stop_engine()
    return asyncio.run(go())


@pytest.mark.parametrize("lang,domain,setup,text", INJECTING,
                         ids=[f"{c[0]}-{c[3][:18]}" for c in INJECTING])
class TestWhatIsRecordedIsWhatWasTyped:
    def test_no_step_records_more_than_the_researcher_wrote(
            self, lang, domain, setup, text):
        _, recorded = _run(lang, domain, setup, text)
        assert recorded, "ステップが1つも記録されていない"
        for kind, got in recorded:
            assert got.strip() == text, (kind, got)

    def test_the_operation_still_resolves(self, lang, domain, setup, text):
        """**足場を記録から外しても、読みは変わらない。**

        足場は解釈器に渡り続ける——記録に入らないだけである。ここが落ちたら
        注入そのものを壊している。
        """
        resp, _ = _run(lang, domain, setup, text)
        assert resp.get("status") in ("completed", "confirmed", "registered",
                                      "needs_confirmation"), resp.get("error")


#: 注入が起き、かつ**操作が失敗する**指示（そのバックエンドが持たない操作）。
#: 成功側だけ測ると、失敗側の記録は変異させても落ちない（実測、変異 M1）。
INJECTING_AND_FAILING = [
    ("en", "linear_algebra", ["A = [2, 1; 1, 2]"], "normal order"),
    ("ja", "linear_algebra", ["A = [2, 1; 1, 2]"], "正規順序にして"),
]


@pytest.mark.parametrize("lang,domain,setup,text", INJECTING_AND_FAILING,
                         ids=[f"{c[0]}-{c[3][:14]}" for c in INJECTING_AND_FAILING])
class TestAFailedInterpretationRecordsTheTypedTextToo:
    """**断られた指示も記録される**——そこも研究者の文であること。

    記録は2箇所（成功側と失敗側）にあり、片方だけ直すと、**失敗した指示の
    ほうにだけ足場が残る**。研究者がいちばん見返すのは失敗したほうである。
    """

    def test_the_failed_step_records_what_was_typed(self, lang, domain, setup, text):
        resp, recorded = _run(lang, domain, setup, text)
        assert resp.get("status") == "failed", resp
        assert recorded, "失敗でもステップは記録されるはず"
        for kind, got in recorded:
            assert got.strip() == text, (kind, got)


class TestTheScaffoldSpeaksTheSessionsLanguage:
    r"""読む相手はその言語の自然言語パーサと LLM。"""

    @staticmethod
    def _what_the_supervisor_was_given(lang, domain, setup, text):
        """**足場の直接の消費者**に当てる —— supervisor が受け取った文字列。

        記録から外した以上、足場は外から見えない。パターンだけを測ると
        「パイプラインが実際に何を渡しているか」は誰も見ておらず、
        接頭辞を日本語直書きに戻す変異が生き残る（実測、変異 M3）。

        当てる先は `_try_rule_short_circuits` —— 注入（2d）の**直後に呼ばれる
        最初の消費者**で、規則で片づく指示も LLM へ行く指示も必ずここを通る。
        `supervisor.interpret` に当てると、規則で片づいた指示では**一度も
        呼ばれず**空振りする（実測: `固有値を求めて`）。
        """
        import ui.interpret_pipeline as P
        seen = []
        original = P._try_rule_short_circuits

        async def spy(req, engine, loop, lang_, supervisor_text, *a, **k):
            seen.append(supervisor_text)
            return await original(req, engine, loop, lang_, supervisor_text, *a, **k)

        async def go():
            nb.start_engine(title="scaffold", mode="standard", language=lang,
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
                seen.clear()
                P._try_rule_short_circuits = spy
                try:
                    await run_interpret_pipeline(
                        InterpretRequest(tokens=[InputToken(type="text", value=text)]),
                        engine, sup)
                finally:
                    P._try_rule_short_circuits = original
            finally:
                nb.stop_engine()
            return seen

        asyncio.run(go())
        return seen

    @pytest.mark.parametrize("lang,domain,setup,text,marker", [
        ("ja", "linear_algebra", ["A = [2, 1; 1, 2]"], "固有値を求めて", "に対して"),
        ("en", "linear_algebra", ["A = [2, 1; 1, 2]"], "simplify", ":"),
    ])
    def test_the_supervisor_is_given_the_prefix_of_its_own_language(
            self, lang, domain, setup, text, marker):
        given = self._what_the_supervisor_was_given(lang, domain, setup, text)
        assert given, "supervisor に届いた文字列が取れていない"
        prefixed = [g for g in given if re.match(r"^E\d+", g.strip())]
        assert prefixed, given
        for g in prefixed:
            assert marker in g, (lang, g)
        if lang == "en":
            for g in prefixed:
                assert "に対して" not in g, g

    def test_both_languages_have_their_own_prefix(self):
        ja = t("interpreter.expr_ref_prefix", "ja", eid="E1", text="X")
        en = t("interpreter.expr_ref_prefix", "en", eid="E1", text="X")
        assert ja != en, (ja, en)
        for got in (ja, en):
            assert "E1" in got and got.rstrip().endswith("X"), got

    def test_the_english_prefix_uses_no_operand_keyword(self):
        r"""`for` / `over` / `by` / `in` は `en_op_var_head(_wide)` の綴り。

        注入した語がそこに当たると、**補った文脈が微分の変数や解く対象として
        読まれる**——`differentiate with respect to x` に `for E1,` を足すと
        E1 が変数の側に見える。だから英語側は語を使わない。
        """
        en = t("interpreter.expr_ref_prefix", "en", eid="E1", text="X")
        assert not re.search(r"(?i)\b(for|over|by|in|with|to)\b", en), en


class TestThePublishedExamplesRecordWhatWasTyped:
    r"""公開実例の `user_text` は、**シナリオに書かれた指示そのもの**であること。

    起票の実害は「注入が公開記録に漏れている」だったが、突き合わせると
    漏れていたのは注入だけではなかった（実測 2026-09-10、**34件**）:

    | 直す前 | 研究者が打った文 |
    |---|---|
    | `E19 に対して G を行列とする` | `G を行列とする`（**注入**、6件） |
    | `E1 と Sx_1 の交換子を計算して` | `H と Sx_1 の交換子を計算して`（**2a-sym の書き換え**、28件） |
    | `Expand E4` | `Expand F` |
    | `normal order E19` | `normal order c^\dagger_{1\uparrow} …` |

    2a-sym は研究者の記号名を式番号へ書き換える——解釈器に渡すためであって、
    **記録のためではない**。同じ欄に2つの意味を入れていたので、両方が一緒に
    漏れていた。綴りで探さず**シナリオの行と突き合わせる**のは、
    `E6 の sin(q) を…` のような研究者自身の `E\d+` を誤って数えないため。
    """

    @staticmethod
    def _typed(stem):
        import sys
        sys.path.insert(0, str(REPO / "scripts"))
        from generate_scenario_examples import _parse_scenario_md, SCENARIOS_DIR
        md = SCENARIOS_DIR / f"{stem}.md"
        parsed = _parse_scenario_md(md) if md.exists() else None
        if not parsed:
            return set()
        out = set()
        for cmd in parsed[1]:
            if isinstance(cmd, str):
                out.add(cmd.strip())
            elif isinstance(cmd, dict):
                for k in ("math", "note"):
                    if cmd.get(k):
                        out.add(str(cmd[k]).strip())
        return out

    def test_every_recorded_user_text_is_a_line_the_scenario_typed(self):
        stray, checked = [], 0
        for p in sorted((REPO / "domains").rglob("scenario_*.json")):
            typed = self._typed(p.stem)
            if not typed:
                continue
            data = json.loads(p.read_text(encoding="utf-8"))

            def walk(node):
                nonlocal checked
                if isinstance(node, dict):
                    u = node.get("user_text")
                    if (isinstance(u, str) and u.strip()
                            and node.get("kind") in ("user_input", "llm_supervision")):
                        checked += 1
                        if u.strip() not in typed:
                            stray.append(f"{p.stem}: {u.strip()[:70]}")
                    for v in node.values():
                        walk(v)
                elif isinstance(node, list):
                    for v in node:
                        walk(v)

            walk(data)
        assert checked > 500, f"{checked} 件しか見ていない——読めていない疑い"
        assert not stray, (
            "公開実例の user_text に、研究者が打っていない文字列がある"
            f"（{len(stray)} 件）:\n  " + "\n  ".join(stray[:20]))

    def test_no_published_user_text_carries_the_injected_prefix(self):
        """上の性質の、いちばん高くついた1つ —— 起票の当の症状。"""
        prefix = re.compile(r"^E\d+\s+に対して\s+")
        leaked = [f"{p.stem}: {u[:60]}"
                  for p in sorted((REPO / "domains").rglob("scenario_*.json"))
                  for u in re.findall(r'"user_text":\s*"([^"]*)"',
                                      p.read_text(encoding="utf-8"))
                  if prefix.match(u)]
        assert not leaked, leaked
