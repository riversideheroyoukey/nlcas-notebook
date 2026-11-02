r"""トレースは一度も計算されていなかった —— 未定義関数は noun のまま成功で返る。

実測 2026-09-06。`backends/maxima.full.json` の `trace` テンプレートは
`mattrace({expr})` だが、この Maxima ビルドの `mattrace` は `nchrpl` パッケージに
あり、**ロードしないと定義されていない**。Maxima は未定義関数を noun のまま返して
`ok=True` を返すので、ステップは `completed` になる:

    cmd:  mattrace(matrix([1, 2], [3, 4]))
    返り: mattrace(matrix([1,2],[3,4]))      ← 5 ではない

`\it arcsin` と同じ形である（未定義なので計算が静かに何もしない）。
**テンプレート関数 32個を全部呼んで測ったところ、これが唯一。**

| | 直す前 | 直した後 |
|---|---|---|
| 操作（`E1 のトレースを求めて` ほか 10綴り） | 0/10 | 10/10 |
| 入力記法（`\text{tr}` `\mathrm{tr}` `\operatorname{tr}` × `tr`/`Tr`） | 0/6 | 6/6 |
| 表示（記号行列で noun のまま残る場合） | `mattrace\left(N+M\right)` | `\mathrm{tr}\left(N+M\right)` |

**入力記法が落ちていた仕組みは「同じ問いに2箇所」である。** `unwrap_command` の
`keep` は**定数表 `_GREEK_MAP` だけ**を受け取っていたので、`\text{tr}` は組版の
ラッパとして先に外れて `tr` になり、その後で `\text{tr}` を探す関数表
（`_TRIG_MAP`）が空振りした。`unwrap_command` の註は「ここで決めるとこの
プロジェクトの再発欠陥になる」と書いており、まさにそれが起きていた。
同じ表の中で `\det` は通り `\mathrm{det}` は落ちる、という割れになっていた。

**綴りは並べずに生成する**（`upright_operator_spellings`）。どのラッパで書くかは
研究者と TeX エディタの好みでしかなく、列挙すると片方に足して他方を忘れる。

**コーパス0行・シナリオ0件。** だから 7500件のテストでも実例の再生成でも出ず、
別の語（`take`）を測っている途中で偶然踏んだ。
"""
import asyncio
import tempfile
from pathlib import Path

import pytest

import ui.notebook_bridge as nb
from notebook.runtime.latex_typography import upright_operator_spellings
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.maxima_adapter import MaximaAdapter
from notebook.runtime.operation_validator import OperationValidator
from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline

DOM = ["classical_mechanics"]
PREP = {"en": ["Define M as a matrix", "M = [1, 2; 3, 4]"],
        "ja": ["M を行列とする", "M = [1, 2; 3, 4]"]}

#: トレースを求める綴り。**どれも 5 を返すこと。**
OPS = [
    ("en", "compute the trace of E1"),
    ("en", "take the trace of E1"),
    ("en", "find the trace of E1"),
    ("en", "the trace of E1"),
    ("en", "trace of E1"),
    ("ja", "E1 のトレースを計算して"),
    ("ja", "E1 のトレースを求めて"),
    ("ja", "E1 のトレースを取って"),
    ("ja", "E1 のトレース"),
    ("ja", "E1 の対角和を求めて"),
]
#: 入力記法。ラッパの違いは組版の好みでしかない。
TRACE_NOTATION = [r"\text{tr}(M)", r"\mathrm{tr}(M)", r"\operatorname{tr}(M)",
                  r"\text{Tr}(M)", r"\mathrm{Tr}(M)", r"\operatorname{Tr}(M)"]
DET_NOTATION = [r"\det(M)", r"\mathrm{det}(M)", r"\text{det}(M)",
                r"\operatorname{det}(M)"]


def _run(eng, sup, text):
    return asyncio.get_event_loop().run_until_complete(
        run_interpret_pipeline(
            InterpretRequest(tokens=[InputToken(type="text", value=text)]), eng, sup))


def _session(lang):
    nb.start_engine(title="tr", mode="standard", language=lang, domains=DOM)
    eng = nb.get_engine()
    sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                        domains=DOM, primary_backend="maxima")
    return eng, sup


@pytest.fixture
def handler():
    nb.start_engine(title="h", mode="standard", language="en", domains=DOM)
    try:
        yield nb.get_engine()._handler
    finally:
        nb.stop_engine()


class TestTheTraceIsActuallyComputed:
    @pytest.mark.parametrize("lang,text", OPS)
    def test_every_spelling_gives_five(self, lang, text):
        eng, sup = _session(lang)
        try:
            for p in PREP[lang]:
                _run(eng, sup, p)
            r = _run(eng, sup, text)
            assert r.get("status") == "completed", (text, r.get("error"))
            got = (r.get("latex") or "").split("=")[-1].strip()
            assert got == "5", (text, r.get("latex"))
        finally:
            nb.stop_engine()

    def test_the_adapter_defines_mattrace(self):
        """**素の Maxima は `mattrace` を知らない。** 知らない関数は noun のまま
        返り `ok=True` になるので、`ok` を見ても計算されたかは分からない。"""
        a = MaximaAdapter()
        a.start()
        try:
            r = a.execute("mattrace(matrix([1,2],[3,4]))")
            assert r.ok and str(r.value).strip() == "5", r.value
            r2 = a.execute("mattrace(matrix([a,b],[c,d]))")
            assert r2.ok and str(r2.value).strip() in ("d+a", "a+d"), r2.value
        finally:
            a.close()


class TestTheNotationReachesTheFunction:
    @pytest.mark.parametrize("src", TRACE_NOTATION)
    def test_every_wrapper_spelling_of_tr(self, handler, src):
        assert handler.latex_to_canonical(src) == "mattrace(M)", src

    @pytest.mark.parametrize("src", DET_NOTATION)
    def test_every_wrapper_spelling_of_det(self, handler, src):
        r"""同じ表の中で `\det` は通り `\mathrm{det}` は落ちていた。"""
        assert handler.latex_to_canonical(src) == "determinant(M)", src

    def test_the_spellings_are_generated_not_listed(self):
        """ラッパの一覧は1箇所（`_FONT_CMDS`）にしかない。"""
        got = upright_operator_spellings({"tr": "mattrace"})
        for cmd in ("text", "mathrm", "operatorname", "mathit", "textrm"):
            assert got[f"\\{cmd}{{tr}}"] == "mattrace", cmd

    @pytest.mark.parametrize("src,want", [
        (r"\mathrm{i}", "%i"),          # 定数（バックエンドの表が読む）
        (r"\text{i}", "%i"),
        (r"\mathrm{g}(x)", "g(x)"),     # ただの組版ラッパ
        (r"\mathrm{tra}(x)", "tra(x)"), # 部分一致で拾わない
    ])
    def test_what_must_not_change(self, handler, src, want):
        assert handler.latex_to_canonical(src) == want, src


class TestTheDisplayIsTheResearchersNotation:
    def test_a_symbolic_trace_publishes_tr_not_mattrace(self):
        r"""記号行列では noun のまま残る——そこで CAS の内部名が公開されていた。"""
        eng, sup = _session("en")
        try:
            for p in ["Define M as a matrix", "Define N as a matrix", "P = M + N"]:
                _run(eng, sup, p)
            r = _run(eng, sup, "compute the trace of E1")
            assert r.get("status") == "completed", r.get("error")
            latex = r.get("latex") or ""
            assert "mattrace" not in latex, latex
            assert r"\mathrm{tr}" in latex, latex
        finally:
            nb.stop_engine()

    def test_the_round_trip_closes(self, handler):
        """公開した綴りは**入力として読み直せる**こと。"""
        assert handler.latex_to_canonical(
            r"\mathrm{tr}\left(N+M\right)") == "mattrace(N+M)"


class TestTheSnapshotIsADeltaFromPristine:
    r"""パッケージを読むと `functions` に入る —— スナップショットが壊れた。

    実測 2026-09-06、`load(nchrpl)` を初期化に足した直後にスナップショット系
    8件が落ちた。`save_snapshot` は `functions` を歩いて**定義本体**を書き出す
    ので、`mattrace` / `ncharpoly` の複数行の `fundef` がファイルに入り、行ごとに
    実行する `load_snapshot` が復元できなくなる。名前が `_` で始まるかどうかで
    足りていたのは、素の Maxima に利用者の関数が1つも無かったからにすぎない。
    """

    def test_a_loaded_package_is_not_saved_as_user_state(self):
        with tempfile.NamedTemporaryFile(suffix=".mac", delete=False) as tf:
            path = Path(tf.name)
        a = MaximaAdapter()
        a.start()
        try:
            a.execute("myfunc(z) := z^2 + 3")
            assert a.save_snapshot(path)
            text = path.read_text(encoding="utf-8")
            assert "myfunc" in text, text
            for pkg_fn in ("mattrace", "ncharpoly"):
                assert pkg_fn not in text, (pkg_fn, text[:400])
        finally:
            a.close()
        b = MaximaAdapter()
        b.start()
        try:
            assert b.load_snapshot(path), path.read_text(encoding="utf-8")[:400]
            assert str(b.execute("myfunc(2)").value).strip() == "7"
            # パッケージは復元ではなく初期化で入る——両方で使えること。
            assert str(b.execute("mattrace(matrix([1,2],[3,4]))").value).strip() == "5"
        finally:
            b.close()
            path.unlink(missing_ok=True)
