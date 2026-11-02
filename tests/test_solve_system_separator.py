r"""連立の並びも1つの読み —— 三十四問目で数え漏らした鍵。

実測 2026-09-07。三十四問目で `inv_transform` / `verify_canon` / `canon_constraint`
の**式番号の並び**を共有 fragment（`list_sep_ja` / `list_sep_en`）へ寄せたが、
`solve_system` は数え漏らしていた。兄弟の `solve_system_vars` は既に共有を使って
いるので、**変数を書けば通り、書かないと落ちる**という形になっていた:

| 打った文 | 直す前 |
|---|---|
| `E1 **および** E2 の連立方程式を解いて` | failed |
| `E1 **ならびに** E2 の連立方程式を解いて` | failed |
| `E1 **や** E2 の連立方程式を解いて` | failed |
| `solve E1 **&** E2` | 「`solve equation` と解釈しました」（別の操作へ） |
| `E1 および E2 を x と y について解いて` | ✔（兄弟は共有を読む） |

**「変数を書かない連立は訊く」のは正しい挙動。** 変数を書かずに連立を解けと言われ
たら「どの変数について解きますか？（例: x, y）」と訊く——だから成功の判定は
`needs_confirmation` である。最初 `completed` を期待して 4/9 と測ってしまった。

**同じ穴を数え漏らさないために、全鍵を機械的に見る。** 「`E\d+` の並びを自前の
区切りで書いている鍵」を数えて 0 であることを直接固定した。
"""
import asyncio
import json
import pathlib
import re

import pytest

import ui.notebook_bridge as nb
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operation_validator import OperationValidator
from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline

CM = ["classical_mechanics"]
_P = json.loads(pathlib.Path("i18n/nl_patterns.json").read_text())

#: 変数を書かない連立。**「どの変数か」を訊くのが正しい。**
ASKS = [
    ("ja", "E1 と E2 の連立方程式を解いて"),
    ("ja", "E1、E2 の連立方程式を解いて"),
    ("ja", "E1 および E2 の連立方程式を解いて"),
    ("ja", "E1 ならびに E2 の連立方程式を解いて"),
    ("ja", "E1 や E2 の連立方程式を解いて"),
    ("en", "solve E1 and E2"),
    ("en", "solve E1, E2"),
    ("en", "solve E1 & E2"),
    ("en", "solve E1, and E2"),
]
#: 変数を書く連立。**x=2, y=1 が出ること。**
SOLVES = [
    ("ja", "E1 と E2 を x と y について解いて"),
    ("ja", "E1 および E2 を x と y について解いて"),
    ("ja", "E1 や E2 を x と y について解いて"),
    ("en", "solve E1 and E2 for x and y"),
    ("en", "solve E1 & E2 for x, y"),
    ("en", "solve E1, and E2 for x, y"),
]


def _ask(lang, prep, text):
    nb.start_engine(title="ss", mode="standard", language=lang, domains=CM)
    try:
        eng = nb.get_engine()
        sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                            domains=CM, primary_backend="maxima")

        def run(x):
            return asyncio.get_event_loop().run_until_complete(
                run_interpret_pipeline(
                    InterpretRequest(tokens=[InputToken(type="text", value=x)]),
                    eng, sup))
        for p in prep:
            run(p)
        r = run(text)
        regs = r.get("registered_expressions") or []
        return r, " | ".join((e.get("latex") or "") for e in regs)
    finally:
        nb.stop_engine()


class TestTheSystemSeparator:
    @pytest.mark.parametrize("lang,text", ASKS)
    def test_without_variables_it_asks(self, lang, text):
        r, _ = _ask(lang, ["x + y = 3", "x - y = 1"], text)
        assert r.get("status") == "needs_confirmation", (text, r.get("status"),
                                                         r.get("error"))
        assert r.get("operation") == "solve_system", (text, r.get("operation"))

    @pytest.mark.parametrize("lang,text", SOLVES)
    def test_with_variables_it_solves(self, lang, text):
        r, body = _ask(lang, ["x + y = 3", "x - y = 1"], text)
        assert r.get("status") == "completed", (text, r.get("error"))
        assert "2" in body and "1" in body, (text, body)


class TestNoKeySpellsTheSeparatorItself:
    r"""**同じ穴を数え漏らさない。** 三十四問目は3鍵を直して `solve_system` を残した。"""

    def test_every_expression_list_reads_the_shared_fragment(self):
        bad = []
        for k, v in _P.items():
            if k.startswith("_") or not isinstance(v, dict):
                continue
            for lang, pat in v.items():
                if lang.startswith("_") or not isinstance(pat, str):
                    continue
                for m in re.finditer(r"E\\d\+\(\?:([^)]*?)E\\d\+\)[+*]", pat):
                    if "list_sep" not in m.group(1):
                        bad.append(f"{k}[{lang}]: {m.group(1)[:40]}")
        assert not bad, ("式番号の並びを自前の区切りで書いている鍵:\n  "
                         + "\n  ".join(sorted(set(bad))))

    @pytest.mark.parametrize("key,lang,frag", [
        ("solve_system", "ja", "list_sep_ja"),
        ("solve_system", "en", "list_sep_en"),
        ("solve_system_vars", "ja", "list_sep_ja"),
        ("solve_system_vars", "en", "list_sep_en"),
    ])
    def test_the_pair_reads_the_same_fragment(self, key, lang, frag):
        assert "{{" + frag + "}}" in _P[key][lang], (key, lang)
