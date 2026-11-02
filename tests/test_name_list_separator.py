"""並んだ名前をどう区切るか —— 同じ問いに6箇所以上、全部ちがう答え。

宣言一族を測っていて出た（2026-09-05）。数えると、名前やE-idの**並び**を切る
規則がこれだけあり、どれも違う部分集合しか知らなかった:

| 場所 | 知っていた区切り |
|---|---|
| `decl_stmt[ja]` / `decl_stmt_wa[ja]` | `,` `、` `と` |
| `decl_stmt[en]` | `,` だけ |
| `solve_system_vars[ja]`（E-id） | `と` `,` `、` `，` |
| `solve_system_vars[en]`（E-id） | `,` `and` `, and` |
| `inv_transform_vars[ja]`（E-id） | `と` `,` `、` |
| `inv_transform_vars[en]`（E-id） | `,` だけ |
| `_solve_vars_from` | `and` を**別の `re.sub` で先に消して**から `[,、と\s]+` |

知らない区切りに当たったときの振る舞いが、どれも**成功を返す**（実測）:

    a、b および c をスカラーとする    → `c` **だけ**登録され confirmed
    a、b、c および d をスカラーとする  → `d` だけ
    Define a and b as scalars      → `b` だけ
    Define a, b and c as scalars   → `c` だけ（ごく普通の英語）
    Define a, b, and c as scalars  → `c` だけ（Oxford comma）
    … を x および y について解いて      → `vars: 'x, および, y'`（助詞が変数になり
                                       Maxima に届いて completed）

**宣言はこの系のほとんどの読みの判別材料**である（`^T` が転置か、`^{-1}` が
逆行列か、添字が index か名前か）。黙って落ちた名前は以降ずっと未宣言の自由変数
として扱われ、Julia セッションなら生の CAS 文言になる。

答えは `notebook/runtime/operand_boundary.split_name_list` と、パターン側の
`_fragments.list_sep_ja` / `list_sep_en` に1つずつ。置き場所が `operand_boundary`
なのは CLAUDE.md の指定（「この族は増える見込みだから次の判別子はそこへ」）。

**区切りを広げると、名前を拾う側の近似が露見した。** `backend_handler` は
区切りの手前を `re.findall(名前の綴り)` で走査していたので、`Define a and b as
scalars` が **`and` を宣言**した——素の findall は区切りの語そのものを名前として
拾う。区切りで切ってから拾う形にした。
"""
import asyncio

import pytest

import ui.notebook_bridge as nb
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.nl_patterns import nlp
from notebook.runtime.operand_boundary import split_name_list
from notebook.runtime.operation_validator import OperationValidator
from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline


class TestTheSplitterIsOneAnswer:
    @pytest.mark.parametrize("lang,text,want", [
        ("ja", "a、b、c", ["a", "b", "c"]),
        ("ja", "a, b, c", ["a", "b", "c"]),
        ("ja", "a と b と c", ["a", "b", "c"]),
        ("ja", "a，b，c", ["a", "b", "c"]),
        ("ja", "a、b および c", ["a", "b", "c"]),
        ("ja", "a、b、c および d", ["a", "b", "c", "d"]),
        ("ja", "a や b", ["a", "b"]),
        ("ja", "a 及び b", ["a", "b"]),
        ("en", "a, b, c", ["a", "b", "c"]),
        ("en", "a and b", ["a", "b"]),
        ("en", "a, b and c", ["a", "b", "c"]),
        ("en", "a, b, and c", ["a", "b", "c"]),
        ("en", "a & b", ["a", "b"]),
        ("ja", r"x_1, x_2, \lambda", ["x_1", "x_2", r"\lambda"]),
        ("en", "c(k,u), c'(-k,d)", ["c(k,u)", "c'(-k,d)"]),
    ])
    def test_it_splits(self, lang, text, want):
        assert split_name_list(text, lang) == want

    def test_a_single_name_is_itself(self):
        for lang in ("ja", "en"):
            assert split_name_list("omega_0", lang) == ["omega_0"]

    def test_it_is_not_narrowed_by_language(self):
        """`E1 と E2 for x and y` のような混じった文でも両方の区切りで切る。

        読みは言語で絞ってよいが、区切りの判定を絞ると相手の綴りを落とす。
        """
        assert split_name_list("a and b", "ja") == ["a", "b"]
        assert split_name_list("a および b", "en") == ["a", "b"]


class TestThePatternsUseTheOneAnswer:
    """パターン側も同じ区切りを使う —— 使っていなければマッチが最後の名前から
    始まり、手前の名前は**そもそも走査されない**。"""

    @pytest.mark.parametrize("lang,text", [
        ("ja", "a、b および c をスカラーとする"),
        ("ja", "a や b をスカラーとする"),
        ("en", "Define a and b as scalars"),
        ("en", "Define a, b, and c as scalars"),
    ])
    def test_the_declaration_spans_the_whole_list(self, lang, text):
        m = nlp("decl_stmt", lang).search(text)
        assert m is not None, text
        assert m.group(0).lstrip().startswith(("a", "Define a", "define a")), \
            (text, m.group(0), "マッチが名前の並びの途中から始まっている")

    @pytest.mark.parametrize("lang,text", [
        ("ja", "E1 と E2 の連立方程式を x および y について解いて"),
        ("en", "solve E1, E2 and E3 for x and y"),
    ])
    def test_the_system_spans_the_whole_list(self, lang, text):
        m = nlp("solve_system_vars", lang).search(text)
        assert m is not None, text
        assert m.group("eids").count("E") >= 2, (text, m.groupdict())


def _run(eng, sup, text):
    return asyncio.get_event_loop().run_until_complete(
        run_interpret_pipeline(
            InterpretRequest(tokens=[InputToken(type="text", value=text)]), eng, sup))


#: (言語, 綴り, 登録されるべき名前)。日英そろえてある。
DECLARATIONS = [
    ("ja", "a、b、c をスカラーとする", ["a", "b", "c"]),
    ("ja", "a と b と c をスカラーとする", ["a", "b", "c"]),
    ("ja", "a，b，c をスカラーとする", ["a", "b", "c"]),
    ("ja", "a、b および c をスカラーとする", ["a", "b", "c"]),
    ("ja", "a、b、c および d をスカラーとする", ["a", "b", "c", "d"]),
    ("ja", "a や b をスカラーとする", ["a", "b"]),
    ("en", "Define a, b, c as scalars", ["a", "b", "c"]),
    ("en", "Define a and b as scalars", ["a", "b"]),
    ("en", "Define a, b and c as scalars", ["a", "b", "c"]),
    ("en", "Define a, b, and c as scalars", ["a", "b", "c"]),
]


@pytest.mark.parametrize("lang,text,names", DECLARATIONS)
def test_every_declared_name_is_registered(lang, text, names):
    """画面まで —— 落ちた名前は「失敗」ではなく**成功として**消えていた。"""
    nb.start_engine(title="name_list", mode="standard", language=lang,
                    domains=["classical_mechanics"])
    try:
        eng = nb.get_engine()
        sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                            domains=["classical_mechanics"], primary_backend="maxima")
        resp = _run(eng, sup, text)
        assert resp.get("status") in ("confirmed", "declared", "completed"), \
            (text, resp.get("status"), resp.get("error"))
        registered = set(eng.symbol_context.registry)
        assert set(names) <= registered, (text, sorted(registered), "落ちた名前がある")
        assert "and" not in registered and "および" not in registered, \
            (text, sorted(registered), "区切りの語が名前として登録されている")
    finally:
        nb.stop_engine()


@pytest.mark.parametrize("lang,text", [
    ("ja", "E1 と E2 の連立方程式を x および y について解いて"),
    ("en", "solve E1, E2 for x and y"),
])
def test_a_separator_never_becomes_a_variable(lang, text):
    nb.start_engine(title="name_list_solve", mode="standard", language=lang,
                    domains=["classical_mechanics"])
    try:
        eng = nb.get_engine()
        sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                            domains=["classical_mechanics"], primary_backend="maxima")
        _run(eng, sup, "x + y = 3")
        _run(eng, sup, "x - y = 1")
        resp = _run(eng, sup, text)
        assert resp.get("status") == "completed", (text, resp.get("error"))
        args = [getattr(s, "operation_args", None) for s in eng.steps.values()
                if getattr(s, "operation", None) == "solve_system"]
        assert args, text
        assert args[-1].get("vars") == "x, y", (text, args[-1])
    finally:
        nb.stop_engine()
