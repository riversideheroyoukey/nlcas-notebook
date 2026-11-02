"""演算子代入の族 —— 同じ問いに**6つ**の規則、動詞も助詞の順序も別々だった。

worklist の `代入`/`substitute`（コーパス 21/20行）の残り。数えると規則は6つあり、
5つは `i18n/nl_patterns.json`、**6つ目は `julia_backend_handler.py` の source 内**に
あった（NL のパターンなのに）。実測 2026-09-05、13綴り中**6つが落ちた**:

| 打った文 | 直す前 |
|---|---|
| `E1 の a(1) を E2 に置換して` | failed（`置き換え` は通る） |
| `In E1, replace a(1) with E2` | failed（`substitute … for …` のみ） |
| `operator_substitute E1 a(1) with E2` | **式として登録**（②の形） |
| `E1 の a(1) を E2 に、a'(1) を E3 に置き換えて` | operator_substitute に届いて**失敗** |
| `E1 の a(1) を E2 に、a'(1) を E3 に置換して` | failed |
| `In E1, replace a(1) with E2 and a'(1) with E3` | failed |

**割れていたのは動詞だけではない。** 助詞の順序が動詞で変わる——代入形は
`X に Y を`、置き換え形は `X を Y に`——のに、`multi`/`pair`/`spin_pair` は
**代入形の順序だけ**を知っていた。だから動詞を揃えるだけでは足りず、`置き換え` の
2つ組は `single` に落ちて「1つの代入＋ゴミの置換先」（`rhs = "E2 に、a'(1) を E3"`）
と読まれ、CAS で失敗していた。研究者が見るのは CAS の文句で、「1つの代入として
読みました」ではない。

**英語側は既に正しい設計をしていた。** source のコメントいわく「英語は3つの形を
同じ綴りで書くので、**1つのパターンが読み、形が行き先を決める**」。日本語が同じ
問いに3パターンを当てているのは、その設計に追いついていないため（下の「残り」）。

**CAS 風の綴りは削除した。** `operator_substitute_single[en]` の綴りは
`operator_substitute E1 a(1) with E2`——操作名そのもので、シナリオ文書が自ら
「不自然で CAS 構文のような綴り」と書いており、原則#1 に反する。しかも実測で
**動いていなかった**（式として登録）。自然な形（`operator_substitute_in_form`）は
動く。単に誤った別名は維持せず削除する、という `laplace` の先例。
"""
import asyncio

import pytest

import ui.notebook_bridge as nb
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.nl_patterns import nlp, pick_group
from notebook.runtime.operation_validator import OperationValidator
from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline


#: (言語, 綴り, 代入の数)。日英そろえてある。
SPELLINGS = [
    ("ja", "E1 の a(1) に E2 を代入する", 1),
    ("ja", "E1 の a(1) に E2 を代入して", 1),
    ("ja", "E1 の a(1) を E2 に置き換えて", 1),
    ("ja", "E1 の a(1) を E2 に置換して", 1),
    ("ja", "E1 の a(1) に E2 を、a'(1) に E3 を代入する", 2),
    ("ja", "E1 の a(1) に E2 を、a'(1) に E3 を代入して", 2),
    ("ja", "E1 の a(1) を E2 に、a'(1) を E3 に置き換えて", 2),
    ("ja", "E1 の a(1) を E2 に、a'(1) を E3 に置換して", 2),
    ("en", "In E1, substitute E2 for a(1)", 1),
    ("en", "In E1, replace a(1) with E2", 1),
    ("en", "In E1, substitute E2 for a(1) and E3 for a'(1)", 2),
    ("en", "In E1, replace a(1) with E2 and a'(1) with E3", 2),
]


# **動詞の一覧は 2026-09-12 に演算子の族の外へ出た**——`op_subst_verb_ja` は
# `ja_subst_verb` に改名され、`substitute_into` / `scalar_substitute_op` も読む。
# 「どの鍵が表を読んでいるか」と「活用が全部入っているか」を数える検査は、
# 上位集合として `tests/test_substitution_verb_is_one_table.py` に在る（ここに
# 部分集合を残すと、同じ問いに2箇所で答えることになる）。


class TestBothParticleOrdersAreRead:
    """助詞の順序は動詞で変わる —— 動詞を揃えるだけでは足りなかった。"""

    @pytest.mark.parametrize("text", [
        "E1 の a(1) に E2 を、a'(1) に E3 を代入する",
        "E1 の a(1) を E2 に、a'(1) を E3 に置き換えて",
        "E1 の a(1) を E2 に、a'(1) を E3 に置換して",
    ])
    def test_multi_reads_all_seven_fields(self, text):
        m = nlp("operator_substitute_multi", "ja").search(text)
        assert m is not None, text
        got = {s: pick_group(m, s) for s in ("eid", "op1", "idx1", "rhs1",
                                             "op2", "idx2", "rhs2")}
        assert got == {"eid": "E1", "op1": "a", "idx1": "1", "rhs1": "E2",
                       "op2": "a'", "idx2": "1", "rhs2": "E3"}, (text, got)

    @pytest.mark.parametrize("text", [
        "E1 の c(k) に E2 を、c(-k) に E3 を代入する",
        "E1 の c(k) を E2 に、c(-k) を E3 に置き換えて",
    ])
    def test_the_nambu_pair_is_read_by_the_same_pattern(self, text):
        """対も同じ1パターンが読む——**行き先を決めるのは形**（下のクラス）。"""
        m = nlp("operator_substitute_multi", "ja").search(text)
        assert m is not None, text
        assert pick_group(m, "op1") == "c" and pick_group(m, "op2") == "c", text
        assert pick_group(m, "idx1") == "k" and pick_group(m, "idx2") == "-k", text
        assert pick_group(m, "rhs1") == "E2" and pick_group(m, "rhs2") == "E3", text

    @pytest.mark.parametrize("text", [
        "In E1, substitute E2 for a(1) and E3 for a'(1)",
        "In E1, replace a(1) with E2 and a'(1) with E3",
    ])
    def test_english_reads_both_word_orders(self, text):
        m = nlp("operator_substitute_multi", "en").search(text)
        assert m is not None, text
        got = {s: pick_group(m, s) for s in
               ("eid", "op1", "idx1", "rhs1", "op2", "idx2", "rhs2")}
        assert got == {"eid": "E1", "op1": "a", "idx1": "1", "rhs1": "E2",
                       "op2": "a'", "idx2": "1", "rhs2": "E3"}, (text, got)


class TestAnOperatorWithTwoIndicesReadsInBothLanguages:
    r"""`c(k,u)` のようにコンマを含む添字 —— **英語だけが落としていた**。

    実測 2026-09-10（JULIA-HANDLER-LANG）。`multi[en]` の添字が `[^,)]`
    ——コンマで止まる——なので `c(k,u)` に当たらず、同じ演算子・`substitute … for …`
    の綴りだけを別の鍵（`operator_substitute_spin_pair`）が拾って埋めていた。
    埋まらない綴りは単数形に落ち、**2つ目の代入が黙って捨てられて** `completed`
    が返っていた:

    | 綴り | 直す前の規則数 |
    |---|---|
    | `In E1, substitute E2 for c(k,u) and E3 for d(-k,d)`（違う演算子） | **1**（要 2） |
    | `In E1, replace c(k,u) with E2 and c(-k,d) with E3` | **1**（要 2） |
    | `In E1, replace c(k,u) with E2 and d(-k,d) with E3` | **1**（要 2） |

    日本語の `multi[ja]` は最初から `[^)]` だった。クラスを揃えて英語 3/6 → 6/6。
    """

    @pytest.mark.parametrize("lang,text", [
        ("ja", "E1 の c(k,u) に E2 を、c(-k,d) に E3 を代入する"),
        ("ja", "E1 の c(k,u) に E2 を、d(-k,d) に E3 を代入する"),
        ("en", "In E1, substitute E2 for c(k,u) and E3 for c(-k,d)"),
        ("en", "In E1, substitute E2 for c(k,u) and E3 for d(-k,d)"),
        ("en", "In E1, replace c(k,u) with E2 and c(-k,d) with E3"),
        ("en", "In E1, replace c(k,u) with E2 and d(-k,d) with E3"),
    ])
    def test_both_rules_survive(self, lang, text):
        m = nlp("operator_substitute_multi", lang).search(text)
        assert m is not None, text
        i1, i2 = pick_group(m, "idx1"), pick_group(m, "idx2")
        assert i1 and "," in i1 and i2 and "," in i2, (text, i1, i2)
        assert pick_group(m, "rhs1") == "E2" and pick_group(m, "rhs2") == "E3", text

    @pytest.mark.parametrize("lang,text", [
        ("ja", "E1 の c(k,u) に E2 を、d(-k,d) に E3 を代入する"),
        ("en", "In E1, substitute E2 for c(k,u) and E3 for d(-k,d)"),
    ])
    def test_two_different_operators_keep_their_names(self, lang, text):
        m = nlp("operator_substitute_multi", lang).search(text)
        assert m is not None, text
        assert pick_group(m, "op1") == "c" and pick_group(m, "op2") == "d", text


class TestTheIndexRelabelUsesOneShapeForBothLanguages:
    r"""`relabel_index` は日英で**同じ群名**——`nl_patterns.py` の設計そのもの。

    2026-09-10 まで位置引数で、しかも**並びが日英で逆**（ja は eid,old,new／
    en は old,new,eid）だったので、読む側が言語ごとに2回呼んで群番号を書き分けて
    いた。群名を揃えると呼び出しは1回になり、並びを取り違える余地が消える。
    """

    @pytest.mark.parametrize("lang,text", [
        ("ja", "E1 の k を -k に指標変換して"),
        ("ja", "E1 の k を -k にリラベルして"),
        ("en", "relabel k to -k in E1"),
    ])
    def test_the_same_group_names_in_both_languages(self, lang, text):
        m = nlp("relabel_index", lang).search(text)
        assert m is not None, text
        assert m.group("eid") == "E1", text
        assert m.group("old") == "k" and m.group("new") == "-k", text

    def test_one_call_serves_both(self):
        """`nlp_search` は全言語を試す——読む側に言語の分岐が要らない。"""
        from notebook.runtime.nl_patterns import nlp_search
        for text in ("E1 の k を -k に指標変換して", "relabel k to -k in E1"):
            m = nlp_search("relabel_index", text)
            assert m and m.group("eid") == "E1" and m.group("old") == "k", text


class TestTheShapeDecidesWhereItGoes:
    """2つ組は**1パターン**が読み、形が行き先を決める —— 日英で同じ答え。

    日本語はここが3パターン（spin_pair / pair / multi）で、英語は最初から1つ
    だった。英語側のコメントが理由を書いている——「英語は3つの形を同じ綴りで
    書くので、1つのパターンが読み、形が行き先を決める」。日本語がその設計に
    追いついていなかっただけで、3つに分ける理由は無かった（2026-09-05）。
    """

    @staticmethod
    def _read(handler, text):
        for lang in ("ja", "en"):
            m = nlp("operator_substitute_multi", lang).search(text)
            if m:
                r = handler._two_rule_substitute_result(m, text, False)
                if r:
                    return r
        return None

    @pytest.fixture
    def handler(self):
        from notebook.runtime.julia_backend_handler import JuliaBackendHandler
        from notebook.models.symbol_context import SymbolContext
        from notebook.models.quantity import Quantity, MathType
        h = JuliaBackendHandler.__new__(JuliaBackendHandler)
        sc = SymbolContext()
        for n in ("a", "b", "c"):
            sc.register(n, Quantity(name=n, kind="operator",
                                    math_type=MathType(kind="scalar")))
        h._context = sc
        return h

    @pytest.mark.parametrize("text", [
        "E1 の c(k) に E2 を、c(-k) に E3 を代入する",
        "E1 の c(k) を E2 に、c(-k) を E3 に置き換えて",
        "In E1, substitute E2 for c(k) and E3 for c(-k)",
    ])
    def test_the_nambu_pair_is_its_own_operation(self, handler, text):
        """同じ演算子の ±X は対（別の操作）——形だけで決まる。"""
        r = self._read(handler, text)
        assert r is not None and r.operation == "operator_substitute_pair", text
        assert r.args == {"expr": "E1", "op_name": "c", "formal_index": "k",
                          "rhs": "E2", "rhs_neg": "E3"}, (text, r.args)

    @pytest.mark.parametrize("text", [
        "E1 の c(k,u) に E7 を、c(-k,d) に E8 を代入する",
        "In E1, substitute E7 for c(k,u) and E8 for c(-k,d)",
    ])
    def test_a_spin_argument_becomes_a_spin_label(self, handler, text):
        r = self._read(handler, text)
        assert r is not None and r.operation == "operator_substitute", text
        assert r.args["op_rules"] == [
            {"op_name": "c", "formal_index": "k", "spin_label": "u", "rhs": "E7"},
            {"op_name": "c", "formal_index": "-k", "spin_label": "d", "rhs": "E8"},
        ], (text, r.args)

    @pytest.mark.parametrize("text", [
        "E1 の a(1) に E2 を、b(2) に E3 を代入する",
        "E1 の a(1) に E2 を、a'(1) に E3 を代入する",
    ])
    def test_anything_else_is_two_independent_rules(self, handler, text):
        r = self._read(handler, text)
        assert r is not None and r.operation == "operator_substitute", text
        assert len(r.args["op_rules"]) == 2, (text, r.args)
        assert all("spin_label" not in rule for rule in r.args["op_rules"]), r.args

    def test_different_operators_at_plus_minus_are_not_a_pair(self, handler):
        """Nambu の対は**同じ演算子**の ±X —— `a(k)` と `b(-k)` は独立な2規則。

        変異試験で見つけた穴（`op1 == op2` を外しても落ちなかった）。
        """
        r = self._read(handler, "E1 の a(k) に E2 を、b(-k) に E3 を代入する")
        assert r is not None and r.operation == "operator_substitute", r
        assert r.args["op_rules"] == [
            {"op_name": "a", "formal_index": "k", "rhs": "E2"},
            {"op_name": "b", "formal_index": "-k", "rhs": "E3"},
        ], r.args

    def test_an_undeclared_name_is_not_an_operator(self, handler):
        """宣言されていない名前を演算子代入として読まない。

        変異試験で見つけた穴（宣言の確認を外しても落ちなかった）。`演算子代入`
        と明示されていればその限りではない——それは研究者が書いたことだから。
        """
        assert self._read(handler, "E1 の zz(1) に E2 を、yy(2) に E3 を代入する") is None
        r = self._read(handler, "E1 の zz(1) に E2 を、yy(2) に E3 を演算子代入する")
        assert r is not None and r.operation == "operator_substitute", r

    def test_the_two_languages_reach_the_same_answer(self, handler):
        ja = self._read(handler, "E1 の c(k,u) に E7 を、c(-k,d) に E8 を代入する")
        en = self._read(handler, "In E1, substitute E7 for c(k,u) and E8 for c(-k,d)")
        assert (ja.operation, ja.args) == (en.operation, en.args)

    def test_the_three_japanese_patterns_are_one(self):
        """`pair[ja]` / `spin_pair[ja]` は multi に吸収された。"""
        import json, pathlib
        d = json.loads(pathlib.Path("i18n/nl_patterns.json").read_text(encoding="utf-8"))
        assert "ja" not in d.get("operator_substitute_spin_pair", {})
        assert "ja" not in d.get("operator_substitute_pair", {})

    def test_the_english_side_is_one_pattern_too(self):
        """**英語も2つあった**（2026-09-10、JULIA-HANDLER-LANG）。

        `spin_pair[en]` が在ったのは綴りのためではなく、`multi[en]` の添字が
        `[^,)]`——コンマで止まる——で `c(k,u)` に当たらなかったから。クラスを
        日本語側（`[^)]`）に揃えたら 1綴りも増やさなくなったので消した。
        """
        import json, pathlib
        d = json.loads(pathlib.Path("i18n/nl_patterns.json").read_text(encoding="utf-8"))
        assert "operator_substitute_spin_pair" not in d
        assert "[^,)]" not in d["operator_substitute_multi"]["en"], (
            "添字の文字クラスが日本語側と揃っていない")


class TestTheCasShapedSpellingIsGone:
    """操作名そのものを綴らせるのは原則#1 に反し、しかも動いていなかった。"""

    def test_it_is_not_a_pattern_any_more(self):
        import json, pathlib
        d = json.loads(pathlib.Path("i18n/nl_patterns.json").read_text(encoding="utf-8"))
        assert "en" not in d["operator_substitute_single"]

    def test_the_natural_form_is_the_one_that_works(self):
        for text in ("In E1, substitute E2 for a(1)", "In E1, replace a(1) with E2"):
            assert nlp("operator_substitute_in_form", "en").search(text), text


def _run(eng, sup, text):
    return asyncio.get_event_loop().run_until_complete(
        run_interpret_pipeline(
            InterpretRequest(tokens=[InputToken(type="text", value=text)]), eng, sup))


@pytest.fixture(scope="module", params=["ja", "en"])
def session(request):
    """`a` をボソン演算子、E1 = a'(1) a(1)、E2 = 2 a(1)、E3 = 3 a'(1)。

    設定はコーパスの `squeezed_state` を簡略化したもの。
    """
    lang = request.param
    nb.start_engine(title="opsub_family", mode="second_quantization",
                    language=lang, domains=["condensed_matter"])
    eng = nb.get_engine()
    sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                        domains=["condensed_matter"], primary_backend="julia_full")
    _run(eng, sup, "a をボソン演算子として定義して" if lang == "ja"
         else "define a as a boson operator")
    _run(eng, sup, "a'(1) a(1)")
    _run(eng, sup, "2 * a(1)")
    _run(eng, sup, "3 * a'(1)")
    yield lang, eng, sup
    nb.stop_engine()


def test_every_spelling_reaches_the_operation(session):
    """画面まで —— 落ちていた綴りは「失敗」か、**式として登録**されていた。"""
    lang, eng, sup = session
    for spell_lang, text, _n in SPELLINGS:
        if spell_lang != lang:
            continue
        n_before = len(eng.expressions)
        resp = _run(eng, sup, text)
        assert resp.get("status") == "completed", (text, resp.get("status"),
                                                   resp.get("error"))
        assert resp.get("operation") == "operator_substitute", (text, resp.get("operation"))
        assert len(eng.expressions) - n_before == 1, (text, "式が余分に登録された")
