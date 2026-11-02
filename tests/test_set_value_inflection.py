"""「〜とする / と置く」の活用 —— 6箇所が別々に、どれも活用を持っていなかった。

十八問目は `ja_decl_suffix` の中の同じ穴を直した（素の `とする` だけ活用形を
持たない）。数えると**同じ穴が他に6箇所**あった——`fp_subst`・`laplace_define`・
`name_assign`・`rhs_set`・`subst_replace_alt`・`pb_pair_momentum` が「〜とする」
「〜と置く」を自前で綴り、**5通りの部分集合**に分かれていて、どれも
`とします`・`とせよ`・`としろ` を知らなかった（実測 2026-09-05、H-5z の右辺設定で
6綴り中3落ち）。

**共有するのは活用だけで、動詞の集合ではない。** 動詞ごと共有する案を先に試して
**実測で覆った**: `subst_replace_alt` に動詞 `とする` を与えると H-5z 自身の文
（`E20 の右辺を 0 とする`）を横取りし、`fp_subst` に `とおく` を与えると
`laplace_define` の文を横取りする——コーパス10行、どれも**他のパターンが正しく
処理している行**だった。**パターン水準の「獲得」は、パイプラインでは横取りに
よる損失になりうる。** 活用だけなら 35綴り獲得・損失0・コーパス差0。

**同じ測定で英語側の割れも出た。** H-5z の英語は `set the right-hand side of E1
to X` の**1綴りだけ**で、`let … be` / `put … to` / `take … to be` は落ち、
`regard … as` は**式として登録**されていた（②の形——`regard` が
`instruction_marker[en]` に無い）。今度は日本語のほうが広いという逆向きの割れ。
"""
import asyncio

import pytest

import ui.notebook_bridge as nb
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.nl_patterns import fragment, nlp, pick_group
from notebook.runtime.operation_validator import OperationValidator
from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline


def _reaches_a_shared_inflection(key: str) -> bool:
    """`key` の日本語パターンが、`{{…}}` 参照を辿って活用の表に届くか。"""
    import json
    import pathlib
    import re
    data = json.loads(pathlib.Path("i18n/nl_patterns.json").read_text(encoding="utf-8"))
    frags = data.get("_fragments", {})
    want = {"ja_suru_infl", "ja_oku_infl"}
    seen: set[str] = set()
    stack = [data[key]["ja"]]
    while stack:
        for name in re.findall(r"\{\{(\w+)\}\}", stack.pop()):
            if name in want:
                return True
            if name not in seen and name in frags:
                seen.add(name)
                stack.append(frags[name])
    return False


class TestTheInflectionIsOneList:
    @pytest.mark.parametrize("infl", ["する", "します", "せよ", "しろ"])
    def test_the_suru_inflections(self, infl):
        import re
        assert re.fullmatch(fragment("ja_suru_infl"), infl), infl

    @pytest.mark.parametrize("infl", ["く", "き", "いて", "いた"])
    def test_the_oku_inflections(self, infl):
        import re
        assert re.fullmatch(fragment("ja_oku_infl"), infl), infl

    @pytest.mark.parametrize("key", ["fp_subst", "laplace_define", "name_assign",
                                     "rhs_set", "subst_replace_alt",
                                     "pb_pair_momentum"])
    def test_every_key_uses_the_lists(self, key):
        r"""**参照は再帰的に辿る。** ローダー自身が不動点まで展開する（断片は断片を
        参照してよい）ので、1段だけ見る検査は「断片を経由して読んでいる鍵」を
        読んでいないと報告してしまう——2026-09-12 の宣言の繋辞の畳み込みで
        `laplace_define` と `pb_pair_momentum` が実際にそう出た（活用は9/9通る
        まま）。辿れば、自前で `と置く` を綴る鍵はやはり落ちる。"""
        assert _reaches_a_shared_inflection(key), key

    def test_the_reference_walk_is_calibrated(self):
        r"""**0 も True も、何も見ていない検査から出る。** 届かない鍵で False に
        なること（`decl_stmt` は `ja_decl_suffix` を読むが、そこに在るのは
        `ja_decl_infl` で活用の表ではない）と、直に綴る形で True になることを
        両方固定する。"""
        assert _reaches_a_shared_inflection("decl_stmt") is False
        assert _reaches_a_shared_inflection("value_definition") is True


class TestOnlyTheInflectionIsShared:
    """動詞ごと共有すると、他のパターンの文を横取りする（実測で覆った案）。"""

    def test_the_replace_family_does_not_claim_a_set_sentence(self):
        """`subst_replace_alt` に `とする` を与えると H-5z の文を取ってしまう。"""
        assert nlp("subst_replace_alt", "ja").search("E20 の右辺を 0 とする") is None

    def test_the_free_parameter_family_does_not_claim_a_laplace_sentence(self):
        """`fp_subst` に `とおく` を与えると `laplace_define` の文を取ってしまう。"""
        assert nlp("fp_subst", "ja").match("x のラプラス変換を X とおく") is None


#: (言語, 綴り)。「E1 の右辺を 0 にする」を言う綴り —— 日英そろえてある。
RHS_SET = [
    ("ja", "E1 の右辺を 0 とする"),
    ("ja", "E1 の右辺を 0 とします"),
    ("ja", "E1 の右辺を 0 とせよ"),
    ("ja", "E1 の右辺を 0 としろ"),
    ("ja", "E1 の右辺を 0 とみなす"),
    ("ja", "E1 の右辺を 0 と置く"),
    ("ja", "E1 の右辺を 0 と置いて"),
    ("ja", "E1 の右辺を 0 とおく"),
    ("ja", "E1 の右辺を 0 として"),
    ("en", "set the right-hand side of E1 to 0"),
    ("en", "put the right-hand side of E1 to 0"),
    ("en", "let the right-hand side of E1 be 0"),
    ("en", "take the right-hand side of E1 to be 0"),
    ("en", "regard the right-hand side of E1 as 0"),
]


class TestThePatternReadsEverySpelling:
    @pytest.mark.parametrize("lang,text", RHS_SET)
    def test_the_eid_and_value_are_read(self, lang, text):
        m = nlp("rhs_set", lang).search(text)
        assert m is not None, text
        assert pick_group(m, "eid") == "E1", (text, m.groupdict())
        assert (pick_group(m, "value") or "").strip() == "0", (text, m.groupdict())

    def test_a_computation_is_not_a_setting(self):
        """`compute the RHS` は値を置く要求ではない。"""
        assert nlp("rhs_set", "en").search("compute the right-hand side of E1") is None
        assert nlp("rhs_set", "ja").search("E1 の右辺を計算して") is None

    def test_regard_is_an_instruction_verb(self):
        """無いと `regard the RHS of E1 as 0` が**式として登録**される（②の形）。"""
        from notebook.runtime.nl_patterns import nlp_any
        assert nlp_any("instruction_marker", "regard the right-hand side of E1 as 0")


def _run(eng, sup, text):
    return asyncio.get_event_loop().run_until_complete(
        run_interpret_pipeline(
            InterpretRequest(tokens=[InputToken(type="text", value=text)]), eng, sup))


@pytest.mark.parametrize("lang,text", RHS_SET)
def test_the_value_actually_reaches_the_notebook(lang, text):
    """画面まで —— 落ちていた綴りは式を1つも作らなかった。"""
    nb.start_engine(title="set_infl", mode="standard", language=lang,
                    domains=["classical_mechanics"])
    try:
        eng = nb.get_engine()
        sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                            domains=["classical_mechanics"], primary_backend="maxima")
        _run(eng, sup, "a をスカラーとする" if lang == "ja" else "Define a as a scalar")
        _run(eng, sup, "E = a*x + 3")
        n_before = len(eng.expressions)
        resp = _run(eng, sup, text)
        assert resp.get("status") in ("completed", "registered"), \
            (text, resp.get("status"), resp.get("error"))
        assert len(eng.expressions) - n_before == 1, (text, "式が作られていない")
    finally:
        nb.stop_engine()
