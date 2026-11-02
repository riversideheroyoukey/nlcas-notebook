r"""英語の宣言は、繋辞を1つの表で答える（NL-GRAMMAR ①、2026-09-12）。

**日本語側は最初から畳まれていた。** `ja_decl_suffix` という断片が「宣言の語尾は
どう綴られるか」に1箇所で答えている。英語側は同じ問いに**鍵ごとに自前で**答えて
おり、語彙は `as` だけのものから `as|is|are|be|to be` まで割れていた。

**足りない綴りは黙って落ちるのではない。** 総称の宣言 `decl_stmt[en]` が最も広い
語彙を持っているので、専用の鍵が読めなかった文はそこに拾われ、
`define_scalar_parameter` になる——文には「演算子」「正準対」「ラプラス変換」
「汎関数」と書いてあるのに、**スカラーが登録されて成功が返る**。

実測（英語、繋辞5 × 先行動詞4 × 族9 = 180通り、**研究者が通る経路**で）::

    畳む前   100/180 が正しい操作に届く
    畳んだ後 180/180
    変わった 80セル——76 が `define_scalar_parameter`（silent-wrong）から
             正しい操作へ、4 が**声のある断り**へ

    let c be an electron annihilation operator  → define_scalar_parameter
    p is the canonical momentum of q            → define_scalar_parameter
    X is the Laplace transform of x             → define_scalar_parameter
    declare X as the Laplace transform of x     → define_scalar_parameter
    F is an expectation functional              → define_scalar_parameter

**日本語も同じ問いで割れていた——断片を持っていながら、である。** 型語を模様の中に
持つ5鍵（スピン・正準対・共役運動量・ラプラス変換の定義・汎関数）が自前の語尾一覧を
綴っており、`pb_pair_canonical[ja]` は31綴り中6しか受けなかった。実測（語尾12 × 族9）
**88/108 → 108/108**、内訳はスカラー誤登録15件と、ラプラス変換の**定義でなく計算**が
走る5件。

**`と置く` は uniform にしない——その非対称が内容である。** これは代入の綴り
（`x を 0 と置く`、SET-TO-UNDECIDED）なので、型語を持たない総称の鍵は受けてはならない。
型語が模様の中に在る鍵には紛れる相手が無いので受ける——コーパスのラプラス変換の
定義カードは実際 `x のラプラス変換を X とおく` と書かれている。

**操作名だけの差分0は安心にならない。** スピンの族は繋辞5通りすべてで
`define_spin_operator` に届いており、粗い格子では「差分0」に見えた。届けていたの
は別名走査で、`spin_operator_decl` の `desc`（核種）は捨てられていた——NMR の
ふつうの綴り `I is a nuclear spin operator of 13C` は**スカラー `I` を登録して
confirmed** を返す。中身まで比べて初めて出た4件である。

**先行動詞も同じ問いだった。** 2つの鍵だけが動詞を必須にしており、しかも一覧が
互いに違った（`laplace_define` は `let|define`、`declare_bracket_functional` は
`declare|let`）。`declare X as the Laplace transform of x` と
`define F as an expectation functional` は、そのどちらの穴にも落ちていた。

**素の `to` は入れない**（`decl_stmt` に付いていた制約をここへ移した。断片に置いた
ので、以後**すべての宣言の鍵**に効く）。入れれば `set x to a scalar` は通るが、同時に
`convert E1 to polar coordinates` が `E1` のスカラー宣言に、`expand E1 to third
order in x` が `exp` の宣言になる（実測 2026-09-10）。`to be` は既に安全である。

**語境界 `\b` は飾りではない。** `decl_suffix[en]` は「型ラベルらしさ」だけを見て
自動登録を迂回させる判定に使われており、境界なしで `be` を足すと
`describe a vector` の `…be` が当たって宣言に見える。
"""
import asyncio
import json
import pathlib
import re

import pytest

import ui.notebook_bridge as nb
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.nl_patterns import nlp
from notebook.runtime.operation_validator import OperationValidator
from ui.interpret_pipeline import (InputToken, InterpretRequest,
                                   run_interpret_pipeline)

_PATTERNS = json.loads(
    (pathlib.Path(__file__).resolve().parents[1] / "i18n" / "nl_patterns.json")
    .read_text(encoding="utf-8"))

#: 宣言の繋辞を読む鍵。**この一覧は表明であり、下の機械的な数え上げが裏を取る。**
DECLARATION_KEYS = [
    "decl_stmt", "decl_suffix", "particle_operator_decl", "bare_quantum_op_decl",
    "bare_ladder_op_decl", "spin_operator_decl", "laplace_define",
    "pb_pair_canonical", "pb_pair_momentum", "declare_bracket_functional",
]

#: 自前で繋辞を綴ってよい鍵と、その理由。**宣言ではないから**である。
COPULA_EXEMPT = {
    "value_definition": "H-1 の値の枝。素の `to` を持つ（`set x to 3`）——宣言の繋辞に"
                        "入れてはならない綴りなので、共有できない。",
    "rhs_set": "H-5z「右辺を X とする」。同じく値の繋辞で `to` を含む。",
    "name_assign": "`store E1 as x`——式に名前を付ける族で、型を宣言していない。",
    "verify_canon": "「この式は…か」を確かめる疑問文。宣言ではない。",
}

_COPULA_LITERAL = re.compile(r"\\s[+*]\(?\??:?(?:as|is|are|be)\b|\(\?:[^()]*\bto\\s\+be\b")


# ---------------------------------------------------------------------------
# 語彙が1箇所にあること
# ---------------------------------------------------------------------------
class TestTheCopulaVocabularyLivesInOneFragment:

    def test_the_fragment_lists_the_spellings(self):
        frag = _PATTERNS["_fragments"]["en_decl_copula"]
        assert re.fullmatch(r"\\b\(\?:as\|is\|are\|be\|to\\s\+be\)\\b", frag), frag

    def test_the_copula_does_not_list_a_bare_to(self):
        r"""`decl_stmt` に書いてあった制約を断片へ移したもの。素の `to` を足すと
        英語でいちばん多い前置詞が宣言の読み手の前に立つ（docstring 参照）。"""
        frag = _PATTERNS["_fragments"]["en_decl_copula"]
        alts = frag.replace(r"\b", "").strip("(?:)").split("|")
        assert "to" not in alts, frag

    def test_the_verb_list_is_shared_too(self):
        assert _PATTERNS["_fragments"]["en_decl_verb"] == r"(?:define|declare|let)"

    def test_the_boundary_is_part_of_the_fragment(self):
        r"""境界がなければ `describe a vector` の `…be` が繋辞になる。"""
        assert _PATTERNS["_fragments"]["en_decl_copula"].startswith(r"\b")

    def test_every_declaration_key_reads_the_fragment(self):
        for key in DECLARATION_KEYS:
            en = _PATTERNS[key]["en"]
            assert "{{en_decl_copula}}" in en, key
            assert not _COPULA_LITERAL.search(
                en.replace("{{en_decl_copula}}", "")), (key, en)

    def test_no_other_key_spells_a_copula_without_saying_why(self):
        r"""**新しい鍵が自前で綴り始めたら、ここが落ちる。** 免除には理由を書く。"""
        rogue = []
        for key, val in _PATTERNS.items():
            if key in DECLARATION_KEYS or key in COPULA_EXEMPT:
                continue
            if not isinstance(val, dict):
                continue
            en = val.get("en")
            if isinstance(en, str) and _COPULA_LITERAL.search(en):
                rogue.append(key)
        assert rogue == [], rogue

    def test_the_detector_is_calibrated(self):
        r"""**0 という結果は「何も見ていない測定器」でも出る。** 報告すべき形と
        沈黙すべき形を両方固定してから、上の 0 を信じる。"""
        for spelled in [r"(?i)X\s+(?:as|is)\s+a", r"foo\s+as\s+bar",
                        r"Y\s*(?:be|to\s+be)\s+", r"(?:to\s+be|=)"]:
            assert _COPULA_LITERAL.search(spelled), spelled
        for quiet in ["{{en_decl_copula}}", r"(?i)the\s+basis\s+of",
                      r"(?:in\s+general|as\s+a\s+differential\s+equation)",
                      r"(?:the|an|and|or|of|as|its)"]:
            assert not _COPULA_LITERAL.search(quiet), quiet

    def test_every_exempt_key_is_actually_detected(self):
        r"""免除の一覧が飾りでないこと——検出されない鍵を免除しても何も言って
        いない（`ode_general` と `operand_stopwords` は `as` を**語として**持つ
        だけなので検出されず、だから一覧にも無い）。"""
        for key in COPULA_EXEMPT:
            assert _COPULA_LITERAL.search(_PATTERNS[key]["en"]), key

    def test_the_exemptions_are_real_keys(self):
        assert set(COPULA_EXEMPT) <= set(_PATTERNS), sorted(
            set(COPULA_EXEMPT) - set(_PATTERNS))


# ---------------------------------------------------------------------------
# 綴りの空間——鍵の正規表現に直接当てる（速い・網羅的）
# ---------------------------------------------------------------------------
_COPULAS = ["as", "is", "are", "be", "to be"]
_VERBS = ["", "define ", "declare ", "let "]

_FAMILY_SENTENCE = {
    "decl_stmt": "x {c} a scalar",
    "particle_operator_decl": "c {c} an electron annihilation operator",
    "bare_quantum_op_decl": "c {c} a fermion operator",
    "bare_ladder_op_decl": "c {c} an annihilation operator",
    "spin_operator_decl": "S {c} a spin operator",
    "laplace_define": "X {c} the Laplace transform of x",
    "pb_pair_canonical": "q and p {c} a canonical pair",
    "pb_pair_momentum": "p {c} the canonical momentum of q",
    "declare_bracket_functional": "F {c} an expectation functional",
}


class TestEveryFamilyAcceptsEverySpelling:
    """畳む前は 80/180、いまは 180/180。"""

    @pytest.mark.parametrize("key", sorted(_FAMILY_SENTENCE))
    @pytest.mark.parametrize("verb", _VERBS)
    @pytest.mark.parametrize("cop", _COPULAS)
    def test_the_grid(self, key, verb, cop):
        text = verb + _FAMILY_SENTENCE[key].format(c=cop)
        rx = nlp(key, "en")
        m = rx.match(text) if key == "laplace_define" else rx.search(text)
        assert m is not None, text


class TestTheBoundaryKeepsOrdinaryWordsOut:
    """`decl_suffix` は自動登録を迂回させる判定に使われる（型ラベルらしさだけを
    見る）ので、語の途中の `be`/`is` を拾ってはならない。"""

    @pytest.mark.parametrize("text", [
        "describe a vector", "probe a scalar", "maybe a matrix", "the tube a vector",
    ])
    def test_a_word_ending_in_a_copula_is_not_a_declaration(self, text):
        assert nlp("decl_suffix", "en").search(text) is None, text

    @pytest.mark.parametrize("text,label", [
        ("as a vector", "vector"), ("to be a scalar", "scalar"),
        ("is a matrix", "matrix"), ("are a tensor", "tensor"),
    ])
    def test_a_real_suffix_still_reads(self, text, label):
        m = nlp("decl_suffix", "en").search(text)
        assert m is not None and m.group(1).strip() == label, text


# ---------------------------------------------------------------------------
# **研究者が通る経路で測る。** 上は「当たる」ことしか言えない——どの規則が先に
# 掴むかはパイプラインが決めるので、横取りの有無は打ってみないと分からない。
# ---------------------------------------------------------------------------
def _ask(lang, domain, text, setup=()):
    async def go():
        nb.start_engine(title="copula", mode="standard", language=lang,
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
            return await run_interpret_pipeline(
                InterpretRequest(tokens=[InputToken(type="text", value=text)]),
                engine, sup)
        finally:
            nb.stop_engine()
    return asyncio.run(go())


class TestASentenceThatSaysOperatorMustNotRegisterAScalar:
    """畳む前はすべて `define_scalar_parameter` が `confirmed` で返っていた。"""

    @pytest.mark.parametrize("lang,domain,text,want", [
        # 英語——繋辞が `as` 以外なら総称の宣言に拾われていた族
        ("en", "differential_equations",
         "X is the Laplace transform of x", "define_laplace_transform"),
        ("en", "differential_equations",
         "declare X as the Laplace transform of x", "define_laplace_transform"),
        ("en", "classical_mechanics",
         "p is the canonical momentum of q", "declare_canonical_pair"),
        ("en", "classical_mechanics",
         "let p be the canonical momentum of q", "declare_canonical_pair"),
        ("en", "classical_mechanics",
         "q and p are a canonical pair", "declare_canonical_pair"),
        ("en", "classical_mechanics",
         "F is an expectation functional", "declare_bracket_functional"),
        ("en", "magnetic_resonance",
         "I is a nuclear spin operator of 13C", "define_spin_operator"),
        # 日本語——語尾の一覧が鍵ごとに違っていた族
        ("ja", "magnetic_resonance",
         "I を ^{13}C の核スピン演算子とします", "define_spin_operator"),
        ("ja", "classical_mechanics",
         "q と p を正準共役対と定義する", "declare_canonical_pair"),
        ("ja", "classical_mechanics",
         "p を q の共役運動量とみなす", "declare_canonical_pair"),
        ("ja", "differential_equations",
         "x のラプラス変換を X と宣言する", "define_laplace_transform"),
    ])
    def test_the_specific_declaration_wins(self, lang, domain, text, want):
        r = _ask(lang, domain, text)
        assert r.get("operation") == want, (text, r)

    def test_the_particle_operator_is_declared_not_a_scalar(self):
        r"""起票の見出しの綴り。Julia セッションなので1件だけ通す（残りは綴りの
        空間の格子が見ている）。"""
        r = _ask("en", "condensed_matter",
                 "let c be an electron annihilation operator")
        assert r.get("operation") == "define_fermion_operator", r


class TestTheNuclideIsWhyTheCoarseGridWasNotEnough:
    r"""**操作名だけの差分0は安心にならない。** `S is a spin operator` は繋辞5通り
    すべてで `define_spin_operator` に届いていた——別名走査が届けており、
    `spin_operator_decl` の `desc`（核種）は捨てられていた。核種を書いた文で
    初めて割れが見える。"""

    @pytest.mark.parametrize("cop", ["as", "is", "are", "be", "to be"])
    def test_the_nuclide_survives_every_copula(self, cop):
        r = _ask("en", "magnetic_resonance",
                 f"I {cop} a nuclear spin operator of 13C")
        assert r.get("operation") == "define_spin_operator", r
        assert "13" in (r.get("message") or ""), r


class TestWhatStaysRefusedIsADecision:

    @pytest.mark.parametrize("cop", ["as", "is", "are", "be", "to be"])
    def test_a_ladder_operator_with_no_particle_is_refused_loudly(self, cop):
        r"""統計（フェルミオンかボソンか）が書かれていないので決めない。畳む前、
        `are` だけは総称に落ちて**スカラーが登録され成功が返って**いた——
        断りが1つ増えたのではなく、silent-wrong が断りに変わった。"""
        r = _ask("en", "condensed_matter", f"c {cop} an annihilation operator")
        assert r.get("operation") != "define_scalar_parameter", r
        assert r.get("status") not in ("completed", "confirmed", "declared"), r

    @pytest.mark.parametrize("text", ["x を 0 と置く", "x をスカラーと置く"])
    def test_the_generic_declaration_still_refuses_oku(self, text):
        r"""`と置く` は**代入の綴り**（SET-TO-UNDECIDED）。総称の宣言には型語が
        模様の中に無いので、値か型かを言い分けられない——だから
        `ja_decl_suffix` は置く／おくを持たず、型語を持つ鍵だけが
        `ja_decl_suffix_named` でそれを受ける。"""
        r = _ask("ja", "classical_mechanics", text)
        assert r.get("operation") != "define_scalar_parameter", (text, r)

    @pytest.mark.parametrize("text,want", [
        # `とおく` はコーパスが実際に打っている綴り（ラプラス変換の定義カード）
        ("x のラプラス変換を X とおく", "define_laplace_transform"),
        ("x のラプラス変換を X と置く", "define_laplace_transform"),
        ("p を q の共役運動量と置く", "declare_canonical_pair"),
    ])
    def test_a_key_whose_type_word_is_written_may_take_oku(self, text, want):
        r"""型語が模様の中に在れば `と置く` に代入の読みは無い——だから
        `ja_decl_suffix_named` はこれを受ける（上の総称の鍵とは逆）。"""
        dom = "differential_equations" if "ラプラス" in text else "classical_mechanics"
        r = _ask("ja", dom, text)
        assert r.get("operation") == want, (text, r)

    @pytest.mark.parametrize("text,want", [
        ("q と p を正準共役対にする", "declare_canonical_pair"),
        ("p を q の共役運動量に定義する", "declare_canonical_pair"),
    ])
    def test_the_ni_particle_is_not_lost_by_the_fold(self, text, want):
        r"""畳む前に通っていた綴り。**畳み込みは狭めてはならない**ので、
        断片は `に` の枝を持つ（生き残った変異が名指した）。"""
        r = _ask("ja", "classical_mechanics", text)
        assert r.get("operation") == want, (text, r)

    def test_set_x_to_a_type_is_still_not_a_declaration(self):
        r"""素の `to` を繋辞に入れない、という決定がまだ効いていること。"""
        r = _ask("en", "classical_mechanics", "set x to a scalar")
        assert r.get("operation") != "define_scalar_parameter", r
