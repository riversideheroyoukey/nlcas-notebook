r"""微分の**種別**を名指す語は、いくつの表に書かれていたか。

実測 2026-09-07（worklist の `total`、8行・3鍵）。同じ語彙が **4箇所** に
別々に綴られていた:

| どこ | 綴っていた語 |
|---|---|
| `diff_rule[en]` の自前綴り | `total｜partial｜ordinary` ＋ 副詞 `totally｜partially｜ordinarily` |
| `total_derivative_marker[en]` | `total｜ordinary` ＋ `totally` |
| `partial_derivative_marker[en]` | `partial` ＋ `partially` |
| `operand_stopwords[en]` | `total｜partial｜totally｜partially｜ordinary` |

**食い違っていたのは `ordinarily` 一語**で、それが起こす症状は
「読めなかった」ではない——`diff_rule` は副詞として**受ける**ので変数も E-id も
正しく取れ、そのうえで marker が種別を知らないため

    differentiate E1 ordinarily with respect to t
    → 「偏微分か全微分か判別できません。どちらか明示してください」

と**研究者が書いた種別を訊き返していた**。前置詞4種（`with respect to`／
`w.r.t.`／`for`／`by`）× 語順2種（副詞が変数の前／後）で **8綴りすべて**。
実測 28/36 → 36/36。

`operand_stopwords` も `ordinarily` だけを知らなかった。あの表の役目は
「この語はオペランドではない」で、種別語がオペランドになる silent-wrong は
2026-09-06 に実測済み（`var='totally'`）——同じ穴が一語だけ残っていた。

**直し方は語を足すことではない。** 足せば5箇所目が生まれるだけで、次に
語彙が増えたときまた割れる。語彙は fragment `deriv_kind_*` にだけ置き、
4つの読み手はそこを差し込む。日本語側は**語彙**としては食い違っていなかったが、
`ja_deriv_word` と marker が同じ接頭辞を二重に綴っていた（＝**割れていない
だけで、割れる形**）ので同じ扱いにした。

**そして日本語の「境界」は実際に割れていた。** 隣の語との衝突（`partial` は
`partial_fraction` の別名語でもある）を固定する検査を書いていて踏んだ——
marker は文全体を `search` するので、日本語には `\b` が無い分だけ:

| 打った文 | 錨を付ける前 |
|---|---|
| `偏微分方程式 E1 を t で微分して` | **kind=partial で completed** |
| `常微分方程式 E1 を t で微分して` | **kind=total で completed**（逆の種別） |
| `E1 を t で微分して` | failed（「どちらか明示してください」＝正しい） |

研究者は素の `微分` と書いており、本来は**訊く**場面。**言及した方程式の種類が
微分の種別を決めていた**、しかも成功として。PDE/ODE はこの分野でもっとも普通の
語なので、これは防御ではない。英語側は `\s+derivative` と `\b` が最初から錨に
なっていた——**日英の割れ**。`(?!方程式)` を3つの ja fragment すべてに付けた。

**コーパスはこれを捕まえられない**——523行に `微分方程式` が0行で、claim 差分が
0 だったのは「安全」ではなく「測っていない」。

**わざと入れていない綴り:**

| 綴り | 理由 |
|---|---|
| `完全微分` | *exact differential*。全微分とは別の概念で、total と読むのは silent-wrong |
| `full derivative` | 英語として実在しない |
| `total derivatives`（複数形） | `derivatives of E1 and E2` と衝突する |
| `常導関数` | 日本語として実在しない（`常微分` は在る） |
"""
import asyncio
import json
import pathlib
import re

import pytest

import ui.notebook_bridge as nb
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.i18n import t
from notebook.runtime.nl_patterns import fragment, nlp
from notebook.runtime.operation_validator import OperationValidator
from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline

_JSON = json.loads(
    (pathlib.Path(__file__).resolve().parent.parent / "i18n" / "nl_patterns.json")
    .read_text(encoding="utf-8"))

#: 語彙が住んでよい唯一の場所。
KIND_FRAGMENTS = ("deriv_kind_total_en", "deriv_kind_partial_en",
                  "deriv_kind_total_adv_en", "deriv_kind_partial_adv_en",
                  "deriv_kind_total_ja", "deriv_kind_partial_ja")


_REF_RE = re.compile(r"\{\{\w+\}\}")


def _literal(pat):
    """fragment 参照を取り除いた残り＝**その場所が自前で綴っている文字**。
    参照名（`{{deriv_kind_total_en}}`）に語が含まれるのは綴りではない。"""
    return _REF_RE.sub("", str(pat))


_TAIL_LOOKAHEAD_RE = re.compile(r"\(\?!.*\)$")


def _words(name):
    """`(?:a|b)(?!c)` → `['a', 'b']`。中身はこちらが書いたデータなので素朴でよい。

    末尾の否定先読みは**語彙ではなく境界**——日本語には `\b` が無いので、
    `常微分方程式`／`偏微分方程式` を種別語と読まないための錨がここに付く。"""
    body = _TAIL_LOOKAHEAD_RE.sub("", fragment(name))
    assert body.startswith("(?:") and body.endswith(")"), (name, body)
    return body[3:-1].split("|")


def _ask(lang, text):
    """(種別, var) を返す。値ではなく**何を頼んだと読まれたか**を見る。"""
    nb.start_engine(title="dkv", mode="standard", language=lang, domains=[])
    seen: list = []
    try:
        eng = nb.get_engine()
        sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                            domains=[], primary_backend="maxima")
        orig = eng.run_operation

        def spy(op, args):
            seen.append((op, dict(args)))
            return orig(op, args)
        eng.run_operation = spy

        def run(x):
            return asyncio.get_event_loop().run_until_complete(
                run_interpret_pipeline(
                    InterpretRequest(tokens=[InputToken(type="text", value=x)]),
                    eng, sup))
        run("q を時間依存関数として定義する" if lang == "ja"
            else "Define q as a time-dependent function")
        run("f = q^2 + t")
        seen.clear()
        resp = run(text)
        got = [(o, a) for o, a in seen if o == "differentiate"]
        if not got:
            return None, None, resp
        a = got[-1][1]
        kind = ("total" if a.get("_is_total_derivative") else
                "partial" if a.get("_is_partial_derivative") else None)
        return kind, a.get("var"), resp
    finally:
        nb.stop_engine()


EN_HEADS = ["with respect to", "w.r.t.", "for", "by"]


class TestEveryKindWordReachesItsKind:
    """**語彙は fragment から取る。** テストが語を書き写すと、書き写した分しか
    守れない——語が増えたとき、増えたことに気づけるのは列挙する側だけ。"""

    @pytest.mark.parametrize("head", EN_HEADS)
    @pytest.mark.parametrize(
        "word,want",
        [(w, "total") for w in _words("deriv_kind_total_en")] +
        [(w, "partial") for w in _words("deriv_kind_partial_en")])
    def test_english_adjective(self, head, word, want):
        kind, var, _ = _ask("en", f"take the {word} derivative of E1 {head} t")
        assert (kind, var) == (want, "t"), (head, word, kind, var)

    @pytest.mark.parametrize("head", EN_HEADS)
    @pytest.mark.parametrize(
        "word,want",
        [(w, "total") for w in _words("deriv_kind_total_adv_en")] +
        [(w, "partial") for w in _words("deriv_kind_partial_adv_en")])
    def test_english_adverb_before_the_variable(self, head, word, want):
        """`ordinarily` はここで 4/4 落ちていた——読めているのに訊き返していた。"""
        kind, var, _ = _ask("en", f"differentiate E1 {word} {head} t")
        assert (kind, var) == (want, "t"), (head, word, kind, var)

    @pytest.mark.parametrize("head", EN_HEADS)
    @pytest.mark.parametrize(
        "word,want",
        [(w, "total") for w in _words("deriv_kind_total_adv_en")] +
        [(w, "partial") for w in _words("deriv_kind_partial_adv_en")])
    def test_english_adverb_after_the_variable(self, head, word, want):
        kind, var, _ = _ask("en", f"differentiate E1 {head} t {word}")
        assert (kind, var) == (want, "t"), (head, word, kind, var)

    @pytest.mark.parametrize(
        "word,want",
        [(w, "total") for w in _words("deriv_kind_total_ja")] +
        [(w, "partial") for w in _words("deriv_kind_partial_ja")])
    @pytest.mark.parametrize("frame", ["E1 を t で{kw}して",
                                       "E1 の t に関する{kw}を求めて"])
    def test_japanese(self, frame, word, want):
        kind, var, _ = _ask("ja", frame.format(kw=word))
        assert (kind, var) == (want, "t"), (frame, word, kind, var)


class TestThePluralIsTheSameRequest:
    r"""**複数形は種別を消さない**（2026-09-13）。

    `take the total derivatives of E1 with respect to t` は、以前は
    `TestWhatWasDeliberatelyLeftOut` に「読まない」として並んでいた。**そのクラスの
    説明は `完全微分`＝exact differential が別概念だという話で、複数形を説明して
    いない**——固定されていたのは決定ではなく、marker が名詞を素の `derivative` と
    綴っていたという**実装の単位**だった（`deriv_word_plain_en` は最初から
    `derivatives?` である）。

    変数が書かれているので要求は一意で、研究者は `total` と書いている。読まなければ
    「偏微分か全微分か明示してください」と訊き返すことになり、それは
    2026-09-06 にこの族が直したばかりの欠陥そのものである。

    `full`／`完全微分` は**除外のまま**——あちらは概念が違う。
    """

    @pytest.mark.parametrize("text,want", [
        ("take the total derivatives of E1 with respect to t", "total"),
        ("take the partial derivatives of E1 with respect to t", "partial"),
        ("take the total derivative of E1 with respect to t", "total"),
    ])
    def test_the_kind_survives_the_plural(self, text, want):
        kind, _, resp = _ask("en", text)
        assert kind == want, (text, kind, resp)


class TestTheVocabularyLivesInOnePlace:
    """四十問目と同じ形——**数える検査そのものにする**。手で数えた「残存なし」は
    2026-09-06 に3鍵取りこぼした。"""

    @pytest.mark.parametrize("word", sorted(
        {w for n in KIND_FRAGMENTS for w in _words(n)}))
    def test_no_key_spells_a_kind_word_itself(self, word):
        offenders = []
        for key, langs in _JSON.items():
            if key == "_fragments" or not isinstance(langs, dict):
                continue
            for lang, pat in langs.items():
                if not lang.startswith("_") and word in _literal(pat):
                    offenders.append(f"{key}[{lang}]")
        assert not offenders, (
            f"{word!r} を自前で綴っている: {offenders}"
            "——語彙は _fragments の deriv_kind_* にだけ置く")

    @pytest.mark.parametrize("word", sorted(
        {w for n in KIND_FRAGMENTS for w in _words(n)}))
    def test_no_fragment_outside_the_family_spells_one(self, word):
        # `_` で始まる欄は**散文の註**であって綴りではない——上の兄弟
        # (`test_no_key_spells_a_kind_word_itself`) は最初からそう読んでおり、
        # こちらだけが読んでいなかった（実測 2026-09-19: 註に「`partial
        # fraction` と書いたら ratchet が鳴った」と**書いた**ら、その文が
        # 鳴らした）。下の `test_prose_can_never_become_a_pattern` が、註が
        # 綴りとして読まれる道が無いことを固定する。
        offenders = [n for n, body in _JSON["_fragments"].items()
                     if not n.startswith("_") and n not in KIND_FRAGMENTS
                     and word in _literal(body)]
        assert not offenders, (f"{word!r}: {offenders}")

    def test_prose_can_never_become_a_pattern(self):
        """`_` で始まる欄を読み飛ばしてよい理由——誰もそれを差し込まない。"""
        import re as _re
        refs = set()
        for body in _JSON["_fragments"].values():
            if isinstance(body, str):
                refs |= set(_re.findall(r"\{\{(\w+)\}\}", body))
        for langs in _JSON.values():
            if isinstance(langs, dict):
                for lang, pat in langs.items():
                    if isinstance(pat, str) and not lang.startswith("_"):
                        refs |= set(_re.findall(r"\{\{(\w+)\}\}", pat))
        assert not [r for r in refs if r.startswith("_")], sorted(refs)

    @pytest.mark.parametrize("word", sorted(
        set(_words("deriv_kind_total_en")) | set(_words("deriv_kind_partial_en")) |
        set(_words("deriv_kind_total_adv_en")) |
        set(_words("deriv_kind_partial_adv_en"))))
    def test_a_kind_word_is_never_an_operand(self, word):
        """`ordinarily` だけがこの表からも漏れていた。"""
        assert nlp("operand_stopwords", "en").match(word), word

    def test_the_vocabulary_is_what_was_measured(self):
        """**ここだけは語を書き写す。** 上の族は fragment から parametrize する
        ので、語を1つ消すと**行が減って緑のまま通る**——2026-09-06 に実測した
        「同じ定数を読むテストは、その定数の誤りを捕まえられない」と同じ罠。
        増減に気づく役はこの1関数が負う。"""
        assert set(_words("deriv_kind_total_en")) == {"total", "ordinary"}
        assert set(_words("deriv_kind_partial_en")) == {"partial"}
        assert set(_words("deriv_kind_total_adv_en")) == {"totally", "ordinarily"}
        assert set(_words("deriv_kind_partial_adv_en")) == {"partially"}
        assert set(_words("deriv_kind_total_ja")) == {"全微分", "常微分", "全導関数"}
        assert set(_words("deriv_kind_partial_ja")) == {"偏微分", "偏導関数"}
        assert set(_words("deriv_word_plain_ja")) == {"微分", "導関数"}

    def test_the_readers_splice_instead_of_spelling(self):
        assert "{{deriv_kind_total_en}}" in _JSON["diff_rule"]["en"]
        assert "{{deriv_kind_total_adv_en}}" in _JSON["diff_rule"]["en"]
        assert "{{deriv_kind_total_en}}" in _JSON["total_derivative_marker"]["en"]
        assert "{{deriv_kind_partial_ja}}" in _JSON["partial_derivative_marker"]["ja"]
        assert "{{deriv_kind_word_en}}" in _JSON["operand_stopwords"]["en"]
        assert "{{deriv_kind_total_ja}}" in _JSON["_fragments"]["ja_deriv_word"]


class TestTheJapaneseReadingWordIsUnchanged:
    """`ja_deriv_word` は fragment 化しても**同じ集合**でなければならない。
    書き換えたのは綴りの置き場所だけで、受ける語ではない。"""

    @pytest.mark.parametrize("word,want", [
        ("微分", True), ("導関数", True), ("偏微分", True), ("全微分", True),
        ("常微分", True), ("偏導関数", True), ("全導関数", True),
        ("常導関数", False), ("積分", False), ("全", False), ("偏微", False),
    ])
    def test_same_set_as_before(self, word, want):
        old = re.compile(r"(?:(?:偏|全|常)?微分|(?:偏|全)?導関数)")
        assert bool(old.fullmatch(word)) is want, word          # 旧の性質を先に固定
        assert bool(re.fullmatch(fragment("ja_deriv_word"), word)) is want, word


class TestWhatWasDeliberatelyLeftOut:
    @pytest.mark.parametrize("text,lang", [
        ("E1 を t で完全微分して", "ja"),
        ("take the full derivative of E1 with respect to t", "en"),
    ])
    def test_not_read_as_a_kind(self, text, lang):
        """`完全微分` は exact differential——**別概念**なので total と読まない。"""
        kind, _, resp = _ask(lang, text)
        assert kind is None or resp.get("status") != "completed", text

    @pytest.mark.parametrize("word", ["完全微分", "常導関数", "full", "exact"])
    def test_not_in_the_vocabulary(self, word):
        assert word not in "|".join(fragment(n) for n in KIND_FRAGMENTS), word


class TestTheKindWordsAreSharedWithOtherOperations:
    """`partial` は `partial_fraction`（部分分数分解）の別名語でもあり、`ordinary`
    は `ordinary differential equation` に出る。**種別を名指すのは「その語」では
    なく「その語＋derivative」**で、その錨が両者を分けている——語彙を fragment へ
    寄せても錨は marker 側に残る、という境界をここで固定する。"""

    @pytest.mark.parametrize("text", [
        "partial fraction decomposition of E1",
        "partial fraction",
        "E1 を部分分数分解して",
        "solve the ordinary differential equation E1",
        "E1 の常微分方程式を解いて",
    ])
    def test_not_read_as_a_derivative_kind(self, text):
        for lang in ("ja", "en"):
            assert not nlp("total_derivative_marker", lang).search(text), (lang, text)
            assert not nlp("partial_derivative_marker", lang).search(text), (lang, text)

    @pytest.mark.parametrize("text", [
        "偏微分方程式 E1 を t で微分して",
        "常微分方程式 E1 を t で微分して",
    ])
    def test_a_mentioned_equation_type_does_not_decide_the_kind(self, text):
        """**日本語には `\\b` が無い。** marker は文全体を `search` するので、
        錨が無いと `偏微分方程式`／`常微分方程式` の中の `偏微分`／`常微分` が
        種別を決めてしまう——研究者は素の `微分` と書いており、本来は訊く場面。
        実測 2026-09-07（錨を外すと `partial` と `total` で **completed**、
        すなわち同じ文が言及した方程式の種類で**逆の微分**を返す silent-wrong）。"""
        kind, var, resp = _ask("ja", text)
        assert kind is None, (text, kind)
        assert resp.get("status") != "completed", text

    def test_the_plain_word_carries_the_boundary_too(self):
        """`微分方程式` の `微分` も種別語と同じ扱い——**3つで1つの語彙**なので
        2つに境界があって1つに無いのが、まさに直している「割れている形」。

        実測 2026-09-07: 錨が無いと `diff_rule` がこの文を**微分の依頼として
        claim** し、`微分方程式に変形して` が「偏微分か全微分か明示してください」
        で終わる。silent-wrong ではない（安全な失敗）ので、変異 M11 が落ちるのは
        この1本だけ。"""
        _, _, resp = _ask("ja", "E1 を t で微分方程式に変形して")
        assert t("interpreter.diff_specify_kind", "ja", var="t") not in str(
            resp.get("error") or ""), resp

    def test_the_plain_word_still_asks(self):
        """錨が「訊く」側を壊していないこと——素の `微分` は曖昧なので訊く。"""
        kind, _, resp = _ask("ja", "E1 を t で微分して")
        assert kind is None and resp.get("status") != "completed"

    def test_the_anchor_is_the_word_derivative(self):
        assert nlp("partial_derivative_marker", "en").search(
            "take the partial derivative of E1 with respect to t")
        assert nlp("total_derivative_marker", "en").search(
            "take the ordinary derivative of E1 with respect to t")
