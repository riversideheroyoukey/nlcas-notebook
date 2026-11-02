r"""DIFF-ORDER-EN-ORDINAL (2026-09-11) — `the third derivative` lost its order.

Japanese reads the order from the COUNTER (`3階微分`, `2回微分`, `三次微分`);
English required the word ``order`` (`the third ORDER derivative`).  So the most
ordinary English spelling for a higher derivative silently differentiated ONCE
and reported success::

    the third derivative of E1 with respect to x   →  order dropped, 1st derivative
    the second derivative of E1 with respect to x  →  同上
    take the 3rd derivative of E1 with respect to x →  同上

Same shape as `十次まで展開` and `thrice`: a number the researcher WROTE is
discarded, and the sentence is otherwise accepted.

Making ``order`` optional on its own would read `the third expression` as an
order too, so the ordinal must be followed by a DERIVATIVE NOUN — the role the
counter (`階`/`次`/`回`) plays on the Japanese side.  That noun's spelling is the
fragment ``deriv_word_plain_en``, the mirror of ``deriv_word_plain_ja``.

**AND THE KIND WORD SITS BETWEEN THEM** (2026-09-13).  The lookahead required the
noun IMMEDIATELY, so as soon as the sentence also named the kind — the ordinary
way to ask for a higher partial derivative — the order was dropped again and the
FIRST derivative came back `completed`::

    the second partial derivative of E1 with respect to x   →  5*x^4  (want 20*x^3)
    take the third total derivative of E1 with respect to x →  5*x^4  (want 60*x^2)

Measured over ordinal × kind × frame: **36 of 64 English spellings returned the
first derivative**, while every Japanese sibling was right — `diff_rule[ja]`
carries the ordinal INSIDE its own pattern, before ``ja_deriv_word`` (which
already composes the kind words).

**This asymmetry was documented AND said it had been measured** — the entry in
``KNOWN_ASYMMETRY`` points at this very file.  What the measurement had not done
was cross the two axes: 18 spellings, none of them with a kind word.  A reason
that names a measurement is only as good as what that measurement varied.
"""
import sys

import pytest

sys.path.insert(0, "tests")

from notebook.runtime.operation_validator import OperationValidator
from test_diff_order_parsing import (_make_engine, _make_expression,     # noqa: E402
                                     _make_supervisor, _run)


def _order_of(text: str, lang: str):
    engine = _make_engine({"E1": _make_expression("E1", "x^2 + 1")})
    engine.lang = lang
    engine._validator = OperationValidator()
    _run(engine, _make_supervisor(), text)
    call = engine.run_operation.call_args
    args = call.args[1] if call else {}
    order = args.get("order")
    return int(order) if order is not None else None


class TestTheOrderIsReadInBothLanguages:
    @pytest.mark.parametrize("text,lang,order", [
        # Japanese reads the counter — this side always worked
        ("E1 を x で微分して",            "ja", None),
        ("E1 を x で3階微分して",         "ja", 3),
        ("E1 を x で三階微分して",         "ja", 3),
        ("E1 を x で2回微分して",         "ja", 2),
        ("E1 を x で3次微分して",         "ja", 3),
        ("E1 を x で十階微分して",         "ja", 10),
        ("E1 を x で3階偏微分して",        "ja", 3),
        # English: the spellings that already worked
        ("differentiate E1 with respect to x",               "en", None),
        ("differentiate E1 twice with respect to x",         "en", 2),
        ("differentiate E1 thrice with respect to x",        "en", 3),
        ("differentiate E1 three times with respect to x",   "en", 3),
        ("differentiate E1 with respect to x 3 times",       "en", 3),
        ("take the third order derivative of E1 with respect to x", "en", 3),
        ("the 3rd order derivative of E1 with respect to x", "en", 3),
        # …and the ordinal-before-the-noun form, which dropped the order
        ("the third derivative of E1 with respect to x",     "en", 3),
        ("the second derivative of E1 with respect to x",    "en", 2),
        ("take the 3rd derivative of E1 with respect to x",  "en", 3),
        ("the tenth derivative of E1 with respect to x",     "en", 10),
    ])
    def test_order(self, text, lang, order):
        assert _order_of(text, lang) == order


class TestTheKindWordDoesNotSwallowTheOrder:
    r"""序数と種別語を**掛けて**測る。片方ずつでは 18 綴りとも緑だった。"""

    @pytest.mark.parametrize("kind", ["partial ", "total ", "ordinary ", ""])
    @pytest.mark.parametrize("word,order", [("second", 2), ("third", 3),
                                            ("2nd", 2), ("3rd", 3)])
    @pytest.mark.parametrize("frame", [
        "the {o} {k}derivative of E1 with respect to x",
        "take the {o} {k}derivative of E1 with respect to x",
        "the {o} {k}derivative of E1 by x",
        "the {o} order {k}derivative of E1 with respect to x",
    ])
    def test_the_order_survives_the_kind(self, frame, word, order, kind):
        assert _order_of(frame.format(o=word, k=kind), "en") == order

    @pytest.mark.parametrize("text,order", [
        ("E1 を x で2階偏微分して", 2), ("E1 を x で三階全微分して", 3),
        ("E1 を x で2階常微分して", 2),
    ])
    def test_the_japanese_sibling_was_always_right(self, text, order):
        r"""日本語が落ちない理由は **`diff_rule[ja]` の序数枠ではない**（最初そう
        書いたが、変異試験が生き残ったので測ったら**外しても値・階数・種別・route
        がすべて同じ**だった——9綴りで確認）。本当の理由は兄弟鍵
        `order_clause[ja]` の先読みが **`ja_deriv_word`（種別語を合成した1つの表）**
        を読んでいることで、英語側の先読みは合成せず素の名詞だけを見ていた。
        **これが真の割れである。**"""
        assert _order_of(text, "ja") == order


class TestTheDerivativeNounIsOneTablePerLanguage:
    r"""日本語は種別と名詞が1語（`偏微分`）なので `ja_deriv_word` が列挙で足りる。
    英語は2語なので**合成**になる——`en_deriv_word` がその鏡である。畳む前は
    この合成が3鍵に自前で書かれ、しかも**名詞が素で綴られていた**（複数形が落ちる）。
    """

    def test_the_english_mirror_exists(self):
        import json
        import pathlib as _p
        d = json.loads((_p.Path(__file__).resolve().parents[1] / "i18n"
                        / "nl_patterns.json").read_text(encoding="utf-8"))
        f = d["_fragments"]
        assert f["en_deriv_word"] == "(?:{{deriv_kind_word_en}}\\s+)?{{deriv_word_plain_en}}"
        for key in ("diff_rule", "partial_derivative_marker", "total_derivative_marker"):
            assert "{{deriv_word_plain_en}}" in d[key]["en"], key
            assert "derivative\\b" not in d[key]["en"], key

    @pytest.mark.parametrize("text,order", [
        ("the derivatives of E1 with respect to x", None),
        ("the second derivatives of E1 with respect to x", 2),
        ("the second partial derivatives of E1 with respect to x", 2),
    ])
    def test_the_plural_is_read(self, text, order):
        r"""断片は複数形を受けるのに、素で綴っていた3鍵は落としていた。"""
        assert _order_of(text, "en") == order


class TestAnOrdinalWithoutADerivativeNounIsNotAnOrder:
    r"""The lookahead is what keeps `order` from being guessed —— **とここには
    書いてあるが、その主張は実測で再現しなかった**（2026-09-13）。

    先読みを外して6綴りを実経路で打っても、値は**全部同じ**だった——
    `differentiate E1 with respect to x in the third step` は先読みの有無に
    かかわらず1階のままである（消費者 `interpret_rules` は文全体を `search` して
    いるので、素朴には拾うはずなのに拾わない）。

    **なぜ拾わないのかを説明できなかったので、guard は消していない。** 消して
    よいと言えるのは「何を守っているか」が言えたときだけで、いまは
    「6綴りでは差が出ない」までしか測れていない。次に触る人へ: まず
    `_rule_order`（上流）がどこで埋まるかを追うこと。
    """

    @pytest.mark.parametrize("text", [
        "the partial derivative of E1 with respect to x",
        "take the derivative of E1 with respect to x",
    ])
    def test_no_order_is_invented(self, text):
        assert _order_of(text, "en") is None
