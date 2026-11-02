r"""素の宣言名と「偏微分して」（2026-09-12、OPERAND-BOUNDARY 残①の続き）。

**`偏微分` / `全微分` は操作の名前でもある。** 直前の作業で宣言済みの名前が
オペランドに渡るようになり、そこで初めて見えた綴りの穴::

    f をスカラー場とする
      f を x で微分して    →  completed  'diff(f,x,1)
      f を x で偏微分して  →  **failed（どの規則にも当たらない）**
      f を x で全微分して  →  **failed**

原因はトークナイザの後の1語だった——`偏微分して` は語尾が落ちて `偏微分` になり、
**別名表にあるのは `微分` だけ**。E 番号を書けば `diff_rule`（インターセプタ）が
先に答えるので届き、H-1 の名前も 2a-sym が式番号へ書き換えるので届く。届かなかった
のは**式番号を持たない宣言済みの名前**だけで、英語は `derivative` が別名なので
最初から通っていた（`take the partial derivative of f …`）——日英の割れ。

**別名を足したら、その経路が種別を運んでいないことが見えた。** `偏微分` と
`全微分` が**どちらもフラグ無し**で操作に届いていた（LIMIT-ONE-SIDED の
`direction` と同じ形——書いた語が捨てられる）。いまは値が割れない（この経路に
届く宣言済みオペランドはスカラー場だけで、座標依存が書かれていないので偏と全は
一致する）が、一致するのは**偶然**なので、`_enrich_args_from_text` で補う。

**PDE は盗まない**（実測）: `偏微分方程式を解いて` はトークナイザが1語のまま
残すので、引かれる語は `偏微分` にならない。
"""
import asyncio

import pytest

import ui.notebook_bridge as nb
from notebook.runtime.llm_supervisor import LLMSupervisor, _tokenize
from notebook.runtime.operation_validator import OperationValidator
from notebook.runtime.token_field_mapping import scan_aliases
from ui.interpret_pipeline import (InputToken, InterpretRequest,
                                   run_interpret_pipeline)


def _ask(lang, domain, setup, text):
    async def go():
        nb.start_engine(title="partial-kind", mode="standard", language=lang,
                        domains=[domain])
        try:
            engine = nb.get_engine()
            seen: list = []
            original = engine.run_operation

            def spy(op, args, **kw):
                seen.append((op, dict(args)))
                return original(op, args, **kw)

            engine.run_operation = spy
            sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                                domains=[domain],
                                primary_backend=engine.session.primary_backend)
            for line in setup:
                await run_interpret_pipeline(
                    InterpretRequest(tokens=[InputToken(type="text", value=line)]),
                    engine, sup)
            seen.clear()
            r = await run_interpret_pipeline(
                InterpretRequest(tokens=[InputToken(type="text", value=text)]),
                engine, sup)
            return r, (seen[-1][1] if seen else {})
        finally:
            nb.stop_engine()
    return asyncio.run(go())


_FIELD = {"ja": ["f をスカラー場とする"], "en": ["define f as a scalar field"]}


class TestTheWordNamesTheOperation:

    @pytest.mark.parametrize("lang,text", [
        ("ja", "f を x で偏微分して"),
        ("ja", "f を x で全微分して"),
        ("ja", "f の x に関する偏微分"),
        ("ja", "f を x で微分して"),
        ("en", "take the partial derivative of f with respect to x"),
        ("en", "take the total derivative of f with respect to x"),
        ("en", "differentiate f with respect to x"),
    ])
    def test_a_declared_field_differentiates(self, lang, text):
        r, _args = _ask(lang, "physics", _FIELD[lang], text)
        assert r.get("status") == "completed", r
        assert "diff(f,x,1)" in (r.get("canonical") or "").replace(" ", ""), r

    @pytest.mark.parametrize("word,operation", [
        ("偏微分", "differentiate"), ("全微分", "differentiate"),
        ("微分", "differentiate"),
    ])
    def test_the_looked_up_word_resolves(self, word, operation):
        r"""**引かれるのは語尾を落とした形**——`偏微分して` も `偏微分する` も
        トークナイザは `偏微分` にする（だから素の形だけを表に入れてある）."""
        v = OperationValidator()
        assert v.resolve_alias(word) == operation

    @pytest.mark.parametrize("text", ["偏微分して", "偏微分する", "偏微分せよ"])
    def test_every_verb_tail_reduces_to_the_bare_word(self, text):
        assert _tokenize(text) == ["偏微分"]


class TestItDoesNotStealThePDE:
    r"""`偏微分方程式を解いて` を `differentiate` に取られてはいけない."""

    @pytest.mark.parametrize("text", [
        "偏微分方程式を解いて", "常微分方程式を解いて", "微分方程式を解いて",
    ])
    def test_the_pde_sentence_names_no_differentiate(self, text):
        v = OperationValidator()
        tokens = _tokenize(text)
        named = [c for _s, _e, c in scan_aliases(tokens, v.resolve_alias,
                                                 v.max_alias_words)]
        assert "differentiate" not in named, (tokens, named)


class TestTheKindTheResearcherWroteReachesTheOperation:
    r"""書いた語（偏／全）は捨てない——**値が一致するのは偶然**である."""

    @pytest.mark.parametrize("lang,text,flag", [
        ("ja", "f を x で偏微分して", "_is_partial_derivative"),
        ("ja", "f を x で全微分して", "_is_total_derivative"),
        ("en", "take the partial derivative of f with respect to x",
         "_is_partial_derivative"),
        ("en", "take the total derivative of f with respect to x",
         "_is_total_derivative"),
    ])
    def test_the_flag_is_set(self, lang, text, flag):
        _r, args = _ask(lang, "physics", _FIELD[lang], text)
        assert args.get(flag) is True, args

    @pytest.mark.parametrize("lang,text", [
        ("ja", "f を x で微分して"), ("en", "differentiate f with respect to x"),
    ])
    def test_an_unqualified_derivative_names_no_kind(self, lang, text):
        """書かれていない種別を補ってはいけない."""
        _r, args = _ask(lang, "physics", _FIELD[lang], text)
        assert "_is_partial_derivative" not in args, args
        assert "_is_total_derivative" not in args, args


class TestTheExpressionIdRouteIsUnchanged:
    r"""E 番号のある綴りは `diff_rule` が答える——そこは偏と全で**値が割れる**."""

    @pytest.mark.parametrize("lang,text,want", [
        ("ja", "E1 を t で偏微分して", "1"),
        ("ja", "E1 を t で全微分して", "2*q*q_dot+1"),
        ("en", "take the partial derivative of E1 with respect to t", "1"),
        ("en", "take the total derivative of E1 with respect to t", "2*q*q_dot+1"),
    ])
    def test_partial_and_total_differ(self, lang, text, want):
        setup = {"ja": ["q を時間依存関数として定義する", "L = q^2 + t"],
                 "en": ["define q as a time-dependent function", "L = q^2 + t"]}[lang]
        r, _args = _ask(lang, "analytical_mechanics", setup, text)
        assert r.get("status") == "completed", r
        assert (r.get("canonical") or "").replace(" ", "") == want, r
