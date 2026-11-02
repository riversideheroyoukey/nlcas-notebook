"""Batch symbol declaration must register EVERY name in the list, in every
language and confirmation path (J-4, J-5 — found via the S-s/S-v practice-loop
scenarios, 2026-07-05).

J-4: an English comma list of 3+ names ("Declare a_1, b_1, ..., d_2 as real.")
only carried the first and last name into raw_names, because m.groups()[:-1]
on a regex with a repeated capturing group only keeps its LAST match — a
standard regex limitation. Fixed by re-scanning the text before the copula
(" as "/" is "/" are ") with re.findall, mirroring the Japanese path.

J-5: when a batch declaration mixes in a symbol needing the subscript-role
confirmation dialog (a single-letter alphabetic subscript, e.g. c_u, c_o),
only the FIRST ambiguous name was carried into _all_names — the rest stayed
undeclared and later resolved as a bogus IndexedSymbol (c_o -> c(o)). Fixed
by applying the chosen interpretation to every ambiguous name in the list.

Also covers the "are" copula ("c_u, c_o are real numbers.") which previously
failed to parse at all (_EN_COPULA only matched "as"/"is").
"""
import asyncio

import ui.notebook_bridge as nb
from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operation_validator import OperationValidator


def _run(eng, sup, value):
    tok = InputToken(type="text", value=value)
    r = asyncio.get_event_loop().run_until_complete(
        run_interpret_pipeline(InterpretRequest(tokens=[tok]), eng, sup))
    if r.get("status") == "needs_confirmation":
        chosen = r["candidates"][-1]   # "part of the name" — the natural choice
        eng.run_operation(chosen["operation"], chosen["args"])
    return r


class TestJ4EnglishCommaList:
    def test_eight_digit_subscripted_names_all_register(self):
        nb.start_engine(title="j4", mode="standard", language="en",
                        domains=["operations_research"])
        try:
            eng = nb.get_engine()
            sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                                domains=["operations_research"],
                                primary_backend="maxima")
            _run(eng, sup,
                "Declare a_1, b_1, c_1, d_1, a_2, b_2, c_2, d_2 as real.")
            for name in ("a_1", "b_1", "c_1", "d_1",
                        "a_2", "b_2", "c_2", "d_2"):
                assert eng.symbol_context.get(name) is not None, name
        finally:
            nb.stop_engine()

    def test_japanese_comma_list_still_works(self):
        """Regression guard: the JP path (already correct) must stay correct."""
        nb.start_engine(title="j4_jp", mode="standard", language="ja",
                        domains=["operations_research"])
        try:
            eng = nb.get_engine()
            sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                                domains=["operations_research"],
                                primary_backend="maxima")
            _run(eng, sup, "a_1, b_1, c_1, d_1, a_2, b_2, c_2, d_2 を実数とする")
            for name in ("a_1", "b_1", "c_1", "d_1",
                        "a_2", "b_2", "c_2", "d_2"):
                assert eng.symbol_context.get(name) is not None, name
        finally:
            nb.stop_engine()


class TestJ5AmbiguousSubscriptBatch:
    def test_two_ambiguous_names_both_register_ja(self):
        nb.start_engine(title="j5_ja", mode="standard", language="ja",
                        domains=["supply_chain_logistics"])
        try:
            eng = nb.get_engine()
            sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                                domains=["supply_chain_logistics"],
                                primary_backend="maxima")
            r = _run(eng, sup, "c_u, c_o を実数とする")
            assert eng.symbol_context.get("c_u") is not None
            assert eng.symbol_context.get("c_o") is not None
        finally:
            nb.stop_engine()

    def test_two_ambiguous_names_both_register_en(self):
        nb.start_engine(title="j5_en", mode="standard", language="en",
                        domains=["supply_chain_logistics"])
        try:
            eng = nb.get_engine()
            sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                                domains=["supply_chain_logistics"],
                                primary_backend="maxima")
            r = _run(eng, sup, "Declare c_u, c_o as real.")
            assert eng.symbol_context.get("c_u") is not None
            assert eng.symbol_context.get("c_o") is not None
        finally:
            nb.stop_engine()

    def test_mixed_ambiguous_and_unambiguous_names(self):
        """c_u (ambiguous) + omega_0 (digit subscript, always kept) in one batch."""
        nb.start_engine(title="j5_mixed", mode="standard", language="ja",
                        domains=[])
        try:
            eng = nb.get_engine()
            sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                                domains=[], primary_backend="maxima")
            _run(eng, sup, "c_u, omega_0 を実数とする")
            assert eng.symbol_context.get("c_u") is not None
            assert eng.symbol_context.get("omega_0") is not None
        finally:
            nb.stop_engine()


class TestEnglishAreCopula:
    def test_are_phrasing_parses_and_registers_both(self):
        nb.start_engine(title="are_copula", mode="standard", language="en",
                        domains=["supply_chain_logistics"])
        try:
            eng = nb.get_engine()
            sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                                domains=["supply_chain_logistics"],
                                primary_backend="maxima")
            r = _run(eng, sup, "c_u, c_o are real numbers.")
            assert eng.symbol_context.get("c_u") is not None
            assert eng.symbol_context.get("c_o") is not None
        finally:
            nb.stop_engine()


class TestJ6EnglishTexGreekCommaList:
    """J-6 (found via the scenario-narrative-notes review, 2026-07-08): an English
    comma list mixing plain names with TeX Greek commands ("Define k, m, \\mu,
    \\hbar as scalars") only registered the LAST name.  The Japanese-only を/は
    boundary gated the \\gamma -> gamma TeX-normalization step, so \\mu/\\hbar
    stayed backslash-prefixed when the English regex ran; _SYM_PAT has no "\\"
    in its leading character class, which broke the comma-repetition chain and
    made search() silently re-anchor inside "\\hbar" (matching bare "hbar" as
    the sole symbol). Fixed by also normalizing TeX Greek commands before the
    English copula (as/is/are) when no Japanese boundary is present.
    """
    def test_tex_greek_commands_mixed_with_plain_names_all_register(self):
        nb.start_engine(title="j6_en", mode="second_quantization", language="en",
                        domains=["condensed_matter"])
        try:
            eng = nb.get_engine()
            sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                                domains=["condensed_matter"],
                                primary_backend=eng.session.primary_backend)
            _run(eng, sup, r"Define k, m, \mu, \hbar as scalars")
            for name in ("k", "m", "mu", "hbar"):
                assert eng.symbol_context.get(name) is not None, name
        finally:
            nb.stop_engine()

    def test_japanese_tex_greek_comma_list_still_works(self):
        """Regression guard: the JP path (already correct) must stay correct."""
        nb.start_engine(title="j6_jp", mode="second_quantization", language="ja",
                        domains=["condensed_matter"])
        try:
            eng = nb.get_engine()
            sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                                domains=["condensed_matter"],
                                primary_backend=eng.session.primary_backend)
            _run(eng, sup, r"k, m, \mu, \hbar をスカラーとする")
            for name in ("k", "m", "mu", "hbar"):
                assert eng.symbol_context.get(name) is not None, name
        finally:
            nb.stop_engine()
