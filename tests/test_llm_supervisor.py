"""Unit tests for LLM Supervisor (Phase 1 rule-based + Phase 2 LLM with mock).

Integration with a real LLM is not tested here; LLMClient is mocked.
"""
import pytest
from unittest.mock import MagicMock
from notebook.runtime.llm_supervisor import LLMSupervisor, _tokenize, _map_tokens_to_fields
from notebook.runtime.llm_client import LLMClient, ChatResponse, _clean_chat_text
from notebook.runtime.operation_validator import OperationValidator


def _make_julia_handler():
    """Build a minimal JuliaBackendHandler for test use."""
    from notebook.runtime.julia_command_builder import JuliaCommandBuilder
    from notebook.runtime.julia_backend_handler import JuliaBackendHandler
    return JuliaBackendHandler(JuliaCommandBuilder())


@pytest.fixture(scope="module")
def v():
    return OperationValidator()


@pytest.fixture(scope="module")
def sup(v):
    return LLMSupervisor(validator=v, llm_client=None, primary_backend="maxima")


# ---------------------------------------------------------------------------
# _tokenize helper
# ---------------------------------------------------------------------------

class TestTokenize:
    def test_english_words(self):
        assert _tokenize("diff E1 x") == ["diff", "E1", "x"]

    def test_japanese_particles_stripped(self):
        tokens = _tokenize("E1を微分する")
        # 「を」 stripped from E1を → E1
        assert "E1" in tokens

    def test_japanese_space_separated(self):
        tokens = _tokenize("E1 を x で微分")
        assert "E1" in tokens
        assert "x" in tokens

    def test_empty_returns_empty(self):
        assert _tokenize("") == []

    def test_multiple_spaces_collapsed(self):
        assert _tokenize("expand   E1") == ["expand", "E1"]


# ---------------------------------------------------------------------------
# _map_tokens_to_fields helper
# ---------------------------------------------------------------------------

class TestMapTokens:
    def test_expr_and_var(self):
        result = _map_tokens_to_fields(["E1", "x"], ["expr", "var"], {"E1"})
        assert result == {"expr": "E1", "var": "x"}

    def test_only_expr(self):
        result = _map_tokens_to_fields(["E1"], ["expr"], {"E1"})
        assert result == {"expr": "E1"}

    def test_two_expr_fields(self):
        result = _map_tokens_to_fields(
            ["E1", "E2"], ["expr", "relation_ref"], {"E1", "E2"}
        )
        assert result == {"expr": "E1", "relation_ref": "E2"}

    def test_missing_field_not_added(self):
        result = _map_tokens_to_fields(["E1"], ["expr", "var"], {"E1"})
        assert "var" not in result
        assert result["expr"] == "E1"


# ---------------------------------------------------------------------------
# Phase 1 — rule-based structured parse
# ---------------------------------------------------------------------------

class TestPhase1SingleMatch:
    def test_expand_english(self, sup):
        r = sup.interpret("expand E1", {"E1"})
        assert r.ok
        assert r.operation == "expand"
        assert r.args == {"expr": "E1"}
        assert not r.requires_confirmation
        assert r.source == "rule"

    def test_diff_english(self, sup):
        r = sup.interpret("diff E1 x", {"E1"})
        assert r.ok
        assert r.operation == "differentiate"
        assert r.args["expr"] == "E1"
        assert r.args["var"] == "x"
        assert not r.requires_confirmation

    def test_japanese_expand(self, sup):
        r = sup.interpret("E1 を展開する", {"E1"})
        assert r.ok
        assert r.operation == "expand"
        assert r.args["expr"] == "E1"

    def test_japanese_diff(self, sup):
        r = sup.interpret("E1 を x で微分", {"E1"})
        assert r.ok
        assert r.operation == "differentiate"
        assert r.args["expr"] == "E1"
        assert r.args["var"] == "x"

    def test_japanese_simplify(self, sup):
        r = sup.interpret("E2 を簡約する", {"E2"})
        assert r.ok
        assert r.operation in ("simplify", "rational_simplify")

    def test_factor_english(self, sup):
        r = sup.interpret("factor E1", {"E1"})
        assert r.ok
        assert r.operation == "factor"

    def test_laplace_english(self, sup):
        r = sup.interpret("laplace E1 t s", {"E1"})
        assert r.ok
        assert r.operation == "laplace_transform"
        assert r.args.get("t") == "t"
        assert r.args.get("s") == "s"

    def test_summation_english(self, sup):
        r = sup.interpret("summation E1 k 1 n", {"E1"})
        assert r.ok
        assert r.operation == "summation"

    def test_transpose_english(self, sup):
        r = sup.interpret("transpose E1", {"E1"})
        assert r.ok
        assert r.operation == "transpose"

    def test_numerical_eval_english(self, sup):
        r = sup.interpret("float E1", {"E1"})
        assert r.ok
        assert r.operation == "numerical_eval"


class TestPhase1NoMatch:
    def test_unknown_returns_none_internally(self, sup):
        r = sup.interpret("frobnigate E1", {"E1"})
        # No LLM configured → error result
        assert not r.ok
        assert "LLM" in r.error or "認識" not in r.error or r.error

    def test_empty_string(self, sup):
        r = sup.interpret("", {"E1"})
        # Empty text → no match
        assert not r.ok


class TestPhase1MultipleAmbiguousOps:
    def test_simplification_group_returns_confirmation(self, sup):
        # "簡約" matches both simplify and rational_simplify (both in simplification group)
        r = sup.interpret("E1 を有理簡約する", {"E1"})
        assert r.ok
        # Should match rational_simplify specifically
        assert r.operation == "rational_simplify"


class TestPhase1IncompleteArgs:
    def test_diff_missing_var_returns_error(self, sup):
        r = sup.interpret("微分する E1", {"E1"})
        # diff has required_fields ["expr", "var"] — var missing → error
        if r.ok:
            # might match something that only needs expr
            pass
        else:
            assert r.error is not None

    def test_known_op_missing_args(self, sup):
        r = sup.interpret("diff E1", {"E1"})
        # "diff" → differentiate; missing "var"
        assert not r.ok or r.requires_confirmation


class TestTaylorPointExtraction:
    """Rule-based extraction of 'point' from natural language for taylor_expand."""

    def test_point_from_x_equals_1(self, sup):
        r = sup.interpret("E1 を x=1 の周りで 3 次テイラー展開", {"E1"})
        assert r.operation == "taylor_expand"
        assert r.args.get("point") == "1"
        assert r.args.get("order") == "3"

    def test_point_from_x_equals_1_alternate_spelling(self, sup):
        # テーラー (alternate romanization) should also be recognized
        r = sup.interpret("E1 を x=1 の周りで 3 次テーラー展開", {"E1"})
        assert r.operation == "taylor_expand"
        assert r.args.get("point") == "1"
        assert r.args.get("order") == "3"

    def test_point_from_x_equals_0(self, sup):
        r = sup.interpret("E1 x=0 まわり 5 次テイラー展開", {"E1"})
        assert r.operation == "taylor_expand"
        assert r.args.get("point") == "0"
        assert r.args.get("order") == "5"

    def test_point_negative(self, sup):
        r = sup.interpret("E1 x=-1 周り 2 次テイラー展開", {"E1"})
        assert r.operation == "taylor_expand"
        assert r.args.get("point") == "-1"

    def test_no_point_falls_through(self, sup):
        # No "x=N" in input → point not set by rule (left for LLM or bridge default)
        r = sup.interpret("E1 を x について 3 次テイラー展開", {"E1"})
        assert r.operation == "taylor_expand"
        assert r.args.get("order") == "3"
        # point may or may not be set — just confirm no crash


# ---------------------------------------------------------------------------
# Phase 2 — LLM mock tests
# ---------------------------------------------------------------------------

def _make_llm_mock(response_text: str, ok: bool = True) -> MagicMock:
    # Apply the same cleaning that LLMClient.chat() performs so that mocks
    # behave consistently with the real client (think-block and fence stripping).
    mock = MagicMock(spec=LLMClient)
    cleaned = _clean_chat_text(response_text) if ok else response_text
    mock.chat.return_value = ChatResponse(ok=ok, text=cleaned)
    return mock


class TestPhase2LLMMock:
    # Inputs that Phase 1 cannot match (no token is a known alias)
    _UNRESOLVABLE = "Can you make E1 simpler please"
    _UNRESOLVABLE2 = "I want to reduce E1 somehow"
    _UNRESOLVABLE3 = "What happens if we apply transformations to E1"

    def _sup(self, v, mock_llm):
        return LLMSupervisor(validator=v, llm_client=mock_llm, primary_backend="maxima")

    def test_llm_success_expand(self, v):
        mock = _make_llm_mock('{"ok": true, "operation": "expand", "args": {"expr": "E1"}, "confidence": 0.9, "candidates": []}')
        sup = self._sup(v, mock)
        r = sup.interpret(self._UNRESOLVABLE, {"E1"})
        assert r.ok
        assert r.operation == "expand"
        assert r.args == {"expr": "E1"}
        assert r.source == "llm"
        assert r.requires_confirmation

    def test_llm_result_always_requires_confirmation(self, v):
        mock = _make_llm_mock('{"ok": true, "operation": "factor", "args": {"expr": "E2"}, "confidence": 1.0, "candidates": []}')
        sup = self._sup(v, mock)
        r = sup.interpret(self._UNRESOLVABLE, {"E2"})
        assert r.ok
        assert r.requires_confirmation  # LLM results always need confirmation

    def test_llm_alias_resolved(self, v):
        # LLM returns "radcan" (alias), should resolve to "radical_simplify"
        mock = _make_llm_mock('{"ok": true, "operation": "radcan", "args": {"expr": "E1"}, "confidence": 0.8, "candidates": []}')
        sup = self._sup(v, mock)
        r = sup.interpret(self._UNRESOLVABLE, {"E1"})
        assert r.ok
        assert r.operation == "radical_simplify"

    def test_llm_unknown_operation_returns_error(self, v):
        mock = _make_llm_mock('{"ok": true, "operation": "frobnigate", "args": {}, "confidence": 0.9, "candidates": []}')
        sup = self._sup(v, mock)
        r = sup.interpret(self._UNRESOLVABLE, set())
        assert not r.ok
        assert "unknown operation" in r.error.lower() or "frobnigate" in r.error

    def test_llm_returns_ok_false(self, v):
        mock = _make_llm_mock('{"ok": false, "error": "Cannot parse"}')
        sup = self._sup(v, mock)
        r = sup.interpret(self._UNRESOLVABLE, set())
        assert not r.ok

    def test_llm_network_error(self, v):
        mock = _make_llm_mock("", ok=False)
        mock.chat.return_value = ChatResponse(ok=False, error="Connection refused")
        sup = self._sup(v, mock)
        r = sup.interpret("expand E1", {"E1"})
        # Phase 1 matches "expand E1" — LLM is never called
        assert r.ok  # Phase 1 handles it

    def test_llm_called_only_when_phase1_fails(self, v):
        mock = _make_llm_mock('{"ok": true, "operation": "expand", "args": {"expr": "E1"}, "confidence": 0.9, "candidates": []}')
        sup = self._sup(v, mock)
        sup.interpret("expand E1", {"E1"})
        mock.chat.assert_not_called()

    def test_llm_called_for_unresolvable_text(self, v):
        mock = _make_llm_mock('{"ok": true, "operation": "expand", "args": {"expr": "E1"}, "confidence": 0.7, "candidates": []}')
        sup = self._sup(v, mock)
        sup.interpret(self._UNRESOLVABLE, {"E1"})
        mock.chat.assert_called_once()

    def test_llm_invalid_json_returns_error(self, v):
        mock = _make_llm_mock("not json at all")
        sup = self._sup(v, mock)
        r = sup.interpret(self._UNRESOLVABLE, {"E1"})
        assert not r.ok
        assert "JSON" in r.error or "json" in r.error.lower()

    def test_llm_markdown_fences_stripped(self, v):
        mock = _make_llm_mock('```json\n{"ok": true, "operation": "expand", "args": {"expr": "E1"}, "confidence": 0.8, "candidates": []}\n```')
        sup = self._sup(v, mock)
        r = sup.interpret(self._UNRESOLVABLE, {"E1"})
        assert r.ok
        assert r.operation == "expand"

    def test_llm_confirmation_prompt_contains_display_ja(self, v):
        mock = _make_llm_mock('{"ok": true, "operation": "expand", "args": {"expr": "E1"}, "confidence": 0.8, "candidates": []}')
        sup = self._sup(v, mock)
        r = sup.interpret(self._UNRESOLVABLE, {"E1"})
        assert r.ok
        assert "展開" in r.confirmation_prompt  # display.ja for expand

    def test_llm_candidates_included(self, v):
        mock = _make_llm_mock(
            '{"ok": true, "operation": "simplify", "args": {"expr": "E1"}, "confidence": 0.6, '
            '"candidates": [{"operation": "rational_simplify", "args": {"expr": "E1"}}]}'
        )
        sup = self._sup(v, mock)
        r = sup.interpret(self._UNRESOLVABLE, {"E1"})
        assert r.ok
        assert len(r.candidates) >= 1

    def test_symbol_updates_missing_defaults_to_empty(self, v):
        mock = _make_llm_mock(
            '{"ok": true, "operation": "expand", "args": {"expr": "E1"}, "confidence": 0.9, "candidates": []}'
        )
        sup = self._sup(v, mock)
        r = sup.interpret(self._UNRESOLVABLE, {"E1"})
        assert r.ok
        assert r.symbol_updates == []

    def test_dependencies_missing_defaults_to_empty(self, v):
        mock = _make_llm_mock(
            '{"ok": true, "operation": "expand", "args": {"expr": "E1"}, "confidence": 0.9, "candidates": []}'
        )
        sup = self._sup(v, mock)
        r = sup.interpret(self._UNRESOLVABLE, {"E1"})
        assert r.ok
        assert r.dependencies == []

    def test_symbol_updates_and_dependencies_both_extracted(self, v):
        mock = _make_llm_mock(
            '{"ok": true, "operation": "expand", "args": {"expr": "E1"}, "confidence": 0.9, "candidates": [], '
            '"symbol_updates": [{"name": "H", "kind": "function", "dependencies": ["p", "q"]}], '
            '"dependencies": [{"dependent": "H", "independent": ["p", "q"]}]}'
        )
        sup = self._sup(v, mock)
        r = sup.interpret(self._UNRESOLVABLE, {"E1"})
        assert r.ok
        assert len(r.symbol_updates) == 1
        assert r.symbol_updates[0]["name"] == "H"
        assert r.symbol_updates[0]["kind"] == "function"
        assert len(r.dependencies) == 1
        assert r.dependencies[0]["dependent"] == "H"
        assert "p" in r.dependencies[0]["independent"]

    def test_llm_abstract_lagrangian_dependencies(self, v):
        mock = _make_llm_mock(
            '{"ok": true, "operation": "differentiate", "args": {"expr": "L", "var": "q"}, "confidence": 0.9, '
            '"symbol_updates": [{"name": "L", "kind": "function", "dependencies": ["q", "dot_q", "t"]}], '
            '"dependencies": [{"dependent": "L", "independent": ["q", "dot_q", "t"]}]}'
        )
        sup = self._sup(v, mock)
        r = sup.interpret("Lからオイラー・ラグランジュ方程式を計算して", {"L"})
        assert r.ok
        assert len(r.symbol_updates) == 1
        assert r.symbol_updates[0]["name"] == "L"
        assert r.symbol_updates[0]["dependencies"] == ["q", "dot_q", "t"]
        assert len(r.dependencies) == 1
        assert r.dependencies[0]["dependent"] == "L"
        assert "q" in r.dependencies[0]["independent"]


# ---------------------------------------------------------------------------
# System prompt generation
# ---------------------------------------------------------------------------

class TestSystemPrompt:
    def test_prompt_contains_all_maxima_ops(self, v):
        sup = LLMSupervisor(validator=v, primary_backend="maxima")
        prompt = sup._build_system_prompt({"E1", "E2"})
        maxima_ops = v.operations_for_backend("maxima")
        for op in maxima_ops:
            assert op in prompt, f"'{op}' missing from system prompt"

    def test_prompt_contains_known_expr_ids(self, v):
        sup = LLMSupervisor(validator=v, primary_backend="maxima")
        prompt = sup._build_system_prompt({"E1", "E3"})
        assert "E1" in prompt
        assert "E3" in prompt

    def test_prompt_no_cas_syntax(self, v):
        sup = LLMSupervisor(validator=v, primary_backend="maxima")
        prompt = sup._build_system_prompt(set())
        # Should not contain raw Maxima commands
        assert "ratsimp(" not in prompt
        assert "trigexpand(" not in prompt

    def test_prompt_includes_symbol_context_registry(self, v):
        from notebook.models.symbol_context import SymbolContext
        ctx = SymbolContext()
        ctx.register_quantity("L", kind="function")
        sup = LLMSupervisor(validator=v, primary_backend="maxima")
        prompt = sup._build_system_prompt(set(), symbol_context=ctx)
        assert "L" in prompt
        assert "function" in prompt
        assert "Current Symbol Context:" in prompt

    def test_prompt_excludes_empty_symbol_context(self, v):
        from notebook.models.symbol_context import SymbolContext
        ctx = SymbolContext()  # empty registry
        sup = LLMSupervisor(validator=v, primary_backend="maxima")
        prompt = sup._build_system_prompt(set(), symbol_context=ctx)
        assert "Current Symbol Context:" not in prompt


# ---------------------------------------------------------------------------
# LLMClient.find_endpoint — scope-based routing
# ---------------------------------------------------------------------------

class TestFindEndpoint:
    """Tests for LLMClient.find_endpoint() using the real models/ directory."""

    @pytest.fixture(scope="class")
    def models_root(self):
        from pathlib import Path
        return Path(__file__).resolve().parent.parent / "models"

    def test_find_local_command_parser(self, models_root):
        from notebook.runtime.llm_client import LLMClient
        client = LLMClient.find_endpoint(
            role="command_parser", scope="local", models_root=models_root
        )
        assert client is not None
        assert client.scope == "local"

    def test_find_local_advisor(self, models_root):
        from notebook.runtime.llm_client import LLMClient
        client = LLMClient.find_endpoint(
            role="advisor", scope="local", models_root=models_root
        )
        assert client is not None
        assert client.scope == "local"

    def test_find_cloud_advisor(self, models_root):
        from notebook.runtime.llm_client import LLMClient
        client = LLMClient.find_endpoint(
            role="advisor", scope="cloud", models_root=models_root
        )
        assert client is not None
        assert client.scope == "cloud"

    def test_cloud_has_no_command_parser(self, models_root):
        from notebook.runtime.llm_client import LLMClient
        # cloud_openai_gpt54 only defines "advisor" role — command_parser should NOT match
        client = LLMClient.find_endpoint(
            role="command_parser", scope="cloud", models_root=models_root
        )
        assert client is None

    def test_find_any_scope_returns_first_match(self, models_root):
        from notebook.runtime.llm_client import LLMClient
        client = LLMClient.find_endpoint(
            role="advisor", scope=None, models_root=models_root
        )
        assert client is not None

    def test_unknown_role_returns_none(self, models_root):
        from notebook.runtime.llm_client import LLMClient
        client = LLMClient.find_endpoint(
            role="nonexistent_role", scope="local", models_root=models_root
        )
        assert client is None

    def test_scope_attribute_set_correctly(self, models_root):
        from notebook.runtime.llm_client import LLMClient
        local = LLMClient.find_endpoint(role="advisor", scope="local", models_root=models_root)
        cloud = LLMClient.find_endpoint(role="advisor", scope="cloud", models_root=models_root)
        assert local.scope == "local"
        assert cloud.scope == "cloud"

    def test_endpoint_name_set(self, models_root):
        from notebook.runtime.llm_client import LLMClient
        client = LLMClient.find_endpoint(
            role="command_parser", scope="local", models_root=models_root
        )
        assert client.endpoint_name  # non-empty


# ---------------------------------------------------------------------------
# Julia-backend Phase 1 rule tests (covers ① E2E automated side)
# ---------------------------------------------------------------------------

class TestJuliaPhase1Rules:
    """Verify that NL instructions for Julia-specific operations resolve via
    Phase 1 (rule-based alias scan) without LLM, covering the alias table and
    Japanese particle stripping used in real sessions."""

    @pytest.fixture(scope="class")
    def jsup(self, v):
        return LLMSupervisor(validator=v, llm_client=None, primary_backend="julia_full")

    @pytest.fixture(scope="class")
    def jhandler(self):
        return _make_julia_handler()

    def test_commutator_ja(self, jsup):
        r = jsup.interpret("交換子を計算して", {"a": None, "b": None})
        assert r.operation == "commutator"

    def test_commutator_with_operator_args(self, jsup):
        # Julia operator expressions a(1) and a'(1) must be extracted as args a/b
        r = jsup.interpret("交換子を計算して a(1) と a'(1)", {})
        assert r.operation == "commutator"
        assert r.args.get("a") == "a(1)"
        assert r.args.get("b") == "a'(1)"
        assert r.ok

    def test_commutator_with_comma_separator(self, jsup):
        # Japanese comma (、) is a valid separator
        r = jsup.interpret("交換子を計算して、 a(1) と a'(1)", {})
        assert r.operation == "commutator"
        assert r.args.get("a") == "a(1)"
        assert r.args.get("b") == "a'(1)"

    def test_commutator_en(self, jsup):
        r = jsup.interpret("commutator a b", {"a": None, "b": None})
        assert r.operation == "commutator"

    def test_anticommutator_ja(self, jsup):
        r = jsup.interpret("反交換子を計算する", {"a": None, "b": None})
        assert r.operation == "anticommutator"

    # -- bracket notation (Phase 0) --

    def test_bracket_commutator_operators(self, jsup):
        r = jsup.interpret("[a(1), a'(1)]", {})
        assert r.ok
        assert r.operation == "commutator"
        assert r.args["a"] == "a(1)"
        assert r.args["b"] == "a'(1)"
        assert r.source == "rule"
        assert not r.requires_confirmation

    def test_bracket_commutator_expr_ids(self, jsup):
        r = jsup.interpret("[E1, E2]", {"E1": None, "E2": None})
        assert r.ok
        assert r.operation == "commutator"
        assert r.args["a"] == "E1"
        assert r.args["b"] == "E2"

    def test_brace_anticommutator_operators(self, jsup):
        r = jsup.interpret("{f(1), f'(1)}", {})
        assert r.ok
        assert r.operation == "anticommutator"
        assert r.args["a"] == "f(1)"
        assert r.args["b"] == "f'(1)"
        assert r.source == "rule"
        assert not r.requires_confirmation

    def test_brace_anticommutator_expr_ids(self, jsup):
        r = jsup.interpret("{E3, E4}", {"E3": None, "E4": None})
        assert r.ok
        assert r.operation == "anticommutator"
        assert r.args["a"] == "E3"
        assert r.args["b"] == "E4"

    def test_bracket_with_nl_suffix(self, jsup):
        # "[X,Y] を計算して" — trailing NL suffix must still resolve via Phase 0
        r = jsup.interpret("[a(1), a'(1)] を計算して", {})
        assert r.ok
        assert r.operation == "commutator"
        assert r.args["a"] == "a(1)"
        assert r.args["b"] == "a'(1)"
        assert r.source == "rule"

    def test_brace_with_nl_suffix(self, jsup):
        r = jsup.interpret("{f(1), f'(1)}を計算して", {})
        assert r.ok
        assert r.operation == "anticommutator"
        assert r.args["a"] == "f(1)"
        assert r.args["b"] == "f'(1)"

    def test_bracket_not_matched_on_maxima(self):
        # commutator is Julia-only — bracket parse must not fire on Maxima backend
        from notebook.runtime.operation_validator import OperationValidator
        v = OperationValidator()
        sup = LLMSupervisor(validator=v, llm_client=None, primary_backend="maxima")
        r = sup.interpret("[a, b]", {})
        assert r.operation != "commutator"

    def test_normal_order_ja(self, jsup):
        r = jsup.interpret("正規順序にして E1", {"E1": None})
        assert r.operation == "normal_order"

    def test_normal_order_canonical(self, jsup):
        r = jsup.interpret("正規順序化 E1", {"E1": None})
        assert r.operation == "normal_order"

    def test_vacuum_expectation_ja(self, jsup):
        r = jsup.interpret("真空期待値を計算する E1", {"E1": None})
        assert r.operation == "vacuum_expectation"

    def test_heisenberg_equation_ja(self, jsup):
        r = jsup.interpret("ハイゼンベルク方程式を作る", {})
        assert r.operation == "heisenberg_equation"

    def test_heisenberg_equation_en(self, jsup):
        r = jsup.interpret("heisenberg E1 H", {"E1": None, "H": None})
        assert r.operation == "heisenberg_equation"

    def test_heisenberg_equation_args_extracted(self, jsup):
        r = jsup.interpret("ハイゼンベルク方程式 a(1) E5", {"E5": None})
        assert r.operation == "heisenberg_equation"
        assert r.args.get("operator") == "a(1)"
        assert r.args.get("hamiltonian") == "E5"

    def test_diff_dot_syntax(self, jsup):
        # Regression for "DIFF. E4" — trailing period must be stripped
        r = jsup.interpret("DIFF. E1", {"E1": None})
        assert r.operation == "differentiate"

    def test_define_boson_operator_ja(self, jsup):
        r = jsup.interpret("ボゾン演算子を定義する", {})
        assert r.operation == "define_boson_operator"

    def test_define_boson_operator_bose(self, jsup):
        # "ボーズ演算子を定義する" — stripped form "ボーズ演算子を定義" must be in alias index
        r = jsup.interpret("ボーズ演算子を定義する", {})
        assert r.operation == "define_boson_operator"

    def test_define_boson_operator_photon(self, jsup):
        r = jsup.interpret("光子演算子を定義する", {})
        assert r.operation == "define_boson_operator"

    def test_define_fermion_operator_ja(self, jsup):
        r = jsup.interpret("フェルミオン演算子を定義する", {})
        assert r.operation == "define_fermion_operator"

    def test_define_fermion_operator_short(self, jsup):
        # "フェルミ演算子を定義する" — shorter form, was hanging (MISS → LLM)
        r = jsup.interpret("フェルミ演算子を定義する", {})
        assert r.operation == "define_fermion_operator"

    # -- particle operator (Phase 0.5 via JuliaBackendHandler.extra_pre_phase1) --

    def test_particle_op_photon_pattern_a(self, jsup, jhandler):
        # "a'(k) を 光子 の 生成演算子 とする" → boson, base name a
        r = jsup.interpret("a'(k)を光子の生成演算子とする", {}, handler=jhandler)
        assert r.ok
        assert r.operation == "define_boson_operator"
        assert r.args["name"] == "a"
        assert r.args["_particle_name"] == "光子"
        assert r.args["_op_type_key"] == "creation"
        assert r.source == "rule"
        assert not r.requires_confirmation

    def test_particle_op_electron_pattern_b(self, jsup, jhandler):
        # "電子の消滅演算子を c と定義して" → fermion, base name c
        r = jsup.interpret("電子の消滅演算子をcと定義して", {}, handler=jhandler)
        assert r.ok
        assert r.operation == "define_fermion_operator"
        assert r.args["name"] == "c"
        assert r.args["_particle_name"] == "電子"
        assert r.args["_op_type_key"] == "annihilation"

    def test_particle_op_phonon_boson(self, jsup, jhandler):
        # フォノン → boson
        r = jsup.interpret("フォノンの消滅演算子をbと定義する", {}, handler=jhandler)
        assert r.ok
        assert r.operation == "define_boson_operator"
        assert r.args["name"] == "b"
        assert r.args["_particle_name"] == "フォノン"

    def test_particle_op_hole_fermion(self, jsup, jhandler):
        # 正孔 → fermion
        r = jsup.interpret("hを正孔の消滅演算子とする", {}, handler=jhandler)
        assert r.ok
        assert r.operation == "define_fermion_operator"
        assert r.args["name"] == "h"

    def test_particle_op_op_expr_display(self, jsup, jhandler):
        # _op_expr_display preserves the full expression including prime and args
        r = jsup.interpret("a'(k)を光子の生成演算子とする", {}, handler=jhandler)
        assert r.args.get("_op_expr_display") == "a'(k)"

    def test_particle_op_base_name_strips_prime(self, jsup, jhandler):
        # base name must be just "a", not "a'" or "a'(k)"
        r = jsup.interpret("a'(k)を光子の生成演算子とする", {}, handler=jhandler)
        assert r.args["name"] == "a"

    def test_particle_op_not_matched_without_handler(self, jsup):
        # without a handler the particle parse hook is never called
        r = jsup.interpret("aを光子の生成演算子とする", {})
        assert r.operation != "define_boson_operator"

    def test_particle_op_not_matched_on_maxima(self):
        # JuliaBackendHandler.extra_pre_phase1 is only wired in Julia sessions;
        # Maxima handler returns None for extra_pre_phase1
        from notebook.runtime.maxima_backend_handler import MaximaBackendHandler
        from notebook.runtime.maxima_adapter import MaximaAdapter
        # MaximaBackendHandler.extra_pre_phase1 inherits the default no-op
        v = OperationValidator()
        sup = LLMSupervisor(validator=v, llm_client=None, primary_backend="maxima")
        from notebook.runtime.backend_handler import BackendHandler
        # Use a plain BackendHandler-derived stub that does nothing
        class _Stub(BackendHandler):
            def build_command(self, *a, **k): ...
            def wrap_command_for_storage(self, *a, **k): return ""
            def canonical_for_result(self, *a, **k): return ""
            def postprocess_result_latex(self, *a, **k): return ""
            def format_constant_for_latex(self, t): return t
            def parse_quantity(self, *a, **k): return None
            def canonical_variables(self, *a, **k): return set()
            def qty_name_from_origin(self, o): return o
            def wrt_qdot_canonical(self, *a, **k): return ""
            def declare_depends(self, *a, **k): pass
            @property
            def supports_equation_source(self): return False
            def latex_to_canonical(self, l): return l
            def parse_ir(self, *a, **k): return None
        stub = _Stub()
        r = sup.interpret("aを光子の生成演算子とする", {}, handler=stub)
        assert r.operation != "define_boson_operator"

    def test_particle_op_expr_id_guard(self, jhandler):
        # If the formula part was auto-registered before reaching extra_pre_phase1,
        # supervisor_text becomes "E7 を光子の生成演算子とする".
        # extra_pre_phase1 must return None (not yield name="E7" which would
        # cause @boson_ops E7 in Julia).
        result = jhandler.extra_pre_phase1("E7 を光子の生成演算子とする", "ja")
        assert result is None

    def test_bypass_auto_register_particle_definition(self, jhandler):
        # should_bypass_auto_register_formula returns True for particle def patterns
        assert jhandler.should_bypass_auto_register_formula("a'(k)", "を光子の生成演算子とする")
        assert jhandler.should_bypass_auto_register_formula("b", "の消滅演算子をbとする")
        assert jhandler.should_bypass_auto_register_formula("c'(1)", "is the creation operator for phonon")
        # Unrelated operation text must not suppress registration
        assert not jhandler.should_bypass_auto_register_formula("a'(k)", "を展開して")
        assert not jhandler.should_bypass_auto_register_formula("a(1)", "を正規順序にして")

    # -- scalar parameter (extra_pre_phase1) --

    def test_scalar_param_ja_basic(self, jhandler):
        r = jhandler.extra_pre_phase1("xi をスカラーパラメータとする", "ja")
        assert r is not None
        assert r.operation == "define_scalar_parameter"
        assert r.args["name"] == "xi"
        assert r.source == "rule"
        assert not r.requires_confirmation

    def test_scalar_param_ja_subscript(self, jhandler):
        # xi_k → base name xi (letter subscript = momentum index, stripped)
        r = jhandler.extra_pre_phase1("xi_k をスカラーパラメータとする", "ja")
        assert r is not None
        assert r.operation == "define_scalar_parameter"
        assert r.args["name"] == "xi"

    def test_scalar_param_ja_short_form(self, jhandler):
        r = jhandler.extra_pre_phase1("Delta をスカラーとする", "ja")
        assert r is not None
        assert r.operation == "define_scalar_parameter"
        assert r.args["name"] == "Delta"

    def test_scalar_param_en_basic(self, jhandler):
        r = jhandler.extra_pre_phase1("define xi as a scalar parameter", "en")
        assert r is not None
        assert r.operation == "define_scalar_parameter"
        assert r.args["name"] == "xi"

    def test_scalar_param_en_is_scalar(self, jhandler):
        r = jhandler.extra_pre_phase1("Delta is a scalar", "en")
        assert r is not None
        assert r.operation == "define_scalar_parameter"
        assert r.args["name"] == "Delta"

    def test_scalar_param_bypass(self, jhandler):
        assert jhandler.should_bypass_auto_register_formula("xi_k", "をスカラーパラメータとする")
        assert jhandler.should_bypass_auto_register_formula("Delta", "をスカラーとする")
        assert jhandler.should_bypass_auto_register_formula("xi", "as a scalar parameter")
        assert not jhandler.should_bypass_auto_register_formula("xi", "を展開して")

    def test_scalar_param_all_names(self, jhandler):
        # _all_names must contain all provided base names
        r = jhandler.extra_pre_phase1("xi をスカラーパラメータとする", "ja")
        assert r is not None
        assert "xi" in r.args.get("_all_names", [])


def _make_maxima_handler():
    """Build a minimal MaximaBackendHandler for test use."""
    from notebook.runtime.maxima_command_builder import MaximaCommandBuilder
    from notebook.runtime.maxima_backend_handler import MaximaBackendHandler
    return MaximaBackendHandler(MaximaCommandBuilder())


class TestTypeDeclarationPatterns:
    """Verify the shared BackendHandler type-declaration patterns (scalar/vector/matrix/tensor)
    work through both Maxima and Julia handlers."""

    @pytest.fixture(scope="class")
    def mh(self):
        return _make_maxima_handler()

    @pytest.fixture(scope="class")
    def jh(self):
        return _make_julia_handler()

    # ── scalar ────────────────────────────────────────────────────────────────

    def test_scalar_ja_maxima(self, mh):
        r = mh.extra_pre_phase1("xi をスカラーパラメータとする", "ja")
        assert r is not None and r.operation == "define_scalar_parameter"
        assert r.args["name"] == "xi"

    def test_scalar_ja_subscript(self, mh):
        # omega_0 → full name omega_0 (digit subscript = part of name, not an index)
        r = mh.extra_pre_phase1("omega_0 をスカラーとする", "ja")
        assert r is not None and r.operation == "define_scalar_parameter"
        assert r.args["name"] == "omega_0"

    def test_scalar_en_maxima(self, mh):
        r = mh.extra_pre_phase1("define lambda as a scalar parameter", "en")
        assert r is not None and r.operation == "define_scalar_parameter"

    def test_scalar_is_scalar_en(self, mh):
        r = mh.extra_pre_phase1("kappa is a scalar", "en")
        assert r is not None and r.operation == "define_scalar_parameter"

    def test_scalar_ja_julia(self, jh):
        # xi_k → base name xi (letter subscript = momentum index)
        r = jh.extra_pre_phase1("xi_k をスカラーとする", "ja")
        assert r is not None and r.operation == "define_scalar_parameter"
        assert r.args["name"] == "xi"

    # ── vector ────────────────────────────────────────────────────────────────

    def test_vector_ja_maxima(self, mh):
        r = mh.extra_pre_phase1("v をベクトルとする", "ja")
        assert r is not None and r.operation == "define_vector"
        assert r.args["name"] == "v"

    def test_vector_en_maxima(self, mh):
        r = mh.extra_pre_phase1("define v as a vector", "en")
        assert r is not None and r.operation == "define_vector"

    def test_vector_is_vector(self, mh):
        r = mh.extra_pre_phase1("A is a vector", "en")
        assert r is not None and r.operation == "define_vector"

    def test_vector_ja_julia(self, jh):
        r = jh.extra_pre_phase1("v をベクトルとする", "ja")
        assert r is not None and r.operation == "define_vector"

    # ── matrix ────────────────────────────────────────────────────────────────

    def test_matrix_ja_maxima(self, mh):
        r = mh.extra_pre_phase1("A を行列とする", "ja")
        assert r is not None and r.operation == "define_matrix"
        assert r.args["name"] == "A"

    def test_matrix_en_maxima(self, mh):
        r = mh.extra_pre_phase1("define M as a matrix", "en")
        assert r is not None and r.operation == "define_matrix"

    def test_matrix_katakana(self, mh):
        r = mh.extra_pre_phase1("H をマトリックスとする", "ja")
        assert r is not None and r.operation == "define_matrix"

    def test_matrix_ja_julia(self, jh):
        r = jh.extra_pre_phase1("M を行列とする", "ja")
        assert r is not None and r.operation == "define_matrix"

    # ── tensor ────────────────────────────────────────────────────────────────

    def test_tensor_ja_maxima(self, mh):
        r = mh.extra_pre_phase1("T をテンソルとする", "ja")
        assert r is not None and r.operation == "define_tensor"
        assert r.args["name"] == "T"

    def test_tensor_en_maxima(self, mh):
        r = mh.extra_pre_phase1("define T as a tensor", "en")
        assert r is not None and r.operation == "define_tensor"

    # ── bypass guard ──────────────────────────────────────────────────────────

    def test_bypass_scalar(self, mh):
        assert mh.should_bypass_auto_register_formula("xi_k", "をスカラーとする")

    def test_bypass_vector(self, mh):
        assert mh.should_bypass_auto_register_formula("v", "をベクトルとする")

    def test_bypass_matrix(self, mh):
        assert mh.should_bypass_auto_register_formula("A", "を行列とする")

    def test_bypass_en_vector(self, mh):
        assert mh.should_bypass_auto_register_formula("v", "as a vector")

    def test_no_bypass_unrelated(self, mh):
        assert not mh.should_bypass_auto_register_formula("v", "を微分して")

    # ── は/とは post-definition style ─────────────────────────────────────────

    def test_wa_scalar_basic(self, mh):
        r = mh.extra_pre_phase1("m は質量", "ja")
        assert r is not None and r.operation == "define_scalar_parameter"
        assert r.args["name"] == "m"

    def test_wa_scalar_dearu(self, mh):
        r = mh.extra_pre_phase1("m は質量である", "ja")
        assert r is not None and r.operation == "define_scalar_parameter"

    def test_wa_scalar_desu(self, mh):
        r = mh.extra_pre_phase1("m は質量です", "ja")
        assert r is not None and r.operation == "define_scalar_parameter"

    def test_wa_vector(self, mh):
        r = mh.extra_pre_phase1("F はベクトルである", "ja")
        assert r is not None and r.operation == "define_vector"
        assert r.args["name"] == "F"

    def test_wa_matrix(self, mh):
        r = mh.extra_pre_phase1("H は行列", "ja")
        assert r is not None and r.operation == "define_matrix"

    def test_towa_scalar(self, mh):
        r = mh.extra_pre_phase1("xi とはスカラーパラメータ", "ja")
        assert r is not None and r.operation == "define_scalar_parameter"
        assert r.args["name"] == "xi"

    def test_wa_multi_symbol(self, mh):
        r = mh.extra_pre_phase1("m と g は実数スカラー", "ja")
        assert r is not None and r.operation == "define_scalar_parameter"
        assert "m" in r.args.get("_all_names", [r.args["name"]])

    def test_wa_no_match_bare_sentence(self, mh):
        # 「結果はベクトル」— no symbol token before は → should NOT match
        r = mh.extra_pre_phase1("結果はベクトル", "ja")
        assert r is None

    def test_wa_no_match_substitution(self, mh):
        # 「x は 0」— label is a bare number → _looks_like_type_label returns False
        r = mh.extra_pre_phase1("x は 0", "ja")
        assert r is None

    def test_wa_no_match_single_char_label(self, mh):
        # 「x は a」— label is a bare single identifier
        r = mh.extra_pre_phase1("x は a", "ja")
        assert r is None

    def test_wa_with_injected_expr_id_prefix(self, mh):
        # Supervisor injects "E1 に対して" prefix — must still match
        r = mh.extra_pre_phase1("E1 に対して m は質量", "ja")
        assert r is not None and r.operation == "define_scalar_parameter"
        assert r.args["name"] == "m"

    def test_wa_vector_with_injected_prefix(self, mh):
        r = mh.extra_pre_phase1("E2 に対して F はベクトルである", "ja")
        assert r is not None and r.operation == "define_vector"
        assert r.args["name"] == "F"

    # ── void messages ─────────────────────────────────────────────────────────

    def test_void_msg_vector_ja(self, mh):
        # Message is now description-only; symbol name lives in symbol_display.
        msg = mh.format_void_op_message("define_vector", {"name": "v", "_raw_names": ["v"]}, "ja")
        assert msg and "ベクトル" in msg
        sym = mh.get_void_op_symbol_display("define_vector", {"name": "v", "_raw_names": ["v"]})
        assert sym == "v"

    def test_void_msg_matrix_en(self, mh):
        msg = mh.format_void_op_message("define_matrix", {"name": "M", "_raw_names": ["M"]}, "en")
        assert msg and "matrix" in msg.lower()
        sym = mh.get_void_op_symbol_display("define_matrix", {"name": "M", "_raw_names": ["M"]})
        assert sym == "M"

    def test_void_msg_tensor_ja(self, mh):
        msg = mh.format_void_op_message("define_tensor", {"name": "T", "_raw_names": ["T"]}, "ja")
        assert msg and "テンソル" in msg
        sym = mh.get_void_op_symbol_display("define_tensor", {"name": "T", "_raw_names": ["T"]})
        assert sym == "T"

    # ── keyword-in-label inference ────────────────────────────────────────────

    def test_momentum_vector_ja(self, mh):
        """'k を運動量ベクトルとする' → ベクトル keyword found → define_vector."""
        r = mh.extra_pre_phase1("k を運動量ベクトルとする", "ja")
        assert r is not None and r.operation == "define_vector"
        assert r.args["name"] == "k"

    def test_metric_tensor_ja(self, mh):
        """'g を計量テンソルとする' → テンソル keyword → define_tensor."""
        r = mh.extra_pre_phase1("g を計量テンソルとする", "ja")
        assert r is not None and r.operation == "define_tensor"

    def test_hamiltonian_matrix_ja(self, mh):
        """'H を行列ハミルトニアンとする' → 行列 keyword → define_matrix."""
        r = mh.extra_pre_phase1("H を行列ハミルトニアンとする", "ja")
        assert r is not None and r.operation == "define_matrix"

    def test_electron_energy_defaults_scalar(self, mh):
        """'xi_k を１電子エネルギーとする' → define_scalar_parameter, base name xi (letter subscript stripped)."""
        r = mh.extra_pre_phase1("xi_k を１電子エネルギーとする", "ja")
        assert r is not None and r.operation == "define_scalar_parameter"
        assert r.args["name"] == "xi"

    def test_order_parameter_defaults_scalar(self, mh):
        """'Delta を BCS 秩序パラメータとする' → no keyword → define_scalar_parameter."""
        r = mh.extra_pre_phase1("Delta を BCS 秩序パラメータとする", "ja")
        assert r is not None and r.operation == "define_scalar_parameter"
        assert r.args["name"] == "Delta"

    def test_momentum_en(self, mh):
        """'define k as momentum vector' → 'vector' keyword → define_vector."""
        r = mh.extra_pre_phase1("define k as momentum vector", "en")
        assert r is not None and r.operation == "define_vector"

    def test_energy_en_defaults_scalar(self, mh):
        """'define xi as single-particle energy' → no type keyword → define_scalar_parameter."""
        r = mh.extra_pre_phase1("define xi as single-particle energy", "en")
        assert r is not None and r.operation == "define_scalar_parameter"

    # ── false-positive guards ─────────────────────────────────────────────────

    def test_numeric_assignment_not_matched(self, mh):
        """'x を 0 とする' (assignment) must not fire the type-declaration hook."""
        assert mh.extra_pre_phase1("x を 0 とする", "ja") is None

    def test_single_letter_label_not_matched(self, mh):
        """'x を a とする' (substitution) must not fire the type-declaration hook."""
        assert mh.extra_pre_phase1("x を a とする", "ja") is None

    # ── bypass with physics labels ────────────────────────────────────────────

    def test_bypass_momentum_vector_suffix(self, mh):
        """Suffix 'を運動量ベクトルとする' must trigger bypass."""
        assert mh.should_bypass_auto_register_formula("k", "を運動量ベクトルとする")

    def test_bypass_energy_suffix(self, mh):
        """Suffix 'を１電子エネルギーとする' must trigger bypass."""
        assert mh.should_bypass_auto_register_formula("xi_k", "を１電子エネルギーとする")

    def test_bypass_en_momentum_vector(self, mh):
        """'as momentum vector' suffix must trigger bypass."""
        assert mh.should_bypass_auto_register_formula("k", "as momentum vector")

    # ── unrelated text returns None ───────────────────────────────────────────

    def test_unrelated_returns_none(self, mh):
        assert mh.extra_pre_phase1("E1 を微分する", "ja") is None
        assert mh.extra_pre_phase1("expand E1", "en") is None


# ---------------------------------------------------------------------------
# needs_expression_injection guard
# ---------------------------------------------------------------------------

class TestNeedsExpressionInjection:
    """Verify that needs_expression_injection() correctly suppresses auto-inject
    when the user already supplied operands or the operation doesn't use expressions."""

    @pytest.fixture(scope="class")
    def jsup(self, v):
        return LLMSupervisor(validator=v, llm_client=None, primary_backend="julia_full")

    def test_define_boson_no_inject(self, jsup):
        # operator-definition takes only 'name' — no expression needed
        assert jsup.needs_expression_injection("ボゾン演算子を定義する") is False

    def test_define_fermion_no_inject(self, jsup):
        assert jsup.needs_expression_injection("フェルミオン演算子を定義する") is False

    def test_commutator_with_args_no_inject(self, jsup):
        # both operands a(1), a'(1) already in text
        assert jsup.needs_expression_injection("交換子を計算して a(1) と a'(1)") is False

    def test_commutator_without_args_needs_inject(self, jsup):
        # no operands → should inject most-recent expression
        assert jsup.needs_expression_injection("交換子を計算して") is True

    def test_normal_order_needs_inject(self, jsup):
        # normal_order requires 'expr' and none is provided
        assert jsup.needs_expression_injection("正規順序にして") is True

    def test_vacuum_expectation_with_arg_no_inject(self, jsup):
        # 'a(1)' fills the single required expr field
        assert jsup.needs_expression_injection("真空期待値 a(1)") is False

    def test_vacuum_expectation_without_arg_needs_inject(self, jsup):
        assert jsup.needs_expression_injection("真空期待値") is True

    def test_heisenberg_with_both_args_no_inject(self, jsup):
        assert jsup.needs_expression_injection("ハイゼンベルク方程式 a(1) E5") is False

    def test_heisenberg_without_args_needs_inject(self, jsup):
        assert jsup.needs_expression_injection("ハイゼンベルク方程式") is True


# ---------------------------------------------------------------------------
# Julia-backend Phase 2 LLM mock tests (covers ② supervisor→Julia mapping)
# ---------------------------------------------------------------------------

class TestJuliaLLMMock:
    """Verify that LLM-returned operation names for Julia operations are
    correctly resolved (or rejected when the hint sends wrong names)."""

    def _jsup(self, v, mock_llm):
        return LLMSupervisor(validator=v, llm_client=mock_llm, primary_backend="julia_full")

    # Phase 1 must NOT claim this, or the LLM is never consulted and these tests
    # stop testing anything.  `A と B の交換子を求めてほしい` used to qualify; since
    # 2026-08-22 the span-based bracket phrase reads it correctly WITHOUT the LLM
    # (that is the point of the rule — a product operand written with spaces was
    # being truncated), so the input moved to a phrasing with no `A と B の…`
    # shape at all.  Verified unclaimed by `RuleBasedIntentParser.parse`.
    _INPUT = "A、B に対する交換関係を出して"

    def test_llm_commutator_correct_name(self, v):
        """LLM returns 'commutator' (the registered name) → resolves OK."""
        mock = _make_llm_mock(
            '{"ok": true, "operation": "commutator", "args": {"a": "A", "b": "B"}, '
            '"confidence": 0.9, "candidates": []}'
        )
        r = self._jsup(v, mock).interpret(self._INPUT, {"A": None, "B": None})
        assert r.ok
        assert r.operation == "commutator"
        assert r.args == {"a": "A", "b": "B"}

    def test_llm_wrong_name_comm_rejected(self, v):
        """LLM returns 'comm' (wrong name from a bad hint) → unknown operation error."""
        mock = _make_llm_mock(
            '{"ok": true, "operation": "comm", "args": {"a": "A", "b": "B"}, '
            '"confidence": 0.9, "candidates": []}'
        )
        r = self._jsup(v, mock).interpret(self._INPUT, {"A": None, "B": None})
        assert not r.ok
        assert "comm" in r.error or "unknown" in r.error.lower()

    def test_llm_wrong_name_normal_form_rejected(self, v):
        """LLM returns 'normal_form' (quantum_optics hint bug) → unknown operation."""
        mock = _make_llm_mock(
            '{"ok": true, "operation": "normal_form", "args": {"expr": "E1"}, '
            '"confidence": 0.9, "candidates": []}'
        )
        r = self._jsup(v, mock).interpret("正規順序にして E1", {"E1": None})
        # Phase 1 matches 'normal_order' via alias, so LLM is never called.
        # Verify Phase 1 wins:
        assert r.ok
        assert r.operation == "normal_order"
        mock.chat.assert_not_called()

    def test_llm_normal_order_correct_name(self, v):
        """LLM returns 'normal_order' for an input Phase 1 cannot handle."""
        mock = _make_llm_mock(
            '{"ok": true, "operation": "normal_order", "args": {"expr": "E1"}, '
            '"confidence": 0.85, "candidates": []}'
        )
        _unresolvable = "Please put this expression in normal ordered form"
        r = self._jsup(v, mock).interpret(_unresolvable, {"E1": None})
        assert r.ok
        assert r.operation == "normal_order"

    def test_llm_vacuum_expectation_correct_name(self, v):
        mock = _make_llm_mock(
            '{"ok": true, "operation": "vacuum_expectation", "args": {"expr": "E1"}, '
            '"confidence": 0.9, "candidates": []}'
        )
        _unresolvable = "What is the vacuum expectation value of E1"
        r = self._jsup(v, mock).interpret(_unresolvable, {"E1": None})
        assert r.ok
        assert r.operation == "vacuum_expectation"

    def test_llm_heisenberg_equation_correct_name(self, v):
        mock = _make_llm_mock(
            '{"ok": true, "operation": "heisenberg_equation", '
            '"args": {"operator": "a", "hamiltonian": "H"}, '
            '"confidence": 0.9, "candidates": []}'
        )
        _unresolvable = "Compute the equation of motion for operator a under H"
        r = self._jsup(v, mock).interpret(_unresolvable, {"a": None, "H": None})
        assert r.ok
        assert r.operation == "heisenberg_equation"


# ---------------------------------------------------------------------------
# Scope policy in config.json
# ---------------------------------------------------------------------------

class TestScopeConfig:
    def test_config_json_is_valid(self):
        import json
        from pathlib import Path
        cfg_path = Path(__file__).resolve().parent.parent / "config" / "default_settings.json"
        assert cfg_path.exists(), "config/default_settings.json missing"
        data = json.loads(cfg_path.read_text())
        assert "supervisor_scope" in data
        assert "adviser_scope" in data

    def test_supervisor_scope_is_local(self):
        import json
        from pathlib import Path
        cfg_path = Path(__file__).resolve().parent.parent / "config" / "default_settings.json"
        data = json.loads(cfg_path.read_text())
        assert data["supervisor_scope"] == "local"


class TestNestedVecOpPhase07:
    """Phase 0.7 — nested vector-calculus notation recognition."""

    @pytest.mark.parametrize("text,expected_op,expected_expr", [
        # Symbolic prefix with unicode
        ("∇·(∇×A)",       "divergence",       "curl(A)"),
        ("∇×(∇×A)",       "curl",             "curl(A)"),
        ("∇·(∇f)",        "divergence",       "gradient(f)"),
        ("∇×(∇f)",        "curl",             "gradient(f)"),
        # Functional two-paren form
        ("div(curl(A))",   "divergence",       "curl(A)"),
        ("curl(curl(A))",  "curl",             "curl(A)"),
        ("div(grad(f))",   "divergence",       "gradient(f)"),
        # Functional one-paren form
        ("div(curl A)",    "divergence",       "curl(A)"),
        ("curl(grad f)",   "curl",             "gradient(f)"),
        # Space-separated prefix
        ("∇· ∇× A",       "divergence",       "curl(A)"),
        # Japanese genitive
        ("Aの回転の発散",  "divergence",       "curl(A)"),
        ("fの勾配の回転",  "curl",             "gradient(f)"),
        # Case-insensitive
        ("DIV(CURL(A))",   "divergence",       "curl(A)"),
        ("Div(Curl(A))",   "divergence",       "curl(A)"),
    ])
    def test_nested_vec_op_recognized(self, text, expected_op, expected_expr):
        from notebook.runtime.llm_supervisor import _try_nested_vec_op
        result = _try_nested_vec_op(text)
        assert result is not None, f"'{text}' was not recognized"
        assert result.operation == expected_op, f"'{text}': got op={result.operation}"
        assert result.args.get("expr") == expected_expr, \
            f"'{text}': got expr={result.args.get('expr')!r}"
        assert result.source == "rule"
        assert result.requires_confirmation is False


class TestManyFieldOpsGetTheirFieldsRight:
    r"""BP-6 sweep, 2026-08-06 — measured, not assumed.

    The premise going in was "4+ required fields ⇒ the positional mapper breaks".
    Measuring all eleven such operations showed that is **false**: with the
    tokens in field order, ten of eleven map correctly.  What actually breaks is
    narrower and specific, and three cases were real:

    * an explicit ``field=value`` phrase was not read at all — and for
      ``kl_divergence`` the word ``KL`` from the sentence landed in ``p``
      (BP-6's original report, still true);
    * ``convolve`` put the integration VARIABLE in ``f`` and dropped both
      expressions, because ``f``/``g`` were not counted as expression fields;
    * ``residue``'s ``x=1`` was consumed for ``point``, leaving ``var`` empty
      even though that same token names the variable.

    ``taylor_expand`` was the fourth and is fixed separately (a named-group NL
    pattern) because its two numeric fields cannot be told apart by position at
    all.
    """

    def _map(self, tokens, fields, known=("E1", "E2")):
        from notebook.runtime.llm_supervisor import _map_tokens_to_fields
        return _map_tokens_to_fields(tokens, fields, set(known))

    def _tok(self, text):
        from notebook.runtime.llm_supervisor import _tokenize
        return _tokenize(text)

    def test_explicit_field_value_wins(self):
        """A researcher who names the fields has removed the ambiguity."""
        got = self._map(self._tok(
            "p=1/6, var=x, lower=1, upper=6 のシャノンエントロピーを計算して"),
            ["p", "var", "lower", "upper"])
        assert got == {"p": "1/6", "var": "x", "lower": "1", "upper": "6"}

    def test_a_word_from_the_sentence_no_longer_becomes_a_value(self):
        """`KL` used to land in `p` — a value invented out of the prose."""
        got = self._map(self._tok(
            "p=1/6, q=1/3, var=x, lower=1, upper=6 の KL ダイバージェンス"),
            ["p", "q", "var", "lower", "upper"])
        assert got["p"] == "1/6" and got["q"] == "1/3"
        assert "KL" not in got.values()

    def test_convolve_operands_are_expressions(self):
        got = self._map(self._tok("E1 と E2 を t について畳み込んで"),
                        ["f", "g", "var", "dummy"])
        assert got["f"] == "E1" and got["g"] == "E2" and got["var"] == "t"

    def test_a_point_written_as_x_equals_names_the_variable_too(self):
        got = self._map(self._tok("E1 の x=1 における留数を求めて"),
                        ["expr", "var", "point"])
        assert got["expr"] == "E1" and got["var"] == "x" and got["point"] == "1"

    @pytest.mark.parametrize("phrase,fields,expected", [
        ("E1 を k について 1 から 5 まで和を取って",
         ["expr", "var", "lower", "upper"],
         {"expr": "E1", "var": "k", "lower": "1", "upper": "5"}),
        ("E1 を k について 1 から 5 まで積を取って",
         ["expr", "var", "lower", "upper"],
         {"expr": "E1", "var": "k", "lower": "1", "upper": "5"}),
        ("E1 の x→0 の極限を求めて",
         ["expr", "var", "point"],
         {"expr": "E1", "var": "x", "point": "0"}),
        ("exp を a について 3 次まで演算子展開して",
         ["func", "op", "var", "order"],
         {"func": "exp", "op": "a", "order": "3"}),
    ])
    def test_the_ones_that_already_worked_still_do(self, phrase, fields, expected):
        """The sweep's real finding: most of these were fine.  Pinned so the
        next change to the mapper cannot quietly break them."""
        got = self._map(self._tok(phrase), fields)
        for k, v in expected.items():
            assert got.get(k) == v, (phrase, k, got)
