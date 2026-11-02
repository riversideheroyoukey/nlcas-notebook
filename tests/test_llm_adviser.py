"""Tests for LLMAdviser.

Covers: availability, ping, prompt construction, result dataclasses,
mock LLM integration (no real network calls).
"""
import pytest
from notebook.runtime.llm_adviser import LLMAdviser, AdviseContext, AdviseResult
from notebook.runtime.llm_client import LLMClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ctx(**kwargs) -> AdviseContext:
    defaults = {
        "expr_id":       "E2",
        "expr_latex":    r"\omega \cos(\omega t)",
        "expr_canonical": "omega*cos(omega*t)",
    }
    defaults.update(kwargs)
    return AdviseContext(**defaults)


class _MockResponse:
    def __init__(self, ok: bool, text: str = "", error: str | None = None):
        self.ok    = ok
        self.text  = text
        self.error = error


class _MockLLMClient:
    def __init__(self, response: _MockResponse, ping_result: bool = True):
        self._response           = response
        self._ping_result        = ping_result
        self.default_temperature = 0.3
        self.default_max_tokens  = 256

    def chat(self, system, user, temperature, max_tokens):
        return self._response

    def ping(self, timeout: int = 5) -> bool:
        return self._ping_result


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------

class TestAvailability:
    def test_no_llm_not_available(self):
        assert LLMAdviser(llm_client=None).available is False

    def test_with_llm_available(self):
        mock = _MockLLMClient(_MockResponse(ok=True, text="解説"))
        assert LLMAdviser(llm_client=mock).available is True


# ---------------------------------------------------------------------------
# advise() without LLM
# ---------------------------------------------------------------------------

class TestAdviseNoLLM:
    def test_returns_error(self):
        result = LLMAdviser(llm_client=None).advise(_ctx())
        assert result.ok is False
        assert result.error is not None

    def test_text_is_empty(self):
        result = LLMAdviser(llm_client=None).advise(_ctx())
        assert result.text == ""


# ---------------------------------------------------------------------------
# System prompt construction
# ---------------------------------------------------------------------------

class TestSystemPrompt:
    def _prompt(self, **kwargs) -> str:
        return LLMAdviser(llm_client=None)._build_system_prompt(_ctx(**kwargs))

    def test_contains_no_cas_constraint(self):
        assert "CASコマンド" in self._prompt()

    def test_contains_japanese_instruction(self):
        assert "日本語" in self._prompt()

    def test_session_title_included(self):
        p = self._prompt(session_title="ラグランジアン力学")
        assert "ラグランジアン力学" in p

    def test_assumptions_included(self):
        p = self._prompt(assumptions=["ω > 0", "t > 0"])
        assert "ω > 0" in p
        assert "t > 0" in p

    def test_recent_exprs_included(self):
        exprs = [{"id": "E1", "latex": r"\sin(\omega t)", "canonical": "sin(omega*t)"}]
        p = self._prompt(recent_exprs=exprs)
        assert "E1" in p

    def test_recent_exprs_capped_at_5(self):
        exprs = [{"id": f"E{i}", "latex": f"x^{i}", "canonical": f"x^{i}"} for i in range(10)]
        p = self._prompt(recent_exprs=exprs)
        assert "E0" in p
        assert "E4" in p
        assert "E5" not in p

    def test_no_session_title_not_in_prompt(self):
        p = self._prompt(session_title="")
        assert "セッション:" not in p

    def test_no_assumptions_section_omitted(self):
        p = self._prompt(assumptions=[])
        assert "前提条件:" not in p


# ---------------------------------------------------------------------------
# User message construction
# ---------------------------------------------------------------------------

class TestUserMessage:
    def _msg(self, **kwargs) -> str:
        return LLMAdviser(llm_client=None)._build_user_message(_ctx(**kwargs))

    def test_contains_expr_latex(self):
        assert r"\omega \cos(\omega t)" in self._msg()

    def test_with_operation_and_source(self):
        msg = self._msg(
            operation_display="微分する",
            source_expr_latex=r"\sin(\omega t)",
        )
        assert "微分する" in msg
        assert r"\sin(\omega t)" in msg

    def test_operation_only_no_source(self):
        msg = self._msg(operation_display="有理簡約")
        assert "有理簡約" in msg

    def test_no_operation_shows_fallback(self):
        msg = self._msg()
        assert "計算結果:" in msg

    def test_question_included(self):
        msg = self._msg(question="これは何を意味するの？")
        assert "これは何を意味するの？" in msg

    def test_no_question_shows_default(self):
        msg = self._msg()
        assert "解説してください" in msg


# ---------------------------------------------------------------------------
# AdviseResult dataclass
# ---------------------------------------------------------------------------

class TestAdviseResult:
    def test_ok_result(self):
        r = AdviseResult(ok=True, text="これは微分です。")
        assert r.ok is True
        assert r.text == "これは微分です。"
        assert r.error is None

    def test_error_result(self):
        r = AdviseResult(ok=False, error="LLM error: timeout")
        assert r.ok is False
        assert r.error == "LLM error: timeout"
        assert r.text == ""


# ---------------------------------------------------------------------------
# advise() with mock LLM
# ---------------------------------------------------------------------------

class TestAdviseWithMockLLM:
    def test_success(self):
        mock = _MockLLMClient(_MockResponse(ok=True, text="これは微分です。"))
        result = LLMAdviser(llm_client=mock).advise(_ctx())
        assert result.ok is True
        assert result.text == "これは微分です。"

    def test_llm_network_error(self):
        mock = _MockLLMClient(_MockResponse(ok=False, error="Network error: timeout"))
        result = LLMAdviser(llm_client=mock).advise(_ctx())
        assert result.ok is False
        assert "LLM error" in result.error

    def test_empty_response_is_error(self):
        mock = _MockLLMClient(_MockResponse(ok=True, text="   "))
        result = LLMAdviser(llm_client=mock).advise(_ctx())
        assert result.ok is False
        assert result.error is not None

    def test_uses_llm_temperature_attribute(self):
        """LLMAdviser passes the client's default_temperature to chat()."""
        calls = []

        class CaptureLLMClient:
            default_temperature = 0.42
            default_max_tokens  = 100

            def chat(self, system, user, temperature, max_tokens):
                calls.append({"temperature": temperature, "max_tokens": max_tokens})
                return _MockResponse(ok=True, text="ok")

        LLMAdviser(llm_client=CaptureLLMClient()).advise(_ctx())
        assert calls[0]["temperature"] == pytest.approx(0.42)
        assert calls[0]["max_tokens"] == 100

    def test_default_max_tokens_fallback(self):
        """When default_max_tokens is None, falls back to 512."""
        calls = []

        class NoMaxTokensClient:
            default_temperature  = 0.3
            default_max_tokens   = None

            def chat(self, system, user, temperature, max_tokens):
                calls.append(max_tokens)
                return _MockResponse(ok=True, text="ok")

        LLMAdviser(llm_client=NoMaxTokensClient()).advise(_ctx())
        assert calls[0] == 512


# ---------------------------------------------------------------------------
# LLMAdviser.ping()
# ---------------------------------------------------------------------------

class TestAdviserPing:
    def test_ping_no_llm_returns_false(self):
        assert LLMAdviser(llm_client=None).ping() is False

    def test_ping_delegates_available(self):
        mock = _MockLLMClient(_MockResponse(ok=True), ping_result=True)
        assert LLMAdviser(llm_client=mock).ping() is True

    def test_ping_delegates_unavailable(self):
        mock = _MockLLMClient(_MockResponse(ok=True), ping_result=False)
        assert LLMAdviser(llm_client=mock).ping() is False


# ---------------------------------------------------------------------------
# LLMClient.ping()
# ---------------------------------------------------------------------------

class TestLLMClientPing:
    def test_ping_returns_true_on_200(self):
        from unittest.mock import patch, MagicMock
        client = LLMClient("http://fake:1234/v1", "model")
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            assert client.ping() is True

    def test_ping_returns_false_on_url_error(self):
        from unittest.mock import patch
        import urllib.error
        client = LLMClient("http://fake:1234/v1", "model")
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("refused")):
            assert client.ping() is False

    def test_ping_returns_false_on_any_exception(self):
        from unittest.mock import patch
        client = LLMClient("http://fake:1234/v1", "model")
        with patch("urllib.request.urlopen", side_effect=OSError("timeout")):
            assert client.ping() is False

    def test_ping_uses_short_timeout(self):
        """ping() must not use the instance's inference timeout."""
        from unittest.mock import patch, MagicMock
        client = LLMClient("http://fake:1234/v1", "model", timeout=180)
        captured = {}
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        def fake_urlopen(req, timeout):
            captured["timeout"] = timeout
            return mock_resp

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            client.ping(timeout=5)
        assert captured["timeout"] == 5
