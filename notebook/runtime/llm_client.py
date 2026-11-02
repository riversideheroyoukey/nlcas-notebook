"""Thin HTTP client for chat-completion APIs (OpenAI-compatible, Ollama, …).

Architecture mirrors ui/vlm_client.py:

  _build_*_payload()   — endpoint-type-specific request payload builders
                          (Ollama quirks such as options.num_ctx live here)
  _extract_*_text()    — endpoint-type-specific response text extractors
  _PAYLOAD_BUILDERS    — registry: endpoint_type → payload builder
  _TEXT_EXTRACTORS     — registry: endpoint_type → text extractor
  LLMClient.chat()     — thin dispatcher; knows nothing about wire format
  LLMClient.ping()     — thin health-check; endpoint-agnostic GET /models

To add a new endpoint type: implement _build_<type>_payload() and
_extract_<type>_text(), then register both in the dicts below.
"""
from __future__ import annotations

import json
import os
import re
import urllib.request
import urllib.error
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ChatResponse:
    ok: bool
    text: str = ""
    error: str | None = None


# ---------------------------------------------------------------------------
# Payload builders  (one per endpoint_type)
# ---------------------------------------------------------------------------

def _build_openai_compatible_payload(
    model: str,
    messages: list,
    temperature: float,
    max_tokens: int | None,
    role_cfg: dict,
) -> dict:
    payload: dict = {"model": model, "temperature": temperature, "messages": messages}
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    return payload


def _build_ollama_payload(
    model: str,
    messages: list,
    temperature: float,
    max_tokens: int | None,
    role_cfg: dict,
) -> dict:
    """OpenAI-compatible Ollama endpoint with Ollama-specific extensions.

    Ollama-specific behaviour isolated here:
      - num_ctx must be inside an "options" sub-object (not a top-level field)
      - keep_alive controls how long the model stays loaded after the request
        (-1 = indefinitely; enables KV cache reuse across calls when the
        system-prompt + history prefix is identical)
    """
    payload = _build_openai_compatible_payload(model, messages, temperature, max_tokens, role_cfg)
    options: dict = {}
    num_ctx = role_cfg.get("num_ctx")
    if num_ctx is not None:
        options["num_ctx"] = num_ctx
    if options:
        payload["options"] = options
    keep_alive = role_cfg.get("keep_alive", -1)
    payload["keep_alive"] = keep_alive
    return payload


# ---------------------------------------------------------------------------
# Text extractors  (one per endpoint_type)
# ---------------------------------------------------------------------------

def _extract_openai_compatible_text(data: dict) -> str:
    return data["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

_PAYLOAD_BUILDERS: dict = {
    "openai_compatible": _build_openai_compatible_payload,
    "ollama":            _build_ollama_payload,
}

_TEXT_EXTRACTORS: dict = {
    "openai_compatible": _extract_openai_compatible_text,
    "ollama":            _extract_openai_compatible_text,  # Ollama /v1 uses OpenAI format
}


# ---------------------------------------------------------------------------
# Response cleaner  (endpoint-agnostic)
# ---------------------------------------------------------------------------

def _clean_chat_text(raw: str) -> str:
    """Strip model-output artefacts before callers parse the text.

    Handles:
      - <think>…</think> blocks from reasoning models (qwen3, gpt-oss …)
      - ```json … ``` or ``` … ``` code fences some models wrap around JSON
    """
    text = re.sub(r"<think>[\s\S]*?</think>", "", raw, flags=re.IGNORECASE).strip()
    text = re.sub(r"```(?:json)?\s*|\s*```", "", text).strip()
    return text


# ---------------------------------------------------------------------------
# LLMClient
# ---------------------------------------------------------------------------

class LLMClient:
    """Sends a single-turn chat request to a supported chat-completion endpoint.

    The endpoint_type attribute selects the payload builder and text extractor.
    Currently supported: "openai_compatible" (default), "ollama".
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        timeout: int = 60,
        endpoint_type: str = "openai_compatible",
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._timeout = timeout
        self._endpoint_type = endpoint_type

    # ------------------------------------------------------------------

    def chat(
        self,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        num_ctx: int | None = None,
        prior_messages: list[dict] | None = None,
    ) -> ChatResponse:
        """Send a chat-completion request.

        *prior_messages* is an optional list of ``{"role": …, "content": …}``
        dicts inserted between the system message and the current user message.
        Pass the conversation history here to enable multi-turn context and
        Ollama KV-cache reuse (the identical prefix is not recomputed).
        """
        build = _PAYLOAD_BUILDERS.get(self._endpoint_type, _build_openai_compatible_payload)
        extract = _TEXT_EXTRACTORS.get(self._endpoint_type, _extract_openai_compatible_text)

        messages = [{"role": "system", "content": system}]
        if prior_messages:
            messages.extend(prior_messages)
        messages.append({"role": "user", "content": user})
        role_cfg = {"num_ctx": num_ctx}
        payload = build(self._model, messages, temperature, max_tokens, role_cfg)

        body = json.dumps(payload).encode()
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        req = urllib.request.Request(
            f"{self._base_url}/chat/completions",
            data=body,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                data = json.loads(resp.read().decode())
            raw_text = extract(data)
            return ChatResponse(ok=True, text=_clean_chat_text(raw_text))
        except urllib.error.URLError as e:
            return ChatResponse(ok=False, error=f"Network error: {e.reason}")
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            return ChatResponse(ok=False, error=f"Response parse error: {e}")

    def ping(self, timeout: int = 5) -> bool:
        """Return True if the server responds to GET /models within *timeout* seconds."""
        req = urllib.request.Request(f"{self._base_url}/models", method="GET")
        if self._api_key:
            req.add_header("Authorization", f"Bearer {self._api_key}")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return 200 <= resp.status < 300
        except Exception:
            return False

    # ------------------------------------------------------------------

    @classmethod
    def from_endpoint_json(
        cls,
        path: str | Path,
        role: str = "command_parser",
    ) -> "LLMClient":
        """Load connection parameters from a models/*/endpoint.json file."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        base_url = data["base_url"]
        model = data["model"]
        timeout = data.get("timeout", 60)
        endpoint_type = data.get("endpoint_type", "openai_compatible")

        api_key_env = data.get("api_key_env")
        api_key = os.environ.get(api_key_env, "") if api_key_env else ""

        client = cls(
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout=timeout,
            endpoint_type=endpoint_type,
        )

        roles = data.get("roles", {})
        role_cfg = roles.get(role, {})
        client.default_temperature: float = float(role_cfg.get("temperature", 0.0))
        client.default_max_tokens: int | None = role_cfg.get("max_tokens", None)
        client.default_num_ctx: int | None = role_cfg.get("num_ctx", None)
        client.scope: str = data.get("scope", "local")
        client.endpoint_name: str = data.get("name", Path(path).parent.name)

        return client

    @classmethod
    def find_endpoint(
        cls,
        role: str,
        scope: str | None = None,
        models_root: str | Path | None = None,
        preferred_name: str | None = None,
    ) -> "LLMClient | None":
        """Search models/ for an endpoint.json that supports *role* and *scope*."""
        root = (
            Path(models_root)
            if models_root
            else Path(__file__).resolve().parent.parent.parent / "models"
        )

        all_paths = sorted(root.glob("*/endpoint.json"))

        if preferred_name:
            def _key(p: Path) -> tuple:
                try:
                    name = json.loads(p.read_text(encoding="utf-8")).get("name", "")
                    return (0 if name == preferred_name else 1, str(p))
                except Exception:
                    return (1, str(p))
            all_paths = sorted(all_paths, key=_key)

        for ep_path in all_paths:
            try:
                data = json.loads(ep_path.read_text(encoding="utf-8"))
                ep_scope = data.get("scope", "local")
                if scope is not None and ep_scope != scope:
                    continue
                if role not in data.get("roles", {}):
                    continue
                return cls.from_endpoint_json(ep_path, role=role)
            except Exception:
                continue
        return None

    @classmethod
    def find_all_endpoints(
        cls,
        role: str,
        scope: str | None = None,
        models_root: str | Path | None = None,
    ) -> list["LLMClient"]:
        """Return all endpoints that support *role*, sorted alphabetically by name."""
        root = (
            Path(models_root)
            if models_root
            else Path(__file__).resolve().parent.parent.parent / "models"
        )
        results: list["LLMClient"] = []
        for ep_path in sorted(root.glob("*/endpoint.json")):
            try:
                data = json.loads(ep_path.read_text(encoding="utf-8"))
                ep_scope = data.get("scope", "local")
                if scope is not None and ep_scope != scope:
                    continue
                if role not in data.get("roles", {}):
                    continue
                results.append(cls.from_endpoint_json(ep_path, role=role))
            except Exception:
                continue
        return results
