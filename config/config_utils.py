# config/config_utils.py
import json
from pathlib import Path
from typing import Any

CONFIG_DIR = Path(__file__).resolve().parent
DEFAULT_SETTINGS_PATH = CONFIG_DIR / "default_settings.json"

_settings_cache = None

def load_settings(force_reload: bool = False) -> dict:
    global _settings_cache
    if _settings_cache is None or force_reload:
        if DEFAULT_SETTINGS_PATH.exists():
            with open(DEFAULT_SETTINGS_PATH, "r", encoding="utf-8") as f:
                _settings_cache = json.load(f)
        else:
            _settings_cache = {}
    return _settings_cache

def get_setting(key: str, default: Any = None) -> Any:
    return load_settings().get(key, default)

def get_default_vlm_target() -> str:
    return get_setting("vlm_target", "ollama_qwen3_vl")

def get_default_refiner_target() -> str:
    return get_setting("refiner_target", "none")

def get_default_language() -> str:
    return get_setting("language", "ja")

def get_default_supervisor_target() -> str:
    return get_setting("supervisor_target", "")

def get_default_adviser_target() -> str:
    return get_setting("adviser_target", "")

def get_default_coder_target() -> str:
    """Preferred endpoint for the numeric-layer coding LLM (role='coder')."""
    return get_setting("coder_target", "")

def get_advise_narrative_max_items() -> int:
    """Item cap for /advise's session-narrative context (ui/routes/advisory.py)."""
    return get_setting("advise_narrative_max_items", 15)

def get_default_numeric_timeout_s() -> float:
    """Default sandbox execution timeout (seconds) for /numeric/run, when
    the caller doesn't pass an explicit override."""
    return get_setting("numeric_timeout_s", 60.0)

def get_plan_narrative_max_items() -> int:
    """Item cap for the C-Ext-5 plan-Adviser's session-narrative context
    (ui/interpret_plan.py::_narrative_context, §11.1 context-supply increment #2)."""
    return get_setting("plan_narrative_max_items", 5)

def get_supervisor_scope() -> str:
    return get_setting("supervisor_scope", "local")

def get_adviser_scope() -> str:
    return get_setting("adviser_scope", "local")

def get_local_llm_server_config() -> dict:
    """Config for the local OpenAI-compatible inference server the app auto-starts.

    Provider-agnostic: switching Ollama -> vLLM / LocalAI / etc. is a config
    change only (command/port/env here + the matching models/*/endpoint.json
    base_url).  Empty or auto_start=false means the app starts nothing and the
    user manages the server themselves.
    """
    return get_setting("local_llm_server", {})
