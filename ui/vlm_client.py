# ui/vlm_client.py

import httpx
import json
import re
from pathlib import Path

from notebook.domain_paths import DOMAINS_ROOT as _DOMAINS_ROOT

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Config / prompt loader
# ---------------------------------------------------------------------------

def load_component(category, model_id):
    base_path = PROJECT_ROOT / "models" / category / model_id
    config_path = base_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found at: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    prompt_file = config.get('paths', {}).get('prompt_file', 'system_prompt.txt')
    prompt_path = base_path / prompt_file
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt = "".join([l for l in f.readlines() if not l.strip().startswith("#")]).strip()
    return config, prompt


# ---------------------------------------------------------------------------
# Transport adapters  (one per endpoint_type)
# Add a new async function here when supporting a new endpoint type,
# then register it in _TRANSPORT below.
# ---------------------------------------------------------------------------

async def _call_ollama(api_config: dict, params: dict, messages: list) -> str:
    """Ollama /api/chat transport. Returns raw content string.

    Ollama-specific behaviour isolated here:
      - think param must be at the top level of the payload (not inside options)
      - /no_think prefix suppresses CoT for qwen3-family models
    """
    base_url = api_config['base_url'].replace("localhost", "127.0.0.1").rstrip('/')
    url = f"{base_url}/api/chat"

    options = {k: v for k, v in params.items() if k != "think"}
    payload = {
        "model": api_config["model_name"],
        "messages": messages,
        "stream": False,
        "options": options,
    }
    if "think" in params:
        payload["think"] = params["think"]
        if not params["think"]:
            last_user = next((m for m in reversed(messages) if m["role"] == "user"), None)
            if last_user and not last_user["content"].startswith("/no_think"):
                last_user["content"] = "/no_think\n" + last_user["content"]

    async with httpx.AsyncClient(timeout=httpx.Timeout(180.0)) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        return response.json().get("message", {}).get("content", "").strip()


# Register adapters here; key must match endpoint_type in config.json
_TRANSPORT: dict = {
    "ollama": _call_ollama,
}


# ---------------------------------------------------------------------------
# Response parser  (endpoint-agnostic)
# ---------------------------------------------------------------------------

def _parse_latex_response(raw_text: str) -> dict:
    """Extract recognized_latex from model output regardless of endpoint type.

    Handles:
      - <think>…</think> blocks from reasoning models (qwen3, InternVL3.5 …)
      - Clean JSON {"recognized_latex": "…"}
      - JSON with LaTeX backslash escapes that break json.loads (\\alpha etc.)
      - ```latex … ``` code fences
      - Raw text fallback
    """
    if not raw_text:
        return {"recognized_latex": "Error: Empty response"}

    # Strip CoT blocks before any parsing
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", raw_text, flags=re.IGNORECASE).strip()

    try:
        data = json.loads(cleaned)
        if isinstance(data, dict) and "recognized_latex" in data:
            return {"recognized_latex": data["recognized_latex"]}
    except Exception:
        pass

    # LaTeX backslashes (\alpha, \beta …) are invalid JSON escape sequences.
    # Extract the value with a regex when json.loads fails.
    m = re.search(r'"recognized_latex"\s*:\s*"((?:[^"\\]|\\.)*)"', cleaned, re.DOTALL)
    if m:
        return {"recognized_latex": m.group(1)}

    bq = "\x60\x60\x60"
    match = re.search(rf"{bq}(?:latex|tex)?\s*([\s\S]*?)\s*{bq}", cleaned, re.IGNORECASE)
    if match:
        return {"recognized_latex": match.group(1).strip()}

    return {"recognized_latex": cleaned}


# ---------------------------------------------------------------------------
# Common call interface
# ---------------------------------------------------------------------------

async def call_model(api_config: dict, params: dict, system_prompt: str,
                     user_content: str, images: list | None = None) -> dict:
    """Route to the correct transport adapter, then parse the response.

    This is the only function the pipeline should call; it knows nothing
    about endpoint-specific wire formats.
    """
    endpoint_type = api_config.get("endpoint_type", "ollama")
    transport = _TRANSPORT.get(endpoint_type)
    if transport is None:
        return {"recognized_latex": f"Error: Unsupported endpoint_type '{endpoint_type}'"}

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_content, "images": images or []},
    ]
    try:
        raw_text = await transport(api_config, params, messages)
        return _parse_latex_response(raw_text)
    except Exception as e:
        return {"recognized_latex": f"Communication Error: {str(e)}"}


# ---------------------------------------------------------------------------
# Domain hints
# ---------------------------------------------------------------------------

def _load_ocr_domain_hints(domains: list[str]) -> str:
    import logging
    hints = []
    for domain in domains:
        hint_path = _DOMAINS_ROOT / domain / "ocr_hint.txt"
        if hint_path.exists():
            hints.append(hint_path.read_text(encoding="utf-8").strip())
        else:
            logging.warning("OCR domain hint not found: %s", hint_path)
    return "\n\n".join(hints)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

async def run_pipeline(engine_id, refiner_id, image_b64, domains: list[str] | None = None):
    eng_conf, eng_prompt = load_component("engines", engine_id)
    if domains:
        ocr_hints = _load_ocr_domain_hints(domains)
        if ocr_hints:
            eng_prompt = eng_prompt + "\n\n" + ocr_hints

    draft_res = await call_model(
        eng_conf["api_config"], eng_conf["parameters"], eng_prompt,
        "Extract LaTeX.", [image_b64]
    )
    draft_latex = draft_res.get("recognized_latex", "")

    if not refiner_id or refiner_id == "none":
        draft_res["_pipeline"] = {"engine": engine_id, "refiner": "none", "draft_latex": draft_latex}
        return draft_res

    ref_conf, ref_prompt = load_component("refiners", refiner_id)
    if ref_conf.get("vision_input", False):
        user_msg = (
            f"The OCR engine produced the following LaTeX from the handwritten image. "
            f"Review the original image and correct any recognition errors:\n\n{draft_latex}"
        )
        ref_images = [image_b64]
    else:
        user_msg = f"Standardize: {draft_latex}"
        ref_images = None

    final_res = await call_model(
        ref_conf["api_config"], ref_conf["parameters"], ref_prompt,
        user_msg, ref_images
    )
    final_res["_pipeline"] = {"engine": engine_id, "refiner": refiner_id, "draft_latex": draft_latex}
    return final_res
