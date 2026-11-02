# profile_loader.py
# Version: v0.1.0
# Last Updated: 2026-04-11
#
# Description:
#   Load model/profile-based endpoint settings and prompts from models/.

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


@dataclass
class ProfileRoleConfig:
    role_name: str
    temperature: float = 0.0


@dataclass
class ProfileConfig:
    name: str
    base_url: str
    model: str
    api_key_env: str | None
    timeout: int
    roles: dict[str, ProfileRoleConfig]
    profile_dir: Path

    @property
    def api_key(self) -> str | None:
        if self.api_key_env is None:
            return None

        value = os.getenv(self.api_key_env)
        if not value:
            raise RuntimeError(
                f"Missing API key: environment variable '{self.api_key_env}' is not set."
            )
        return value


def load_profile(profile_name: str) -> ProfileConfig:
    profile_dir = MODELS_DIR / profile_name
    endpoint_path = profile_dir / "endpoint.json"

    if not endpoint_path.exists():
        raise FileNotFoundError(f"Missing endpoint config: {endpoint_path}")

    data = json.loads(endpoint_path.read_text(encoding="utf-8"))

    roles_raw = data.get("roles", {})
    roles = {
        role_name: ProfileRoleConfig(
            role_name=role_name,
            temperature=role_data.get("temperature", 0.0),
        )
        for role_name, role_data in roles_raw.items()
    }

    return ProfileConfig(
        name=data["name"],
        base_url=data["base_url"],
        model=data["model"],
        api_key_env=data.get("api_key_env"),
        timeout=data.get("timeout", 120),
        roles=roles,
        profile_dir=profile_dir,
    )


def load_prompt(profile_name: str, role_name: str) -> str:
    profile = load_profile(profile_name)

    # ① profile固有
    prompt_path = profile.profile_dir / "prompts" / f"{role_name}_system.txt"
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8")

    # ② default fallback
    default_path = MODELS_DIR / "default" / "prompts" / f"{role_name}_system.txt"
    if default_path.exists():
        return default_path.read_text(encoding="utf-8")

    raise FileNotFoundError(
        f"Missing prompt: role='{role_name}' not found in profile '{profile_name}' nor default."
    )