"""Loads and resolves domain-cluster notation profiles from notation_profiles.json.

The profiles define which notation tokens have unambiguous meanings in each
domain cluster, enabling the LLM Supervisor to skip confirmation dialogs and
the SemanticResolver to resolve ambiguous notation without LLM fallback.

Inheritance chain: second_quantization_profile → quantum_standard →
                   classical_physics → calculus_base
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

from notebook.domain_paths import DOMAINS_ROOT


_DEFAULT_PROFILES_PATH = DOMAINS_ROOT / "notation_profiles.json"


class NotationRule(TypedDict, total=False):
    meaning: str
    exclusive: bool
    condition: str
    note: str


class NotationProfileLoader:
    """Resolves notation rules for a given set of active session domains.

    Usage::

        loader = NotationProfileLoader()
        exclusive, meaning = loader.is_exclusive("prime", ["condensed_matter"])
        # → (True, "creation_operator")

        hint = loader.notation_hint_text(["condensed_matter"])
        # → multi-line string injected into LLM system prompt
    """

    def __init__(self, profiles_path: Path | str | None = None) -> None:
        path = Path(profiles_path) if profiles_path else _DEFAULT_PROFILES_PATH
        data = json.loads(path.read_text(encoding="utf-8"))
        # Preserve insertion order so that more-specific profiles (listed later)
        # override less-specific ones when multiple domains match.
        self._profiles: dict[str, dict] = {
            p["id"]: p for p in data.get("profiles", [])
        }

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def profile_for_domains(self, domains: list[str]) -> tuple[str | None, dict[str, NotationRule]]:
        """Return (profile_id, merged_rules) for the most specific matching profile.

        When multiple profiles match (e.g. both ``condensed_matter`` and
        ``analytical_mechanics`` are active), the most specific one (deepest
        inheritance chain) takes precedence.  Its rules override ancestors.
        """
        best_id: str | None = None
        best_depth: int = -1
        for pid in self._profiles:
            profile = self._profiles[pid]
            if any(d in profile.get("domains", []) for d in domains):
                depth = self._inheritance_depth(pid)
                if depth > best_depth:
                    best_id = pid
                    best_depth = depth

        if best_id is None:
            return None, {}
        return best_id, self._resolve_rules(best_id)

    def is_exclusive(self, token: str, domains: list[str]) -> tuple[bool, str | None]:
        """Return ``(True, meaning)`` when the token has an exclusive (unambiguous)
        meaning in the given domains.  Returns ``(False, None)`` otherwise.

        An ``exclusive`` rule means the token has only one possible interpretation
        in this domain cluster — the confirmation dialog should be suppressed.
        """
        _, rules = self.profile_for_domains(domains)
        rule = rules.get(token)
        if rule and rule.get("exclusive"):
            return True, rule.get("meaning")
        return False, None

    def get_rule(self, token: str, domains: list[str]) -> NotationRule | None:
        """Return the full rule dict for *token* in the given domains, or None."""
        _, rules = self.profile_for_domains(domains)
        return rules.get(token)  # type: ignore[return-value]

    def notation_hint_text(self, domains: list[str]) -> str:
        """Return a compact notation hint for injection into the LLM system prompt.

        Exclusive rules are marked as such so the LLM knows not to ask for
        confirmation.  Non-exclusive rules are listed as context.
        """
        profile_id, rules = self.profile_for_domains(domains)
        if not rules:
            return ""
        lines = [f"Notation profile ({profile_id}):"]
        for token, rule in rules.items():
            meaning = rule.get("meaning", "?")
            excl_tag = " [exclusive]" if rule.get("exclusive") else ""
            cond = rule.get("condition", "")
            cond_tag = f" (when: {cond})" if cond else ""
            lines.append(f"  {token}: {meaning}{excl_tag}{cond_tag}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_rules(self, profile_id: str) -> dict[str, NotationRule]:
        """Recursively merge rules from the inheritance chain (child overrides parent)."""
        profile = self._profiles.get(profile_id, {})
        rules: dict[str, NotationRule] = {}
        parent_id = profile.get("extends")
        if parent_id:
            rules.update(self._resolve_rules(parent_id))
        rules.update(profile.get("rules", {}))
        return rules

    def _inheritance_depth(self, profile_id: str) -> int:
        """Return the depth of the inheritance chain (0 = no parent)."""
        profile = self._profiles.get(profile_id, {})
        parent_id = profile.get("extends")
        if parent_id:
            return 1 + self._inheritance_depth(parent_id)
        return 0
