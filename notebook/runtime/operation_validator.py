from __future__ import annotations
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_SCHEMA_PATH = PROJECT_ROOT / "schemas" / "common_operations.schema.json"

# Which argument fields may hold an expression id is decided in ONE place, read
# by the command builders as well — notebook/runtime/expr_ref_fields.py explains
# why the allowlist that used to live here was the wrong shape, and what the
# invariant actually is (EXPR-REF-FIELD-TYPES).
from notebook.runtime.expr_ref_fields import iter_expr_refs


@dataclass
class ValidationResult:
    valid: bool
    operation: str | None = None
    args: dict = field(default_factory=dict)
    backend: str | None = None
    error: str | None = None
    error_code: str | None = None
    missing_fields: list[str] = field(default_factory=list)
    unknown_exprs: list[str] = field(default_factory=list)
    unknown_fields: list[str] = field(default_factory=list)


class OperationValidator:
    """Validates a common operation against the registry and current session state.

    Checks (in order):
    1. Operation is registered in common_operations.schema.json.
    2. All required fields are present in args.
    3. Expression references (E1, E2, ...) exist in the session's expr table.
    4. The primary backend supports the operation; if not, fallback must be confirmed.
    """

    def __init__(self, schema_path: Path | str | None = None) -> None:
        path = Path(schema_path) if schema_path else DEFAULT_SCHEMA_PATH
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._registry: dict[str, dict] = {
            op["operation"]: op for op in data["operations"]
        }
        self._alias_index: dict[str, str] = _build_alias_index(self._registry)
        self._max_alias_words: int | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(
        self,
        operation: str,
        args: dict,
        primary_backend: str,
        known_expr_ids: set[str] | None = None,
        fallback_confirmed: bool = False,
    ) -> ValidationResult:
        """Validate a single operation candidate.

        Args:
            operation: Common operation name (e.g. "expand").
            args: Operation arguments dict (e.g. {"expr": "E1"}).
            primary_backend: Session's primary backend ("maxima" or "julia_full").
            known_expr_ids: Set of display_ids currently in the session (e.g. {"E1","E2"}).
                            Pass None to skip expression-existence checks.
            fallback_confirmed: True if the user already confirmed a cross-backend fallback.

        Returns:
            ValidationResult with valid=True and resolved backend on success,
            or valid=False with error_code and human-readable error on failure.
        """
        # 1. Operation registered?
        if operation not in self._registry:
            return ValidationResult(
                valid=False,
                error=f"Unknown operation: '{operation}'",
                error_code="UNKNOWN_OPERATION",
            )

        spec = self._registry[operation]

        # 2. Required fields present?
        missing = [f for f in spec["required_fields"] if f not in args]
        if missing:
            return ValidationResult(
                valid=False,
                operation=operation,
                args=args,
                error=f"Missing required fields: {missing}",
                error_code="MISSING_REQUIRED_FIELDS",
                missing_fields=missing,
            )

        # 2.5 No unknown fields?
        allowed = frozenset(spec["required_fields"]) | frozenset(spec.get("optional_fields", []))
        unexpected = sorted(k for k in args if k not in allowed)
        if unexpected:
            return ValidationResult(
                valid=False,
                operation=operation,
                args=args,
                error=f"Unknown fields for '{operation}': {unexpected}. Allowed: {sorted(allowed)}",
                error_code="UNKNOWN_FIELDS",
                unknown_fields=unexpected,
            )

        # 3. Expression references exist?
        if known_expr_ids is not None:
            unknown = _collect_unknown_expr_refs(args, known_expr_ids)
            if unknown:
                return ValidationResult(
                    valid=False,
                    operation=operation,
                    args=args,
                    error=f"Unknown expression references: {unknown}",
                    error_code="UNKNOWN_EXPRESSION",
                    unknown_exprs=unknown,
                )

        # 4. Backend supports this operation?
        supported = spec["supported_backends"]
        if primary_backend not in supported:
            if not fallback_confirmed:
                return ValidationResult(
                    valid=False,
                    operation=operation,
                    args=args,
                    error=(
                        f"Operation '{operation}' is not supported by '{primary_backend}'. "
                        f"Supported: {supported}. Confirm fallback to proceed."
                    ),
                    error_code="BACKEND_NOT_SUPPORTED",
                )
            backend = supported[0]  # first supported backend after confirmed fallback
        else:
            backend = primary_backend

        return ValidationResult(
            valid=True,
            operation=operation,
            args=args,
            backend=backend,
        )

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def resolve_alias(self, name: str) -> str | None:
        """Return the canonical operation name for an alias, or None if unknown.

        Lookup is case-insensitive.  Both English and Japanese aliases are checked.
        The canonical operation name itself is always a valid input.
        """
        return self._alias_index.get(name.lower())

    @property
    def known_operations(self) -> list[str]:
        return list(self._registry.keys())

    @property
    def known_aliases(self) -> dict[str, str]:
        return dict(self._alias_index)

    @property
    def max_alias_words(self) -> int:
        """The longest alias, in words.

        The n-gram alias scan needs a ceiling; taking it from the table means a
        new five-word alias works the day it is added, instead of silently never
        matching (open_items EN-ALIAS-MULTIWORD, which is that failure exactly).
        """
        if self._max_alias_words is None:
            self._max_alias_words = max(
                (len(a.split()) for a in self._alias_index), default=1)
        return self._max_alias_words

    def get_spec(self, operation: str) -> dict | None:
        return self._registry.get(operation)

    def field_names(self, operation: str) -> "frozenset[str]":
        """Every field *operation* accepts (required + optional), or empty."""
        spec = self._registry.get(operation)
        if not spec:
            return frozenset()
        return (frozenset(spec.get("required_fields", []))
                | frozenset(spec.get("optional_fields", [])))

    def all_field_names(self) -> "frozenset[str]":
        """Every field name ANY operation accepts.

        Used to tell a researcher's ``lower=0`` (a field they named) from a
        substitution's ``x=2`` (a value): only names in this set are read as
        field assignments, so ordinary mathematics is never mistaken for one.
        """
        names: set[str] = set()
        for spec in self._registry.values():
            names.update(spec.get("required_fields", []))
            names.update(spec.get("optional_fields", []))
        return frozenset(names)

    def operations_for_alias(self, name: str) -> list[str]:
        """Every operation that LISTS *name* as an alias, in registry order.

        `resolve_alias` is a ``dict[str, str]`` and resolves a collision by
        first-come — so an alias listed by two operations can only ever reach
        one of them, and the other is unreachable by that word.  Measured
        2026-08-16 there are nine such words (``integrate``/``積分``,
        ``laplace``, ``product``, …): the spec claims a spelling the index
        cannot deliver.

        This accessor reads the SPECS rather than the index, so the caller can
        see the whole family.  It exists to be used only where the ambiguity is
        already resolved by something the researcher wrote — never to guess.
        """
        key = name.lower()
        out = [
            canonical for canonical, spec in self._registry.items()
            if canonical.lower() == key
            or any(key == a.lower()
                   for al in spec.get("aliases", {}).values() for a in al)
        ]
        return out

    def operations_for_backend(self, backend: str) -> list[str]:
        return [
            name for name, spec in self._registry.items()
            if backend in spec["supported_backends"]
        ]


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _build_alias_index(registry: dict[str, dict]) -> dict[str, str]:
    """Build a lowercase alias → canonical operation name mapping.

    Canonical names always win over aliases, so they are added last.
    Within the alias list, earlier entries win on collision.
    """
    index: dict[str, str] = {}
    for canonical, spec in registry.items():
        for lang_aliases in spec.get("aliases", {}).values():
            for alias in lang_aliases:
                key = alias.lower()
                if key not in index:
                    index[key] = canonical
        # Canonical name always overrides any alias with the same spelling.
        index[canonical.lower()] = canonical
    return index


def _collect_unknown_expr_refs(args: dict, known: set[str]) -> list[str]:
    """Return expression IDs referenced in args that are not in known."""
    return [eid for _field, eid in iter_expr_refs(args) if eid not in known]
