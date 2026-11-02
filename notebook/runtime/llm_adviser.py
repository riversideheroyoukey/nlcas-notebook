"""LLM Adviser — generates mathematical/physical commentary on CAS results.

Direction: CAS result → natural-language insight → user.
The inverse of LLMSupervisor (user text → operation).

The adviser never generates CAS commands; it returns natural language only.
Triggered explicitly when the user requests commentary.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from notebook.runtime.llm_client import LLMClient

from notebook.runtime.i18n import t


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AdviseContext:
    """Context passed to LLMAdviser.advise()."""

    # Target expression
    expr_id: str
    expr_latex: str
    expr_canonical: str
    expr_cas_repr: str | None = None  # Backend secondary repr when canonical is opaque

    # Derivation info (optional — how the expression was produced)
    operation: str | None = None
    operation_display: str | None = None  # lang-resolved display label (set by caller)
    operation_args: dict = field(default_factory=dict)

    # Source expression (optional — what was operated on)
    source_expr_id: str | None = None
    source_expr_latex: str | None = None

    # User's optional follow-up question
    question: str = ""

    # Session context
    session_title: str = ""
    assumptions: list[str] = field(default_factory=list)  # display strings

    # Recent expressions for context (newest first, capped at 5 by caller)
    recent_exprs: list[dict] = field(default_factory=list)
    # each dict: {"id": "E1", "latex": "...", "canonical": "..."}

    # Compressed session narrative (chronological, most recent ~15 events before target)
    # Supported kinds: "user_input", "registration", "cas_op", "note", "assumption"
    # Each dict has at minimum {"kind": str}; other keys depend on kind.
    session_context: list[dict] = field(default_factory=list)

    # Display language for system prompt and user message generation
    lang: str = "ja"


@dataclass
class AdviseResult:
    ok: bool
    text: str = ""
    error: str | None = None


# ---------------------------------------------------------------------------
# LLMAdviser
# ---------------------------------------------------------------------------

class LLMAdviser:
    """Generates mathematical/physical commentary on a CAS result.

    Args:
        llm_client: Optional LLMClient configured with the 'advisor' role.
                    When None, .available is False and advise() returns an error.
    """

    def __init__(self, llm_client: "LLMClient | None" = None) -> None:
        self._llm = llm_client

    @property
    def available(self) -> bool:
        return self._llm is not None

    def ping(self, timeout: int = 5) -> bool:
        """Return True if the adviser LLM server responds within *timeout* seconds."""
        if self._llm is None:
            return False
        return self._llm.ping(timeout=timeout)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def advise(self, ctx: AdviseContext) -> AdviseResult:
        """Return natural-language commentary on the CAS result in *ctx*."""
        if self._llm is None:
            return AdviseResult(ok=False, error=t("adviser.not_configured", ctx.lang))

        system = self._build_system_prompt(ctx)
        user_msg = self._build_user_message(ctx)

        temperature = getattr(self._llm, "default_temperature", 0.3)
        max_tokens = getattr(self._llm, "default_max_tokens", None) or 512

        resp = self._llm.chat(
            system=system,
            user=user_msg,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        if not resp.ok:
            return AdviseResult(ok=False, error=f"LLM error: {resp.error}")

        text = resp.text.strip()
        if not text:
            return AdviseResult(ok=False, error=t("adviser.no_response", ctx.lang))

        return AdviseResult(ok=True, text=text)

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_system_prompt(self, ctx: AdviseContext) -> str:
        lang = ctx.lang
        lines = [
            t("adviser.system_role", lang),
            t("adviser.system_task", lang),
            "",
            t("adviser.system_constraints_header", lang),
            t("adviser.system_no_code", lang),
            t("adviser.system_explain", lang),
            t("adviser.system_math_format", lang),
            t("adviser.system_tex_compat", lang),
            t("adviser.system_uncertainty", lang),
            t("adviser.system_language", lang),
        ]

        if ctx.session_title:
            lines += ["", t("adviser.context_session", lang, title=ctx.session_title)]

        if ctx.assumptions:
            lines += ["", t("adviser.context_assumptions", lang)]
            for a in ctx.assumptions:
                lines.append(f"  - {a}")

        if ctx.session_context:
            lines += ["", t("adviser.context_history", lang)]
            for item in ctx.session_context:
                k = item.get("kind")
                if k == "user_input":
                    lines.append("  " + t("adviser.history_user", lang, text=item.get("text", "")))
                elif k == "registration":
                    lines.append("  " + t("adviser.history_expr", lang,
                                          expr_id=item["expr_id"], latex=item["latex"]))
                elif k == "cas_op":
                    src = f" ({item['source_id']} →)" if item.get("source_id") else ""
                    op_label = item.get("operation_label") or item.get("operation", "")
                    julia_note = f" (Julia: `{item['cas_repr']}`)" if item.get("cas_repr") else ""
                    lines.append(f"  [{item['expr_id']}]{src} {op_label}: ${item['latex']}${julia_note}")
                elif k == "note":
                    lines.append("  " + t("adviser.history_memo", lang, text=item.get("text", "")))
                elif k == "assumption":
                    lines.append("  " + t("adviser.history_assumption", lang, text=item.get("text", "")))

        elif ctx.recent_exprs:
            lines += ["", t("adviser.context_latest", lang)]
            for e in ctx.recent_exprs[:5]:
                lines.append(f"  {e['id']}: ${e['latex']}$")

        return "\n".join(lines)

    def _build_user_message(self, ctx: AdviseContext) -> str:
        lang = ctx.lang
        lines: list[str] = []

        if ctx.operation_display and ctx.source_expr_latex:
            lines.append(t("adviser.operation_applied", lang,
                           operation=ctx.operation_display,
                           latex=ctx.source_expr_latex))
        elif ctx.operation_display:
            lines.append(t("adviser.operation_result", lang, operation=ctx.operation_display))
        else:
            lines.append(t("adviser.calculation_result", lang))

        lines.append(f"$${ctx.expr_latex}$$")

        if ctx.expr_cas_repr:
            lines.append(f"(CAS: `{ctx.expr_cas_repr}`)")

        if ctx.question:
            lines += ["", t("adviser.question", lang, question=ctx.question)]
        else:
            lines += ["", t("adviser.explain_request", lang)]

        return "\n".join(lines)
