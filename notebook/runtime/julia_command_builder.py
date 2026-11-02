"""Julia-specific CAS command builder."""
from __future__ import annotations

from pathlib import Path

from notebook.runtime.command_builder import CommandBuilder

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_SPEC = PROJECT_ROOT / "backends" / "julia.full.json"


class JuliaCommandBuilder(CommandBuilder):
    """Builds Julia command strings from CasOperation IR nodes.

    Uses the template spec in backends/julia.full.json.
    Scalar argument resolution reuses the shared IR-to-string logic in
    CommandBuilder, which is backend-agnostic (Symbol, Number, FunctionCall, etc.).
    """

    _backend_label = "Julia"

    def __init__(self, spec_path: Path | str | None = None) -> None:
        super().__init__(spec_path or DEFAULT_SPEC)
