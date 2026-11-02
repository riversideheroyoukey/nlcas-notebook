"""Maxima-specific CAS command builder."""
from __future__ import annotations

from pathlib import Path

from notebook.runtime.command_builder import CommandBuilder

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_SPEC = PROJECT_ROOT / "backends" / "maxima.full.json"


class MaximaCommandBuilder(CommandBuilder):
    """Builds Maxima command strings from CasOperation IR nodes.

    Deterministic: same inputs always produce the same command.
    Uses the template spec in backends/maxima.full.json.
    """

    _backend_label = "Maxima"

    def __init__(self, spec_path: Path | str | None = None) -> None:
        super().__init__(spec_path or DEFAULT_SPEC)
