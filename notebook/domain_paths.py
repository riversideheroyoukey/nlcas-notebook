"""Canonical filesystem location of the research-domain configs.

Domains were originally nested under ``models/`` (an LLM-centric tree: role
endpoint configs, OCR engines, prompts).  They have since grown into the
**first-class entry point to research fields** — each domain carries its label,
category, supervisor/OCR hints, notation profile and worked-example set — so
they now live at the repository top level: ``<repo>/domains/``.

Import :data:`DOMAINS_ROOT` instead of rebuilding ``.../ "domains"`` by hand, so
any future relocation is a one-line change here.  ``notebook`` is the core
package that both ``ui`` and ``scripts`` depend on, so this constant is safe to
import from all of them without creating an import cycle.
"""
from __future__ import annotations

from pathlib import Path

# notebook/domain_paths.py → parent (notebook/) → parent (repository root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOMAINS_ROOT = PROJECT_ROOT / "domains"
