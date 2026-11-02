"""Dialogue-driven NumericSpec proposal (slice 2 論点B, docs/25 §3).

Mirrors ``plan_adviser.py``'s proven skeleton — a ``chat(system, user) -> str``
callable, strict parsing (a broken reply never becomes a spec), and a
gatekeeper before anything is shown to the researcher — with the contract
swapped from a Plan to a ``NumericSpec`` and the gatekeeper swapped from
``validate_plan`` to ``validate_numeric_spec`` (both already implemented,
docs/24). The module is LLM-agnostic: tests inject a fake ``chat`` so the
deterministic skeleton (prompt shape, parsing, gatekeeper, refine-turn wiring)
is fully covered without any model.

B-γ "propose-then-refine" (docs/25 §1): ``propose_spec`` is called once per
turn. The first call has no prior spec; a refine call passes the previous
spec + the researcher's natural-language correction, and the LLM returns a
revised spec. Unlike ``plan_adviser`` (which returns MULTIPLE candidates),
this returns a SINGLE spec — decision §7-1: propose one, only add multiple
candidates if refinement stalls in practice.

A spec with gatekeeper issues is NOT discarded (unlike plan_adviser's silent
per-candidate rejection): it is returned alongside the issues so the
researcher sees exactly what's unresolved and can refine — that back-and-forth
IS the point of B-γ.
"""
from __future__ import annotations

import json

from notebook.models.numeric import NumericSpec
from notebook.numeric.intent_registry import known_kinds
from notebook.numeric.spec_validate import validate_numeric_spec

_SYSTEM_PROMPT = """You are the numeric-computation spec adviser of a theoretical-research notebook.
Given a goal, the target expression, its free symbols, and the notebook's
context, propose ONE numeric-computation spec as a JSON object. Rules:
- Ground the spec in the notebook context given (definitions, assumptions,
  prior results) — do not invent parameters the expression doesn't have.
- Every input "name" MUST be one of the expression's free symbols.
- Output ONLY the JSON object, no prose. Schema:
  {"intent": {"kind": str, "goal": str},
   "inputs": [{"name": str, "role": "sweep"|"fixed"|"free"|"boundary",
               "spec": {...}}],
   "outputs": [{"name": str, "shape": str, "over": [str], "role": str}],
   "method": {"name": str, "params": {...}},
   "presentation": {"kind": str, "options": {...}}}
- role="sweep" spec is {"min": number, "max": number, "points": int}.
- role="fixed" spec is {"value": number}. role="free" spec is
  {"initial": number}. role="boundary" spec is {"at": number, "value": number}.
- Known intent kinds (prefer these when applicable; a novel kind is fine too):
  __KNOWN_KINDS__
"""


def build_spec_prompt(goal_text: str, target_summary: str,
                      free_symbols: "set[str] | None", narrative: "list[dict] | None" = None,
                      prior_spec: "NumericSpec | None" = None,
                      refinement: str = "", expression_catalog: "list[dict] | None" = None) -> str:
    """Compose the user turn.  A refine turn (``prior_spec`` given) includes
    the previous proposal and the researcher's correction; the LLM revises
    rather than starting over.  ``free_symbols=None`` (the expression isn't
    whitelist-lowerable) omits that line — the adviser then works from the
    expression text alone, and the gatekeeper's membership check is skipped
    downstream (mirrors ``validate_numeric_spec``'s own None policy)."""
    parts = [
        f"Goal: {goal_text}",
        f"Target expression: {target_summary}",
    ]
    if expression_catalog is not None:
        parts.append("Available expressions (reference these IDs; do not invent formulas or coordinates): "
                     + json.dumps(expression_catalog, ensure_ascii=False))
    if free_symbols is not None:
        parts.append(f"Free symbols: {', '.join(sorted(free_symbols))}")
    if narrative:
        parts.append(f"Notebook context (chronological): {json.dumps(narrative, ensure_ascii=False)}")
    if prior_spec is not None:
        parts.append(f"Previous proposal: {prior_spec.model_dump_json()}")
        parts.append(f"Researcher's correction: {refinement}")
        parts.append("Revise the previous proposal to address the correction.")
    else:
        parts.append("Propose the spec now.")
    parts.append("Respond with the JSON object only.")
    return "\n\n".join(parts)


def parse_spec(raw_text: str) -> "NumericSpec | None":
    """Strictly parse the adviser's reply into a NumericSpec.

    Anything malformed (no JSON object, invalid JSON, fields that don't fit
    the model) yields None — a broken reply must never become a spec the
    researcher is asked to approve.
    """
    if not raw_text:
        return None
    # Decode the object rather than greedily matching through trailing prose.
    # Require exactly one candidate spec: multiple alternatives need refinement.
    decoder = json.JSONDecoder()
    candidates = []
    offset = 0
    while offset < len(raw_text):
        start = raw_text.find('{', offset)
        if start < 0:
            break
        try:
            value, size = decoder.raw_decode(raw_text[start:])
        except ValueError:
            return None
        if isinstance(value, dict) and ('intent' in value or 'outputs' in value):
            candidates.append(value)
        offset = start + size
    if len(candidates) != 1:
        return None
    data = candidates[0]
    if not isinstance(data, dict):
        return None
    try:
        return NumericSpec.model_validate(data)
    except Exception:
        return None


def propose_spec(chat, goal_text: str, target_summary: str,
                 free_symbols: "set[str] | None", narrative: "list[dict] | None" = None,
                 prior_spec: "NumericSpec | None" = None, refinement: str = "",
                 expression_catalog: "list[dict] | None" = None,
                 ) -> "tuple[NumericSpec | None, list[dict]]":
    """Consult the adviser for one B-γ turn.

    Returns (spec, issues). ``spec`` is None only when the reply was
    unparseable (never shown to the researcher). A parseable spec is ALWAYS
    returned even with gatekeeper ``issues`` — the researcher sees what's
    unresolved and refines in the next turn (docs/25 §3, unlike
    plan_adviser's silent per-candidate rejection).  ``free_symbols=None``
    (expression not whitelist-lowerable) skips the membership check, same
    policy as ``validate_numeric_spec`` itself.
    """
    prompt = build_spec_prompt(goal_text, target_summary, free_symbols,
                               narrative, prior_spec, refinement, expression_catalog)
    # str.replace, not .format(): the prompt's JSON schema example is full of
    # literal {..} braces that .format() would misparse as placeholders.
    system = _SYSTEM_PROMPT.replace("__KNOWN_KINDS__", ", ".join(known_kinds()))
    from notebook.numeric.plot_prompt import CARTESIAN_PROMPT
    system += CARTESIAN_PROMPT
    raw = chat(system, prompt)
    spec = parse_spec(raw or "")
    if spec is None:
        return None, []
    # Layer sources can contain parameters absent from the selected expression;
    # fork-time validation resolves the actual referenced expression universe.
    membership = None if spec.presentation.kind == "cartesian" else free_symbols
    issues = validate_numeric_spec(spec, free_symbols=membership)
    return spec, issues
