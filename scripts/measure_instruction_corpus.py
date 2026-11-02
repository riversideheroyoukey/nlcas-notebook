#!/usr/bin/env python
"""Replay every scenario and RECORD what each instruction line reached — do not abort.

Why this is not ``generate_scenario_examples.py``: that script ``assert``s on the
first command whose status is not a success, so it answers "did the whole corpus
run?" and nothing else.  The question here is per-line — *which operation did this
spelling reach* — and a line that fails must not hide the ~40 lines after it
(measured 2026-08-22: four consecutive regenerations aborted at the same line).

Every line is executed through the SAME rule-based pipeline the researcher's own
typing goes through (``run_interpret_pipeline`` + the no-LLM supervisor), in the
SAME scenario context (the preceding lines are replayed first), because a bare
``E3 を p で微分して`` in a fresh session fails for reasons that have nothing to do
with reading the instruction.

Output: JSON on stdout (or --out), one record per NON-note command line:

    {scenario, lang, index, kind, text, status, operation, args, error,
     after_failure}

``operation`` is read from the engine's own timeline (the ``llm_supervision`` /
``cas_operation`` step the line produced), not from the response dict, so it is
the same field the saved sessions carry.

    PATH="$HOME/.juliaup/bin:$PATH" .venv/bin/python scripts/measure_instruction_corpus.py
"""
import argparse
import asyncio
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import ui.notebook_bridge as nb  # noqa: E402
from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline  # noqa: E402
from scripts.generate_scenario_examples import _parse_scenario_md, _sup, SCENARIOS_DIR  # noqa: E402

_OK = ("completed", "registered", "declared", "confirmed", "set_notation",
       "assumption_added")


def _steps_after(engine, seen: set) -> list:
    """The timeline steps this command added, in order."""
    out = []
    for sid, st in engine.steps.items():
        if sid not in seen:
            seen.add(sid)
            out.append(st)
    return out


def _operation_of(steps) -> "tuple[str | None, dict]":
    """The operation the line reached, preferring the interpretation step.

    ``llm_supervision`` carries the args as READ (the operands); ``cas_operation``
    carries them as EXECUTED.  A line that only registers an expression has
    neither, and reports ``(expression)`` — which is itself an answer, and the
    shape of the 2026-08-22 defects where an instruction was silently registered
    as a formula.

    The step kind is ``expression_registration``.  Spelling it
    ``expression_registered`` here (measured 2026-08-23) reported 437 of 963 text
    lines as "reached no operation at all" instead of "registered an expression"
    — a 45% hole in the measurement that looked like a finding about the system.
    """
    for st in steps:
        if getattr(st, "kind", None) == "llm_supervision" and st.operation:
            return st.operation, dict(st.operation_args or {})
    for st in steps:
        if st.operation:
            return st.operation, dict(st.operation_args or {})
    if any(getattr(st, "kind", None) == "expression_registration" for st in steps):
        return "(expression)", {}
    return None, {}


def _created_exprs(engine, steps) -> list:
    """(display_id, canonical, latex) for every expression this line registered.

    The LATEX is here because comparing operation and operands is not enough:
    measured 2026-08-23, the Japanese and English sessions produced the same
    Heisenberg-equation VALUE and displayed it differently — Japanese as
    ``\frac{d a(1)}{d t} = …`` and English as the bare right-hand side, a result
    published without saying what it is the derivative of.  Both languages
    agreed on operation and operands, so that divergence was invisible to this
    measurement; only regenerating the worked examples caught it.

    The E-ids are what later lines REFER to, so a language that registers one
    extra expression shifts every following reference — open_items
    OPERAND-BOUNDARY ② says the two languages already differ this way.  Recording
    them is what turns that from a note into a measurement.
    """
    out = []
    for st in steps:
        for eid in (getattr(st, "output_refs", None) or []):
            expr = engine.expressions.get(eid)
            out.append([eid, (expr.canonical if expr else None),
                        (expr.latex if expr else None)])
    return out


def _card_labels(r: dict) -> list:
    """The label of every card the UI will DRAW for this response.

    Replays ``notebook_main.js``'s own branch: each ``registered_expressions``
    entry is drawn, and ``data`` only when none of them already showed its
    ``expr_id``.  Recording the response fields instead is not the same thing —
    measured 2026-08-27, ``substitute_into`` returned an ``operation_label``
    that never reached a card, because its result was ALSO appended to
    ``registered_expressions`` and that entry (no label) was drawn instead.

    ``None`` in this list is a card the researcher sees with no operation label.
    """
    cards = [e.get("operation_label")
             for e in (r.get("registered_expressions") or [])]
    if not any(e.get("expr_id") == r.get("expr_id")
               for e in (r.get("registered_expressions") or [])):
        cards.append(r.get("operation_label"))
    return cards


def _public_args(args: dict) -> dict:
    """The operands as written — internal ``_``-prefixed keys are bookkeeping."""
    return {k: v for k, v in (args or {}).items() if not k.startswith("_")}


async def _replay(title: str, domain: str, commands: list, language: str,
                  lang_tag: str, records: list):
    nb.start_engine(title=title, mode="standard", language=language,
                    domains=[domain])
    try:
        eng = nb.get_engine()
        sup = _sup(domain, eng.session.primary_backend)
        seen = set(eng.steps)
        failed_before = False
        for idx, cmd in enumerate(commands):
            if isinstance(cmd, dict) and "note" in cmd:
                eng.record_manual_note(cmd["note"])
                _steps_after(eng, seen)
                continue
            if isinstance(cmd, dict) and "op" in cmd:
                try:
                    eng.run_operation(cmd["op"], cmd["args"])
                    status, err = "completed", None
                except Exception as exc:                    # noqa: BLE001
                    status, err = "failed", f"{type(exc).__name__}: {exc}"
                    failed_before = True
                _steps_after(eng, seen)
                records.append(dict(scenario=title, lang=lang_tag, index=idx,
                                    kind="op", text=f"@op: {cmd['op']}",
                                    status=status, operation=cmd["op"],
                                    args=_public_args(cmd["args"]), route=None,
                                    routes=[], created=[],
                                    error=err, after_failure=failed_before))
                continue
            if isinstance(cmd, dict) and "math" in cmd:
                tok = InputToken(type="math", value=cmd["math"],
                                 latex=cmd["math"], canonical="")
                kind, text = "math", cmd["math"]
            else:
                tok = InputToken(type="text", value=cmd)
                kind, text = "text", cmd
            try:
                r = await run_interpret_pipeline(
                    InterpretRequest(tokens=[tok]), eng, sup)
            except Exception as exc:                        # noqa: BLE001 — recorded
                records.append(dict(scenario=title, lang=lang_tag, index=idx,
                                    kind=kind, text=text, status="raised",
                                    operation=None, args={}, route=None,
                                    routes=[], created=[],
                                    error=f"{type(exc).__name__}: {exc}",
                                    after_failure=failed_before))
                failed_before = True
                continue
            new = _steps_after(eng, seen)
            status = r.get("status")
            if status == "needs_confirmation" and r.get("candidates"):
                chosen = r["candidates"][-1]
                try:
                    eng.run_operation(chosen["operation"], chosen["args"])
                except Exception as exc:                    # noqa: BLE001
                    r = dict(r, error=f"{type(exc).__name__}: {exc}")
                new += _steps_after(eng, seen)
            op, args = _operation_of(new)
            # WHICH rule claimed this line.  `run_interceptors` already records
            # the winner (ctx.route, the response dict, and every step it
            # produced), so this needs no instrumentation — and it is what makes
            # the DECLINES measurable: every interceptor listed before the winner
            # in the registry was offered this exact sentence and passed on it.
            # Those (sentence, rule) pairs are the negative examples NL-GRAMMAR ②
            # needs, and they are real spellings rather than invented ones.
            # A line can be touched by MORE THAN ONE rule: step 0 registers the
            # leading formula (`inline_formula_ja` / `_en`) and an interceptor
            # then handles the rest.  One field cannot say that, so record both
            # — `route` is who ANSWERED, `routes` is every rule that acted.
            # Measured 2026-08-23: recording only the winner hid the step-0
            # attribution on every line where an interceptor also fired, which
            # is most of them.
            routes = list(dict.fromkeys(
                [st.route for st in new if getattr(st, "route", None)]))
            route = r.get("route") or (routes[0] if routes else None)
            ok = status in _OK or (status == "needs_confirmation"
                                   and r.get("candidates"))
            if not ok:
                failed_before = True
            records.append(dict(scenario=title, lang=lang_tag, index=idx,
                                kind=kind, text=text, status=status,
                                operation=op, args=_public_args(args),
                                route=route, routes=routes,
                                created=_created_exprs(eng, new),
                                cards=_card_labels(r),
                                error=(r.get("error") if not ok else None),
                                after_failure=failed_before))
    finally:
        nb.stop_engine()


async def main(out: "pathlib.Path | None", only: "str | None"):
    records: list = []
    for jp in sorted(SCENARIOS_DIR.glob("scenario_*_jp.md")):
        title = jp.stem[:-3]
        if only and only not in title:
            continue
        parsed = _parse_scenario_md(jp)
        if not parsed:
            continue
        domain, commands = parsed
        await _replay(title, domain, commands, "ja", "jp", records)
        en = SCENARIOS_DIR / f"{title}_en.md"
        ep = _parse_scenario_md(en) if en.exists() else None
        if ep:
            await _replay(title, ep[0] or domain, ep[1], "en", "en", records)
        print(f"  {title}: {len(records)} records so far", file=sys.stderr)
    text = json.dumps(records, ensure_ascii=False, indent=1)
    if out:
        out.write_text(text, encoding="utf-8")
        print(f"--- wrote {len(records)} record(s) to {out}", file=sys.stderr)
    else:
        print(text)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=pathlib.Path, default=None)
    ap.add_argument("--only", default=None, help="substring of a scenario title")
    a = ap.parse_args()
    asyncio.run(main(a.out, a.only))
