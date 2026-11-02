#!/usr/bin/env python
"""Generate worked-example sessions (.json) and TeX exports (.tex) for the scenarios.

The single source of each scenario's commands is its doc under docs/scenarios/:
a fenced ``` ```scenario domain=<key> ``` ``` block listing one command per line.
`scenario_*_jp.md` holds the Japanese commands, `scenario_*_en.md` the English ones.
This generator reads those blocks and runs them — there is NO command list in this
file, so the docs and the generated examples cannot drift apart.

Command line conventions inside the block:
  - a plain line                  → a natural-language / equation TEXT command
  - a line wrapped in $ … $       → a MATH chip (sent with empty canonical, as the
                                    UI does; needed for H-1 compound-LHS ODEs and
                                    LaTeX integrals)
  - a line starting with @note:   → a manual research note (engine.record_manual_note),
                                    the same "📝 メモ" the UI writes — narrative
                                    prose that should reach the exported .tex
                                    alongside the equations, not just the .md doc
  - a line starting with @op:     → a direct engine.run_operation(operation, args)
                                    call: "@op: <operation> <json-args>". For
                                    generic alias-driven ops whose required_fields
                                    hold raw expression VALUES (not just E-id
                                    refs) — e.g. shannon_entropy's `p`, or
                                    implicit_derivative's `expr` — the rule-based
                                    field mapper (llm_supervisor._map_tokens_to_fields)
                                    is positional and cannot recover a "field=value"
                                    phrase, so there is no reliable no-LLM NL
                                    trigger for them yet. This mirrors the
                                    needs_confirmation branch below (which already
                                    calls run_operation directly for a confirmed
                                    candidate) — it is a generator/test-tooling
                                    extension only, not a new NL capability.

Output: domains/<domain>/examples/scenario_<title>_{jp,en}.{json,tex}.  The English
variant is produced only when a scenario_<title>_en.md exists.

Reproducible: commands are rule-based dispatched (no LLM), so a no-LLM supervisor
suffices and everything runs in a standard (Maxima) session.

    PATH="$HOME/.juliaup/bin:$PATH" .venv/bin/python scripts/generate_scenario_examples.py
"""
import asyncio
import json
import pathlib
import re
import shutil
import sys

# Python puts THIS file's directory (scripts/) on the path, not the repo root, so
# the command in the docstring above — the one CLAUDE.md and docs/scenarios.md
# both give — died with `ModuleNotFoundError: No module named 'ui'` (measured
# 2026-08-15).  Fixed here rather than by adding PYTHONPATH to the two documented
# command lines: a reader who copies either one should get a working run.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import ui.notebook_bridge as nb  # noqa: E402
from ui import session_io
from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline
from notebook.domain_paths import DOMAINS_ROOT
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operation_validator import OperationValidator

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCENARIOS_DIR = ROOT / "docs" / "scenarios"

_BLOCK_RE = re.compile(r"```scenario([^\n]*)\n(.*?)\n```", re.DOTALL)


def _parse_scenario_md(path: pathlib.Path) -> "tuple[str, list] | None":
    """Extract (domain, commands) from a scenario doc's ```scenario block, or None.

    A line wrapped in $…$ becomes a math chip {"math": …}; a line starting with
    "@note:" becomes a manual note {"note": …}; any other non-empty, non-comment
    line is a text command."""
    m = _BLOCK_RE.search(path.read_text(encoding="utf-8"))
    if not m:
        return None
    dm = re.search(r"domain=(\S+)", m.group(1))
    if not dm:
        return None
    commands: list = []
    for line in m.group(2).splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("@note:"):
            commands.append({"note": s[len("@note:"):].strip()})
        elif s.startswith("@op:"):
            _op_body = s[len("@op:"):].strip()
            _op_name, _op_json = _op_body.split(" ", 1)
            commands.append({"op": _op_name, "args": json.loads(_op_json)})
        elif s.startswith("$") and s.endswith("$") and len(s) > 2:
            commands.append({"math": s[1:-1].strip()})
        else:
            commands.append(line.rstrip())
    return dm.group(1), commands


def _sup(domain: str, primary_backend: str):
    # Real rule-based supervisor (llm_client=None), as debug_repl --no-llm uses, so
    # non-rule-based commands (e.g. symbol declarations) get the graceful no-LLM path
    # instead of a MagicMock result the pipeline cannot consume.  The backend MUST
    # match the engine's (start_engine infers julia_full for second-quantization
    # domains), so Julia scenarios work too — not just Maxima.
    return LLMSupervisor(validator=OperationValidator(), llm_client=None,
                         domains=[domain], primary_backend=primary_backend)


async def _run_commands(title: str, domain: str, commands: list,
                        language: str, suffix: str,
                        out_root: "pathlib.Path | None" = None):
    """Execute a scenario's commands and write {title}_{suffix}.json + .tex.

    `language` sets the session language (English → English visible_action / i18n /
    TeX preamble); `suffix` is "jp" or "en".  The commands are run verbatim — the
    English ones (from scenario_*_en.md) are a genuine English execution through the
    SAME rule-based pipeline, not a relabelled Japanese one.

    `out_root` overrides the domains root the examples are written under.  It
    exists so the drift check (EXAMPLES-DRIFT, tests/test_examples_up_to_date.py)
    can generate into a scratch directory and compare, rather than overwriting
    the committed examples in order to find out whether they are stale."""
    out_dir = (out_root or DOMAINS_ROOT) / domain / "examples"
    out_dir.mkdir(parents=True, exist_ok=True)
    nb.start_engine(title=title, mode="standard", language=language, domains=[domain])
    try:
        eng = nb.get_engine()
        sup = _sup(domain, eng.session.primary_backend)
        for cmd in commands:
            if isinstance(cmd, dict) and "note" in cmd:
                # A manual research note (the UI's "📝 メモ" button) — bypasses NL
                # interpretation entirely, exactly like ui/notebook_bridge.py's
                # /note endpoint does.
                eng.record_manual_note(cmd["note"])
                continue
            if isinstance(cmd, dict) and "op" in cmd:
                # Direct run_operation call — see the module docstring's "@op:"
                # convention. Same call the needs_confirmation branch below makes
                # for a confirmed candidate.
                eng.run_operation(cmd["op"], cmd["args"])
                continue
            if isinstance(cmd, dict) and "math" in cmd:
                # Drive math input EXACTLY as the UI does: LaTeX with an EMPTY
                # canonical so the server converts it (keeps the log faithful).
                tok = InputToken(type="math", value=cmd["math"],
                                 latex=cmd["math"], canonical="")
            else:
                tok = InputToken(type="text", value=cmd)
            r = await run_interpret_pipeline(InterpretRequest(tokens=[tok]), eng, sup)
            if r["status"] == "needs_confirmation" and r.get("candidates"):
                # Mirror the UI's confirm click: call engine.run_operation directly
                # with the chosen candidate's (operation, args) — exactly what
                # ui/notebook_bridge.py's /operation endpoint does when the user
                # picks a button, bypassing NL re-parsing.  For the subscript-role
                # dialog (backend_handler.py extra_pre_phase1), the LAST candidate
                # is always "名前の一部" (keep the full subscripted name, e.g.
                # k_B, xi_k) — the natural choice for a named physics constant.
                _chosen = r["candidates"][-1]
                eng.run_operation(_chosen["operation"], _chosen["args"])
                continue
            assert r["status"] in ("completed", "registered", "declared", "confirmed",
                                   "set_notation", "assumption_added"), f"{cmd!r} -> {r}"
        # Record the (Maxima) domain so the saved session is self-describing.
        eng.session.domains = [domain]
        eng.save(out_dir / f"{title}_{suffix}.json")
        res = await nb.export_latex()
        tex_src = session_io.sessions_dir() / pathlib.Path(res["path"]).name
        shutil.copy(tex_src, out_dir / f"{title}_{suffix}.tex")
        # A standalone lattice figure («格子図を保存して») is language-independent.
        _fig_svg = session_io.sessions_dir() / f"{title}_lattice.svg"
        _extra = ""
        if _fig_svg.exists():
            shutil.copy(_fig_svg, out_dir / f"{title}_lattice.svg")
            _extra = " + lattice.svg"
        print(f"{domain}: {title}_{suffix}.json + .tex{_extra}  ({len(commands)} steps)")
    finally:
        nb.stop_engine()


async def main(out_root: "pathlib.Path | None" = None):
    """Regenerate every scenario's example, and SAY whether it finished.

    A failing command raises out of `_run_commands`, so the run stops with the
    rest of the corpus left STALE — and a stale `.tex` still compiles and still
    shows a large, plausible `git diff`.  Measured 2026-08-22: four consecutive
    runs aborted at the same line without being noticed, and two commits claimed
    "全実例を再生成" while everything after the abort point was untouched.  The
    summary below is what makes "it ran" and "it FINISHED" different things to
    look at.
    """
    _planned = _done = 0
    try:
        for jp in sorted(SCENARIOS_DIR.glob("scenario_*_jp.md")):
            title = jp.stem[:-3]                   # strip the "_jp" suffix
            parsed = _parse_scenario_md(jp)
            if not parsed:
                continue                           # doc with no scenario block
            domain, commands = parsed
            en = SCENARIOS_DIR / f"{title}_en.md"
            ep = _parse_scenario_md(en) if en.exists() else None
            _planned += 1 + (1 if ep else 0)
            await _run_commands(title, domain, commands, "ja", "jp", out_root)
            _done += 1
            if ep:
                await _run_commands(title, ep[0] or domain, ep[1], "en", "en",
                                    out_root)
                _done += 1
    finally:
        print(f"--- regenerated {_done} example(s); "
              f"{'FINISHED' if _done >= _planned else 'STOPPED EARLY — the rest are STALE'}")


if __name__ == "__main__":
    import argparse
    _ap = argparse.ArgumentParser(description=__doc__)
    _ap.add_argument("--out-root", type=pathlib.Path, default=None,
                     help="write examples under this domains root instead of "
                          "domains/ (used by the drift check)")
    asyncio.run(main(_ap.parse_args().out_root))
