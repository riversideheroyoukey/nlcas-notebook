"""Debug REPL — stdin/stdout mirror of the notebook browser UI.

Lets you drive the full interpret pipeline without a browser, making
the system's responses machine-checkable and diff-testable.

Usage
-----
    python -m ui.debug_repl [--no-llm] [--lang ja|en] [--title TITLE]
                            [--session FILENAME] [--domains DOMAIN ...]

Flags
-----
  --no-llm              Use rule-based interpretation only (no LLM calls).
                        Confirmation dialogs auto-select the first candidate.
  --lang ja|en          Override session language (default: from config).
  --title TITLE         Title for the initial session (default: "Notebook").
  --session FILE        Load a saved session at startup instead of a blank one.
  --domains D [D ...]   One or more domain names to activate (e.g.
                        analytical_mechanics quantum_mechanics).
                        Injects the corresponding supervisor_hint.txt into
                        the LLM system prompt so domain-specific guidance
                        (e.g. Lagrangian ∂L/∂q̇ workflow) is available.

Output prefixes
---------------
  [reg:En]  <latex>    — expression registered (no operation)
  [cas:En]  <latex>    — CAS operation result
  [note]    <text>     — manual note stored
  [asm]     <cond>     — assumption added
  [-asm]    <cond>     — assumption removed
  [msg]     <text>     — confirmation / notation-change message
  [sys]     <text>     — infrastructure message (save, load, sessions list)
  [conf]    <prompt>   — confirmation dialog (before showing choices)
  [err]     <text>     — error

Commands
--------
  new [title] [ja|en]   Start a fresh session.
  load <filename>       Load a saved session.
  save [filename]       Save current session to new file.
  save!                 Overwrite the current file.
  export [filename]     Write a LaTeX document to disk.
  note <text>           Add a manual memo.
  sessions / ls         List saved sessions.
  status                Show active session info.
  help                  Show this help.
  quit / exit / q       Exit.

  <anything else>       Sent to the interpret pipeline.
"""
from __future__ import annotations

import argparse
import asyncio
import re
import sys
from functools import partial
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from notebook.engine import NotebookEngine
from notebook.models.session import SessionMode
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operation_validator import OperationValidator

from ui.interpret_pipeline import InputToken, InterpretRequest, run_interpret_pipeline
from ui.notebook_bridge import _infer_session_mode
import ui.session_io as session_io
from config.config_utils import (get_default_language, get_default_supervisor_target,
                                  get_supervisor_scope)


# ---------------------------------------------------------------------------
# Engine / supervisor factory
# ---------------------------------------------------------------------------

def _make_engine(title: str, language: str, domains: list[str] | None = None) -> NotebookEngine:
    mode = _infer_session_mode(domains or [])
    engine = NotebookEngine(title=title, mode=mode).__enter__()
    engine.session.language = [language, "en"] if language != "en" else ["en", "ja"]
    if domains:
        engine.session.domains = domains
        engine.symbol_context.domains = list(domains)   # sync (start() already ran)
    return engine


def _make_supervisor(no_llm: bool, domains: list[str] | None = None, *,
                     primary_backend: str) -> LLMSupervisor:
    """primary_backend is REQUIRED — see notebook_bridge._make_supervisor."""
    llm = None
    if not no_llm:
        try:
            from notebook.runtime.llm_client import LLMClient
            llm = LLMClient.find_endpoint(
                role="supervisor",
                scope=get_supervisor_scope() or None,
                models_root=_PROJECT_ROOT / "models",
                preferred_name=get_default_supervisor_target() or None,
            )
        except Exception:
            pass
    return LLMSupervisor(validator=OperationValidator(), llm_client=llm, domains=domains or [],
                         primary_backend=primary_backend)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _out(tag: str, text: str) -> None:
    print(f"[{tag}] {text}", flush=True)


def _print_registered(registered: list[dict]) -> None:
    for item in registered:
        latex = item.get("latex") or item.get("canonical", "")
        _out(f"reg:{item['expr_id']}", latex)


def _print_result(result: dict) -> None:
    """Print the non-confirmation parts of a pipeline result."""
    _print_registered(result.get("registered_expressions") or [])
    status = result.get("status", "")

    if status == "registered":
        pass  # already printed above

    elif status == "completed":
        msg = result.get("message", "")
        if msg:
            _out("msg", msg)
        expr_id = result.get("expr_id", "")
        latex   = result.get("latex") or result.get("canonical", "")
        if expr_id or latex:
            label  = result.get("operation_label", "")
            suffix = f"  ← {label}" if label else ""
            _out(f"cas:{expr_id}" if expr_id else "cas", f"{latex}{suffix}")

    elif status in ("confirmed", "set_notation", "set_latex_name",
                    "declare_time_dependent"):
        msg = result.get("message", "")
        if msg:
            _out("msg", msg)
        for item in result.get("updated_exprs") or []:
            _out(f"cas:{item['expr_id']}", item.get("latex", ""))

    elif status == "assumption_added":
        cond = result.get("condition", "")
        msg  = result.get("message", "") or f"assumed: {cond}"
        _out("asm", msg)

    elif status == "assumption_removed":
        cond = result.get("condition", "")
        _out("-asm", cond)

    elif status == "failed":
        _out("err", result.get("error", "(unknown error)"))


# ---------------------------------------------------------------------------
# Confirmation helpers
# ---------------------------------------------------------------------------

async def _read_line(prompt: str = "") -> str:
    """Read a line from stdin, with optional prompt if interactive."""
    loop = asyncio.get_event_loop()
    if sys.stdin.isatty() and prompt:
        sys.stdout.write(prompt)
        sys.stdout.flush()
    raw = await loop.run_in_executor(None, sys.stdin.readline)
    return raw.rstrip("\n")


def _show_candidates(candidates: list[dict], lang: str) -> None:
    for i, c in enumerate(candidates):
        label = c.get(f"display_{lang}") or c.get("display_en") or c.get("operation", "")
        _out("conf", f"  [{i+1}] {label}")


async def _pick_candidate(
    candidates: list[dict],
    lang: str,
    auto: bool,
) -> int | None:
    """Return 0-based index of chosen candidate, or None to cancel."""
    if auto:
        _out("conf", f"auto: [{candidates[0].get(f'display_{lang}') or candidates[0].get('display_en') or candidates[0].get('operation', '')}]")
        return 0
    _show_candidates(candidates, lang)
    choice = await _read_line("choice> ")
    choice = choice.strip()
    if choice.lower() in ("", "q", "cancel"):
        return None
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(candidates):
            return idx
    except ValueError:
        pass
    _out("err", "Invalid choice — cancelled")
    return None


async def _handle_needs_dep_confirmation(
    result: dict,
    engine: NotebookEngine,
    auto: bool,
    lang: str,
) -> None:
    """Handle dependency-conflict confirmation (total vs partial derivative)."""
    dep_vars  = result.get("dep_vars", [])
    operation = result.get("operation", "")
    args      = dict(result.get("args") or {})
    diff_var  = result.get("diff_var", "")
    vars_str  = ", ".join(dep_vars)

    _out("conf", f"{vars_str} is declared as a function of {diff_var}.")

    candidates = [
        {"label": "total",   "text": "Compute as total derivative"},
        {"label": "partial", "text": f"Treat {vars_str} as independent (partial derivative)"},
    ]
    for i, c in enumerate(candidates):
        _out("conf", f"  [{i+1}] {c['text']}")

    if auto:
        choice_label = candidates[0]["label"]
        _out("conf", f"auto: [{candidates[0]['text']}]")
    else:
        raw = await _read_line("choice> ")
        raw = raw.strip()
        if raw.lower() in ("", "q", "cancel"):
            _out("err", "Cancelled")
            return
        try:
            idx = int(raw) - 1
            choice_label = candidates[idx]["label"]
        except (ValueError, IndexError):
            _out("err", "Invalid choice — cancelled")
            return

    remove_deps_for = dep_vars if choice_label == "partial" else []
    dep_confirmed = True
    if remove_deps_for:
        loop = asyncio.get_event_loop()
        for var in remove_deps_for:
            await loop.run_in_executor(None, partial(engine.remove_variable_dependency, var))
    if dep_confirmed and not remove_deps_for:
        args["_is_total_derivative"] = True

    await _execute_operation(engine, operation, args)


async def _execute_operation(
    engine: NotebookEngine,
    operation: str,
    args: dict,
) -> None:
    """Run a validated operation and print the result."""
    loop = asyncio.get_event_loop()
    implicit_lhs = args.pop("_implicit_lhs", None)
    args.pop("_is_total_derivative", None)
    if implicit_lhs:
        var = args.get("var", "")
        if var:
            await loop.run_in_executor(
                None, partial(engine.declare_depends, implicit_lhs, [var])
            )
    if operation == "differentiate" and not implicit_lhs:
        var = args.get("var", "")
        expr_id = args.get("expr", "")
        src_expr = engine.expressions.get(expr_id)
        if var and src_expr:
            m = re.match(r"^'diff\(\s*([A-Za-z_]\w*)\s*,", src_expr.canonical)
            if m:
                dep_var = m.group(1)
                await loop.run_in_executor(
                    None, partial(engine.declare_depends, dep_var, [var])
                )
    try:
        step, value = await loop.run_in_executor(
            None, partial(engine.run_operation, operation, args)
        )
    except Exception as e:
        _out("err", str(e))
        return
    expr_id = step.output_refs[0] if step.output_refs else None
    expr    = engine.expressions.get(expr_id) if expr_id else None
    latex   = (expr.latex if expr else "") or value
    label   = step.visible_action or ""
    suffix  = f"  ← {label}" if label else ""
    tag     = f"cas:{expr_id}" if expr_id else "cas"
    _out(tag, f"{latex}{suffix}")


# ---------------------------------------------------------------------------
# LaTeX export (self-contained, no HTTP)
# ---------------------------------------------------------------------------

def _export_latex(engine: NotebookEngine, dest: Path | None = None) -> str:
    """Write a .tex file and return the filename used."""
    from notebook.runtime.replay import compute_lineage
    from notebook.models.step import StepKind as SK

    def _escape(text: str) -> str:
        for ch, rep in [('\\', r'\textbackslash{}'), ('&', r'\&'), ('%', r'\%'),
                        ('$', r'\$'), ('#', r'\#'), ('^', r'\^{}'), ('_', r'\_'),
                        ('{', r'\{'), ('}', r'\}'), ('~', r'\textasciitilde{}')]:
            text = text.replace(ch, rep)
        return text

    def _note_to_latex(text: str) -> str:
        parts = re.split(r'(\$\$[\s\S]*?\$\$|\$[^$\n]+?\$)', text)
        out = []
        for part in parts:
            if (part.startswith('$$') and part.endswith('$$') and len(part) > 4):
                out.append(part)
            elif (part.startswith('$') and part.endswith('$') and len(part) > 2):
                out.append(part)
            else:
                out.append(_escape(part).replace('\n', '\\\\\n'))
        return ''.join(out)

    lang  = engine.lang
    title = _escape(engine.session.title)
    lineage = compute_lineage(engine.active_branch_id, engine.steps, engine.branches)

    if lang == 'ja':
        preamble = [
            r'% Compile with: uplatex file.tex && dvipdfmx file.dvi',
            r'%           or: latexmk -uplatex file.tex',
            r'\documentclass[uplatex,dvipdfmx]{jsarticle}',
            r'\usepackage{amsmath, amssymb}',
        ]
    else:
        preamble = [
            r'\documentclass{article}',
            r'\usepackage{amsmath, amssymb}',
            r'\usepackage[T1]{fontenc}',
            r'\usepackage[utf8]{inputenc}',
        ]

    lines = preamble + [
        r'% \instruction{} displays NL instructions in the PDF as italic text.',
        r'% To hide all instruction lines from the PDF output, replace the next line with:',
        r'%   \newcommand{\instruction}[1]{}',
        r'% 指示行を PDF に表示しない場合は、次の行を以下のように書き換えてください：',
        r'%   \newcommand{\instruction}[1]{}',
        r'\newcommand{\instruction}[1]{\noindent\textit{\small #1}\par\medskip}',
        '',
        f'\\title{{{title}}}',
        r'\date{\today}',
        '',
        r'\begin{document}',
        r'\maketitle',
        '',
    ]

    for step in lineage:
        if step.kind == SK.user_input and step.user_text:
            lines.append(f'\\instruction{{{_escape(step.user_text)}}}')
            lines.append('')
        elif step.kind in (SK.expression_registration, SK.cas_operation):
            if step.visible_action and step.kind == SK.cas_operation:
                lines.append(f'% {_escape(step.visible_action)}')
            for ref in step.output_refs:
                expr = engine.expressions.get(ref)
                if expr:
                    lines.append(r'\begin{equation}')
                    lines.append(f'  {expr.latex}')
                    lines.append(f'  \\label{{eq:{expr.display_id}}}')
                    lines.append(r'\end{equation}')
                    lines.append('')
        elif step.kind == SK.assumption_add:
            # Prefer the TeX-facing display ("\gamma > 0") over the bare
            # Maxima-safe canonical ("gamma > 0") — matches ui/notebook_bridge.py's
            # export_latex.
            display_tex = (step.operation_args.get("display")
                           or step.operation_args.get("canonical", ""))
            lines.append(f'\\noindent\\textbf{{Assumption:}}\\ $\\displaystyle {display_tex}$\\par')
            lines.append('')
        elif step.kind == SK.manual_note and step.user_text:
            lines.append(_note_to_latex(step.user_text))
            lines.append('')

    lines.append(r'\end{document}')
    tex = '\n'.join(lines)

    if dest is None:
        safe = re.sub(r'[^\w\-]', '_', engine.session.title) or 'notebook'
        dest = Path(f"{safe}.tex")
    elif dest.suffix != ".tex":
        dest = dest.with_suffix(".tex")

    dest.write_text(tex, encoding="utf-8")
    return str(dest)


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

async def _cmd_new(
    parts: list[str],
    engine_ref: list[NotebookEngine],
    supervisor_ref: list[LLMSupervisor],
    no_llm: bool,
) -> None:
    title = parts[1] if len(parts) > 1 and parts[1] not in ("ja", "en") else "Notebook"
    lang  = "en"
    for p in parts[1:]:
        if p in ("ja", "en"):
            lang = p
    if engine_ref[0] is not None:
        engine_ref[0].__exit__(None, None, None)
    engine_ref[0] = _make_engine(title, lang)
    supervisor_ref[0] = _make_supervisor(no_llm, primary_backend=engine_ref[0].session.primary_backend)
    session_io.reset_save_state()
    _out("sys", f"New session: \"{title}\" (lang={lang})")


async def _cmd_load(
    parts: list[str],
    engine_ref: list[NotebookEngine],
    supervisor_ref: list[LLMSupervisor],
    no_llm: bool,
) -> None:
    if len(parts) < 2:
        _out("err", "Usage: load <filename>")
        return
    filename = parts[1]
    if not filename.endswith(".json"):
        filename += ".json"
    try:
        if engine_ref[0] is not None:
            engine_ref[0].__exit__(None, None, None)
        engine_ref[0] = await session_io.do_load(filename)
        supervisor_ref[0] = _make_supervisor(no_llm, domains=engine_ref[0].session.domains,
                                             primary_backend=engine_ref[0].session.primary_backend)
        expr_count = len(engine_ref[0].expressions)
        _out("sys", f"Loaded: {filename} ({expr_count} expressions)")
    except FileNotFoundError:
        _out("err", f"Session not found: {filename}")
    except Exception as e:
        _out("err", f"Load failed: {e}")


async def _cmd_save(
    parts: list[str],
    engine: NotebookEngine,
    overwrite: bool,
) -> None:
    filename = parts[1] if len(parts) > 1 else ""
    try:
        fname = await session_io.do_save(engine, filename=filename, overwrite=overwrite)
        _out("sys", f"Saved: {fname}")
    except Exception as e:
        _out("err", f"Save failed: {e}")


async def _cmd_export(
    parts: list[str],
    engine: NotebookEngine,
) -> None:
    dest = Path(parts[1]) if len(parts) > 1 else None
    try:
        fname = _export_latex(engine, dest)
        _out("sys", f"Exported: {fname}")
    except Exception as e:
        _out("err", f"Export failed: {e}")


async def _cmd_note(
    text: str,
    engine: NotebookEngine,
) -> None:
    loop = asyncio.get_event_loop()
    if not text.strip():
        _out("err", "Note text is empty")
        return
    try:
        step = await loop.run_in_executor(None, partial(engine.record_manual_note, text.strip()))
        _out("note", text.strip())
    except Exception as e:
        _out("err", str(e))


async def _cmd_sessions() -> None:
    sessions = await session_io.do_list_sessions()
    if not sessions:
        _out("sys", "No saved sessions.")
        return
    for s in sessions:
        _out("sys",
             f"{s['filename']} — {s['modified_at']} — {s['expr_count']} exprs / {s['step_count']} steps")


def _cmd_status(engine: NotebookEngine) -> None:
    lang    = engine.lang
    title   = engine.session.title
    branch  = engine.active_branch_id
    exprs   = len(engine._active_expr_ids())
    steps   = engine._step_counter
    fname   = session_io.get_current_filename() or "(not saved)"
    mode    = engine.session.mode.value
    backend = engine.session.primary_backend
    domains = engine.session.domains or []
    _out("sys", f"title={title!r}  lang={lang}  mode={mode}  backend={backend}  domains={domains}")
    _out("sys", f"branch={branch}  exprs={exprs}  steps={steps}  file={fname}")


def _help_text() -> str:
    return (
        "Commands:\n"
        "  new [title] [ja|en]   New session\n"
        "  load <file>           Load session\n"
        "  save [file]           Save to new file\n"
        "  save!                 Overwrite current file\n"
        "  export [file]         Export LaTeX\n"
        "  note <text>           Add memo\n"
        "  sessions / ls         List saved sessions\n"
        "  status                Session info\n"
        "  help                  This help\n"
        "  quit / exit / q       Exit\n"
        "  <anything else>       Interpret pipeline"
    )


# ---------------------------------------------------------------------------
# Interpret dispatch
# ---------------------------------------------------------------------------

_EXPR_REF_RE = re.compile(r'\b(E\d+)\b')


def _extract_selected_expr_id(line: str, engine: NotebookEngine) -> str:
    """Return first expression ID mentioned in line that exists in the engine."""
    for m in _EXPR_REF_RE.finditer(line):
        eid = m.group(1)
        if eid in engine.expressions:
            return eid
    return ""


async def _handle_interpret(
    line: str,
    engine: NotebookEngine,
    supervisor: LLMSupervisor,
    no_llm: bool,
) -> None:
    """Send a line to the interpret pipeline and handle the result."""
    tokens = [InputToken(type="text", value=line)]
    req    = InterpretRequest(
        tokens=tokens,
        selected_expr_id=_extract_selected_expr_id(line, engine),
    )

    try:
        result = await run_interpret_pipeline(req, engine, supervisor)
    except Exception as e:
        _out("err", str(e))
        return

    status = result.get("status", "")
    lang   = engine.lang

    if status == "needs_dep_confirmation":
        _print_registered(result.get("registered_expressions") or [])
        await _handle_needs_dep_confirmation(result, engine, auto=no_llm, lang=lang)
        return

    if status == "needs_confirmation":
        _print_registered(result.get("registered_expressions") or [])
        prompt     = result.get("confirmation_prompt", "Please confirm.")
        operation  = result.get("operation", "")
        args       = dict(result.get("args") or {})
        candidates = result.get("candidates") or []
        pending    = result.get("pending_dependencies") or []

        _out("conf", prompt)

        if candidates:
            idx = await _pick_candidate(candidates, lang, auto=no_llm)
            if idx is None:
                return
            chosen = candidates[idx]
            operation = chosen.get("operation", operation)
            args = dict(chosen.get("args") or args)
        else:
            # Single confirmed operation — just execute
            if no_llm:
                _out("conf", "auto: confirmed")

        # Apply pending dependency declarations
        loop = asyncio.get_event_loop()
        for dep in pending:
            dep_name = dep.get("dependent")
            dep_indep = dep.get("independent", [])
            if dep_name and dep_indep:
                await loop.run_in_executor(
                    None, partial(engine.declare_depends, dep_name, dep_indep)
                )

        # Check for dep conflict on the confirmed differentiate
        if operation == "differentiate":
            diff_var = args.get("var", "")
            src_expr = engine.expressions.get(args.get("expr", ""))
            if src_expr and diff_var:
                conflicts = await loop.run_in_executor(
                    None, partial(engine.check_diff_context, src_expr.canonical, diff_var)
                )
                if conflicts:
                    dep_vars = list({v for v, _ in conflicts})
                    dep_result = {
                        "dep_vars": dep_vars,
                        "operation": operation,
                        "args": args,
                        "diff_var": diff_var,
                    }
                    await _handle_needs_dep_confirmation(dep_result, engine, auto=no_llm, lang=lang)
                    return

        await _execute_operation(engine, operation, args)
        return

    _print_result(result)


# ---------------------------------------------------------------------------
# Main REPL loop
# ---------------------------------------------------------------------------

async def repl(args: argparse.Namespace) -> None:
    no_llm  = args.no_llm
    lang    = args.lang or get_default_language()
    domains = args.domains or []

    engine_ref:     list[NotebookEngine | None]  = [None]
    supervisor_ref: list[LLMSupervisor | None]   = [None]

    if args.session:
        filename = args.session
        if not filename.endswith(".json"):
            filename += ".json"
        try:
            engine_ref[0]     = await session_io.do_load(filename)
            # --domains overrides saved session domains when explicitly specified
            if domains:
                engine_ref[0].session.domains = domains
            supervisor_ref[0] = _make_supervisor(no_llm, domains=engine_ref[0].session.domains,
                                                 primary_backend=engine_ref[0].session.primary_backend)
            _out("sys", f"Loaded: {filename} ({len(engine_ref[0].expressions)} expressions)")
        except Exception as e:
            _out("err", f"Load failed: {e} — starting blank session")
            engine_ref[0]     = _make_engine(args.title, lang, domains=domains)
            supervisor_ref[0] = _make_supervisor(no_llm, domains=domains,
                                                 primary_backend=engine_ref[0].session.primary_backend)
            session_io.reset_save_state()
    else:
        engine_ref[0]     = _make_engine(args.title, lang, domains=domains)
        supervisor_ref[0] = _make_supervisor(no_llm, domains=domains,
                                             primary_backend=engine_ref[0].session.primary_backend)
        session_io.reset_save_state()
        _domains_str = f"  domains={domains}" if domains else ""
        _out("sys", f"New session: \"{args.title}\" (lang={lang}{'  --no-llm' if no_llm else ''}{_domains_str})")

    _out("sys", "Type 'help' for commands, 'quit' to exit.")
    print("", flush=True)

    while True:
        try:
            line = await _read_line("> ")
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            if sys.stdin.isatty():
                continue
            # Non-interactive: empty line → skip
            continue

        engine     = engine_ref[0]
        supervisor = supervisor_ref[0]
        if engine is None or supervisor is None:
            _out("err", "No active session — use 'new' or 'load'")
            continue

        parts = line.split()
        cmd   = parts[0].lower() if parts else ""

        if cmd in ("quit", "exit") or (cmd == "q" and len(parts) == 1):
            break

        elif cmd == "help":
            print(_help_text(), flush=True)

        elif cmd == "new":
            await _cmd_new(parts, engine_ref, supervisor_ref, no_llm)

        elif cmd == "load":
            await _cmd_load(parts, engine_ref, supervisor_ref, no_llm)

        elif cmd == "save!" or (cmd == "save" and line.strip() == "save!"):
            await _cmd_save(parts, engine, overwrite=True)

        elif cmd == "save":
            await _cmd_save(parts, engine, overwrite=False)

        elif cmd == "export":
            await _cmd_export(parts, engine)

        elif cmd == "note":
            text = line[len("note"):].strip()
            await _cmd_note(text, engine)

        elif cmd in ("sessions", "ls"):
            await _cmd_sessions()

        elif cmd == "status":
            _cmd_status(engine)

        else:
            await _handle_interpret(line, engine, supervisor, no_llm)

        print("", flush=True)

    # Cleanup
    if engine_ref[0] is not None:
        engine_ref[0].__exit__(None, None, None)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Debug REPL — stdin/stdout interface for the research notebook.",
    )
    parser.add_argument("--no-llm", action="store_true",
                        help="Use rule-based interpretation only.")
    parser.add_argument("--lang", choices=["ja", "en"], default=None,
                        help="Session language (default: from config).")
    parser.add_argument("--title", default="Notebook",
                        help="Title for the initial session.")
    parser.add_argument("--session", default=None,
                        help="Load a saved session at startup.")
    parser.add_argument("--domains", nargs="+", default=None, metavar="DOMAIN",
                        help="Domain names to activate (e.g. analytical_mechanics).")
    args = parser.parse_args()
    asyncio.run(repl(args))


if __name__ == "__main__":
    main()
