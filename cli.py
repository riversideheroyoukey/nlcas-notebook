#!/usr/bin/env python3
"""Minimal REPL CLI for the Research Notebook MVP.

Usage:
    python cli.py
    python cli.py --title "Hamiltonian derivation"
    python cli.py --script calc.nb
    python cli.py --script calc.nb --interactive
    python cli.py --load session.json

All operations are expressed mathematically — no CAS syntax is required.
Script files use canonical math expressions (e.g. x^2-1, not LaTeX).
Lines starting with # are comments; blank lines are ignored.
"""
import json
import os
import sys
from pathlib import Path

# Windows' console defaults to the legacy codepage (e.g. cp932 for Japanese
# locales), not UTF-8 - printing a non-ASCII symbol (e.g. the "✗" below)
# raises UnicodeEncodeError there. macOS/Linux terminals are UTF-8 by
# default already, so this branch never touches them.
if os.name == "nt":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from notebook.engine import NotebookEngine
from notebook.models.session import SessionMode
from notebook.runtime.operation_validator import OperationValidator
from notebook.runtime.llm_supervisor import LLMSupervisor

_validator = OperationValidator()


def _read_scope_config() -> dict:
    """Read models/default/config.json; return {} on any error."""
    cfg_path = Path(__file__).resolve().parent / "models" / "default" / "config.json"
    try:
        return json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _make_supervisor(backend: str = "maxima") -> LLMSupervisor:
    """Build an LLMSupervisor backed by a local LLM (scope=local only).

    If no local endpoint is found, returns a rule-only supervisor.
    The scope policy is fixed to 'local' — cloud LLMs are never used
    for command interpretation.
    """
    from notebook.runtime.llm_client import LLMClient
    client = LLMClient.find_endpoint(role="command_parser", scope="local")
    return LLMSupervisor(validator=_validator, llm_client=client, primary_backend=backend)


def _make_adviser(backend: str = "maxima"):
    """Build an LLM client for the advisor role.

    Scope is read from models/default/config.json:
      adviser_scope = "local"  → only local LLMs (default)
      adviser_scope = "any"    → local preferred, cloud allowed as fallback
    Returns None if no matching endpoint is found.
    """
    from notebook.runtime.llm_client import LLMClient
    cfg = _read_scope_config()
    adviser_scope_policy = cfg.get("adviser_scope", "local")

    scope_filter = None if adviser_scope_policy == "any" else "local"
    client = LLMClient.find_endpoint(role="advisor", scope=scope_filter)
    if client is None and adviser_scope_policy == "any":
        # Explicit cloud fallback
        client = LLMClient.find_endpoint(role="advisor", scope="cloud")
    return client

HELP = """
Commands:
  reg <expr>                      Register expression (e.g. reg (x^2-1)/(x-1))
  expand <En>                     Expand expression
  factor <En>                     Factor expression
  simplify <En>                   Simplify (rationally)
  diff <En> <var>                 Differentiate with respect to var
  integrate <En> <var>            Indefinite integral
  limit <En> <var> <point>        Limit
  taylor <En> <var> <pt> <order>  Taylor expansion
  solve <En> <var>                Solve equation for var
  subst <En> <lhs> <rhs>          Substitute lhs → rhs in En
  latex <En>                      Show LaTeX for expression

  exprs                           Show all registered expressions
  history                         Show step history
  branches                        List branches
  fork <Sn>                       Fork a new branch at step Sn
  switch <branch_id>              Switch active branch
  replay                          Replay current branch from scratch

  save <file>                     Save session to JSON file

  help                            Show this message
  exit / quit                     Exit
"""


def fmt_exprs(engine: NotebookEngine) -> str:
    if not engine.expressions:
        return "  (no expressions yet)"
    lines = []
    for eid, expr in engine.expressions.items():
        branch_tag = (f"  [{expr.branch_id}]"
                      if expr.branch_id != "main" else "")
        lines.append(f"  {eid}: {expr.canonical}{branch_tag}")
    return "\n".join(lines)


def fmt_history(engine: NotebookEngine) -> str:
    if not engine.steps:
        return "  (no steps yet)"
    lines = []
    for sid, step in engine.steps.items():
        out = f"  → {', '.join(step.output_refs)}" if step.output_refs else ""
        status = "✓" if step.status.value == "success" else "✗"
        branch_tag = (f"  [{step.branch_id}]"
                      if step.branch_id != "main" else "")
        lines.append(f"  {status} {sid}  {step.visible_action}{out}{branch_tag}")
    return "\n".join(lines)


def _print_result(engine: NotebookEngine) -> None:
    if engine.expressions:
        last_id = list(engine.expressions)[-1]
        expr = engine.expressions[last_id]
        print(f"  {last_id} = {expr.canonical}")


def _confirm_operation(
    engine: NotebookEngine,
    result,
    auto_confirm: bool = False,
) -> None:
    """Show confirmation UI and execute if user agrees (or auto_confirm=True)."""
    if result.candidates and len(result.candidates) > 1:
        # Numbered menu for multiple candidates
        print("  複数の候補があります。番号を選んでください (N でキャンセル):")
        for i, cand in enumerate(result.candidates, 1):
            spec = _validator.get_spec(cand["operation"])
            disp = spec["display"]["ja"] if spec else cand["operation"]
            args_str = ", ".join(f"{k}={v}" for k, v in cand["args"].items())
            print(f"    {i}. {disp}: {args_str}")
        if auto_confirm:
            choice = "1"
        else:
            choice = input("  選択 [1-{}/N]: ".format(len(result.candidates))).strip()
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(result.candidates):
                cand = result.candidates[idx]
                _execute_supervised(engine, cand["operation"], cand["args"])
            else:
                print("  キャンセルしました。")
        else:
            print("  キャンセルしました。")
        return

    # Single candidate — Y/N confirmation
    if result.confirmation_prompt:
        print(f"\n  {result.confirmation_prompt}")
    else:
        spec = _validator.get_spec(result.operation)
        disp = spec["display"]["ja"] if spec else result.operation
        args_str = ", ".join(f"{k}={v}" for k, v in result.args.items())
        print(f"  操作 '{disp}' として解釈しました: {args_str}")

    if auto_confirm:
        ans = "y"
    else:
        ans = input("  実行しますか？ [y/N] ").strip().lower()
    if ans == "y":
        _execute_supervised(engine, result.operation, result.args)
    else:
        print("  キャンセルしました。")


def _execute_supervised(
    engine: NotebookEngine,
    operation: str,
    args: dict,
) -> None:
    """Execute an operation that was resolved by the supervisor."""
    spec = _validator.get_spec(operation)
    if operation == "to_latex":
        _, val = engine.run_operation(operation, args)
        print(f"  LaTeX: {val}")
    else:
        engine.run_operation(operation, args)
        _print_result(engine)


def _dispatch(engine: NotebookEngine, raw: str, auto_confirm: bool = False) -> bool:
    """Execute one command line.  Returns False when the session should end."""
    raw = raw.strip()
    if not raw or raw.startswith("#"):
        return True

    tokens = raw.split()
    cmd, args = tokens[0].lower(), tokens[1:]

    try:
        if cmd in ("exit", "quit"):
            print("Goodbye.")
            return False

        elif cmd == "help":
            print(HELP)

        elif cmd == "exprs":
            print(fmt_exprs(engine))

        elif cmd == "history":
            print(fmt_history(engine))

        elif cmd == "branches":
            for bid, b in engine.branches.items():
                active = " ← active" if bid == engine.active_branch_id else ""
                fork_info = (f"  (forked from {b.forked_from_step_id})"
                             if b.forked_from_step_id else "")
                head = f"  head: {b.head_step_id}" if b.head_step_id else ""
                print(f"  {bid}{active}{fork_info}{head}")

        elif cmd == "reg":
            if not args:
                print("Usage: reg <canonical_expr>"); return True
            canonical = " ".join(args)
            expr = engine.register_expression(canonical)
            print(f"  Registered {expr.display_id}: {expr.canonical}")

        elif _validator.resolve_alias(cmd) is not None:
            canonical = _validator.resolve_alias(cmd)
            spec = _validator.get_spec(canonical)
            required = spec["required_fields"]
            if len(args) < len(required):
                usage = " ".join(f"<{f}>" for f in required)
                print(f"Usage: {cmd} {usage}"); return True
            op_args = dict(zip(required, args))
            if canonical == "to_latex":
                _, val = engine.run_operation(canonical, op_args)
                print(f"  LaTeX: {val}")
            else:
                engine.run_operation(canonical, op_args)
                _print_result(engine)

        elif cmd == "save":
            if not args:
                print("Usage: save <file.json>"); return True
            engine.save(args[0])
            print(f"  Session saved to: {args[0]}")

        elif cmd == "replay":
            print("  Replaying current branch...")
            result = engine.replay_current_branch()
            if result.ok:
                print(f"  Replayed {result.replayed_steps} steps "
                      f"({result.skipped_steps} skipped). State restored.")
                for eid, val in result.expr_values.items():
                    print(f"    {eid}: {val}")
            else:
                print(f"  Replay completed with {len(result.errors)} error(s):")
                for err in result.errors:
                    print(f"    ✗ {err}")

        else:
            # Unknown structured command — try supervisor interpretation
            supervisor = _make_supervisor(engine.session.primary_backend)
            known_ids = set(engine.expressions.keys())
            result = supervisor.interpret(raw, known_ids)
            if not result.ok:
                print(f"  認識できませんでした: '{raw}'")
                if result.error:
                    print(f"  ({result.error})")
                print("  'help' でコマンド一覧を確認してください。")
            elif not result.requires_confirmation:
                # High-confidence rule-based hit — execute directly
                _execute_supervised(engine, result.operation, result.args)
            else:
                _confirm_operation(engine, result, auto_confirm=auto_confirm)

    except ValueError as e:
        print(f"  Error: {e}")
    except Exception as e:
        print(f"  Unexpected error: {e}")

    return True


def run_script(engine: NotebookEngine, path: str | Path,
               interactive: bool = False) -> None:
    """Execute commands from a script file, then optionally drop into REPL."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    print(f"Running script: {path}  ({len(lines)} lines)\n")

    for lineno, raw in enumerate(lines, start=1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            if stripped.startswith("#"):
                print(f"  {stripped}")   # echo comments for readability
            continue

        print(f"[{engine.active_branch_id}:{lineno}] > {stripped}")
        if not _dispatch(engine, stripped, auto_confirm=True):
            break

    print("\n--- script complete ---")
    if interactive:
        print('Dropping into interactive mode. Type "help" for commands.\n')
        run_repl(engine)


def run_repl(engine: NotebookEngine) -> None:
    print(f'\nSession: "{engine.session.title}"  backend: {engine.session.primary_backend}')
    print('Type "help" for commands.\n')

    while True:
        try:
            raw = input(f"[{engine.active_branch_id}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not _dispatch(engine, raw):
            break


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Research Notebook CLI")
    parser.add_argument("--title", default="Untitled", help="Session title")
    parser.add_argument("--mode", choices=["standard", "condensed"],
                        default="standard", help="Calculation mode")
    parser.add_argument("--load", metavar="FILE", help="Load session from JSON file")
    parser.add_argument("--script", metavar="FILE", help="Run commands from script file")
    parser.add_argument("--interactive", action="store_true",
                        help="Drop into REPL after --script completes")
    ns = parser.parse_args()

    print("Starting Maxima...")
    if ns.load:
        engine_ctx = NotebookEngine.load(ns.load)
        print(f'  Loaded session: "{engine_ctx.session.title}"')
    else:
        mode = (SessionMode.standard_symbolic if ns.mode == "standard"
                else SessionMode.second_quantization)
        engine_ctx = NotebookEngine(title=ns.title, mode=mode)

    with engine_ctx as engine:
        if ns.script:
            run_script(engine, ns.script, interactive=ns.interactive)
        else:
            run_repl(engine)


if __name__ == "__main__":
    main()
