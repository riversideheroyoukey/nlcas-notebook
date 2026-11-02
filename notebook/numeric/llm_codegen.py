"""Vibe-coding path: LLM-generated numeric code (N-1 phase 2 centerpiece).

This is the ORIGINAL intent of the numeric layer (design principles doc):
expressions coming back from the CAS usually resist further analytic work,
so the researcher wants them EVALUATED — and the code that does it is
written by an LLM from the TeX/canonical form, reviewed by the researcher,
and executed in the sandbox.  The deterministic templates remain as the
fast path for whitelist-covered cases; THIS path is the general one.

Role separation (責務分離) is enforced, not advisory:
  - the LLM writes code against an explicit contract (NB_RESULT line,
    output dir, Agg backend) — it never touches the CAS;
  - a compile() gate rejects non-Python replies immediately;
  - LLM-generated code CANNOT run until the researcher approves it
    (engine-side review gate, not a UI convention);
  - the sandbox treats the code as adversarial regardless of approval;
  - when the deterministic template can also express the task, a
    differential check runs both codes and compares their NB_RESULT
    summaries — a mistranslated formula shows up as a numeric mismatch.

Like plan_adviser, everything except the real model's output quality is
deterministic and tested against a fake chat callable.
"""
from __future__ import annotations

import json
import re

from notebook.models.numeric import NumericTask
from notebook.runtime.shared_parser import UnknownNotationError

_FENCED_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.S)

# A stray bare language-tag line ("python"/"py") sometimes ends up as the
# literal first line of the extracted source — observed with gpt-oss-120b
# (S-ac eval, 2026-07-10): the reply starts with the word "python" on its
# own line and never opens a ``` fence at all, so _FENCED_RE never matches
# and the raw-text fallback keeps the stray line.  "python" alone is a
# syntactically valid no-op statement, so it slips past the compile() gate
# below and only fails at RUNTIME with NameError — invisible until the
# sandbox actually executes the script.  Strip it before the compile gate;
# a genuine first line of real code is never exactly "python"/"py" with
# nothing else on it, so this can only help.
_STRAY_LANG_TAG_RE = re.compile(r"\A[ \t]*(python|py)[ \t]*\r?\n", re.I)

_SYSTEM_PROMPT = """You write small, reviewable Python scripts that numerically evaluate a
mathematical expression from a theoretical-physics notebook.  Hard contract:
- Use numpy (as np) and matplotlib (Agg backend) only; both are available.
- Read the output directory from os.environ["NB_NUMERIC_OUT"] and write ALL
  files (figure.png, table.csv, ...) inside it.  Never touch the network.
- Bind every free symbol either from the sweep variable (np.linspace over the
  given range) or from the given fixed values.
- Finish by printing EXACTLY one line: NB_RESULT {json} where {json} contains
  at least {"target": str, "sweep_var": str, "min": {"x":..,"y":..},
  "max": {"x":..,"y":..}} computed from the swept curve.  If the task is a
  single-point evaluation (no sweep), print {"target": str, "value": number}
  instead.
- After NB_RESULT, print EXACTLY one line: NB_MANIFEST {json} declaring the
  produced artifacts and diagnostics, e.g. {"artifacts": [{"name": str,
  "value_kind": "scalar"|"series"|"vector"|"matrix"|"trace", "value_type":
  "real_float64"|"complex_float64"|"integer", "value": <small value>} or
  {..., "file": "table.csv"}], "diagnostics": {"success": bool,
  "method": str, "warnings": [str]}}.  Declare every value a later
  computation could consume; be honest in diagnostics.
- Output ONLY the Python code, in a single ```python fenced block.  No prose.
- The script must be plain, readable, and editable by a human reviewer —
  no cleverness, no dynamic exec, no imports beyond os/json/numpy/matplotlib.
"""


def build_codegen_prompt(task: NumericTask, latex: str,
                         scaffold: "str | None" = None) -> str:
    spec = {
        "problem_type": task.problem_type,
        "target_quantity": task.target_quantity or "y",
        "sweep_parameters": task.sweep_parameters,
        "fixed_values": task.fixed_values,
        "approximations": task.approximations,
        "spec": task.spec.model_dump(mode="json") if task.spec else None,
        "plot_sources": task.plot_sources,
    }
    parts = [
        f"Expression (LaTeX): {latex or task.source_latex or task.source_canonical}",
        f"Expression (CAS canonical): {task.source_canonical}",
        f"Task spec: {json.dumps(spec, ensure_ascii=False)}",
    ]
    if scaffold:
        parts.append(
            "A deterministic template produced this scaffold for the same "
            "task — use it as a starting point and keep its output contract:\n"
            f"```python\n{scaffold}\n```")
    parts.append("Write the script now.")
    return "\n\n".join(parts)


def extract_code(raw: str) -> str:
    """Extract and syntax-gate the Python code from the LLM reply.

    Raises UnknownNotationError (i18n key) when there is no code block or
    the code does not compile — a broken reply must never become a stored
    artifact."""
    if not raw or not raw.strip():
        raise UnknownNotationError("numeric.no_code_block")
    m = _FENCED_RE.search(raw)
    source = (m.group(1) if m else raw).strip()
    source = _STRAY_LANG_TAG_RE.sub("", source, count=1).strip()
    if not source:
        raise UnknownNotationError("numeric.no_code_block")
    try:
        compile(source, "<llm-generated>", "exec")
    except SyntaxError as exc:
        raise UnknownNotationError("numeric.code_syntax_error",
                                   error=str(exc)) from exc
    return source


def generate_llm_code(chat, task: NumericTask, latex: str = "",
                      scaffold: "str | None" = None) -> "tuple[str, dict]":
    """Ask the coding LLM for the script; return (source, provenance_meta)."""
    prompt = build_codegen_prompt(task, latex, scaffold)
    raw = chat(_SYSTEM_PROMPT, prompt)
    source = extract_code(raw or "")
    meta = {
        "source": "llm",
        "reviewed": False,               # the review gate keys off this
        "had_template_scaffold": scaffold is not None,
        "prompt_chars": len(prompt),
    }
    return source, meta


# ---------------------------------------------------------------------------
# Differential check: LLM code vs deterministic template
# ---------------------------------------------------------------------------

def _result_summary(stdout: str) -> "dict | None":
    m = re.search(r"^NB_RESULT (\{.*\})\s*$", stdout or "", re.M)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except ValueError:
        return None


def differential_check(runner, llm_source: str, template_source: str,
                       workdir, rel_tol: float = 1e-6) -> dict:
    """Run BOTH codes in the sandbox and compare their NB_RESULT summaries.

    A mistranslated formula in the LLM code shows up as a min/max mismatch
    against the deterministic template (which is only available for
    whitelist-covered tasks — exactly the cases where we have a ground
    truth to compare against).  Returns a verdict dict; never raises on a
    mismatch — the researcher reads the verdict.
    """
    from pathlib import Path
    workdir = Path(workdir)
    results = {}
    for name, source in (("llm", llm_source), ("template", template_source)):
        d = workdir / name
        d.mkdir(parents=True, exist_ok=True)
        code = d / "code.py"
        code.write_text(source)
        sb = runner.run_file(code, d)
        results[name] = {"ok": sb.ok, "summary": _result_summary(sb.stdout),
                         "stderr_tail": (sb.stderr or "")[-300:]}
    verdict: dict = {"runs": results, "match": False}
    a, b = results["llm"]["summary"], results["template"]["summary"]
    if results["llm"]["ok"] and results["template"]["ok"] and a and b:
        def _close(p, q):
            try:
                return abs(float(p) - float(q)) <= rel_tol * max(
                    1.0, abs(float(p)), abs(float(q)))
            except (TypeError, ValueError):
                return False
        checks = []
        for key in ("min", "max"):
            for axis in ("x", "y"):
                checks.append(_close(a.get(key, {}).get(axis),
                                     b.get(key, {}).get(axis)))
        verdict["match"] = all(checks)
        verdict["checks"] = checks
    return verdict
