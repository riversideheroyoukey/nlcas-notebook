"""N-1 phase 2, vibe-coding path — everything except the real model's output
quality, tested against FAKE chat callables:

  - the codegen prompt carries the TeX form, the canonical, the spec, and
    (when the whitelist covers the task) the deterministic scaffold;
  - extract_code's gates: fenced/raw extraction, no-code and syntax errors;
  - the engine review gate: LLM code cannot run until approved;
  - the path works where the template CANNOT (whitelist-external function);
  - the differential check catches a mistranslated formula numerically.
"""
import pytest

from notebook.engine import NotebookEngine
from notebook.models.numeric import NumericTask
from notebook.numeric.llm_codegen import (
    build_codegen_prompt, differential_check, extract_code, generate_llm_code,
)
from notebook.numeric.sandbox import SandboxRunner
from notebook.runtime.i18n import t
from notebook.runtime.shared_parser import UnknownNotationError

_runner = SandboxRunner()
needs_sandbox = pytest.mark.skipif(
    not _runner.available(),
    reason="macOS sandbox-exec or .venv-numeric python unavailable",
)

# A hand-written CORRECT script honoring the contract, for sin(2x) on [0, 2π].
GOOD_CODE = '''import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.environ.get("NB_NUMERIC_OUT", ".")
k = 2.0
x = np.linspace(0, 6.283185307179586, 200)
y = np.sin(k * x)
fig, ax = plt.subplots()
ax.plot(x, y)
fig.savefig(os.path.join(OUT, "figure.png"))
_imin, _imax = int(np.argmin(y)), int(np.argmax(y))
print("NB_RESULT", json.dumps({
    "target": "y", "sweep_var": "x", "n_points": 200,
    "min": {"x": float(x[_imin]), "y": float(y[_imin])},
    "max": {"x": float(x[_imax]), "y": float(y[_imax])},
}))
'''

# Same contract but the WRONG formula (cos instead of sin) — the
# differential check must catch this as a numeric mismatch.
WRONG_CODE = GOOD_CODE.replace("np.sin(k * x)", "np.cos(k * x)")


def _fenced(code: str) -> str:
    return f"Sure! Here's the script:\n```python\n{code}```\n"


def _task(**over) -> NumericTask:
    base = dict(
        task_id="N1", source_expression_ids=["E1"],
        source_canonical="sin(k*x)", source_latex=r"\sin(kx)",
        problem_type="parameter_sweep",
        sweep_parameters={"x": {"min": 0, "max": 6.283185307179586,
                                "points": 200}},
        fixed_values={"k": "2.0"},
    )
    base.update(over)
    return NumericTask(**base)


# ---------------------------------------------------------------------------
# Prompt content and code extraction
# ---------------------------------------------------------------------------

class TestPromptAndExtraction:

    def test_prompt_carries_tex_canonical_and_spec(self):
        p = build_codegen_prompt(_task(), latex=r"\sin(kx)")
        assert r"\sin(kx)" in p               # the TeX the researcher saw
        assert "sin(k*x)" in p                # the CAS canonical
        assert "sweep_parameters" in p and '"k": "2.0"' in p

    def test_prompt_includes_scaffold_when_given(self):
        p = build_codegen_prompt(_task(), latex="", scaffold="# SCAFFOLD")
        assert "# SCAFFOLD" in p and "starting point" in p

    def test_fenced_and_raw_extraction(self):
        assert extract_code(_fenced(GOOD_CODE)).startswith("import json")
        assert extract_code(GOOD_CODE).startswith("import json")

    def test_no_code_rejected(self):
        with pytest.raises(UnknownNotationError) as ei:
            extract_code("I cannot help with that.")
        # prose does not compile -> syntax gate; empty -> no_code_block
        assert ei.value.i18n_key in ("numeric.no_code_block",
                                     "numeric.code_syntax_error")
        with pytest.raises(UnknownNotationError):
            extract_code("")

    def test_stray_bare_language_tag_line_stripped(self):
        """Regression (S-ac eval, 2026-07-10): gpt-oss-120b sometimes replies
        with NO ``` fence at all, just the word "python" on its own first
        line followed by the script.  "python" alone is a syntactically
        valid no-op statement, so unstripped it slips past compile() and
        only fails at runtime with NameError.  Reproduced 3/3 trials."""
        raw = "python\n" + GOOD_CODE
        src = extract_code(raw)
        assert not src.splitlines()[0].strip() == "python"
        assert src.startswith("import json")
        compile(src, "<t>", "exec")   # would already pass; the point is no NameError at runtime

    def test_stray_language_tag_inside_fence_also_stripped(self):
        # defensive: the tag drifting onto its own line INSIDE a fence too
        fenced = f"```\npython\n{GOOD_CODE}```\n"
        assert extract_code(fenced).startswith("import json")

    def test_legitimate_leading_import_named_python_like_untouched(self):
        # only a BARE "python"/"py" line (nothing else) is stripped —
        # a real first line is never mistaken for the tag
        raw = "import python_utils as pu\nprint(1)\n"
        assert extract_code(raw).startswith("import python_utils")

    def test_syntax_error_rejected(self):
        with pytest.raises(UnknownNotationError) as ei:
            extract_code(_fenced("def broken(:\n    pass"))
        assert ei.value.i18n_key == "numeric.code_syntax_error"
        for lang in ("ja", "en"):
            assert t(ei.value.i18n_key, lang, **ei.value.fmt_kwargs)

    def test_generate_marks_unreviewed(self):
        source, meta = generate_llm_code(lambda s, u: _fenced(GOOD_CODE),
                                         _task())
        assert source.startswith("import json")
        assert meta["source"] == "llm" and meta["reviewed"] is False


# ---------------------------------------------------------------------------
# Engine: review gate and the beyond-whitelist capability
# ---------------------------------------------------------------------------

class TestReviewGate:

    def _engine_with_llm_code(self, chat):
        e = NotebookEngine(title="vibe gate")
        ex = e.register_expression("sin(k*x)")
        task = e.fork_numeric_task(
            ex.display_id,
            sweep_parameters={"x": {"min": 0, "max": 6.28, "points": 200}},
            fixed_values={"k": 2.0})
        art = e.generate_numeric_code_llm(task.task_id, chat=chat)
        return e, task, art

    def test_unreviewed_llm_code_refuses_to_run(self):
        e, task, art = self._engine_with_llm_code(
            lambda s, u: _fenced(GOOD_CODE))
        with pytest.raises(ValueError) as ei:
            e.run_numeric_task(task.task_id)
        assert art.artifact_id in str(ei.value)
        for lang in ("ja", "en"):
            assert t("numeric.code_not_reviewed", lang,
                     artifact_id=art.artifact_id)

    @needs_sandbox
    def test_approved_llm_code_runs(self, tmp_path):
        e, task, art = self._engine_with_llm_code(
            lambda s, u: _fenced(GOOD_CODE))
        e.save(tmp_path / "approved.json")
        e.approve_numeric_code(art.artifact_id)
        execution = e.run_numeric_task(task.task_id)
        assert execution.ok, execution.stderr

    def test_no_chat_gives_clear_error(self):
        e = NotebookEngine(title="vibe no chat")
        ex = e.register_expression("sin(k*x)")
        task = e.fork_numeric_task(
            ex.display_id, sweep_parameters={"x": {"min": 0, "max": 1}},
            fixed_values={"k": 1.0})
        with pytest.raises(ValueError) as ei:
            e.generate_numeric_code_llm(task.task_id, chat=None)
        assert "LLM" in str(ei.value)

    def test_whitelist_external_expression_still_gets_code(self):
        """The vibe path's home turf: the deterministic template CANNOT
        lower zeta(x), but the LLM path still produces a runnable artifact
        (scaffold is simply absent)."""
        e = NotebookEngine(title="vibe beyond")
        ex = e.register_expression("zeta(x)")
        task = e.fork_numeric_task(
            ex.display_id, sweep_parameters={"x": {"min": 2, "max": 5}})
        # deterministic path fails as expected
        with pytest.raises(ValueError):
            e.generate_numeric_code(task.task_id)
        # vibe path succeeds
        art = e.generate_numeric_code_llm(
            task.task_id, chat=lambda s, u: _fenced(GOOD_CODE))
        assert art.metadata["had_template_scaffold"] is False

    def test_scaffold_present_for_whitelist_covered_task(self):
        e, task, art = self._engine_with_llm_code(
            lambda s, u: _fenced(GOOD_CODE))
        assert art.metadata["had_template_scaffold"] is True
        assert "np.sin" in art.metadata["template_scaffold"]


# ---------------------------------------------------------------------------
# Differential check: numeric ground truth vs LLM code
# ---------------------------------------------------------------------------

@needs_sandbox
class TestDifferentialCheck:

    def test_correct_llm_code_matches_template(self, tmp_path):
        e = NotebookEngine(title="vibe diff ok")
        e.save(tmp_path / "diff_ok.json")
        ex = e.register_expression("sin(k*x)")
        task = e.fork_numeric_task(
            ex.display_id,
            sweep_parameters={"x": {"min": 0, "max": 6.283185307179586,
                                    "points": 200}},
            fixed_values={"k": 2.0})
        e.generate_numeric_code_llm(task.task_id,
                                    chat=lambda s, u: _fenced(GOOD_CODE))
        verdict = e.differential_check_llm_code(task.task_id)
        assert verdict["runs"]["llm"]["ok"]
        assert verdict["runs"]["template"]["ok"]
        assert verdict["match"] is True

    def test_mistranslated_formula_detected(self, tmp_path):
        """cos instead of sin: both codes RUN fine — only the numeric
        comparison against the deterministic template exposes the error."""
        e = NotebookEngine(title="vibe diff bad")
        e.save(tmp_path / "diff_bad.json")
        ex = e.register_expression("sin(k*x)")
        task = e.fork_numeric_task(
            ex.display_id,
            sweep_parameters={"x": {"min": 0, "max": 6.283185307179586,
                                    "points": 200}},
            fixed_values={"k": 2.0})
        e.generate_numeric_code_llm(task.task_id,
                                    chat=lambda s, u: _fenced(WRONG_CODE))
        verdict = e.differential_check_llm_code(task.task_id)
        assert verdict["runs"]["llm"]["ok"]          # it runs…
        assert verdict["match"] is False             # …but it is WRONG


# ---------------------------------------------------------------------------
# Coder-role resolution (role="coder" preferred, advisor fallback)
# ---------------------------------------------------------------------------

class TestCoderResolution:

    def test_coder_role_registered_with_4096_tokens(self):
        """Both configured endpoints carry the coder role; qwen is default."""
        from notebook.runtime.llm_client import LLMClient
        from config.config_utils import get_default_coder_target
        assert get_default_coder_target() == "ollama_qwen25_coder_32b"
        clients = LLMClient.find_all_endpoints(role="coder")
        names = {c.endpoint_name for c in clients}
        assert {"ollama_qwen25_coder_32b",
                "lmstudio_remote_gpt-oss-120b"} <= names
        for c in clients:
            assert c.default_max_tokens == 4096
            assert c.default_temperature == 0.2

    def test_preferred_name_selects_endpoint(self):
        from notebook.runtime.llm_client import LLMClient
        for name in ("ollama_qwen25_coder_32b", "lmstudio_remote_gpt-oss-120b"):
            llm = LLMClient.find_endpoint(role="coder", preferred_name=name)
            assert llm is not None and llm.endpoint_name == name

    def test_default_resolution_picks_qwen(self):
        from ui.notebook_bridge import _resolve_coder_chat
        chat = _resolve_coder_chat()
        assert callable(chat)
        assert chat.endpoint_name == "ollama_qwen25_coder_32b"

    def test_explicit_name_overrides_default(self):
        from ui.notebook_bridge import _resolve_coder_chat
        chat = _resolve_coder_chat("lmstudio_remote_gpt-oss-120b")
        assert chat.endpoint_name == "lmstudio_remote_gpt-oss-120b"

    def test_advisor_fallback_when_no_coder_role(self, tmp_path):
        """A models tree WITHOUT any coder role falls back to the advisor
        chain (which yields None in this test process — no crash)."""
        from ui.notebook_bridge import _resolve_coder_chat
        (tmp_path / "empty").mkdir()
        chat = _resolve_coder_chat(models_root=tmp_path)
        assert chat is None or callable(chat)
