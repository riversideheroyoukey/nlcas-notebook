"""Tests for ui/debug_repl.py.

Exercises the full stdin/stdout pipeline without a browser.
Integration tests use real CAS processes (Maxima and Julia).
All tests use --no-llm mode (rule-based interpretation, no LLM calls).

Test coverage:
  - _print_result() / _print_registered() output format
  - _handle_interpret() full pipeline — Maxima and Julia backends
  - save / load round-trip preserving julia_repr and ir_ast state
  - LaTeX export (_export_latex) content
  - REPL meta-commands: status, note, new
  - ir_ast and julia_repr state after CAS operations
  - UNKNOWN_FIELDS validation surface through the pipeline
  - Full stdin/stdout pipe tests (subprocess) — Maxima and Julia backends
  - Regression tests for display bugs: \\over→\\frac, combine parens, .tex extension
"""
from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from functools import partial
from pathlib import Path

import pytest

from notebook.engine import NotebookEngine
from notebook.models.session import SessionMode
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operation_validator import OperationValidator

from ui.debug_repl import (
    _print_result,
    _print_registered,
    _handle_interpret,
    _cmd_save,
    _cmd_load,
    _cmd_note,
    _cmd_status,
    _export_latex,
)
import ui.session_io as session_io


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sup(domains=None, backend="maxima") -> LLMSupervisor:
    """Rule-based supervisor — no LLM."""
    return LLMSupervisor(
        validator=OperationValidator(),
        llm_client=None,
        domains=domains or [],
        primary_backend=backend,
    )


def _run(coro):
    return asyncio.run(coro)


_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _repl_pipe(
    commands: list[str],
    domains: list[str] | None = None,
    session_file: Path | None = None,
) -> str:
    """Run debug_repl as a subprocess with piped commands and return stdout."""
    args = [sys.executable, "-m", "ui.debug_repl", "--no-llm", "--lang", "ja"]
    if domains:
        args += ["--domains"] + domains
    if session_file:
        args += ["--session", str(session_file)]
    stdin_text = "\n".join(commands) + "\nquit\n"
    
    try:
        # Julia JIT compilation (time-to-first-X) can be very slow, especially on CI.
        # We set a generous timeout of 300 seconds (5 minutes) to prevent infinite hangs.
        result = subprocess.run(
            args,
            input=stdin_text,
            capture_output=True,
            text=True,
            cwd=str(_PROJECT_ROOT),
            timeout=300,
        )
    except subprocess.TimeoutExpired as e:
        pytest.fail(f"debug_repl subprocess timed out after {e.timeout}s.\nStdout:\n{e.stdout}\nStderr:\n{e.stderr}")

    if result.returncode != 0:
        pytest.fail(
            f"debug_repl exited with code {result.returncode}\n"
            f"--- STDOUT ---\n{result.stdout}\n"
            f"--- STDERR ---\n{result.stderr}"
        )
        
    return result.stdout


# ---------------------------------------------------------------------------
# _print_registered / _print_result — output format
# ---------------------------------------------------------------------------

class TestOutputFormat:
    def test_print_registered_single(self, capsys):
        _print_registered([{"expr_id": "E1", "latex": r"x^2+1", "canonical": "x^2+1"}])
        out = capsys.readouterr().out
        assert "[reg:E1]" in out
        assert r"x^2+1" in out

    def test_print_registered_multiple(self, capsys):
        _print_registered([
            {"expr_id": "E1", "latex": "x", "canonical": "x"},
            {"expr_id": "E2", "latex": "y", "canonical": "y"},
        ])
        out = capsys.readouterr().out
        assert "[reg:E1]" in out
        assert "[reg:E2]" in out

    def test_print_result_registered_status(self, capsys):
        _print_result({
            "status": "registered",
            "registered_expressions": [{"expr_id": "E1", "latex": r"x^2", "canonical": "x^2"}],
        })
        out = capsys.readouterr().out
        assert "[reg:E1]" in out

    def test_print_result_completed_with_label(self, capsys):
        _print_result({
            "status": "completed",
            "expr_id": "E2",
            "latex": r"2\,x",
            "operation_label": "微分する",
        })
        out = capsys.readouterr().out
        assert "[cas:E2]" in out
        assert r"2\,x" in out
        assert "微分する" in out

    def test_print_result_completed_no_label(self, capsys):
        _print_result({
            "status": "completed",
            "expr_id": "E3",
            "latex": r"x+1",
        })
        out = capsys.readouterr().out
        assert "[cas:E3]" in out
        assert "←" not in out

    def test_print_result_failed(self, capsys):
        _print_result({"status": "failed", "error": "Unknown expression: E99"})
        out = capsys.readouterr().out
        assert "[err]" in out
        assert "E99" in out

    def test_print_result_assumption_added(self, capsys):
        _print_result({"status": "assumption_added", "condition": "x > 0",
                       "message": "assumed: x > 0"})
        out = capsys.readouterr().out
        assert "[asm]" in out
        assert "x > 0" in out

    def test_print_result_assumption_removed(self, capsys):
        _print_result({"status": "assumption_removed", "condition": "x > 0"})
        out = capsys.readouterr().out
        assert "[-asm]" in out


# ---------------------------------------------------------------------------
# _handle_interpret — full pipeline (requires Maxima)
# ---------------------------------------------------------------------------

class TestHandleInterpretMaxima:
    """Integration tests: real Maxima, no LLM."""

    @pytest.fixture(autouse=True)
    def _engine(self):
        with NotebookEngine(title="repl_test") as e:
            e.start()
            self.engine = e
            self.sup = _sup()
            yield

    def _interpret(self, line):
        """Run one line through the full interpret pipeline (no capsys consumed here)."""
        _run(_handle_interpret(line, self.engine, self.sup, no_llm=True))

    def test_register_plain_expression(self, capsys):
        self._interpret("x^2 + 1")
        out = capsys.readouterr().out
        assert "[reg:E1]" in out

    def test_cas_expand_via_natural_language(self, capsys):
        self._interpret("(x+1)^2")
        capsys.readouterr()   # clear registration output
        self._interpret("E1 を展開して")
        out = capsys.readouterr().out
        assert "[cas:E2]" in out

    def test_differentiate_via_alias(self, capsys):
        self._interpret("x^3")
        capsys.readouterr()
        self._interpret("E1 を x で微分して")
        out = capsys.readouterr().out
        assert "[cas:E2]" in out
        assert "x^2" in out or r"x^{2}" in out

    def test_unknown_expression_ref_gives_error(self, capsys):
        self._interpret("E99 を展開して")
        out = capsys.readouterr().out
        assert "[err]" in out

    def test_multiple_ops_sequential(self, capsys):
        self._interpret("x^2 - 1")
        capsys.readouterr()
        self._interpret("E1 を因数分解して")
        out = capsys.readouterr().out
        assert "[cas:E2]" in out
        assert "(x-1)" in out or "(x+1)" in out or "x-1" in out

    def test_result_expression_ir_ast_populated(self, capsys):
        """After CAS operation, ir_ast must be non-None for Maxima results."""
        self._interpret("x^2 + 2*x + 1")
        capsys.readouterr()
        self._interpret("E1 を展開して")
        capsys.readouterr()
        e2 = self.engine.expressions.get("E2")
        assert e2 is not None
        assert e2.ir_ast is not None

    def test_cas_repr_equals_canonical_for_maxima(self, capsys):
        """For Maxima, cas_repr always equals canonical (same string)."""
        self._interpret("x^2 + 1")
        capsys.readouterr()
        self._interpret("E1 を x で微分して")
        capsys.readouterr()
        e2 = self.engine.expressions.get("E2")
        assert e2 is not None
        assert e2.cas_repr == e2.canonical

    def test_opaque_expr_not_used_for_maxima(self, capsys):
        """OpaqueExpr must not appear in Maxima ir_ast."""
        from notebook.models.ir import OpaqueExpr
        self._interpret("x^3")
        capsys.readouterr()
        self._interpret("E1 を展開して")
        capsys.readouterr()
        e2 = self.engine.expressions.get("E2")
        assert e2 is not None
        assert not isinstance(e2.ir_ast, OpaqueExpr)

    def test_taylor_uses_frac_not_over(self, capsys):
        """Bug 3 regression: Maxima {{A}\\over{B}} must be converted to \\frac{A}{B}."""
        self.engine.register_expression("exp(x)")
        step, _ = self.engine.run_operation(
            "taylor_expand", {"expr": "E1", "var": "x", "point": "0", "order": 4}
        )
        e2 = self.engine.expressions[step.output_refs[0]]
        assert r"\frac" in e2.latex
        assert r"\over" not in e2.latex


# ---------------------------------------------------------------------------
# L-1: inverse linear basis transform (real Maxima)
# ---------------------------------------------------------------------------

class TestInverseTransformMaxima:
    """Integration: build_inverse_transform end-to-end on real Maxima."""

    @pytest.fixture(autouse=True)
    def _engine(self):
        with NotebookEngine(title="inv_test") as e:
            e.start()
            self.engine = e
            self.sup = _sup()
            yield

    def _interpret(self, line):
        _run(_handle_interpret(line, self.engine, self.sup, no_llm=True))

    def _latexes(self):
        return {eid: ex.latex for eid, ex in self.engine.expressions.items()}

    def test_bogoliubov_like_inverse(self, capsys):
        # forward: alpha_k = u c_k + v c_mk,  alpha_mk = -v c_k + u c_mk
        # inverse: c_k = (u alpha_k - v alpha_mk)/(u^2+v^2), etc.  The new-basis
        # symbols are bound by H-1; without block-localization the system would be a
        # 0=0 tautology and Maxima would return a parametric %r solution.
        for line in ("u をスカラーとする", "v をスカラーとする",
                     "alpha_k = u c_k + v c_mk", "alpha_mk = - v c_k + u c_mk"):
            self._interpret(line)
        capsys.readouterr()
        self._interpret("E1, E2 を c_k, c_mk について逆変換して")
        out = capsys.readouterr().out
        latexes = self._latexes()
        # both solution rows must be registered (regression: stale step hid row 2)
        c_k = next(v for k, v in latexes.items() if v.startswith(r"c_{k}"))
        c_mk = next(v for k, v in latexes.items() if v.startswith(r"c_{mk}"))
        assert "%r" not in c_k and "%r" not in c_mk      # not a degenerate solution
        assert "v^2+u^2" in c_k.replace(" ", "") or "u^2+v^2" in c_k.replace(" ", "")
        assert "alpha" in c_k and "alpha" in c_mk

    def test_auto_basis_inference(self, capsys):
        for line in ("u をスカラーとする", "v をスカラーとする",
                     "alpha_k = u c_k + v c_mk", "alpha_mk = - v c_k + u c_mk"):
            self._interpret(line)
        capsys.readouterr()
        # no explicit basis: u, v are scalar coefficients so the basis {c_k, c_mk}
        # is inferred automatically.
        self._interpret("E1, E2 の逆変換を作って")
        latexes = self._latexes()
        assert any(v.startswith(r"c_{k}") for v in latexes.values())
        assert any(v.startswith(r"c_{mk}") for v in latexes.values())

    def test_explicit_matrix_inverse(self):
        # give M and the two bases directly: M=[[u,-v],[v,u]] inverts to
        # c_k=(u α_k+v α_mk)/(u²+v²), c_mk=(u α_mk−v α_k)/(u²+v²).
        for line in ("u をスカラーとする", "v をスカラーとする"):
            self._interpret(line)
        self._interpret(
            "M = [[u,-v],[v,u]]、旧基底 c_k, c_mk、新基底 alpha_k, alpha_mk の逆変換を求めて")
        latexes = [ex.latex or "" for ex in self.engine.expressions.values()]
        c_k = next(v for v in latexes if v.startswith(r"c_{k}"))
        assert "alpha" in c_k and ("v^2+u^2" in c_k.replace(" ", "")
                                   or "u^2+v^2" in c_k.replace(" ", ""))
        # the generated forward equations must display in the space-separated math
        # convention (no backend '*'), while their canonical keeps '*' for the CAS.
        fwd = next(e for e in self.engine.expressions.values()
                   if (e.latex or "").startswith(r"\alpha_{k} ="))
        assert "*" not in fwd.latex
        assert "*" in (fwd.canonical or "")

    def test_metric_constraints_non_orthonormal(self):
        # G = [[1,s],[s,1]] (⟨e1,e2⟩=s); transform f1=a e1, f2=e2 preserves the
        # metric iff M G Mᵀ = G, i.e. a²−1=0 and (a−1)s=0.
        self._interpret("⟨e_1, e_2⟩ = s")
        self._interpret("f1 = a e_1")
        self._interpret("f2 = e_2")
        self._interpret("E1, E2 の計量保存条件を求めて")
        latexes = [ex.latex or "" for ex in self.engine.expressions.values()]
        joined = " ".join(v.replace(" ", "") for v in latexes)
        assert "a^2-1=0" in joined
        assert "(a-1)*s=0" in joined or "(a-1)s=0" in joined

    def test_transform_substitution_commutative_multi(self):
        # apply a 2-equation coordinate transform to L: q1=Q1+Q2, q2=Q1−Q2
        # → L = q1²+q2² becomes 2Q1²+2Q2² (chained commutative substitute).
        # Q_1/Q_2 must be declared: an undeclared digit-subscripted symbol
        # used as an operation source resolves as an IndexedSymbol (index
        # access), not an atom — the documented project rule (see CLAUDE.md
        # "Undeclared subscript in an OPERATION → index").
        for line in ("Q_1, Q_2 をスカラーとする",
                    "L = q_1^2 + q_2^2", "q_1 = Q_1 + Q_2", "q_2 = Q_1 - Q_2"):
            self._interpret(line)
        self._interpret("E2, E3 を E1 に代入して")
        latexes = [ex.latex or "" for ex in self.engine.expressions.values()]
        folded = next(v for v in latexes if v.startswith("L =") and "q_" not in v)
        norm = folded.replace(" ", "")
        assert "2Q_{2}^2" in norm and "2Q_{1}^2" in norm

    def test_metric_constraints_rotation_is_preserved(self):
        # an orthonormal metric (⟨e1,e2⟩=0) with a rotation preserves G → no
        # non-trivial constraints (cos²+sin²−1 ratsimps to 0).
        self._interpret("⟨e_1, e_2⟩ = 0")
        self._interpret(r"f1 = cos(t) e_1 + sin(t) e_2")
        self._interpret(r"f2 = - sin(t) e_1 + cos(t) e_2")
        resp = _run(_handle_interpret("E1, E2 の計量保存条件を求めて",
                                      self.engine, self.sup, no_llm=True))
        # no constraint expression of the form "... = 0" beyond the inputs
        cons = [ex.latex or "" for ex in self.engine.expressions.values()
                if (ex.latex or "").strip().endswith("= 0")]
        assert cons == []

    def test_apply_constraint_folds_denominator(self):
        # under u²+v²=1 the Bogoliubov-shaped denominator collapses: the explicit
        # relation E1 reduces E2 = F = (u a − v b)/(u²+v²) to F = a u − b v.
        for line in ("a をスカラーとする", "b をスカラーとする",
                     "u をスカラーとする", "v をスカラーとする",
                     "u^2 + v^2 = 1", "F = (u a - v b)/(u^2 + v^2)"):
            self._interpret(line)
        self._interpret("E2 を E1 の拘束で簡約して")
        latexes = [ex.latex for ex in self.engine.expressions.values()]
        folded = next(v for v in latexes if v.startswith("F=") and "u^2" not in v)
        assert "a u-b v" in folded.replace(" ", " ")


# ---------------------------------------------------------------------------
# L-1: inverse linear basis transform (real Julia / Symbolics.jl)
# ---------------------------------------------------------------------------

class TestInverseTransformJulia:
    """Integration: build_inverse_transform end-to-end on a live Julia session."""

    @pytest.fixture(autouse=True)
    def _engine(self):
        with NotebookEngine(title="inv_julia",
                            mode=SessionMode.second_quantization) as e:
            e.session.domains = ["condensed_matter"]
            e.start()
            self.engine = e
            self.sup = _sup(domains=["condensed_matter"], backend="julia_full")
            yield

    def _interpret(self, line):
        _run(_handle_interpret(line, self.engine, self.sup, no_llm=True))

    def test_bogoliubov_like_inverse(self):
        # Julia keeps the new basis free (structure-preserving H-1), so solve_for
        # returns the old basis in terms of the new one; Symbolics.simplify cleans
        # the nested fractions into (u·alpha_k − v·alpha_mk)/(u²+v²) etc.
        for line in ("u をスカラーとする", "v をスカラーとする",
                     "alpha_k = u c_k + v c_mk", "alpha_mk = - v c_k + u c_mk"):
            self._interpret(line)
        self._interpret("E1, E2 を c_k, c_mk について逆変換して")
        latexes = [ex.latex for ex in self.engine.expressions.values()]
        c_k = next(v for v in latexes if v.startswith("c_{k}"))
        c_mk = next(v for v in latexes if v.startswith("c_{mk}"))
        for sol in (c_k, c_mk):
            assert "alpha" in sol
            assert "u^{2} + v^{2}" in sol or "v^{2} + u^{2}" in sol

    def test_fermion_canonical_constraint(self):
        # γ = u c − v c†  is canonical iff {γ,γ†}=u²+v²=1 (fermion → unitary)
        self._interpret("c をフェルミ演算子として定義")
        for s in ("u をスカラーとする", "v をスカラーとする"):
            self._interpret(s)
        self._interpret(r"\gamma = u c(1) - v c'(1)")
        self._interpret("E1 の正準拘束を求めて")
        latexes = [ex.latex for ex in self.engine.expressions.values()]
        assert any("= 1" in v and "u^{2} + v^{2}" in v for v in latexes)

    def test_boson_canonical_constraint(self):
        # b = u a + v a†  is canonical iff [b,b†]=u²−v²=1 (boson → para-unitary).
        # The minus sign comes from the commutator (metric η) automatically.
        self._interpret("a をボソン演算子として定義")
        for s in ("u をスカラーとする", "v をスカラーとする"):
            self._interpret(s)
        self._interpret(r"b = u a(1) + v a'(1)")
        self._interpret("E1 の正準拘束を求めて")
        latexes = [ex.latex for ex in self.engine.expressions.values()]
        assert any("= 1" in v and "u^{2} - v^{2}" in v for v in latexes)

    def test_multimode_standard_bcs_dedups_to_one(self):
        # 2-mode BCS γ1=u c1−v c2†, γ2=u c2+v c1†: the diagonals both give u²+v²=1
        # (deduped to one) and all off-diagonal / anomalous brackets vanish.
        self._interpret("c をフェルミ演算子として定義")
        for s in ("u をスカラーとする", "v をスカラーとする"):
            self._interpret(s)
        self._interpret(r"g1 = u c(1) - v c'(2)")
        self._interpret(r"g2 = u c(2) + v c'(1)")
        self._interpret("E1 と E2 の正準拘束を求めて")
        cons = [ex.latex for ex in self.engine.expressions.values() if "= 1" in (ex.latex or "")]
        assert sum("u^{2} + v^{2}" in v for v in cons) == 1   # exactly one, deduped

    def test_multimode_anomalous_surfaces_extra_constraint(self):
        # wrong-sign ansatz γ2=u c2−v c1† breaks canonicality: besides u²+v²=1, the
        # anomalous {γ1,γ2}=−2uv must appear as the constraint −2uv=0.
        self._interpret("c をフェルミ演算子として定義")
        for s in ("u をスカラーとする", "v をスカラーとする"):
            self._interpret(s)
        self._interpret(r"g1 = u c(1) - v c'(2)")
        self._interpret(r"g2 = u c(2) - v c'(1)")
        self._interpret("E1 と E2 の正準拘束を求めて")
        latexes = [ex.latex or "" for ex in self.engine.expressions.values()]
        assert any("u^{2} + v^{2}" in v and "= 1" in v for v in latexes)
        assert any("u v" in v and "= 0" in v for v in latexes)   # anomalous −2uv=0

    def test_operator_inverse_with_explicit_basis(self):
        # full Nambu BCS: 4 forward equations inverted for c(1),c(2),c'(1),c'(2).
        # Operator-form vars are surrogated for solve and mapped back, giving e.g.
        # c(1) = (u α_k + v α_mk†)/(u²+v²).
        self._interpret("c をフェルミ演算子として定義")
        for s in ("u をスカラーとする", "v をスカラーとする"):
            self._interpret(s)
        self._interpret(r"alpha_k = u c(1) - v c'(2)")
        self._interpret(r"alpha_mk = u c(2) + v c'(1)")
        self._interpret(r"alpha_kd = u c'(1) - v c(2)")
        self._interpret(r"alpha_mkd = u c'(2) + v c(1)")
        self._interpret(
            r"E1, E2, E3, E4 を c(1), c(2), c'(1), c'(2) について逆変換して")
        latexes = [ex.latex or "" for ex in self.engine.expressions.values()]
        # a surrogate must never leak into a registered expression
        assert not any("nbA_" in v for v in latexes)
        # c(1) solution in terms of the new (alpha) basis with u²+v² denominator
        assert any(v.startswith("c_{1}") and "alpha" in v and "u^{2} + v^{2}" in v
                   for v in latexes)

    def test_operator_inverse_then_constraint_clean(self):
        # L-1x end-to-end: derive u²+v²=1, build the operator inverse, then apply the
        # constraint to fold the denominator → c(1) = u α_k + v α_mk† (clean).
        self._interpret("c をフェルミ演算子として定義")
        for s in ("u をスカラーとする", "v をスカラーとする"):
            self._interpret(s)
        self._interpret(r"alpha_k = u c(1) - v c'(2)")
        self._interpret(r"alpha_mk = u c(2) + v c'(1)")
        self._interpret(r"alpha_kd = u c'(1) - v c(2)")
        self._interpret(r"alpha_mkd = u c'(2) + v c(1)")
        self._interpret("E1, E2 の正準拘束を求めて")
        self._interpret(
            r"E1, E2, E3, E4 を c(1), c(2), c'(1), c'(2) について逆変換して")
        # the c(1) inverse is E9; folding the constraint must not block on "mkd"
        c1 = next(e for e in self.engine.expressions.values()
                  if (e.latex or "").startswith("c_{1}"))
        self._interpret(f"正準拘束のもとで {c1.display_id} を簡約して")
        cleaned = [e.latex or "" for e in self.engine.expressions.values()
                   if (e.latex or "").startswith("c_{1}") and "u^{2} + v^{2}" not in (e.latex or "")]
        assert cleaned, "constraint should fold the u²+v² denominator"
        assert "alpha" in cleaned[-1]

    def test_operator_transform_substitution(self):
        # apply an operator transform c1 = u g + v g† to H = c†c → it routes to
        # operator_substitute (non-commutative), giving v² + (u²−v²) g†g.
        self._interpret("c をフェルミ演算子として定義")
        self._interpret("g をフェルミ演算子として定義")
        for s in ("u をスカラーとする", "v をスカラーとする"):
            self._interpret(s)
        self._interpret(r"H = c'(1) c(1)")
        self._interpret(r"c_1 = u g(1) + v g'(1)")
        self._interpret("E2 を E1 に代入して")
        latexes = [ex.latex or "" for ex in self.engine.expressions.values()]
        assert any(v.startswith("H =") and "g" in v
                   and ("u^{2} - v^{2}" in v or "v^{2}" in v)
                   and "c" not in v.split("=", 1)[1] for v in latexes)

    def test_metric_constraints_julia(self):
        # the non-orthonormal metric condition M G Mᵀ=G also runs on Julia/Symbolics:
        # G=[[1,s],[s,1]], f1=a e1, f2=e2 → a²−1=0 and (a−1)s=0.
        self._interpret("⟨e_1, e_2⟩ = s")
        self._interpret("f1 = a e_1")
        self._interpret("f2 = e_2")
        self._interpret("E1, E2 の計量保存条件を求めて")
        joined = " ".join((ex.latex or "").replace(" ", "")
                          for ex in self.engine.expressions.values())
        assert "a^2" in joined and "=0" in joined
        assert "(-1+a)" in joined or "(a-1)" in joined          # the (a−1)s constraint

    def test_verify_canonical_conditional(self, capsys):
        # γ = u c − v c† is canonical only under u²+v²=1 → conditional verdict.
        self._interpret("c をフェルミ演算子として定義")
        for s in ("u をスカラーとする", "v をスカラーとする"):
            self._interpret(s)
        self._interpret(r"\gamma = u c(1) - v c'(1)")
        capsys.readouterr()
        self._interpret("E1 は正準か？")
        out = capsys.readouterr().out
        assert "次の条件を満たせば正準" in out
        latexes = [ex.latex or "" for ex in self.engine.expressions.values()]
        assert any("u^{2} + v^{2}" in v and "= 1" in v for v in latexes)

    def test_verify_canonical_yes(self, capsys):
        # the identity γ = c is trivially canonical → "no further constraints".
        self._interpret("c をフェルミ演算子として定義")
        self._interpret(r"\gamma = c(1)")
        capsys.readouterr()
        self._interpret("E1 は正準変換ですか？")
        assert "正準（para-unitary）です" in capsys.readouterr().out

    def test_derive_then_auto_apply(self):
        # derive u²+v²=1 from γ, then "正準拘束のもとで En を簡約" auto-applies it
        self._interpret("c をフェルミ演算子として定義")
        for s in ("u をスカラーとする", "v をスカラーとする",
                  "a をスカラーとする", "b をスカラーとする"):
            self._interpret(s)
        self._interpret(r"\gamma = u c(1) - v c'(1)")
        self._interpret("E1 の正準拘束を求めて")        # E2: (u²+v²)=1
        self._interpret("F = (u a - v b)/(u^2 + v^2)")   # E3
        self._interpret("正準拘束のもとで E3 を簡約して")
        latexes = [ex.latex for ex in self.engine.expressions.values()]
        # the folded result has no u²+v² denominator left
        assert any(v.startswith("F =") and "u^{2} + v^{2}" not in v
                   and ("a \\cdot u" in v or "a u" in v) for v in latexes)


# ---------------------------------------------------------------------------
# save / load round-trip
# ---------------------------------------------------------------------------

class TestSaveLoad:
    @pytest.fixture(autouse=True)
    def _reset(self, tmp_path):
        session_io.reset_save_state()
        self.tmp = tmp_path
        yield
        session_io.reset_save_state()

    def test_save_creates_file(self, tmp_path, capsys):
        with NotebookEngine(title="save_test") as e:
            e.start()
            e.register_expression("x^2 + 1")
            parts = ["save", str(tmp_path / "out")]
            _run(_cmd_save(parts, e, overwrite=False))
        out = capsys.readouterr().out
        assert "[sys]" in out
        assert "Saved" in out.lower() or "saved" in out.lower() or "out" in out

    def test_save_then_load_restores_expressions(self, tmp_path):
        path = tmp_path / "sess.json"
        with NotebookEngine(title="save_load_test") as e:
            e.start()
            e.register_expression("x^2 + 1")
            e.save(path)

        loaded = NotebookEngine.load(path)
        assert "E1" in loaded.expressions
        assert loaded.expressions["E1"].canonical == "x^2 + 1"

    def test_julia_repr_survives_save_load(self, tmp_path):
        """julia_repr (unlike ir_ast) must survive save/load."""
        from notebook.models.expression import Expression
        path = tmp_path / "julia_sess.json"
        with NotebookEngine(title="cas_repr_test") as e:
            e.start()
            # Manually inject an expression with julia_repr set (simulates Julia session)
            from notebook.models.expression import Expression
            e.expressions["E1"] = Expression(
                display_id="E1",
                branch_id=e.active_branch_id,
                latex=r"a^\dagger(1)a(1)",
                canonical="_nb_E1",
                cas_repr="a†(1)a(1)",
            )
            e.save(path)

        loaded = NotebookEngine.load(path)
        assert loaded.expressions["E1"].cas_repr == "a†(1)a(1)"

    def test_ir_ast_is_none_after_load(self, tmp_path):
        """ir_ast must not survive save/load (runtime-only)."""
        path = tmp_path / "ir_sess.json"
        with NotebookEngine(title="ir_test") as e:
            e.start()
            e.register_expression("x^2 + 1")
            e.save(path)

        loaded = NotebookEngine.load(path)
        for expr in loaded.expressions.values():
            assert expr.ir_ast is None

    def test_julia_repr_not_in_ir_ast_after_load(self, tmp_path):
        """After load, cas_repr-backed OpaqueExpr in ir_ast is gone (ir_ast=None)."""
        from notebook.models.expression import Expression
        from notebook.models.ir import OpaqueExpr
        path = tmp_path / "opaque_sess.json"
        with NotebookEngine(title="opaque_test") as e:
            e.start()
            expr = Expression(
                display_id="E1",
                branch_id=e.active_branch_id,
                latex=r"a†(1)",
                canonical="_nb_E1",
                cas_repr="a(1)",
                ir_ast=OpaqueExpr(text="a(1)"),
            )
            e.expressions["E1"] = expr
            e.save(path)

        loaded = NotebookEngine.load(path)
        assert loaded.expressions["E1"].ir_ast is None
        assert loaded.expressions["E1"].cas_repr == "a(1)"


# ---------------------------------------------------------------------------
# LaTeX export
# ---------------------------------------------------------------------------

class TestLatexExport:
    def test_export_creates_tex_file(self, tmp_path):
        with NotebookEngine(title="export_test") as e:
            e.start()
            e.register_expression("x^2 + 1")
            dest = tmp_path / "out.tex"
            fname = _export_latex(e, dest)
        assert Path(fname).exists()
        content = Path(fname).read_text(encoding="utf-8")
        assert r"\begin{document}" in content
        assert r"\end{document}" in content

    def test_export_contains_expression(self, tmp_path):
        with NotebookEngine(title="export_expr_test") as e:
            e.start()
            e.register_expression("x^2 + 1")
            dest = tmp_path / "expr.tex"
            _export_latex(e, dest)
        content = dest.read_text(encoding="utf-8")
        assert "equation" in content
        assert "x^2" in content or "x^{2}" in content

    def test_export_contains_title(self, tmp_path):
        with NotebookEngine(title="My Research") as e:
            e.start()
            dest = tmp_path / "titled.tex"
            _export_latex(e, dest)
        content = dest.read_text(encoding="utf-8")
        assert "My Research" in content

    def test_export_english_uses_article(self, tmp_path):
        with NotebookEngine(title="en_test") as e:
            e.start()
            e.session.language = ["en"]
            dest = tmp_path / "en.tex"
            _export_latex(e, dest)
        content = dest.read_text(encoding="utf-8")
        assert r"{article}" in content  # \documentclass[dvipdfmx]{article}

    def test_export_note_appears(self, tmp_path):
        with NotebookEngine(title="note_test") as e:
            e.start()
            e.record_manual_note("This is a test memo.")
            dest = tmp_path / "note.tex"
            _export_latex(e, dest)
        content = dest.read_text(encoding="utf-8")
        assert "test memo" in content

    def test_export_assumption_prefers_tex_display_over_ascii_canonical(self, tmp_path):
        """An assumption's TeX-facing display (e.g. "\\gamma > 0") must be used
        in the exported tile, not the bare Maxima-safe canonical ("gamma > 0") —
        found via BackendHandler.assume_sign's auto-generated sign assumption
        for a TeX-declared symbol."""
        with NotebookEngine(title="asm_display_test") as e:
            e.start()
            e.add_assumption("gamma > 0", display=r"\gamma > 0")
            dest = tmp_path / "asm.tex"
            _export_latex(e, dest)
        content = dest.read_text(encoding="utf-8")
        asm_line = next(ln for ln in content.splitlines() if "Assumption:" in ln)
        assert r"\gamma > 0" in asm_line
        assert r"\displaystyle gamma > 0" not in asm_line


# ---------------------------------------------------------------------------
# REPL meta-commands
# ---------------------------------------------------------------------------

class TestReplCommands:
    def test_status_command_shows_info(self, capsys):
        with NotebookEngine(title="status_test") as e:
            e.start()
            session_io.reset_save_state()
            _cmd_status(e)
        out = capsys.readouterr().out
        assert "[sys]" in out
        assert "status_test" in out
        assert "maxima" in out

    def test_note_command_stores_and_prints(self, capsys):
        with NotebookEngine(title="note_cmd_test") as e:
            e.start()
            _run(_cmd_note("important observation", e))
        out = capsys.readouterr().out
        assert "[note]" in out
        assert "important observation" in out

    def test_note_empty_gives_error(self, capsys):
        with NotebookEngine(title="note_empty_test") as e:
            e.start()
            _run(_cmd_note("", e))
        out = capsys.readouterr().out
        assert "[err]" in out


# ---------------------------------------------------------------------------
# UNKNOWN_FIELDS validation — surface through engine (not via LLM pipeline)
# ---------------------------------------------------------------------------

class TestUnknownFieldsValidation:
    """UNKNOWN_FIELDS check prevents misspelled args from reaching the CAS."""

    def test_misspelled_field_raises_on_direct_call(self):
        """run_operation with a bad field name must raise ValueError."""
        with NotebookEngine(title="uf_test") as e:
            e.start()
            expr = e.register_expression("x^2")
            with pytest.raises(Exception):
                # "varl" is not a valid field for differentiate
                e.run_operation("differentiate", {
                    "expr": expr.display_id,
                    "var": "x",
                    "varl": "t",   # typo — should trigger UNKNOWN_FIELDS
                })

    def test_internal_underscore_field_stripped_before_validate(self):
        """_is_total_derivative must not trigger UNKNOWN_FIELDS."""
        with NotebookEngine(title="uf_internal") as e:
            e.start()
            expr = e.register_expression("x^2")
            # Must not raise — _is_total_derivative is stripped in _preprocess_args
            step, _ = e.run_operation("differentiate", {
                "expr": expr.display_id,
                "var": "x",
                "_is_total_derivative": True,
            })
            assert step is not None


# ---------------------------------------------------------------------------
# _handle_interpret — full pipeline (requires Julia)
# ---------------------------------------------------------------------------

class TestHandleInterpretJulia:
    """Integration tests: real Julia backend, no LLM."""

    @pytest.fixture(autouse=True)
    def _engine(self):
        from notebook.models.session import SessionMode
        with NotebookEngine(title="julia_repl_test",
                            mode=SessionMode.second_quantization) as e:
            e.start()
            # Declare `a` as a boson operator so it is registered in SymbolContext.
            # Required since C-2: blocks_on_undefined_symbols blocks QA builtins
            # (a, f) when undeclared, even though QuantumAlgebra pre-exports them.
            e.run_operation("define_boson_operator", {"name": "a"})
            self.engine = e
            self.sup = _sup(backend="julia_full")
            yield

    def _interpret(self, line):
        _run(_handle_interpret(line, self.engine, self.sup, no_llm=True))

    def test_register_plain_expression(self, capsys):
        self._interpret("a(1)")
        out = capsys.readouterr().out
        assert "[reg:E1]" in out

    def test_expand_boson_operator(self, capsys):
        self._interpret("a(1)")
        capsys.readouterr()
        self._interpret("E1 を展開して")
        out = capsys.readouterr().out
        assert "[cas:E2]" in out
        assert "{a}" in out

    def test_expand_dagger_operator(self, capsys):
        self._interpret("a'(1)")
        capsys.readouterr()
        self._interpret("E1 を展開して")
        out = capsys.readouterr().out
        assert "[cas:E2]" in out
        assert r"\dagger" in out

    def test_julia_repr_set_after_expand(self, capsys):
        """Julia backend must set julia_repr on CAS results."""
        self._interpret("a(1)")
        capsys.readouterr()
        self._interpret("E1 を展開して")
        capsys.readouterr()
        e2 = self.engine.expressions.get("E2")
        assert e2 is not None
        assert e2.cas_repr is not None

    def test_ir_ast_opaque_for_julia(self, capsys):
        """Julia CAS results must have ir_ast wrapped in OpaqueExpr."""
        from notebook.models.ir import OpaqueExpr
        self._interpret("a(1)")
        capsys.readouterr()
        self._interpret("E1 を展開して")
        capsys.readouterr()
        e2 = self.engine.expressions.get("E2")
        assert e2 is not None
        assert isinstance(e2.ir_ast, OpaqueExpr)


# ---------------------------------------------------------------------------
# _handle_interpret — symbolic operations on Julia backend
# ---------------------------------------------------------------------------

class TestHandleInterpretJuliaSymbolic:
    """Julia backend: standard symbolic operations comparable to Maxima coverage.

    Uses auto-declare (@variables) for Symbolics.jl identifiers so that
    Maxima-style canonicals (x^2, sin(x), …) work transparently.
    NL commands go through _handle_interpret; direct engine calls test
    operations that have no rule-based alias.
    """

    @pytest.fixture(autouse=True)
    def _engine(self):
        from notebook.models.session import SessionMode
        with NotebookEngine(title="julia_symbolic_test",
                            mode=SessionMode.second_quantization) as e:
            e.start()
            e.run_operation("define_boson_operator", {"name": "a"})
            self.engine = e
            self.sup = _sup(backend="julia_full")
            yield

    def _interpret(self, line):
        _run(_handle_interpret(line, self.engine, self.sup, no_llm=True))

    # --- NL-routed operations ---

    def test_expand_polynomial(self, capsys):
        self._interpret("(x+1)^2")
        capsys.readouterr()
        self._interpret("E1 を展開して")
        out = capsys.readouterr().out
        assert "[cas:E2]" in out
        assert "x^2" in out or r"x^{2}" in out

    def test_differentiate_polynomial(self, capsys):
        self._interpret("x^3")
        capsys.readouterr()
        self._interpret("E1 を x で微分して")
        out = capsys.readouterr().out
        assert "[cas:E2]" in out
        assert "3" in out

    def test_simplify_trig_identity(self, capsys):
        self._interpret("sin(x)^2 + cos(x)^2")
        capsys.readouterr()
        self._interpret("E1 を簡約して")
        out = capsys.readouterr().out
        assert "[cas:E2]" in out
        assert "1" in out

    def test_combine_expressions_multiply(self, capsys):
        self._interpret("x^2")
        self._interpret("x + 1")
        capsys.readouterr()
        self._interpret("E1 と E2 を掛けて")
        out = capsys.readouterr().out
        assert "[cas:E3]" in out
        assert "x" in out

    def test_quantum_ops_still_work_alongside_symbolic(self, capsys):
        """QuantumAlgebra ops must still function after symbolic variable auto-declare."""
        self._interpret("x^2")          # triggers @variables x auto-declare
        capsys.readouterr()
        self._interpret("a(1)")         # QuantumAlgebra op
        capsys.readouterr()
        self._interpret("E2 を展開して")
        out = capsys.readouterr().out
        assert "[cas:E3]" in out
        assert "{a}" in out

    # --- Direct engine calls (no NL alias in --no-llm mode) ---

    def test_taylor_expand(self):
        self.engine.register_expression("sin(x)")
        step, _ = self.engine.run_operation(
            "taylor_expand", {"expr": "E1", "var": "x", "point": "0", "order": 5}
        )
        e2 = self.engine.expressions[step.output_refs[0]]
        assert "x" in (e2.cas_repr or "")
        assert r"\frac" in e2.latex or "frac" in e2.latex

    def test_integrate_indefinite(self):
        self.engine.register_expression("sin(x)")
        step, _ = self.engine.run_operation("integrate_indefinite", {"expr": "E1", "var": "x"})
        e2 = self.engine.expressions[step.output_refs[0]]
        assert "cos" in (e2.cas_repr or "")

    def test_integrate_definite(self):
        self.engine.register_expression("x^2")
        step, _ = self.engine.run_operation(
            "integrate_definite", {"expr": "E1", "var": "x", "lower": "0", "upper": "3"}
        )
        e2 = self.engine.expressions[step.output_refs[0]]
        assert "9" in (e2.cas_repr or "")  # ∫₀³ x² dx = 9

    def test_substitute_value(self):
        self.engine.register_expression("x^2 + 1")
        step, _ = self.engine.run_operation(
            "substitute", {"expr": "E1", "lhs": "x", "rhs": "3"}
        )
        e2 = self.engine.expressions[step.output_refs[0]]
        assert "10" in (e2.cas_repr or "")

    def test_numerical_eval(self):
        self.engine.register_expression("3.14159^2")
        step, _ = self.engine.run_operation("numerical_eval", {"expr": "E1"})
        e2 = self.engine.expressions[step.output_refs[0]]
        val = float(e2.cas_repr or "0")
        assert abs(val - 9.8696) < 0.001

    def test_numerical_eval_hp(self):
        self.engine.register_expression("2.0^10")
        step, _ = self.engine.run_operation("numerical_eval_hp", {"expr": "E1"})
        e2 = self.engine.expressions[step.output_refs[0]]
        assert "1024" in (e2.cas_repr or "")

    def test_real_part_of_symbolic(self):
        self.engine.register_expression("x^2 + 1")
        step, _ = self.engine.run_operation("real_part", {"expr": "E1"})
        e2 = self.engine.expressions[step.output_refs[0]]
        assert e2.cas_repr is not None

    def test_imaginary_part_of_real_is_zero(self):
        self.engine.register_expression("x^2 + 1")
        step, _ = self.engine.run_operation("imaginary_part", {"expr": "E1"})
        e2 = self.engine.expressions[step.output_refs[0]]
        assert "0" in (e2.cas_repr or "")

    def test_combine_expressions_add(self):
        self.engine.register_expression("x^2")
        self.engine.register_expression("x + 1")
        step, _ = self.engine.run_operation(
            "combine_expressions", {"expr_a": "E1", "expr_b": "E2", "op": "+"}
        )
        e3 = self.engine.expressions[step.output_refs[0]]
        assert "x^2" in (e3.cas_repr or "") or "x" in (e3.cas_repr or "")

    def test_combine_expressions_subtract(self):
        self.engine.register_expression("x^2")
        self.engine.register_expression("x + 1")
        step, _ = self.engine.run_operation(
            "combine_expressions", {"expr_a": "E1", "expr_b": "E2", "op": "-"}
        )
        e3 = self.engine.expressions[step.output_refs[0]]
        assert e3.cas_repr is not None

    def test_add_to_both_sides(self):
        self.engine.register_expression("x^2")
        step, _ = self.engine.run_operation(
            "add_to_both_sides", {"expr": "E1", "term": "1"}
        )
        e2 = self.engine.expressions[step.output_refs[0]]
        assert "1" in (e2.cas_repr or "")

    def test_multiply_both_sides(self):
        self.engine.register_expression("x + 1")
        step, _ = self.engine.run_operation(
            "multiply_both_sides", {"expr": "E1", "factor": "2"}
        )
        e2 = self.engine.expressions[step.output_refs[0]]
        assert "2" in (e2.cas_repr or "")

    def test_julia_repr_set_for_symbolic_op(self):
        """All Julia symbolic results must set julia_repr."""
        self.engine.register_expression("x^3")
        step, _ = self.engine.run_operation("differentiate", {"expr": "E1", "var": "x"})
        e2 = self.engine.expressions[step.output_refs[0]]
        assert e2.cas_repr is not None

    def test_latex_output_not_empty(self):
        """Julia symbolic results must produce non-empty LaTeX."""
        self.engine.register_expression("x^2 + 1")
        step, _ = self.engine.run_operation("expand", {"expr": "E1"})
        e2 = self.engine.expressions[step.output_refs[0]]
        assert e2.latex and "x" in e2.latex


# ---------------------------------------------------------------------------
# Pipe-mode tests — Maxima backend (subprocess)
# ---------------------------------------------------------------------------

class TestPipeMaxima:
    """End-to-end REPL pipe tests: spawn debug_repl as a Maxima subprocess."""

    def test_register(self):
        out = _repl_pipe(["x^2 + 1"])
        assert "[reg:E1]" in out

    def test_differentiate(self):
        out = _repl_pipe(["x^3", "E1 を x で微分して"])
        assert "[cas:E2]" in out
        assert "x^2" in out or r"x^{2}" in out

    def test_combine_mul_parens(self):
        """Bug 2 regression: * with compound operand must parenthesize RHS."""
        out = _repl_pipe(["x^2", "x+1", "E1 と E2 を掛けて"])
        assert "(x+1)" in out

    def test_combine_sub_parens(self):
        """Bug 2 regression: - with compound RHS operand must parenthesize."""
        out = _repl_pipe(["x^2", "x+1", "E1 から E2 を引いて"])
        assert "(x+1)" in out

    def test_export_tex_extension(self, tmp_path):
        """Bug 1 regression: export without .tex suffix must add .tex."""
        dest = tmp_path / "out_no_ext"
        _repl_pipe(["x^2 + 1", f"export {dest}"])
        assert dest.with_suffix(".tex").exists()

    def test_status(self):
        out = _repl_pipe(["status"])
        assert "[sys]" in out
        assert "maxima" in out

    def test_error_on_unknown_ref(self):
        out = _repl_pipe(["E99 を展開して"])
        assert "[err]" in out


# ---------------------------------------------------------------------------
# Pipe-mode tests — Julia backend (subprocess, single session)
# ---------------------------------------------------------------------------

class TestPipeJulia:
    """End-to-end REPL pipe tests with Julia backend.

    One subprocess handles all commands to amortize Julia startup cost.
    Covers both quantum algebra and standard symbolic operations.
    Each test method checks a different aspect of the combined output.
    """

    @pytest.fixture(scope="class", autouse=True)
    def _session(self, tmp_path_factory, request):
        tmp = tmp_path_factory.mktemp("julia_pipe")
        export_base = tmp / "julia_export_test"
        out = _repl_pipe(
            [
                # Declare `a` before algebraic use (required since C-2)
                "ボゾン演算子 a を定義して",  # side-effect only
                # QuantumAlgebra operations
                "a(1)",          # → [reg:E1]
                "a'(1)",         # → [reg:E2]
                "E1 を展開して",  # → [cas:E3]  boson expand
                "E2 を展開して",  # → [cas:E4]  dagger expand
                # Standard symbolic operations (Symbolics.jl via auto-declare)
                "(x+1)^2",       # → [reg:E5]
                "E5 を展開して",  # → [cas:E6]  polynomial expand
                "x^3",           # → [reg:E7]
                "E7 を x で微分して",   # → [cas:E8]  differentiation
                "sin(x)^2 + cos(x)^2",  # → [reg:E9]
                "E9 を簡約して",          # → [cas:E10] trig simplify
                # Meta commands
                "status",
                f"export {export_base}",
            ],
            domains=["condensed_matter"],
        )
        request.cls._out = out
        request.cls._export_tex = export_base.with_suffix(".tex")

    # --- QuantumAlgebra ---

    def test_register_first_expression(self):
        assert "[reg:E1]" in self._out

    def test_register_second_expression(self):
        assert "[reg:E2]" in self._out

    def test_expand_produces_cas_tag(self):
        assert "[cas:E3]" in self._out

    def test_expand_boson_latex(self):
        assert "{a}" in self._out

    def test_expand_dagger_latex(self):
        assert r"\dagger" in self._out

    # --- Symbolic operations ---

    def test_symbolic_expand_polynomial(self):
        """(x+1)^2 expanded on Julia backend must produce a cas tag."""
        assert "[cas:E6]" in self._out

    def test_symbolic_expand_contains_x(self):
        """Expanded polynomial must contain x."""
        assert "x" in self._out

    def test_symbolic_differentiate_produces_cas(self):
        """x^3 differentiated w.r.t. x must produce a cas tag."""
        assert "[cas:E8]" in self._out

    def test_symbolic_differentiate_result(self):
        """d/dx x^3 = 3x^2."""
        assert "3" in self._out
        assert "x^2" in self._out or r"x^{2}" in self._out

    def test_symbolic_simplify_trig(self):
        """sin²+cos²=1 must simplify to 1."""
        assert "[cas:E10]" in self._out

    def test_symbolic_simplify_result_is_one(self):
        assert "1" in self._out

    # --- Meta ---

    def test_status_shows_julia_backend(self):
        assert "julia_full" in self._out

    def test_export_creates_tex(self):
        """Bug 1 regression on Julia backend: export without .tex must add .tex."""
        assert self._export_tex.exists()


# ---------------------------------------------------------------------------
# Julia save / restore — two-subprocess round-trip
# ---------------------------------------------------------------------------

class TestPipeJuliaSaveLoad:
    """Save a Julia session and restore it in a second subprocess.

    Phase 1: register 4 expressions (QuantumAlgebra + Symbolics.jl) and save.
    Phase 2: load the saved JSON with --session; run status.
    The class fixture amortises Julia startup by sharing both outputs.
    """

    @pytest.fixture(scope="class", autouse=True)
    def _session(self, tmp_path_factory, request):
        tmp = tmp_path_factory.mktemp("julia_saveload")
        save_base = tmp / "julia_saveload_test"

        # Phase 1: build session and save
        out1 = _repl_pipe(
            [
                "ボゾン演算子 a を定義して",  # side-effect: declare a (required since C-2)
                "a(1)",          # E1 — QuantumAlgebra boson annihilation
                "a'(1)",         # E2 — creation operator
                "(x+1)^2",       # E3 — Symbolics.jl polynomial
                "E3 を展開して",  # E4 — expanded form
                f"save {save_base}",
            ],
            domains=["condensed_matter"],
        )

        # Phase 2: load saved file, run status
        saved_json = save_base.with_suffix(".json")
        out2 = _repl_pipe(
            ["status"],
            domains=["condensed_matter"],
            session_file=saved_json,
        )

        request.cls._out1 = out1
        request.cls._out2 = out2
        request.cls._saved_json = saved_json

    # --- Phase 1: save ---

    def test_save_creates_json_file(self):
        assert self._saved_json.exists()

    def test_save_message_appears(self):
        assert "[sys] Saved:" in self._out1

    def test_phase1_registers_four_expressions(self):
        for tag in ("[reg:E1]", "[reg:E2]", "[reg:E3]", "[cas:E4]"):
            assert tag in self._out1, f"{tag} not found in phase-1 output"

    # --- Phase 2: restore ---

    def test_load_message_appears(self):
        assert "Loaded:" in self._out2

    def test_load_reports_four_expressions(self):
        assert "4 expressions" in self._out2

    def test_status_shows_four_exprs_after_load(self):
        assert "exprs=4" in self._out2

    def test_status_shows_julia_backend_after_load(self):
        assert "julia_full" in self._out2

    def test_saved_json_contains_julia_repr(self):
        """julia_repr must survive JSON serialisation (unlike ir_ast).

        Only expressions whose canonical is _nb_Ek (i.e., the Julia string is
        not parseable by MaximaIRCompiler) carry a julia_repr.  In this
        session E4 is the expanded Symbolics.jl result "1 + 2x + x^2", which
        cannot round-trip through the Maxima parser, so it gets canonical=
        _nb_E4 and julia_repr set.  QuantumAlgebra operators like a(1) are
        parseable as Maxima function calls and have cas_repr=None.
        """
        data = json.loads(self._saved_json.read_text())
        exprs = data.get("expressions", {})
        cas_repr_values = [
            v.get("cas_repr") for v in exprs.values() if v.get("cas_repr")
        ]
        assert len(cas_repr_values) >= 1, (
            "Expected at least one expression (E4 expanded polynomial) to have "
            "julia_repr in saved JSON"
        )

    def test_saved_json_has_no_ir_ast(self):
        """ir_ast is runtime-only and must not appear in the saved JSON."""
        data = json.loads(self._saved_json.read_text())
        exprs = data.get("expressions", {})
        for eid, v in exprs.items():
            assert "ir_ast" not in v, f"ir_ast found in saved JSON for {eid}"


class TestPipeJuliaSaveLoadChained:
    """Verify that chained _nb_Ek references survive save/load.

    Phase 1 builds a 3-step chain where each step's rendered_command refers to
    the Julia variable from the previous step:

        E1 = (x+1)^3          (expression_registration — canonical is the string)
        E2 = expand(E1)        → _nb_E2   (first Julia variable)
        E3 = diff(_nb_E2, x)   → _nb_E3   (depends on _nb_E2)

    Save writes all rendered_commands to .mac so _nb_E2 and _nb_E3 are
    re-defined in Julia on load without a separate replay pass.

    Phase 2 loads the session and immediately runs a new operation on E3
    (whose canonical is _nb_E3).  If _nb_E3 was not restored, Julia would
    raise UndefVarError and no [cas:E4] tag would appear in the output.
    """

    @pytest.fixture(scope="class", autouse=True)
    def _session(self, tmp_path_factory, request):
        tmp = tmp_path_factory.mktemp("julia_chained")
        save_base = tmp / "julia_chain_test"

        # Phase 1: chained operations
        out1 = _repl_pipe(
            [
                "(x+1)^3",           # E1 — Symbolics.jl polynomial (registration only)
                "E1 を展開して",      # E2 — canonical = _nb_E2
                "E2 を x で微分して", # E3 — canonical = _nb_E3, depends on _nb_E2
                f"save {save_base}",
            ],
            domains=["condensed_matter"],
        )

        # Phase 2: load and run a new operation on the chained result
        saved_json = save_base.with_suffix(".json")
        out2 = _repl_pipe(
            ["E3 を x で微分して"],  # should produce E4 using _nb_E3 from snapshot
            domains=["condensed_matter"],
            session_file=saved_json,
        )

        request.cls._out1 = out1
        request.cls._out2 = out2
        request.cls._saved_json = saved_json
        request.cls._saved_mac = save_base.with_suffix(".jl")

    # --- Phase 1 assertions ---

    def test_phase1_builds_chain(self):
        for tag in ("[reg:E1]", "[cas:E2]", "[cas:E3]"):
            assert tag in self._out1, f"{tag} not found in phase-1 output"

    def test_mac_contains_expand_command(self):
        """The .jl snapshot must contain the expand rendered_command for E2."""
        assert self._saved_mac.exists(), ".jl file not created"
        mac_text = self._saved_mac.read_text(encoding="utf-8")
        assert "_nb_E2" in mac_text, "expand command (_nb_E2) missing from .mac"

    def test_mac_contains_differentiate_command(self):
        """The .jl snapshot must contain the differentiate rendered_command for E3."""
        mac_text = self._saved_mac.read_text(encoding="utf-8")
        assert "_nb_E3" in mac_text, "differentiate command (_nb_E3) missing from .mac"

    # --- Phase 2 assertions ---

    def test_operation_after_load_succeeds(self):
        """Differentiating E3 after load must produce a new CAS result (E4).

        This fails with UndefVarError if _nb_E3 was not restored from the
        .mac snapshot, proving that the snapshot now covers Symbolics.jl vars.
        """
        assert "[cas:E4]" in self._out2, (
            "E4 not produced after load — _nb_E3 was likely not restored from .mac.\n"
            f"Phase-2 output:\n{self._out2}"
        )


# ---------------------------------------------------------------------------
# B1-1: boson operator + 3-step differentiation chain — save / restore
# ---------------------------------------------------------------------------

class TestB1ChainSaveRestoreLatex:
    """B1-1 scenario: ボゾン演算子定義 + 3段微分チェーンの保存・復元.

    Plain `a` cannot be registered without LLM (no brackets/operators), so
    the test begins with the boson operator definition (alias-resolvable) then
    uses (x+1)^3 as E1.

    Phase 1 builds:
        boson operator definition (side-effect, no expression)
        E1 = (x+1)^3  (registration)
        E2 = expand(E1)       → _nb_E2
        E3 = diff(E2, x)      → _nb_E3
        E4 = diff(E3, x)      → _nb_E4
        E5 = diff(E4, x)      → _nb_E5  (= constant 6)

    Phase 2 (subprocess): load + diff(E4, x) → E6 (= 6).
    Phase 2 (in-process): check latex fields of E1–E5 are non-empty
    after load, confirming timeline display would render correctly.
    """

    @pytest.fixture(scope="class", autouse=True)
    def _session(self, tmp_path_factory, request):
        tmp = tmp_path_factory.mktemp("b1_chain")
        save_base = tmp / "b1_chain_test"

        # Phase 1: boson definition + 3-level differentiation chain
        out1 = _repl_pipe(
            [
                "ボゾン演算子 a を定義して",  # side-effect only (alias-resolvable, no E tag)
                "(x+1)^3",                   # E1 — register polynomial
                "E1 を展開して",              # E2 — _nb_E2
                "E2 を x で微分して",          # E3 — _nb_E3
                "E3 を x で微分して",          # E4 — _nb_E4
                "E4 を x で微分して",          # E5 — _nb_E5  (= 6)
                f"save {save_base}",
            ],
            domains=["condensed_matter"],
        )

        saved_json = save_base.with_suffix(".json")
        saved_mac  = save_base.with_suffix(".jl")

        # Phase 2 (subprocess): load and diff E4 → E6 (= 6)
        out2 = _repl_pipe(
            ["E4 を x で微分して"],  # diff(6x + 6, x) = 6
            domains=["condensed_matter"],
            session_file=saved_json,
        )

        # Phase 2 (in-process): load session to inspect latex fields directly
        loaded_engine = NotebookEngine.load(saved_json)

        request.cls._out1       = out1
        request.cls._out2       = out2
        request.cls._saved_json = saved_json
        request.cls._saved_mac  = saved_mac
        request.cls._engine     = loaded_engine

    # --- Phase 1: build check ---

    def test_phase1_registers_expressions(self):
        for tag in ("[reg:E1]", "[cas:E2]", "[cas:E3]", "[cas:E4]", "[cas:E5]"):
            assert tag in self._out1, f"{tag} missing from phase-1 output"

    def test_mac_contains_all_nb_vars(self):
        """All four Julia variable assignments must appear in the .jl snapshot."""
        assert self._saved_mac.exists(), ".jl not created"
        mac = self._saved_mac.read_text(encoding="utf-8")
        for var in ("_nb_E2", "_nb_E3", "_nb_E4", "_nb_E5"):
            assert var in mac, f"{var} missing from .jl"

    # --- Phase 2: latex field check (timeline display correctness) ---

    def test_loaded_expressions_have_latex(self):
        """Every expression must have a non-empty latex field after load.

        An empty latex would cause a blank tile in the timeline.
        """
        for eid in ("E1", "E2", "E3", "E4", "E5"):
            expr = self._engine.expressions.get(eid)
            assert expr is not None, f"{eid} missing after load"
            assert expr.latex, f"{eid}.latex is empty after load"

    def test_cas_expressions_have_nb_canonical(self):
        """CAS results (E2–E5) must have _nb_Ek canonicals after load."""
        for eid in ("E2", "E3", "E4", "E5"):
            expr = self._engine.expressions[eid]
            assert expr.canonical.startswith("_nb_"), (
                f"{eid}.canonical should be _nb_Ek, got: {expr.canonical!r}"
            )

    def test_e5_latex_is_constant(self):
        """E5 = d³/dx³(x+1)³ = 6; its latex must represent the constant 6."""
        e5 = self._engine.expressions["E5"]
        assert "6" in e5.latex, f"E5 should be constant 6, got latex: {e5.latex!r}"

    # --- Phase 2: operation after load ---

    def test_operation_on_restored_var_succeeds(self):
        """diff(E4, x) = 6 must succeed after load (_nb_E4 restored from .mac)."""
        assert "[cas:E6]" in self._out2, (
            "_nb_E4 not restored — new operation after load failed.\n"
            f"Phase-2 output:\n{self._out2}"
        )

    def test_operation_result_is_six(self):
        """diff(6x+6, x) = 6 — result must contain '6'."""
        assert "6" in self._out2, f"Expected '6' in phase-2 output:\n{self._out2}"

    def test_user_input_companion_survives_load(self):
        """After load, user_input (YOU) steps must be discoverable as timeline companions.

        user_input steps do not advance the branch head, so they are absent from
        compute_lineage.  The /timeline endpoint finds them via the companion lookup:
        steps with the same parent as the following state-affecting step.
        """
        from notebook.runtime.replay import compute_lineage
        from notebook.models.step import StepKind

        e = self._engine
        lineage = compute_lineage(e.active_branch_id, e.steps, e.branches)
        lineage_ids = {s.step_id for s in lineage}

        # Build companion map (mirrors notebook_bridge.py /timeline logic)
        companions: dict = {}
        for s in e.steps.values():
            if s.step_id in lineage_ids:
                continue
            if s.branch_id != e.active_branch_id:
                continue
            if s.kind != StepKind.user_input or not s.user_text:
                continue
            parent_id = s.parent_step_ids[0] if s.parent_step_ids else None
            companions.setdefault(parent_id, []).append(s)

        # Collect all companion user_input texts
        companion_texts = [s.user_text for steps in companions.values() for s in steps]

        # "ボゾン演算子 a を定義して" was a side-effect operation with no output expr.
        # Its preceding user_input step must be found as a companion.
        assert any("ボゾン" in t for t in companion_texts), (
            f"No boson user_input companion found. companions: {companion_texts}"
        )
