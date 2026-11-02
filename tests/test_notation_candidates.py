"""R-8: notation candidates + confirmation + notation_usage wiring.

Two ambiguous tokens get the candidate/confirmation treatment (minimal scope
per docs/archive/audit/fable_implementation_roadmap.md R-8):

  |X|            → absolute value vs determinant
  undeclared Δ   → Laplacian vs symbol (fires only when no domain narrows it)

Resolution priority: declaration > domain > notation_usage prior record >
candidate list.  The chosen role is written to SymbolContext.notation_usage
by the `resolve_notation` pseudo-operation, persists with the session, and
suppresses re-confirmation — the wiring CLAUDE.md described but which the
2026-07-03 audit found unimplemented (finding AR-2).
"""
import asyncio

import pytest

from notebook.engine import NotebookEngine
from notebook.models.quantity import MathType
from notebook.models.step import StepKind
from ui.interpret_pipeline import InputToken, _try_notation_candidates
from ui.notebook_bridge import OperationRequest, _handle_side_effect_operation


def _run_nc(engine, text, tokens=None, lang="ja"):
    async def _inner():
        loop = asyncio.get_event_loop()
        toks = tokens if tokens is not None else [InputToken(type="text", value=text)]
        return await _try_notation_candidates(engine, loop, lang, text, toks, [])
    return asyncio.run(_inner())


def _run_bridge(engine, req):
    async def _inner():
        loop = asyncio.get_event_loop()
        return await _handle_side_effect_operation(req, engine, loop)
    return asyncio.run(_inner())


@pytest.fixture
def engine():
    return NotebookEngine(title="notation candidates")


# ---------------------------------------------------------------------------
# |X| : abs vs determinant
# ---------------------------------------------------------------------------

class TestPipeCandidates:

    def test_undeclared_inner_asks(self, engine):
        res, _ = _run_nc(engine, "|A|")
        assert res is not None and res["status"] == "needs_confirmation"
        roles = [c["args"]["role"] for c in res["candidates"]]
        assert roles == ["abs", "determinant"]
        assert all(c["operation"] == "resolve_notation" for c in res["candidates"])
        assert all(c["display_ja"] and c["display_en"] for c in res["candidates"])

    def test_prompt_is_translated(self, engine):
        res_ja, _ = _run_nc(engine, "|A|", lang="ja")
        res_en, _ = _run_nc(engine, "|A|", lang="en")
        assert "解釈" in res_ja["confirmation_prompt"]
        assert "ambiguous" in res_en["confirmation_prompt"]

    def test_declared_matrix_resolves_to_determinant_silently(self, engine):
        engine.register_quantity("A", kind="anonymous",
                                 math_type=MathType(algebra="matrix"))
        res, raw = _run_nc(engine, "|A|")
        assert res is None
        assert raw == "determinant(A)"

    def test_declared_scalar_resolves_to_abs_silently(self, engine):
        engine.register_quantity("x", kind="scalar")
        res, raw = _run_nc(engine, "|x|")
        assert res is None
        assert raw == "abs(x)"

    def test_prior_choice_applies_silently(self, engine):
        engine.symbol_context.notation_usage["|·|"] = "abs"
        res, raw = _run_nc(engine, "|y + 1|")
        assert res is None
        assert raw == "abs(y + 1)"

    def test_math_token_keeps_display_latex(self, engine):
        """The canonical is rewritten; the latex (what the user sees) stays |A|."""
        engine.symbol_context.notation_usage["|·|"] = "abs"
        tok = InputToken(type="math", latex="|A|", canonical="|A|")
        res, _ = _run_nc(engine, "|A|", tokens=[tok])
        assert res is None
        assert tok.canonical == "abs(A)"
        assert tok.latex == "|A|"

    def test_nl_instruction_does_not_fire_jp(self, engine):
        res, raw = _run_nc(engine, "E1 を展開して", lang="ja")
        assert res is None and raw == "E1 を展開して"

    def test_nl_instruction_does_not_fire_en(self, engine):
        res, raw = _run_nc(engine, "Expand E1 fully", lang="en")
        assert res is None and raw == "Expand E1 fully"

    def test_multiple_pairs_now_ask_once_for_all_of_them(self, engine):
        """Was: "out of the minimal scope — R-3's loud error remains".

        `|a| + |b|` is ordinary mathematics, and being outside the single-pair
        regex meant it was refused with no chance to confirm — the substance of
        open_items ABS-PIPE-RESIDUAL, closed 2026-08-18.  ONE decision covers
        every pair: the bars in one expression are the same notation, so asking
        per pair would ask the same question twice.  Shapes and the norm/bra-ket
        cases are pinned in tests/test_pipe_notation_shapes.py.
        """
        res, _ = _run_nc(engine, "|a| + |b|")
        assert res is not None and res["status"] == "needs_confirmation"
        assert res["candidates"][0]["args"]["rewritten"] == "abs(a) + abs(b)"


# ---------------------------------------------------------------------------
# Undeclared Δ : Laplacian vs symbol
# ---------------------------------------------------------------------------

class TestDeltaCandidates:

    def test_no_domain_undeclared_asks(self, engine):
        res, _ = _run_nc(engine, "Delta f")
        assert res is not None and res["status"] == "needs_confirmation"
        roles = [c["args"]["role"] for c in res["candidates"]]
        assert roles == ["laplacian", "symbol"]
        assert res["candidates"][1]["args"]["declare"] == "Delta"

    def test_domain_narrows_no_dialog(self, engine):
        engine.session.domains = ["electromagnetism"]
        res, _ = _run_nc(engine, "Delta f")
        assert res is None  # existing resolver behavior stands

    def test_declared_delta_no_dialog(self, engine):
        engine.register_quantity("Delta", kind="scalar")
        res, _ = _run_nc(engine, "Delta f")
        assert res is None

    def test_prior_laplacian_choice_silent(self, engine):
        engine.symbol_context.notation_usage["Delta"] = "laplacian"
        res, _ = _run_nc(engine, "Delta g")
        assert res is None

    def test_prior_symbol_choice_restores_declaration(self, engine):
        engine.symbol_context.notation_usage["Delta"] = "symbol"
        res, _ = _run_nc(engine, "Delta g")
        assert res is None
        assert engine.symbol_context.get("Delta") is not None


# ---------------------------------------------------------------------------
# resolve_notation pseudo-operation (bridge side)
# ---------------------------------------------------------------------------

class TestResolveNotationBridge:

    def test_abs_choice_registers_and_records(self, engine):
        req = OperationRequest(
            operation="resolve_notation",
            args={"token": "|·|", "role": "abs",
                  "rewritten": "abs(A)", "latex": "|A|"},
            user_text="|A|",
            candidates=[{"operation": "resolve_notation", "args": {"role": "abs"}}],
        )
        resp = _run_bridge(engine, req)
        assert resp is not None and resp["ok"]
        assert engine.symbol_context.notation_usage["|·|"] == "abs"
        expr = engine.expressions[resp["expr_id"]]
        assert expr.canonical == "abs(A)"
        assert expr.latex == "|A|"  # display keeps the user's notation
        kinds = {s.kind for s in engine.steps.values()}
        assert StepKind.ambiguity_resolution in kinds

    def test_delta_symbol_choice_declares_scalar(self, engine):
        req = OperationRequest(
            operation="resolve_notation",
            args={"token": "Delta", "role": "symbol",
                  "rewritten": "Delta*f", "latex": "", "declare": "Delta"},
        )
        resp = _run_bridge(engine, req)
        assert resp is not None and resp["ok"]
        assert engine.symbol_context.notation_usage["Delta"] == "symbol"
        prop = engine.symbol_context.get("Delta")
        assert prop is not None and prop.quantity.kind == "scalar"

    def test_choice_survives_save_load_and_stays_silent(self, engine, tmp_path):
        """The remembered role must suppress the dialog after reload too."""
        req = OperationRequest(
            operation="resolve_notation",
            args={"token": "|·|", "role": "determinant",
                  "rewritten": "determinant(M)", "latex": "|M|"},
        )
        assert _run_bridge(engine, req)["ok"]
        path = tmp_path / "notation.json"
        engine.save(path)
        loaded = NotebookEngine.load(path)
        res, raw = _run_nc(loaded, "|N|")
        assert res is None
        assert raw == "determinant(N)"

    def test_missing_fields_rejected(self, engine):
        req = OperationRequest(operation="resolve_notation",
                               args={"token": "|·|"})
        resp = _run_bridge(engine, req)
        assert resp is not None and not resp["ok"]
