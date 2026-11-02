"""Multi-symbol declarations must show ALL names, not just the first.

"R、L、w、C をスカラーとする" declares four scalars; the declaration card and the
TeX export must render "R, L, w, C", not "R".  The TeX export had its own
first-name-only logic (it took _raw_names[0]); the persisted cas_operation step
carries registered_names with name = first only.
"""
import asyncio
import pathlib

from notebook.engine import NotebookEngine
from ui import notebook_bridge as nb
from ui.interpret_pipeline import run_interpret_pipeline
from ui.notebook_bridge import InterpretRequest, InputToken
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operation_validator import OperationValidator


def test_void_display_latex_lists_all_registered_names():
    e = NotebookEngine(title="t")
    e.start()
    h = e._handler
    # persisted cas_operation step carries registered_names (name = first only)
    assert h.get_void_op_display_latex(
        "define_scalar_parameter",
        {"name": "R", "registered_names": ["R", "L", "w", "C"]}) == "R, L, w, C"
    # single declaration unchanged (and Greek mapped)
    assert h.get_void_op_display_latex(
        "define_scalar_parameter", {"name": "xi"}) == r"\xi"


def test_tex_export_lists_all_declared_scalars():
    async def _run():
        nb.start_engine(title="multiscalar_tex", mode="standard", language="ja",
                        domains=[])
        try:
            eng = nb.get_engine()
            sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                                domains=[], primary_backend="maxima")
            await run_interpret_pipeline(
                InterpretRequest(tokens=[InputToken(type="text",
                                                    value="R、L、w、Cをスカラーとする")]),
                eng, sup)
            res = await nb.export_latex()
            return pathlib.Path(res["path"]).read_text(encoding="utf-8")
        finally:
            nb.stop_engine()

    tex = asyncio.get_event_loop().run_until_complete(_run())
    assert "R, L, w, C" in tex, tex


def test_scalar_named_with_spin_component_suffix_not_misrendered_as_superscript():
    """J-7 (scenario-narrative-notes review, 2026-07-08): a plain scalar whose
    name happens to end in x/y/z/p/m (e.g. "mz", the mean-field magnetization)
    must render as the literal name, not be reinterpreted as a spin-operator
    component (S^z-style superscript) just because of its trailing letter.
    The heuristic in _canonical_name_to_latex must be gated on the name (and
    its stem) actually being registered with kind=="operator" — i.e. a real
    define_spin_operator component — not on name shape alone.
    """
    e = NotebookEngine(title="t2")
    e.start()
    h = e._handler
    # No registration at all: falls back to the literal name.
    assert h._canonical_name_to_latex("mz") == "mz"
    # Registered as a plain scalar (kind="scalar"): still the literal name.
    from notebook.models.symbol_context import Quantity
    e.symbol_context.register("mz", quantity=Quantity(name="mz", kind="scalar"))
    assert h._canonical_name_to_latex("mz") == "mz"


def test_genuine_spin_component_still_renders_as_superscript():
    async def _run():
        nb.start_engine(title="spin_suffix_ok", mode="second_quantization",
                        language="en", domains=["spin_systems"])
        try:
            eng = nb.get_engine()
            sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                                domains=["spin_systems"],
                                primary_backend=eng.session.primary_backend)
            await run_interpret_pipeline(
                InterpretRequest(tokens=[InputToken(
                    type="text", value="S as the spin operator")]),
                eng, sup)
            return eng._handler._canonical_name_to_latex("Sz")
        finally:
            nb.stop_engine()

    assert asyncio.get_event_loop().run_until_complete(_run()) == "S^{z}"
