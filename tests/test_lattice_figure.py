"""Tests for lattice figure generation (F-1).

Confirms a generated lattice can be rendered to TikZ (for papers) and SVG (for
viewing), and saved to disk — so the user can verify the lattice is the shape
they intended.
"""
import math

import pytest

from notebook.runtime.lattice_figure import (
    lattice_layout, lattice_to_tikz, lattice_to_svg,
    lattice_to_standalone_tex, save_lattice_figure, _find_latex_engine,
)


class TestLayout:
    def test_square_positions_and_bonds(self):
        pos, bonds = lattice_layout("square_2d", 9, "open", "3x3")
        assert len(pos) == 9
        assert pos[1] == (0.0, 0.0) and pos[5] == (1.0, 1.0)
        assert len(bonds) == 12  # 3x3 open square

    def test_triangular_is_sheared(self):
        pos, _ = lattice_layout("triangular_2d", 9, "open", "3x3")
        # site 4 = grid (0,1) → sheared x=0.5, y=√3/2
        assert pos[4][0] == pytest.approx(0.5)
        assert pos[4][1] == pytest.approx(math.sqrt(3) / 2)

    def test_all_pairs_on_circle(self):
        pos, bonds = lattice_layout("all_pairs", 4, "open")
        # 4 points on the unit circle, all 6 chords
        for p in pos.values():
            assert math.hypot(*p) == pytest.approx(1.0)
        assert len(bonds) == 6

    def test_chain_on_a_line(self):
        pos, bonds = lattice_layout("nearest_neighbor", 4, "open")
        assert all(y == 0.0 for _, y in pos.values())
        assert len(bonds) == 3

    def test_honeycomb_grid_layout(self):
        pos, bonds = lattice_layout("honeycomb_2d", 9, "open", "3x3")
        assert len(pos) == 9 and len(bonds) >= 1

    def test_custom_on_circle(self):
        pos, bonds = lattice_layout("custom", None, edges="(1,2),(2,3),(1,3)")
        assert len(pos) == 3 and len(bonds) == 3
        for p in pos.values():
            assert math.hypot(*p) == pytest.approx(1.0)

    def test_custom_without_edges_raises(self):
        with pytest.raises(ValueError):
            lattice_layout("custom", None)

    def test_unknown_constraint_raises(self):
        with pytest.raises(ValueError):
            lattice_layout("bogus", 4)

    def test_bad_shape_raises(self):
        with pytest.raises(ValueError):
            lattice_layout("square_2d", 9, "open", "3-3")


class TestTikz:
    def test_tikz_has_nodes_and_bonds(self):
        pos, bonds = lattice_layout("square_2d", 4, "open", "2x2")
        tikz = lattice_to_tikz(pos, bonds)
        assert tikz.startswith("\\begin{tikzpicture}")
        assert tikz.count("\\node[site]") == 4
        assert tikz.count("\\draw") == len(bonds)

    def test_periodic_wrap_is_dashed(self):
        pos, bonds = lattice_layout("square_2d", 9, "periodic", "3x3")
        assert "[dashed]" in lattice_to_tikz(pos, bonds)

    def test_standalone_wraps_tikz(self):
        pos, bonds = lattice_layout("nearest_neighbor", 3, "open")
        doc = lattice_to_standalone_tex(lattice_to_tikz(pos, bonds))
        assert "\\documentclass[tikz,border=4pt]{standalone}" in doc
        assert "\\begin{document}" in doc and "\\end{document}" in doc


class TestSvg:
    def test_svg_circle_and_line_counts(self):
        pos, bonds = lattice_layout("square_2d", 4, "open", "2x2")
        svg = lattice_to_svg(pos, bonds)
        assert svg.startswith("<svg xmlns=")
        assert svg.count("<circle") == 4
        assert svg.count("<line") == len(bonds)

    def test_svg_periodic_wrap_dashed(self):
        pos, bonds = lattice_layout("square_2d", 9, "periodic", "3x3")
        assert "stroke-dasharray" in lattice_to_svg(pos, bonds)


class TestFigureForOperation:
    def test_lattice_op_yields_figure(self):
        from notebook.runtime.lattice_figure import tikz_figure_for_operation
        fig = tikz_figure_for_operation(
            "expand_neighbor_sum",
            {"pair_constraint": "square_2d", "n_sites": 4, "lattice_shape": "2x2"})
        assert fig and "tikzpicture" in fig and fig.startswith("\\begin{center}")

    def test_non_lattice_op_is_none(self):
        from notebook.runtime.lattice_figure import tikz_figure_for_operation
        assert tikz_figure_for_operation("expand", {"expr": "x"}) is None

    def test_bad_geometry_is_none(self):
        from notebook.runtime.lattice_figure import tikz_figure_for_operation
        # 6 sites, square, no shape → not a perfect square → no figure (no crash)
        assert tikz_figure_for_operation(
            "expand_neighbor_sum",
            {"pair_constraint": "square_2d", "n_sites": 6}) is None


class TestExportIntegration:
    """The TeX export drops the bond-structure figure next to a lattice sum."""

    def test_export_inserts_tikz_figure(self, tmp_path):
        import asyncio
        import ui.notebook_bridge as nb
        from pathlib import Path
        nb.start_engine(title="lattice_fig_export", mode="standard", language="en")
        try:
            eng = nb.get_engine()
            eng.run_operation("expand_neighbor_sum",
                {"body": "s_i*s_j", "index1": "i", "index2": "j", "n_sites": 4,
                 "pair_constraint": "square_2d", "lattice_shape": "2x2"})
            res = asyncio.run(nb.export_latex())
            assert res.get("ok"), res
            content = Path(res["path"]).read_text(encoding="utf-8")
            assert "\\usepackage{tikz}" in content
            assert "tikzpicture" in content
            assert "\\begin{center}" in content
        finally:
            nb.stop_engine()


class TestSaveTrigger:
    """Explicit "格子図を保存して" command saves the latest lattice figure."""

    def test_trigger_regex(self):
        from ui.interpret_pipeline import _is_save_lattice_figure as f
        assert f("格子図を保存して")
        assert f("save the lattice figure", "en")   # docs/28 §3.1: EN session
        assert not f("最近接和を展開して")
        assert not f("格子を展開")

    def test_save_for_engine_no_lattice(self):
        from notebook.runtime.lattice_figure import save_lattice_figure_for_engine
        eng = type("E", (), {"steps": {}, "session": None})()
        r = save_lattice_figure_for_engine(eng)
        assert r["ok"] is False and r["error"] == "no_lattice"

    def test_pipeline_saves_after_lattice(self, tmp_path):
        import asyncio
        from unittest.mock import MagicMock
        import ui.notebook_bridge as nb
        from notebook.runtime.llm_supervisor import LLMSupervisor
        from notebook.runtime.operation_validator import OperationValidator
        from ui.interpret_pipeline import (
            InterpretRequest, InputToken, run_interpret_pipeline)
        import ui.session_io as session_io

        sup = MagicMock(spec=LLMSupervisor)
        sup._v = OperationValidator()
        sup.needs_expression_injection.return_value = False
        nb.start_engine(title="figsave", mode="standard", language="en")
        try:
            with mock_sessions_dir(session_io, tmp_path):
                eng = nb.get_engine()
                eng.run_operation("expand_neighbor_sum",
                    {"body": "s_i*s_j", "index1": "i", "index2": "j",
                     "n_sites": 4, "pair_constraint": "square_2d",
                     "lattice_shape": "2x2"})
                req = InterpretRequest(
                    tokens=[InputToken(type="text", value="save the lattice figure")])
                res = asyncio.run(run_interpret_pipeline(req, eng, sup))
            assert res["status"] == "completed"
            assert res["operation"] == "save_lattice_figure"
            assert (tmp_path / "figsave_lattice.svg").exists()
            assert (tmp_path / "figsave_lattice.tex").exists()
        finally:
            nb.stop_engine()

    def test_pipeline_no_lattice_message(self):
        import asyncio
        from unittest.mock import MagicMock
        import ui.notebook_bridge as nb
        from notebook.runtime.llm_supervisor import LLMSupervisor
        from notebook.runtime.operation_validator import OperationValidator
        from ui.interpret_pipeline import (
            InterpretRequest, InputToken, run_interpret_pipeline)

        sup = MagicMock(spec=LLMSupervisor)
        sup._v = OperationValidator()
        sup.needs_expression_injection.return_value = False
        nb.start_engine(title="nofig", mode="standard", language="en")
        try:
            req = InterpretRequest(
                tokens=[InputToken(type="text", value="save the lattice figure")])
            res = asyncio.run(run_interpret_pipeline(req, nb.get_engine(), sup))
            assert res["status"] == "failed"
            assert "lattice" in res["error"].lower()
        finally:
            nb.stop_engine()


from contextlib import contextmanager


@contextmanager
def mock_sessions_dir(session_io, path):
    orig = session_io.sessions_dir
    session_io.sessions_dir = lambda: path
    try:
        yield
    finally:
        session_io.sessions_dir = orig


class TestSave:
    def test_writes_svg_and_tex(self, tmp_path):
        r = save_lattice_figure("triangular_2d", 9, "open", "3x3",
                                out_dir=tmp_path, name="tri", compile_pdf=False)
        assert (tmp_path / "tri.svg").exists()
        assert (tmp_path / "tri.tex").exists()
        assert "tikzpicture" in r["tikz"]

    def test_pdf_when_engine_available(self, tmp_path):
        r = save_lattice_figure("square_2d", 4, "open", "2x2",
                                out_dir=tmp_path, name="sq", compile_pdf=True)
        if _find_latex_engine() is None:
            assert r["pdf"] is None
            assert "no LaTeX engine" in r["pdf_error"]
        else:
            assert r["pdf"] and (tmp_path / "sq.pdf").exists()
