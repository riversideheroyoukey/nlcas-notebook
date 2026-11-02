"""F-1 stage 3c-graph: the adjacency matrix A of a finite graph and its algebra.

Unlike the infinite-lattice dispersion (symbolic cos entries → cubic walls), a
finite graph's A is a concrete integer matrix, so powers / walk counts / degrees /
triangle counts are exact, and registering A as an ordinary matrix expression lets
the existing eigenvalues operation chain for free.
"""
import asyncio
from unittest.mock import MagicMock

import ui.notebook_bridge as nb
from notebook.runtime.backend_handler import (
    build_adjacency_matrix, adjacency_to_canonical, adjacency_to_latex,
    graph_degree_sequence, graph_triangle_count, graph_walk_trace,
    graph_edge_count, graph_identity_statements, graph_identity_check)
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operation_validator import OperationValidator
from ui.interpret_pipeline import (
    InterpretRequest, InputToken, run_interpret_pipeline, _parse_graph_spec)


class TestBuildAdjacency:
    def test_chain_open(self):
        a, err = build_adjacency_matrix("nearest_neighbor", n_sites=4)
        assert err is None
        assert a == [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]]

    def test_chain_periodic(self):
        a, err = build_adjacency_matrix("nearest_neighbor", n_sites=4,
                                        boundary="periodic")
        assert err is None
        assert a[0][3] == 1 and a[3][0] == 1          # wrap bond

    def test_custom_triangle(self):
        a, err = build_adjacency_matrix("custom", edges="(1,2),(2,3),(1,3)")
        assert err is None
        assert a == [[0, 1, 1], [1, 0, 1], [1, 1, 0]]

    def test_square_2x2(self):
        a, err = build_adjacency_matrix("square_2d", lattice_shape="2x2")
        assert err is None
        assert graph_degree_sequence(a) == [2, 2, 2, 2]   # 4-cycle

    def test_custom_needs_edges(self):
        a, err = build_adjacency_matrix("custom", edges="")
        assert a is None and "custom" in err


class TestGraphInvariants:
    def test_canonical_and_latex(self):
        a = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
        assert adjacency_to_canonical(a) == "matrix([0,1,1],[1,0,1],[1,1,0])"
        assert adjacency_to_latex(a) == (
            r"\begin{pmatrix} 0 & 1 & 1 \\ 1 & 0 & 1 \\ 1 & 1 & 0 \end{pmatrix}")

    def test_degree_sequence(self):
        a, _ = build_adjacency_matrix("nearest_neighbor", n_sites=4)
        assert graph_degree_sequence(a) == [1, 2, 2, 1]

    def test_triangle_count(self):
        tri, _ = build_adjacency_matrix("custom", edges="(1,2),(2,3),(1,3)")
        assert graph_triangle_count(tri) == 1
        chain, _ = build_adjacency_matrix("nearest_neighbor", n_sites=4)
        assert graph_triangle_count(chain) == 0

    def test_walk_trace_is_twice_edges(self):
        # tr(A²) = Σ degrees = 2|E|; chain of 4 has 3 edges → 6
        a, _ = build_adjacency_matrix("nearest_neighbor", n_sites=4)
        assert graph_walk_trace(a, 2) == 6
        assert graph_walk_trace(a, 1) == 0            # tr A = 0 (no self-loops)


class TestGraphIdentities:
    def test_edge_count(self):
        tri, _ = build_adjacency_matrix("custom", edges="(1,2),(2,3),(1,3)")
        assert graph_edge_count(tri) == 3

    def test_statements_are_symbolic(self):
        s = graph_identity_statements()
        assert s == (r"\operatorname{tr} A = 0,\ "
                     r"\operatorname{tr} A^2 = 2|E|,\ "
                     r"\operatorname{tr} A^3 = 6 N_\triangle")

    def test_check_holds_on_triangle(self):
        tri, _ = build_adjacency_matrix("custom", edges="(1,2),(2,3),(1,3)")
        latex, ok = graph_identity_check(tri)
        assert ok is True
        assert r"\operatorname{tr} A^2 = 2|E| = 6" in latex
        assert r"\operatorname{tr} A^3 = 6 N_\triangle = 6" in latex   # 1 triangle

    def test_check_holds_on_square_lattice(self):
        a, _ = build_adjacency_matrix("square_2d", lattice_shape="3x3")
        latex, ok = graph_identity_check(a)
        assert ok is True
        assert r"\operatorname{tr} A^2 = 2|E| = 24" in latex          # 12 edges
        assert r"\operatorname{tr} A^3 = 6 N_\triangle = 0" in latex  # no triangles


class TestParseGraphSpec:
    def test_lattice_shape(self):
        spec = _parse_graph_spec("正方格子 3x3 の隣接行列を求めて")
        assert spec["pair_constraint"] == "square_2d"
        assert spec["lattice_shape"] == "3x3" and spec["n_sites"] == 9

    def test_chain_sites(self):
        spec = _parse_graph_spec("5サイトの鎖の次数列を求めて")
        assert spec["pair_constraint"] == "nearest_neighbor"
        assert spec["n_sites"] == 5

    def test_custom_edges(self):
        spec = _parse_graph_spec("グラフ (1,2),(2,3),(1,3) の隣接行列")
        assert spec["pair_constraint"] == "custom"
        assert spec["edges"] == [(1, 2), (2, 3), (1, 3)]

    def test_no_spec(self):
        assert _parse_graph_spec("隣接行列を求めて") is None


def _sup():
    s = MagicMock(spec=LLMSupervisor)
    s._v = OperationValidator()
    s.needs_expression_injection.return_value = False
    return s


def _run(engine, text):
    req = InterpretRequest(tokens=[InputToken(type="text", value=text)])
    return asyncio.run(run_interpret_pipeline(req, engine, _sup()))


class TestGraphPipeline:
    def test_adjacency_matrix_command(self):
        nb.start_engine(title="G1", mode="standard", language="ja")
        try:
            eng = nb.get_engine()
            r = _run(eng, "正方格子 2x2 の隣接行列を求めて")
            assert r["status"] == "completed"
            assert r["operation"] == "adjacency_matrix"
            ex = eng.expressions[r["expr_id"]]
            assert ex.canonical.startswith("matrix([")
            assert r"\begin{pmatrix}" in ex.latex
        finally:
            nb.stop_engine()

    def test_degree_sequence_command(self):
        nb.start_engine(title="G2", mode="standard", language="ja")
        try:
            eng = nb.get_engine()
            r = _run(eng, "5サイトの鎖の次数列を求めて")
            assert r["status"] == "completed"
            assert eng.expressions[r["expr_id"]].latex == r"\deg = (1, 2, 2, 2, 1)"
        finally:
            nb.stop_engine()

    def test_triangle_count_command(self):
        nb.start_engine(title="G3", mode="standard", language="ja")
        try:
            eng = nb.get_engine()
            r = _run(eng, "グラフ (1,2),(2,3),(1,3) の三角形の数を求めて")
            assert r["status"] == "completed"
            assert eng.expressions[r["expr_id"]].latex == r"\#\triangle = 1"
        finally:
            nb.stop_engine()

    def test_adjacency_spectrum_command(self):
        nb.start_engine(title="G4", mode="standard", language="ja")
        try:
            eng = nb.get_engine()
            # triangle K3: adjacency eigenvalues are 2, -1, -1
            r = _run(eng, "グラフ (1,2),(2,3),(1,3) の隣接行列の固有値を求めて")
            assert r["status"] == "completed"
            assert r["operation"] == "adjacency_spectrum"
            latex = eng.expressions[r["expr_id"]].latex
            assert "2" in latex and "-1" in latex
        finally:
            nb.stop_engine()

    def test_identity_statements_command(self):
        nb.start_engine(title="G5", mode="standard", language="ja")
        try:
            eng = nb.get_engine()
            r = _run(eng, "グラフの恒等式を表示して")
            assert r["status"] == "completed"
            assert r["operation"] == "graph_identity"
            assert r"\operatorname{tr} A^2 = 2|E|" in eng.expressions[r["expr_id"]].latex
        finally:
            nb.stop_engine()

    def test_identity_verify_command(self):
        nb.start_engine(title="G6", mode="standard", language="ja")
        try:
            eng = nb.get_engine()
            r = _run(eng, "グラフ (1,2),(2,3),(1,3) で恒等式を確認して")
            assert r["status"] == "completed"
            assert r"\operatorname{tr} A^2 = 2|E| = 6" in eng.expressions[r["expr_id"]].latex
        finally:
            nb.stop_engine()


class TestSpinSpectralPipeline:
    """R1c batch 2 (docs/27): end-to-end coverage for the sibling F-1 stage 3c
    spin/spectral NL routes that the batch extracted into the early-intercept
    registry (previously exercised only at the spin_algebra layer, never through
    run_interpret_pipeline).  A standard Maxima session is fine — the spin
    algebra is self-contained Python (CAS-bypass)."""

    def test_spin_commutator_command(self):
        nb.start_engine(title="GSC", mode="standard", language="ja")
        try:
            eng = nb.get_engine()
            r = _run(eng, "ハイゼンベルグ模型 3サイト で S^z_1 と H の交換子を求めて")
            assert r["status"] == "completed"
            assert r["operation"] == "spin_commutator"
        finally:
            nb.stop_engine()

    def test_specific_heat_command(self):
        nb.start_engine(title="GSH", mode="standard", language="ja")
        try:
            eng = nb.get_engine()
            r = _run(eng, "ハイゼンベルグ 三角格子 3サイト の比熱を高温展開で求めて")
            assert r["status"] == "completed"
            assert r["operation"] == "specific_heat"
        finally:
            nb.stop_engine()
