"""F-1 (second increment): natural-language routing to expand_neighbor_sum.

"S^z_i S^z_j の最近接和を 4 サイト 周期境界で展開して" is intercepted rule-based
(before the LLM) and dispatched to the expand_neighbor_sum operation.
"""
import asyncio

import pytest

from unittest.mock import MagicMock

from notebook.engine import NotebookEngine
from notebook.models.session import SessionMode
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operation_validator import OperationValidator
from ui.interpret_pipeline import _parse_neighbor_sum


class TestParseNeighborSum:
    def test_basic_periodic(self):
        op, args = _parse_neighbor_sum(
            "S^z_i S^z_j の最近接和を 4 サイト 周期境界で展開して")
        assert op == "expand_neighbor_sum"
        assert args == {"body": "S^z_i S^z_j", "index1": "i", "index2": "j",
                        "n_sites": 4, "boundary": "periodic"}

    def test_open_default_and_star_body(self):
        _, args = _parse_neighbor_sum("Sz_i*Sz_j の最近接和を3サイトで展開して")
        assert args["boundary"] == "open"
        assert args["n_sites"] == 3
        assert args["body"] == "Sz_i*Sz_j"

    def test_fullwidth_digit_and_coefficient(self):
        _, args = _parse_neighbor_sum(
            "-J S^z_i S^z_j の最近接相互作用和を ５ サイト 開放端で展開して")
        assert args["n_sites"] == 5
        assert args["boundary"] == "open"
        assert args["body"] == "-J S^z_i S^z_j"

    def test_heisenberg_body_picks_first_two_indices(self):
        _, args = _parse_neighbor_sum(
            "Sx_i Sx_j + Sy_i Sy_j + Sz_i Sz_j の隣接和を4サイト周期境界で展開")
        assert args["index1"] == "i" and args["index2"] == "j"

    def test_non_matching_returns_none(self):
        assert _parse_neighbor_sum("E1 を展開して") is None
        assert _parse_neighbor_sum("S^z_1 S^z_2 の交換子") is None
        # nearest-neighbor mentioned but no two indices in body
        assert _parse_neighbor_sum("Sz_i の最近接和を4サイトで展開して") is None


class TestParseSumNotation:
    """A: \\sum_{<ij>} math-notation trigger (in addition to the keyword form)."""

    def test_tex_langle_rangle(self):
        op, args = _parse_neighbor_sum(
            r"\sum_{\langle i j \rangle} S^z_i S^z_j を 4 サイト 周期境界で展開して")
        assert op == "expand_neighbor_sum"
        assert args == {"body": "S^z_i S^z_j", "index1": "i", "index2": "j",
                        "n_sites": 4, "boundary": "periodic"}

    def test_ascii_angle_brackets(self):
        _, args = _parse_neighbor_sum(r"\sum_{<ij>} Sz_i Sz_j を3サイトで展開して")
        assert args["index1"] == "i" and args["index2"] == "j"
        assert args["n_sites"] == 3 and args["boundary"] == "open"
        assert args["body"] == "Sz_i Sz_j"

    def test_unicode_sigma_and_coefficient(self):
        _, args = _parse_neighbor_sum(
            r"\sum_{\langle i,j \rangle} -J Sz_i Sz_j を4サイト周期境界で展開")
        assert args["body"] == "-J Sz_i Sz_j"
        assert args["boundary"] == "periodic"

    def test_single_index_sum_not_a_bond(self):
        # \sum_{k} is an ordinary single-index sum, not a nearest-neighbor bond
        assert _parse_neighbor_sum(r"\sum_{k} c'(k) c(k) を展開") is None

    # Regression: the notation-form body cut (_ns_extract_body_after) only
    # recognized the Japanese を/の particles as a boundary, so English text
    # after \sum_{<ij>} kept trailing words ("over a 4-site periodic chain")
    # inside "body" instead of stopping at the preposition.
    def test_en_over_preposition(self):
        op, args = _parse_neighbor_sum(
            r"\sum_{\langle i j \rangle} S^z_i S^z_j over a 4-site periodic chain, expand", "en")
        assert op == "expand_neighbor_sum"
        assert args == {"body": "S^z_i S^z_j", "index1": "i", "index2": "j",
                        "n_sites": 4, "boundary": "periodic"}

    def test_en_for_preposition_open_boundary(self):
        _, args = _parse_neighbor_sum(
            r"expand \sum_{\langle i j \rangle} S^z_i S^z_j for a 3-site chain", "en")
        assert args["body"] == "S^z_i S^z_j"
        assert args["n_sites"] == 3 and args["boundary"] == "open"


class TestParseAllPairs:
    """B (molecular): \\sum_{i<j} / 全ペア和 → pair_constraint=all_pairs."""

    def test_allpairs_notation(self):
        _, args = _parse_neighbor_sum(r"\sum_{i<j} q_i q_j を 3 サイトで展開して")
        assert args["pair_constraint"] == "all_pairs"
        assert args["index1"] == "i" and args["index2"] == "j"
        assert args["body"] == "q_i q_j" and args["n_sites"] == 3

    def test_allpairs_keyword(self):
        _, args = _parse_neighbor_sum("q_i q_j の全ペア和を3サイトで展開して")
        assert args["pair_constraint"] == "all_pairs"

    def test_nearest_neighbor_has_no_pair_constraint_key(self):
        # backward compatibility: nn results omit pair_constraint
        _, args = _parse_neighbor_sum("S^z_i S^z_j の最近接和を4サイトで展開して")
        assert "pair_constraint" not in args


class TestParse2DLattice:
    """2D lattice keyword → square_2d / triangular_2d + lattice_shape."""

    def test_square_lattice(self):
        _, args = _parse_neighbor_sum("Sz_i Sz_j の正方格子 3x3 の最近接和を展開して")
        assert args["pair_constraint"] == "square_2d"
        assert args["lattice_shape"] == "3x3"
        assert args["n_sites"] == 9
        assert args["body"] == "Sz_i Sz_j"

    def test_triangular_lattice_periodic(self):
        _, args = _parse_neighbor_sum("S^z_i S^z_j の三角格子 3x3 周期境界で展開して")
        assert args["pair_constraint"] == "triangular_2d"
        assert args["boundary"] == "periodic"

    def test_english_square_lattice_rectangle(self):
        # Session-language lookup (docs/28 §3.1): an English lattice name goes
        # with an English command in an English session.
        _, args = _parse_neighbor_sum(
            "expand the nearest-neighbor sum of Sz_i Sz_j over a 2x4 square lattice", "en")
        assert args["pair_constraint"] == "square_2d"
        assert args["lattice_shape"] == "2x4" and args["n_sites"] == 8

    def test_honeycomb_lattice(self):
        _, args = _parse_neighbor_sum(
            "Sz_i Sz_j のハニカム 3x3 の最近接和を展開して")
        assert args["pair_constraint"] == "honeycomb_2d"

    def test_cylinder_boundary(self):
        _, args = _parse_neighbor_sum(
            "Sz_i Sz_j の正方格子 3x3 円筒境界で展開して")
        assert args["boundary"] == "cylinder" and args["pair_constraint"] == "square_2d"


def _mock_supervisor() -> MagicMock:
    sup = MagicMock(spec=LLMSupervisor)
    sup._v = OperationValidator()
    sup.needs_expression_injection.return_value = False
    return sup


class TestPipelineDispatch:
    """End-to-end through the real pipeline + a live Julia engine."""

    def _run(self, engine, text: str) -> dict:
        from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline
        req = InterpretRequest(tokens=[InputToken(type="text", value=text)])
        return asyncio.run(run_interpret_pipeline(req, engine, _mock_supervisor()))

    def test_nl_open_chain(self):
        with NotebookEngine(title="f1nl", mode=SessionMode.second_quantization) as e:
            e.start()
            e.run_operation("define_spin_operator", {"name": "S"})
            res = self._run(e, "S^z_i S^z_j の最近接和を3サイトで展開して")
            assert res["status"] == "completed"
            assert res["operation"] == "expand_neighbor_sum"
            latex = e.expressions[res["expr_id"]].latex
            assert "S^z_1 S^z_2" in latex and "S^z_2 S^z_3" in latex
            assert "S^z_3 S^z_1" not in latex  # open: no wrap-around bond

    def test_nl_periodic_chain(self):
        with NotebookEngine(title="f1nl2", mode=SessionMode.second_quantization) as e:
            e.start()
            e.run_operation("define_spin_operator", {"name": "S"})
            res = self._run(e, "Sz_i Sz_j の最近接和を3サイト周期境界で展開して")
            assert res["status"] == "completed"
            latex = e.expressions[res["expr_id"]].latex
            assert "Sz_3 Sz_1" in latex  # periodic adds the wrap-around bond

    def test_sum_notation_periodic_chain(self):
        with NotebookEngine(title="f1sum", mode=SessionMode.second_quantization) as e:
            e.start()
            e.run_operation("define_spin_operator", {"name": "S"})
            res = self._run(e, r"\sum_{\langle i j \rangle} S^z_i S^z_j を3サイト周期境界で展開して")
            assert res["status"] == "completed"
            assert res["operation"] == "expand_neighbor_sum"
            latex = e.expressions[res["expr_id"]].latex
            assert "S^z_1 S^z_2" in latex and "S^z_3 S^z_1" in latex

    def test_nl_2d_square_lattice(self):
        with NotebookEngine(title="f1_2d", mode=SessionMode.second_quantization) as e:
            e.start()
            e.run_operation("define_spin_operator", {"name": "S"})
            res = self._run(e, "Sz_i Sz_j の正方格子 2x2 の最近接和を展開して")
            assert res["status"] == "completed"
            assert res["operation"] == "expand_neighbor_sum"
            latex = e.expressions[res["expr_id"]].latex
            # 2×2 square: 4 bonds
            assert latex.count("Sz_") == 8

    def test_nl_maxima_scalar_chain(self):
        # The same NL routing works in a Maxima (standard_symbolic) session with
        # a scalar lattice body — quantum-chemistry / lattice-energy use case.
        with NotebookEngine(title="f1max", mode=SessionMode.standard_symbolic) as e:
            e.start()
            res = self._run(e, "-J*s_i*s_j の最近接和を4サイト周期境界で展開して")
            assert res["status"] == "completed"
            assert res["operation"] == "expand_neighbor_sum"
            flat = e.expressions[res["expr_id"]].latex.replace(" ", "")
            assert "s\\left(1\\right)" in flat and "s\\left(4\\right)" in flat

    def test_unrecognized_lattice_name_errors_instead_of_corrupting(self):
        # A silent-lattice-fallback bugfix: an unrecognized lattice name (e.g.
        # "ダイヤモンド格子"/"diamond lattice") used to be silently ignored,
        # falling back to the 1D chain AND leaking stray text (here "9サイト")
        # into the substituted body as a bogus numeric coefficient. It must
        # now fail with a clear error naming the unrecognized lattice,
        # produced BEFORE any body substitution happens.
        with NotebookEngine(title="f1unk1", mode=SessionMode.second_quantization) as e:
            e.start()
            e.run_operation("define_spin_operator", {"name": "S"})
            res = self._run(e, "Sz_i Sz_j のダイヤモンド格子で9サイトの最近接和を展開して")
            assert res["status"] == "failed"
            assert "unrecognized lattice" in res["error"]
            assert "diamond" in res["error"].lower() or "ダイヤモンド" in res["error"]

    def test_unrecognized_en_lattice_name_errors(self):
        with NotebookEngine(title="f1unk2", mode=SessionMode.second_quantization) as e:
            e.start()
            # English input → English session (NL patterns are looked up by
            # session language, docs/28 §3.1).
            e.session.language = ["en"]
            e.run_operation("define_spin_operator", {"name": "S"})
            res = self._run(
                e, "Expand the nearest-neighbor sum of Sz_i Sz_j over a "
                   "diamond lattice with 9 sites")
            assert res["status"] == "failed"
            assert "unrecognized lattice" in res["error"]
            assert "diamond" in res["error"].lower()

    def test_named_lattices_still_work_after_the_guard(self):
        # The unrecognized-lattice guard must not regress any of the
        # currently-supported named lattices.
        with NotebookEngine(title="f1unk3", mode=SessionMode.second_quantization) as e:
            e.start()
            e.run_operation("define_spin_operator", {"name": "S"})
            res = self._run(e, "Sz_i Sz_j の正方格子 3x3 の最近接和を展開して")
            assert res["status"] == "completed"


class TestCustomGraphBonds:
    """A custom/abstract bond list ("グラフ (1,2),(2,3),(1,3)" / "graph
    (1,2),(2,3),(1,3)") lets expand_neighbor_sum reach an arbitrary lattice,
    the same escape hatch the dispersion-relation feature already has via
    its abstract crystal descriptor."""

    def test_parse_custom_edges_jp(self):
        op, args = _parse_neighbor_sum(
            "Sz_i Sz_j のグラフ (1,2),(2,3),(1,3) の最近接和を展開して")
        assert op == "expand_neighbor_sum"
        assert args["pair_constraint"] == "custom"
        assert args["edges"] == [(1, 2), (2, 3), (1, 3)]
        assert args["body"] == "Sz_i Sz_j"
        assert "n_sites" not in args   # bonds alone determine the sites

    def test_parse_custom_edges_en(self):
        op, args = _parse_neighbor_sum(
            "Expand the nearest-neighbor sum of Sz_i Sz_j over the graph "
            "(1,2),(2,3),(1,3)", "en")
        assert op == "expand_neighbor_sum"
        assert args["pair_constraint"] == "custom"
        assert args["edges"] == [(1, 2), (2, 3), (1, 3)]
        assert args["body"] == "Sz_i Sz_j"

    def test_custom_edges_triangle_pipeline_jp(self):
        with NotebookEngine(title="f1graph1", mode=SessionMode.second_quantization) as e:
            e.start()
            e.run_operation("define_spin_operator", {"name": "S"})
            res = self._run(e, "Sz_i Sz_j のグラフ (1,2),(2,3),(1,3) の最近接和を展開して")
            assert res["status"] == "completed"
            assert res["operation"] == "expand_neighbor_sum"
            latex = e.expressions[res["expr_id"]].latex
            assert "Sz_1 Sz_2" in latex
            assert "Sz_2 Sz_3" in latex
            assert "Sz_1 Sz_3" in latex

    def test_custom_edges_triangle_pipeline_en(self):
        with NotebookEngine(title="f1graph2", mode=SessionMode.second_quantization) as e:
            e.start()
            e.session.language = ["en"]
            e.run_operation("define_spin_operator", {"name": "S"})
            res = self._run(
                e, "Expand the nearest-neighbor sum of Sz_i Sz_j over the "
                   "graph (1,2),(2,3),(1,3)")
            assert res["status"] == "completed"
            latex = e.expressions[res["expr_id"]].latex
            assert "Sz_1 Sz_2" in latex
            assert "Sz_2 Sz_3" in latex
            assert "Sz_1 Sz_3" in latex

    def _run(self, engine, text: str) -> dict:
        from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline
        req = InterpretRequest(tokens=[InputToken(type="text", value=text)])
        return asyncio.run(run_interpret_pipeline(req, engine, _mock_supervisor()))
