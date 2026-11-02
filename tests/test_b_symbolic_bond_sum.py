"""B stage 1: symbolic bond sums kept unexpanded, then concretized.

A bond-sum notation with no expand verb / site count is registered as a
symbolic expression (display round-trip + a lattice descriptor); "En を N
サイト…で展開" later concretizes it via expand_neighbor_sum.
"""
import asyncio
from unittest.mock import MagicMock

import pytest

import ui.notebook_bridge as nb
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operation_validator import OperationValidator
from ui.interpret_pipeline import (
    InterpretRequest, InputToken, run_interpret_pipeline,
    _parse_symbolic_bond_sum, _parse_concretize,
    _parse_bond_sum_substitute, _token_sub, _relabel_sites)


class TestStage2Helpers:
    def test_token_sub_plain(self):
        assert _token_sub("-J \\sum s_i s_j", "J", "2t") == "-2t \\sum s_i s_j"

    def test_token_sub_tex_command(self):
        assert _token_sub(r"\xi \sum c_i", r"\xi", "mu") == r"mu \sum c_i"

    def test_token_sub_word_boundary(self):
        # must not touch 'J' inside another token
        assert _token_sub("Jx + J", "J", "K") == "Jx + K"

    def test_substitute_parser_needs_symbolic_target(self):
        from unittest.mock import MagicMock
        eng = MagicMock()
        ex = MagicMock()
        ex.lattice_descriptor = None  # not a symbolic bond sum
        eng.expressions = {"E1": ex}
        eng.lhs_alias_map = {}
        assert _parse_bond_sum_substitute("E1 の J を 2t に置き換え", eng) is None


class TestStage3bRelabel:
    def test_relabels_subscripts_only(self):
        # coefficient digits must NOT be relabelled
        out = _relabel_sites(r"- 2 J \mathit{i} S^y_1 S^z_2", {1: "n", 2: "j"})
        assert out == r"- 2 J \mathit{i} S^y_n S^z_j"

    def test_braced_subscript(self):
        assert _relabel_sites("A_{1} B_{2}", {1: "n", 2: "j"}) == "A_{n} B_{j}"


class TestStage3bCommutator:
    """B stage 3b: [H, S^α_n] → symbolic neighbour-sum EOM, then concretize at a
    site (live re-derivation) — Julia spin session."""

    def test_symbolic_commutator_and_concretize(self):
        nb.start_engine(title="B3b", mode="condensed_matter", language="ja")
        try:
            eng = nb.get_engine()
            eng.run_operation("define_spin_operator", {"name": "S"})
            eng.run_operation("define_scalar_parameter", {"name": "J"})
            _run(eng, r"H = -J \sum_{\langle i j \rangle} S^z_i S^z_j")

            rc = _run(eng, "[H, S^x_n]")
            assert rc["status"] == "completed"
            assert rc["operation"] == "bond_sum_commutator"
            eom = eng.expressions[rc["expr_id"]]
            assert r"\sum_{j \in \mathrm{nn}(n)}" in eom.latex
            assert "S^y_n" in eom.latex and "S^z_j" in eom.latex   # relabelled
            assert eom.lattice_descriptor["kind"] == "neighbor_sum"

            rk = _run(eng, f"{rc['expr_id']} を 正方格子 3x3 のサイト5で展開して")
            assert rk["status"] == "completed"
            res = eng.expressions[rk["expr_id"]].latex
            # centre site 5 of a 3×3 square has neighbours 2,4,6,8
            for j in (2, 4, 6, 8):
                assert f"S^z_{j}" in res
            assert res.count("S^y_5") == 4
        finally:
            nb.stop_engine()

    def test_commutator_needs_symbolic_bond_sum(self):
        # a non-bond-sum E-id is not a commutator target
        from unittest.mock import MagicMock
        from ui.interpret_pipeline import _parse_bond_sum_commutator
        eng = MagicMock()
        ex = MagicMock(); ex.lattice_descriptor = None
        eng.expressions = {"E1": ex}
        eng.lhs_alias_map = {}
        assert _parse_bond_sum_commutator("[E1, S^x_n]", eng) is None


class TestParseSymbolic:
    def test_coefficient_nn_notation(self):
        latex, desc = _parse_symbolic_bond_sum(
            r"-J \sum_{\langle i j \rangle} S^z_i S^z_j")
        assert r"\sum_{\langle ij \rangle}" in latex
        assert desc["pair_constraint"] == "nearest_neighbor"
        assert desc["index1"] == "i" and desc["index2"] == "j"
        assert desc["body"] == "(-J)*(S^z_i S^z_j)"

    def test_all_pairs_notation(self):
        _, desc = _parse_symbolic_bond_sum(r"\sum_{i<j} q_i q_j")
        assert desc["pair_constraint"] == "all_pairs"

    def test_expand_verb_is_not_symbolic(self):
        assert _parse_symbolic_bond_sum(
            r"\sum_{\langle i j \rangle} S^z_i S^z_j を3サイトで展開") is None

    def test_site_count_is_not_symbolic(self):
        assert _parse_symbolic_bond_sum(
            r"\sum_{\langle i j \rangle} S^z_i S^z_j 4サイト") is None

    def test_equation_form_keeps_lhs(self):
        # 1b: equation form is accepted, LHS kept in the display
        latex, desc = _parse_symbolic_bond_sum(
            r"H = -J \sum_{\langle i j \rangle} S^z_i S^z_j")
        assert latex.startswith("H = ")
        assert desc["body"] == "(-J)*(S^z_i S^z_j)"

    def test_complex_lhs_rejected(self):
        assert _parse_symbolic_bond_sum(r"a + b = \sum_{i<j} q_i q_j") is None

    def test_plain_expression_not_symbolic(self):
        assert _parse_symbolic_bond_sum("S^z_1 S^z_2") is None


class TestRegisterBondSum:
    def test_engine_method_sets_descriptor(self):
        from notebook.engine import NotebookEngine
        with NotebookEngine(title="b1") as e:
            e.start()
            expr = e.register_bond_sum(
                r"\sum_{\langle ij \rangle} S^z_i S^z_j",
                {"body": "S^z_i*S^z_j", "index1": "i", "index2": "j",
                 "pair_constraint": "nearest_neighbor"})
            assert expr.lattice_descriptor["pair_constraint"] == "nearest_neighbor"
            assert r"\sum" in expr.latex
            assert e.expressions[expr.display_id] is expr


def _sup():
    s = MagicMock(spec=LLMSupervisor)
    s._v = OperationValidator()
    s.needs_expression_injection.return_value = False
    return s


def _run(engine, text):
    req = InterpretRequest(tokens=[InputToken(type="text", value=text)])
    return asyncio.run(run_interpret_pipeline(req, engine, _sup()))


class TestPipelineRoundTrip:
    def test_register_then_concretize(self):
        nb.start_engine(title="Bround", mode="standard", language="ja")
        try:
            eng = nb.get_engine()
            r1 = _run(eng, r"-J \sum_{\langle i j \rangle} s_i s_j")
            assert r1["status"] == "registered"
            assert r1["operation"] == "register_bond_sum"
            eid = r1["expr_id"]
            ex = eng.expressions[eid]
            assert r"\sum_{\langle ij \rangle}" in ex.latex      # display round-trip
            assert ex.lattice_descriptor is not None             # kept symbolic

            r2 = _run(eng, f"{eid} を正方格子 2x2 で展開して")
            assert r2["status"] == "completed"
            res = eng.expressions[r2["expr_id"]].latex
            # 2×2 square Ising: 4 bonds, each with the -J coefficient
            assert res.count("J") == 4
        finally:
            nb.stop_engine()

    def test_concretize_parser_needs_descriptor(self):
        # an E-id without a lattice_descriptor is not a concretize target
        eng = MagicMock()
        eng.expressions = {}
        eng.lhs_alias_map = {}
        assert _parse_concretize("E1 を 4 サイトで展開", eng) is None

    def test_substitute_into_symbolic_sum_then_concretize(self):
        # B stage 2: Op(Σ body) = Σ Op(body) — substitute a parameter, keep the sum
        nb.start_engine(title="Bsub", mode="standard", language="ja")
        try:
            eng = nb.get_engine()
            _run(eng, r"H = -J \sum_{\langle i j \rangle} s_i s_j")
            r2 = _run(eng, "H の J を 2t に置き換えて")
            assert r2["status"] == "registered"
            assert r2["operation"] == "bond_sum_substitute"
            new = eng.expressions[r2["expr_id"]]
            assert "2t" in new.latex and "-J" not in new.latex   # symbolic, sum kept
        finally:
            nb.stop_engine()

    def test_substitute_into_symbolic_sum_en(self):
        # Regression: _BOND_SUBST_RE was Japanese-only ("H の J を 2t に置き換え");
        # English fell through to a generic substitute that loses the
        # lattice_descriptor, breaking later expand/dispersion chaining.
        nb.start_engine(title="Bsub_en", mode="standard", language="en")
        try:
            eng = nb.get_engine()
            _run(eng, r"H = -J \sum_{\langle i j \rangle} s_i s_j")
            r2 = _run(eng, "replace J with 2t in H")
            assert r2["status"] == "registered"
            assert r2["operation"] == "bond_sum_substitute"
            new = eng.expressions[r2["expr_id"]]
            assert "2t" in new.latex and "-J" not in new.latex   # symbolic, sum kept
            assert new.lattice_descriptor is not None             # tracking preserved
            assert r"\sum" in new.latex
            assert eng.lhs_alias_map["H"] == r2["expr_id"]       # name now points here

            # English session → English expand command (NL patterns are looked
            # up by session language, docs/28 §3.1).
            r3 = _run(eng, "expand H over a 2x2 square lattice")
            res = eng.expressions[r3["expr_id"]].latex
            assert "t" in res and "J" not in res                 # substitution propagated
        finally:
            nb.stop_engine()

    def test_equation_form_register_and_concretize_by_name(self):
        nb.start_engine(title="Bname", mode="standard", language="ja")
        try:
            eng = nb.get_engine()
            r1 = _run(eng, r"H = -J \sum_{\langle i j \rangle} s_i s_j")
            assert r1["status"] == "registered"
            assert eng.lhs_alias_map.get("H") == r1["expr_id"]
            assert eng.expressions[r1["expr_id"]].latex.startswith("H = ")

            r2 = _run(eng, "H を 三角格子 3x3 で展開して")  # by NAME, triangular
            assert r2["status"] == "completed"
            # triangular 3×3 has 16 bonds
            assert eng.expressions[r2["expr_id"]].latex.count("J") == 16
        finally:
            nb.stop_engine()
