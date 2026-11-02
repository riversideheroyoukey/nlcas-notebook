"""F-1 (first increment): expand a nearest-neighbor bond sum on a 1D chain.

`expand_neighbor_sum` turns a compact `Σ_{⟨ij⟩} body(i,j)` into the explicit
sum of per-site operator terms on an N-site chain, with open or periodic
boundary conditions.  The expanded expression reuses the existing per-site
spin machinery, so it can be fed straight into commutator / mean-field / etc.
"""
import pytest

from notebook.engine import NotebookEngine
from notebook.models.session import SessionMode
from notebook.runtime.operation_validator import OperationValidator


@pytest.fixture
def v():
    return OperationValidator()


@pytest.fixture(scope="module")
def engine():
    with NotebookEngine(title="f1", mode=SessionMode.second_quantization) as e:
        e.start()
        e.run_operation("define_spin_operator", {"name": "S"})
        e.run_operation("define_scalar_parameter", {"name": "J"})
        yield e


class TestSchema:
    def test_operation_known(self, v):
        assert "expand_neighbor_sum" in v.known_operations

    def test_supported_backends(self, v):
        spec = v.get_spec("expand_neighbor_sum")
        assert "julia_full" in spec["supported_backends"]
        assert "maxima" in spec["supported_backends"]

    def test_aliases(self, v):
        assert v.resolve_alias("最近接和を展開") == "expand_neighbor_sum"
        assert v.resolve_alias("nearest neighbor sum") == "expand_neighbor_sum"

    def test_required_fields(self, v):
        r = v.validate("expand_neighbor_sum",
                       {"body": "Sz_i*Sz_j", "index1": "i", "index2": "j", "n_sites": 3},
                       "julia_full", set())
        assert r.valid, r.error

    def test_missing_field_rejected(self, v):
        # index2 is required; n_sites is optional (custom/2D-shape derive it)
        r = v.validate("expand_neighbor_sum",
                       {"body": "Sz_i*Sz_j", "index1": "i"},
                       "julia_full", set())
        assert not r.valid

    def test_n_sites_optional_for_custom(self, v):
        r = v.validate("expand_neighbor_sum",
                       {"body": "Sz_i*Sz_j", "index1": "i", "index2": "j",
                        "pair_constraint": "custom", "edges": "(1,2),(2,3)"},
                       "julia_full", set())
        assert r.valid, r.error


class TestBuildValidation:
    """Builder-level validation (handler.build_command, no live Julia)."""

    def _build(self, engine, **kw):
        return engine._handler.build_command("expand_neighbor_sum", kw, {})

    def test_bad_boundary_rejected(self, engine):
        r = self._build(engine, body="Sz_i*Sz_j", index1="i", index2="j",
                        n_sites=3, boundary="helical")
        assert not r.ok and "boundary" in r.error

    def test_n_sites_too_small_rejected(self, engine):
        r = self._build(engine, body="Sz_i*Sz_j", index1="i", index2="j", n_sites=1)
        assert not r.ok

    def test_same_index_rejected(self, engine):
        r = self._build(engine, body="Sz_i*Sz_i", index1="i", index2="i", n_sites=3)
        assert not r.ok

    def test_non_integer_n_sites_rejected(self, engine):
        r = self._build(engine, body="Sz_i*Sz_j", index1="i", index2="j", n_sites="lots")
        assert not r.ok


class TestEngineExpansion:
    """End-to-end through a live Julia session."""

    def test_open_chain_three_sites(self, engine):
        _, v = engine.run_operation("expand_neighbor_sum",
            {"body": "Sz_i*Sz_j", "index1": "i", "index2": "j", "n_sites": 3})
        assert "Sᶻ(1) Sᶻ(2)" in v and "Sᶻ(2) Sᶻ(3)" in v
        assert "Sᶻ(3) Sᶻ(1)" not in v  # open: no wrap-around bond

    def test_periodic_chain_three_sites(self, engine):
        _, v = engine.run_operation("expand_neighbor_sum",
            {"body": "Sz_i*Sz_j", "index1": "i", "index2": "j",
             "n_sites": 3, "boundary": "periodic"})
        assert "Sᶻ(3) Sᶻ(1)" in v  # periodic adds the wrap-around bond

    def test_periodic_two_sites_no_double_bond(self, engine):
        # a 2-site ring has a single bond; periodic must not add (2,1) again
        _, v = engine.run_operation("expand_neighbor_sum",
            {"body": "Sz_i*Sz_j", "index1": "i", "index2": "j",
             "n_sites": 2, "boundary": "periodic"})
        # only one Sz(1)Sz(2) bond, no Sz(2)Sz(1) duplicate term
        assert v.count("Sᶻ") == 2

    def test_coefficient_carried(self, engine):
        _, v = engine.run_operation("expand_neighbor_sum",
            {"body": "-J*Sz_i*Sz_j", "index1": "i", "index2": "j", "n_sites": 4})
        assert v.count("J Sᶻ") == 3  # three bonds on a 4-site open chain

    def test_heisenberg_body(self, engine):
        _, v = engine.run_operation("expand_neighbor_sum",
            {"body": "Sx_i*Sx_j + Sy_i*Sy_j + Sz_i*Sz_j",
             "index1": "i", "index2": "j", "n_sites": 2})
        assert "Sˣ(1) Sˣ(2)" in v and "Sʸ(1) Sʸ(2)" in v and "Sᶻ(1) Sᶻ(2)" in v

    def test_expanded_result_is_operable(self, engine):
        """The expanded chain Hamiltonian can be used in a downstream commutator."""
        step, _ = engine.run_operation("expand_neighbor_sum",
            {"body": "Sz_i*Sz_j", "index1": "i", "index2": "j", "n_sites": 3})
        _, c = engine.run_operation(
            "commutator", {"a": step.output_refs[0], "b": "Sx_1"})
        # [Sz_1 Sz_2, Sx_1] = (i Sy_1) Sz_2 etc. — non-empty, no raw error
        assert c and "MethodError" not in c


class TestMaximaExpansion:
    """F-1 Maxima parity: scalar lattice bodies (classical Ising, lattice
    energy, quantum-chemistry terms) expand on a 1D chain via Maxima."""

    @pytest.fixture(scope="class")
    def meng(self):
        from notebook.models.session import SessionMode as _SM
        with NotebookEngine(title="f1max", mode=_SM.standard_symbolic) as e:
            e.start()
            yield e

    def test_open_chain_scalar_ising(self, meng):
        _, v = meng.run_operation("expand_neighbor_sum",
            {"body": "-J*s_i*s_j", "index1": "i", "index2": "j", "n_sites": 4})
        flat = v.replace(" ", "")
        # three open-chain bonds: (1,2),(2,3),(3,4)
        assert "s(1)*s(2)" in flat and "s(2)*s(3)" in flat and "s(3)*s(4)" in flat
        assert "J" in flat

    def test_periodic_adds_wrap_bond(self, meng):
        _, v = meng.run_operation("expand_neighbor_sum",
            {"body": "s_i*s_j", "index1": "i", "index2": "j",
             "n_sites": 3, "boundary": "periodic"})
        flat = v.replace(" ", "")
        # wrap-around bond (3,1) — Maxima may order it as s(1)*s(3)
        assert "s(1)*s(3)" in flat or "s(3)*s(1)" in flat

    def test_validator_accepts_maxima(self):
        r = OperationValidator().validate(
            "expand_neighbor_sum",
            {"body": "s_i*s_j", "index1": "i", "index2": "j", "n_sites": 3},
            "maxima", set())
        assert r.valid, r.error


class TestAllPairsConstraint:
    """B (extensible bond structure): the all_pairs generator covers molecular /
    quantum-chemistry pairwise sums Σ_{i<j} body, in addition to the default
    nearest_neighbor 1D-chain bonds.  The pair structure is pluggable, so new
    domains (2D lattices, molecular graphs) register another generator."""

    def test_julia_all_pairs_three_sites(self, engine):
        # N=3 all_pairs → (1,2),(1,3),(2,3)
        _, v = engine.run_operation("expand_neighbor_sum",
            {"body": "Sz_i*Sz_j", "index1": "i", "index2": "j",
             "n_sites": 3, "pair_constraint": "all_pairs"})
        assert "Sᶻ(1) Sᶻ(2)" in v and "Sᶻ(1) Sᶻ(3)" in v and "Sᶻ(2) Sᶻ(3)" in v

    def test_maxima_all_pairs_scalar(self):
        from notebook.models.session import SessionMode as _SM
        with NotebookEngine(title="ap_max", mode=_SM.standard_symbolic) as e:
            e.start()
            _, v = e.run_operation("expand_neighbor_sum",
                {"body": "q_i*q_j", "index1": "i", "index2": "j",
                 "n_sites": 3, "pair_constraint": "all_pairs"})
            flat = v.replace(" ", "")
            # three unordered pairs of charges
            assert "q(1)*q(2)" in flat and "q(1)*q(3)" in flat and "q(2)*q(3)" in flat

    def test_all_pairs_count_four_sites(self, engine):
        # N=4 all_pairs → 6 pairs
        _, v = engine.run_operation("expand_neighbor_sum",
            {"body": "Sz_i*Sz_j", "index1": "i", "index2": "j",
             "n_sites": 4, "pair_constraint": "all_pairs"})
        assert v.count("Sᶻ(") == 12  # 6 pairs × 2 operators

    def test_unknown_constraint_rejected(self, engine):
        # The error message was reworded (2026-07-03, the expand_neighbor_sum
        # sibling of the dispersion-relation silent-lattice-fallback fix) to
        # name the unrecognized value directly and point at the custom-graph
        # escape hatch, rather than the raw parameter name "pair_constraint".
        r = engine._handler.build_command("expand_neighbor_sum",
            {"body": "Sz_i*Sz_j", "index1": "i", "index2": "j",
             "n_sites": 3, "pair_constraint": "bogus"}, {})
        assert not r.ok
        assert "unrecognized lattice" in r.error and "'bogus'" in r.error
        assert "graph" in r.error   # points at the custom-bond escape hatch

    def test_default_is_nearest_neighbor(self, engine):
        # No pair_constraint → nearest_neighbor: N=3 open gives only 2 bonds
        _, v = engine.run_operation("expand_neighbor_sum",
            {"body": "Sz_i*Sz_j", "index1": "i", "index2": "j", "n_sites": 3})
        assert "Sᶻ(1) Sᶻ(3)" not in v  # all_pairs would include (1,3); nn does not


class TestPinSite:
    """B stage 3a: pin_site concretizes a neighbour sum Σ_{j∈nn(n)} body(n,j) at
    a concrete site n — keep only the bonds touching n, the other endpoint = j."""

    def _terms(self, **kw):
        from notebook.runtime.backend_handler import build_neighbor_sum_terms
        s, e = build_neighbor_sum_terms(
            kw.get("body", "A_n*B_j"), "n", "j", kw.get("n_sites"),
            kw.get("boundary", "open"), kw["pair_constraint"],
            kw.get("lattice_shape"), kw.get("edges"), kw.get("pin_site"))
        assert e is None, e
        return s

    def test_square_center_has_four_neighbours(self):
        s = self._terms(pair_constraint="square_2d", lattice_shape="3x3", pin_site=5)
        assert s.count("A_5") == 4
        for j in (2, 4, 6, 8):
            assert f"B_{j}" in s

    def test_square_corner_has_two_neighbours(self):
        s = self._terms(pair_constraint="square_2d", lattice_shape="3x3", pin_site=1)
        assert s.count("A_5".replace("5", "1")) == 2  # site 1 corner: 2 neighbours

    def test_triangular_center_has_six(self):
        s = self._terms(pair_constraint="triangular_2d", lattice_shape="3x3", pin_site=5)
        assert s.count("A_5") == 6

    def test_pin_via_custom_edges(self):
        s = self._terms(pair_constraint="custom", edges="(1,2),(2,3),(2,4)", pin_site=2)
        assert s.count("A_2") == 3  # site 2 connects to 1,3,4

    def test_isolated_site_rejected(self):
        from notebook.runtime.backend_handler import build_neighbor_sum_terms
        _, e = build_neighbor_sum_terms("A_n*B_j", "n", "j", None, "open",
                                        "custom", None, "(1,2)", 9)
        assert e and "no neighbours" in e


class TestTwoIndexPairSymbol:
    """Multi-index substitution: a two-index pair symbol such as a molecular
    distance r_{ij} gets BOTH indices substituted, so the full Coulomb sum
    Σ_{i<j} q_i q_j / r_{ij} expands.  r_{ij} / r_i_j / r_{i,j} are equivalent."""

    @pytest.fixture(scope="class")
    def meng(self):
        from notebook.models.session import SessionMode as _SM
        with NotebookEngine(title="coulomb", mode=_SM.standard_symbolic) as e:
            e.start()
            yield e

    def _run(self, meng, body):
        _, v = meng.run_operation("expand_neighbor_sum",
            {"body": body, "index1": "i", "index2": "j",
             "n_sites": 3, "pair_constraint": "all_pairs"})
        return v.replace(" ", "")

    def test_coulomb_braced_adjacent(self, meng):
        flat = self._run(meng, "q_i*q_j/r_{ij}")
        assert "r(1,2)" in flat and "r(1,3)" in flat and "r(2,3)" in flat
        assert "(q(1)*q(2))" in flat

    def test_coulomb_chained(self, meng):
        flat = self._run(meng, "q_i*q_j/r_i_j")
        assert "r(1,2)" in flat and "r(2,3)" in flat

    def test_coulomb_comma_braced(self, meng):
        flat = self._run(meng, "q_i*q_j/r_{i,j}")
        assert "r(1,2)" in flat and "r(1,3)" in flat

    def test_bond_dependent_coupling(self, meng):
        # a pair-indexed coupling J_{ij} is also substituted on both indices
        flat = self._run(meng, "J_{ij}*s_i*s_j")
        assert "J(1,2)" in flat and "J(2,3)" in flat


class TestTwoDLattices:
    """2D lattices are just another registered bond generator: square_2d /
    triangular_2d, sized by an Lx×Ly lattice_shape (or √n_sites for a square).
    Sites are mapped to a linear index, so the body notation is unchanged."""

    def _count_bonds(self, engine, **kw):
        from notebook.runtime.backend_handler import build_neighbor_sum_terms
        s, e = build_neighbor_sum_terms(
            kw.get("body", "Sz_i*Sz_j"), "i", "j",
            kw.get("n_sites"), kw.get("boundary", "open"),
            kw["pair_constraint"], kw.get("lattice_shape"))
        assert e is None, e
        return s.count("(Sz") if s else 0

    def test_square_open_3x3_bond_count(self, engine):
        # 3x3 open: 3*2 horizontal + 3*2 vertical = 12 bonds
        _, v = engine.run_operation("expand_neighbor_sum",
            {"body": "Sz_i*Sz_j", "index1": "i", "index2": "j", "n_sites": 9,
             "pair_constraint": "square_2d", "lattice_shape": "3x3"})
        assert v.count("Sᶻ(") == 24  # 12 bonds × 2 operators

    def test_square_torus_3x3_bond_count(self, engine):
        # 3x3 periodic (torus): 9 horizontal + 9 vertical = 18 bonds
        _, v = engine.run_operation("expand_neighbor_sum",
            {"body": "Sz_i*Sz_j", "index1": "i", "index2": "j", "n_sites": 9,
             "pair_constraint": "square_2d", "lattice_shape": "3x3",
             "boundary": "periodic"})
        assert v.count("Sᶻ(") == 36  # 18 bonds × 2

    def test_triangular_open_3x3_bond_count(self, engine):
        # square(12) + (Lx-1)(Ly-1)=4 diagonals = 16 bonds
        _, v = engine.run_operation("expand_neighbor_sum",
            {"body": "Sz_i*Sz_j", "index1": "i", "index2": "j", "n_sites": 9,
             "pair_constraint": "triangular_2d", "lattice_shape": "3x3"})
        assert v.count("Sᶻ(") == 32  # 16 bonds × 2

    def test_square_2x2_grid_structure(self, engine):
        # explicit small grid: bonds (1,2),(3,4) horizontal, (1,3),(2,4) vertical
        _, v = engine.run_operation("expand_neighbor_sum",
            {"body": "Sz_i*Sz_j", "index1": "i", "index2": "j", "n_sites": 4,
             "pair_constraint": "square_2d", "lattice_shape": "2x2"})
        for pair in ("Sᶻ(1) Sᶻ(2)", "Sᶻ(3) Sᶻ(4)", "Sᶻ(1) Sᶻ(3)", "Sᶻ(2) Sᶻ(4)"):
            assert pair in v

    def test_n_squared_sugar_without_shape(self, engine):
        # n_sites=4 with no lattice_shape → 2×2 square
        _, v = engine.run_operation("expand_neighbor_sum",
            {"body": "Sz_i*Sz_j", "index1": "i", "index2": "j", "n_sites": 4,
             "pair_constraint": "square_2d"})
        assert v.count("Sᶻ(") == 8  # 4 bonds × 2

    def test_non_square_without_shape_rejected(self, engine):
        r = engine._handler.build_command("expand_neighbor_sum",
            {"body": "Sz_i*Sz_j", "index1": "i", "index2": "j", "n_sites": 6,
             "pair_constraint": "square_2d"}, {})
        assert not r.ok and "perfect square" in r.error

    def test_bad_lattice_shape_rejected(self, engine):
        r = engine._handler.build_command("expand_neighbor_sum",
            {"body": "Sz_i*Sz_j", "index1": "i", "index2": "j", "n_sites": 9,
             "pair_constraint": "square_2d", "lattice_shape": "3-3"}, {})
        assert not r.ok and "lattice_shape" in r.error

    def test_maxima_square_scalar(self):
        from notebook.models.session import SessionMode as _SM
        with NotebookEngine(title="sq_max", mode=_SM.standard_symbolic) as e:
            e.start()
            _, v = e.run_operation("expand_neighbor_sum",
                {"body": "-J*s_i*s_j", "index1": "i", "index2": "j", "n_sites": 4,
                 "pair_constraint": "square_2d", "lattice_shape": "2x2"})
            assert v.count("J") == 4  # 2×2 square has 4 bonds


class TestHoneycombCylinderCustom:
    """Honeycomb / cylinder boundary / arbitrary-graph (custom) generators —
    all additive on the pluggable registry."""

    def _bonds(self, **kw):
        from notebook.runtime.backend_handler import build_neighbor_sum_terms
        s, e = build_neighbor_sum_terms(
            "Sz_i*Sz_j", "i", "j", kw.get("n_sites"),
            kw.get("boundary", "open"), kw["pair_constraint"],
            kw.get("lattice_shape"), kw.get("edges"))
        assert e is None, e
        # count bond terms = number of '+'-joined groups
        return s.count("(Sz")

    def test_honeycomb_bulk_is_three_coordinated(self):
        from notebook.runtime.backend_handler import _BOND_GENERATORS
        from collections import Counter
        bonds = _BOND_GENERATORS["honeycomb_2d"](16, "open", (4, 4))
        deg = Counter()
        for a, b in bonds:
            deg[a] += 1
            deg[b] += 1
        assert max(deg.values()) == 3  # honeycomb: 3-fold coordination

    def test_cylinder_between_open_and_torus(self):
        op = self._bonds(pair_constraint="square_2d", lattice_shape="3x3", boundary="open")
        cy = self._bonds(pair_constraint="square_2d", lattice_shape="3x3", boundary="cylinder")
        pe = self._bonds(pair_constraint="square_2d", lattice_shape="3x3", boundary="periodic")
        assert op < cy < pe  # cylinder wraps one direction only

    def test_custom_edges_string(self):
        from notebook.runtime.backend_handler import build_neighbor_sum_terms
        s, e = build_neighbor_sum_terms("s_i*s_j", "i", "j", None, "open",
                                        "custom", None, "(1,2),(2,3),(1,3)")
        assert e is None
        flat = s.replace(" ", "")
        assert "(s_1*s_2)" in flat and "(s_2*s_3)" in flat and "(s_1*s_3)" in flat

    def test_custom_edges_list_julia(self, engine):
        _, v = engine.run_operation("expand_neighbor_sum",
            {"body": "Sz_i*Sz_j", "index1": "i", "index2": "j",
             "pair_constraint": "custom", "edges": [(1, 2), (2, 4), (1, 4)]})
        assert "Sᶻ(1) Sᶻ(2)" in v and "Sᶻ(2) Sᶻ(4)" in v and "Sᶻ(1) Sᶻ(4)" in v

    def test_custom_without_edges_rejected(self, engine):
        r = engine._handler.build_command("expand_neighbor_sum",
            {"body": "Sz_i*Sz_j", "index1": "i", "index2": "j",
             "pair_constraint": "custom", "n_sites": 4}, {})
        assert not r.ok and "edges" in r.error

    def test_bad_boundary_rejected(self, engine):
        r = engine._handler.build_command("expand_neighbor_sum",
            {"body": "Sz_i*Sz_j", "index1": "i", "index2": "j", "n_sites": 9,
             "pair_constraint": "square_2d", "lattice_shape": "3x3",
             "boundary": "klein"}, {})
        assert not r.ok and "boundary" in r.error
