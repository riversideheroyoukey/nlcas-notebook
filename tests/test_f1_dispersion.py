"""F-1 stage 3c-fourier: single-band tight-binding dispersion ε(k).

A translation-invariant lattice with one site per unit cell has a closed-form
Bloch dispersion ε(k) = Σ_{ΔR} 2t cos(k·ΔR), built directly from the unit-cell
hopping vectors — no commutator, so it bypasses the opaque-canonical ceiling.
"""
import asyncio
from unittest.mock import MagicMock

import ui.notebook_bridge as nb
from notebook.runtime.backend_handler import (
    build_dispersion_single_band, build_dispersion_multiband,
    build_dispersion_nband, build_dispersion,
    _fold_two_coeff, _abs_coeff, _sq)
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operation_validator import OperationValidator
from ui.interpret_pipeline import (
    InterpretRequest, InputToken, run_interpret_pipeline,
    _parse_dispersion, _bond_sum_coeff)


class TestFoldTwoCoeff:
    def test_negative_hopping(self):
        assert _fold_two_coeff("-t") == "-2t"

    def test_plain_hopping(self):
        assert _fold_two_coeff("t") == "2t"
        assert _fold_two_coeff("J") == "2J"

    def test_unit_and_empty(self):
        assert _fold_two_coeff("") == "2"
        assert _fold_two_coeff("1") == "2"
        assert _fold_two_coeff("-1") == "-2"

    def test_compound_falls_back(self):
        assert _fold_two_coeff("t + u") == r"2 \left( t + u \right)"


class TestBuildDispersion:
    def test_chain(self):
        latex, err = build_dispersion_single_band("-t", "nearest_neighbor")
        assert err is None
        assert latex == r"\varepsilon(k) = -2t \cos k"

    def test_square(self):
        latex, err = build_dispersion_single_band("-t", "square_2d")
        assert err is None
        assert latex == r"\varepsilon(\mathbf{k}) = -2t \left( \cos k_x + \cos k_y \right)"

    def test_triangular(self):
        latex, err = build_dispersion_single_band("-t", "triangular_2d")
        assert err is None
        assert r"\cos(k_x + k_y)" in latex
        assert latex.count(r"\cos") == 3

    def test_square_t_tprime(self):
        latex, err = build_dispersion_single_band("-t", "square_2d", nnn_coeff="-t'")
        assert err is None
        assert latex == (
            r"\varepsilon(\mathbf{k}) = -2t \left( \cos k_x + \cos k_y \right) "
            r"-2t' \left( \cos(k_x + k_y) + \cos(k_x - k_y) \right)")

    def test_chain_t_tprime(self):
        latex, err = build_dispersion_single_band("-t", "nearest_neighbor",
                                                  nnn_coeff="-t'")
        assert err is None
        assert latex == r"\varepsilon(k) = -2t \cos k -2t' \cos(2 k)"

    def test_triangular_t_tprime(self):
        latex, err = build_dispersion_single_band("-t", "triangular_2d",
                                                  nnn_coeff="-t'")
        assert err is None
        assert latex == (
            r"\varepsilon(\mathbf{k}) = "
            r"-2t \left( \cos k_x + \cos k_y + \cos(k_x + k_y) \right) "
            r"-2t' \left( \cos(k_x - k_y) + \cos(2 k_x + k_y) "
            r"+ \cos(k_x + 2 k_y) \right)")

    def test_honeycomb_routes_to_multiband(self):
        # single-band builder rejects it; the dispatcher handles it
        latex, err = build_dispersion_single_band("-t", "honeycomb_2d")
        assert latex is None and "multi-band" in err

    def test_unknown_lattice_error(self):
        latex, err = build_dispersion_single_band("-t", "all_pairs")
        assert latex is None
        assert "translation-invariant" in err


class TestAbsCoeff:
    def test_strips_sign(self):
        assert _abs_coeff("-t") == "t"
        assert _abs_coeff("t") == "t"
        assert _abs_coeff("") == ""
        assert _abs_coeff("-1") == ""


class TestMultiband:
    def test_honeycomb_graphene_dispersion(self):
        latex, err = build_dispersion_multiband("-t", "honeycomb_2d")
        assert err is None
        assert latex == (r"\varepsilon_\pm(\mathbf{k}) = \pm t "
                         r"\sqrt{ 3 + 2\cos k_1 + 2\cos k_2 + 2\cos(k_1 - k_2) }")

    def test_dispatch_routes_honeycomb_to_multiband(self):
        latex, err = build_dispersion("-t", "honeycomb_2d")
        assert err is None
        assert r"\varepsilon_\pm" in latex and r"\sqrt" in latex

    def test_dispatch_routes_chain_to_single_band(self):
        latex, err = build_dispersion("-t", "nearest_neighbor")
        assert err is None
        assert latex == r"\varepsilon(k) = -2t \cos k"

    def test_unknown_multiband_error(self):
        latex, err = build_dispersion_multiband("-t", "square_2d")
        assert latex is None and "multi-band" in err


class TestMultibandNNN:
    def test_honeycomb_with_next_nearest(self):
        latex, err = build_dispersion_multiband("-t", "honeycomb_2d", nnn_coeff="-t'")
        assert err is None
        assert latex == (
            r"\varepsilon_\pm(\mathbf{k}) = "
            r"-2t' \left( \cos k_1 + \cos k_2 + \cos(k_1 - k_2) \right) "
            r"\pm t \sqrt{ 3 + 2\cos k_1 + 2\cos k_2 + 2\cos(k_1 - k_2) }")

    def test_dispatch_threads_nnn(self):
        latex, err = build_dispersion("-t", "honeycomb_2d", nnn_coeff="-t'")
        assert err is None
        assert r"-2t' \left(" in latex and r"\pm t \sqrt" in latex

    def test_no_nnn_is_pure_band(self):
        latex, _ = build_dispersion_multiband("-t", "honeycomb_2d")
        assert r"\left(" not in latex.split(r"\sqrt")[0]  # no diagonal term


class TestMultibandOnsite:
    def test_gapped_dirac_mass_term(self):
        latex, err = build_dispersion_multiband("-t", "honeycomb_2d", onsite=r"\Delta")
        assert err is None
        assert latex == (
            r"\varepsilon_\pm(\mathbf{k}) = \pm \sqrt{ \Delta^2 + t^2 "
            r"\left( 3 + 2\cos k_1 + 2\cos k_2 + 2\cos(k_1 - k_2) \right) }")

    def test_mass_and_nnn_combined(self):
        latex, err = build_dispersion_multiband(
            "-t", "honeycomb_2d", nnn_coeff="-t'", onsite=r"\Delta")
        assert err is None
        # symmetric NNN sits outside as the band average, mass under the root
        assert latex.startswith(r"\varepsilon_\pm(\mathbf{k}) = -2t' \left(")
        assert r"\pm \sqrt{ \Delta^2 + t^2 \left(" in latex

    def test_dispatch_threads_onsite(self):
        latex, err = build_dispersion("-t", "honeycomb_2d", onsite=r"\Delta")
        assert err is None
        assert r"\Delta^2 + t^2" in latex


class TestMultibandPerSublattice:
    def test_distinct_nnn_amplitudes(self):
        latex, err = build_dispersion_multiband(
            "-t", "honeycomb_2d", nnn_coeff="-t_A'", nnn_b="-t_B'")
        assert err is None
        S = r"\left( \cos k_1 + \cos k_2 + \cos(k_1 - k_2) \right)"
        # symmetric part (t_A'+t_B') outside, antisymmetric (t_A'-t_B') under root
        assert latex == (
            rf"\varepsilon_\pm(\mathbf{{k}}) = \left( -t_A' - t_B' \right) {S} "
            rf"\pm \sqrt{{ \left( \left( -t_A' + t_B' \right) {S} \right)^2 "
            r"+ t^2 \left( 3 + 2\cos k_1 + 2\cos k_2 + 2\cos(k_1 - k_2) \right) }")

    def test_per_sublattice_with_mass(self):
        latex, err = build_dispersion_multiband(
            "-t", "honeycomb_2d", nnn_coeff="-t_A'", nnn_b="-t_B'", onsite=r"\Delta")
        assert err is None
        assert r"\left( -t_A' - t_B' \right)" in latex          # band average
        assert r"\right) + \Delta \right)^2" in latex           # mass under root

    def test_dispatch_threads_nnn_b(self):
        latex, err = build_dispersion("-t", "honeycomb_2d",
                                      nnn_coeff="t_A'", nnn_b="t_B'")
        assert err is None
        assert r"\left( t_A' + t_B' \right)" in latex

    def test_combine_coeffs(self):
        from notebook.runtime.backend_handler import _combine_coeffs
        assert _combine_coeffs("-t_A'", "-t_B'", "+") == "-t_A' - t_B'"
        assert _combine_coeffs("-t_A'", "-t_B'", "-") == "-t_A' + t_B'"
        assert _combine_coeffs("t_A'", "t_B'", "+") == "t_A' + t_B'"


class TestNbandStar:
    def test_lieb_flat_plus_dispersive(self):
        latex, err = build_dispersion_nband("-t", "lieb_2d")
        assert err is None
        assert latex == (r"\varepsilon_0 = 0, \quad \varepsilon_\pm(\mathbf{k}) = "
                         r"\pm t \sqrt{ 4 + 2\cos k_x + 2\cos k_y }")

    def test_dispatch_routes_lieb(self):
        latex, err = build_dispersion("-t", "lieb_2d")
        assert err is None
        assert r"\varepsilon_0 = 0" in latex      # flat band

    def test_kagome_flat_plus_dirac(self):
        latex, err = build_dispersion_nband("-t", "kagome_2d")
        assert err is None
        assert latex == (
            r"\varepsilon_{\mathrm{flat}} = 2t, \quad \varepsilon_\pm(\mathbf{k}) = "
            r"-t \left( 1 \pm \sqrt{ 3 + 2\cos k_x + 2\cos k_y "
            r"+ 2\cos(k_x + k_y) } \right)")

    def test_dispatch_routes_kagome(self):
        latex, err = build_dispersion("-t", "kagome_2d")
        assert err is None
        assert r"\varepsilon_{\mathrm{flat}} = 2t" in latex

    def test_unknown_nband_error(self):
        latex, err = build_dispersion_nband("-t", "checkerboard_2d")
        assert latex is None and "3-sublattice" in err


class TestSqHelper:
    """`build_dispersion_ssh` was deleted 2026-08-14 (SCENARIO-MACRO case 6):
    it returned a hardcoded LaTeX string for one named model, and
    `TestCrystalDescriptor::test_ssh_via_bonds` below already asserted that the
    GENERAL bond route produces the same band expression **byte for byte** —
    plus the Bloch matrix, which the canned route did not.  Its interface tests
    went with it; `_sq` stays because the general route uses it.
    """

    def test_sq_helper(self):
        assert _sq("v") == "v^2"
        assert _sq("-t") == "t^2"
        assert _sq("a+b") == r"\left( a+b \right)^2"


class TestCrystalDescriptor:
    def test_ssh_via_bonds(self):
        from notebook.runtime.backend_handler import build_crystal_bloch
        mat, band, err = build_crystal_bloch(
            [(0, 1, (0,), "v"), (0, 1, (-1,), "w")])
        assert err is None
        assert mat == (r"\tilde H(k) = \begin{pmatrix} "
                       r"0 & v + w e^{-i k} \\ v + w e^{i k} & 0 "
                       r"\end{pmatrix}")
        assert band == r"\varepsilon_\pm(k) = \pm \sqrt{ v^2 + w^2 + 2 v w \cos k }"

    def test_single_band_chain(self):
        from notebook.runtime.backend_handler import build_crystal_bloch
        mat, band, err = build_crystal_bloch([(0, 0, (1,), "-t")])
        assert err is None
        assert mat is None                       # scalar, no matrix
        assert band == r"\varepsilon(k) = -2t \cos k"

    def test_2d_mass_via_diagonal(self):
        from notebook.runtime.backend_handler import build_crystal_bloch
        # staggered onsite ±M on A/B, single A–B bond t → gapped 2-band
        mat, band, err = build_crystal_bloch(
            [(0, 0, (0, 0), "M"), (1, 1, (0, 0), "-M"), (0, 1, (0, 0), "-t")])
        assert err is None
        assert band == (r"\varepsilon_\pm(\mathbf{k}) = "
                        r"\frac{ M - M }{2} \pm \sqrt{ "
                        r"\left( \frac{ M + M }{2} \right)^2 + t^2 }")

    def test_three_band_matrix_plus_secular_equation(self):
        from notebook.runtime.backend_handler import build_crystal_bloch
        mat, band, err = build_crystal_bloch(
            [(0, 1, (0,), "a"), (1, 2, (0,), "b"), (0, 2, (0,), "c")])
        assert err is None
        assert mat.startswith(r"\tilde H(k) = \begin{pmatrix}")
        # N≥3: no closed form (Abel–Ruffini); the band equation is the secular det
        assert band == r"\det\!\left(\tilde H(k) - \varepsilon I\right) = 0"

    def test_dimension_mismatch_error(self):
        from notebook.runtime.backend_handler import build_crystal_bloch
        mat, band, err = build_crystal_bloch([(0, 1, (0,), "v"), (0, 1, (0, 0), "w")])
        assert mat is None and band is None and "dimension" in err


class TestBondSumCoeff:
    def test_extracts_prefactor(self):
        assert _bond_sum_coeff("(-t)*(c'_i c_j)") == "-t"

    def test_no_prefactor(self):
        assert _bond_sum_coeff("S^z_i S^z_j") == ""


class TestParseDispersionGuards:
    def test_needs_dispersion_verb(self):
        eng = MagicMock(); eng.expressions = {}; eng.lhs_alias_map = {}
        assert _parse_dispersion("E1 を展開して", eng) is None

    def test_needs_bond_sum_target(self):
        eng = MagicMock()
        ex = MagicMock(); ex.lattice_descriptor = None
        eng.expressions = {"E1": ex}; eng.lhs_alias_map = {}
        assert _parse_dispersion("E1 の分散関係を求めて", eng) is None


def _sup():
    s = MagicMock(spec=LLMSupervisor)
    s._v = OperationValidator()
    s.needs_expression_injection.return_value = False
    return s


def _run(engine, text):
    req = InterpretRequest(tokens=[InputToken(type="text", value=text)])
    return asyncio.run(run_interpret_pipeline(req, engine, _sup()))


class TestDispersionPipeline:
    def test_chain_dispersion_by_name(self):
        nb.start_engine(title="Dchain", mode="standard", language="ja")
        try:
            eng = nb.get_engine()
            r1 = _run(eng, r"H = -t \sum_{\langle i j \rangle} c'_i c_j")
            assert r1["status"] == "registered"
            r2 = _run(eng, "H の分散関係を求めて")
            assert r2["status"] == "completed"
            assert r2["operation"] == "dispersion_relation"
            latex = eng.expressions[r2["expr_id"]].latex
            assert latex == r"\varepsilon(k) = -2t \cos k"
        finally:
            nb.stop_engine()

    def test_square_lattice_override(self):
        nb.start_engine(title="Dsq", mode="standard", language="ja")
        try:
            eng = nb.get_engine()
            _run(eng, r"H = -t \sum_{\langle i j \rangle} c'_i c_j")
            r2 = _run(eng, "H の正方格子で分散関係を求めて")
            assert r2["status"] == "completed"
            latex = eng.expressions[r2["expr_id"]].latex
            assert r"\cos k_x + \cos k_y" in latex
            assert "-2t" in latex
        finally:
            nb.stop_engine()

    def test_square_t_tprime_command(self):
        nb.start_engine(title="Dsqn", mode="standard", language="ja")
        try:
            eng = nb.get_engine()
            _run(eng, r"H = -t \sum_{\langle i j \rangle} c'_i c_j")
            r2 = _run(eng, "H の正方格子で次近接 -t' を含めて分散関係を求めて")
            assert r2["status"] == "completed"
            latex = eng.expressions[r2["expr_id"]].latex
            assert r"-2t' \left( \cos(k_x + k_y)" in latex
        finally:
            nb.stop_engine()

    def test_honeycomb_two_band_dispersion(self):
        nb.start_engine(title="Dhc", mode="standard", language="ja")
        try:
            eng = nb.get_engine()
            _run(eng, r"H = -t \sum_{\langle i j \rangle} c'_i c_j")
            r2 = _run(eng, "H のハニカム格子で分散関係を求めて")
            assert r2["status"] == "completed"
            latex = eng.expressions[r2["expr_id"]].latex
            assert r"\varepsilon_\pm" in latex          # two bands
            assert r"\sqrt{ 3 + 2\cos k_1" in latex
        finally:
            nb.stop_engine()

    def test_honeycomb_with_nnn_command(self):
        nb.start_engine(title="Dhcn", mode="standard", language="ja")
        try:
            eng = nb.get_engine()
            _run(eng, r"H = -t \sum_{\langle i j \rangle} c'_i c_j")
            r2 = _run(eng, "H のハニカム格子で次近接 -t' を含めて分散関係を求めて")
            assert r2["status"] == "completed"
            latex = eng.expressions[r2["expr_id"]].latex
            assert r"-2t' \left(" in latex              # diagonal term present
            assert r"\pm t \sqrt" in latex
        finally:
            nb.stop_engine()

    def test_honeycomb_with_mass_command(self):
        nb.start_engine(title="Dhcm", mode="standard", language="ja")
        try:
            eng = nb.get_engine()
            _run(eng, r"H = -t \sum_{\langle i j \rangle} c'_i c_j")
            r2 = _run(eng, r"H のハニカム格子で質量 \Delta を含めて分散関係を求めて")
            assert r2["status"] == "completed"
            latex = eng.expressions[r2["expr_id"]].latex
            assert r"\sqrt{ \Delta^2 + t^2 \left(" in latex   # gapped Dirac
        finally:
            nb.stop_engine()

    def test_lieb_lattice_command(self):
        nb.start_engine(title="Dlieb", mode="standard", language="ja")
        try:
            eng = nb.get_engine()
            _run(eng, r"H = -t \sum_{\langle i j \rangle} c'_i c_j")
            r2 = _run(eng, "H の Lieb 格子で分散関係を求めて")
            assert r2["status"] == "completed"
            latex = eng.expressions[r2["expr_id"]].latex
            assert r"\varepsilon_0 = 0" in latex                       # flat band
            assert r"\pm t \sqrt{ 4 + 2\cos k_x + 2\cos k_y }" in latex
        finally:
            nb.stop_engine()

    def test_crystal_bonds_command(self):
        nb.start_engine(title="Dcry", mode="standard", language="ja")
        try:
            eng = nb.get_engine()
            r = _run(eng, "次のボンドで結晶の分散関係を求めて: (0,1,(0),v),(0,1,(-1),w)")
            assert r["status"] == "completed"
            # both the Bloch matrix and the bands are registered
            latexes = [x["latex"] for x in r["registered_expressions"]]
            assert any(r"\tilde H(k) = \begin{pmatrix}" in x for x in latexes)
            assert any(r"\pm \sqrt{ v^2 + w^2 + 2 v w \cos k }" in x for x in latexes)
        finally:
            nb.stop_engine()

    def test_the_ssh_model_is_the_bond_route_with_custom_symbols(self):
        """`SSH 模型の分散関係を求めて` was its own NL route into a hardcoded
        LaTeX string, deleted 2026-08-14 (SCENARIO-MACRO case 6).  Everything it
        offered — including renaming the two amplitudes (`v=t_1, w=t_2`) — is the
        general bond route with those amplitudes written in, and that route also
        returns the Bloch matrix.  Measured byte-identical before deleting;
        pinned here in BOTH languages because the deleted route was JA-only.
        """
        for lang, cmd in (
                ("ja", "次のボンドで結晶の分散関係を求めて: (0,1,(0),t_1),(0,1,(-1),t_2)"),
                ("en", "crystal dispersion from bonds: (0,1,(0),t_1),(0,1,(-1),t_2)")):
            nb.start_engine(title=f"Dssh_{lang}", mode="standard", language=lang)
            try:
                eng = nb.get_engine()
                r = _run(eng, cmd)
                assert r["status"] == "completed", (lang, r)
                latexes = [x["latex"] for x in r["registered_expressions"]]
                assert any(r"\pm \sqrt{ t_1^2 + t_2^2 + 2 t_1 t_2 \cos k }" in x
                           for x in latexes), (lang, latexes)
                assert any(r"\tilde H(k) = \begin{pmatrix}" in x for x in latexes), lang
            finally:
                nb.stop_engine()

    def test_kagome_lattice_command(self):
        nb.start_engine(title="Dkag", mode="standard", language="ja")
        try:
            eng = nb.get_engine()
            _run(eng, r"H = -t \sum_{\langle i j \rangle} c'_i c_j")
            r2 = _run(eng, "H のカゴメ格子で分散関係を求めて")
            assert r2["status"] == "completed"
            latex = eng.expressions[r2["expr_id"]].latex
            assert r"\varepsilon_{\mathrm{flat}} = 2t" in latex        # flat band
            assert r"-t \left( 1 \pm \sqrt{ 3 + 2\cos k_x" in latex    # Dirac bands
        finally:
            nb.stop_engine()

    def test_honeycomb_per_sublattice_nnn_command(self):
        nb.start_engine(title="Dhcab", mode="standard", language="ja")
        try:
            eng = nb.get_engine()
            _run(eng, r"H = -t \sum_{\langle i j \rangle} c'_i c_j")
            r2 = _run(eng, r"H のハニカム格子で次近接 -t_A', -t_B' を含めて分散関係を求めて")
            assert r2["status"] == "completed"
            latex = eng.expressions[r2["expr_id"]].latex
            assert r"\left( -t_A' - t_B' \right)" in latex   # band average
            assert r"\left( -t_A' + t_B' \right)" in latex   # under the root
        finally:
            nb.stop_engine()


class TestUnrecognizedLatticeNameGuard:
    """An explicit but unsupported lattice name (e.g. "ダイヤモンド格子"/"diamond
    lattice") must be rejected with a clear error, not silently fall back to
    the underlying bond sum's default pair_constraint (normally the plain 1D
    chain) — see the silent-lattice-fallback bugfix."""

    def test_unrecognized_jp_lattice_name_errors(self):
        nb.start_engine(title="Dunk1", mode="standard", language="ja")
        try:
            eng = nb.get_engine()
            _run(eng, r"H = -t \sum_{\langle i j \rangle} c'_i c_j")
            r2 = _run(eng, "H のダイヤモンド格子で分散関係を求めて")
            assert r2["status"] == "failed"
            assert "unrecognized lattice name" in r2["error"]
            assert "diamond" in r2["error"].lower() or "ダイヤモンド" in r2["error"]
        finally:
            nb.stop_engine()

    def test_unrecognized_en_lattice_name_errors(self):
        # English input → English session (nl_patterns are looked up by session
        # language, docs/28 §3.1).
        nb.start_engine(title="Dunk2", mode="standard", language="en")
        try:
            eng = nb.get_engine()
            _run(eng, r"H = -t \sum_{\langle i j \rangle} c'_i c_j")
            r2 = _run(eng, "Find the dispersion relation of H on a diamond lattice")
            assert r2["status"] == "failed"
            assert "unrecognized lattice name" in r2["error"]
            assert "diamond" in r2["error"].lower()
        finally:
            nb.stop_engine()

    def test_no_lattice_name_still_defaults_without_error(self):
        nb.start_engine(title="Dunk3", mode="standard", language="ja")
        try:
            eng = nb.get_engine()
            _run(eng, r"H = -t \sum_{\langle i j \rangle} c'_i c_j")
            r2 = _run(eng, "H の分散関係を求めて")
            assert r2["status"] == "completed"
            latex = eng.expressions[r2["expr_id"]].latex
            assert latex == r"\varepsilon(k) = -2t \cos k"
        finally:
            nb.stop_engine()

    def test_mass_keyword_does_not_false_positive(self):
        nb.start_engine(title="Dunk4", mode="standard", language="ja")
        try:
            eng = nb.get_engine()
            _run(eng, r"H = -t \sum_{\langle i j \rangle} c'_i c_j")
            r2 = _run(eng, r"H のハニカム格子で質量 \Delta を含めて分散関係を求めて")
            assert r2["status"] == "completed"
        finally:
            nb.stop_engine()

    def test_sublattice_potential_keyword_does_not_false_positive(self):
        nb.start_engine(title="Dunk5", mode="standard", language="ja")
        try:
            eng = nb.get_engine()
            _run(eng, r"H = -t \sum_{\langle i j \rangle} c'_i c_j")
            r2 = _run(eng, r"H のハニカム格子で副格子ポテンシャル m0 を含めて分散関係を求めて")
            assert r2["status"] == "completed"
        finally:
            nb.stop_engine()

    def test_all_five_known_lattice_names_still_work(self):
        commands = {
            "H の正方格子で分散関係を求めて": r"\cos k_x + \cos k_y",
            "H の三角格子で分散関係を求めて": r"\cos(k_x + k_y)",
            "H のハニカム格子で分散関係を求めて": r"\varepsilon_\pm(\mathbf{k})",
            "H のカゴメ格子で分散関係を求めて": r"\varepsilon_{\mathrm{flat}} = 2t",
            "H の Lieb 格子で分散関係を求めて": r"\varepsilon_0 = 0",
        }
        for text, expect_fragment in commands.items():
            nb.start_engine(title="Dunk6", mode="standard", language="ja")
            try:
                eng = nb.get_engine()
                _run(eng, r"H = -t \sum_{\langle i j \rangle} c'_i c_j")
                r2 = _run(eng, text)
                assert r2["status"] == "completed", (text, r2)
                latex = eng.expressions[r2["expr_id"]].latex
                assert expect_fragment in latex, (text, latex)
            finally:
                nb.stop_engine()
