"""S-f scenario: given E,B (plane EM wave), Maxwell's equations are consistent.

Guards docs/scenario_maxwell_consistency.md and — importantly — the documented
recipe for giving explicit components to a vector field: define_vector_field
registers abstract components, then each component is bound via an H-1 equation
(E_y = cos(k x - w t)).  div/curl then expand using those bound values.

Plane wave (E0=1): E=(0, cos(kx-wt), 0), B=(0, 0, cos(kx-wt)/c).
"""
from notebook.engine import NotebookEngine
from notebook.models.session import SessionMode


def _session():
    return NotebookEngine(
        title="s_f maxwell",
        mode=SessionMode.standard_symbolic,
        coordinate_system="cartesian",
    )


def _setup_plane_wave(e):
    e.run_operation("define_vector_field",
                    {"name": "E", "coordinate_system": "cartesian",
                     "natural_vars": ["x", "y", "z"]})
    e.run_operation("define_vector_field",
                    {"name": "B", "coordinate_system": "cartesian",
                     "natural_vars": ["x", "y", "z"]})
    # Explicit components via H-1 equation form (the recipe under test).
    for eq in ("E_x = 0", "E_y = cos(k*x - w*t)", "E_z = 0",
               "B_x = 0", "B_y = 0", "B_z = cos(k*x - w*t)/c"):
        e.register_expression(eq)


def test_divergences_vanish():
    with _session() as e:
        e.start()
        _setup_plane_wave(e)
        _, div_e = e.run_operation("divergence", {"expr": "E"})
        _, div_b = e.run_operation("divergence", {"expr": "B"})
        assert div_e.strip() == "0"   # ∇·E = 0 (Gauss, no charge)
        assert div_b.strip() == "0"   # ∇·B = 0 (no monopoles)


def test_curls_are_the_plane_wave_values():
    with _session() as e:
        e.start()
        _setup_plane_wave(e)
        _, curl_e = e.run_operation("curl", {"expr": "E"})
        _, curl_b = e.run_operation("curl", {"expr": "B"})
        ce = curl_e.replace(" ", "")
        cb = curl_b.replace(" ", "")
        # ∇×E = [0, 0, -k sin(kx-wt)]
        assert ce.startswith("[0,0,") and "k*sin(k*x-t*w)" in ce
        # ∇×B = [0, k sin(kx-wt)/c, 0]
        assert cb.endswith(",0]") and "sin(k*x-t*w))/c" in cb


def test_time_derivative_componentwise_and_reassembly():
    """∂B/∂t is built component-by-component (∂/∂t is NOT a vector operator) and
    reassembled into a vector that is usable downstream — the accepted workflow.
    """
    with _session() as e:
        e.start()
        _setup_plane_wave(e)
        # per-component time derivative (the vector itself can't be t-diff'd)
        bz = e.lhs_alias_map["B_z"]
        _, dbz = e.run_operation("differentiate", {"expr": bz, "var": "t"})
        assert dbz.replace(" ", "") == "(w*sin(k*x-t*w))/c"
        # reassemble ∂B/∂t as a vector and use it (divergence)
        e.run_operation("define_vector_field",
                        {"name": "dBdt", "coordinate_system": "cartesian",
                         "natural_vars": ["x", "y", "z"]})
        for eq in ("dBdt_x = 0", "dBdt_y = 0", "dBdt_z = w*sin(k*x - w*t)/c"):
            e.register_expression(eq)
        _, div_dbdt = e.run_operation("divergence", {"expr": "dBdt"})
        assert div_dbdt.strip() == "0"


def test_faraday_z_component_equation_gives_dispersion_relation():
    """Faraday is compared component-by-component: the z-component equation
    (curl E)_z = (−∂B/∂t)_z solves to the dispersion relation ω = ck."""
    with _session() as e:
        e.start()
        eq = e.register_expression("-k*sin(k*x - w*t) = -w*sin(k*x - w*t)/c")
        _, sol = e.run_operation("solve_equation", {"equation": eq.display_id, "var": "w"})
        assert "w=c*k" in sol.replace(" ", "")
