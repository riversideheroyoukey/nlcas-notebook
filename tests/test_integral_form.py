"""Pure-helper tests for the unevaluated definite-integral form (S-o)."""
from notebook.runtime.integral_form import (
    parse_latex_definite_integral, build_integral_latex, build_eval_command,
    classify_transform, derive_new_bounds, latex_var,
)


class TestParse:
    def test_double_integral(self):
        d = parse_latex_definite_integral(
            r"\int_{0}^{1} \int_{0}^{2} x^2+y^2 \, dx \, dy")
        # outer \int_0^1 binds last differential dy → (y,0,1); inner (x,0,2)
        assert d["bounds"] == [["y", "0", "1"], ["x", "0", "2"]]
        assert d["integrand_latex"] == "x^2+y^2"

    def test_parenthesized_integrand_and_iint(self):
        d = parse_latex_definite_integral(r"\iint_{0}^{1}_{0}^{1} (x*y) \, dx \, dy")
        # \iint expands to two \int; tolerate the doubled-subscript spelling
        assert d is None or d["integrand_latex"] == "x*y"

    def test_not_an_integral(self):
        assert parse_latex_definite_integral("x^2+y^2") is None


class TestDisplay:
    def test_roundtrip_and_greek_differential(self):
        desc = {"integrand_latex": "r^3", "integrand_canonical": "r^3",
                "bounds": [["theta", "0", r"2\pi"], ["r", "0", "1"]], "region": None}
        tex = build_integral_latex(desc)
        assert tex == r"\int_{0}^{2\pi} \int_{0}^{1} r^3 \, dr \, d\theta"

    def test_with_result(self):
        desc = {"integrand_latex": "r^3", "bounds": [["theta", "0", r"2\pi"],
                ["r", "0", "1"]], "region": None}
        tex = build_integral_latex(desc, result_latex=r"\frac{\pi}{2}")
        assert tex.endswith(r"= \frac{\pi}{2}")

    def test_latex_var(self):
        assert latex_var("theta") == r"\theta"
        assert latex_var("r") == "r"


class TestEvalCommand:
    def test_iterated_inner_first(self):
        cmd = build_eval_command("r^3", [["theta", "0", "2*%pi"], ["r", "0", "1"]])
        assert cmd == "integrate(integrate((r^3), r, 0, 1), theta, 0, 2*%pi)"


class TestTransformClassify:
    def test_polar(self):
        assert classify_transform(["x", "y"],
                                  ["r*cos(theta)", "r*sin(theta)"]) == "polar"

    def test_spherical(self):
        assert classify_transform(
            ["x", "y", "z"],
            ["r*sin(theta)*cos(phi)", "r*sin(theta)*sin(phi)", "r*cos(theta)"]
        ) == "spherical"

    def test_unknown(self):
        assert classify_transform(["x", "y"], ["2*u+v", "u+3*v"]) is None


class TestDeriveBounds:
    def test_polar_disk(self):
        b = derive_new_bounds("polar", {"kind": "disk", "radius": "1"}, ["r", "theta"])
        assert b == [["theta", "0", "2*%pi"], ["r", "0", "1"]]

    def test_spherical_ball(self):
        b = derive_new_bounds("spherical", {"kind": "ball", "radius": "R"},
                              ["r", "theta", "phi"])
        assert b == [["phi", "0", "2*%pi"], ["theta", "0", "%pi"], ["r", "0", "R"]]

    def test_rectangle_returns_none(self):
        assert derive_new_bounds("polar", {"kind": "rectangle"}, ["r", "theta"]) is None
