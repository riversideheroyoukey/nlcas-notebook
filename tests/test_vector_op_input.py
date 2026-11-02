"""Vector-operator input computes: nabla notation AND named unary operators.

`\\nabla \\cdot E`, `div E`, `div(E)`, `\\nabla \\times E`, `curl E`, `rot E`,
`\\nabla f`, `grad f`, `\\nabla^2 f`, `laplacian f` are dispatched to the
divergence/curl/gradient/scalar_laplacian operation so they COMPUTE — instead of
registering an unevaluated `nabla . E` / `div(E)`.  (Root cause + the named unary
operators div/rot/curl/grad requested for S-f.)
"""
import asyncio

from ui import notebook_bridge as nb
from ui.interpret_pipeline import run_interpret_pipeline, _try_single_vec_op
from ui.notebook_bridge import InterpretRequest, InputToken
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operation_validator import OperationValidator


def test_recognizer_maps_all_forms():
    cases = {
        r"\nabla \cdot E": ("divergence", "E"), "div E": ("divergence", "E"),
        "div(E)": ("divergence", "E"), "divergence E": ("divergence", "E"),
        r"\nabla \times E": ("curl", "E"), "curl E": ("curl", "E"),
        "curl(E)": ("curl", "E"), "rot E": ("curl", "E"),
        r"\nabla f": ("gradient", "f"), "grad f": ("gradient", "f"),
        r"\nabla^2 f": ("scalar_laplacian", "f"), "laplacian f": ("scalar_laplacian", "f"),
    }
    for text, (op, field) in cases.items():
        r = _try_single_vec_op(text)
        assert r is not None and r.operation == op and r.args["expr"] == field, text
    # must NOT match a bare scalar/number or unrelated text
    assert _try_single_vec_op("x + 1") is None
    assert _try_single_vec_op("divergence theorem の説明") is None


def _result(domains, setup, command):
    nb.start_engine(title="vop", mode="standard", language="ja", domains=domains)
    try:
        eng = nb.get_engine()
        sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                            domains=domains, primary_backend="maxima")
        setup(eng)
        r = asyncio.get_event_loop().run_until_complete(
            run_interpret_pipeline(
                InterpretRequest(tokens=[InputToken(type="text", value=command)]),
                eng, sup))
        return r
    finally:
        nb.stop_engine()


def _plane_wave_E(eng):
    eng.run_operation("define_vector_field",
                      {"name": "E", "coordinate_system": "cartesian",
                       "natural_vars": ["x", "y", "z"]})
    for eq in ("E_x = 0", "E_y = cos(k*x - w*t)", "E_z = 0"):
        eng.register_expression(eq)


def test_divergence_forms_compute_zero():
    for cmd in (r"\nabla \cdot E", "div E", "div(E)"):
        r = _result(["electromagnetism"], _plane_wave_E, cmd)
        assert r["status"] == "completed" and r["operation"] == "divergence"
        assert r.get("canonical", "").strip() == "0", (cmd, r)


def test_curl_forms_compute_the_vector():
    for cmd in (r"\nabla \times E", "curl E", "rot E"):
        r = _result(["electromagnetism"], _plane_wave_E, cmd)
        assert r["status"] == "completed" and r["operation"] == "curl"
        c = r.get("canonical", "").replace(" ", "")
        assert c.startswith("[0,0,") and "k*sin(k*x-t*w)" in c, (cmd, r)


def test_gradient_and_laplacian_forms_compute():
    def _scalar_f(eng):
        eng.run_operation("define_scalar_field",
                          {"name": "f", "coordinate_system": "cartesian",
                           "natural_vars": ["x", "y", "z"]})
        eng.register_expression("f = x^2 + y^2 + z^2")

    for cmd in (r"\nabla f", "grad f"):
        r = _result(["electromagnetism"], _scalar_f, cmd)
        assert r["status"] == "completed" and r["operation"] == "gradient"
        assert r.get("canonical", "").replace(" ", "") == "[2*x,2*y,2*z]", (cmd, r)
    for cmd in (r"\nabla^2 f", "laplacian f"):
        r = _result(["electromagnetism"], _scalar_f, cmd)
        assert r["status"] == "completed" and r["operation"] == "scalar_laplacian"
        assert r.get("canonical", "").strip() == "6", (cmd, r)


def test_div_of_plain_scalar_not_intercepted():
    # `div t` where t is just a scalar must NOT dispatch divergence (type error);
    # the strict guard requires a vector field for div/curl.
    def _scalar_t(eng):
        eng.run_operation("define_scalar_parameter", {"name": "t"})
    r = _result([], _scalar_t, "div t")
    assert not (r.get("operation") == "divergence" and r.get("status") == "completed")
