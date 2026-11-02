r"""The Laplace transform PAIR declaration, and the S-e transfer function.

**Rewritten 2026-08-15 (SCENARIO-MACRO case 7).**  `laplace_by_definition` was
deleted as a macro — measured, both of its scenarios reach their published
results through general operations — and with it went every test that asserted
its interface: the explicit-function branch (now `integrate_definite`), the
by-parts branch (now a theorem the researcher writes), and the linearity split
(now nothing at all, so there is no code path left that could distribute
𝓛{x²} to X² or pass a term through untransformed).

Two things survive, for different reasons:

* `define_laplace_transform` — a DECLARATION of the transform pair.  It is the
  researcher's naming (`X` is the image of `x`), which is not knowledge the CAS
  holds under that symbol; `inverse_laplace`'s result label reads it.  Whether
  the notebook should hold it at all is open_items LAPLACE-PAIR-ANCHOR.
* the S-e transfer function itself, rewritten onto the route the scenario now
  takes: the researcher writes the derivative images as ordinary expressions,
  composes them into the s-domain equation by E-id (H-2), states the zero
  initial conditions as equations, and solves.  Its asserted VALUES are
  unchanged, which is what makes it evidence that the macro was one.
"""
import asyncio

import ui.notebook_bridge as nb
from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operation_validator import OperationValidator


def _sup(domains=None):
    return LLMSupervisor(validator=OperationValidator(), llm_client=None,
                         domains=domains or [], primary_backend="maxima")


def _run(eng, sup, value, ttype="text"):
    tok = (InputToken(type="math", value=value, latex=value, canonical="")
           if ttype == "math" else InputToken(type="text", value=value))
    return asyncio.get_event_loop().run_until_complete(
        run_interpret_pipeline(InterpretRequest(tokens=[tok]), eng, sup))


def _eid(r):
    return r.get("expr_id") or r["registered_expressions"][-1]["expr_id"]


class TestTransformPairDeclaration:
    """"q のラプラス変換を Q とおく" / "Let Q be the Laplace transform of q".

    English word order names the CAPITAL (s-domain) symbol first and the base
    last — the reverse of the Japanese — which is why both languages are pinned.
    """

    def test_declaration_ja(self):
        nb.start_engine(title="lapdefine_ja", mode="standard", language="ja",
                        domains=[])
        eng = nb.get_engine()
        try:
            _run(eng, _sup(), "q を時間依存関数として定義する")
            d = _run(eng, _sup(), "q のラプラス変換を Q とおく")
            assert d["operation"] == "define_laplace_transform"
            assert eng.expressions[_eid(d)].latex == \
                r"Q(s) = \int_0^{\infty} q(t)\, e^{-s t}\, dt"
        finally:
            nb.stop_engine()

    def test_declaration_en(self):
        nb.start_engine(title="lapdefine_en", mode="standard", language="en",
                        domains=[])
        eng = nb.get_engine()
        try:
            _run(eng, _sup(), "q を時間依存関数として定義する")
            d = _run(eng, _sup(), "Let Q be the Laplace transform of q")
            assert d["operation"] == "define_laplace_transform"
            assert eng.expressions[_eid(d)].latex == \
                r"Q(s) = \int_0^{\infty} q(t)\, e^{-s t}\, dt"
        finally:
            nb.stop_engine()


class TestSeTransferFunction:
    """The S-e RLC transfer function, general operations only.

    The researcher writes 𝓛{q̇} and 𝓛{q̈} — a theorem, with no choice to make in
    it — composes them into the s-domain equation, states the zero initial
    conditions the transfer function is DEFINED at, solves for Q and substitutes
    into H = Q/V.  Nothing here is Laplace-specific except the pair declaration.
    """

    def test_rlc_transfer_function(self):
        nb.start_engine(title="se", mode="standard", language="ja",
                        domains=["control_theory"])
        eng = nb.get_engine()
        sup = _sup(["control_theory"])

        def run(v, tt="text"):
            return _run(eng, sup, v, tt)
        try:
            run("q を時間依存関数として定義する")
            run("v を時間依存関数として定義する")
            d = run("q のラプラス変換を Q とおく")
            assert d["operation"] == "define_laplace_transform"
            run("v のラプラス変換を V とおく")

            # ① the transform rules, written by the researcher
            d1 = _eid(run(r"s Q - q_0", "math"))
            d2 = _eid(run(r"s^2 Q - s q_0 - u_0", "math"))

            # ② the s-domain equation, composing them by E-id (H-2)
            eq = _eid(run(rf"L {d2} + R {d1} + Q/C = V", "math"))
            eqx = eng.expressions[eq]
            assert eqx.canonical.replace(" ", "") == \
                "L*(s^2*Q-s*q_0-u_0)+R*(s*Q-q_0)+Q/C=V"
            assert "laplace" not in eqx.canonical      # no builtin transform

            # ③ the zero initial conditions the transfer function is defined at
            run(r"q_0 = 0", "math")
            run(r"u_0 = 0", "math")

            # ④ the transfer function's definition
            hid = _eid(run(r"H = Q/V", "math"))

            # ⑤ solve for Q, then substitute into H
            r5 = run(f"{eq} を Q について解いて")
            assert r5["operation"] == "solve_equation"
            qx = eng.expressions[_eid(r5)]
            assert qx.canonical.replace(" ", "") == "(C*V)/(C*L*s^2+C*R*s+1)"

            r6 = run(f"{_eid(r5)} を {hid} に代入して")
            x6 = eng.expressions[_eid(r6)]
            assert x6.canonical.replace(" ", "") == "C/(C*L*s^2+C*R*s+1)"
            assert x6.latex.startswith("H =")
        finally:
            nb.stop_engine()

    def test_ic_symbols_are_gone_from_the_solution(self):
        """The zero conditions must actually reach the solved equation.

        They are ordinary H-1 definitions now, so the CAS applies them when the
        solve runs (P2).  If they did not, `q_0` would still be in the answer —
        the transfer function would silently be the wrong quantity.
        """
        nb.start_engine(title="se_ic", mode="standard", language="ja",
                        domains=["control_theory"])
        eng = nb.get_engine()
        sup = _sup(["control_theory"])
        try:
            d1 = _eid(_run(eng, sup, r"s Q - q_0", "math"))
            d2 = _eid(_run(eng, sup, r"s^2 Q - s q_0 - u_0", "math"))
            eq = _eid(_run(eng, sup, rf"L {d2} + R {d1} + Q/C = V", "math"))
            _run(eng, sup, r"q_0 = 0", "math")
            _run(eng, sup, r"u_0 = 0", "math")
            r = _run(eng, sup, f"{eq} を Q について解いて")
            canon = eng.expressions[_eid(r)].canonical.replace(" ", "")
            assert "q_0" not in canon and "u_0" not in canon, canon
        finally:
            nb.stop_engine()
