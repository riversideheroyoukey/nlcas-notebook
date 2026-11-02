r"""A declared number domain has to reach the CAS, not just the notebook.

COMPLEX-DECL-NOT-SENT.  `z を複素数とする` recorded `domain="complex"` on the
Quantity, put an acceptance in the timeline, and sent the CAS nothing — so the
researcher declared a symbol, was told it had been registered, and got a
different answer than they asked for.  **A declaration that is shown accepted
and changes nothing is worse than one that is refused.**

Measured 2026-08-16, three domains were recorded and none was sent, and all
three change answers::

    「z を複素数とする」  ratsimp(conjugate(z)*z) → z^2        (the conjugate is GONE)
    「w を実数とする」    ratsimp(conjugate(w)*w) → w^2        (right, but by luck —
                                                    Maxima's default, not the
                                                    declaration)
    「n を整数とする」    ratsimp(sin(n*%pi))     → sin(%pi*n) (it is 0)

The repair is a sibling of `assume_sign`, which had already made exactly this
argument for the SIGN half of the same declaration: a constraint is not display
metadata.  The domain half had simply never been wired.
"""
import asyncio

import pytest

import ui.notebook_bridge as nb
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operation_validator import OperationValidator
from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline


def _sup():
    return LLMSupervisor(validator=OperationValidator(), llm_client=None,
                         domains=[], primary_backend="maxima")


def _run(eng, sup, value, ttype="text"):
    tok = (InputToken(type="math", value=value, latex=value, canonical="")
           if ttype == "math" else InputToken(type="text", value=value))
    return asyncio.get_event_loop().run_until_complete(
        run_interpret_pipeline(InterpretRequest(tokens=[tok]), eng, sup))


class TestTheDeclarationReachesTheCas:
    @pytest.mark.parametrize("declaration,name,feature", [
        ("z を複素数とする", "z", "complex"),
        ("w を実数とする",   "w", "real"),
        ("n を整数とする",   "n", "integer"),
    ], ids=["complex", "real", "integer"])
    def test_featurep_says_so(self, declaration, name, feature):
        nb.start_engine(title="dom", mode="standard", language="ja", domains=[])
        try:
            eng, sup = nb.get_engine(), _sup()
            _run(eng, sup, declaration)
            got = eng._adapter.execute(f"featurep({name}, {feature})")
            assert got.ok and got.value.strip() == "true", got.value
        finally:
            nb.stop_engine()

    def test_it_coexists_with_the_scalar_declaration(self):
        """The definition already sends `declare(name, scalar)`; the domain is
        an addition, not a replacement."""
        nb.start_engine(title="dom2", mode="standard", language="ja", domains=[])
        try:
            eng, sup = nb.get_engine(), _sup()
            _run(eng, sup, "z を複素数とする")
            got = eng._adapter.execute("[featurep(z, scalar), featurep(z, complex)]")
            assert got.value.replace(" ", "") == "[true,true]", got.value
        finally:
            nb.stop_engine()


class TestItChangesTheAnswer:
    r"""Reaching the CAS is only half of it — the point is the value."""

    def test_a_complex_symbol_keeps_its_conjugate(self):
        nb.start_engine(title="dom3", mode="standard", language="ja", domains=[])
        try:
            eng, sup = nb.get_engine(), _sup()
            _run(eng, sup, "z を複素数とする")
            _run(eng, sup, r"z^* z", "math")
            r = _run(eng, sup, "E1 を簡約して")
            got = r["canonical"].replace(" ", "")
            assert "conjugate" in got, got
            assert got != "z^2"
        finally:
            nb.stop_engine()

    def test_an_integer_symbol_makes_sin_n_pi_vanish(self):
        nb.start_engine(title="dom4", mode="standard", language="ja", domains=[])
        try:
            eng, sup = nb.get_engine(), _sup()
            _run(eng, sup, "n を整数とする")
            _run(eng, sup, r"\sin(n \pi)", "math")
            r = _run(eng, sup, "E1 を簡約して")
            assert r["canonical"].strip() == "0", r["canonical"]
        finally:
            nb.stop_engine()

    def test_a_real_symbol_still_folds_its_conjugate(self):
        """Maxima's default for an undeclared symbol is already real, so this
        one is about the declaration being HONEST rather than about changing
        the answer — `featurep` above is what pins that it was sent."""
        nb.start_engine(title="dom5", mode="standard", language="ja", domains=[])
        try:
            eng, sup = nb.get_engine(), _sup()
            _run(eng, sup, "w を実数とする")
            _run(eng, sup, r"w^* w", "math")
            r = _run(eng, sup, "E1 を簡約して")
            assert r["canonical"].replace(" ", "") == "w^2", r["canonical"]
        finally:
            nb.stop_engine()


class TestTheHookIsBackendNeutral:
    def test_the_base_handler_no_ops(self):
        """Backend Encapsulation Rule: a backend without a predicate mechanism
        must not need to know this exists (the same stance `assume_sign` takes
        for Julia/Symbolics, which has no sign predicate)."""
        from notebook.runtime.backend_handler import BackendHandler
        assert BackendHandler.declare_domain(None, None, "n", "integer") is None

    def test_an_unknown_domain_sends_nothing(self):
        """`declare(x, <anything>)` is not a thing to guess at."""
        nb.start_engine(title="dom6", mode="standard", language="ja", domains=[])
        try:
            eng = nb.get_engine()
            seen = []
            orig = eng._adapter.execute
            eng._adapter.execute = lambda c, *a, **k: (seen.append(c), orig(c, *a, **k))[1]
            eng._handler.declare_domain(eng, "x", "quaternion")
            assert not [c for c in seen if "declare" in c]
        finally:
            nb.stop_engine()


class TestEveryDeclarationActuallyApplies:
    r"""Found while measuring the one above: **no declaration this app sent had
    ever taken effect.**

    The five declaration templates ended in ``; true``, and inside the adapter's
    ``errcatch(_res_:( … ))`` wrapper Maxima's statement separator is ``,``, not
    ``;`` — the adapter's own header note says so, about a different symptom.
    So the command LOOKED like a declaration, returned truthy, and declared
    nothing (measured 2026-08-16)::

        declare(g, antisymmetric); true  → featurep false,  g(2,1) stays g(2,1)
        declare(h, antisymmetric)        → featurep true,   h(2,1) → -h(1,2)

    `declare_symmetry` is a user-facing operation with its own NL trigger, and
    it had never done anything.  That is the same shape as the ticket this file
    is named for: **a declaration the researcher watches the system accept,
    which changes nothing.**

    Turning them on moved factor ORDER in two pinned canonicals (declared
    scalars sort last) — mathematically identical, and noted where they are
    asserted.
    """

    def test_no_declaration_template_carries_a_statement_separator(self):
        """A ratchet: the suffix is invisible in the command's return value, so
        only reading the templates catches it coming back."""
        import json
        with open("backends/maxima.full.json", encoding="utf-8") as f:
            ops = json.load(f).get("operations", {})
        offenders = [k for k, v in ops.items()
                     if isinstance(v, dict) and ";" in str(v.get("template", ""))]
        assert not offenders, offenders

    def test_a_symmetry_declaration_now_applies(self):
        """The operation, not its NL trigger — what had never worked is the
        command itself."""
        nb.start_engine(title="sym", mode="standard", language="ja", domains=[])
        try:
            eng = nb.get_engine()
            eng.run_operation("declare_symmetry",
                              {"name": "g", "symmetry": "antisymmetric"})
            got = eng._adapter.execute("g(2,1)")
            assert got.ok and got.value.replace(" ", "") == "-g(1,2)", got.value
        finally:
            nb.stop_engine()
