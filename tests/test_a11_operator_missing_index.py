"""A-11: site-unspecified quantum operators give a friendly error.

Writing a spin component without a site (`Sx`) — or a fermion/boson operator
without an index (`c`) — leaves a QuantumAlgebra `QuExprConstructor`, and using
it in a product raises an opaque Julia `MethodError`.  The Julia handler now
recognizes that signature (`classify_cas_error`) and the engine raises a
friendly, i18n'd message telling the user to specify an index/site instead.

Case A of A-11 (error-message improvement).  Auto-completion (cases B/C) is
deliberately not done — it is only safe for single-site models and is unsafe
once multi-site lattices (F-1) exist.
"""
import pytest

from notebook.engine import NotebookEngine
from notebook.models.session import SessionMode
from notebook.runtime.julia_backend_handler import JuliaBackendHandler


class TestClassifyCasError:
    """Unit-level: the Julia handler maps the MethodError signature to a key."""

    def _h(self):
        return JuliaBackendHandler(None)

    def test_constructor_methoderror_classified(self):
        err = ("MethodError: no method matching *(::QuExpr, "
               "::QuantumAlgebra.QuExprConstructor{...tlsxyz_ops...})")
        assert self._h().classify_cas_error(err) == "error.operator_missing_index"

    def test_unrelated_error_not_classified(self):
        assert self._h().classify_cas_error("UndefVarError: `xi` not defined") is None

    def test_empty_error_not_classified(self):
        assert self._h().classify_cas_error("") is None


class TestEngineFriendlyMessage:
    """End-to-end through a live Julia session."""

    def test_no_site_anywhere_gives_friendly_message(self):
        # both operands site-unspecified and no site in the session → genuinely
        # ambiguous (0 sites), so case-B does not fire and case-A errors.
        with NotebookEngine(title="a11ja", mode=SessionMode.second_quantization) as e:
            e.session.language = ["ja"]
            e.start()
            e.run_operation("define_spin_operator", {"name": "S"})
            with pytest.raises(ValueError) as ei:
                e.run_operation("commutator", {"a": "Sx", "b": "Sy"})
            assert "MethodError" not in str(ei.value)

    def test_no_site_anywhere_english(self):
        with NotebookEngine(title="a11en", mode=SessionMode.second_quantization) as e:
            e.session.language = ["en"]
            e.start()
            e.run_operation("define_spin_operator", {"name": "S"})
            with pytest.raises(ValueError) as ei:
                e.run_operation("commutator", {"a": "Sx", "b": "Sy"})
            assert "site" in str(ei.value).lower()

    def test_site_specified_still_works(self):
        with NotebookEngine(title="a11ok", mode=SessionMode.second_quantization) as e:
            e.start()
            e.run_operation("define_spin_operator", {"name": "S"})
            _, result = e.run_operation("commutator", {"a": "Sz_1", "b": "Sx_1"})
            assert result and "MethodError" not in result


class TestEngineAutoCompleteSite:
    """A-11 case B: a single unambiguous site auto-completes the bare operator."""

    def test_single_site_autocompletes(self):
        # the Larmor case: the operand Sz_1 fixes the site, so Sx → Sx_1
        with NotebookEngine(title="a11b", mode=SessionMode.second_quantization) as e:
            e.start()
            e.run_operation("define_spin_operator", {"name": "S"})
            _, result = e.run_operation("commutator", {"a": "Sz_1", "b": "Sx"})
            # [Sz_1, Sx_1] = i Sy_1 — succeeds, no opaque error
            assert result and "MethodError" not in result


class TestSiteCompletionHooks:
    """Unit-level: the Julia handler's site detection / completion."""

    def _h(self):
        from notebook.models.symbol_context import SymbolContext
        h = JuliaBackendHandler(None)
        ctx = SymbolContext()
        for n in ("Sx", "Sy", "Sz"):
            ctx.register_quantity(n)
        h.set_symbol_context(ctx)
        return h

    def test_spin_sites_in(self):
        h = self._h()
        assert h.spin_sites_in("Sz_1 Sx_1") == {"1"}
        assert h.spin_sites_in("Sz_1 Sx_2") == {"1", "2"}
        assert h.spin_sites_in("Sx") == set()

    def test_complete_bare_only_when_unsited(self):
        h = self._h()
        assert h.complete_bare_spin_sites("Sx", "1") == "Sx_1"
        assert h.complete_bare_spin_sites("Sx Sy", "2") == "Sx_2 Sy_2"
        assert h.complete_bare_spin_sites("Sz_1", "1") == "Sz_1"   # already sited

    def test_no_registered_spins_is_noop(self):
        from notebook.models.symbol_context import SymbolContext
        h = JuliaBackendHandler(None)
        h.set_symbol_context(SymbolContext())
        assert h.complete_bare_spin_sites("Sx", "1") == "Sx"
        assert h.spin_sites_in("Sx_1") == set()
