"""The backend CAS is pluggable: shared layers hold no CAS-specific behaviour.

Core Principle #10 / CLAUDE.md "Backend Encapsulation Rule".  The count of
mechanical violations (backend imports, backend-identity branches, handler
isinstance checks) is pinned at zero by
``tests/test_code_hygiene.py::test_shared_layers_have_no_cas_specific_code``.
This file covers the three hooks that replaced the leaks found in the
2026-07-29 audit (open_items AUDIT-SHARED-CAS-LEAK), so a future "simplification"
that folds one back into shared code fails on behaviour, not just on a count.
"""
import pytest

from notebook.runtime.backend_handler import _TEX_GREEK_TO_CANONICAL
from notebook.runtime.julia_backend_handler import JuliaBackendHandler
from notebook.runtime.julia_latex_converter import JuliaLatexConverter
from notebook.runtime.julia_semantic_resolver import JuliaSemanticResolver
from notebook.runtime.latex_converter import TEX_GREEK_TO_NAME
from notebook.runtime.maxima_backend_handler import MaximaBackendHandler
from notebook.runtime.maxima_latex_converter import MaximaLatexConverter
from notebook.runtime.maxima_semantic_resolver import MaximaSemanticResolver
from notebook.runtime.semantic_resolver import BaseSemanticResolver


def _handler(cls):
    """A handler instance without touching a CAS process."""
    return cls.__new__(cls)


class TestNeutralGreekTable:
    """Leak 1: the shared base built its TeX-Greek table from
    ``MaximaLatexConverter._GREEK_MAP``, so a Julia-only deployment imported the
    Maxima converter and got Maxima's canonicalization as the default."""

    def test_neutral_table_has_no_cas_constants(self):
        for tex in (r"\pi", r"\infty", r"\mathrm{i}", r"\i"):
            assert tex not in TEX_GREEK_TO_NAME, (
                f"{tex} spells differently per CAS — it belongs in each "
                "converter's _GREEK_MAP, not the neutral table")

    def test_both_converters_agree_on_the_letters(self):
        for tex, name in TEX_GREEK_TO_NAME.items():
            assert MaximaLatexConverter._GREEK_MAP[tex] == name
            assert JuliaLatexConverter._GREEK_MAP[tex] == name

    def test_constants_still_differ_per_backend(self):
        assert MaximaLatexConverter._GREEK_MAP[r"\pi"] == "%pi"
        assert JuliaLatexConverter._GREEK_MAP[r"\pi"] == "pi"
        assert MaximaLatexConverter._GREEK_MAP[r"\mathrm{i}"] == "%i"
        assert JuliaLatexConverter._GREEK_MAP[r"\mathrm{i}"] == "im"

    def test_declaration_parser_table_is_letters_only(self):
        assert _TEX_GREEK_TO_CANONICAL["gamma"] == "gamma"
        assert _TEX_GREEK_TO_CANONICAL["hbar"] == "hbar"
        assert "pi" not in _TEX_GREEK_TO_CANONICAL
        assert "i" not in _TEX_GREEK_TO_CANONICAL


class TestMissingVariableRecovery:
    """Leak 2: the engine matched Julia's ``UndefVarError`` wording and its
    ``_nb_E<n>`` naming itself."""

    @pytest.mark.parametrize("error,expected", [
        ("UndefVarError: `_nb_E12` not defined", "E12"),
        ("UndefVarError: '_nb_E3'", "E3"),
        ('UndefVarError: "_nb_E7" not defined', "E7"),
        ("UndefVarError: `x` not defined", None),   # not a stored expression
        ("MethodError: no method matching foo()", None),
    ])
    def test_julia_recognises_its_own_error(self, error, expected):
        assert _handler(JuliaBackendHandler).missing_variable_expr_id(error) == expected

    def test_maxima_has_nothing_to_recover(self):
        """Maxima keeps no per-expression CAS variables, so it must not claim
        recovery just because the string happens to appear."""
        assert _handler(MaximaBackendHandler).missing_variable_expr_id(
            "UndefVarError: `_nb_E1`") is None


class TestMultiStatementExecution:
    """Leak 3: the engine special-cased ``define_coordinate_system`` and built a
    ``MaximaResult`` itself."""

    class _FakeAdapter:
        def __init__(self, fail_on=None):
            self.fail_on, self.executed = fail_on, []

        def execute_script(self, command):
            from notebook.runtime.maxima_adapter import MaximaResult
            for line in (l.strip() for l in command.strip().splitlines()):
                if not line:
                    continue
                self.executed.append(line)
                if line == self.fail_on:
                    return MaximaResult(ok=False, value="", error=f"bad: {line}")
            return MaximaResult(ok=True, value="true", error=None)

    def test_multi_line_coordinate_system_runs_line_by_line(self):
        adapter = self._FakeAdapter()
        r = _handler(MaximaBackendHandler).execute_multi_statement(
            adapter, "define_coordinate_system", "nb_a: 1$\nnb_b: 2$\n")
        assert r.ok and adapter.executed == ["nb_a: 1$", "nb_b: 2$"]

    def test_first_failing_line_stops_and_fails(self):
        adapter = self._FakeAdapter(fail_on="nb_b: 2$")
        r = _handler(MaximaBackendHandler).execute_multi_statement(
            adapter, "define_coordinate_system", "nb_a: 1$\nnb_b: 2$\nnb_c: 3$\n")
        assert not r.ok and "nb_b" in r.error
        assert "nb_c: 3$" not in adapter.executed

    @pytest.mark.parametrize("operation,command", [
        ("define_coordinate_system", "nb_a: 1$"),      # single line
        ("expand", "nb_a: 1$\nnb_b: 2$"),              # other operation
    ])
    def test_not_applicable_returns_none(self, operation, command):
        assert _handler(MaximaBackendHandler).execute_multi_statement(
            self._FakeAdapter(), operation, command) is None

    def test_other_backends_opt_out(self):
        assert _handler(JuliaBackendHandler).execute_multi_statement(
            self._FakeAdapter(), "define_coordinate_system", "a\nb") is None


class TestBackendNamesHaveOneOwner:
    """Leak 5: ``engine.py`` held its own mode→backend table (and
    ``.get(mode, "maxima")``), duplicating knowledge that ``backend_factory``
    declares itself the owner of."""

    def test_both_modes_resolve(self):
        from notebook.models.session import SessionMode
        from notebook.runtime.backend_factory import default_backend_for_mode
        assert default_backend_for_mode(SessionMode.standard_symbolic) == "maxima"
        assert default_backend_for_mode(SessionMode.second_quantization) == "julia_full"
        # A plain value string works too (config files carry strings, not enums).
        assert default_backend_for_mode("standard_symbolic") == "maxima"

    def test_unlisted_mode_raises_instead_of_defaulting(self):
        """The point of the change: a new SessionMode with no entry used to get
        Maxima silently, and the backend cannot be switched mid-session
        (Core Principle #6), so the researcher would have to start over."""
        from notebook.runtime.backend_factory import default_backend_for_mode
        with pytest.raises(ValueError) as exc:
            default_backend_for_mode("a_mode_nobody_registered")
        assert "backend_factory" in str(exc.value)   # says where to fix it

    def test_the_two_directions_agree(self):
        from notebook.runtime.backend_factory import (
            _DEFAULT_BACKEND_FOR_MODE, default_backend_for_mode,
            session_mode_for_backend)
        for mode in _DEFAULT_BACKEND_FOR_MODE:
            assert session_mode_for_backend(default_backend_for_mode(mode)) == mode

    def test_unknown_backend_maps_to_no_mode(self):
        from notebook.runtime.backend_factory import session_mode_for_backend
        assert session_mode_for_backend("some_other_cas") is None
        assert session_mode_for_backend("") is None

    def test_engine_uses_the_factory_default(self):
        from notebook.engine import NotebookEngine
        from notebook.models.session import SessionMode
        with NotebookEngine(title="dflt") as e:
            assert e.session.primary_backend == "maxima"
        eng = NotebookEngine(title="dflt2", mode=SessionMode.second_quantization)
        assert eng.session.primary_backend == "julia_full"

    @pytest.mark.parametrize("domain,expected", [
        ("condensed_matter", "second_quantization"),
        ("quantum_field_theory", "second_quantization"),
        ("nuclear_physics", "second_quantization"),
        ("classical_mechanics", "standard_symbolic"),
    ])
    def test_domain_category_inference_goes_through_the_factory(self, domain, expected):
        """``_infer_session_mode`` read ``default_backend`` from
        domains/categories.json and compared it to ``"julia_full"`` itself."""
        import ui.notebook_bridge as nb
        assert nb._infer_session_mode([domain]).value == expected

    def test_inference_falls_back_when_nothing_asks_otherwise(self):
        import ui.notebook_bridge as nb
        assert nb._infer_session_mode([]).value == "standard_symbolic"
        assert nb._infer_session_mode(["no_such_domain"]).value == "standard_symbolic"


class TestBackendMustBeStated:
    """Leak 6: three constructors defaulted ``primary_backend`` to ``"maxima"``.
    A caller that forgot it silently got Maxima's operation set — in a Julia
    session that offers operations the backend cannot run.  All 72 call sites
    already passed it, so nothing depended on the default."""

    def test_supervisor_requires_it_and_by_keyword(self):
        from notebook.runtime.llm_supervisor import LLMSupervisor
        with pytest.raises(TypeError):
            LLMSupervisor()
        assert LLMSupervisor(primary_backend="julia_full")._backend == "julia_full"

    def test_rule_parser_requires_it(self):
        from notebook.runtime.llm_supervisor import RuleBasedIntentParser
        from notebook.runtime.operation_validator import OperationValidator
        with pytest.raises(TypeError):
            RuleBasedIntentParser(OperationValidator())
        assert RuleBasedIntentParser(OperationValidator(), "maxima")._backend == "maxima"

    @pytest.mark.parametrize("module_name,func", [
        ("ui.notebook_bridge", "_make_supervisor"),
        ("ui.debug_repl", "_make_supervisor"),
    ])
    def test_supervisor_factories_require_it(self, module_name, func):
        import importlib
        import inspect
        fn = getattr(importlib.import_module(module_name), func)
        param = inspect.signature(fn).parameters["primary_backend"]
        assert param.default is inspect.Parameter.empty, "must not default to a CAS"
        assert param.kind is inspect.Parameter.KEYWORD_ONLY


class TestNoSharedDefaultForCasTokens:
    """Leak 4: ``imaginary_unit_name`` defaulted to Maxima's ``%i`` in the shared
    base, so a new backend that forgot to declare it emitted Maxima's token."""

    def test_base_has_no_default(self):
        assert not hasattr(BaseSemanticResolver, "imaginary_unit_name")

    def test_each_backend_declares_its_own(self):
        assert MaximaSemanticResolver.imaginary_unit_name == "%i"
        assert JuliaSemanticResolver.imaginary_unit_name == "im"
