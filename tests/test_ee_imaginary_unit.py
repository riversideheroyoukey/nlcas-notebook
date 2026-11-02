"""Tests for the electrical-engineering imaginary-unit `j` handling (C-4).

In the electrical_engineering domain `j` is the imaginary unit (√-1): the Julia
code generator emits `im`, and `j` is not flagged as an undefined symbol.  In
other domains `j` keeps its physics meaning (a summation index) and is left
untouched.
"""
import json
from pathlib import Path

from notebook.models.symbol_context import SymbolContext
from notebook.models.ir import Symbol
from notebook.runtime.julia_code_generator import JuliaCodeGenerator
from notebook.runtime.julia_backend_handler import JuliaBackendHandler

_ROOT = Path(__file__).resolve().parent.parent


def _ctx(*domains: str) -> SymbolContext:
    ctx = SymbolContext()
    ctx.domains = list(domains)
    return ctx


class TestCodeGenImaginaryUnit:
    def test_j_becomes_im_in_ee_domain(self):
        out = JuliaCodeGenerator(context=_ctx("electrical_engineering")).generate(Symbol(name="j"))
        assert out == "im"

    def test_j_unchanged_in_physics_domain(self):
        out = JuliaCodeGenerator(context=_ctx("condensed_matter")).generate(Symbol(name="j"))
        assert out == "j"

    def test_j_unchanged_with_no_domain(self):
        out = JuliaCodeGenerator(context=_ctx()).generate(Symbol(name="j"))
        assert out == "j"

    def test_i_is_still_imaginary_unit(self):
        # i remains the imaginary unit regardless of domain
        assert JuliaCodeGenerator(context=_ctx("electrical_engineering")).generate(Symbol(name="i")) == "im"
        assert JuliaCodeGenerator(context=_ctx("condensed_matter")).generate(Symbol(name="i")) == "im"

    def test_declared_j_not_converted(self):
        # if the user explicitly declares j, respect it (don't force im)
        from notebook.models.quantity import Quantity
        ctx = _ctx("electrical_engineering")
        ctx.register("j", quantity=Quantity(name="j", kind="scalar"))
        assert JuliaCodeGenerator(context=ctx).generate(Symbol(name="j")) == "j"


class TestUndefinedWarning:
    def _handler(self) -> JuliaBackendHandler:
        return JuliaBackendHandler.__new__(JuliaBackendHandler)

    def test_j_not_flagged_in_ee_domain(self):
        und = self._handler().find_undefined_symbols(Symbol(name="j"), _ctx("electrical_engineering"))
        assert "j" not in und

    def test_j_flagged_in_physics_domain(self):
        und = self._handler().find_undefined_symbols(Symbol(name="j"), _ctx("condensed_matter"))
        assert "j" in und

    def test_i_never_flagged(self):
        und = self._handler().find_undefined_symbols(Symbol(name="i"), _ctx("condensed_matter"))
        assert "i" not in und


class TestMaximaImaginaryUnit:
    """EE sessions default to the Maxima backend, so `j` must also resolve to
    Maxima's imaginary unit %i there (unambiguous within the EE domain)."""

    def _resolve_j(self, *domains: str, declared: bool = False):
        from notebook.runtime.maxima_semantic_resolver import MaximaSemanticResolver
        from notebook.models.quantity import Quantity
        ctx = _ctx(*domains)
        if declared:
            ctx.register("j", quantity=Quantity(name="j", kind="scalar"))
        out = MaximaSemanticResolver().resolve(Symbol(name="j"), ctx)
        return getattr(out, "name", out)

    def test_j_becomes_percent_i_in_ee_domain(self):
        assert self._resolve_j("electrical_engineering") == "%i"

    def test_j_unchanged_in_physics_domain(self):
        assert self._resolve_j("condensed_matter") == "j"

    def test_declared_j_respected(self):
        assert self._resolve_j("electrical_engineering", declared=True) == "j"


class TestDomainConfig:
    def test_meta_json_valid(self):
        meta = json.loads(
            (_ROOT / "domains" / "electrical_engineering" / "meta.json").read_text(encoding="utf-8"))
        assert meta["category"] == "engineering"
        assert meta["label_en"] == "Electrical Engineering"
        assert meta["label_ja"] == "電気工学"

    def test_supervisor_hint_mentions_imaginary_j(self):
        hint = (_ROOT / "domains" / "electrical_engineering"
                / "supervisor_hint.txt").read_text(encoding="utf-8")
        assert "imaginary unit" in hint
        assert "`j`" in hint
