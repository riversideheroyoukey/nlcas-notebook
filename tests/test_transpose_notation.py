"""AUDIT-XTX / R-11: prime as matrix transpose (X' → transpose(X)).

A single standalone prime resolves to a Transpose when the base is a declared
matrix/vector/tensor (declaration wins) OR the active domains' notation profile
maps prime→transpose (statistics / econometrics — R-11 profile-driven
resolution).  Otherwise the prime keeps its existing meaning (derivative /
creation operator / bare Symbol).
"""
from notebook.runtime.shared_parser import SharedParser
from notebook.runtime.maxima_semantic_resolver import MaximaSemanticResolver
from notebook.runtime.maxima_code_generator import MaximaCodeGenerator
from notebook.models.symbol_context import SymbolContext
from notebook.models.quantity import Quantity, MathType


def _gen(text, ctx):
    ast = SharedParser().parse(text)
    ir = MaximaSemanticResolver().resolve(ast, ctx)
    return MaximaCodeGenerator(ctx).generate(ir)


def _matrix_ctx(*names, domains=None):
    ctx = SymbolContext()
    ctx.domains = list(domains or [])
    for n in names:
        ctx.register(n, Quantity(name=n, kind="anonymous",
                                 math_type=MathType(algebra="matrix")))
    return ctx


class TestPrimeTransposeByDeclaration:
    def test_matrix_prime_is_transpose(self):
        assert _gen("X'", _matrix_ctx("X")) == "transpose(X)"

    def test_xtx_is_matrix_product(self):
        # X'X → transpose(X) . X (the Maxima matrix product, not element-wise *)
        assert _gen("X'X", _matrix_ctx("X")) == "transpose(X) . X"

    def test_xty_is_matrix_product(self):
        assert _gen("X' y", _matrix_ctx("X", "y")) == "transpose(X) . y"

    def test_scalar_prime_is_not_transpose(self):
        # undeclared / scalar symbol, no domain → prime stays part of the name.
        # The name is SPELLED `X\'` in Maxima code: a bare quote is Maxima's
        # noun-form operator, not an identifier character, so the unescaped
        # form was a syntax error — and the adapter reported syntax errors as
        # success, so `simplify` on `x\' + x` returned `_res_` (2026-08-20).
        assert _gen("X'", SymbolContext()) == "X\\'"


class TestPrimeTransposeByProfile:
    def test_statistics_domain_prime_is_transpose(self):
        # R-11: statistics_profile maps prime→transpose for econometrics etc.
        assert _gen("X'", _matrix_ctx(domains=["econometrics"])) == "transpose(X)"

    def test_probability_statistics_domain(self):
        assert _gen("X'", _matrix_ctx(domains=["probability_statistics"])) == "transpose(X)"

    def test_non_statistics_domain_unchanged(self):
        # quantum profile maps prime→creation_operator, NOT transpose
        assert _gen("X'", _matrix_ctx(domains=["quantum_mechanics"])) == "X\\'"

    def test_no_domain_unchanged(self):
        assert _gen("X'", _matrix_ctx(domains=[])) == "X\\'"


def test_double_prime_not_transpose():
    # X'' is not a transpose (order > 1); keep the existing behaviour.
    out = _gen("X''", _matrix_ctx("X"))
    assert "transpose" not in out
