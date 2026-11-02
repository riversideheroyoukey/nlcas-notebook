"""R-9: matrix element access A[i,j].

When the base symbol is a declared matrix / tensor / vector, ``A[i,j]`` resolves
to an IndexedSymbol and the Maxima generator renders it as ``A[i, j]`` (array
access).  Undeclared bases keep the existing behaviour (declaration-driven, the
(A) policy); a standalone ``[A,B]`` is still a commutator.
"""
from notebook.runtime.shared_parser import SharedParser
from notebook.runtime.maxima_semantic_resolver import MaximaSemanticResolver
from notebook.runtime.maxima_code_generator import MaximaCodeGenerator
from notebook.models.symbol_context import SymbolContext
from notebook.models.quantity import Quantity, MathType


def _cas(text, ctx):
    ir = MaximaSemanticResolver().resolve(SharedParser().parse(text), ctx)
    return MaximaCodeGenerator(ctx).generate(ir)


def _matrix_ctx(*names, algebra="matrix"):
    ctx = SymbolContext()
    for n in names:
        ctx.register(n, Quantity(name=n, kind="anonymous",
                                 math_type=MathType(algebra=algebra)))
    return ctx


class TestMatrixElement:
    def test_two_index_element(self):
        assert _cas("A[i,j]", _matrix_ctx("A")) == "A[i, j]"

    def test_numeric_element(self):
        assert _cas("A[1,2]", _matrix_ctx("A")) == "A[1, 2]"

    def test_single_index(self):
        assert _cas("A[i]", _matrix_ctx("A")) == "A[i]"

    def test_vector_element(self):
        assert _cas("v[i]", _matrix_ctx("v", algebra="vector")) == "v[i]"

    def test_commutator_unaffected(self):
        assert _cas("[A,B]", SymbolContext()) == "((A)*(B) - (B)*(A))"

    def test_undeclared_not_element_access(self):
        # undeclared A[i,j] is NOT treated as element access (declaration-driven)
        assert "A[i" not in _cas("A[i,j]", SymbolContext())
