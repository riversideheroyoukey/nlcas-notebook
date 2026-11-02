"""What ``[a, b]`` means is a per-BACKEND convention (open_items AUDIT-BRACKET-BACKEND).

``BaseSemanticResolver._resolve_bracketexpr`` used to return ``Commutator`` for
every backend — Julia/QuantumAlgebra's reading, applied in Maxima sessions where
``[...]`` is a list and a matrix is a list of rows.  Consequence, reproduced from
a canonical that exists in ``sessions/S-y_avoided_crossing``: a 2×2 Hamiltonian
``expand``ed to ``0`` and the step was recorded as a success.

The convention now lives in ``_square_bracket_role``, one of the hooks the
resolver's docstring lists as the only permitted backend differences (Core
Principle #10).  These tests pin both readings and the nesting rule that
separates them.
"""
import pytest

from notebook.engine import NotebookEngine
from notebook.models.ir import BracketExpr, Commutator, Symbol
from notebook.models.quantity import MathType, Quantity
from notebook.models.symbol_context import SymbolContext
from notebook.runtime.julia_semantic_resolver import JuliaSemanticResolver
from notebook.runtime.maxima_semantic_resolver import MaximaSemanticResolver
from notebook.runtime.shared_parser import SharedParser

_HAMILTONIAN = "[[epsilon, Delta],[Delta, -epsilon]]"


def _square(*args):
    return BracketExpr(args=list(args), kind="square")


class TestMaximaReading:

    def test_nested_bracket_is_a_list(self):
        """A commutator of two lists is meaningless in any CAS, so a bracket
        whose elements are brackets can only be a nested list."""
        res = MaximaSemanticResolver()._resolve_node(
            SharedParser().parse(_HAMILTONIAN), SymbolContext())
        assert isinstance(res, BracketExpr)
        assert all(isinstance(row, BracketExpr) for row in res.args), (
            "rows must stay lists — resolving them as commutators is what "
            "collapsed the matrix to [0, 0]")

    def test_flat_two_element_bracket_stays_a_commutator(self):
        """``[A, B]`` on matrices is a real Maxima workflow (R-4/SR-4), and the
        commutator reading is the established behaviour when nothing is
        declared."""
        res = MaximaSemanticResolver()._resolve_node(
            _square(Symbol(name="A"), Symbol(name="B")), SymbolContext())
        assert isinstance(res, Commutator)

    def test_declared_matrix_commutator_still_works(self):
        ctx = SymbolContext()
        for n in ("A", "B"):
            ctx.register(n, Quantity(name=n, kind="anonymous",
                                     math_type=MathType(algebra="matrix")))
        res = MaximaSemanticResolver()._resolve_node(
            _square(Symbol(name="A"), Symbol(name="B")), ctx)
        assert isinstance(res, Commutator)


class TestJuliaReading:

    def test_bracket_is_a_commutator(self):
        res = JuliaSemanticResolver()._resolve_node(
            _square(Symbol(name="A"), Symbol(name="B")), SymbolContext())
        assert isinstance(res, Commutator)

    def test_nesting_stays_a_commutator_of_commutators(self):
        """The Jacobi identity is ordinary operator algebra — nesting must NOT
        be read as a list here.  This is exactly why the reading is a hook."""
        res = JuliaSemanticResolver()._resolve_node(
            SharedParser().parse("[[A,B],[C,D]]"), SymbolContext())
        assert isinstance(res, Commutator)
        assert isinstance(res.a, Commutator) and isinstance(res.b, Commutator)


class TestEndToEnd:

    @pytest.mark.parametrize("operation", ["expand", "simplify"])
    def test_two_level_hamiltonian_survives(self, operation):
        """The regression itself: before the fix both operations returned '0'."""
        with NotebookEngine(title="bracket") as e:
            ex = e.register_expression(_HAMILTONIAN)
            st, val = e.run_operation(operation, {"expr": ex.display_id})
            assert st.status.value == "success"
            assert val.replace(" ", "") == "[[epsilon,Delta],[Delta,-epsilon]]"

    def test_matrix_semicolon_form_is_unaffected(self):
        with NotebookEngine(title="bracket m") as e:
            assert (e._handler.to_backend_code("[a, b; c, d]").code
                    == "matrix([a, b], [c, d])")

    def test_three_element_list_is_unaffected(self):
        with NotebookEngine(title="bracket l") as e:
            assert e._handler.to_backend_code("[a,b,c]").code == "[a, b, c]"
