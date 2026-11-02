"""R-4 (SR-4): matrix commutators must use Maxima's `.` product, not `*`.

Maxima's `*` is ELEMENTWISE on matrices, so the expanded commutator
((A)*(B) - (B)*(A)) of two symmetric-pattern matrices like the Pauli
matrices is the ZERO matrix — a silently wrong answer.  Verified against
real Maxima on 2026-07-03: [σx, σy] with `*` → matrix([0,0],[0,0]);
with `.` → matrix([2*%i,0],[0,-2*%i]) = 2i·σz (correct).

Scalars keep `*` (historical rendered-command stability).
"""
import pytest

from notebook.models.ir import Commutator, Anticommutator, Symbol
from notebook.models.quantity import Quantity, MathType
from notebook.models.symbol_context import SymbolContext
from notebook.runtime.maxima_code_generator import MaximaCodeGenerator


def _matrix_ctx(*names: str) -> SymbolContext:
    ctx = SymbolContext()
    for n in names:
        ctx.register(n, Quantity(
            name=n, kind="anonymous", math_type=MathType(algebra="matrix")))
    return ctx


class TestCommutatorProductOperator:
    def test_scalar_commutator_keeps_star(self):
        gen = MaximaCodeGenerator(SymbolContext())
        cmd = gen.generate(Commutator(a=Symbol(name="a"), b=Symbol(name="b")))
        assert cmd == "((a)*(b) - (b)*(a))"

    def test_no_context_keeps_star(self):
        gen = MaximaCodeGenerator()
        cmd = gen.generate(Commutator(a=Symbol(name="A"), b=Symbol(name="B")))
        assert cmd == "((A)*(B) - (B)*(A))"

    def test_matrix_commutator_uses_dot(self):
        gen = MaximaCodeGenerator(_matrix_ctx("A", "B"))
        cmd = gen.generate(Commutator(a=Symbol(name="A"), b=Symbol(name="B")))
        assert cmd == "((A) . (B) - (B) . (A))"

    def test_one_matrix_operand_is_enough(self):
        gen = MaximaCodeGenerator(_matrix_ctx("A"))
        cmd = gen.generate(Commutator(a=Symbol(name="A"), b=Symbol(name="B")))
        assert " . " in cmd and "*" not in cmd

    def test_matrix_anticommutator_uses_dot(self):
        gen = MaximaCodeGenerator(_matrix_ctx("A", "B"))
        cmd = gen.generate(Anticommutator(a=Symbol(name="A"), b=Symbol(name="B")))
        assert cmd == "((A) . (B) + (B) . (A))"

    def test_vector_operand_uses_dot(self):
        ctx = SymbolContext()
        ctx.register("v", Quantity(
            name="v", kind="anonymous", math_type=MathType(algebra="vector")))
        gen = MaximaCodeGenerator(ctx)
        cmd = gen.generate(Commutator(a=Symbol(name="v"), b=Symbol(name="M")))
        assert " . " in cmd


class TestPauliCommutatorAgainstMaxima:
    """Integration: the generated command must give [σx, σy] = 2i·σz, not 0."""

    def test_pauli_commutator_nonzero(self):
        from notebook.runtime.maxima_adapter import MaximaAdapter
        gen = MaximaCodeGenerator(_matrix_ctx("A", "B"))
        cmd = gen.generate(Commutator(a=Symbol(name="A"), b=Symbol(name="B")))
        with MaximaAdapter() as m:
            m.execute("A: matrix([0,1],[1,0])$")       # σx
            m.execute("B: matrix([0,-%i],[%i,0])$")    # σy
            result = m.execute(cmd)
        assert result.ok
        value = result.value.replace(" ", "")
        assert value != "matrix([0,0],[0,0])", "commutator collapsed to zero (elementwise *)"
        assert "2*%i" in value  # [σx, σy] = 2i σz

    def test_pauli_anticommutator_identity(self):
        """{σx, σx} = 2I."""
        from notebook.runtime.maxima_adapter import MaximaAdapter
        gen = MaximaCodeGenerator(_matrix_ctx("A"))
        cmd = gen.generate(Anticommutator(a=Symbol(name="A"), b=Symbol(name="A")))
        with MaximaAdapter() as m:
            m.execute("A: matrix([0,1],[1,0])$")
            result = m.execute(cmd)
        assert result.ok
        assert result.value.replace(" ", "") == "matrix([2,0],[0,2])"
