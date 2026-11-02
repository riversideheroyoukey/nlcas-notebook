"""AUDIT-MATMUL-STAR: general matrix `Mul` must use Maxima's `.` product, not `*`.

Sibling bug to R-4 (tests/test_matrix_commutator.py), which fixed the SAME
elementwise-`*`-on-matrices issue but only for expanded Commutator/
Anticommutator.  A plain product `A*B` of two declared matrix symbols went
through the shared, type-agnostic `ExprGenerator._generate_mul` and still
rendered `*`, silently giving Maxima's elementwise product instead of the
matrix product.

Verified against real Maxima: sigma_x * sigma_y with `*` (elementwise) →
matrix([0,-%i],[%i,0]) = sigma_y itself (silently wrong); with `.` (matrix
product) → matrix([%i,0],[0,-%i]) = i*sigma_z (correct).
"""
from notebook.models.ir import Mul, Symbol
from notebook.models.quantity import Quantity, MathType
from notebook.models.symbol_context import SymbolContext
from notebook.runtime.maxima_code_generator import MaximaCodeGenerator


def _matrix_ctx(*names: str) -> SymbolContext:
    ctx = SymbolContext()
    for n in names:
        ctx.register(n, Quantity(
            name=n, kind="anonymous", math_type=MathType(algebra="matrix")))
    return ctx


class TestMulProductOperator:
    def test_scalar_mul_keeps_star(self):
        gen = MaximaCodeGenerator(SymbolContext())
        cmd = gen.generate(Mul(args=[Symbol(name="a"), Symbol(name="b")]))
        assert cmd == "a * b"

    def test_no_context_keeps_star(self):
        gen = MaximaCodeGenerator()
        cmd = gen.generate(Mul(args=[Symbol(name="A"), Symbol(name="B")]))
        assert cmd == "A * B"

    def test_matrix_mul_uses_dot(self):
        gen = MaximaCodeGenerator(_matrix_ctx("A", "B"))
        cmd = gen.generate(Mul(args=[Symbol(name="A"), Symbol(name="B")]))
        assert cmd == "A . B"

    def test_one_matrix_operand_is_enough(self):
        gen = MaximaCodeGenerator(_matrix_ctx("A"))
        cmd = gen.generate(Mul(args=[Symbol(name="A"), Symbol(name="B")]))
        assert cmd == "A . B"

    def test_scalar_times_matrix_uses_dot(self):
        """A scalar coefficient in a 3-factor product with a matrix still
        needs `.` for the whole chain — Maxima's `.` degrades to ordinary
        multiplication when either factor is scalar, so `2 . A . B` is safe."""
        ctx = _matrix_ctx("A", "B")
        gen = MaximaCodeGenerator(ctx)
        cmd = gen.generate(Mul(args=[Symbol(name="k"), Symbol(name="A"), Symbol(name="B")]))
        assert cmd == "k . A . B"

    def test_negation_shortcut_unaffected(self):
        """Mul([-1, A]) → "-A" regardless of operator (no product to choose)."""
        from notebook.models.ir import Number
        gen = MaximaCodeGenerator(_matrix_ctx("A"))
        cmd = gen.generate(Mul(args=[Number(value=-1), Symbol(name="A")]))
        assert cmd == "-A"


class TestPauliMulAgainstMaxima:
    """Integration: A*B for Pauli matrices must give the real matrix product."""

    def test_pauli_mul_matches_matrix_product_not_elementwise(self):
        from notebook.runtime.maxima_adapter import MaximaAdapter
        gen = MaximaCodeGenerator(_matrix_ctx("A", "B"))
        cmd = gen.generate(Mul(args=[Symbol(name="A"), Symbol(name="B")]))
        with MaximaAdapter() as m:
            m.execute("A: matrix([0,1],[1,0])$")       # sigma_x
            m.execute("B: matrix([0,-%i],[%i,0])$")    # sigma_y
            result = m.execute(cmd)
        assert result.ok
        value = result.value.replace(" ", "")
        # Elementwise (the bug) would silently reproduce sigma_y itself.
        assert value != "matrix([0,-%i],[%i,0])", "product collapsed to elementwise *"
        assert value == "matrix([%i,0],[0,-%i])"  # sigma_x . sigma_y = i*sigma_z
