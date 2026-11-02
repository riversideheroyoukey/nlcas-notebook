"""Tests for the operator_substitute IR-level quantum operator substitution."""
from __future__ import annotations
import pytest
from notebook.models.ir import (
    Add, Mul, FunctionCall, Adjoint, Symbol, Number,
)
from notebook.runtime.julia_backend_handler import _adjoint_ir, _subst_quantum_op


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sym(name: str) -> Symbol:
    return Symbol(name=name)


def fc(name: str, *args) -> FunctionCall:
    return FunctionCall(func=sym(name), args=list(args))


OPERATOR_NAMES = {"c", "alpha", "beta"}


# ---------------------------------------------------------------------------
# _adjoint_ir
# ---------------------------------------------------------------------------

class TestAdjointIr:
    def test_operator_fc_gets_adjoint(self):
        expr = fc("c", sym("k"))
        result = _adjoint_ir(expr, OPERATOR_NAMES)
        assert isinstance(result, Adjoint)
        assert isinstance(result.expr, FunctionCall)

    def test_adjoint_of_adjoint_cancels(self):
        expr = Adjoint(expr=fc("c", sym("k")))
        result = _adjoint_ir(expr, OPERATOR_NAMES)
        # (c†)† = c
        assert isinstance(result, FunctionCall)
        assert result.func.name == "c"

    def test_scalar_fc_unchanged(self):
        expr = fc("u", sym("k"))  # u(k) is scalar
        result = _adjoint_ir(expr, OPERATOR_NAMES)
        assert result is expr

    def test_add_distributes(self):
        a = fc("c", sym("k"))
        b = fc("alpha", sym("k"))
        expr = Add(args=[a, b])
        result = _adjoint_ir(expr, OPERATOR_NAMES)
        assert isinstance(result, Add)
        assert all(isinstance(t, Adjoint) for t in result.args)

    def test_mul_reverses_order(self):
        a = fc("alpha", sym("k"))
        b = fc("c", sym("k"))
        expr = Mul(args=[a, b])
        result = _adjoint_ir(expr, OPERATOR_NAMES)
        # adjoint reverses: [adjoint(b), adjoint(a)]
        assert isinstance(result, Mul)
        assert isinstance(result.args[0], Adjoint)
        assert result.args[0].expr.func.name == "c"
        assert isinstance(result.args[1], Adjoint)
        assert result.args[1].expr.func.name == "alpha"


# ---------------------------------------------------------------------------
# _subst_quantum_op
# ---------------------------------------------------------------------------

class TestSubstQuantumOp:
    def test_simple_substitution(self):
        # c(k) → u(k)*alpha(k), substitute into c(k)
        main = fc("c", sym("k"))
        rhs = Mul(args=[fc("u", sym("k")), fc("alpha", sym("k"))])
        result = _subst_quantum_op(main, "c", rhs, "k", OPERATOR_NAMES)
        assert isinstance(result, Mul)
        assert result.args[0].func.name == "u"
        assert result.args[1].func.name == "alpha"

    def test_index_propagation_negative(self):
        # c(-k): actual_idx = Mul([Number(-1), Symbol("k")])
        neg_k = Mul(args=[Number(value=-1), sym("k")])
        main = fc("c", neg_k)
        rhs = Mul(args=[fc("u", sym("k")), fc("alpha", sym("k"))])
        result = _subst_quantum_op(main, "c", rhs, "k", OPERATOR_NAMES)
        # Result should be u(-k) * alpha(-k)
        assert isinstance(result, Mul)
        u_call = result.args[0]
        assert u_call.func.name == "u"
        # u's arg should have been substituted: Symbol("k") → Mul([-1, k])
        from notebook.models.ir import Mul as MulIR, Number as NumberIR
        assert isinstance(u_call.args[0], MulIR)

    def test_adjoint_op_replaced_by_rhs_adjoint(self):
        # c†(k) → adjoint(u(k)*alpha(k) + v(k)*beta†(-k))
        # For a simple case: adjoint of (u(k)*alpha(k)) should be (adjoint(alpha(k))*u(k))
        rhs = Mul(args=[fc("u", sym("k")), fc("alpha", sym("k"))])
        main = Adjoint(expr=fc("c", sym("k")))
        result = _subst_quantum_op(main, "c", rhs, "k", OPERATOR_NAMES)
        # adjoint(u(k)*alpha(k)) = adjoint(alpha(k))*u(k) [reversed, scalar u unchanged]
        assert isinstance(result, Mul)
        assert isinstance(result.args[0], Adjoint)
        assert result.args[0].expr.func.name == "alpha"
        assert result.args[1].func.name == "u"

    def test_non_matching_op_unchanged(self):
        main = fc("alpha", sym("k"))
        rhs = fc("beta", sym("k"))
        result = _subst_quantum_op(main, "c", rhs, "k", OPERATOR_NAMES)
        # alpha(k) is not c(k) — should be unchanged
        assert isinstance(result, FunctionCall)
        assert result.func.name == "alpha"

    def test_add_of_operators(self):
        # c(k) + c(-k) each get substituted
        neg_k = Mul(args=[Number(value=-1), sym("k")])
        main = Add(args=[fc("c", sym("k")), fc("c", neg_k)])
        rhs = fc("alpha", sym("k"))
        result = _subst_quantum_op(main, "c", rhs, "k", OPERATOR_NAMES)
        assert isinstance(result, Add)
        assert len(result.args) == 2

    def test_indexsum_body_traversed(self):
        # sum_k(c(k)) — IndexSum with dummy index "k" same as formal_idx "k"
        # Substitution must descend into the body despite the name collision.
        from notebook.models.ir import IndexSum, IndexSymbol
        dummy_k = IndexSymbol(name="k")
        inner = fc("c", sym("k"))
        main = IndexSum(body=inner, dummy_indices=[dummy_k])
        rhs = fc("alpha", sym("k"))
        result = _subst_quantum_op(main, "c", rhs, "k", OPERATOR_NAMES)
        assert isinstance(result, IndexSum)
        body = result.body
        assert isinstance(body, FunctionCall)
        assert body.func.name == "alpha"
