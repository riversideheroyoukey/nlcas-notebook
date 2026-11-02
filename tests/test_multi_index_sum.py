r"""A sum over several indices must sum over all of them.

Found 2026-08-07 while preparing the Green's-function perturbation-expansion
topic: the general (arbitrary-basis, spin-dependent) Hamiltonian is written

    H_0   = \sum_{m,n} h_{m,n} c'_m c_n
    H_int = \sum_{m,n,p,q} V_{m,n,p,q} c'_m c'_n c_p c_q

and neither could be written.  Two silent failures, both from the same cause —
the sum's subscript was handed to the GENERIC resolver, which does not know it
is looking at an index list:

* ``\sum_{m,n}`` — the same curly braces are the ANTICOMMUTATOR ``{A,B}``
  everywhere else, and with exactly two symbols that is what came back.  An
  Anticommutator matched none of the caller's branches, the sum was not
  recognised at all, and the expression degraded to a PRODUCT:
  ``\sum_{m,n} h_{m,n}`` reached Maxima as ``sum(m, n) * h(m, n)``.

* ``\sum_{m,n,p}`` — three or more args stayed a BracketExpr and the caller
  took ``args[0]``, dropping the rest in silence: a triple sum became
  ``sum(h(m, n), m)``.  The worse shape of the two, because a plausible answer
  comes back.

The fix reads the subscript in the position where it appears.  Lowering was
already right: the code generator emits one nested ``sum(...)`` per dummy —
the iterated form the researcher proposed, and the same shape
``evaluate_integral_form`` uses for a multiple integral.  Measured in Maxima:
nesting order does not change the result (``ratsimp(difference) = 0``), and the
CAS even lifts index-independent factors out of the inner sum.
"""
import pytest

from notebook.models.symbol_context import SymbolContext
from notebook.runtime.maxima_code_generator import MaximaCodeGenerator
from notebook.runtime.maxima_semantic_resolver import MaximaSemanticResolver
from notebook.runtime.julia_semantic_resolver import JuliaSemanticResolver
from notebook.runtime.shared_parser import SharedParser


def _resolve(text, ctx=None):
    return MaximaSemanticResolver().resolve(SharedParser().parse(text),
                                            ctx or SymbolContext())


def _cas(text, ctx=None):
    ctx = ctx or SymbolContext()
    return MaximaCodeGenerator(context=ctx).generate(_resolve(text, ctx))


def _dummies(ir):
    return [d.name for d in getattr(ir, "dummy_indices", [])]


@pytest.mark.parametrize("text,names", [
    (r"\sum_k h_k", ["k"]),
    (r"\sum_{k} h_k", ["k"]),
    (r"\sum_{m,n} h_{m,n}", ["m", "n"]),
    (r"\sum_{m,n,p} h_{m,n}", ["m", "n", "p"]),
    (r"\sum_{m,n,p,q} V_{m,n,p,q}", ["m", "n", "p", "q"]),
])
def test_every_index_becomes_a_dummy(text, names):
    ir = _resolve(text)
    assert type(ir).__name__ == "IndexSum", f"{text} resolved to {type(ir).__name__}"
    assert _dummies(ir) == names


@pytest.mark.parametrize("text,expected", [
    (r"\sum_k h_k", "sum(h(k), k)"),
    (r"\sum_{m,n} h_{m,n}", "sum(sum(h(m, n), n), m)"),
    (r"\sum_{m,n,p} h_{m,n}", "sum(sum(sum(h(m, n), p), n), m)"),
])
def test_lowering_nests_one_sum_per_index(text, expected):
    """Outermost is the first index written — reading order, the same
    OUTER→INNER convention `integral_descriptor` uses for its bounds."""
    assert _cas(text) == expected


def test_a_multi_index_sum_is_not_a_product():
    """The observed symptom, pinned directly."""
    code = _cas(r"\sum_{m,n} h_{m,n}")
    assert "sum(m, n)" not in code, code
    assert code.startswith("sum(sum("), code


def test_the_body_keeps_all_its_factors():
    code = _cas(r"\sum_{m,n} h_{m,n} x_m y_n")
    assert code == "sum(sum(h(m, n) * x(m) * y(n), n), m)", code


class TestTheAnticommutatorIsUntouched:
    r"""`{A,B}` is the anticommutator everywhere EXCEPT a sum's subscript.
    The fix narrows where the generic reading is asked for; it must not change
    what that reading is."""

    @pytest.mark.parametrize("text", [r"\{A,B\}", "{A,B}"])
    @pytest.mark.parametrize("resolver", [MaximaSemanticResolver,
                                          JuliaSemanticResolver])
    def test_braces_still_mean_anticommutator(self, text, resolver):
        ir = resolver().resolve(SharedParser().parse(text), SymbolContext())
        assert type(ir).__name__ == "Anticommutator"


class TestEveryDummyIsBound:
    """Inside `\\sum_{m,n}` BOTH m and n are indices because the sum says so.

    Binding only the first would leave the second to the subscript policy's
    guess — which for a single subscript deliberately does NOT assume an index
    (`p_x(x)` is a label, CLAUDE.md).  Before the fix only one name was bound
    because only one was read.
    """

    def test_a_later_index_resolves_as_an_index(self):
        ctx = SymbolContext()
        code = MaximaCodeGenerator(context=ctx).generate(
            _resolve(r"\sum_{m,n} A_m B_n", ctx))
        # Both subscripts lowered to index applications, neither to a name.
        assert "A(m)" in code and "B(n)" in code, code


class TestTheSingleIndexFormsAreUnchanged:
    """The regression surface: these all worked before and must still."""

    @pytest.mark.parametrize("text,expected", [
        (r"\sum_i x_i", "sum(x(i), i)"),
        (r"\sum_{i} x_i", "sum(x(i), i)"),
        # 2026-08-08: this row used to expect `sum(x(i), i)` — the bounds the
        # researcher had written were being dropped, which is why Maxima (whose
        # `sum` takes four arguments) rejected every bounded sum.  Fixed as
        # open_items B-10 ①; the row is kept, with the value it should always
        # have had.
        (r"\sum_{i=1}^{N} x_i", "sum(x(i), i, 1, N)"),
    ])
    def test_unchanged(self, text, expected):
        assert _cas(text) == expected
