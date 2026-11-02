r"""form_lagrangian builds CAS text; the bond-sum route claims text by its shape.

ONE-QUESTION-MANY-ANSWERS ratchet, tenth lowering: `maxima_econ_mixin.py` 2→0,
`ui/interpret_f1.py` 1→0, `ui/interpret_integral.py` 1→0.

Measured 2026-08-13 on `\sum_{i=1}^{3} c_i = u*x + v*y`:

    form_lagrangian, as the objective
        (1}^{3} c_i = u*x + v*y) - (lambda_1*((g)-(x + y)))
    form_lagrangian, as the constraint
        (x + y) - (lambda_1*((\sum_{i)-(1}^{3} c_i = u*x + v*y)))

and in the F-1 bond-sum interceptor, on
`\sum_{i=1}^{3} S_i = \sum_{\langle i,j \rangle} V_{i,j}`:

    body        (1}^{3} S_i =)*(V_{i,j})
    lhs_prefix  \sum_{i = 1}^{3} S_i = …

The F-1 one is the sharpest of the three.  `_SYM_LHS_RE` accepts `\` and `{`, so
the naive split's `\sum_{i` LOOKED like a simple name and **the route claimed an
input it should have declined** — then multiplied a fragment (carrying its own
`=`) into the bond term and wrote the doubled equation into the display.

`ui/interpret_integral.py`'s single site decides between the named and the
positional transform spec.  No realistic input separates the two readings (a
bracketed `=` would have to sit in the OLD variable's name), so it was routed
through the one definition without a measured fix behind it — recorded here so
nobody has to re-derive that.
"""
import pytest

import ui.interpret_f1 as f1


# The `form_lagrangian` half of this file was DELETED on 2026-08-14 with the
# operation itself (SCENARIO-MACRO case 3): those two tests asserted how that
# BUILDER split its operands, and a builder that no longer exists cannot be
# wrong.  The defect they were a vehicle for — a naive `=` split mangling
# `\sum_{i=1}^{3} c_i = u*x + v*y` into `1}^{3} c_i =` — is not theirs to keep:
# it lives in `shared_parser.split_relation`/`strip_lhs`, and the two routes
# below still exercise it on the same input.  With the macro gone, the
# researcher writes the Lagrangian as an ordinary expression, so the general
# registration path answers the same question (tests/test_sum_family_ring2.py).


class TestTheBondSumRoute:
    r"""It must DECLINE what it cannot name, not claim it with a fragment."""

    def test_a_bounded_sum_on_the_left_is_declined(self):
        assert f1._parse_symbolic_bond_sum(
            r"\sum_{i=1}^{3} S_i = \sum_{\langle i,j \rangle} V_{i,j}") is None

    def test_a_simple_name_is_still_claimed(self):
        parsed = f1._parse_symbolic_bond_sum(
            r"H = \sum_{\langle i,j \rangle} V_{i,j}")
        assert parsed is not None
        prefix, descriptor = parsed
        assert prefix.startswith("H = ")
        assert descriptor["body"] == "V_{i,j}"

    def test_no_equation_at_all_still_works(self):
        parsed = f1._parse_symbolic_bond_sum(
            r"\sum_{\langle i,j \rangle} V_{i,j}")
        assert parsed is not None
        # No `=`, so no label to prepend — the prefix is the redisplayed sum
        # itself (measured; the route always returns what it will display).
        assert "=" not in parsed[0]
