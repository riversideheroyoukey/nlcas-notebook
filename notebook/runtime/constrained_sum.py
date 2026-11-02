r"""How a CONSTRAINED Σ/Π lowers — the DECISION only, no rendering.

Split out for the reason ``julia_index_policy`` and ``subscript_policy`` give
for theirs: the material for the decision (which relation, which two indices,
which range, Σ or Π) kept growing inside the generator's ``IndexSum`` branch,
and the branch is not where a notation rule belongs.

**The measurement that moved this out of "refuse" and into "compute".**
``\sum_{i \ne j}``, ``\sum_{i<j}`` and friends were refused wholesale with
"Maxima's ``sum`` has no place for a condition".  That sentence is true and the
conclusion drawn from it was not: the CAS does not need a place for the
condition, because **a constraint between two integer indices IS a pair of
ranges**.  Measured against a live Maxima on 2026-08-17, over ``1..3``:

    sum(sum(V(i,j), j, i+1, 3), i, 1, 3)                 V(1,2)+V(1,3)+V(2,3)
    sum(sum(V(i,j), j, i,   3), i, 1, 3)                 + the diagonal
    sum(sum(V(i,j), j, 1, i-1), i, 1, 3)                 V(2,1)+V(3,1)+V(3,2)
    sum(sum(V(i,j), j, 1, 3), i, 1, 3) - sum(V(i,i),i,1,3)   all six ordered pairs

and the rewrites check out under the CAS's own differential test
(``ratsimp((i<j) + (i>j) + diagonal - full) = 0``, and the ``\ne`` form equals
``(i<j) + (i>j)``).  This is the ``\int d^3k`` shape again, which is why
CLAUDE.md's rule opens by saying to MEASURE the boundary: it is not where the
refusal said it was.

**A summation index is an integer by construction**, so ``i < j`` really is
``j`` from ``i+1`` — the same warrant ``_sum_dummy_indices`` already cites for
rewriting ``\sum_{j>i}^{M}``.  Nothing here is a guess, which is the whole test
for whether a notation may be computed rather than refused.

**``\ne`` is NOT ``<``, and that distinction is the point.**  CLAUDE.md records
that the two were once routed to a single ``all_pairs`` constraint and
``\sum_{i \ne j}`` over three sites expanded to THREE terms instead of six.
Here they are different rewrites with different term counts (N(N-1)/2 against
N(N-1)), pinned by a term count in the tests rather than by inspection.

**What is still refused, and why it is not the same question:**

* ``\langle i,j \rangle`` — nearest neighbours.  This one really has no CAS
  form, because adjacency is not a relation between the index VALUES: it is a
  property of a lattice that the notation does not name.  It keeps its refusal
  and its existing home, the ``lattice_descriptor`` route (F-1).
* ``\ne`` under a Π.  Dividing the full product by the diagonal is only a
  rewrite when no diagonal factor vanishes, and nothing in the notation says
  that.  The ORDERING relations need no division and lower under Π exactly as
  they do under Σ, so they are allowed for both.
* Anything that is not two dummy indices.  ``\sum_{i<j<k}`` is a chain and
  stays refused for the reason the chained-relation guard gives.
"""
from __future__ import annotations

from dataclasses import dataclass

#: Constraint strings as ``_CONSTRAINT_MARKS`` in ``sum_integral_prefix``
#: spells them, mapped to what the INNER index's range becomes.  ``lo``/``hi``
#: are the shared range's ends and ``i`` the outer dummy.
_ORDERING = {
    "<":      lambda lo, hi, i: (f"{i} + 1", hi),
    r"\le":   lambda lo, hi, i: (i, hi),
    ">":      lambda lo, hi, i: (lo, f"{i} - 1"),
    r"\ge":   lambda lo, hi, i: (lo, i),
}

#: The nearest-neighbour marker, which is NOT a relation between index values.
NEAREST_NEIGHBOUR = r"\langle\,\rangle"


@dataclass(frozen=True)
class ConstrainedPlan:
    """What the generator should emit for one constrained ``IndexSum``.

    ``kind`` is ``nested`` (two ranges, one shared body) or ``minus_diagonal``
    (the full double sum less its diagonal), or ``refuse`` with a reason.
    """
    kind: str                        # nested | minus_diagonal | refuse
    inner_lo: str = ""
    inner_hi: str = ""
    outer_lo: str = ""
    outer_hi: str = ""
    i18n_key: str = ""


def classify_constrained(constraint: str, dummy_names: "list[str]",
                         lo: str, hi: str, op: str) -> ConstrainedPlan:
    """Decide how ``Σ_{i REL j}`` over the shared range ``[lo, hi]`` lowers.

    *dummy_names* are the two dummies IN WRITTEN ORDER, so ``i<j`` and ``j<i``
    are told apart by position rather than by rewriting the relation — the
    researcher's spelling decides which index is the outer one.
    """
    if constraint == NEAREST_NEIGHBOUR:
        # Adjacency is a property of a lattice, not of the index values; there
        # is nothing to rewrite it INTO.  The lattice route is where this
        # notation is actually carried, and the message points at it.
        return ConstrainedPlan(
            kind="refuse",
            i18n_key="parser.unsupported_notation.adjacency_sum")
    if len(dummy_names) != 2:
        # A chain (`i<j<k`) or a lone index with a stray marker.  More than one
        # relation cannot be read as one pair of ranges, and guessing which
        # pair was meant is exactly what this module exists not to do.
        return ConstrainedPlan(
            kind="refuse",
            i18n_key="parser.unsupported_notation.constrained_sum")

    outer = dummy_names[0]
    if constraint in _ORDERING:
        inner_lo, inner_hi = _ORDERING[constraint](lo, hi, outer)
        return ConstrainedPlan(kind="nested", inner_lo=str(inner_lo),
                               inner_hi=str(inner_hi), outer_lo=lo, outer_hi=hi)
    if constraint == r"\ne":
        if op == "product":
            # Π_{i≠j} would be the full product DIVIDED by the diagonal, and a
            # division is a rewrite only when no diagonal factor is zero.  The
            # notation does not say that, so this is refused rather than
            # answered with a form that is wrong exactly when it matters.
            return ConstrainedPlan(
                kind="refuse",
                i18n_key="parser.unsupported_notation.constrained_product_ne")
        return ConstrainedPlan(kind="minus_diagonal", inner_lo=lo, inner_hi=hi,
                               outer_lo=lo, outer_hi=hi)
    return ConstrainedPlan(kind="refuse",
                           i18n_key="parser.unsupported_notation.constrained_sum")


def is_relational(constraint: "str | None") -> bool:
    """True when *constraint* relates the index VALUES (so it has a rewrite).

    Used by the two places that have to know a constrained pair sum ties its
    indices into ONE range: the resolver, which may then attribute a written
    superscript bound to both, and the placeholder allocator, which must hand
    them a SINGLE pair rather than one each.
    """
    return constraint in _ORDERING or constraint == r"\ne"
