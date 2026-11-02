# Practice-loop sample scenario: Battle of the Sexes, pure and mixed equilibria

Find the pure-strategy Nash equilibria and the mixed-strategy equilibrium of the classic
two-player game Battle of the Sexes.

**Backend:** standard (Maxima) / **domain:** game theory.

## SCENARIO-MACRO, case 4 — only the macro came off; the primitive stayed

**On 2026-08-14 `mixed_strategy_indifference_eq` /
`mixed_strategy_indifference_conditions` were removed** (open_items SCENARIO-MACRO).
Inside was a fixed formula assembling the expected-payoff equations from matrix entries —
something **the researcher can write**.

**`pure_nash_equilibria` stayed.**  It scans a bimatrix asking whether each cell is both a
row maximum and a column maximum, which no composition of the existing general operations
expresses — the ticket's "genuine primitive".  **Removing every macro is not the goal.**

**The side finding was the larger one:** that same `pure_nash_equilibria` was
**completely unreachable from natural language**.  Its aliases were registered, but the
positional field mapper did not count `payoff_a`/`payoff_b` as **expression** fields, so
neither could be filled and every phrasing failed — including typing the alias itself.
The defect BP-6 recorded for `convolve`'s `f`/`g` had simply never been fixed for this
operation.  Worse, `_EXPR_FIELDS` existed in **two places** and the BP-6 fix had landed in
only one of them.  Unified into one definition with `payoff_a`/`payoff_b` added, and
**Japanese now reaches it** (step 3).

## Steps and expected results

$A$ holds player 1's (row) payoffs and $B$ player 2's (column) payoffs.  Row 1 / column 1
is "opera", row 2 / column 2 is "ballet".

| # | Input (natural language / math) | Result |
|---|---|---|
| 1 | `$A = [2, 0; 0, 1]$` | player 1's payoff matrix |
| 2 | `$B = [1, 0; 0, 2]$` | player 2's payoff matrix |
| 3 | `find the pure Nash equilibria of E1 and E2` | $\{(1,1),(2,2)\}$ — the two pure equilibria where both choose the same event  |
| 4 | `$2 q + 0 (1-q) = 0 q + 1 (1-q)$` | **the researcher writes player 1's indifference condition**: with the opponent playing column 1 with probability $q$, rows 1 and 2 have equal expected payoff |
| 5 | `solve E4 for q` | $q = \dfrac{1}{3}$ |
| 6 | `$1 p + 0 (1-p) = 0 p + 2 (1-p)$` | player 2's indifference condition, against an opponent playing row 1 with probability $p$ |
| 7 | `solve E6 for p` | $p = \dfrac{2}{3}$ |

The mixed equilibrium is $(p,q)=(2/3,\,1/3)$, matching the macro's values (measured before
deleting it).

## Confirmed behaviour and caveats

- **An indifference condition is an equation between expected payoffs — game theory, not
  system behaviour.**  Nothing needs to assemble it from matrix entries.  The researcher
  writes it, and because the written equation stays on the timeline, **which payoffs are
  being compared is readable**.
- **The macro's defensive form is no longer needed.**  It deliberately built
  `lhs - rhs = 0` because a result collapsing to a bare `q = <expr>` would make H-1's P2
  bind `q` CAS-side.  The root-cause guard (an operation's result never P2-binds, BP-7) is
  untouched and is pinned by
  `tests/test_game_theory.py::test_bp7_root_cause_operation_result_never_p2_binds` —
  **removing the macro's workaround leaves the real fix in place.**
- **Step 3 is typed the way a researcher would say it.**  Until 2026-08-15 it had to type
  the operation's own name instead: all three of this operation's English aliases are
  multi-word, and the alias scan read one token at a time, so in English **the operation
  may as well not have existed** (open_items EN-ALIAS-MULTIWORD, fixed at the root — the
  scan now matches phrases, longest first).

```scenario domain=game_theory
@note: The classic two-player game Battle of the Sexes. Register player 1's (row) payoff matrix $A$ and player 2's (column) payoff matrix $B$. Row 1 / column 1 is opera, row 2 / column 2 is ballet.
$A = [2, 0; 0, 1]$
$B = [1, 0; 0, 2]$
@note: Find the pure-strategy Nash equilibria. This scans the bimatrix for cells that are both a row and a column maximum — not expressible as a composition of general operations, so it is a genuine primitive and stays.
find the pure Nash equilibria of E1 and E2
@note: The mixed equilibrium comes from the indifference conditions. With the opponent playing column 1 (opera) with probability $q$, player 1's rows 1 and 2 have equal expected payoff — the researcher writes that equation. Assembling it from $A$'s entries is game theory, not the system's job.
$2 q + 0 (1-q) = 0 q + 1 (1-q)$
@note: Solve for $q$.
solve E4 for q
@note: Likewise, write player 2's indifference between columns from $B$'s entries.
$1 p + 0 (1-p) = 0 p + 2 (1-p)$
@note: Solve for $p$. The mixed equilibrium is $(p,q)=(2/3, 1/3)$.
solve E6 for p
```
