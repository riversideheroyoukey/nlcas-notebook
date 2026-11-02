# Practice-loop sample scenario: 2-variable LP vertex optimization (binding-constraint intersection → objective evaluation)

Detailed version of practice-loop scenario menu **S-s (operations research)**. For the
2-variable linear program $\max z=c_1x+c_2y$ subject to
$a_1x+b_1y\le d_1,\ a_2x+b_2y\le d_2,\ x,y\ge0$, this scenario **finds the intersection
(vertex) of the two binding constraints via `solve_system`** and evaluates the objective
$z$ at that vertex (the standard LP fact that the optimum is attained at a vertex of the
feasible region).

**Backend:** standard (Maxima) / **Domain:** operations research (operations_research).

> A minimal sample that completes with just a simultaneous solve and substitution.
> All operations are interpreted rule-based (no LLM required).

## Background

$$\max z=c_1x+c_2y \quad \text{s.t. } a_1x+b_1y\le d_1,\ a_2x+b_2y\le d_2,\ x,y\ge0.$$

The optimum is attained at a vertex of the feasible region (a point where some
constraint holds with equality). Setting the two binding constraints as equalities,
solving for the intersection (vertex), and substituting into the objective:

$$a_1x+b_1y=d_1,\quad a_2x+b_2y=d_2 \;\Longrightarrow\; (x^*,y^*) \;\Longrightarrow\; z^*=c_1x^*+c_2y^*.$$

This matches Cramer's rule: $x^*=\dfrac{d_1b_2-d_2b_1}{a_1b_2-a_2b_1}$,
$y^*=\dfrac{a_1d_2-a_2d_1}{a_1b_2-a_2b_1}$.

## Session setup / symbols

- Mode **standard (Maxima)**, domain **operations research**.
- $a_1,b_1,c_1,d_1,a_2,b_2,c_2,d_2$ are coefficients (real); $x,y$ are decision variables
  (free variables, no declaration needed).
- **The coefficients are digit-subscripted symbols (`a_1`, etc.), so declare them as
  real before use** — an undeclared digit-subscripted symbol used as an operation
  source resolves as an `IndexedSymbol` (index access) per the documented rule
  ("Undeclared subscript in an OPERATION → index"). Declaring it registers it as a
  plain symbol instead.
- **Each constraint is entered in "= 0" form** (`... - d_1 = 0`), not `... = d_1`
  (a bare-number-free RHS). With a compound LHS (a sum like `a_1 x + b_1 y`), the
  latter form routes through H-1's named-definition path, which stores only the RHS
  in canonical; `solve_system` then cannot recover the LHS and returns the empty
  solution `[]` (see "Observed behavior / caveats" below).

## Input steps

### Step 1: declare the coefficients as real

```
Declare a_1, b_1, c_1, d_1, a_2, b_2, c_2, d_2 as real.
```

**Expected output:** all eight coefficients registered as real scalars in `SymbolContext`.

### Step 2: define the objective (H-1)

```
z = c_1 x + c_2 y
```

**Expected output:** registered as E1. $z=c_1x+c_2y$.

### Steps 3-4: the two binding constraints ("= 0" form)

```
a_1 x + b_1 y - d_1 = 0
a_2 x + b_2 y - d_2 = 0
```

**Expected output:** E2 $=a_1x+b_1y-d_1=0$, E3 $=a_2x+b_2y-d_2=0$.

### Step 5: solve the system (H-9, vertex coordinates)

```
Solve E2 and E3 for x and y
```

**Expected output:** the two solutions registered as separate expressions:

$$x^*=E4=-\frac{b_2d_1-b_1d_2}{a_2b_1-a_1b_2}\;\left(=\frac{d_1b_2-d_2b_1}{a_1b_2-a_2b_1}\right),
\qquad y^*=E5=\frac{a_2d_1-a_1d_2}{a_2b_1-a_1b_2}.$$

Matches Cramer's rule (E5 is the raw solution-set bracket, consumed internally, not
shown as a separate registered expression).

### Step 6: substitute $x^*$ into the objective

```
Substitute E4 into E1
```

**Expected output:** E6 $=z=\dfrac{(a_1b_2-a_2b_1)c_2y-b_1c_1d_2+b_2c_1d_1}{a_1b_2-a_2b_1}$
(intermediate form, $y$ still free).

### Step 7: substitute $y^*$ too — objective value at the vertex

```
Substitute E5 into E6
```

**Expected output:** E7 $=z^*=\dfrac{(a_1c_2-b_1c_1)d_2+(b_2c_1-a_2c_2)d_1}{a_1b_2-a_2b_1}$.

Expanding gives $z^*=\dfrac{c_1(b_2d_1-b_1d_2)+c_2(a_1d_2-a_2d_1)}{a_1b_2-a_2b_1}$, matching
$z^*=c_1x^*+c_2y^*$ (the objective evaluated at the vertex).

## Confirmation points

- The vertex (intersection of the two binding constraints) is found correctly via
  `solve_system` and matches Cramer's rule.
- Substituting the vertex coordinates into the objective **twice in a row**
  ($x^*$ then $y^*$) leaves the coefficient symbols intact (see SUBST-DIGIT-SUB below).

## Observed behavior / caveats

- **H-1's RHS-only canonical storage vs. `solve_system` (a documented, designed
  workaround).** Entering a constraint as `a_1 x + b_1 y = d_1` (RHS not a bare
  number) routes through H-1's "named definition" path, storing only the RHS
  (`d_1`) as canonical. A compound LHS (`a_1 x + b_1 y`) is outside `_SIMPLE_LHS_RE`
  and is never reconstructed into `expr_lookup`, so `solve_system` returns the empty
  solution `[]`. Writing the constraint as `... = 0` (a bare-number RHS) skips H-1
  entirely (guard 8: a bare-number RHS is treated as a plain Maxima equation), so the
  full equality is stored as canonical and the problem does not arise — this scenario
  uses that form.
- **SUBST-DIGIT-SUB (found and root-caused in this scenario, 2026-07-05).** Even with
  the workaround above, substituting a solve result (an expression containing
  digit-subscripted coefficients like `a_1`, `b_2`) into another expression corrupted
  every **purely-digit-subscripted symbol** into a bogus function call (`a(1)`,
  `b(2)`, ...) — not just cosmetically: the stored `canonical` itself was wrong,
  a different mathematical object. This reproduced even for symbols explicitly
  declared as real beforehand. Root cause: `build_sub_name` in
  `notebook/runtime/semantic_resolver.py` (a helper that reconstructs a combined
  subscripted name to check the symbol registry) only handled a `Symbol` subscript
  node, not the `Number` node a bare digit tokenizes to — so the registry lookup for
  a digit-subscripted name always failed and silently fell through to
  `IndexedSymbol` (function-call form). Fixed by handling `Number` nodes
  (`tests/test_digit_subscript_resolution.py`). The impact was not limited to this
  scenario: any "substitute a solve result into another expression" pattern using
  digit-subscripted coefficients was affected.
- **Also found in passing, root-caused and fixed (2026-07-05):** the English
  bulk-declaration form `"Declare a_1, b_1, ..., d_2 as real."` only registered the
  **first and last** symbol in the comma list (the middle ones were silently
  dropped). Root cause: `m.groups()[:-1]` on the English define-statement regex
  only keeps the LAST match of a repeated capturing group (a standard regex
  limitation) — the Japanese path already avoided this by re-scanning the raw
  text with `re.findall` instead of relying on capture groups. Fixed by applying
  the same re-scan (bounded by the English copula "as"/"is"/"are") to the
  English path too (`notebook/runtime/backend_handler.py`,
  `tests/test_batch_symbol_declaration.py`). Filed as J-4 in open_items.md
  (now closed).

## Example files

The executed session and exported document:
[../../domains/operations_research/examples/scenario_lp_vertex_optimization_en.json](../../domains/operations_research/examples/scenario_lp_vertex_optimization_en.json)
/ `.tex` (the Japanese version is `_jp.{json,tex}`).

**Regenerate:** `PYTHONPATH=. .venv/bin/python scripts/generate_scenario_examples.py`

## Generator commands (this `scenario` block is the single source for generation)

```scenario domain=operations_research
@note: Prepare the coefficients $a_1,b_1,c_1,d_1,a_2,b_2,c_2,d_2$ of the two-variable linear program $\max z=c_1x+c_2y$ (subject to $a_1x+b_1y\le d_1,\ a_2x+b_2y\le d_2$) as real numbers.
Declare a_1, b_1, c_1, d_1, a_2, b_2, c_2, d_2 as real.
@note: Define the objective function $z=c_1x+c_2y$.
z = c_1 x + c_2 y
@note: The optimum is attained at a vertex of the feasible region (a point where some constraint holds with equality). Register the first binding constraint $a_1x+b_1y=d_1$ as an equation.
a_1 x + b_1 y - d_1 = 0
@note: Register the second binding constraint $a_2x+b_2y=d_2$.
a_2 x + b_2 y - d_2 = 0
@note: Solve the two constraints as a simultaneous system for $x,y$ to find their intersection (the vertex coordinates $x^*,y^*$).
Solve E2 and E3 for x and y
@note: Substitute the $x^*$ just found into the objective function $z$ (leaving an intermediate form still containing $y$).
Substitute E4 into E1
@note: Substitute $y^*$ as well, obtaining the value of the objective function at the vertex, $z^*=c_1x^*+c_2y^*$.
Substitute E5 into E6
```
