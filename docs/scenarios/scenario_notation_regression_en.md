# Notation regression: the spellings that once broke, typed again every time

## Purpose

**This scenario is not a line of research — it is a list of notations.** Every
spelling below is one this system read wrongly while believing it read it, and
it is here so that the example regeneration checks, on every run, that it has
not gone back to the old reading.

> **Why it is needed (measured 2026-09-11).** Regenerating the examples is what
> catches display regressions here — but it is **structurally blind to a
> spelling no scenario uses**. Three defects were found on one day (every prime
> shown as a dagger; `\vec A(t)` read as the vector times the time; `vec(A)_x`
> handed to the CAS as raw text), and **all 96 published examples were
> untouched**, with 5000+ tests and the static scan green. What found them was
> typing them (handoff (reference omitted from the public snapshot) §5.11).

**Its limit, stated plainly.** This catches REGRESSION, not undiscovered
defects — the same property the instruction corpus has, being built from
instructions that work. New defects still come from ① typing the spellings the
grammar itself claims (`scripts/nl_spelling_coverage.py`), ② re-typing what the
system printed, and ③ the practice loop.

**How to read it.** What matters is not "no error" but **what it became**.
`\vec{A}(t)` registered successfully while broken; what differed was the
meaning. The generated `.json` / `.tex` carry the canonical and the result, so a
change of meaning shows up as a diff.

## What each spelling checks

| spelling | what it once became | ticket |
|---|---|---|
| `\psi_\sigma` | `\psi_{sigma}` — the italic word | GREEK-SUB-ON-GREEK-BASE |
| `G(t, t')` | `G(t, t^{\dagger})` on the Julia side | PRIME-DISPLAY-DAGGER |
| `\mathbf r` | displayed as `vec(r)` — CAS spelling on screen | PRIME-DISPLAY-DAGGER |
| `\vec{A}(t)` | `vec(A) * t` — the vector times the time | VEC-APPLIED |
| `\nabla \times \vec{A}` | registered as an expression, never computed | VEC-APPLIED |
| `\lim_{x \to 0} \frac{\sin x}{x}` | `limx,to,0 (reference omitted from the public snapshot)` — a subscripted function | LIMIT-ONE-SIDED |
| `\lim_{x \to 0^+} \frac{1}{x}` | the side never reached the CAS | LIMIT-ONE-SIDED |
| `\lim_{t' \to t^+} G(t,t')` | one undefined symbol for the whole right side | GF-NOTATION |
| `the third derivative` | English dropped the order and differentiated once | DIFF-ORDER-EN-ORDINAL |
| `compute the determinant of [4, 0; 0, 1]` | English answered with **another matrix's determinant** | MATRIX-INLINE-OPERAND-EN |
| `the norm of A` (a declared name) | "required arguments are missing: ['expr']" — for an operand that was written | OPERAND-BOUNDARY 残① |
| `f を x で偏微分して` (a declared name) | "could not interpret" — `偏微分` was not an alias, so JAPANESE alone fell through | OPERAND-BOUNDARY 残①（続） |
| `p is the canonical momentum of q` | `define_scalar_parameter` — **a SCALAR `p` registered, with success** | DECL-COPULA |
| `X is the Laplace transform of x` | the transform was COMPUTED instead of defined; the name `X` bound nothing | DECL-COPULA |
| `replace E28 in E27` | "required arguments are missing: ['lhs', 'rhs']" — fields the sentence never had | SUBST-VERB |
| `in E27, replace 5 for v` | worked in English; the Japanese sibling `置き換えて` did not | SUBST-VERB |
| `solve for u and z in E31 and E32` | **solved the FIRST equation only**, as `completed` | SOLVE-ORDER |
| `the second partial derivative of E35 with respect to x` | **silently the FIRST derivative**, `completed` | DERIV-ORDER-KIND |
| `the curl of \vec{A}` | **raw `Maxima evaluation error`** — the leading article kept the written field from being registered | REFUSAL-NOT-REACHED ④ |
| `\nabla \times \vec{A}` typed again AFTER `y = x^3` | **raw `Maxima evaluation error`** — and so did every other vector-calculus spelling, in both languages | COORD-NAME-BOUND |
| `define B as a vector field` after `y = x^3` | **raw `Maxima evaluation error`** — the declaration names its variables of dependence (`depends([B_x,…],[x,y,z])`) | COORD-NAME-BOUND (third exit) |
| `the dot product of \vec{A} and \vec{B}` | **switched A's derivative display to dot notation**, `confirmed` — no product computed | VEC-BINARY-OPERAND-WHITESPACE |
| `the cross product of \nabla \times \vec{A} and \vec{B}` | the operands split on whitespace: `\nabla` × `\times`, `completed` | VEC-BINARY-OPERAND-WHITESPACE |
| `\nabla \cdot (\nabla \times \vec{A})` | registered, never computed — every LaTeX nested spelling, both languages | NESTED-VEC-OP-LATEX |
| `the curl of the curl of A` | **the curl applied ONCE**, `completed` | NESTED-VEC-OP-LATEX |
| `\nabla s` (`s` defined by `s = x^5`) | registered — the interceptor read only a declared KIND | NESTED-VEC-OP-LATEX |
| `the Laplacian of s` | **`20*x^3 = 20*x^3`** — an equation, `completed` | NESTED-VEC-OP-LATEX |
| `the curl of 2 P` | **the curl of `P`** — the coefficient dropped, `completed` (six operations, both languages) | OPERAND-COEFFICIENT-DROPPED |
| `the divergence of P + 2 B` | refused: "E… is a scalar" — `P` holds a vector value, recorded as a scalar | H1-VECTOR-ALGEBRA-DEFAULT |
| `the divergence of \vec{P} \times \vec{A}` | the divergence of `P` alone, `completed` | OPERAND-COEFFICIENT-DROPPED |
| `differentiate E61 with respect to m totally`, after `2 m = 8` was solved | — (**a sentence that works**. Refusing a value-bearing name in the variable slot from the LEDGER alone takes this one with it) | DEFINED-NAME-AS-VARIABLE |
| `integrate E61 over m`, after `2 m = 8` was solved | **"required arguments are missing: ['var']"** — 2a-sym rewrote `m` to an E-id, so the refusal was about a field the researcher HAD written (9 spellings, both languages) | DEFINED-NAME-VAR-REWRITTEN |
| `differentiate E66 with respect to \\bar{y} partially` | **`0`** — the expression's side was `y_bar` and the variable's side a function call `bar(y)` (both languages; `\\hat{H}`, `\\tilde{x}` too) | DECORATED-NAME-AS-DIFF-VAR |
| `differentiate E66 with respect to t' partially` | **`0`**, `completed` — the shared `symbol` class had no `'`, so `t'` was TRUNCATED to `t` (Japanese failed loudly) | DECORATED-NAME-AS-DIFF-VAR |
| `integrate E66 over \\bar{y}` | "required arguments are missing: ['var']" — the **alias-scan path** could only take a plain identifier as the variable, so `\\bar{y}`, `y'` and `x_1` were all dropped (24 cells, both languages) | DECORATED-VAR-NOT-TOKENIZED |
| `change variable r to u in E70` | the integral reached the CAS unlowered and came back **unchanged**, `completed`; and in this scenario `u` has been SOLVED, so 2a-sym rewrote it — the English change-of-variables frame was missing from the guard | CHANGE-VARIABLE-NOOP |
| `2 (x)^3` / `2^2 (x)` | **`8 x^3`** and **`2^{2x}`** — the exponent wrapped the whole juxtaposition, and the exponent swallowed the following factor (both languages) | JUXT-POW-PRECEDENCE |
| `solve E72 and E73 in \\alpha and \\beta` | **`\\alpha = 3-\\beta`, `completed`** — a SYSTEM answered by solving the first equation alone for one variable. `in` was spelled in `solve_single` only, so this sentence matched no key and the alias scan answered (Japanese was 6/6 right) | NL-GRAMMAR ① question 54 |

## Input steps

```scenario domain=electromagnetism
@note: Notation regression. Spellings that once broke, typed again so the example diff shows any change of meaning.
define A as a vector field
define f as a scalar field
\psi_\sigma
simplify E1
G(t, t')
simplify E3
\mathbf r
simplify E5
\vec{A}(t)
\nabla \times \vec{A}
\nabla \cdot \vec{A}
\nabla \vec{f}
the curl of \vec{A}
\lim_{x \to 0} \frac{\sin x}{x}
simplify E13
\lim_{x \to 0^+} \frac{1}{x}
simplify E15
\lim_{t' \to t^+} G(t,t')
y = x^3
the third derivative of E18 with respect to x
D = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}
compute the determinant of [4, 0; 0, 1]
the norm of A
take the partial derivative of f with respect to x
p is the canonical momentum of q
the Poisson bracket of q and p
X is the Laplace transform of x
w = a v + b
v = 2 t
replace E28 in E27
in E27, replace 5 for v
u + z = 3
u - z = 1
solve for u and z in E31 and E32
s = x^5
the second partial derivative of E35 with respect to x
\nabla \times \vec{A}
define B as a vector field
the dot product of \vec{A} and \vec{B}
the cross product of \nabla \times \vec{A} and \vec{B}
\nabla \cdot (\nabla \times \vec{A})
the curl of the curl of A
\nabla s
the Laplacian of s
P = [0, x, 0]
the curl of 2 P
the divergence of P + 2 B
the divergence of \vec{P} \times \vec{A}
Q = [z, 0, y]
the divergence of 2 (\vec{P} \times \vec{Q})
the divergence of \vec{P} \times \vec{Q} + \vec{A}
integrate a x^2 over x
differentiate a x^2 in x totally
K = 3 (\vec{P} \times \vec{Q})
simplify K
m^2 + 3
2 m = 8
solve E62 for m
differentiate E61 with respect to m totally
integrate E61 over m
\bar{y}^2 + t'
differentiate E66 with respect to \bar{y} partially
differentiate E66 with respect to t' partially
integrate E66 over \bar{y}
\int r^2 \, dr
change variable r to u in E70
\alpha + \beta = 3
\alpha - \beta = 1
solve E72 and E73 in \alpha and \beta
expand the neighbour sum of Sz_i Sz_j over a 4-site periodic chain
expand the nearest-neighbor sum of Sz_i Sz_j in a 3x3 honeycomb lattice
y'^3
differentiate E78 with respect to y'
2 (x)^3
simplify E80
2^2 (x)
simplify E82
```

## The readings this pins

- `\psi_\sigma` → `psi_sigma`, and **the simplified result displays as `\psi_{\sigma}`**
- `G(t, t')` → the simplified result displays as `G\left(t , t'\right)`
  (a prime marks a DISTINCT VARIABLE, not an adjoint)
- `\mathbf r` → the simplified result displays as `\mathbf{r}` (`vec(` never reaches the screen)

> **Why the `simplify` lines are there.** An expression that was only registered
> is displayed using **the researcher's own LaTeX**, so a fix on the display side
> never reaches the published artifact — noticed immediately after writing this
> file, which had missed two of its own families that way. Running one operation
> makes the display be BUILT, and only then does a display regression show up in
> the diff.
- `\vec{A}(t)` → `vec(A(t))` (the wrapper goes around the APPLICATION; written
  with a space, `\vec{A} (t)` stays a product)
- `\nabla \times \vec{A}` / `\nabla \cdot \vec{A}` / `\nabla \vec{f}` → **computed**
- `the curl of \vec{A}` → **computed**, and to the same value as the notation spelling above. The bare article stood in front of the operation's alias, so the written field was never registered and a raw `Maxima evaluation error` came back; the Japanese sibling `\vec{A} の回転を求めて` was right throughout
- `\lim_{x \to 0} \frac{\sin x}{x}` simplified → **1**
- `\lim_{x \to 0^+} \frac{1}{x}` simplified → **inf** (proof the side reached the
  CAS; two-sided would be `infinity`)
- `\lim_{t' \to t^+} G(t,t')` → kept as `limit(G(t, t'), t', t, plus)`
- the third derivative of `x^3` → **6**
- `differentiate E61 with respect to m totally`, after `2 m = 8` was solved for `m`
  → **`2*m`**. **A solved variable is still a variable** — a solve result carries a
  `defining_expr_id` and yet binds nothing CAS-side (measured), so writing the refusal
  "a name that holds a value is not a variable" against the LEDGER refuses this line for
  no reason. What decides is the CAS
- `integrate E61 over m`, after `2 m = 8` was solved for `m` → **`m^3/3+3*m+C_1`**.
  2a-sym's guard for "is this symbol the operation's variable?" was a HAND-WRITTEN list of
  frames, narrower than the grammar's, so the name became an E-id for integrate, limit,
  residue, partial fractions, change-of-variables and substitute. **The same sentence works
  before the solve** — that is why this line stands after it
- `differentiate E66 with respect to \bar{y} partially` → **`2*y_bar`**, and with respect to
  `t'` → **`1`**. The expression goes through `latex_to_canonical`; **the variable did not**, so
  one name became two CAS objects and the answer was `0`. The prime was truncated by the shared
  `symbol` class. `y = x^3` stands above, so this line also shows **`\bar{y}` is not `y`**
- `integrate E66 over \bar{y}` → **`y_bar^3/3+t'*y_bar+C_2`**. The derivative was already
  fixed because an interceptor reads it; the INDEFINITE integral is answered by the alias
  scan, which carried a hand-written rule saying "a plain identifier, or `\alpha`".
  `x_1` fell into the same hole — and that is an ordinary variable
- `change variable r to u in E70` → **`'integrate(u^2,u)`**. The operand was not lowered, and
  `changevar` only acts on an integral NOUN (Maxima evaluates `integrate(...)` first). With no
  integral the operation is refused, naming `substitute`. **This scenario has SOLVED `u`**, which
  is what exposed the second half: the guard's frame table had the change of variables in
  Japanese only
- `solve E72 and E73 in \alpha and \beta` → **`\alpha = 2`, `\beta = 1`**. This is the
  line the fix is about: it used to return `\alpha = 3-\beta` — one equation, one variable,
  `completed`. The three solve keys now read ONE table for the word that introduces the
  variable, which is what the Japanese side always did
- the partial derivative of the declared `f` → **`'diff(f,x,1)`**. English always
  worked here (`derivative` is an alias); the Japanese sibling
  `f を x で偏微分して` did not, because the verb tail is stripped to `偏微分`
  and only `微分` was in the table — and only for a declared name, since an E-id
  is claimed by `diff_rule` first
- the norm of the declared `A` → **`sqrt(A_z^2+A_y^2+A_x^2)`**. A declared object
  has no `defining_expr_id`, so its name was never rewritten to an E-id and could
  not be passed as an operand (operations that need a VALUE still refuse — stating
  the condition that is not met)
- the determinant of `[4, 0; 0, 1]` → **4**. The `D` above it (determinant 3) is a
  DECOY — the leading compute verb kept the written matrix from being registered,
  so this line answered **3, D's determinant**, as `completed`. Without the decoy
  the defect only looks like a failure and stays invisible
- `p is the canonical momentum of q` → **the canonical pair (q, p) is registered**,
  and the `the Poisson bracket of q and p` line that follows returns **1** — which
  is the proof that it really was registered (without a pair the bracket fails
  asking for one). Before the copula was folded, `is` alone sent this sentence to
  the generic declaration, which registered **a scalar named `p`** and reported
  success
- `X is the Laplace transform of x` → the canonical is
  **`'integrate(x*exp(-(s)*(t)),t,0,inf)`**, the definition itself. Before, the key
  required a leading `let`/`define`, so this spelling became a scalar
- `replace E28 in E27` → **`2*a*t+b`**. Before, only `substitute` was in the
  expression-into-expression frame's verb table, so `replace` was refused with
  "required arguments are missing: ['lhs', 'rhs']" — a refusal naming two fields
  the sentence never had
- `in E27, replace 5 for v` → **`5*a+b`**. English always worked here; the
  Japanese sibling `E27 の v に 5 を置き換えて` did not, because `substitute_into`
  spelled inflections for `入れ` and not for `置き換え`
- `solve for u and z in E31 and E32` → **two equations, u = 2 and z = 1**. The
  vars-first word order had no key on the English side, so the alias scan answered
  it and **solved the FIRST equation alone** — `3-z`, reported as `completed`. The
  Japanese sibling failed loudly, so this was a ja/en split too
- `the second partial derivative of E35 with respect to x` → **`20*x^3`**
  ($\partial^2/\partial x^2$). The ordinal's lookahead required the derivative NOUN
  immediately, so a KIND word in between dropped the order and the FIRST derivative
  came back `completed` — 36 of 64 ordinal × kind × frame spellings. Japanese was
  right throughout, because its sibling key reads `ja_deriv_word`, which already
  composes the kind words
- **`\nabla \times \vec{A}` is typed a SECOND time, at the very end** — and it must
  land on the value the first one did. What stands between them is `y = x^3`, which
  H-1's P2 sends to the CAS as `y : x^3`; the `nb_curl_*` functions spell the
  coordinate variables inside themselves, so `diff(A_z,y)` became "differentiate by
  `x^3`" and the researcher got a raw `Maxima evaluation error`. Every vector-calculus
  spelling fell that way, in both languages — divergence, gradient and the Laplacian
  too. `∇` BINDS the coordinates the way `∑` binds its index, and the researcher never
  wrote them, so they are shadowed for the call (COORD-NAME-BOUND). **The position is
  the point**: the identical spelling higher up cannot catch this, because nothing had
  bound a coordinate name yet
- `define B as a vector field` → **declared**, although `y = x^3` stands above it. A
  declaration names its variables of dependence, so a binding on `y` reached
  `depends([B_x,B_y,B_z],[x,y,z])` and the researcher got a raw
  `Maxima evaluation error` — only "declare, then bind" had been measured
- `the dot product of \vec{A} and \vec{B}` → **`A_z*B_z+A_y*B_y+A_x*B_x`**. Before, the bare word
  `dot` fired the NOTATION switch (`notation` was optional in English, required in
  Japanese), so this line changed a session setting and reported `confirmed`; and
  had it reached the product, `\vec{A}` would have become the invented component
  `\vec{A}_x` — a raw `Maxima evaluation error`. Japanese hit the second defect only
- `the cross product of \nabla \times \vec{A} and \vec{B}` → **computed**: the curl of `A`
  crossed with `B`. The operands used to split on whitespace (`\nabla` × `\times`,
  `completed`). It also stands after `y = x^3`: reading a ∇ operand puts `nb_curl_*`
  inside the product, a NEW exit for COORD-NAME-BOUND, so only that operand is shadowed
- `\nabla \cdot (\nabla \times \vec{A})` → **0**. The nested reader knew only Unicode and names,
  so the LaTeX spelling — what a researcher writes, and what the system prints — was
  registered as a formula while `∇·(∇×A)` computed 0
- `the curl of the curl of A` → **the curl applied twice**. English had no chain key; the alias
  scan saw one operation word and applied it ONCE, `completed`
- `\nabla s` → **`[5*x^4,0,0]`**. `s` was given a value by `s = x^5`, and the notation's
  interceptor checked only a declared kind, so it registered while `s の勾配` computed
- `the Laplacian of s` → **`20*x^3`**. It was `20*x^3 = 20*x^3`: the scalar builder read the
  equation instead of its right-hand side, for every scalar defined by `=`
- `the curl of 2 P` → **`[0,0,2]`**. The operand gate counted only operator SYMBOLS as mathematics,
  so `2 P` — a juxtaposition, this system's multiplication — was one name: `P` was rewritten to its
  expression id and the `2` was dropped. Six operations fell that way, in both languages
- `the divergence of P + 2 B` → **`2*'diff(B_z,z,1)+…`**. A name given a VECTOR value was recorded
  with the algebra `scalar`, and the type check read that default as a declaration; the type is now
  read from the value, and from the expression's structure (a sum of vectors is a vector)
- `the divergence of \vec{P} \times \vec{A}` → **computed**. The cross product was dropped after
  its first name; the components now come from the system's own `nb_cross`, because Maxima's `~`
  does not evaluate on lists
- `the divergence of 2 (\vec{P} \times \vec{Q})` → **`2*y-2*x`**. A cross product ON ITS OWN was
  read; buried in a larger expression it was refused, because which text `nb_cross` replaces was
  decided by splitting on a top-level `~` (CROSS-PRODUCT-INSIDE-OPERAND). What an operator binds
  is a question of syntax, so the rewrite now happens on the TREE
- `the divergence of \vec{P} \times \vec{Q} + \vec{A}` → the divergence of **`(P×Q) + A`**.
  That is the point of this line: with no parentheses written, the cross product still binds first.
  Splitting the string on `~` gives `P × (Q + A)` — a different quantity, answered `completed`
- `simplify K` → **`[3*x^4,0,-(3*x)]`**. `K = 3 (\vec{P} \times \vec{Q})` registered and
  displayed, but every GENERIC operation on it returned a raw `Maxima evaluation error` — the
  `~` the generator writes is not evaluable in Maxima (list or atom alike, measured). Six
  operations fell that way; only the three ∇ ones worked, because they go through the component
  reading. Only the text HANDED TO THE CAS is swapped (`nb_cross` over component lists); the
  researcher's own canonical still carries `~`
- `differentiate a x^2 in x totally` → **`2*a*x`**. The derivative frame spelled the preposition by
  HAND (`with respect to|w.r.t.|by`), so `in` / `for` / `over` never reached it and the fields
  were filled by POSITION: `differentiate(expr='x^2', var='a')` → **`0`**, `completed` — 3 of 6
  prepositions. It now reads the SHARED table, the one `def_integral[en]` already read
- `integrate a x^2 over x` → **`(a*x^3)/3+C`**. English had no inline frame for the integral
  (the derivative has one), so the integrand was never registered and the fields were filled by
  POSITION: `integrate(expr='x^2', var='a')` — the coefficient became the variable and the `x`
  the researcher wrote was dropped, `completed`. **The pair with the Japanese line is the point**:
  the two must land on the same value

- `expand the neighbour sum of Sz_i Sz_j over a 4-site periodic chain` → **four bonds**.
  Japanese was right and **the English pair was broken**: the two languages' sum-word tables
  answered DIFFERENT questions — Japanese held the sum NOUN, English only the adjective
  `nearest neighbou?r`, with `sum` and `lattice sum` spelled inside the keys — so the mirror
  of `隣接和` existed nowhere. In the 1D form it failed loudly
- `expand the nearest-neighbor sum of Sz_i Sz_j in a 3x3 honeycomb lattice` → **nine bonds**.
  **The same hole is SILENT in the 2D form**: when the body cannot be read, the whole English
  sentence became the summand and `3 Sz(8) Sz(9) Expand a neighbor of over sum the …` came back
  `completed`. `in` was missing from the body's own terminator list. This pair is the asymmetry
  itself — the Japanese boundary is grammar (particles and the script change), the English one
  is vocabulary

- `differentiate E78 with respect to y'` → **`3*y'^2`**. A primed name was not in
  `canonical_variables`' identifier scan, so `y'^3`'s variables read as `{y}` — and the
  system then asked "partial or total?" about an expression **with one symbol in it**, in
  both languages. The parser has always treated `'` as part of an identifier (`f'(x)`);
  only the scan did not. **The `… with respect to t' partially` line above spells
  `partially` because it was written around this defect.**

## The contract for adding to this file

1. **Write both languages in the same order, the same count.** The instruction
   corpus pairs the two by `(scenario, line index)`, so adding to one side alone
   breaks the pairing.
2. **Write it so the meaning is visible.** When registering alone does not fix
   the meaning, run an operation after it and land on a value (that is what the
   `simplify` lines are for).
3. **A different backend goes in a different file.** The second-quantization
   (Julia) side is `scenario_notation_regression_sq_en.md`.
