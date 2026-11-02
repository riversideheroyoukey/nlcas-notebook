# Practice-loop sample scenario: constrained utility maximisation and comparative statics

Maximise a Cobb–Douglas utility $U(x_1,x_2)=x_1^ax_2^b$ subject to the budget constraint
$p_1x_1+p_2x_2=m$, and obtain the demand functions and their income sensitivity
(comparative statics).

**Backend:** standard (Maxima) / **domain:** mathematical economics.

## SCENARIO-MACRO, case 3 — rewritten with general operations only

**On 2026-08-14 this scenario was rewritten** from a form that used the dedicated
`form_lagrangian` / `stationary_conditions` / `implicit_derivative` operations (open_items
SCENARIO-MACRO).  Inside they were, respectively, "objective − multiplier × constraint",
a loop calling `differentiate` once per variable, and the one-line template
`-diff(F,p)/diff(F,x)` — **all of them expressions a researcher can write**.

**Where the knowledge went:** both the shape of the Lagrangian and the implicit function
theorem are now **one step the researcher types** (steps 6 and 16).  Both are economics;
they do not belong to the system.  The automatic multiplier naming (`lambda_1`) is gone
too — **the researcher names it**.

**Fixed in passing:** the old scenario wrote `U = x1^a * x2^b` / `p1*x1 + p2*x2 = m` — CAS
syntax, with `*` and unsubscripted `x1`.  That was not a stylistic choice: it looks like a
spelling **chosen to dodge the declaration described below**.  The rewrite uses the real
notation, $x_1,x_2,p_1,p_2$.

## Steps and expected results

| # | Input (natural language / math) | Result |
|---|---|---|
| 1–4 | `declare x_1 as a scalar`, and likewise $x_2,p_1,p_2$ | registered as subscripted NAMES (choose "part of the name"; why below) |
| 5 | `$U = x_1^a x_2^b$` | the utility function |
| 6 | `$L = U - \lambda (p_1 x_1 + p_2 x_2 - m)$` | **the researcher writes the Lagrangian**, and names the multiplier $\lambda$ |
| 7 | `take the partial derivative of E2 with respect to x_1` | $\dfrac{\partial L}{\partial x_1}=a x_1^{a-1}x_2^{b}-p_1\lambda$ |
| 8 | `take the partial derivative of E2 with respect to x_2` | $\dfrac{\partial L}{\partial x_2}=b x_1^{a}x_2^{b-1}-p_2\lambda$ |
| 9 | `take the partial derivative of E2 with respect to \lambda` | $\dfrac{\partial L}{\partial \lambda}=m-p_1x_1-p_2x_2$ |
| 10–12 | `set the right-hand side of E3 to 0`, likewise E4, E5 | the three first-order conditions |
| 13 | `$F = a m - (a+b) p_1 x_1$` | the demand relation written explicitly as $F(x_1,m)=0$ |
| 14 | `take the partial derivative of E9 with respect to m` | $\dfrac{\partial F}{\partial m}=a$ |
| 15 | `take the partial derivative of E9 with respect to x_1` | $\dfrac{\partial F}{\partial x_1}=-(a+b)p_1$ |
| 16 | `$\frac{dx_1}{dm} = -\frac{E10}{E11}$` | **the researcher writes the implicit function theorem**: $\dfrac{dx_1}{dm}=\dfrac{a}{(a+b)p_1}$ |
| 17 | `solve E6, E7 and E8 for x_1, x_2, \lambda` | the demand functions $x_1=\dfrac{am}{(a+b)p_1}$, $x_2=\dfrac{bm}{(a+b)p_2}$, and $\lambda$ |
| 18 | `take the partial derivative of E13 with respect to m` | $\dfrac{\partial x_1}{\partial m}=\dfrac{a}{(a+b)p_1}$ — **agrees with step 16** |

## Confirmed behaviour and caveats

- **The agreement between steps 16 and 18 is what this scenario produces.**  The
  sensitivity from the implicit function theorem and the one obtained by differentiating
  the solved demand function come out equal.  **The macro could not do this** —
  `implicit_derivative` returned a single answer with nothing to compare it against.
- **⚠ Step 17 (solving) must come after steps 13–16.**  The solved result $x_1=\dots$ is
  registered as an H-1 definition, and P2 **binds $x_1$ on the CAS side**.  Writing
  $F=am-(a+b)p_1x_1$ afterwards collapses $F$ identically to 0, and
  $\partial F/\partial m=0$ comes out — a **plausible-looking wrong answer** (measured).
  The ordering is part of what this scenario says.
- **A subscripted name must be declared.**  Used without declaration, $x_1$ resolves at
  operation time as the **component index** `x(1)`, the $U$ term contributes nothing to
  the derivative, and the first-order condition comes out as
  $\partial L/\partial x_1=-p_1\lambda$ — wrong (measured).  This is the documented
  behaviour ("an undeclared subscript in an OPERATION is an index"), not a defect; steps
  1–4 exist for it.
- **The researcher names the multiplier.**  The macro auto-numbered `lambda_1` to avoid
  collisions; writing $\lambda$ simply uses that name.  The system not inventing names is
  the simpler arrangement.
- **The system solve is written the way English enumerates: `solve E6, E7 and E8 for …`.**
  Until 2026-08-16 that spelling **fell through to a single-equation solve and returned
  an unrelated answer as a success**, so this scenario used `and` throughout.  Measuring
  the repair found three more defects behind it, all reached by an ordinary system that
  contains an equation like `z = 2` — the solve-for variables were being rewritten to
  expression ids, and a symbol bound CAS-side by H-1 could not be solved for at all.
  Two of those cancelled, which is why the `and` form had looked correct.

```scenario domain=mathematical_economics
@note: Maximise the Cobb–Douglas utility $U(x_1,x_2)=x_1^ax_2^b$ subject to $p_1x_1+p_2x_2=m$. First declare the subscripted symbols as NAMES — undeclared, they resolve at operation time as the component index $x(1)$ and the first-order conditions come out wrong.
declare x_1 as a scalar
declare x_2 as a scalar
declare p_1 as a scalar
declare p_2 as a scalar
@note: Register the utility function.
$U = x_1^a x_2^b$
@note: The researcher writes the Lagrangian, and names the multiplier (the system need not auto-number one).
$L = U - \lambda (p_1 x_1 + p_2 x_2 - m)$
@note: For the first-order conditions, differentiate the Lagrangian with respect to each variable.
take the partial derivative of E2 with respect to x_1
take the partial derivative of E2 with respect to x_2
take the partial derivative of E2 with respect to \lambda
@note: Set each right-hand side to zero to get the first-order conditions.
set the right-hand side of E3 to 0
set the right-hand side of E4 to 0
set the right-hand side of E5 to 0
@note: Do the comparative statics FIRST. Write the demand relation explicitly as $F(x_1,m)=am-(a+b)p_1x_1=0$. This step must precede the solve: solving binds $x_1$ on the CAS side, which collapses $F$ identically to 0.
$F = a m - (a+b) p_1 x_1$
@note: Obtain the two partial derivatives the implicit function theorem needs.
take the partial derivative of E9 with respect to m
take the partial derivative of E9 with respect to x_1
@note: The researcher writes the implicit function theorem $dx_1/dm=-F_m/F_{x_1}$, pulling in the previous results by expression reference.
$\frac{dx_1}{dm} = -\frac{E10}{E11}$
@note: Solve the first-order conditions for the demand functions. (Written with `and` rather than a comma list — see open_items SOLVE-LIST-COMMA-EN.)
solve E6, E7 and E8 for x_1, x_2, \lambda
@note: Differentiate the solved demand function with respect to $m$ directly, and confirm it agrees with the implicit-function-theorem result. The macro returned one answer with nothing to compare it against.
take the partial derivative of E13 with respect to m
```
