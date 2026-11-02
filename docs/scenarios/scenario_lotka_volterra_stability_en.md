# Practice-loop sample scenario: Lotka–Volterra stability analysis (S-q)

> (2026-07-02: an English translation of the Japanese scenario, verified against the
> real Maxima backend through the same rule-based, no-LLM generator pipeline. See
> [scenario_lotka_volterra_stability_jp.md](scenario_lotka_volterra_stability_jp.md)
> for the Japanese version.)

Detailed write-up of practice-loop scenario-menu item **S-q (mathematical biology)**.
Find the **fixed points** of the predator-prey model and determine stability (a
neutral center) from the **eigenvalues of the Jacobian matrix** there. Completes
entirely with existing general operations (`solve_system`, partial differentiation,
matrix eigenvalues).

**Backend:** standard (Maxima) / **Domain:** mathematical biology
(mathematical_biology).

## Physical background

Lotka–Volterra equations (prey $x$, predator $y$):

$$\dot x = a x - b x y,\qquad \dot y = d x y - c y.$$

- **Fixed points** (intersections of the nullclines $\dot x=\dot y=0$): the extinction
  equilibrium $(0,0)$ and the **coexistence equilibrium** $(c/d,\ a/b)$.
- **Jacobian at the coexistence equilibrium**:

$$J=\begin{pmatrix}\partial_x\dot x & \partial_y\dot x\\ \partial_x\dot y & \partial_y\dot y\end{pmatrix}
=\begin{pmatrix}a-by & -bx\\ dy & dx-c\end{pmatrix}
\xrightarrow{(c/d,\,a/b)}
\begin{pmatrix}0 & -bc/d\\ ad/b & 0\end{pmatrix}.$$

- **Eigenvalues** $\lambda=\pm\sqrt{-ac}=\pm i\sqrt{ac}$ (purely imaginary) = a
  **neutral center** (Lyapunov stable but not asymptotically stable). Populations
  orbit around the fixed point.

## Steps and expected results

| Step | Input (natural language) | Expected result |
|---|---|---|
| 1 Nullclines | `a*x - b*x*y = 0` (E1) / `d*x*y - c*y = 0` (E2) | registered as equations |
| 2 Fixed points | `Solve E1 and E2 for x and y` | **two fixed points**: $x=0,\ y=0$ and $x=c/d,\ y=a/b$ |
| 3 Partial derivatives (Jacobian entries) | `a*x - b*x*y` differentiated w.r.t. `x`, plus 3 more | $a-by,\ -bx,\ dy,\ dx-c$ |
| 4 Jacobian (evaluated at the coexistence equilibrium) | `[0, -b*c/d; a*d/b, 0]` | $\begin{pmatrix}0&-bc/d\\ ad/b&0\end{pmatrix}$ |
| 5 Eigenvalues | eigenvalues of the matrix's E-id | $\pm\sqrt{-ac}=\pm i\sqrt{ac}$ (neutral center) |

> **Matrix input notation:** a numeric matrix is entered as `[a, b; c, d]` (`;`
> separates rows, `,` separates columns). The Jacobian is hand-assembled from the
> partial derivatives evaluated at the coexistence equilibrium (a dedicated operation
> for this is an optional future extension).

## Observed behaviour / notes

- **SILENT-DIFF-VAR-ALIAS (found and root-caused, 2026-07-05).** A raw formula (not
  an E-id) as the target of an English differentiate request used to have no
  equivalent to the Japanese "formula + trailing instruction" auto-split — and worse,
  it was not just a routing failure but a genuine silent-corruption bug. In Japanese,
  `"a*x - b*x*y を x で偏微分して"` works as a single input because
  `_auto_register_inline_formula` (`ui/interpret_pipeline.py`) detects the
  ASCII-formula / Japanese-instruction script boundary (`_FORMULA_OP_RE` requires the
  instruction part to start with a non-ASCII character) and splits it into a math
  token (registered without going through the general symbol→E-id substitution pass)
  plus a separate differentiate instruction referencing the new E-id. That boundary
  heuristic cannot recognize an all-ASCII English sentence, so
  `"Take the partial derivative of a*x - b*x*y with respect to x"` used to fall
  through to the general pipeline, where the "known symbol → E-id" substitution pass
  (intended for cases like `H` → its defining equation) also fired on `x` and `y` —
  and since `Solve E1 and E2 for x and y` (the preceding step) had already registered
  `x=0`/`x=c/d` and `y=0`/`y=a/b` as aliases, the formula's own free variables were
  silently replaced by their solved VALUES before differentiation ever ran
  (`differentiate a*x-b*x*y with respect to x` returned `0` instead of `a-b*y`, with
  no error). Fixed with a dedicated English recognizer,
  `_DIFF_INLINE_FORMULA_EN_RE` in `ui/interpret_pipeline.py`, that splits on the
  explicit keyword boundary ("of ... with respect to") the same way the Japanese
  script-boundary split does, registering the formula first (bypassing the alias
  substitution) before differentiating the resulting E-id
  (`tests/test_silent_diff_var_alias.py`). The plain natural-language form below now
  works directly, matching the Japanese scenario's conciseness — no math-chip
  workaround needed.

**Verified:** 2026-07-05 (real Maxima backend, generator run, all 9 steps completed
end-to-end — matching the Japanese version's step count now that the workaround is
no longer needed — same results as the Japanese version: fixed points $(0,0)$ and
$(c/d,a/b)$, Jacobian eigenvalues $\pm\sqrt{-ac}$).

## generator commands (this `scenario` block is the single source used to generate the example)

```scenario domain=mathematical_biology
@note: Register the nullcline (the condition $\dot x=0$) of the Lotka-Volterra equation $\dot x=ax-bxy$ (the prey population's rate of change) as an equation.
a*x - b*x*y = 0
@note: Register the nullcline of the other equation $\dot y=dxy-cy$ (the predator population's rate of change).
d*x*y - c*y = 0
@note: Solve the two nullclines as a simultaneous system for $x,y$ to find the fixed points (where the populations do not change). A single call to Maxima's `solve([...],[x,y])` genuinely returns BOTH solution branches at once (confirmed directly against the raw Maxima output: `[[x=0,y=0],[x=c/d,y=a/b]]`) — this is not two separate answers stitched together by this project's own code; it is exactly how `solve` on a system of two low-degree polynomial equations behaves. This should give two fixed points: the extinction equilibrium $(0,0)$ and the coexistence equilibrium $(c/d,\,a/b)$.
Solve E1 and E2 for x and y
@note: To examine stability at the coexistence equilibrium, find the components of the Jacobian matrix one at a time. First, differentiate $\dot x$ with respect to $x$. This command's operand ($a x-bxy$) is typed as a raw formula, not an existing $E_n$ reference — the pipeline registers it as its own numbered equation first (visible below as the equation immediately preceding the derivative), THEN differentiates it, so this one command produces two numbered equations, not one. Re-differentiating the same raw formula again below (with respect to $y$) reuses that same registration instead of duplicating it.
Take the partial derivative of a*x - b*x*y with respect to x
@note: Differentiate $\dot x$ with respect to $y$.
Take the partial derivative of a*x - b*x*y with respect to y
@note: Differentiate $\dot y$ with respect to $x$. As with the first differentiation above, this command's raw-formula operand ($dxy-cy$) is registered as its own equation before being differentiated, so two numbered equations again appear from this one command.
Take the partial derivative of d*x*y - c*y with respect to x
@note: Differentiate $\dot y$ with respect to $y$.
Take the partial derivative of d*x*y - c*y with respect to y
@note: Using the four partial derivatives just found, evaluated at the coexistence equilibrium $(c/d,a/b)$, assemble the Jacobian matrix $J=\begin{pmatrix}0&-bc/d\\ad/b&0\end{pmatrix}$.
[0, -b*c/d; a*d/b, 0]
@note: Find the eigenvalues of this Jacobian matrix. If they are purely imaginary, the coexistence equilibrium is a neutral center (populations orbit around the fixed point).
Find the eigenvalues of E13
```
