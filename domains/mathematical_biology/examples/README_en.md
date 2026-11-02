# mathematical_biology — worked example (Lotka–Volterra stability analysis)

This folder holds an **educational worked example** obtained by actually running the
mathematical-biology scenario. It reproduces in a standard (**Maxima**) session with the
`mathematical_biology` domain.

| Scenario | Files | Content |
|---|---|---|
| [scenario_lotka_volterra_stability](../../../docs/scenarios/scenario_lotka_volterra_stability_jp.md) | `scenario_lotka_volterra_stability_jp.{json,tex}` | Prey–predator model $\dot x=ax-bxy,\ \dot y=dxy-cy$: fixed points ($(0,0)$ and the coexistence equilibrium $(c/d,a/b)$) → Jacobian → eigenvalues $\pm i\sqrt{ac}$ (a neutral center). Uses only existing operations: `solve_system`, partial differentiation, matrix eigenvalues |

For each: `.json` is the session after execution (loadable from `sessions/`), `.tex`
is the exported document (compiles with uplatex/dvipdfmx).

**Input tips:**
- register the nullclines as equations `a*x - b*x*y = 0`, and find the fixed points with
  `solve the simultaneous equations E1 and E2 for x and y` (the **multiple fixed
  points** are correctly registered, one per solution set).
- obtain the Jacobian entries by partial differentiation, e.g. `partially differentiate
  a*x - b*x*y by x`.
- enter the matrix as `[0, -b*c/d; a*d/b, 0]` (`;` row separator, `,` column separator)
  and get the eigenvalues with `find the eigenvalues`.
