# electromagnetism — worked examples (Maxwell consistency)

This folder holds **educational worked examples** obtained by actually running the
electromagnetism scenarios. They reproduce in a standard (**Maxima**) session with the
`electromagnetism` domain.

| Scenario | Files | Content |
|---|---|---|
| [scenario_maxwell_consistency](../../../docs/scenarios/scenario_maxwell_consistency_jp.md) | `scenario_maxwell_consistency_jp.{json,tex}` | For the plane wave $\boldsymbol{E}=(0,\cos(kx-\omega t),0)$, $\boldsymbol{B}=(0,0,\cos(kx-\omega t)/c)$: confirm $\nabla\!\cdot\!E=\nabla\!\cdot\!B=0$, compute $\nabla\!\times\!E$/$\nabla\!\times\!B$, build $\partial\boldsymbol{B}/\partial t$ component-by-component and reassemble, then derive the dispersion relation $\omega=ck$ from the $z$-component of Faraday's law |
| [scenario_coordinate_jacobian](../../../docs/scenarios/scenario_coordinate_jacobian_jp.md) | `scenario_coordinate_jacobian_jp.{json,tex}` | **Change of variables in a definite integral.** A range-bearing definite integral is given as a **display-only, unevaluated object** (region predicate `x^2+y^2<=1` or LaTeX `\int\int`); a change of variables builds and displays a **new range-bearing integral** (integrand·\|J\| + new bounds), evaluated to a number. Polar over the unit disk $\to\pi/2$, spherical volume of the unit ball $\to 4\pi/3$, and a general Jacobian (user-supplied new bounds). Both the transform and the evaluation are **LLM-free** |

For each: `.json` is the session after execution (loadable from `sessions/`), `.tex`
is the exported document (compiles with uplatex/dvipdfmx).

**Input tips:**
- declare a vector field with `define E as a vector field` (the components
  `E_x,E_y,E_z` are registered abstractly); bind each component with an equation
  `E_y = …`.
- enter divergence/curl as **nabla notation as a formula** (`\nabla \cdot E`,
  `\nabla \times E`) and it is computed in place (`div E`, `curl E` also work).
- build $\partial\boldsymbol{B}/\partial t$ by **partial-differentiating each
  component** (`partially differentiate B_z by t` — $B_z(x,t)$ is multivariable, so
  the partial derivative is explicit), then reassemble into a new vector field `dBdt`.
- compare vector equations **component-wise** (register the $z$-component equation →
  `solve for w` → $\omega=ck$).
