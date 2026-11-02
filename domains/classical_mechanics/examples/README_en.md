# classical_mechanics — worked examples (central-force invariants, Poisson brackets)

This folder holds **educational worked examples** obtained by actually running the
classical (Newtonian) mechanics scenarios. They reproduce in a standard (**Maxima**)
session.

| Scenario | Files | Content |
|---|---|---|
| [scenario_central_force_conservation](../../../docs/scenarios/scenario_central_force_conservation_jp.md) | `scenario_central_force_conservation_jp.{json,tex}` | Invariants of motion in a central potential: angular momentum $L_z=mr^2\dot\theta$ and energy $E=\tfrac12 m(\dot r^2+r^2\dot\theta^2)-k/r$, confirmed via $d/dt$ → substitute the equations of motion → 0 |
| [scenario_poisson_bracket](../../../docs/scenarios/scenario_poisson_bracket_jp.md) | `scenario_poisson_bracket_jp.{json,tex}` | Poisson brackets: declare the canonical pairs $(r,p_r),(\theta,p_\theta)$ and confirm $\{r,p_r\}=1$, $\{H,L_z\}=0$ (angular-momentum conservation) for the central-force Hamiltonian, and $\{H,r\}=-p_r/m$ (Hamilton's equation) |

For each: `.json` is the session after execution (drop it into `sessions/` to load),
`.tex` is the exported document (compiles with uplatex/dvipdfmx).

Follows the workflow invariant → $\frac{d}{dt}$ → substitute the equations of motion
($\ddot\theta,\ddot r$) → $0$. Velocities are entered as `\dot{q}` (math notation).
In the classical-mechanics domain velocities are dependent quantities $\dot q=dq/dt$,
represented by Maxima's depends form `'diff(q,t,N)` — in contrast to analytical
mechanics' independent-atom model
([doc 19](../../../docs/developer_manuals/19_analytical_mechanics_independent_variables.md)).

**Regenerate:** `PATH="$HOME/.juliaup/bin:$PATH" .venv/bin/python scripts/generate_scenario_examples.py`
