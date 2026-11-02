# analytical_mechanics — worked examples (Lagrangian → Euler-Lagrange)

This folder holds **educational worked examples** obtained by actually running the
analytical-mechanics scenarios. They reproduce in a standard (**Maxima**) session.

| Scenario | Files | Content |
|---|---|---|
| [scenario_double_pendulum_lagrangian](../../../docs/scenarios/scenario_double_pendulum_lagrangian_jp.md) | `scenario_double_pendulum_lagrangian_jp.{json,tex}` | Double pendulum (with velocity coupling): Lagrangian → Euler-Lagrange equations |
| [scenario_coupled_oscillations](../../../docs/scenarios/scenario_coupled_oscillations_jp.md) | `scenario_coupled_oscillations_jp.{json,tex}` | Coupled oscillations: equations of motion → normal modes (sum/difference) → inverse of the normal-coordinate transform (L-1) |

For each: `.json` is the session after execution (drop it into `sessions/` to load),
`.tex` is the exported document (compiles with uplatex/dvipdfmx).

Follows the Euler-Lagrange workflow $\partial L/\partial\dot q$ → $\frac{d}{dt}$ →
$\partial L/\partial q$ → equation of motion. Velocities are entered as `\dot{q}`
(math notation). In the analytical-mechanics domain the coordinate $q$ and velocity
$\dot q$ are treated as independent atoms (independent-atom model =
[doc 19](../../../docs/developer_manuals/19_analytical_mechanics_independent_variables.md)).

**Regenerate:** `PYTHONPATH=. .venv/bin/python scripts/generate_scenario_examples.py`
