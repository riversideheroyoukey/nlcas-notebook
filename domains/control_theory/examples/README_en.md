# control_theory — worked example (RLC transfer function)

This folder holds an **educational worked example** obtained by actually running the
control-theory scenario. It reproduces in a standard (**Maxima**) session with the
`control_theory` domain.

| Scenario | Files | Content |
|---|---|---|
| [scenario_rlc_transfer_function](../../../docs/scenarios/scenario_rlc_transfer_function_jp.md) | `scenario_rlc_transfer_function_jp.{json,tex}` | Laplace-transform the series-RLC equation of motion $L\ddot q+R\dot q+q/C=v$ to derive the transfer function $H(s)=\dfrac{1}{Ls^2+Rs+1/C}$. Time derivatives are formed with `total derivative` to get a genuine $d/dt$, and the transfer function is extracted under zero initial conditions (no leakage) per the definition |

For each: `.json` is the session after execution (loadable from `sessions/`), `.tex`
is the exported document (compiles with uplatex/dvipdfmx).

**Input tips:**
- declare $q(t)$ with `define q as a time-dependent function`, and form the genuine
  time derivatives $\dot q,\ddot q$ with `total derivative` (the `\ddot{q}` notation
  becomes an independent atom in analytical mechanics and is not a Laplace-ready
  derivative).
- register the ODE as a formula such as `L*E3 + R*E2 + E1/C = v` (to keep the LHS).
- `compute the transfer function q/v of E5` extracts the $Q,V$ coefficients from the
  s-domain expression to give $H(s)$ (the initial-condition terms drop out
  automatically; the general Laplace transform is unaffected).
