# electrical_engineering — worked example (AC impedance and resonance)

This folder holds an **educational worked example** obtained by actually running the
electrical-engineering scenario. It reproduces in a standard (**Maxima**) session with
the `electrical_engineering` domain.

| Scenario | Files | Content |
|---|---|---|
| [scenario_ac_impedance](../../../docs/scenarios/scenario_ac_impedance_jp.md) | `scenario_ac_impedance_jp.{json,tex}` | Separate the real part $R$ and imaginary part $\omega L-\frac{1}{\omega C}$ of the series-RLC complex impedance $Z=R+j\omega L+\frac{1}{j\omega C}$, then solve $\operatorname{Im}Z=0$ to obtain the resonance $\omega_0=\frac{1}{\sqrt{LC}}$ |

For each: `.json` is the session after execution (loadable from `sessions/`), `.tex`
is the exported document (compiles with uplatex/dvipdfmx).

In the `electrical_engineering` domain **`j` is the imaginary unit** (C-4; `%i` in
Maxima). `take the real part` / `take the imaginary part` act on the **RHS (value)** of
the equation `Z = …` (giving `Re Z = R`, `Im Z = ωL−1/(ωC)`), and solving the
reactance `for w` yields the resonant angular frequency (`solve_equation` treats a bare
expression as `=0`).

**Regenerate:** `PYTHONPATH=. .venv/bin/python scripts/generate_scenario_examples.py`
