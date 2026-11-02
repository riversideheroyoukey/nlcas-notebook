# operations_research — worked example (M/M/1 queue)

This folder holds an **educational worked example** obtained by actually running the
operations-research (OR) scenario. It reproduces in a standard (**Maxima**) session with
the `operations_research` domain.

| Scenario | Files | Content |
|---|---|---|
| [scenario_mm1_queue](../../../docs/scenarios/scenario_mm1_queue_jp.md) | `scenario_mm1_queue_{jp,en}.{json,tex}` | M/M/1 queue: substitute $\rho$ into the utilization $\rho=\lambda/\mu$ and mean number $L=\rho/(1-\rho)$ to get $L=\lambda/(\mu-\lambda)$, then **divide both sides by $\lambda$** to derive Little's law $W=L/\lambda=1/(\mu-\lambda)$ |

For each: `.json` is the session after execution (loadable from `sessions/`), `.tex`
is the exported document (compiles with uplatex/dvipdfmx). Both the **Japanese (`_jp`)
and English (`_en`)** variants are included (thanks to input-NL i18n, the English
variant is produced by running the same pipeline).

**Input tips:**
- register the utilization and mean number as equations `\rho = \lambda/\mu` and
  `L = \rho/(1-\rho)` (H-1).
- `In E2, substitute \lambda/\mu for \rho` (symbol→value substitution) → $L=\lambda/(\mu-\lambda)$.
- `Divide both sides of E3 by \lambda` (H-8) → Little's law $W=1/(\mu-\lambda)$. It closes
  under **substitution and simplification alone**.

**Regenerate:** `PYTHONPATH=. .venv/bin/python scripts/generate_scenario_examples.py`
