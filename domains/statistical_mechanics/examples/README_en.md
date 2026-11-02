# statistical_mechanics — worked examples (statistical mechanics of quantum spin systems)

This folder holds **educational worked examples** of **quantum-spin-system statistical
mechanics** that can be evaluated by the spin algebra (no CAS), obtained by actually
running the scenarios. All reproduce in a standard (**Maxima**) session.

| Scenario | Files | Content |
|---|---|---|
| [scenario_spin_chain_spectrum](../../../docs/scenarios/scenario_spin_chain_spectrum_jp.md) | `scenario_spin_chain_spectrum_jp.{json,tex}` | **Dynamic spectrum** of a quantum spin chain: commutator → moments → continued fraction → $S(\omega)$ → high-temperature expansion (↔ triangular numbers) |
| [scenario_ising_partition_function](../../../docs/scenarios/scenario_ising_partition_function_jp.md) | `scenario_ising_partition_function_jp.{json,tex}` | **Partition function** of the 2D Ising model (small lattice, high-temperature expansion): cumulant expansion of the specific heat; odd order $\beta^3$ ↔ triangular numbers for square vs triangular |
| [scenario_lattice_figures](../../../docs/scenarios/scenario_lattice_figures_jp.md) | `scenario_lattice_figures_jp.{json,tex}` + `_lattice.svg` | **Lattice figure drawing** (honeycomb / square, periodic boundary) → auto-embedded into the TeX export. Figure generation bypasses the CAS (backend-independent) |

For each: `.json` is the session after execution (loadable from `sessions/`), `.tex`
is the exported document (compiles with uplatex/dvipdfmx).

**About the domain:** these are **quantum-spin-system statistical mechanics** via the
spin algebra (moments, continued fraction, high-temperature expansion, $S(\omega)$);
since they bypass the CAS they are evaluated in standard (Maxima). They are placed in
`statistical_mechanics` (Maxima-consistent backend), not in the julia_full domain
`spin_systems`, which presupposes second quantization (many-body, Julia).

**Regenerate:** `PYTHONPATH=. .venv/bin/python scripts/generate_scenario_examples.py`
