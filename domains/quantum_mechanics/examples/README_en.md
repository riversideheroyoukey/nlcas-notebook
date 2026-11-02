# quantum_mechanics — worked example (tight-binding band structure)

This folder holds an **educational worked example** obtained by actually running the
scenario [docs/scenarios/scenario_tight_binding_bands_jp.md](../../../docs/scenarios/scenario_tight_binding_bands_jp.md).

| File | Content |
|---|---|
| `scenario_tight_binding_bands_jp.json` | the session after execution (loadable from `sessions/`) |
| `scenario_tight_binding_bands_jp.tex` | the exported LaTeX document (compiles with uplatex/dvipdfmx) |

Rigorously follows the dispersion relation $\varepsilon(k)$ for chain / square /
triangular / honeycomb / kagome / Lieb / SSH / abstract crystal descriptor.

**About the domain:** the tight-binding dispersion is a **non-interacting, single-particle**
computation, evaluated in a standard (**Maxima**) session via the F-1 3c-fourier
machinery (no CAS). It is placed in `quantum_mechanics` (single-particle lattice),
whose Maxima backend is consistent — not in the julia_full domain `condensed_matter`,
which presupposes second quantization (many-body).

**Regenerate:** `PYTHONPATH=. .venv/bin/python scripts/generate_scenario_examples.py`
