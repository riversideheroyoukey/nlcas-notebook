# magnetic_resonance — worked example (rotating-frame effective Hamiltonian)

This folder holds an **educational worked example** obtained by actually running the
magnetic-resonance scenario. It reproduces in a standard (**Maxima**) session with the
`magnetic_resonance` domain (the spin algebra bypasses the CAS).

| Scenario | Files | Content |
|---|---|---|
| [scenario_rotating_frame](../../../docs/scenarios/scenario_rotating_frame_jp.md) | `scenario_rotating_frame_{jp,en}.{json,tex}` | The NMR/ESR rotating-frame transformation as a sequence of **general spin operations**: register the custom spin H $H_{\mathrm{lab}}=-\omega_0 S^z-\omega_1 S^x$ → `rotating_frame` transform → $H_{\mathrm{rot}}=(\omega-\omega_0) S^z-\omega_1 S^x$ → `effective_field` $\omega_{\mathrm{eff}}=\sqrt{(\omega-\omega_0)^2+\omega_1^2}$ → general `commutator` for the precession EOM $[H_{\mathrm{rot}},S^z]=i\omega_1 S^y$ |

For each: `.json` is the session after execution (loadable from `sessions/`), `.tex`
is the exported document (compiles with uplatex/dvipdfmx); both `_jp` and `_en`.

**Input tips:**
- solve the derivation in **four general operations**: `H_{lab} = -\omega_0 S^z_1 -
  \omega_1 S^x_1` (register a spin Hamiltonian) → `Move E1 to the rotating frame at
  \omega` (`rotating_frame` transform) → `Find the effective field of E2`
  (`effective_field`) → `Find the commutator of E2 and S^z_1` (general `commutator` =
  the precession EOM).
- **No per-scenario baked-in answer**: each is a general op reusable on any spin
  expression, and spin expressions chain through `parse_spin_expression`. The detuning
  $\Omega=\omega_0-\omega$ is a display shorthand; insert one general substitution
  $(\omega-\omega_0)\to\Omega$ if you want it.
