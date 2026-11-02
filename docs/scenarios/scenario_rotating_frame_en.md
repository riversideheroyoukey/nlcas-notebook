# Practice-loop sample scenario: rotating-frame effective Hamiltonian (S-m)

A detailed write-up of practice-loop scenario-menu item **S-m (magnetic resonance)**.
The standard NMR/ESR **rotating-frame transformation** is carried out entirely as a
**sequence of general spin operations**: register a custom spin Hamiltonian, add the
frame term, take the magnitude of the effective field, and obtain the precession
equation of motion from a commutator. The spin algebra bypasses the CAS (a
self-contained Python implementation).

> **Design principle (important):** a scenario exists to *surface latent bugs*, so it
> must complete on **general, reusable operations** (parse a spin expression, **add
> expressions**, **norm of the coefficient vector**, general `commutator`) — not on a
> feature that bakes in this scenario's answer. "Moving to the rotating frame" is just
> **adding** the frame term $+\omega S^z$, so there is no dedicated transform op — the
> addition is written explicitly. The magnitude of the effective field is read directly
> from the components (coefficients) of $H = -\mathbf B\cdot\mathbf S$ (no field vector
> is materialised). The detuning $\Omega=\omega_0-\omega$ is a display shorthand; this
> scenario keeps the raw $(\omega-\omega_0)$ form the addition returns (no substitution
> to $\Omega$).

**Backend:** standard (Maxima) / **Domain:** magnetic resonance (magnetic_resonance).

## Physical background

Moving to a frame rotating at $\omega$ ($U=e^{-i\omega t S^z}$) adds a frame term
$+\omega S^z$ to the Zeeman part, so the lab Hamiltonian

$$H_{\mathrm{lab}} = -\omega_0 S^z - \omega_1 S^x$$

becomes, with the detuning $\Omega=\omega_0-\omega$,

$$H_{\mathrm{rot}} = -\Omega S^z - \omega_1 S^x .$$

The magnitude of the **effective field** the spin feels is
$\omega_{\mathrm{eff}}=\sqrt{\Omega^2+\omega_1^2}$, and the precession in the rotating
frame is given by the equation of motion $[H_{\mathrm{rot}},S^z]=i\omega_1 S^y$ (at
resonance $\Omega=0$ this is a Rabi precession about $\omega_1$).

## Steps and expected results

The derivation runs as **four general operations** (each its own input = its own YOU
card); each command returns one expression.

| # | Input (natural language) | Operation | Expected result |
|---|---|---|---|
| 1 | `H_{lab} = -\omega_0 S^z_1 - \omega_1 S^x_1` | register spin H (parse) | $H_{\mathrm{lab}} = -\omega_0 S^z_1 - \omega_1 S^x_1$ (E1) |
| 2 | `H_{rot} = E1 + \omega S^z_1` | **expression addition** (add the frame term) | $H_{\mathrm{rot}} = (\omega-\omega_0) S^z_1 - \omega_1 S^x_1$ (E2) |
| 3 | `Find the magnitude of the effective field of E2` | **norm of the coefficient vector** | $\omega_{\mathrm{eff}} = \sqrt{(\omega-\omega_0)^2+\omega_1^2}$ (E3) |
| 4 | `Find the commutator of E2 and S^z_1` | general `commutator` | $[H_{\mathrm{rot}}, S^z_1] = i\omega_1 S^y_1$ (E4) |

Register the custom spin Hamiltonian (an H-1 `LHS = RHS` whose RHS parses as a spin
expression) as E1, **add the frame term $+\omega S^z$ to E1** to get E2 (the honest
operation for "ride the rotating frame"). Since E2 is the component form of
$-\mathbf B\cdot\mathbf S$, read the **norm directly from its coefficients** for E3, and
take the general `commutator` for the precession EOM E4. **Each is a general operation
reusable on any spin expression** — not a per-scenario baked-in answer.

## Notes / observed behaviour

- **The spin algebra bypasses the CAS** (`notebook/runtime/spin_algebra.py`). A
  Hamiltonian is kept as structured data (a term list), so E2 can be lexically parsed
  (`parse_spin_expression`) and its coefficients (the field components) read directly.
  The commutator, the norm and the addition are all general operations on the term list
  (same family as S-a / S-p).
- Each command registers one expression (via `register_dispersion`; a derived quantity,
  so no LHS alias).
- To use the detuning $\Omega=\omega_0-\omega$, insert one general substitution
  $(\omega-\omega_0)\to\Omega$ (this scenario shows the raw form the addition returns).
  At resonance $\omega=\omega_0$, $H_{\mathrm{rot}}=-\omega_1 S^x$ — a Rabi precession
  about $\omega_1$.
- The Japanese variant runs through the SAME pipeline (`E2 の有効磁場の大きさを求めて`,
  etc.; the equation `H_{rot} = E1 + \omega S^z_1` is shared).
  → [scenario_rotating_frame_jp.md](scenario_rotating_frame_jp.md)

**Worked example:** [`domains/magnetic_resonance/examples/`](../../domains/magnetic_resonance/examples/)
(`scenario_rotating_frame_{jp,en}.{json,tex}`).

## generator コマンド（この `scenario` ブロックが生成の単一ソース）

```scenario domain=magnetic_resonance
@note: Register the lab-frame spin Hamiltonian $H_{\rm lab}=-\omega_0S^z_1-\omega_1S^x_1$ ($\omega_0$: the Zeeman frequency, $\omega_1$: the strength of the rotating field).
H_{lab} = -\omega_0 S^z_1 - \omega_1 S^x_1
@note: To move into a frame rotating at angular frequency $\omega$, add the frame term $+\omega S^z$ to the lab-frame Hamiltonian, giving the rotating-frame Hamiltonian $H_{\rm rot}$.
H_{rot} = E1 + \omega S^z_1
@note: From the coefficients (field components) of $H_{\rm rot}=-\mathbf B\cdot\mathbf S$, directly find the magnitude of the effective field felt by the spin, $\omega_{\rm eff}=\sqrt{(\omega-\omega_0)^2+\omega_1^2}$.
Find the magnitude of the effective field of E2
@note: Compute the commutator of the rotating-frame Hamiltonian $H_{\rm rot}$ and $S^z$ to obtain the equation of motion for precession in the rotating frame.
Find the commutator of E2 and S^z_1
```
