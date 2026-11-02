# Practice-loop scenario S-c: the BCS gap equation (a scalar handoff, Julia → Maxima)

> Run 2026-08-03.  This continues [scenario_bcs_hamiltonian_en.md](scenario_bcs_hamiltonian_en.md)
> (the operator half).  Where that one stops at the Bogoliubov condition, this one goes on to the
> anomalous average, **crosses a session boundary**, and reaches the BCS closed form
> $\Delta=\omega_D/\sinh\frac{1}{N_0V}$ in Maxima.

**Backend:** Julia (QuantumAlgebra.jl) → **Maxima** (the handoff target) / **Domain:** condensed matter.

### Why two sessions

Operator algebra is QuantumAlgebra's job and $(k,i\omega_n)$-space scalar algebra is Maxima's, and
**the backend is fixed when the session is created** (Core Principle #6).  A calculation that has
finished with the operators therefore crosses a boundary.  The crossing is a **fork, not a switch**:
no live state is carried over — what travels is the researcher's own writing plus the declarations
needed to re-establish it, and the target session executes every step for real
([docs/34 §5](../developer_manuals/34_field_operators_pair_representation_and_scalar_handoff.md)).

### The physics

1. Diagonalise the mean-field BCS Hamiltonian by a Bogoliubov transformation (the operator half)
2. Obtain the anomalous average $\langle c_{-k\downarrow}c_{k\uparrow}\rangle = \dfrac{uv}{(u^2+v^2)^2}$
   — **operator algebra ends here**; everything after is c-numbers
3. Write the self-consistency condition $\Delta = V\sum_k \langle c_{-k\downarrow}c_{k\uparrow}\rangle$
   in the continuum limit as $1 = N_0V\displaystyle\int_0^{\omega_D}\frac{d\xi}{\sqrt{\xi^2+\Delta^2}}$
4. Do the integral, $\mathrm{asinh}\dfrac{\omega_D}{\Delta}$, and solve for $\Delta$:
   $\Delta = \dfrac{\omega_D}{\sinh\frac{1}{N_0V}}$

Steps 3–4 belong to Maxima, steps 1–2 to Julia.  **The handoff is that seam.**

---

### Steps

#### ST-1–7: the operator half

Run ST-1 through ST-7 of [scenario_bcs_hamiltonian_en.md](scenario_bcs_hamiltonian_en.md)
(the generator block below repeats every line, so this scenario runs on its own).
It ends with the diagonalised $H$ and the pair-creation coefficient
$-\dfrac{\Delta(u^2-v^2)+2uv\xi}{(u^2+v^2)^2}$.

---

#### ST-8: the anomalous average (the Cooper-pair expectation value)

| Item | Content |
|---|---|
| Input | `c(-k,d) c(k,u)` (register the pair product) |
| Input | `In E12, substitute E7 for c(k,u) and E8 for c(-k,d)` (move to the quasiparticle basis) |
| Input | `@op: vacuum_expectation {"expr": "E13"}` |
| Expected | $\langle c_{-k\downarrow}c_{k\uparrow}\rangle = \dfrac{uv}{(u^2+v^2)^2}$ |
| Check | the result is a **labelled equation**: the left side says what the value is OF, the right side is the c-number |
| Result | |

---

#### ST-9: hand it to a Maxima session (SCALAR-HANDOFF)

| Item | Content |
|---|---|
| Input | `hand E14 off to a Maxima session` |
| Expected | "carrying 5 declarations" + the filename + **"the values agree on both sides (checked at 3 points)"** |
| Check | only the **value** crosses.  The operator label has no meaning in a scalar CAS, so it is not registered; it survives in the target's provenance note as "value of …" |
| Check | the source session stays **open** (a file is written, not switched to), and a `scalar_handoff` step is recorded on its timeline |
| Result | |

**⚠ This step exposed two defects on its first run (see "What this run found").**  Before the fix
it was refused outright and S-c could not be completed.

---

#### ST-10: open the target session and declare what is left

| Item | Content |
|---|---|
| Action | open the created `H1_maxima.json` (the "Open the created session" button in the completion message, or Load) |
| Check | `Delta`, `u`, `v`, `xi`, `H_BCS` are already declared **with their signs** (carried from the Julia session) |
| Input | `Define V as a positive real` / `Define N_0 as a positive real` / `Define omega_D as a positive real` |
| Note | these three did not exist on the Julia side (coupling strength, density of states, Debye frequency).  **The sign is not optional** — without it Maxima stops mid-integral to ask, and the session hangs (SIGN-DIALOG-ASSUME) |
| Result | |

---

#### ST-11: the gap integral

| Item | Content |
|---|---|
| Input | `1/sqrt(xi^2 + Delta^2)` |
| Input | `Integrate E2 with respect to xi from 0 to omega_D` |
| Expected | $\displaystyle\int_0^{\omega_D}\frac{d\xi}{\sqrt{\xi^2+\Delta^2}} = \mathrm{asinh}\frac{\omega_D}{\Delta}$ |
| Result | |

---

#### ST-12: solve the gap equation

| Item | Content |
|---|---|
| Input | `1 = N_0 * V * (asinh(omega_D/Delta))` |
| Input | `Solve E4 for Delta` |
| Expected | $\Delta = \dfrac{\omega_D}{\sinh\frac{1}{N_0V}}$ (**the BCS closed form**) |
| Result | |

---

### Verification checklist

- [x] the anomalous average $uv/(u^2+v^2)^2$ comes out as a vacuum expectation value
- [x] a labelled result (`⟨cc⟩ = value`) crosses — the **value** is checked, the operator label is dropped and kept as provenance
- [x] five declarations are replayed in the target **with their signs**
- [x] the crossing check answers **agreed (3 points)** — comparing the source's own reading of the exported LaTeX, not the opaque `_nb_E16` handle
- [x] the target session does the definite integral without a sign question and returns `asinh(omega_D/Delta)`
- [x] solving the gap equation reaches the BCS closed form
- [x] the source session stays open and keeps its `scalar_handoff` step

---

## What this run found

**The first run of this scenario exposed two defects.  The first one stopped S-c completely.**

| # | Symptom | Cause | Fix |
|---|---|---|---|
| 1 | `hand E14 off to a Maxima session` was **refused**: "E14 still contains an operator (c)".  S-c could not complete | a vacuum-expectation result is the labelled equation `c(-k,d) c(k,u) = uv/(u²+v²)²`, whose left side says what the value is of.  The operator check read the whole line and found `c` — but **an expectation value is a c-number by definition**, so the mechanism turned away the very expression it was built to carry | check the **value** (the right side) instead (`split_export_form`).  Not a loosening: an operator in the value is still refused.  The operator label is not registered; the target's note records it as "value of …" |
| 2 | the crossing check reported "⚠ the expression does not match the source (`_nb_E16` vs `u, v`)" — a **false alarm** | the source side of the comparison was `source_canonical`, which for a Julia result is the opaque handle `_nb_E16`.  It **could not be checked**, and that was reported as a mismatch | compare against the source's own reading of the exported LaTeX, produced by the source backend's handler (`source_value_canonical`).  Two independent readings of the same LaTeX is exactly the round trip this check exists to test.  With nothing but a handle, say `not_checkable` — never "differed" |

**"Cannot check" and "does not match" are different things.**  The design doc (docs/34 §5.3) had
already ruled that an unverifiable crossing must not be filed as agreement; **the converse turned out
to matter just as much** — telling a researcher a correct result is wrong is worse than saying nothing.

**The manual measurement missed both.**  The 2026-07-31 run (docs/34 §6) was transcribed by hand, so
it went through neither the operator check nor the value check.  **Mechanised work has to be run
through the mechanism before it counts as verified.**

**Verified:** 2026-08-03 (real Julia + Maxima, ST-1 through ST-12).

---

## Generator commands (this `scenario` block is the single source)

> **Note:** the generator produces one session per scenario, so this block runs the **Julia side
> (ST-1–9)**.  The handoff writes the target session as a separate file, so ST-10–12 continue by
> opening that file (as the tables above describe).

```scenario domain=condensed_matter
@note: Prepare the operators and parameters for the BCS Hamiltonian. $c$ is the electron annihilation operator with momentum $k$ and spin $\sigma$ ($u$ = up, $d$ = down); $g$ is the Bogoliubov quasiparticle. $u, v$ are the transformation coefficients, $\xi$ the kinetic energy, $\Delta$ the superconducting gap.
c as the annihilation operator for electron
g as a fermion operator
Define u, v as scalars
Define xi, Delta as scalars
@note: Register the mean-field BCS Hamiltonian.
H_BCS = xi * (c'(k,u) c(k,u) + c'(-k,d) c(-k,d)) + Delta * (c'(k,u) c'(-k,d) + c(-k,d) c(k,u))
@note: The equation of motion shows how c(k,u) is coupled to H_BCS.
Heisenberg equation of E1 for c(k,u)
@note: Define the Bogoliubov transformation as all four Nambu-spinor components together.
g(k,u) = u * c(k,u) - v * c'(-k,d)
g(-k,d) = u * c(-k,d) + v * c'(k,u)
g'(k,u) = u * c'(k,u) - v * c(-k,d)
g'(-k,d) = u * c'(-k,d) + v * c(k,u)
@note: Solve the four for the electron operators (the inverse transform).
Invert E3, E4, E5, E6 for c(k,u), c(-k,d), c'(k,u), c'(-k,d)
@note: Substitute the inverse transform into the Hamiltonian. Choosing u, v so the pair-creation coefficient vanishes is the Bogoliubov condition.
In E1, substitute E7 for c(k,u) and E8 for c(-k,d)
@note: Now head for the gap equation. To get the Cooper-pair anomalous average, first register the pair product.
c(-k,d) c(k,u)
@note: Move it to the quasiparticle basis.
In E12, substitute E7 for c(k,u) and E8 for c(-k,d)
@note: Take the expectation value in the quasiparticle vacuum (the BCS ground state). Operator algebra ends here; everything after this is c-numbers.
@op: vacuum_expectation {"expr": "E13"}
@note: The operators are done, so the remaining scalar algebra belongs to Maxima. Cross the session boundary — a fork, not a switch: this session stays as it is and the target becomes a new file. What travels is the value on the right plus the declarations (signs included) needed to re-establish it. The operator label on the left has no meaning in a scalar CAS, so what the value is OF is recorded in the target's note.
hand E14 off to a Maxima session
```
