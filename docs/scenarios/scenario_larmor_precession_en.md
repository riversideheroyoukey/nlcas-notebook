# Practice-loop sample scenario: Larmor precession of a spin

> (2026-07-02: an English translation of the Japanese scenario, verified against the
> real Julia backend through the same rule-based, no-LLM generator pipeline. See
> [scenario_larmor_precession_jp.md](scenario_larmor_precession_jp.md) for the Japanese
> version. No new i18n gaps were found here — all the English NL patterns this
> scenario needs (spin operator declaration, `commutator`, `heisenberg_equation`) were
> already fixed while building the earlier translations.)

**Physical theme:** the equation of motion of a spin-1/2 in a magnetic field (the basis
of NMR / ESR)
**Backend:** Julia (second-quantization session)
**Verification goal:** confirm the basic Julia/QA workflow (operator declaration →
commutator → equation of motion) works smoothly.

---

## Physical background

The Zeeman Hamiltonian of a spin $S$ in a magnetic field $B$:

$$H = -\gamma B \, S_z$$

The Heisenberg equation of motion gives the precession equations:

$$\frac{dS_x}{dt} = 2\gamma B \, S_y, \quad \frac{dS_y}{dt} = -2\gamma B \, S_x, \quad \frac{dS_z}{dt} = 0$$

(QuantumAlgebra uses the Pauli-matrix convention $[S_x, S_y] = 2i S_z$, hence the
factor of 2.)

Finally, $[H, S_z] = 0$ confirms $S_z$ is a conserved quantity (step 9).

---

## Session setup

1. Open **"New session"** in the browser.
2. Title: `Larmor precession`.
3. Domain: select `quantum mechanics` (or `condensed matter physics`)
   → confirm the **Julia backend** is automatically selected.
4. Click "Start".

---

## Input steps

> **Notation:** operator site indices are written uniformly as `Sz_1` (underscore
> suffix form). `Sz(1)` also works, but `Sz_1` is recommended for ease of typing.

---

### Step 1: declare the spin operator

```
S as the spin operator
```

**Expected output:** `Sx`, `Sy`, `Sz` (and `S`) are registered as operators. No
expression number is assigned.

---

### Step 2: declare the scalar parameters

```
Define gamma, B as real scalars
```

**Expected output:** `gamma` and `B` are registered as declared scalars.
`@variables gamma::Real` and `@variables B::Real` are sent to Julia.

---

### Step 3: register the Hamiltonian

```
H = -\gamma * B * Sz_1
```

**Expected output:** registered as a new expression (an E-id). Displayed roughly as
$H = -\gamma B S_{z_1}$.

---

### Step 4: commutator [H, Sx_1]

```
Compute the commutator of H and Sx_1
```

**Expected output:** $-2i\gamma B \cdot S_{y_1}$

Physically: $[H, S_x] = -\gamma B [S_z, S_x] = -\gamma B (2i S_y) = -2i\gamma B S_y$

---

### Step 5: commutator [H, Sy_1]

```
Compute the commutator of H and Sy_1
```

**Expected output:** $+2i\gamma B \cdot S_{x_1}$

Physically: $[H, S_y] = -\gamma B [S_z, S_y] = -\gamma B (-2i S_x) = 2i\gamma B S_x$

---

### Step 6: Heisenberg equation (Sx)

```
Heisenberg equation of H for Sx_1
```

**Expected output:** $\displaystyle\frac{dS_x}{dt} = 2\gamma B \cdot S_{y_1}$

---

### Step 7: Heisenberg equation (Sy)

```
Heisenberg equation of H for Sy_1
```

**Expected output:** $\displaystyle\frac{dS_y}{dt} = -2\gamma B \cdot S_{x_1}$

---

### Step 8: Heisenberg equation (Sz)

```
Heisenberg equation of H for Sz_1
```

**Expected output:** $\displaystyle\frac{dS_z}{dt} = 0$

---

### Step 9: confirm the conserved quantity (via the commutator)

```
Compute the commutator of H and Sz_1
```

**Expected output:** $0$

$[H, S_z] = -\gamma B [S_z, S_z] = 0$ — $S_z$ is indeed conserved. Confirms step 8.

---

## Known caveats

| Caveat | Detail |
|---|---|
| **Pauli convention** | QuantumAlgebra uses $[S_x, S_y] = 2i S_z$ (a factor of 2 compared to the textbook spin-1/2 convention $i S_z$) |
| **The first input fixes the notation style** | Whether you first type `Sz_1` (suffix) or `S^z_1` (superscript) fixes the display style for the whole session. Mixing is not recommended |
| **Site specification** | Either `Sz_1` (suffix) or `Sz(1)` (parenthesized) works, but keep the notation consistent |

**Verified:** 2026-07-02 (real Julia backend, generator run, all 9 steps completed
end-to-end, same results as the Japanese version).

---

## Workflow replay (S-z, C-Ext-5 `heisenberg_components`, verified 2026-07-17)

Verified whether steps 4-9 (two commutators + three Heisenberg equations + one
confirmation) can be replayed with a **single instruction** via the fixed C-Ext-5
workflow `heisenberg_components` (NL trigger: "heisenberg equations for all
components of S"). Steps 1-3 (declare S, declare gamma/B, register H) stay manual.

**Happy path:** after steps 1-3, entering "heisenberg equations for all components
of S" runs plan P1 (three steps, one per `Sx_1`/`Sy_1`/`Sz_1`) and the results
**exactly match** steps 6-8 of the manual scenario:

| Component | Workflow-replay result | Manual scenario (steps 6-8) |
|---|---|---|
| $dS_x/dt$ | `2B*gamma * Sy(1)` | $2\gamma B \cdot S_{y_1}$ ✓ |
| $dS_y/dt$ | `-2B*gamma * Sx(1)` | $-2\gamma B \cdot S_{x_1}$ ✓ |
| $dS_z/dt$ | `0` | $0$ ✓ |

**Stop-safe + retry_plan verification:** deliberately deferred step 1 (declaring S
as the spin operator) and instead declared gamma/B, registered H (with `Sz_1`
undeclared — registers with an undefined-symbols warning), then fired the workflow
trigger. As expected:

- Plan P1 failed at step s1 (`Sx_1`'s Heisenberg equation) with
  `Undefined symbols: Sz. Please define them before computing.`; s2/s3 were marked
  `skipped`; `plan.status = "stopped"` (0/3 succeeded, partial results kept). The
  timeline UI correctly rendered the ✗/⊘ breakdown.
- Declared "S as the spin operator" to fix the root cause, then called
  `engine.retry_plan("P1")` (no UI button exists for this — see the gap noted
  below — verified via a direct engine call against the saved session). `retry_plan`
  created a **new, separate plan P2** (`retried_from="P1"`, per the B-3 design of
  immutable plan history), and all three steps succeeded with values matching the
  table above. P1 itself remained `stopped` and unchanged in storage.

**Gap discovered (new, logged to open_items.md):** `ui/timeline_serializer.py` never
handles `StepKind.plan_execution` (nor `branch_fork`/`ambiguity_resolution`/
`assumption_remove`), so a fixed-workflow plan-execution summary — success or
failure alike — **disappears from the timeline after a page reload**. The live
render right after the confirm click works correctly (verified visually in this
session), but `GET /timeline` (used on reload) never includes an entry for that
step, even though the data survives intact in `engine.plans`/`engine.steps`.
Separately, `retry_plan` itself has no HTTP route or UI button — it is only
reachable as an engine-level method today.

**Fix confirmed (D-5, 2026-07-17, same day):** both gaps were fixed by adding a
`plan_execution` branch to `ui/timeline_serializer.py`, a new
`POST /plan/retry/{plan_id}` route, and a 🔁 Retry button in the UI. Verified live
in the browser against this exact scenario: the stopped P1 plan card survives a
page reload instead of disappearing, and clicking the new 🔁 Retry button (after
declaring `S`) produces the same result as calling `engine.retry_plan()` directly
(new plan P2, all three steps succeeded, same values as the table above) — this
time reachable entirely through the UI. See the open_items.md archive (D-5) for
details.

## generator commands (this `scenario` block is the single source used to generate the example)

```scenario domain=spin_systems
@note: Consider the Larmor precession of a spin-1/2 in a magnetic field. Prepare the spin operator $S$ (components $S_x,S_y,S_z$).
S as the spin operator
@note: Prepare the gyromagnetic ratio $\gamma$ and the magnetic field strength $B$ as real scalar parameters.
Define gamma, B as real scalars
@note: Register the Zeeman Hamiltonian $H=-\gamma BS_z$ (the energy of a spin in a magnetic field).
H = -\gamma * B * Sz_1
@note: Compute the commutator of the Hamiltonian $H$ with the $x$-component of spin $S_x$. This commutator determines part of the time evolution of $S_x$ (QuantumAlgebra uses the Pauli-matrix convention $[S_x,S_y]=2iS_z$, so the coefficient is twice the textbook $iS_z$).
Compute the commutator of H and Sx_1
@note: Likewise, compute the commutator of the Hamiltonian $H$ with the $y$-component of spin $S_y$.
Compute the commutator of H and Sy_1
@note: Use the Heisenberg equation $dS_x/dt=i[H,S_x]$ to find the equation of motion for $S_x$. Precession should produce a term proportional to $S_y$.
Heisenberg equation of H for Sx_1
@note: Likewise, find the equation of motion for $S_y$. A term proportional to $S_x$, with the opposite sign, should appear.
Heisenberg equation of H for Sy_1
@note: Find the equation of motion for $S_z$. Since $H$ is written purely in terms of $S_z$, $S_z$ should be conserved and this should vanish.
Heisenberg equation of H for Sz_1
@note: Directly compute the commutator of $H$ and $S_z$ to confirm, by another route, the conservation result from the previous step.
Compute the commutator of H and Sz_1
```
