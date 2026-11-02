# Practice-loop sample scenario: dynamical spectrum of a quantum spin chain (recursion method)

> (2026-07-02: an English translation of the Japanese scenario, verified against the
> real Maxima backend through the same rule-based, no-LLM generator pipeline. See
> [scenario_spin_chain_spectrum_jp.md](scenario_spin_chain_spectrum_jp.md) for the
> Japanese version.)

**Physical theme:** the spin model's equation of motion → moments → continued fraction
→ spectral function, plus the relationship between the high-temperature expansion and
lattice loops (triangles)
**Backend:** standard (Maxima) suffices. Spin algebra is computed exactly in Python
without going through the CAS, so Julia is not needed.
**Verification goal:** confirm that the whole spin-algebra pipeline (commutator,
moments, continued fraction, spectral function, high-temperature expansion) returns
**exact closed forms**.

---

## Physical background

The Laplace image (Green's function) of the dynamical correlation function
$\langle S^\alpha(t) S^\alpha\rangle$ of a spin operator $S^\alpha$ can be expanded, via
the Mori–Zwanzig recursion (Lanczos) method, into a continued fraction
$G(z)=1/(z-b_1^2/(z-b_2^2/\cdots))$. On a finite cluster, the continued fraction
**terminates at finite depth**, so $G(z)$ is a rational function — a discrete spectrum.
The coefficients $b_n^2$ are determined by the infinite-temperature moments
$M_{2k}=(L^kO,L^kO)$ ($L=[H,\cdot]$). Furthermore, the high-temperature expansion of
the energy $\langle H^n\rangle$ can be written in terms of the lattice's loop counts —
**edge count $|E|$, triangle count $N_\triangle$** (triangles show up in
$\langle H^3\rangle$). This scenario traces this whole pipeline with exact values.

> Everywhere assumes spin-1/2, infinite temperature (inner products / traces reduce to
> $(S^a)^2=1/4$ etc. on a single site).

---

## Session setup

- Mode: **standard (Maxima)** (the spin-algebra commands are backend-independent).
- Domain: any **Maxima-family domain** (the worked example is saved under
  `statistical_mechanics`). `spin_systems` is julia_full (second quantization), so it
  is not used as the save location for this Maxima scenario.
- Verification harness: `./scripts/debug_repl.sh --no-llm --lang en` (reproducible via
  rule-based dispatch).

---

## Input steps

### Step 1: equation of motion (1st-order commutator)

```
Find the commutator of S^x_1 for the Ising model with 2 sites
```

**Expected output:** `[H, S^{x}_{1}] = i J S^{y}_{1} S^{z}_{2}`

**Check points:**
- The coefficient `i J` and the operator product `S^y_1 S^z_2` come out exactly (Ising-z's
  $S^x$ couples to the neighboring $S^z$ and precesses).

### Step 2: acceleration (2nd-order commutator = composition)

```
Find the double commutator of S^x_1 for the Ising model with 2 sites
```

**Expected output:** `[H, [H, S^{x}_{1}]] = J^{2} S^{x}_{1} \left(S^{z}_{2}\right)^{2}`

**Check points:**
- The 1st-order result is **composed directly into the 2nd** (resolving the old
  "problem B" — impossible with opaque canonicals).
- $(S^z_2)^2$ stays as an operator (not numerically reduced by the spin magnitude).

### Step 3: infinite-temperature moments

```
Find the moments up to 2nd order of S^x_1 for the Ising model with 2 sites
```

**Expected output:** `M_{0} = \frac{1}{4},\ M_{2} = \frac{1}{16} J^{2}`

**Check points:**
- $M_0=\langle (S^x)^2\rangle=1/4$ (the spin-1/2 normalization), $M_2=J^2/16$.
- **"up to 2nd order" stops at $M_2$** (before 2026-09-10 it returned $M_4$ too —
  the written number was used as the number of recursion STEPS; ticket
  MOMENT-ORDER-MEANING). To ask in steps, write "up to depth 2", which gives $M_4$.

### Step 4: recursion coefficients (continued fraction)

```
Find the continued fraction of S^x_1 for the Ising model with 2 sites
```

**Expected output:** `b_{1}^2 = \frac{1}{4} J^{2}` (the Krylov space is then exhausted,
terminating early)

**Check points:**
- $b_1^2=M_2/M_0=(J^2/16)/(1/4)=J^2/4$. With 2 sites, it terminates after one
  coefficient.

### Step 5: spectral function (single mode = exact delta functions)

```
Find the spectral function of S^x_1 for the Ising model with 2 sites
```

**Expected output:**
`G(z) = \cfrac{1}{z - \cfrac{\frac{1}{4} J^{2}}{z}},\quad S(\omega) = \tfrac{1}{2}\left[\delta(\omega - \frac{1}{2} J) + \delta(\omega + \frac{1}{2} J)\right]`

**Check points:**
- Excitation frequency $\omega_0=\sqrt{b_1^2}=J/2$ (spin precession). Weight $1/2$ each.

### Step 6: multi-mode (two coefficients = a central peak plus side peaks)

```
Find the spectral function of S^x_2 for the Heisenberg model on a 3-site chain
```

**Expected output (spectral part):**
`S(\omega) = \frac{5}{9}\,\delta(\omega) + \frac{2}{9}\,\delta(\omega - \frac{3}{2} J) + \frac{2}{9}\,\delta(\omega + \frac{3}{2} J)`

**Check points:**
- For the center site (coordination number 2), $b_1^2+b_2^2=9J^2/4$ → side peaks at
  $\pm 3J/2$. The weights sum to $5/9+2/9+2/9=1$ (sum rule).

### Step 7: the general formula via the coordination number z (no lattice specified)

```
Find the 2nd moment of S^x for the Heisenberg model on a general lattice
```

**Expected output:** `M_0 = \frac{1}{4},\ M_2 = \frac{1}{8} J^{2} z`

**Check points:**
- Substituting $z=2$ (interior of a chain) gives $M_2=J^2/4$, consistent with the
  concrete value in step 6.

### Step 8: high-temperature expansion of the energy (general formula, up to 4th order)

```
Find the high-temperature expansion of the Heisenberg model in general form up to 4th order
```

**Expected output (key terms):**
- `\langle H^2 \rangle = \frac{3}{16} J^{2} |E|`
- `\langle H^3 \rangle = \frac{9}{32} J^{3} N_\triangle - \frac{3}{32} J^{3} |E|`
- `\langle H^4 \rangle = \frac{9}{32} J^{4} N_\square - \cdots` ($N_\square$ = the square
  count)

**Check points:**
- The **triangle count $N_\triangle$** shows up in $\langle H^3\rangle$, and the
  **square count $N_\square$** in $\langle H^4\rangle$.

### Step 9: connecting to the 3c-graph (cross-checked on a concrete lattice)

```
Find the high-temperature expansion of the Heisenberg model on a 3x3 square lattice
```

**Expected output:** the concrete values are displayed along with
`(|E| = 12,\ N_\triangle = 0)`, and `\langle H^3 \rangle = -\frac{9}{8} J^{3}`

**Check points:**
- The square lattice has zero triangles → $\langle H^3\rangle=c_3|E|=-\tfrac{3}{32}\cdot 12\,J^3=-\tfrac{9}{8}J^3$,
  reproducing the general formula.

### Step 10: counting triangles directly (3c-graph connection, direct check)

```
Find the number of triangles of the graph (1,2),(2,3),(1,3)
```

**Expected output:** `\#\triangle = 1` ($=\tfrac16\mathrm{tr}\,A^3$).

**Check points:**
- For a 3-vertex graph that is a single closed loop (one triangle), the triangle
  count correctly comes out to 1 — a direct check backing up step 9's
  high-temperature expansion (corresponds to `graph_triangle_count` in the
  coverage table).

### Step 11 (optional): the continuous band of an infinite system

```
Find the continuous spectral density of the square-root terminator
```

**Expected output:** the square-root terminator $t(z)=(z-\sqrt{z^2-4b_\infty^2})/2$ and
the Wigner semicircle
`S(\omega) = \frac{\sqrt{4 b_\infty^2 - \omega^2}}{2\pi b_\infty^2}\ (|\omega|\le 2 b_\infty)`

**Check points:**
- Contrasts the discrete delta functions of a finite cluster (steps 5–6) with the
  continuous band of an infinite system, side by side as closed-form expressions.

---

## Coverage (features verified by this scenario)

| Feature | Example input | Implementation |
|---|---|---|
| Nested commutator (resolves problem B) | double commutator | `spin_algebra.nested_commutator` |
| ∞-temperature moment | moments up to order m ($M_m$) / up to depth m (m recursion steps) | `moments` |
| Recursion continued fraction | continued fraction | `continued_fraction` |
| Spectral function (single/multi mode) | spectral function | `spectral_peaks` / `spectral_latex` |
| General formula via coordination number z | ... moment for a general lattice | `general_low_moments` |
| High-T expansion ↔ lattice loops | high-temperature expansion in general form up to 4th order | `energy_moment_general_latex` |
| Triangle count (3c-graph connection) | number of triangles of a graph | `graph_triangle_count` |
| Square-root terminator (continuous band) | square-root terminator ... | `terminator_spectral_latex` |

**Related design docs:** (the spin algebra lives in `notebook/runtime/spin_algebra.py`)

## Observed behaviour / notes

`ui/interpret_f1.py` (the F-1 lattice/spin-algebra parsers) already had substantial
bilingual support from earlier work in this codebase — commutator, moment, continued
fraction, spectral function, general-lattice, and square-root-terminator triggers all
already accepted English, including "double commutator" for the 2nd-order case. Only
one narrow gap remained:

- **A hyphenated site count ("3-site chain") didn't match the site-count pattern.**
  `_NS_NSITE_RE` (`ui/interpret_f1.py`) required whitespace (or nothing) between the
  digit and "site"/"サイト", so "2 sites" matched but "3-site chain" did not (the
  hyphen isn't whitespace). Extended the pattern to also accept a hyphen.

(Note: a genuinely unfixed gap remains for 3rd-order-and-above *named* nested
commutators — `_ORDER_N_RE`, e.g. Japanese "3段交換子"/"三重交換子" — which has no
English equivalent. This scenario doesn't exercise it (only 1st and 2nd order, the
latter already covered by the English "double commutator" alias), so it's left as a
known gap rather than fixed here.)

**Verified:** 2026-07-02 (real Maxima backend, generator run, all 11 steps completed
end-to-end, same results as the Japanese version).

### Reproducibility and a known scaling limit (2026-07-08 re-audit)

A researcher reported being unable to reproduce these results by typing the commands
above directly, and asked whether the results shown here are genuinely computed by
the CAS-bypass spin-algebra engine or were fabricated in Python. Both claims were
re-verified from scratch:

- **The exact commands above ARE genuinely reproduced**, byte-for-byte, by driving
  the real interpretation pipeline (`ui.interpret_pipeline.run_interpret_pipeline`,
  the same function the `/interpret` HTTP endpoint and the generator both call) with
  a fresh session and no LLM in the loop — all 11 steps, same LaTeX, same coefficients,
  confirmed independently on both the Maxima and (for the same commands) the Julia
  backend, and with the command entered as both `text` and `math` input tokens.
  `notebook/runtime/spin_algebra.py` performs a real, deterministic, closed-form
  su(2)-algebra calculation (exact Levi-Civita commutators + rational-coefficient
  bookkeeping) — it is not a lookup of canned answers.
- **A genuine, separate bug was found while probing beyond the documented parameters**:
  requesting the spectral function for a **4-site** Heisenberg chain (one step past
  this scenario's validated 3-site case) triggered a runaway computation — term counts
  growing 1 → 4 → 27 → 164 → 967 in four commutator steps (confirmed by direct
  profiling), heading toward a multi-minute, multi-gigabyte-memory hang with no
  result. Root cause: the module has no reduction rule for two or more operator
  factors landing on the same site (`notebook/runtime/spin_algebra.py`'s `_collect`);
  on a chain, adjacent bonds share a site, so repeated commutators can keep
  re-introducing same-site factors that never cancel. This is a *performance* defect
  in the underlying algebra, not evidence that the documented results are fabricated
  — the 2-3 site cases in this scenario terminate their recursion (hit zero norm)
  long before the problem can manifest, which is exactly why they reproduce reliably.
  **Fixed defensively** (not root-caused — implementing a correct same-site product
  reduction is a nontrivial extension of the su(2) algebra and was judged too risky to
  attempt without a much larger validation pass): `_collect` now raises
  `ComplexityLimitExceeded` once a term list passes a safety threshold (3000 terms,
  comfortably above anything the validated 2-3 site scenarios ever reach), which
  `ui/interpret_pipeline.py` catches and turns into an immediate `{"status":
  "failed", ...}` response instead of an unbounded hang.
- **[Added 2026-08-06] The "not root-caused" above no longer holds (open_items A-5d).**
  Implementing the spin-½ same-site reduction ($S^aS^a=1/4$,
  $S^aS^b=(i/2)\epsilon_{abc}S^c$) changed the term count **from growing to
  saturating** — the 4-site chain's repeated commutators went from
  2→12→76→474→2758→abort to **2→6→14→21→24→28→30**, and the 4-site spectral function
  described as "runaway" above now goes from **failing after 185.9 s to completing in
  0.1 s**. What changed is the growth law, not just the constant: the spin-½ operator
  space is finite ($4^N$), so at a fixed lattice size the count saturates and no
  number of further commutators reaches the limit.
  **None of this scenario's 2-3 site results change** (verified differentially — the
  moments agree with and without the reduction).
  The reduction applies on the **evaluation** path (moments, continued fraction,
  spectral function), where no declaration is needed because the inner product is
  already the spin-½ trace. The path that **displays a commutator itself** (this
  scenario's opening $[H,[H,S^x_1]] = J^2 S^x_1 (S^z_2)^2$) stays general-spin and
  reduces only where per-site spins are declared — that $(S^z_2)^2$ is not an
  unsimplified leftover but the correct general-spin answer.
  The threshold guard remains: the wall moved from **4 sites to 8**
  (measured: 4 sites 0.11 s, 6 sites 4.9 s, 8 sites hits the limit in ~1 s).
- **Practical guidance if you cannot reproduce a result**: use the exact wording shown
  in the `generator commands` block below (not a paraphrase), start from a *fresh*
  session with no prior state, and confirm the session's primary backend/domain match
  "Session setup" above. If a command instead hangs for more than a few seconds, it is
  very likely the same-site term-count issue above (as of 2026-08-06 that means 8+
  sites; try a smaller lattice, or a smaller depth) rather than a discrepancy in the
  documented 2-3 site results.

## generator commands (this `scenario` block is the single source used to generate the example)

```scenario domain=statistical_mechanics
@note: For the Ising model (2 sites), compute the commutator $[H,S^x_1]$ of the spin operator $S^x_1$ — the first step of finding how $S^x$ evolves in time (its equation of motion).
Find the commutator of S^x_1 for the Ising model with 2 sites
@note: Compute the double commutator $[H,[H,S^x_1]]$ of the same operator (the first-order result feeds directly into the second, corresponding to an "acceleration").
Find the double commutator of S^x_1 for the Ising model with 2 sites
@note: Find the infinite-temperature moments $M_0=\langle(S^x)^2\rangle,\,M_2$, in preparation for expanding the operator's time evolution in a Krylov subspace.
Find the moments up to 2nd order of S^x_1 for the Ising model with 2 sites
@note: From the moments, find the continued-fraction coefficient $b_1^2$ representing the Green's function of the dynamical correlation function (the recursion method).
Find the continued fraction of S^x_1 for the Ising model with 2 sites
@note: From the continued fraction, find the spectral function $S(\omega)$. Since this is a finite system, it should be a sum of discrete delta functions.
Find the spectral function of S^x_1 for the Ising model with 2 sites
@note: Now find the spectral function at the central site of a 3-site Heisenberg chain. There should be two coefficients this time, giving a central peak plus side peaks.
Find the spectral function of S^x_2 for the Heisenberg model on a 3-site chain
@note: Without specifying a lattice shape, find the second moment of the Heisenberg model as a general formula in the coordination number $z$.
Find the 2nd moment of S^x for the Heisenberg model on a general lattice
@note: Find the high-temperature expansion of the Heisenberg model's energy, up to 4th order, as a general formula (in graph invariants such as edge count and triangle count) without specifying a lattice shape.
Find the high-temperature expansion of the Heisenberg model in general form up to 4th order
@note: Apply the general formula to a concrete $3\times3$ square lattice and check consistency with the general formula from the previous step.
Find the high-temperature expansion of the Heisenberg model on a 3x3 square lattice
@note: Directly count the number of triangles in a 3-vertex graph (a direct check backing up the triangle count $N_\triangle$ appearing in the 3rd-order term of the high-temperature expansion).
Find the number of triangles of the graph (1,2),(2,3),(1,3)
@note: To contrast with the discrete spectrum (sum of delta functions) of a finite cluster, find the continuous spectral density of an infinite system (the square-root terminator).
Find the continuous spectral density of the square-root terminator
```
