# Practice-loop sample scenario: partition function of the 2D Ising lattice (small lattices, high-temperature expansion)

> (2026-07-02: an English translation of the Japanese scenario, verified against the
> real Maxima backend through the same rule-based, no-LLM generator pipeline. See
> [scenario_ising_partition_function_jp.md](scenario_ising_partition_function_jp.md)
> for the Japanese version.)

Detailed write-up of practice-loop scenario-menu item **S-g (statistical mechanics)**.
Confirm the **high-temperature expansion of the partition function (free energy)** of
the 2D Ising model on small lattices, alongside exact expectation values. See that the
expansion coefficients are determined by **lattice loop counts** (edge count $|E|$,
triangle count $N_\triangle$, square count $N_\square$), and that the **triangular
lattice has odd-order terms** (the signature of geometric frustration). Evaluated via
spin-algebra (this project's own spin algebra, bypassing the CAS).

## Physical background

The partition function $Z=\mathrm{Tr}\,e^{-\beta H}$ of the Ising model
$H = -J\sum_{\langle ij\rangle} s_i s_j$ ($s=\pm\tfrac12$) expands at high temperature
($\beta\to0$) as

$$\ln Z = N\ln 2 + \tfrac12\kappa_2\beta^2 - \tfrac1{3!}\kappa_3\beta^3 + \tfrac1{4!}\kappa_4\beta^4 + \cdots$$

and the specific heat $C(\beta)=\beta^2\,\partial^2(\ln Z/N)/\partial\beta^2|_{\text{intensive}}$
can be written in terms of the **connected energy cumulants** $\kappa_n=\langle H^n\rangle_c$.
The infinite-temperature moments $\langle H^n\rangle=\mathrm{Tr}\,H^n/\mathrm{Tr}\,1$ can
be expressed as graph invariants:

$$\langle H^2\rangle = c_2\,|E|,\qquad \langle H^3\rangle = c_3\,|E| + d_3\,N_\triangle,\qquad
\langle H^4\rangle = (\cdots)\,N_\square + \cdots$$

For the Ising model, $\langle H\rangle=0$ (spin-flip symmetry) and $c_3=0$ (the 3rd order
of a single bond vanishes), so $\langle H^3\rangle = d_3 N_\triangle$ is **proportional
to the triangle count alone**. Hence:

- **Bipartite (square) lattice**: $N_\triangle=0\Rightarrow$ the odd-order $\beta^3$
  term vanishes.
- **Triangular lattice**: $N_\triangle>0\Rightarrow$ the $\beta^3$ term appears
  (geometric frustration).

This is the core fact this scenario confirms.

## Session setup

- Backend: **standard symbolic (Maxima)**. spin-algebra's moments/high-temperature
  expansion are evaluated without going through the CAS (registered via
  `register_dispersion`). No domain needs to be specified.
- Assumes the infinite-temperature trace of spin-½ (a single site has $(s)^2=1/4$).

## Input steps

### Step 1: the general formula for the energy moments (graph invariants)

```
Find the high-temperature expansion of the Ising model in general form
→ ⟨H²⟩ = (1/16) J² |E|,  ⟨H³⟩ = (3/32) J³ N△
```

The 3rd order is proportional **only to the triangle count $N_\triangle$** (the Ising
model's $c_3=0$). This is the bridge connecting "3rd moment ↔ 3c-graph's
$\mathrm{tr}\,A^3$".

### Step 2: specialize to a 2×2 square lattice (a small lattice)

```
Find the high-temperature expansion of the Ising model on a 2x2 square lattice
→ ⟨H²⟩ = (1/4) J²,  ⟨H³⟩ = 0    (|E| = 4,  N△ = 0)
```

The 2×2 periodic boundary is a 4-cycle ($|E|=4$, no triangles). Matches the value
obtained by substituting $|E|=4,\,N_\triangle=0$ into the general formula.
**Hand-calculation cross-check**: $\langle H^2\rangle = J^2\sum_{b}\langle (s_is_j)^2\rangle = 4\cdot\tfrac1{16}J^2 = \tfrac14J^2$
(cross terms between different bonds vanish at odd order).

### Step 3: high-temperature expansion of the specific heat (= the partition function)

```
Find the high-temperature expansion of the specific heat of the Ising model on a 2x2 square lattice
→ C(β) = (1/4) J² β² + (1/32) J⁴ β⁴ + O(β⁵)
```

**No $\beta^3$ term** ($N_\triangle=0$). Corresponds to the second derivative of the
free energy $\ln Z$.

### Step 4: up to 4th order (the square count $N_\square$ appears)

```
Find the high-temperature expansion of the Ising model on a 2x2 square lattice up to 4th order
→ ⟨H²⟩ = (1/4) J²,  ⟨H³⟩ = 0,  ⟨H⁴⟩ = (1/4) J⁴   (|E| = 4,  N△ = 0,  N□ = 1)
```

The 2×2 lattice has exactly one 4-cycle ($N_\square=1$). The origin of the $\beta^4$
term.

### Step 5: hand-calculation cross-check of the $\beta^4$ coefficient via cumulants

Since $\langle H\rangle=0$, the 4th cumulant is $\kappa_4 = \langle H^4\rangle - 3\langle H^2\rangle^2$.

$$\kappa_4 = \tfrac14 J^4 - 3\left(\tfrac14 J^2\right)^2 = \tfrac14 J^4 - \tfrac3{16}J^4 = \tfrac1{16}J^4,
\qquad \tfrac12\kappa_4 = \tfrac1{32}J^4.$$

Matches the $\beta^4$ coefficient $\tfrac1{32}J^4$ from step 3. Likewise the $\beta^2$
coefficient is $\kappa_2=\langle H^2\rangle=\tfrac14J^2$.

### Step 6: a 3×3 triangular lattice (frustration → odd-order terms appear)

```
Find the high-temperature expansion of the Ising model on a 3x3 triangular lattice
→ ⟨H²⟩ = J²,  ⟨H³⟩ = (3/4) J³    (|E| = 16,  N△ = 8)
```

Substituting $|E|=16,\,N_\triangle=8$ into the general formula:
$\tfrac1{16}J^2\cdot16=J^2$, $\tfrac3{32}J^3\cdot8=\tfrac34J^3$.
**With triangles present, $\langle H^3\rangle\ne0$.**

### Step 7: the triangular lattice's specific heat (the $\beta^3$ term appears)

```
Find the high-temperature expansion of the specific heat of the Ising model on a 3x3 triangular lattice
→ C(β) = J² β² − (3/4) J³ β³ + (5/16) J⁴ β⁴ + O(β⁵)
```

In contrast to the square lattice (step 3), **the $\beta^3$ term $-\tfrac34J^3\beta^3$
appears**. The coefficient $-\kappa_3\beta^3 = -\langle H^3\rangle\beta^3 = -\tfrac34J^3\beta^3$
(for the Ising model, $\kappa_3=\langle H^3\rangle$) is consistent with step 6.
**The presence or absence of odd-order terms is a signature of the triangle count
(geometric frustration).**

## Summary of expected values (exact, cross-checked)

| Quantity | Square 2×2 | Triangular 3×3 |
|---|---|---|
| Invariants | $|E|=4,\ N_\triangle=0,\ N_\square=1$ | $|E|=16,\ N_\triangle=8$ |
| $\langle H^2\rangle$ | $\tfrac14J^2$ | $J^2$ |
| $\langle H^3\rangle$ | $0$ | $\tfrac34J^3$ |
| $\langle H^4\rangle$ | $\tfrac14J^4$ | — |
| $C(\beta)$ | $\tfrac14J^2\beta^2+\tfrac1{32}J^4\beta^4$ | $J^2\beta^2-\tfrac34J^3\beta^3+\tfrac5{16}J^4\beta^4$ |

## Coverage (features verified by this scenario)

- **F-1 lattice-sum specialization**: cross-checking the general formula (graph
  invariants $|E|,N_\triangle,N_\square$) against a concrete lattice (square /
  triangular).
- **Energy moments**: $\langle H^2\rangle,\langle H^3\rangle,\langle H^4\rangle$
  (infinite-temperature trace).
- **High-temperature expansion of the specific heat**: the free-energy (partition
  function) expansion in terms of connected cumulants $\kappa_n$.
- **Visualizing the physics**: the presence/absence of the odd-order $\beta^3$ term =
  the triangle count = the signature of geometric frustration.
- Backend: standard symbolic (Maxima), spin-algebra evaluated without going through
  the CAS.

## Observed behaviour / notes

- **`ui/interpret_f1.py` (the F-1 lattice/spin-algebra parsers) already had
  substantial bilingual support** from earlier work in this codebase — the Ising/
  Heisenberg model-name detection, the "high-temperature expansion"/"energy moment"
  trigger, and the "square lattice"/"triangular lattice" keywords all already accepted
  English. Only one narrow gap remained: **the "up to Nth order" depth specifier**
  (`_SPEC_DEPTH_RE`) only recognized the Japanese 次/段/深さ suffixes, so "up to 4th
  order" silently fell back to the default depth (3rd order) instead of the requested
  4th — a **silent wrong-answer**, not an error (step 4 would have returned the same
  3-term result as step 2, missing $\langle H^4\rangle$ entirely). Added English "up to
  Nth order" / "to order N" / "depth N" alternatives.

**Verified:** 2026-07-02 (real Maxima backend, generator run, all 6 steps completed
end-to-end, same results as the Japanese version, including the 4th-order term).

## generator commands (this `scenario` block is the single source used to generate the example)

```scenario domain=statistical_mechanics
@note: For the Ising model $H=-J\sum_{\langle ij\rangle}s_is_j$ ($s=\pm\frac12$), find the infinite-temperature energy moments $\langle H^2\rangle,\langle H^3\rangle$ without specifying a lattice shape, as a general formula in graph invariants (edge count $|E|$, triangle count $N_\triangle$). For the Ising model, $\langle H^3\rangle$ is proportional to the triangle count $N_\triangle$ alone.
Find the high-temperature expansion of the Ising model in general form
@note: Apply the general formula to a specific lattice: a $2\times2$ square lattice with periodic boundaries. This lattice is a 4-cycle ($|E|=4$, no triangles, so $N_\triangle=0$), so $\langle H^3\rangle$ should vanish.
Find the high-temperature expansion of the Ising model on a 2x2 square lattice
@note: From the energy moments, find the high-temperature expansion (power series in $\beta$) of the specific heat $C(\beta)=\beta^2\,\partial^2(\ln Z/N)/\partial\beta^2$. This is NOT a literal Maxima command: the recursion/moment method has no generic CAS routine for it, so it is computed by this project's own spin-algebra module (`notebook/runtime/spin_algebra.py`) as an exact, deterministic closed-form calculation — the same graph-invariant energy moments $\langle H^2\rangle,\langle H^3\rangle,\langle H^4\rangle$ from the previous steps are converted to connected cumulants $\kappa_n$ ($\kappa_2=\langle H^2\rangle$, $\kappa_4=\langle H^4\rangle-3\langle H^2\rangle^2$ since $\langle H\rangle=0$) and assembled into the power series $C(\beta)=\kappa_2\beta^2-\tfrac1{3!}\kappa_3\beta^3+\tfrac1{2\cdot2!}\kappa_4\beta^4+\cdots$. For this $2\times2$ lattice, $\kappa_2=\langle H^2\rangle=\tfrac14J^2$ and $\kappa_4=\tfrac14J^4-3(\tfrac14J^2)^2=\tfrac1{16}J^4$, giving the $\beta^2$ and $\beta^4$ coefficients below by hand — a cross-check anyone can redo, not a black box. Since this lattice has no triangles, no odd-order $\beta^3$ term should appear.
Find the high-temperature expansion of the specific heat of the Ising model on a 2x2 square lattice
@note: Extend the expansion to 4th order. The $2\times2$ lattice has exactly one 4-cycle (square count $N_\square=1$), which is the origin of the $\beta^4$ term.
Find the high-temperature expansion of the Ising model on a 2x2 square lattice up to 4th order
@note: Now repeat the calculation on a $3\times3$ triangular lattice. Since this lattice has triangles ($N_\triangle=8$), $\langle H^3\rangle$ should be nonzero (geometric frustration).
Find the high-temperature expansion of the Ising model on a 3x3 triangular lattice
@note: Repeat the same moments-to-cumulants-to-power-series computation (see the note on the square-lattice specific heat above — same spin-algebra module, no CAS command involved) for the triangular lattice. In contrast to the square lattice (no triangles), an odd-order $\beta^3$ term should actually appear, with coefficient $-\kappa_3\beta^3=-\langle H^3\rangle\beta^3=-\tfrac34J^3\beta^3$ (for the Ising model $\kappa_3=\langle H^3\rangle$ exactly, since $\langle H\rangle=0$).
Find the high-temperature expansion of the specific heat of the Ising model on a 3x3 triangular lattice
```
