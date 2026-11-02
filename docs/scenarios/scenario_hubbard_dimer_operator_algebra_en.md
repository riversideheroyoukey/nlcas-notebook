# Practice-loop scenario: computing the two-site Hubbard model all the way through, in non-commuting operators

> Run 2026-08-22. **Nothing in this scenario is an answer typed in as TeX by
> someone who already knew it.** What is registered is the Hamiltonian and the
> definitions of the operators being compared; **every result below is what the
> operator algebra computed.**

**Backend:** Julia full (QuantumAlgebra) / **domain:** condensed matter.

### Why this scenario was added

The Green's-function scenarios established that the notation can be **written
and displayed**. But most of the quantum steps had become checks on input
correctness, with few lines the CAS actually computed. Measured 2026-08-22:
`scenario_greens_function_from_hamiltonian` computed 10 of 30 written
expressions; `spinful_matrix_elements`, `tight_binding_bands` and
`spin_chain_spectrum` computed **none**. Other domains run at 60–89%.
**Operator algebra is where this kind of session should be strongest**, so the
gap was in the notation layer, not the physics.

Measured, the blockage was not QuantumAlgebra but the layer that reads the
instruction. QA answered `{c(1,↑), c†(1,↑)} = 1` correctly all along; four
spellings never reached it:

| spelling | what arrived |
|---|---|
| `anticommutator of c(1,u) and c'(1,u)` | `b="c'(1"` → a raw `ParseError:` |
| `[c(1,u), c'(1,u)]` | `a="c(1"`, `b="u), c'(1,u)"` — **split silently in the wrong place** |
| `c_{1\uparrow}` | refused as the index `1 * u` (the arrow was collapsed to a letter first, erasing the separator it WAS) |
| `commutator of n_{1\uparrow} and n_{1\downarrow}` | `[c^†_{1↑}, c_{1↑}]` — **a different pair of operators, answered as a success** |

Each approximated "where does the operand end?" with whitespace and commas
instead of brackets and particles. One root.

### A range that was written is computed as written

The first version of this scenario wrote the Hamiltonian out longhand. Told that
this was unnatural, we measured — and the sum notation had not been "unused", it
had been **silently wrong**:

| spelling | measured (first run) |
|---|---|
| `\sum_{i=1}^{2}` | the range 1..2 **disappears**: `[U\sum_{i=1}^{2}\hat n_{i\uparrow}\hat n_{i\downarrow}, \hat c_{1\uparrow}]` answered `\sum_i[U c^†_{1↓}c_{1↑}c_{1↓}]` as a success — a $\sum_i$ over a summand with no $i$ in it |
| `\sum_{\sigma}` | the **raw CAS text** `ArgumentError("index names must be single character, got sigma.")` |
| `\{A, B\}` | not read as an anticommutator at all; registered as an ordinary expression |

QuantumAlgebra's $\sum$ is a **formal** sum with nowhere to put a range. That is
no licence to drop the range: **a concrete integer range is finitely many terms,
and expanding is mechanical and exact** (this system already does exactly that in
three other places). A `\sum_\sigma` with no range written is NOT expanded — there
the formal sum is precisely what the notation says.

### An arrow is a value; σ is a variable

`\uparrow` goes straight into a QuantumAlgebra index (QA's rule is "one unicode
character", which an arrow is). And **an arrow cannot be bound by a sum** —
nobody writes `\sum_\uparrow`; they write `\sum_\sigma`. So two different arrows
are two different **values**, and `δ(↑,↓) = 0` follows by exactly the warrant
that makes `δ(1,2) = 0` for integer sites. `\sigma` stays a letter, so `δ(s,s')`
inside `\sum_\sigma` keeps its sifting property — **the distinction the physics
needs falls straight out of what was written.**

Without it every commutator below carried `δ_{↑↓}` terms that the researcher
knows are zero and the system could not.

### What this establishes

- **Conserved quantities come out of the computation**: $[\hat H,\hat N]=0$,
  $[\hat H,\hat S^z]=0$, $[\hat H,\hat S^+]=0$
- **The SU(2) algebra is derived**: $[\hat S^+,\hat S^-]=2\hat S^z$ is computed
  rather than assumed, and the CAS confirms the difference is 0
- **What is NOT conserved comes out in the form that says why**: the commutator
  with the double occupancy $\hat D$ keeps only the four hopping terms; $U$ drops out
- **Pauli's identity** $\hat n^2=\hat n$ falls out of normal ordering
- **The exact time evolution in the atomic limit** (PAIR-REP):
  $\hat c_{1\uparrow}(t)=\hat c_{1\uparrow}+(1-e^{-iUt})\hat c^\dagger_{1\downarrow}\hat c_{1\uparrow}\hat c_{1\downarrow}$

```scenario domain=condensed_matter lang=en
@note: Declare the electron annihilation operator, the hopping $w$, the on-site repulsion $U$ and the time $t$.
define c as an electron annihilation operator
define w as a scalar parameter
define U as a scalar parameter
define t as a scalar parameter
@note: **The two-site Hubbard model, written in the textbook sum notation.** The spin sum $\sum_\sigma$ carries no range, so it goes to QuantumAlgebra's formal sum as written; the site sum $\sum_{i=1}^{2}$ has a concrete range and is therefore **expanded into terms** before it reaches the backend (QA's $\sum$ has nowhere to put a range — which is no reason to drop the range silently and compute something other than what was written).
H = -w \sum_{\sigma} (c^\dagger_{1\sigma} c_{2\sigma} + c^\dagger_{2\sigma} c_{1\sigma}) + U \sum_{i=1}^{2} c^\dagger_{i\uparrow} c_{i\uparrow} c^\dagger_{i\downarrow} c_{i\downarrow}
@note: Ask for the underlying anticommutators **in the bracket notation itself**: 1 for the same site and spin, 0 for a different site, and **0 for a different spin** — that last one stays as $\delta_{\uparrow\downarrow}$ unless the arrow is read as a value.
\{c_{1\uparrow}, c^\dagger_{1\uparrow}\}
\{c_{1\uparrow}, c^\dagger_{2\uparrow}\}
\{c_{1\uparrow}, c^\dagger_{1\downarrow}\}
@note: **Particle-number conservation.** Write the total number $\hat N=\sum_{i\sigma}\hat c^\dagger_{i\sigma}\hat c_{i\sigma}$ and compute $[\hat H,\hat N]$. The $\sum_\sigma$ expands over the projections of the particle that was DECLARED (an electron, $m=\pm1/2$). Neither hopping nor $U$ changes $\hat N$, so it should be 0.
N = \sum_{i=1}^{2} \sum_{\sigma} c^\dagger_{i\sigma} c_{i\sigma}
[E1, N]
@note: **Conservation of $S^z$.** Written the textbook way, $\hat S^z=\sum_{i\sigma}\sigma\,\hat n_{i\sigma}$: the $\sigma$ in the COEFFICIENT position gets the **value** ($\pm1/2$, an exact rational), the one in the INDEX position gets the **spelling** (the arrow). Also 0.
S_z = \sum_{i=1}^{2} \sum_{\sigma} \sigma c^\dagger_{i\sigma} c_{i\sigma}
[E1, S_z]
@note: **Derive the SU(2) algebra.** Write the ladder operators $\hat S^\pm$ and compute $[\hat S^+,\hat S^-]$. It should be $2\hat S^z$ — and the point is to compute that rather than assume it.
S_p = \sum_{i=1}^{2} c^\dagger_{i\uparrow} c_{i\downarrow}
S_m = \sum_{i=1}^{2} c^\dagger_{i\downarrow} c_{i\uparrow}
[S_p, S_m]
@note: Write $2\hat S^z$, subtract it, and let the CAS confirm the identity (it gives 0).
T = 2 \sum_{i=1}^{2} \sum_{\sigma} \sigma c^\dagger_{i\sigma} c_{i\sigma}
subtract E12 from E11
@note: **SU(2) symmetry of $\hat H$**: $[\hat H,\hat S^+]=0$. The $U$ term $\hat n_\uparrow\hat n_\downarrow$ is invariant under flipping the spins, and hopping conserves spin.
[E1, S_p]
@note: **Something that is NOT conserved.** The double occupancy $\hat D=\sum_i\hat n_{i\uparrow}\hat n_{i\downarrow}$ is not a constant of the motion. Only the **hopping** terms survive; $U$ drops out — creating and destroying double occupancy is hopping's doing.
D = \sum_{i=1}^{2} c^\dagger_{i\uparrow} c_{i\uparrow} c^\dagger_{i\downarrow} c_{i\downarrow}
[E1, D]
@note: **The right-hand side of the equation of motion**, the core of $i\hbar\,\partial_t\hat c_{1\uparrow}=[\hat c_{1\uparrow},\hat H]$: hopping carries it to the neighbouring site, $U$ brings in the opposite-spin number operator.
[E1, c_{1\uparrow}]
@note: **A nested commutator is read as written**: $[\hat c_{1\uparrow},[\hat c^\dagger_{1\uparrow},\hat c_{1\uparrow}]]=2\hat c_{1\uparrow}$. Which comma separates is decided by bracket depth, so the inner bracket is part of the operand.
[c_{1\uparrow}, [c^\dagger_{1\uparrow}, c_{1\uparrow}]]
@note: **Pauli's identity.** Normal-ordering $\hat n^2$ gives $\hat n$ back — two electrons cannot occupy the same state. Another algebraic result, not a transcription.
normal order c^\dagger_{1\uparrow} c_{1\uparrow} c^\dagger_{1\uparrow} c_{1\uparrow}
@note: **Exact time evolution in the atomic limit** (PAIR-REP, docs/34 §2). Switch off the hopping and keep the $U$ term: the $\mathrm{ad}_H$ chain closes on finitely many operators, so $\hat c_{1\uparrow}(t)$ comes out **with no truncation**.
H_U = U c^\dagger_{1\uparrow} c_{1\uparrow} c^\dagger_{1\downarrow} c_{1\downarrow}
@note: The coefficient $1-e^{-iUt}$ is a scalar and cannot live in QuantumAlgebra (which decomposes c-numbers into cos/sin) — the substance is the (coefficient, operator) PAIR, and only the display looks like $\hat c_{1\uparrow}(t)$. This is the Hubbard atom's exact result.
evolve c_{1\uparrow} under H_U
```
