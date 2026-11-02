# Practice-loop scenario: computing the equation-of-motion commutator in operator algebra

> Run 2026-08-21. ONE step of the sister scenario
> [scenario_greens_function_from_hamiltonian](scenario_greens_function_from_hamiltonian_en.md)
> (Maxima), done on the backend that has an operator algebra.

**Backend:** Julia full (QuantumAlgebra) / **domain:** condensed matter (`condensed_matter`).

### Why it is a separate session (**this split was decided on 2026-07-30/31**)

The division of labour is not a decision made here; it is on the record:

> **Implication for the choice of backend.** … the condensed-matter domain
> defaults to Julia and the backend is fixed when the session is created (Core
> Principle #6), so **if you are doing Matsubara Green's-function algebra the
> right move today is to choose a Maxima session**, and Julia only where operator
> algebra (Wick contraction, normal ordering) is needed.
> — [docs/33 §7](../developer_manuals/33_time_dependent_operators_and_green_functions.md) (2026-07-30)

The main line is **SCALAR-HANDOFF** (Julia → Maxima, [docs/34 §5](../developer_manuals/34_field_operators_pair_representation_and_scalar_handoff.md),
decided 2026-07-31, implemented 2026-08-02). The scenario that actually crosses
that boundary is [S-c (the BCS gap equation)](scenario_bcs_gap_equation_en.md),
which hands over **after** `vacuum_expectation` has reduced the expression to
c-numbers.

**The handoff cannot be used here** (measured 2026-08-21): what would have to
cross is the commutator's result, and that is still operator-valued —

> E2 still contains operators (c, c'), so it cannot be handed to a scalar-CAS
> session. Only expressions free of operators can cross (expectation values,
> coefficients — quantities from after the operator algebra is done).

**The other direction is closed too**: the two-time Green's function
$c_{n}(t)c^\dagger_{m}(t')$ cannot be operated on in Julia at all (a
time-carrying operator has no place in QuantumAlgebra — docs/33 §5; open_items
PAIR-REP is unimplemented). So this derivation's boundary does **not** coincide
with the boundary the mechanism can carry — which is why the hand-over into the
sister scenario is a note.

### Why it is a separate session (the physics)

The core of the Green's function's equation of motion is $[\hat c_{m\sigma}, \hat H]$.
**Maxima has no operator algebra**, so that session carries operator pairs as single
names and leaves this one commutator uncomputed (the diagonalisation, the move into
the basis, Dyson's formal solution, the Fourier transform and the substitutions are
all computed there). **The backend is fixed when the session is created and cannot be
switched** (Core Principle #6), so the commutator is computed here, in its own
session, and the result is written across — and **the history records which session
computed what**, which is what makes writing it across legitimate.

### What this confirms

- **A commutator with a summed Hamiltonian is computable**: against
  $\sum_n \varepsilon_n c^\dagger_n c_n$ the delta contracts and $-\varepsilon_m c_m$
  is left (the sum is gone)
- The **four-index interaction** opens into two terms with $V$'s indices intact
- Indexed scalars $\varepsilon_n$, $V_{p,q,s,n}$ run over the bound summation indices

```scenario domain=condensed_matter
@note: Declare the electron annihilation operator and the scalar coefficients. $\varepsilon$ is the mode energy, $V$ the two-body matrix element.
define c as an electron annihilation operator
define \varepsilon as a scalar parameter
define V as a scalar parameter
@note: The Hamiltonian. The one-body term is a sum over modes, the two-body term a sum over four indices with an indexed scalar coefficient (which lowers to QuantumAlgebra's `param`). These correspond to $\hat H_0$ (diagonal) and $\hat V$ in the sister scenario.
H = \sum_n \varepsilon_n c'(n) c(n) + \sum_{p q s n} V_{p,q,s,n} c'(p) c'(q) c(s) c(n)
@note: **Compute the commutator.** By hand, the one-body term contracts through $\delta_{nm}$ and leaves $-\varepsilon_m c_m$ (the sum disappears); the two-body term gives two $c^\dagger c c$ terms.
commutator of E1 and c(m)
@note: This is the right-hand side of the Green's function's equation of motion — the step that is written across into the sister scenario. The sign is for $[\hat H,\hat c]$; $[\hat c,\hat H]$ is its negative.
@note: **From here on the time-carrying operator itself is the object** (PAIR-REP, docs/34 §2). $\hat c(t)=e^{i\hat Ht}\hat ce^{-i\hat Ht}$ is not a new kind of quantity but the sum of nested commutators $\sum_n (it)^n/n!\,\mathrm{ad}_H^n(\hat c)$. With a diagonal $\hat H_0$, $[\hat H_0,\hat c_m]=-\varepsilon_m\hat c_m$ closes on ONE operator, so the sum resums into an exponential **exactly** — this is not a truncation.
@note: Take the one-body Hamiltonian alone (the interaction does not close the ad-chain; that is slice 2's territory).
H_0 = \sum_n \varepsilon_n c'(n) c(n)
@note: Evolve the annihilation operator. The coefficient $e^{-i\varepsilon_m t}$ is a SCALAR and cannot live in QuantumAlgebra (which decomposes a c-number into cos/sin), so the substance is the (coefficient, operator) PAIR and only the display looks like $\hat c_m(t)$.
evolve c(m) under E3
@note: The creation operator too, at a different time $u$. The sign flips ($[\hat H_0,\hat c^\dagger_m]=+\varepsilon_m\hat c^\dagger_m$).
evolve c'(m) in time u under E3
@note: **The different-time anticommutator.** This is what the representation is for: $\{\hat c_m(t),\hat c^\dagger_m(u)\}=f(t)\overline{f(u)}\{\hat c_m,\hat c^\dagger_m\}$ — coefficient product times equal-time algebra. Folding time into an index would have collapsed it to $\delta(t,u)$. The equal-time $\{\hat c,\hat c^\dagger\}=1$ cancels and what remains is a SCALAR with no operator in it: the time dependence of the free propagator.
the anticommutator of E4 and E5
@note: The commutator keeps its operator ($1-2\hat n_m$), so its result is again a coefficient-times-operator pair.
the commutator of E4 and E5
@note: The anticommutator's result carries no operator, so **this is the session boundary**: once it is a scalar it can cross to Maxima (SCALAR-HANDOFF). The remaining algebra into $\omega$ space is that side's work.
@note: **A chain that does not close on ONE operator is still exact if it closes on finitely many.** Try the two-site model with hopping: $\mathrm{ad}$ closes on the two-dimensional span of $\hat c_1$ and $\hat c_2$, and the coefficients become a matrix exponential.
H_w = w (c'(1) c(2) + c'(2) c(1))
@note: Evolving the annihilation operator now MIXES two operators ($\cos$ and $-i\sin$). That is what differs from the diagonal case, which closes on one.
evolve c(1) under E10
evolve c'(1) in time u under E10
@note: The different-time anticommutator. Two of the four combinations survive, $\cos\cos+\sin\sin$ is collected on the scalar side, and what comes out is a function of the TIME DIFFERENCE alone: $\cos(w(t-u))$ — the free propagator of the two-site model.
the anticommutator of E11 and E12
```
