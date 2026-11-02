# Practice-loop sample scenario: transverse-field Ising model and mean-field approximation (Julia backend verification)

> (2026-07-02: an English translation of the Japanese scenario, verified against the
> real Julia backend through the SAME rule-based, no-LLM generator pipeline. See
> [scenario_transverse_field_ising_jp.md](scenario_transverse_field_ising_jp.md) for the
> Japanese version and the shared "確認された挙動・注意点" bug notes.)

**Backend:** Julia (QuantumAlgebra.jl) / **Domain:** spin systems (spin_systems).

### Physical background

Two-site transverse-field Ising model:

$$H = -J\, S^z_1 S^z_2 - \Gamma \left( S^x_1 + S^x_2 \right)$$

From the SU(2) commutation relations:

$$[H,\, S^z_1] = -\Gamma\, [S^x_1,\, S^z_1] = 2i\Gamma\, S^y_1$$

A mean-field approximation replaces site 2's spin with the mean field (c-number)
$m_z=\langle S^z\rangle$:

$$S^z_2 \;\to\; m_z \quad\Longrightarrow\quad
H_{\rm MF} = -J\, m_z\, S^z_1 - \Gamma\left(S^x_1 + S^x_2\right)$$

---

### Building it from general operations (★ the point of this scenario)

1. $S$ is declared via "S as the spin operator" (`Sx`, `Sy`, `Sz` become usable).
2. $H$ is registered in H-1 equation form directly, using the suffix component
   notation `Sz_1`.
3. The commutator is entered directly as `[E_n, Sz_1]` in the math input field (bracket
   notation resolves to a commutator as a general operation).
4. The mean-field approximation (replacing a specific site's operator with a c-number)
   is expressed as a general substitution: "In $E_n$, substitute $m_z$ for $S^z_2$".
   Since $S^z_2$'s index is an integer (a site label), it is automatically detected as
   an exact-site substitution (replacing only that site).

---

### Steps

#### ST-1: Start the session

| Item | Content |
|---|---|
| Action | New session, domain: "spin systems" or "condensed matter physics" → the Julia backend is auto-selected |
| Result | |

---

#### ST-2: Declare the parameters and spin operator

| Item | Content |
|---|---|
| Input | `Define J, \Gamma as scalars` |
| Input | `S as the spin operator` |
| Check | `Sx`, `Sy`, `Sz` become usable (`Sp`, `Sm` are not auto-defined) |
| Result | |

---

#### ST-3: Register the Hamiltonian (H-1)

| Item | Content |
|---|---|
| Input | `H = -J * Sz_1 * Sz_2 - \Gamma * (Sx_1 + Sx_2)` |
| Check | Component normalization equivalent to $S^z_1$ and the `\Gamma (...)` multiplication conversion are both handled correctly |
| Result | |

---

#### ST-4: Verify the commutation relation

| Item | Content |
|---|---|
| Input | `[E1, Sz_1]` |
| Expected | $2i\Gamma\, S^y_1$ |
| Check | The SU(2) commutation relation, the imaginary unit, and Kronecker $\delta(1,2)=0$ evaluation are all handled correctly |
| Result | |

---

#### ST-5: Mean-field approximation (site-specific operator substitution)

| Item | Content |
|---|---|
| Input | `Define mz as a scalar` |
| Input | `In E1, substitute mz for S^z_2` |
| Check | Only `Sz_2` is replaced by `mz`; `Sz_1`, `Sx_1`, `Sx_2` are unchanged (an integer index is auto-detected as a site-specific substitution) |
| Expected | $H_{\rm MF} = -\Gamma S^x_1 - \Gamma S^x_2 - J\, m_z\, S^z_1$ |
| Result | |

---

### Verification checklist

- [x] Bulk spin-operator declaration works on the Julia backend
- [x] $S^z_i$ component normalization and `\Gamma (...)` multiplication conversion are correct
- [x] $[H, S^z_1] = 2i\Gamma S^y_1$ is obtained correctly (SU(2) commutation relation, $\delta(1,2)=0$ evaluation)
- [x] The mean-field approximation (`S^z_2 \to m_z`) applies to **site 2 only**, leaving the other operator terms unchanged

---

## Observed behaviour / notes

- **The generic substitution short-circuit ignored a quantum operator's site index
  (root-cause bug, fixed).** `ui/interpret_pipeline.py`'s two generic
  "substitute/replace X with/for Y" short-circuits (`substitute_into` and the 2e4
  short-circuit) dispatched to the plain `substitute` operation regardless of whether
  the LHS names a scalar or a quantum operator. For a site-indexed target like
  `S^z_2`, that path silently falls back to whole-operator identity replacement
  (ignoring the site index) instead of the exact-site `operator_substitute` — and errors
  outright once the RHS is declared as a plain scalar (`Sym mz is not callable`).
  Fixed by guarding both short-circuits to bail out when the LHS starts with a
  registered operator-kind symbol name, deferring to the backend's site-aware
  `operator_substitute` (with its automatic integer-index-means-exact-site detection)
  instead. Same bug class as SUBST-DOUBLE-SHADOW found in the BCS Hamiltonian scenario.
- **English-specific gaps found while building this translation:** the spin-operator
  declaration pattern (`_EN_SPIN_PARSE_RE`) only accepted "as **the** spin operator",
  not "as **a** spin operator" — fixed to accept both, matching the equivalent
  fermion/boson declaration pattern. A natural single-clause English
  `operator_substitute` trigger ("In E<i>n</i>, substitute RHS for op(idx)") also had
  to be added — the only English pattern that previously existed
  (`operator_substitute E1 c(k) with RHS`) used unnatural, CAS-syntax-like phrasing.

**Verified:** 2026-07-02 (real Julia backend, generator run, all 6 steps completed
end-to-end with the same physics as the Japanese version).

---

## generator commands (this `scenario` block is the single source used to generate the example)

```scenario domain=spin_systems
@note: Consider the two-site transverse-field Ising model. Prepare the coupling constant $J$ (the strength of the Ising interaction) and the transverse-field strength $\Gamma$ as scalar parameters.
Define J, \Gamma as scalars
@note: Prepare the spin operator $S$ used on both sites.
S as the spin operator
@note: Register the transverse-field Ising Hamiltonian $H=-JS^z_1S^z_2-\Gamma(S^x_1+S^x_2)$ (the Ising coupling term and the transverse-field term).
H = -J * Sz_1 * Sz_2 - \Gamma * (Sx_1 + Sx_2)
@note: Compute the commutator of the Hamiltonian $H$ and $S^z$ on site 1, to check part of the equation of motion following from the SU(2) commutation relations.
[E1, Sz_1]
@note: To carry out a mean-field approximation, prepare the mean field (a c-number) $m_z=\langle S^z\rangle$, which will replace the spin on site 2, as a scalar parameter.
Define mz as a scalar
@note: Replace only the operator $S^z_2$ on site 2 with the mean field $m_z$, obtaining the mean-field Hamiltonian $H_{\rm MF}$ (the operators on site 1 and the transverse-field term should be unchanged).
In E1, substitute mz for S^z_2
```
