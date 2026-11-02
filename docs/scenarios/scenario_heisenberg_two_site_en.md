# Practice-loop sample scenario: the Heisenberg model (2-site spin chain)

> (2026-07-02: an English translation of the Japanese scenario, verified against the
> real Julia backend through the same rule-based, no-LLM generator pipeline. See
> [scenario_heisenberg_two_site_jp.md](scenario_heisenberg_two_site_jp.md) for the
> Japanese version. No new i18n gaps were found here — all the English NL patterns
> this scenario needs were already fixed while building the earlier translations.)

> (Split off from the former "E-3 research-workflow verification scenario" into its
> own **1 scenario = 1 session** unit.)

**Backend:** Julia (second quantization) / **Domain:** spin systems (spin_systems).

### Physical background

The most fundamental model of magnetism. A 2-site spin-1/2 chain admits exact
diagonalization. High real-world relevance (magnetic resonance, quantum information,
condensed matter physics) and educationally important.

$$H = J\, \mathbf{S}_1 \cdot \mathbf{S}_2 = J\left( S_1^z S_2^z + \frac{1}{2}\left( S_1^+ S_2^- + S_1^- S_2^+ \right) \right)$$

where $J > 0$: antiferromagnetic coupling, $J < 0$: ferromagnetic coupling.

Exact energy eigenvalues:
- singlet $|S=0\rangle$: $E = -3J/4$
- triplet $|S=1\rangle$: $E = +J/4$ (3-fold degenerate)

Physics to verify:
1. $[H, S_z^{\rm tot}] = 0$ (total z-spin is conserved)
2. $[H, S_z(1)]$ is nonzero (an individual site is not conserved)
3. Confirming the equation of motion via the Heisenberg EOM

---

### Steps

#### HB-1: start the session

| Item | Content |
|---|---|
| Action | New session, domain: "spin systems" or "condensed matter physics", backend: Julia |
| Result | |

---

#### HB-2: declare the scalar parameter and the operator

| Item | Content |
|---|---|
| Input | `Define J as a real parameter` |
| Input | `S as the spin operator` |
| Expected | `@variables J::Real`, `@tlsxyz_ops S` are issued |
| Check | `Sx`, `Sy`, `Sz` become usable (`Sp`, `Sm` are not auto-defined) |
| Result | |

---

#### HB-2b: manually define $S^+$/$S^-$

`Sp`/`Sm` are not auto-defined, to avoid a bug from mixing `@tlsxyz_ops` and
`@tlspm_ops`. When $S^\pm$ is needed, register it as a linear combination of $S^x, S^y$.

**Formulas to enter:**
```
Sp_1 = Sx_1 + i * Sy_1
Sm_1 = Sx_1 - i * Sy_1
Sp_2 = Sx_2 + i * Sy_2
Sm_2 = Sx_2 - i * Sy_2
```

| Item | Content |
|---|---|
| Input | register the above in sequence (E1–E4) |
| Check | each formula displays as $S_1^x + iS_1^y$ etc. |
| Result | |

---

#### HB-3: enter the Hamiltonian

**The canonical notation to enter (written directly in Sx/Sy/Sz):**
```
H = J * Sz_1 * Sz_2 + J/4 * (Sx_1 * Sx_2 + Sy_1 * Sy_2)
```
This is $J/4 (S_1^x S_2^x + S_1^y S_2^y) = J/2 (S_1^+ S_2^- + S_1^- S_2^+)$ written in
the QA convention.

> **Note:** an expression ID (E1, E4, etc.) cannot be embedded inside a formula in the
> math input field. Expression IDs can only be used in operation instructions such as
> `[E1, E2]` or "differentiate E1".

| Item | Content |
|---|---|
| Input | register the above in the math input field (E5) |
| Check | displays as $J S_1^z S_2^z + J/4(S_1^x S_2^x + S_1^y S_2^y)$ |
| Result | |

---

#### HB-4: confirm the total-spin conservation law

**After registering E6 = Sz(1) + Sz(2) (total z-spin):**

| Item | Content |
|---|---|
| Input | `[H, Sztot]` (the commutator of H and $S_z^{\rm tot}$, referenced by name — the bracket notation resolves named symbols via `lhs_alias_map`, not just `E`-ids) |
| Expected result | $0$ (total spin is a conserved quantity) |
| Check | the result is 0 (correctly obtained using only `@tlsxyz_ops`) |
| Result | |

---

#### HB-5: individual-site spin (not conserved)

| Item | Content |
|---|---|
| Input | register `Sz(1)` (E7) |
| Input | `[E5, E7]` (the commutator of H and site 1's $S^z$) |
| Expected result | nonzero (an expression involving $S_1^x, S_1^y, S_2^x, S_2^y$) |
| Physical meaning | an individual site's spin is not conserved (spin-flip terms exist) |
| Result | |

---

#### HB-6: the Heisenberg equation (equation of motion)

Since `Sp(1)` is undefined, compute the Heisenberg equation for $S^+_1$ (E1),
registered in HB-2b.

| Item | Content |
|---|---|
| Input | `Heisenberg equation of E5 for Sx_1` |
| Expected result | an expression involving products with site-2 operators |
| Check | a term proportional to $J(S_1^y S_2^z - S_1^z S_2^y)$ is present |
| Result | |

---

#### HB-7: discussion via the LLM Adviser

| Item | Content |
|---|---|
| Action | select HB-4's result ($[H, S_z^{\rm tot}] = 0$) and click "💡 Discuss" |
| Expected | a physical explanation such as "total-spin conservation stems from SU(2) symmetry" |
| Result | |

---

#### HB-8: add a note

| Item | Content |
|---|---|
| Action | enter the following via "📝 Note": |
| Content | `Exact solution of the 2-site model: the singlet (S=0) has E = -3J/4, the triplet (S=1) has E = +J/4. For J>0, the ground state is the singlet (antiferromagnetic).` |
| Result | |

---

#### HB-9: LaTeX export / verification

| Item | Content |
|---|---|
| Action | the "⬇ LaTeX" button |
| Check | HB-3–HB-6's computed results are in math environments |
| Check | HB-8's note is in a plain-text paragraph |
| Result | |

---

### Verification checklist (scenario 2)

- [x] the `J` parameter and `Sz`, `Sx`, `Sy` become usable (`Sp`, `Sm` defined manually)
- [x] $S^+, S^-$ can be registered as formulas $S^x \pm iS^y$
- [x] $[H, S_z^{\rm tot}] = 0$ is obtained (confirming the conservation law)
- [x] $[H, S_z(1)] \neq 0$ is obtained (individual-site non-conservation)
- [x] the site-coupling term comes out correctly in the Heisenberg equation
- [ ] the LLM Adviser responds in the context of SU(2) symmetry
- [ ] the LaTeX output compiles

**Verified:** 2026-07-02 (real Julia backend, generator run, all 11 steps completed
end-to-end, same results as the Japanese version).

## generator commands (this `scenario` block is the single source used to generate the example)

```scenario domain=spin_systems
@note: Consider the two-site spin-1/2 Heisenberg model. Prepare the coupling constant $J$ ($J>0$ is antiferromagnetic coupling, $J<0$ is ferromagnetic coupling) and the spin operator $S$ used on both sites.
Define J as a real parameter
S as the spin operator
@note: Explicitly define the ladder operators $S^\pm=S^x\pm iS^y$ on sites 1 and 2 as linear combinations of the existing $S^x,S^y$.
Sp_1 = Sx_1 + i * Sy_1
Sm_1 = Sx_1 - i * Sy_1
Sp_2 = Sx_2 + i * Sy_2
Sm_2 = Sx_2 - i * Sy_2
@note: Register the Heisenberg Hamiltonian $H=J\mathbf S_1\cdot\mathbf S_2=JS_1^zS_2^z+\frac{J}{4}(S_1^xS_2^x+S_1^yS_2^y)$ (the second term is the $S^x,S^y$ form of $S_1^+S_2^-+S_1^-S_2^+$).
H = J * Sz_1 * Sz_2 + J/4 * (Sx_1 * Sx_2 + Sy_1 * Sy_2)
@note: Register the total $z$-spin $S_z^{\rm tot}=S_1^z+S_2^z$.
Sztot = Sz_1 + Sz_2
@note: Compute the commutator of the Hamiltonian $H$ and the total spin $S_z^{\rm tot}$, referencing both by the names just given them (the bracket notation resolves named symbols the same way it resolves E-ids). If it vanishes, the total $z$-spin is a conserved quantity (a consequence of SU(2) symmetry).
[H, Sztot]
@note: Individual site spins need not be conserved. Compute the Heisenberg equation of motion for $S^z$ on site 1 and check that a nonzero coupling term appears.
Heisenberg equation of E5 for Sz_1
@note: Likewise, compute the Heisenberg equation of motion for $S^x$ on site 1.
Heisenberg equation of E5 for Sx_1
```
