# Practice-loop sample scenario: BCS Hamiltonian and the Bogoliubov transform (Julia backend verification)

> (2026-07-02: an English translation of the Japanese scenario, verified against the
> real Julia backend through the SAME rule-based, no-LLM generator pipeline. See
> [scenario_bcs_hamiltonian_jp.md](scenario_bcs_hamiltonian_jp.md) for the Japanese
> version and the shared "確認された挙動・注意点" bug notes — several of which were
> found while building THIS English translation, not the Japanese original.)

**Backend:** Julia (QuantumAlgebra.jl) / **Domain:** condensed matter physics
(condensed_matter).

### Physical background

Mean-field BCS Hamiltonian (momentum $k$, spin $\uparrow(u)/\downarrow(d)$):

$$H_{\rm BCS} = \xi_k\left(c^\dagger_{k\uparrow} c_{k\uparrow} + c^\dagger_{-k\downarrow} c_{-k\downarrow}\right)
+ \Delta\left(c^\dagger_{k\uparrow} c^\dagger_{-k\downarrow} + c_{-k\downarrow} c_{k\uparrow}\right)$$

The Bogoliubov forward transform is defined as the full Nambu-spinor set (all four
creation/annihilation combinations):

$$g_{k\uparrow} = u\,c_{k\uparrow} - v\,c^\dagger_{-k\downarrow}, \qquad
g_{-k\downarrow} = u\,c_{-k\downarrow} + v\,c^\dagger_{k\uparrow}$$

$$g^\dagger_{k\uparrow} = u\,c^\dagger_{k\uparrow} - v\,c_{-k\downarrow}, \qquad
g^\dagger_{-k\downarrow} = u\,c^\dagger_{-k\downarrow} + v\,c_{k\uparrow}$$

Inverting these four for $c_{k\uparrow},c_{-k\downarrow},c^\dagger_{k\uparrow},c^\dagger_{-k\downarrow}$
(general operation `build_inverse_transform`) gives:

$$c_{k\uparrow} = \frac{v\,g^\dagger_{-k\downarrow} + u\,g_{k\uparrow}}{u^2+v^2}, \qquad
c_{-k\downarrow} = \frac{u\,g_{-k\downarrow} - v\,g^\dagger_{k\uparrow}}{u^2+v^2}$$

Substituting these into $H_{\rm BCS}$, the pair-creation term $g^\dagger_{-k\downarrow} g^\dagger_{k\uparrow}$
gets the coefficient

$$-\frac{\Delta(u^2-v^2) + 2uv\xi_k}{(u^2+v^2)^2}$$

Setting this to zero (the Bogoliubov condition, under $u^2+v^2=1$) —
$\Delta(u^2-v^2) = -2uv\xi_k$ — diagonalizes $H_{\rm BCS}$ into pure quasiparticle-number
terms $g^\dagger g$.

---

### Building it from general operations (★ the point of this scenario)

1. $c$ (electron) is declared via "c as the annihilation operator for electron" (the
   particle-name dictionary in `models/particles.json` resolves "electron" to a
   fermion).
2. $g$ (the Bogoliubov quasiparticle) is not a specific particle species, so it is
   declared via the bare form "g as a fermion operator".
3. $H_{\rm BCS}$ is registered in H-1 equation form directly, using the 2-argument
   operator $c(k,\sigma)$ ($\sigma=u/d$).
4. The Heisenberg equation is obtained via "Heisenberg equation of $E_n$ for $c(k,u)$".
5. The Bogoliubov forward transform is registered as all four Nambu-spinor components
   in H-1 equation form (no automatic adjoint generation by design — see
   [doc 18](../developer_manuals/18_linear_transform_design.md) §4).
6. "Invert $E_a,E_b,E_c,E_d$ for $c(k,u),c(-k,d),c'(k,u),c'(-k,d)$"
   (`build_inverse_transform`, L-1) gives the old basis ($c$ side) in terms of the new
   basis ($g$ side).
7. The resulting $c(k,u)=\dots$ / $c(-k,d)=\dots$ are substituted into $H_{\rm BCS}$ by
   E-id reference: "In $E_1$, substitute $E_a$ for $c(k,u)$ and $E_b$ for $c(-k,d)$"
   (`operator_substitute`'s `op_rules`; the $c^\dagger$ side is handled automatically
   as the adjoint, so it does not need a separate rule).

---

### Steps

#### ST-1: Start the session

| Item | Content |
|---|---|
| Action | New session, domain: "condensed matter physics" → the Julia backend is auto-selected |
| Result | |

---

#### ST-2: Declare the operators and parameters

| Item | Content |
|---|---|
| Input | `c as the annihilation operator for electron` |
| Input | `g as a fermion operator` |
| Input | `Define u, v as scalars` |
| Input | `Define xi, Delta as scalars` |
| Check | Both `c` and `g` are declared via `@fermion_ops` (a name absent from the particle dictionary uses the bare fermion form) |
| Result | |

---

#### ST-3: Register the mean-field BCS Hamiltonian (H-1)

| Item | Content |
|---|---|
| Input | `H_BCS = xi * (c'(k,u) c(k,u) + c'(-k,d) c(-k,d)) + Delta * (c'(k,u) c'(-k,d) + c(-k,d) c(k,u))` |
| Check | `c'(k,u)` is recognized as the creation operator; the `-k` argument is handled correctly as a site (momentum) label |
| Result | |

---

#### ST-4: Heisenberg equation of motion (optional)

| Item | Content |
|---|---|
| Input | `Heisenberg equation of E1 for c(k,u)` |
| Expected | $\dfrac{dc_{k\uparrow}}{dt} = -i\xi_k\, c_{k\uparrow} - i\Delta\, c^\dagger_{-k\downarrow}$ |
| Result | |

---

#### ST-5: Register the Bogoliubov forward transform (full 4-component Nambu spinor, L-1 step ①)

| Item | Content |
|---|---|
| Input | `g(k,u) = u * c(k,u) - v * c'(-k,d)` |
| Input | `g(-k,d) = u * c(-k,d) + v * c'(k,u)` |
| Input | `g'(k,u) = u * c'(k,u) - v * c(-k,d)` |
| Input | `g'(-k,d) = u * c'(-k,d) + v * c(k,u)` |
| Check | All four are registered via H-1 and get an E-id |
| Result | |

---

#### ST-6: Generate the inverse transform (L-1 step ②)

| Item | Content |
|---|---|
| Input | `Invert E3, E4, E5, E6 for c(k,u), c(-k,d), c'(k,u), c'(-k,d)` (`build_inverse_transform`; the E-ids depend on how many optional steps were run — omitting ST-4 shifts everything down by one) |
| Expected | $c_{k\uparrow} = \dfrac{v g^\dagger_{-k\downarrow} + u g_{k\uparrow}}{u^2+v^2}$, $c_{-k\downarrow} = \dfrac{u g_{-k\downarrow} - v g^\dagger_{k\uparrow}}{u^2+v^2}$, and two more |
| Check | The inverse transform is solved correctly even with 2-argument operators (`c(k,u)` etc.) |
| Result | |

---

#### ST-7: Substitute into the Hamiltonian (L-1 step ③)

| Item | Content |
|---|---|
| Input | `In E1, substitute E7 for c(k,u) and E8 for c(-k,d)` (`E7`/`E8` are the E-ids of $c(k,u)=\dots$/$c(-k,d)=\dots$ from ST-6; actual numbers depend on session progress) |
| Check | The simultaneous operator double-substitution (`op_rules`) applies correctly via E-id reference, and the result is normal-ordered |
| Expected | Diagonal $g^\dagger g$ terms, plus a pair-creation term $g^\dagger_{-k\downarrow} g^\dagger_{k\uparrow}$ with coefficient $-\dfrac{\Delta(u^2-v^2)+2uv\xi_k}{(u^2+v^2)^2}$ |
| Result | |

---

### Verification checklist

- [x] `c` (electron) and `g` (quasiparticle, bare fermion form) are both correctly declared fermionic
- [x] H-1 registration and display of the 2-argument operator $c(k,\sigma)$ is correct
- [x] The Heisenberg equation returns the correct leading BCS terms
- [x] Registering the full 4-component Nambu-spinor forward transform → `build_inverse_transform` solves correctly even with 2-argument operators
- [x] Substituting the inverse-transform result into the Hamiltonian by E-id reference (`operator_substitute`'s `op_rules`) gives the pair-creation coefficient $-(\Delta(u^2-v^2)+2uv\xi_k)/(u^2+v^2)^2$ (the starting point of the Bogoliubov condition)

---

## Observed behaviour / notes

Building this English translation surfaced **five more root-cause bugs**, on top of
the four already documented in
[scenario_bcs_hamiltonian_jp.md](scenario_bcs_hamiltonian_jp.md) §確認された挙動・注意点
— every one of them an English-language gap in a rule-based (no-LLM) NL interceptor
that the Japanese generator run never exercised:

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 1 | `X as fermion operator` (no particle name) never matched | `_BARE_QUANTUM_OP_RE` (`julia_handler_common.py`) required the literal Japanese particle を even for the English alternative | Added an `as`/`is` connector alternative alongside を |
| 2 | `Invert E3, E4, E5, E6 for ...` routed to the single-matrix `inverse_matrix` operation instead of `build_inverse_transform` | `_parse_inverse_transform` (`interpret_l1.py`) was Japanese-only; the generic Phase-1 alias scan's `inverse_matrix` alias list (`invert`, `inverse`) intercepted first | Added English patterns to `_parse_inverse_transform`, checked before the generic matrix-ops alias scan |
| 3 | `Heisenberg equation of E1 for c(k,u)` fed `operator='c(k', hamiltonian='u)'` to the op, splitting the 2-argument operator on its internal comma | No dedicated rule-based Heisenberg-equation trigger existed in either language — it relied entirely on the generic alias-scan + positional-token-filler, which shreds a comma-containing token when the surrounding sentence has no Japanese particles to anchor tokenization | Added a dedicated rule-based Heisenberg-equation trigger (Japanese and English) in `extra_pre_phase1`, bypassing the generic filler entirely |
| 4 | `In E1, substitute E7 for c(k,u) and E8 for c(-k,d)` (double-clause operator substitution) produced `lhs='In', rhs='for'` | Same SUBST-DOUBLE-SHADOW class of bug as the Japanese case, but via `substitute_into`'s English pattern this time — its `lhs` group swallowed the second clause (`"c(k,u) and E8 for c(-k,d)"`), which the existing Japanese-only guard didn't detect | Added an English guard (`and ... for` inside lhs) plus a dedicated English spin-index double-substitution pattern (`_OP_SUBST_SPIN_RE_EN`) in `extra_pre_phase1` |
| 5 | `In E1, substitute mz for S^z_2` (single-site operator substitution, exercised more directly in the transverse-field-Ising scenario but from the same code path) routed to plain `substitute` (whole-operator identity) instead of the index-aware `operator_substitute` | `substitute_into`'s guard only checked for E-id/double-clause forms, not whether the LHS names a registered quantum operator | Extended the guard to bail when LHS starts with a registered operator-kind symbol name, and added a natural-English single-clause `operator_substitute` pattern (`_OP_SUBST_RE_EN2`) |

Also found along the way: the `instruction_marker` verb allowlist (used to distinguish
an NL instruction from a bare formula) was missing several verbs actually used across
these scenarios ("simplify", "set", "heisenberg", "commutator", "factor",
"rationalize"), causing those requests to be silently mis-registered as literal new
expressions instead of being recognized as operations.

See `docs/open_items.md` for the consolidated list of fixes (all changes are additive —
existing Japanese-only patterns are unchanged; English forms are tried as fallback
alternatives).

**Verified:** 2026-07-02 (real Julia backend, generator run, all 12 steps completed
end-to-end with the same physics as the Japanese version).

---

## generator commands (this `scenario` block is the single source used to generate the example)

```scenario domain=condensed_matter
@note: Prepare the operators and parameters needed to build the BCS Hamiltonian. $c$ is the annihilation operator for an electron with momentum $k$ and spin $\sigma$ ($u$ = up, $d$ = down). $g$ is the annihilation operator for the Bogoliubov quasiparticle introduced by the transformation below (declared as a generic fermion operator, separate from the electron, since it is not a named particle species). $u$ and $v$ are the coefficients of the Bogoliubov transformation (a condition such as $u^2+v^2=1$ is imposed later). $\xi_k$ (xi) is the electron kinetic energy measured from the chemical potential, and $\Delta$ (Delta) is the pair potential (the superconducting gap) that binds Cooper pairs.
c as the annihilation operator for electron
g as a fermion operator
Define u, v as scalars
Define xi, Delta as scalars
@note: Register the mean-field BCS Hamiltonian. The first term xi*(c'(k,u) c(k,u) + c'(-k,d) c(-k,d)) is the electron kinetic energy; the second term Delta*(c'(k,u) c'(-k,d) + c(-k,d) c(k,u)) is the pair potential describing the creation and annihilation of Cooper pairs.
H_BCS = xi * (c'(k,u) c(k,u) + c'(-k,d) c(-k,d)) + Delta * (c'(k,u) c'(-k,d) + c(-k,d) c(k,u))
@note: Compute the Heisenberg equation of motion dc/dt = i[H, c] for c(k,u), to see how it is coupled to H_BCS.
Heisenberg equation of E1 for c(k,u)
@note: Define the Bogoliubov transformation (the change of basis to quasiparticle operators) as all four components of the Nambu-spinor set of electron operators and their adjoints together (the dagger is not auto-generated, so the creation-operator equations are written out explicitly).
g(k,u) = u * c(k,u) - v * c'(-k,d)
g(-k,d) = u * c(-k,d) + v * c'(k,u)
g'(k,u) = u * c'(k,u) - v * c(-k,d)
g'(-k,d) = u * c'(-k,d) + v * c(k,u)
@note: Solve the four transformation equations just defined for the electron operators c(k,u), c(-k,d), c'(k,u), c'(-k,d) (the inverse transform), expressing them in terms of the quasiparticle operators g.
Invert E3, E4, E5, E6 for c(k,u), c(-k,d), c'(k,u), c'(-k,d)
@note: Substitute the inverse-transform results into the BCS Hamiltonian to obtain it in terms of the quasiparticle operators g. Choosing u, v so that the coefficient of the pair-creation term vanishes (the Bogoliubov condition) brings the Hamiltonian to diagonal form, containing only the quasiparticle number operator g'g.
In E1, substitute E7 for c(k,u) and E8 for c(-k,d)
```
