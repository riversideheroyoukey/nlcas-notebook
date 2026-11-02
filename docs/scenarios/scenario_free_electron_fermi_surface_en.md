# Practice-loop sample scenario: Fermi surface of the free-electron model (Julia backend verification)

> (2026-07-02: an English translation of the Japanese scenario, verified against the
> real Julia backend through the SAME rule-based, no-LLM generator pipeline. See
> [scenario_free_electron_fermi_surface_jp.md](scenario_free_electron_fermi_surface_jp.md)
> for the Japanese version and the shared "確認された挙動・注意点" bug notes.)

**Backend:** Julia (Symbolics.jl scalar algebra) / **Domain:** condensed matter physics
(condensed_matter).

### Physical background

Free-electron energy dispersion (measured from the chemical potential $\mu$):

$$\xi_k = \frac{\hbar^2 k^2}{2m} - \mu$$

Solving the Fermi-surface condition $\xi_k = 0$ gives the Fermi wavevector:

$$k_F = \frac{\sqrt{2m\mu}}{\hbar}$$

Substituting this back into $\xi_k$ identically gives $\xi_{k_F} = 0$.

The effective mass $m^*$ follows from the second derivative of the dispersion:

$$\frac{1}{m^*} = \frac{1}{\hbar^2} \frac{d^2 \xi_k}{dk^2} = \frac{1}{m}$$

(for free electrons, $m^* = m$ — the effective mass equals the bare mass)

---

### Building it from general operations (★ the point of this scenario)

No CAS syntax is written — the whole derivation uses only general operations (H-1
registration, H-5z "set the right-hand side", H-9 solve, substitute, differentiate),
with the physical-background TeX notation typed as-is:

1. $\xi_k$ is registered in H-1 equation form (`\xi_k = \hbar^2 * k^2 / (2 * m) - \mu`).
2. The Fermi-surface condition comes from "Set the right-hand side of E1 to 0" (H-5z),
   producing the new equation $\xi_k=0$.
3. "Solve E<i>n</i> for k" (H-9) gives the Fermi wavevector. Julia's `solve_for` is
   linear-only, so this equation — quadratic in $k$ — automatically falls back to
   SymPy.solve.
4. "Substitute E<i>n</i> into E1" plugs $k=k_F$ back into $\xi_k$; "Simplify the
   right-hand side of E<i>m</i>" confirms it collapses to 0 identically.
5. "Differentiate E1 twice with respect to k" gives the second derivative, from which
   $m^*=\hbar^2/(d^2\xi_k/dk^2)$ is registered as a new expression.

---

### Steps

#### ST-1: Start the session

| Item | Content |
|---|---|
| Action | New session, domain: "condensed matter physics" → the Julia backend is auto-selected |
| Result | |

---

#### ST-2: Declare the symbols

| Item | Content |
|---|---|
| Input | `Define k, m, \mu, \hbar as scalars` |
| Check | All 4 symbols are registered as scalars on the Julia backend |
| Result | |

---

#### ST-3: Register the free-electron energy (H-1)

| Item | Content |
|---|---|
| Input | `\xi_k = \hbar^2 * k^2 / (2 * m) - \mu` |
| Check | Recognized as H-1 form (LHS = RHS); `xi_k` is registered in `lhs_alias_map` |
| Expected | $\xi_k = \dfrac{\hbar^2 k^2}{2m} - \mu$ |
| Result | |

---

#### ST-4: Simplify the right-hand side (optional)

| Item | Content |
|---|---|
| Input | `Simplify the right-hand side of E1` |
| Expected | $\xi_k = \dfrac{-2m\mu + k^2\hbar^2}{2m}$ (simplified form, physically unchanged) |
| Result | |

---

#### ST-5: Register the Fermi-surface condition (H-5z check)

| Item | Content |
|---|---|
| Input | `Set the right-hand side of E1 to 0` |
| Check | Handled rule-based as H-5z (no LLM call); a new equation is registered |
| Expected | $\dfrac{\hbar^2 k^2}{2m} - \mu = 0$ |
| Result | |

---

#### ST-6: Solve for the Fermi wavevector (H-9 Julia solve check)

| Item | Content |
|---|---|
| Input | `Solve E3 for k` |
| Check | Since the equation is quadratic in $k$, `Symbolics.solve_for` (linear-only) fails and automatically falls back to SymPy.solve |
| Expected | $k = \dfrac{\sqrt{2m\mu}}{\hbar}$ |
| Result | |

---

#### ST-7: Confirm the Fermi surface by substitution

| Item | Content |
|---|---|
| Input | `Substitute E4 into E1` |
| Input | `Simplify the right-hand side of E5` |
| Check | The solve result's canonical key `k` is picked up from `lhs_alias_map`; after substitution, simplifying the RHS collapses it to 0 identically |
| Expected | $\xi_{k_F} = 0$ |
| Result | |

---

#### ST-8: Compute the effective mass (2nd derivative + H-1)

| Item | Content |
|---|---|
| Input | `Differentiate E1 twice with respect to k` |
| Input | `m_star = \hbar^2 / E7` |
| Check | The Julia second derivative is computed correctly; the H-1 registration expands the E7 reference |
| Expected | $\dfrac{d^2\xi_k}{dk^2} = \dfrac{\hbar^2}{m}$, then $m^* = m$ |
| Result | |

---

### Verification checklist

- [x] Bulk scalar declaration works on the Julia backend
- [x] H-1 registration (`\xi_k = ...`) displays correctly under Julia
- [x] H-5z ("set the right-hand side to 0") is handled rule-based
- [x] H-9 solve falls back to SymPy for the quadratic and returns the correct root
- [x] Substituting the solve result gives $\xi_{k_F}=0$ identically
- [x] The second derivative gives the effective mass $m^*=m$

---

## Observed behaviour / notes

See [scenario_free_electron_fermi_surface_jp.md](scenario_free_electron_fermi_surface_jp.md)
§確認された挙動・注意点 for the root-cause bug found and fixed while building this
scenario (`evaluate_formula_with_latex`/`_cas_value_and_latex` unconditionally
re-issuing a Maxima-only `tex1(...)` call). That fix is backend-generic (keyed off
whether `result.latex` is already populated), so it applies identically here.

This English translation additionally exercised — and fixed — several English-language
gaps in the rule-based (no-LLM) NL interceptors that the Japanese version's generator run
had never touched: the `instruction_marker` verb allowlist was missing "simplify"/"set"
(so those requests were silently mis-registered as literal new expressions instead of
being recognized as operations); H-5z, H-9 solve-for, RHS-focused operations
(expand/simplify/factor), the "En を Em に代入して" formula-substitution shortcut, and
differentiation-order counting ("twice"/"N times") had no English pattern at all. See
`docs/open_items.md` for the consolidated list of fixes.

**Verified:** 2026-07-02 (real Julia backend, generator run, all 9 steps completed
end-to-end with the same physics as the Japanese version).

---

## generator commands (this `scenario` block is the single source used to generate the example)

```scenario domain=condensed_matter
@note: Consider the free-electron-model dispersion. Prepare the wavevector $k$, electron mass $m$, chemical potential $\mu$, and reduced Planck constant $\hbar$ as scalar parameters.
Define k, m, \mu, \hbar as scalars
@note: Define the free-electron energy dispersion $\xi_k=\hbar^2k^2/(2m)-\mu$ (energy measured relative to the chemical potential).
\xi_k = \hbar^2 * k^2 / (2 * m) - \mu
@note: Simplify the right-hand side into a tidier form (no physical change).
Simplify the right-hand side of E1
@note: Fix the Fermi-surface condition $\xi_k=0$ (the energy coincides with the Fermi level) as an equation, by setting the right-hand side to 0.
Set the right-hand side of E1 to 0
@note: Solve this equation for the wavevector $k$ to obtain the Fermi wavevector $k_F$.
Solve E3 for k
@note: Substitute the Fermi wavevector just found back into the original expression for $\xi_k$, to confirm that the energy is identically zero on the Fermi surface.
Substitute E4 into E1
@note: Simplify the substitution result to make the zero explicit.
Simplify the right-hand side of E5
@note: To find the effective mass $m^*$, take the second derivative of the dispersion $\xi_k$ with respect to the wavevector $k$.
Differentiate E1 twice with respect to k
@note: From the second derivative just obtained, define the effective mass $m^*=\hbar^2/(d^2\xi_k/dk^2)$ (for the free-electron model this should equal $m$, the bare mass).
m_star = \hbar^2 / E7
```
