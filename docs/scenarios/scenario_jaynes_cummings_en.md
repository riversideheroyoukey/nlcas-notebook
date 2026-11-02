# Practice-loop sample scenario: the Jaynes-Cummings model (photon and 2-level molecule interaction)

> (2026-07-02: an English translation of the Japanese scenario, verified against the
> real Julia backend through the same rule-based, no-LLM generator pipeline. See
> [scenario_jaynes_cummings_jp.md](scenario_jaynes_cummings_jp.md) for the Japanese
> version. No new i18n gaps were found here — all the English NL patterns this
> scenario needs (real-parameter declaration, particle-name operator declaration,
> spin operator declaration, `heisenberg_equation`, bracket-notation commutator) were
> already fixed while building the earlier translations.)

> (Split off from the former "E-3 research-workflow verification scenario" into its own
> **1 scenario = 1 session** unit.)

**Backend:** Julia (second quantization) / **Domain:** quantum optics
(quantum_optics).

### Physical background

The interaction between a molecule (or a 2-level atom) and a single photon mode. A
basic model in quantum optics and molecular spectroscopy.

$$H_{\rm JC} = \omega\, a^\dagger a + \omega_0\, S_z + g\left( a\, S^+ + a^\dagger\, S^- \right)$$

where:
- $a$, $a^\dagger$: the photon's annihilation/creation operators (bosonic)
- $S_z$, $S^\pm = S_x \pm iS_y$: the molecule's 2-level operators (spin-1/2)
- $\omega$: photon frequency, $\omega_0$: molecular transition frequency, $g$:
  light-matter coupling constant

The equations of motion from the Heisenberg equation:

$$\dot{a} = -i\omega\, a - ig\, S^-$$
$$\dot{S}^- = -i\omega_0\, S^- + 2ig\, S_z\, a$$

At resonance $\omega = \omega_0$, Rabi oscillations appear.

---

### Steps

#### JC-1: start the session

| Item | Content |
|---|---|
| Action | New session, domain: "quantum optics", backend: Julia |
| Check | the Julia backend is running (status display) |
| Result | |

---

#### JC-2: declare the scalar parameters

| Item | Content |
|---|---|
| Input | `Define omega as a real parameter` |
| Input | `Define \omega_0 as a real parameter` |
| Input | `Define g as a real parameter` |
| Expected | `@variables omega::Real` etc. are sent to Julia |
| Check | registered as scalars in SymbolContext |
| Result | |

---

#### JC-3: declare the operators

| Item | Content |
|---|---|
| Input | `a as the creation operator for photon` (particle-name form) |
| Input | `S as the spin operator` |
| Expected | `@boson_ops a`, `@tlsxyz_ops S; @tlspm_ops S` are issued |
| Check | `a`, `S`, `Sz`, `Sp`, `Sm` are registered in SymbolContext |
| Result | |

---

#### JC-4: enter the Hamiltonian

**The canonical notation to enter:**
```
omega * a'(1)*a(1) + \omega_0 * Sz(1) + g*(a(1)*Sp(1) + a'(1)*Sm(1))
```

| Item | Content |
|---|---|
| Input | enter the above canonical form in the math input field to register it |
| Check | registered as E1, displayed correctly via KaTeX |
| Check | displays as $\omega a^\dagger a + \omega_0 S_z + g(aS^+ + a^\dagger S^-)$ |
| Result | |

---

#### JC-5: the photon's Heisenberg equation

| Item | Content |
|---|---|
| Input | `Heisenberg equation of E1 for a(1)` |
| Expected result | $-i\omega\, a(1) - ig\, S^-(1)$ |
| Check | `Sm(1)` or $S^-$ is present |
| Check | the coefficients `omega` ($\omega$) and `g` are correct |
| Result | |

---

#### JC-6: the molecule's ($S^-$) Heisenberg equation

| Item | Content |
|---|---|
| Input | `Heisenberg equation of E1 for Sm(1)` |
| Expected result | $-i\omega_0\, S^-(1) + 2ig\, S_z(1)\, a(1)$ |
| Check | the product `Sz(1)` and `a(1)` is present |
| Result | |

**Physical interpretation:** the equation for $S^-$ has the product $S_z \cdot a$ — this
is the Jaynes-Cummings model's nonlinearity, the origin of photon-number-dependent
Rabi oscillations.

---

#### JC-7: discussion via the LLM Adviser

| Item | Content |
|---|---|
| Action | select JC-5's result (the photon's equation of motion) and click "💡 Discuss" |
| Expected | the LLM explains the physical meaning (photon decay, stimulated emission by the molecule) |
| Check | no CAS syntax leaks into the output |
| Result | |

---

#### JC-8: add a note

| Item | Content |
|---|---|
| Action | enter the following via "📝 Note": |
| Content | `At resonance ω = ω₀, Rabi oscillations occur under the rotating-wave approximation (RWA). The Rabi frequency is ΩR = g√(n+1) (n = photon number)` |
| Check | the note is displayed on the timeline |
| Result | |

---

#### JC-9: confirm the total-excitation conservation law

| Item | Content |
|---|---|
| Input | register `Sz(1) + a'(1)*a(1)` (the sum of the photon number and the spin z-component) |
| Input | `[E_H, E_excitation]` (the commutator of the excitation-number operator and H) |
| Expected | $[H, N_{\rm ex}] = 0$ (the excitation number $N_{\rm ex} = a^\dagger a + S_z + 1/2$ is conserved) |
| Result | |

---

#### JC-10: LaTeX export

| Item | Content |
|---|---|
| Action | click the "⬇ LaTeX" button |
| Check | formulas are output inside `\begin{equation}` environments |
| Check | no internal macros such as `\instruction{}` or `\operation{}` leak through |
| Check | JC-8's note is inserted as a plain-text paragraph between the formulas |
| Check | the output actually compiles with pdflatex |
| Result | |

---

### Verification checklist (scenario 1)

- [x] natural-language declaration of Greek-letter parameters (ω, ω₀, g) works
- [x] a mixed boson + spin Hamiltonian is entered and displayed correctly
- [x] the Heisenberg equation results match the known results
- [ ] the LLM Adviser returns a meaningful discussion in the physical context
- [ ] the note is retained on the timeline and included in the LaTeX
- [ ] the LaTeX is output in a compilable form

**Verified:** 2026-07-02 (real Julia backend, generator run, all 10 steps completed
end-to-end, same results as the Japanese version: $[H, N_{\rm ex}] = 0$).

## generator commands (this `scenario` block is the single source used to generate the example)

```scenario domain=quantum_optics
@note: Consider the Jaynes-Cummings model (a photon interacting with a two-level molecule). Prepare the photon frequency $\omega$, the molecular transition frequency $\omega_0$, and the light-matter coupling constant $g$ as real parameters.
Define omega as a real parameter
Define \omega_0 as a real parameter
Define g as a real parameter
@note: Prepare the photon annihilation/creation operators $a,a^\dagger$ as boson operators, and the molecule's two-level (spin-1/2) operator $S$ as a spin operator.
a as the creation operator for photon
S as the spin operator
@note: Register the Jaynes-Cummings Hamiltonian $H=\omega a^\dagger a+\omega_0S_z+g(aS^++a^\dagger S^-)$, entered with $S^\pm=S_x\pm iS_y$ expanded out. The first term is the photon energy, the second the molecular excitation energy, and the third the light-matter interaction from photon absorption/emission.
omega * a'(1)*a(1) + \omega_0 * Sz(1) + g*(a(1)*(Sx(1) + i*Sy(1)) + a'(1)*(Sx(1) - i*Sy(1)))
@note: Compute the Heisenberg equation of motion for the photon operator $a$.
Heisenberg equation of E1 for a(1)
@note: Compute the Heisenberg equation of motion for the molecular operator $S_x$.
Heisenberg equation of E1 for Sx(1)
@note: Define the excitation-number operator $N_{\rm ex}=a^\dagger a+S_z/2$ (the sum of the photon number and the molecule's spin $z$-component).
a'(1)*a(1) + Sz(1)/2
@note: Compute the commutator of the Hamiltonian $H$ and the excitation-number operator $N_{\rm ex}$. Physically, each of $H$'s three terms individually conserves the total excitation number: the photon-energy and molecule-energy terms are diagonal in the excitation-number basis, and the interaction term $g(aS^++a^\dagger S^-)$ only ever converts one photon into one molecular excitation or vice versa (photon absorption/emission), never changing $a^\dagger a+S_z$'s total. This is why the CAS-computed commutator collapses to exactly 0 rather than some residual expression — a genuinely computed cancellation, not a fixed answer. (The CAS applies operator identities and returns the final simplified value directly; it does not expose the intermediate term-by-term expansion as a separate visible step.)
[E1, E4]
```
