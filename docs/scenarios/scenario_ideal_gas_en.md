# Practice-loop sample scenario: statistical mechanics of the ideal gas (classical statistics)

> (2026-07-03: an English translation of the Japanese scenario, verified against the
> real Maxima backend through the same rule-based, no-LLM generator pipeline. See
> [scenario_ideal_gas_jp.md](scenario_ideal_gas_jp.md) for the Japanese version. No new
> i18n gaps were found — all the English NL patterns this scenario needs (multi-symbol
> parameter declaration, `\log`/`\sqrt`/`N!` TeX-literal input, "expand En", "take the
> partial derivative of En with respect to X", "simplify the right-hand side of En")
> were already fixed while building the earlier translations.)

> (Split off from the former "E-3 research-workflow verification scenario" into its own
> **1 scenario = 1 session** unit. 2026-07-02: the old version, which exposed CAS syntax
> (`diff(F,V)`, `%pi`, etc.), was rebuilt using only general operations — parameter
> declaration, expression registration, partial differentiation, expand, simplify.
> Two root-cause bugs were also fixed so that TeX notation exactly as written —
> `k_B` (a subscripted symbol), `\lambda`, `N!` — runs end-to-end with no errors; see
> "Observed behaviour / notes" below.)

**Backend:** Maxima / **Domain:** statistical mechanics (statistical_mechanics).

### Physical background

A monatomic ideal gas of $N$ particles. Thermodynamic quantities are derived from the
classical-statistics partition function.

$$Z_1 = \frac{V}{\lambda^3}, \quad \lambda = \sqrt{\frac{h^2}{2\pi m k_B T}} \quad \text{(de Broglie thermal wavelength)}$$

$$Z_N = \frac{Z_1^N}{N!} \quad \text{(indistinguishable particles, the Gibbs correction)}$$

$$F = -k_B T \ln Z_N \quad \text{(Helmholtz free energy)}$$

Deriving the thermodynamic quantities:
$$P = -\left(\frac{\partial F}{\partial V}\right)_T = \frac{Nk_BT}{V} \quad \text{(the ideal-gas equation of state)}$$

$$U = -T^2\frac{\partial}{\partial T}\!\left(\frac{F}{T}\right) = \frac{3}{2}Nk_BT \quad \text{(internal energy, the Gibbs–Helmholtz relation)}$$

$$C_V = \left(\frac{\partial U}{\partial T}\right)_V = \frac{3}{2}Nk_B \quad \text{(heat capacity at constant volume)}$$

**Note:** in deriving $P,U,C_V$, the $V,T$-dependence of $N!$ is zero, so it drops out
automatically during differentiation. The Stirling approximation ($\ln N!\approx N\ln
N-N$) is only needed when directly computing the entropy $S=-(\partial F/\partial T)_V$,
which is not included in this scenario's checklist.

---

### How this is built from general operations (★ the heart of this scenario)

Without writing any CAS syntax, using only H-1 equation registration, E-id reference
chains, and the general `differentiate` operation, **the TeX notation from the physical
background above is entered verbatim**:

1. $\lambda,Z_1,Z_N,F$ are registered in sequence using H-1's equation form (e.g.
   `\lambda = \sqrt{h^2 / (2 * \pi * m * k_B * T)}`). Each expression can directly
   reference the E-id of an earlier one (`Z_1 = V / E1^3`'s `E1` is the definition of
   $\lambda$), so Maxima automatically binds it as a symbol definition (the H-1 P2
   mechanism). Later expressions automatically expand the earlier values.
2. Use the TeX command `\pi`, not the Unicode character, for $\pi$ (project convention).
3. $P=-\partial F/\partial V$ is entered as "take the partial derivative of $F$ with
   respect to V" → register the sign-flipped result as a new expression `P = -E6` (no
   need to write the Maxima syntax `diff(F,V)`).
4. $U=-T^2\,\partial(F/T)/\partial T$ is built by first creating $F/T$ as a new
   expression via `FoverT = F / T`, then taking its partial derivative with respect to
   T, then registering the result multiplied by $-T^2$.

---

### Steps

#### ST-1: start the session

| Item | Content |
|---|---|
| Action | New session, domain: "statistical mechanics" or unspecified, backend: Maxima |
| Result | |

---

#### ST-2: declare the parameters

| Item | Content |
|---|---|
| Input | `Define N, V, T, m, h, k_B as positive real parameters` |
| Check | multiple parameters are registered in SymbolContext at once |
| Check | the subscript `B` of `k_B` triggers the disambiguation dialog (choose "part of the name") |
| Result | |

---

#### ST-3: define the thermal de Broglie wavelength and partition function

| Item | Content |
|---|---|
| Input | `\lambda = \sqrt{h^2 / (2 * \pi * m * k_B * T)}` |
| Input | `Z_1 = V / E1^3` ($E1$ is $\lambda$) |
| Input | `Z_N = E2^N / N!` ($E2$ is $Z_1$) |
| Check | $\lambda$, $Z_1$, $Z_N$ display correctly via KaTeX ($N!$ is preserved) |
| Result | |

---

#### ST-4: the Helmholtz free energy

| Item | Content |
|---|---|
| Input | register `F = -k_B * T * \log(E3)` ($E3$ is $Z_N$) |
| Input | `Expand F` (the registered definitions of $Z_N,Z_1,\lambda$ are automatically substituted and expanded) |
| Expected result | $F$ becomes a closed-form expression in only $N,V,T,m,h,k_B$ (containing $N!$) |
| Result | |

---

#### ST-5: derive the equation of state

| Item | Content |
|---|---|
| Input | `Take the partial derivative of F with respect to V` |
| Input | `P = -E6` (register with the sign flipped) |
| Input | `Simplify the right-hand side of E7` |
| Expected result | $P = Nk_BT/V$ (the ideal-gas equation of state) |
| Result | |

---

#### ST-6: derive the internal energy

| Item | Content |
|---|---|
| Input | `FoverT = F / T` (register $F/T$ as a new expression) |
| Input | `Take the partial derivative of FoverT with respect to T` |
| Input | `U = -T^2 * E10` |
| Input | `Simplify the right-hand side of E11` |
| Expected result | $U = \frac{3}{2}Nk_BT$ |
| Result | |

---

#### ST-7: the heat capacity at constant volume

| Item | Content |
|---|---|
| Input | `Take the partial derivative of U with respect to T` |
| Expected result | $C_V = \frac{3}{2}Nk_B$ |
| Check | the correct value for a monatomic ideal gas is obtained |
| Result | |

---

#### ST-8: discussion via the LLM Adviser

| Item | Content |
|---|---|
| Action | select ST-5's result (the equation of state) and click "💡 Discuss" |
| Expected | an explanation of the relation between the ideal-gas assumptions (no intermolecular interaction) and the equation of state |
| Result | |

---

#### ST-9: LaTeX export

| Item | Content |
|---|---|
| Action | the "⬇ LaTeX" button |
| Check | $\ln$, $\partial$ are output with the correct LaTeX commands |
| Check | the output compiles with `pdflatex` |
| Result | |

---

### Verification checklist (scenario 4)

- [x] declaring multiple parameters at once (including `k_B`) works in natural language
- [x] registering and displaying expressions containing `\log`, `\sqrt`, `N!` is correct (TeX notation entered verbatim)
- [x] $-\partial F/\partial V = Nk_BT/V$ is obtained (the equation of state)
- [x] $U = 3Nk_BT/2$ is obtained (internal energy)
- [x] $C_V = 3Nk_B/2$ is obtained (heat capacity)
- [ ] the LLM Adviser returns a meaningful discussion in the thermodynamic context
- [ ] the LaTeX output compiles

---

## Observed behaviour / notes

- **A postfix `!` (factorial) was completely ignored by the SharedParser (a root-cause
  bug, now fixed).** The tokenizer regex `_TOKEN_RE` in
  `notebook/runtime/shared_parser.py` had no pattern for `!`, so `N!` was silently
  lexed as plain `N` (no error — the formula is **silently corrupted**, the hardest
  class of bug to notice). This corruption did not surface at "direct H-1
  registration" (at registration time the raw text is sent to Maxima almost as-is), but
  since **general operations** (`expand`, `differentiate`, etc.) re-parse the
  expression through the SharedParser before regenerating the CAS command, $N!$ — which
  displayed correctly right after computing $Z_N$ — turned into $N$ the moment `Expand
  F` was run afterward. Fix: added a `_T_BANG` token and made `parse_primary` convert a
  trailing `!` into `FunctionCall(factorial, [operand])` (`factorial` was already a
  known function registered in `MaximaSemanticResolver`, so the parser-side addition
  alone routes both prefix and postfix notation through the same path).
- **A single-letter subscript symbol (e.g. `k_B`) triggers the disambiguation dialog at
  declaration.** This is intended behaviour (see the Subscript Classification section
  of `CLAUDE.md`), but integration with non-interactive generator runs was missing.
  `scripts/generate_scenario_examples.py` was extended so that, on receiving
  `needs_confirmation`, it selects the "part of the name" candidate through the same
  path the real UI's confirm button uses (calling `engine.run_operation(candidate.operation,
  candidate.args)` directly). This avoids having to use a workaround name with the
  subscript stripped, like `kB`.

**Verified:** 2026-07-03 (real Maxima backend, generator run, all 14 steps completed
end-to-end, TeX notation `k_B`, `\lambda`, `N!` entered verbatim).

## generator commands (this `scenario` block is the single source used to generate the example)

```scenario domain=statistical_mechanics
@note: Consider a monatomic ideal gas of $N$ particles. Prepare the particle count $N$, volume $V$, temperature $T$, particle mass $m$, Planck constant $h$, and Boltzmann constant $k_B$ as positive real parameters.
Define N, V, T, m, h, k_B as positive real parameters
@note: Define the thermal de Broglie wavelength $\lambda=\sqrt{h^2/(2\pi mk_BT)}$ (the length scale at which quantum effects become important).
\lambda = \sqrt{h^2 / (2 * \pi * m * k_B * T)}
@note: Define the single-particle partition function $Z_1=V/\lambda^3$ (the volume divided by the cube of the thermal de Broglie wavelength, corresponding to the number of quantum states a particle can occupy).
Z_1 = V / E1^3
@note: Define the $N$-particle partition function $Z_N=Z_1^N/N!$. Dividing by $N!$ is the Gibbs correction for overcounting due to permutations of indistinguishable particles.
Z_N = E2^N / N!
@note: Define the Helmholtz free energy $F=-k_BT\ln Z_N$.
F = -k_B * T * \log(E3)
@note: Substitute the definitions of $Z_N$, $Z_1$, and $\lambda$ registered so far and expand $F$ into a closed-form expression in $N,V,T,m,h,k_B$ alone.
Expand F
@note: To find the pressure $P=-(\partial F/\partial V)_T$, first take the partial derivative of $F$ with respect to the volume $V$.
Take the partial derivative of F with respect to V
@note: Flip the sign and register the result as the pressure $P$.
P = -E6
@note: Simplify the resulting expression.
Simplify the right-hand side of E7
@note: To find the internal energy $U=-T^2\partial(F/T)/\partial T$ (the Gibbs-Helmholtz relation), first form $F/T$ as a new expression.
FoverT = F / T
@note: Take the partial derivative of $F/T$ with respect to the temperature $T$.
Take the partial derivative of FoverT with respect to T
@note: Multiply by $-T^2$ and register the result as the internal energy $U$.
U = -T^2 * E10
@note: Simplify the resulting expression.
Simplify the right-hand side of E11
@note: To find the constant-volume heat capacity $C_V=(\partial U/\partial T)_V$, take the partial derivative of the internal energy $U$ with respect to the temperature $T$.
Take the partial derivative of U with respect to T
```
