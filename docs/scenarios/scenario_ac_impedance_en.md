# Practice-loop sample scenario: AC circuit impedance and the resonance condition

> (2026-07-02: an English translation of the Japanese scenario, verified against the
> real Maxima backend through the same rule-based, no-LLM generator pipeline. See
> [scenario_ac_impedance_jp.md](scenario_ac_impedance_jp.md) for the Japanese version.)

Detailed write-up of practice-loop scenario-menu item **S-n (electrical engineering)**.
Enter the complex impedance of a series RLC circuit $Z=R+j\omega L+\dfrac{1}{j\omega C}$,
**separate it into real and imaginary parts**, and solve **imaginary part $=0$
(the resonance condition)** to obtain the resonant angular frequency
$\omega_0=\dfrac{1}{\sqrt{LC}}$.

**Backend:** standard (Maxima) / **Domain:** electrical engineering
(electrical_engineering).

## Physical background

Series RLC impedance:

$$Z=R+j\omega L+\frac{1}{j\omega C}=R+j\!\left(\omega L-\frac{1}{\omega C}\right)$$

- Real part $\operatorname{Re}Z=R$
- Imaginary part (reactance) $\operatorname{Im}Z=\omega L-\dfrac{1}{\omega C}$
- **Resonance condition** $\operatorname{Im}Z=0\ \Rightarrow\ \omega L=\dfrac{1}{\omega C}\ \Rightarrow\ \omega_0=\dfrac{1}{\sqrt{LC}}$

## The point of this scenario: the electrical-engineering imaginary unit `j`

**In an electrical-engineering (electrical_engineering) domain session, `j` is treated
as the imaginary unit** (C-4). It can be used directly in input and is converted to
`%i` under Maxima (in other domains `j` keeps its physical meaning, e.g. as a summation
index). The user never needs to be aware of the CAS's `%i`.

## Steps and expected results

| Step | Input | Expected result |
|---|---|---|
| 1 Register | `Z = R + j*w*L + 1/(j*w*C)` (E1) | $Z=R+j\omega L+\frac{1}{j\omega C}$ (`j` is the imaginary unit in the EE domain) |
| 2 Real part | `Find the realpart of E1` (`real_part`) | $\operatorname{Re}Z=R$ |
| 3 Imaginary part | `Find the imagpart of E1` (`imaginary_part`) → E3 | $\operatorname{Im}Z=\omega L-\dfrac{1}{\omega C}$ (the reactance) |
| 4 Resonance condition | `E3 = 0` | $\omega L-\dfrac{1}{\omega C}=0$ (displayed as an **explicit equation with E3's value expanded**) |
| 5 Solve | Solve the previous equation `for w` (`solve_equation`) | $\omega=\pm\dfrac{1}{\sqrt{LC}}$ → resonance $\omega_0=\dfrac{1}{\sqrt{LC}}$ |

> **What `E_n = X` means:** when the LHS is an existing expression ID, its **value is
> expanded and an explicit equation `<value> = X` is created**. `E3 = 0` is displayed
> as `ωL−1/(ωC) = 0` (the resonance condition). $X$ may be 0, any other value, or
> another expression ID. In the notebook, "explicitly giving an expression a value and
> displaying it" is meaningful in its own right (solving directly by treating E3 as
> `=0` also works, but this makes the reasoning step visible).

## Observed behaviour / notes

- **Completes with existing functionality only** (no new feature needed) — just
  `real_part` / `imaginary_part` / `solve_equation` and the EE domain's `j→%i`
  conversion.
- **English single-site substitution/lookup aliases must be a single token —
  multi-word aliases with a space cannot match the rule-based (no-LLM) Phase-1 alias
  scan.** The scan tokenizes on whitespace and looks up each token verbatim in the
  alias index (`OperationValidator.resolve_alias`, exact lowercase match), so a
  space-containing English alias like `"real part"` (added to
  `schemas/common_operations.schema.json` alongside the existing `"realpart"`) can
  never match: the tokenizer splits it into `"real"` and `"part"` as two separate
  tokens, neither of which alone is a known alias. This is a structural property of
  the rule-based scan, not a bug to "fix" without a larger tokenizer change — the
  existing single-word English aliases (`realpart`, `imagpart`) are what actually work
  in a no-LLM context and are what this scenario's generator block uses. (A
  space-containing alias is still harmless to keep in the schema — it's picked up when
  building LLM prompts.)

**Verified:** 2026-07-02 (real Maxima backend, generator run, all 6 steps completed
end-to-end: Re Z=R, Im Z equation `Lw-1/(Cw)=0`, solve → $w=\pm\sqrt{1/(CL)}$, same as
the Japanese version).

---

## generator commands (this `scenario` block is the single source used to generate the example)

```scenario domain=electrical_engineering
@note: To work with a series RLC circuit's impedance, prepare the resistance R, inductance L, angular frequency w (omega), and capacitance C as real scalar parameters.
Define R, L, w, C as scalars
@note: The impedance of a resistor R, an inductor L, and a capacitor C connected in series is the complex quantity Z = R + j*omega*L + 1/(j*omega*C), where j is the imaginary unit (in this electrical_engineering domain, following engineering convention, j denotes the imaginary unit and i denotes current).
Z = R + j*w*L + 1/(j*w*C)
@note: Extract the real part of the impedance Z (the resistive component).
Find the realpart of E1
@note: Extract the imaginary part of the impedance Z (the reactive component), representing the phase shift between current and voltage caused by the energy-storing elements L and C.
Find the imagpart of E1
@note: Resonance is the condition where the reactance (the imaginary part) vanishes. Form the equation that sets this reactance to zero.
E3 = 0
@note: Solving this equation for the angular frequency w gives the resonant angular frequency. Of the two roots that differ in sign, the physically meaningful one is the positive root omega_0 = 1/sqrt(LC).
Solve E4 for w
```
