# Practice-loop sample scenario: Itô's lemma and the Black–Scholes equation

Apply **Itô's lemma** to a function $f(S,t)$ of an asset following geometric Brownian
motion $dS=\mu S\,dt+\sigma S\,dW$, and write down the Black–Scholes equation.

**Backend:** standard (Maxima) / **domain:** financial mathematics.

## SCENARIO-MACRO, case 1 — rewritten with general operations only

**On 2026-08-14 this scenario was rewritten** from a form that used the dedicated
`apply_ito_lemma` operation to one that uses only general operations (open_items
SCENARIO-MACRO).

The reason is not that the macro failed.  **It broke the instrument precisely because it
worked**: running this scenario with an operation added for this scenario makes the
question "are the features sufficient?" answer itself, and all the practice loop can then
verify is that the macro works on the one input it was written for.

**Removed:** `apply_ito_lemma` (`notebook/engine_finance_mixin.py` +
`ui/interpret_finance.py` — a dedicated entrance that was not even in the common
operation schema).  Inside, it called `differentiate` three times and combined the
results in a **fixed linear combination**.

**Where the knowledge went:** Itô's formula, which lived in the macro's docstring, is now
**one step the researcher types** (step 5 below).  The formula is finance; it does not
belong to the system.

**Measured equal before and after:**

| $f$ | old (macro) | new (general operations) |
|---|---|---|
| $\ln S$ | $df=\left(-\frac{\sigma^2-2\mu}{2}\right)dt+\sigma\,dW$ | $df=\left(\mu-\frac{\sigma^2}{2}\right)dt+\sigma\,dW$ |
| $S^2$ | $df=\left(S^2\sigma^2+2S^2\mu\right)dt+2S^2\sigma\,dW$ | $dg=\left(S^2\sigma^2+2S^2\mu\right)dt+2S^2\sigma\,dW$ |

The same values; only the arrangement differs (the old path ran `ratsimp`).

**Fixed in passing:** the old scenario wrote its SDE and Black–Scholes equation as
`diff(V,t) + (1/2)*sigma^2*S^2*diff(V,S,2) + r*S*diff(V,S) - r*V = 0` — **raw Maxima
syntax**, i.e. a Core Principle #1 violation sitting in a published example.  Both are
LaTeX now.

## Mathematical background

When $S$ follows geometric Brownian motion $dS=\mu S\,dt+\sigma S\,dW$, the differential
of $f(S,t)$ obeys **Itô's lemma** rather than the ordinary chain rule:

$$df=\left(\frac{\partial f}{\partial t}+\mu S\frac{\partial f}{\partial S}
  +\frac{1}{2}\sigma^2S^2\frac{\partial^2f}{\partial S^2}\right)dt
  +\sigma S\frac{\partial f}{\partial S}\,dW.$$

The $\frac12\sigma^2S^2\partial^2_Sf$ term is what separates it from the classical chain
rule, and comes from $(dW)^2=dt$.  **The formula itself is finance**; what the system
supplies is the partial derivatives (`differentiate`) and the means to assemble them.

## Steps and expected results

Each step is an independent YOU card.  Two functions are checked, $f=\ln S$ and $g=S^2$.

| # | Input (natural language / math) | Result |
|---|---|---|
| 1 | `$f = \ln S$` (math input) | $f=\ln S$ (H-1 registers $f$) |
| 2 | `take the partial derivative of E1 with respect to t` | $\dfrac{\partial f}{\partial t}=0$ |
| 3 | `take the partial derivative of E1 with respect to S` | $\dfrac{\partial f}{\partial S}=\dfrac{1}{S}$ |
| 4 | `take the 2nd order partial derivative of E1 with respect to S` | $\dfrac{\partial^2f}{\partial S^2}=-\dfrac{1}{S^2}$ |
| 5 | `$df = (E2 + \mu S E3 + \frac{1}{2}\sigma^2 S^2 E4) dt + \sigma S E3 dW$` | $df=\left(\mu-\dfrac{\sigma^2}{2}\right)dt+\sigma\,dW$ — **the researcher writes Itô's formula, pulling in steps 2–4 by expression reference (H-2)** |
| 6 | `$g = S^2$` (math input) | $g=S^2$ |
| 7 | `take the partial derivative of E6 with respect to t` | $\dfrac{\partial g}{\partial t}=0$ |
| 8 | `take the partial derivative of E6 with respect to S` | $\dfrac{\partial g}{\partial S}=2S$ |
| 9 | `take the 2nd order partial derivative of E6 with respect to S` | $\dfrac{\partial^2g}{\partial S^2}=2$ |
| 10 | `$dg = (E7 + \mu S E8 + \frac{1}{2}\sigma^2 S^2 E9) dt + \sigma S E8 dW$` | $dg=\left(S^2\sigma^2+2S^2\mu\right)dt+2S^2\sigma\,dW$ |
| 11 | `$dS = \mu S dt + \sigma S dW$` (math input) | the SDE — no dedicated operation needed, an ordinary registration |
| 12 | `$\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + r S \frac{\partial V}{\partial S} - r V = 0$` | the Black–Scholes equation, likewise an ordinary equation |

## Confirmed behaviour and caveats

- **Partial vs total must be stated.**  `differentiate E1 with respect to t` **fails**
  with "cannot tell whether this is a partial or a total derivative"
  (`differentiate_is_ambiguous`).  For $f(S,t)$ with $S$ depending on $t$ the ambiguity is
  real, so it is right that the system does not decide; the researcher says "partial".
- **Expression references (H-2) do the assembling.**  Step 5 refers to steps 2–4 as
  `E2`–`E4`, expanded into the canonical at registration.  A reference to an expression
  that does not exist lands in `undefined_symbols` and raises the ⚠ banner (measured), so
  it never silently becomes a symbol.
- **The result's canonical keeps the unexpanded form.**  Step 5's canonical is
  `((0) + mu*S*(1/S) + …)*dt + …`; only the display is simplified, by Maxima's `tex1`.
  That follows the rule that `canonical` is *the researcher's own writing* — what they
  wrote is Itô's formula, not its simplification.  Later operations meet a
  mathematically equal expression.
- **Solving the Black–Scholes equation symbolically for a closed-form price is out of
  scope.**  This scenario builds the equation; going further is the numeric layer's (N-1)
  job or a separate design.

## The shape to follow when rewriting with general operations only

1. **The researcher writes the formula.**  A fixed formula that a macro embedded becomes
   one step assembled from expression references.  The system supplies only general
   operations such as `differentiate` / `simplify`.
2. **Type LaTeX, not CAS syntax.**  No `diff(V,t)` in a scenario (Principle #1).  A
   spelling that cannot be typed is a defect — and finding it is what the loop is for.
3. **Move only the tests that pin a trap.**  Tests asserting the operation's interface go
   away with the operation.  Here
   `test_feeding_result_into_another_call_does_not_hang` was the **only** test naming the
   trap in the Fourier mixin's `_transform_source_body`, so it moved to
   `tests/test_fourier_transform.py`.  The tests asserting VALUES (the textbook result)
   were rewritten as assertions about the general-operation route.
4. **Length is not a cost.**  Twelve steps is longer than one `apply_ito_lemma` line, and
   those twelve steps passing is exactly what the practice loop should produce.
5. **Never write a workaround into a scenario.**  A spelling chosen to dodge a defect
   becomes **permanent** once published — a worked example is read as "this is how it is
   written", so nobody rewrites it after the fix.  If one is unavoidable, mark it as
   temporary and write **both** halves: the reason and the ticket id in the scenario, and
   "when fixed, come back to this scenario" in the ticket.  With only one half, whoever
   fixes the defect never learns the scenario exists.  (Live examples: bits in
   [information theory](scenario_information_theory_en.md), the English inverse transform
   in [the forced oscillator](scenario_forced_oscillator_laplace_en.md).)

```scenario domain=financial_mathematics
@note: Apply Itô's lemma to a function of an asset following geometric Brownian motion $dS=\mu S\,dt+\sigma S\,dW$. Start by registering the logarithm $f=\ln S$.
$f = \ln S$
@note: Obtain, with general operations, the three partial derivatives Itô's formula needs. First $\partial f/\partial t$, which is 0 since $t$ does not appear explicitly in $f$. Whether this is a partial or a total derivative is stated by the researcher (the ambiguity is real, since $S$ depends on $t$).
take the partial derivative of E1 with respect to t
@note: Then $\partial f/\partial S$.
take the partial derivative of E1 with respect to S
@note: The second-order term $\partial^2 f/\partial S^2$ specific to Itô's lemma — this is what separates it from the classical chain rule.
take the 2nd order partial derivative of E1 with respect to S
@note: Assemble Itô's formula $df=(\partial_t f+\mu S\partial_S f+\frac12\sigma^2S^2\partial_S^2 f)dt+\sigma S\partial_S f\,dW$ from those expression references. The formula itself is finance, and the researcher writes it.
$df = (E2 + \mu S E3 + \frac{1}{2}\sigma^2 S^2 E4) dt + \sigma S E3 dW$
@note: Repeat the same steps for a different function $g=S^2$, to confirm the same formula works on another input.
$g = S^2$
@note: $\partial g/\partial t$.
take the partial derivative of E6 with respect to t
@note: $\partial g/\partial S$.
take the partial derivative of E6 with respect to S
@note: $\partial^2 g/\partial S^2$, a constant 2.
take the 2nd order partial derivative of E6 with respect to S
@note: Assemble the same Itô formula for $g$ — identical in shape to step 5, only the expression references change.
$dg = (E7 + \mu S E8 + \frac{1}{2}\sigma^2 S^2 E9) dt + \sigma S E8 dW$
@note: The SDE notation itself needs no dedicated operation — it registers as an ordinary expression ($dt$ and $dW$ are plain symbols).
$dS = \mu S dt + \sigma S dW$
@note: The Black–Scholes equation is likewise an ordinary equation. Solving it symbolically for a closed-form price is out of this scenario's scope.
$\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + r S \frac{\partial V}{\partial S} - r V = 0$
```
