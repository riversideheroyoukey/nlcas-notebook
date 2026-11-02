# Practice-loop sample scenario: Fourier transform, convolution and Z transform

Obtain the Fourier transform of a Gaussian and its inverse, the convolution of two
Gaussians, and the Z transform of a discrete signal.

**Backend:** standard (Maxima) / **domain:** signal processing.

## SCENARIO-MACRO, case 5 — three macros removed, one primitive kept

**On 2026-08-14 `fourier_transform_by_definition` / `inverse_fourier_by_definition` /
`convolve` were removed** (open_items SCENARIO-MACRO).  All three assembled an
`integrate(...)` call, which the researcher can write as a definite integral.

**`z_transform_by_definition` stayed.**  Inside it is Maxima's `nusum` (closed-form
infinite summation), which **no general operation expresses** — measured,
`\sum_{n=0}^{\infty}(1/2)^n z^{-n}` comes back as `sum(..., n, 0, inf)`, unclosed.  The
ticket's "genuine primitive gap" is real; this is the second "keep" decision after
`pure_nash_equilibria` in case 4.

**The real gain is that the convergence assumption became the researcher's.**  The macro
called `_positive_params_assume` to **assume every symbolic parameter in the expression
positive**, integrate, and forget it again.  Against the rule that problem-dependent
choices must not be baked into shared code (CLAUDE.md, no-hardcoded-modeling-choices),
that is a **hidden modelling choice**.  Written with general operations, $a>0$ becomes an
**explicit step** (step 5) that stays on the timeline.

**The defect this rewrite uncovered (fixed in a separate commit):** writing the
convolution over $\tau$ showed that `\int_{-\infty}^{\infty} f d\tau` raised a **raw
pydantic ValidationError**, and `\int_0^{2\pi} r\,dr\,d\theta` — **a spelling this system
itself prints** — silently became a **product**.  The measure was read by two
implementations and only one of them handled `d\theta` (open_items MEASURE-READ-TWICE).
**Every Greek integration variable is written that way**: dθ, dφ, dτ, dω.

## Steps and expected results

| # | Input (natural language / math) | Result |
|---|---|---|
| 1 | `$F = \int_{-\infty}^{\infty} e^{-t^2} e^{-i \omega t} dt$` | the Fourier transform written **straight from the definition** |
| 2 | `compute the right-hand side of E1` | $F = e^{-\omega^2/4}\sqrt{\pi}$ |
| 3 | `$f = \frac{1}{2\pi}\int_{-\infty}^{\infty} \sqrt{\pi} e^{-\frac{\omega^2}{4}} e^{i \omega t} d\omega$` | the inverse, likewise from the definition |
| 4 | `compute the right-hand side of E3` | $f = e^{-t^2}$ — **the round trip closes** |
| 5 | `Assume a > 0` | **the researcher states the convergence condition** (the macro silently assumed every parameter positive) |
| 6 | `$G = \int_{-\infty}^{\infty} e^{-a t^2} e^{-i \omega t} dt$` | a parametrised Gaussian |
| 7 | `compute the right-hand side of E5` | $G = \dfrac{e^{-\omega^2/(4a)}\sqrt{\pi}}{\sqrt{a}}$ |
| 8 | `$C = \int_{-\infty}^{\infty} e^{-\tau^2} e^{-(t-\tau)^2} d\tau$` | the convolution, from its definition (now that $d\tau$ reads) |
| 9 | `compute the right-hand side of E7` | $C = \dfrac{e^{-t^2/2}\sqrt{\pi}}{\sqrt{2}}$ — **twice the variance** |
| 10 | `z transform (1/2)^n by definition` | $\mathcal Z\{2^{-n}\}(z)=\sum_{n=0}^{\infty}2^{-n}z^{-n}=\dfrac{2z}{2z-1}$ (**the kept primitive**) |

## Confirmed behaviour and caveats

- **The closing round trip is what steps 1–4 produce.**  $e^{-t^2}$ transforms to
  $e^{-\omega^2/4}\sqrt\pi$ and inverts back to $e^{-t^2}$ exactly.  The macro produced
  the same values, but here **the defining integral itself stays on the timeline**, so
  what was computed is readable.
- **The convergence assumption is a step.**  Omit step 5 and Maxima asks about the sign of
  $a$; the operation returns as a failed step with that reason (the persistent adapter
  does not hang).  **Asking is correct** — for $a<0$ the integral diverges.
- **The Z transform is the kept primitive.**  Wrapping `nusum` in
  `limit(nusum(...,N),N,inf)` makes Maxima ask an interactive sign question ("Is z-1
  positive…") for a symbolic base and **hangs the persistent adapter** (measured).  Plain
  `nusum` never hangs but may leave a boundary term for a symbolic base, which the
  researcher folds with their own region-of-convergence assumption.  **That knowledge
  lives in the macro's docstring** — the macro was not deleted, so it has not evaporated.
- **Writing `\sum_{n=0}^{\infty}` does not close it.**  Measured, it comes back as
  `sum(...,n,0,inf)`: Maxima's `sum` produces no closed form, so an infinite sum needs
  `nusum` — that is what the "primitive gap" actually is.  Extending the general
  `summation` to infinite bounds would make the Z transform a general operation too, but
  it has to get past the adapter-freezing trap above.

```scenario domain=signal_processing
@note: Write the Fourier transform of the Gaussian $e^{-t^2}$ as the definition's own definite integral, $F(\omega)=\int_{-\infty}^{\infty} f(t)e^{-i\omega t}dt$. No dedicated operation is used.
$F = \int_{-\infty}^{\infty} e^{-t^2} e^{-i \omega t} dt$
@note: Evaluate the right-hand side.
compute the right-hand side of E1
@note: Write the inverse from its definition too, $\frac{1}{2\pi}\int F(\omega)e^{i\omega t}d\omega$.
$f = \frac{1}{2\pi}\int_{-\infty}^{\infty} \sqrt{\pi} e^{-\frac{\omega^2}{4}} e^{i \omega t} d\omega$
@note: Evaluating returns exactly the original $e^{-t^2}$ — the round trip closes.
compute the right-hand side of E3
@note: Before the parametrised Gaussian, state the convergence condition $a>0$ explicitly. The macro silently assumed every symbolic parameter in the expression positive, but that is a problem-dependent judgement and does not belong in shared code. Omit it and Maxima asks about the sign of $a$, and the operation fails with that reason — which is correct, since the integral diverges for $a<0$.
Assume a > 0
@note: The Fourier transform of a parametrised Gaussian.
$G = \int_{-\infty}^{\infty} e^{-a t^2} e^{-i \omega t} dt$
@note: Evaluate the right-hand side.
compute the right-hand side of E5
@note: Write the convolution of two Gaussians as the definition's own integral, $\int f(\tau)g(t-\tau)d\tau$.
$C = \int_{-\infty}^{\infty} e^{-\tau^2} e^{-(t-\tau)^2} d\tau$
@note: Evaluating gives a Gaussian of twice the variance.
compute the right-hand side of E7
@note: The Z transform of the discrete signal $(1/2)^n$. This is the kept primitive: inside is Maxima's `nusum` (closed-form infinite summation), which no general operation expresses — measured, writing `\sum_{n=0}^{\infty}` does not close.
z transform (1/2)^n by definition
```
