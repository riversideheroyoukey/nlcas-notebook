# Practice-loop scenario: writing the integration variable your own way (θ, \theta, theta)

> Run 2026-08-03.  Can **what the notebook displayed be typed straight back in**?  Three spellings of
> the integration variable, checked to give the same result.

**Backend:** Maxima / **Domain:** electromagnetism (a polar area integral).

### Why check this

The notebook shows results in TeX, and the researcher types what was shown back in as the next input.
That is the natural way to work, and it is what the user-form ⇄ CAS-form boundary
([docs/32](../developer_manuals/32_user_cas_boundary_repair_plan.md)) exists to make safe.  The
integration variable was the last place where **only one of three spellings got through**:

| Spelling | Before 2026-08-02 | Now |
|---|---|---|
| `theta` (plain) | works | works |
| `\theta` (TeX — the displayed form itself) | no match: the pattern's `[A-Za-z_]\w*` cannot match it | works |
| `θ` (Unicode, straight from the keyboard) | no match: normalised only AFTER the rule-based interception | works |

The Unicode case is the instructive one: **the converter existed and worked correctly**, but it was
called at the top of `LLMSupervisor.interpret()` — after the rule-based interception.  Moving
normalisation to a single entry point fixed it (`interpret_common.normalize_request_greek`) **without
adding one line to any spelling table**, and it fixed Maxima and Julia at the same time.

### The maths

$\iint_{x^2+y^2\le1}(x^2+y^2)\,dx\,dy$ over the unit disc (the polar moment of inertia of a uniform
disc per unit density).  In polar coordinates the Jacobian $|J|=r$ multiplies the integrand and
$\int_0^{2\pi}\!\!\int_0^1 r^3\,dr\,d\theta = \pi/2$.

---

### Steps

#### ST-1: register the region integral

| Item | Content |
|---|---|
| Input | `definite-integrate x^2 + y^2 over the region x^2 + y^2 <= 1` |
| Expected | $\displaystyle\iint_{x^2+y^2\le1}(x^2+y^2)\,dx\,dy$ — **display-only**; the CAS is not called yet |
| Result | |

---

#### ST-2: change to polar coordinates (with a Unicode θ)

| Item | Content |
|---|---|
| Input | `change of variables in E1 from x,y to r,θ via x=r*cos(θ), y=r*sin(θ)` |
| Expected | $\displaystyle\int_0^{2\pi}\!\!\int_0^1 r^3\,dr\,d\theta$ — the Jacobian $r$ is applied and **the new bounds are derived automatically** |
| Check | the `θ` typed from the keyboard goes through as-is; no need to respell it as `theta` |
| Result | |

---

#### ST-3: evaluate

| Item | Content |
|---|---|
| Input | `evaluate this integral` |
| Expected | $\pi/2$ |
| Result | |

---

#### ST-4: the same calculation in TeX spelling (typing the display back in)

| Item | Content |
|---|---|
| Input | `definite-integrate x^2 + y^2 over the region x^2 + y^2 <= 1` |
| Input | `change of variables in E4 from x,y to r,\theta via x=r*\cos(\theta), y=r*\sin(\theta)` |
| Input | `evaluate this integral` |
| Expected | **byte-for-byte identical** to ST-2 and ST-3 |
| Result | |

---

#### ST-5: a single-variable integral behaves the same

| Item | Content |
|---|---|
| Input | `sin(θ)^2` |
| Input | `integrate E7 with respect to θ from 0 to π` |
| Expected | $\displaystyle\int_0^{\pi}\sin^2\theta\,d\theta = \frac{\pi}{2}$ |
| Check | the Unicode `π` also works as a bound (`normalize_bound` maps it to `%pi`) |
| Result | |

---

### Verification checklist

- [x] `θ` (Unicode), `\theta` (TeX) and `theta` (plain) give the same result
- [x] the change of variables applies the Jacobian $r$ and derives the new bounds
- [x] evaluation gives $\pi/2$
- [x] a single-variable integral takes a Unicode variable and a Unicode bound (`θ`, `π`)

---

## What this run found

**Building this scenario turned up a silent wrong answer.**

Writing the change of variables **without the `from … to …` clause** —
`change of variables in E1 via x = r cos(θ), y = r sin(θ)` — routed to the single-variable
`change_variable`, which handed the **display-only** region integral straight to Maxima's
`changevar`.  The result was `_res_` — the adapter's own scratch variable name — and it was
**registered as a successful expression** the researcher could go on to reference.  Nothing to do
with Greek letters: plain names and a linear `u,v` map did the same.

A range-bearing definite integral is display-only **by design** (the CAS is not invoked at input,
INTEGRAL-COV).  Its own operations (`change_variables_integral_form`, `evaluate_integral_form`) are
engine methods that never go through `run_operation`, so **anything that gets there is a generic
operation being handed a picture**.  `_reject_display_only_operands` now refuses it, naming the right
phrasing — the same guard F-9 put in front of display-only angular-momentum states, extended.

**The failed attempt is recorded too.**  A guard was first put in the adapter, on the reasoning that
"a result equal to our own scratch variable can never be an answer".  Measured, **five passing tests
failed**: `_res_` is the ordinary outcome of any command that is not a single expression, and every
declaration the app makes has that shape (`errcatch(_res_:(declare(Delta, scalar); true))` never
binds the name, because inside parentheses Maxima's separator is `,` and not `;`).  Generalising from
the shape alone was wrong.  It was withdrawn, and the reason is written into `maxima_adapter.py` so
the same guard is not added again.

**A second silent wrong answer, this one English-only.**  With no integral form present,
`evaluate this integral` used to decline here and be picked up by the alias scan, where the word
**`evaluate` was taken as the integration variable** and the result was registered as a success
(Japanese produced an empty confirmation dialog instead).  **Fixed 2026-08-04**: the interception no
longer declines — it answers with a reason (open_items archive, EVAL-INTEGRAL-EN-FALLTHROUGH).

**Verified:** 2026-08-03 (real Maxima, ST-1 through ST-5, all three spellings agreeing).

---

## Generator commands (this `scenario` block is the single source)

```scenario domain=electromagnetism
@note: Register $\iint(x^2+y^2)\,dx\,dy$ over the unit disc $x^2+y^2\le1$. The CAS is not called yet — this is a range-bearing, display-only object.
definite-integrate x^2 + y^2 over the region x^2 + y^2 <= 1
@note: Change to polar coordinates, writing the angle as the Greek letter typed straight from the keyboard — that is also what the notebook displays, so being able to type it back is the natural thing. The Jacobian $|J|=r$ multiplies the integrand and the new bounds are derived automatically.
change of variables in E1 from x,y to r,θ via x=r*cos(θ), y=r*sin(θ)
@note: Evaluate the transformed integral.
evaluate this integral
@note: Now the same calculation in TeX spelling, to confirm that typing the displayed form back in gives the same answer.
definite-integrate x^2 + y^2 over the region x^2 + y^2 <= 1
change of variables in E4 from x,y to r,\theta via x=r*\cos(\theta), y=r*\sin(\theta)
evaluate this integral
@note: The same for a single-variable definite integral, with both the variable and the bound written as Greek letters.
sin(θ)^2
integrate E7 with respect to θ from 0 to π
```
