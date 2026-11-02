# Practice-loop sample scenario: EOQ (Economic Order Quantity) optimization — differentiate → solve → substitute

Detailed version of practice-loop scenario menu **S-t (production optimization)**. The
total cost $C(Q)=KD/Q+hQ/2$ ($Q$: order quantity) is differentiated with respect to $Q$
and set to zero to find the optimal order quantity $Q^*=\sqrt{2KD/h}$, which is then
substituted back into $C$ to obtain the minimum total cost $C(Q^*)$. A classic
**differentiate → solve → substitute** optimization pattern where a CAS shines.

**Backend:** standard (Maxima) / **Domain:** production optimization (production_optimization).

## Background

A basic inventory-management model (Economic Order Quantity). Given annual demand $D$,
ordering cost $K$ per order, and holding cost $h$ per unit, the total cost is

$$C(Q)=\frac{KD}{Q}+\frac{hQ}{2}$$

(ordering cost $KD/Q$: the number of orders $D/Q$ times the cost per order $K$, plus
holding cost $hQ/2$: average inventory $Q/2$ times the holding cost per unit $h$).
Differentiating $C$ with respect to $Q$ and setting it to zero gives the optimal order
quantity:

$$\frac{dC}{dQ}=-\frac{KD}{Q^2}+\frac{h}{2}=0 \;\Longrightarrow\; Q^*=\sqrt{\frac{2KD}{h}}.$$

Substituting $Q^*$ into $C(Q)$ gives the minimum total cost $C(Q^*)=\sqrt{2KDh}$.

## Session setup / symbols

- Mode **standard (Maxima)**, domain **production optimization**.
- $K,D,h,Q$ are free variables (Maxima treats them as symbols without declaration; an
  undefined-symbol warning may appear but does not block processing in a Maxima
  session).

## Input steps

### Step 1: define the total cost function (H-1)

```
C = K D / Q + h Q / 2
```

**Expected output:** registered as E1. $C=\dfrac{KD}{Q}+\dfrac{hQ}{2}$.

### Step 2: total derivative with respect to $Q$

```
Take the total derivative of E1 with respect to Q
```

**Expected output:** E2 $=\dfrac{dC}{dQ}=\dfrac{h}{2}-\dfrac{DK}{Q^2}$.

> **"Total derivative" must be stated explicitly.** $C$ depends only on $Q$, but the
> system cannot tell partial from total derivative from the phrasing alone, so it
> requires an explicit choice (a bare "differentiate E1 with respect to Q" would fail
> with "cannot tell whether this is a partial or total derivative"). For single-variable
> optimization, always use "total derivative."

### Step 3: solve for $Q$ (H-9, optimal order quantity)

```
Solve E2 for Q
```

**Expected output:** two solutions are registered: $Q=-\sqrt2\sqrt{DK/h}$ (E3, the
physically meaningless negative root) and $Q^*=$ E4 $=\sqrt2\sqrt{DK/h}\;
\left(=\sqrt{2DK/h}\right)$ (the positive order quantity).

### Step 4: substitute $Q^*$ into the total cost

```
Substitute E4 into E1
```

**Expected output:** E5 $=C(Q^*)=\dfrac{\sqrt2\,DK}{\sqrt{DK/h}}$.

## Confirmation points

- Of the two roots from the derivative-then-solve step, the **positive root** (E4)
  corresponds to the optimal order quantity $Q^*$ (the negative root E3 is
  mathematically valid but economically meaningless — an order quantity must be
  positive).
- Substituting $Q^*$ into $C(Q)$ (E5) is mathematically equivalent to the textbook
  form $\sqrt{2DKh}$, even though the radical form differs (see "Observed behavior"
  below).

## Observed behavior / caveats

- **Maxima does not automatically rationalize radicals.** E5's canonical stays as
  `(sqrt(2)*D*K)/sqrt((D*K)/h)` ($=\sqrt2\,DK/\sqrt{DK/h}$) rather than the textbook
  $\sqrt{2KDh}$ form (`ratsimp`/simplify does not rationalize a radical denominator).
  The two are algebraically equivalent (squaring both sides gives
  $2D^2K^2/(DK/h)=2DKh$ on both). Getting the textbook form would need a dedicated
  radical-simplification operation (`radical_simplify`, see BP-2 in open_items.md);
  no rule-based NL trigger exists for it yet (LLM path only).
- **A bare "differentiate" does not work.** Even for a single-variable function, the
  partial-vs-total choice must be stated explicitly (a deliberate disambiguation
  design). Say "take the total derivative ... with respect to Q."

## Example files

The executed session and exported document:
[../../domains/production_optimization/examples/scenario_eoq_optimal_order_quantity_en.json](../../domains/production_optimization/examples/scenario_eoq_optimal_order_quantity_en.json)
/ `.tex` (the Japanese version is `_jp.{json,tex}`).

**Regenerate:** `PYTHONPATH=. .venv/bin/python scripts/generate_scenario_examples.py`

## Generator commands (this `scenario` block is the single source for generation)

```scenario domain=production_optimization
@note: Given annual demand $D$, ordering cost $K$ per order, holding cost $h$ per unit, and order quantity $Q$, define the total cost $C(Q)=KD/Q+hQ/2$ (the sum of the ordering cost, "number of orders $D/Q$ times $K$", and the holding cost, "average inventory $Q/2$ times $h$").
C = K D / Q + h Q / 2
@note: To find the order quantity that minimizes the total cost, take the total derivative of $C$ with respect to $Q$.
Take the total derivative of E1 with respect to Q
@note: Set the derivative to zero and solve for $Q$ to find the candidate order quantities that minimize the cost.
Solve E2 for Q
@note: Substitute the economically meaningful positive order quantity $Q^*$ from the solutions just found into the total-cost expression $C(Q)$ to obtain the minimum total cost.
Substitute E4 into E1
```
