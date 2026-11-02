# Practice-loop sample scenario: line balancing (minimum stations, line efficiency)

Detailed version of practice-loop scenario menu **S-u (production optimization)**. From
the sum of task times $T$ on a production line with cycle time $c$, this scenario finds
the minimum number of stations $n=\lceil T/c\rceil$ (ceiling function), the line
efficiency $\eta=T/(nc)$, and the balance delay $1-\eta$. A small, easy-to-follow
scenario that completes with **arithmetic and substitution alone**.

**Backend:** standard (Maxima) / **Domain:** production optimization (production_optimization).

## Background

For a production line where the sum of all task element times is $T$ and the cycle
time (allowed time per station) is $c$:

- Minimum number of stations: $n=\left\lceil\dfrac{T}{c}\right\rceil$ (you cannot build
  fewer stations than $T/c$, so round up)
- Line efficiency: $\eta=\dfrac{T}{nc}$ (the ratio of ideal total work time $T$ to the
  actual total available time $nc$)
- Balance delay (idle fraction): $1-\eta$

Concrete example ($T=88$, $c=15$): $n=\lceil88/15\rceil=\lceil5.867\rceil=6$,
$\eta=88/(6\times15)=88/90=44/45\approx0.978$, balance delay $=1/45\approx0.022$.

## Session setup / symbols

- Mode **standard (Maxima)**, domain **production optimization**.
- $T,c,n,\eta$ are free variables. The ceiling function uses Maxima's built-in
  `ceiling(...)` (not `ceil`, matching the known-function-name list).

## Input steps

### Steps 1-2: define the model (H-1)

```
n = ceiling(T / c)
eta = T / (n c)
```

**Expected output:** E1 $=n=\lceil T/c\rceil$, E2 $=\eta=T/(nc)$ (each registered as an
H-1 named quantity — "n" and "eta" stay usable as symbols in later expressions).

### Steps 3-4: substitute concrete values ($T=88,\ c=15$) to fix the station count

```
In E1, substitute 88 for T
In E3, substitute 15 for c
```

**Expected output:** E3 $=n=\lceil88/c\rceil$, E4 $=n=6$.

### Steps 5-6: substitute the same concrete values into the efficiency formula

```
In E2, substitute 88 for T
In E5, substitute 15 for c
```

**Expected output:** E5 $=\eta=\dfrac{44}{3c}$, E6 $=\eta=\dfrac{44}{45}$.

> Note that already at step 5 the denominator is $3c$ (with $n=6$ folded in) —
> see "Observed behavior" below.

### Step 7: balance delay (H-2 formula-reference expansion)

```
d = 1 - E6
```

**Expected output:** E7 $=d=\dfrac{1}{45}$ ($1-\eta$ computed directly by expanding
$E6$'s value inline).

## Confirmation points

- The `ceiling` built-in evaluates correctly in Maxima ($\lceil5.867\rceil=6$).
- H-2 (formula-reference expansion) lets a **new expression that directly embeds an
  expression id** (`d = 1 - E6`) go straight through to a computed result.

## Observed behavior / caveats

- **H-1's P2 auto-substitute propagates into other expressions too.** Registering
  `n = ceiling(T/c)` via H-1 also sends `n : ceiling(T/c)$` to Maxima (P2).
  Afterwards, when substituting into a DIFFERENT expression that references `n`
  (`eta = T / (n c)`), Maxima evaluates `n` using **whatever definition of `n` is
  current at that point** (if `n=6` has already been established, that value is
  used). So step 5's result is not the naively-expected `88/(n*c)` (with `n` still
  free) but already has `n=6` folded in, giving `44/(3c)`. **This is not a bug — it
  is P2 working as designed** ("once a symbol is defined, later operations
  automatically pick up its value") — but the **order** of substitutions (fixing
  $n$'s value before moving on to the $\eta$ formula) does affect the intermediate
  results you see.

## Example files

The executed session and exported document:
[../../domains/production_optimization/examples/scenario_line_balancing_en.json](../../domains/production_optimization/examples/scenario_line_balancing_en.json)
/ `.tex` (the Japanese version is `_jp.{json,tex}`).

**Regenerate:** `PYTHONPATH=. .venv/bin/python scripts/generate_scenario_examples.py`

## Generator commands (this `scenario` block is the single source for generation)

```scenario domain=production_optimization
@note: From the total work-element time $T$ and the cycle time $c$ (the time allowed per station), define the minimum number of stations $n=\lceil T/c\rceil$ (ceiling function, since fewer than $T/c$ stations cannot suffice).
n = ceiling(T / c)
@note: Define the line efficiency $\eta=T/(nc)$ (the ratio of the ideal total working time $T$ to the actual total available time $nc$).
eta = T / (n c)
@note: As a concrete example, substitute the total work time $T=88$ into the station-count expression.
In E1, substitute 88 for T
@note: Substitute the cycle time $c=15$ as well, to fix the minimum number of stations $n$ as a number.
In E3, substitute 15 for c
@note: Substitute the same total work time $T=88$, this time into the efficiency expression.
In E2, substitute 88 for T
@note: Substitute the same cycle time $c=15$, to fix the line efficiency $\eta$ as a number.
In E5, substitute 15 for c
@note: Compute the balance delay (idle ratio) $d=1-\eta$, plugging in the value of $\eta$ fixed in the previous step directly.
d = 1 - E6
```
