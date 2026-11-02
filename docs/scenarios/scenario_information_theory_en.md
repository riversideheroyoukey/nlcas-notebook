# Practice-loop sample scenario: three information-theoretic quantities

Obtain the Shannon entropy, the Kullback–Leibler divergence and the mutual
information by **writing them as ordinary sums**.

**Backend:** standard (Maxima) / **domain:** information theory.

## SCENARIO-MACRO, case 2 — rewritten with general operations only

**On 2026-08-14 this scenario was rewritten** from a form that used the dedicated
`shannon_entropy` / `kl_divergence` / `mutual_information` operations (open_items
SCENARIO-MACRO).  All three were a single `sum(...)` template inside — something the
researcher can write with a summation sign.

**The rewrite improved reachability itself.**  The old scenario drove those three through
`@op:`, a **generator-only backdoor**: their required fields hold raw VALUES (`p`,
`lower`, …), which the positional field mapper cannot recover from a phrase, so **no
natural-language trigger existed at all**.  Operations a researcher could not reach from
the UI were sitting in a published example.  Written as sums, the problem disappears.

**The silent-wrong this rewrite uncovered (fixed):** the entropy of a fair die, written
$-\sum_{x=1}^{6}\frac16\log\frac16$, returned **0** (it is $\log 6$).  A fraction was
emitted as `(a)/(b)`, which is **not one operand**, so `\log \frac{1}{6}` was read as
`log(1)/6` — and `log(1)=0`.  The same shape:

| input | what it meant |
|---|---|
| `\log \frac{1}{6}` | `log(1)/6` = **0** |
| `\sin \frac{\pi}{2}` | `sin(π)/2` = **0** (it is 1) |
| `\log \frac{p(x)}{q(x)}` | `log(p)/q` — the KL integrand itself |
| `\frac{a}{b}^2` | `a/b²` |
| `x^\frac{1}{2}` | `x/2` (it is $\sqrt x$) |

**The macros were hiding it**: all three build the `log` call themselves, so the
researcher's own spelling never reached the converter.  This is exactly the
"the macro breaks the instrument" argument in the ticket, and it surfaced the moment the
macro came off.

## Steps and expected results

| # | Input (natural language / math) | Result |
|---|---|---|
| 1 | `$H = -\sum_{x=1}^{6} \frac{1}{6}\log_2 \frac{1}{6}$` | the entropy of a fair die, straight from the definition, in bits |
| 2 | `compute the right-hand side of E1` | $\dfrac{\log 6}{\log 2} = \log_2 6 \approx 2.585$ bits |
| 3 | `$H_b = -p\log p - (1-p)\log(1-p)$` | the entropy of a Bernoulli distribution |
| 4 | `differentiate E3 with respect to p` | $\dfrac{dH_b}{dp}=\log(1-p)-\log p$ |
| 5 | `solve E4 for p` | $\log p=\log(1-p)$ — **the CAS stops here** (see below) |
| 6 | `$D = p\log \frac{p}{q} + (1-p)\log \frac{1-p}{1-q}$` | the KL divergence between two Bernoullis |
| 7 | `substitute p for q in E6` | $D=0$ — this **checks the property** $D(p\Vert p)=0$ |
| 8 | `$I = \sum_{x=1}^{2}\sum_{y=1}^{2} p_{xy}(x,y)\log \frac{p_{xy}(x,y)}{p_x(x) p_y(y)}$` | the definition of mutual information (a double sum) |
| 9 | `$I_0 = \sum_{x=1}^{2}\sum_{y=1}^{2} p_x(x) p_y(y)\log \frac{p_x(x) p_y(y)}{p_x(x) p_y(y)}$` | the independent case, $p_{xy}=p_xp_y$ |
| 10 | `compute the right-hand side of E9` | $0$ — **independence means zero mutual information** |

## Confirmed behaviour and caveats

- **Step 5 does not reach $p=1/2$.**  What comes back is $\log p=\log(1-p)$; Maxima does
  not take the injectivity-of-log step.  **That is the right behaviour** — the system
  returns the condition that holds instead of guessing "probably $p=1/2$".  The
  researcher reads it off.
- **Steps 7 and 10 check properties rather than evaluate definitions.**  The old
  scenario's three steps each merely displayed a definition with values plugged in;
  written with general operations, they compose naturally with `differentiate` and
  `substitute`, so $D(p\Vert p)=0$ and "independence ⇒ $I=0$" come out **as
  computations**.
- **Bits can be written now (2026-08-16).**  Until then `\log_2 6` became
  **`log(2) * 6` — the logarithm OF 2, times 6** (≈ 4.16 where the value is ≈ 2.585).
  Not an undefined symbol but **valid Maxima whose value is simply different**, so
  neither an error nor a warning appeared.  **This scenario was in nats because of that,
  not by choice**, and the workaround $\log 6/\log 2$ was deliberately kept out: put it
  in and nobody rewrites it after the fix.  The based spelling now goes through.
- **The result displays as $\frac{\log 6}{\log 2}$, not $\log_2 6$.**  A change of base
  is a REWRITE (`log(X)/log(b)`), so that is what the CAS holds and what `tex1` gives
  back.  Not wrong, a different spelling — and the displayed form types straight back in
  (the round trip closes).  Reading it as $\log_2$ would need a display bridge.
- **The summation machinery itself is fine** (B-10): `\sum_{x=1}^{6}x` measures 21 and
  `\sum_{x=1}^{3}x^2` measures 14.  Only the fraction in the summand was broken.

```scenario domain=information_theory
@note: Write the Shannon entropy $H=-\sum_x p(x)\log_2 p(x)$ for a fair die ($p=1/6$, six faces) as the definition's own sum, in bits. No dedicated operation is used.
$H = -\sum_{x=1}^{6} \frac{1}{6}\log_2 \frac{1}{6}$
@note: The right-hand side evaluates to $\log 6/\log 2 = \log_2 6 \approx 2.585$ bits — information theory's default unit, so this is the form the definition should be written in. Until 2026-08-16 the based spelling was silent-wrong: it became `log(2)*6`, the logarithm of 2 times 6.
compute the right-hand side of E1
@note: Next the entropy of a Bernoulli distribution, $H_b(p)$. From here the general operations compose naturally.
$H_b = -p\log p - (1-p)\log(1-p)$
@note: Differentiate with respect to $p$ to look for the condition that maximises the entropy.
differentiate E3 with respect to p
@note: Set the derivative to zero and solve for $p$. The CAS returns $\log p=\log(1-p)$ — it does not take the injectivity-of-log step, so $p=1/2$ is read off by the researcher. Not filling that in is the correct behaviour.
solve E4 for p
@note: Write the Kullback–Leibler divergence between two Bernoulli distributions straight from the definition. The point is that the summand's $\log\frac{p}{q}$ can simply be typed.
$D = p\log \frac{p}{q} + (1-p)\log \frac{1-p}{1-q}$
@note: Substitute $p$ for $q$. The basic KL property $D(p\Vert p)=0$ comes out as a computation, not as a displayed definition.
substitute p for q in E6
@note: Write the mutual information as the definition's own double sum.
$I = \sum_{x=1}^{2}\sum_{y=1}^{2} p_{xy}(x,y)\log \frac{p_{xy}(x,y)}{p_x(x) p_y(y)}$
@note: Write the independent case, i.e. with $p_{xy}(x,y)=p_x(x)p_y(y)$ substituted in.
$I_0 = \sum_{x=1}^{2}\sum_{y=1}^{2} p_x(x) p_y(y)\log \frac{p_x(x) p_y(y)}{p_x(x) p_y(y)}$
@note: Evaluating the right-hand side gives 0: independence makes the mutual information vanish, verified by computation.
compute the right-hand side of E9
```
