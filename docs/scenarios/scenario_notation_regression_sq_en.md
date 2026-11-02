# Notation regression (second quantization): primes and operand spellings

## Purpose

The second-quantization half of the
[notation regression](scenario_notation_regression_en.md). A different backend
means a different reading layer, so it is a separate file (a domain whose
category is `second_quantization` runs on Julia).

Two families are pinned here.

**① A prime is an adjoint only on an OPERATOR.** `a'(k)` is a creation
operator, but `t'`, `\sigma'` and `\mathbf r'` are the researcher's mark for a
DISTINCT VARIABLE — which `latex_converter` already reads that way. Only the
Julia display made no distinction and turned **every prime into a dagger**
(measured 2026-09-11: `G(t, t')` came back as `G(t, t^{\dagger})`). The
discriminator is the DECLARATION, and the published Julia sessions say so: every
primed operator base carries `kind == "operator"`, and `sigma` — whose prime
marks a distinct variable — is declared nowhere.

**② One table answers "how is an operand spelled".** The Heisenberg family
answered that question with THREE hand-written character classes, and 18 of 72
combinations (7 word orders × 9 operand shapes × both languages) were broken —
`c(k,u)` silently truncated to `u)`, `c_{1\uparrow}` not matched, and `c(1, u)`
with the space (**what Maxima's own `tex1` prints**) not matched either.

## What each spelling checks

| spelling | what it once became | ticket |
|---|---|---|
| `a'(k)` (a declared operator) | — (correctly `a^{\dagger}(k)`) | PRIME-DISPLAY-DAGGER |
| `G(t, t')` | `G(t, t^{\dagger})` | PRIME-DISPLAY-DAGGER |
| `\sigma'` | `\sigma^{\dagger}` | PRIME-DISPLAY-DAGGER |
| `c(k,u)` as an operand | silently truncated to `u)` | OPERAND-SPELLING |
| `c(1, u)` (with a space) | matched by no word order | OPERAND-SPELLING |
| `c_{1\uparrow}` | not matched by two of the three | OPERAND-SPELLING |
| a photon's `\sum_\sigma` (spin 1) | not expanded — a formal sum with 4-operator terms left over | OPERAND-BOUNDARY ③ |
| `a_{1,+}` (**typing back what the system printed**) | a raw Julia `ParseError:` reached the researcher | PROJECTION-ROUND-TRIP |
| `a_{1+}` (the glued form) | same | PROJECTION-ROUND-TRIP |

## Input steps

```scenario domain=condensed_matter
@note: Notation regression (second quantization). How a prime reads, and three operand spellings.
define c as an electron annihilation operator
define a as a photon annihilation operator
a'(k)
G(t, t')
\sigma'
H = a'(k) a(k)
evolve c(k,u) under E4
evolve c(1, u) under E4
evolve c_{1\uparrow} under E4
P = \sum_\sigma a^\dagger_{1\sigma} a_{2\sigma}
Q = \sum_\sigma a^\dagger_{1\sigma} a_{1\sigma}
the commutator of P and Q
the commutator of a_{1,+} and a^\dagger_{1,+}
the commutator of a_{1+} and a^\dagger_{1-}
\lim_{x \to 0} \frac{\sin x}{x}
```

## The readings this pins

- `a'(k)` → displayed `a^{\dagger}(k)` (a DECLARED operator, so the prime is the adjoint)
- `G(t, t')` → displayed `G(t, t')` (`G` is not declared an operator, so the prime stays)
- `\sigma'` → displayed `\sigma'` (same rule; a primed spin index names a different value)
- `c(k,u)` / `c(1, u)` / `c_{1\uparrow}` → **each read whole as the operand**
  (truncated, the evolution would answer about a different operator, or fail)
- the commutator of two photon (spin 1, helicity $\pm1$) $\sum_\sigma$ sums →
  **`- a^\dagger_{1+} a_{2+} - a^\dagger_{1-} a_{2-}`**: one term per projection,
  no $\sum$ left. Integer spin is spelled with the plain signs, as
  $\pi^+\pi^0\pi^-$ is, so it is handled exactly as the electron's arrows are.
  Before the fix the sum stayed formal and carried `\sum_{s,s_1}` 4-operator terms
  — not a wrong answer, but the same instruction came back in a different shape
  for a different particle
- `the commutator of a_{1,+} and a^\dagger_{1,+}` → **1**, and with `a^\dagger_{1-}`
  → **0**. These type back **what the commutator above prints**
  (`a^\dagger_{1,+}`); before, a raw Julia `ParseError:` reached the researcher,
  and m=0 was printed as `b_{10}` — **indistinguishable from site 10**. The
  arrows always worked because `\uparrow` is a COMMAND and therefore its own
  index boundary; only the signs had no input spelling
- `\lim_{x \to 0} \frac{\sin x}{x}` → **registers and displays** in a Julia
  session. Symbolics.jl has no limit, so it cannot be COMPUTED — but this
  codebase's stance is "write it down first, refuse at the point of use", and
  before the fix using it answered **"undefined symbols: lim, to"**. The
  researcher wrote neither: they are the two halves of the limit's own canonical
  spelling, and the refusal this codebase wrote for exactly this case
  (`julia_limit`) sat behind that guard and was never reached
  (REFUSAL-NOT-REACHED ①)

## The contract for adding to this file

The same as the Maxima file — **both languages, same order, same count**,
**written so the meaning is visible**, and **a different backend goes in a
different file**.
