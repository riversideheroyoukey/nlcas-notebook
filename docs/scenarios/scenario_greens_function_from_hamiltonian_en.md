# Practice-loop scenario: from a field-operator Hamiltonian to the Green's function

> Run 2026-08-20; **rewritten 2026-08-21 so that the derivation is COMPUTED**.
> The many-body Green's function in the spelling a textbook uses — and each step
> of the derivation carried out by the CAS: introducing the eigenvalues by
> diagonalisation, moving the Green's function into the basis, the formal
> solution of the Dyson equation, the Hartree–Fock decoupling of the higher
> Green's function, the Fourier transform from time to frequency, and solving
> for $G$.

**Why it was rewritten (the researcher's point):** the first version only PUT
DEFINITIONS DOWN. The formulae were written as a textbook writes them, but each
step of the derivation was typed by someone who already knew the answer — so
nothing measured whether the system could COMPUTE any of it. To ask what the CAS
is good for, each step has to go through as an operation, and the steps that do
not go through have to be measured and stated. The breakdown is below.

| step | what happens | computed? |
|---|---|---|
| Diagonalising $\hat H_0$ | substitute "the basis diagonalises it" ($T=\varepsilon\delta$) into the general $\sum_{nm}T\hat\rho$ and simplify | ✅ **CAS** (the delta collapses the $m$ sum; **the eigenvalue first appears here**) |
| Moving $G$ into the basis | substitute the mode expansion into the definition, twice | ✅ **CAS** (the sums are carried inside the time-ordered expectation) |
| Dyson's formal solution | multiply both sides of $(1-G^0\Sigma)G=G^0$ by $(1-G^0\Sigma)^{-1}$ **from the left** | ✅ **CAS** (the left side reduces to 1 and $G=(1-G^0\Sigma)^{-1}G^0$ comes out) |
| The EOM commutator | $[\hat c_{m\sigma},\hat H]$ | ❌ Maxima has no operator algebra → **computed in the sister scenario** (Julia/QuantumAlgebra) and written across |
| Hartree–Fock decoupling | split four operators into pairs ($\approx$) | ⬜ the approximation itself (one line). An $\approx$ **can be written and displayed; computing with it is refused** |
| Fourier transform | write the inverse transform's own integral and differentiate both sides | ✅ **CAS** ($-i\omega$ comes down: $i\hbar\partial_t\to\hbar\omega$) |
| Solving for $G$ | solve the frequency-domain equation | ✅ **CAS** (the Hartree–Fock Green's function) |
| Checking the approximation | substitute $\Sigma^{(HF)}=0$ | ✅ **CAS** (the free propagator comes back) |

What stays a definition is what there is nothing to evaluate against — matrix
elements in an arbitrary basis, all-space integrals, expectation values. What is
computed is the algebra of indices and coefficients, which is the part a machine
can do here.

**Backend:** Maxima / **domain:** quantum mechanics (`quantum_mechanics`).

### Why this is worth confirming

This scenario exists because measuring the right question gave zero. On the same
day, eighteen lines of T = 0 many-body notation were typed in as a textbook
writes them, and **not one kept its meaning**. Most did not even fail — they
**succeeded as a different expression**:

| spelling | measured (before the repair) |
|---|---|
| `G = G^0 + G^0\Sigma G` | **`G*Sigma+1`** (`G^0` is 1 to Maxima) |
| `v(\mathbf r-\mathbf r')` | **`v(0)`** (the prime was silently dropped) |
| `\hat\psi` | **`p_hatsi`** (`hat` wedged inside `psi`) |
| the `\psi^\dagger` family | five spellings, **five different things** |
| `\langle\ldots\rangle` | the product of two undefined symbols `langle`, `rangle` |
| `c_{k+q}` | **`c(k) + q`** (the braces were dropped) |
| `\delta(\mathbf r-\mathbf r')` | `KroneckerDelta requires exactly 2 arguments` |
| `T[\ldots]` (time ordering) | a raw pydantic ValidationError |

Underneath them sat a deeper one: **the Maxima adapter reported every syntax
error as a success** (`errcatch` does not catch a syntax error; the guard
variable kept the previous command's value and the unbound `_res_` — the
adapter's own scratch name — was printed as the answer). While that stood, every
notation this system could not lower correctly came back as a plausible answer.

This scenario checks that the repair holds **on the real thing**. The question is
not "can it compute" but **does it mean what was written**.

**Writing the continuation produced six more of the same shape** (measured
2026-08-21). The equation of motion asks for spellings the first half never used
— a time derivative, an approximate relation, a solve-for variable carrying
indices — and those had not been measured:

| spelling | measured (before the repair) |
|---|---|
| `\frac{\partial}{\partial t}\hat c_{n\sigma}(t)` | **`'diff(\hat,t,1)*c_{n,sigma}(t)`** — the decoration command is differentiated and the name becomes a separate factor |
| `\frac{\partial \hat\psi(x,t)}{\partial t}` | **`psi_hat(x,t)/t`** — not read as a derivative at all, so $\partial$ cancels (BP-9 itself) |
| `\frac{d}{dx}\sin x` | **`'diff(sin,x,1)*x`** = **0** |
| `\hat T(\mathbf r)` | **`time_order(…, hat)`** — the one-body kinetic operator becomes a time-ordering operator. **It was in the first half's $\hat H_0$ (E4), published** |
| `A \approx B + C` | **`A*B*approx+C`**, returned by `simplify` as a SUCCESS (the same shape `\neq` was repaired for) |
| `solve E32 for G^{(HF)}_{n\sigma}(\omega)` | not interpreted — neither a braced subscript nor a superscript label was read as a solve-for variable. Once they were, the name — being a DEFINED one — was rewritten to its expression id, so Maxima was asked to solve for a name the researcher never wrote (`solve(…, E22_psupHFn,sigma (reference omitted from the public snapshot))`) |

As with the first five, **not one of them was an error**. The first three have a
single cause: the scan that reads what is being differentiated took `\hat` and
`\sin` to be complete names on their own. A command is not a quantity by
itself, so taking its argument is grammar, not a guess.

### The physics

1. Spin-dependent field operators and their anticommutator
2. The Hamiltonian — one-body and two-body terms, written with field operators
3. A spin-dependent orthonormal set (orthonormality and completeness); the
   orthonormality is **opened into components**
4. Expand the field operators in that set
5. **Move the Hamiltonian into the new basis** — the one-body term becomes a
   matrix element $T_{n\sigma,m\sigma}$ times an operator pair, the two-body term
   a matrix element with **six indices: four modes and two spins** (two, because
   the field-operator form carries two). Up to here the basis is ARBITRARY
6. **Diagonalise its zeroth-order part** — assuming the orthonormal set
   introduced at the start diagonalises $\hat H_0$ is a single equation about the
   matrix element, $T_{n\sigma,m\sigma}=\varepsilon_{n\sigma}\delta_{nm}$.
   Substituting and simplifying, the delta collapses the sum and $\hat H_0$
   becomes a sum of number operators — **the eigenvalue $\varepsilon_{n\sigma}$
   appears as the RESULT of that computation**
7. Define the Green's function **with this basis's creation and annihilation
   operators**, the ground state **written out**:
   $\langle\Phi_0|T[\cdots]|\Phi_0\rangle$. One could write it with field
   operators and move it into the basis afterwards, but that costs two
   substitutions and buys nothing (**the user's point, 2026-08-22** — this
   scenario used to take that route)
8. **Fourier transform in the time difference** to introduce $\omega$ and write
   the free propagator $G^0$ — the sign of $i\eta$ **flips across the Fermi
   surface**
9. Write the Dyson equation **in components** — the second term on the right
   is a sum over the basis indices. Pull out the common factor, **declare $G$
   and $\Sigma$ to be matrices**, and **write the general solution with the
   inverse**
10. **Only then approximate** — substitute the first-order (Hartree)
    self-energy for the general $\Sigma$

That is the Dyson route. Only its last step is a computation; everything before
it puts a definition down. As with matrix elements in an arbitrary basis, the
integrals and expectation values here are not things to evaluate — there is
nothing to evaluate them against. What the notebook must do is let them be
**written, displayed as written, and used later**.

### The continuation: fixing the self-energy from the equation of motion

**The Dyson equation does not say what $\Sigma$ is.** It was put down in general
form, and the first-order shape (E20) came from outside. The other route — the
Green's function's **equation of motion** — gets $\Sigma$ out of the Hamiltonian.

11. Creation/annihilation operators in the Heisenberg picture. The exponent
    carries the **full** Hamiltonian (that is what differs from the interaction
    picture of E10)
12. The operator equation of motion $i\hbar\partial_t\hat c=[\hat c,\hat H]$.
    Opened with the anticommutator, $\hat H_0$ gives the eigenvalue and
    $\hat V$ gives a **product of three operators**
13. The Green's function's equation of motion. The time derivative **also hits
    the step functions of the time ordering**, and the equal-time anticommutator
    comes out of that as the $\hbar\delta(t-t')$ term. The two-body term
    produces the expectation of **four operators** — the higher Green's function
    $G^{(2)}$
14. **The equation is not closed.** The one-body equation contains the two-body
    one, whose equation contains the three-body one. **An approximation is a cut
    in that chain**
15. The lowest cut (**Hartree–Fock decoupling**) — split the four operators into
    pairs and replace the equal-time pair by an occupation number. **It is not
    an equality, so it is written with $\approx$.** The approximation is that
    one line; everything after it is algebra
16. Substituting collapses two of the sums and **the self-energy appears**. Its
    first term is E20 — the Hartree term the Dyson route took from outside —
    **exactly**; its second is the exchange (Fock) term, where being a fermion
    first matters. Fourier transform and **solve for $G$**:
    $G_{n\sigma}(\omega)=\hbar/(\hbar\omega-\varepsilon_{n\sigma}-\Sigma^{(HF)}_{n\sigma})$,
    and setting the self-energy back to zero returns the free propagator

**The approximation is confined to one line**, which is what this route is worth
looking at for: every other line is an equality, so where generality was given
up can still be read off afterwards.

```scenario domain=quantum_mechanics
@note: Start from the spin-dependent field operators. The hat is a decoration of the NAME: $\hat\psi$ is a different quantity from $\psi$, which is exactly why physics writes it that way.
\hat\psi_\sigma(\mathbf{r})
\hat\psi^\dagger_\sigma(\mathbf{r})
@note: The equal-time anticommutator. The two deltas on the right are DIFFERENT: the spin one is Kronecker, the position one is Dirac. A subscript against an argument list tells them apart — physics writes them apart, so no guess is needed.
\{\hat\psi_\sigma(\mathbf{r}), \hat\psi^\dagger_{\sigma'}(\mathbf{r}')\} = \delta_{\sigma\sigma'}\delta(\mathbf{r}-\mathbf{r}')
@note: Declare that the 0 in $\hat H_0$ is PART OF THE NAME. Undeclared, a subscript is read as a component index (this notebook's rule), and a computed result would come back as $\hat H(0)$. A declaration is the discriminator for most of the readings in this system.
declare \hat H_0 as a scalar
@note: The one-body term. $\hat T$ is kinetic energy plus external field as one operator. $\int d^3r$ is a volume measure, not a one-variable integral.
\hat H_0 = \int d^3r\, \hat\psi^\dagger_\sigma(\mathbf{r}) \hat T(\mathbf{r}) \hat\psi_\sigma(\mathbf{r})
@note: The two-body term — six dimensions, and $\mathbf r$ and $\mathbf r'$ are DIFFERENT points. While the prime was being dropped, $v(\mathbf r-\mathbf r')$ was reaching the CAS as $v(0)$.
\hat V = \frac{1}{2}\int d^3r \int d^3r' \hat\psi^\dagger_\sigma(\mathbf{r}) \hat\psi^\dagger_{\sigma'}(\mathbf{r}') v(\mathbf{r}-\mathbf{r}') \hat\psi_{\sigma'}(\mathbf{r}') \hat\psi_\sigma(\mathbf{r})
@note: The spin-dependent orthonormal set. Orthonormality is an integral over all space and the basis is arbitrary, so it is NOT something to evaluate — the integral is the condition.
\int d^3r\, \varphi^*_{n\sigma}(\mathbf{r}) \varphi_{m\sigma'}(\mathbf{r}) = \delta_{nm}\delta_{\sigma\sigma'}
@note: Open it into components: an iterated Cartesian integral over all space.
expand into components
@note: Completeness. The right-hand side is the Dirac delta; the sum on the left runs over BOTH mode and spin.
\sum_{n\sigma} \varphi_{n\sigma}(\mathbf{r}) \varphi^*_{n\sigma}(\mathbf{r}') = \delta(\mathbf{r}-\mathbf{r}')
@note: Expand the field operators in that set. From here the algebra splits into scalar coefficients and the operators $\hat c_{n\sigma}$.
\hat\psi_\sigma(\mathbf{r}) = \sum_n \varphi_{n\sigma}(\mathbf{r}) \hat c_{n\sigma}
@note: **The computation starts here.** Put the expansion (E9) into $\hat H$ and write **the Hamiltonian in the new basis**. The one-body term first: the matrix element $T_{n\sigma,m\sigma}$ is an all-space integral of the same shape as E6, and is not something to evaluate. The operator PAIR $\hat c^\dagger_{n\sigma}\hat c_{m\sigma}$ is carried as ONE name — Maxima has no operator algebra, so it is never asked to order a product (the operator algebra itself is done in the sister scenario's Julia session).
\hat\rho_{n\sigma,m\sigma} = \hat c^\dagger_{n\sigma} \hat c_{m\sigma}
\hat H_0 = \sum_{n m} T_{n\sigma,m\sigma} \hat\rho_{n\sigma,m\sigma}
@note: The two-body term moves over the same way. **There are only TWO spins**: the field-operator form carries $\sigma,\sigma'$ and expanding in the basis multiplies the MODES only, so the independent indices are **four modes plus two spins — six**. Summation indices are written SEPARATED (`\sum_{nmpq}` would be one index named $nmpq$).
\hat V = \frac{1}{2}\sum_{n m p q}\sum_{\sigma \sigma'} V_{n\sigma,m\sigma',p\sigma',q\sigma} \hat c^\dagger_{n\sigma} \hat c^\dagger_{m\sigma'} \hat c_{p\sigma'} \hat c_{q\sigma}
@note: Up to here the basis is ARBITRARY. Now CHOOSE one: assuming that the orthonormal set introduced at the start diagonalises $\hat H_0$ is a single equation about the matrix element. Unlike specialising to plane waves, almost no generality is lost.
T_{n\sigma,m\sigma} = \varepsilon_{n\sigma} \delta_{nm}
@note: Substitute it into the zeroth-order term.
substitute E13 into E11
@note: Simplifying, **the Kronecker delta collapses the sum over $m$** and the diagonal form is left — **the eigenvalue $\varepsilon_{n\sigma}$ appears as the RESULT of this computation**, not copied in by hand. $\hat\rho_{n\sigma,n\sigma}$ is the number operator.
simplify E14
@note: **The zeroth-order term is diagonal now, so define the Green's function HERE, with this basis's creation and annihilation operators.** One could write it with field operators and move it into the basis afterwards, but that costs two substitutions and buys nothing — write it in this basis from the start. $T[\cdots]$ is the time ordering (a bare $T$ also reads as kinetic energy, so the bracket is written), and with the same state on both sides this is the same quantity as $\langle A\rangle_{\Phi_0}$.
\hat c_{n\sigma}(t) = e^{i\hat H t/\hbar} \hat c_{n\sigma} e^{-i\hat H t/\hbar}
G_{n\sigma,m\sigma'}(t,t') = -i\langle\Phi_0| T[\hat c_{n\sigma}(t) \hat c^\dagger_{m\sigma'}(t')]|\Phi_0\rangle
@note: The free propagator. The superscript 0 of $G^0$ is a **perturbation-order label**, not an exponent. The sign of $i\eta$ **flips across the Fermi surface**.
G^0_{n\sigma}(\omega) = \frac{1}{\omega - \varepsilon_{n\sigma}/\hbar + i\eta\,\mathrm{sgn}(\varepsilon_{n\sigma}-\varepsilon_F)}
@note: The Dyson equation **in components** — the second term on the right is a sum over the basis indices, and $G$ appears on the right as well.
G_{n\sigma,m\sigma'}(\omega) = G^0_{n\sigma}(\omega)\delta_{nm}\delta_{\sigma\sigma'} + \sum_{p \sigma''} G^0_{n\sigma}(\omega) \Sigma_{n\sigma,p\sigma''}(\omega) G_{p\sigma'',m\sigma'}(\omega)
@note: Drop the indices and write it as a matrix. **Declaring $G$, $\Sigma$ and $G^0$ to be matrices** makes the products matrix products and $^{-1}$ the INVERSE. Without the declaration $^{-1}$ is a reciprocal and the product is commutative — both are different quantities.
declare G as a matrix
declare \Sigma as a matrix
declare G^0 as a matrix
@note: Write it with $G$ collected on the left. **Do NOT write $G = G^0 + G^0\Sigma G$**: that form is an H-1 DEFINITION, and from then on the name $G$ stands for that expression instead of being the unknown we want to solve for (measured: written that way, the next step comes back with its left side expanded to $G^0\Sigma G+G^0$).
(1 - G^0 \Sigma) G = G^0
@note: **Compute the formal solution.** Multiply both sides by $(1-G^0\Sigma)^{-1}$ **from the left** — the product does not commute, so the two sides are different equations and which one is meant is the researcher's to say. Reducing $(1-G^0\Sigma)^{-1}(1-G^0\Sigma)$ to 1 is the CAS's job.
multiply both sides of E20 by (1 - G^0 \Sigma)^{-1} from the left
@note: That is the Dyson route. **The Dyson equation does not say what $\Sigma$ is.** The other route — the Green's function's **equation of motion** — gets $\Sigma$ out of the Hamiltonian.
@note: The Heisenberg-picture operator. Unlike the interaction picture (E10), the exponent carries the **full** Hamiltonian $\hat H=\hat H_0+\hat V$.
@note: The operator equation of motion. **What is differentiated is $\hat c$, not $c$** — the hat decorates the NAME, so it goes inside the derivative with it.
i\hbar \frac{\partial}{\partial t} \hat c_{n\sigma}(t) = [\hat c_{n\sigma}(t), \hat H]
@note: **Maxima cannot compute this commutator** — it has no operator algebra (which is also why the operator pair is carried as one name in this session). **That division of labour was decided on 2026-07-30**: Green's-function algebra in Maxima, Julia only where operator algebra is needed (docs/33 §7). The SAME commutator IS computed, by QuantumAlgebra, in the sister scenario scenario_greens_function_eom_commutator, which answers $[\hat H,\hat c_{m\sigma}]$ with $-\varepsilon_{m\sigma}\hat c_{m\sigma} + \sum V(\ldots)$. The line below writes that result down (with the sign turned round for $[\hat c,\hat H]$).
[\hat c_{m\sigma}, \hat H] = -\varepsilon_{m\sigma}\hat c_{m\sigma} - \sum_{p q s}\sum_{\sigma''} V_{m\sigma,p\sigma'',q\sigma'',s\sigma} \hat c^\dagger_{p\sigma''} \hat c_{q\sigma''} \hat c_{s\sigma}
@note: The Green's function with basis indices (the core of what E19 computed, with the coefficients dropped).
@note: The Green's function's equation of motion. The time derivative **also hits the step functions of the time ordering**, and that is where the equal-time anticommutator — the $\hbar\delta(t-t')$ term — comes from. The two-body term produces the expectation of **four operators**.
i\hbar \frac{\partial}{\partial t} G_{n\sigma,m\sigma'}(t,t') = \hbar\delta(t-t')\delta_{nm}\delta_{\sigma\sigma'} + \varepsilon_{n\sigma} G_{n\sigma,m\sigma'}(t,t') + \sum_{p q s}\sum_{\sigma''} V_{n\sigma,p\sigma'',q\sigma'',s\sigma} G^{(2)}_{p\sigma'',q\sigma'',s\sigma,m\sigma'}(t,t')
@note: The definition of that higher Green's function. The superscript $(2)$ is a **label**, like the $0$ of $G^0$, not an exponent. Three operators sit at $t$; only the last one sits at $t'$.
G^{(2)}_{p\sigma'',q\sigma'',s\sigma,m\sigma'}(t,t') = -i\langle\Phi_0| T[\hat c^\dagger_{p\sigma''}(t) \hat c_{q\sigma''}(t) \hat c_{s\sigma}(t) \hat c^\dagger_{m\sigma'}(t')]|\Phi_0\rangle
@note: **The equation is not closed.** The one-body equation contains a two-body Green's function, whose equation contains a three-body one — the hierarchy does not end. **An approximation is a cut in that chain.**
@note: The lowest cut (Hartree–Fock decoupling): split the four operators **into pairs** and replace the equal-time pair by an expectation value. The second term is negative because the fermions were exchanged. **This is not an equality, so it is written with $\approx$** — the approximation is this one line, and everything after it is algebra. An $\approx$ can be written and displayed, and computing with it is refused with a reason (a CAS has no "approximately equal").
G^{(2)}_{p\sigma'',q\sigma'',s\sigma,m\sigma'}(t,t') \approx \langle \hat c^\dagger_{p\sigma''} \hat c_{q\sigma''}\rangle G_{s\sigma,m\sigma'}(t,t') - \delta_{\sigma\sigma''} \langle \hat c^\dagger_{p\sigma''} \hat c_{s\sigma''}\rangle G_{q\sigma'',m\sigma'}(t,t')
@note: The basis diagonalises $\hat H_0$ (E15), so the equal-time expectation is diagonal too.
\langle \hat c^\dagger_{p\sigma''} \hat c_{q\sigma''}\rangle = \delta_{pq}\langle \hat n_{p\sigma''}\rangle
@note: Substituting collapses two of the sums, leaving one over the mode $p$ and the spin $\sigma''$. This is the self-energy — **the equation of motion has fixed $\Sigma$.** The first term is the Hartree term the Dyson route puts in by hand as its first-order approximation; the second is the exchange (Fock) term, where being a fermion first matters.
\Sigma^{(HF)}_{n\sigma,s\sigma} = \sum_{p \sigma''}\left[V_{n\sigma,p\sigma'',p\sigma'',s\sigma} - \delta_{\sigma\sigma''}V_{n\sigma,p\sigma'',s\sigma'',p\sigma}\right]\langle \hat n_{p\sigma''} \rangle
@note: The higher Green's function is gone and the equation has **closed**. It is no longer the exact $G$, though: it is the quantity that came out of the decoupling, so it is written $G^{(HF)}$.
i\hbar \frac{\partial}{\partial t} G^{(HF)}_{n\sigma,m\sigma'}(t,t') = \hbar\delta(t-t')\delta_{nm}\delta_{\sigma\sigma'} + \varepsilon_{n\sigma} G^{(HF)}_{n\sigma,m\sigma'}(t,t') + \sum_{s} \Sigma^{(HF)}_{n\sigma,s\sigma} G^{(HF)}_{s\sigma,m\sigma'}(t,t')
@note: **Fourier transform in the time difference — with no dedicated operation.** Write the inverse transform's own definition and differentiate both sides with respect to $t$. The space between $\omega$ and $(t-t')$ is the product sign (without it, $\omega$ reads as a function applied to $(t-t')$, and that reading is refused).
G^{(HF)}_{n\sigma}(t-t') = \frac{1}{2\pi}\int_{-\infty}^{\infty} G^{(HF)}_{n\sigma}(\omega) e^{-i\omega (t-t')} d\omega
@note: The derivative goes under the integral sign and brings $-i\omega$ down — that is the content of $i\hbar\partial_t \to \hbar\omega$. The same transform turns $\delta(t-t')$ into 1.
differentiate both sides of E30 with respect to t
@note: The frequency-domain equation. Both $\hat H_0$ and $\Sigma^{(HF)}$ are diagonal in this basis, so write the diagonal element. **Mind where $\hbar$ sits**: the $\Sigma$ of Dyson (E22) has the dimension of a frequency, the $\Sigma^{(HF)}$ here that of an energy, and the two differ by exactly $\hbar$.
(\omega - \varepsilon_{n\sigma}/\hbar) G^{(HF)}_{n\sigma}(\omega) = 1 + \Sigma^{(HF)}_{n\sigma} G^{(HF)}_{n\sigma}(\omega)/\hbar
@note: Solve for $G^{(HF)}$. **The solve-for variable is given in the researcher's own spelling** — label, indices and argument all still on it.
solve E32 for G^{(HF)}_{n\sigma}(\omega)
@note: The Hartree–Fock Green's function. The pole has moved from $\varepsilon_{n\sigma}$ to $\varepsilon_{n\sigma}+\Sigma^{(HF)}_{n\sigma}$ — **the interaction shifts the single-particle energy**, which is what this approximation says.
@note: Finally, set the self-energy back to zero. If the approximation was put in correctly, switching the interaction off must return the free propagator (E21, without the $i\eta$).
\Sigma^{(HF)}_{n\sigma} = 0
substitute E34 into E33
```
