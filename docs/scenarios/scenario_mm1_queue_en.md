# Practice-loop sample scenario: M/M/1 queue (utilization, mean number in system, Little's law)

A detailed write-up of practice-loop scenario-menu item **S-r (operations & management
science)**. For an M/M/1 queue with arrival rate $\lambda$ and service rate $\mu$, enter
the **utilization** $\rho=\lambda/\mu$ and the **mean number in system**
$L=\rho/(1-\rho)$, substitute $\rho$ to get $L=\lambda/(\mu-\lambda)$, and then **divide
both sides by $\lambda$** to derive Little's law $W=L/\lambda=1/(\mu-\lambda)$ (the mean
time in system).

**Backend:** standard (Maxima) / **Domain:** operations & management science
(operations_research).

> As a mathematical model, M/M/1 closes under **substitution and simplification alone**,
> which makes it an easy minimal showcase for a CAS-assisted notebook. Every operation
> is interpreted rule-based (no LLM needed).

## Model background

A single-server queue (M/M/1) with Poisson arrivals (rate $\lambda$) and exponential
service times (rate $\mu$). In the steady state ($\rho=\lambda/\mu<1$):

- utilization $\rho=\dfrac{\lambda}{\mu}$
- mean number in system $L=\dfrac{\rho}{1-\rho}=\dfrac{\lambda}{\mu-\lambda}$
- mean time in system $W=\dfrac{L}{\lambda}=\dfrac{1}{\mu-\lambda}$ (**Little's law**
  $L=\lambda W$)

This scenario substitutes $\rho=\lambda/\mu$ into the definition of $L$, then applies
Little's law by **dividing both sides by $\lambda$** to obtain $W$.

## Session setup and symbols

- mode **standard (Maxima)**, domain **operations & management science**.
- enter $\lambda,\mu,\rho$ in TeX notation (`\lambda`, etc.).

## Input steps

### Step 1: define the utilization (H-1)

```
\rho = \lambda/\mu
```

**Expected output:** registered as E1. $\rho=\lambda/\mu$.

### Step 2: define the mean number in system (H-1)

```
L = \rho/(1-\rho)
```

**Expected output:** registered as E2. $L=\rho/(1-\rho)$.

### Step 3: substitute $\lambda/\mu$ for $\rho$ (symbol→value substitution)

```
In E2, substitute \lambda/\mu for \rho
```

**Expected output:** E3 $= L=-\dfrac{\lambda}{\lambda-\mu}\ \left(=\dfrac{\lambda}{\mu-\lambda}\right)$.

### Step 4: divide both sides by $\lambda$ (Little's law, H-8)

```
Divide both sides of E3 by \lambda
```

**Expected output:** E4 $= \dfrac{L}{\lambda}=-\dfrac{1}{\lambda-\mu}\ \left(=\dfrac{1}{\mu-\lambda}=W\right)$.

> The sign is shown in Maxima's canonical form ($-1/(\lambda-\mu)$). For $\mu>\lambda$
> (the steady-state condition) it is positive and equals $W=1/(\mu-\lambda)$.

## Checkpoints

- substituting $\rho=\lambda/\mu$ into $L=\rho/(1-\rho)$ gives $\lambda/(\mu-\lambda)$
  (step 3).
- Little's law $L=\lambda W$ is obtained by dividing both sides of $L$ by $\lambda$:
  $W=L/\lambda=1/(\mu-\lambda)$ (step 4). This confirms that symbol→value substitution
  (H-1 family) and both-sides operations (H-8) work correctly **even with a Greek
  operand**.

## Example files

The session after execution and the exported document:
[../../domains/operations_research/examples/scenario_mm1_queue_en.json](../../domains/operations_research/examples/scenario_mm1_queue_en.json)
and `.tex` (the Japanese variant is `_jp.{json,tex}`).

**Regenerate:** `PYTHONPATH=. .venv/bin/python scripts/generate_scenario_examples.py`

## generator コマンド（この `scenario` ブロックが生成の単一ソース）

```scenario domain=operations_research
@note: Consider an M/M/1 queue with arrival rate $\lambda$ and service rate $\mu$. Define the utilization $\rho=\lambda/\mu$ (the fraction of time the server is busy).
$\rho = \lambda/\mu$
@note: Define the mean number in system $L=\rho/(1-\rho)$.
$L = \rho/(1-\rho)$
@note: Replace $\rho$ with $\lambda/\mu$ so that $L$ is expressed purely in terms of $\lambda,\mu$.
In E2, substitute \lambda/\mu for \rho
@note: To apply Little's law $L=\lambda W$, divide both sides by the arrival rate $\lambda$ to obtain the mean time in system $W=L/\lambda$.
Divide both sides of E3 by \lambda
```
