## Definition

Uniform Convergence is the phenomenon of a model's empirical risk
converging consistently toward its true generalization error, as sample
size grows — and the mathematical machinery used to prove it. It
answers two central questions: (1) $\hat\varepsilon(h)$ vs $\varepsilon(h)$
— training error vs. generalization error, and (2) $\varepsilon(\hat h)$
vs $\varepsilon(h^*)$ — the generalization error we achieve vs. the best
possible within our hypothesis class.

## Two Tools

**Union Bound:** for events $A_1,\dots,A_k$ (need not be independent):

$$P(A_1 \cup A_2 \cup \dots \cup A_k) \leq P(A_1) + P(A_2) + \dots + P(A_k)$$

**Hoeffding's Inequality:** let $Z_1,\dots,Z_m \sim \text{Bernoulli}(\phi)$,
$\hat\phi = \frac{1}{m}\sum Z_i$, and $\gamma > 0$ (the margin):

$$\boxed{P(|\hat\phi - \phi| > \gamma) \leq 2\exp(-2\gamma^2 m)}$$

As $m$ increases, the probability of the estimate deviating from the
true value shrinks exponentially.

## Connecting to a Single Hypothesis

For one sample $h_i$: $\varepsilon(h_i)$ is the generalization error, and
$\hat\varepsilon(h_i)$ the empirical (estimation) error. Applying
Hoeffding directly:

$$\boxed{P\left(|\hat\varepsilon(h_i) - \varepsilon(h_i)| > \gamma\right) \leq 2\exp(-2\gamma^2 m)}$$

The gap between generalization error and empirical error being greater
than margin $\gamma$ is bounded by $2\exp(-2\gamma^2 m)$. Increase $m$,
and this bound — and the empirical curve itself — concentrates more
tightly around the true generalization curve. This convergence of
empirical risk toward generalization error is **uniform convergence**.

## Uniform Convergence for Finite Hypothesis Classes

The above is pointwise — for one fixed $h$. To bound *every* $h$ in a
finite class $\mathcal{H}$ of size $k$ simultaneously, combine Hoeffding
with the Union Bound:

$$\text{With probability} \geq 1-\Delta,\ \text{for all } h \in \mathcal{H}: \quad |\varepsilon(h) - \hat\varepsilon(h)| \leq \gamma, \quad \text{where } \Delta = 2k\exp(-2\gamma^2 m)$$

**Sample complexity:** solving for $m$ given a target $\Delta$ and
margin $\gamma$:

$$\boxed{m \geq \frac{1}{2\gamma^2}\log\left(\frac{2k}{\Delta}\right)}$$

This guarantees that with enough data $m$, empirical risk converges
uniformly to the true generalization error curve across the *entire*
hypothesis class.

## The Generalization Bound

We want to bound $\varepsilon(\hat h)$ — our learned [[Empirical Risk Minimization|ERM]]
hypothesis's error — relative to $\varepsilon(h^*)$, the best possible
within $\mathcal{H}$.

$$\varepsilon(\hat h) \leq \varepsilon(h^*) + \gamma \quad \text{[by uniform convergence]}$$
$$\hat\varepsilon(\hat h) \leq \hat\varepsilon(h^*) \quad \text{[since ERM minimizes empirical risk]}$$
$$\hat\varepsilon(h^*) \leq \varepsilon(h^*) + \gamma \quad \text{[uniform convergence applied to } h^*\text{]}$$

Combining all three:

$$\boxed{\varepsilon(\hat h) \leq \varepsilon(h^*) + 2\gamma}$$

With probability at least $1-\Delta$, the generalization error of our
ERM hypothesis is bounded by the best-in-class error plus $2\gamma$.

## Implementation

- [[../Implementation/Generalization/hoeffding_uniform_convergence.py]]

---

## Related Notes
- [[Learning Theory]]
- [[Empirical Risk Minimization]]
- [[Bias-Variance Tradeoff]]
- [[VC Dimension]]

## References
- *Mathematics for Machine Learning* — Deisenroth et al. (mml-book.github.io)
- Stanford CS229 Lecture Notes — cs229.stanford.edu
- *Dive into Deep Learning* — d2l.ai

---

**Next:** [[VC Dimension]]
