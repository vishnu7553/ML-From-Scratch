## Definition

Learning Theory gave us $\hat\theta$ as a random variable and $\theta^*$
as an unknown constant. That is enough to state the bias-variance
tradeoff exactly, rather than by intuition: bias and variance become the
first two moments of one sampling distribution, and generalization error
splits into three named pieces.

## The Parameter View

Part 1 read bias and variance off *curves* — how well a line or wiggly
polynomial fits its points. Now read them off $\hat\theta$'s sampling
distribution, across repeated experiments on fresh draws of a fixed-size
dataset $m$:

$$\text{Bias} = \mathbb{E}[\hat\theta] - \theta^* \qquad \text{Var}[\hat\theta] = \mathbb{E}\left[\left(\hat\theta - \mathbb{E}[\hat\theta]\right)^2\right]$$

- **Bias** — where the center of the cloud of estimates sits relative to $\theta^*$.
- **Variance** — how tightly that cloud clusters.

Two named properties follow:

- If $\hat\theta \to \theta^*$ as $m \to \infty$: **consistency**.
- If $\mathbb{E}[\hat\theta] = \theta^*$ for all $m$: the estimator is **unbiased**.
- The rate at which $\text{Var}[\hat\theta] \to 0$: **statistical efficiency**.

## Decomposing the Error

Take a **hypothesis space** $\mathcal{H}$ — everything your algorithm is
able to express. Define the optimal elements in that framing:

- $g$ — the best possible hypothesis in the universe (unconstrained); it may lie outside $\mathcal{H}$. This is the Bayes-optimal predictor (unknown in general).
- $h^* \in \mathcal{H}$ — the best hypothesis *within* the hypothesis class $\mathcal{H}$ (the minimizer of true risk over $\mathcal{H}$).
- $\hat h$ — the hypothesis actually learned from the finite training set $S$ (a random variable, depending on $S$).

The gap between these three gives the standard decomposition of generalization error:

$$\varepsilon(h^*) - \varepsilon(g) = \text{approximation error} \qquad \varepsilon(\hat h) - \varepsilon(h^*) = \text{estimation error}$$

Since $\varepsilon(g)$ is irreducible (Bayes error), the true risk of the learned hypothesis is:

$$\boxed{\varepsilon(\hat h) = \underbrace{\text{estimation error}}_{\text{variance}} + \underbrace{\text{approximation error}}_{\text{bias}} + \varepsilon(g)}$$

Widening $\mathcal{H}$ typically lowers approximation error (reduces bias),
but raises estimation error (increases variance): more room for $\hat h$
to fluctuate with the training sample. Narrowing $\mathcal{H}$ does the
opposite. This is the bias–variance tradeoff.

This is also *why* [[Regularization]] reduces variance: it shrinks the
effective hypothesis space by penalizing large $\theta$'s, trading a
small increase in bias for a larger reduction in variance:

$$\text{approximation error} \propto \text{bias} \qquad \text{estimation error} \propto \text{variance}$$

## Implementation

- [[../Implementation/Regression/bias_variance_tradeoff.py]]

---

## Related Notes
- [[Learning Theory]]
- [[Empirical Risk Minimization]]

## References
- *Mathematics for Machine Learning* — Deisenroth et al. (mml-book.github.io)
- Stanford CS229 Lecture Notes — cs229.stanford.edu
- *Dive into Deep Learning* — d2l.ai

---

**Next:** [[Empirical Risk Minimization]]
