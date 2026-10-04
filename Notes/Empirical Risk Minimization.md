## Definition

Empirical Risk Minimization (ERM) is a learning algorithm — like
gradient descent — not a model. It picks the hypothesis within a class
$\mathcal{H}$ that minimizes the average error on the training set.

$$\boxed{\hat h_{ERM} = \arg\min_{h \in \mathcal{H}}\ \frac{1}{m}\sum_{i=1}^{m}\mathbb{1}\{h(x^{(i)}) \neq y^{(i)}\}}$$

> In gradient ascent/descent you **maximize** correct predictions. In
> ERM you **minimize** the miss rate — incorrect predictions. Same goal,
> opposite-signed framing.

## Risk / Generalization Error

$$\varepsilon(h) = \mathbb{E}_{(x,y)\sim D}\left[\mathbb{1}\{h(x) \neq y\}\right]$$

This is the **true risk** — expectation over the *entire* distribution
$D$, which we never fully observe.

$$\hat\varepsilon_S(h) = \frac{1}{m}\sum_{i=1}^{m}\mathbb{1}\{h(x^{(i)}) \neq y^{(i)}\}$$

This is the **empirical risk** — computed on a finite sample $S$, where
the indicator $\mathbb{1}\{\cdot\}$ outputs 1 when the prediction is
wrong.

## ERM vs. Gradient Descent — Not Opposites

They're easy to conflate, but they serve different roles in the
pipeline:

- **ERM is the "What":** it defines the *goal* — find a hypothesis
  within a class that minimizes average training error.
- **Gradient Descent is the "How":** the optimization algorithm used to
  actually reach that goal, once the ERM objective (the loss function)
  is defined.

## Implementation

- [[../Implementation/Generalization/erm_vs_true_risk.py]]

---

## Related Notes
- [[Learning Theory]]
- [[Bias-Variance Tradeoff]]
- [[Uniform Convergence]]

## References
- *Mathematics for Machine Learning* — Deisenroth et al. (mml-book.github.io)
- Stanford CS229 Lecture Notes — cs229.stanford.edu
- *Dive into Deep Learning* — d2l.ai

---

**Next:** [[Uniform Convergence]]
