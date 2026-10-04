## Definition

VC Dimension extends [[Uniform Convergence]] to hypothesis classes with
an **infinite** number of hypotheses (e.g. neural networks, linear
decision boundaries) — cases where the finite class size $k$ used in the
earlier bound no longer applies. VC dimension $d_{VC}$ replaces $k$ as
the measure of a hypothesis class's effective complexity.

## The Generalization Bound via VC Dimension

$$\boxed{\varepsilon(\hat h) \leq \hat\varepsilon(\hat h) + O\left(\sqrt{\frac{d_{VC}\log(m/d_{VC})}{m}}\right)}$$

The gap between true and empirical risk shrinks at rate
$O\left(\sqrt{d_{VC}\log(m/d_{VC})/m}\right)$ as $m \to \infty$. This
bound holds uniformly over all hypotheses in the class (with high
probability), and follows from **Sauer-Shelah's Lemma** (which bounds
the growth function by $(me/d_{VC})^{d_{VC}}$), combined with Hoeffding's
Inequality and a symmetrization argument.

## Core Takeaway

The required sample complexity scales proportionally with the VC
dimension of the hypothesis space — confirming that more expressive
(higher-capacity) models require more data to generalize effectively.
This is the formal version of the intuition from
[[Bias-Variance Tradeoff]]: a bigger hypothesis class reduces bias but
demands more data to keep variance in check.

## Implementation

- [[../Implementation/Generalization/vc_dimension_shattering.py]]

---

## Related Notes
- [[Uniform Convergence]]
- [[Learning Theory]]
- [[Bias-Variance Tradeoff]]

## References
- *Mathematics for Machine Learning* — Deisenroth et al. (mml-book.github.io)
- Stanford CS229 Lecture Notes — cs229.stanford.edu
- *Dive into Deep Learning* — d2l.ai

---

**Next:** [[Model Selection and Cross-Validation]]
