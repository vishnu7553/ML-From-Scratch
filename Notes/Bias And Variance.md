## Definition

The Bias-Variance Tradeoff is the fundamental tension between a model
being too simple to capture the true pattern in the data (**bias**) and
being too sensitive to the noise in whatever data it happened to see
(**variance**) — two failure modes that pull in opposite directions as
model complexity changes.

## From Data to a Model

Data is a collection of raw facts, numbers, words, or measurements that
doesn't have much meaning on its own. Once organized so patterns and
trends become visible, it turns into useful information — and that's
exactly what fitting a model to data is: finding the pattern buried in
the raw numbers.

We've already seen how to fit a model — a classification or a
regression — to data and make predictions on unseen data. That raises
the natural next question: does *any* model with less training error
count as a good fit, or even a perfect fit? To answer that, look at what
happens when the same data gets fit by models of increasing complexity.

## Three Fits, One Dataset

Consider fitting three different models to the same data:

![[bv_test.png]]

Training MSE falls as complexity rises — degree 1 gives 0.2049, degree 3
gives 0.0367, degree 15 gives 0.0086 — yet the degree-15 fit generalizes
worst of the three on held-out data. That gap between training and
generalization error is the whole story of this note.

**Graph (a) — high bias, low variance:**
- Very rigid (no flexibility)
- Minimal loss, but not the lowest possible
- **Underfit** — the model can do better

**Graph (b):**
- Just enough flexibility
- Very minimal loss
- **Just right** — generalized

**Graph (c) — high variance, low bias:**
- Extremely flexible
- Zero loss with respect to training data
- **Overfit** — cannot generalize to a new point

## Bias

**Bias** is the strong initial assumptions or rigid preconceptions that
a learning algorithm holds about how data should be fit (such as
assuming a strictly linear relationship).

> High bias leads directly to underfitting (Graph a), where the model
> lacks the flexibility to capture the true underlying trends in the
> data.

## Variance

**Variance** is the extreme sensitivity and variability of a model's
predictions when trained on slightly different random draws of data,
caused by an overly complex hypothesis (such as a high-order
polynomial).

> High variance leads directly to overfitting (Graph c), where the
> model achieves zero training loss but fails completely to generalize
> to new, unseen data.

## The Relationship Between Bias and Variance

They sit on opposite ends of the model complexity spectrum, representing
a fundamental trade-off: models with high bias tend to underfit because
they are too simple, whereas models with high variance tend to overfit
because they are overly complex. Adjusting model complexity or applying
[[Regularization]] helps balance this interplay so the algorithm avoids
both extremes.

## Implementation

- [[../Implementation/Regression/bias_variance_tradeoff.py]]

---

## Related Notes
- [[Overfitting and Underfitting]]
- [[Regularization]]
- [[Model Selection and Cross-Validation]]
- [[Linear Regression]]

## References
- *Mathematics for Machine Learning* — Deisenroth et al. (mml-book.github.io)
- Stanford CS229 Lecture Notes — cs229.stanford.edu
- *Dive into Deep Learning* — d2l.ai

---

**Next:** [[Regularization]]
