## Definition

A loss function $J(\theta)$ quantifies how wrong the model's predictions are, given the current parameters. Training is the search for the $\theta$ that minimizes it.

## Why the Loss Is Not Arbitrary

The loss is the mathematical shadow of the assumed data distribution (see [[Exponential family]]):

|Distribution|Loss|Algorithm|
|---|---|---|
|Gaussian|Mean Squared Error|[[Linear Regression]]|
|Bernoulli|Binary Cross-Entropy|[[Logistic Regression]]|
|Multinomial|Categorical Cross-Entropy|[[Softmax Regression]]|

> The loss follows from the likelihood via [[Probabilistic Interpretation]] — you don't *choose* it, you *derive* it.

## Desired Properties

- **Convex** — one global optimum, so [[Gradient Descent]] reliably converges
- **Differentiable** — the gradient drives the optimization
- **Aligned with the task** — e.g., for classification, accuracy itself is non-differentiable, so we use the surrogate cross-entropy loss

## Related Notes

- [[Linear Regression]]
- [[Logistic Regression]]
- [[Probabilistic Interpretation]]
- [[Exponential family]]
- [[Gradient Descent]]
