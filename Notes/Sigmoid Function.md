## Definition

The sigmoid (logistic) function squashes any real number into the range $(0, 1)$:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

## Properties

- Output always lies strictly between 0 and 1
- Smooth and differentiable everywhere
- Saturates near 0 and 1 — extreme inputs don't shift the boundary

## Where It Comes From and Appears

- The **backbone** of [[Logistic Regression]] — converts the linear score $\theta^T x$ into a probability.
- Falls out of the **algebra** of the [[Exponential family]] Bernoulli member, not chosen by hand.
- Emerges "for free" from [[Gaussian Discriminant Analysis]] — two Gaussians + Bayes' Rule produce an S-shaped $p(y=1|x)$.

## Related Notes

- [[Logistic Regression]]
- [[Exponential family]]
- [[Gaussian Discriminant Analysis]]
