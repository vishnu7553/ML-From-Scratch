## Definition

The Exponential Family is a class of probability distributions that share a common mathematical "DNA." Instead of viewing each distribution (Gaussian, Bernoulli, Poisson, etc.) as a random, isolated formula, we recognize them as specific members of a single, universal template.

## Intuition

Every "different" loss function you've derived so far — MSE for [[Linear Regression]], Binary Cross-Entropy for [[Logistic Regression]] — wasn't actually a separate invention. They're all the same underlying template, wearing different distributional clothes. The Exponential Family is that template.

## The Canonical Blueprint

Any distribution in this family can be written in this precise form:

$$\boxed{P(y; \eta) = b(y) \cdot \exp\left(\eta^T T(y) - a(\eta)\right)}$$

|Term|Name|Meaning|
|---|---|---|
|$\eta$|Natural parameter|The "control knob" — the parameter the model actually learns. Dictates the distribution's behavior. e.g. $\eta = \theta^T x$|
|$T(y)$|Sufficient statistic|The core data that matters. In most ML cases, $T(y) = y$ (the actual data itself)|
|$a(\eta)$|Log-partition function|The normalizer — ensures total probability integrates to 1|
|$b(y)$|Base measure|A fixed "baseline" function of $y$ that doesn't depend on $\eta$ — effectively a scalar for the probability|

## Why This Matters

Picking a distribution is not arbitrary — it dictates the entire learning process. From the Exponential Family we get a probability density function (PDF) for the distribution we chose, so the natural instinct is to take the log of that PDF to get the log-likelihood. When we take the log of a PDF, **we define our loss function.**

The loss function falls out of the log-likelihood based on which distribution (Gaussian, Bernoulli, etc.) we chose. We don't _choose_ the loss function — it's the mathematical shadow cast by the distribution we selected:

- If you assume **Gaussian** → Mean Squared Error follows naturally
- If you assume **Bernoulli** → Logistic loss (Cross-Entropy) follows naturally

## Worked Example — Bernoulli

Take the Bernoulli distribution — it models the probability of a certain event happening or not, with $\phi$ = probability of the event:

$$p(y; \phi) = \phi^y \cdot (1-\phi)^{(1-y)}$$

Apply $\exp(\log(\cdot))$ to both sides to force it into exponential form:

$$= \exp\left(\log\left(\phi^y \cdot (1-\phi)^{(1-y)}\right)\right)$$

$$= \exp\left[y\log(\phi) + (1-y)\log(1-\phi)\right]$$

$$= \exp\left[y\log(\phi) - y\log(1-\phi) + \log(1-\phi)\right]$$

$$\boxed{p(y;\phi) = \exp\left[y \cdot \log\left(\frac{\phi}{1-\phi}\right) + \log(1-\phi)\right]}$$

Now compare against the canonical template $p(y;\eta) = b(y)\exp(\eta^T T(y) - a(\eta))$ — matching term by term:

$$b(y) = 1 \qquad T(y) = y \qquad \eta = \log\left(\frac{\phi}{1-\phi}\right) \implies \phi = \frac{1}{1+e^{-\eta}}$$

$$a(\eta) = \log(1-\phi) = \log\left(1 - \frac{1}{1+e^{-\eta}}\right) = \log(1+e^{\eta})$$

> Notice $\phi = \frac{1}{1+e^{-\eta}}$ — that's the sigmoid function, falling straight out of the algebra, not chosen by hand.

## Worked Example — Gaussian

Assuming $\sigma^2 = 1$ for simplicity, and $\mu = \theta^T x$:

$$p(y;\mu) = \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{(y-\mu)^2}{2}\right) = \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{y^2 + \mu^2 - 2\mu y}{2}\right)$$

$$= \frac{1}{\sqrt{2\pi}} \cdot e^{-y^2/2} \cdot \exp\left(\mu y - \frac{\mu^2}{2}\right)$$

Matching to the template:

$$b(y) = \frac{1}{\sqrt{2\pi}} e^{-y^2/2} \qquad T(y) = y \qquad \eta = \mu \qquad a(\eta) = \frac{\mu^2}{2} = \frac{\eta^2}{2}$$

## Properties

1. **Maximising Log Likelihood Estimation MLE with respect to $\eta$ is concave** (equivalently, Negative Log-Likelihood with respect to $\eta$ is convex) — this is _why_ gradient-based optimization reliably converges for every GLM: there's only one global optimum to find, never multiple local ones.
2. $\mathbb{E}[y;\eta] = \frac{\partial}{\partial \eta} a(\eta)$ — the mean of the distribution is the first derivative of the log-partition function.
3. $\text{Var}[y;\eta] = \frac{\partial^2}{\partial \eta^2} a(\eta)$ — the variance is the second derivative.

## Members of the Family

|Data Type|Distribution|
|---|---|
|Real-valued|Gaussian|
|Binary|Bernoulli|
|Count|Poisson|
|Real positive ($\mathbb{R}^+$)|Gamma, Exponential|
|Probability of a probability|Beta, Dirichlet (Bayesian)|

---

## Related Notes

- [[Generalized linear models]]
- [[Linear Regression]]
- [[Logistic Regression]]
- [[Probabilistic Interpretation]]
- [[Softmax Regression]]

## References

- _Mathematics for Machine Learning_ — Deisenroth et al. (mml-book.github.io)
- Stanford CS229 Lecture Notes — cs229.stanford.edu
- _Dive into Deep Learning_ — d2l.ai