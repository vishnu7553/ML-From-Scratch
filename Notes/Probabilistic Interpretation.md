## Definition

The Probabilistic Interpretation answers a question usually left unexamined: **why do we minimize squared error at all?** It shows that Least Squared Error (LSE) is not an arbitrary convenient choice — it falls directly out of assuming your prediction errors are Gaussian and deriving the parameters via Maximum Likelihood Estimation (MLE).

## Why Use Least Squared Error?

Least Squared Error (or Mean Squared Error) is the most common loss family in machine learning — but why this family, and not some other error measure?

Start by assuming any linear algorithm's true relationship is:

$$y^{(i)} = \theta^T x^{(i)} + \varepsilon^{(i)}, \qquad \varepsilon^{(i)} \sim \mathcal{N}(0, \sigma^2)$$

Where $\varepsilon^{(i)}$ is an **error term** capturing unmodeled effects or random noise — e.g. in house price prediction, this might be the seller's mood or the weather on closing day. Things you can't feature-engineer away.

## The IID Assumption

We assume every $\varepsilon^{(i)}$ is **IID** — Independently and Identically Distributed. Concretely: $\varepsilon^{(1)}$ is independent of $\varepsilon^{(2)}$, but both are drawn from the identical distribution.

This assumption is what lets the probability of the whole dataset factor into a clean product later — independence turns a joint probability into a multiplication.

Because the noise is Gaussian, the prediction $y^{(i)}$ given $x^{(i)}$ and $\theta$ is itself Gaussian-distributed:

$$p(y^{(i)} \mid x^{(i)}; \theta) = \frac{1}{\sigma \sqrt{2\pi}} \exp\left(-\frac{(y^{(i)} - \theta^T x^{(i)})^2}{2\sigma^2}\right)$$

$$p(y^{(i)} \mid x^{(i)}; \theta) \sim \mathcal{N}(\theta^T x^{(i)}, \sigma^2)$$

In plain words: we're adding noise $\sigma^2$ to the linear equation $\theta^T x^{(i)}$, and treating the probability of $y^{(i)}$ occurring as a normal distribution centered on our prediction.

## Likelihood vs. Probability — Why Two Words for "The Same Thing"?

> This distinction is easy to gloss over, but it matters for correctly reading any ML paper afterward.

At first glance $\mathcal{L}(\theta) = p(y \mid x; \theta)$ looks like we're just renaming probability. We aren't:

||Varying|Fixed|Called|
|---|---|---|---|
|**Likelihood** $\mathcal{L}(\theta)$|Parameters $\theta$|Data|"Likelihood of the parameters"|
|**Probability** $p(\cdot)$|Data|Parameters|"Probability of the data"|

- **Likelihood of parameters** ✅ — **Likelihood of data** ❌ — we're estimating parameters, not data.
- **Probability of data** ✅ — **Probability of parameters** ❌ — same reasoning, reversed.

So: when $\theta$ varies and the dataset is fixed, it's called $\mathcal{L}(\theta)$. When $\theta$ is fixed and the data point varies (like the query point in [[Locally weighted regression]]), it's called a probability function $p(\cdot)$. Same underlying math, different name depending on what's being held constant.

## Maximum Likelihood Estimation

To estimate the optimal $\theta$, define the likelihood as the probability of observing the entire training set:

$$\mathcal{L}(\theta) = p(y \mid x; \theta) = \prod_{i=1}^{m} p(y^{(i)} \mid x^{(i)}; \theta)$$

> **Why a product ($\prod$) and not a sum ($\sum$)?** Because the IID assumption means every error term is independent of the others — and the probability of several independent events all happening is the product of their individual probabilities.

Substituting the Gaussian PDF:

$$\mathcal{L}(\theta) = \prod_{i=1}^{m} \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(y^{(i)} - \theta^T x^{(i)})^2}{2\sigma^2}\right)$$

## Log-Likelihood

Products are hard to differentiate — apply $\log$ to convert the product into a sum, using $\log(A \cdot B) = \log(A) + \log(B)$ and $\log(\prod) = \sum$:

$$\ell(\theta) = \log(\mathcal{L}(\theta)) = m\log\left(\frac{1}{\sigma\sqrt{2\pi}}\right) + \sum_{i=1}^{m} -\frac{(y^{(i)} - \theta^T x^{(i)})^2}{2\sigma^2}$$

## Dropping Constants

The goal is to **maximize** the likelihood estimation of $\theta$ — equivalently, choose $\theta$ to optimize $\ell(\theta)$. Since $m\log\left(\frac{1}{\sigma\sqrt{2\pi}}\right)$ and $2\sigma^2$ don't depend on $\theta$ at all, they're constants with respect to the optimization and can be dropped:

$$\boxed{\ell(\theta) = -\frac{1}{2}\sum_{i=1}^{m}\left(y^{(i)} - \theta^T x^{(i)}\right)^2}$$

## The Punchline

What's left is, up to a constant factor and a sign flip, **exactly the least squares objective**. Maximizing $\ell(\theta)$ is the same as minimizing $\sum (y^{(i)} - \theta^T x^{(i)})^2$ — which is precisely [[Linear Regression]]'s MSE loss.

This concludes _when_ to reach for least squared error as your loss function:

- When the error is assumed to follow a **Normal Distribution** under the **IID** assumption
- And the goal is to estimate the **likelihood** of $\theta$, i.e. $\mathcal{L}(\theta)$

> Least squares was never an arbitrary convenience — it's the mathematical consequence of assuming Gaussian noise. Choose a different noise distribution (e.g. Bernoulli, for classification) and a _different_ loss function falls out just as naturally. This is exactly the same "distribution → natural loss" logic behind [[Generalized linear models]].

Now we're ready to jump to classification.

---

## Related Notes

- [[Linear Regression]]
- [[Locally weighted regression]]
- [[Logistic Regression]]
- [[Generalized linear models]]
- [[Loss Functions]]

## References

- _Mathematics for Machine Learning_ — Deisenroth et al. (mml-book.github.io)
- Stanford CS229 Lecture Notes — cs229.stanford.edu
- _Dive into Deep Learning_ — d2l.ai