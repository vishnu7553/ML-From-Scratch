## Definition

- **Underfitting** — the model is too simple to capture the structure in the data (high bias): it performs poorly on *both* training and new data.
- **Overfitting** — the model is too flexible and memorizes noise (high variance): it performs very well on training data but generalizes poorly.

## How It Shows Up in This Vault

- [[Locally weighted regression]] — bandwidth $\tau$ too large underfits (smooth, biased); too small overfits (jagged, noisy).
- [[Linear Regression]] — more features / higher-degree polynomials risk overfitting.
- [[Logistic Regression]] — a too-strong model boundary chases every outlier.

## The Bias–Variance Trade-off

Every model trades these two failure modes. The goal is the sweet spot where total error (bias² + variance + irreducible noise) is minimized — often found by tuning a single hyperparameter like $\tau$ (LWR) or $C$ (SVM soft margin).

## Related Notes

- [[Supervised Learning]]
- [[Linear Regression]]
- [[Logistic Regression]]
- [[Locally weighted regression]]
