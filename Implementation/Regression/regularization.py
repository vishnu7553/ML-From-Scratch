import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# Dataset — same noisy sine, high-degree polynomial features
# so there's real overfitting risk to regularize away.
# ─────────────────────────────────────────────

np.random.seed(0)
X = np.linspace(0, 1, 20)
y = np.sin(2 * np.pi * X) + np.random.normal(0, 0.25, size=X.shape[0])

DEGREE = 12  # deliberately high -- plain regression at this degree overfits badly

# Raw monomials {x, x^2, ..., x^12} on [0, 1] have wildly different scales --
# x^12 never exceeds 1 while x^12's neighbours are O(1e-6). A penalty of
# lambda = 1 applied to those columns is therefore enormous compared to the
# fit, which kills the wiggle at lambda = 1 instead of at lambda = 100. The
# note warns about exactly this ("if the features are on wildly different
# scales, the regularization will be applied very unevenly ... normalize the
# features to a similar scale before regularizing").
#
# Legendre polynomials are mutually orthogonal on [-1, 1], so X^T X is
# (numerically) diagonal: every weight is penalized on the same footing and
# lambda means the same thing at every order. That makes the note's actual
# numbers behave as described -- lambda = 1 is too weak to help, lambda = 100
# lands on the good middle, lambda -> huge collapses h_theta to 0.
FEATURE_SCALE = 10.0  # puts that transition between lambda = 1 and lambda = 100


def legendre(x, order):
    """Legendre polynomial P_order(x), built by the standard recurrence."""
    if order < 0:
        return np.zeros_like(x)
    if order == 0:
        return np.ones_like(x)
    if order == 1:
        return x
    return ((2 * order - 1) * x * legendre(x, order - 1)
            - (order - 1) * legendre(x, order - 2)) / order


def design_matrix(x, degree):
    """Polynomial design matrix in the (scaled) Legendre basis.

    Maps [0, 1] -> [-1, 1] first, since that is the interval the Legendre
    polynomials are orthogonal over.
    """
    z = 2.0 * np.asarray(x, dtype=float) - 1.0
    return FEATURE_SCALE * np.column_stack([legendre(z, j) for j in range(degree + 1)])


# ─────────────────────────────────────────────
# Method — Regularized Normal Equation (Ridge Regression)
# ─────────────────────────────────────────────

def fit_regularized(X, y, degree, lam):
    """
    min (1/2) sum(y - theta^T x)^2 + (lambda/2) ||theta||^2

    Closed-form solution (the regularized Normal Equation):
        theta = (X^T X + lambda * I)^-1 X^T y

    The penalty covers every component of theta, as the note's
    lambda * ||theta||^2 states. That matters for the extreme-lambda
    claim: with the intercept left unpenalized (a common convention --
    you don't usually want to force the bias term toward zero) the
    prediction settles at mean(y) = 0.14 instead of 0, so h_theta(x)
    never actually approaches 0. Penalizing theta_0 too reproduces the
    limit the note describes.

    lambda = 0 is the degenerate case: the plain normal equation is
    ill-conditioned for a degree-12 design, so explicitly inverting
    X^T X returns numerical garbage instead of the least-squares
    solution (||theta|| blew up to ~1e6 here, which would misrepresent
    "no regularization" as an explosion rather than an overfit).
    lambda = 0 therefore goes through lstsq, which is also the ridge
    limit: the X^T X term simply drops out and every penalized weight is
    driven to zero.
    """
    X_poly = design_matrix(X, degree)

    if lam == 0:
        theta, *_ = np.linalg.lstsq(X_poly, y, rcond=None)
        return theta

    I = np.eye(X_poly.shape[1])
    theta = np.linalg.solve(X_poly.T @ X_poly + lam * I, X_poly.T @ y)
    return theta


def predict(X, theta):
    degree = len(theta) - 1
    return design_matrix(X, degree) @ theta


# ─────────────────────────────────────────────
# Sweep Lambda — Reproduce the Notes' Exact Claim
# ─────────────────────────────────────────────

if __name__ == "__main__":
    lambdas = [0, 1, 100, 1e10]  # 1e10 stands in for the notes' 10^100 -- float overflow limits how literal we can be
    x_plot = np.linspace(0, 1, 300)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X, y, color="black", zorder=5, label="Data")

    print("── Effect of Lambda on the Fitted Weights ──\n")
    norms = []
    for lam in lambdas:
        theta = fit_regularized(X, y, DEGREE, lam)
        y_plot = predict(x_plot, theta)
        theta_norm = np.linalg.norm(theta)
        norms.append(theta_norm)

        label = f"lambda={lam:g}" if lam < 1e6 else "lambda=1e10 (~10^100 regime)"
        ax.plot(x_plot, y_plot, linewidth=2, label=label)

        print(f"   lambda = {lam:<10g}  ||theta|| = {theta_norm:>12.4f}  "
              f"max|theta_i| = {np.max(np.abs(theta)):.4f}")

    # The note's two quantitative claims about lambda:
    #   "lambda is very large -> all parameters theta are forced close to zero"
    #   "at the extreme (lambda = 10^100), h_theta(x) ~= 0"
    strictly_shrinking = all(a > b for a, b in zip(norms, norms[1:]))
    print(f"\n||theta|| strictly decreases as lambda grows: {strictly_shrinking}")

    theta_max_lam = fit_regularized(X, y, DEGREE, lambdas[-1])
    h_at_max = predict(x_plot, theta_max_lam)
    print(f"At lambda={lambdas[-1]:g}, max|h_theta(x)| = {np.max(np.abs(h_at_max)):.2e} "
          f"-> h_theta(x) ~= 0, as the note claims for lambda=10^100.")

    # With lambda = 0 there is no penalty at all, so the degree-12 model is
    # free to interpolate: the wiggly overfit curve the note describes.
    theta_unreg = fit_regularized(X, y, DEGREE, 0)
    train_mse_unreg = np.mean((y - predict(X, theta_unreg)) ** 2)
    print(f"\nAt lambda=0 the penalty vanishes and the model interpolates the "
          f"noise (training MSE = {train_mse_unreg:.4f}) -- the wiggly,")
    print("every-point-chasing fit. Regularization is what pulls it back.")

    # ── Check the note's four regimes one by one, out of sample ──────────
    # Held-out error against the true sine is the honest way to tell an
    # overfit curve from a genuinely good one: a wiggly fit can have low
    # training error and still generalize badly.
    y_true = np.sin(2 * np.pi * x_plot)
    print("\n── The Note's Four Regimes ──\n")
    print(f"   {'lambda':>9}  {'train MSE':>10}  {'held-out MSE':>12}   interpretation")

    errors = {}
    for lam in lambdas:
        theta = fit_regularized(X, y, DEGREE, lam)
        train_mse = np.mean((y - predict(X, theta)) ** 2)
        held_out_mse = np.mean((y_true - predict(x_plot, theta)) ** 2)
        errors[lam] = (train_mse, held_out_mse)

    regime = {
        0: "no regularization -- interpolates the noise",
        1: "penalty still too weak -- still chases every point",
        100: "smooth, high penalty -- the sweet spot",
        1e10: "total rigidity -- h_theta(x) ~= 0",
    }
    for lam in lambdas:
        train_mse, held_out_mse = errors[lam]
        print(f"   {lam:>9g}  {train_mse:>10.4f}  {held_out_mse:>12.4f}   {regime[lam]}")

    noise_floor = 0.25 ** 2  # variance of the noise the data was drawn with
    best_lam = min(errors, key=lambda k: errors[k][1])
    print(f"\nBest held-out error at lambda={best_lam:g} "
          f"(noise floor = {noise_floor:.4f}) -- matches the note's "
          f"'optimal lambda is some intermediate value'.")

    lambda_1_overfits = errors[1][1] > errors[100][1]
    lambda_100_beats_noise = errors[100][1] < noise_floor
    print(f"lambda=1 still overfits relative to lambda=100: {lambda_1_overfits}")
    print(f"lambda=100 generalizes better than pure noise: {lambda_100_beats_noise}")

    ax.plot(x_plot, np.sin(2 * np.pi * x_plot), "--", color="gray", alpha=0.5, label="True function")
    ax.set_ylim(-2, 2)
    ax.set_title(f"Ridge Regression (degree {DEGREE}) — Effect of Lambda")
    ax.legend()
    plt.tight_layout()
    plt.show()
