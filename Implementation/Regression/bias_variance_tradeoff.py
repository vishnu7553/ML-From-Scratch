import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

# A noisy quadratic-ish curve -- enough structure that a straight line
# genuinely underfits, and enough noise that a high-degree polynomial
# genuinely overfits by chasing it.
np.random.seed(0)
X = np.linspace(0, 1, 20)
y_true = np.sin(2 * np.pi * X)
y = y_true + np.random.normal(0, 0.25, size=X.shape[0])


# ─────────────────────────────────────────────
# Method — Polynomial Fit via Normal Equation
# ─────────────────────────────────────────────

def fit_polynomial(X, y, degree):
    """
    Fits y = theta0 + theta1*x + ... + theta_d*x^d as linear regression on
    engineered features [1, x, x^2, ..., x^d] -- same model as Linear
    Regression, just with more columns.

    Solved with lstsq rather than the literal Normal Equation
    (X^T X)^-1 X^T y. Both find the same least-squares solution, but a
    high-degree Vandermonde X^T X is severely ill-conditioned, so
    explicitly inverting it returns garbage at the higher degrees and
    destroys the very effect this script is meant to show. lstsq goes
    through a numerically stable QR/SVD path instead.
    """
    X_poly = np.vander(X, degree + 1, increasing=True)
    theta, *_ = np.linalg.lstsq(X_poly, y, rcond=None)
    return theta


def predict_polynomial(X, theta):
    degree = len(theta) - 1
    X_poly = np.vander(X, degree + 1, increasing=True)
    return X_poly @ theta


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# ─────────────────────────────────────────────
# Fit Three Models — (a) High Bias, (b) Balanced, (c) High Variance
# ─────────────────────────────────────────────

if __name__ == "__main__":
    degrees = {"(a) Underfit": 1, "(b) Just Right": 3, "(c) Overfit": 15}
    x_plot = np.linspace(0, 1, 300)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    print("── Bias-Variance Tradeoff Across Model Complexity ──\n")
    for ax, (label, degree) in zip(axes, degrees.items()):
        theta = fit_polynomial(X, y, degree)
        y_pred_train = predict_polynomial(X, theta)
        y_pred_plot = predict_polynomial(x_plot, theta)

        train_error = mse(y, y_pred_train)
        print(f"   {label}  (degree={degree}):  training MSE = {train_error:.4f}")

        ax.scatter(X, y, color="black", zorder=3, label="Data")
        ax.plot(x_plot, y_pred_plot, color="tab:red", linewidth=2, label=f"degree {degree} fit")
        ax.plot(x_plot, np.sin(2 * np.pi * x_plot), "--", color="gray", alpha=0.6, label="True function")
        ax.set_title(f"{label}\ntraining MSE = {train_error:.4f}")
        ax.set_ylim(-2, 2)
        ax.legend(fontsize=8)

    print("\nNotice: training MSE keeps dropping as degree increases --")
    print("that's exactly why 'lowest training error' is the wrong")
    print("criterion for picking a model. Degree 15 wins on training MSE")
    print("but is clearly overfitting the noise, not the true sine curve.")

    # ── Does that hold up on data the model has never seen? ──
    # The panels above only show training error, which by construction
    # always favours the most complex model. Score the same sweep
    # against a held-out grid of the true function to show what the
    # training number hides: overfitting buys a lower training MSE while
    # generalization error gets steadily worse.
    x_test = np.linspace(0, 1, 200)
    y_test = np.sin(2 * np.pi * x_test)

    print("\n── Training vs. Generalization Error Across Degrees ──\n")
    print("   degree   training MSE   held-out MSE")
    rows = []
    for degree in range(1, 16):
        theta = fit_polynomial(X, y, degree)
        train_err = mse(y, predict_polynomial(X, theta))
        test_err = mse(y_test, predict_polynomial(x_test, theta))
        rows.append((degree, train_err, test_err))
        print(f"   {degree:>6}   {train_err:>12.4f}   {test_err:>12.4f}")

    train_curve = [r[1] for r in rows]
    test_curve = [r[2] for r in rows]
    best_train = min(rows, key=lambda r: r[1])[0]
    best_test = min(rows, key=lambda r: r[2])[0]

    print(f"\n   Training MSE is lowest at degree {best_train} (the most")
    print(f"   complex model tried), but generalization error is lowest at")
    print(f"   degree {best_test}. The two curves point at different models --")
    print("   picking by training error alone guarantees the overfit one.")

    plt.suptitle("Bias-Variance Tradeoff: Underfit -> Just Right -> Overfit")
    plt.tight_layout()
    plt.show()
