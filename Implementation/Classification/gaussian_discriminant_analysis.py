import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# ─────────────────────────────────────────────
# Dataset — two 2D Gaussian-shaped classes
# ─────────────────────────────────────────────

np.random.seed(7)
n_per_class = 100

# Two classes with different means but a SHARED covariance structure —
# GDA's core assumption is one common Sigma across all classes.
true_cov = np.array([[1.2, 0.4], [0.4, 0.8]])
class0 = np.random.multivariate_normal([2, 2], true_cov, n_per_class)
class1 = np.random.multivariate_normal([6, 5], true_cov, n_per_class)

X = np.vstack([class0, class1])
y = np.array([0]*n_per_class + [1]*n_per_class)


# ─────────────────────────────────────────────
# Method — Closed-Form MLE Fit
# ─────────────────────────────────────────────

def fit_gda(X, y):
    """
    Fits GDA's four parameters via their closed-form MLE solutions —
    no iterative optimization needed, unlike Logistic Regression.

        phi    = fraction of examples with y=1
        mu_0   = mean of all x where y=0
        mu_1   = mean of all x where y=1
        Sigma  = shared covariance, pooled across both classes
    """
    m = X.shape[0]

    phi = np.mean(y == 1)
    mu_0 = X[y == 0].mean(axis=0)
    mu_1 = X[y == 1].mean(axis=0)

    mu_y = np.where(y[:, None] == 0, mu_0, mu_1)
    diffs = X - mu_y
    Sigma = (diffs.T @ diffs) / m

    return phi, mu_0, mu_1, Sigma


# ─────────────────────────────────────────────
# Prediction — Bayes' Rule + arg max
# ─────────────────────────────────────────────

def predict_proba(X, phi, mu_0, mu_1, Sigma):
    """
    p(y=1 | x) via Bayes' Rule. p(x) is computed explicitly here (rather
    than dropped, as the notes do for classification) purely so we can
    plot the actual probability curve, not just the winning class.
    """
    p_x_given_0 = multivariate_normal.pdf(X, mean=mu_0, cov=Sigma)
    p_x_given_1 = multivariate_normal.pdf(X, mean=mu_1, cov=Sigma)

    p_y1 = p_x_given_1 * phi
    p_y0 = p_x_given_0 * (1 - phi)

    return p_y1 / (p_y1 + p_y0)


def predict(X, phi, mu_0, mu_1, Sigma):
    """arg max version — only the winning class, p(x) dropped entirely."""
    return (predict_proba(X, phi, mu_0, mu_1, Sigma) >= 0.5).astype(int)


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────

def evaluate(X, y, phi, mu_0, mu_1, Sigma):
    preds = predict(X, phi, mu_0, mu_1, Sigma)
    accuracy = np.mean(preds == y)

    print("── Evaluation ───────────────────────────")
    print(f"   phi   : {phi:.4f}")
    print(f"   mu_0  : {mu_0}")
    print(f"   mu_1  : {mu_1}")
    print(f"   Accuracy: {accuracy * 100:.2f}%")
    print("─────────────────────────────────────────\n")


# ─────────────────────────────────────────────
# Visualisation 1 — Fitted Gaussian Contours + Decision Boundary
# ─────────────────────────────────────────────

def plot_contours_and_boundary(X, y, phi, mu_0, mu_1, Sigma):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                          np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]

    p1 = predict_proba(grid, phi, mu_0, mu_1, Sigma).reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contour(xx, yy, multivariate_normal.pdf(grid, mu_0, Sigma).reshape(xx.shape),
                levels=5, colors="tab:blue", alpha=0.5)
    plt.contour(xx, yy, multivariate_normal.pdf(grid, mu_1, Sigma).reshape(xx.shape),
                levels=5, colors="tab:orange", alpha=0.5)
    plt.contour(xx, yy, p1, levels=[0.5], colors="black", linewidths=2)

    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], c="tab:blue", edgecolor="black", label="Class 0")
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c="tab:orange", edgecolor="black", label="Class 1")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("GDA — Fitted Gaussian Contours and Decision Boundary")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# Visualisation 2 — The Sigmoid Experiment (1D)
# ─────────────────────────────────────────────

def sigmoid_emergence_experiment():
    """
    Reproduces the notes' single-feature experiment: fit GDA to a 1D
    two-class dataset, then plot p(y=1|x) across the full range of x.

    The claim being tested: this curve should be an S-shape identical
    in form to the logistic sigmoid, despite GDA never being told to
    use one — it falls out purely from two Gaussians and Bayes' Rule.
    """
    np.random.seed(3)
    x0 = np.random.normal(-2, 1.2, 150).reshape(-1, 1)
    x1 = np.random.normal(3, 1.2, 150).reshape(-1, 1)
    X1d = np.vstack([x0, x1])
    y1d = np.array([0]*150 + [1]*150)

    phi, mu_0, mu_1, Sigma = fit_gda(X1d, y1d)

    x_range = np.linspace(X1d.min() - 2, X1d.max() + 2, 400).reshape(-1, 1)
    p1 = predict_proba(x_range, phi, mu_0, mu_1, Sigma)

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    # Best-fit logistic sigmoid overlay, purely for visual comparison
    from scipy.optimize import curve_fit
    def logistic(x, a, b):
        return sigmoid(a * x + b)
    popt, _ = curve_fit(logistic, x_range.flatten(), p1, p0=[1, 0])

    plt.figure(figsize=(8, 5))
    plt.plot(x_range, p1, linewidth=3, label="GDA's p(y=1|x)", color="tab:blue")
    plt.plot(x_range, logistic(x_range.flatten(), *popt), "--",
              linewidth=2, label="Fitted logistic sigmoid", color="black")
    plt.axhline(0.5, color="gray", linestyle=":", linewidth=1)
    plt.xlabel("x")
    plt.ylabel("p(y=1 | x)")
    plt.title("GDA Rediscovers the Sigmoid — No One Told It To")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────

if __name__ == "__main__":
    phi, mu_0, mu_1, Sigma = fit_gda(X, y)
    evaluate(X, y, phi, mu_0, mu_1, Sigma)
    plot_contours_and_boundary(X, y, phi, mu_0, mu_1, Sigma)
    sigmoid_emergence_experiment()