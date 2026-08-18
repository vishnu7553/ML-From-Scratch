import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

# LWR shines on non-linear data — synthesize a noisy sine wave rather than
# reuse the linear datasets/, since a straight line has no "local"
# structure worth weighting.
np.random.seed(42)
X = np.linspace(0, 2 * np.pi, 100)
y = np.sin(X) + np.random.normal(0, 0.15, size=X.shape[0])


# ─────────────────────────────────────────────
# Weight Function
# ─────────────────────────────────────────────

def get_weights(x_query, X_train, tau):
    """
    Computes the weight of every training point relative to a single
    query point, using a Gaussian-shaped (but non-probabilistic) kernel.

        w(i) = exp( -(x_i - x_query)^2 / (2 * tau^2) )

    Nearby points get a weight close to 1, distant points decay toward 0.
    tau controls the bandwidth of "nearby":
        - large tau -> over-saturated local selection (underfitting)
        - small tau -> narrow, small area of prediction (overfitting)
    """
    return np.exp(-((X_train - x_query) ** 2) / (2 * tau ** 2))


# ─────────────────────────────────────────────
# Method — Weighted Normal Equation
# ─────────────────────────────────────────────

def fit_local(x_query, X_train, y_train, tau):
    """
    Fits a fresh weighted linear regression local to x_query.

    Minimises:
        sum_i  w(i) * (y_i - theta^T x_i)^2

    Which has the closed-form weighted Normal Equation solution:
        theta = (X^T W X)^-1 X^T W y

    Where W is the diagonal matrix of weights for this query point.
    Unlike ordinary Linear Regression, this is re-solved for every
    single prediction — the training data is never discarded.
    """
    m = X_train.shape[0]
    X_b = np.c_[np.ones(m), X_train]          # prepend bias column
    x_query_b = np.array([1, x_query])

    w = get_weights(x_query, X_train, tau)
    W = np.diag(w)

    theta = np.linalg.inv(X_b.T @ W @ X_b) @ X_b.T @ W @ y_train
    return x_query_b @ theta


# ─────────────────────────────────────────────
# Prediction over a range
# ─────────────────────────────────────────────

def predict_curve(X_train, y_train, tau, n_points=200):
    """
    Predicts across a smooth range of query points, refitting a local
    model at every single one — this is what makes LWR non-parametric:
    there is no single global theta, only many local ones.
    """
    x_range = np.linspace(X_train.min(), X_train.max(), n_points)
    predictions = np.array([fit_local(x, X_train, y_train, tau) for x in x_range])
    return x_range, predictions


# ─────────────────────────────────────────────
# Visualisation — Effect of Bandwidth (tau)
# ─────────────────────────────────────────────

def plot_bandwidth_comparison(X_train, y_train, taus):
    """
    Plots the fitted curve for several values of tau side by side,
    to make the underfitting/overfitting trade-off visible directly.
    """
    plt.figure(figsize=(9, 5))
    plt.scatter(X_train, y_train, color="gray", alpha=0.5, label="Data", zorder=1)

    for tau in taus:
        x_range, preds = predict_curve(X_train, y_train, tau)
        plt.plot(x_range, preds, linewidth=2, label=f"tau = {tau}")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Locally Weighted Regression — Effect of Bandwidth (tau)")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────

if __name__ == "__main__":
    taus = [0.1, 0.5, 3.0]

    print("── Locally Weighted Regression ──────────")
    for tau in taus:
        _, preds = predict_curve(X, y, tau)
        mse = np.mean((preds - np.sin(np.linspace(X.min(), X.max(), len(preds)))) ** 2)
        print(f"   tau = {tau:>4}  |  MSE vs true sine: {mse:.4f}")
    print("─────────────────────────────────────────\n")

    plot_bandwidth_comparison(X, y, taus)