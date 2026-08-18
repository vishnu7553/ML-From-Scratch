from pathlib import Path

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
# Dataset — reuse the same student pass/fail data
# as Logistic Regression, so convergence speed is
# directly comparable on identical data.
# ─────────────────────────────────────────────

DATASETS_DIR = Path(__file__).resolve().parent.parent / "datasets"

data = pd.read_csv(DATASETS_DIR / "student.csv")

X = data["study_hours"].values.reshape(-1, 1)
y = data["pass_fail"].values.reshape(-1, 1)

X_b = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# ─────────────────────────────────────────────
# Gradient Ascent — baseline for comparison
# (same as Implementation/Classification/logistic_regression.py)
# ─────────────────────────────────────────────

def gradient_ascent(X, y, epochs=1000, alpha=0.01, tol=1e-8):
    """
    Standard first-order method: takes a small, fixed-size step in the
    direction of the gradient every iteration. Cheap per step, but needs
    many iterations to converge — linear convergence.
    """
    m = len(X)
    theta = np.zeros((X.shape[1], 1))
    history = []

    for i in range(epochs):
        predictions = sigmoid(X @ theta)
        grad = X.T @ (y - predictions)
        theta_new = theta + (alpha / m) * grad
        history.append(log_likelihood(X, y, theta_new))
        if np.linalg.norm(theta_new - theta) < tol:
            theta = theta_new
            break
        theta = theta_new

    return theta, history


# ─────────────────────────────────────────────
# Newton's Method
# ─────────────────────────────────────────────

def log_likelihood(X, y, theta):
    """Log-likelihood of logistic regression, for tracking convergence."""
    h = sigmoid(X @ theta)
    eps = 1e-12  # avoid log(0)
    return float(np.sum(y * np.log(h + eps) + (1 - y) * np.log(1 - h + eps)))


def gradient(X, y, theta):
    """First-order derivative of the log-likelihood: grad = X^T (y - h)"""
    h = sigmoid(X @ theta)
    return X.T @ (y - h)


def hessian(X, theta):
    """
    Second-order derivative of the log-likelihood.

        H = -X^T S X,   where S = diag( h * (1 - h) )

    H is negative semi-definite at the optimum (since we're at a maximum
    of a concave log-likelihood), which is exactly what makes the
    Newton update theta - H^-1 grad move toward the maximum.
    """
    h = sigmoid(X @ theta).flatten()
    S = np.diag(h * (1 - h))
    return -X.T @ S @ X


def newtons_method(X, y, iterations=10, tol=1e-8):
    """
    Root-finds grad(theta) = 0 by repeatedly jumping to where the
    tangent of the gradient function crosses zero:

        theta_(t+1) = theta_(t) - H^-1 * grad(theta_(t))

    Converges in far fewer iterations than gradient ascent, at the cost
    of inverting the (n+1) x (n+1) Hessian every step.
    """
    m, n = X.shape
    theta = np.zeros((n, 1))
    history = []

    for i in range(iterations):
        grad = gradient(X, y, theta)
        H = hessian(X, theta)
        theta_new = theta - np.linalg.inv(H) @ grad
        history.append(log_likelihood(X, y, theta_new))
        if np.linalg.norm(theta_new - theta) < tol:
            theta = theta_new
            break
        theta = theta_new

    return theta, history


# ─────────────────────────────────────────────
# Compare Convergence Speed
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("── Gradient Ascent ──────────────────────")
    theta_ga, hist_ga = gradient_ascent(X_b, y, epochs=50000, alpha=0.05)
    print(f"   Converged in {len(hist_ga)} iterations")
    print(f"   Final theta: {theta_ga.T}")
    print(f"   Final log-likelihood: {hist_ga[-1]:.6f}")
    print("─────────────────────────────────────────\n")

    print("── Newton's Method ──────────────────────")
    theta_nm, hist_nm = newtons_method(X_b, y, iterations=10)
    print(f"   Converged in {len(hist_nm)} iterations")
    print(f"   Final theta: {theta_nm.T}")
    print(f"   Final log-likelihood: {hist_nm[-1]:.6f}")
    print("─────────────────────────────────────────\n")

    print(f"Newton's Method reached an equivalent solution in "
          f"{len(hist_nm)} steps vs Gradient Ascent's {len(hist_ga)} steps.")