import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
# The Point of This File
# ─────────────────────────────────────────────
# Linear Regression, Logistic Regression, and Poisson Regression look
# like three separate algorithms. Under the GLM framework they are the
# *same* update rule:
#
#     theta_j := theta_j + alpha * (y - h_theta(x)) * x_j
#
# only the hypothesis h_theta(x) changes, because only the canonical
# response function g(eta) changes. This file proves that by using one
# single training loop across all three tasks.


def train_glm(X, y, response_function, epochs=2000, alpha=0.01):
    """
    One gradient ascent loop for *any* GLM.

    response_function is the canonical response g(eta) = E[y; eta] —
    swap it out and you get a completely different algorithm, with
    zero other code changes:
        - identity   -> Linear Regression   (Gaussian)
        - sigmoid    -> Logistic Regression (Bernoulli)
        - exp        -> Poisson Regression  (Poisson)
    """
    m, n = X.shape
    theta = np.zeros((n, 1))

    for _ in range(epochs):
        eta = X @ theta                       # Part 1: linear predictor
        h = response_function(eta)            # Part 2 + 3: link + distribution -> h_theta(x)
        grad = X.T @ (y - h)                  # the one true GLM gradient
        theta += (alpha / m) * grad

    return theta


# ─────────────────────────────────────────────
# The Three Canonical Response Functions
# ─────────────────────────────────────────────

def identity(eta):
    """Gaussian -> Linear Regression. g(eta) = eta"""
    return eta


def sigmoid(eta):
    """Bernoulli -> Logistic Regression. g(eta) = 1 / (1 + e^-eta)"""
    return 1 / (1 + np.exp(-eta))


def poisson_response(eta):
    """Poisson -> Poisson Regression (count data). g(eta) = e^eta"""
    return np.exp(eta)


# ─────────────────────────────────────────────
# Demo — Same Loop, Three Different Tasks
# ─────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(0)

    print("── GLM 1: Regression (Gaussian) ─────────")
    X_reg = np.random.uniform(0, 10, (100, 1))
    y_reg = 3 * X_reg + 5 + np.random.normal(0, 1, (100, 1))
    X_reg_b = np.c_[np.ones((100, 1)), X_reg]
    theta_reg = train_glm(X_reg_b, y_reg, identity, epochs=2000, alpha=0.01)
    print(f"   Learned: y = {theta_reg[1][0]:.2f}x + {theta_reg[0][0]:.2f}"
          f"   (true: y = 3.00x + 5.00)")
    print("─────────────────────────────────────────\n")

    print("── GLM 2: Classification (Bernoulli) ────")
    X_clf = np.random.uniform(0, 10, (100, 1))
    y_clf = (X_clf.flatten() + np.random.normal(0, 1, 100) > 5).astype(int).reshape(-1, 1)
    X_clf_b = np.c_[np.ones((100, 1)), X_clf]
    theta_clf = train_glm(X_clf_b, y_clf, sigmoid, epochs=2000, alpha=0.1)
    preds = (sigmoid(X_clf_b @ theta_clf) >= 0.5).astype(int)
    acc = np.mean(preds == y_clf) * 100
    print(f"   Learned theta: {theta_clf.T}")
    print(f"   Accuracy: {acc:.1f}%")
    print("─────────────────────────────────────────\n")

    print("── GLM 3: Counts (Poisson) ──────────────")
    X_cnt = np.random.uniform(0, 5, (200, 1))
    true_rate = np.exp(0.5 * X_cnt.flatten() + 0.2)
    y_cnt = np.random.poisson(true_rate).reshape(-1, 1)
    X_cnt_b = np.c_[np.ones((200, 1)), X_cnt]
    theta_cnt = train_glm(X_cnt_b, y_cnt, poisson_response, epochs=3000, alpha=0.01)
    print(f"   Learned: rate = exp({theta_cnt[1][0]:.2f}x + {theta_cnt[0][0]:.2f})"
          f"   (true: rate = exp(0.50x + 0.20))")
    print("─────────────────────────────────────────\n")

    print("Same train_glm() loop. Same gradient. Three algorithms.")
    print("Only the response function (g) changed.")