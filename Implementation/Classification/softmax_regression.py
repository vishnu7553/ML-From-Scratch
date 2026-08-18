import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

# 3-class synthetic dataset (circle / triangle / square), matching the
# notes' own running example. Each class is a Gaussian blob in 2D
# feature space so the decision boundaries are easy to visualise.
np.random.seed(42)
n_per_class = 60

circle    = np.random.randn(n_per_class, 2) * 0.6 + [2, 2]
triangle  = np.random.randn(n_per_class, 2) * 0.6 + [6, 2]
square    = np.random.randn(n_per_class, 2) * 0.6 + [4, 6]

X = np.vstack([circle, triangle, square])
labels = np.array([0]*n_per_class + [1]*n_per_class + [2]*n_per_class)
K = 3  # number of classes


def one_hot(labels, k):
    """Converts integer class labels into one-hot row vectors."""
    m = labels.shape[0]
    Y = np.zeros((m, k))
    Y[np.arange(m), labels] = 1
    return Y


Y = one_hot(labels, K)
X_b = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)  # bias column


# ─────────────────────────────────────────────
# Softmax Function
# ─────────────────────────────────────────────

def softmax(Z):
    """
    Converts raw class scores (logits) into a probability distribution
    per example, via exponentiation + normalization:

        p(y=k | x) = exp(theta_k^T x) / sum_j exp(theta_j^T x)

    Subtracts the row-wise max before exponentiating purely for
    numerical stability — this does not change the result, since
    softmax is shift-invariant (the same constant cancels in the
    numerator and denominator).
    """
    Z_shifted = Z - np.max(Z, axis=1, keepdims=True)
    exp_Z = np.exp(Z_shifted)
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)


# ─────────────────────────────────────────────
# Method — Gradient Descent on Cross-Entropy
# ─────────────────────────────────────────────

def cross_entropy_loss(Y, P):
    """
    Cross-entropy between the one-hot target Y and predicted
    distribution P — the "distance" between prediction and ground truth
    that training minimises.
    """
    m = Y.shape[0]
    eps = 1e-12
    return -np.sum(Y * np.log(P + eps)) / m


def gradient_descent(X, Y, K, epochs=2000, alpha=0.1):
    """
    theta is now a matrix (n_features x K) — one column per class.

    The gradient of cross-entropy w.r.t. theta has the same clean form
    as Logistic Regression's, just matrix-shaped across all K classes
    simultaneously:

        grad = X^T (P - Y) / m

    Every step pushes the correct class's score up and every other
    class's score down at once, via the normalization in softmax.
    """
    m, n = X.shape
    theta = np.zeros((n, K))

    print("── Training ─────────────────────────────")
    for i in range(epochs):
        Z = X @ theta
        P = softmax(Z)
        grad = X.T @ (P - Y) / m
        theta -= alpha * grad
        if i % 200 == 0:
            loss = cross_entropy_loss(Y, P)
            print(f"   Epoch {i:>4}  |  Cross-Entropy Loss: {loss:.4f}")
    print("─────────────────────────────────────────\n")

    return theta


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────

def evaluate(X, Y, theta):
    P = softmax(X @ theta)
    predictions = np.argmax(P, axis=1)
    true_labels = np.argmax(Y, axis=1)
    accuracy = np.mean(predictions == true_labels)

    print("── Evaluation ───────────────────────────")
    print(f"   Accuracy: {accuracy * 100:.2f}%")
    print("─────────────────────────────────────────\n")
    return predictions


# ─────────────────────────────────────────────
# Visualisation — Decision Boundaries
# ─────────────────────────────────────────────

def plot_decision_boundaries(X, labels, theta):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                          np.linspace(y_min, y_max, 300))

    grid = np.c_[np.ones(xx.ravel().shape[0]), xx.ravel(), yy.ravel()]
    Z = np.argmax(softmax(grid @ theta), axis=1).reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.25, cmap="viridis")
    class_names = ["Circle", "Triangle", "Square"]
    markers = ["o", "^", "s"]
    for k in range(3):
        pts = X[labels == k]
        plt.scatter(pts[:, 0], pts[:, 1], label=class_names[k],
                    marker=markers[k], edgecolor="black")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Softmax Regression — Decision Boundaries")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────

if __name__ == "__main__":
    theta = gradient_descent(X_b, Y, K, epochs=2000, alpha=0.1)
    evaluate(X_b, Y, theta)
    plot_decision_boundaries(X, labels, theta)