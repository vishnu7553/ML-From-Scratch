import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

data = pd.read_csv("../datasets/student.csv")

X = data["study_hours"].values.reshape(-1, 1)
y = data["pass_fail"].values.reshape(-1, 1)

# Prepend a column of ones to X so the bias term b is handled
# automatically via matrix multiplication — same trick as Linear Regression.
# Note: x0 is always 1, so theta[0] acts as the intercept.
X_b = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)


# ─────────────────────────────────────────────
# Sigmoid Function
# ─────────────────────────────────────────────

def sigmoid(z):
    """
    Squashes any real value into the range (0, 1).
    This is the core of Logistic Regression — it converts
    the raw linear score into a probability.

        σ(z) = 1 / (1 + e^(-z))
    """
    return 1 / (1 + np.exp(-z))


# ─────────────────────────────────────────────
# Method — Gradient Ascent
# ─────────────────────────────────────────────

def gradient_ascent(X, y, epochs=100, alpha=0.01):
    """
    Maximises the log-likelihood by iteratively updating theta
    in the direction of the gradient.

    Unlike Linear Regression which minimises MSE (gradient descent),
    here we maximise log-likelihood (gradient ascent). The math is
    identical — just the sign flips.

    Update rule:
        theta_j ← theta_j + (alpha / m) * sum( (y - h(x)) * x_j )

    In matrix form:
        theta ← theta + (alpha / m) * X^T · (y - sigmoid(X · theta))

    The 1/m scaling keeps gradient values proportional to dataset
    size — without it, larger datasets cause gradient explosion.

    Parameters:
        X      : feature matrix with bias column (m x n+1)
        y      : true labels (m x 1), values in {0, 1}
        epochs : number of iterations
        alpha  : learning rate — controls step size per iteration
    """
    m = len(X)
    theta = np.zeros((X.shape[1], 1))

    print("── Training ─────────────────────────────")
    for i in range(epochs):
        predictions = sigmoid(X @ theta)
        theta = theta + (alpha / m) * (X.T @ (y - predictions))
        if i % 10 == 0:
            print(f"   Epoch {i:>4}  |  theta: {theta.T}")
    print("─────────────────────────────────────────\n")

    return theta


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────

def evaluate(X, y, theta):
    """
    Evaluates the trained model using classification metrics.

    Threshold: if sigmoid output >= 0.5 → class 1, else class 0.

    Metrics:
        Accuracy  — overall correctness
        Precision — of all predicted positives, how many are correct
        Recall    — of all actual positives, how many did we catch
        F1 Score  — harmonic mean of precision and recall
    """
    probabilities = sigmoid(X @ theta)
    predictions   = (probabilities >= 0.5).astype(int)

    tp = np.sum((predictions == 1) & (y == 1))
    fp = np.sum((predictions == 1) & (y == 0))
    fn = np.sum((predictions == 0) & (y == 1))

    accuracy  = np.mean(predictions == y)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    print("── Evaluation ───────────────────────────")
    print(f"   Accuracy  : {accuracy * 100:.2f}%")
    print(f"   Precision : {precision:.4f}")
    print(f"   Recall    : {recall:.4f}")
    print(f"   F1 Score  : {f1:.4f}")
    print("─────────────────────────────────────────\n")

    return predictions


# ─────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────

def plot_sigmoid_fit(X, y, theta):
    """
    Plots the training data alongside the fitted sigmoid curve.
    The curve shows the model's predicted probability of class 1
    across the full range of the input feature.
    """
    x_range    = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
    x_range_b  = np.concatenate((np.ones((x_range.shape[0], 1)), x_range), axis=1)
    probability = sigmoid(x_range_b @ theta)

    plt.figure(figsize=(8, 5))
    plt.scatter(X, y, color="blue", label="Data points", zorder=3)
    plt.plot(x_range, probability, color="red", linewidth=2, label="Sigmoid fit")
    plt.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="Decision boundary (0.5)")
    plt.xlabel("Study Hours")
    plt.ylabel("Probability of Passing")
    plt.title("Logistic Regression — Sigmoid Fit")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

theta = gradient_ascent(X_b, y, epochs=100, alpha=0.01)

print(f"   Bias   (theta_0) : {theta[0][0]:.4f}")
print(f"   Weight (theta_1) : {theta[1][0]:.4f}\n")

predictions = evaluate(X_b, y, theta)

plot_sigmoid_fit(X, y, theta)