from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

DATASETS_DIR = Path(__file__).resolve().parent.parent / "datasets"

dataset = pd.read_csv(DATASETS_DIR / "Salary_dataset.csv")

X = dataset["YearsExperience"].values
y = dataset["Salary"].values

# ─────────────────────────────────────────────
# Visualisation Helpers
# ─────────────────────────────────────────────

def show_scatter(x_val, y_val):
    """Plot raw data points."""
    plt.scatter(x_val, y_val, c="blue")
    plt.xlabel("Years of Experience")
    plt.ylabel("Salary")
    plt.title("Data Distribution")
    plt.show()


def show_regression_line(x_val, y_val, m, b):
    """Plot data points alongside the fitted regression line."""
    plt.scatter(x_val, y_val, c="red", label="Actual")
    plt.plot(x_val, [m * x + b for x in x_val], label="Predicted", color="blue")
    plt.xlabel("Years of Experience")
    plt.ylabel("Salary")
    plt.title("Linear Regression Fit")
    plt.legend()
    plt.show()


# ─────────────────────────────────────────────
# Loss — Inspection Tool
# ─────────────────────────────────────────────

def percentage_loss(x_vals, y_vals, m, b):
    """
    Mean Absolute Percentage Error (MAPE) expressed as a percentage.
    Used only for human-readable loss inspection during training.
    This is not the loss function being optimised — that is MSE.
    """
    loss = 0
    for i in range(len(x_vals)):
        loss += ((y_vals[i] - (m * x_vals[i] + b)) / y_vals[i]) ** 2
    loss /= len(x_vals)
    return np.sqrt(loss) * 100


# ─────────────────────────────────────────────
# Method 1 — Gradient Descent (Iterative)
# ─────────────────────────────────────────────

def gradient_descent(m, b, x, y, learning_rate):
    """
    Computes gradients of MSE with respect to m (weight) and b (bias),
    then returns updated parameters after one step in the descent direction.

    Update rule:
        m ← m - α · (∂MSE/∂m)
        b ← b - α · (∂MSE/∂b)

    Where α is the learning rate.
    """
    m_gradient = 0
    b_gradient = 0
    n = len(x)

    for i in range(n):
        error = y[i] - (m * x[i] + b)
        m_gradient += -(2 / n) * error * x[i]
        b_gradient += -(2 / n) * error

    m = m - m_gradient * learning_rate
    b = b - b_gradient * learning_rate

    return m, b, percentage_loss(x, y, m, b)


# ─────────────────────────────────────────────
# Method 2 — Normal Equation (Closed Form)
# ─────────────────────────────────────────────

def normal_equation(x, y):
    """
    Computes the optimal parameters β directly using the closed-form solution:

        β = (XᵀX)⁻¹ · Xᵀ · y

    A column of ones is prepended to X so that the bias term b
    is handled automatically via matrix multiplication.

    Note: Computationally exact but scales as O(n³) with dataset size.
    Gradient descent is preferred for large datasets.
    """
    x = np.c_[np.ones((x.shape[0], 1)), x]
    beta = np.linalg.inv(x.T @ x) @ x.T @ y
    return beta


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────

def evaluate(x_vals, y_vals, m, b):
    """
    Evaluates the trained model using three standard regression metrics:

        R²   — proportion of variance in y explained by the model (0 to 1)
        RMSE — error in the same units as y, penalises large errors
        MAE  — average absolute error, robust to outliers
    """
    predictions = m * x_vals + b

    ss_res = np.sum((y_vals - predictions) ** 2)
    ss_tot = np.sum((y_vals - np.mean(y_vals)) ** 2)

    r2   = 1 - (ss_res / ss_tot)
    rmse = np.sqrt(np.mean((y_vals - predictions) ** 2))
    mae  = np.mean(np.abs(y_vals - predictions))

    print("\n── Evaluation ───────────────────────────")
    print(f"   R²   : {r2:.4f}  (1.0 = perfect fit)")
    print(f"   RMSE : {rmse:,.2f}")
    print(f"   MAE  : {mae:,.2f}")
    print("─────────────────────────────────────────\n")


# ─────────────────────────────────────────────
# Training — Gradient Descent
# ─────────────────────────────────────────────

m = 0
b = 0
learning_rate = 0.0001
epochs = 100000

print("── Training ─────────────────────────────")
for i in range(epochs):
    m, b, loss = gradient_descent(m, b, X, y, learning_rate)
    if i % 10000 == 0:
        print(f"   Epoch {i:>6}  |  Loss: {loss:.2f}%")

print("─────────────────────────────────────────")
print(f"\n   Weight (m) : {m:.4f}")
print(f"   Bias   (b) : {b:.4f}")

evaluate(X, y, m, b)
show_regression_line(X, y, m, b)


# ─────────────────────────────────────────────
# Normal Equation — Cross Verification
# ─────────────────────────────────────────────

beta = normal_equation(X, y)
print("── Normal Equation ──────────────────────")
print(f"   Bias   (β₀) : {beta[0]:.4f}")
print(f"   Weight (β₁) : {beta[1]:.4f}")
print("─────────────────────────────────────────")