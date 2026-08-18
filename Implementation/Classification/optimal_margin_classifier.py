import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# The Point of This File
# ─────────────────────────────────────────────
# Solve the SAME separable-case SVM problem two different ways:
#   1. The PRIMAL:  min 1/2||w||^2  s.t. y_i(w^Tx_i+b) >= 1
#   2. The DUAL:    via alpha_i, using only inner products <x_i, x_j>
#
# Then verify the Representer Theorem directly: reconstruct w from the
# dual's alphas as w = sum(alpha_i * y_i * x_i), and confirm it exactly
# matches the w found by solving the primal directly. This is the
# concrete proof that w really does live in the span of the training
# data -- not an abstract claim.


# ─────────────────────────────────────────────
# Dataset — small, linearly separable, 2D
# ─────────────────────────────────────────────

X = np.array([
    [2, 3], [3, 3.5], [2.5, 4], [1.5, 3.2],   # class -1
    [6, 6], [7, 6.5], [6.5, 7.5], [7.5, 6],   # class +1
])
y = np.array([-1, -1, -1, -1, 1, 1, 1, 1])
m, n = X.shape


# ─────────────────────────────────────────────
# 1. Solve the PRIMAL directly
# ─────────────────────────────────────────────

def solve_primal(X, y):
    """
    min 1/2||w||^2   s.t.   y_i(w^Tx_i + b) >= 1  for all i

    Solved directly via SLSQP -- fine for small datasets, though real
    SVM solvers use dedicated quadratic programming methods, not
    general-purpose nonlinear optimizers.
    """
    def objective(params):
        w = params[:n]
        return 0.5 * np.dot(w, w)

    def constraint(params, i):
        w = params[:n]
        b = params[n]
        return y[i] * (np.dot(w, X[i]) + b) - 1

    constraints = [{"type": "ineq", "fun": constraint, "args": (i,)} for i in range(m)]
    x0 = np.zeros(n + 1)

    result = minimize(objective, x0, constraints=constraints, method="SLSQP")
    w = result.x[:n]
    b = result.x[n]
    return w, b


# ─────────────────────────────────────────────
# 2. Solve the DUAL — only inner products, never raw features alone
# ─────────────────────────────────────────────

def solve_dual(X, y):
    """
    max sum(alpha_i) - 1/2 sum_i sum_j alpha_i alpha_j y_i y_j <x_i,x_j>
    s.t.  alpha_i >= 0,  sum(alpha_i * y_i) = 0

    Notice the objective touches x_i and x_j ONLY as an inner product
    <x_i, x_j> -- never as standalone vectors. This is exactly the
    property that makes the kernel trick possible later: swap the inner
    product for a kernel function, and everything else stays identical.
    """
    K = X @ X.T  # Gram matrix of inner products <x_i, x_j>

    def neg_dual_objective(alpha):
        return -(np.sum(alpha) - 0.5 * np.sum(
            np.outer(alpha, alpha) * np.outer(y, y) * K
        ))

    constraints = [{"type": "eq", "fun": lambda a: np.dot(a, y)}]
    bounds = [(0, None) for _ in range(m)]
    alpha0 = np.zeros(m)

    result = minimize(neg_dual_objective, alpha0, bounds=bounds,
                       constraints=constraints, method="SLSQP")
    alpha = result.x
    alpha[alpha < 1e-6] = 0  # clean up numerical noise
    return alpha


def reconstruct_w_from_alpha(alpha, X, y):
    """The Representer Theorem, applied directly: w = sum(alpha_i * y_i * x_i)"""
    return np.sum((alpha * y)[:, None] * X, axis=0)


def recover_b(w, X, y, alpha, tol=1e-4):
    """
    b is recovered from any support vector (alpha_i > 0), using the
    fact that support vectors sit EXACTLY on the margin:
    y_i(w^Tx_i + b) = 1.
    """
    sv_idx = np.where(alpha > tol)[0]
    b_values = y[sv_idx] - X[sv_idx] @ w
    return np.mean(b_values)


# ─────────────────────────────────────────────
# Run — Solve Both, Compare
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("── Solving the Primal Directly ──────────")
    w_primal, b_primal = solve_primal(X, y)
    print(f"   w = {w_primal}")
    print(f"   b = {b_primal:.4f}")
    print("─────────────────────────────────────────\n")

    print("── Solving the Dual ─────────────────────")
    alpha = solve_dual(X, y)
    for i, a in enumerate(alpha):
        marker = " <- support vector" if a > 1e-6 else ""
        print(f"   alpha_{i} = {a:.4f}{marker}")
    print("─────────────────────────────────────────\n")

    print("── Representer Theorem Check ────────────")
    w_from_dual = reconstruct_w_from_alpha(alpha, X, y)
    b_from_dual = recover_b(w_from_dual, X, y, alpha)
    print(f"   w reconstructed from alphas: {w_from_dual}")
    print(f"   w from direct primal solve : {w_primal}")
    print(f"   Difference: {np.linalg.norm(w_from_dual - w_primal):.6f}  (should be ~0)")
    print(f"\n   b reconstructed from alphas: {b_from_dual:.4f}")
    print(f"   b from direct primal solve : {b_primal:.4f}")
    print("─────────────────────────────────────────\n")

    n_support = np.sum(alpha > 1e-6)
    print(f"Only {n_support} of {m} training points have alpha > 0 --")
    print(f"those are the support vectors. Every other point could be")
    print(f"deleted and w would not change at all.")

    # ── Visualization ──
    plt.figure(figsize=(8, 6))
    plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], c="tab:blue",
                s=80, edgecolor="black", label="Class -1")
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c="tab:orange",
                s=80, edgecolor="black", label="Class +1")

    sv_idx = np.where(alpha > 1e-6)[0]
    plt.scatter(X[sv_idx][:, 0], X[sv_idx][:, 1], s=200,
                facecolors="none", edgecolor="black", linewidth=2,
                label="Support vectors (alpha > 0)")

    xx = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100)
    yy_boundary = -(w_primal[0] * xx + b_primal) / w_primal[1]
    plt.plot(xx, yy_boundary, "k-", linewidth=2, label="Decision boundary")

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Optimal Margin Classifier — Primal vs Dual Agree")
    plt.legend()
    plt.tight_layout()
    plt.show()