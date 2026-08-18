import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time

# ─────────────────────────────────────────────
# Part 1 — The Kernel Trick: O(n) vs O(n^2)
# ─────────────────────────────────────────────

def explicit_quadratic_mapping(x):
    """
    Brute-force phi(x): all pairwise products x_i * x_j.
    This is the O(n^2) way -- explicitly building the high-dimensional
    vector before taking any inner product.
    """
    return np.outer(x, x).flatten()


def kernel_trick_quadratic(x, z):
    """k(x,z) = (x^T z)^2 -- computes the SAME inner product in O(n)."""
    return np.dot(x, z) ** 2


def demonstrate_speedup():
    print("── Kernel Trick: O(n) vs O(n^2) ─────────\n")
    for n in [10, 100, 1000, 5000]:
        x = np.random.randn(n)
        z = np.random.randn(n)

        t0 = time.perf_counter()
        phi_x = explicit_quadratic_mapping(x)
        phi_z = explicit_quadratic_mapping(z)
        brute_force_result = phi_x @ phi_z
        t_brute = time.perf_counter() - t0

        t0 = time.perf_counter()
        kernel_result = kernel_trick_quadratic(x, z)
        t_kernel = time.perf_counter() - t0

        match = np.isclose(brute_force_result, kernel_result)
        print(f"   n={n:>5}  brute-force: {t_brute*1000:>8.4f}ms  "
              f"kernel: {t_kernel*1000:>8.4f}ms  "
              f"speedup: {t_brute/max(t_kernel,1e-9):>8.1f}x  "
              f"match: {match}")
    print()


# ─────────────────────────────────────────────
# Part 2 — Mercer's Theorem: Is a Function a Valid Kernel?
# ─────────────────────────────────────────────

def linear_kernel(X):
    return X @ X.T


def rbf_kernel(X, sigma=1.0):
    sq_dists = np.sum(X**2, axis=1)[:, None] + np.sum(X**2, axis=1)[None, :] - 2 * X @ X.T
    return np.exp(-sq_dists / (2 * sigma**2))


def fake_invalid_kernel(X):
    """
    A function that measures SOME notion of 'difference' but is NOT a
    valid inner product -- included specifically to fail Mercer's test,
    proving the test actually discriminates real kernels from invented
    similarity functions.
    """
    n = X.shape[0]
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K[i, j] = np.sin(X[i] @ X[j])  # sin() breaks positive semi-definiteness
    return K


def check_mercers_theorem(K, name, tol=1e-8):
    """
    The test from the notes: the Gram matrix must be symmetric AND
    positive semi-definite (all eigenvalues >= 0).
    """
    is_symmetric = np.allclose(K, K.T, atol=tol)
    eigenvalues = np.linalg.eigvalsh(K)
    is_psd = np.all(eigenvalues >= -tol)

    print(f"   {name}:")
    print(f"      Symmetric: {is_symmetric}")
    print(f"      Min eigenvalue: {eigenvalues.min():.6f}")
    print(f"      Positive semi-definite: {is_psd}")
    print(f"      -> {'VALID kernel (Mercer satisfied)' if (is_symmetric and is_psd) else 'INVALID kernel'}\n")


def demonstrate_mercers_theorem():
    print("── Mercer's Theorem: Testing Kernel Validity ──\n")
    np.random.seed(0)
    X = np.random.randn(20, 3)

    check_mercers_theorem(linear_kernel(X), "Linear kernel")
    check_mercers_theorem(rbf_kernel(X, sigma=1.0), "RBF (Gaussian) kernel")
    check_mercers_theorem(fake_invalid_kernel(X), "Invented sin()-based 'kernel'")


# ─────────────────────────────────────────────
# Part 3 — Soft Margin: Effect of C
# ─────────────────────────────────────────────

def solve_soft_margin_dual(X, y, C):
    """
    Same dual as the Optimal Margin Classifier, but alpha is now boxed
    to [0, C] instead of [0, inf) -- the one-line change that makes SVM
    tolerant of outliers.
    """
    m = X.shape[0]
    K = X @ X.T

    def neg_dual_objective(alpha):
        return -(np.sum(alpha) - 0.5 * np.sum(
            np.outer(alpha, alpha) * np.outer(y, y) * K
        ))

    constraints = [{"type": "eq", "fun": lambda a: np.dot(a, y)}]
    bounds = [(0, C) for _ in range(m)]
    alpha0 = np.zeros(m)

    result = minimize(neg_dual_objective, alpha0, bounds=bounds,
                       constraints=constraints, method="SLSQP")
    alpha = result.x
    alpha[alpha < 1e-6] = 0
    return alpha


def demonstrate_soft_margin():
    print("── Soft Margin: Effect of C on an Outlier ──\n")
    np.random.seed(1)
    X = np.vstack([
        np.random.randn(15, 2) * 0.6 + [2, 2],
        np.random.randn(15, 2) * 0.6 + [6, 6],
        [[6.2, 1.8]],  # a single outlier, mislabeled into class -1's territory
    ])
    y = np.array([-1]*15 + [1]*15 + [-1])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, C in zip(axes, [0.1, 1, 100]):
        alpha = solve_soft_margin_dual(X, y, C)
        w = np.sum((alpha * y)[:, None] * X, axis=0)
        sv_idx = np.where(alpha > 1e-6)[0]
        b_vals = y[sv_idx] - X[sv_idx] @ w
        b = np.mean(b_vals) if len(b_vals) > 0 else 0

        xx = np.linspace(0, 8, 100)
        yy = -(w[0]*xx + b) / w[1]

        ax.scatter(X[y == -1][:, 0], X[y == -1][:, 1], c="tab:blue", edgecolor="black")
        ax.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c="tab:orange", edgecolor="black")
        ax.scatter(X[-1, 0], X[-1, 1], c="tab:blue", edgecolor="red",
                   linewidth=2, s=150, label="outlier")
        ax.plot(xx, yy, "k-", linewidth=2)
        ax.set_xlim(0, 8)
        ax.set_ylim(0, 8)
        ax.set_title(f"C = {C}  ({np.sum(alpha>1e-6)} support vectors)")
        ax.legend()

    plt.suptitle("Soft Margin — Low C Ignores the Outlier, High C Chases It")
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────

if __name__ == "__main__":
    demonstrate_speedup()
    demonstrate_mercers_theorem()
    demonstrate_soft_margin()