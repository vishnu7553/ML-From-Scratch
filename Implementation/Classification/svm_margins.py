import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# The Point of This File
# ─────────────────────────────────────────────
# Full SVM training (the Optimal Margin Classifier's dual problem) comes
# in a later file, once the optimization objective is derived. This
# file focuses purely on this window's new concept: the functional vs.
# geometric margin distinction, and *why* SVM optimizes the geometric
# one.


# ─────────────────────────────────────────────
# Dataset — a manually separable 2D toy set
# ─────────────────────────────────────────────

X = np.array([
    [2, 3], [3, 3.5], [2.5, 4],       # class -1
    [5, 5], [6, 6], [5.5, 6.5],       # class +1
])
y = np.array([-1, -1, -1, 1, 1, 1])

# A hand-picked separating hyperplane: w^T x + b = 0
w = np.array([1.0, 1.0])
b = -8.0


# ─────────────────────────────────────────────
# Functional and Geometric Margin
# ─────────────────────────────────────────────

def functional_margin(X, y, w, b):
    """
    functional_margin_i = y_i * (w^T x_i + b)

    Positive when the point is correctly classified; its magnitude is
    NOT a reliable measure of confidence on its own, because it can be
    inflated arbitrarily by scaling w and b.
    """
    return y * (X @ w + b)


def geometric_margin(X, y, w, b):
    """
    geometric_margin_i = functional_margin_i / ||w||

    This is the actual Euclidean distance from each point to the
    hyperplane -- invariant to scaling w and b, because the same scale
    factor appears in both the numerator and ||w||, and cancels.
    """
    return functional_margin(X, y, w, b) / np.linalg.norm(w)


# ─────────────────────────────────────────────
# Demo 1 — Functional Margin Can Be "Cheated"
# ─────────────────────────────────────────────

def demonstrate_scale_invariance():
    print("── Scaling w and b by increasing constants ──\n")
    print(f"{'Scale c':>8} | {'Functional Margin (min)':>24} | {'Geometric Margin (min)':>23}")
    print("-" * 62)

    for c in [1, 2, 5, 10, 100]:
        w_scaled = w * c
        b_scaled = b * c
        fm = functional_margin(X, y, w_scaled, b_scaled).min()
        gm = geometric_margin(X, y, w_scaled, b_scaled).min()
        print(f"{c:>8} | {fm:>24.4f} | {gm:>23.4f}")

    print("\nThe functional margin inflates linearly with c -- easy to")
    print("'cheat' without moving the decision boundary at all.")
    print("The geometric margin stays IDENTICAL across every scale --")
    print("it's measuring the actual, unchanging distance to the boundary.")
    print("This is exactly why SVM optimizes the geometric margin.\n")


# ─────────────────────────────────────────────
# Demo 2 — Visualizing the Margin Geometrically
# ─────────────────────────────────────────────

def plot_margin(X, y, w, b):
    x_min, x_max = X[:, 0].min() - 2, X[:, 0].max() + 2
    y_min, y_max = X[:, 1].min() - 2, X[:, 1].max() + 2

    xx = np.linspace(x_min, x_max, 200)
    # decision boundary: w0*x + w1*y + b = 0  =>  y = -(w0*x + b) / w1
    yy_boundary = -(w[0] * xx + b) / w[1]

    margin_dist = 1 / np.linalg.norm(w)  # distance for functional margin = 1
    yy_margin_pos = -(w[0] * xx + b - 1) / w[1]
    yy_margin_neg = -(w[0] * xx + b + 1) / w[1]

    plt.figure(figsize=(8, 6))
    plt.plot(xx, yy_boundary, "k-", linewidth=2, label="Decision boundary (w^Tx+b=0)")
    plt.plot(xx, yy_margin_pos, "k--", linewidth=1, label="Margin boundaries")
    plt.plot(xx, yy_margin_neg, "k--", linewidth=1)

    plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], c="tab:blue", s=80,
                edgecolor="black", label="Class -1")
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c="tab:orange", s=80,
                edgecolor="black", label="Class +1")

    # annotate geometric margin of the closest point
    gm = geometric_margin(X, y, w, b)
    closest_idx = np.argmin(gm)
    plt.annotate(f"closest point\ngeometric margin = {gm[closest_idx]:.3f}",
                 xy=X[closest_idx], xytext=(X[closest_idx][0]+0.5, X[closest_idx][1]-1.5),
                 arrowprops=dict(arrowstyle="->"))

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Decision Boundary and Margin")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("── Margins for the Hand-Picked Hyperplane ───\n")
    fm = functional_margin(X, y, w, b)
    gm = geometric_margin(X, y, w, b)
    for i in range(len(X)):
        print(f"   Point {X[i]}  y={y[i]:+d}  |  "
              f"functional={fm[i]:.3f}  geometric={gm[i]:.3f}")
    print()

    demonstrate_scale_invariance()
    plot_margin(X, y, w, b)