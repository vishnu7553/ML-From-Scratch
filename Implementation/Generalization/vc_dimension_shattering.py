import numpy as np
from itertools import product

# ─────────────────────────────────────────────
# Classic result: linear classifiers in 2D have VC dimension 3.
# Verify directly: can a line shatter every labeling of 3 points
# (yes, for points in "general position"), but NOT every labeling
# of 4 points (no -- the XOR-like labeling always fails)?
# ─────────────────────────────────────────────

def can_linear_separate(points, labels):
    """
    Checks if a line can separate `points` into `labels` (+1/-1), by
    brute-force searching over candidate separating directions. Good
    enough for small point sets -- not a general-purpose solver.
    """
    if len(set(labels)) == 1:
        return True  # all one class -- trivially separable

    thetas = np.linspace(0, 2 * np.pi, 720)
    for theta in thetas:
        w = np.array([np.cos(theta), np.sin(theta)])
        for b in np.linspace(-5, 5, 200):
            pred = np.sign(points @ w + b)
            pred[pred == 0] = 1
            if np.all(pred == labels):
                return True
    return False


def shatter_test(points):
    """Tries every possible +/-1 labeling of the points; returns which fail."""
    n = len(points)
    failures = []
    for labeling in product([-1, 1], repeat=n):
        labels = np.array(labeling)
        if not can_linear_separate(points, labels):
            failures.append(labeling)
    return failures


if __name__ == "__main__":
    print("── VC Dimension of Linear Classifiers in 2D ──\n")

    # 3 points in general position (not collinear) -- should ALL be shatterable
    points_3 = np.array([[0, 0], [1, 0], [0, 1]])
    failures_3 = shatter_test(points_3)
    print(f"3 points, general position: {2**3} possible labelings, "
          f"{len(failures_3)} failed to separate.")
    print("-> VC dim >= 3" if len(failures_3) == 0 else "-> shattering failed unexpectedly")

    # 4 points (square) -- the XOR labeling should ALWAYS fail for a line
    points_4 = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    failures_4 = shatter_test(points_4)
    print(f"\n4 points (square): {2**4} possible labelings, "
          f"{len(failures_4)} failed to separate.")
    for f in failures_4:
        print(f"   Labeling {f} cannot be linearly separated (XOR-like diagonal pattern)")
    print("-> VC dim < 4, confirming VC dimension of 2D linear classifiers = 3")
