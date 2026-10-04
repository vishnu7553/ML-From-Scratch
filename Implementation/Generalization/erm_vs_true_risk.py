import numpy as np

# ─────────────────────────────────────────────
# True distribution D: y = 1 if x > 0.5 else 0, with 10% label noise
# ─────────────────────────────────────────────

def true_risk(h, n_samples=200000, seed=1):
    """Monte Carlo estimate of E[1{h(x) != y}] over the full distribution D."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n_samples)
    y = (x > 0.5).astype(int)
    flip = rng.random(n_samples) < 0.1
    y = np.where(flip, 1 - y, y)
    return np.mean(h(x) != y)


def empirical_risk(h, x, y):
    """hat_epsilon_S(h) = (1/m) sum 1{h(x_i) != y_i}"""
    return np.mean(h(x) != y)


def erm_threshold_classifier(x_train, y_train, candidate_thresholds):
    """
    h_ERM = arg min_{h in H} empirical risk, where H = {step functions
    at threshold t, for t in candidates}. Brute-force search over H.
    """
    best_t, best_err = None, np.inf
    for t in candidate_thresholds:
        h = lambda x, t=t: (x > t).astype(int)
        err = empirical_risk(h, x_train, y_train)
        if err < best_err:
            best_t, best_err = t, err
    return best_t, best_err


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for m in [20, 200, 2000]:
        x_train = rng.uniform(0, 1, m)
        y_train = (x_train > 0.5).astype(int)
        flip = rng.random(m) < 0.1
        y_train = np.where(flip, 1 - y_train, y_train)

        candidates = np.linspace(0, 1, 101)
        best_t, emp_err = erm_threshold_classifier(x_train, y_train, candidates)
        h_best = lambda x, t=best_t: (x > t).astype(int)
        gen_err = true_risk(h_best)

        print(f"m={m:>5}: ERM threshold={best_t:.2f}  "
              f"empirical risk={emp_err:.4f}  true risk={gen_err:.4f}  "
              f"gap={abs(gen_err-emp_err):.4f}")

    print("\nAs m grows, the ERM-selected threshold converges to the true")
    print("boundary (0.5) and the empirical/true risk gap shrinks --")
    print("exactly the uniform convergence behavior the notes predict.")
