import numpy as np

# ─────────────────────────────────────────────
# Dataset — noisy sine, same shape as the other files in this batch,
# so all three CV methods can be compared choosing the same thing:
# best polynomial degree.
# ─────────────────────────────────────────────

np.random.seed(0)
X_full = np.linspace(0, 1, 60)
y_full = np.sin(2 * np.pi * X_full) + np.random.normal(0, 0.25, size=X_full.shape[0])

DEGREES = list(range(1, 11))


def fit_polynomial(X, y, degree):
    X_poly = np.vander(X, degree + 1, increasing=True)
    theta = np.linalg.inv(X_poly.T @ X_poly + 1e-8 * np.eye(degree + 1)) @ X_poly.T @ y
    return theta


def predict(X, theta):
    degree = len(theta) - 1
    X_poly = np.vander(X, degree + 1, increasing=True)
    return X_poly @ theta


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# ─────────────────────────────────────────────
# 1. Hold-Out Cross Validation
# ─────────────────────────────────────────────

def holdout_cv(X, y, degrees, test_frac=0.3, seed=0):
    """
    Randomly split into S_train (~70%) and S_test (~30%). Train each
    candidate degree on S_train only, pick the one with lowest S_test
    error. Simple, but wastes test_frac of the data for every model
    you don't end up keeping.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_test = int(len(X) * test_frac)
    test_idx, train_idx = idx[:n_test], idx[n_test:]

    errors = {}
    for d in degrees:
        theta = fit_polynomial(X[train_idx], y[train_idx], d)
        errors[d] = mse(y[test_idx], predict(X[test_idx], theta))

    best_degree = min(errors, key=errors.get)
    return best_degree, errors


# ─────────────────────────────────────────────
# 2. k-Fold Cross Validation
# ─────────────────────────────────────────────

def k_fold_cv(X, y, degrees, k=10, seed=0):
    """
    Split into k folds. For each degree, train on k-1 folds and test
    on the held-out fold, k times, then average. Every point is used
    for training in k-1 rounds and validation in exactly 1 -- far less
    wasteful than hold-out CV, at the cost of k times the training.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    folds = np.array_split(idx, k)

    avg_errors = {}
    for d in degrees:
        fold_errors = []
        for j in range(k):
            test_idx = folds[j]
            train_idx = np.concatenate([folds[i] for i in range(k) if i != j])
            theta = fit_polynomial(X[train_idx], y[train_idx], d)
            fold_errors.append(mse(y[test_idx], predict(X[test_idx], theta)))
        avg_errors[d] = np.mean(fold_errors)

    best_degree = min(avg_errors, key=avg_errors.get)
    return best_degree, avg_errors


# ─────────────────────────────────────────────
# 3. Leave-One-Out Cross Validation
# ─────────────────────────────────────────────

def leave_one_out_cv(X, y, degrees):
    """
    The special case k = m: every single point gets held out exactly
    once. Standard practice for very small datasets, where holding out
    even 10% for a dev/test set is unaffordable.
    """
    m = len(X)
    avg_errors = {}
    for d in degrees:
        point_errors = []
        for i in range(m):
            train_idx = np.delete(np.arange(m), i)
            theta = fit_polynomial(X[train_idx], y[train_idx], d)
            pred = predict(X[i:i+1], theta)
            point_errors.append((y[i] - pred[0]) ** 2)
        avg_errors[d] = np.mean(point_errors)

    best_degree = min(avg_errors, key=avg_errors.get)
    return best_degree, avg_errors


# ─────────────────────────────────────────────
# 4. Feature Selection — Forward and Backward Search
# ─────────────────────────────────────────────

def make_feature_selection_dataset(n=200, seed=1):
    """
    8 features: only 3 (indices 0, 2, 5) actually drive y.
    The rest are pure noise -- a controlled setting where we KNOW the
    right answer in advance, so forward/backward search's output can
    be checked against ground truth.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(n, 8))
    true_coef = np.zeros(8)
    true_coef[0] = 2.0
    true_coef[2] = -1.5
    true_coef[5] = 1.0
    y = X @ true_coef + rng.normal(0, 0.5, size=n)
    return X, y, true_coef


def cv_error_for_subset(X, y, feature_idx, k=5, seed=0):
    """k-fold CV error using only the given subset of feature columns."""
    if len(feature_idx) == 0:
        return np.var(y)  # predicting the mean -- baseline error with no features

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    folds = np.array_split(idx, k)

    X_sub = X[:, feature_idx]
    X_b = np.c_[np.ones(len(X_sub)), X_sub]

    fold_errors = []
    for j in range(k):
        test_idx, train_idx = folds[j], np.concatenate([folds[i] for i in range(k) if i != j])
        theta = np.linalg.inv(X_b[train_idx].T @ X_b[train_idx]) @ X_b[train_idx].T @ y[train_idx]
        pred = X_b[test_idx] @ theta
        fold_errors.append(mse(y[test_idx], pred))
    return np.mean(fold_errors)


def forward_search(X, y, n_features):
    """Start empty, greedily add whichever feature helps CV error most."""
    selected = []
    remaining = list(range(n_features))
    history = []

    best_error = cv_error_for_subset(X, y, selected)
    history.append((tuple(selected), best_error))

    while remaining:
        candidate_errors = {f: cv_error_for_subset(X, y, selected + [f]) for f in remaining}
        best_candidate = min(candidate_errors, key=candidate_errors.get)

        if candidate_errors[best_candidate] >= best_error:
            break  # adding anything else no longer helps -- stop

        selected.append(best_candidate)
        remaining.remove(best_candidate)
        best_error = candidate_errors[best_candidate]
        history.append((tuple(selected), best_error))

    return selected, history


def backward_search(X, y, n_features):
    """Start with everything, greedily remove whichever feature hurts least."""
    selected = list(range(n_features))
    history = []

    best_error = cv_error_for_subset(X, y, selected)
    history.append((tuple(selected), best_error))

    while len(selected) > 1:
        candidate_errors = {}
        for f in selected:
            subset = [s for s in selected if s != f]
            candidate_errors[f] = cv_error_for_subset(X, y, subset)

        best_removal = min(candidate_errors, key=candidate_errors.get)

        if candidate_errors[best_removal] > best_error * 1.05:
            break  # removing anything else hurts too much -- stop

        selected.remove(best_removal)
        best_error = candidate_errors[best_removal]
        history.append((tuple(selected), best_error))

    return selected, history


# ─────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("── 1. Hold-Out Cross Validation ─────────")
    best_d, errors = holdout_cv(X_full, y_full, DEGREES)
    for d, e in errors.items():
        marker = " <- best" if d == best_d else ""
        print(f"   degree {d:>2}: test MSE = {e:.4f}{marker}")
    print(f"   Selected degree: {best_d}")
    print("─────────────────────────────────────────\n")

    print("── 2. 10-Fold Cross Validation ──────────")
    best_d, errors = k_fold_cv(X_full, y_full, DEGREES, k=10)
    for d, e in errors.items():
        marker = " <- best" if d == best_d else ""
        print(f"   degree {d:>2}: avg CV MSE = {e:.4f}{marker}")
    print(f"   Selected degree: {best_d}")
    print("─────────────────────────────────────────\n")

    print("── 3. Leave-One-Out Cross Validation ────")
    best_d, errors = leave_one_out_cv(X_full, y_full, DEGREES[:6])  # capped -- LOOCV is O(m) fits per degree
    for d, e in errors.items():
        marker = " <- best" if d == best_d else ""
        print(f"   degree {d:>2}: LOOCV MSE = {e:.4f}{marker}")
    print(f"   Selected degree: {best_d}")
    print("─────────────────────────────────────────\n")

    print("── 4. Feature Selection (ground truth: features 0, 2, 5) ──\n")
    X_feat, y_feat, true_coef = make_feature_selection_dataset()

    print("Forward Search:")
    selected_f, history_f = forward_search(X_feat, y_feat, 8)
    for subset, err in history_f:
        print(f"   {subset}  CV MSE = {err:.4f}")
    print(f"   Final selected features: {selected_f}\n")

    print("Backward Search:")
    selected_b, history_b = backward_search(X_feat, y_feat, 8)
    for subset, err in history_b:
        print(f"   {subset}  CV MSE = {err:.4f}")
    print(f"   Final selected features: {selected_b}\n")

    print(f"Ground truth relevant features: [0, 2, 5]")
    print(f"Forward search found:  {sorted(selected_f)}")
    print(f"Backward search found: {sorted(selected_b)}")
