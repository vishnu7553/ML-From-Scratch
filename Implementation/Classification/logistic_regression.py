from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# Dataset — heart.csv (918 rows, 11 features)
# ─────────────────────────────────────────────
# The original toy set here was 19 rows of study-hours vs pass/fail,
# far too small to say anything about a fitted model. This uses the
# full heart-disease set instead, and holds out a stratified 20% of it
# that gradient ascent never sees.

DATASETS_DIR = Path(__file__).resolve().parent.parent / "datasets"

RANDOM_SEED = 0
TEST_FRACTION = 0.2

data = pd.read_csv(DATASETS_DIR / "heart.csv")
TARGET = "HeartDisease"

# Categorical columns are listed explicitly rather than inferred from
# dtype, so the encoding below is decided by meaning, not by accident.
# Binary columns name the level that maps to 1, rather than relying on
# whichever value happens to appear in the first row.
BINARY_COLUMNS = {"Sex": "M", "ExerciseAngina": "Y"}
CATEGORICAL_COLUMNS = ["ChestPainType", "RestingECG", "ST_Slope"]
CONTINUOUS_COLUMNS = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]


# ─────────────────────────────────────────────
# Feature Encoding
# ─────────────────────────────────────────────

def encode_features(df):
    """
    Turns the raw frame into a numeric matrix, without pandas' own
    get_dummies — the encoding is the model, so it is written out.

    Binary columns map straight onto 0/1. Multi-category columns get
    one-hot columns with the first level dropped: a full one-hot block
    plus the bias column would be linearly dependent, and the normal
    equations would be singular. Dropping one level per categorical
    keeps the design matrix full rank.

    Returns:
        X       : (m x n) float matrix
        names   : the feature names, in column order
    """
    columns, names = [], []

    for col in CONTINUOUS_COLUMNS:
        columns.append(df[col].to_numpy(dtype=float).reshape(-1, 1))
        names.append(col)

    for col, positive in BINARY_COLUMNS.items():
        values = (df[col] == positive).to_numpy(dtype=float).reshape(-1, 1)
        columns.append(values)
        names.append(f"{col}={positive}")

    for col in CATEGORICAL_COLUMNS:
        levels = sorted(df[col].unique())[1:]  # drop the first level
        for level in levels:
            columns.append((df[col] == level).to_numpy(dtype=float).reshape(-1, 1))
            names.append(f"{col}={level}")

    return np.hstack(columns), names


def add_bias_column(X):
    """
    Prepends a column of ones so the bias term is handled by matrix
    multiplication. x0 is always 1, so theta[0] acts as the intercept.
    """
    return np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)


# ─────────────────────────────────────────────
# Train / Test Split
# ─────────────────────────────────────────────

def stratified_split(y, test_fraction=0.2, seed=0):
    """
    Splits indices into train and test while keeping the class balance
    of each identical to the whole. A plain random split can leave the
    test set lopsided on a small set, which makes the held-out number
    noisier than it needs to be.

    Stratifying per class is not the same as stratifying the whole set,
    so this loops over classes and holds out the same fraction of each.
    """
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []

    for label in np.unique(y):
        idx = np.flatnonzero(y == label)
        rng.shuffle(idx)
        cut = int(round(len(idx) * test_fraction))
        test_idx.extend(idx[:cut])
        train_idx.extend(idx[cut:])

    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return np.array(train_idx), np.array(test_idx)


def standardise(X_train, X_test):
    """
    Zero-mean unit-variance scaling, needed here for two reasons.

    Gradient ascent takes a single learning rate for every parameter,
    and raw Cholesterol (~200-600) would produce a gradient thousands
    of times larger than raw Oldpeak (~0-6). One shared alpha either
    diverges on the first or crawls on the second; scaling puts every
    feature on the same footing.

    The mean and std come from the TRAIN ROWS ONLY and are then applied
    to both sets. Computing them over the full dataset would leak test
    information into the fit — a quiet way to make held-out numbers look
    better than they are.

    Note the scaled test set does not have a mean of exactly zero. That
    offset is expected, and is the visible sign that these statistics
    were fitted on training rows only.
    """
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0)
    sd[sd == 0] = 1.0  # a constant column would otherwise divide by zero
    return (X_train - mu) / sd, (X_test - mu) / sd


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
# Loss — Log Loss (Binary Cross-Entropy)
# ─────────────────────────────────────────────

def log_loss(X, y, theta):
    """
    Binary cross-entropy, averaged over the dataset:

        L(theta) = -(1/m) * sum( y*log(h) + (1-y)*log(1-h) )

    This is the negative of the mean log-likelihood, so it is exactly
    the quantity that is *minimised*, while gradient ascent below
    maximises the log-likelihood. Expect this number to fall every
    epoch and settle where theta stops moving.

    h is clipped inside the log by eps, because a saturated sigmoid
    returns exactly 0 or 1 and log(0) would produce inf/nan.
    """
    h = sigmoid(X @ theta)
    eps = 1e-12
    return float(-np.mean(y * np.log(h + eps) + (1 - y) * np.log(1 - h + eps)))


# ─────────────────────────────────────────────
# Method — Gradient Ascent
# ─────────────────────────────────────────────

def gradient_ascent(X, y, X_holdout=None, y_holdout=None,
                    epochs=3000, alpha=0.5):
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

    Only the training rows take part in the update. The held-out rows
    are scored alongside purely to watch the gap open up; if they were
    fed into the loop they would stop being held out.

    Parameters:
        X           : training features with bias column (m x n+1)
        y           : training labels (m x 1), values in {0, 1}
        X_holdout   : held-out features, scored but never trained on
        epochs      : number of iterations
        alpha       : learning rate — controls step size per iteration

    Returns:
        theta, list of (train_loss, holdout_loss) per epoch
    """
    m = len(X)
    theta = np.zeros((X.shape[1], 1))
    history = []

    for _ in range(epochs):
        predictions = sigmoid(X @ theta)
        theta = theta + (alpha / m) * (X.T @ (y - predictions))

        if X_holdout is None:
            history.append((log_loss(X, y, theta), np.nan))
        else:
            history.append((log_loss(X, y, theta),
                            log_loss(X_holdout, y_holdout, theta)))

    return theta, history


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────

def evaluate(X, y, theta, title):
    """
    Reports accuracy plus the threshold-sensitive metrics.

    Accuracy alone is misleading on an unbalanced target, so precision,
    recall and F1 are shown too: a model that predicts every row as
    negative scores 45% accuracy here while being useless.

    Threshold: sigmoid output >= 0.5 → class 1, else class 0.
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

    print(f"── {title} " + "─" * max(0, 41 - len(title) - 4))
    print(f"   Rows           : {len(y)}")
    print(f"   Log Loss       : {log_loss(X, y, theta):.4f}")
    print(f"   Accuracy       : {accuracy * 100:.2f}%")
    print(f"   Precision      : {precision:.4f}")
    print(f"   Recall         : {recall:.4f}")
    print(f"   F1 Score       : {f1:.4f}")
    print("─────────────────────────────────────────\n")

    return accuracy, predictions


# ─────────────────────────────────────────────
# Visualisation — Loss Curves
# ─────────────────────────────────────────────

def plot_loss_curves(history, title="Logistic Regression — Log Loss"):
    """
    Plots training loss against held-out loss, epoch by epoch.

    Both curves fall together while the model is underfitting, then
    level off. The distance between them is the generalisation gap:
    the point after which extra epochs only fit the training rows more
    precisely without helping unseen data. A single loss curve cannot
    show that, which is why both are drawn.
    """
    train = np.array([h[0] for h in history])
    holdout = np.array([h[1] for h in history])
    epochs = np.arange(1, len(history) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train, color="tab:blue", linewidth=1.5,
             label="Training loss")
    plt.plot(epochs, holdout, color="tab:red", linewidth=1.5,
             label="Held-out loss")
    plt.axhline(np.log(2), color="gray", linestyle=":", linewidth=1,
                label="Predict-the-majority baseline (ln 2 = 0.693)")
    plt.xlabel("Epoch")
    plt.ylabel("Log Loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────

if __name__ == "__main__":
    X, names = encode_features(data)
    y = data[TARGET].to_numpy(dtype=float).reshape(-1, 1)

    train_idx, test_idx = stratified_split(
        y.flatten(), test_fraction=TEST_FRACTION, seed=RANDOM_SEED)

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Split first, scale second — the scaler must never see test rows.
    X_train_s, X_test_s = standardise(X_train, X_test)
    X_train_b = add_bias_column(X_train_s)
    X_test_b  = add_bias_column(X_test_s)

    print("── Setup ────────────────────────────────")
    print(f"   Dataset        : heart.csv ({len(y)} rows)")
    print(f"   Features       : {len(names)} "
          f"(+1 bias) -> {len(names) + 1} parameters")
    print(f"   Positive rate  : {y.mean() * 100:.1f}%")
    print(f"   Train / test   : {len(train_idx)} / {len(test_idx)} "
          f"stratified, seed={RANDOM_SEED}")
    print("─────────────────────────────────────────\n")

    theta, history = gradient_ascent(X_train_b, y_train,
                                     X_holdout=X_test_b, y_holdout=y_test,
                                     epochs=3000, alpha=0.5)

    print("── Training ─────────────────────────────")
    for i in range(0, len(history), 500):
        train_loss, holdout_loss = history[i]
        print(f"   Epoch {i + 1:>4}  |  Train Log Loss: {train_loss:.4f}"
              f"  |  Held-out Log Loss: {holdout_loss:.4f}")
    print("─────────────────────────────────────────\n")

    # Coefficients are on standardised inputs, so their magnitudes are
    # directly comparable — the largest movers on this dataset are the
    # signal, and anything near zero contributed almost nothing.
    coefficients = {name: float(theta[i + 1][0])
                    for i, name in enumerate(names)}

    print("── Learned Parameters (standardised inputs) ─")
    print(f"   Intercept (theta_0) : {theta[0][0]:+.4f}")
    for name, value in sorted(coefficients.items(), key=lambda kv: -abs(kv[1])):
        bar = "#" * int(round(abs(value) * 20))
        print(f"   {name:<22} {value:+.4f}  {bar}")
    print("─────────────────────────────────────────\n")

    train_acc, _ = evaluate(X_train_b, y_train, theta,
                            "Evaluation on Training Set")
    test_acc, _ = evaluate(X_test_b, y_test, theta,
                           "Evaluation on Held-Out Test Set")

    print("── Summary ──────────────────────────────")
    print(f"   Training accuracy : {train_acc * 100:.2f}%")
    print(f"   Held-out accuracy: {test_acc * 100:.2f}%")
    print(f"   Generalisation gap: {(train_acc - test_acc) * 100:.2f} points")
    print("─────────────────────────────────────────")
    print("   The held-out figure is the one to read. It comes from the")
    print("   stratified 20% that gradient ascent never saw, using a")
    print("   scaler fitted on training rows only.")
    print("─────────────────────────────────────────\n")

    plot_loss_curves(history)