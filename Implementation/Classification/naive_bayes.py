import numpy as np

# ─────────────────────────────────────────────
# Dataset — tiny synthetic spam/not-spam corpus
# ─────────────────────────────────────────────

# Small enough to read by eye, but big enough to trigger the zero-
# probability problem the notes describe (the word "prize" only ever
# appears in spam, never in non-spam, in this training set).
emails = [
    ("win money now claim prize",        1),
    ("free money click here now",        1),
    ("claim your free prize today",      1),
    ("urgent click to win cash",         1),
    ("meeting scheduled for tomorrow",   0),
    ("project report due friday",        0),
    ("lunch with team tomorrow",         0),
    ("please review the attached report",0),
]

texts = [e[0] for e in emails]
labels = np.array([e[1] for e in emails])


def build_vocabulary(texts):
    """Builds the vocabulary from every unique word in the training set."""
    vocab = sorted(set(word for text in texts for word in text.split()))
    return {word: i for i, word in enumerate(vocab)}


def text_to_vector(text, vocab):
    """
    x_i = 1 if word i (from the vocabulary) appears in this email, else 0.
    This is the same {0,1}^n binary feature vector the notes describe —
    presence/absence, not word count.
    """
    x = np.zeros(len(vocab))
    for word in text.split():
        if word in vocab:
            x[vocab[word]] = 1
    return x


vocab = build_vocabulary(texts)
X = np.array([text_to_vector(t, vocab) for t in texts])


# ─────────────────────────────────────────────
# Method — Fit via Closed-Form MLE (+ Laplace Smoothing)
# ─────────────────────────────────────────────

def fit_naive_bayes(X, y, laplace=True):
    """
    Fits phi_y and phi_j|y for both classes via their closed-form MLE
    solutions, with optional Laplace smoothing.

    Without smoothing:
        phi_j|y=1 = count(x_j=1, y=1) / count(y=1)

    With Laplace smoothing (+1 numerator, +2 denominator — pretending
    we saw the word once in each class):
        phi_j|y=1 = (count(x_j=1, y=1) + 1) / (count(y=1) + 2)
    """
    m, n = X.shape

    phi_y = np.mean(y == 1)

    if laplace:
        phi_j_given_1 = (X[y == 1].sum(axis=0) + 1) / (np.sum(y == 1) + 2)
        phi_j_given_0 = (X[y == 0].sum(axis=0) + 1) / (np.sum(y == 0) + 2)
    else:
        phi_j_given_1 = X[y == 1].sum(axis=0) / np.sum(y == 1)
        phi_j_given_0 = X[y == 0].sum(axis=0) / np.sum(y == 0)

    return phi_y, phi_j_given_1, phi_j_given_0


# ─────────────────────────────────────────────
# Prediction — Naive Bayes Product Rule
# ─────────────────────────────────────────────

def predict_proba(x, phi_y, phi_j_given_1, phi_j_given_0):
    """
    p(x|y) = product over all words of p(x_j|y) — the naive
    conditional-independence assumption applied directly.

    Computed in log-space to avoid numerical underflow from multiplying
    many small probabilities together (a real concern with a 10,000-
    word vocabulary, even though this toy example is tiny).
    """
    log_p_x_given_1 = np.sum(
        x * np.log(phi_j_given_1) + (1 - x) * np.log(1 - phi_j_given_1)
    )
    log_p_x_given_0 = np.sum(
        x * np.log(phi_j_given_0) + (1 - x) * np.log(1 - phi_j_given_0)
    )

    log_p_1 = log_p_x_given_1 + np.log(phi_y)
    log_p_0 = log_p_x_given_0 + np.log(1 - phi_y)

    # normalize back out of log-space for a readable probability
    max_log = max(log_p_1, log_p_0)
    p1 = np.exp(log_p_1 - max_log)
    p0 = np.exp(log_p_0 - max_log)
    return p1 / (p1 + p0)


def predict(x, phi_y, phi_j_given_1, phi_j_given_0):
    """arg max version — the winning class only, matching the notes."""
    return int(predict_proba(x, phi_y, phi_j_given_1, phi_j_given_0) >= 0.5)


# ─────────────────────────────────────────────
# Demo — The Zero-Probability Problem, With and Without Smoothing
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Vocabulary ({len(vocab)} words): {list(vocab.keys())}\n")

    # A new email containing "prize" — which in this training set only
    # ever appeared in spam emails, never in non-spam.
    test_email = "team prize tomorrow"
    x_test = text_to_vector(test_email, vocab)
    print(f"Test email: '{test_email}'\n")

    print("── Without Laplace Smoothing ────────────")
    phi_y, phi1, phi0 = fit_naive_bayes(X, labels, laplace=False)
    prize_idx = vocab["prize"]
    print(f"   phi_'prize'|y=0 (non-spam) = {phi0[prize_idx]:.4f}  <- exactly zero")
    with np.errstate(divide="ignore", invalid="ignore"):
        p = predict_proba(x_test, phi_y, phi1, phi0)
    if np.isnan(p):
        print("   FAILED: p(spam | email) = NaN")
        print("   log(0) from the zero-probability word collapses the")
        print("   entire product -> the classifier cannot score this email at all.")
    else:
        print(f"   p(spam | email) = {p:.4f}")
    print("─────────────────────────────────────────\n")

    print("── With Laplace Smoothing ───────────────")
    phi_y, phi1, phi0 = fit_naive_bayes(X, labels, laplace=True)
    print(f"   phi_'prize'|y=0 (non-spam) = {phi0[prize_idx]:.4f}  <- no longer zero")
    p = predict_proba(x_test, phi_y, phi1, phi0)
    pred = predict(x_test, phi_y, phi1, phi0)
    print(f"   p(spam | email) = {p:.4f}")
    print(f"   Prediction: {'SPAM' if pred == 1 else 'NOT SPAM'}")
    print("─────────────────────────────────────────\n")

    # Full evaluation on the training set
    print("── Evaluation on Training Set ───────────")
    correct = 0
    for text, true_label in zip(texts, labels):
        x = text_to_vector(text, vocab)
        pred = predict(x, phi_y, phi1, phi0)
        correct += (pred == true_label)
        print(f"   '{text[:35]:<35}' -> {'SPAM' if pred else 'NOT SPAM':<9} "
              f"(true: {'SPAM' if true_label else 'NOT SPAM'})")
    print(f"\n   Accuracy: {correct}/{len(texts)} = {100*correct/len(texts):.1f}%")
    print("─────────────────────────────────────────")