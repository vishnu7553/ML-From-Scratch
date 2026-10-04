# ML From Scratch

A study repo for classical machine learning, built while working through
Stanford CS229. Each topic has notes on the math and, where it makes sense,
a NumPy implementation.

This is a learning project, not a library and not a benchmark. The code is
written to be read.

---

## How to use this repo

1. Read the math: the [handwritten notes](HandWritten/ML%20Notes.pdf) (PDF),
   or the typed notes in `Notes/`.
2. Read the matching script in `Implementation/`.
3. Run it and compare the output with what the notes predict.

---

## Contents

`—` means notes only, no code yet.

| Topic | Notes | Code |
| --- | --- | --- |
| **Foundations** | | |
| Machine Learning | [Introduction](Notes/Machine%20Learning.md) | — |
| Supervised Learning | [Overview](Notes/Supervised%20Learning.md) | — |
| Loss Functions | [Concept](Notes/Loss%20Functions.md) | — |
| Gradient Descent | [Concept](Notes/Gradient%20Descent.md) | — |
| **Regression** | | |
| Linear Regression | [Derivation](Notes/Linear%20Regression.md) | [Code](Implementation/Regression/linear_regression.py) |
| Locally Weighted Regression | [Derivation](Notes/Locally%20weighted%20regression.md) | [Code](Implementation/Regression/locally_weighted_regression.py) |
| Probabilistic Interpretation (MLE) | [Derivation](Notes/Probabilistic%20Interpretation.md) | — |
| **Classification** | | |
| Sigmoid Function | [Concept](Notes/Sigmoid%20Function.md) | — |
| Logistic Regression | [Derivation](Notes/Logistic%20Regression.md) | [Code](Implementation/Classification/logistic_regression.py) |
| Newton's Method | [Derivation](Notes/Newton's%20method.md) | [Code](Implementation/Classification/newtons_method.py) |
| Exponential Family | [Derivation](Notes/Exponential%20family.md) | — |
| Generalized Linear Models | [Derivation](Notes/Generalized%20linear%20models.md) | [Code](Implementation/Classification/glm_unified_update.py) |
| Softmax Regression | [Derivation](Notes/Softmax%20Regression.md) | [Code](Implementation/Classification/softmax_regression.py) |
| **Generative Learning Algorithms** | | |
| Generative Learning Algorithms | [Overview](Notes/Generative%20Learning%20Algorithms.md) | — |
| Gaussian Discriminant Analysis | [Derivation](Notes/Gaussian%20Discriminant%20Analysis.md) | [Code](Implementation/Classification/gaussian_discriminant_analysis.py) |
| Naive Bayes | [Derivation](Notes/Naive%20bayes.md) | [Code](Implementation/Classification/naive_bayes.py) |
| **Support Vector Machines** | | |
| Support Vector Machines | [Derivation](Notes/Support%20Vector%20Machines.md) | [Code](Implementation/Classification/svm_margins.py) |
| Optimal Margin Classifier | [Derivation](Notes/Optimal%20margin%20classifier.md) | [Code](Implementation/Classification/optimal_margin_classifier.py) |
| Kernels | [Derivation](Notes/Kernels.md) | [Code](Implementation/Classification/kernels.py) |
| **Generalization** | | |
| Overfitting and Underfitting | [Concept](Notes/Overfitting%20and%20Underfitting.md) | — |
| Bias and Variance | [Concept](Notes/Bias%20And%20Variance.md) | [Code](Implementation/Regression/bias_variance_tradeoff.py) |
| Bias-Variance Tradeoff | [Derivation](Notes/Bias-Variance%20Tradeoff.md) | — |
| Regularization | [Derivation](Notes/Regularization.md) | [Code](Implementation/Regression/regularization.py) |
| Model Selection and Cross-Validation | [Derivation](Notes/Model%20Selection%20and%20Cross-Validation.md) | [Code](Implementation/Regression/model_selection_cv.py) |
| **Learning Theory** | | |
| Learning Theory | [Concept](Notes/Learning%20Theory.md) | — |
| Empirical Risk Minimization | [Derivation](Notes/Empirical%20Risk%20Minimization.md) | [Code](Implementation/Generalization/erm_vs_true_risk.py) |
| Uniform Convergence | [Derivation](Notes/Uniform%20Convergence.md) | [Code](Implementation/Generalization/hoeffding_uniform_convergence.py) |
| VC Dimension | [Derivation](Notes/VC%20Dimension.md) | [Code](Implementation/Generalization/vc_dimension_shattering.py) |

---

## Layout

- `HandWritten/`: handwritten notes (PDF)
- `Notes/`: typed notes, one file per topic
- `Implementation/`: Python scripts in `Regression/`, `Classification/`
  and `Generalization/`, with small CSV files in `datasets/`

---

## Conventions and limits

- The algorithm math is written in NumPy, with no sklearn. pandas loads the
  CSVs and matplotlib draws the plots. The exception is the SVM files, which
  hand their hand-written objective and constraints to
  `scipy.optimize.minimize` — a dedicated quadratic-programming solver for
  SVMs is out of scope, and the code notes this in a comment in
  `optimal_margin_classifier.py`.
- Datasets are deliberately small, so the data stays easy to inspect and the
  output is about how the algorithm works. Metrics are computed on the
  training data unless a script says otherwise. They show the algorithm
  working, not how it generalizes.
- Scripts are standalone demos, run one at a time. There are no automated
  tests yet.
- Not covered: neural networks and backpropagation, decision trees,
  unsupervised learning, reinforcement learning.

---

## Running the code

```bash
git clone https://github.com/vishnu7553/ML-From-Scratch
cd ML-From-Scratch/Implementation

pip install uv
uv sync

python Regression/linear_regression.py
python Classification/logistic_regression.py
```

Some scripts open plot windows.

---

## Philosophy

> Understand the math. Write the code. Then and only then, use the library.

---

## Author

**Vishnu Vardhan** · [@vishnu7553](https://github.com/vishnu7553) · MIT license