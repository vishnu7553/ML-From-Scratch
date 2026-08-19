# ML From Scratch
Machine learning implemented from scratch — no sklearn, no shortcuts. Every algorithm is built from its mathematical foundation, with derivations in the notes and corresponding Python implementations.

<p align="center">
  <a href="https://vishnu7553.github.io/ML-From-Scratch/" target="_blank" title="Open interactive graph (full screen)">
    <img src="https://raw.githubusercontent.com/vishnu7553/ML-From-Scratch/main/docs/graph.svg?v=3" alt="ML concept graph" />
  </a>
  <p>Click open the interactive graph.</p>
</p>

---

## Start Here

> **Recommended:** read the [handwritten notes](HandWritten/ML%20Notes.pdf) for a solid understanding of the math behind each algorithm.

| Topic                              | Notes                                                     | Implementation                                                          |
| ---------------------------------- | --------------------------------------------------------- | ----------------------------------------------------------------------- |
| **Foundations**                    |                                                           |                                                                         |
| Machine Learning                   | [Introduction](Notes/Machine%20Learning.md)               | —                                                                       |
| Supervised Learning                | [Overview](Notes/Supervised%20Learning.md)                | —                                                                       |
| **Regression**                     |                                                           |                                                                         |
| Linear Regression                  | [Derivation](Notes/Linear%20Regression.md)                | [Code](Implementation/Regression/linear_regression.py)                  |
| Locally Weighted Regression        | [Derivation](Notes/Locally%20weighted%20regression.md)    | [Code](Implementation/Regression/locally_weighted_regression.py)        |
| Probabilistic Interpretation (MLE) | [Derivation](Notes/Probabilistic%20Interpretation.md)     | —                                                                       |
| **Classification**                 |                                                           |                                                                         |
| Logistic Regression                | [Derivation](Notes/Logistic%20Regression.md)              | [Code](Implementation/Classification/logistic_regression.py)            |
| Newton's Method                    | [Derivation](Notes/Newton's%20method.md)                  | [Code](Implementation/Classification/newtons_method.py)                 |
| Exponential Family                 | [Derivation](Notes/Exponential%20family.md)               | —                                                                       |
| Generalized Linear Models          | [Derivation](Notes/Generalized%20linear%20models.md)      | [Code](Implementation/Classification/glm_unified_update.py)             |
| Softmax Regression                 | [Derivation](Notes/Softmax%20Regression.md)               | [Code](Implementation/Classification/softmax_regression.py)             |
| **Generative Learning Algorithms** |                                                           |                                                                         |
| Generative Learning Algorithms     | [Overview](Notes/Generative%20Learning%20Algorithms.md)   | —                                                                       |
| Gaussian Discriminant Analysis     | [Derivation](Notes/Gaussian%20Discriminant%20Analysis.md) | [Code](Implementation/Classification/gaussian_discriminant_analysis.py) |
| Naive Bayes                        | [Derivation](Notes/Naive%20bayes.md)                      | [Code](Implementation/Classification/naive_bayes.py)                    |
| **Support Vector Machines**        |                                                           |                                                                         |
| Support Vector Machines            | [Derivation](Notes/Support%20Vector%20Machines.md)        | [Code](Implementation/Classification/svm_margins.py)                    |
| Optimal Margin Classifier          | [Derivation](Notes/Optimal%20margin%20classifier.md)      | [Code](Implementation/Classification/optimal_margin_classifier.py)      |
| Kernels                            | [Derivation](Notes/Kernels.md)                            | [Code](Implementation/Classification/kernels.py)                        |

---

## Structure

```
ML/
├── Notes/                              # Mathematical derivations and concept notes
│   ├── Machine Learning.md
│   ├── Supervised Learning.md
│   ├── Linear Regression.md
│   ├── Locally Weighted Regression.md
│   ├── Probabilistic Interpretation.md
│   ├── Logistic Regression.md
│   ├── Newton's method.md
│   ├── Exponential family.md
│   ├── Generalized linear models.md
│   ├── Softmax Regression.md
│   ├── Generative Learning Algorithms.md
│   ├── Gaussian Discriminant Analysis.md
│   ├── Naive bayes.md
│   ├── Support Vector Machines.md
│   ├── Optimal margin classifier.md
│   └── Kernels.md
│
└── Implementation/                     # From-scratch Python implementations
    ├── Regression/
    │   ├── linear_regression.py
    │   └── locally_weighted_regression.py
    ├── Classification/
    │   ├── logistic_regression.py
    │   ├── newtons_method.py
    │   ├── glm_unified_update.py
    │   ├── softmax_regression.py
    │   ├── gaussian_discriminant_analysis.py
    │   ├── naive_bayes.py
    │   ├── svm_margins.py
    │   ├── optimal_margin_classifier.py
    │   └── kernels.py
    └── datasets/
```

---

## Philosophy

> Understand the math. Write the code. Then and only then, use the library.

Each implementation follows this sequence:

1. Derive the algorithm mathematically in the notes
2. Implement from scratch in Python using only NumPy
3. Verify against a known dataset with evaluation metrics

---

## Running the Code

```bash
# Clone the repo
git clone https://github.com/vishnu7553/ML-From-Scratch
cd ML-From-Scratch/Implementation

# Install dependencies
pip install uv
uv sync

# Run linear regression
python Regression/linear_regression.py

# Run logistic regression
python Classification/logistic_regression.py
```

---

## Author

**Vishnu Vardhan**

- GitHub: [@vishnu7553](https://github.com/vishnu7553)
