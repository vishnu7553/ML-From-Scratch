# ML From Scratch

Machine learning implemented from scratch — no sklearn, no shortcuts.
Every algorithm is built from its mathematical foundation, with
derivations in the notes and corresponding Python implementations.

---

## Start Here

The notes are the primary resource. Each topic is derived mathematically
before any code is written.

| Topic | Notes | Implementation |
|---|---|---|
| Machine Learning | [Introduction](Notes/Machine%20Learning.md) | — |
| Supervised Learning | [Overview](Notes/Supervised%20Learning.md) | — |
| Linear Regression | [Derivation](Notes/Linear%20Regression.md) | [Code](Implementation/Regression/linear_regression.py) |

---

## Structure

```
ML/
├── Notes/               # Mathematical derivations and concept notes
│   ├── Machine Learning.md
│   ├── Supervised Learning.md
│   └── Linear Regression.md
│
└── Implementation/      # From-scratch Python implementations
    ├── Regression/
    │   └── linear_regression.py
    ├── Classification/
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
git clone https://github.com/vishnu7553/ml-from-scratch
cd ml-from-scratch/Implementation

# Install dependencies
pip install uv
uv sync

# Run linear regression
python Regression/linear_regression.py
```

---

## Author

**Vishnu Vardhan**
- GitHub: [@vishnu7553](https://github.com/vishnu7553)
