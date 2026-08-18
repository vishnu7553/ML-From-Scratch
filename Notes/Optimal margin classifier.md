## Definition

The Optimal Margin Classifier is the formal optimization problem that turns [[Support Vector Machines|SVM's]] geometric margin maximization into a solvable convex problem — the concrete algorithm that finds the best-fit separating hyperplane, for the case where the data is linearly separable.

## From Margins to an Optimization Problem

Recap of both margins (see [[Support Vector Machines]]):

- **Functional margin:** $\hat\gamma^{(i)} = y^{(i)}(w^Tx^{(i)}+b)$ — measures prediction confidence, but suffers from a scaling flaw.
- **Geometric margin:** $\gamma^{(i)} = \frac{\hat\gamma^{(i)}}{|w|}$ — fixes the flaw by converting the measure into a true physical Euclidean distance, immune to parameter scaling.

The core objective: find parameters $w$ and $b$ that maximize the geometric margin across the _entire_ training set, focusing on the worst case (the closest point):

$$\max_{w,b}\ \gamma \quad \text{subject to} \quad \frac{\hat\gamma^{(i)}}{|w|} \geq \gamma \ \text{ for all } i=1,\dots,n$$

## Eliminating the Scaling Ambiguity

Maximizing this directly involves both the functional margin $\hat\gamma$ and the weight norm $|w|$ — the scaling ambiguity of the functional margin creates an optimization challenge, since any solution can be infinitely scaled up.

To eliminate this, we take advantage of the fact that scaling $w$ and $b$ doesn't affect the decision boundary at all — we can freely choose a scale for our parameters that neutralizes the ambiguity. **We set the functional margin of the closest training example to a fixed constant, typically 1:**

$$\boxed{\hat\gamma = \min_i\ y^{(i)}(w^Tx^{(i)}+b) = 1}$$

By fixing the functional margin to 1, every other training example is constrained to have a functional margin of at least 1:

$$y^{(i)}(w^Tx^{(i)}+b) \geq 1 \quad \text{for all } i$$

## The Optimization Problem Simplifies

With the functional margin fixed at 1, the geometric margin equation simplifies dramatically:

$$\gamma = \frac{\hat\gamma}{|w|} = \frac{1}{|w|}$$

Maximizing $\gamma = \frac{1}{|w|}$ is mathematically identical to **minimizing** the denominator $|w|$. For convenience in calculus and optimization, minimizing $|w|$ is equivalent to minimizing its squared half-norm:

$$\gamma = \frac{1}{2}|w|^2$$

This boils everything down into the final optimization problem — the Optimal Margin Classifier:

$$\boxed{\min_{w,b}\ \frac{1}{2}|w|^2 \quad \text{subject to} \quad y^{(i)}(w^Tx^{(i)}+b) \geq 1 \ \text{ for } i=1,\dots,m}$$

## Summary

1. **Functional margin** gives a baseline measure of confidence.
2. **Geometric margin** turns that confidence into a true, scale-invariant physical distance.
3. **Optimal margin** leverages a clever scaling choice ($\hat\gamma=1$) to convert the geometric margin maximization task into a clean **convex quadratic programming problem** ($\min \frac{1}{2}|w|^2$) that yields a unique, perfectly-distanced decision boundary.

The Optimal Margin Classifier is just $\frac{1}{3}$ of SVM — adding [[Kernels]] and solving inseparable cases is what finally makes a complete SVM model.

## The Representer Theorem — The Key to Infinite Dimensions

The optimal margin classifier finds the best-fit line between classes. But what if classes aren't neatly separated by a straight line — what if the data is a mess, like a circle inside a square? We already know the fix: move to a higher-dimensional space (3D, 4D, or even infinite $\infty$D), where we could find the optimal classification line. But this higher-dimension computation — literally $\infty$D — becomes extremely expensive, or outright impossible. This leads to the core of SVM.

**The Representer Theorem states:** the minimizer of a regularized empirical risk function over a reproducing kernel Hilbert space (or any high-dimensional feature space) can always be represented as a **finite linear combination of the training data points**:

$$\boxed{w = \sum_{i=1}^{m} \alpha_i, y^{(i)}, \phi(x^{(i)})}$$

In short: even if we map data into an infinitely complex space, the optimal weight vector $w$ never wanders off into some weird, unconstrained direction — it lives entirely within the span of our training examples.

**Linear combination assumption:**

$$w = \sum_{i=1}^{m} \alpha_i, y^{(i)}, x^{(i)}$$

**Why does this work?** Even if features $x$ are 100-trillion dimensional, the Representer Theorem proves that at the optimal values of $w$, we don't lose any performance by making this assumption.

## Two Intuitions for Why $w$ Lives in the Span of the Data

**Intuition 1 — Gradient Descent:** Recall Logistic Regression's update rule: $\theta := \theta - \alpha(h_\theta(x^{(i)}) - y^{(i)})x^{(i)}$. Every time you update the weights, you're adding or subtracting a multiple of a training example $x$. By induction, no matter how many iterations you run, $w$ will always be a linear combination of your training examples. This holds for batch gradient descent too:

$$\theta := \theta - \alpha\sum_{i=1}^{m}(h_\theta(x^{(i)}) - y^{(i)})x^{(i)}$$

**Intuition 2 — Geometry:** The vector $w$ pins the direction of the decision boundary and is always **perpendicular** to it. In high-dimensional space, $w$ naturally aligns with the span of your data points.

Example: $g([2,1]\cdot x - 2)$ gives decision boundary $w^Tx+b = 2x_1 + 1x_2 - 2 = 0$. Visualizing this, the weight vector $w$ always pins the decision boundary perpendicular to itself, and the bias term $b$ moves the boundary back and forth along that direction — $w$ sets the _direction_ of the decision boundary, $b$ _shifts_ it.

## Implementation

- [[../Implementation/Classification/optimal_margin_classifier.py]]

---

## Related Notes

- [[Support Vector Machines]]
- [[Kernels]]
- [[Logistic Regression]]

## References

- _Mathematics for Machine Learning_ — Deisenroth et al. (mml-book.github.io)
- Stanford CS229 Lecture Notes — cs229.stanford.edu
- _Dive into Deep Learning_ — d2l.ai