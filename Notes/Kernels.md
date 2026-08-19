## Definition

Kernels are mathematical functions that let [[Support Vector Machines|SVM]] find a decision boundary in an extremely high — even infinite — dimensional space, while keeping the actual computation as cheap as working in the original input space. They are what completes the SVM:

$$\boxed{\text{SVM} = \text{Optimal Margin} + \text{Kernel}}$$

## Why the Dual Problem, Not the Raw Expansion

Once we prove the optimization objective touches $x_i, x_j$ only as an inner product ($\alpha_i\alpha_j y_iy_j (x_i^Tx_j)$, from [[Optimal margin classifier]]), you might think it's ready to code. But textbooks and papers don't leave it in that raw expanded form — they use the **Dual Optimization Problem** instead. Two reasons:

1. **Messy constraints:** even with the inner product ready for a kernel, the original setup's per-point margin constraints are still there. `Lagrange Duality` is the tool that folds those constraints into the objective, turning a painful constrained minimization into a cleaner maximization over just the $\alpha$ values.
2. **The standardized form is what you'll see in every paper:**

$$\boxed{\max_\alpha \ \sum_{i=1}^{m}\alpha_i - \frac{1}{2}\sum_{i=1}^{m}\sum_{j=1}^{m}\alpha_i\alpha_j, y_iy_j, k(x_i,x_j)}$$

subject to:

$$0 \leq \alpha_i \leq C \quad \text{(bounds and soft margin)} \qquad \sum_{i=1}^{m}\alpha_i y_i = 0$$

**Why researchers love this standardized view:**

- **The kernel slot is glowing:** the formula explicitly isolates $k(x_i,x_j)$ — swapping kernels (linear, polynomial, Gaussian) is as simple as plugging a new function into that one slot.
- **The magic of sparsity:** when optimization software solves this exact dual equation, most $\alpha_i$ values turn out to be exactly zero. The few points with non-zero $\alpha_i$ are the **support vectors** — everything else gets ignored.

## Training and Prediction — Both in Terms of Inner Products

**To train:** solve for $\alpha_i$ and $b$.

**To predict:** compute $h_{w,b}(x)$ for a new test example. Substituting the Representer Theorem's $w = \sum \alpha_i y^{(i)}x^{(i)}$:

$$h_{w,b}(x) = g(w^Tx+b) = g\left(\left(\sum_{i=1}^m \alpha_i y^{(i)} x^{(i)}\right)^T x + b\right)$$

$$\boxed{h_{w,b}(x) = g\left(\sum_{i=1}^{m} \alpha_i, y^{(i)}, \langle x^{(i)}, x\rangle + b\right)}$$

Once the $\alpha$ values are learned, predictions use just inner products again — so **the entire algorithm, both training and prediction, is expressed only in terms of inner products.** That's exactly the property that makes swapping in a kernel function possible everywhere.

## Formulating a Kernel Function — The Recipe

1. Write the algorithm in terms of $\langle x_i, x_j\rangle$ (or $\langle x,z\rangle$).
2. Let there be a mapping from $x$ to a function $\phi(x)$ — some higher dimension. E.g. $x=[x_1,x_2] \rightarrow \phi(x)=[x_1,x_2,x_1x_2,x_1^2x_2,\dots]$
3. Find a way to compute $k(x_i,x_j) = \phi(x_i)^T\phi(x_j)$.
4. Replace $\langle x_i,x_j\rangle$ in the algorithm with $k(x_i,x_j)$.

## Worked Example — The Quadratic Kernel

If $x \in \mathbb{R}^n$, then a mapping to all pairwise products $\phi(x) = [x_ix_j]_{i,j=1}^n$ lives in $\mathbb{R}^{n^2}$. Computing $\phi(x)^T\phi(z)$ explicitly needs $O(n^2)$ time — the brute-force way.

But there's a better way:

$$\boxed{k(x,z) = \phi(x)^T\phi(z) = (x^Tz)^2}$$

Expanding $(x^Tz)^2$ by hand confirms this matches the brute-force $O(n^2)$ sum exactly — but computing $(x^Tz)^2$ directly only costs $O(n)$: compute the inner product $x^Tz$ once ($O(n)$), then square a single number ($O(1)$).

> Though the kernel be any type or order, as long as it follows the inner product format, the computation stays $O(n)$ — this is the entire point of the kernel trick.

## The General Polynomial Kernel

$$\boxed{k(x,z) = (x^Tz+c)^d}$$

For $d=5$ this corresponds to a mapping $\phi(x)$ containing $\binom{n+2}{2}$ features — every monomial up to degree $d$. For a 2D input $x=[x_1,x_2]^T$, $\phi(x)$ builds a vector containing all combinations where the sum of exponents is $\leq d$: $x_1^5$, $\sqrt{5c},x_1^4x_2$, $\sqrt{10c},x_1^3x_2^2$, ..., down through every lower-degree term. The square roots and constants like $c$ pop out naturally during the algebraic expansion, purely to make $\phi(x)^T\phi(z)$ match the kernel formula precisely.

## What Non-Linearity Actually Means Here

SVM is finding an optimal margin in a very high-dimensional space, while keeping computation linear ($O(n)$) using kernels. So where does the _non-linear_ decision boundary in the original space come from?

**SVM finds a linear decision boundary in the higher dimension — not in the base dimension of the input features.** When you project that linear boundary back down to the original dimension, it appears as a non-linear boundary. A flat plane slicing through a 3D "bowl" shape, viewed back down in 2D, traces out a circle — the boundary was linear the entire time, just not in the space you're looking at it from.

## Kernels as a Similarity Measure

If $(x,z)$ in a kernel function are similar/close to each other, the inner product $k(x,z) = \phi(x)^T\phi(z)$ is **large**. If they're dissimilar/distant, $k(x,z)$ is **small**.

**Important distinction — a kernel is a metric, not a probability.** Even for the Gaussian kernel below, whose shape resembles a normal distribution, its job is not to compute probability density — it's to act as a similarity metric, telling the algorithm how close two points are so it can construct the optimal margin.

## The Gaussian (RBF) Kernel

$$\boxed{k(x,z) = \exp\left(-\frac{|x-z|^2}{2\sigma^2}\right)}$$

If $x$ and $z$ are close, the output is close to 1 (since $e^0 \approx 1$). If they're far apart, the output is close to 0.

## Proving the Gaussian Kernel Is Valid — Mercer's Theorem

**The big question:** can we just invent any similarity function and call it a kernel? **The answer is no** — for a function to be a valid kernel, there must exist a feature mapping $\phi(x)$ such that $k(x,z) = \phi(x)^T\phi(z)$. If a function doesn't correspond to a real inner product, SVM's optimization math breaks.

For powerful kernels like Gaussian (RBF), $\phi(x)$ is infinite- dimensional — we literally cannot write it down manually. So how do we know it's legal?

**Mercer's Theorem (the solution):** instead of forcing you to construct an infinite-dimensional $\phi(x)$ to prove its existence, Mercer's Theorem gives a shortcut to verify the _function itself_.

**The test:** take a set of $m$ training examples and compute the **Kernel (Gram) Matrix**, where every entry is $k(x^{(i)}, x^{(j)})$ for all combinations of points:

- The resulting $m\times m$ matrix must be **symmetric**.
- The matrix must be **positive semi-definite**.

If a kernel function passes this test for _any_ set of points, Mercer's Theorem guarantees a corresponding $\phi(x)$ exists. You never need to know what $\phi(x)$ actually is — only that it exists.

### Proving the Gaussian Kernel, via Taylor Series

Expand $|x-z|^2 = |x|^2 + |z|^2 - 2x^Tz$ and substitute back:

$$k(x,z) = \exp\left(-\frac{|x|^2}{2\sigma^2}\right) \cdot \exp\left(-\frac{|z|^2}{2\sigma^2}\right) \cdot \exp\left(\frac{x^Tz}{\sigma^2}\right)$$

The first two factors depend only on $x$ alone or $z$ alone. The only part where $x$ and $z$ mix is the final exponential term $\exp\left(\frac{x^Tz}{\sigma^2}\right)$.

Using the Taylor series $e^u = \sum_{k=0}^{\infty}\frac{u^k}{k!}$ with $u = \frac{x^Tz}{\sigma^2}$:

$$\exp\left(\frac{x^Tz}{\sigma^2}\right) = \sum_{k=0}^{\infty}\frac{1}{k!}\left(\frac{x^Tz}{\sigma^2}\right)^k$$

**Connecting the dots:** inside that infinite sum are powers of dot products, $(x^Tz)^k$. We already proved that powers of dot products like $(x^Tz+c)^d$ correspond to a valid polynomial feature mapping. Adding an infinite number of valid polynomial mappings together still results in a valid mapping.

**Conclusion:** $k(x,z)=\exp\left(-\frac{|x-z|^2}{2\sigma^2}\right)$ is a valid kernel function. Even though we can't write down a finite vector for $\phi(x)$, expanding it via Taylor series proves it acts as an infinite sum of polynomial inner products — satisfying Mercer's Theorem automatically, making it a 100% legal kernel. This kernel is called the **Gaussian kernel** because it uses the same mathematical mechanism as the Gaussian probability density function (without actually being one).

## Common Kernel Types

|Kernel|Formula|Notes|
|---|---|---|
|Linear|$k(x,z) = x^Tz$, $\phi(x)=x$|No higher-dimensional mapping at all — doesn't use the kernel trick's real power, but is technically still a valid kernel|
|Polynomial|$k(x,z) = (x^Tz+c)^d$|Similarity over polynomials of the original variables|
|Radial Basis Function (RBF / Gaussian)|$k(x,z) = \exp(-\|x-z\|^2/2\sigma^2)$|Popular default; infinite-dimensional feature space|
|Sigmoid|—|Equivalent to a two-layer perceptron neural network|

**The kernel trick is general** — any algorithm that can be written in terms of $\langle x_i,x_j\rangle$ can use kernels, not just SVM. This applies to any [[Generalized linear models|GLM]] (Linear, Logistic, etc.), fitting the model to an infinite-dimensional space to produce non-linear decision boundaries.

## Soft Margin — Handling Outliers and Noise

The hard-margin constraint $y^{(i)}(w^Tx^{(i)}+b) \geq 1$ forces the algorithm to focus on every training example without exception. The problem: real-world data is chaotic, not the clean, perfectly-separable data theory assumes.

- **The noise trap:** forcing a clean separation of every noisy point produces a jagged, hyper-overfit boundary that fails on new data.
- **The lone outlier:** the hard-margin classifier optimizes for the worst case. A single outlier sitting far from its class will violently tilt the entire decision boundary just to satisfy that one point.

### The Fix — Slack Variables (L1 Norm Soft Margin)

Relax the per-sample constraint by introducing a slack variable $\varepsilon_i \geq 0$:

$$\varepsilon_i \geq 0 \quad \text{such that} \quad y^{(i)}(w^Tx^{(i)}+b) \geq 1-\varepsilon_i$$

For any sample where $\varepsilon_i > 0$, the margin constraint is violated proportionally. Without a penalty, $\varepsilon_i$ would drift toward infinity, making the optimization ill-posed — so a penalty term is added to the objective:

$$\boxed{\min_{w,b,\varepsilon}\ \frac{1}{2}|w|^2 + C\sum_{i=1}^{n}\varepsilon_i}$$

**The hyperparameter $C$** controls the trade-off between margin width and classification error:

||Effect|
|---|---|
|**Higher $C$**|Penalizes violations heavily — pushes the model to fit every training example (higher variance, risk of overfitting)|
|**Lower $C$**|Allows more violations, favoring a wider, smoother margin (higher bias, better generalization)|

### The Dual Form With Soft Margin

The dual objective stays the same as the hard-margin case, but with one critical added bound on the Lagrange multipliers:

$$\boxed{0 \leq \alpha_i \leq C}$$

This box constraint prevents any single training point from exerting too much influence over the final boundary. Solving this quadratic program used to be expensive; modern optimization libraries handle it efficiently, making soft-margin SVM a practical, plug-and-play tool.

**This completes the SVM derivation** — margins, the optimal margin classifier, kernels, and soft margin together are the full model.

## Real-World Applications

- **Handwritten digit recognition (MNIST):** 28×28 pixel images become a 784-dimensional feature vector of pixel intensities, $x \in \mathbb{R}^{784}$. SVM classifies which digit (0–9) the image represents.
- **Protein sequence analysis:** feature vectors built from amino-acid composition can reach tens of thousands of dimensions, making kernels essential for keeping classification computationally tractable.

## Implementation

- [[../Implementation/Classification/kernels.py]]

---

## Related Notes

- [[Support Vector Machines]]
- [[Optimal margin classifier]]
- [[Generalized linear models]]

## References

- _Mathematics for Machine Learning_ — Deisenroth et al. (mml-book.github.io)
- Stanford CS229 Lecture Notes — cs229.stanford.edu
- _Dive into Deep Learning_ — d2l.ai