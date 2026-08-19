## Definition

Gradient Descent is an iterative first-order optimization algorithm used to find the parameters $\theta$ that minimize a loss function $J(\theta)$.

$$\theta \leftarrow \theta - \alpha \frac{\partial}{\partial \theta} J(\theta)$$

- $\alpha$ — **learning rate**: controls step size per iteration
  - Too small → painfully slow convergence
  - Too large → overshoot, oscillation, divergence
- The gradient points in the direction of **steepest increase**, so we subtract it to descend toward a minimum.

## When It Is Used

- [[Linear Regression]] — minimizes MSE (closed form exists, but GD scales to large data)
- [[Logistic Regression]] — maximizes log-likelihood via the *ascent* form (sign flip)
- [[Newton's method]] — a second-order alternative that uses curvature for faster convergence

## Key Intuition

> The terrain is the loss surface, the gradient is gravity, and $\alpha$ is the joystick. Too aggressive a joystick and you fly off the map; too gentle and you crawl.

## Related Notes

- [[Linear Regression]]
- [[Logistic Regression]]
- [[Newton's method]]
