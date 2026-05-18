# Behavioral Sensitivity Quantization (BSQ): Function-Aware Mixed-Precision Quantization via Adaptive Allocation

## Abstract

Neural network quantization is often driven by parameter-space heuristics that treat magnitude or curvature as proxies for functional importance. In overparameterized models, however, parameter fidelity does not necessarily imply behavioral fidelity. We propose **Behavioral Sensitivity Quantization (BSQ)**, a post-training mixed-precision framework that allocates bit-widths according to empirical sensitivity measured in function space. BSQ estimates the relative fragility of structural units by probing output deviation under small parameter perturbations, using efficient Jacobian-vector products and stochastic trace estimation on a lightweight calibration set. This sensitivity score is then used to solve a constrained bit-allocation problem that minimizes an approximate functional distortion objective under a global memory budget. Unlike methods that rely directly on Hessian estimation, BSQ avoids explicit second-order computation; unlike replay-heavy approaches, it requires only a small unlabeled calibration set. The resulting method is simple, hardware-friendly, and compatible with standard post-training quantization pipelines. Empirically and theoretically, BSQ provides a grounded framework expected to improve the accuracy–efficiency trade-off in low-bit regimes for both vision and language models.

---

## 1. Introduction

Deploying large neural networks on resource-constrained hardware requires compression methods that preserve task behavior under severe memory limits. Quantization is one of the most effective approaches, but most post-training methods still operate primarily in parameter space, using weight magnitude, activation scale, or curvature as indirect indicators of importance. These signals are useful, but they are not equivalent to the model's actual functional sensitivity.

This distinction matters most in modern overparameterized architectures, where many parameter perturbations have little effect on output while a small subset of units can be behaviorally fragile. In such settings, preserving every parameter equally is wasteful, and preserving the largest weights is often insufficient. What matters is not only how large a parameter is, but how much the model's output changes when that parameter is perturbed.

We take this view literally and recast quantization as a **function-aware resource allocation problem**. Instead of assigning bits uniformly or by weight statistics, we estimate how sensitive each structural unit is to small perturbations and allocate precision accordingly. Units with high behavioral sensitivity receive more bits; robust units are quantized more aggressively.

Our contributions are threefold. First, we formalize mixed-precision quantization as a constrained optimization problem over function-space distortion, with memory cost weighted by parameter count per unit. Second, we introduce a lightweight behavioral sensitivity estimator based on stochastic Jacobian probing, requiring only a small calibration set and no fine-tuning. Third, we derive a practical bit-allocation rule that turns sensitivity estimates into discrete precision assignments compatible with standard deployment workflows.

The result is a post-training quantization method that is conceptually simple, mathematically grounded as a first-order surrogate, and practically deployable.

---

## 2. Problem Formulation

Let $f_\theta : \mathcal{X} \rightarrow \mathcal{Y}$ be a pretrained model with parameters $\theta$, partitioned into $K$ structural units:

$$
\theta = \{\theta_1, \dots, \theta_K\}, \qquad \theta_k \in \mathbb{R}^{N_k}.
$$

We seek a quantized model $f_{\theta_Q}$ in which each unit $k$ is assigned a bit-width $b_k \in \{b_{\min}, \dots, b_{\max}\}$. The goal is to minimize behavioral distortion under a total bit budget:

$$
\min_{\{b_k\}} \mathbb{E}_{x \sim \mathcal{D}} \left[ d\left(f_\theta(x), f_{\theta_Q}(x)\right) \right]
$$

subject to

$$
\sum_{k=1}^{K} N_k b_k \le B_{\text{total}}.
$$

Here $d(\cdot,\cdot)$ is a task-appropriate divergence, such as squared error for regression or KL divergence for classification. The memory constraint includes the number of parameters per unit, which is essential because bit-width choices are not cost-equivalent across layers of different sizes.

---

## 3. Behavioral Sensitivity Estimation

### 3.1 Definition

We define the behavioral sensitivity of unit $k$ as the expected output deviation induced by a small perturbation to that unit:

$$
S_k = \mathbb{E}_{x \sim \mathcal{D}, \delta_k} \left[\left\| f_\theta(x) - f_{\theta + \Delta_k(\delta_k)}(x) \right\|_2^2 \right],
$$

where $\Delta_k(\delta_k)$ applies the perturbation only to $\theta_k$.

Intuitively, a unit with large $S_k$ is functionally fragile and should be quantized more conservatively.

### 3.2 First-Order Approximation

For small perturbations, we linearize the model around $\theta$:

$$
f_{\theta + \Delta_k(\delta_k)}(x) \approx f_\theta(x) + J_k(x)\delta_k,
$$

where $J_k(x) = \frac{\partial f_\theta(x)}{\partial \theta_k}$ is the Jacobian with respect to unit $k$.

This gives the approximation

$$
S_k \approx \mathbb{E}_{x,\delta_k} \left[\|J_k(x)\delta_k\|_2^2\right].
$$

If $\delta_k$ is isotropic with zero mean and covariance $\sigma^2 I$, then

$$
S_k \propto \mathbb{E}_x \left[\|J_k(x)\|_F^2\right].
$$

### 3.3 Stochastic Estimation

We estimate this quantity using Hutchinson-style probing. For random vectors $v_m$ with zero mean and unit covariance,

$$
\|J_k(x)\|_F^2 = \operatorname{tr}\left(J_k(x)^\top J_k(x)\right) \approx \frac{1}{M}\sum_{m=1}^{M} \|J_k(x)v_m\|_2^2.
$$

This provides a scalable estimate of sensitivity without materializing the full Jacobian. In practice, the estimator is computed on a small calibration set and averaged across samples:

$$
\widehat{S}_k = \frac{1}{|\mathcal{D}_{\text{calib}}|} \sum_{x \in \mathcal{D}_{\text{calib}}} \frac{1}{M}\sum_{m=1}^{M} \|J_k(x)v_m\|_2^2.
$$

This makes the estimator cheap enough for large models while remaining tied directly to output stability.

---

## 4. Precision Allocation Strategy

### 4.1 Surrogate Optimization

We model the distortion induced by quantizing unit $k$ at bit-width $b_k$ as

$$
D_k(b_k) \approx C_k 2^{-2b_k},
$$

where $C_k$ is a unit-specific constant proportional to $\widehat{S}_k$.

This yields the approximate optimization problem

$$
\min_{\{b_k\}} \sum_{k=1}^{K} \widehat{S}_k 2^{-2b_k}
\quad \text{s.t.} \quad
\sum_{k=1}^{K} N_k b_k \le B_{\text{total}}.
$$

This is not an exact rate-distortion theorem for neural quantization; it is a first-order surrogate motivated by classical uniform quantization behavior and the local linearization above.

### 4.2 Closed-Form Relaxation

Ignoring integrality for the moment, the Lagrangian is

$$
\mathcal{L} = \sum_{k=1}^{K} \widehat{S}_k 2^{-2b_k} + \lambda\left(\sum_{k=1}^{K}N_k b_k - B_{\text{total}}\right).
$$

Setting $\partial \mathcal{L}/\partial b_k = 0$ gives

$$
-2\ln(2)\widehat{S}_k 2^{-2b_k} + \lambda N_k = 0,
$$

hence

$$
b_k^* = \frac{1}{2}\log_2\left(\frac{2\ln(2)\widehat{S}_k}{\lambda N_k}\right).
$$

So the optimal relaxed allocation is monotone in $\widehat{S}_k/N_k$: units with larger sensitivity per parameter receive more bits.

### 4.3 Discrete Allocation

Since hardware supports only discrete precisions, we project the relaxed solution onto the available set, e.g. $\{2,3,4,8\}$ bits. A practical procedure is:

1. Compute scores $r_k = \log(\widehat{S}_k/N_k + \varepsilon)$.
2. Normalize $r_k$ across units.
3. Map to the allowed bit set using monotone binning.
4. Adjust greedily until $\sum_k N_k b_k \le B_{\text{total}}$.

This retains the ranking induced by sensitivity while satisfying the exact memory constraint.

---

## 5. Related Work

Post-training quantization methods such as GPTQ and AWQ estimate quantization error using weight- or activation-based surrogates. Hessian-aware methods like HAWQ and related variants use curvature information to guide precision assignment, but they can become expensive at scale. Recent mixed-precision approaches incorporate learned masks, calibration-based salience, or channel-wise heuristics.

BSQ differs in two ways. First, it measures fragility directly in function space rather than relying on parameter statistics as proxies. Second, it uses a lightweight stochastic estimator that avoids explicit Hessian computation and does not require fine-tuning. In that sense, BSQ sits between purely heuristic PTQ and heavier curvature-based methods: it is behavioral, but still cheap enough to be practical.

---

## 6. Proposed Evaluation Protocol

### 6.1 Setup

To validate BSQ empirically, we propose the following evaluation protocol on representative vision and language models, including a convolutional backbone, a ViT-style architecture, and a large decoder-only language model. The calibration set should be small, unlabeled when possible, and fixed across methods to ensure fair comparison.

Baselines should include strong uniform PTQ methods, at least one activation-aware method, and one Hessian-aware or mixed-precision baseline. Metrics should include task accuracy or perplexity, memory footprint, calibration time, and sensitivity of the results to the calibration set size.

### 6.2 Expected Analysis

The key empirical questions are:

* Does behavioral sensitivity correlate better with post-quantization degradation than magnitude or activation scale?
* Does sensitivity-guided allocation improve the accuracy–bit trade-off under the same budget?
* How stable is the allocation across calibration subsets?
* How much calibration data is actually needed before the ranking becomes reliable?

### 6.3 Reporting Guidelines

All reported numbers should be accompanied by the exact model checkpoint, tokenizer, sequence length, calibration protocol, and quantization backend. Comparisons across papers should be made only when the evaluation setup is genuinely matched. If a result depends on a particular implementation detail, that detail should be stated explicitly rather than buried in the appendix.

---

## 7. Theoretical Analysis

### 7.1 Local Distortion Bound

Under the first-order approximation,

$$
\|f_\theta(x)-f_{\theta_Q}(x)\|_2 \approx \left\| \sum_{k=1}^{K} J_k(x)\delta_k \right\|_2.
$$

If each unit's quantization error is bounded by $\|\delta_k\|_2 \le \epsilon_k$, then

$$
\|f_\theta(x)-f_{\theta_Q}(x)\|_2 \le \sum_{k=1}^{K} \|J_k(x)\|_2 \epsilon_k,
$$

and, after expectation over $x$, sensitivity acts as a proxy for local functional fragility.

This does not constitute a global guarantee for arbitrary nonlinear networks; it is a local bound that justifies sensitivity-based allocation as a principled heuristic.

### 7.2 Complexity

Let $M$ be the number of probe vectors and $|\mathcal{D}_{\text{calib}}|$ the calibration set size. Sensitivity estimation scales approximately linearly in both:

$$
\mathcal{O}\left(M \cdot |\mathcal{D}_{\text{calib}}| \cdot \text{cost of a forward-mode probe}\right).
$$

This is substantially cheaper than full fine-tuning and typically lighter than exact second-order methods on large models.

---

## 8. Limitations and Future Work

BSQ is still a first-order method. Its quality depends on how well local linearization captures the behavior of the model around the pretrained weights. In highly nonlinear regions, or for units with strong activation outliers, the approximation may understate true distortion. The method also assumes that per-unit bit allocation is a sufficient abstraction; in practice, tensor-level, channel-level, or group-wise allocation may be more appropriate depending on the backend. Future work should explore second-order corrections, better handling of outliers, and tighter integration with deployment kernels.

---

## 9. Conclusion

Behavioral Sensitivity Quantization reframes mixed-precision compression as a function-aware allocation problem. By measuring how much output behavior changes under small unit-wise perturbations, BSQ assigns precision where it matters most and compresses robust units more aggressively. The resulting method is post-training, calibration-light, and compatible with existing quantization pipelines. Its main value is not that it replaces all prior quantization work, but that it provides a cleaner bridge between model behavior and compression decisions.
