# Math.md

### The Complete Mathematics Reference for Machine Learning and AI — Every Concept, Every Intuition, Every Place It Shows Up in the Pipeline

> **Why this file exists:** Most ML practitioners can name "linear algebra, calculus, probability, statistics" as the math they need. Far fewer can say *which specific operations in which specific algorithms* each branch powers — or recognize when a model failure has a mathematical root cause (singular matrix, non-convex optimization, ill-conditioned gradients). This file is the systematic reference: every mathematical concept that actually matters in modern ML, with where it shows up, why it matters, and what fails when you don't understand it.
>
> **How to read this file:** Use §2 (the dependency map) to find what you need. The math is organized in dependency order — earlier sections enable later ones. If you're rusty, read §3-6 (the foundations) in sequence. If you're targeting a specific topic (e.g., transformers), use the cross-reference in §2.

---

## Table of Contents

1. [Philosophy — What Math You Actually Need for ML](#1-philosophy--what-math-you-actually-need-for-ml)
2. [The Mathematics-to-ML Dependency Map](#2-the-mathematics-to-ml-dependency-map)
3. [Linear Algebra](#3-linear-algebra)
4. [Calculus & Multivariate Calculus](#4-calculus--multivariate-calculus)
5. [Probability Theory](#5-probability-theory)
6. [Statistics](#6-statistics)
7. [Optimization](#7-optimization)
8. [Information Theory](#8-information-theory)
9. [Numerical Methods & Stability](#9-numerical-methods--stability)
10. [Discrete Math & Combinatorics](#10-discrete-math--combinatorics)
11. [Graph Theory](#11-graph-theory)
12. [Differential Equations (for Diffusion, Neural ODEs)](#12-differential-equations-for-diffusion-neural-odes)
13. [Functional Analysis (for Kernels, RKHS)](#13-functional-analysis-for-kernels-rkhs)
14. [Measure Theory (Where It Actually Matters)](#14-measure-theory-where-it-actually-matters)
15. [Math for Specific ML Areas](#15-math-for-specific-ml-areas)
16. [Common Math Failure Modes in ML](#16-common-math-failure-modes-in-ml)
17. [Implementation Quick Reference](#17-implementation-quick-reference)

---

## 1. Philosophy — What Math You Actually Need for ML

### The Three Laws of ML Mathematics

```
LAW 1 — The Law of Operational Math
    You don't need to prove the Spectral Theorem. You need to know
    that when a covariance matrix is singular, your model breaks,
    and why. Math for ML is operational: it tells you what's
    happening, what can go wrong, and how to fix it. The math
    you can't compute or apply is decoration.

LAW 2 — The Law of Numerical Reality
    Mathematics on paper is exact. Mathematics on a computer is not.
    Every formula in a textbook becomes a numerical approximation
    when implemented — and the gap is where bugs, instabilities,
    and silent precision losses live. Understanding the numerical
    behavior of an operation is as important as understanding the
    operation itself.

LAW 3 — The Law of Inductive Bias as Math
    Every algorithm encodes assumptions about the world through
    mathematical structure. Linear models assume affine relationships.
    Trees assume axis-aligned partitions. CNNs assume translation
    equivariance. Transformers assume permutation-equivariant
    relational structure. The math IS the assumption. Picking the
    wrong math means fighting the data.
```

### The Three Tiers of Mathematical Knowledge for ML

```
TIER 1 — OPERATIONAL (must know to do ML work):
  - Vectors, matrices, dot products, matrix multiplication
  - Derivatives, gradients, chain rule
  - Probability distributions, expectation, variance
  - Mean, variance, hypothesis testing
  - Gradient descent
  - Log, exp, softmax, sigmoid

TIER 2 — STRUCTURAL (must know to understand why ML works):
  - Eigenvalues, eigenvectors, SVD
  - Hessian, Jacobian
  - Bayes' theorem, conditional probability
  - Maximum likelihood, MAP estimation
  - Convexity, KKT conditions
  - Entropy, KL divergence, cross-entropy

TIER 3 — RESEARCH (must know to read papers and design new methods):
  - Tensor algebra, manifolds
  - Functional derivatives, calculus of variations
  - Stochastic processes, SDEs
  - Measure theory, σ-algebras
  - Convex analysis, duality
  - Information geometry
```

**Most practitioners need Tier 1 fluent and Tier 2 understood.** Tier 3 is for research and specialized work. This file covers all three but emphasizes the first two.

### Why Math Failures Look Like Engineering Failures

Most "the model is broken" debugging traces to mathematical roots:

| Engineering Symptom | Mathematical Cause |
|---|---|
| Training diverges (NaN loss) | Exploding gradients, log of zero, division by near-zero |
| Loss is flat | Vanishing gradients, saddle points, dead ReLU neurons |
| Predictions all the same | Mode collapse, ill-conditioned features |
| Wildly different results across runs | Non-convex optimization, poor initialization |
| Coefficients enormous in magnitude | Multicollinearity (singular X^T X), no regularization |
| PCA failing | Features not centered, scale mismatch |
| Logistic regression won't converge | Perfect separability, infinite likelihood |
| Confidence intervals nonsensical | Distributional assumption violated |
| Cross-entropy returns inf | log(0) — predicted probability of true class was zero |

Engineers who don't recognize math symptoms debug for days. Those who do diagnose in minutes.

---

## 2. The Mathematics-to-ML Dependency Map

Cross-reference: which math powers which ML concepts.

### By ML Topic → Required Math

```
LINEAR REGRESSION
  ← Linear algebra: matrix multiplication, transpose, inverse
  ← Calculus: partial derivatives, gradient
  ← Optimization: normal equation OR gradient descent
  ← Probability: Gaussian assumption for inference
  ← Statistics: OLS estimator, confidence intervals

LOGISTIC REGRESSION
  ← Linear algebra: same as above
  ← Calculus: chain rule (sigmoid composition)
  ← Probability: Bernoulli distribution, log-likelihood
  ← Optimization: gradient descent (no closed form)
  ← Information theory: cross-entropy as loss

NEURAL NETWORKS
  ← Linear algebra: matrix-vector products at every layer
  ← Calculus: chain rule (backpropagation), Jacobian
  ← Optimization: SGD, Adam, momentum
  ← Probability: weight initialization (Xavier, He)
  ← Numerical: gradient clipping, batch normalization

CONVOLUTIONAL NETWORKS
  ← Linear algebra: matrix form of convolution
  ← Calculus: gradient of convolution operation
  ← Discrete math: kernel sliding, padding, stride arithmetic
  ← Group theory (intuition): translation equivariance

TRANSFORMERS
  ← Linear algebra: matrix multiplication is 80% of compute
  ← Calculus: softmax derivative, attention gradient
  ← Probability: softmax as distribution
  ← Optimization: AdamW, learning rate schedules
  ← Numerical: layer norm, attention numerical stability

GRADIENT BOOSTING
  ← Calculus: first and second derivatives of loss
  ← Optimization: Newton's method (XGBoost uses 2nd order)
  ← Information theory: information gain, Gini

PCA / DIMENSIONALITY REDUCTION
  ← Linear algebra: covariance matrix, eigendecomposition, SVD
  ← Statistics: variance, covariance interpretation

K-MEANS / CLUSTERING
  ← Linear algebra: Euclidean distance, vector norms
  ← Optimization: alternating minimization (EM-flavor)

SVMs
  ← Linear algebra: dot products, hyperplane equations
  ← Optimization: convex quadratic programming, Lagrange duality
  ← Functional analysis (RBF kernel): inner product spaces

BAYESIAN METHODS
  ← Probability: Bayes' theorem, conjugate priors
  ← Calculus: integration over parameters
  ← Numerical: MCMC sampling, variational inference

DIFFUSION MODELS
  ← Stochastic processes: Brownian motion, SDEs
  ← Probability: forward / reverse Markov chains
  ← Calculus: score functions, Tweedie's formula

REINFORCEMENT LEARNING
  ← Probability: Markov decision processes
  ← Optimization: policy gradient theorem
  ← Calculus: gradients of expectations
  ← Dynamic programming: Bellman equation

GRAPH NEURAL NETWORKS
  ← Linear algebra: adjacency matrix, Laplacian
  ← Graph theory: connectivity, spectral properties
```

### By Math → ML Applications

```
EIGENVALUES / EIGENVECTORS
  → PCA (principal components)
  → Spectral clustering
  → PageRank
  → Stability of optimization (Hessian eigenvalues)
  → Graph Laplacian methods

SINGULAR VALUE DECOMPOSITION
  → Recommender systems (matrix factorization)
  → Image compression
  → Pseudo-inverse for ill-conditioned regression
  → Latent semantic analysis

CHAIN RULE
  → Backpropagation (the entire thing)
  → Automatic differentiation
  → Composition of any differentiable model

BAYES' THEOREM
  → Naive Bayes classifier
  → Bayesian neural networks
  → Posterior inference everywhere
  → Generative models

KL DIVERGENCE
  → VAE training
  → Variational inference
  → RL policy regularization (PPO, TRPO)
  → Distribution drift quantification

GAUSSIAN DISTRIBUTION
  → Linear regression (errors)
  → Gaussian processes
  → VAE latent space
  → Diffusion model noise

SOFTMAX
  → Multi-class classification
  → Attention weights
  → Policy networks in RL
  → Mixture model assignments

CONVEX OPTIMIZATION
  → Linear regression, logistic regression, SVMs
  → L1/L2 regularization
  → Lasso, ElasticNet
```

---

## 3. Linear Algebra

> **The single most important mathematical area for modern ML.** Every neural network is, at its core, a sequence of matrix operations. Every embedding is a vector. Every dataset is a matrix. Linear algebra is the language ML is written in.

### 3.1 Vectors

A vector is an ordered list of numbers: **x** = [x₁, x₂, ..., xₙ] ∈ ℝⁿ

**In ML:** Every input sample is a vector. Every embedding is a vector. Every weight column is a vector.

#### Vector Operations

| Operation | Formula | Use |
|---|---|---|
| Addition | x + y | Combining representations |
| Scalar multiplication | αx | Scaling |
| Dot product | x · y = Σ xᵢyᵢ | Similarity, attention scores |
| Norm (L2) | ‖x‖₂ = √(Σ xᵢ²) | Vector magnitude |
| Norm (L1) | ‖x‖₁ = Σ \|xᵢ\| | Sparsity-inducing |
| Norm (L∞) | ‖x‖∞ = max\|xᵢ\| | Worst-case magnitude |
| Cosine similarity | (x · y) / (‖x‖ ‖y‖) | Direction similarity |

**Why this matters:** Embedding similarity, attention weights, all gradient updates, every distance metric — all built from these.

### 3.2 Matrices

A matrix is a 2D array: **A** ∈ ℝᵐˣⁿ

**In ML:**
- Datasets: rows = samples, columns = features
- Linear layer weights: W ∈ ℝᵒᵘᵗˣⁱⁿ
- Covariance matrices, similarity matrices, attention matrices

#### Matrix Operations

| Operation | Properties | Use |
|---|---|---|
| Matrix-vector product | y = Ax | Forward pass of linear layer |
| Matrix multiplication | C = AB | Composing transformations |
| Transpose | (Aᵀ)ᵢⱼ = Aⱼᵢ | Backprop, dot products |
| Trace | tr(A) = Σ Aᵢᵢ | Loss functions, regularization |
| Determinant | det(A) | Volume scaling, invertibility |
| Inverse | A⁻¹ | Solving Ax = b (rarely computed directly!) |
| Rank | dim of column space | Information content |
| Pseudo-inverse | A⁺ | OLS solution for non-square / singular A |

**Critical practical note:** Never compute matrix inverses for solving linear systems. Use `np.linalg.solve` or LU decomposition. Computing the inverse is numerically unstable and slower.

### 3.3 Special Matrices

| Matrix | Property | Where |
|---|---|---|
| **Identity** I | Ix = x | Multiplicative identity |
| **Diagonal** | Off-diagonal = 0 | Scaling, batch norm |
| **Symmetric** | A = Aᵀ | Covariance, Gram matrices |
| **Orthogonal** | AᵀA = I | Rotations, RNN init |
| **Positive definite** | xᵀAx > 0 ∀ x≠0 | Covariance, kernel matrices |
| **Positive semi-definite (PSD)** | xᵀAx ≥ 0 | Valid kernels, covariances |
| **Sparse** | Mostly zeros | Recommender data, text |
| **Toeplitz** | Constant diagonals | Convolution as matrix |

### 3.4 Eigenvalues and Eigenvectors

For a square matrix A, eigenvector **v** and eigenvalue λ satisfy:
```
Av = λv
```

**Geometric meaning:** v is a direction A only stretches (by λ), not rotates.

**Why it matters in ML:**

| Use | Connection |
|---|---|
| **PCA** | Eigenvectors of covariance matrix = principal directions; eigenvalues = variance explained |
| **Optimization stability** | Eigenvalues of the Hessian determine convergence speed |
| **Spectral clustering** | Eigenvectors of graph Laplacian reveal cluster structure |
| **PageRank** | Eigenvector of transition matrix |
| **Stability of RNNs** | Eigenvalues > 1 → exploding; < 1 → vanishing |

#### Eigendecomposition
For diagonalizable A:
```
A = VΛV⁻¹
```
where V's columns are eigenvectors and Λ is diagonal of eigenvalues.

For **symmetric** A (covariance matrices, Gram matrices):
```
A = QΛQᵀ   (Q is orthogonal)
```
This is the **spectral theorem**. Used everywhere in ML.

### 3.5 Singular Value Decomposition (SVD)

The most useful matrix factorization in ML. For **any** A ∈ ℝᵐˣⁿ:
```
A = UΣVᵀ
```
- U ∈ ℝᵐˣᵐ: left singular vectors (orthogonal)
- Σ ∈ ℝᵐˣⁿ: diagonal of singular values (≥ 0, descending)
- V ∈ ℝⁿˣⁿ: right singular vectors (orthogonal)

**Why SVD beats eigendecomposition for ML:**
- Works for any matrix (not just square)
- Numerically stable
- Singular values are always real and non-negative
- Reveals rank, range, null space

**Applications:**
- **Low-rank approximation:** Keep top k singular values → best rank-k approximation
- **Matrix factorization:** Collaborative filtering (Netflix Prize)
- **Pseudo-inverse:** A⁺ = VΣ⁺Uᵀ
- **PCA:** SVD of centered data = principal components
- **Image compression**
- **LSA (Latent Semantic Analysis):** SVD on term-document matrix

```python
U, S, Vt = np.linalg.svd(A, full_matrices=False)
# Reconstruct low-rank approximation
A_k = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
```

### 3.6 Matrix Calculus

The single most-used set of identities in deep learning:

| Expression | Gradient |
|---|---|
| f(x) = aᵀx | ∇f = a |
| f(x) = xᵀAx | ∇f = (A + Aᵀ)x; if A symmetric: 2Ax |
| f(x) = ‖x‖² | ∇f = 2x |
| f(x) = ‖Ax - b‖² | ∇f = 2Aᵀ(Ax - b) |
| f(X) = tr(AX) | ∇f = Aᵀ |
| f(X) = log det(X) | ∇f = (X⁻¹)ᵀ |

**Convention:** Most ML literature uses **denominator layout** (gradient has same shape as parameter). Some math literature uses numerator layout. Be consistent.

### 3.7 Vector Spaces and Subspaces

Concepts that show up implicitly in ML:

- **Span:** Set of all linear combinations of vectors
- **Linear independence:** No vector is a linear combination of others
- **Basis:** Linearly independent vectors that span the space
- **Rank of a matrix:** Dimension of column space = number of independent columns
- **Null space (kernel):** Vectors x with Ax = 0

**Why these matter:**
- **Rank deficiency** = multicollinearity in regression
- **Low-dimensional manifolds** = the heart of representation learning
- **Embedding spaces** are quotient spaces (you don't care about scale or rotation)

### 3.8 Norms and Distances

| Norm | Formula | ML Use |
|---|---|---|
| L1 | Σ\|xᵢ\| | Lasso regularization, sparsity |
| L2 | √(Σxᵢ²) | Ridge regularization, default distance |
| L∞ | max\|xᵢ\| | Robust distance, adversarial bounds |
| Frobenius | √(Σᵢⱼ Aᵢⱼ²) | Matrix L2 |
| Nuclear | Σᵢ σᵢ | Low-rank regularization |

**Equivalence:** For finite-dimensional spaces, all norms are equivalent up to constants. They induce the same topology — but **different geometries** that lead to different solutions in optimization.

### 3.9 Tensors

Generalization: scalar (0-D), vector (1-D), matrix (2-D), tensor (n-D).

**In ML:**
- Batch of images: (batch, height, width, channels) — 4-D tensor
- Sequence batch: (batch, seq_len, hidden_dim) — 3-D tensor
- Convolutional weight: (out_channels, in_channels, kernel_h, kernel_w) — 4-D

**Tensor operations:**
- Element-wise: `A * B` (Hadamard product)
- Reduction: sum, mean over axes
- Reshape, transpose, broadcast
- Einsum: general contraction (`einsum("bij,bjk->bik")` = batch matrix multiply)

**Einsum is the unified language for tensor operations.** Master it.

---

## 4. Calculus & Multivariate Calculus

### 4.1 Derivatives — The Fundamental Idea

The derivative of f at x is:
```
f'(x) = lim[h→0] (f(x+h) - f(x)) / h
```

**Intuition:** Instantaneous rate of change. Slope of tangent line.

**In ML:** Every gradient descent step is "move parameters in the direction the derivative points downhill."

### 4.2 Essential Derivative Rules

| Rule | Formula |
|---|---|
| Power | d/dx[xⁿ] = nxⁿ⁻¹ |
| Exponential | d/dx[eˣ] = eˣ |
| Logarithm | d/dx[ln x] = 1/x |
| Sum | (f + g)' = f' + g' |
| Product | (fg)' = f'g + fg' |
| Quotient | (f/g)' = (f'g - fg') / g² |
| Chain | (f(g(x)))' = f'(g(x)) · g'(x) |

**The chain rule is the most important.** Backpropagation is the chain rule applied recursively.

### 4.3 Key Derivatives in ML

| Function | Derivative |
|---|---|
| Sigmoid σ(x) = 1/(1+e⁻ˣ) | σ(x)(1 - σ(x)) |
| Tanh | 1 - tanh²(x) |
| ReLU | 1 if x>0, 0 if x<0, undefined at 0 |
| Softplus log(1+eˣ) | σ(x) |
| Softmax sᵢ = eˣⁱ/Σeˣʲ | ∂sᵢ/∂xⱼ = sᵢ(δᵢⱼ - sⱼ) |
| Cross-entropy CE(p,q) = -Σpᵢlog(qᵢ) | -p/q (element-wise) |

**Memorize sigmoid and softmax derivatives.** They appear in every classification gradient.

### 4.4 Partial Derivatives and Gradients

For f: ℝⁿ → ℝ, the gradient is:
```
∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]ᵀ
```

**Properties:**
- ∇f points in the direction of steepest ascent
- ‖∇f‖ is the maximum rate of change
- ∇f = 0 at critical points (minima, maxima, saddle points)

**Gradient descent:**
```
x_{t+1} = x_t - η ∇f(x_t)
```

### 4.5 Jacobian

For f: ℝⁿ → ℝᵐ, the Jacobian is the matrix of all partial derivatives:
```
J ∈ ℝᵐˣⁿ,  Jᵢⱼ = ∂fᵢ/∂xⱼ
```

**In ML:**
- Linear layer: Jacobian = weight matrix W
- Activation function (element-wise): Jacobian = diagonal of derivatives
- Backpropagation: Vector-Jacobian product (VJP) at each layer

**Reverse-mode auto-diff** computes vector-Jacobian products. **Forward-mode** computes Jacobian-vector products. Backprop is reverse-mode.

### 4.6 Hessian

The matrix of second derivatives:
```
H ∈ ℝⁿˣⁿ,  Hᵢⱼ = ∂²f/∂xᵢ∂xⱼ
```

**Properties:**
- Symmetric (when f is twice continuously differentiable)
- Positive definite ⟺ local minimum (sufficient condition)
- Negative definite ⟺ local maximum
- Indefinite ⟺ saddle point

**In ML:**
- **Newton's method:** x_{t+1} = x_t - H⁻¹∇f (used in XGBoost!)
- **Optimization stability:** Condition number of Hessian = ratio of largest to smallest eigenvalue. High condition number = slow convergence.
- **Curvature information** in advanced optimizers (K-FAC, Shampoo)

For neural networks, H is too large to compute. **Hessian-vector products** can be computed efficiently and are used in some optimizers.

### 4.7 Chain Rule for Vector Functions

For composition y = g(f(x)) where f: ℝⁿ → ℝᵐ, g: ℝᵐ → ℝᵏ:
```
∂y/∂x = (∂y/∂f) · (∂f/∂x) = J_g · J_f
```

**Backpropagation IS this rule.** Applied recursively backward through the network, with shapes worked out so we never form the full Jacobians explicitly.

### 4.8 Directional Derivatives

The rate of change of f at x in direction u:
```
D_u f(x) = ∇f(x) · u
```

**Used in:** Line search methods, understanding gradient direction.

### 4.9 Taylor Series

Approximate f near x₀:
```
f(x) ≈ f(x₀) + ∇f(x₀)ᵀ(x - x₀) + ½(x - x₀)ᵀH(x₀)(x - x₀) + ...
```

**Uses in ML:**
- **First-order methods (gradient descent):** Use only the linear term
- **Second-order methods (Newton):** Use up to quadratic
- **Trust region methods:** Bound the validity of the approximation
- **Convergence proofs:** Mostly based on Taylor expansion

### 4.10 Integration (Brief — Less Common in ML)

Integration shows up in:
- **Expected values:** E[f(X)] = ∫ f(x) p(x) dx
- **Marginalization:** p(x) = ∫ p(x,y) dy
- **Continuous distributions:** Probabilities are integrals of densities
- **Bayesian inference:** Posterior involves integrals (often intractable)

Most ML uses **Monte Carlo** or **variational** approximations to integrals rather than computing them analytically.

---

## 5. Probability Theory

> Probability is how ML quantifies uncertainty, defines loss functions, and models data-generating processes.

### 5.1 Sample Spaces, Events, and Probability

- **Sample space (Ω):** Set of all possible outcomes
- **Event:** Subset of Ω
- **Probability:** Function P: events → [0, 1] satisfying Kolmogorov's axioms:
  1. P(A) ≥ 0
  2. P(Ω) = 1
  3. P(A ∪ B) = P(A) + P(B) for disjoint A, B

### 5.2 Random Variables

A random variable X is a function from Ω to ℝ (or ℝⁿ).

**Discrete RV:** Takes countably many values (dice, classification labels)
**Continuous RV:** Takes uncountably many values (real-valued features)

#### Probability Mass Function (PMF) — discrete
```
p(x) = P(X = x)
```

#### Probability Density Function (PDF) — continuous
```
P(a < X < b) = ∫_a^b f(x) dx
```

#### Cumulative Distribution Function (CDF)
```
F(x) = P(X ≤ x)
```

### 5.3 Expectation and Variance

#### Expectation
```
E[X] = Σ x · p(x)        (discrete)
E[X] = ∫ x · f(x) dx      (continuous)
```

**Linearity:** E[aX + bY] = aE[X] + bE[Y]. Always holds, even when X, Y are dependent.

#### Variance
```
Var(X) = E[(X - E[X])²] = E[X²] - E[X]²
```

#### Standard Deviation
```
σ(X) = √Var(X)
```

#### Covariance
```
Cov(X, Y) = E[(X - E[X])(Y - E[Y])] = E[XY] - E[X]E[Y]
```

#### Correlation
```
ρ(X, Y) = Cov(X, Y) / (σ(X) σ(Y))   ∈ [-1, 1]
```

### 5.4 Conditional Probability and Independence

#### Conditional Probability
```
P(A | B) = P(A ∩ B) / P(B)
```

#### Independence
X and Y are independent iff:
```
P(X = x, Y = y) = P(X = x) · P(Y = y)   for all x, y
```

**In ML:** Naive Bayes assumes conditional independence given the class. The assumption is almost always wrong but the model often works anyway.

### 5.5 Bayes' Theorem

The single most important equation in probability for ML:
```
P(A | B) = P(B | A) · P(A) / P(B)
```

Or in ML notation:
```
posterior = (likelihood × prior) / evidence
P(θ | data) = P(data | θ) · P(θ) / P(data)
```

**Used in:**
- Naive Bayes classifiers
- Bayesian neural networks
- Probabilistic programming
- All of Bayesian inference

### 5.6 Common Probability Distributions

#### Discrete

| Distribution | PMF | When |
|---|---|---|
| **Bernoulli(p)** | p^k (1-p)^(1-k) | Binary outcome |
| **Binomial(n, p)** | C(n,k) p^k (1-p)^(n-k) | Count of successes in n trials |
| **Categorical** | pₖ | Multi-class label |
| **Multinomial(n, p)** | ... | Counts in K categories |
| **Poisson(λ)** | e^-λ λ^k / k! | Rare events, counts |
| **Geometric(p)** | (1-p)^(k-1) p | First success at trial k |

#### Continuous

| Distribution | PDF | When |
|---|---|---|
| **Uniform(a, b)** | 1/(b-a) | No information beyond bounds |
| **Normal(μ, σ²)** | (1/√(2πσ²)) exp(-(x-μ)²/2σ²) | Default for continuous, CLT |
| **Multivariate Normal(μ, Σ)** | ... | Vector continuous, GP, VAE |
| **Exponential(λ)** | λe^(-λx) | Waiting times, durations |
| **Gamma(α, β)** | ... | Positive continuous, durations |
| **Beta(α, β)** | ... | Probabilities, [0,1] valued |
| **Dirichlet(α)** | ... | Distribution over probabilities (LDA) |
| **Student-t** | ... | Heavy-tailed alternative to Normal |
| **Laplace(μ, b)** | (1/2b) exp(-\|x-μ\|/b) | L1 / sparsity prior |

### 5.7 The Gaussian (Normal) Distribution

The most important continuous distribution in ML.

**Univariate:**
```
N(μ, σ²): f(x) = (1/√(2πσ²)) exp(-(x-μ)²/(2σ²))
```

**Multivariate:**
```
N(μ, Σ): f(x) = (1/√((2π)^k det(Σ))) exp(-½(x-μ)ᵀ Σ⁻¹ (x-μ))
```

**Key properties:**
- Closed under linear transformations: AX + b ~ N(Aμ + b, AΣAᵀ)
- Sum of independent Gaussians is Gaussian
- Conditional and marginal distributions of MVN are Gaussian
- Maximum entropy distribution given mean and variance
- Central Limit Theorem makes it the limiting distribution of averages

**Where it shows up:**
- Errors in linear regression
- Weight initialization in neural networks
- Variational autoencoders (latent prior)
- Gaussian processes
- Diffusion model noise

### 5.8 Central Limit Theorem (CLT)

If X₁, X₂, ..., Xₙ are iid with mean μ and variance σ², then:
```
(X̄ - μ) / (σ/√n)  →  N(0, 1)  as n → ∞
```

**The mean of any (well-behaved) distribution becomes approximately Gaussian for large samples.**

**In ML:** Justifies Gaussian assumptions in many estimators, A/B testing, confidence intervals.

### 5.9 Law of Large Numbers (LLN)

Sample averages converge to expected values:
```
X̄ₙ → E[X]   as n → ∞
```

**In ML:** Why Monte Carlo estimation works. Why more training data generally helps. Why mini-batch SGD's gradient is an unbiased estimate of the full gradient.

### 5.10 Joint, Marginal, and Conditional Distributions

For random variables X, Y:

| Concept | Formula |
|---|---|
| Joint | p(x, y) |
| Marginal | p(x) = ∫ p(x, y) dy |
| Conditional | p(x \| y) = p(x, y) / p(y) |

**Chain rule of probability:**
```
p(x₁, x₂, ..., xₙ) = p(x₁) p(x₂|x₁) p(x₃|x₁,x₂) ... p(xₙ|x₁,...,xₙ₋₁)
```

**In ML:** This is how autoregressive models (GPT, image autoregressive) factor distributions.

### 5.11 Change of Variables

If Y = g(X) for invertible g:
```
p_Y(y) = p_X(g⁻¹(y)) · |det(J_{g⁻¹}(y))|
```

**In ML:** Normalizing flows are built entirely on this formula. Every transformation must be invertible with tractable Jacobian.

### 5.12 Expectation of Functions

```
E[f(X)] = ∫ f(x) p(x) dx
```

**In ML:** Loss functions are expectations: E[L(y, f(x; θ))]. We can't compute this exactly so we estimate with samples (mini-batch).

### 5.13 Monte Carlo Estimation

To estimate E[f(X)]:
1. Draw N samples x₁, ..., xₙ from p
2. Compute (1/N) Σ f(xᵢ)

Variance of estimator: σ²/N. **Halving error needs 4× samples.**

**In ML:** Mini-batch gradients, Monte Carlo dropout, REINFORCE policy gradients.

---

## 6. Statistics

### 6.1 Descriptive vs. Inferential

- **Descriptive:** Summarize observed data (mean, variance, histograms)
- **Inferential:** Draw conclusions about a population from a sample

**ML straddles both.** Training is descriptive (fit observed data). Generalization is inferential (predict population behavior).

### 6.2 Estimators

An **estimator** is a function of data that estimates some parameter.

**Properties:**
- **Bias:** E[θ̂] - θ
- **Variance:** Var(θ̂)
- **MSE:** Bias² + Variance (decomposition)
- **Consistency:** θ̂ → θ as n → ∞
- **Efficiency:** Achieves the Cramér-Rao lower bound

**Bias-variance tradeoff in ML:** Higher model complexity = lower bias, higher variance. The art is balancing them.

### 6.3 Maximum Likelihood Estimation (MLE)

Given data D and a parametric model p(D | θ), MLE finds:
```
θ̂_MLE = argmax_θ p(D | θ) = argmax_θ Σ log p(xᵢ | θ)
```

**In ML:**
- Linear regression with Gaussian errors → MLE = OLS
- Logistic regression → MLE = minimize cross-entropy
- Most neural networks → MLE = minimize negative log-likelihood

### 6.4 Maximum A Posteriori (MAP) Estimation

```
θ̂_MAP = argmax_θ p(θ | D) = argmax_θ p(D | θ) p(θ)
```

**MAP = MLE + prior.** Regularization can be derived as MAP with specific priors:
- L2 regularization ⟺ Gaussian prior on weights
- L1 regularization ⟺ Laplace prior on weights

### 6.5 Bayesian Estimation

Don't pick a single θ. Maintain the full posterior:
```
p(θ | D) = p(D | θ) p(θ) / p(D)
```

For prediction:
```
p(y_new | x_new, D) = ∫ p(y_new | x_new, θ) p(θ | D) dθ
```

**Pros:** Uncertainty quantification, principled regularization, robust with small data.
**Cons:** Intractable integrals (need MCMC or variational methods).

### 6.6 Hypothesis Testing

Setup:
- **Null hypothesis (H₀):** Default assumption (e.g., "no effect")
- **Alternative (H₁):** What we want to detect
- **Test statistic:** Function of data
- **p-value:** P(observing data this extreme | H₀ true)
- **Significance level (α):** Threshold for rejection (typically 0.05)

**Type I error:** Reject H₀ when true (false positive). Rate = α.
**Type II error:** Fail to reject H₀ when false (false negative). Rate = β.
**Power:** 1 - β = ability to detect a real effect.

**In ML:** A/B testing, comparing models, fairness audits.

### 6.7 Confidence Intervals

A 95% CI for θ is an interval that contains the true θ with 95% probability **across repeated experiments**.

**Common CI:**
```
θ̂ ± 1.96 · SE(θ̂)
```

**Critical misunderstanding:** A specific CI either contains θ or doesn't. The 95% refers to the procedure, not any single interval.

### 6.8 Common Statistical Tests

| Test | Use |
|---|---|
| **t-test** | Compare means (one-sample, two-sample, paired) |
| **ANOVA** | Compare more than two group means |
| **Chi-squared** | Independence of categorical variables |
| **Kolmogorov-Smirnov** | Distribution comparison (drift detection) |
| **Mann-Whitney U** | Non-parametric two-group comparison |
| **Wilcoxon signed-rank** | Non-parametric paired comparison |
| **Permutation test** | General, distribution-free |
| **Bootstrap** | Estimate sampling distribution |

### 6.9 Bootstrap

Resample with replacement to estimate sampling distributions:

```python
estimates = []
for _ in range(B):
    sample = np.random.choice(data, size=len(data), replace=True)
    estimates.append(statistic(sample))
ci = np.percentile(estimates, [2.5, 97.5])
```

**In ML:** Confidence intervals for metrics, comparing models, prediction intervals.

### 6.10 Regression Diagnostics

For classical linear models:

| Diagnostic | Checks |
|---|---|
| Residual plot | Linearity, homoscedasticity |
| Q-Q plot | Normality of residuals |
| VIF | Multicollinearity |
| Cook's distance | Influential points |
| Leverage | Points with extreme features |
| R², adjusted R² | Variance explained |
| AIC, BIC | Model comparison |

**Modern ML often ignores these** — but for high-stakes inferential work, they remain essential.

### 6.11 Multiple Testing

Run K tests at level α — false positive rate inflates to ~Kα.

**Corrections:**
- **Bonferroni:** Use α/K (conservative)
- **Benjamini-Hochberg (FDR):** Less conservative, controls false discovery rate

**In ML:** Crucial for feature selection, A/B testing platforms, hyperparameter sweeps.

---

## 7. Optimization

> Training an ML model IS solving an optimization problem.

### 7.1 The General Setup

Minimize a loss:
```
θ* = argmin_θ L(θ)
```

**In ML, L is usually:**
```
L(θ) = (1/N) Σᵢ ℓ(yᵢ, f(xᵢ; θ)) + λ R(θ)
        └─────empirical risk─────┘   └─regularization─┘
```

### 7.2 Convexity

f is **convex** if for all x, y, λ ∈ [0,1]:
```
f(λx + (1-λ)y) ≤ λf(x) + (1-λ)f(y)
```

**Equivalent for twice-differentiable f:** Hessian is PSD everywhere.

**Why convexity matters:**
- Every local minimum is a global minimum
- Gradient descent converges (no saddle points to worry about)
- Many algorithms have provable bounds

**Convex losses in ML:** Linear regression (MSE), logistic regression, SVMs.
**Non-convex losses in ML:** Neural networks (almost always).

### 7.3 First-Order Methods

#### Gradient Descent
```
θ_{t+1} = θ_t - η ∇L(θ_t)
```

- **Batch GD:** Use full dataset. Stable, slow per epoch.
- **Stochastic GD (SGD):** Single sample. Noisy, fast iterations.
- **Mini-batch GD:** Batch of B samples. Standard for deep learning.

#### Momentum
```
v_{t+1} = βv_t + ∇L(θ_t)
θ_{t+1} = θ_t - η v_{t+1}
```
Accelerates in consistent directions, dampens oscillations.

#### Nesterov Accelerated Gradient
"Look ahead" momentum. Better theoretical convergence.

#### Adam
Adaptive learning rates per parameter:
```
m_t = β₁ m_{t-1} + (1-β₁) g_t              # 1st moment
v_t = β₂ v_{t-1} + (1-β₂) g_t²             # 2nd moment
m̂_t = m_t / (1 - β₁^t)                    # Bias correction
v̂_t = v_t / (1 - β₂^t)
θ_{t+1} = θ_t - η m̂_t / (√v̂_t + ε)
```

**Default for deep learning.** AdamW adds decoupled weight decay.

### 7.4 Second-Order Methods

#### Newton's Method
```
θ_{t+1} = θ_t - H⁻¹ ∇L(θ_t)
```

Uses curvature for faster convergence. **Quadratic convergence near optimum.**

**Cost:** Computing and inverting H is O(n³). Prohibitive for neural networks.

**XGBoost uses Newton's method** because trees have few parameters per node.

#### Quasi-Newton (BFGS, L-BFGS)
Approximate the Hessian inverse using gradient differences. Used for medium-sized problems.

### 7.5 Convergence Rates

For convex f:
- **Gradient descent:** O(1/t) (sublinear)
- **GD with momentum / Nesterov:** O(1/t²)
- **Newton's:** O(1/t²) or faster (quadratic near optimum)
- **Strongly convex GD:** O(λ^t) where λ < 1 (linear / exponential)

### 7.6 Learning Rate

**Too small:** Slow convergence
**Too large:** Diverge or oscillate

**Scheduling strategies:**
- **Constant:** Simple but often suboptimal
- **Step decay:** η × γ every K epochs
- **Exponential decay:** η_t = η₀ · e^(-kt)
- **Cosine annealing:** η_t = η_min + ½(η_max - η_min)(1 + cos(πt/T))
- **One-cycle:** Warmup then anneal — strong for deep learning
- **Reduce on plateau:** Drop when validation loss stalls

### 7.7 Constrained Optimization and Lagrange Multipliers

For:
```
minimize f(x) subject to g(x) = 0
```

Form the **Lagrangian**:
```
L(x, λ) = f(x) - λ g(x)
```

At optimum: ∇_x L = 0 and ∇_λ L = 0.

#### KKT Conditions (inequality constraints)
For h(x) ≤ 0:
1. ∇f - λ∇h = 0 (stationarity)
2. h(x) ≤ 0 (primal feasibility)
3. λ ≥ 0 (dual feasibility)
4. λ h(x) = 0 (complementary slackness)

**In ML:**
- SVMs are solved via Lagrange duality
- Constrained optimization for fairness ML

### 7.8 Stochastic Optimization

Most ML optimization is **stochastic** — the gradient is a noisy estimate (mini-batch).

**Properties:**
- Gradient is unbiased but high variance
- Noise can help escape saddle points
- Convergence proofs require decreasing learning rates or averaging
- "Implicit regularization" of SGD: tends to find flat minima

### 7.9 Non-Convex Optimization

Neural network losses are non-convex. Critical implications:

- Many local minima — but most are roughly equivalent in deep nets
- **Saddle points** dominate in high dimensions (more than minima)
- Initialization matters
- Learning rate matters for which basin you find

**Empirical observation:** Modern deep learning works despite non-convexity, partly because:
- Overparameterization makes loss landscape smoother
- SGD has implicit regularization toward flat minima
- Skip connections smooth the landscape (ResNet)

### 7.10 Regularization as Constrained Optimization

L2 regularization:
```
min ‖y - Xβ‖² + λ‖β‖²    ⟺    min ‖y - Xβ‖² subject to ‖β‖² ≤ C
```

L1 regularization (Lasso):
```
min ‖y - Xβ‖² + λ‖β‖₁    ⟺    min ‖y - Xβ‖² subject to ‖β‖₁ ≤ C
```

**Geometric intuition:** L1 ball has corners on axes → solutions tend to lie on axes (sparse). L2 ball is smooth → solutions tend to be small but nonzero.

---

## 8. Information Theory

> Information theory provides the loss functions of classification, the regularization of generative models, and the metrics for distribution comparison.

### 8.1 Entropy

The Shannon entropy of a discrete distribution:
```
H(X) = -Σ p(x) log p(x)
```

**Interpretation:** Average number of bits (log base 2) or nats (log base e) needed to encode a sample from p. Measure of uncertainty.

**Properties:**
- H(X) ≥ 0
- H(X) = 0 ⟺ X is deterministic
- Maximum entropy: uniform distribution → log(n)

**In ML:** Decision tree splits maximize information gain (entropy reduction).

### 8.2 Cross-Entropy

```
H(p, q) = -Σ p(x) log q(x)
```

**Interpretation:** Expected number of bits to encode samples from p using a code optimized for q.

**In ML:** **The standard loss for classification.** For one-hot targets:
```
CE = -log(predicted_probability_of_true_class)
```

### 8.3 KL Divergence

```
KL(p || q) = Σ p(x) log(p(x) / q(x))
           = H(p, q) - H(p)
```

**Properties:**
- KL(p || q) ≥ 0, with equality iff p = q
- **NOT symmetric:** KL(p || q) ≠ KL(q || p)
- **NOT a metric** (no triangle inequality)

**In ML:**
- VAE loss includes KL between encoder distribution and prior
- Variational inference minimizes KL to approximate intractable posteriors
- Drift detection: KL between training and current distribution
- PPO uses KL constraints between policy versions

### 8.4 Jensen-Shannon Divergence

Symmetric, bounded variant of KL:
```
JS(p, q) = ½ KL(p || m) + ½ KL(q || m),  where m = ½(p + q)
```

**Bounded by log(2).** Used in GAN training (original formulation).

### 8.5 Mutual Information

```
I(X; Y) = KL(p(x,y) || p(x) p(y))
        = H(X) - H(X | Y)
        = H(Y) - H(Y | X)
```

**Interpretation:** How much knowing one variable reduces uncertainty about the other.

**In ML:**
- Feature selection (mutual information vs target)
- Information bottleneck principle
- Representation learning objectives

### 8.6 Cross-Entropy as Maximum Likelihood

For classification with one-hot targets and softmax outputs:
```
MLE for categorical → minimize cross-entropy
```

**The loss function isn't arbitrary.** It's the negative log-likelihood under a categorical distribution. This is why cross-entropy is the "right" loss for classification.

### 8.7 Information Gain (Decision Trees)

When splitting at feature f with threshold t:
```
IG = H(parent) - [weighted average of H(children)]
```

Decision trees greedily select splits maximizing IG.

### 8.8 The Information Bottleneck

Optimal representations T of input X for predicting Y maximize:
```
I(T; Y) - β I(T; X)
```

**Intuition:** Capture all information about Y, no more about X than necessary.

**Influences modern representation learning and theoretical analysis of deep nets.**

---

## 9. Numerical Methods & Stability

> The math you write is exact. The math your computer runs is not. This section is the difference between "math works" and "model trains."

### 9.1 Floating Point Reality

Computers use finite-precision floating point:
- **float64:** ~15-16 significant decimal digits
- **float32:** ~7 significant digits (default for ML)
- **float16:** ~3-4 significant digits
- **bfloat16:** Same range as float32, less precision

**Implications:**
- 0.1 + 0.2 ≠ 0.3 exactly
- Large + small + (-large) can lose the small value
- Operations not associative: (a + b) + c ≠ a + (b + c)

### 9.2 Catastrophic Cancellation

Subtracting nearly-equal numbers amplifies error:
```
(1.0000001 - 1.0000000) = 0.0000001  — but lose 7 digits of precision
```

**Fix:** Reformulate to avoid the subtraction.

### 9.3 Numerically Stable Softmax

Naive:
```python
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))
```

If x has large values, `exp` overflows.

**Stable:**
```python
def softmax(x):
    x = x - np.max(x)  # Doesn't change result, prevents overflow
    return np.exp(x) / np.sum(np.exp(x))
```

**Built into every deep learning framework.** But you'll see this when implementing from scratch.

### 9.4 Log-Sum-Exp Trick

To compute log(Σ exp(xᵢ)) without overflow:
```python
def logsumexp(x):
    m = np.max(x)
    return m + np.log(np.sum(np.exp(x - m)))
```

**Used in:** Cross-entropy, attention computation, mixture model likelihoods.

### 9.5 Avoiding log(0)

```python
# Bad: can give -inf
loss = -y * np.log(p) - (1 - y) * np.log(1 - p)

# Good: clip
eps = 1e-7
loss = -y * np.log(np.clip(p, eps, 1-eps)) - (1 - y) * np.log(np.clip(1-p, eps, 1-eps))
```

### 9.6 Condition Number

For matrix A:
```
κ(A) = σ_max(A) / σ_min(A)
```

**Interpretation:** How much output errors are amplified relative to input errors. Linear systems with high condition number are hard to solve accurately.

**In ML:**
- Ill-conditioned features → unstable regression
- Hessian condition number → slow convergence
- Batch normalization improves conditioning of forward pass

### 9.7 Gradient Clipping

When gradients explode:
```python
# By norm
total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# By value
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
```

**Essential for:** Training RNNs, transformers, any deep model with potential gradient instability.

### 9.8 Mixed Precision Training

Use float16 / bfloat16 for forward/backward, float32 for weight updates:
- 2× memory savings
- 2-4× speedup on modern GPUs
- Requires "loss scaling" for fp16 to prevent gradient underflow

### 9.9 Numerical Differentiation (and Why You Don't Use It)

Finite differences:
```
f'(x) ≈ (f(x + h) - f(x)) / h
```

**Why ML doesn't use this:**
- Numerical error grows for small h (cancellation)
- Truncation error grows for large h
- Cost: O(n) for n parameters

**ML uses automatic differentiation** — exact derivatives, computed efficiently via the chain rule applied to the computation graph.

### 9.10 Numerical Linear Algebra Tips

| Don't | Do |
|---|---|
| Compute A⁻¹ then multiply | `np.linalg.solve(A, b)` |
| Compute (AᵀA)⁻¹Aᵀb (normal equation) | Use QR or SVD-based solver |
| Stack vectors then transpose | Use views / reshape |
| Convert sparse to dense | Use scipy.sparse |
| Loop element-wise in Python | Vectorize with NumPy / use BLAS |

---

## 10. Discrete Math & Combinatorics

### 10.1 Counting

| Concept | Formula |
|---|---|
| **Permutations (order matters)** | n! / (n-k)! |
| **Combinations (order doesn't)** | n! / (k!(n-k)!) = C(n, k) |
| **Multinomial coefficient** | n! / (n₁! n₂! ... nₖ!) |

**In ML:** Computing class distributions, multi-label combinations, combinatorial features.

### 10.2 Boolean Logic

ML uses:
- Indicator functions (1 if condition true, 0 otherwise)
- Logical features (categorical encoding, masks)
- Decision tree rules

### 10.3 Modular Arithmetic and Hashing

| Concept | Use |
|---|---|
| **Hash functions** | Feature hashing, locality-sensitive hashing |
| **Bloom filters** | Efficient set membership |
| **MinHash** | Document similarity |
| **HyperLogLog** | Cardinality estimation |

### 10.4 Algorithmic Complexity

| Notation | Meaning |
|---|---|
| O(1) | Constant |
| O(log n) | Logarithmic |
| O(n) | Linear |
| O(n log n) | "Log linear" — sorting, FFT |
| O(n²) | Quadratic — pairwise comparisons |
| O(n³) | Cubic — matrix multiplication (naive) |

**Common ML complexities:**
- Linear regression (normal equation): O(n²p + p³)
- Logistic regression (per iter): O(np)
- k-NN query: O(n) naive, O(log n) with KD-tree (low dim)
- Decision tree training: O(np log n)
- Neural network forward pass: O(parameters × batch)
- Attention: O(n²d) in sequence length n

---

## 11. Graph Theory

### 11.1 Basics

- **Graph G = (V, E):** Set of vertices and edges
- **Directed / undirected**
- **Weighted / unweighted**
- **Adjacency matrix A:** Aᵢⱼ = 1 if edge from i to j

### 11.2 Graph Laplacian

```
L = D - A
```
where D is the degree matrix (diagonal of degrees).

**Properties:**
- Symmetric, PSD
- Smallest eigenvalue is 0
- Number of zero eigenvalues = number of connected components
- Eigenvectors of L encode cluster structure

**In ML:**
- **Spectral clustering:** Cluster using eigenvectors of L
- **Graph neural networks:** Propagate features via L (GCN: AĤ = D⁻¹ᐟ²(A+I)D⁻¹ᐟ²)
- **Manifold learning:** Approximate manifold structure

### 11.3 Random Walks and PageRank

For a graph, the **transition matrix** P normalizes A by row sums.

**Stationary distribution π satisfies:** πᵀP = πᵀ. This is PageRank.

**Power iteration:** Repeatedly compute π_{t+1} = πₜᵀP. Eigenvalue interpretation: π is the dominant left eigenvector.

### 11.4 Graph Neural Networks (Brief)

Per-layer update:
```
h_v^{(l+1)} = σ(W · AGG({h_u^{(l)} : u ∈ N(v)}))
```

Where AGG is mean, sum, or attention-weighted.

**Key insight:** Stacking L layers means each node sees L-hop neighborhood.

---

## 12. Differential Equations (for Diffusion, Neural ODEs)

### 12.1 Ordinary Differential Equations (ODEs)

```
dx/dt = f(x, t)
```

**In ML:**
- **Neural ODEs:** Treat neural network as continuous depth, integrate
- **Flow-based models:** Continuous normalizing flows

### 12.2 Stochastic Differential Equations (SDEs)

```
dx = f(x, t) dt + g(x, t) dW
```

Where dW is Brownian motion (random walk).

**In ML:**
- **Diffusion models:** Forward process adds noise via SDE; reverse process denoises
- **Langevin dynamics:** Sampling via SDE

### 12.3 The Score Function

```
s(x) = ∇_x log p(x)
```

**The gradient of the log density.** Doesn't require knowing the normalization constant.

**In ML:** **Score-based generative models / diffusion models** estimate the score and use it for generation. Replaces explicit density estimation.

### 12.4 Tweedie's Formula

Connects noisy observations to clean estimates:
```
E[x | y] = y + σ² ∇_y log p(y)
```

**The fundamental equation of denoising diffusion.**

---

## 13. Functional Analysis (for Kernels, RKHS)

### 13.1 Kernel Functions

A kernel k(x, y) is a function such that there exists a feature map φ with:
```
k(x, y) = ⟨φ(x), φ(y)⟩
```

**Mercer's theorem:** A symmetric function is a valid kernel iff its Gram matrix is PSD for any input set.

**Common kernels:**
- Linear: k(x, y) = xᵀy
- Polynomial: k(x, y) = (xᵀy + c)^d
- RBF (Gaussian): k(x, y) = exp(-‖x-y‖²/2σ²)
- Sigmoid: k(x, y) = tanh(αxᵀy + c)

### 13.2 The Kernel Trick

Many algorithms can be written purely in terms of dot products:
- SVM
- PCA
- Ridge regression
- Gaussian processes

Replace ⟨x, y⟩ with k(x, y) and you've effectively worked in a (possibly infinite-dimensional) feature space without ever computing φ explicitly.

### 13.3 Reproducing Kernel Hilbert Space (RKHS)

The Hilbert space of functions where the kernel "reproduces" function evaluation:
```
f(x) = ⟨f, k(·, x)⟩
```

**Why it matters:** Provides theoretical justification for kernel methods, generalization bounds for SVMs, and connection to Gaussian processes.

---

## 14. Measure Theory (Where It Actually Matters)

> Most ML practitioners never need measure theory. But understanding *why* it exists clarifies several things modern ML relies on.

### 14.1 Why Probability Needs Measure Theory

For uncountable sample spaces (like the reals), not all subsets can be assigned a probability consistently. Measure theory formalizes which sets are "measurable" and how to integrate over them.

**Practical impact:** Approximately none for most ML work. Probability distributions, expectations, and densities behave as you'd expect.

### 14.2 Where It Surfaces

| Topic | Connection |
|---|---|
| **Radon-Nikodym derivative** | Foundation of density ratios, importance sampling |
| **Pushforward measures** | Change of variables in flows |
| **Convergence in distribution** | Theoretical limits, CLT, etc. |
| **Almost sure convergence** | Stronger guarantees in stochastic optimization |
| **Stochastic integrals** | Rigorous foundation of SDEs |

### 14.3 The Practical Takeaway

You don't need to know measure theory to do ML. You need to know:
- "Probability distribution" is well-defined even when densities exist
- Integration over continuous spaces is well-defined
- "Almost surely" means "with probability 1"
- Some theorems require measure-theoretic statements to be precise

For 99% of practitioners, this is sufficient.

---

## 15. Math for Specific ML Areas

### 15.1 Neural Network Training (consolidated)

Every iteration of training uses:
- **Linear algebra:** Matrix-vector products in forward pass
- **Calculus:** Chain rule for backward pass
- **Probability:** Loss = negative log-likelihood
- **Optimization:** SGD / Adam variant
- **Numerical methods:** Mixed precision, gradient clipping

### 15.2 Transformers

| Operation | Math |
|---|---|
| **Attention** | softmax(QKᵀ/√d) V |
| **Multi-head** | Concat of multiple attention outputs |
| **Position encoding** | Sinusoidal or learned |
| **Layer norm** | (x - μ) / √(σ² + ε) |
| **Feed-forward** | Two linear layers + activation |
| **Residual connection** | Add input to output |

**Why √d in attention:** Variance of QKᵀ scales with d; division keeps softmax in non-saturated region.

### 15.3 Convolutional Networks

| Operation | Math |
|---|---|
| **Convolution** | (f * g)[n] = Σ f[m] g[n - m] |
| **Output size** | (W - K + 2P) / S + 1 |
| **Pooling** | Max or average over window |
| **Backprop through conv** | Convolution with flipped kernel |

### 15.4 Variational Autoencoders

Objective (ELBO):
```
log p(x) ≥ E_q[log p(x|z)] - KL(q(z|x) || p(z))
```

- E_q[log p(x|z)]: reconstruction term
- KL(q(z|x) || p(z)): regularization toward prior

**Reparameterization trick:** Sample z = μ + σ·ε where ε ~ N(0, 1). Allows gradients to flow through stochastic node.

### 15.5 Diffusion Models

Forward process (adds noise):
```
q(x_t | x_{t-1}) = N(√(1-β_t) x_{t-1}, β_t I)
```

Closed form: x_t = √(ᾱ_t) x_0 + √(1-ᾱ_t) ε

Reverse process (learned):
```
p_θ(x_{t-1} | x_t) = N(μ_θ(x_t, t), Σ_θ(x_t, t))
```

Training objective (simplified): predict the noise ε at each step.
```
L = E [‖ε - ε_θ(x_t, t)‖²]
```

### 15.6 Reinforcement Learning

**Bellman equation:**
```
V(s) = E_π[r + γ V(s')]
Q(s, a) = E[r + γ max_{a'} Q(s', a')]
```

**Policy gradient theorem:**
```
∇J(θ) = E[∇log π_θ(a|s) · Q(s, a)]
```

**REINFORCE:** Replace Q with Monte Carlo return.
**Actor-Critic:** Use learned baseline (value function).
**PPO:** Constrain policy updates via clipped objective + KL.

### 15.7 Gaussian Processes

Prior: f ~ GP(0, k)
Posterior given observations (X, y):
```
f* | X, y, X* ~ N(K*ᵀ K⁻¹ y, K** - K*ᵀ K⁻¹ K*)
```

Where K, K*, K** are kernel matrices.

**Key property:** Uncertainty quantification for free (predictive variance).

**Bottleneck:** O(n³) inverse — limits to ~10K points.

### 15.8 Matrix Factorization (Recommenders)

```
R ≈ U Vᵀ
```
where R is the rating matrix, U is user latent factors, V is item latent factors.

**SGD update:**
```
e_{ui} = r_{ui} - u_u · v_i
u_u ← u_u + η(e_{ui} v_i - λ u_u)
v_i ← v_i + η(e_{ui} u_u - λ v_i)
```

### 15.9 LLM-Specific Math

| Concept | Math |
|---|---|
| **Token logits → probabilities** | Softmax over vocabulary |
| **Temperature** | Divide logits by T before softmax |
| **Top-k sampling** | Truncate to k highest probabilities |
| **Top-p (nucleus) sampling** | Truncate to smallest set with cumulative prob ≥ p |
| **Perplexity** | exp(cross-entropy) |
| **RLHF reward** | Learned reward model + PPO optimization |
| **DPO** | Bradley-Terry preference modeling |

---

## 16. Common Math Failure Modes in ML

### 16.1 Diagnostic Table

| Symptom | Math Cause | Fix |
|---|---|---|
| Loss is NaN | log(0), 0/0, overflow | Clipping, log-sum-exp, mixed precision care |
| Loss is flat at start | Bad initialization, dead ReLU | Better init (He / Xavier), Leaky ReLU |
| Gradient explodes | Recurrent dynamics, no normalization | Gradient clipping, layer norm |
| Gradient vanishes | Deep network, saturating activations | Skip connections, batch norm, ReLU |
| Weights grow without bound | No regularization, momentum overshoot | L2 reg, weight decay |
| Model can't learn XOR | Linear model on non-linear problem | Add non-linearity / kernel / depth |
| Logistic regression won't converge | Perfectly separable classes | Add regularization |
| OLS coefficients enormous | Multicollinearity, X is rank-deficient | Ridge, drop correlated features, SVD-based solver |
| PCA components meaningless | Not centered, not scaled | StandardScaler before PCA |
| Softmax outputs all 1/n | Logits are zero (or all equal) | Check for layer collapse |
| Variational posterior collapses | KL term dominates | β-VAE, free bits |
| Mode collapse in GAN | Generator finds one good output | Different loss (WGAN), regularization |
| Training accuracy → 100%, val flat | Memorization | More data, regularization, augmentation |
| Confidence calibration off | Model trained with cross-entropy is overconfident | Temperature scaling, label smoothing |
| Bias high in fp16 | Loss scale, gradient underflow | Mixed precision with proper loss scaling |

### 16.2 The Three Main Failure Categories

```
NUMERICAL:
  Overflow, underflow, division by zero, log of zero,
  catastrophic cancellation, ill-conditioning

OPTIMIZATION:
  Saddle points, local minima, plateaus, oscillation,
  divergence, slow convergence

MODELING:
  Wrong distribution, wrong loss function, leakage,
  mismatch between train and inference distributions
```

Most "ML bugs" are one of these three. Diagnosing which is the first step to fixing.

---

## 17. Implementation Quick Reference

### NumPy Essentials

```python
import numpy as np

# Linear algebra
A @ B                             # Matrix multiplication
A.T                               # Transpose
np.linalg.inv(A)                  # Inverse (avoid in practice)
np.linalg.solve(A, b)             # Solve Ax = b (preferred)
np.linalg.lstsq(A, b)             # Least squares solution
np.linalg.eig(A)                  # Eigenvalues, eigenvectors
np.linalg.svd(A)                  # SVD
np.linalg.norm(x, ord=2)          # L2 norm
np.linalg.matrix_rank(A)          # Rank

# Statistics
np.mean(x), np.median(x), np.std(x), np.var(x)
np.percentile(x, [25, 50, 75])
np.cov(X.T)                       # Covariance matrix
np.corrcoef(X.T)                  # Correlation matrix

# Distributions
np.random.normal(mu, sigma, size)
np.random.binomial(n, p, size)
np.random.choice(items, size, p=probabilities)
```

### SciPy for Statistics

```python
from scipy import stats

stats.norm.pdf(x, loc=0, scale=1)
stats.norm.cdf(x, loc=0, scale=1)
stats.t.ppf(0.975, df=n-1)        # Critical value for 95% CI

# Tests
stats.ttest_ind(a, b)
stats.ks_2samp(a, b)              # Distribution comparison
stats.chi2_contingency(table)

# Bootstrap
from scipy.stats import bootstrap
res = bootstrap((data,), np.mean, confidence_level=0.95)
```

### PyTorch Math

```python
import torch
import torch.nn.functional as F

# Tensor operations
x.shape, x.dtype, x.device
torch.matmul(A, B)                # Or A @ B
torch.einsum("ij,jk->ik", A, B)   # General contraction
torch.norm(x, p=2)

# Differentiable operations
y = torch.sigmoid(x)
y = F.softmax(x, dim=-1)
y = F.log_softmax(x, dim=-1)      # Numerically stable
loss = F.cross_entropy(logits, targets)
loss = F.mse_loss(pred, target)
loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1))

# Autograd
loss.backward()                   # Computes gradients
torch.autograd.grad(loss, params) # Manual gradient computation
torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)

# Optimization
optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
```

### Common Patterns

```python
# Stable softmax (manual)
def stable_softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / exp_x.sum(axis=-1, keepdims=True)

# Stable cross-entropy
def cross_entropy(logits, targets):
    log_probs = logits - logsumexp(logits, axis=-1, keepdims=True)
    return -log_probs[range(len(targets)), targets].mean()

# Cosine similarity
def cosine_sim(x, y):
    return (x @ y.T) / (np.linalg.norm(x, axis=1)[:, None] * np.linalg.norm(y, axis=1)[None, :])

# PCA from scratch
def pca(X, k):
    X_centered = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    return X_centered @ Vt[:k].T   # Top-k components

# KL divergence (discrete)
def kl_div(p, q, eps=1e-10):
    return np.sum(p * np.log((p + eps) / (q + eps)))
```

### Calculus Sanity Checks

```python
# Numerical gradient check (debugging autograd)
def numerical_grad(f, x, eps=1e-5):
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy(); x_plus[i] += eps
        x_minus = x.copy(); x_minus[i] -= eps
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * eps)
    return grad

# Verify analytical gradient matches numerical
analytical = compute_gradient_analytically(x)
numerical = numerical_grad(loss_fn, x)
assert np.allclose(analytical, numerical, rtol=1e-4)
```

---

## Summary — The Math You Actually Need to Know Cold

If you remember nothing else, master these 15:

1. **Matrix multiplication** — what shape goes in, what shape comes out, how to compute it
2. **Dot product as similarity** — and cosine similarity
3. **Eigendecomposition / SVD** — what they are, what they reveal
4. **Chain rule** — manually, fluently, for arbitrary compositions
5. **Gradient of a scalar wrt a vector** — and what the shape must be
6. **Bayes' theorem** — and which term is which in any application
7. **Gaussian distribution** — both univariate and multivariate
8. **MLE = minimize negative log-likelihood** — the bridge between probability and loss functions
9. **Cross-entropy = log loss = MLE for classification** — these are the same thing
10. **Convexity** — what it guarantees and what its absence means
11. **Gradient descent and its variants** — SGD, momentum, Adam mechanics
12. **KL divergence** — definition, asymmetry, and where it appears
13. **Bias-variance decomposition** — and the tradeoff
14. **Numerical stability tricks** — log-sum-exp, stable softmax, clipping
15. **The chain rule applied to compute graphs** — the entire content of backpropagation

Everything else is built on these 15. Master them and you can read papers, debug models, and reason about systems. Stay fuzzy on them and you'll forever be cargo-culting algorithms you don't understand.

---

## Further Reading

### Foundational Books
- **Strang** — *Introduction to Linear Algebra* (foundation)
- **Strang** — *Linear Algebra and Learning from Data* (ML-focused)
- **Boyd, Vandenberghe** — *Convex Optimization* (free online)
- **Wasserman** — *All of Statistics*
- **MacKay** — *Information Theory, Inference, and Learning Algorithms* (free online)
- **Bishop** — *Pattern Recognition and Machine Learning*
- **Murphy** — *Probabilistic Machine Learning: An Introduction* and *Advanced Topics*

### Online Resources
- **3Blue1Brown** — *Essence of Linear Algebra*, *Essence of Calculus* (visual intuition)
- **MIT 18.06** — Linear Algebra (Gilbert Strang's classic lectures)
- **Stanford CS229** — Machine Learning (Andrew Ng's notes)
- **Matrix Cookbook** — reference for matrix calculus identities
- **distill.pub** — interactive ML explanations
- **The Deep Learning Book** (Goodfellow, Bengio, Courville) — free online

### Papers Worth Reading for the Math
- "Attention Is All You Need" — the original Transformer
- "Adam: A Method for Stochastic Optimization"
- "Auto-Encoding Variational Bayes" — VAEs
- "Denoising Diffusion Probabilistic Models"
- "Score-Based Generative Modeling through Stochastic Differential Equations"

---

*`Math.md` — Version 1.0*
*Scope: Comprehensive mathematics reference for machine learning and AI*
*Companion to `Metrics.md`, `Audit.md`, `Algorithms.md`, `Features.md`, `Data.md`, `MLOps.md`*
*Use this as the single source of truth for "what's the math behind this technique?"*
